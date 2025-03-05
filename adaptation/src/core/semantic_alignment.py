from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import random
import numpy as np
from latentis.transform.translate import Translator, Procrustes
from latentis.transform.translate.aligner import MatrixAligner, ZeroPadding, SGDAffineAligner
from latentis.transform.translate.functional import lstsq_align_state
from latentis.transform import TransformSequence
from latentis.transform.base import StandardScaling, MeanLPNorm
import json

# GLOBAL VARIABLES
## ALIGNERS ALIASES
ALIGNERS_embedding = {
    "matrix": lambda _: MatrixAligner(name="affine", align_fn_state=lstsq_align_state),
    "sgd": lambda seed: SGDAffineAligner(num_steps=1000, lr=1e-3, random_seed=seed),
    "ortho": lambda _: Procrustes()
}

ALIGNERS_lm_head = {
    "matrix": lambda _: MatrixAligner(name="affine", align_fn_state=lstsq_align_state),
    "sgd": lambda seed: SGDAffineAligner(num_steps=1000, lr=1e-3, random_seed=seed),
    "ortho": lambda _: Procrustes()
}

INTERSECTION_WORDNET_ITA_PATH = "/home/luca/llm-cva-tatent/stats_anchors/intersection_wordnet_ita.txt"
INTERSECTION_WORDNET_ENG_PATH = "/home/luca/llm-cva-tatent/stats_anchors/intersection_wordnet_eng.txt"
FREQUENCIES_ENG_PATH = "/home/luca/llm-cva-tatent/stats_anchors/token_counts_en_str.json"

def reproducibility(seed):
    # numpy random number generator
    np.random.seed(seed)
    # python random number generator
    random.seed(seed)
    # pytorch random number generator
    torch.manual_seed(seed)


class SemanticAlignmentEmbeddingInitializer:
    """Initialize the target model with Naive Average of missing tokens."""

    def __init__(
            self,
            source_model: AutoModelForCausalLM,
            source_tokenizer: AutoTokenizer,
            helper_model: AutoModelForCausalLM,
            target_tokenizer: AutoTokenizer,
            seed: int = 42,
            tie_weights: bool = True,
            aligner: str = "ortho",
            num_anchors: int = 5000,
            substitute_intersection: bool = False,
            target_vocabulary_strategy: str = "union",
            anchor_selection: str = "random"
    ):
        self.source_model = source_model
        self.source_tokenizer: AutoTokenizer = source_tokenizer
        self.helper_model = helper_model
        self.target_tokenizer = target_tokenizer
        self.seed = seed
        self.tie_weights = tie_weights
        self.aligner = aligner
        self.num_anchors = num_anchors
        self.substitute_intersection = substitute_intersection
        self.target_vocabulary_strategy = target_vocabulary_strategy
        self.anchor_selection = anchor_selection

        print("Tie Weghts -> ", tie_weights)

        #  intersection vocabulary
        self.vocabulary_intersection = {}

        #  target missed vocabulary
        self.target_vocabulary_out = {}

        # Initialize the translators
        self.translator_embedding = Translator(
            aligner=ALIGNERS_embedding[self.aligner](seed),
            x_transform=TransformSequence([StandardScaling(), MeanLPNorm(p=2)]),
            y_transform=TransformSequence([StandardScaling(), MeanLPNorm(p=2)]),
            dim_matcher=ZeroPadding()
        )

        if not self.tie_weights:
            self.translator_lm_head = Translator(
                aligner=ALIGNERS_lm_head[self.aligner](seed),
                x_transform=TransformSequence([StandardScaling(), MeanLPNorm(p=2)]),
                y_transform=TransformSequence([StandardScaling(), MeanLPNorm(p=2)]),
                dim_matcher=ZeroPadding()
            )

        # enable reproducibility
        reproducibility(self.seed)

    def _wordnet_sample_anchors(self) -> list[tuple]:
        print("WORDNET anchor selection")

        intersection = list(self.vocabulary_intersection.items())

        wordnet_tokens = []

        # load the italian words from stats
        with open(INTERSECTION_WORDNET_ITA_PATH, "r") as f:
            italian_tokens = [t.strip() for t in f.readlines()]

        # load the english words from stats
        with open(INTERSECTION_WORDNET_ENG_PATH, "r") as f:
            english_tokens_full = [t.strip() for t in f.readlines()]

        english_tokens = []

        # remove the english tokens that are present even in italian
        for en_tok in english_tokens_full:
            if en_tok in italian_tokens:
                continue

            english_tokens.append(en_tok)

        ENGLISH_TOK_NUM = self.num_anchors - len(italian_tokens)
        
        # sort english tokens by their frequency
        ## pair each token with its frequency in the frequency english file
        with open(FREQUENCIES_ENG_PATH, "r") as f:
            english_tok_frequencies = json.load(f) # are already sorted

        # fill wordnet_tokens with ENGLISH_TOK_NUM english tokens
        for k, _ in english_tok_frequencies.items():
            if k in english_tokens:
                wordnet_tokens.append(k)

            if len(wordnet_tokens) == ENGLISH_TOK_NUM:
                break
        
        wordnet_tokens += italian_tokens

        # Generate anchor_tokens, from intersection...
        anchor_tokens = []
        for tok_mapping in intersection:
            if tok_mapping[0] in wordnet_tokens:
                anchor_tokens.append(tok_mapping)

        return anchor_tokens


    def _prefixes_sample_anchors(self) -> list[tuple]:

        print("PREFIX anchor selection")

        intersection = list(self.vocabulary_intersection.items())

        prefixes = []

        for tok, mapping in intersection:
            if tok.startswith("▁"):
                prefixes.append((tok, mapping))
        
        prefixes.sort(reverse=True, key=lambda x: len(x[0]))

        anchor_tokens = prefixes[:self.num_anchors]

        return anchor_tokens

    def _random_sample_anchors(self) -> list[tuple]:
        anchor_tokens = random.sample(list(self.vocabulary_intersection.items()), self.num_anchors)

        print(f"len(anchor_tokens) = {len(anchor_tokens)}")

        with open(f"random_{len(anchor_tokens)}_selected.txt", "w") as f:
            f.writelines([str(a)+'\n' for (a,_) in anchor_tokens])

        return anchor_tokens

    def _full_intersection_anchors(self) -> list[tuple]:
        print("Take as anchor the full intersection...")

        anchor_tokens = list(self.vocabulary_intersection.items())

        print(f"len(anchor_tokens) = {len(anchor_tokens)}")

        self.num_anchors = len(anchor_tokens)

        return anchor_tokens


    def _map_llama_to_minerva(self, token_str: str):

        if token_str == "<|begin_of_text|>": # begin of sentence token
            return "<s>"

        if token_str == "<|end_of_text|>": # end of sentence token
            return "</s>"

        return token_str.replace("Ġ", "▁")


    def __call__(self) -> AutoModelForCausalLM:
        """Initialize the target model with Naive Average."""
        #####
        # Get the token to index mappings
        #####
        target_token_to_idx = {t: i for t, i in self.target_tokenizer.get_vocab().items()}
        source_token_to_idx = {self._map_llama_to_minerva(t): i for t, i in self.source_tokenizer.get_vocab().items()}

        if self.target_vocabulary_strategy == "union":
            self.target_vocabulary_out = source_token_to_idx.copy()
        else:
            self.target_vocabulary_out = target_token_to_idx.copy()

        new_token_count = 0
        for token, target_idx in target_token_to_idx.items():
            if token in source_token_to_idx:  # intersection
                source_idx = source_token_to_idx[token]
                self.vocabulary_intersection[token] = (source_idx, target_idx)
            else:
                if self.target_vocabulary_strategy == "union":
                    self.target_vocabulary_out[token] = len(source_token_to_idx) + new_token_count
                    new_token_count += 1

        print(f"Intersection length : ({len(self.vocabulary_intersection)})")

        ## get anchor vectors

        if self.anchor_selection == "random":
            anchor_tokens = self._random_sample_anchors()
        elif self.anchor_selection == "prefix":
            anchor_tokens = self._prefixes_sample_anchors()
        elif self.anchor_selection == "wordnet":
            anchor_tokens = self._wordnet_sample_anchors()
        elif self.anchor_selection == "full":
            anchor_tokens = self._full_intersection_anchors()

        #####
        # Generate random embeddings
        #####
        # Get the source and helper embeddings
        #####
        source_embeddings = self.source_model.get_input_embeddings().weight.detach()
        helper_embeddings = self.helper_model.get_input_embeddings().weight.detach()

        ## get intersection spaces
        helper_embeddings_intersection, source_embeddings_intersection = self.get_spaces_intersection(
            helper_embeddings, source_embeddings)

        target_embeddings = torch.nn.Embedding(len(self.target_vocabulary_out), source_embeddings.size(1))
        helper_embeddings_anchor, source_embeddings_anchor = self.extract_anchor_embeddings(anchor_tokens,
                                                                                            helper_embeddings,
                                                                                            source_embeddings)

        # Compute the transformation on the embedding vectors
        self.compute_transformation_embeddings(helper_embeddings, helper_embeddings_anchor,
                                               helper_embeddings_intersection, source_embeddings,
                                               source_embeddings_anchor, source_embeddings_intersection,
                                               source_token_to_idx, target_embeddings, target_token_to_idx,
                                               self.translator_embedding)

        if not self.tie_weights:
            source_lm_head = self.source_model.get_output_embeddings().weight.detach()
            helper_lm_head = self.helper_model.get_output_embeddings().weight.detach()

            # get intersection spaces
            helper_lm_head_intersection, source_lm_head_intersection = self.get_spaces_intersection(
                helper_lm_head, source_lm_head)

            helper_lm_head_anchor, source_lm_head_anchor = self.extract_anchor_embeddings(anchor_tokens,
                                                                                                helper_lm_head,
                                                                                                source_lm_head)

            target_lm_head = torch.nn.Linear(source_embeddings.size(1), len(self.target_vocabulary_out), bias=False)

            # Compute the transformation on the embedding vectors
            self.compute_transformation_embeddings(helper_lm_head, helper_lm_head_anchor,
                                                   helper_lm_head_intersection, source_lm_head,
                                                   source_lm_head_anchor, source_lm_head_intersection,
                                                   source_token_to_idx, target_lm_head,
                                                   target_token_to_idx, self.translator_lm_head)


        # Initialize the target model
        self.source_model.set_input_embeddings(target_embeddings)
        if not self.tie_weights:
            self.source_model.set_output_embeddings(target_lm_head)
        else:
            self.source_model.tie_weights()

        if self.target_vocabulary_strategy == "union":
            # update tokenizier
            rev_target_vocab_out = {idx: t for t, idx in self.target_vocabulary_out.items()}
            new_tokens = []
            for i in range(new_token_count):
                new_tokens.append(rev_target_vocab_out[len(source_token_to_idx) + i])

            self.source_tokenizer.add_tokens(new_tokens)

        # Update the config
        self.source_model.config.vocab_size = len(self.target_vocabulary_out)
        self.source_model.config.pad_token_id = self.target_tokenizer.pad_token_id
        if self.target_tokenizer.unk_token_id is not None:
            self.source_model.config.unk_token_id = self.target_tokenizer.unk_token_id
        if self.target_tokenizer.eos_token_id is not None:
            self.source_model.config.eos_token_id = self.target_tokenizer.eos_token_id
        if self.target_tokenizer.bos_token_id is not None:
            self.source_model.config.bos_token_id = self.target_tokenizer.bos_token_id

        return self.source_model

    def extract_anchor_embeddings(self, anchor_tokens, helper_embeddings, source_embeddings):
        source_embeddings_anchor = torch.zeros((self.num_anchors, source_embeddings.size(1)))
        helper_embeddings_anchor = torch.zeros((self.num_anchors, helper_embeddings.size(1)))
        # select anchors in the embedding space
        for i, (token, (source_idx, target_idx)) in enumerate(anchor_tokens):
            source_embeddings_anchor[i] = source_embeddings[source_idx]
            helper_embeddings_anchor[i] = helper_embeddings[target_idx]
        return helper_embeddings_anchor, source_embeddings_anchor

    def compute_transformation_embeddings(self, helper_embeddings, helper_embeddings_anchor,
                                          helper_embeddings_intersection, source_embeddings, source_embeddings_anchor,
                                          source_embeddings_intersection, source_token_to_idx, target_embeddings,
                                          target_token_to_idx, translator):
        translator.fit(x=helper_embeddings_anchor, y=source_embeddings_anchor)
        #if self.substitute_intersection and self.target_vocabulary_strategy == "substitution":
            # new embedding will be the transformer embedding of the helper model
            # target_embeddings.weight.data = translator(helper_embeddings)["x"]
        #else:
            # if self.target_vocabulary_strategy == "substitution":
            #    for token, (source_idx, target_idx) in self.vocabulary_intersection.items():
            #        target_embeddings.weight.data[target_idx] = source_embeddings[source_idx]

        for token, target_idx in self.target_vocabulary_out.items():
            # if self.target_vocabulary_strategy == "union" and token in source_token_to_idx:
            if token in source_token_to_idx:
                target_embeddings.weight.data[target_idx] = source_embeddings[source_token_to_idx[token]]
            else:
                target_embeddings.weight.data[target_idx] = \
                    translator(helper_embeddings[target_token_to_idx[token]].unsqueeze(0))["x"].flatten()

        ## Print Statistics
        print(
            f"MEAN SQUARED ERROR - EMBEDDING : {(source_embeddings_intersection - translator(helper_embeddings_intersection)['x']).abs().mean()}")
        print(
            f"COSINE - EMBEDDING : {F.cosine_similarity(source_embeddings_intersection, translator(helper_embeddings_intersection)['x']).abs().mean()}")

    def get_spaces_intersection(self, helper_embeddings, source_embeddings):
        source_embeddings_intersection = torch.zeros((len(self.vocabulary_intersection), source_embeddings.size(1)))
        helper_embeddings_intersection = torch.zeros((len(self.vocabulary_intersection), helper_embeddings.size(1)))
        for i, (_, (source_idx, target_idx)) in enumerate(self.vocabulary_intersection.items()):
            source_embeddings_intersection[i] = source_embeddings[source_idx]
            helper_embeddings_intersection[i] = helper_embeddings[target_idx]
        return helper_embeddings_intersection, source_embeddings_intersection
