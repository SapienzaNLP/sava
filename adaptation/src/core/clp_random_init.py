from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import torch.nn.functional as F
import random
import numpy as np


def reproducibility(seed):
    # numpy random number generator
    np.random.seed(seed)
    # python random number generator
    random.seed(seed)
    # pytorch random number generator
    torch.manual_seed(seed)
    # transformer random number generator
    set_seed(seed)

class CLPRandomEmbeddingInitializer:
    """Initialize the target model with Naive Average of missing tokens."""

    def __init__(
            self,
            source_model: AutoModelForCausalLM,
            source_tokenizer: AutoTokenizer,
            target_tokenizer: AutoTokenizer,
            seed: int = 42,
            tie_weights: bool = True,
            target_vocabulary_strategy: str = "union"
    ):
        self.source_model = source_model
        self.source_tokenizer: AutoTokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.seed = seed
        self.tie_weights = tie_weights
        self.target_vocabulary_strategy = target_vocabulary_strategy

        #  intersection vocabulary
        self.vocabulary_intersection = {}

        #  target missed vocabulary
        self.target_vocabulary_out = {}

        # enable reproducibility
        reproducibility(self.seed)

    def __call__(self) -> AutoModelForCausalLM:
        """Initialize the target model with Naive Average."""
        #####
        # Get the token to index mappings
        #####
        target_token_to_idx = {t: i for t, i in self.target_tokenizer.get_vocab().items()}
        source_token_to_idx = {t: i for t, i in self.source_tokenizer.get_vocab().items()}

        if self.target_vocabulary_strategy == "union":
            self.target_vocabulary_out = source_token_to_idx.copy()
            new_token_count = 0
            for token, _ in target_token_to_idx.items():
                if not token in source_token_to_idx:
                    self.target_vocabulary_out[token] = len(source_token_to_idx) + new_token_count
                    new_token_count += 1
        else:
            self.target_vocabulary_out = target_token_to_idx.copy()

        #####
        # Generate random embeddings
        #####
        # Get the source embeddings
        #####
        source_embeddings = self.source_model.get_input_embeddings().weight.detach()

        target_embeddings = np.random.normal(
            torch.mean(source_embeddings, axis=0), 
            torch.std(source_embeddings, axis=0), 
            (
                len(self.target_vocabulary_out), 
                source_embeddings.shape[1]
            )
        )

        for token, idx in self.target_vocabulary_out.items(): # run over the target vocabulary
            if token in source_token_to_idx: # substitute the tokens in the source vocabulary with their source representation
                target_embeddings[idx] = source_embeddings[source_token_to_idx[token]]
            
        if not self.tie_weights:
            source_lm_head = self.source_model.get_output_embeddings().weight.detach()
            target_lm_head = np.random.normal(
                torch.mean(source_lm_head, axis=0), 
                torch.std(source_lm_head, axis=0), 
                (
                    len(self.target_vocabulary_out), 
                    source_lm_head.shape[1]
                )
            )

            for token, idx in self.target_vocabulary_out.items(): # run over the target vocabulary
                if token in source_token_to_idx: # substitute the tokens in the source vocabulary with their source representation
                    target_lm_head[idx] = source_lm_head[source_token_to_idx[token]]

        # Initialize the target model
        target_embeddings_emb = torch.nn.Embedding(len(self.target_vocabulary_out), source_embeddings.size(1))
        target_embeddings_emb.weight.data = torch.from_numpy(target_embeddings)
        self.source_model.set_input_embeddings(target_embeddings_emb)
        if not self.tie_weights:
            target_lm_head_emb = torch.nn.Embedding(len(self.target_vocabulary_out), source_lm_head.size(1))
            target_lm_head_emb.weight.data = torch.from_numpy(target_lm_head)
            self.source_model.set_output_embeddings(target_lm_head_emb)
        else:
            self.source_model.tie_weights()

        # update tokenizier
        if self.target_vocabulary_strategy == "union":
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