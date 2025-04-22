import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

class CLPEmbeddingInitializer:
    """Initialize the target model with CLP."""
    def __init__(
        self,
        source_model: AutoModelForCausalLM,
        helper_model: AutoModelForCausalLM,
        source_tokenizer: AutoTokenizer,
        target_tokenizer: AutoTokenizer,
        copy_special_tokens: bool = False,
        seed: int = 42,
        tie_weights: bool = False,
        target_vocabulary_strategy: str = "union"
    ):
        self.source_model = source_model
        self.helper_model = helper_model
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.copy_special_tokens = copy_special_tokens
        self.seed = seed
        self.target_vocabulary_strategy=target_vocabulary_strategy
        self.tie_weights=tie_weights

        print("Tie weights ->", self.tie_weights)


    def _initialize_missing_embeddings(self, helper_embeddings, target_embeddings, source_embeddings):
            # Get the embeddings for the missing tokens and overlapping tokens in the helper model
            helper_missing_token_embeddings = \
                helper_embeddings[[self.target_token_to_idx[t] for t in self.missing_tokens_list], :]
            helper_overlapping_token_embeddings = \
                helper_embeddings[[self.target_token_to_idx[t] for t in self.overlapping_tokens_list], :]
            
            # Get the embeddings for the overlapping tokens in the source model
            overlapping_tokens_idxs = \
                [self.source_token_to_idx[t] for t in self.overlapping_tokens_list]
            overlapping_token_vecs = source_embeddings[overlapping_tokens_idxs, :] # -> (len(overlapping_tokens), embedding_dim)

            # Calculate the cosine similarity between the missing tokens and overlapping tokens in the helper model
            cos_sims = cosine_similarity(
                helper_missing_token_embeddings, 
                helper_overlapping_token_embeddings
            ) # -> (len(missing_tokens), len(overlapping_tokens))

            # Initialize the target embeddings with the weighted mean of the overlapping tokens in the source model
            for index, token in enumerate(tqdm(self.missing_tokens_list)):
                # Get the cosine similarity scores for the missing token
                token_cos_sim = cos_sims[index] # -> (len(overlapping_tokens),)
                normed_token_cos_sim = token_cos_sim / token_cos_sim.sum() # -> (len(overlapping_tokens),)
                
                # Calculate the weighted mean of the overlapping tokens in the source model
                target_vec = np.average(
                    overlapping_token_vecs, axis=0, weights=normed_token_cos_sim
                ) # -> (embedding_dim,)

                # Set the embedding of the current missing token to the weighted mean
                target_embeddings[self.target_vocabulary_out[token]] = target_vec


    def __call__(self) -> AutoModelForCausalLM:
        """Initialize the target model with CLP.

        Raises:
            ValueError: No overlapping tokens between source and target model.

        Returns:
            AutoModelForCausalLM: The target model initialized with CLP.
        
        References:
            - https://github.com/malteos/clp-transfer
        """
        #####
        # Get the source and helper embeddings
        #####
        source_embeddings = self.source_model.get_input_embeddings().weight.detach().numpy()
        helper_embeddings = self.helper_model.get_input_embeddings().weight.detach().numpy()

        #####
        # Get the token to index mappings
        #####
        self.target_token_to_idx = {t: i for t, i in self.target_tokenizer.get_vocab().items()} # vocabulary index of the helper model
        self.source_token_to_idx = {t: i for t, i in self.source_tokenizer.get_vocab().items()}

        if self.target_vocabulary_strategy == "union":
            self.target_vocabulary_out = self.source_token_to_idx.copy()

            new_token_count = 0
            for token, _ in self.target_token_to_idx.items():
                    if not token in self.source_token_to_idx:
                        self.target_vocabulary_out[token] = len(self.source_token_to_idx) + new_token_count
                        new_token_count += 1
        else:
            self.target_vocabulary_out = self.target_token_to_idx.copy()

        #####
        # Generate random embeddings
        # See https://github.com/malteos/clp-transfer
        #####
        np.random.seed(self.seed)
        target_embeddings = np.random.normal(
            np.mean(source_embeddings, axis=0), 
            np.std(source_embeddings, axis=0), 
            (
                len(self.target_vocabulary_out), 
                source_embeddings.shape[1]
            )
        )
        logger.info(f"source embedding shape {source_embeddings.shape}")
        logger.info(f"target embedding shape {target_embeddings.shape}")

        #####
        # Initialize the target embeddings with the source embeddings for overlapping tokens
        #####
        # Get the overlapping tokens
        target_tokens = set(self.target_tokenizer.get_vocab().keys())
        source_tokens = set(self.source_tokenizer.get_vocab().keys())
        overlapping_tokens = target_tokens & source_tokens
        self.overlapping_tokens_list = list(overlapping_tokens)
        if not overlapping_tokens:
            raise ValueError('No overlapping tokens! Cannot initialize the target model with CLP.')

        if self.target_vocabulary_strategy == "substitution":
            # Copy the source embeddings for overlapping tokens
            for token in overlapping_tokens:
                target_embeddings[self.target_token_to_idx[token]] = \
                    source_embeddings[self.source_token_to_idx[token]]
        else:
            # Copy the source embeddings for all the source tokens
            for token, _ in self.target_vocabulary_out.items(): # run over the target vocabulary
                if token in self.source_token_to_idx: # substitute the tokens in the source vocabulary with their source representation
                    target_embeddings[self.source_token_to_idx[token]] = source_embeddings[self.source_token_to_idx[token]]
                
        # LM_HEAD
        if not self.tie_weights:
            source_lm_head = self.source_model.get_output_embeddings().weight.detach().numpy()
            helper_lm_head = self.helper_model.get_output_embeddings().weight.detach().numpy()

            target_lm_head = np.random.normal(
                np.mean(source_lm_head, axis=0), 
                np.std(source_lm_head, axis=0), 
                (
                    len(self.target_vocabulary_out), 
                    source_lm_head.shape[1]
                )
            )

            if self.target_vocabulary_strategy == "substitution":
                # Copy the source embeddings for overlapping tokens
                for token in overlapping_tokens:
                    target_lm_head[self.target_token_to_idx[token]] = \
                        source_lm_head[self.source_token_to_idx[token]]
            else:
                # Copy the source embeddings for all the source tokens
                for token, idx in self.target_vocabulary_out.items(): # run over the target vocabulary
                    if token in self.source_token_to_idx: # substitute the tokens in the source vocabulary with their source representation
                        target_lm_head[self.source_token_to_idx[token]] = source_lm_head[self.source_token_to_idx[token]]

        
        #####
        # Initialize the missing tokens with the weighted mean of the source embeddings
        #####
        # Get missing tokens
        missing_tokens = target_tokens - source_tokens
        self.missing_tokens_list = list(missing_tokens)

        if not missing_tokens:
            logger.info('No missing tokens!')
        else:
            self._initialize_missing_embeddings(helper_embeddings, target_embeddings, source_embeddings)

            # LM_HEAD
            if not self.tie_weights:
                self._initialize_missing_embeddings(helper_lm_head, target_lm_head, source_lm_head)
            
        #####
        # Set the embeddings for the special tokens
        # - See https://github.com/cmdowney88/embeddingstructure
        #####
        if self.copy_special_tokens:
            # Get the special tokens
            source_special_tokens_map = self.source_tokenizer.special_tokens_map
            target_special_tokens_map = self.target_tokenizer.special_tokens_map
            
            # Copy the source embeddings for the special tokens
            for special_token_name, target_special_token in target_special_tokens_map.items():
                if special_token_name in source_special_tokens_map:
                    source_special_token = source_special_tokens_map[special_token_name]
                    source_special_token_idx = self.source_token_to_idx[source_special_token]
                    target_special_token_idx = self.target_token_to_idx[target_special_token]
                    target_embeddings[target_special_token_idx] = \
                        source_embeddings[source_special_token_idx]
                    
                    if not self.tie_weights:
                        target_lm_head[target_special_token_idx] = \
                            source_lm_head[source_special_token_idx]
        #####
        # Initialize the target model
        #####
        target_embeddings_emb = torch.nn.Embedding(len(self.target_vocabulary_out), source_embeddings.shape[1])
        target_embeddings_emb.weight.data = torch.from_numpy(target_embeddings)
        self.source_model.set_input_embeddings(target_embeddings_emb)
        if not self.tie_weights:
            target_lm_head_emb = torch.nn.Embedding(len(self.target_vocabulary_out), source_lm_head.shape[1])
            target_lm_head_emb.weight.data = torch.from_numpy(target_lm_head)
            self.source_model.set_output_embeddings(target_lm_head_emb)
        else:
            self.source_model.tie_weights()

        if self.target_vocabulary_strategy == "union":
            # update tokenizier
            rev_target_vocab_out = {idx: t for t, idx in self.target_vocabulary_out.items()}
            new_tokens = []
            for i in range(new_token_count):
                new_tokens.append(rev_target_vocab_out[len(self.source_token_to_idx) + i])

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