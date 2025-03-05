import logging
import math
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import entmax
import torch
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer


def round_to_nearest_multiple(vocabulary_size, multiple):
    rounded_size = math.ceil(vocabulary_size / multiple) * multiple
    return rounded_size


class CLPPlusEmbeddingInitializer:
    """Initialize the target model with CLP."""
    def __init__(
        self,
        source_model: AutoModelForCausalLM,
        helper_model: AutoModelForCausalLM,
        source_tokenizer: AutoTokenizer,
        target_tokenizer: AutoTokenizer,
        copy_special_tokens: bool = False,
        seed: int = 42,
        tie_weights: bool = True
    ):
        self.source_model = source_model
        self.helper_model = helper_model
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.copy_special_tokens = copy_special_tokens
        self.seed = seed
        self.tie_weights = tie_weights

    def clp_plus(self, source_embeddings, helper_embeddings):
        """Initialize the target model with CLP+."""
        #####
        # Get the token to index mappings
        #####
        target_token_to_idx = {t: i for t, i in self.target_tokenizer.get_vocab().items()}
        source_token_to_idx = {t: i for t, i in self.source_tokenizer.get_vocab().items()}

        #####
        # Generate random embeddings
        # See https://github.com/malteos/clp-transfer
        #####
        np.random.seed(self.seed)
        target_embeddings = np.random.normal(
            np.mean(source_embeddings, axis=0), 
            np.std(source_embeddings, axis=0), 
            (
                round_to_nearest_multiple(len(self.target_tokenizer), 8),
                source_embeddings.shape[1]
            )
        )
        print(source_embeddings.shape)
        print(target_embeddings.shape)
        print(helper_embeddings.shape)
        
        #####
        # Initialize the target embeddings with the source embeddings for overlapping tokens
        #####
        # Get the overlapping tokens
        target_tokens = set(self.target_tokenizer.get_vocab().keys())
        source_tokens = set(self.source_tokenizer.get_vocab().keys())
        overlapping_tokens = target_tokens & source_tokens
        overlapping_tokens_list = list(overlapping_tokens)
        if not overlapping_tokens:
            raise ValueError('No overlapping tokens! Cannot initialize the target model with CLP.')

        # Copy the source embeddings for overlapping tokens
        for token in overlapping_tokens:
            target_embeddings[target_token_to_idx[token]] = \
                source_embeddings[source_token_to_idx[token]]
        
        #####
        # Initialize the missing tokens with the weighted mean of the source embeddings
        #####
        # Get missing tokens
        missing_tokens = target_tokens - source_tokens
        missing_tokens_list = list(missing_tokens)

        if not missing_tokens:
            logger.info('No missing tokens!')
        else:
            # Get the embeddings for the missing tokens and overlapping tokens in the helper model
            # IndexError: index 60946 is out of bounds for axis 0 with size 60928
            helper_missing_token_embeddings = \
                helper_embeddings[[target_token_to_idx[t] for t in missing_tokens_list], :]
            helper_overlapping_token_embeddings = \
                helper_embeddings[[target_token_to_idx[t] for t in overlapping_tokens_list], :]
            
            # Get the embeddings for the overlapping tokens in the source model
            overlapping_tokens_idxs = \
                [source_token_to_idx[t] for t in overlapping_tokens_list]
            overlapping_token_vecs = torch.from_numpy(source_embeddings[overlapping_tokens_idxs, :]) # -> (len(overlapping_tokens), source_embedding_dim)

            # Calculate the cosine similarity between the missing tokens and overlapping tokens in the helper model
            cos_sims = cosine_similarity(
                helper_missing_token_embeddings, 
                helper_overlapping_token_embeddings
            ) # -> (len(missing_tokens), len(overlapping_tokens))

            # Initialize the target embeddings with the weighted mean of the overlapping tokens in the source model
            for index, token in enumerate(tqdm(missing_tokens_list)):
                # Get the cosine similarity scores for the missing token
                token_cos_sim = entmax.sparsemax(torch.from_numpy(cos_sims[index])) # -> (len(overlapping_tokens),)
                logger.info(f"token_cos_sim: {token_cos_sim.shape}")
                
                # Get the weighted mean of the overlapping tokens in the source model
                mask = token_cos_sim > 0.0
                masked_token_cos_sim = token_cos_sim[mask] # -> (num_token_cos_sim_positive,)
                masked_overlapping_token_vecs = overlapping_token_vecs[mask] # -> (num_token_cos_sim_positive, source_embedding_dim)
                logger.info(f"masked_token_cos_sim: {masked_token_cos_sim.shape}")
                logger.info(f"masked_overlapping_token_vecs: {masked_overlapping_token_vecs.shape}")
                weighted_src_embs = torch.mul(
                    masked_overlapping_token_vecs, 
                    masked_token_cos_sim.unsqueeze(1)
                ) # -> (num_token_cos_sim_positive, source_embedding_dim)
                logger.info(f"weighted_src_embs: {weighted_src_embs.shape}")
                logger.info("=" * 10)
                weighted_mean = torch.sum(weighted_src_embs, dim=0) # -> (source_embedding_dim,)
                
                # Set the embedding of the current missing token to the weighted mean
                target_embeddings[target_token_to_idx[token]] = weighted_mean.detach().numpy()

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
                    source_special_token_idx = source_token_to_idx[source_special_token]
                    target_special_token_idx = target_token_to_idx[target_special_token]
                    target_embeddings[target_special_token_idx] = \
                        source_embeddings[source_special_token_idx]

        return target_embeddings

    def __call__(self) -> AutoModelForCausalLM:
        """Initialize the target model with CLP.

        Raises:
            ValueError: No overlapping tokens between source and target model.

        Returns:
            AutoModelForCausalLM: The target model initialized with CLP.
        
        References:
            - https://github.com/malteos/clp-transfer
            - https://github.com/konstantinjdobler/focus
        """
        #####
        # Get the source and helper embeddings
        #####
        source_embeddings = self.source_model.get_input_embeddings().weight.detach().numpy()
        helper_embeddings = self.helper_model.get_input_embeddings().weight.detach().numpy()

        target_embeddings = self.clp_plus(source_embeddings, helper_embeddings)

        #####
        # Initialize the target model
        #####
        target_model = self.source_model
        #move them to same precision as the source model
        target_embeddings = torch.from_numpy(target_embeddings).to(target_model.get_input_embeddings().weight.dtype) 
        target_model.set_input_embeddings(target_embeddings)
        if self.tie_weights:
            target_model.tie_weights()
            logger.info(target_model.get_input_embeddings().weight.data.shape)
        else:
            source_output_embeddings = self.source_model.get_output_embeddings().weight.detach().numpy()
            helper_output_embeddings = self.helper_model.get_output_embeddings().weight.detach().numpy()

            target_output_embeddings = self.clp_plus(source_output_embeddings, helper_output_embeddings)
            target_output_embeddings = torch.from_numpy(target_output_embeddings).to(target_model.get_output_embeddings().weight.dtype)
            target_model.set_output_embeddings(target_output_embeddings)
            logger.info(target_model.get_output_embeddings().weight.data.shape)

        #####
        # Update tokenizer and config
        #####
        target_model.config.vocab_size = len(self.target_tokenizer)
        target_model.config.pad_token_id = self.target_tokenizer.pad_token_id
        if self.target_tokenizer.unk_token_id is not None:
            target_model.config.unk_token_id = self.target_tokenizer.unk_token_id
        if self.target_tokenizer.eos_token_id is not None:
            target_model.config.eos_token_id = self.target_tokenizer.eos_token_id
        if self.target_tokenizer.bos_token_id is not None:
            target_model.config.bos_token_id = self.target_tokenizer.bos_token_id

        return target_model