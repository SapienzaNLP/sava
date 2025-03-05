import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RandomEmbeddingInitializer:
    """Initialize the target model with random embeddings."""
    def __init__(
        self,
        source_model: AutoModelForCausalLM,
        source_tokenizer: AutoTokenizer,
        target_tokenizer: AutoTokenizer,
        seed: int = 42,
        tie_weights: bool = False,
    ):
        self.source_model = source_model
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.seed = seed
        self.tie_weights = tie_weights


    def __call__(self) -> AutoModelForCausalLM:
        # Get the source embeddings
        source_embeddings = self.source_model.get_input_embeddings().weight.detach().numpy()
        
        # Generate random embeddings
        # See https://github.com/malteos/clp-transfer
        np.random.seed(self.seed)
        target_embeddings = np.random.normal(
            np.mean(source_embeddings, axis=0), 
            np.std(source_embeddings, axis=0), 
            (
                len(self.target_tokenizer.get_vocab()), 
                source_embeddings.shape[1]
            )
        )

        logger.info(source_embeddings.shape)
        logger.info(target_embeddings.shape)

        if not self.tie_weights:
            source_lm_head = self.source_model.get_output_embeddings().weight.detach().numpy()
            target_lm_head = np.random.normal(
                np.mean(source_lm_head, axis=0), 
                np.std(source_lm_head, axis=0), 
                (
                    len(self.target_tokenizer.get_vocab()), 
                    source_lm_head.shape[1]
                )
            )

            logger.info(target_lm_head.shape)

        # Initialize the target model
        target_embeddings_emb = torch.nn.Embedding(len(self.target_tokenizer.get_vocab()), source_embeddings.shape[1])
        target_embeddings_emb.weight.data = torch.from_numpy(target_embeddings)
        self.source_model.set_input_embeddings(target_embeddings_emb)

        if not self.tie_weights:
            target_lm_head_emb = torch.nn.Embedding(len(self.target_tokenizer.get_vocab()), source_lm_head.shape[1])
            target_lm_head_emb.weight.data = torch.from_numpy(target_lm_head)
            self.source_model.set_output_embeddings(target_lm_head_emb)
        else:
            self.source_model.tie_weights()

        self.source_model.config.vocab_size = len(self.target_tokenizer.get_vocab())
        self.source_model.config.pad_token_id = self.target_tokenizer.pad_token_id
        if self.target_tokenizer.unk_token_id is not None:
            self.source_model.config.unk_token_id = self.target_tokenizer.unk_token_id
        if self.target_tokenizer.eos_token_id is not None:
            self.source_model.config.eos_token_id = self.target_tokenizer.eos_token_id
        if self.target_tokenizer.bos_token_id is not None:
            self.source_model.config.bos_token_id = self.target_tokenizer.bos_token_id

        return self.source_model
