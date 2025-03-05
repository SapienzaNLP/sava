from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class AverageEmbeddingInitializer:
    """Initialize the target model with Naive Average of missing tokens."""
    def __init__(
        self,
        target_model: AutoModelForCausalLM,
        source_tokenizer: AutoTokenizer,
        target_tokenizer: AutoTokenizer,
        copy_special_tokens: bool = False,
        seed: int = 42,
        tie_weights: bool = True
    ):
        self.target_model = target_model
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.copy_special_tokens = copy_special_tokens
        self.seed = seed
        self.tie_weights = tie_weights


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

        #####
        # Generate random embeddings
                #####
        # Get the source and helper embeddings
        #####
        source_embeddings = self.target_model.get_input_embeddings().weight.detach()
        target_embeddings = torch.nn.Embedding(len(self.target_tokenizer), source_embeddings.shape[1])
        if not self.tie_weights:
            source_lm_head = self.target_model.get_output_embeddings().weight.detach()
            target_lm_head = torch.nn.Linear(source_lm_head.shape[1], len(self.target_tokenizer), bias=False)
        else:
            source_lm_head = source_embeddings
            target_lm_head = torch.nn.Linear(source_embeddings.shape[1], len(self.target_tokenizer), bias=False)

        for token, idx in target_token_to_idx.items():
            if token in source_token_to_idx:
                source_idx = source_token_to_idx[token]
                target_embeddings.weight.data[idx] = source_embeddings[source_idx]
                target_lm_head.weight.data[idx] = source_lm_head[source_idx]
            else:
                source_tokens = self.source_tokenizer.tokenize(token.replace("_", " "))
                source_idxs = self.source_tokenizer.convert_tokens_to_ids(source_tokens)
                target_embeddings.weight.data[idx] = source_embeddings[source_idxs].mean(dim=0)
                target_lm_head.weight.data[idx] = source_lm_head[source_idxs].mean(dim=0)

        # move them to same precision as the source model
        target_embeddings = target_embeddings.to(source_embeddings.dtype)
        target_lm_head = target_lm_head.to(source_lm_head.dtype)

        # Initialize the target model
        self.target_model.set_input_embeddings(target_embeddings)
        if not self.tie_weights:
            self.target_model.set_output_embeddings(target_lm_head)
        else:
            self.target_model.tie_weights()
        
        # Update the config
        self.target_model.config.vocab_size = len(self.target_tokenizer)
        self.target_model.config.pad_token_id = self.target_tokenizer.pad_token_id
        if self.target_tokenizer.unk_token_id is not None:
            self.target_model.config.unk_token_id = self.target_tokenizer.unk_token_id
        if self.target_tokenizer.eos_token_id is not None:
            self.target_model.config.eos_token_id = self.target_tokenizer.eos_token_id
        if self.target_tokenizer.bos_token_id is not None:
            self.target_model.config.bos_token_id = self.target_tokenizer.bos_token_id

        return self.target_model

        


                