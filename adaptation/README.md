# Adaptation

Here you can find the code used to perform the adaptation of pretrained base models.

## Implemented Approaches
* Random
* [CLP](https://arxiv.org/abs/2301.09626)
* [FVT](https://aclanthology.org/2022.emnlp-industry.41/)
* SAVA *\**

## Usage
```bash
$ python src/main.py -h
usage: main.py [-h] --initialization_method
               {random,clp,clp_plus,heuristics,lapt,focus}
               --source_model_name_or_path SOURCE_MODEL_NAME_OR_PATH
               --source_tokenizer_name_or_path SOURCE_TOKENIZER_NAME_OR_PATH
               [--helper_model_name_or_path HELPER_MODEL_NAME_OR_PATH]
               --target_tokenizer_name_or_path TARGET_TOKENIZER_NAME_OR_PATH
               [--cache_dir CACHE_DIR] [--seed SEED] [--copy_special_tokens]
               --output_dir OUTPUT_DIR [--dataset_path DATASET_PATH]
               [--output_data_dir OUTPUT_DATA_DIR]
               [--substitute_intersection] 
               [--num_anchors NUM_ANCHORS]
               [--tie_weights]
               [--anchor_selection ANCHOR_SELECTION]
options:
  -h, --help            show this help message and exit
  --initialization_method {clp,clp_random,average_naive,semantic_alignment}
                        The embedding initialization method to use.
  --source_model_name_or_path SOURCE_MODEL_NAME_OR_PATH
                        The source model to initialize the target model with.
  --source_tokenizer_name_or_path SOURCE_TOKENIZER_NAME_OR_PATH
                        The source tokenizer to initialize the target
                        tokenizer with.
  --helper_model_name_or_path HELPER_MODEL_NAME_OR_PATH
                        [clp, semantic] The helper model to help initialize a
                        terget model.
  --target_tokenizer_name_or_path TARGET_TOKENIZER_NAME_OR_PATH
                        The target
                        tokenizer name or path.
  --cache_dir CACHE_DIR
                        The cache directory to save the pretrained models.
  --seed SEED           The random seed.
  --copy_special_tokens
                        [clp, clp_plus] Whether to copy the special tokens'
                        embeddings from the source model to the target model.
  --output_dir OUTPUT_DIR
                        The output directory to save the target model and
                        tokenizer.
  --num_anchors NUM_ANCHORS
                        [semantic] The number of tokens used to tune the semantic aligner, to be set if anchor_selection is 'random'.
  --anchor_selection {random, full}
                        [semantic] How the anchor tokes are selected from the vocabularies intersection.
  --tie_weights
                        If the source model use tie weights architecture.
  --substitute_intersection
                        [semantic] If map throught the trained aligned even the intersection tokens.
```

## Reproduction
**Example**: The following is to initialize Llama-3.1-8B model for Italian using SAVA.
```bash
#!/bin/bash

cd /path/to/llm-vocab-adaptation/adaptation/src

python main.py \
    --source_model_name_or_path meta-llama/Llama-3.1-8B \
    --source_tokenizer_name_or_path meta-llama/Llama-3.1-8B \
    --helper_model_name_or_path sapienzanlp/Minerva-3B-base-v1.0 \
    --target_tokenizer_name_or_path sapienzanlp/Minerva-3B-base-v1.0 \
    --seed 42 \
    --initialization_method semantic_alignment \
    --aligner sgd \
    --output_dir /path/to/output/model/dir/llama-3.1-italian-sava 
```

## Acknowledgement
This code is based on the repository published by Atsuki Yamaguchi et al. https://github.com/gucci-j/llm-cva?tab=readme-ov-file. We are very gratefull to the 