# Embedding Analysis

In this folder there are the scripts use to analyze the embedding structure of the adapted models

## Usage

Compute fertility of a give tokenizer for CulturaX or Wikipedia texts.

```bash
#!/bin/bash

cd /path/to/sava/embedding_analysis

python compute_fertility.py \
    --tokenizer_name TOKENIZER \
    --lang {it, en} \
    --dataset {CULTURAX, WIKIPEDIA}
```

Compute absolute embedding vocabulary similarities between two models that share the same tokenizer.

```bash
#!/bin/bash

cd /path/to/sava/embedding_analysis

python compute_embedding_vocabulary_similarity.py \
    --model_1 MODEL_NAME \
    --model_1 MODEL_NAME \
    --num_neigh NUM_NEIGH \
    --output_dir OUTPUT_DIR \
    --cosine
```

Create a set of anchor tokens and save them, necessary to compute the relative similarities between two models that share the same vocabulary.
```bash
#!/bin/bash

cd /path/to/sava/embedding_analysis

python create_anchors_from_prefixies_to_embedding-sim.py \
    --model_name MODEL_NAME \
    --output_path OUTPUT_PATH \
    --num_anchors NUM_ANCHORS
```

Compute absolute embedding vocabulary relative similarities between two models that share the same tokenizer, given a predefined set of common anchor tokens.

```bash
#!/bin/bash

cd /path/to/sava/embedding_analysis

python compute_embedding_vocabulary_similarity-relative.py \
    --model_1 MODEL_NAME \
    --model_1 MODEL_NAME \
    --output_dir OUTPUT_DIR \
    --anchor_path ANCHOR_PATH
```