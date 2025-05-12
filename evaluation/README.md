# Evaluation

This folder contains the scripts used to evaluate Italian adapted models.

## Generative Benchmarks

For the generative benchmarks we relied on SQUAD-it and FLORES.

### SQUAD-it

Evaluate a model on SQUAD-it dataset.

```bash
#!/bin/bash

cd /path/to/sava/evaluation

python prompting_squad/squad_vllm.py \
    --model_name MODEL_NAME \
    --output_dir OUTPUT_DIR \
```

### FLORES

Transform FLORES dataset into a proper format.

```bash
#!/bin/bash

cd /path/to/sava/evaluation

python download_and_process_flores.py \
    --output_folder OUTPUT_FOLDER
```

Evaluate a model on FLORES dataset.

```bash
#!/bin/bash

cd /path/to/sava/evaluation

python prompting_squad/tranlsation_vllm.py \
    --model_name MODEL_NAME \
    --dataset_path DATASET_PATH \
    --output_dir OUTPUT_DIR \
    --from_lang FROM_LANG \
    --to_lang TO_LANG
```

## Multi-choice Benchmarks

For the multiple-choices we used [ITA-Bench](https://github.com/SapienzaNLP/ita-bench) suite.

###  ⚠ Warning: ⚠

*the paper results are based on the first version of ITA-Bench, we will upload here the results with the newer version of the benchmark. (The new results do not change our findings).*