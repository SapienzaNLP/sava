# Training

This folder contains some file used to make the training recipe reproducible.

## Configs

Under the `yamls` folder there are the config files used by the [LLM-Foundry](https://github.com/mosaicml/llm-foundry/tree/v0.8.0) library.

In our experiments we relied on the LLM-Foundry library at version *v0.8.0*.

## Data used for the adaptation

The adapted model were trained on a collection of Italian and English data extracted from [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX).
The data were extracted to be skewed toward Italian language with a ration of *one over four*. Extracting the **first 9B tokens from Italian part of CulturaX and the first 3B tokens from English part of CulturaX.**