<div align="center">

# Optimizing LLMs for Italian: Reducing Token Fertility and Enhancing Efficiency Through Vocabulary Adaptation

<img src="https://github.com/Andrew-Wyn/images/blob/master/sava/italian_adapt-img.jpg?raw=true" width="400" style="border-radius:10%"/>

<br>

[![Conference](https://img.shields.io/badge/NAACL-2025-4b44ce)](https://2025.naacl.org/)
[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2504.17025v1)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection_Mistral-FCD21D)](https://huggingface.co/collections/SemanticAlignment/mistral-7b-v01-adapted-679243206cec8a21f75435dd)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection_Llama-FCD21D)](https://huggingface.co/collections/SemanticAlignment/llama-31-adapted-67924314d8957c78a3e7bcaf)
</div>

A repository containing the original code and models for the paper:

Luca Moroni, Giovanni Puccetti, Pere-Lluís Huguet Cabot, Andrei Stefan Bejgu, Alessio Miaschi, Edoardo Barba, Felice Dell’Orletta, Andrea Esuli, Roberto Navigli. 
[Optimizing LLMs for Italian: Reducing Token Fertility and Enhancing Efficiency Through Vocabulary Adaptation](https://aclanthology.org/2025.findings-naacl.371.pdf), in *Findings of NAACL 2025*. 

# Usage

This repository is divided in four parts, `adaptation`, `embedding analysis`, `train`, and `analysis`.

Each part is implemented and documented in the respective folder of this repository.

* The *Adaptation* part constains the code to reproduce the adaptation of english LLMs on a given tokenizer.
* The *Embedding Analysis* part contains the script used to analyze the embedding structure of the adapted models.
* The *Train* folder contains the code and the reference for the library used to train adapted models.
* The *Evaluation* folder contains the code and the reference of the dataset and libraries used to evaluate adapted models during the further stage of training.

## Cite this work

If you use any part of this work, please consider citing the paper as follows:

```bibtex
@inproceedings{moroni2025optimizing,
  title={Optimizing LLMs for Italian: Reducing Token Fertility and Enhancing Efficiency Through Vocabulary Adaptation},
  author={Moroni, Luca and Puccetti, Giovanni and Cabot, Pere-Llu{\'\i}s Huguet and Bejgu, Andrei Stefan and Miaschi, Alessio and Barba, Edoardo and Dell’Orletta, Felice and Esuli, Andrea and Navigli, Roberto},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
  pages={6646--6660},
  year={2025}
}
```

## 🪪 License

The data and software are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgements
We gratefully acknowledge the support of Future AI Research ([PNRR MUR project PE0000013-FAIR](https://fondazione-fair.it/en/)). 
Partially financed by the European Union - NextGenerationEU through the Italian Ministry of University and Research under PNRR - PRIN 2022 (2022EPTPJ9) "WEMB: Word Embeddings from Cognitive Linguistics to Language Engineering and back" and by the PNRR project ITSERR (CUP B53C22001770006). We acknowledge the support of the ISCRA project TRAVEL (HP10CY9V7K) for awarding access to the LEONARDO supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CINECA (Italy) and thank Giuseppe Fiameni for his support.
