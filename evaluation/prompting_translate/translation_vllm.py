import torch
import evaluate
from vllm import LLM, SamplingParams
from datasets import load_dataset
from random import sample
from tqdm import tqdm
import json
import os
import argparse
import csv

os.environ ['CUDA_LAUNCH_BLOCKING'] ='1'

# VARS
CACHE_DATASETS=""
PROMPT_MAPPING = { 
    "it": "Traduci dall'Inglese all'Italiano",
    "en": "Translate from Italian to English"
}

# Definition of evaluation metric
comet_metric = evaluate.load('comet') ## import the scorer
# ----

def create_few_shot_example(text, train_data, from_lang, to_lang, num_shots=5):
    prompt = ""

    # random pick few shot samples from training data
    few_shots_subset_idx = sample(list(range(len(train_data))), num_shots)

    for idx in few_shots_subset_idx:
        prompt += f"{PROMPT_MAPPING[to_lang]}\nText: {train_data[from_lang][idx]}\nTranslation: {train_data[to_lang][idx]}\n\n"

    return prompt + f"{PROMPT_MAPPING[to_lang]}\nText: {text}\nTranslation:"


def formatting_prompts_func(examples, train_data, from_lang, to_lang):
    output_texts = []
    for i in tqdm(range(len(examples["en"]))):

        text = create_few_shot_example(examples[from_lang][i], train_data, from_lang, to_lang, 5)

        output_texts.append(text)
    return output_texts


def map_opus_100(example):
    example["it"] = example["translation"]["it"]
    example["en"] = example["translation"]["en"]

    del example["translation"]

    return example


def main():
    # arguments
    parser = argparse.ArgumentParser(description='Prompting translation experiments')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--from_lang', type=str)
    parser.add_argument('--to_lang', type=str)

    args = parser.parse_args()

    model_name = args.model_name
    from_lang = args.from_lang
    to_lang = args.to_lang

    llm = LLM(model=model_name, gpu_memory_utilization=0.7, max_model_len=2048, dtype="bfloat16", trust_remote_code=True)

    print(llm)

    # Load_dataset: FLORES
    train_data = load_dataset("csv", data_files=f"/leonardo/home/userexternal/lmoroni0/__Work/minerva_sft/Flores/flores.{from_lang}-{to_lang}/dev.csv", index_col=False, cache_dir=CACHE_DATASETS)["train"]
    train_data = train_data.rename_column("src", from_lang)
    train_data = train_data.rename_column("ref", to_lang)

    test_data = load_dataset("csv", data_files=f"/leonardo/home/userexternal/lmoroni0/__Work/minerva_sft/Flores/flores.{from_lang}-{to_lang}/test.csv", index_col=False, cache_dir=CACHE_DATASETS)["train"]
    test_data = test_data.rename_column("src", from_lang)
    test_data = test_data.rename_column("ref", to_lang)

    print(f"Train data lenght {len(train_data)}")
    print(f"Test data lenght {len(test_data)}")
    
    texts = formatting_prompts_func(test_data, train_data, from_lang, to_lang)

    sources = [test_data[from_lang][i] for i in range(len(test_data["en"]))]
    references = [test_data[to_lang][i] for i in range(len(test_data["en"]))]

    sampling_params = SamplingParams(temperature=0, max_tokens=512)

    predicted = [
            out.outputs[0].text  # type: ignore
            for out in llm.generate(
                texts,
                sampling_params
            )  # type: ignore
        ]

    predicted = [p.split("\n\n")[0] for p in predicted]

    with open(f"/leonardo/home/userexternal/lmoroni0/__Work/minerva_sft/prompting_translations/{from_lang}_{to_lang}-{model_name.split('/')[-4]}-{model_name.split('/')[-1]}.csv","w") as f:
        f.writelines(f"SOURCE\tPREDICTED\tREFERENCE\n")
        for s, p, r in zip(sources, predicted, references):
            f.writelines(f"{s}\t{p}\t{r}\n")

    # some clearning trick
    del llm
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    comet_result = comet_metric.compute(predictions=predicted, references=references, sources=sources)
    result = {"comet": round(comet_result["mean_score"] * 100, 4)}

    with open(f"/leonardo/home/userexternal/lmoroni0/__Work/minerva_sft/prompting_translate_results/{from_lang}_{to_lang}-{model_name.split('/')[-4]}-{model_name.split('/')[-1]}.csv","w") as f:
        f.writelines(json.dumps(result))

    print(result)


if __name__ == "__main__":
    main()