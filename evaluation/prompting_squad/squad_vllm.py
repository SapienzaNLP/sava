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

# Definition of evaluation metric
rouge = evaluate.load('rouge') ## import the scorer
# ----

def create_few_shot_example(e, train_data, num_shots=2):
    prompt = ""

    # random pick few shot samples from training data
    few_shots_subset_idx = sample(list(range(len(train_data))), num_shots)

    for idx in few_shots_subset_idx:
        prompt += f"Contesto: {train_data['context'][idx]}\nDomanda: {train_data['question'][idx]}\nRisposta: {train_data['answer'][idx]}\n\n"

    return prompt + f"Contesto: {e['context']}\nDomanda: {e['question']}\nRisposta:"


def formatting_prompts_func(examples, train_data):
    output_texts = []
    for i in tqdm(range(len(examples["question"]))):
        text = create_few_shot_example(examples[i], train_data, 2)

        output_texts.append(text)
    return output_texts


def format_squad(e):

    a = e["answers"]

    e["answer"] = a["text"][0]

    return e


def main():
    # arguments
    parser = argparse.ArgumentParser(description='Prompting neutral transformation experiments')
    parser.add_argument('--model_name', type=str)

    args = parser.parse_args()

    model_name = args.model_name

    llm = LLM(model=model_name, gpu_memory_utilization=0.7, max_model_len=2048, dtype="bfloat16", trust_remote_code=True)

    print(llm)

    # Load_dataset: GENTE
    ds = load_dataset("crux82/squad_it")

    ds = ds.map(format_squad)

    train_data = ds["train"].select(list(range(1000)))
    test_data = ds["test"]

    print(f"Train data lenght {len(train_data)}")
    print(f"Test data lenght {len(test_data)}")
    
    texts = formatting_prompts_func(test_data, train_data)

    context = [test_data["context"][i] for i in range(len(test_data["question"]))]
    question = [test_data["question"][i] for i in range(len(test_data["question"]))]
    answer = [test_data["answer"][i] for i in range(len(test_data["question"]))]

    sampling_params = SamplingParams(temperature=0, max_tokens=512)

    predicted = [
            out.outputs[0].text  # type: ignore
            for out in llm.generate(
                texts,
                sampling_params
            )  # type: ignore
        ]

    predicted = [p.split("\n\n")[0] for p in predicted]

    with open(f"/leonardo/home/userexternal/lmoroni0/__Work/minerva_sft/prompting_squad_examples/{model_name.split('/')[-4]}-{model_name.split('/')[-1]}.csv","w") as f:
        f.writelines(f"CONTEXT\tQUESTION\tPREDICTED\tANSWER\n")
        for c, q, p, a in zip(context, question, predicted, answer):
            f.writelines(f"{c}\t{q}\t{p}\t{a}\n")

    # some clearning trick
    del llm
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    results = rouge.compute(predictions=predicted, references=answer)

    with open(f"/leonardo/home/userexternal/lmoroni0/__Work/minerva_sft/prompting_squad_results/{model_name.split('/')[-4]}-{model_name.split('/')[-1]}.csv","w") as f:
        f.writelines(json.dumps(results))

    print(results)


if __name__ == "__main__":
    main()