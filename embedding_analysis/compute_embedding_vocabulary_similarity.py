from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import numpy as np
import torch
import faiss
import json
from sklearn.metrics.pairwise import cosine_similarity

#MODEL_1_NAME = "SemanticAlignment/mistral_7B-adapted_3B_sgd_full_substitution"
#MODEL_2_NAME = "sapienzanlp/Minerva-3B-base-v1.0"
#PATH_TO_SAVE = "/home/luca/llm-cva-tatent/similarities"

def compute_nearest_neighboor_cosine(embedding_matrix, num_neigh=11):
    print("compute cosine similarity")
    sims = cosine_similarity(embedding_matrix)

    print("computed similairity")
    indices = np.argsort(-sims, axis=1)

    return indices[:, :num_neigh]


def compute_nearest_neighboor(embedding_matrix, num_neigh=11):
    print("train")
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])

    index.add(embedding_matrix)

    print("search")

    _, neighbors = index.search(embedding_matrix, num_neigh)

    return neighbors


def intersection(lst1, lst2):

    return len(list(set(lst1) & set(lst2)))


def compute_similarity(indices_1, indices_2, num_neigh):
    similairities = []

    for i in range(indices_1.shape[0]):
        sim = intersection(indices_1[i][1:], indices_2[i][1:]) / (num_neigh - 1)

        similairities.append(sim)

    return np.mean(similairities), np.std(similairities)


def main(args):

    # parameters
    offset = 259
    num_neigh = int(args.num_neigh) + 1 # an element A is 1nn of itself
    model_1_name = args.model_1
    model_2_name = args.model_2
    output_dir = args.output_dir
    rev_1 = args.rev_1
    rev_2 = args.rev_2
    cosine = args.cosine

    if rev_1:
        model_1 = AutoModelForCausalLM.from_pretrained(model_1_name, revision=rev_1)
    else:
        model_1 = AutoModelForCausalLM.from_pretrained(model_1_name)
    #tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_1_NAME)

    if rev_2:
        model_2 = AutoModelForCausalLM.from_pretrained(model_2_name, revision=rev_2)
    else:
        model_2 = AutoModelForCausalLM.from_pretrained(model_2_name)

    model_1_embedding = model_1.get_input_embeddings().weight.detach().numpy()
    #tokens_1 = {i: t for t,i in tokenizer_1.get_vocab().items()}
    model_2_embedding = model_2.get_input_embeddings().weight.detach().numpy()

    model_1_embedding = model_1_embedding[offset:] # remove special tokens
    model_2_embedding = model_2_embedding[offset:] # remove special tokens

    del model_1
    del model_2

    if cosine:
        indices_1 = compute_nearest_neighboor_cosine(model_1_embedding, num_neigh=num_neigh)
    else:
        indices_1 = compute_nearest_neighboor(model_1_embedding, num_neigh=num_neigh)
    
    if cosine:
        indices_2 = compute_nearest_neighboor_cosine(model_2_embedding, num_neigh=num_neigh)
    else:
        indices_2 = compute_nearest_neighboor(model_2_embedding, num_neigh=num_neigh)

    #compute similarity between the two embeddings
    similairity_mean, similairity_std = compute_similarity(indices_1, indices_2, num_neigh=num_neigh)

    # save into a file the similarityscore of the two models

    output_file_name = output_dir+f"/{model_1_name.split('/')[-1]}"

    if rev_1:
        output_file_name += f"_{rev_1}"
    
    output_file_name += f"+{model_2_name.split('/')[-1]}"

    if rev_2:
        output_file_name += f"_{rev_2}"

    if cosine:
        output_file_name += "_cosine_similairity"

    output_file_name += f"{args.num_neigh}.jsonl"

    with open(output_file_name, "w") as f:
        f.writelines(json.dumps({"similarity_mean": similairity_mean, "similarity_std": similairity_std}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Similariter',
                    description='Compute embedding similarities between two different models')    
    parser.add_argument('-m1', '--model_1')
    parser.add_argument('-r1', '--rev_1', required=False)
    parser.add_argument('-m2', '--model_2')
    parser.add_argument('-r2', '--rev_2', required=False)
    parser.add_argument('-n', '--num_neigh')
    parser.add_argument('-o', '--output_dir')
    parser.add_argument('-c', '--cosine', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    main(args)