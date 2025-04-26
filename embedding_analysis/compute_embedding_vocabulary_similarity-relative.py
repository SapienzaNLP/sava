from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import numpy as np

import json
from numpy import dot
from numpy.linalg import norm


def get_anchors_idx(anchor_path):
    with open(anchor_path, "r") as f:
        anchor_idx = json.load(f)

    return anchor_idx


def map_anchors_to_embedding(model_embedding, anchors_idx):
    print("map anchors...")
    output = []

    for _, i in anchors_idx.items():
        output.append(model_embedding[i])

    return np.array(output)


def compute_similarity(model_1_positioning, model_2_positioning):
    print("compute similarities...")
    similairities = []

    for i in range(model_1_positioning.shape[0]):
        a = model_1_positioning[i]
        b = model_2_positioning[i]
        cos_sim = dot(a, b)/(norm(a)*norm(b))

        if cos_sim == np.nan:
            print("error")

        similairities.append(cos_sim)

    return np.nanmean(similairities), np.nanstd(similairities)


def get_relative_positioning(model_embedding, model_anchors):
    print("get relative positions...")
    return model_embedding @ model_anchors.T


def main(args):

    # parameters
    offset = 259
    model_1_name = args.model_1
    model_2_name = args.model_2
    output_dir = args.output_dir
    rev_1 = args.rev_1
    rev_2 = args.rev_2
    anchor_path = args.anchor_path

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

    del model_1
    del model_2

    anchors_idx = get_anchors_idx(anchor_path)

    model_1_anchors = map_anchors_to_embedding(model_1_embedding, anchors_idx)
    model_2_anchors = map_anchors_to_embedding(model_2_embedding, anchors_idx)

    model_1_embedding = model_1_embedding[offset:] # remove special tokens
    model_2_embedding = model_2_embedding[offset:] # remove special tokens

    model_1_relative_positioning = get_relative_positioning(model_1_embedding, model_1_anchors)
    model_2_relative_positioning = get_relative_positioning(model_2_embedding, model_2_anchors)

    #compute similarity between the two embeddings
    similairity_mean, similairity_std = compute_similarity(model_1_relative_positioning, model_2_relative_positioning)

    print("saving files...")

    # save into a file the similarityscore of the two models

    output_file_name = output_dir+f"/{model_1_name.split('/')[-1]}"

    if rev_1:
        output_file_name += f"_{rev_1}"
    
    output_file_name += f"+{model_2_name.split('/')[-1]}"

    if rev_2:
        output_file_name += f"_{rev_2}"

    output_file_name += f".jsonl"

    with open(output_file_name, "w") as f:
        f.writelines(json.dumps({"similarity_mean": str(similairity_mean), "similarity_std": str(similairity_std)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog="Compute the similarity between two models' embedding space. Use a precomputed set of anchor tokens.",
                    description='')    
    parser.add_argument('-m1', '--model_1')
    parser.add_argument('-r1', '--rev_1', required=False)
    parser.add_argument('-m2', '--model_2')
    parser.add_argument('-r2', '--rev_2', required=False)
    parser.add_argument('-o', '--output_dir')
    parser.add_argument('-a', '--anchor_path')
    args = parser.parse_args()

    main(args)