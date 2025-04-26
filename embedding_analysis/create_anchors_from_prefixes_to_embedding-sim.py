from transformers import AutoTokenizer
import random
import json
import argparse

def save_anchors(args):
    model_name = args.model_name
    num_anchors = args.num_anchors
    output_path = args.output_path

    t = AutoTokenizer.from_pretrained(model_name)

    # get only prefix_tokens

    target_token_to_idx_prefix = {t: i for t, i in t.get_vocab().items() if "▁" in t} # vocabulary index of the helper model

    target_token_to_idx_noprefix = {t: i for t, i in t.get_vocab().items() if not "▁" in t and i > 259} # vocabulary index of the helper model

    random_tokens_prefix = random.sample(list(target_token_to_idx_prefix.keys()), num_anchors)

    random_tokens_noprefix = random.sample(list(target_token_to_idx_noprefix.keys()), num_anchors)

    selected_token_to_idx_prefix = {t: i for t, i in target_token_to_idx_prefix.items() if t in random_tokens_prefix}

    selected_token_to_idx_noprefix = {t: i for t, i in target_token_to_idx_noprefix.items() if t in random_tokens_noprefix}

    selected_token_to_idx_prefix.update(selected_token_to_idx_noprefix)

    with open(output_path, "w") as f:
        json.dump(selected_token_to_idx_prefix, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Save Anchors',
                    description='Select anchors tokens and save locally.\nThis script will select num_anchors random prefix tokens and num_tokens random non-prefix tokens, from the vocabulary of a given model.')
    parser.add_argument('-m', '--model_name')
    parser.add_argument('-o', '--output_path')
    parser.add_argument('-n', '--num_anchors')
    args = parser.parse_args()

    save_anchors(args)