from transformers import AutoTokenizer
import random
import json

MODEL_NAME = "sapienzanlp/Minerva-3B-base-v1.0"
NUM_ANCHORS = 100
SAVE_ANCHORS_PATH = "/home/luca/llm-cva-tatent/similairities_anchors.json"

def compute_nearest_neighboor_cosine():
    t = AutoTokenizer.from_pretrained(MODEL_NAME)

    # get only prefix_tokens

    target_token_to_idx_prefix = {t: i for t, i in t.get_vocab().items() if "▁" in t} # vocabulary index of the helper model

    target_token_to_idx_noprefix = {t: i for t, i in t.get_vocab().items() if not "▁" in t and i > 259} # vocabulary index of the helper model

    random_tokens_prefix = random.sample(list(target_token_to_idx_prefix.keys()), 128)

    random_tokens_noprefix = random.sample(list(target_token_to_idx_noprefix.keys()), 128)

    selected_token_to_idx_prefix = {t: i for t, i in target_token_to_idx_prefix.items() if t in random_tokens_prefix}

    selected_token_to_idx_noprefix = {t: i for t, i in target_token_to_idx_noprefix.items() if t in random_tokens_noprefix}

    selected_token_to_idx_prefix.update(selected_token_to_idx_noprefix)

    with open(SAVE_ANCHORS_PATH, "w") as f:
        json.dump(selected_token_to_idx_prefix, f)

if __name__ == "__main__":
    compute_nearest_neighboor_cosine()