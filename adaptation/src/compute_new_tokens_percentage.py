from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

#TOKENIZER_NAME = "SemanticAlignment/mistral_7B-adapted_random_union"
TOKENIZER_NAME = "SemanticAlignment/mistral_7B-adapted_3B_sgd_5k"
TOKENIZER_NAME = "SemanticAlignment/mistral_7B-adapted_350M_clp"

LANG = "en"
LANG_LONG = "english"
LIMIT_DOCS = 10_000
DS = "CULTURAX"
# DS = "WIKIPEDIA"

def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    if DS == "CULTURAX":
        DS_STREAM = load_dataset('uonlp/CulturaX', LANG, split='train', streaming=True)
    else:
        DS_STREAM = load_dataset("wikimedia/wikipedia", f"20231101.{LANG}", split='train', streaming=True)

    percentage_new_tokens = 0

    for it, sample in enumerate(tqdm(DS_STREAM)):
        if it == LIMIT_DOCS:
            break

        if it > 0 and it % 10_000 == 0:
            print(f"Runtime new tokens: {percentage_new_tokens / (it + 1)}")

        sample_text = sample["text"]

        tokens_id = tokenizer(sample_text, add_special_tokens=False).input_ids

        sample_percentage = np.sum(np.array(tokens_id) >= 32000) / len(tokens_id)
        percentage_new_tokens += sample_percentage

    percentage_new_tokens = percentage_new_tokens / LIMIT_DOCS

    print(f"{TOKENIZER_NAME} - percentage new tokens over {LIMIT_DOCS} over {DS} {LANG_LONG}: {percentage_new_tokens}")


if __name__ == "__main__":
    main()