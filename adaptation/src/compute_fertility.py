from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import nltk

nltk.download('punkt')

# TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"
# TOKENIZER_NAME = "google/gemma-7b"
# TOKENIZER_NAME = "sapienzanlp/minestral-1B-100B_it-100B_en-cx-04032024"
# TOKENIZER_NAME = "meta-llama/Meta-Llama-3-8B"
# TOKENIZER_NAME = "/home/luca/llm-cva-tatent/adaptation/mistral_7B-adapted_3B_sgd_8k_substitution"
TOKENIZER_NAME = "sapienzanlp/Minerva-7B-base-v1.0"


LANG = "it"
LANG_LONG = "italian"
LIMIT_DOCS = 50_000
# DS = "CULTURAX"
DS = "WIKIPEDIA"

def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    if DS == "CULTURAX":
        DS_STREAM = load_dataset('uonlp/CulturaX', LANG, split='train', streaming=True)
    else:
        DS_STREAM = load_dataset("wikimedia/wikipedia", f"20231101.{LANG}", split='train', streaming=True)

    fertility = 0

    for it, sample in enumerate(tqdm(DS_STREAM)):
        if it == LIMIT_DOCS:
            break

        if it > 0 and it % 10_000 == 0:
            print(f"Runtime fertility: {fertility / it}")

        sample_text = sample["text"]

        words = word_tokenize(sample_text, language=LANG_LONG)
        tokens_id = tokenizer(sample_text, add_special_tokens=False).input_ids

        sample_fertility = len(tokens_id) / len(words)
        fertility += sample_fertility

    fertility = fertility / LIMIT_DOCS

    print(f"{TOKENIZER_NAME} - FERTILITY over {LIMIT_DOCS} over {DS} {LANG_LONG}: {fertility}")


if __name__ == "__main__":
    main()