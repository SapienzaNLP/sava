from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import nltk
import argparse

nltk.download('punkt')

LANG_LONG = {"it": "italian",
            "en": "english"}

LIMIT_DOCS = 50_000

def main(args):
    tokenizer_name = args.tokenizer_name
    lang = args.lang
    dataset = args.dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if dataset == "CULTURAX":
        DS_STREAM = load_dataset('uonlp/CulturaX', lang, split='train', streaming=True)
    elif dataset == "WIKIPEDIA":
        DS_STREAM = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split='train', streaming=True)

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

    print(f"{tokenizer_name} - FERTILITY over {LIMIT_DOCS} over {dataset} {LANG_LONG[lang]}: {fertility}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Fertility',
                    description='Compute fertility of a given tokenizer for a predefined dataset.')    
    parser.add_argument('-t', '--tokenizer_name')
    parser.add_argument('-l', '--lang')
    parser.add_argument('-d', '--dataset')
    args = parser.parse_args()

    main(args)