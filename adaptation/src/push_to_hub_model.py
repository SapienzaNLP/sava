from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse


def main(args):
    local_folder = args.local_folder
    hf_model_name = args.hf_model_name

    model = AutoModelForCausalLM.from_pretrained(local_folder, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(local_folder)

    model.push_to_hub(hf_model_name, private=True)
    tokenizer.push_to_hub(hf_model_name, private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_folder",
        type=str,
        required=True,
        help="The path of local folder where the model is saved."
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        required=True,
        help="The name of remote repository to push the model."
    )
    
    args = parser.parse_args()
    main(args)