from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# model_name = "/home/luca/llm-cva-tatent/adaptation/mistral_7B-adapted_3B_sgd_full_substitution"
model_name = "/home/luca/llm-cva-tatent/adaptation/llama-3_1-adapted_3B_sgd_full_substitution"
# model_name = "/home/luca/llm-cva-tatent/adaptation/llama-3_1-adapted_avg_substitution"


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.bfloat16)

    tokenizer.pad_token_id = tokenizer.unk_token_id

    prompt = """Mi chiamo Luca e adoro fare cose divertenti con"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("input token_ids:", inputs['input_ids'])

    outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs["attention_mask"], max_length=30)

    prompt_length = inputs['input_ids'].shape[1]

    generated_ids = outputs[0][prompt_length:]

    generated_answer = tokenizer.decode(generated_ids)

    print(f"generated ids {generated_ids}")
    print(f"generated answer {generated_answer}")
    

if __name__ == "__main__":
    main()