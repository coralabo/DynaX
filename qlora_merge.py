import os
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def merge_lora_to_LLM(model_name_or_path, adapter_name_or_path, save_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        model = PeftModel.from_pretrained(model, adapter_name_or_path)
        model = model.merge_and_unload()
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)

        print(f"Model and tokenizer saved to {save_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with a pre-trained language model.")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B", 
                        help="Hugging Face model name or local path.")
    parser.add_argument("--adapter_name_or_path", type=str, default="output", 
                        help="Path to the LoRA adapter.")
    parser.add_argument("--save_path", type=str, default="newModels", 
                        help="Path to save the merged model.")

    args = parser.parse_args()

    print("Merging beginning...")
    merge_lora_to_LLM(args.model_name_or_path, args.adapter_name_or_path, args.save_path)
    print("Merge over.")