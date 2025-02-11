import os
import argparse
import torch
from transformers import AutoTokenizer
from models.llama_modeling import LlamaForCausalLM
from dataUtil import perplexity

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(model_name_or_path, dataset, max_seq_len):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        ppl = perplexity(model, tokenizer, max_seq_len, dataset)
        print(f'Perplexity on {dataset}: {ppl.item()}')

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate perplexity of a language model on a dataset.")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Hugging Face model name or local path.")
    parser.add_argument("--dataset", type=str, default="ptb",
                        choices=["wiki", "c4", "ptb"],
                        help="Dataset to evaluate perplexity on.")
    parser.add_argument("--max_seq_len", type=int, default=4096,
                        help="Maximum sequence length.")

    args = parser.parse_args()

    main(args.model_name_or_path, args.dataset, args.max_seq_len)