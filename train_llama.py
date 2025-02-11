import json
import os
import argparse
import logging
from pathlib import Path

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

from models.llama_modeling import LlamaForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IB_DISABLE'] = '1'

max_seq_len = 8192
train_batch_size = 2
eval_batch_size = 2
epochs = 1
out_model_path = './output'
model_name_or_path = 'meta-llama/Meta-Llama-3-8B'
learning_rate = 1e-4  # 1e-5, 1e-4, 5e-4

def load_and_tokenize_dataset(dataset_name, tokenizer, train_num):
    if(dataset_name == "wiki"):
        train_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        validation_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding='max_length',
                truncation=True,
                max_length=max_seq_len,
            )
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns='text')
        tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns='text')

    else if(dataset_name == "ptb"):
        train_dataset = load_dataset("ptb_text_only", "penn_treebank", split="train")
        validation_dataset = load_dataset("ptb_text_only", "penn_treebank", split="validation")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        def tokenize_function(examples):
            return tokenizer(
                examples["sentence"],
                padding='max_length',
                truncation=True,
                max_length=max_seq_len,
            )
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns='sentence')
        tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns='sentence')

    else if(dataset_name == "c4"):
        train_dataset = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        validation_dataset = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding='max_length',
                truncation=True,
                max_length=max_seq_len,
            )
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns='text')
        tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns='text')

    return tokenized_train_dataset, tokenized_validation_dataset

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_train_dataset, tokenized_validation_dataset = load_and_tokenize_dataset(args.dataset, tokenizer, args.train_num)

    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    model.add_adapter(lora_config)

    latest_checkpoint = None
    if os.path.exists(args.out_model_path):
        checkpoints = [d for d in os.listdir(args.out_model_path) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = os.path.join(args.out_model_path, max(checkpoints, key=lambda x: int(x.split("-")[1])))

    training_args = TrainingArguments(
        output_dir=args.out_model_path,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        fp16=False,
        bf16=False,
        weight_decay=0.01,
        save_total_limit=1,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        group_by_length=True,
        max_steps=-1,
        save_steps=1000,
        logging_steps=200,
        resume_from_checkpoint=latest_checkpoint,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(args.out_model_path)

    eval_results = trainer.evaluate()
    logger.info(f"eval_loss: {eval_results['eval_loss']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wiki", choices=["wiki", "ptb", "c4"])
    parser.add_argument("--model_name_or_path", type=str, default=model_name_or_path, help="Hugging Face model name or local path.")
    parser.add_argument("--out_model_path", type=str, default=out_model_path)
    parser.add_argument("--learning_rate", type=float, default=learning_rate)
    parser.add_argument("--train_batch_size", type=int, default=train_batch_size)
    parser.add_argument("--eval_batch_size", type=int, default=eval_batch_size)
    parser.add_argument("--epochs", type=int, default=epochs)
    args = parser.parse_args()

    main(args)