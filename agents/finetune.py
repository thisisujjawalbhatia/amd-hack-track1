#!/usr/bin/python3
"""
Fine-tune Q-Agent (Mistral-7B-Instruct-v0.3) and A-Agent (Qwen2.5-14B-Instruct)
using Unsloth LoRA. Saves adapters back into hf_models/.

Usage:
    # Fine-tune question model
    python -m agents.finetune --agent q

    # Fine-tune answer model
    python -m agents.finetune --agent a

    # Fine-tune both
    python -m agents.finetune --agent both

Prerequisites:
    - Run `python -m agents.build_finetune_data` first to generate training JSONL
    - Unsloth must be installed (pre-installed in container)
    - Models must exist in hf_models/
"""

import argparse
import json
import torch
from pathlib import Path
from datasets import Dataset

BASE_DIR = Path(__file__).parent.parent
HF_MODELS = BASE_DIR / "hf_models"
DATA_DIR = Path(__file__).parent / "data"


def load_jsonl(path: Path):
    """Load JSONL file into list of dicts."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_for_training(examples, tokenizer):
    """Convert chat messages to formatted text using the tokenizer's chat template."""
    texts = []
    for ex in examples:
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return texts


def finetune_q_agent():
    """Fine-tune Mistral-7B-Instruct-v0.3 for question generation."""
    from unsloth import FastLanguageModel

    model_path = str(HF_MODELS / "mistral_7b_base")
    output_path = str(HF_MODELS / "mistral_7b_base-qlora")

    print(f"Loading Q-Agent model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load training data
    data_path = DATA_DIR / "q_train.jsonl"
    if not data_path.exists():
        print("Training data not found. Run `python -m agents.build_finetune_data` first.")
        return

    raw_data = load_jsonl(data_path)
    texts = format_for_training(raw_data, tokenizer)
    dataset = Dataset.from_dict({"text": texts})

    print(f"Training Q-Agent on {len(dataset)} examples...")

    from trl import SFTTrainer
    from transformers import TrainingArguments

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=True,
        args=TrainingArguments(
            output_dir=output_path + "_checkpoints",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            save_steps=30,
            report_to="none",
        ),
    )

    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Q-Agent LoRA adapter saved to {output_path}")


def finetune_a_agent():
    """Fine-tune Qwen2.5-14B-Instruct for answer generation."""
    from unsloth import FastLanguageModel

    model_path = str(HF_MODELS / "qwen_14b_base")
    output_path = str(HF_MODELS / "qwen_14b_base-qlora")

    print(f"Loading A-Agent model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load training data
    data_path = DATA_DIR / "a_train.jsonl"
    if not data_path.exists():
        print("Training data not found. Run `python -m agents.build_finetune_data` first.")
        return

    raw_data = load_jsonl(data_path)
    texts = format_for_training(raw_data, tokenizer)
    dataset = Dataset.from_dict({"text": texts})

    print(f"Training A-Agent on {len(dataset)} examples...")

    from trl import SFTTrainer
    from transformers import TrainingArguments

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=True,
        args=TrainingArguments(
            output_dir=output_path + "_checkpoints",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=1e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            save_steps=30,
            report_to="none",
        ),
    )

    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"A-Agent LoRA adapter saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Q/A agents with LoRA")
    parser.add_argument(
        "--agent",
        type=str,
        choices=["q", "a", "both"],
        default="both",
        help="Which agent to fine-tune: q (question), a (answer), or both",
    )
    args = parser.parse_args()

    if args.agent in ("q", "both"):
        finetune_q_agent()
        # Free GPU memory before next model
        if args.agent == "both":
            torch.cuda.empty_cache()

    if args.agent in ("a", "both"):
        finetune_a_agent()

    print("Fine-tuning complete!")


if __name__ == "__main__":
    main()
