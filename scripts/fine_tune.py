#!/usr/bin/env python3
"""Fine-tune a LoRA-adapted Qwen model on HotPotQA SFT artifacts."""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

from arag import Config
from arag.data.hotpotqa import build_hotpotqa_artifacts
from arag.utils.device import format_device_message


def fine_tune(config: Config):
    data_paths = build_hotpotqa_artifacts(
        output_dir=config.get("data.prepared_dir", "data/hotpotqa"),
        seed=config.get("data.split_seed", 42),
        embed_sample_limit=config.get("data.embed_sample_limit"),
        question_sample_limit=config.get("data.question_sample_limit"),
    )

    print(format_device_message(config.get("runtime.device") or config.get("llm.device")))

    qwen_model_name = config.get("llm.model", "Qwen/Qwen3-4B-Instruct-2507")
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        qwen_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_dataset = load_dataset("json", data_files=data_paths["sft_train"], split="train")
    eval_dataset = load_dataset("json", data_files=data_paths["sft_val"], split="train")

    sft_config = SFTConfig(
        output_dir=config.get("training.output_dir", "outputs/qwen_hotpotqa_lora"),
        num_train_epochs=config.get("training.num_train_epochs", 1),
        learning_rate=config.get("training.learning_rate", 2e-5),
        per_device_train_batch_size=config.get("training.per_device_train_batch_size", 2),
        per_device_eval_batch_size=config.get("training.per_device_eval_batch_size", 2),
        gradient_accumulation_steps=config.get("training.gradient_accumulation_steps", 4),
        logging_steps=config.get("training.logging_steps", 100),
        eval_strategy="steps",
        eval_steps=config.get("training.eval_steps", 1000),
        save_strategy="steps",
        save_steps=config.get("training.save_steps", 1000),
        save_total_limit=config.get("training.save_total_limit", 2),
        load_best_model_at_end=config.get("training.load_best_model_at_end", True),
        dataset_text_field="text",
        packing=False,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(config.get("training.output_dir", "outputs/qwen_hotpotqa_lora"))
    tokenizer.save_pretrained(config.get("training.output_dir", "outputs/qwen_hotpotqa_lora"))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen LoRA on HotPotQA")
    parser.add_argument("--config", "-c", required=True, help="Config file path")
    args = parser.parse_args()
    fine_tune(Config.from_yaml(args.config))


if __name__ == "__main__":
    main()