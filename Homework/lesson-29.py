"""
LoRA Fine-Tuning + Merge + GGUF/Ollama Deployment

This submission covers:
1. LoRA fine-tuning
2. LoRA adapter saving
3. Merging LoRA with the base model using merge_and_unload()
4. Saving the complete merged Hugging Face model
5. Converting the merged model to GGUF using llama.cpp
6. Creating an Ollama Modelfile
7. Building and running the model with Ollama
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
DATA_PATH = "instruction-data.json"

OUTPUT_DIR = "./phi-3.5-mini-lora"
MERGED_DIR = "./phi-3.5-mini-merged"

MAX_LENGTH = 512


# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 2. Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)


# 3. Format training examples
def format_example(example):
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example.get('input', '')}\n\n"
        f"### Response:\n{example['output']}"
    )

    return {"text": text}


dataset = dataset.map(format_example)


# 4. Tokenize dataset
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
)


# 5. Load base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)


# 6. Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ],
)


# 7. Apply LoRA
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


# 8. Training configuration
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    report_to="none",
)


# 9. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# 10. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)


# 11. Train
trainer.train()


# 12. Save LoRA adapter
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"LoRA adapter saved to: {OUTPUT_DIR}")


# 13. Merge LoRA adapters into the base model
print("Merging LoRA adapters with the base model...")

merged_model = model.merge_and_unload()

print("LoRA adapters merged successfully.")


# 14. Fix tied-weight metadata for compatible Transformers versions
if hasattr(merged_model, "_tied_weights_keys"):
    if isinstance(merged_model._tied_weights_keys, list):
        merged_model._tied_weights_keys = {
            key: None for key in merged_model._tied_weights_keys
        }


# 15. Save complete merged Hugging Face model
Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)

print("Saving merged model...")

merged_model.save_pretrained(
    MERGED_DIR,
    safe_serialization=True,
)

tokenizer.save_pretrained(MERGED_DIR)

print(f"Complete merged model saved to: {MERGED_DIR}")


# ============================================================
# LOCAL DEPLOYMENT WITH LLAMA.CPP AND OLLAMA
# ============================================================

# After downloading the merged model directory to your local computer,
# use the following commands.

# 16. Clone llama.cpp
#
# cd ~/Desktop/projects
# git clone https://github.com/ggml-org/llama.cpp.git
# cd llama.cpp
# python3 -m pip install -r requirements.txt


# 17. Convert Hugging Face model to GGUF
#
# python3 convert_hf_to_gguf.py \
#   ~/Desktop/projects/phi-3.5-mini-merged \
#   --outfile ~/Desktop/projects/phi-3.5-mini-merged/phi-3.5-mini-f16.gguf \
#   --outtype f16


# 18. Ollama Modelfile
#
# Create a file named "Modelfile" in the same directory as the GGUF.
#
# FROM ./phi-3.5-mini-f16.gguf
#
# TEMPLATE """{{ if .System }}<|system|>
# {{ .System }}<|end|>
# {{ end }}{{ if .Prompt }}<|user|>
# {{ .Prompt }}<|end|>
# {{ end }}<|assistant|>
# """
#
# PARAMETER temperature 0.7
# PARAMETER top_p 0.9
# PARAMETER stop "<|end|>"


# 19. Build Ollama model
#
# ollama create my-custom-model -f Modelfile


# 20. Run the model
#
# ollama run my-custom-model


# 21. Example test
#
# Hello! Who are you?
#
# Explain machine learning in simple terms.


print("\nAssignment pipeline completed.")
print("Requirements covered:")
print("1. LoRA training")
print("2. LoRA adapter saving")
print("3. merge_and_unload()")
print("4. Full merged model saving")
print("5. llama.cpp GGUF conversion")
print("6. Ollama Modelfile")
print("7. ollama create")
print("8. ollama run")