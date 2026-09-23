import json
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

from peft import LoraConfig, get_peft_model


# Configuration

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
DATA_PATH = "instruction-data.json"
OUTPUT_DIR = "./phi-3.5-mini-lora"

MAX_LENGTH = 512


# Device

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using Device:", device)


# Load Tokenizer

print("\nLoading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Load Instruction Dataset

print("\nLoading Instruction Dataset...")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Number of examples:", len(data))

dataset = Dataset.from_list(data)

print(dataset)


# Chat Template Preprocessing

def format_example(example):

    messages = [
        {
            "role": "user",
            "content": example["instruction"]
        }
    ]

    if example.get("input"):
        messages[0]["content"] += "\n" + example["input"]

    messages.append(
        {
            "role": "assistant",
            "content": example["output"]
        }
    )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {
        "text": text
    }


print("\nApplying chat template...")

dataset = dataset.map(format_example)

print("\nExample formatted text:")
print(dataset[0]["text"])


# Tokenization

def tokenize_function(example):

    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens


print("\nTokenizing Dataset...")

tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names
)

print(tokenized_dataset)


# Load Pretrained Model

print("\nLoading pretrained model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

model.config.pad_token_id = tokenizer.pad_token_id


# Parameters Before LoRA

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params_before = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nBefore LoRA:")
print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params_before)


# LoRA Configuration

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    target_modules=[
        "qkv_proj",
        "o_proj"
    ],

    bias="none",
    task_type="CAUSAL_LM"
)


# Apply LoRA

print("\nApplying LoRA...")

model = get_peft_model(
    model,
    lora_config
)

print("\nLoRA parameter statistics:")

model.print_trainable_parameters()


# Data Collator

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)


# Training Arguments

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    logging_steps=10,

    save_strategy="epoch",

    fp16=True if device == "cuda" else False,

    report_to="none",

    remove_unused_columns=False
)


# Trainer

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_dataset,

    data_collator=data_collator
)


# Train

print("\nStarting LoRA fine-tuning...")

trainer.train()


# Save LoRA Adapter

print("\nSaving LoRA adapter...")

model.save_pretrained(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\nLoRA adapter saved to:")
print(OUTPUT_DIR)