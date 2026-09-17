import json
import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# 1. Configuration & Hyperparameters
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
MAX_LENGTH = 512
OUTPUT_DIR = "./phi-3.5-mini-instruct"

# Fallback path logic to support running from any working directory
DATA_PATH = "/Users/mac/Desktop/Machine Learning/DL/Homework/instruction-data.json"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "instruction-data.json"


# 2. Load Instruction Dataset
print("Loading instruction dataset...")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    
print("Number of examples:", len(data))
print("First example:")
print(data[0])


# 3. Load Tokenizer & Model
print("\nLoading tokenizer...")
# Phi-3 / Phi-3.5 models require trust_remote_code=True for custom tokenization logic
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("Tokenizer vocabulary size:", len(tokenizer))

print("\nLoading model...")
# Load in float16 precision and map automatically across available hardware (MPS / CUDA / CPU)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded successfully.")
print("Total number of parameters:", sum(p.numel() for p in model.parameters()))
print("Model device:", model.device)



# 4. Format Conversations into Chat Template Schema
def format_conversation(example):
    """
    Converts raw instruction-input-output dicts into standardized
    chat message dictionaries with 'user' and 'assistant' roles.
    """
    instruction = example["instruction"].strip()
    input_text = example.get("input", "").strip()
    output = example["output"].strip()
    
    # If additional input context is provided, combine with instruction
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction
        
    messages = [
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": output
        }
    ]
    
    return {"messages": messages}

# Convert python list to Hugging Face Dataset
dataset = Dataset.from_list(
    [format_conversation(example) for example in data]
)

print("\nFormatted conversation example:")
print(dataset[0]["messages"])


# 5. Tokenization & Prompt Loss Masking
def prepare_input(example):
    """
    Applies Phi-3's official chat template, tokenizes sequences,
    and sets label values of user prompt tokens to -100 so that
    loss is computed ONLY on the assistant's response tokens.
    """
    messages = example["messages"]
    
    # Full conversation rendered into chat template string
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    
    # Render only the user prompt prefix (with generation prompt appended)
    user_messages = [messages[0]]
    prompt_text = tokenizer.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize full conversation without static padding (dynamic padding in collator)
    full_encoding = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )
    
    # Tokenize user prompt to find its exact token length
    prompt_encoding = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )
    
    input_ids = full_encoding["input_ids"]
    attention_mask = full_encoding["attention_mask"]
    
    prompt_length = min(
        len(prompt_encoding["input_ids"]),
        len(input_ids)
    )
    
    # Create target labels: duplicate input_ids, then mask prompt tokens with -100
    labels = input_ids.copy()
    for i in range(prompt_length):
        labels[i] = -100
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

print("\nTokenizing dataset...")

tokenized_ds = dataset.map(
    prepare_input,
    remove_columns=["messages"]
)

print("Tokenization completed.")
print("Number of tokenized examples:", len(tokenized_ds))

# Inspect label masking on the first sample
first_example = tokenized_ds[0]
print("\nFirst example verification:")
print("Input IDs (first 30):", first_example["input_ids"][:30])
print("Labels (first 30):   ", first_example["labels"][:30])
print(
    "Number of active target tokens used for loss computation:",
    sum(label != -100 for label in first_example["labels"])
)



# 6. Data Collator & Training Setup
# DataCollatorForSeq2Seq pads inputs and masks padding tokens in labels with -100 dynamically per batch
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
    return_tensors="pt"
)

# Training configuration with gradient accumulation for memory optimization
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    fp16=True,
    remove_unused_columns=False
)

# Initialize Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=data_collator,
    processing_class=tokenizer
)


# 7. Model Training & Checkpoint Saving
print("\nStarting instruction fine-tuning...")
trainer.train()

print("\nSaving fine-tuned model and tokenizer...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining completed successfully.")
print("Model saved to:", OUTPUT_DIR)
