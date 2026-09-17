import json
import torch

from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast, # type:ignore
    Trainer,
    TrainingArguments
)

from datasets import Dataset

## Configuration
MODEL_NAME = "openai-community/gpt2"
MAX_LENGTH = 512
OUTPUT_DIR = "./gpt2-instruct"

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

## Load instruction dataset
with open("/Users/mac/Desktop/Machine Learning/DL/Homework/instruction-data.json", "r", encoding="utf-8") as f:
    data=json.load(f)
    
print("Number of example:", len(data))
print("Sample data", data[:5])


## Format the conversations
def format_input(example):
    instruction=example["instruction"]
    input_text=example.get("input", "").strip()
    output=example["output"].strip()
    
    if input_text:
        user_text=f"{instruction}\n\n{input_text}"
    else:
        user_text=instruction
        
    text = (
        f"{USER_TOKEN}"
        f"{user_text}"
        f"{ASSISTANT_TOKEN}"
        f"{output}"
    )
    
    return {"text":text}

dataset=Dataset.from_list(
    [format_input(example) for example in data]
)

print("\nExample formatted conversation:")
print(dataset[0]["text"])


### Load GPT-2 model and Tokenizer
tokenizer=GPT2TokenizerFast.from_pretrained(MODEL_NAME)
model=GPT2LMHeadModel.from_pretrained(MODEL_NAME)

print("\nOriginal vocabulary size:", len(tokenizer))


## Add Special tokens
tokenizer.add_special_tokens({
    "pad_token":tokenizer.eos_token,
    "additional_special_tokens":[
        USER_TOKEN, ASSISTANT_TOKEN
    ]
})

print("New vocabulary size:", len(tokenizer))
print("Padding token:", tokenizer.pad_token)
print("User token ID:", tokenizer.convert_tokens_to_ids(USER_TOKEN))
print("Assistant token ID:", tokenizer.convert_tokens_to_ids(ASSISTANT_TOKEN))


### Resize GPT-2 embedding matrix
model.resize_token_embeddings(len(tokenizer))

print(
    "Model vocabulary size after resizing:",
    model.get_input_embeddings().weight.shape[0] #type: ignore
)


### Prepare tokenized inputs and labels
def prepare_input(example):
    text=example["text"]
    
    encoding=tokenizer(
        text, truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    
    input_ids=encoding["input_ids"]
    attention_mask=encoding["attention_mask"]
    
    labels=[-100]*len(input_ids)
    
    assistant_id=tokenizer.convert_tokens_to_ids(ASSISTANT_TOKEN)
    
    # Find the assistant token
    try:
        assistant_position=input_ids.index(assistant_id)
    except ValueError:
        return {
            "input_ids":input_ids,
            "attention_mask": attention_mask,
            "labels":labels
        }
        
    # Only calculate loss on assistant output tokens
    # User promt and the <|assistant|> token remain masked

    start=assistant_position+1
    
    for i in range(start, len(input_ids)):
        if attention_mask[i]==1:
            labels[i]=input_ids[i]
            
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels":labels
    }
        
tokenized_ds=dataset.map(
    prepare_input,
    remove_columns=["text"]
)

print("\nTokenization completed.")
print("Number of tokenized example:", len(tokenized_ds))

# Check label masking
print("\nFirst example label check:")

first_example=tokenized_ds[0]

print("Input IDs:", first_example["input_ids"][:30])
print("Labels:", first_example["labels"][:30])

print(
    "\nNumber of tokens used for loss:",
    sum(label !=-100 for label in first_example["labels"])
)


### Training configuration
training_args=TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    num_train_epochs=4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    
    learning_rate=5e-5,
    
    warmup_steps=200,
    lr_scheduler_type="cosine",
    
    optim="adamw_torch",
    
    logging_steps=50,
    save_strategy="epoch",
    
    report_to='none',
    
    fp16=False  
)

## Create Trainer
trainer=Trainer(
    model=model, 
    args=training_args,
    train_dataset=tokenized_ds,
    processing_class=tokenizer
)

# Start training
print("\nStart instruction fine-tuning...")

trainer.train()


## Save model and tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining completed.")
print("Model saved to:", OUTPUT_DIR)
