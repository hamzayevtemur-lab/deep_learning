# ==============================================================================
# Lesson 25: Emotion Classification using BERT (PyTorch & Hugging Face)
# ==============================================================================
# Objective:
#   1. Load and preprocess the SMILE Twitter Emotion dataset.
#   2. Filter relevant emotion categories (happy, angry, sad, surprise, disgust).
#   3. Create custom PyTorch Dataset & DataLoader objects.
#   4. Fine-tune pre-trained BERT (`bert-base-uncased`) for sequence classification.
#   5. Train, evaluate, and save the best model based on weighted F1 score.
#   6. Upload the fine-tuned model and tokenizer to Hugging Face Hub.
# ==============================================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)

from huggingface_hub import login

# ------------------------------------------------------------------------------
# 1. Configuration & Hyperparameters
# ------------------------------------------------------------------------------
DATA_PATH = "/Users/mac/Desktop/Machine Learning/DL/Homework/smile-annotations-final.csv"
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./bert-smile-emotion"
HF_MODEL_NAME = "TemurbekHamzaev/bert-smile-emotion"
RANDOM_STATE = 42

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_LENGTH = 128

# ------------------------------------------------------------------------------
# 2. Hardware Acceleration & Reproducibility Setup
# ------------------------------------------------------------------------------
# Set device hierarchy: Apple Silicon MPS -> CUDA GPU -> CPU
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

# Ensure reproducibility across runs
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# ------------------------------------------------------------------------------
# 3. Data Loading & Preprocessing
# ------------------------------------------------------------------------------
print("\nLoading Smile Twitter Emotion dataset...")

# Load raw CSV (No header in raw file, assigning column names)
df = pd.read_csv(
    DATA_PATH,
    names=["id", "text", "label"],
    header=None
) 

# Keep only necessary columns
df = df[["text", "label"]]

# Drop missing values
df = df.dropna(subset=["text", "label"])

# Ensure text and label are strings
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(str)

print("Dataset Shape:", df.shape)
print("\nRaw emotion distribution:")
print(df["label"].value_counts())

# Define target emotion subset to filter
VALID_EMOTIONS = [
    "happy",
    "angry",
    "sad",
    "surprise",
    "disgust"
]

# Filter dataset to include only target emotions
df = df[df["label"].isin(VALID_EMOTIONS)].copy()

print("\nFiltered dataset shape:", df.shape)
print("\nEmotion distribution after filtering:")
print(df["label"].value_counts())

# Build label encoding mappings (String -> Integer & Integer -> String)
labels = sorted(df["label"].unique())
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

# Map string labels to numeric integers
df["label"] = df["label"].map(label2id)

num_labels = len(labels)
print("\nLabel mapping:", label2id)
print("Number of classes:", num_labels)

# ------------------------------------------------------------------------------
# 4. Train / Validation Split
# ------------------------------------------------------------------------------
# Stratified split to maintain class balance in train and validation sets
train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=RANDOM_STATE,
    stratify=df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("\nDataset split:")
print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))

# ------------------------------------------------------------------------------
# 5. Tokenizer & Dataset Initialization
# ------------------------------------------------------------------------------
print("\nLoading BERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class EmotionDataset(Dataset):
    """
    Custom PyTorch Dataset for text classification.
    Tokenizes raw text samples on the fly during training/evaluation.
    """
    def __init__(self, dataframe, tokenizer, max_length):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        
        # Tokenize text sequence (Fixed typo: max_length instead of max_lenght)
        encoding = self.tokenizer(
            text, 
            truncation=True,
            max_length=self.max_length
        )
        
        item = {
            "input_ids": encoding['input_ids'],
            "attention_mask": encoding['attention_mask'],
            "labels": label
        }
        
        return item


train_dataset = EmotionDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset = EmotionDataset(val_df, tokenizer, MAX_LENGTH)

# DataCollator automatically handles dynamic padding within each mini-batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator
)

# ------------------------------------------------------------------------------
# 6. Model Initialization & Optimization Setup
# ------------------------------------------------------------------------------
print("\nLoading pre-trained BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label
)

model.to(device)

# AdamW optimizer with L2 weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

# Calculate total optimization steps & warmup schedule
total_training_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_training_steps * 0.1)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps
)

# ------------------------------------------------------------------------------
# 7. Training & Evaluation Functions
# ------------------------------------------------------------------------------
def train_one_epoch(model, data_loader, optimizer, scheduler, device):
    """
    Trains the model for one epoch over the dataset.
    """
    model.train()
    
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (computes cross-entropy loss automatically when labels are passed)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_batch
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Get index of maximum logit value (predicted class index)
        predictions = torch.argmax(logits, dim=-1)
        
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels_batch.detach().cpu().numpy())
        
    average_loss = total_loss / len(data_loader)
        
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # Fixed bug: returning computed variable 'accuracy' instead of function reference 'accuracy_score'
    return average_loss, accuracy, f1


def evaluate(model, data_loader, device):
    """
    Evaluates model performance on the validation dataset.
    """
    model.eval()
    
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_batch
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            
    average_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    
    return average_loss, accuracy, f1

# ------------------------------------------------------------------------------
# 8. Main Training Loop
# ------------------------------------------------------------------------------
print("\nStarting BERT fine-tuning...")

best_f1 = 0.0

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}") 
    
    train_loss, train_accuracy, train_f1 = train_one_epoch(
        model, 
        train_loader, 
        optimizer, 
        scheduler, 
        device
    )
          
    val_loss, val_accuracy, val_f1 = evaluate(
        model, 
        val_loader,
        device
    )
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Train F1: {train_f1:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation F1: {val_f1:.4f}")
    
    # Save model weights if validation F1 score improves
    if val_f1 > best_f1:
        best_f1 = val_f1
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print("Best Model Saved")

print("\nTraining Completed.")
print(f"Best validation F1: {best_f1:.4f}")

# ------------------------------------------------------------------------------
# 9. Load Best Saved Model & Final Evaluation
# ------------------------------------------------------------------------------
print("\nLoading best model for final evaluation...")

model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
model.to(device)

print("\nFinal evaluation...")
val_loss, val_accuracy, val_f1 = evaluate(model, val_loader, device)

print("\nFinal validation results:")
print(f"Loss: {val_loss:.4f}")
print(f"Accuracy: {val_accuracy:.4f}")
print(f"F1-score: {val_f1:.4f}")

# ------------------------------------------------------------------------------
# 10. Hugging Face Hub Upload
# ------------------------------------------------------------------------------
print("\nLogging into Hugging Face...")
# Note: Ensure you are logged in via `huggingface-cli login` or supply a token parameter `login(token="...")`
login()

print("\nUploading model to Hugging Face Hub...")
model.push_to_hub(HF_MODEL_NAME)
tokenizer.push_to_hub(HF_MODEL_NAME)

print("\nModel successfully uploaded.")
print(f"https://huggingface.co/{HF_MODEL_NAME}")