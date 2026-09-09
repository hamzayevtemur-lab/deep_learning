import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from unidecode import unidecode

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

torch.manual_seed(42)
random.seed(42)

# Read Dataset

DATA_DIR = "/Users/mac/Desktop/Machine Learning/DL/DB/Names2"

X_names = []
Y_labels = []

def replace(text, chars, target):
    for char in chars:
        text=text.replace(char, target)
        
    return text

for file in sorted(os.listdir(DATA_DIR)):
    if not file.endswith(".txt"):
        continue
    
    label=os.path.splitext(file)[0]
    path=os.path.join(DATA_DIR, file)
    
    with open(path, encoding="utf-8") as f:
        for line in f:
            name=line.strip().lower()
            name=unidecode(name)
            
            if name=="":
                continue
            
            name = replace(name, [",", "1", "/", ":"], "")
            name = replace(name, ["-"], " ")
            
            X_names.append(name)
            Y_labels.append(label)
            

print(f"Total Samples: {len(X_names)}")
print(f"Number of Classes: {len(set(Y_labels))}") 


# Create Character Vocabulary
pad_token="<pad>"

unique_chars=sorted(set("".join(X_names))) 

idx2char=[pad_token]+unique_chars
char2idx={
    char:idx
    for idx, char in enumerate(idx2char)
}   

pad_idx=char2idx[pad_token]

print(f"Vocabulary Size: {len(idx2char)}")
print(f"Padding Index: {pad_idx}")  

# Encode Names
def encode(name):
    return [char2idx[ch] for ch in name]

X=[encode(name) for name in X_names]

# Encode labels
unique_labels=sorted(set(Y_labels))

label2idx={
    label:idx
    for idx, label in enumerate(unique_labels)
}

Y=[label2idx[label] for label in Y_labels]

print(f"Labels: {label2idx}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y,
    shuffle=True
)

print("Data Split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# Custom Dataset
class NamesDataset(Dataset):
    def __init__(self, X, Y):
        self.X=X
        self.Y=Y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
train_dataset = NamesDataset(X_train, y_train)
test_dataset = NamesDataset(X_test, y_test)


print("Dataset Sizes:")
print(f"Train Dataset: {len(train_dataset)}")
print(f"Test Dataset:  {len(test_dataset)}")   


## Custom Collate Function
def collate_fn(batch):
    names = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    max_length = max(len(name) for name in names)

    padded_names = torch.full(
        (len(names), max_length),
        pad_idx,
        dtype=torch.long
    )

    for i, name in enumerate(names):
        padded_names[i, :len(name)] = torch.tensor(
            name,
            dtype=torch.long
        )

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    return padded_names, labels
    
    
print("Checking encoded data:")
print("X[0]:", X[0])
print("Type of X[0]:", type(X[0]))
print("Type of X[0][0]:", type(X[0][0]))

# DataLoaders
BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)



# Verify Train DataLoader
print("Train DataLoader:")
print(f"Number of batches: {len(train_loader)}")

x_train_batch, y_train_batch = next(iter(train_loader))

print(f"Input Shape:  {x_train_batch.shape}")
print(f"Labels Shape: {y_train_batch.shape}")
print(f"First Encoded Name: {x_train_batch[0]}")
print(f"First Label: {y_train_batch[0]}")

# Verify Test DataLoader
print("Test DataLoader:")
print(f"Number of batches: {len(test_loader)}")

x_test_batch, y_test_batch = next(iter(test_loader))
print(f"Input Shape:  {x_test_batch.shape}")
print(f"Labels Shape: {y_test_batch.shape}")
print(f"First Encoded Name: {x_test_batch[0]}")
print(f"First Label: {y_test_batch[0]}")