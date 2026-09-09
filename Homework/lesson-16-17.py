import os
import random

import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from unidecode import unidecode

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

## Read Dataset
DATA_DIR="/Users/mac/Desktop/Machine Learning/DL/DB/Names2"

X_names=[]
Y_labels=[]

def replace(name, chars, target):
    for char in chars:
        name = name.replace(char, target)
    return name

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
            
            name = replace(name, [",", '1','/',  ":"], '')
            name = replace(name, ['-'], ' ')
            
            X_names.append(name)
            Y_labels.append(label)


print(f"Total Samples : {len(X_names)}")
print(f"Number of Classes : {len(set(Y_labels))}")


pad_token="<pad>"

unique_chars=sorted(set("".join(X_names)))

idx2char=[pad_token]+unique_chars

char2idx={char: idx for idx, char in enumerate(idx2char)}

print(f"Vocabulary Size:{len(idx2char)}")

print(idx2char[:20])
print("Unique chars:", unique_chars)

## Encode function
def encode(name):
    return [char2idx[ch] for ch in name]

## Decode function
def decode(index):
    return "".join(idx2char[idx] for idx in index)



## Encode Labels
unique_labels=sorted(set(Y_labels))

label2idx={label:idx for idx, label in enumerate(unique_labels)}

idx2label = {idx: label for label, idx in label2idx.items()}

print(label2idx)

## Encode entire dataset
X=[encode(name) for name in X_names]
Y=[label2idx[label] for label in Y_labels]

# Test
for i in range(5):
    print(f"Name:{X_names[i]}")
    print(f"Encoded:{X[i]}")
    print(f"Label    : {Y_labels[i]}")
    print(f"Class ID : {Y[i]}")
    print("-" * 40)
    
    

# Train Test Split
X_train, X_test, y_train, y_test=train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y, shuffle=True
)

## Custom Dataset
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

print(f"Train Dataset Size : {len(train_dataset)}")
print(f"Test Dataset Size  : {len(test_dataset)}") 



### Custom Collate function
pad_idx=char2idx[pad_token]

def collate_fn(batch):
    names, labels=zip(*batch)
    
    max_length=max(len(name) for name in names)
    
    padded_names=torch.full(
        (len(names), max_length),
        pad_idx, dtype=torch.long
    )
    
    for i, name in enumerate(names):
        padded_names[i, :len(name)]=torch.tensor(
            name, dtype=torch.long
        )
        
    labels=torch.tensor(labels, dtype=torch.long)
    
    return padded_names, labels



##### DataLoader

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

for x, y in train_loader:

    print("Input Shape :", x.shape)
    print("Labels Shape:", y.shape)

    print()
    print("First Encoded Name:")
    print(x[0])

    print()
    print("First Label:")
    print(y[0])

    break



@dataclass
class Config:
    vocab_size:int=len(char2idx)
    embedding_dim: int=32
    
    num_filters: int=64
    kernel_size: int=3
    
    dropout: float=0.5
    
    num_classes: int=len(label2idx)
    
    
config=Config()


##### CNN Model
class CNNClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embedding=nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=pad_idx
        )
        
        self.conv=nn.Conv1d(
            in_channels=config.embedding_dim,
            out_channels=config.num_filters,
            kernel_size=config.kernel_size
        )
        
        self.relu=nn.ReLU()
        
        self.pool=nn.AdaptiveMaxPool1d(1)
        
        self.dropout=nn.Dropout(config.dropout)
        
        self.fc=nn.Linear(
            config.num_filters,
            config.num_classes
        )
        
    def forward(self, x):
        # (B, T)
        x=self.embedding(x)
        
        # (B, T, C)->(B, C, T)
        x=x.permute(0, 2, 1)
        
        x=self.conv(x)
        
        x=self.relu(x)
        x=self.pool(x)
        
        x=x.squeeze(-1)
        
        x=self.dropout(x)
        x=self.fc(x)
        
        return x


model=CNNClassifier(config).to(device)
print(model)


### Loss function and Optimzer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Test one forward pass

x, y=next(iter(train_loader))

x=x.to(device)

logits=model(x)

print('Input Shape:', x.shape)
print("Output Shape:", logits.shape)


##### Training Loop
best_accuracy = 0.0
epochs = 10

for epoch in range(epochs):

    # Training
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predictions = logits.argmax(dim=1)

        train_correct += (predictions == y).sum().item()
        train_total += y.size(0)

    train_loss /= len(train_loader)
    train_accuracy = train_correct / train_total


    # Evaluation
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            test_loss += loss.item()
            predictions = logits.argmax(dim=1)

            test_correct += (predictions == y).sum().item()
            test_total += y.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = test_correct / test_total


    # Save Best Model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(
            model.state_dict(),
            "best_cnn_model.pth"
        )
        print(f"Best model saved! Accuracy: {best_accuracy:.4f}")

    # Print Epoch Results
    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.4f}"
    )
    
print(f"\nBest Test Accuracy: {best_accuracy:.4f}")



### Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

print("Confusion Matrix:")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=unique_labels
)

fig, ax = plt.subplots(figsize=(10,10))

disp.plot(ax=ax, xticks_rotation=90)

plt.show()

### Classification Report
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=unique_labels
    )
)