import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

## Select Device
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(device)

# Random Seed
torch.manual_seed(42)
random.seed(42)

## Load TinyStories
dataset=load_dataset("roneneldan/TinyStories")

train_data=dataset["train"]
validation_data=dataset["validation"]

print(train_data[0])

# Small Subset
train_text = " ".join(train_data[:1000]["text"])
valid_text = " ".join(validation_data[:200]["text"])

## Build Vocabulary
all_text = train_text + valid_text

chars = sorted(list(set(all_text)))

stoi={c:i for i, c in enumerate(chars)}
itos={i:c for c, i in stoi.items()}

vocab_size=len(chars)
print(vocab_size)

## Encode
train_encoded=torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
valid_encoded=torch.tensor([stoi[c] for c in valid_text], dtype=torch.long)

### Custom Dataset
class TinyStoriesDataset(Dataset):
    def __init__(self, data, block_size):
        self.data=data
        self.block_size=block_size
        
    def __len__(self):
        return len(self.data)-self.block_size
    
    def __getitem__(self, idx):
        x=self.data[idx:idx+self.block_size]
        y=self.data[idx+self.block_size]
        
        return x, y
  
  
BLOCK_SIZE = 8
BATCH_SIZE = 256

## Dataset Object
train_dataset=TinyStoriesDataset(
    train_encoded, BLOCK_SIZE
)

valid_dataset=TinyStoriesDataset(
    valid_encoded, BLOCK_SIZE
)

## Dataloader
train_loader=DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader=DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE
)



### MLP without BatchNorm
class MLPWithoutBN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, block_size):
        super().__init__()
        
        self.embedding=nn.Embedding(
            vocab_size, embedding_dim
        )
        
        self.network=nn.Sequential(
            nn.Linear(
                embedding_dim*block_size, hidden_size
            ),
            nn.ReLU(),
            
            nn.Linear(
                hidden_size, vocab_size
            )
        )
        
    def forward(self, x):
        x=self.embedding(x)
        
        x=x.view(x.size(0), -1)
        
        return self.network(x)
    
    
    
### MLP With BatchNorm
class MLPWithBN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, block_size):
        super().__init__()
        
        self.embedding=nn.Embedding(
            vocab_size, embedding_dim
        )   
        
        self.network=nn.Sequential(
            nn.Linear(
                embedding_dim*block_size, 
                hidden_size
            ),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            
            nn.Linear(
                hidden_size, vocab_size
            )
        ) 
        
    def forward(self, x):
        x=self.embedding(x)
        
        x=x.view(x.size(0), -1)
        
        return self.network(x)
    
    

EMBEDDING_DIM = 24
HIDDEN_SIZE = 256

EPOCHS = 10

LEARNING_RATE = 0.001

### create model
model_without_bn = MLPWithoutBN(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    block_size=BLOCK_SIZE
).to(device)

model_with_bn = MLPWithBN(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    block_size=BLOCK_SIZE
).to(device)

## Loss Function
criterion = nn.CrossEntropyLoss()

## Optimizers
optimizer_without_bn = torch.optim.Adam(
    model_without_bn.parameters(),
    lr=LEARNING_RATE
)

optimizer_with_bn = torch.optim.Adam(
    model_with_bn.parameters(),
    lr=LEARNING_RATE
)


### Evaluation Function
def evaluate_model(model, dataloader, criterion):
    model.eval()
    
    total_loss=0
    
    with torch.no_grad():
        for x, y in dataloader:
            x=x.to(device)
            y=y.to(device)
            
            logits=model(x)
            
            loss=criterion(logits, y)
            
            total_loss+=loss.item()
            
    return total_loss/len(dataloader)



### Training Function
def train_model(model, train_loader, valid_loader, optimizer, criterion, epochs):
    train_losses=[]
    valid_losses=[]
    
    for epoch in range(epochs):
        model.train()
        
        running_loss=0
        
        for x, y in train_loader:
            x=x.to(device)
            y=y.to(device)
            
            optimizer.zero_grad()
            logits=model(x)
            
            loss=criterion(logits, y)
            
            loss.backward()
            optimizer.step()
            
            running_loss+=loss.item()
            
            
        train_loss=running_loss/len(train_loader)
        
        valid_loss=evaluate_model(
            model, valid_loader, criterion
        )
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(
            f"Epoch {epoch+1}/{epochs}"
            f" | Train Loss: {train_loss:.4f}"
            f" | Validation Loss: {valid_loss:.4f}"
        )
        
    return train_losses, valid_losses


### Train without BatchNorm
print("="*60)
print("Training WITHOUT Batch Normalization")

train_loss_without_bn, valid_loss_without_bn=train_model(
    model=model_without_bn,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer_without_bn,
    criterion=criterion,
    epochs=EPOCHS
)

### Train with BatchNorm
print("="*60)
print("Training WITH Batch Normalization")

train_loss_with_bn, valid_loss_with_bn=train_model(
    model=model_with_bn,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer_with_bn,
    criterion=criterion,
    epochs=EPOCHS
)



### Plot Loss Curves
plt.figure(figsize=(10,6))

plt.plot(
    train_loss_without_bn,
    label="Train (Without BN)"
)

plt.plot(
    valid_loss_without_bn,
    label="Validation (Without BN)"
)

plt.plot(
    train_loss_with_bn,
    label="Train (With BN)"
)

plt.plot(
    valid_loss_with_bn,
    label="Validation (With BN)"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Batch Normalization Comparison")
plt.legend()
plt.grid(True)

plt.show()


### Text Generation Function
def generate_text(model, start_text, length=200):
    model.eval()
    
    context=[stoi[c] for c in start_text]
    
    for _ in range(length):
        x=torch.tensor(context[-BLOCK_SIZE:],
                       dtype=torch.long).unsqueeze(0).to(device)
        
        if x.shape[1]<BLOCK_SIZE:
            padding=torch.zeros(
                (1, BLOCK_SIZE-x.shape[1]),
                dtype=torch.long
            ).to(device)
            
            x=torch.cat([padding, x], dim=1)
            
        with torch.no_grad():
            logits=model(x)
            
            probs=torch.softmax(logits, dim=1)
            
            next_char=torch.multinomial(
                probs, num_samples=1
            ).item()
            
        context.append(next_char)
        
    return "".join(itos[i] for i in context)



### Generate without BatchNorm
print("="*60)
print("WITHOUT BatchNorm")

print(generate_text(
    model_without_bn,
    "Once "
))


### Generate with BatchNorm
print("="*60)
print("WITH BatchNorm")

print(generate_text(
    model_with_bn,
    "Once "
))