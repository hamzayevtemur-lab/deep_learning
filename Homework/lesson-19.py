import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

torch.manual_seed(42)
random.seed(42)

### Load the Dataset
DATA_PATH = "/Users/mac/Desktop/Machine Learning/DL/DB/Names/names.txt"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    names=[ line.strip().lower() for line in f if line.strip() ]
    

print("Number of names:", len(names))
print("First 10 names:")

for name in names[:10]:
    print(name)
    
names = [f".{name}." for name in names]

## Build the Vocabulary
vocab=sorted(set("".join(names)))

stoi={ch:i for i, ch in enumerate(vocab)}
itos={i:ch for ch, i in stoi.items()}

VOCAB_SIZE = len(vocab)

print("Vocabulary:", vocab)
print("Vocabulary size:", VOCAB_SIZE)


## Encode 
def encode(text):
    return [stoi[ch] for ch in text]

## Decode
def decode(idx):
    return "".join(itos[i] for i in idx)

example = names[0]

encoded = encode(example)

print("Original:", example)
print("Encoded:", encoded)
print("Decoded:", decode(encoded))

    
## Train Test Split
train_names, test_names=train_test_split(
    names, test_size=0.2,random_state=42
)

print("Training names:", len(train_names))
print("Testing names:", len(test_names))

### Custom Dataset
class NamesDataset(Dataset):
    def __init__(self, names):
        self.names=names
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name=self.names[idx]
        
        encoded=torch.tensor(
            encode(name),
            dtype=torch.long
        )
        
        x=encoded[:-1]
        y=encoded[1:]
        
        return x, y
    
train_dataset = NamesDataset(train_names)
test_dataset = NamesDataset(test_names)

## Check one
x, y = train_dataset[0]

print("X:", x)
print("Y:", y)

print("X decoded:", decode(x.tolist()))
print("Y decoded:", decode(y.tolist()))


### Dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

x, y = next(iter(train_loader))

print("X shape:", x.shape)
print("Y shape:", y.shape)

## Hyperparameters
EMBEDDING_DIM = 16
HIDDEN_SIZE = 64

## Create the RNN Class
class CharRNN(nn.Module):
    def __init__(
        self, 
        vocab_size,
        embedding_dim,
        hidden_size
    ):
        super().__init__()
        
        self.E=nn.Parameter(
            torch.randn(
                vocab_size, embedding_dim
            )*0.01
        )
        
        self.hidden_size=hidden_size
        
        # Input-> Hidden
        self.W_xh=nn.Parameter(
            torch.randn(
                embedding_dim, hidden_size
            )*0.01
        )
        
        # Hidden -> hidden
        self.W_hh=nn.Parameter(
            torch.randn(
                hidden_size, hidden_size
            )*0.01
        )
        
        # Hidden bias
        self.b_h=nn.Parameter(
            torch.zeros(hidden_size)
        )
        
        # Hidden->Output
        self.W_hy=nn.Parameter(
            torch.randn(
                hidden_size,vocab_size
            )*0.01
        )
        
        # Output bias
        self.b_y=nn.Parameter(
            torch.zeros(vocab_size)
        )
    def forward(self, x):
        batch_size, sequence_length=x.shape
        
        # Convert character IDs into embeddings
        embeddings = self.E[x]
        
        # Initial hidden state
        h = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device
        )
        
        outputs=[]
        for t in range(sequence_length):
            # Current character embedding
            x_t=embeddings[:,t,:]
            
            # Recurrent equation
            h=torch.tanh(
                x_t @ self.W_xh+h @ self.W_hh + self.b_h
            )
            
            # Hidden->output
            logits=(
                h @ self.W_hy+self.b_y
            )
            
            outputs.append(logits)
            
        # Combine all time-step outputs
        outputs=torch.stack(
            outputs, dim=1
        )
        return outputs
    


### Create the Model
model=CharRNN(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE
).to(device)

print(model)

## Model Parameters
for name, param in model.named_parameters():
    print(name, param.shape)
        
        
### Loss function & Optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

## Test one forward pass
x, y = next(iter(train_loader))

x = x.to(device)
y = y.to(device)

logits = model(x)

print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Output shape:", logits.shape)


### Training loop
EPOCHS=20
train_losses=[]
test_losses=[]

for epoch in range(EPOCHS):
    
    ### Training
    model.train()
    total_train_loss=0.0
    
    for x, y in train_loader:
        x=x.to(device)
        y=y.to(device)
        
        optimizer.zero_grad()
        logits=model(x)
        
        loss=criterion(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        
        total_train_loss+=loss.item()
        
    avg_train_loss=(
        total_train_loss/len(train_loader)
    )
    
    ### Evaluation
    model.eval()
    
    total_test_loss=0.0
    
    with torch.no_grad():
        for x, y in test_loader:
            x=x.to(device)
            y=y.to(device)
            
            logits=model(x)
            
            loss=criterion(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1)
            )
            total_test_loss+=loss.item()
            
    avg_test_loss=(
        total_test_loss/len(test_loader)
    )
    
    # Save losses
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    
    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Train Loss: {avg_train_loss:.4f} "
        f"| Test Loss: {avg_test_loss:.4f}"
    )
    
    
model.eval()
x, y=test_dataset[0]
x=x.unsqueeze(0).to(device)

with torch.no_grad():
    logits=model(x)
    
predictions=logits.argmax(dim=-1)

print("Input:      ", decode(x[0].cpu().tolist()))
print("Target:     ", decode(y.tolist()))
print("Prediction: ", decode(predictions[0].cpu().tolist()))


### Plot the Losses
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,label="Train Loss"
)

plt.plot(
    test_losses,
    label="Test Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Character-level RNN Training")

plt.legend()
plt.show()


## Generation Function
def generate_name(model, max_length=20):
    model.eval()
    
    # Start with "."
    current_char=stoi["."]
    
    # Initial hidden state
    h=torch.zeros(
        1, model.hidden_size,
        device=device
    )
    generated=[]
    
    with torch.no_grad():
        for _ in range(max_length):
            # Current character ->tensor
            x=torch.tensor(
                [[current_char]],
                dtype=torch.long,
                device=device
            )
            
            # Character ->embedding
            x_t=model.E(x[:,0])
            
            # Recurrent step
            h=torch.tanh(
                x_t @ model.W_xh+h@ model.W_hh+model.b_h
            )
            #Hidden -> output
            logits=(
                h @ model.W_hy+model.b_y
            )
            # Logits->probabilities
            probs=torch.softmax(
                logits, dim=-1
            )
            # Sample next character 
            next_char=torch.multinomial(
                probs, num_samples=1
            ).item()
            
            # Stop at "."
            if next_char==stoi["."]:
                break
            
            generated.append(
                itos[next_char] # type:ignore
            )
            
            # Feed prediction back into the RNN
            current_char=next_char
            
    return "".join(generated)


for _ in range(20):
    print(generate_name(model))

