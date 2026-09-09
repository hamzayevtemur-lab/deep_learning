import random 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using Device:{device}")

## Load Dataset

dataset_path="/Users/mac/Desktop/Machine Learning/DL/DB/Names/names.txt"

with open(dataset_path, "r", encoding="utf-8") as f:
    words=f.read().splitlines()
    
print(f"Total names: {len(words)}")


## Build Vocabulary
chars=sorted(list(set("".join(words))))

stoi={ch:i+1 for i, ch in enumerate(chars)}
stoi["."]=0

itos={i:ch for ch, i in stoi.items()}

vocab_size=len(stoi)

print(f"Vocabulary size:{vocab_size}")
print(stoi)

## Hyperparameter
block_size=3

def build_dataset(words):
    X=[]
    Y=[]
    
    for word in words:
        context=[0]*block_size
        
        for ch in word+".":
            ix=stoi[ch]
            X.append(context)
            Y.append(ix)
            
            context=context[1:]+[ix]
            
    X=torch.tensor(X)
    Y=torch.tensor(Y)
    
    return X, Y


## Shuffle Dataset
random.seed(42)
random.shuffle(words)

n1=int(0.8*len(words))
n2=int(0.9*len(words))

train_words=words[:n1]
val_words=words[n1:n2]
test_words=words[n2:]

X_train, Y_train = build_dataset(train_words)
X_val, Y_val = build_dataset(val_words)
X_test, Y_test = build_dataset(test_words)

print("Training examples :", len(X_train))
print("Validation examples:", len(X_val))
print("Testing examples   :", len(X_test))

## Custom Dataset
class NamesDataset(Dataset):
    def __init__(self, X, Y):
        self.X=X
        self.Y=Y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    

## Dataloader
batch_size=256

train_loader=DataLoader(
    NamesDataset(X_train, Y_train),
    batch_size=batch_size,
    shuffle=True
)

val_loader=DataLoader(
    NamesDataset(X_val, Y_val),
    batch_size=batch_size,
    shuffle=False
)

test_loader=DataLoader(
    NamesDataset(X_test, Y_test),
    batch_size=batch_size,
    shuffle=False
)

# check one batch
x, y=next(iter(train_loader))

print("Input shape :", x.shape)
print("Target shape:", y.shape)

print("Example Context (indices)")
print(x[0])

print("Target Index")
print(y[0])

print("Context Characters:")
print([itos[i.item()] for i in x[0]])

print("Target Character:")
print(itos[y[0].item()])


### N-Gram MLP Language Model

# Hyperparameters
embedding_dim = 10
hidden_dim = 200
num_hidden_layers = 2

class NGramMLP(nn.Module):
    def __init__(self, 
                 vocab_size,
                 block_size,
                 embedding_dim,
                 hidden_dim,
                 num_hidden_layers=1
                ):
        
        super().__init__()
        
        self.embedding=nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        
        layers=[]
        
        input_dim=block_size*embedding_dim
        
        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            
            input_dim=hidden_dim
            
        # Output layer
        layers.append(nn.Linear(hidden_dim, vocab_size))
        
        self.network=nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: (batch_size, block_size)
        x=self.embedding(x)
        
        # (batch, block_size, embedding_dim)
        x=x.view(x.shape[0], -1)
        
        # (batch, block_size*embedding_dim)
        logits=self.network(x)
        
        return logits 
    
    
### Model
model=NGramMLP(
    vocab_size=vocab_size,
    block_size=block_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim, 
    num_hidden_layers=num_hidden_layers
)     

model=model.to(device)

print(model)

# Number of Parameteres
total_params=sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")

### Test Forward Pass
x, y=next(iter(train_loader)) 

x=x.to(device)
y=y.to(device)

logits=model(x)

print("Input shape :", x.shape)
print("Output shape:", logits.shape)

## Loss

criterion=nn.CrossEntropyLoss()
loss=criterion(logits, y)

print(f"Initial Loss:{loss.item():.4f}")

## Optimizer
optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


### Evaluation Function
def evaluate(model, dataloader):
    model.eval()
    
    total_loss=0
    
    with torch.no_grad():
        for X, Y in dataloader:
            X=X.to(device)
            Y=Y.to(device)
            
            logits=model(X)
            loss=criterion(logits, Y)
            
            total_loss+=loss.item()
            
    return total_loss/len(dataloader)


### Training Loop

epochs=20
train_losses=[]
val_losses=[]

best_val_loss=float("inf")

checkpoint_path = "ngram_mlp_best.pth"

for epoch in range(epochs):
    model.train()
    
    running_loss=0
    
    for X, Y in train_loader:
        X=X.to(device)
        Y=Y.to(device)
        
        optimizer.zero_grad()
        
        logits=model(X)
        loss=criterion(logits, Y)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss+=loss.item()
        
    train_loss=running_loss/len(train_loader)
    
    val_loss=evaluate(model, val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(
        f"Epoch [{epoch+1}/{epochs}]"
        f"Train Loss: {train_loss:.4f}"
        f"Validation Loss: {val_loss:.4f}"
    )
    
    # Save best model
    if val_loss<best_val_loss:
        best_val_loss=val_loss
        
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_loss": val_loss,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "num_hidden_layers": num_hidden_layers,
                "block_size": block_size,
                "vocab_size": vocab_size,
            },
            checkpoint_path
        )
        print("Best Model Saved")
        

## Test Performance
test_loss=evaluate(model, test_loader)
print(f"\nTest Loss:{test_loss:.4f}")

### Plot Loss Curves
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")

plt.title("Training History")

plt.legend()
plt.grid(True)

plt.show()



### Load Saved model

checkpoint=torch.load(
    checkpoint_path, 
    map_location=device
)



loaded_model = NGramMLP(
    vocab_size=checkpoint["vocab_size"],
    block_size=checkpoint["block_size"],
    embedding_dim=checkpoint["embedding_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    num_hidden_layers=checkpoint["num_hidden_layers"]
)

loaded_model.load_state_dict(checkpoint["model_state_dict"])

loaded_model=loaded_model.to(device)
loaded_model.eval()

print("Checkpoint Loaded Successfully!")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Validation Loss: {checkpoint['validation_loss']:.4f}")


## Generate Names

def generate_name(model, block_size, stoi, itos, device):
    model.eval()
    
    context = [0] * block_size
    out = []
    
    while True:
        x=torch.tensor([context], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits=model(x)
            probs=torch.softmax(logits, dim=1)
            ix=torch.multinomial(
                probs, num_samples=1
            ).item()
            
        if ix==0:
            break
        
        out.append(itos[ix])
        
        context=context[1:]+[ix]
        
    return "".join(out)


print("Generated Names")

for _ in range(20):
    print(generate_name(
        loaded_model,
        block_size,
        stoi,
        itos,
        device
    ))
    
    
    

############## Experiments
def train_configuration(
    embedding_dim,
    hidden_dim,
    num_hidden_layers,
    epochs=10
):

    model = NGramMLP(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(epochs):

        model.train()

        for X, Y in train_loader:

            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()

            logits = model(X)

            loss = criterion(logits, Y)

            loss.backward()

            optimizer.step()

    val_loss = evaluate(model, val_loader)

    return val_loss




experiments = [

    (10,100,1),

    (10,200,1),

    (20,200,1),

    (20,200,2),

    (30,300,2)

]

results = []

for emb, hidden, layers in experiments:

    print(f"Running {emb}-{hidden}-{layers}")

    loss = train_configuration(
        emb,
        hidden,
        layers
    )

    results.append(
        (emb, hidden, layers, loss)
    )
    

print("="*60)

print("Experiment Results")

print(f"{'Embedding':<12}"
      f"{'Hidden':<10}"
      f"{'Layers':<10}"
      f"{'Val Loss'}")

for emb, hidden, layers, loss in results:

    print(f"{emb:<12}"
          f"{hidden:<10}"
          f"{layers:<10}"
          f"{loss:.4f}")