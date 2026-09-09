import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

### Device
device=torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

torch.manual_seed(42)
random.seed(42)

### Hyperparameters
BLOCK_SIZE = 8
BATCH_SIZE = 256
EMBEDDING_DIM = 64
HIDDEN_SIZE = 512
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 10

### Load Dataset
print("=" * 60)
print("Loading TinyStories Dataset...")

dataset=load_dataset("roneneldan/TinyStories")

train_data=dataset["train"]
validation_data=dataset['validation']

print(f"Training stories   : {len(train_data)}")
print(f"Validation stories : {len(validation_data)}")

train_size=5000
valid_size=1000

train_text="".join(train_data[:train_size]["text"])
valid_text="".join(validation_data[:valid_size]["text"])

### Build Vocabulary
all_text=train_text+valid_text
chars=sorted(list(set(all_text)))

stoi={c:i for i, c in enumerate(chars)}
itos={i:c for c, i in stoi.items()}

vocab_size=len(chars)
print(f"Vocabulary Size: {vocab_size}")

## Encode Text
train_encoded = torch.tensor(
    [stoi[c] for c in train_text],
    dtype=torch.long
)

valid_encoded = torch.tensor(
    [stoi[c] for c in valid_text],
    dtype=torch.long
)

print(f"Training Characters   : {len(train_encoded)}")
print(f"Validation Characters : {len(valid_encoded)}")

print("First 20 Encoded Characters:")

print(train_encoded[:20])

### Cutom Dataset
class TinyStoriesDataset(Dataset):
    def __init__(self, encoded_text, block_size):
        self.data=encoded_text
        self.block_size=block_size
        
    def __len__(self):
        return len(self.data)-self.block_size
    
    def __getitem__(self, index):
        x=self.data[index:index+self.block_size]
        y=self.data[index+self.block_size]
        
        return x, y
    

## Dataset object
train_dataset=TinyStoriesDataset(
    train_encoded, BLOCK_SIZE
)
valid_dataset=TinyStoriesDataset(
    valid_encoded,BLOCK_SIZE
)

### Dataloader
train_loader=DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

valid_loader=DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False
)

### Test
x, y = next(iter(train_loader))

print("Input Shape :", x.shape)
print("Target Shape:", y.shape)


sample_x, sample_y = train_dataset[0]

print("Input indices:")
print(sample_x)

print("Target index:")
print(sample_y)

### Model

class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, block_size, dropout):
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
            nn.Dropout(dropout),
            
            nn.Linear(
                hidden_size, hidden_size
            ),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(
                hidden_size, vocab_size
            )
        )
    
    def forward(self, x):
        x=self.embedding(x)
        x=x.view(x.size(0), -1)
        
        logits=self.network(x)
        
        return logits
    

## Model
model=MLP(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT
).to(device)

print(model)

# Test the model
x, y=next(iter(train_loader))
x=x.to(device)

logits=model(x)

print('Input Shape:', x.shape)
print("Output Shape:", logits.shape)


### Loss and Optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


### Initialization Function
def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            
        elif isinstance(layer, nn.Embedding):
            nn.init.normal_(
                layer.weight,mean=0.0, std=0.02
            )
            
            
initialize_weights(model)

## Verify Initialization 
print(model.network[0].weight[:2])


### Train One Epoch
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss=0
    
    for x, y in dataloader:
        x=x.to(device)
        y=y.to(device)
        
        optimizer.zero_grad()
        logits=model(x)
        loss=criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
        
    return running_loss/len(dataloader)


### Validation Function
def validate(model, dataloader, criterion):
    model.eval()
    
    running_loss=0
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            running_loss += loss.item()

    return running_loss / len(dataloader)
                    


###### TRAIN Model

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs, config):
    train_losses = []
    valid_losses = []
    
    best_loss=float("inf")
    for epoch in range(epochs):
        train_loss=train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        valid_loss=validate(
            model, valid_loader, criterion
        )
        
        train_losses.append(train_loss)

        valid_losses.append(valid_loss)
        
        print(
            f"Epoch {epoch+1}/{epochs}"
            f" | Train Loss: {train_loss:.4f}"
            f" | Validation Loss: {valid_loss:.4f}"
        )
        
        if valid_loss<best_loss:
            best_loss=valid_loss
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "vocab_size": vocab_size,
                "block_size": BLOCK_SIZE,
                "stoi": stoi,
                "itos": itos
            }, "best_model_14.pth")
            
            print("Best model updated.")
            
    return train_losses, valid_losses, best_loss



## Configurations
configs = [

    {
        "embedding_dim": 32,
        "hidden_size": 256,
        "dropout": 0.2,
        "learning_rate": 0.001
    },

    {
        "embedding_dim": 64,
        "hidden_size": 256,
        "dropout": 0.3,
        "learning_rate": 0.001
    },

    {
        "embedding_dim": 64,
        "hidden_size": 512,
        "dropout": 0.3,
        "learning_rate": 0.0005
    }

]

results = []
best_validation_loss = float("inf")
best_configuration = None
best_train_losses = None
best_valid_losses = None

for config in configs:
    print("\n" + "=" * 70)
    print("Training Configuration")
    print(config)
    
    model = MLP(
        vocab_size=vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_size=config["hidden_size"],
        block_size=BLOCK_SIZE,
        dropout=config["dropout"]
    ).to(device)
    
    initialize_weights(model)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )
    
    train_losses, valid_losses, validation_loss = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        config=config
    )
     
    results.append({
        "config": config,
        "validation_loss": validation_loss
    })
    
    

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_configuration = config
        
        best_train_losses = train_losses
        best_valid_losses = valid_losses



    
print("\n" + "=" * 70)
print("Experiment Results")
print("=" * 70)

for result in results:
    print(result)

print("\n" + "=" * 70)
print("Best Configuration")
print("=" * 70)

print(best_configuration)

print(f"\nBest Validation Loss: {best_validation_loss:.4f}")


### Plot
plt.figure(figsize=(10, 6))

plt.plot(
    best_train_losses, 
    marker="o",
    label='Training Loss'
)

plt.plot(
    best_valid_losses,
    marker="s",
    label="Validation Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")

plt.show()



### Load Saved Model
checkpoint=torch.load(
    "best_model_14.pth",
    map_location=device
)

### Rebuild the Model
model = MLP(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["config"]["embedding_dim"],
    hidden_size=checkpoint["config"]["hidden_size"],
    block_size=checkpoint["block_size"],
    dropout=checkpoint["config"]["dropout"]
).to(device)

## load the weights
model.load_state_dict(
    checkpoint["model_state_dict"]
)
model.eval()


### Generate Story Function
def generate_text(model, start_text, stoi, itos, block_size, device, max_lenght=300):
    model.eval()
    
    context=[stoi[c] for c in start_text if c in stoi]
    
    if len(context)==0:
        raise ValueError("Prompt contains no known characters.")
    
    generated=context.copy()
    
    with torch.no_grad():
        for _ in range(max_lenght):
            x=generated[-block_size:]
            
            if len(x)<block_size:
                x=[0]*(block_size-len(x))+x
                
            x=torch.tensor(
                x, dtype=torch.long
            ).unsqueeze(0).to(device)
            
            logits=model(x)
            
            probs=torch.softmax(
                logits, dim=1
            )
            
            next_index=torch.multinomial(
                probs, num_samples=1
            ).item()
            
            generated.append(next_index)
            
    return "".join(itos[i] for i in generated)


prompt=input("\nEnter a prompt:")

story=generate_text(
    model, prompt, checkpoint["stoi"],
    checkpoint["itos"], checkpoint["block_size"],
    device
)

print("\nGenerated Story:\n")
print(story)