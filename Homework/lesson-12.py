import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

from sklearn.decomposition import PCA

random.seed(42)
torch.manual_seed(42)

## Load Dataset
dataset_path="/Users/mac/Desktop/Machine Learning/DL/DB/Names/names.txt"
with open(dataset_path, "r") as f:
    words=f.read().splitlines()
    
print("Total Names:", len(words))
print("First 10 Names:", words[:10])

## Build Vocabulary

chars=sorted(list(set("".join(words))))

stoi={c:i+1 for i, c in enumerate(chars)}
stoi["."]=0

itos={i:c for c, i in stoi.items()}

vocab_size=len(stoi)

print(f"Vocabulary Size:{vocab_size}")
print(stoi)

### Shuffle and Split Dataset
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

train_words = words[:n1]
val_words = words[n1:n2]
test_words = words[n2:]

print(f"Training Names  : {len(train_words)}")
print(f"Validation Names: {len(val_words)}")
print(f"Test Names      : {len(test_words)}")


### Custom Dataset
block_size=3

class NamesDataset(Dataset):
    def __init__(self, words, stoi, block_size):
        self.X=[]
        self.Y=[]
        
        for word in words:
            ## Initial context: [..]
            context=[0]*block_size
            
            # add "." to indicate the end of the word
            for ch in word+'.':
                target=stoi[ch]
                
                self.X.append(context.copy())
                self.Y.append(target)
                
                # Slide the context window
                context=context[1:]+[target]
                
        self.X=torch.tensor(self.X, dtype=torch.long)
        self.Y=torch.tensor(self.Y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
### Dataset Object
train_dataset = NamesDataset(
    train_words,
    stoi,
    block_size
)

val_dataset = NamesDataset(
    val_words,
    stoi,
    block_size
)

test_dataset = NamesDataset(
    test_words,
    stoi,
    block_size
)

### Dataloader
batch_size = 256

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Test
X_batch, Y_batch = next(iter(train_loader))

print("Input Shape :", X_batch.shape)
print("Target Shape:", Y_batch.shape)

print("\nFirst Input:")
print(X_batch[0])

print("\nFirst Target:")
print(Y_batch[0])


### MLP Model

class MLP(nn.Module):
    def __init__(self, 
                vocab_size, 
                block_size, 
                embedding_dim,
                hidden_size, 
                num_hidden_layer
        ):
        super().__init__()
        
        # Character Embedding
        self.embedding=nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        
        layers=[]
        
        input_size=block_size*embedding_dim
        
        # Hidden Layers
        
        for _ in range(num_hidden_layer):
            layers.append(
                nn.Linear(input_size, hidden_size)
            )   
            
            layers.append(
                nn.ReLU()
            )    
            input_size=hidden_size
            
        # output layer
        layers.append(
            nn.Linear(hidden_size, vocab_size)
        )
        self.network=nn.Sequential(*layers)
        
        
    def forward(self, x):
        # x-> (batch_size, block_size)
        x=self.embedding(x)
        
        # (batch_size, block_size, embedding_dim)
        x=x.view(x.size(0), -1)
        
        # (batch_size, block_size * embedding_dim)
        logits = self.network(x)
        
        return logits
        

## Model
embedding_dim = 20
hidden_size = 200
num_hidden_layers = 2

model = MLP(
    vocab_size=vocab_size,
    block_size=block_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_hidden_layer=num_hidden_layers
)

print(model)

## Check input and Output shapes
X_batch, Y_batch = next(iter(train_loader))

print("Input Shape :", X_batch.shape)

logits = model(X_batch)

print("Output Shape:", logits.shape)


## Weight Initialization

# Xavier Initialization

def initialize_xavier(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            

def initialize_kaiming(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(
                layer.weight,
                nonlinearity="relu"
            )
            nn.init.zeros_(layer.bias)
   
   
         
initialize_xavier(model)
print(model.network[0].weight[:5])


## Loss Function
criterion=nn.CrossEntropyLoss()

## Training Function
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    
    model.train()
    running_loss=0.0
    
    for X_batch, Y_batch in dataloader:
        
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, Y_batch)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


## Validation Function
def evaluate(model, dataloader, criterion, device):

    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            logits = model(X_batch)

            loss = criterion(logits, Y_batch)

            running_loss += loss.item()

    return running_loss / len(dataloader)


## Select Device
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(device)


## Move model to device
model.to(device)

## Optimizer
learning_rate = 0.01

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

### Complete Training Loop
epochs=20
train_losses=[]
val_losses=[]

for epoch in range(epochs):
    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device
    )

    val_loss = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Validation Loss: {val_loss:.4f}"
    )
    
    
## Test the Model
test_loss=evaluate(
    model, test_loader, criterion, device
)
print(f"Test Loss: {test_loss:.4f}")



### Hyperparameter Tunning
configs = [

    {
        "block_size": 3,
        "embedding_dim": 10,
        "hidden_size": 100,
        "num_hidden_layers": 1,
        "learning_rate": 0.01,
        "initialization": "xavier"
    },

    {
        "block_size": 3,
        "embedding_dim": 20,
        "hidden_size": 200,
        "num_hidden_layers": 2,
        "learning_rate": 0.01,
        "initialization": "kaiming"
    },

    {
        "block_size": 5,
        "embedding_dim": 20,
        "hidden_size": 300,
        "num_hidden_layers": 2,
        "learning_rate": 0.005,
        "initialization": "kaiming"
    },

    {
        "block_size": 5,
        "embedding_dim": 30,
        "hidden_size": 300,
        "num_hidden_layers": 3,
        "learning_rate": 0.003,
        "initialization": "xavier"
    }

]

# Variable for saving results
results = []
best_model = None
best_config = None
best_validation_loss = float("inf")

## Train the model
for config in configs:
    train_dataset = NamesDataset(
        train_words,
        stoi,
        config["block_size"]
    )

    val_dataset = NamesDataset(
        val_words,
        stoi,
        config["block_size"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False
    )

    model = MLP(
        vocab_size,
        config["block_size"],
        config["embedding_dim"],
        config["hidden_size"],
        config["num_hidden_layers"]
    ).to(device)


    if config["initialization"] == "xavier":
        initialize_xavier(model)
    else:
        initialize_kaiming(model)



    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )

    epochs = 20
    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        validation_loss = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

    results.append({
        "config": config,
        "validation_loss": validation_loss

    })

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_model = model
        best_config = config
        
        
## Result
print("\nResults")

for result in results:
    print(result["config"])
    print(
        "Validation Loss:",
        round(result["validation_loss"], 4)
    )

    print("-" * 50)
   
   
## Print the Best Configuration 
print("\nBest Configuration")
print(best_config)
print(f"Best Validation Loss: {best_validation_loss:.4f}")


## Evaluate the Best Model on the Test Set
test_dataset = NamesDataset(
    test_words,
    stoi,
    best_config["block_size"]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)

# evaluate
test_loss = evaluate(
    best_model,
    test_loader,
    criterion,
    device
)

print(f"Test Loss: {test_loss:.4f}")

## Generate New Names
best_model.eval()
number_of_names=20
for _ in range(number_of_names):
    context = [0] * best_config["block_size"]
    generated_name = []
    
    while True:
        x=torch.tensor([context], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits=best_model(x)
            
        probabilities=F.softmax(logits, dim=1)
        
        index=torch.multinomial(
            probabilities, 
            num_samples=1
        ).item()
        
        context=context[1:]+[index]
        
        if index==0:
            break
        
        generated_name.append(itos[index])
        
    print("".join(generated_name))
    



## Visualize Character Embeddings
embeddings = best_model.embedding.weight.detach().cpu().numpy()

## Apply PCA
pca = PCA(n_components=2)

reduced_embeddings = pca.fit_transform(embeddings)

## Plot
plt.figure(figsize=(8,8))

for i in range(vocab_size):

    plt.scatter(
        reduced_embeddings[i,0],
        reduced_embeddings[i,1]
    )

    plt.text(
        reduced_embeddings[i,0],
        reduced_embeddings[i,1],
        itos[i],
        fontsize=12
    )

plt.title("Character Embeddings (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid(True)

plt.show()

## Plot Validation Results
labels = []

losses = []

for i, result in enumerate(results):

    labels.append(f"Model {i+1}")

    losses.append(result["validation_loss"])

plt.figure(figsize=(8,5))

plt.bar(labels, losses)

plt.ylabel("Validation Loss")

plt.title("Hyperparameter Comparison")

plt.show()

## Print final summary
print("=" * 60)
print("BEST MODEL")

print(best_config)

print(f"Validation Loss : {best_validation_loss:.4f}")
print(f"Test Loss       : {test_loss:.4f}")
print("=" * 60)