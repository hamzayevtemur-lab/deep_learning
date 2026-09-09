import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

## Load Dataset

dataset_path="/Users/mac/Desktop/Machine Learning/DL/DB/Names/names.txt"

with open(dataset_path, "r", encoding="utf-8") as f:
    words=f.read().splitlines()
    
print("Number of names:", len(words))
print(words[:10])

## Build Vocabulary
chars=sorted(list(set("".join(words))))

stoi={s:i+1 for i, s in enumerate(chars)}
stoi['.']=0

itos={i:s for s, i in stoi.items()}
vocab_size=len(stoi)

print(vocab_size)
print(stoi)

## Build Dataset
block_size=3

X=[]
Y=[]

for word in words:
    context=[0]*block_size
    
    for ch in word+'.':
        ix=stoi[ch]
        
        X.append(context)
        Y.append(ix)
        
        context=context[1:]+[ix]
        
X=torch.tensor(X)
Y=torch.tensor(Y)

print(X.shape)
print(Y.shape)

## Split Dataset
random.seed(42)
random.shuffle(words)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

train_words = words[:n1]
val_words = words[n1:n2]
test_words = words[n2:]

## Dataset Builder

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

    return torch.tensor(X),torch.tensor(Y)

Xtr,Ytr = build_dataset(train_words)
Xdev,Ydev = build_dataset(val_words)
Xte,Yte = build_dataset(test_words)

###  Language Model with Embeddings

class NameGenerator(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim=10,
                 hidden=200):
        
        super().__init__()
        
        self.embedding=nn.Embedding(
            vocab_size, embedding_dim
        )
        
        self.fc1=nn.Linear(
            block_size*embedding_dim,
            hidden
        )
        
        self.fc2=nn.Linear(
            hidden, vocab_size
        )
        
    def forward(self, x):
        x=self.embedding(x)
        
        x=x.view(x.shape[0], -1)
        
        x=F.relu(self.fc1(x))
        
        logits=self.fc2(x)
        
        return logits
    

## Initialize Model
model=NameGenerator(vocab_size)
print(model)


## Loss & Optimizer
criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.01
)



### Training Loop

epochs=20
batch_size=256

for epoch in range(epochs):
    permutation=torch.randperm(Xtr.size(0))
    
    total_loss=0
    
    model.train()
    
    for i in range(0, Xtr.size(0), batch_size):
        indices=permutation[i:i+batch_size]
        
        xb=Xtr[indices]
        yb=Ytr[indices]
        
        logits=model(xb)
        loss=criterion(logits, yb)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        total_loss+=loss.item()
        
    model.eval()
    
    with torch.no_grad():
        val_logits=model(Xdev)
        
        val_loss=criterion(val_logits, Ydev)
        
        print(
        f"Epoch {epoch+1:2d} | "
        f"Train Loss {total_loss:.3f} | "
        f"Val Loss {val_loss:.4f}"
    )


## Test Loss
model.eval()
with torch.no_grad():
    logits=model(Xte)
    loss=criterion(logits, Yte)
    
print("Test Loss:", loss.item())



### Generate Names
model.eval()

for _ in range(20):
    context=[0]*block_size
    
    out=[]
    
    while True:
        x=torch.tensor([context])
        
        logits=model(x)
        
        probs=F.softmax(logits, dim=1)
        
        ix=torch.multinomial(probs, 1).item()
        
        context=context[1:]+[ix]
        
        if ix==0:
            break
        
        out.append(itos[ix])
        
    print("".join(out))
    
    
    
### Visualize Embeddings

embeddings=model.embedding.weight.detach().numpy()

pca=PCA(n_components=2)

reduced=pca.fit_transform(embeddings)

plt.figure(figsize=(8, 8))

for i in range(vocab_size):
    plt.scatter(reduced[i, 0], reduced[i, 1])
    
    plt.text(
        reduced[i,0],
        reduced[i, 1],
        itos[i],
        fontsize=12
    )
    
plt.title("Character Embeddings")
plt.show()


        