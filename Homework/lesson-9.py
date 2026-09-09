import random 
import torch
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

dataset_dir="/Users/mac/Desktop/Machine Learning/DL/DB/Names/names.txt"

with open(dataset_dir, "r", encoding="utf-8") as f:
    names=[line.strip().lower() for line in f if line.strip()]
    

print("Dataset Information")
print(f"Total Names: {len(names)}")
print(f"Shortest Name:{min(len(name) for name in names)}")
print(f"Longest Name:{max(len(name) for name in names)}")
print(f"Average Lenght: {sum(len(name) for name in names)/len(names)}")

## build vocabulary
chars=sorted(list(set("".join(names))))

stoi={ch:i+1 for i, ch in enumerate(chars)}
stoi["."]=0

itos={i:ch for ch, i in stoi.items()}
vocab_size=len(stoi)

print("Vocabulary Size:", vocab_size)
print(stoi)

## Bigram count matrix
N=torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

## Fill the Count matrix
for name in names:
    chs=["."]+list(name)+["."]
    
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        
        N[ix1, ix2]+=1
        

## visualize the Count matrix
plt.figure(figsize=(12, 12))
plt.imshow(N, cmap="Blues")

for i in range(vocab_size):
    for j in range(vocab_size):
        chstr=itos[i]+itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
        
plt.axis("off")
plt.show()
    
### Convert Counts into Probabilities
P=(N+1).float()
P/=P.sum(dim=1, keepdim=True)


### Generate Names
g = torch.Generator().manual_seed(2147483647)

print("BIGRAM Samples names:")
for _ in range(20):
    out=[]
    
    ix=0
    
    while True:
        p=P[ix]
        
        ix=torch.multinomial(
            p, num_samples=1, 
            replacement=True,
            generator=g
        ).item()
        
        out.append(itos[ix])
        
        if ix==0:
            break
        
        
    print("".join(out))
    
    
### TRIGRAM

trigram_counts = defaultdict(lambda: torch.zeros(vocab_size, dtype=torch.int32))
for name in names:

    chs = [".", "."] + list(name) + ["."]

    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):

        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]

        trigram_counts[(ix1, ix2)][ix3] += 1
        
        
## Convert counts to Probabilities
trigram_probs = {}

for context, counts in trigram_counts.items():

    probs = (counts + 1).float()
    probs /= probs.sum()

    trigram_probs[context] = probs
 
 
###  Generate Names Using the Trigram Model 

g = torch.Generator().manual_seed(2147483647)

print("TRIGRAM Sample names:")
for _ in range(20):
    out2=[]
    
    ix1=0
    ix2=0
    
    while True:
        context=(ix1, ix2)
        
        if context not in trigram_probs:
            break
        
        p=trigram_probs[context]
        
        ix3 = torch.multinomial(
            p,
            num_samples=1,
            replacement=True,
            generator=g
        ).item()
        
        if ix3==0:
            break
        
        out2.append(itos[ix3])
        
        ix1=ix2
        ix2=ix3
        
        
    print("".join(out2))
    
