import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import time
import os

from model import NameGenerator     

### Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Config — change these to experiment 
CONFIG = {
    "block_size": 5,
    "n_embd":     32,
    "n_hidden":   256,
    "n_layers":   2,
    "optimizer":  "adam",
    "lr":         1e-3,
    "epochs":     50,
    "batch_size": 64,
    "dropout":    0.2,
}


## Data loading
def load_data(path):
    names=open(path).read().splitlines()
    vocab=sorted(set("".join(names)+"."))
    
    stoi={ch:i for i, ch in enumerate(vocab)}
    itos={i:ch for i, ch in enumerate(vocab)}
    
    print(f"Names: {len(names):,} | Vocab size: {len(vocab)}")
    return names, vocab, stoi, itos


## Dataset
class NamesDataset(Dataset):
    def __init__(self, names_list, block_size, stoi):
        X, Y=[], []
        
        for name in names_list:
            context=[0]*block_size
            for ch in name+".":
                ix=stoi[ch]
                X.append(context)
                Y.append(ix)
                context=context[1:]+[ix]
                
        self.X=torch.tensor(X, dtype=torch.long)
        self.Y=torch.tensor(Y, dtype=torch.long)
        
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    



###### Train function
def train(model, config, train_loader, val_loader):
    model=model.to(device)
    
    if config["optimizer"]=='adam':
        optimizer=torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
        
    scheduler     = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    train_losses  = []
    val_losses    = []
    best_val_loss = float("inf")
    t_start       = time.time()
    
    for epoch in range(1, config["epochs"]+1):
        
        ## training
        model.train()
        total_loss=0.0
        
        for X_batch, Y_batch in train_loader:
            X_batch=X_batch.to(device)
            Y_batch=Y_batch.to(device)
            loss=F.cross_entropy(model(X_batch), Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            
        
        avg_train=total_loss/len(train_loader)
        train_losses.append(avg_train)
        
        
        ## validation
        model.eval()
        total_val=0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                total_val += F.cross_entropy(model(X_batch), Y_batch).item()
                
        avg_val=total_val/len(val_loader)
        val_losses.append(avg_val)
        
        if avg_val<best_val_loss:
            best_val_loss=avg_val
            
        scheduler.step()
        print(f"epoch {epoch:3d} | train {avg_train:.4f} | val {avg_val:.4f}", flush=True)
        
    
    elapsed=time.time()-t_start
    
    print(f"\nDone in {elapsed:.1f}s  |  best val: {best_val_loss:.4f}", flush=True)

    return {
        "train_losses":  train_losses,
        "val_losses":    val_losses,
        "best_val_loss": best_val_loss,
        "training_time": elapsed
    }
    


#### Save
def save_model(model, config, result, stoi, itos, vocab_size, 
               path="saved_models/best_model.pt"):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
        "model_state": model.state_dict(),
        "config":      config,
        "vocab": {
            "stoi":       stoi,
            "itos":       itos,
            "vocab_size": vocab_size
        },
        "result": {
            "best_val_loss": result["best_val_loss"],
            "train_losses":  result["train_losses"],
            "val_losses":    result["val_losses"],
        }
    }, path)
    
    mb = os.path.getsize(path) / 1e6
    print(f"Saved → {path}  ({mb:.2f} MB)")



### Plot

def plot(result, config, model):
    fig, axes=plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(result["train_losses"], label="Train Loss",  color="#4fc3f7", lw=2)
    axes[0].plot(result["val_losses"],   label="Val Loss",   color="#ff8a65", lw=2, linestyle="--")
    axes[0].set_title("Train & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].axis("off")
    
    lines=[f"{k:<15} {v}" for k, v in config.items()]
    
    lines += ["",
              f"{'best_val_loss':<15} {result['best_val_loss']:.4f}",
              f"{'training_time':<15} {result['training_time']:.1f}s",
              f"{'parameters':<15} {sum(p.numel() for p in model.parameters()):,}"]
    
    axes[1].text(0.1, 0.9, "\n".join(lines),
                 transform=axes[1].transAxes,
                 fontsize=12, verticalalignment="top",
                 fontfamily="monospace")
    
    axes[1].set_title("Model Config")
    
    plt.tight_layout()
    
    os.makedirs("saved_models", exist_ok=True)
    plt.savefig("saved_models/training_result.png", dpi=150)
    plt.show()
    print("Plot saved")

    
    


#### Main

if __name__=="__main__":
    
    ## load data
    names, vocab, stoi, itos=load_data("/Users/mac/Desktop/Machine Learning/DL/MyModels/MyApp/DB/Names/names.txt")
    vocab_size=len(vocab)
    
    # split
    random.seed(42)
    random.shuffle(names)
    n1 = int(0.8 * len(names))
    n2 = int(0.9 * len(names))
    
    train_ds = NamesDataset(names[:n1], CONFIG["block_size"], stoi)
    val_ds   = NamesDataset(names[n1:n2], CONFIG["block_size"], stoi)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
    
    ## Build model
    model = NameGenerator(
        vocab_size = vocab_size,
        n_embd     = CONFIG["n_embd"],
        block_size = CONFIG["block_size"],
        n_hidden   = CONFIG["n_hidden"],
        n_layers   = CONFIG["n_layers"],
        dropout    = CONFIG["dropout"]
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    ## Train
    result=train(model, CONFIG, train_loader, val_loader)
    
    ## Save
    save_model(model, CONFIG, result, stoi, itos, vocab_size)
    
    ## Plot
    plot(result, CONFIG, model)
    



    

  
   
    