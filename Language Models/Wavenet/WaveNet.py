import math
import random
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

## Device
device=torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", device)

torch.manual_seed(42)
random.seed(42)


# Hyperparameters
BLOCK_SIZE = 128
BATCH_SIZE = 32
EMBED_DIM = 128
HIDDEN_CHANNELS = 128
NUM_BLOCKS = 6
LEARNING_RATE = 3e-4
EPOCHS = 5
VOCAB_SIZE = 50257
MODEL_PATH = "wavenet_tinystories.pt"

train_size=5000
valid_size=1000

## Load TinyStories Dataset
print("Loading TinyStories dataset------")
dataset=load_dataset("roneneldan/TinyStories")

train_data = dataset["train"].select(range(train_size))
validation_data = dataset["validation"].select(range(valid_size))

print("Training Stroies:", len(train_data))
print("Validation Stories:", len(validation_data))


### GPT-2 Tokenizer
enc=tiktoken.get_encoding("gpt2")


### Custom Dataset
class TinyStoriesDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, block_size=128):
        self.block_size=block_size
        print("Tokenizing dataset...")
        
        all_tokens=[]
        
        for sample in hf_dataset:
            tokens=tokenizer.encode(sample['text'])
            
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.eot_token) ## end of the story
            
        self.tokens=torch.tensor(all_tokens, dtype=torch.long)
        
        print('Total tokens:', len(self.tokens))
        
    def __len__(self):
        return len(self.tokens)-self.block_size-1
    
    def __getitem__(self, idx):
        x=self.tokens[idx:idx+self.block_size]
        y=self.tokens[idx+1:idx+self.block_size+1]
        
        return x, y
    

### Dataset objects
train_dataset=TinyStoriesDataset(
    train_data, enc, BLOCK_SIZE
)

val_dataset=TinyStoriesDataset(
    validation_data, enc, BLOCK_SIZE
)


### DataLoaders
train_loader=DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    drop_last=True
)

val_loader=DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True
)

x, y=next(iter(train_loader))

print("Input shape:", x.shape)
print("Target shape:", y.shape)

print(x[0][:20])
print(y[0][:20])



#### Causal Convolution Layer
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        
        self.padding=(kernel_size-1)*dilation
        
        self.conv=nn.Conv1d(
            in_channels,out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        
    def forward(self, x):
        x=self.conv(x)
        
        # Remove extra padded values on the right 
        if self.padding > 0:
            return x[:, :, :-self.padding]
        
        return x
    
    


#### Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        
        self.conv=CausalConv1d(
            channels, 
            channels,
            kernel_size=2, 
            dilation=dilation
        )
        
        self.relu=nn.ReLU()
        
    
    def forward(self, x):
        residual=x
        out=self.conv(x)
        out=self.relu(out)
        out=out+residual
        
        return out    



#### WaveNet Class

class WaveNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_channels,
        num_blocks
    ):
        
        super().__init__()
        
        # Token embedding
        self.embedding=nn.Embedding(
            vocab_size, embed_dim
        )
        
        # Convert embedding dimension to hidden channels
        self.input_projection=nn.Conv1d(
            embed_dim,hidden_channels,kernel_size=1
        )
        
        # Residual blocks with exponentially increasing dilations
        dilations=[2**i for i in range(num_blocks)]
        
        self.residual_blocks=nn.ModuleList([
            ResidualBlock(
                hidden_channels, dilation=d
            ) 
            for d in dilations
        ])
        
        # Final 1x1 convolution 
        self.output_projection=nn.Conv1d(
            hidden_channels,hidden_channels, kernel_size=1
        )
        
        # vocabulary prediction
        self.fc=nn.Linear(
            hidden_channels, vocab_size
        )
    
    def forward(self, x):
        # Embedding
        x=self.embedding(x)
        
        # (B, L, C)-> (B, C, L)
        x=x.transpose(1, 2)
        
        # Project channels
        x=self.input_projection(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x=block(x)
            
        # Final projection
        x=self.output_projection(x)
        
        # (B,C,L) → (B,L,C)
        x = x.transpose(1, 2)
        
        # vocabulary prediction
        logits=self.fc(x)
        
        return logits
        

# Model
model=WaveNet(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_channels=HIDDEN_CHANNELS,
    num_blocks=NUM_BLOCKS
).to(device)

print(model)

# Test the model
x, y=next(iter(train_loader))
x=x.to(device)
logits=model(x)

print("Input shape:", x.shape)
print("Output shape:", logits.shape)



### Training the WaveNet
# Loss Function & Optimizer
criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)



## Train One Epoch
def train_one_epoch():
    model.train()
    
    total_loss=0
    
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
        
        total_loss+=loss.item()
        
    return total_loss/len(train_loader)


### Validation
def validate():
    model.eval()
    
    total_loss=0
    
    with torch.no_grad():
        for x, y in val_loader:
            x=x.to(device)
            y=y.to(device)
            
            logits=model(x)
            
            loss=criterion(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1)
            )
            
            total_loss+=loss.item()
            
    return total_loss/len(val_loader)

### Complete Training Loop
best_val_loss=float("inf")

train_losses=[]
val_losses=[]

for epoch in range(EPOCHS):
    
    train_loss=train_one_epoch()
    val_loss=validate()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(
        f"Epoch {epoch+1}/{EPOCHS}"
        f" | Train Loss: {train_loss:.4f}"
        f" | Val Loss: {val_loss:.4f}"
    )
    
    if val_loss<best_val_loss:
        best_val_loss=val_loss
        
        torch.save(
            model.state_dict(),
            MODEL_PATH
        )
        
        print("Model saved")
    
    
### Plot the Loss Curves
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("WaveNet Training")

plt.legend()

plt.show()


### load best model
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()


### Text Generation Function
@torch.no_grad()
def generate(prompt, max_new_tokens=100):
    model.eval()
    
    tokens=enc.encode(prompt)
    
    for _ in range(max_new_tokens):
        # Keep only the latest BLOCK_SIZE tokens
        context = tokens[-BLOCK_SIZE:]

        if len(context) < BLOCK_SIZE:
            context = [enc.eot_token] * (BLOCK_SIZE - len(context)) + context
        
        x=torch.tensor(
            context, dtype=torch.long,device=device
        ).unsqueeze(0)
        
        logits=model(x)
        
        # Prediction for the final position
        logits=logits[:, -1, :]
        
        probs=torch.softmax(logits, dim=-1)
        
        next_token=torch.multinomial(
            probs, num_samples=1
        ).item()
        
        tokens.append(next_token)
        
    return enc.decode(tokens)


## Generate Story
prompt=input("\nEnter a prompt:")

generated_text=generate(
    prompt, max_new_tokens=100
)

print("\nGenerated Story:\n")
print(generated_text)