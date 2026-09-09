import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from unidecode import unidecode

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

DATA_PATH = "/Users/mac/Desktop/Machine Learning/DL/DB/Names2"

names = []
labels = []
class_names = []

for label, filename in enumerate(sorted(os.listdir(DATA_PATH))):
    if not filename.endswith(".txt"):
        continue
    
    class_name=filename.replace(".txt", "")
    class_names.append(class_name)
    
    file_path=os.path.join(DATA_PATH, filename)
    
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            name=line.strip().lower()
            
            if not name:
                continue
            
            name = unidecode(name)
            name = name.replace(",", "")
            name = name.replace("1", "")
            name = name.replace("/", "")
            name = name.replace(":", "")
            name = name.replace("-", " ")
            
            name = name.strip()
            
            if len(name)>=2:
                names.append(name)
                labels.append(label)
                

print("Total Samples:", len(names))
print("Number of Classes:", len(class_names))   


#### Create the character vocabulary
characters = sorted(set("".join(names)))

pad_token = "<pad>"
eos_token = "<eos>"

pad_idx = 0
eos_idx = 1

char_to_idx = {
    char: idx + 2
    for idx, char in enumerate(characters)
}

char_to_idx[pad_token] = pad_idx
char_to_idx[eos_token] = eos_idx

idx_to_char = {
    idx: char
    for char, idx in char_to_idx.items()
}

vocab_size = len(char_to_idx)

print("Vocabulary Size:", vocab_size)
print("Padding Index:", pad_idx)
print("EOS Index:", eos_idx)
    
    
### Convert names into character sequences
encoded_names=[]

for name in names:
    encoded=[
        char_to_idx[char]
        for char in name
        if char in char_to_idx
    ]
    
    encoded_names.append(encoded)
    
    
### Split the names into training and testing sets
train_names, test_names = train_test_split(
    encoded_names,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", len(train_names))
print("Testing Samples:", len(test_names))



##  Create input and target sequences (Dataset Class)
class CharSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences=sequences
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence=self.sequences[idx]
        
        input_sequence = sequence
        target_sequence = sequence[1:] + [eos_idx]
        
        return input_sequence, target_sequence
    



### Custom Collate Function
def collate_fn(batch):
    input_sequences=[item[0] for item in batch]
    target_sequences=[item[1] for item in batch]
    
    max_length=max(len(sequence) for sequence in input_sequences)
    
    padded_inputs=torch.full(
        (len(batch), max_length),
        pad_idx,
        dtype=torch.long
    )
    
    padded_targets = torch.full(
        (len(batch), max_length),
        pad_idx,
        dtype=torch.long
    )
    
    for i, (input_sequence, target_sequence) in enumerate(batch):
        padded_inputs[i,:len(input_sequence)]=torch.tensor(
            input_sequence, dtype=torch.long
        )
        
        padded_targets[i, :len(target_sequence)]=torch.tensor(
            target_sequence,dtype=torch.long
        )
        
    return padded_inputs, padded_targets


## Dataset and Dataloder
train_dataset = CharSequenceDataset(train_names)
test_dataset = CharSequenceDataset(test_names)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn
)


print("Train Batches:", len(train_loader))
print("Test Batches:", len(test_loader))


## Test one Batch
sample_inputs, sample_targets = next(iter(train_loader))

print("Input Shape:", sample_inputs.shape)
print("Target Shape:", sample_targets.shape)


### RNN
class CharacterRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super().__init__()
        
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        
        self.embedding=nn.Parameter(
            torch.randn(vocab_size, embedding_dim)*0.1
        )
        
        self.W_xh = nn.Parameter(
            torch.randn(embedding_dim, hidden_dim) * 0.1
        )

        self.W_hh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim) * 0.1
        )

        self.b_h = nn.Parameter(
            torch.zeros(hidden_dim)
        )

        self.W_hy = nn.Parameter(
            torch.randn(hidden_dim, vocab_size) * 0.1
        )

        self.b_y = nn.Parameter(
            torch.zeros(vocab_size)
        )
        
    def forward(self, x):
        batch_size, sequence_length=x.shape
        
        hidden=torch.zeros(
            batch_size, 
            self.hidden_dim,
            device=x.device
        )
        
        outputs=[]
        
        for t in range(sequence_length):
            current_char=x[:, t]
            
            x_t=self.embedding[current_char]
            
            hidden=torch.tanh(
                x_t @ self.W_xh+hidden @ self.W_hh + self.b_h
            )
            output=(
                hidden@self.W_hy+self.b_y
            )
            
            outputs.append(output)
            
        outputs=torch.stack(outputs, dim=1)
        
        return outputs
    
    


### Initialize the model
model = CharacterRNN(
    vocab_size=vocab_size,
    embedding_dim=32,
    hidden_dim=64
).to(device)

print(model)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Total Parameters:", total_parameters)


# Test the forward pass
sample_inputs = sample_inputs.to(device)
sample_targets = sample_targets.to(device)

outputs = model(sample_inputs)

print("Model Output Shape:", outputs.shape)



        
## Loss function
criterion=nn.CrossEntropyLoss(
    ignore_index=pad_idx
)

## Define optimizer
optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


### Training loop
num_epochs=20

train_losses=[]
test_losses=[]

for epoch in range(num_epochs):
    model.train()
    
    total_train_loss=0.0
    
    for inputs, targets in train_loader:
        inputs=inputs.to(device)
        targets=targets.to(device)
        
        optimizer.zero_grad()
        
        outputs=model(inputs)
        
        loss=criterion(
            outputs.reshape(-1, vocab_size),
            targets.reshape(-1)
        )
        
        loss.backward()
        
        optimizer.step()
        
        total_train_loss+=loss.item()
        
    average_train_loss=total_train_loss/len(train_loader)
    
    model.eval()
    
    total_test_loss=0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs=inputs.to(device)
            targets=targets.to(device)
            
            outputs=model(inputs)
            
            loss=criterion(
                outputs.reshape(-1, vocab_size),
                targets.reshape(-1)
            )
            
            total_test_loss+=loss.item()
            
    average_test_loss=total_test_loss/len(test_loader)
    
    train_losses.append(average_train_loss)
    test_losses.append(average_test_loss)
    
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {average_train_loss:.4f} "
        f"Test Loss: {average_test_loss:.4f}"
    )
    
    


### Generate New names
def generate_name(model, start_char=None, max_length=20):
    model.eval()
    
    if start_char is None:
        valid_chars=[
            char for char in char_to_idx
            if char not in [pad_token, eos_token]
        ]
        
        start_char=random.choice(valid_chars)
        
    current_idx=char_to_idx[start_char]
    
    generated_name=start_char
    
    hidden=torch.zeros(
        1, model.hidden_dim, device=device
    )
    
    with torch.no_grad():
        for _ in range(max_length):
            current_input=torch.tensor(
                [[current_idx]],
                dtype=torch.long,
                device=device
            )
            
            current_char_embedding=model.embedding[
                current_input[:, 0]
            ]
            hidden=torch.tanh(
                current_char_embedding @ model.W_xh+hidden@model.W_hh+model.b_h
            )
            
            logits=(
                hidden@model.W_hy+model.b_y
            )
            
            probabilities=torch.softmax(
                logits, dim=-1
            )
            
            next_idx=torch.multinomial(
                probabilities, num_samples=1
            ).item()
            
            if next_idx==eos_idx:
                break
            
            if next_idx==pad_idx:
                break
            
            next_char=idx_to_char[next_idx] # type: ignore
            
            generated_name+=next_char
            
            current_idx=next_idx
            
    return generated_name



### Generated Some Names 
print("\nGenerated Names:")
for _ in range(10):
    print(generate_name(model))
    
  
### Names starting Char A  
print("\nNames starting with A:")
for _ in range(5):
    print(generate_name(model, start_char="a"))