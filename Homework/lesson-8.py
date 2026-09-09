import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## Load Dataset
data=load_breast_cancer()

X=data.data # type:ignore
y=data.target # type:ignore

print("X shape:", X.shape)
print("y shape:", y.shape)

## Test Train Split
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

## Scaling
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

## Convert to tensor
X_train=torch.tensor(X_train, dtype=torch.float32)
X_test=torch.tensor(X_test, dtype=torch.float32)

y_train=torch.tensor(y_train, dtype=torch.long)
y_test=torch.tensor(y_test, dtype=torch.long)


### DataLoader
train_loader=DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)
test_loader=DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=32,
    shuffle=False
)

### Custom Dropout layer
class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p=p
        
    def forward(self, x):
        if self.training:
            mask=(torch.rand_like(x)>self.p).float()
            return x*mask/(1-self.p)
        
        else:
            return x
        
### MLP Without Dropout
class MLP_NoDropout(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network=nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        return self.network(x)
    
### MLP with Custom Dropout
class MLP_CustomDropout(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network=nn.Sequential(
            nn.Linear(30,64),
            nn.ReLU(),
            CustomDropout(0.5),

            nn.Linear(64,32),
            nn.ReLU(),
            CustomDropout(0.5),

            nn.Linear(32,2)
        )
        
    def forward(self, x):
        return self.network(x)


### MLP with nn.Dropout

class MLP_PyTorchDropout(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network=nn.Sequential(
            nn.Linear(30,64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32,2)
        )
        
    def forward(self, x):
        return self.network(x)
    


### Training Function
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion=nn.CrossEntropyLoss()

def train_model(model, epochs=30):
    model.to(device)
    
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        
        total_loss=0
        
        for X_batch, y_batch in train_loader:
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            outputs=model(X_batch)
            
            loss=criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss+=loss.item()
            
        print(f"Epoch {epoch+1:2d} Loss: {total_loss/len(train_loader):.4f}")
        

### Evaluation Function

def evaluate(model):
    model.eval()
    
    correct=0
    total=0
    loss_sum=0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs=model(X_batch)
            
            loss=criterion(outputs, y_batch)
            
            loss_sum+=loss.item()
            
            prediction=outputs.argmax(1)
            
            correct+=(prediction==y_batch).sum().item()
            
            total+=y_batch.size(0)
            
    accuracy=100*correct/total
    loss=loss_sum/len(test_loader)
    
    return loss, accuracy


### Train all model
print("Model WITHOUT Dropout")
model1 = MLP_NoDropout()
train_model(model1)


print("Model with CUSTOM Dropout")
model2 = MLP_CustomDropout()
train_model(model2)


print("Model with nn.Dropout")
model3 = MLP_PyTorchDropout()
train_model(model3)


### Evaluate
loss1, acc1 = evaluate(model1)
loss2, acc2 = evaluate(model2)
loss3, acc3 = evaluate(model3)

print("RESULTS")
print(f"No Dropout       Loss={loss1:.4f} Accuracy={acc1:.2f}%")
print(f"Custom Dropout   Loss={loss2:.4f} Accuracy={acc2:.2f}%")
print(f"nn.Dropout       Loss={loss3:.4f} Accuracy={acc3:.2f}%")