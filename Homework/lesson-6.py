import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset, DataLoader


### Load and Preprocess Dataset
data=load_breast_cancer()

X=data.data
y=data.target
print("X Shape:", X.shape)
print("y shape:", y.shape)

# train-test split 
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

### Standardize features
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# Convert to tensor 
X_train=torch.tensor(X_train, dtype=torch.float32)
X_test=torch.tensor(X_test, dtype=torch.float32)

y_train=torch.tensor(y_train, dtype=torch.long)
y_test=torch.tensor(y_test, dtype=torch.long)

## Create TensorDatasets
train_set=TensorDataset(X_train, y_train)
test_set=TensorDataset(X_test, y_test)

## Data Loader
train_loader=DataLoader(
    train_set, batch_size=32, shuffle=True
)

test_loader=DataLoader(
    test_set, batch_size=32, shuffle=False
)

#### Custom Linear Layer

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.weight=nn.Parameter(
            torch.randn(out_features, in_features)* 0.01
        )
        self.bias=nn.Parameter(
            torch.zeros(out_features)
        )
        
    def forward(self, x):
        return x @ self.weight.t() + self.bias
    
    
## Compare Custom layer with nn.Linear
custom=CustomLinear(30, 16)
builtin=nn.Linear(30, 16)

x=torch.randn(5, 30)

print("Custom output shape:", custom(x).shape)
print("nn.Linear output shape:", builtin(x).shape)


### NN with custom Linear
class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network=nn.Sequential(
            CustomLinear(30, 64),
            nn.ReLU(),
            
            CustomLinear(64, 32),
            nn.ReLU(),
            
            CustomLinear(32, 2)
        )
        
    def forward(self, x):
        return self.network(x)
    
    
### NN with nn.Linear
class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)
    
    
### Train Function

def train_model(model, train_loader, val_loader, epochs=30):
    criterion=nn.CrossEntropyLoss()
    
    optimizer=optim.Adam(
        model.parameters(),
        lr=0.001
    )
    
    train_losses=[]
    test_losses=[]
    test_accuracy=[]
    
    for epoch in range(epochs):
        model.train()
        running_loss=0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            outputs=model(X_batch)
            loss=criterion(outputs, y_batch)
            loss.backward()
            
            optimizer.step()
            
            running_loss+=loss.item()
            
        train_loss=running_loss/len(train_loader)
        
        
        ## Validation
        model.eval()
        
        running_test_loss=0
        correct=0
        total=0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs=model(X_batch)
                loss=criterion(outputs, y_batch)
                
                running_test_loss+=loss.item()
                
                predictions=outputs.argmax(dim=1)
                
                correct+=(predictions==y_batch).sum().item()
                
                total+=y_batch.size(0)
                
        test_loss=running_test_loss/len(test_loader)
        accuracy=correct/total
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracy.append(accuracy)
        
        print(
            f"Epoch {epoch+1:2d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {test_loss:.4f} | "
            f"Val Acc: {accuracy:.4f}"
        )
        
    return train_losses, test_losses, test_accuracy

    
    
    
## Train models
print("Training Custom Network")
custom_model = CustomNet()

custom_history = train_model(
    custom_model,
    train_loader,
    test_loader
)


print("Training nn.Linear Network")
linear_model = LinearNet()

linear_history = train_model(
    linear_model,
    train_loader,
    test_loader
)           

## Compare final results
custom_train_loss = custom_history[0][-1]
custom_val_loss = custom_history[1][-1]
custom_acc = custom_history[2][-1]

linear_train_loss = linear_history[0][-1]
linear_val_loss = linear_history[1][-1]
linear_acc = linear_history[2][-1]

print("\nComparison Results")

print(f"{'Model':20} {'Train Loss':12} {'Val Loss':12} {'Val Acc'}")

print(f"{'Custom Linear':20} "
      f"{custom_train_loss:.4f} "
      f"{custom_val_loss:.4f} "
      f"{custom_acc:.4f}")

print(f"{'nn.Linear':20} "
      f"{linear_train_loss:.4f} "
      f"{linear_val_loss:.4f} "
      f"{linear_acc:.4f}")