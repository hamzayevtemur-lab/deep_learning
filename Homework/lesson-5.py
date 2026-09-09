import torch
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# load dataset
housing=fetch_california_housing()

X=housing.data
y=housing.target

print("X Shape:",X.shape)
print("y Shape:",y.shape)

## Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

## Scaller
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


## Convert to PyTorch Tensors
X_train=torch.tensor(X_train, dtype=torch.float32)
X_test=torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


## Trainable Parameters
W1 = torch.randn(8, 32, requires_grad=True)
b1 = torch.zeros(32, requires_grad=True)

W2 = torch.randn(32, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

## model
def model(x):
    hidden=torch.relu(x@W1+b1)
    output=hidden@W2 + b2
    return output


# hyperparameters
learning_rate = 0.001
epochs = 200
batch_size = 64

num_samples=X_train.shape[0]

for epoch in range(epochs):
    permutation=torch.randperm(num_samples)
    
    X_train = X_train[permutation]
    y_train = y_train[permutation]
    
    epoch_loss=0
    
    for i in range(0, num_samples, batch_size):
        X_batch=X_train[i:i+batch_size]
        y_batch=y_train[i:i+batch_size]
        
        pred=model(X_batch)
        
        loss=((pred-y_batch)**2).mean()
        
        loss.backward()
        
        with torch.no_grad():
            W1 -= learning_rate * W1.grad
            b1 -= learning_rate * b1.grad

            W2 -= learning_rate * W2.grad
            b2 -= learning_rate * b2.grad
            
        
        W1.grad.zero_()
        b1.grad.zero_()

        W2.grad.zero_()
        b2.grad.zero_()
        
        epoch_loss+=loss.item()
        
        
    if (epoch+1)%20==0:
        print(f"Epoch {epoch+1}/{epochs} Loss={epoch_loss:.4f}")
        
        
        
## Evaluate
with torch.no_grad():
    pred=model(X_test)
    
y_true=y_test.numpy()
y_pred=pred.numpy()

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
    
print(f"MSE : {mse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R2  : {r2:.4f}")