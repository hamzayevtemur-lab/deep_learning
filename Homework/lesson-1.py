import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.neural_network import MLPClassifier



#### Load Dataset
digits=load_digits()

X=digits.data # type: ignore
y=digits.target # type: ignore

print("NUmber of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Number of classes:", len(np.unique(y)))

print("\nClass Labels:")
print(np.unique(y))


### Display
plt.figure(figsize=(10,4))

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X[i].reshape(8,8), cmap="gray")
    plt.title(f"Label: {y[i]}")
    plt.axis("off")
    
plt.tight_layout()
plt.show()


### Split Dataset
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


#### Feature Scaling
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


#### One-Hot Encoding
num_classes=len(np.unique(y))

y_train_onehot=np.eye(num_classes)[y_train]
y_test_onehot=np.eye(num_classes)[y_test]


## Shapes
print("\nTraining Data")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("One-hot:", y_train_onehot.shape)
print("\nTesting Data")
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)



#### Activation Functions
def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x>0).astype(float)

def softmax(x):
    x=x-np.max(x, axis=1, keepdims=True)
    
    exp=np.exp(x)
    
    return exp/np.sum(exp, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    epsilon=1e-12
    
    y_pred=np.clip(y_pred, epsilon, 1.0-epsilon)
    
    loss=-np.mean(np.sum(y_true*np.log(y_pred), axis=1))
    
    return loss


### FNN Class

class FNN:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, learning_rate=0.01):
        self.learning_rate=learning_rate
        
        # He initialization 
        self.W1=np.random.randn(input_size, hidden1_size)*np.sqrt(2/input_size)
        self.b1=np.zeros((1, hidden1_size))
        
        self.W2=np.random.randn(hidden1_size, hidden2_size)*np.sqrt(2/hidden1_size)
        self.b2=np.zeros((1, hidden2_size))
        
        self.W3=np.random.randn(hidden2_size, output_size)*np.sqrt(2/hidden2_size)
        self.b3=np.zeros((1, output_size))
        
        
    def forward(self, X):
        
        # layer 1
        self.Z1=X @ self.W1+self.b1
        self.A1=relu(self.Z1)
        
        # layer 2
        self.Z2=self.A1 @ self.W2 + self.b2
        self.A2=relu(self.Z2)
        
        # output layer
        self.Z3=self.A2 @ self.W3 + self.b3
        self.A3=softmax(self.Z3)
        
        return self.A3
    
    def backward(self, X, y_true):
        
        m=X.shape[0]
        
        # Output Layer
        dZ3=self.A3-y_true
        
        dW3=self.A2.T @ dZ3 /m
        db3=np.sum(dZ3, axis=0, keepdims=True) /m
        
        # Hidden layer 2
        dA2=dZ3 @ self.W3.T
        dZ2=dA2*relu_derivative(self.Z2)
        
        dW2=self.A1.T @ dZ2 / m
        db2=np.sum(dZ2, axis=0, keepdims=True)/ m
        
        # Hidden Layer 1
        dA1=dZ2 @ self.W2.T
        dZ1=dA1* relu_derivative(self.Z1)
        
        dW1=X.T @ dZ1 / m
        db1=np.sum(dZ1, axis=0, keepdims=True) / m
        
        ## Gradient Dexcent Update
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
    
    def predict(self, X):
        probabilities=self.forward(X)
        return np.argmax(probabilities, axis=1)
    
    
    def train(self, X, y, epochs=300):
        losses=[]
        
        for epoch in range(epochs):
            # forward pass
            predictions=self.forward(X)
            
            # Compute loss
            loss=cross_entropy(y, predictions)
            
            # Backward pass
            self.backward(X, y)
            losses.append(loss)
            
            if (epoch+1)%20==0:
                print(f" Epoch {epoch+1}/{epochs} Loss: {loss:.4f}")
                
            
        return losses
    


#### Model
model=FNN(
    input_size=64, 
    hidden1_size=128, 
    hidden2_size=64, 
    output_size=10,
    learning_rate=0.01
)

## Train the model
losses=model.train(
    X_train,
    y_train_onehot,
    epochs=300
)

### Plot the loss
plt.figure(figsize=(8, 5))

plt.plot(losses)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.grid(True)

plt.show()


### Test the model
y_pred=model.predict(X_test)

## Acuracy
accuracy=accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.4f}")

## Precision Recall and F1 score
precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


### Classification
print(classification_report(y_test, y_pred))

### Confusion Matrix
cm=confusion_matrix(y_test, y_pred)

disp=ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=digits.target_names # type:ignore
)

disp.plot(cmap="Blues")
plt.title("Custom FNN Confusion Matrix")

plt.show()


#### Sklearn MLPClassifier

mlp=MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="sgd",
    learning_rate_init=0.01,
    max_iter=300,
    random_state=42
)

# Train
mlp.fit(X_train, y_train)

# Prediction
mlp_pred=mlp.predict(X_test)

print(f"Sklearn MLP Accuracy : {accuracy_score(y_test, mlp_pred):.4f}")

print(classification_report(y_test,mlp_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, mlp_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=digits.target_names # type:ignore
)

disp.plot(cmap="Greens")

plt.title("MLPClassifier Confusion Matrix")

plt.show()


### Final Comparison
custom_accuracy = accuracy_score(y_test, y_pred)
custom_precision = precision_score(y_test, y_pred, average="weighted")
custom_recall = recall_score(y_test, y_pred, average="weighted")
custom_f1 = f1_score(y_test, y_pred, average="weighted")

mlp_accuracy = accuracy_score(y_test, mlp_pred)
mlp_precision = precision_score(y_test, mlp_pred, average="weighted")
mlp_recall = recall_score(y_test, mlp_pred, average="weighted")
mlp_f1 = f1_score(y_test, mlp_pred, average="weighted")

print(f"{'Metric':<15}{'Custom FNN':<15}{'MLPClassifier'}")
print(f"{'Accuracy':<15}{custom_accuracy:.4f}{'':<7}{mlp_accuracy:.4f}")
print(f"{'Precision':<15}{custom_precision:.4f}{'':<7}{mlp_precision:.4f}")
print(f"{'Recall':<15}{custom_recall:.4f}{'':<7}{mlp_recall:.4f}")
print(f"{'F1-score':<15}{custom_f1:.4f}{'':<7}{mlp_f1:.4f}")
