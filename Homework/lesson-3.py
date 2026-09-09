import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

from sklearn.metrics import (
    accuracy_score, precision_score, f1_score,
    recall_score, confusion_matrix, classification_report
)


### Load Fashion MNIST Daraset
(x_train, y_train), (x_test, y_test)=fashion_mnist.load_data()

print("Training Images Shape :", x_train.shape)
print("Training Labels Shape :", y_train.shape)
print("Testing Images Shape  :", x_test.shape)
print("Testing Labels Shape  :", y_test.shape)


### Explore Dataset
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

print("Number of Classes:", len(class_names))



#### Display Sample Image
plt.figure(figsize=(10, 8))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()


#### Preprocess Data
# Normalize Images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Pixel Range:", x_train.min(), "to", x_train.max())




##### Building Sequential Model
model=Sequential([
    # Convert 28x28 image into 784 features
    Flatten(input_shape=(28, 28)),
    
    # Hidden Layer 1
    Dense(256, activation="relu"),
    Dropout(0.3),
    
    # Hidden layer 2
    Dense(128, activation="relu"),
    Dropout(0.3),
    
    # Hidden layer 3
    Dense(64, activation="relu"),
    
    # Output layer
    Dense(10, activation="softmax")
])

model.summary()

### Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

### Train model
history=model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)



#### Plot Accuracy and Loss

plt.figure(figsize=(12, 5))

## Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

## Loss
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

## Prediction
y_prob=model.predict(x_test)

y_pred=np.argmax(y_prob, axis=1)




#### Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)

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

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")



# Classification Report
print(classification_report(
    y_test,
    y_pred,
    target_names=class_names
))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.show()


### Train on Random Images
plt.figure(figsize=(12, 8))

indices=np.random.choice(len(x_test), 16)

for i, idx in enumerate(indices):
    plt.subplot(4, 4, i+1)
    plt.imshow(x_test[idx], cmap="gray")
    true_label = class_names[y_test[idx]]
    pred_label = class_names[y_pred[idx]]
    
    color = "green" if true_label == pred_label else "red"

    plt.title(f"T:{true_label}\nP:{pred_label}", color=color)

    plt.axis("off")
    
plt.tight_layout()
plt.show()


### Save Model
model.save("fashion_mnist_sequential.keras")
print("Model Saved Successfully!")
