import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import Model as KerasModel

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


## Load Dataset
(x_train, y_train),(x_test, y_test)=fashion_mnist.load_data()

print("X train shape:",x_train.shape)
print("Y train shape:",y_train.shape)
print("X test shape:",x_test.shape)
print("Y test shape:",y_test.shape)

## Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


class_names = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Boot"
]



#### Functional API Model
inputs=Input(shape=(28, 28))

x=Flatten()(inputs)

x=Dense(256, activation="relu")(x)
x=Dropout(0.3)(x)

x=Dense(128, activation="relu")(x)
x=Dropout(0.3)(x)

x=Dense(64, activation="relu")(x)

outputs=Dense(10, activation="softmax")(x)

functional_model=Model(inputs, outputs)


### Compile 
functional_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

## Train
history_functional=functional_model.fit(
    x_train, y_train, epochs=20, batch_size=128, validation_split=0.2
)



#### Model Subclassing API

class FashionClassifier(KerasModel):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.flatten=Flatten()
        
        self.dense1 = Dense(256, activation="relu")
        self.dropout1 = Dropout(0.3)

        self.dense2 = Dense(128, activation="relu")
        self.dropout2 = Dropout(0.3)

        self.dense3 = Dense(64, activation="relu")

        self.output_layer = Dense(10, activation="softmax")
        
    
    def call(self, inputs):
        x = self.flatten(inputs)

        x = self.dense1(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.dropout2(x)

        x = self.dense3(x)

        return self.output_layer(x)
    
    
## Create model
subclass_model=FashionClassifier()

dummy = tf.zeros((1,28,28))
subclass_model(dummy)

subclass_model.summary()

## Compile
subclass_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

## Train
history_subclass=subclass_model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2
)



### Evaluation Function

def evaluate_model(model, x_test, y_test):
    prob=model.predict(x_test)
    pred=np.argmax(prob, axis=1)
    
    accuracy=accuracy_score(y_test, pred)
    
    precision = precision_score(
        y_test,
        pred,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        pred,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        pred,
        average="weighted"
    )
    
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    
    print(classification_report(
        y_test, pred, target_names=class_names
    ))
    
    cm=confusion_matrix(y_test, pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True,
                fmt="d", cmap="Blues")
    
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    plt.show()
    
    
    
## Evaluate Both Models

print("Functional API")
evaluate_model(
    functional_model,
    x_test,
    y_test
)


print("Subclass API")
evaluate_model(
    subclass_model,
    x_test,
    y_test
)
    
    
# Save Models
functional_model.save("functional_model.keras")

subclass_model.save("subclass_model.keras")
    
    

### Load saved model
model=tf.keras.models.load_model("functional_model.keras")



### Load dataset
(_, _), (x_test, y_test) = fashion_mnist.load_data()

x_test = x_test.astype("float32") / 255.0

## Predict one image

index=25
image=np.expand_dims(x_test[index], axis=0)
prediction=model.predict(image)

predicted_class=np.argmax(prediction)

print("Predicted Class Functional API:", class_names[predicted_class])
print("Actual Class   Functional AP :", class_names[y_test[index]])


### Load Subclass Model
subclass_model=tf.keras.models.load_model(
    "subclass_model.keras",
    custom_objects={"FashionClassifier": FashionClassifier}
)

idx=10
image2=np.expand_dims(x_test[idx], axis=0)
prediction_sub=subclass_model.predict(image2)

pred_class=np.argmax(prediction_sub)


print("Predicted Class Subclass API :", class_names[pred_class])
print("Actual Class  Subclass API   :", class_names[y_test[index]])