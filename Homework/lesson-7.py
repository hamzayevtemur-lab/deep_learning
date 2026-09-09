import torch
import torch.nn as nn
import torch.optim as optim
import os
import scipy.io as sio
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


## Dataset path
data_dir="/Users/mac/Desktop/Machine Learning/DL/DB/102flowers"
image_dir=os.path.join(data_dir, "jpg")
label_file=os.path.join(data_dir, "imagelabels.mat")
setid_file=os.path.join(data_dir, "setid.mat")

## read matlab files
labels=sio.loadmat(label_file)
setid=sio.loadmat(setid_file)

print("Labels:",labels.keys())
print("ID:",setid.keys())

## Extract labels and indices
labels=labels["labels"].squeeze()
train_idx=setid["trnid"].squeeze()
valid_idx = setid["valid"].squeeze()
test_idx = setid["tstid"].squeeze()

print("Labels Len:",len(labels))
print("Train id Len:",len(train_idx))
print("validation id Len:",len(valid_idx))
print("Test id Len:",len(test_idx))

### Custom Dataset

class FlowerDataset(Dataset):
    def __init__(self, image_dir, labels, image_indeces):
        self.image_dir=image_dir
        self.labels=labels
        self.image_indeces=image_indeces
        
    def __len__(self):
        return len(self.image_indeces)
    
    def __getitem__(self, idx):
        
        # image id from MATLAB
        image_id=self.image_indeces[idx]
        
        # Image path
        image_name=f"image_{image_id:05d}.jpg"
        image_path=os.path.join(self.image_dir, image_name)
        
        # Read image
        image=Image.open(image_path).convert("RGB")
        
        ## Image Preprocessing
        image=image.resize((128, 128))
        
        # Convert PIl image to Tensor
        image=transforms.ToTensor()(image)
        
        # Normalize image
        image=transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)
        
        # Labels in Matlab start from 1
        label=int(self.labels[image_id-1])-1
        
        return image, label
    

## Dataset Object
train_dataset=FlowerDataset(
    image_dir,
    labels,
    train_idx
)

valid_dataset=FlowerDataset(
    image_dir,
    labels,
    valid_idx
)

test_dataset=FlowerDataset(
    image_dir,
    labels,
    test_idx
)

print("Training :", len(train_dataset))
print("Validation:", len(valid_dataset))
print("Testing :", len(test_dataset))     


### DataLoader

batch_size=32

train_loader=DataLoader(
    train_dataset, batch_size=batch_size,
    shuffle=True
)

valid_loader=DataLoader(
    valid_dataset, batch_size=batch_size,
    shuffle=False
)

test_loader=DataLoader(
    test_dataset, batch_size=batch_size,
    shuffle=False
)

# Verify DataLoader
images, labels=next(iter(train_loader))

print("Image Shape:", images.shape)
print("Labels Shape:", labels.shape)

## Display batch of images
classes = 102

fig, axes = plt.subplots(2, 4, figsize=(10, 6))

for i in range(8):

    img = images[i]

    img = img * torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    img = img + torch.tensor([0.485,0.456,0.406]).view(3,1,1)

    img = img.permute(1,2,0)

    axes[i//4, i%4].imshow(img.clamp(0,1))
    axes[i//4, i%4].set_title(f"Class {labels[i].item()+1}")
    axes[i//4, i%4].axis("off")

plt.tight_layout()
plt.show()


### Model

class FlowerCNN(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()
        
        # Feature Extraction
        self.features=nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 
            nn.Dropout2d(0.25),  
        )
        
        # Global Average Pooling
        self.pool=nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier=nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x=self.features(x)
        
        x=self.pool(x)
        
        x=self.classifier(x)
        
        return x
    

# Model  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FlowerCNN().to(device)

print(model)

## Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

## Training Function
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    
    running_loss=0
    correct=0
    total=0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc
        
## Evaulate Function
def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


### Training Loop

epochs = 20

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

best_accuracy = 0

for epoch in range(epochs):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, val_acc = evaluate(
        model,
        valid_loader,
        criterion,
        device
    )

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        torch.save(model.state_dict(), "best_flower_model.pth")

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )
    


## Test the model
test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    criterion,
    device
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.2f}%")

## Plot Training Curves
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accuracies, label="Train")
plt.plot(val_accuracies, label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()