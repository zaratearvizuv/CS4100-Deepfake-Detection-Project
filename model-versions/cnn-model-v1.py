# CNN Model Skeleton for transfer learning

# Step 1: Importing Libraries and Load Data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# If we used the csv file, then we would've had to manually parse data
# CSV more useful once we need the metadata
# Using the filepath is more efficient in it being automatically done
CNN_FLICKR_DATASET = "datasets/flickr-gan-dataset/real_vs_fake/real-vs-fake"

# Hyperparameters
# Batch size determines how many pictures go in per training epoc
BATCH_SIZE = 32
# Epochs are a training session. Too much causes overfitting
EPOCHS = 10
# Learning Rate is the step size the model takes to learn the optimum
# Too high jumps over optimums. Too low takes too long to find the optimum
LR = 0.001
# Uses the GPU if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Preprocess the Data

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(CNN_FLICKR_DATASET, "train"), transform=transform)
valid_dataset = datasets.ImageFolder(os.path.join(CNN_FLICKR_DATASET, "valid"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(CNN_FLICKR_DATASET, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Step 3: Define the CNN Architecture

# Load pre-trained ResNet and modify for binary classification
model = models.resnet50(pretrained=True)

# Freeze early layers
# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for binary classification (fake vs real)
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)

model = model.to(DEVICE)

# Step 4: Set up loss and optimizer

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# Training function
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i % 100 == 0:
            print(f"  Batch [{i}/{len(loader)}] Loss: {loss.item():.4f}")
    
    return running_loss / len(loader), 100. * correct / total

# Validation function
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# Step 5: Train the Model

# Training loop
print("\nStarting training...")
best_acc = 0.0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    valid_loss, valid_acc = validate(model, valid_loader, criterion)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
    
    # Save best model
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), "cnn_detector_best.pth")
        print(f"  Saved best model (acc: {best_acc:.2f}%)")

# Step 6: Evaluate the Model

# Test the model
print("\nTesting...")
model.load_state_dict(torch.load("cnn_detector_best.pth"))
test_loss, test_acc = validate(model, test_loader, criterion)
print(f"Test Accuracy: {test_acc:.2f}%")

print("\nDone! Model saved as 'cnn_detector_best.pth'")