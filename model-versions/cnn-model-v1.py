# CNN Model Skeleton for transfer learning





# Step 1: Importing Libraries and Load Data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# Creates folder with results with automatic renaming
def experiments_folder(base_name="cnn-model-v1-experiment"):
    results_path = "experiments"
    os.makedirs(results_path, exist_ok=True)
    
    experiment_num = 1
    while os.path.exists(os.path.join(results_path, f"{base_name}-{experiment_num}")):
        experiment_num += 1
    
    folder_path = os.path.join(results_path, f"{base_name}-{experiment_num}")
    os.makedirs(folder_path)
    return folder_path

RESULTS_FOLDER = experiments_folder()
print(f"Results saved to: {RESULTS_FOLDER}")

# If we used the csv file, then we would've had to manually parse data
# CSV more useful once we need the metadata
# Using the filepath is more efficient in it being automatically done
CNN_FLICKR_DATASET = os.path.join("datasets", "flickr-gan-dataset", "real_vs_fake", "real-vs-fake")

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

PRE_TRAINED_BOOL = True

# Step 2: Preprocess the Data

# Transform prepares images before sending off to model
# .compose means make images with these changes
transform = transforms.Compose([
    # Resizes image resolution to 224x224. ResNet expects this input size
    transforms.Resize((224, 224)),
    # Tensor ?
    transforms.ToTensor(),
    # Normalize ?
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Applies transform on the images in the dataset
# ImageFolder puts images in a tuple (tensor_image? , 0 for fake and 1 for real image)
train_dataset = datasets.ImageFolder(os.path.join(CNN_FLICKR_DATASET, "train"), transform=transform)
valid_dataset = datasets.ImageFolder(os.path.join(CNN_FLICKR_DATASET, "valid"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(CNN_FLICKR_DATASET, "test"), transform=transform)

# DataLoader(dataset, batch size, shuffle images)
# Feeds the datasets into the model by grabbing batch size images per epoch
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Step 3: Define the CNN Architecture

# Load pre-trained ResNet with transfer learning
model = models.resnet50(pretrained=PRE_TRAINED_BOOL)

# Freezes layers, which keeps the trained layers
for param in model.parameters():
    # controls if a layer learns/updates during training
    # We dont need to train it since we're using its knowledge for prediction
    param.requires_grad = False

# Replace final layer to prepare it for prediction
# Has requires_grad set to true by default
# Performs a straight linear feature search, then curve, then randomly loses neurons,
# then linear feature search again
# Learns more complex relationships with more without overfitting
# More complexity layers means can learn more patterns but risk overfitting.
# More layers are needed if accuracy low, more classification, a lot of data
model.fc = nn.Sequential(
    # 2048 inputs, 512 outputs
    # 2048 inputs since ResNet50's last layer outputs 2048 features
    # Features are patterns or characteristics the model detects in an image
    # Features in early layers: lines, color
    # Features in later layers: faces, hair, nose
    # Performs linear math operation: output = (input * weight) + bias
    nn.Linear(2048, 512),
    # Activation function
    # Rectified Linear Unit: Replaces negative numbers with 0
    # Replacing negatives with 0 as a way to indicate something is not there
    # We only care about something being there (above 0) or not (equal 0)
    # Like saying something is more than not there
    # Also learns curving patterns rather than just linear
    nn.ReLU(), 
    # 30% dropout to prevent overfitting
    # 30% of neurons randomly turned off each batch
    nn.Dropout(0.3),
    # 512 inputs, 2 outputs
    nn.Linear(512, 2)
)

# Moves model to GPU or CPU so it can run there
# Data and model must be on the same device to work together
model = model.to(DEVICE)

# Step 4: Set up loss and optimizer

# Criterion/Loss Function
# CrossEntropyLoss is how it measures predicitions
# Close score to classification is a low loss number
# Far score to classification is a high loss number
# Used for calculating error so optimizer knows how to adjust weights
criterion = nn.CrossEntropyLoss()
# How the model updates weights to reduce loss
# Weights come in: input * weight = output or image pixels * weight = prediction
# Weights change to get a correct prediction
# Adam (Adaptive Moment Estimation) is an algorithm that decides how to adjust the weights
# model.fc.parameters() only changes the final layer weights since we froze the previous layers
# Learning rate is how much the weights change/step size
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# Step 5: Train the Model

# Training function
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(loader):
        # Moves model to GPU or CPU so it can run there
        # Data and model must be on the same device to work together
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
            print(f"\rBatch [{i}/{len(loader)}] Loss: {loss.item():.4f}", end="")
    
    print("\nCurrent batch training complete")
    
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

# Training loop
best_acc = 0.0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    valid_loss, valid_acc = validate(model, valid_loader, criterion)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
    
    # Save best model
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), os.path.join(RESULTS_FOLDER, "model.pth"))
        print(f"Saved best model (acc: {best_acc:.2f}%)")

# Step 6: Evaluate the Model

# Test the model
print("\nTesting Model")
model.load_state_dict(torch.load(os.path.join(RESULTS_FOLDER, "model.pth")))
test_loss, test_acc = validate(model, test_loader, criterion)
print(f"Test Accuracy: {test_acc:.2f}%")

# Save results to file
results = f"""CNN ResNet Results

Test Accuracy: {test_acc:.2f}%
Best Validation Accuracy: {best_acc:.2f}%

Hyperparameters:
- Epochs: {EPOCHS}
- Batch Size: {BATCH_SIZE}
- Learning Rate: {LR}
- Pre-trained: {PRE_TRAINED_BOOL}

Dataset: {CNN_FLICKR_DATASET}
"""

with open(os.path.join(RESULTS_FOLDER, "results.txt"), "w") as f:
    f.write(results)

print(f"Results saved to {RESULTS_FOLDER}")