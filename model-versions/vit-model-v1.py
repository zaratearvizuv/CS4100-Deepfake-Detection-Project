# ViT deepfake detection on 140k Real and Fake Faces (same dataset train and test)
# Uses vit_base_patch16_224 from timm, fine tuned with BCEWithLogitsLoss
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time

# Dirs (140k Real and Fake Faces)
DATASET_ROOT = os.path.join("datasets", "140k-real-and-fake-faces")
IMAGE_ROOT = os.path.join(DATASET_ROOT, "real_vs_fake", "real-vs-fake")
TRAIN_CSV = os.path.join(DATASET_ROOT, "train.csv")
VALID_CSV = os.path.join(DATASET_ROOT, "valid.csv")
TEST_CSV = os.path.join(DATASET_ROOT, "test.csv")

BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Experiment folder
def experiments_folder(base_name):
    results_path = "experiments"
    os.makedirs(results_path, exist_ok=True)
    experiment_num = 1
    while os.path.exists(os.path.join(results_path, base_name + "-" + str(experiment_num))):
        experiment_num += 1
    folder_path = os.path.join(results_path, base_name + "-" + str(experiment_num))
    os.makedirs(folder_path)
    return folder_path

# Dataset class
class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(IMAGE_ROOT, self.data.iloc[idx, -1])
        label = self.data.iloc[idx, 3]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
print("Loading datasets...")
train_dataset = ImageDataset(TRAIN_CSV, transform=transform)
valid_dataset = ImageDataset(VALID_CSV, transform=transform)
test_dataset = ImageDataset(TEST_CSV, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print("Train images:", len(train_dataset))
print("Valid images:", len(valid_dataset))
print("Test images:", len(test_dataset))

# Model
print("Loading ViT base patch16 224 model...")
model = timm.create_model("vit_base_patch16_224", pretrained=True)
num_features = model.head.in_features
model.head = torch.nn.Linear(num_features, 1)
model = model.to(device)

param_count = sum(p.numel() for p in model.parameters())
print("Parameters:", param_count)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training
start_time = time.time()

train_losses = []
train_accuracies = []

print("Training ViT model...")
for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 32 == 0:
            batch_acc = running_corrects / total
            batch_loss = running_loss / total
            print("  Epoch [" + str(epoch+1) + "/" + str(NUM_EPOCHS) + "], Batch [" + str(batch_idx+1) + "/" + str(len(train_loader)) + "], Loss: " + str(round(batch_loss, 4)) + ", Accuracy: " + str(round(batch_acc, 4)))

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    epoch_time = time.time() - epoch_start
    print("Epoch [" + str(epoch+1) + "/" + str(NUM_EPOCHS) + "] | Loss: " + str(round(epoch_loss, 4)) + " | Accuracy: " + str(round(epoch_acc * 100, 2)) + "% | Time: " + str(int(epoch_time // 60)) + "m " + str(int(epoch_time % 60)) + "s")

print("Training complete.")

# Evaluate on validation set
print("\nEvaluating on validation set...")
model.eval()
val_loss = 0.0
val_corrects = 0
val_true = []
val_pred = []

with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        val_corrects += (preds == labels).sum().item()
        val_true.extend(labels.cpu().numpy().flatten().astype(int).tolist())
        val_pred.extend(preds.cpu().numpy().flatten().astype(int).tolist())

val_loss = val_loss / len(valid_dataset)
val_acc = val_corrects / len(valid_dataset)

print("Validation Loss: " + str(round(val_loss, 4)))
print("Validation Accuracy: " + str(round(val_acc * 100, 2)) + "%")

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss = 0.0
test_corrects = 0
test_true = []
test_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        test_corrects += (preds == labels).sum().item()
        test_true.extend(labels.cpu().numpy().flatten().astype(int).tolist())
        test_pred.extend(preds.cpu().numpy().flatten().astype(int).tolist())

test_loss = test_loss / len(test_dataset)
test_acc = test_corrects / len(test_dataset)

train_acc = train_accuracies[-1]

print("Test Loss: " + str(round(test_loss, 4)))
print("Test Accuracy: " + str(round(test_acc * 100, 2)) + "%")
print("Train Accuracy: " + str(round(train_acc * 100, 2)) + "%")
print("Difference: " + str(round(abs(train_acc - test_acc) * 100, 2)) + "%")

print("\nClassification Report:")
print(classification_report(test_true, test_pred, target_names=["Fake", "Real"], digits=4))

cm = confusion_matrix(test_true, test_pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix:")
print("                  Predicted Fake  Predicted Real")
print("  Actual Fake     " + str(tn) + "               " + str(fp))
print("  Actual Real     " + str(fn) + "               " + str(tp))
print("\nTrue Positives  (Real correctly detected):", tp)
print("True Negatives  (Fake correctly detected):", tn)
print("False Positives (Fake misclassified as Real):", fp)
print("False Negatives (Real misclassified as Fake):", fn)

# Plots
folder_path = experiments_folder("vit-model-v1-experiment")

# Confusion matrix plot
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix (ViT base patch16 224)")
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ["Fake", "Real"])
plt.yticks(tick_marks, ["Fake", "Real"])
thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

total_time = time.time() - start_time

# Save results
with open(os.path.join(folder_path, "results.txt"), "w") as f:
    f.write("Experiment: ViT Deepfake Detection\n")
    f.write("Model: vit_base_patch16_224 (timm, pretrained ImageNet, fine tuned)\n")
    f.write("Dataset: 140k Real and Fake Faces\n")
    f.write("Device: " + str(device) + "\n")
    f.write("Train images: " + str(len(train_dataset)) + "\n")
    f.write("Valid images: " + str(len(valid_dataset)) + "\n")
    f.write("Test images: " + str(len(test_dataset)) + "\n")
    f.write("\nTrain Accuracy: " + str(round(train_acc * 100, 2)) + "%\n")
    f.write("Validation Accuracy: " + str(round(val_acc * 100, 2)) + "%\n")
    f.write("Test Accuracy: " + str(round(test_acc * 100, 2)) + "%\n")
    f.write("Difference (overfit check): " + str(round(abs(train_acc - test_acc) * 100, 2)) + "%\n")
    f.write("Batch size: " + str(BATCH_SIZE) + "\n")
    f.write("Epochs: " + str(NUM_EPOCHS) + "\n")
    f.write("Learning rate: " + str(LEARNING_RATE) + "\n")
    f.write("Optimizer: Adam\n")
    f.write("Loss: BCEWithLogitsLoss\n")
    f.write("Parameters: " + str(param_count) + "\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(test_true, test_pred, target_names=["Fake", "Real"], digits=4))
    f.write("\nConfusion Matrix:\n")
    f.write("                  Predicted Fake  Predicted Real\n")
    f.write("  Actual Fake     " + str(tn) + "               " + str(fp) + "\n")
    f.write("  Actual Real     " + str(fn) + "               " + str(tp) + "\n")
    f.write("\nTrue Positives  (Real correctly detected): " + str(tp) + "\n")
    f.write("True Negatives  (Fake correctly detected): " + str(tn) + "\n")
    f.write("False Positives (Fake misclassified as Real): " + str(fp) + "\n")
    f.write("False Negatives (Real misclassified as Fake): " + str(fn) + "\n")
    f.write("\nTotal Time: " + str(round(total_time / 60, 2)) + " minutes\n")

torch.save(model.state_dict(), os.path.join(folder_path, "vit_binary_classification.pth"))

print("Results saved to " + folder_path)
