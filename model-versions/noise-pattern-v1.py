import os
import time
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Dataset and results paths
DATASET_PATH = os.path.join("datasets", "real_vs_fake", "real-vs-fake")
RESULTS_PATH = "experiments"
os.makedirs(RESULTS_PATH, exist_ok=True)

IMAGE_SIZE = (64, 64)
CHUNK_SIZE = 5000

def experiments_folder(base_name="noise-pattern-v4-experiment"):
    experiment_num = 1
    while os.path.exists(os.path.join(RESULTS_PATH, f"{base_name}-{experiment_num}")):
        experiment_num += 1
    folder_path = os.path.join(RESULTS_PATH, f"{base_name}-{experiment_num}")
    os.makedirs(folder_path)
    return folder_path

RESULTS_FOLDER = experiments_folder()
print(f"Results will be saved to: {RESULTS_FOLDER}")

def extract_noise(img_array):
    img_float = img_array.astype(np.float32)
    blurred = uniform_filter(img_float, size=3)
    return img_float - blurred

def extract_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    features = []
    for c in range(3):
        noise = extract_noise(img_array[:, :, c])
        features.extend(noise.flatten())
        features.append(np.mean(noise))
        features.append(np.std(noise))
        features.append(np.var(noise))
    return np.array(features, dtype=np.float32)

def load_split(split):
    print(f"\nLoading {split} split...")
    split_path = os.path.join(DATASET_PATH, split)
    chunk_dir = os.path.join(RESULTS_FOLDER, f"{split}_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_idx = 0
    for label_idx, label_name in enumerate(sorted(os.listdir(split_path))):
        label_path = os.path.join(split_path, label_name)
        if not os.path.isdir(label_path):
            continue
        files = os.listdir(label_path)
        print(f"  {label_name}: {len(files)} images (label={label_idx})")

        X_batch, y_batch = [], []
        for i, fname in enumerate(files):
            if i % 1000 == 0:
                print(f"    Processing {i}/{len(files)}...")
            fpath = os.path.join(label_path, fname)
            try:
                features = extract_features(fpath)
                X_batch.append(features)
                y_batch.append(label_idx)
            except Exception as e:
                print(f"    Skipping {fname}: {e}")

            if len(X_batch) >= CHUNK_SIZE:
                chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx}.npz")
                np.savez_compressed(chunk_path, X=np.array(X_batch), y=np.array(y_batch))
                print(f"    Saved chunk {chunk_idx} to disk")
                chunk_idx += 1
                X_batch, y_batch = [], []

        if X_batch:
            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx}.npz")
            np.savez_compressed(chunk_path, X=np.array(X_batch), y=np.array(y_batch))
            chunk_idx += 1

    print(f"  Loading {chunk_idx} chunks from disk...")
    all_X, all_y = [], []
    for i in range(chunk_idx):
        chunk = np.load(os.path.join(chunk_dir, f"chunk_{i}.npz"))
        all_X.append(chunk["X"])
        all_y.append(chunk["y"])

    return np.concatenate(all_X), np.concatenate(all_y)

start_time = time.time()

X_train, y_train = load_split("train")
X_valid, y_valid = load_split("valid")
X_test, y_test = load_split("test")

print(f"\nDone loading in {(time.time() - start_time) / 60:.1f} minutes")
print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
print("Features scaled!")

print("\nTraining XGBoost...")
train_start = time.time()
clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    verbosity=1,
    n_jobs=-1
)
clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=50)
print(f"Training done in {(time.time() - train_start) / 60:.1f} minutes")

# Evaluate
valid_preds = clf.predict(X_valid)
valid_acc = accuracy_score(y_valid, valid_preds) * 100

test_preds = clf.predict(X_test)
test_acc = accuracy_score(y_test, test_preds) * 100
test_precision = precision_score(y_test, test_preds) * 100
test_recall = recall_score(y_test, test_preds) * 100
test_f1 = f1_score(y_test, test_preds) * 100
test_cm = confusion_matrix(y_test, test_preds)

total_time = time.time() - start_time

print(f"\nValidation Accuracy: {valid_acc:.2f}%")
print(f"Test Accuracy:  {test_acc:.2f}%")
print(f"Test Precision: {test_precision:.2f}%")
print(f"Test Recall:    {test_recall:.2f}%")
print(f"Test F1 Score:  {test_f1:.2f}%")
print(f"\nConfusion Matrix:")
print(f"                 Predicted Fake  Predicted Real")
print(f"Actual Fake      {test_cm[0][0]}          {test_cm[0][1]}")
print(f"Actual Real      {test_cm[1][0]}          {test_cm[1][1]}")
print(f"\nTotal Time: {total_time / 60:.1f} minutes")

joblib.dump(clf, os.path.join(RESULTS_FOLDER, "model.pkl"))
joblib.dump(scaler, os.path.join(RESULTS_FOLDER, "scaler.pkl"))

results = f"""Noise Pattern Analysis Results

Test Accuracy:  {test_acc:.2f}%
Test Precision: {test_precision:.2f}%
Test Recall:    {test_recall:.2f}%
Test F1 Score:  {test_f1:.2f}%
Validation Accuracy: {valid_acc:.2f}%
Total Training Time: {total_time / 60:.1f} minutes

Confusion Matrix:
                 Predicted Fake  Predicted Real
Actual Fake      {test_cm[0][0]}          {test_cm[0][1]}
Actual Real      {test_cm[1][0]}          {test_cm[1][1]}

Hyperparameters:
- Image Size: {IMAGE_SIZE}
- XGBoost n_estimators: 200
- XGBoost max_depth: 6
- XGBoost learning_rate: 0.1

Dataset: {DATASET_PATH}
"""

with open(os.path.join(RESULTS_FOLDER, "results.txt"), "w") as f:
    f.write(results)

print(f"\nResults saved to {RESULTS_FOLDER}")
