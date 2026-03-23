# This model trains and tests on same stable diffusion dataset. Sttable diffusion against itself
# Uses a black and white color conversion

# Step 1: Importing Libraries and Load Data

import os
import cv2
import numpy as np
from scipy.stats import laplace
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time

TRAIN_FAKE_DIR = os.path.join("datasets", "stable-diffusion-dataset", "text2img", "text2img")
TRAIN_REAL_DIR = os.path.join("datasets", "stable-diffusion-dataset", "wiki", "wiki")
TEST_FAKE_DIR = os.path.join("datasets", "stable-diffusion-dataset", "text2img", "text2img")
TEST_REAL_DIR = os.path.join("datasets", "stable-diffusion-dataset", "wiki", "wiki")
# This number determines how many images are loaded per real or fake class
MAX_IMAGES = 30000


# Creates folder with results with automatic renaming
def experiments_folder(base_name):
    results_path = "experiments"
    os.makedirs(results_path, exist_ok=True)
    
    experiment_num = 1
    while os.path.exists(os.path.join(results_path, f"{base_name}-{experiment_num}")):
        experiment_num += 1
    
    folder_path = os.path.join(results_path, f"{base_name}-{experiment_num}")
    os.makedirs(folder_path)
    return folder_path

def extract_dct_features(img_path):
    # Load image and convert to grayscale
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Make dimensions multiple of 8 via cropping
    h, w = img.shape
    h = (h // 8) * 8
    w = (w // 8) * 8
    img = img[:h, :w].astype(np.float32)
    
    # Collect AC coefficients across all 8x8 blocks
    ac_coeffs = [[] for _ in range(63)]
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = img[i:i+8, j:j+8]
            dct_block = cv2.dct(block)
            
            # Flatten in zigzag order, skip DC (position 0)
            flat = dct_block.flatten()
            for k in range(1, 64):  # skip index 0 (DC)
                ac_coeffs[k-1].append(flat[k])
    
    # Fit Laplacian to each AC coefficient → extract β
    betas = []
    for coeffs in ac_coeffs:
        loc, scale = laplace.fit(coeffs)
        betas.append(scale)  # scale = β
    
    return np.array(betas)  # 63 β values

def load_images_from_nested_dir(root_dir, max_images=MAX_IMAGES):
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(('.jpg', '.png')):
                all_files.append(os.path.join(dirpath, f))
    return all_files[:max_images]

def build_dataset(real_dir, fake_dir, max_per_class=MAX_IMAGES):
    X, y = [], []
    
    real_files = load_images_from_nested_dir(real_dir, max_per_class)
    fake_files = load_images_from_nested_dir(fake_dir, max_per_class)
    
    total_files = len(real_files) + len(fake_files)
    dataset_start = time.time()

    print("Extracting DCT features from real images...")
    for i, path in enumerate(real_files):
        features = extract_dct_features(path)
        X.append(features)
        y.append(0)
        if (i+1) % 500 == 0:
            elapsed = time.time() - dataset_start
            done = i + 1
            remaining = (elapsed / done) * (total_files - done)
            print(f"  {i+1}/{total_files} | Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | Remaining: {int(remaining//60)}m {int(remaining%60)}s")

    print("Extracting DCT features from fake images...")
    for i, path in enumerate(fake_files):
        features = extract_dct_features(path)
        X.append(features)
        y.append(1)
        offset = len(real_files) + i + 1
        if (i+1) % 500 == 0:
            elapsed = time.time() - dataset_start
            remaining = (elapsed / offset) * (total_files - offset)
            print(f"  {offset}/{total_files} | Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | Remaining: {int(remaining//60)}m {int(remaining%60)}s")
    
    return np.array(X), np.array(y)

start_time = time.time()

print("Building training dataset...")
X_train, y_train = build_dataset(TRAIN_REAL_DIR, TRAIN_FAKE_DIR)
print(f"Dataset shape: {X_train.shape}")

print("Building test dataset...")
X_test, y_test = build_dataset(TEST_REAL_DIR, TEST_FAKE_DIR)
print(f"Test dataset shape: {X_test.shape}")

# Train Gradient Boosting classifier
print("Training Gradient Boosting classifier...")
clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.6,
    max_depth=2,
    random_state=42
)
clf.fit(X_train, y_train)
print("Training complete.")

# Evaluate
print("Evaluating...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
      target_names=['Real', 'Fake']))

folder_path = experiments_folder("dct-model-v3-experiment")

total_time = time.time() - start_time

with open(os.path.join(folder_path, "results.txt"), "w") as f:
    f.write(f"Dataset: stable-diffusion-dataset (Stable Diffusion V1.5)\n")
    f.write(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    f.write(f"Train images per class: {MAX_IMAGES}\n")
    f.write(f"Test images per class: {MAX_IMAGES}\n")
    f.write(f"GB n_estimators: 100\n")
    f.write(f"GB learning_rate: 0.6\n")
    f.write(f"GB max_depth: 2\n")
    f.write(f"\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    f.write(f"\nTotal Time: {total_time/60:.2f} minutes\n")

print(f"Results saved to {folder_path}")