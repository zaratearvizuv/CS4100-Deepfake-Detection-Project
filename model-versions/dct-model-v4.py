# Cross-GAN test: Train on StyleGAN1 (flickr), Test on Stable Diffusion
# Uses black and white color conversion

import os
import cv2
import numpy as np
from scipy.stats import laplace
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time

# Train on StyleGAN1
TRAIN_REAL_DIR = os.path.join("datasets", "flickr-gan-dataset", "real_vs_fake", "real-vs-fake", "train", "real")
TRAIN_FAKE_DIR = os.path.join("datasets", "flickr-gan-dataset", "real_vs_fake", "real-vs-fake", "train", "fake")

# Test on Stable Diffusion
TEST_REAL_DIR = os.path.join("datasets", "stable-diffusion-dataset", "wiki", "wiki")
TEST_FAKE_DIR = os.path.join("datasets", "stable-diffusion-dataset", "text2img", "text2img")

MAX_IMAGES = 25000

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
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    h = (h // 8) * 8
    w = (w // 8) * 8
    img = img[:h, :w].astype(np.float32)
    ac_coeffs = [[] for _ in range(63)]
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = img[i:i+8, j:j+8]
            dct_block = cv2.dct(block)
            flat = dct_block.flatten()
            for k in range(1, 64):
                ac_coeffs[k-1].append(flat[k])
    betas = []
    for coeffs in ac_coeffs:
        loc, scale = laplace.fit(coeffs)
        betas.append(scale)
    return np.array(betas)

def load_images_flat(real_dir, fake_dir, max_per_class):
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                  if f.endswith(('.jpg', '.png'))][:max_per_class]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                  if f.endswith(('.jpg', '.png'))][:max_per_class]
    return real_files, fake_files

def load_images_nested(real_dir, fake_dir, max_per_class):
    def walk(d):
        files = []
        for dirpath, _, filenames in os.walk(d):
            for f in filenames:
                if f.endswith(('.jpg', '.png')):
                    files.append(os.path.join(dirpath, f))
        return files[:max_per_class]
    return walk(real_dir), walk(fake_dir)

def build_features(real_files, fake_files, label=""):
    X, y = [], []
    total = len(real_files) + len(fake_files)
    start = time.time()
    print(f"Extracting DCT features from real images {label}...")
    for i, path in enumerate(real_files):
        X.append(extract_dct_features(path))
        y.append(0)
        if (i+1) % 500 == 0:
            elapsed = time.time() - start
            remaining = (elapsed / (i+1)) * (total - (i+1))
            print(f"  {i+1}/{total} | Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | Remaining: {int(remaining//60)}m {int(remaining%60)}s")
    print(f"Extracting DCT features from fake images {label}...")
    for i, path in enumerate(fake_files):
        X.append(extract_dct_features(path))
        y.append(1)
        offset = len(real_files) + i + 1
        if (i+1) % 500 == 0:
            elapsed = time.time() - start
            remaining = (elapsed / offset) * (total - offset)
            print(f"  {offset}/{total} | Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | Remaining: {int(remaining//60)}m {int(remaining%60)}s")
    return np.array(X), np.array(y)

start_time = time.time()

# Build training set (StyleGAN1 - flat folders)
print("Building training dataset (StyleGAN1)...")
train_real, train_fake = load_images_flat(TRAIN_REAL_DIR, TRAIN_FAKE_DIR, MAX_IMAGES)
X_train, y_train = build_features(train_real, train_fake, "(StyleGAN1)")
print(f"Train shape: {X_train.shape}")

# Build test set (Stable Diffusion - nested folders)
print("Building test dataset (Stable Diffusion)...")
test_real, test_fake = load_images_nested(TEST_REAL_DIR, TEST_FAKE_DIR, MAX_IMAGES)
X_test, y_test = build_features(test_real, test_fake, "(Stable Diffusion)")
print(f"Test shape: {X_test.shape}")

# Train
print("Training Gradient Boosting classifier...")
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.6, max_depth=2, random_state=42)
clf.fit(X_train, y_train)
print("Training complete.")

# Overfitting check
y_train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"Train Accuracy: {train_acc * 100:.2f}%")

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# Plot
folder_path = experiments_folder("dct-model-v4-experiment")
train_scores, test_scores = [], []
for yt, yv in zip(clf.staged_predict(X_train), clf.staged_predict(X_test)):
    train_scores.append(accuracy_score(y_train, yt))
    test_scores.append(accuracy_score(y_test, yv))
plt.figure(figsize=(10, 6))
plt.plot(train_scores, label='Train Accuracy (StyleGAN1)')
plt.plot(test_scores, label='Test Accuracy (Stable Diffusion)')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Cross-GAN: StyleGAN1 → Stable Diffusion')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(folder_path, 'training_progress.png'))
plt.show()

total_time = time.time() - start_time

with open(os.path.join(folder_path, "results.txt"), "w") as f:
    f.write(f"Experiment: Cross-GAN Generalization\n")
    f.write(f"Train Dataset: flickr-gan-dataset (StyleGAN1)\n")
    f.write(f"Test Dataset: stable-diffusion-dataset (Stable Diffusion V1.5)\n")
    f.write(f"Train images: {len(X_train)}\n")
    f.write(f"Test images: {len(X_test)}\n")
    f.write(f"\nTrain Accuracy: {train_acc * 100:.2f}%\n")
    f.write(f"Test Accuracy: {acc * 100:.2f}%\n")
    f.write(f"Difference (overfit check): {(train_acc - acc) * 100:.2f}%\n")
    f.write(f"GB n_estimators: 100\n")
    f.write(f"GB learning_rate: 0.6\n")
    f.write(f"GB max_depth: 2\n")
    f.write(f"Color Type: Black & White\n")
    f.write(f"\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    f.write(f"\nTotal Time: {total_time/60:.2f} minutes\n")

print(f"Results saved to {folder_path}")