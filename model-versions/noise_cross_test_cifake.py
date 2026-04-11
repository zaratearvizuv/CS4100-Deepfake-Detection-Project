import os
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Paths
MODEL_PATH = "experiments/noise-pattern-v4-experiment-1/model.pkl"
SCALER_PATH = "experiments/noise-pattern-v4-experiment-1/scaler.pkl"
RESULTS_PATH = "experiments/noise-pattern-v4-experiment-1/cross_test_cifake_results.txt"

# CIFAKE has its own FAKE/REAL split
FAKE_PATH = "datasets/cifake/test/FAKE"
REAL_PATH = "datasets/cifake/test/REAL"

IMAGE_SIZE = (64, 64)
MAX_IMAGES = 5000

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

def load_folder(folder, label, max_images=MAX_IMAGES):
    X, y = [], []
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files = files[:max_images]
    print(f"  Loading {len(files)} images from {folder} (label={label})")
    for i, fname in enumerate(files):
        if i % 1000 == 0:
            print(f"    {i}/{len(files)}...")
        try:
            features = extract_features(os.path.join(folder, fname))
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"    Skipping {fname}: {e}")
    return np.array(X), np.array(y)

print("Loading model and scaler...")
clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Loaded!")

print("\nLoading CIFAKE fake images...")
X_fake, y_fake = load_folder(FAKE_PATH, label=0)

print("\nLoading CIFAKE real images...")
X_real, y_real = load_folder(REAL_PATH, label=1)

X = np.concatenate([X_fake, X_real])
y = np.concatenate([y_fake, y_real])

print("\nScaling features...")
X = scaler.transform(X)

print("\nRunning predictions...")
preds = clf.predict(X)
acc = accuracy_score(y, preds) * 100
report = classification_report(y, preds, target_names=["fake", "real"])

print(f"\nCIFAKE Cross-Test Accuracy: {acc:.2f}%")
print(report)

results = f"""Noise Pattern Cross-Test: CIFAKE (Stable Diffusion)
=====================================================
Model trained on: StyleGAN1 (real_vs_fake)
Tested on: CIFAKE (Stable Diffusion generated images)

Accuracy: {acc:.2f}%

Classification Report:
{report}
"""

with open(RESULTS_PATH, "w") as f:
    f.write(results)

print(f"\nResults saved to {RESULTS_PATH}")
