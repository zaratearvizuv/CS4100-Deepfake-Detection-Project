import os
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Paths
MODEL_PATH = "experiments/noise-pattern-v4-experiment-1/model.pkl"
SCALER_PATH = "experiments/noise-pattern-v4-experiment-1/scaler.pkl"
RESULTS_PATH = "experiments/noise-pattern-v4-experiment-1/cross_test_deepfakeface_results.txt"

FAKE_DIRS = [
    "datasets/deepfakeface/inpainting/inpainting",
    "datasets/deepfakeface/insight/insight",
    "datasets/deepfakeface/text2img/text2img",
]
REAL_PATH = "datasets/real_vs_fake/real-vs-fake/test/real"

IMAGE_SIZE = (64, 64)
MAX_IMAGES_PER_DIR = 300
MAX_REAL = 900

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

def load_folder_recursive(root, label, max_images=300):
    """Walk all subfolders to find images"""
    X, y = [], []
    all_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(dirpath, f))
    all_files = all_files[:max_images]
    print(f"  Loading {len(all_files)} images from {root} (label={label})")
    for i, fpath in enumerate(all_files):
        if i % 100 == 0:
            print(f"    {i}/{len(all_files)}...")
        try:
            features = extract_features(fpath)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"    Skipping {fpath}: {e}")
    return np.array(X), np.array(y)

def load_flat_folder(folder, label, max_images=MAX_REAL):
    X, y = [], []
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files = files[:max_images]
    print(f"  Loading {len(files)} images from {folder} (label={label})")
    for i, fname in enumerate(files):
        if i % 100 == 0:
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

X_fake_all, y_fake_all = [], []
for d in FAKE_DIRS:
    if os.path.exists(d):
        X_f, y_f = load_folder_recursive(d, label=0, max_images=MAX_IMAGES_PER_DIR)
        if len(X_f) > 0:
            X_fake_all.append(X_f)
            y_fake_all.append(y_f)
    else:
        print(f"  Skipping {d} - not found")

X_fake = np.concatenate(X_fake_all)
y_fake = np.concatenate(y_fake_all)

print("\nLoading real images...")
X_real, y_real = load_flat_folder(REAL_PATH, label=1, max_images=MAX_REAL)

X = np.concatenate([X_fake, X_real])
y = np.concatenate([y_fake, y_real])

print("\nScaling features...")
X = scaler.transform(X)

print("\nRunning predictions...")
preds = clf.predict(X)
acc = accuracy_score(y, preds) * 100
report = classification_report(y, preds, target_names=["fake", "real"])

print(f"\nDeepFakeFace Cross-Test Accuracy: {acc:.2f}%")
print(report)

results = f"""Noise Pattern Cross-Test: DeepFakeFace (HuggingFace)
=====================================================
Model trained on: StyleGAN1 (real_vs_fake)
Tested on: OpenRL/DeepFakeFace (inpainting, insight, text2img)

Accuracy: {acc:.2f}%

Classification Report:
{report}
"""

with open(RESULTS_PATH, "w") as f:
    f.write(results)

print(f"\nResults saved to {RESULTS_PATH}")
