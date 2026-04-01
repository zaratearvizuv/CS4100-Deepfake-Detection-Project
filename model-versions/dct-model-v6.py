import os
import cv2
import numpy as np
from scipy.stats import laplace
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import time

# ── Dirs ─────────────────────────────────────────────────────────────────────
REAL_DIR = os.path.join("datasets", "flickr-gan-dataset", "real_vs_fake", "real-vs-fake", "train", "real")
FAKE_DIR = os.path.join("datasets", "sfhq-part3", "images", "images")

MAX_IMAGES = 30000

# ── Experiment folder ────────────────────────────────────────────────────────
def experiments_folder(base_name):
    results_path = "experiments"
    os.makedirs(results_path, exist_ok=True)
    experiment_num = 1
    while os.path.exists(os.path.join(results_path, f"{base_name}-{experiment_num}")):
        experiment_num += 1
    folder_path = os.path.join(results_path, f"{base_name}-{experiment_num}")
    os.makedirs(folder_path)
    return folder_path

# ── DCT feature extraction ────────────────────────────────────────────────────
def extract_dct_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
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

# ── Load images ───────────────────────────────────────────────────────────────
def build_features(real_dir, fake_dir, max_per_class):
    import random
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                  if f.endswith(('.jpg', '.png'))]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                  if f.endswith(('.jpg', '.png'))]
    random.shuffle(real_files)
    random.shuffle(fake_files)
    real_files = real_files[:max_per_class]
    fake_files = fake_files[:max_per_class]

    X, y = [], []
    total = len(real_files) + len(fake_files)
    start = time.time()

    print(f"Extracting DCT features from real images...")
    for i, path in enumerate(real_files):
        feats = extract_dct_features(path)
        if feats is not None:
            X.append(feats)
            y.append(0)
        if (i+1) % 500 == 0:
            elapsed = time.time() - start
            remaining = (elapsed / (i+1)) * (total - (i+1))
            print(f"  {i+1}/{total} | Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | Remaining: {int(remaining//60)}m {int(remaining%60)}s")

    print(f"Extracting DCT features from fake images (StyleGAN2)...")
    for i, path in enumerate(fake_files):
        feats = extract_dct_features(path)
        if feats is not None:
            X.append(feats)
            y.append(1)
        offset = len(real_files) + i + 1
        if (i+1) % 500 == 0:
            elapsed = time.time() - start
            remaining = (elapsed / offset) * (total - offset)
            print(f"  {offset}/{total} | Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s | Remaining: {int(remaining//60)}m {int(remaining%60)}s")

    return np.array(X), np.array(y)

# ── Main ──────────────────────────────────────────────────────────────────────
start_time = time.time()

print("Building dataset (StyleGAN2 same-GAN test)...")
X, y = build_features(REAL_DIR, FAKE_DIR, MAX_IMAGES)
print(f"Dataset shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

print("Training Gradient Boosting classifier...")
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.6, max_depth=2, random_state=42)
clf.fit(X_train, y_train)
print("Training complete.")

y_train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"Train Accuracy: {train_acc * 100:.2f}%")

y_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:")
print(f"                  Predicted Real  Predicted Fake")
print(f"  Actual Real     {tn:<15} {fp}")
print(f"  Actual Fake     {fn:<15} {tp}")
print(f"\nTrue Positives  (Fake correctly detected): {tp}")
print(f"True Negatives  (Real correctly detected): {tn}")
print(f"False Positives (Real misclassified as Fake): {fp}")
print(f"False Negatives (Fake misclassified as Real): {fn}")

# ── Plot ──────────────────────────────────────────────────────────────────────
folder_path = experiments_folder("dct-model-v6-experiment")
train_scores, test_scores = [], []
for yt, yv in zip(clf.staged_predict(X_train), clf.staged_predict(X_test)):
    train_scores.append(accuracy_score(y_train, yt))
    test_scores.append(accuracy_score(y_test, yv))

plt.figure(figsize=(10, 6))
plt.plot(train_scores, label='Train Accuracy')
plt.plot(test_scores, label='Test Accuracy (StyleGAN2)')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('DCT + Gradient Boosting — StyleGAN2 (SFHQ) Same-GAN Test')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(folder_path, 'training_progress.png'))
plt.show()

total_time = time.time() - start_time

# ── Save results ──────────────────────────────────────────────────────────────
with open(os.path.join(folder_path, "results.txt"), "w") as f:
    f.write(f"Experiment: StyleGAN2 Same-GAN Test\n")
    f.write(f"Fake Dataset: sfhq-part3 (StyleGAN2)\n")
    f.write(f"Real Dataset: flickr-gan-dataset (Flickr real faces)\n")
    f.write(f"Train images: {len(X_train)}\n")
    f.write(f"Test images: {len(X_test)}\n\n")
    f.write(f"Train Accuracy: {train_acc * 100:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
    f.write(f"Difference (overfit check): {abs(train_acc - test_acc) * 100:.2f}%\n")
    f.write(f"GB n_estimators: 100\n")
    f.write(f"GB learning_rate: 0.6\n")
    f.write(f"GB max_depth: 2\n")
    f.write(f"Color Type: Black & White\n\n")
    f.write(f"Classification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    f.write(f"\nConfusion Matrix:\n")
    f.write(f"                  Predicted Real  Predicted Fake\n")
    f.write(f"  Actual Real     {tn:<15} {fp}\n")
    f.write(f"  Actual Fake     {fn:<15} {tp}\n")
    f.write(f"\nTrue Positives  (Fake correctly detected): {tp}\n")
    f.write(f"True Negatives  (Real correctly detected): {tn}\n")
    f.write(f"False Positives (Real misclassified as Fake): {fp}\n")
    f.write(f"False Negatives (Fake misclassified as Real): {fn}\n")
    f.write(f"\nTotal Time: {total_time/60:.2f} minutes\n")

print(f"Results saved to {folder_path}")