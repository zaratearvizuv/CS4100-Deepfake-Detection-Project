import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import torch
from diffusers import DDPMPipeline

# Paths
FAKE_PATH = "datasets/real_vs_fake/real-vs-fake/test/fake"
REAL_PATH = "datasets/real_vs_fake/real-vs-fake/test/real"
RESULTS_PATH = "experiments/dire_stylegan1_results.txt"
os.makedirs("experiments", exist_ok=True)

IMAGE_SIZE = (32, 32)
MAX_IMAGES = 1000
TIMESTEP = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print("Loading DDPM model...")
pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(DEVICE)
scheduler = pipe.scheduler
unet = pipe.unet
print("Model loaded!")

def compute_dire(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
    img_tensor = torch.tensor(np.array(img)).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    t = torch.tensor([TIMESTEP], device=DEVICE)
    noise = torch.randn_like(img_tensor)
    noisy = scheduler.add_noise(img_tensor, noise, t)

    with torch.no_grad():
        pred_noise = unet(noisy, t).sample

    reconstructed = noisy - pred_noise
    error = torch.mean((img_tensor - reconstructed) ** 2).item()
    return error

def load_folder(folder, label, max_images=MAX_IMAGES):
    X, y = [], []
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files = files[:max_images]
    print(f"  Processing {len(files)} images from {folder} (label={label})")
    for i, fname in enumerate(files):
        if i % 100 == 0:
            print(f"    {i}/{len(files)}...")
        try:
            error = compute_dire(os.path.join(folder, fname))
            X.append([error])
            y.append(label)
        except Exception as e:
            print(f"    Skipping {fname}: {e}")
    return np.array(X), np.array(y)

print("\nProcessing StyleGAN1 fake images...")
X_fake, y_fake = load_folder(FAKE_PATH, label=0)

print("\nProcessing real images...")
X_real, y_real = load_folder(REAL_PATH, label=1)

X = np.concatenate([X_fake, X_real])
y = np.concatenate([y_fake, y_real])

print("\nTraining classifier on DIRE scores...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
clf = LogisticRegression()
clf.fit(X_scaled, y)

preds = clf.predict(X_scaled)
acc = accuracy_score(y, preds) * 100
report = classification_report(y, preds, target_names=["fake", "real"])

print(f"\nDIRE StyleGAN1 Accuracy: {acc:.2f}%")
print(report)

results = f"""DIRE Results: StyleGAN1
=======================
Dataset: real_vs_fake (StyleGAN1)
Diffusion Model: google/ddpm-cifar10-32
Timestep: {TIMESTEP}
Max images per class: {MAX_IMAGES}

Accuracy: {acc:.2f}%

Classification Report:
{report}
"""

with open(RESULTS_PATH, "w") as f:
    f.write(results)

joblib.dump(clf, "experiments/dire_stylegan1_clf.pkl")
joblib.dump(scaler, "experiments/dire_stylegan1_scaler.pkl")
print(f"\nResults saved to {RESULTS_PATH}")
