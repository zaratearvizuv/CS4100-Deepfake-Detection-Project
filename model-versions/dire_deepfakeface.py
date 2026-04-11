import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import torch
from diffusers import DDPMPipeline

FAKE_DIRS = [
    "datasets/deepfakeface/inpainting/inpainting",
    "datasets/deepfakeface/insight/insight",
    "datasets/deepfakeface/text2img/text2img",
]
REAL_PATH = "datasets/real_vs_fake/real-vs-fake/test/real"
RESULTS_PATH = "experiments/dire_deepfakeface_results.txt"
os.makedirs("experiments", exist_ok=True)

IMAGE_SIZE = (32, 32)
MAX_IMAGES_PER_DIR = 300
MAX_REAL = 900
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

def load_folder_recursive(root, label, max_images=300):
    X, y = [], []
    all_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(dirpath, f))
    all_files = all_files[:max_images]
    print(f"  Processing {len(all_files)} images from {root} (label={label})")
    for i, fpath in enumerate(all_files):
        if i % 100 == 0:
            print(f"    {i}/{len(all_files)}...")
        try:
            error = compute_dire(fpath)
            X.append([error])
            y.append(label)
        except Exception as e:
            print(f"    Skipping {fpath}: {e}")
    return np.array(X), np.array(y)

def load_flat_folder(folder, label, max_images=MAX_REAL):
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

print("Loading model and scaler...")
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

print("\nProcessing real images...")
X_real, y_real = load_flat_folder(REAL_PATH, label=1, max_images=MAX_REAL)

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

print(f"\nDIRE DeepFakeFace Accuracy: {acc:.2f}%")
print(report)

results = f"""DIRE Results: DeepFakeFace (HuggingFace)
=========================================
Dataset: OpenRL/DeepFakeFace (inpainting, insight, text2img)
Diffusion Model: google/ddpm-cifar10-32
Timestep: {TIMESTEP}
Max images per subdir: {MAX_IMAGES_PER_DIR}

Accuracy: {acc:.2f}%

Classification Report:
{report}
"""

with open(RESULTS_PATH, "w") as f:
    f.write(results)

print(f"\nResults saved to {RESULTS_PATH}")
