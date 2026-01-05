import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

from torchvision import models, transforms
from torchvision.utils import make_grid

from .generator import Generator
from .data_loader import get_dataloader

# -------------------------------------------------
# Paths
# -------------------------------------------------
FIGURES_DIR = "figures"
CHECKPOINTS_DIR = "checkpoints"
LOGS_DIR = "logs"

os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------------------------------
# Device
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# 1️⃣ LOSS CURVES
# -------------------------------------------------
loss_df = pd.read_csv(os.path.join(LOGS_DIR, "training_losses.csv"))

plt.figure(figsize=(6, 4))
plt.plot(loss_df["epoch"], loss_df["D_loss"], label="Discriminator Loss")
plt.plot(loss_df["epoch"], loss_df["G_loss"], label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Training Loss Curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "loss_curves.png"))
plt.close()

# -------------------------------------------------
# 2️⃣ LOAD TRAINED GENERATOR
# -------------------------------------------------
latent_dim = 100

G = Generator(latent_dim=latent_dim)
G.load_state_dict(
    torch.load(
        os.path.join(CHECKPOINTS_DIR, "G_v1_epoch_50.pth"),
        map_location=device
    )
)
G.to(device)
G.eval()

# -------------------------------------------------
# 3️⃣ GENERATED IMAGE GRID
# -------------------------------------------------
z = torch.randn(25, latent_dim, device=device)

with torch.no_grad():
    fake_images = G(z).cpu()

grid = make_grid(fake_images, nrow=5, normalize=True)

plt.figure(figsize=(5, 5))
plt.imshow(grid.permute(1, 2, 0), cmap="gray")
plt.axis("off")
plt.title("Generated Synthetic Images")
plt.savefig(os.path.join(FIGURES_DIR, "generated_images.png"))
plt.close()

# -------------------------------------------------
# 4️⃣ DIVERSITY SCORE
# -------------------------------------------------
flat_imgs = fake_images.view(fake_images.size(0), -1)
pairwise_dist = euclidean_distances(flat_imgs)
diversity_score = pairwise_dist.mean()

print(f"Diversity Score: {diversity_score:.4f}")

# -------------------------------------------------
# 5️⃣ FID PROXY (CNN FEATURE DISTANCE)
# -------------------------------------------------
feature_extractor = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)
feature_extractor.fc = torch.nn.Identity()
feature_extractor.to(device)
feature_extractor.eval()

resize_transform = transforms.Resize((224, 224))

def extract_features(images):
    images = images.to(device)
    images = images.repeat(1, 3, 1, 1)     # grayscale → RGB
    images = resize_transform(images)
    with torch.no_grad():
        feats = feature_extractor(images)
    return feats.cpu().numpy()

real_loader = get_dataloader(batch_size=32)
real_imgs, _ = next(iter(real_loader))

real_features = extract_features(real_imgs[:25])
fake_features = extract_features(fake_images[:25])

fid_proxy = np.linalg.norm(
    real_features.mean(axis=0) - fake_features.mean(axis=0)
)

print(f"FID Proxy Score: {fid_proxy:.4f}")

# -------------------------------------------------
# 6️⃣ t-SNE VISUALIZATION
# -------------------------------------------------
features = np.vstack([real_features, fake_features])
labels = np.array([0] * 25 + [1] * 25)

tsne = TSNE(n_components=2, random_state=42, perplexity=10)
embeddings = tsne.fit_transform(features)

plt.figure(figsize=(5, 4))
plt.scatter(
    embeddings[labels == 0, 0],
    embeddings[labels == 0, 1],
    label="Real",
    alpha=0.6
)
plt.scatter(
    embeddings[labels == 1, 0],
    embeddings[labels == 1, 1],
    label="Fake",
    alpha=0.6
)
plt.legend()
plt.title("t-SNE: Real vs Fake Embeddings")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "tsne_real_vs_fake.png"))
plt.close()

# -------------------------------------------------
# 7️⃣ PRIVACY / ANTI-MEMORIZATION CHECK
# -------------------------------------------------
real_flat = real_imgs[:25].view(25, -1).numpy()
fake_flat = fake_images[:25].view(25, -1).numpy()

similarity_matrix = cosine_similarity(fake_flat, real_flat)
max_similarity = similarity_matrix.max()

print(f"Max Real–Fake Similarity (Privacy Check): {max_similarity:.4f}")

if max_similarity < 0.9:
    print("Privacy Check PASSED: No memorization detected.")
else:
    print("WARNING: Potential memorization detected.")

print("Evaluation completed. All figures saved in /figures")
