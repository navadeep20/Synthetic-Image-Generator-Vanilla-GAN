import torch
import torchvision.utils as vutils
from generator import Generator
import os

# Get absolute path of current file (src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define correct paths
SAMPLES_DIR = os.path.join(BASE_DIR, "..", "samples")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "..", "checkpoints")

# Create samples folder if not exists
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Load trained generator
G = Generator()
G.load_state_dict(torch.load(os.path.join(CHECKPOINTS_DIR, "G_final.pth")))
G.eval()

# Generate noise
z = torch.randn(25, 100)

# Generate images
fake_images = G(z)

# Save image grid
output_path = os.path.join(SAMPLES_DIR, "generated_samples.png")

vutils.save_image(
    fake_images,
    output_path,
    nrow=5,
    normalize=True
)

print(f"Sample images saved at: {output_path}")
