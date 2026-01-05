import torch
import time
from .generator import Generator
from .monitoring import log_inference

MODEL_VERSION = "G_v1"

def generate_images(num_images=10, latent_dim=100, device="cpu"):
    start = time.time()

    G = Generator(latent_dim)
    G.load_state_dict(torch.load("checkpoints/G_v1_epoch_50.pth"))
    G.to(device)
    G.eval()

    z = torch.randn(num_images, latent_dim, device=device)
    with torch.no_grad():
        images = G(z)

    latency_ms = (time.time() - start) * 1000
    log_inference(
        model_version=MODEL_VERSION,
        num_images=num_images,
        latency_ms=round(latency_ms, 2)
    )

    return images
