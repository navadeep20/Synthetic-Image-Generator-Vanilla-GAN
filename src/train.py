import os
import csv
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

from .data_loader import get_dataloader
from .vanilla_gan import VanillaGAN

# -------------------------
# Load configuration
# -------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

latent_dim = config["training"]["latent_dim"]
batch_size = config["training"]["batch_size"]
epochs = config["training"]["epochs"]
lr = config["training"]["learning_rate"]
beta1 = config["training"]["beta1"]

paths = config["paths"]
samples_dir = paths["samples_dir"]
checkpoints_dir = paths["checkpoints_dir"]
logs_dir = paths["logs_dir"]

os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# -------------------------
# Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Data Loader
# -------------------------
dataloader = get_dataloader(batch_size)

# -------------------------
# Initialize GAN
# -------------------------
gan = VanillaGAN(
    latent_dim=latent_dim,
    lr=lr,
    beta1=beta1,
    device=device
)

# -------------------------
# Logging setup
# -------------------------
csv_log_path = os.path.join(logs_dir, "training_losses.csv")
tensorboard_writer = SummaryWriter(log_dir=os.path.join(logs_dir, "tensorboard"))

with open(csv_log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "D_loss", "G_loss"])

print("Training started...")

# -------------------------
# Training Loop
# -------------------------
for epoch in range(epochs):
    d_loss_epoch = 0.0
    g_loss_epoch = 0.0

    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.to(device)
        batch_size_curr = real_imgs.size(0)

        z = torch.randn(batch_size_curr, latent_dim, device=device)
        fake_imgs = gan.generator(z)

        # Train Discriminator
        d_loss = gan.train_discriminator(real_imgs, fake_imgs)

        # Train Generator
        g_loss = gan.train_generator(fake_imgs)

        d_loss_epoch += d_loss
        g_loss_epoch += g_loss

    d_loss_epoch /= len(dataloader)
    g_loss_epoch /= len(dataloader)

    # -------------------------
    # Logging
    # -------------------------
    tensorboard_writer.add_scalar("Loss/Discriminator", d_loss_epoch, epoch)
    tensorboard_writer.add_scalar("Loss/Generator", g_loss_epoch, epoch)

    with open(csv_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, d_loss_epoch, g_loss_epoch])

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"D Loss: {d_loss_epoch:.4f} "
        f"G Loss: {g_loss_epoch:.4f}"
    )

    # -------------------------
    # Save sample images
    # -------------------------
    if (epoch + 1) % 5 == 0:
        from torchvision.utils import save_image
        save_image(
            fake_imgs[:25],
            os.path.join(samples_dir, f"epoch_{epoch+1}.png"),
            nrow=5,
            normalize=True
        )

    # -------------------------
    # Checkpointing
    # -------------------------
    torch.save(
        gan.generator.state_dict(),
        os.path.join(checkpoints_dir, f"G_v1_epoch_{epoch+1}.pth")
    )
    torch.save(
        gan.discriminator.state_dict(),
        os.path.join(checkpoints_dir, f"D_v1_epoch_{epoch+1}.pth")
    )

tensorboard_writer.close()

print("Training finished with logging and  checkpoints.")
