import torch
import torch.nn as nn

from .generator import Generator
from .discriminator import Discriminator


class VanillaGAN:
    """
    Vanilla GAN wrapper class
    Combines Generator and Discriminator
    """

    def __init__(
        self,
        latent_dim=100,
        lr=0.0002,
        beta1=0.5,
        device="cpu"
    ):
        self.device = device

        # Models
        self.generator = Generator(latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Loss
        self.criterion = nn.BCELoss()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(beta1, 0.999)
        )

        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(beta1, 0.999)
        )

    def train_discriminator(self, real_imgs, fake_imgs):
        batch_size = real_imgs.size(0)

        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        real_loss = self.criterion(
            self.discriminator(real_imgs), valid
        )

        fake_loss = self.criterion(
            self.discriminator(fake_imgs.detach()), fake
        )

        d_loss = real_loss + fake_loss

        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()

        return d_loss.item()

    def train_generator(self, fake_imgs):
        batch_size = fake_imgs.size(0)
        valid = torch.ones(batch_size, 1, device=self.device)

        g_loss = self.criterion(
            self.discriminator(fake_imgs), valid
        )

        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        return g_loss.item()
