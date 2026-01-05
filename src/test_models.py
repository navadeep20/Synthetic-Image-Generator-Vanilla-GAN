import torch
from generator import Generator
from discriminator import Discriminator

z = torch.randn(4, 100)

G = Generator()
D = Discriminator()

fake_images = G(z)
output = D(fake_images)

print("Fake image shape:", fake_images.shape)
print("Discriminator output shape:", output.shape)
