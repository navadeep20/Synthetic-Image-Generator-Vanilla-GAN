from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size=64):
    """
    Loads and preprocesses dataset for GAN training.
    Dataset: MNIST (privacy-safe grayscale images)
    """

    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # scale to [-1, 1]
    ])

    dataset = datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader
