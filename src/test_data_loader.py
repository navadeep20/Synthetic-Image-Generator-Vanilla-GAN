from data_loader import get_dataloader

loader = get_dataloader()
images, labels = next(iter(loader))

print("Batch shape:", images.shape)
print("Min value:", images.min().item())
print("Max value:", images.max().item())
