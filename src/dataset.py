import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as t

def get_dataloader(data_dir, batch_size=128, image_size=64):
    stats = (0.5,), (0.5,)
    
    ds = ImageFolder(data_dir, transform=t.Compose([
        t.Grayscale(1),
        t.Resize(image_size),
        t.CenterCrop(image_size),
        t.ToTensor(),
        t.Normalize(*stats)
    ]))
    
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return dl, stats
