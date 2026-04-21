import torch
import os
from torchvision.utils import save_image

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [to_device(i, device) for i in x]
    return x.to(device)

def save_samples(generator, latent, epoch, folder="results"):
    import os
    from torchvision.utils import save_image

    os.makedirs(folder, exist_ok=True)
    fake = generator(latent)


    fake = (fake + 1) / 2  

    save_image(fake, f"{folder}/img_{epoch}.png", nrow=8)
