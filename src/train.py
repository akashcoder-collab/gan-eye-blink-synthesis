import torch
import torch.nn.functional as F
from dataset import get_dataloader
from model import get_generator, get_discriminator
from utils import get_device, to_device, save_samples

def train():
    data_dir = "data/"
    batch_size = 128
    latent_size = 128
    lr = 0.0002
    epochs = 10

    dl, _ = get_dataloader(data_dir, batch_size)
    device = get_device()

    G = get_generator(latent_size).to(device)
    D = get_discriminator().to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

    fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)

    for epoch in range(epochs):
        for real, _ in dl:
            real = to_device(real, device)

            opt_d.zero_grad()
            real_preds = D(real)
            real_loss = F.binary_cross_entropy(real_preds, torch.ones_like(real_preds))

            z = torch.randn(real.size(0), latent_size,1,1, device=device)
            fake = G(z)
            fake_preds = D(fake.detach())
            fake_loss = F.binary_cross_entropy(fake_preds, torch.zeros_like(fake_preds))

            loss_d = real_loss + fake_loss
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad()
            preds = D(fake)
            loss_g = F.binary_cross_entropy(preds, torch.ones_like(preds))
            loss_g.backward()
            opt_g.step()

        print(f"Epoch {epoch}: G={loss_g.item():.4f}, D={loss_d.item():.4f}")
        save_samples(G, fixed_latent, epoch)

    torch.save(G.state_dict(), "models/generator.pth")
    torch.save(D.state_dict(), "models/discriminator.pth")

if __name__ == "__main__":
    train()
