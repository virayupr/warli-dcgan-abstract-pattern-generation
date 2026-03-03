# ============================================================
# train_dcgan.py
# DCGAN Training Script (Reproducible Version)
# ============================================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.dcgan import DCGANGenerator, DCGANDiscriminator


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Configuration
# ============================================================

DATA_ROOT = "../data/warli_dataset"  # dataset path
OUTPUT_DIR = "../results"
IMAGE_SIZE = 64  # 64 or 128
BATCH_SIZE = 128
NZ = 100
NGF = 64
NDF = 64
NUM_EPOCHS = 1000
LR = 0.0002
BETA1 = 0.5
SEED = 42
SAVE_INTERVAL = 50


# ============================================================
# 3. Setup
# ============================================================

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "samples"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)


# ============================================================
# 4. Dataset
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

print("Dataset size:", len(dataset))


# ============================================================
# 5. Models
# ============================================================

netG = DCGANGenerator(nz=NZ, ngf=NGF, nc=1, image_size=IMAGE_SIZE).to(device)
netD = DCGANDiscriminator(ndf=NDF, nc=1, image_size=IMAGE_SIZE, sigmoid=False).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

fixed_noise = torch.randn(25, NZ, 1, 1, device=device)


# ============================================================
# 6. Training Loop
# ============================================================

G_losses = []
D_losses = []

print("Starting Training...")

for epoch in range(1, NUM_EPOCHS + 1):

    for real, _ in dataloader:

        real = real.to(device)
        b_size = real.size(0)

        real_labels = torch.ones(b_size, device=device)
        fake_labels = torch.zeros(b_size, device=device)

        # ---------------------------
        # Train Discriminator
        # ---------------------------

        netD.zero_grad()

        output_real = netD(real)
        loss_real = criterion(output_real, real_labels)

        noise = torch.randn(b_size, NZ, 1, 1, device=device)
        fake = netG(noise)

        output_fake = netD(fake.detach())
        loss_fake = criterion(output_fake, fake_labels)

        lossD = loss_real + loss_fake
        lossD.backward()
        optimizerD.step()

        # ---------------------------
        # Train Generator
        # ---------------------------

        netG.zero_grad()

        output = netD(fake)
        lossG = criterion(output, real_labels)

        lossG.backward()
        optimizerG.step()

    G_losses.append(lossG.item())
    D_losses.append(lossD.item())

    print(f"Epoch [{epoch}/{NUM_EPOCHS}]  "
          f"Loss_D: {lossD.item():.4f}  "
          f"Loss_G: {lossG.item():.4f}")

    # Save samples
    if epoch % SAVE_INTERVAL == 0 or epoch == 1:
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        grid = vutils.make_grid(fake, padding=2, normalize=True)
        vutils.save_image(grid, os.path.join(OUTPUT_DIR, "samples",
                                             f"samples_epoch_{epoch:04d}.png"))

    # Save checkpoint
    if epoch % SAVE_INTERVAL == 0:
        torch.save({
            "epoch": epoch,
            "netG_state_dict": netG.state_dict(),
            "netD_state_dict": netD.state_dict(),
            "optimizerG_state_dict": optimizerG.state_dict(),
            "optimizerD_state_dict": optimizerD.state_dict(),
        }, os.path.join(OUTPUT_DIR, "checkpoints",
                        f"dcgan_epoch_{epoch:04d}.pth"))


# ============================================================
# 7. Save Final Model
# ============================================================

torch.save(netG.state_dict(), os.path.join(OUTPUT_DIR, "generator_final.pth"))
torch.save(netD.state_dict(), os.path.join(OUTPUT_DIR, "discriminator_final.pth"))

print("Training Complete.")


# ============================================================
# 8. Plot Loss Curves
# ============================================================

plt.figure(figsize=(8,5))
plt.title("Generator and Discriminator Loss")
plt.plot(G_losses, label="Generator")
plt.plot(D_losses, label="Discriminator")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.png"))
plt.close()
