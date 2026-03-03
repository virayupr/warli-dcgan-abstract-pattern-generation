# models/dcgan.py
# ============================================================
# DCGAN architecture (PyTorch) for 1-channel (grayscale) images
# Supports 64x64 and 128x128 via image_size parameter.
# ============================================================

from __future__ import annotations
import torch
import torch.nn as nn


def weights_init_dcgan(m: nn.Module) -> None:
    """
    DCGAN weight initialization:
    - Conv/ConvTranspose: N(0, 0.02)
    - BatchNorm weight: N(1, 0.02), bias = 0
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class DCGANGenerator(nn.Module):
    """
    DCGAN Generator.
    Produces (nc, image_size, image_size) images from latent vector z.

    image_size: 64 or 128 (common DCGAN sizes).
    nc: number of output channels (1 for grayscale).
    nz: latent dimension (100 typical).
    ngf: feature map base width (64 typical).
    """
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 1, image_size: int = 64):
        super().__init__()
        assert image_size in (64, 128), "image_size must be 64 or 128 for this implementation."

        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.image_size = image_size

        if image_size == 64:
            # z -> 4 -> 8 -> 16 -> 32 -> 64
            self.net = nn.Sequential(
                # (nz) x 1 x 1 -> (ngf*8) x 4 x 4
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                # (ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                # -> (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                # -> (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                # -> (nc) x 64 x 64
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        else:  # 128
            # z -> 4 -> 8 -> 16 -> 32 -> 64 -> 128
            self.net = nn.Sequential(
                # (nz) x 1 x 1 -> (ngf*16) x 4 x 4
                nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),

                # -> (ngf*8) x 8 x 8
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                # -> (ngf*4) x 16 x 16
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                # -> (ngf*2) x 32 x 32
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                # -> (ngf) x 64 x 64
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                # -> (nc) x 128 x 128
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        self.apply(weights_init_dcgan)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z expected shape: (B, nz, 1, 1)
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN Discriminator.
    Returns a probability (after sigmoid) or logits if sigmoid=False.

    image_size: 64 or 128
    nc: input channels (1 for grayscale)
    ndf: feature map base width (64 typical)
    """
    def __init__(self, ndf: int = 64, nc: int = 1, image_size: int = 64, sigmoid: bool = True):
        super().__init__()
        assert image_size in (64, 128), "image_size must be 64 or 128 for this implementation."
        self.sigmoid = sigmoid

        if image_size == 64:
            # 64 -> 32 -> 16 -> 8 -> 4 -> 1
            layers = [
                # (nc) x 64 x 64 -> (ndf) x 32 x 32
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                # -> (ndf*2) x 16 x 16
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                # -> (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                # -> (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                # -> 1 x 1 x 1 (logit)
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            ]
        else:
            # 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 1
            layers = [
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            ]

        if sigmoid:
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        self.apply(weights_init_dcgan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: (B, 1, 1, 1) -> flatten to (B,)
        out = self.net(x)
        return out.view(-1)


# Quick sanity test (optional)
if __name__ == "__main__":
    z = torch.randn(8, 100, 1, 1)
    G = DCGANGenerator(nz=100, ngf=64, nc=1, image_size=64)
    x = G(z)
    D = DCGANDiscriminator(ndf=64, nc=1, image_size=64, sigmoid=True)
    y = D(x)
    print("G out:", x.shape, "D out:", y.shape)
