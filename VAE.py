"""NICE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

class Model(nn.Module):
    def __init__(self, latent_dim, device):
        """Initialize a VAE.

        Args:
            latent_dim: dimension of embedding
            device: run on cpu or gpu
        """
        super(Model, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 1, 2),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  7, 7
        )

        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.upsample = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 1, 2),  # B, 1, 28, 28
            nn.Sigmoid()
        )

    # def sample(self, sample_size: int = None, mu=None, logvar=None) -> torch.Tensor:
    #     """
    #     :param sample_size: Number of samples
    #     :param mu: z mean, None for init with zeros
    #     :param logvar: z logstd, None for init with zeros
    #     :return:
    #     """
    #     if mu==None:
    #         mu = torch.zeros((sample_size, self.latent_dim)).to(self.device)
    #     if logvar == None:
    #         logvar = torch.zeros((sample_size, self.latent_dim)).to(self.device)
    #     pass

    def sample(self, sample_size: int = None):
        sampled_x = torch.rand((sample_size, self.latent_dim)).to(self.device)
        recon_x = self.decoder(self.upsample(sampled_x).view(-1, 64, 7, 7))
        save_image(recon_x, "output.jpg")

    def z_sample(self, mu, logvar) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std).to(self.device)
        return mu + eps * std

    @staticmethod
    def loss(x, recon, mu, logvar):
        # Binary cross entropy loss
        BCE = F.binary_cross_entropy(recon, x, reduction='sum')
        # KL Divergence loss
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KL

    def forward(self, x: torch.Tensor):
        x_latent = self.encoder(x).view(-1, 64*7*7)
        mu, logvar = self.mu(x_latent), self.logvar(x_latent)
        z = self.z_sample(mu=mu, logvar=logvar)
        return self.decoder(self.upsample(z).view(-1, 64, 7, 7)), mu, logvar
