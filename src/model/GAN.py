import pytorch_lightning as pl
from .generator import Generator
from .discriminator import Discriminator
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.0002):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim=self.hpparams.latent_dim)
        self.validation_z = torch.randn(6, self.hpparams.latent_dim)
