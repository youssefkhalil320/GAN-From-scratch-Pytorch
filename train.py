from src.data_loader.MNISTDataModule import MNISTDataModule
from src.models.GAN import GAN
import torch
import torchvision
import pytorch_lightning as pl


AVAIL_GPUS = min(1, torch.cuda.device_count())


dm = MNISTDataModule()
model = GAN()

trainer = pl.Trainer(max_epochs=20, gpus=AVAIL_GPUS)
trainer.fit(model, dm)
