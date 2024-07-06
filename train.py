from src.data_loader.MNISTDataModule import MNISTDataModule
from src.models.GAN import GAN


dm = MNISTDataModule()
model = GAN()
model.plot_imgs()
