import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST


random_seed = 42
torch.manual_seed(random_seed)

BATCH_SIZE = 128
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = int(os.cpu_count() / 2)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data",
                 batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True,
                               transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000])

        # Assign test dataset
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    # Instantiate the data module
    data_module = MNISTDataModule()

    # Prepare the data (download if necessary)
    data_module.prepare_data()

    # Set up the data (train/val/test splits)
    data_module.setup(stage='fit')
    data_module.setup(stage='test')

    # Check the train dataloader
    train_dataloader = data_module.train_dataloader()
    print("Train DataLoader length:", len(train_dataloader))

    # Iterate through the training data loader
    for batch_idx, (data, target) in enumerate(train_dataloader):
        print(
            f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
        if batch_idx == 1:  # Print only the first two batches
            break

    # Check the validation dataloader
    val_dataloader = data_module.val_dataloader()
    print("Validation DataLoader length:", len(val_dataloader))

    # Iterate through the validation data loader
    for batch_idx, (data, target) in enumerate(val_dataloader):
        print(
            f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
        if batch_idx == 1:  # Print only the first two batches
            break

    # Check the test dataloader
    test_dataloader = data_module.test_dataloader()
    print("Test DataLoader length:", len(test_dataloader))

    # Iterate through the test data loader
    for batch_idx, (data, target) in enumerate(test_dataloader):
        print(
            f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
        if batch_idx == 1:  # Print only the first two batches
            break
