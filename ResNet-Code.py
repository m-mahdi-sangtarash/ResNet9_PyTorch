import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm, trange
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive


# Downloading From Source
dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
download_and_extract_archive(dataset_url, download_root="./data")

print(os.listdir("./data/cifar10/train"))
print(os.listdir("./data/cifar10/test"))

mean_std_values = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_transform = transforms.Compose([transforms.RandomCrop(32),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.AugMix(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(*mean_std_values, inplace=True)])
val_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(*mean_std_values)])

train_set = ImageFolder("./data/cifar10", train_transform)
valid_set = ImageFolder("./data/cifar10", val_transform)

batch_size = 256
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)