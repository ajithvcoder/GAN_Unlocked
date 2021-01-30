import argparse
import os
import numpy as np 
import math 
import torchvision.transforms as transforms 
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn 
import torch.nn.functional as F 
import torch 

os.makedirs("images",exist_ok=True)

# constants
n_epochs = 200
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 8
latent_dim = 100
img_size = 28
channels = 1
sample_interval = 400
img_shape = (channels, img_size, img_size)
if torch.cuda.is_available():
  cuda = True
else:
  cuda = False 


"""
What is nn.Module ?

- https://github.com/torch/nn/blob/master/doc/module.md

- https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py

What is super(Generator,self).__init__() ?

- https://stackoverflow.com/a/33469090

- https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods

nn.linear 

- https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

"""

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    def block(in_feat, out_feat, normalize=True):
      layers = [nn.Linear(in_feat, out_feat)]
      if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
      layers.append(nn.LeakyReLU(0.2, inplace=True))
      return layers 
    self.model = nn.Sequential(
        # latent vector of a particular size , 128 channels
        *block(latent_dim, 128, normalize=False),
        *block(128, 256),
        *block(256,512),
        *block(512,1024),
        nn.Linear(1024, int(np.prod(img_shape))),
        nn.Tanh()
    )
  def forward(self, z):
    # only when forward pass is initated it will be running
    img = self.model(z)
    print(img.shape)
    # reshaping the img
    img = img.view(img.size(0), *img_shape)
    return img 

# loss function
adversarial_loss = torch.nn.BCELoss()

# initalizing generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
  generator.cuda()
  discriminator.cuda()
  adversarial_loss.cuda()

# data loader
os.makedirs("data/mnist",exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train = True,
        download = True,
        transform = transforms.Compose( \
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]
        ),
    ),
    batch_size = batch_size,
    shuffle = True,
)


#optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1,b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1,b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


