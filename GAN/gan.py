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

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(int(np.prod(img_shape)),512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 1),
        nn.Sigmoid(),

    )
  def forward(self, img):
    # flattening tensors
    img_flat = img.view(img.size(0), -1)
    validity = self.model(img_flat)
    return validity

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


for param in generator.parameters():
  print(type(param), param.size())


for epoch in range(n_epochs):
  for i, (imgs, _) in enumerate(dataloader):
    #ground truths definition
    # valid
    valid = Variable(Tensor(imgs.size(0),1).fill_(1.0), requires_grad=False )
    # fake
    fake = Variable(Tensor(imgs.size(0),1).fill_(0.0), requires_grad= False)

    # Configure inuput
    real_imgs = Variable(imgs.type(Tensor))

    ########## Train generator ################

    # clears the output of all variables from previous iteration
    optimizer_G.zero_grad()

    # sample noise as gen input
    z = Variable(Tensor(np.random.normal(0,1,(imgs.shape[0], latent_dim))))
    print(z.shape)

    gen_imgs = generator(z)
    # checking how much genrators image the discrimainator is able to classify as valid and then taking it as a loss and back propagating
    g_loss = adversarial_loss(discriminator(gen_imgs), valid)
    # backpropagation
    g_loss.backward()

    ######## training discriminator ###############

    optimizer_D.zero_grad()

    real_loss = adversarial_loss(discriminator(real_imgs), valid)

    # tensor.detach() creates a tensor that shares storage with tensor that does not require grad. It detaches the output from the computational graph
    # https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch#:~:text=detach()%20creates%20a%20tensor,output%20from%20the%20computational%20graph.

    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()),fake)
    d_loss = (real_loss + fake_loss)/2
    # backpropagation
    d_loss.backward()

    optimizer_D.step()

    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
          (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

    batches_done = epoch * len(dataloader) + i
    if batches_done % sample_interval == 0:
      save_image(gen_imgs.data[25], "images/%d.png"%batches_done, nrow=5, normalize=True)

