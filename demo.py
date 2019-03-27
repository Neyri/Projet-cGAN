
import matplotlib.pyplot as plt
import imageio
import glob
import random
import os
import numpy as np
import math
import itertools
import time
import datetime
from pathlib import Path
from PIL import Image
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from helper import *
from model import *
from datasets import *

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

# Load Gen and Discriminator
generator.load_state_dict(torch.load('2nd_try_G.pth'))
discriminator.load_state_dict(torch.load('2nd_try_D.pth'))

# Eval mode
discriminator.eval()
generator.eval()

# Define Tensor
Tensor = torch.FloatTensor

#parmaeters
epoch = 0 #  epoch to start training from
n_epoch = 200  #  number of epochs of training
batch_size =10  #  size of the batches
lr = 0.0002 #  adam: learning rate
b1 =0.5  #  adam: decay of first order momentum of gradient
b2 = 0.999  # adam: decay of first order momentum of gradient
decay_epoch = 100  # epoch from which to start lr decay
img_height = 256  # size of image height
img_width = 256  # size of image width
channels = 3  # number of image channels
sample_interval = 500 # interval between sampling of images from generators
checkpoint_interval = -1 # interval between model checkpoints


def run(args):
    path = args.path 
    n_display = args.n_display
    do_store = args.do_store
    output = args.output

    for i in range(n_display) :
        # Configure dataloaders
        transforms_ = [ transforms.Resize((img_height, img_width), Image.BICUBIC),
                transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

        # Load data
        val_dataloader = DataLoader(ImageDataset(path, transforms_=transforms_, mode='val'),
                            batch_size=n_display, shuffle=False)
        image, mask = next(iter(val_dataloader))
        output = generator(mask.type(Tensor))
        output = output.view(n_display, 3, 256, 256)
        output = output.cpu().detach()
        image = reverse_transform(image[i])
        output = reverse_transform(output[i])
        mask = reverse_transform(mask[i])
        if  do_store == False:
            plot2x3Array(image, mask,output)
        else:
            print('Data stored in folder abc')


def main():
    parser=argparse.ArgumentParser(description="generate results from trained cGAN")
    parser.add_argument("-in",help="path to input file" ,dest="path", type=str,required=True)
    parser.add_argument("-n",help="number of data to display" ,dest="n_display",type=int, default=1)
    parser.add_argument("-s",help="store the data on a specific folder",dest="do_store",type=bool,default=False)
    parser.add_argument("-out",help="output filename",dest="output",type=str,required=False)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
	main()


