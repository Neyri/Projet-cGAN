
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


from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from model import *
from datasets import *


# Load Gen and Discriminator
discriminator.load_state_dict(torch.load('2nd_try_D.pth'))
generator.load_state_dict(torch.load('2nd_try_G.pth'))
# Eval mode
discriminator.eval()
generator.eval()

# Load data
val_dataloader = DataLoader(ImageDataset("val", transforms_=transforms_, mode='val'),
                            batch_size=1, shuffle=False)

try:
    image, mask = next(iter(dataloader))
    output = generator(mask.type(Tensor))
    output = output.view(16, 3, 256, 256)
    output = output.cpu().detach()
    image = reverse_transform(image[0])
    output = reverse_transform(output[0])
    mask = reverse_transform(mask[0])
    plot2x3Array(mask,image,output)
except:
    pass