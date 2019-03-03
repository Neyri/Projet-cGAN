import os
import torch
# import pandas as pd
from skimage import io, transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ImageDatasets(Dataset):
    """Image Datasets with side to side image and labels"""

    def __init__(self, path, transform=None, dtype='jpg'):
        """
        Args:
            path (string): the path to access the directory with the images
            transform (callable, optional): Optional transform to be applied on a sample.
            dtype (string, optional): Type of the images to read. Default is jpg
        """
        self.path = path
        self.transform = transform
        self.img_list = glob.glob(os.path.join(path, '*.' + dtype))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img = Image.open(img_name)
        w, h = img.size
        w_split = int(w / 2)
        # TODO: change this to PIL
        photo = img.crop((0, 0, w_split, h))
        labels = img.crop((w_split, 0, w, h))
        sample = {'image': photo, 'labels': labels}

        if self.transform:
            # make a seed with numpy generator
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)  # apply this seed to img tranfsorms
            random.seed(seed)
            sample['image'] = self.transform(sample['image'])
            sample['labels'] = self.transform(sample['labels'])
            # sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    path = 'dataset/edges2shoes/train'
    ds = ImageDatasets(path)
    # ds2 = ImageDatasets(path, transform=transforms.ToTensor())
    idx = np.random.randint(len(ds))
    sample = ds[idx]
    idx2 = np.random.randint(len(ds))
    sample2 = ds[idx2]
    plt.subplot(221)
    plt.imshow(sample['image'])
    plt.title('image')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(sample['labels'])
    plt.title('labels')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(sample2['image'])
    plt.title('image')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(sample2['labels'])
    plt.title('labels')
    plt.axis('off')
    plt.show()
