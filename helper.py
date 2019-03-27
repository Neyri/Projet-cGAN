
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

def plot2x2Array(image, mask, save = False,direc = None ,fname= None):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image)
    axarr[1].imshow(mask)
    axarr[0].set_title('Image')
    axarr[1].set_title('Mask')
    if save != True:
        plt.show()
    else:
        if os.path.isdir(direc):
            plt.savefig(direc+"/"+fname+".png")
        else:
            os.mkdir(direc)
            plt.savefig(direc+"/"+fname+".png")

def reverse_transform(image):
    image = image.numpy().transpose((1, 2, 0))
    image = ((image+1)/2*255).astype(np.uint8)
    image = np.clip(image, 0,255)    
    return image

def plot2x3Array(image, mask,predict, save=False,direc = None, fname= None):
    f, axarr = plt.subplots(1,3,figsize=(15,15))
    axarr[0].imshow(image)
    axarr[1].imshow(mask)
    axarr[2].imshow(predict)
    axarr[0].set_title('input')
    axarr[1].set_title('real')
    axarr[2].set_title('fake')
    if save != True:
        plt.show()
    else:
        if os.path.isdir(direc):
            plt.savefig(direc+"/"+fname+".png")
        else:
            os.mkdir(direc)
            plt.savefig(direc+"/"+fname+".png")