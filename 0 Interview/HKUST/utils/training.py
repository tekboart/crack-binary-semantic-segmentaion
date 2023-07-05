import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
# import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

import os
from tqdm import tqdm

# load my custom Classes/Functions/etc.
from utils.models.unet import UnetScratch


# set Hyperparameters
device = 'cuda:0'
lr = 1e-4
batch_size = 32
epochs = 3
num_workers = 2
image_height = 360
image_width = 640
pin_mem = True
load_model = False
train_img_dir = os.path.join("data", "traindata", "img")
train_mask_dir = os.path.join("data", "traindata", "mask")
val_img_dir = os.path.join("data", "valdata", "img")
val_mask_dir = os.path.join("data", "valdata", "mask")



def train_fn(loader, model, optimizer, loss_fn, scaler):
    pass

###############################################################################
# For testing
###############################################################################
if __name__ == "__main__":
    main()
