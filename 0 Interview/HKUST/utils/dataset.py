import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

import albumentations as A
# import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

import numpy as np
from PIL import Image


# TODO: where is the shuffle part? maybe in DataLoader


class SegmentaionDataset(Dataset):
    """
    # TODO: add docstring
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_ext: str = "jpg",
        mask_ext: str = "png",
        mask_suffix: str = "_mask",
        transform=None,
        num_classes: int = 1,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.mask_suffix = mask_suffix
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self) -> int:
        """
        # TODO: add docstring
        """
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        # TODO: add docstring
        """
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(
            self.mask_dir,
            self.images[idx].replace(
                f".{self.img_ext}", f"{self.mask_suffix}.{self.mask_ext}"
            ),
        )

        # NumPy implementation (channels_last format)
        # we used np.array, as opposed to PIL, as it will be converted to other formats (e.g., torch.Tensor) much easier
        # TODO: as numpy cannot run on GPU maybe it's better to use pure torch tensors instead.
        # TODO: why we define each element in channels_last format???
        # .convert('RGB') as img is RGB (3 channels)
        image = np.array(Image.open(img_path).convert("RGB"))
        # make channels_first
        # image = np.moveaxis(image, -1, 0)
        # print('img shape', image.shape)
        # .convert('L') as mask is grayscale
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # print('mask shape', mask.shape)
        # (H, W) --> (C=1, H, W) (an img must be 3darray not matrix)
        # mask = np.expand_dims(mask, 0)
        # print('mask shape', mask.shape)
        if self.num_classes == 1:
            # We need 0/1 for each pixel (not 0.0 or 255.), this is needed for sigmoid/softmax fn.
            mask = np.where(mask == 255.0, 1.0, mask)
        elif self.num_classes > 1:
            # convert a gray img with pixel values [0,1,..,C] to a (C, H, W) 3Dtensor
            # in each channel pixels must be either 0 or 1 (0: is_not_class_obj, 1:is_class_obj)
            pass

        # # PyTorch implementation (channels_first format)
        # image = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.RGB).float()
        # # print('img shape', image.shape)
        # mask = torchvision.io.read_image(mask_path, torchvision.io.ImageReadMode.GRAY).float()
        # # print('mask shape', mask.shape)

        # if self.num_classes == 1:
        #     # We need 0/1 for each pixel (not 0.0 or 255.), this is needed for sigmoid/softmax fn.
        #     mask = torch.where(mask == 255.0, 1.0, mask)
        # elif self.num_classes > 1:
        #     # convert a gray img with pixel values [0,1,..,C] to a (C, H, W) 3Dtensor
        #     # in each channel pixels must be either 0 or 1 (0: is_not_class_obj, 1:is_class_obj)
        #     pass

        if self.transform:
            # This implementation works with transfomrs that input both img and mask (and return both)
            # e.g., alumentations and/or torchvision transformers
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations.get("image")
            # print('augmentations image:', augmentations['image'].shape)
            mask = augmentations.get("mask")
            # print('augmentations mask:', augmentations['mask'].shape)
            # print('type mask', augmentations['mask'].__class__)

        # make the image, mask channels_first
        # Only if not using alumentations's ToTensorV2 (in self.transform)
        # which automatically reshape's the img, mask for pytorch models (caused error and shape mismatch)
        image = np.moveaxis(image, -1, 0)
        # print('image shape', image.shape)
        mask = np.expand_dims(mask, 0)
        # print('mask shape', mask.shape)

        return image, mask


def get_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    '''
    Preparing your data for training with DataLoaders
    1. put samples in “minibatches”
    2. reshuffle the data at every epoch to reduce model overfitting (only for train set)
    3. use Python’s multiprocessing to speed up data retrieval
    '''
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_dataloader, val_dataloader


###############################################################################
# For testing
###############################################################################
if __name__ == "__main__":
    pass
