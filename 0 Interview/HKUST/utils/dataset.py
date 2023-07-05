import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

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
        transform = None,
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

        # we used np.array, as opposed to PIL, as it will be converted to other formats (e.g., torch.Tensor) much easier
        # TODO: as numpy cannot run on GPU maybe it's better to use pure torch tensors instead.
        # .convert('RGB') as img is RGB (3 channels)
        image = np.array(Image.open(img_path).convert("RGB"))
        # .convert('L') as mask is grayscale
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # (H, W) --> (C=1, H, W) (an img must be 3darray not matrix)
        mask = np.expand_dims(mask, 0)
        if self.num_classes == 1:
            # We need 0/1 for each pixel (not 0.0 or 255.), this is needed for sigmoid/softmax fn.
            mask = np.where(mask == 255.0, 1.0, mask)
        elif self.num_classes > 1:
            # convert a gray img with pixel values [0,1,..,C] to a (C, H, W) 3Dtensor
            # in each channel pixels must be either 0 or 1 (0: is_not_class_obj, 1:is_class_obj)
            pass

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            if isinstance(augmentations, (list, tuple)):
                image = augmentations[0]
                mask = augmentations[1]
            elif isinstance(augmentations, dict):
                image = augmentations.get("image")
                mask = augmentations.get("mask")

        return image, mask


def get_loaders(
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers: int = 4,
        pin_memory: bool = True,
):
    # create our datasets
    train_ds = SegmentaionDataset(image_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_transform)
    val_ds = SegmentaionDataset(image_dir=val_img_dir, mask_dir=val_mask_dir, transform=val_transform)

    # Preparing your data for training with DataLoaders
    # 1. ass samples in “minibatches”
    # 2. reshuffle the data at every epoch to reduce model overfitting (only for train set)
    # 3. use Python’s multiprocessing to speed up data retrieval
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_dataloader, val_dataloader


###############################################################################
# For testing
###############################################################################
if __name__ == "__main__":
    pass
