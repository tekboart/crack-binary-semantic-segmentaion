import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from typing import Union

class SegmentaionDataset(Dataset):
    """
    # TODO: add docstring

    Parameters
    ----------
    subset: tuple|list
        Set the start and end indices, to avoid
        >>> SegmentationDataset(image_dir, mask_dir, subset=[0, 10])
        takes only 10 samples to create the dataset (i.e., [0:10])
    data_format: st
        define whether the final (img, mask) be channels_first (default) or channels_last
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
        data_format: str = 'channels_first',
        subset: Union[tuple, list] = None,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = (
            os.listdir(image_dir)
            if not subset
            else os.listdir(image_dir)[subset[0] : subset[1]]
        )
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.mask_suffix = mask_suffix
        self.num_classes = num_classes
        self.transform = transform
        self.data_format = data_format

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

        # TODO: why we define each element in channels_last format??? for self.transform
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        if self.num_classes == 1:
            # make the mask binary (float) (needed for sigmoid/softmax)
            mask = np.where(mask == 255.0, 1.0, mask)
            # make the mask a 3DTensor: (H, W) --> (H, W, C=1)
            mask = np.expand_dims(mask, -1)
        elif self.num_classes > 1:
            # convert a gray img with pixel values [0,1,..,C] to a (H, W, C=num_classes) 3Dtensor
            # in each channel pixels must be binary float (0.0: is_not_class_obj, 1.0:is_class_obj)
            pass

        if self.transform:
            # This implementation works with transfomrs that input both img and mask (and return both)
            # e.g., alumentations and/or torchvision transformers
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations.get("image")
            mask = augmentations.get("mask")

        # make the (image, mask) channels_first (as needed by pytorch models)
        # Only if not using alumentations's ToTensorV2 (in self.transform)
        # which automatically reshape's the img, mask for pytorch models (caused error and shape mismatch)
        if self.data_format == 'channels_first':
            image = np.moveaxis(image, -1, 0)
            assert image.shape[0] == 3
            mask = np.moveaxis(mask, -1, 0)
            assert mask.shape[0] == self.num_classes
        # elif self.data_format == 'channels_last':
        #     assert image.shape[-1] == 3
        #     mask = np.expand_dims(mask, -1)

        return image, mask

    # TODO: create the __iter__ methodd (must be like tf.Dataset to use next(iter()))
    # def __iter__(self):
    #     pass


def get_loaders(
    train_ds: Dataset,
    val_ds: Dataset = None,
    test_ds: Dataset = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Preparing your data for training with DataLoaders
    1. put samples in “minibatches”
    2. reshuffle the data at every epoch to reduce model overfitting (only for train set)
    3. use Python’s multiprocessing to speed up data retrieval
    """
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    if val_ds:
        val_dataloader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
    else:
        val_dataloader = None

    if test_ds:
        test_dataloader = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
    else:
        test_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader


###############################################################################
# For testing
###############################################################################
if __name__ == "__main__":
    pass
