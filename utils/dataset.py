import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from typing import Union


class SegmentaionDataset(Dataset):
    """
    # TODO: add docstring

    Attributes
    ----------
    subset: tuple|list
        Set the start and end indices, to avoid
        >>> SegmentationDataset(image_dir, mask_dir, subset=[0, 10])
        >>> # takes only 10 samples to create the dataset (i.e., [0:10])
    data_format: str ('channels_first' (default) | 'channels_last')
        define the final channel order for (img, mask) be channels_first (default) or channels_last

    pretrained_preprocess_fn: Callable
        A func the simulates the same preprocessing steps that inputs of the pretrained model have undergone.

        \* must take and output channels_last image.

        \** If use this, then no normalization must be done in self.transform (i.e., image augmentation)

    Methods
    -------
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_ext: str = "jpg",
        mask_ext: str = "png",
        mask_suffix: str = "_mask",
        transform=None,
        preprocess_fn=None,
        num_classes: int = 1,
        data_format: str = "channels_first",
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
        self.preprocess_fn = preprocess_fn
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
            #FIXME: use Regex to get the name of mask images
            # for an image like 23432.jpg.jpg it would cause an error but with Regex it will work
            # pattern (not tested): r".+\.(P<ext>\w+)\b)""
            self.images[idx].replace(
                f".{self.img_ext}", f"{self.mask_suffix}.{self.mask_ext}"
            ),
        )

        # TODO: why we define each element in channels_last format??? for self.transform
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.num_classes == 1:
            # make the mask binary (float) (needed for sigmoid/softmax)
            # method 1: doesn't work if the mask's pixel are not 0/255 or 0/1
            # *** e.g., the Kaggle Crack Segmentation Dataset's masks were not binary
            # np.unique(<a_mask>) = array([0, 1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255], dtype=uint8)
            # mask = np.where(mask == 255.0, 1.0, mask)
            # method 2: is robust
            mask = np.where(mask == 0.0, 0.0, 1.0)
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

        if self.preprocess_fn:
            image = self.preprocess_fn(image)
            assert isinstance(image, np.ndarray)

        # make the (image, mask) channels_first (as needed by pytorch models)
        # Only if not using alumentations's ToTensorV2 (in self.transform)
        # which automatically reshape's the img, mask for pytorch models (caused error and shape mismatch)
        if self.data_format == "channels_first":
            image = np.moveaxis(image, -1, 0)
            assert image.shape[0] == 3
            mask = np.moveaxis(mask, -1, 0)
            assert mask.shape[0] == self.num_classes

        return image, mask

    # TODO: create the __iter__ methodd (must be like tf.Dataset to use next(iter()))
    # def __iter__(self):
    #     pass

    def pred_num_batches(self, batch_size: int) -> int:
        """
        Predict the #mini_batches after getting sliced to batches (e.g., by Loader).
        """
        return np.ceil(len(self) / batch_size).astype("int")


def get_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    worker_init_fn=None,
    generator=None,
):
    """
    Preparing your data for training with DataLoaders
    1. put samples in “minibatches”
    2. reshuffle the data at every epoch to reduce model overfitting (only for train set)
    3. use Python’s multiprocessing to speed up data retrieval


    Returns
    -------
    list
        The #loaders in the output depends on which of the train_ds, val_ds, or test_ds were provided in input.
    """
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    return train_dataloader, val_dataloader


###############################################################################
# For testing
###############################################################################
if __name__ == "__main__":
    pass
