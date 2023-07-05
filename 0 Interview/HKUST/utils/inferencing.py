import torch
import torchvision.transforms.v2 as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import os

# Type Hinting
from typing import Union, Optional, TypeAlias, Callable, Container, Generator


def img_to_inference_tensor(
    img_addr: str,
    make_batch: bool = True,
    size: tuple[int, int] = None,
    data_format: str = "channels_first",
):
    '''
    Take an image path (str) and output a torch.Tensor (ready for inference).


    Parameters
    ----------

    size: tuple[int, int]
        A shape to resize the image up/down. (height, width)
    '''
    # TODO: instead of PIL use pure torch:
    # from torchvision.io import read_image, ImageReadMode
    # img = read_image('dog.jpeg', mode=ImageReadMode.GRAY)

    # load the img as PIL img
    img = Image.open(img_addr)

    if size:
        # height = size[0]
        # width = size[1]
        # remember PIL uses (W, H) instead of (H, W)
        img = img.resize(reversed(size))

    # method 1
    # this convert matrix (H, W) to tensor (1, H, W) (for gray imgs)
    img_tensor = TF.functional.pil_to_tensor(img)
    # method 2
    # img_tensor = TF.ToTensor()(img)

    if make_batch:
        img_tensor = img_tensor.unsqueeze(dim=0)

    if data_format == 'channels_last' and make_batch:
        img_tensor = img_tensor.moveaxis(1, -1)
    elif data_format == 'channels_last' and not make_batch:
        img_tensor = img_tensor.moveaxis(0, -1)

    return img_tensor


# TODO: create a func to normalize a torch.Tensor
def normalize_tensor(
    img_tensor: Union[torch.Tensor, np.ndarray],
    normalization_range: Union[tuple, list] = (-1, 1),
):
    """
    Normalizes an Image Tensor
        > needed for both tranining and inference
    """
    if tuple(normalization_range) == (-1, 1):
        img_tensor = (img_tensor / 127.5) - 1
    elif tuple(normalization_range) == (0, 1):
        img_tensor = img_tensor / 255.0

    return img_tensor


def inference_segmentation(
    img_batch: torch.Tensor,
    model: Callable,
    num_classes: int = 1,
    thresh: float = 0.5,
    data_format: str = "channels_first",
    normalize: bool = True,
    preprocess_func: Callable = None,
):
    """
    Passes an image to a segmentation model and returns its predicted mask Tensor.
        * This is for PyTorch, as keras has model.predict

    Paramaters
    ----------
    img_batch: torch.Tensor
        a 4DTensor of shape (m, C, H, W)
    model:
        a Segmentation model

    normalize: bool
        Should the input tensor get normalized?

    Returns
    -------
    """
    # normalize the image batch
    if normalize and not preprocess_func:
        img_batch = normalize_tensor(img_batch)
    elif normalize and preprocess_func:
        img_batch = preprocess_func(img_batch)

    # add the batch axis, if input is a 3Dtensor (single image)
    if len(img_batch.shape) == 3:
        img_batch = img_batch.unsqueeze(dim=0)

    yhat = model(img_batch)

    # construct the final mask (a gray img)
    if num_classes > 1:
        if data_format == "channels_first":
            # resulting tensor's shape: (m, 1, H, W)
            yhat_single_channel = torch.argmax(yhat, dim=1)
        elif data_format == "channels_first":
            # resulting tensor's shape: (m, H, W, 1)
            yhat_single_channel = torch.argmax(yhat, dim=-1)
    elif num_classes == 1:
        # all pixels below thresh are "not class" (i.e., background)
        yhat_single_channel = torch.where(yhat >= thresh, 255, 0)

    return yhat_single_channel


# TODO: write a inference_detection func (e.g., for YOLO)
def inference_detection():
    pass
