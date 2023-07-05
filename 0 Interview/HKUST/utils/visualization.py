import torch
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import os

# Type Hinting
from typing import Union, Optional, TypeAlias

PILImage: TypeAlias = Image.Image


def plot_segment_inference(
    img: Union[Image.Image, np.ndarray],
    mask: Union[PILImage, np.ndarray],
    mask_target: Optional[Union[PILImage, np.ndarray]],
):
    """
    plot the image and its corresponding
        all input image Tensors must be channels_first (as needed by plt)
    """
    pass


def torch_tensor_for_plt(
    tensor: torch.Tensor,
    data_format: str = "channels_first",
    to_numpy: bool = True,
):
    """
    make a torch tensor ready to be plotted by plt

    Parameters
    ----------
    tensor: torch.Tensor
        a 3D or 4D tensor of shape: ([m,] C, H, W) or ([m,], H, W, C)
    data_format: st
        The channel format. either 'channels_first' (m, C, H, W) or 'channels_last' (m, H, W, C)
    """
    tensor_rank = len(tensor.shape)

    if tensor_rank not in [3, 4]:
        raise ValueError("the tensor's rank must be either 3 or 4")

    # define the channel's axis
    if tensor_rank == 3:
        channel_axis = 0
    elif tensor_rank == 4:
        channel_axis = 1

    # make it channels_last (as needed by plt)
    if data_format == "channels_first":
        tensor = tensor.moveaxis(channel_axis, -1)

    if to_numpy:
        if isinstance(tensor, torch.Tensor):
            # use detach() if tensor requires grad
            tensor.detach().numpy()
        elif isinstance(tensor, tf.Tensor):
            tensor.numpy()

    return tensor


def plot_segmentation_inference(img_batch, mask_batch, yhat_mask_batch, limit: int = None):

    '''
    Take three Tensors with equal batch_size and plot them side by side (each record in on row)

    Parameters
    ----------

    truncate: int
        Set a value to plot only a limited number of examples. (must be >= 1)
    '''

    img_ndim = len(img_batch.shape)
    mask_ndim = len(img_batch.shape)
    yhat_ndim = len(img_batch.shape)

    if not all(dim == 4 for dim in (img_ndim, mask_ndim, yhat_ndim)):
        raise ValueError("All input tensors must be of shape (m, H, W, C)")

    if limit:
        batch_size = limit
    else:
        batch_size = img_batch.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(30, 10))

    if batch_size == 1:
        axes[0].imshow(img_batch[0])
        axes[0].set_title('Image', fontsize=16)
        axes[1].imshow(yhat_mask_batch[0], cmap='gray')
        axes[1].set_title('Mask (predicted)', fontsize=16)
        axes[2].imshow(mask_batch[0], cmap='gray')
        axes[2].set_title('Mask (target)', fontsize=16)
    elif batch_size > 1:
        for row in range(batch_size):
            axes[row][0].imshow(img_batch[row])
            axes[row][0].set_title('Image', fontsize=16)
            axes[row][1].imshow(yhat_mask_batch[row], cmap='gray')
            axes[row][1].set_title('Mask (predicted)', fontsize=16)
            axes[row][2].imshow(mask_batch[row], cmap='gray')
            axes[row][2].set_title('Mask (target)', fontsize=16)

    plt.show()