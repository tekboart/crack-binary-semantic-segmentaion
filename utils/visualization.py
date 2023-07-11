import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Type Hinting
from typing import Union, Optional, TypeAlias

PILImage: TypeAlias = Image.Image


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
        # use detach() if tensor requires grad
        tensor.detach().cpu().numpy()

    return tensor


#TODO: I've written a better fn --> delete it
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
        img = img_batch[0]
        yhat_mask = yhat_mask_batch[0]
        mask = mask_batch[0]
        axes[0].imshow(img)
        axes[0].set_title(f'Image {[*img.shape]}', fontsize=16)
        axes[0].axis('off')
        axes[1].imshow(yhat_mask, cmap='gray')
        axes[1].set_title(f'Mask (predicted) {[*yhat_mask.shape]}', fontsize=16)
        axes[1].axis('off')
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title(f'Mask (target) {[*mask.shape]}', fontsize=16)
        axes[2].axis('off')
    elif batch_size > 1:
        for row in range(batch_size):
            img = img_batch[row]
            yhat_mask = yhat_mask_batch[row]
            mask = mask_batch[row]
            axes[row][0].imshow(img)
            axes[row][0].set_title(f'Image {[*img.shape]}', fontsize=16)
            axes[row][0].axis('off')
            axes[row][1].imshow(yhat_mask, cmap='gray')
            axes[row][1].set_title(f'Mask (predicted) {[*yhat_mask.shape]}', fontsize=16)
            axes[row][1].axis('off')
            axes[row][2].imshow(mask, cmap='gray')
            axes[row][2].set_title(f'Mask (target) {[*mask.shape]}', fontsize=16)
            axes[row][2].axis('off')

    plt.show()

class ImageAntiStandardize:
    """
    revert the z-scored image to un-zscored values, needed for displaying it (e.g., plt.imshow).

    Parameters
    ----------
    img: np.ndarray
    channels_mean: list
        a list mean values for each channel in the input img.
        The len(channels_mean) must match the #C in input img.
        Default values are for ImageNet (need RGB (C=3) image).
    channels_std: list
        a list std values for each channel in the input img.
        The len(channels_mean) must match the #C in input img.
        Default values are for ImageNet (need RGB (C=3) image).
    clip: list
        We must put the unstandardized values to an acceptable range for image data. Best values are [0, 1] (default) and [0, 255]
        * as Z-score works with probability and reversing the process doesn't give us the original values.
        But, their relationship is restored and all we need to is to clip them.
        >>> img_zscore = (img - mean) / std
        >>> 'values are mostly (68.3%) in [-1, 1] but not necessarily
        >>> img = (img - mean) / std
        >>> img == img_restored
    """
    def __init__(
        self,
        channels_mean: list = [0.485, 0.456, 0.406],
        channels_std: list = [0.229, 0.224, 0.225],
        clip: list = [0, 1],
    ) -> None:
        self.channels_mean = channels_mean
        self.channels_std = channels_std
        self.clip = clip

    def __call__(self, image):
        '''
        Parameters
        ----------
        img: np.ndarray
        '''
        mean = np.array(self.channels_mean)
        std = np.array(self.channels_std)
        image = std * image + mean
        # force the values to be in a specific range
        image = np.clip(image, self.clip[0], self.clip[1])

        return image


def image_batch_to_ndarray_channels_first(batched_img_tensor, data_format: str):
    """
    Convert a batched image tensor to ndarray and make it channels_first.

    Parameters
    ----------
    batched_img_tensor: Union[np.ndarray, torch.Tensor, tf.Tensor]
        a batch of images of shape (m, H, W, C) or (m, C, H, W)
    data_format: str | 'channels_first' or 'channels_last'
    """
    tensor_rank = len(batched_img_tensor.shape)

    if tensor_rank == 3:
        channel_first_to_channel_last_order = (1, 2, 0)
    elif tensor_rank == 4:
        channel_first_to_channel_last_order = (0, 2, 3, 1)
    else:
        raise ValueError("the #axes (aka rank) for input tensor must be either 3 or 4")

    if isinstance(batched_img_tensor, np.ndarray):
        if data_format == "channels_first":
            batched_img_tensor = batched_img_tensor.transpose(
                channel_first_to_channel_last_order
            )
    elif isinstance(batched_img_tensor, torch.Tensor):
        if data_format == "channels_first":
            # TODO: add .detach().cpu() (if need for batches from inference)
            batched_img_tensor = batched_img_tensor.numpy().transpose(
                channel_first_to_channel_last_order
            )
    #TODO: Should I add support for tf.Tensor
    # I have to import the entire tensorflow to just use tf.Tensor type???
    # get so much MEM with no use
    #TODO: if not remove tf.Tensor from ValueError (below) as well
    # elif isinstance(batched_img_tensor, Tensor):
    #     if data_format == "channels_first":
    #         batched_img_tensor = batched_img_tensor.numpy().transpose(
    #             channel_first_to_channel_last_order
    #         )
    else:
        raise ValueError(
            "the type/class of the input tensor must be either (1) np.ndarray, (2) torch.Tensor, (3) tf.Tensor."
        )

    return batched_img_tensor


def image_mask_plot(
    batch_list: list[np.ndarray],
    data_format: str,
    num_rows: int = 1,
    titles: list = ["image", "mask (target)", "mask (predict)"],
    plot_axes: bool = False,
    anti_standardize_fn = None,
):
    """
    plot image, mask pairs of a given batch of data.

    Parameters
    ----------
    batch_list: list
        a list of [torch.Tensors] with shape (N, H, W, C) or (N, C, W, C)
        e.g., [img_batch, mask_batch, yhat_batch]
    data_format: str | 'channels_first' or 'channels_last'
    num_rows: int
    titles: list
    plot_axes: bool
        Should the axes be plotted (to show image dimensions)
    anti_standardize_fn: bool
        if the input image (the first elem of batch_list) was standardized/normalized (e.g., by z-score), then give us the anti_fn to reverse it.
        The resulting image must be in range of either [0, 1] (float) or [0, 255] (int)
    """
    batch_list = list(batch_list)
    num_cols = len(batch_list)

    # convert batch_tensors to ndarray + make it channels_last
    for i in range(num_cols):
        batch_list[i] = image_batch_to_ndarray_channels_first(
            batch_list[i], data_format
        )

    # plot img, mask (side-by-side)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    # make the axes a 2darray (in case num_rows=1, hence axes is 1darray)
    if axes.ndim == 1:
        axes = np.expand_dims(axes, 0)
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            img = batch_list[col_idx][row_idx]

            # different setting for images and masks
            if img.shape[-1] == 1:
                axes[row_idx][col_idx].imshow(img, cmap="gray")
            else:
                if anti_standardize_fn:
                    img = anti_standardize_fn(img)
                axes[row_idx][col_idx].imshow(img)

            # set the col titles only for the first row
            if row_idx == 0:
                axes[row_idx][col_idx].set_title(titles[col_idx], fontsize=14)

            if not plot_axes:
                axes[row_idx][col_idx].axis("off")