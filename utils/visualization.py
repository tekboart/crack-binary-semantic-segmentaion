import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil
from random import randint

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


# TODO: I've written a better fn --> delete it
def plot_segmentation_inference(
    img_batch, mask_batch, yhat_mask_batch, limit: int = None
):
    """
    Take three Tensors with equal batch_size and plot them side by side (each record in on row)

    Parameters
    ----------

    truncate: int
        Set a value to plot only a limited number of examples. (must be >= 1)
    """

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
        axes[0].set_title(f"Image {[*img.shape]}", fontsize=16)
        axes[0].axis("off")
        axes[1].imshow(yhat_mask, cmap="gray")
        axes[1].set_title(f"Mask (predicted) {[*yhat_mask.shape]}", fontsize=16)
        axes[1].axis("off")
        axes[2].imshow(mask, cmap="gray")
        axes[2].set_title(f"Mask (target) {[*mask.shape]}", fontsize=16)
        axes[2].axis("off")
    elif batch_size > 1:
        for row in range(batch_size):
            img = img_batch[row]
            yhat_mask = yhat_mask_batch[row]
            mask = mask_batch[row]
            axes[row][0].imshow(img)
            axes[row][0].set_title(f"Image {[*img.shape]}", fontsize=16)
            axes[row][0].axis("off")
            axes[row][1].imshow(yhat_mask, cmap="gray")
            axes[row][1].set_title(
                f"Mask (predicted) {[*yhat_mask.shape]}", fontsize=16
            )
            axes[row][1].axis("off")
            axes[row][2].imshow(mask, cmap="gray")
            axes[row][2].set_title(f"Mask (target) {[*mask.shape]}", fontsize=16)
            axes[row][2].axis("off")

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
        """
        Parameters
        ----------
        img: np.ndarray
        """
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
    # TODO: Should I add support for tf.Tensor
    # I have to import the entire tensorflow to just use tf.Tensor type???
    # get so much MEM with no use
    # TODO: if not remove tf.Tensor from ValueError (below) as well
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
    anti_standardize_fn=None,
    shuffle: bool = True,
):
    """
    plot image, mask pairs of a given batch of data.

    Parameters
    ----------
    batch_list: list
        a list of [torch.Tensors] with shape (N, H, W, C) or (N, C, W, C).
        The order is important so either [img_batch, mask_batch] or [img_batch, mask_batch, yhat_batch]
    data_format: str | 'channels_first' or 'channels_last'
    num_rows: int
    titles: list
    plot_axes: bool
        Should the axes be plotted (to show image dimensions)
    anti_standardize_fn: bool
        if the input image (the first elem of batch_list) was standardized/normalized (e.g., by z-score), then give us the anti_fn to reverse it.
        The resulting image must be in range of either [0, 1] (float) or [0, 255] (int)
    shuffle: bool
        Should take image by order or by random.
    """
    batch_list = list(batch_list)  # make sure it's a list
    num_cols = len(batch_list)
    num_recs = batch_list[0].shape[0]

    # convert batch_tensors to ndarray + make it channels_last
    for i in range(num_cols):
        batch_list[i] = image_batch_to_ndarray_channels_first(
            batch_list[i], data_format
        )

    # used, to avoid plotting repeated images
    used_indices = []

    # plot img, mask (side-by-side)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    # make the axes a 2darray (in case num_rows=1, hence axes is 1darray)
    if axes.ndim == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx in range(num_rows):

        if shuffle:
            # get a random index of images
            img_idx = randint(0, num_recs - 1)
            # generate a new img_indx until it's unique
            while img_idx in used_indices:
                img_idx = randint(0, num_recs - 1)
            # add the unique img_idx to used_indices
            used_indices.append(img_idx)
        else:
            img_idx = row_idx

        for col_idx in range(num_cols):
            img = batch_list[col_idx][img_idx]

            # different setting for target_masks
            if col_idx in [1, 2] and img.shape[-1] == 1:
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


def plot_metrics(
    history: dict,
    metrics: list[str] = ["loss"],
    y_scale: str = "linear",
    x_interval: int = 5,
    crop_yaxis: bool = True,
) -> None:
    """
    Plot the metrics (for both the train and val)

    Parameters
    ----------
    history: dict
        the output of model.fit()
    metrics: list[str]
        a list of metrics to be plotted.
    y_scale: str
        Define the scale of the y-axis. Possible values are 'linear', 'log', 'symlog', 'asinh', and 'logit'.
    x_interval: int
        Define the intervals between x-axis ticks.
    crop_yaxis: bool
        Whether to zoom-in the y-axis to show only the relevant areas.

    returns
    -------
    None
        a multi plot of the performance of the model + a green line to differentiate before and after finetuning

    Examples
    --------
    >>> plot_metrics(history, metrics_list, y_scale="linear", x_interval=5, crop_yaxis=True)
    >>> plt.suptitle('both preliminary & finetune phases')
    >>> plt.tight_layout()
    >>> plt.show()
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # calculate the #epochs
    num_epochs = len(list(history.values())[0])

    plt.figure(figsize=(12, len(metrics) * 2))
    # take the ceiling of #metrics provided as the #rows
    plot_height = int(len(metrics) / 2) + 1

    for i, metric in enumerate(metrics):
        name = metric.replace("_", " ").title()
        plt.subplot(plot_height, 2, i + 1)
        plt.plot(history[metric], color=colors[0], label="Train")
        plt.plot(
            history["val_" + metric],
            color=colors[1],
            linestyle="--",
            label="Val",
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)

        # Set the scale
        plt.yscale(y_scale)

        # Show only the relevant range
        if crop_yaxis:
            # take the min value for each metric (considering both train an val)
            min_metric = min(history[metric])
            min_metric_val = min(history[f"val_{metric}"])
            min_yaxis = min(min_metric, min_metric_val)
            # take the max value for each metric (considering both train an val)
            max_metric = max(history[metric])
            max_metric_val = max(history[f"val_{metric}"])
            max_yaxis = max(max_metric, max_metric_val)
            # set the limit of y-axis
            border = 0.05
            plt.ylim([min_yaxis - border, max_yaxis + border])
        else:
            if metric == "loss":
                plt.ylim([0, plt.ylim()[1]])
            else:
                plt.ylim([0, 1])

        # set the xaxis ticks
        plt.xticks(range(0, num_epochs, x_interval))

        plt.legend()
        plt.tight_layout()


def plot_metrics_finetune(
    history,
    history_finetune,
    metrics: list[str] = ["loss"],
    y_scale: str = "linear",
    crop_yaxis: bool = True,
    x_interval: int = 5,
):
    """
    #TODO: Complete docstring

    Parameters
    ----------
    history: dict
        the output of original model.fit()
    history_finetuen: dict
        the history of model after finetune (un-freezing layers)
    metrics: list[str]
        a list of metrics to be plotted.
    y_scale: str
        Define the scale of the y-axis. Possible values are 'linear', 'log', 'symlog', 'asinh', and 'logit'.
    x_interval: int
        Define the intervals between x-axis ticks.
    crop_yaxis: bool
        Whether to zoom-in the y-axis to show only the relevant areas.

    returns
    -------
    None
        a multi plot of the performance of the model + a green line to differentiate before and after finetuning

    Examples
    --------
    >>> plot_metrics_finetune(MobileNet_tl_history,
                      MobileNet_tl_finetune_history,
                      initial_epochs_real,
                      metrics=metrics_names)
    >>> plt.suptitle('both preliminary & finetune phases')
    >>> plt.tight_layout()
    >>> plt.show()
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # calculate the #epochs
    num_epochs_train = len(list(history.values())[0])
    num_epochs_finetune = len(list(history_finetune.values())[0])
    num_epochs = num_epochs_train + num_epochs_finetune

    # the #epochs for TL phase 1 (useful if use earlystoppage)
    num_epochs_train = len(history["loss"])
    num_epochs_finetune = len(history_finetune["loss"])

    plt.figure(figsize=(12, len(metrics) * 2))
    # take the ceiling of #metrics provided as the #rows
    # cal the #rows
    # num_rows = int(len(metrics) / 2) + 1
    num_rows = np.ceil(len(metrics) / 2).astype("int") + 1

    for i, metric in enumerate(metrics):
        # name = metric.replace("_", " ").capitalize()
        name = metric.replace("_", " ").title()
        plt.subplot(num_rows, 2, i + 1)
        history_train_total = history[metric] + history_finetune[metric]
        history_val_total = history["val_" + metric] + history_finetune["val_" + metric]
        # plot the train metrics
        plt.plot(history_train_total, color=colors[0], label="Train")
        # plot the val metrics
        # plt.plot(range(1, epoch+1), metric_val, color=colors[1], linestyle="-", label="Val")
        plt.plot(
            history_val_total,
            color=colors[1],
            linestyle="--",
            label="Val",
        )
        # plot the vertical line (to denote fine-tune section)
        plt.plot(
            [num_epochs_train, num_epochs_train],
            plt.ylim([0, 1]),
            color=colors[2],
            linestyle=":",
            label="Start Fine Tuning",
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.yscale(y_scale)

        if crop_yaxis:
            # take the min value for each metric (considering both train an val)
            min_metric = min(history_train_total)
            min_metric_val = min(history_val_total)
            min_yaxis = min(min_metric, min_metric_val)
            # take the max value for each metric (considering both train an val)
            max_metric = max(history_train_total)
            max_metric_val = max(history_val_total)
            max_yaxis = max(max_metric, max_metric_val)
            # set the limit of y-axis
            border = 0.05
            plt.ylim([min_yaxis - border, max_yaxis + border])
        else:
            if metric == "loss":
                plt.ylim([0, plt.ylim()[1]])
            else:
                plt.ylim([0, 1])

        # set the xaxis ticks
        plt.xticks(range(0, num_epochs, x_interval))

        plt.legend()
        plt.tight_layout()
