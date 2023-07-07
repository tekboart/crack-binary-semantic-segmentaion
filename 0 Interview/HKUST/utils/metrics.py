import torch


# TODO: maybe create a class for each custom eval metrics and set a __str__ if we print it
# we can subclass pytorch lighnting's TensorMetric to implement metrics
# as this subclass handles automated DDP syncing and converts all inputs and outputs to tensors.


def dice_binary_segment(
    preds: torch.Tensor, targets: torch.Tensor, from_logits: bool, smooth: float = 1.0
) -> int:
    """
    Calculate the dice metric (aka Sørensen–Dice coefficient) for binary segmentation

    Parameters
    ----------
    preds: torch.Tensor
    targets: torch.Tensor
    from_logits: bool
        Are predictions logits (output without activation_fn)?
    smooth: float, optional

    Returns
    -------
    int
        The binary segmentation dice for a batch of images.
    """
    if from_logits:
        preds = torch.sigmoid(preds)

    # flatten label and prediction tensors
    preds = preds.view(-1).float()
    targets = targets.view(-1).float()

    # calc |A ∩ B|
    intersection = (preds * targets).sum().float()
    # calc |A| + |B|
    summation = (preds + targets).sum()
    # calc 2 * |A ∩ B| / |A| + |B|
    dice = (2.0 * intersection + smooth) / (summation + smooth)

    return dice


def jaccard_binary_segment(
    preds: torch.Tensor, targets: torch.Tensor, from_logits: bool, smooth: float = 1.0
) -> int:
    """
    Calculate the Jaccard metric (aka IOU) for binary segmentation

    Parameters
    ----------
    preds: torch.Tensor
    targets: torch.Tensor
    from_logits: bool
        Are predictions logits (output without activation_fn)?
    smooth: float, optional

    Returns
    -------
    int
        The binary segmentation jaccard for a batch of images.
    """
    if from_logits:
        preds = torch.sigmoid(preds)

    # flatten label and prediction tensors
    preds = preds.view(-1).float()
    targets = targets.view(-1).float()

    # calc |A ∩ B|
    intersection = (preds * targets).sum().float()
    # calc |A| + |B|
    total = (preds + targets).sum()
    # calc |A ∪ B| = |A| + |B| - |A ∩ B|
    union = total - intersection

    jaccard = (intersection + smooth) / (union + smooth)

    return jaccard


def accuracy_binary_segment(
    preds, targets, from_logits: bool, thresh: float = 0.5
) -> int:
    """
    Calculate the accuracy for binary segmentation

    Parameters
    ----------
    preds: torch.Tensor
    targets: torch.Tensor
    from_logits: bool
        Are predictions logits (output without activation_fn)?
    thresh: float, optional

    Returns
    -------
    int
        The binary segmentation accuracy for a batch of images.
    """
    if from_logits:
        preds = torch.sigmoid(preds)

    # make each pixel values binary (0 or 1)
    preds = (preds > thresh).float()
    # calc #pixels in this img_batch that were classified correctly
    num_correct_pixels = (preds == targets).sum()
    # calc the total #pixels in this img_batch
    num_pixels = torch.numel(preds)

    accu = num_correct_pixels / num_pixels

    return accu
