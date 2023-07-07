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
    sum = (preds + targets).sum()
    # calc 2 * |A ∩ B| / |A| + |B|
    dice = (2.0 * intersection + smooth) / (sum + smooth)

    return 1 - dice


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

    intersection = (preds * targets).sum().float()
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

    # calc |A ∩ B|
    intersection = (preds * targets).sum().float()
    # calc |A| + |B|
    total = (preds + targets).sum()
    # calc |A ∪ B| = |A| + |B| - |A ∩ B|
    union = total - intersection

    jaccard = (intersection + smooth) / (union + smooth)

    return 1 - jaccard


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
        Define whether the preds are logits or an activation_fn (e.g., sigmoid) has been applied to it.
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


# changed the name to 'validation_fn' and moved to utils/training.py
# def check_accuracy(
#     loader,
#     model,
#     num_classes: int = 1,
#     from_logits: bool = True,
#     thresh: float = 0.5,
#     device: str = "cuda",
# ):
#     '''
#     Does the forward pass + eval_metrics (aka inference for val/test set)
#     '''
#     # TODO: make it modular by taking the metrics and metrics_fn args
#     num_correct = 0
#     num_pixels = 0
#     dice_score = 0

#     # TODO: why this line?
#     # probab to set training=False, so do only forward
#     model.eval()

#     with torch.no_grad():
#         for x, y in loader:
#             # load our x:data, y:targets to device's MEM (e.g., GPU VRAM)
#             x = x.to(device)
#             y = y.to(device)

#             # calc predictions
#             # if used act_fn on the last layer no need for this line
#             if from_logits:
#                 preds = torch.sigmoid(model(x))
#             else:
#                 preds = model(x)

#             # make pixel values for predictions binary
#             # a pixel is either part of a class or not
#             # use .float() to have 0.0/1.0 as our data are of type float not int
#             preds = (preds > thresh).float()
#             # calc #pixels in this img_batch that were classified correctly
#             num_correct += (preds == y).sum()
#             # calc the total #pixels in this img_batch
#             num_pixels += torch.numel(preds)

#             # calc dice score
#             # method 1: (didn't work)
#             # dice_score += (2 * (preds * y).sum()) / (preds + y).sum() + 1e-8
#             # method 2
#             dice_score += dice_segment(preds, y)

#     accu = num_correct / num_pixels
#     dice = dice_score / len(loader)

#     # TODO: why this line?
#     # probab to set training=True (for future training)
#     model.train()

#     return accu, dice
