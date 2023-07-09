import torch
from torchmetrics.metric import Metric

# make this class a subset of torchmetrics.metric.Metric
# needs update() and compute() overrriding (its an ABC class)
import torch.nn as nn
class DiceBinarySegment(nn.Module):
    """
    Calculate the dice metric (aka Sørensen–Dice coefficient) for binary segmentation

    Atributes
    ---------
    from_logits: bool
        Are predictions logits (output without activation_fn)?
    smooth: float, optional
    """

    def __init__(
        self, from_logits: bool = True, smooth: float = 1e-5, thresh: float = 0.5
    ):
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.thresh = thresh

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor):
        """
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
        if self.from_logits:
            preds = torch.sigmoid(preds)

        # make each pixel values binary (0 or 1)
        preds = (preds >= self.thresh).float()

        assert preds.shape == targets.shape
        # flatten label and prediction tensors
        preds = preds.view(-1).float()
        targets = targets.view(-1).float()

        # calc |A ∩ B|
        intersection = (preds * targets).sum().float()
        # calc |A| + |B|
        # summation = (preds + targets).sum()
        summation = (preds.square() + targets.square()).sum().float()
        # calc 2 * |A ∩ B| / |A| + |B|
        dice = (2.0 * intersection + self.smooth) / (summation + self.smooth)

        return dice


class JaccardBinarySegment:
    """
    Calculate the Jaccard metric (aka IOU) for binary segmentation

    Atributes
    ---------
    from_logits: bool
        Are predictions logits (output without activation_fn)?
    smooth: float, optional
    """

    def __init__(
        self, from_logits: bool = True, smooth: float = 1e-5, thresh: float = 0.5
    ):
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.thresh = thresh

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor):
        """
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
        if self.from_logits:
            preds = torch.sigmoid(preds)

        # flatten label and prediction tensors
        preds = preds.view(-1).float()
        targets = targets.view(-1).float()

        # make each pixel values binary (0 or 1)
        preds = (preds >= self.thresh).float()

        # calc |A ∩ B|
        intersection = (preds * targets).sum().float()
        # calc |A| + |B|
        total = (preds + targets).sum()
        # calc |A ∪ B| = |A| + |B| - |A ∩ B|
        union = total - intersection

        jaccard = (intersection + self.smooth) / (union + self.smooth)

        return jaccard


class AccuracyBinarySegment:
    """
    Calculate the Accuracy for binary segmentation

    Atributes
    ---------
    from_logits: bool
        Are predictions logits (output without activation_fn)?
    smooth: float, optional
    """

    def __init__(
        self, from_logits: bool = True, smooth: float = 1e-5, thresh: float = 0.5
    ):
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.thresh = thresh

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor):
        """
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
        if self.from_logits:
            preds = torch.sigmoid(preds)

        # make each pixel values binary (0 or 1)
        preds = (preds >= self.thresh).float()

        # calc #pixels in this img_batch that were classified correctly
        num_correct_pixels = (preds == targets).sum()
        # calc the total #pixels in this img_batch
        num_pixels = torch.numel(preds)

        accu = num_correct_pixels / num_pixels

        return accu


###############################################################################
# For testing
###############################################################################
if __name__ == "__main__":
    from torchmetrics.classification import Dice, BinaryJaccardIndex

    # Compare our custom metircs with torchmetrics counterparts
    dice_torch = Dice(num_classes=1, average="micro")
    jaccard_torch = BinaryJaccardIndex()

    # generate pred and target masks of shape (N=32, C=5, H=64, W=64)
    preds = torch.randint(
        low=0, high=2, size=(32, 5, 64, 64), dtype=torch.float16
    ).view(-1)
    targets = torch.randint(
        low=0, high=2, size=(32, 5, 64, 64), dtype=torch.int16
    ).view(-1)

    print(f"{'dice (me):':<20} {DiceBinarySegment(from_logits=False)(preds, targets)}")
    # outputs: 4995531141757965
    print(f"{'dice (torch):':<20} {dice_torch(preds, targets)}")
    # outputs: 0.4995531141757965
    print(f"{'jaccard (me):':<20} {JaccardBinarySegment(from_logits=False)(preds, targets)}")
    # outputs: 0.33293622732162476
    print(f"{'jaccard (torch):':<20} {jaccard_torch(preds, targets)}")
    # outputs: 0.33293622732162476

    # Check wether they output 1.0, when predictions are 100% accurate (complete overlap)
    assert all(
        [
            1.0,
            DiceBinarySegment(from_logits=False)(preds=targets.float(), targets=targets),
            dice_torch(targets.float(), targets),
            JaccardBinarySegment(from_logits=False)(preds=targets.float(), targets=targets),
            jaccard_torch(targets.float(), targets),
        ]
    )
