import torch
from torchmetrics.metric import Metric

# make this class a subset of torchmetrics.metric.Metric
# needs update() and compute() overrriding (its an ABC class)
import torch.nn as nn


class BasicMetricsBinarySegment(nn.Module):
    """
    Calculate the binary metrics () for binary segmentation

    Atributes
    ---------
    from_logits: bool
        Are predictions logits (output without activation_fn)?
    """

    def __init__(self, from_logits: bool = True, thresh: float = 0.5):
        super().__init__()
        self.from_logits = from_logits
        self.thresh = thresh

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        preds: torch.Tensor
        targets: torch.Tensor

        Returns
        -------
        tuple
            A tuple of (FP, FN, TP, TN).
        """
        if self.from_logits:
            preds = torch.sigmoid(preds)

        # make each pixel values binary (0 or 1)
        preds = (preds >= self.thresh).float()

        assert preds.shape == targets.shape
        # flatten label and prediction tensors
        preds = preds.view(-1).float()
        targets = targets.view(-1).float()

        # calc the False Positive
        fp = torch.sum((preds == 1) & (targets == 0))
        # calc the False Negative
        fn = torch.sum((preds == 0) & (targets == 1))
        # calc the True Positve
        tp = torch.sum((preds == 1) & (targets == 1))
        # calc the True Negative
        tn = torch.sum((preds == 0) & (targets == 0))

        return fp, fn, tp, tn


class PrecisionBinarySegment(nn.Module):
    """
    Calculate the Precision metric for binary segmentation.

                       TP
    Precision = ---------------
                    TP + FP

    Atributes
    ---------
    from_logits: bool
        Are predictions logits (output without activation_fn)?
    smooth: float, optional
    """

    def __init__(
        self, from_logits: bool = True, thresh: float = 0.5, smooth: float = 1e-5
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

        Returns
        -------
        int
            The binary segmentation dice for a batch of images.
        """
        fp, fn, tp, tn = BasicMetricsBinarySegment(
            self.from_logits, thresh=self.thresh
        )(preds, targets)

        precision = tp / (tp + fp)

        return precision


class RecallBinarySegment(nn.Module):
    """
    Calculate the Recall metric for binary segmentation.

                   TP
    Recall = ---------------
                 TP + FN

    Atributes
    ---------
    from_logits: bool
        Are predictions logits (output without activation_fn)?
    smooth: float, optional
    """

    def __init__(
        self, from_logits: bool = True, thresh: float = 0.5, smooth: float = 1e-5
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

        Returns
        -------
        int
            The binary segmentation dice for a batch of images.
        """
        fp, fn, tp, tn = BasicMetricsBinarySegment(
            self.from_logits, thresh=self.thresh
        )(preds, targets)

        recall = tp / (tp + fn)

        return recall


class F1BinarySegment(nn.Module):
    """
    Calculate the F1-Score (aka F-Measure) metric for binary segmentation.

    This is the same as DiceBinarySegment, so is redundant.

                 2 * Precision * Recall
    F1_Score = ---------------------------
                   Precision + Recall

                       2TP
    F1_Score = -----------------
                 2TP + FN + FP

                       TP
    F1_Score = -----------------
                 TP + 1/2(FN + FP)

    Atributes
    ---------
    from_logits: bool
        Are predictions logits (output without activation_fn)?
    smooth: float, optional
    """

    def __init__(
        self, from_logits: bool = True, thresh: float = 0.5, smooth: float = 1e-5
    ):
        super().__init__()
        self.from_logits = from_logits
        self.smooth = (
            smooth  # TODO: if not needed delete it (do it for precision and Recall too)
        )
        self.thresh = thresh

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Parameters
        ----------
        preds: torch.Tensor
        targets: torch.Tensor

        Returns
        -------
        int
            The binary segmentation dice for a batch of images.
        """
        fp, fn, tp, tn = BasicMetricsBinarySegment(
            self.from_logits, thresh=self.thresh
        )(preds, targets)

        # precision = PrecisionBinarySegment(self.from_logits, self.thresh)(
        #     preds, targets
        # )

        # formula 1: 100% works
        # f1_score = tp / (tp + (0.5 * (fp + fn)))

        # formula 2: 100% works
        f1_score = (2*tp) / (2*tp + fp + fn)

        return f1_score


class AccuracyBinarySegment(nn.Module):
    """
    Calculate the Accuracy for binary segmentation

    Accuracy = (TP + TN) / (TP + FN + TN + FP)

    Atributes
    ---------
    from_logits: bool
        Are predictions logits (output without activation_fn)?
    smooth: float, optional
    """

    def __init__(
        self, from_logits: bool = True, thresh: float = 0.5, smooth: float = 1e-5
    ):
        super().__init__()
        self.from_logits = from_logits
        self.smooth = (
            smooth  # TODO: if not needed delete it (do it for precision and Recall too)
        )
        self.thresh = thresh

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Parameters
        ----------
        preds: torch.Tensor
        targets: torch.Tensor

        Returns
        -------
        int
            The binary segmentation dice for a batch of images.
        """
        fp, fn, tp, tn = BasicMetricsBinarySegment(
            self.from_logits, thresh=self.thresh
        )(preds, targets)

        # precision = PrecisionBinarySegment(self.from_logits, self.thresh)(
        #     preds, targets
        # )

        accuracy = (tp + tn) / (tp + fn + tn + fp)

        return accuracy


class AccuracyBinarySegmentDepricated:
    """
    (Depricated): there is a better/more efficient implementation above

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
    ) -> None:
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.thresh = thresh

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> int:
        """
        Parameters
        ----------
        preds: torch.Tensor
        targets: torch.Tensor

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


class JaccardBinarySegment(nn.Module):
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
    print(
        f"{'jaccard (me):':<20} {JaccardBinarySegment(from_logits=False)(preds, targets)}"
    )
    # outputs: 0.33293622732162476
    print(f"{'jaccard (torch):':<20} {jaccard_torch(preds, targets)}")
    # outputs: 0.33293622732162476

    # Check wether they output 1.0, when predictions are 100% accurate (complete overlap)
    assert all(
        [
            1.0,
            DiceBinarySegment(from_logits=False)(
                preds=targets.float(), targets=targets
            ),
            dice_torch(targets.float(), targets),
            JaccardBinarySegment(from_logits=False)(
                preds=targets.float(), targets=targets
            ),
            jaccard_torch(targets.float(), targets),
        ]
    )
