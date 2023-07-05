import torch


# TODO: maybe create a class for each custom eval metrics and set a __str__ if we print it
# we can subclass pytorch lighnting's TensorMetric to implement metrics
# This class handles automated DDP syncing and converts all inputs and outputs to tensors.


# TODO: first create FP, FN, etc then use them to calc diff metrics: accuracy, recall, precision, etc.


def check_accuracy(
    loader,
    model,
    num_classes: int = 1,
    from_logits: bool = True,
    threshdevice: str = "cuda",
):
    # TODO: This func is not very good as it's reduntant and calces the preds again.
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    # TODO: why this line?
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            # load our x:data, y:targets to device's MEM (e.g., GPU VRAM)
            x = x.to(device)
            y = y.to(device)

            # calc predictions
            # if used act_fn on the last layer no need for this line
            if from_logits:
                preds = torch.sigmoid(model(x))
            else:
                preds = model(x)

            # make pixel values for predictions binary
            # a pixel is either part of a class or not
            # use .float() to have 0.0/1.0 as our data are of type float not int
            preds = (preds > 0.5).float()
            # calc #pixels in this img_batch that were classified correctly
            num_correct += (preds == y).sum()
            # calc the total #pixels in this img_batch
            num_pixels += torch.numel(preds)

            # calc dice score
            dice_score += (2 * (preds * y).sum()) / (preds + y).sum() + 1e-8

    accu = num_correct / num_pixels
    dice = dice_score / len(loader)

    # TODO: why this line?
    model.train()

    return  accu, dice

def accuracy(targets, predictions, from_logits: bool = True):
    pass
