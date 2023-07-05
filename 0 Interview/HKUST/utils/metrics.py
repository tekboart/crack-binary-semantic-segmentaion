import torch


# TODO: maybe create a class for each custom eval metrics and set a __str__ if we print it
# we can subclass pytorch lighnting's TensorMetric to implement metrics
# This class handles automated DDP syncing and converts all inputs and outputs to tensors.


# TODO: first create FP, FN, etc then use them to calc diff metrics: accuracy, recall, precision, etc.


def dice_segment(pred, target):
    # TODO: the values are not in range [0, 1]
    # I think we must apply act_func(pred)
    smooth = 1.0
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2.0 * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


# TODO: create accu eval metric BOn dice_coeff (above)
def accuracy_segment(preds, targets, from_logits: bool = True, thresh: float = 0.5):
    if from_logits:
        preds = torch.sigmoid(preds)

    # make pixel values for predictions binary
    # a pixel is either part of a class or not
    # use .float() to have 0.0/1.0 as our data are of type float not int
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
