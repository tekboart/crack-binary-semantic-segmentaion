import torch
import torch.nn as nn
import torch.optim as optim


import os
from tqdm import tqdm

# load my custom Classes/Functions/etc.
# from utils.metrics import check_accuracy, dice_coeff


def save_checkpoint(state, dirname: str = "", filename: str = None) -> None:
    """
    Save the trained model (aka checkpoint)
    """
    from datetime import datetime

    print(" Saving Checkpoint (In progress) ".center(79, "-"))

    if not filename:
        # get the date+time (of currect TimeZone)
        time = datetime.today().strftime("%Y.%m.%d@%H-%M-%S")
        # get the date+time (of UTC TimeZone)
        # time = datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S')

        filename = f"{dirname}{time}-model_checkpoint.pth.tar"

    torch.save(state, filename)
    print(f"\nCheckpoint was saved as: {filename}\n")

    print(" Saving Checkpoint (Done) ".center(79, "-"))


def load_checkpoint(checkpoint, model) -> None:
    """
    Load the weights from a the trained model to another model.

    Parameters
    ----------
    checkpoint
        A previously saved model checkpoint (e.g., using torch.save())
        e.g., my_checkpoint.pth.tar
    """
    print(" Loading Checkpoint (In progress) ".center(79, "-"))

    model.load_state_dict(checkpoint["state_dict"])

    print(" Loading Checkpoint (Done) ".center(79, "-"))


def train_fn(
    loader,
    model,
    optimizer,
    loss_fn,
    scaler,
    metrics: dict = {},
    metrics_fn: dict = {},
    device: str = "cuda:0",
):
    """
    does one epoch of training
    """
    loop = tqdm(loader)
    # tqdm() returns an iterator so never access its content to avoid exhaustion
    # that's why we wrapped it in iter()
    # count the #iteration/steps in each epoch
    step_counter = sum(1 for _ in iter(loop))

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().to(device)

        # forward
        # we use float16 to reduce VRAM and MEM usage
        with torch.cuda.amp.autocast():
            predictions = model(data)

            # must check the shape of preds and target masks before giving them to loss_fn
            assert predictions.shape == targets.shape
            loss = loss_fn(predictions, targets)

        # calc eval metrics (for training)
        for key in metrics:
            if eval_fn := metrics_fn.get(key):
                metrics[key] += eval_fn(predictions, targets)

        # backprop
        # init all grads az zero/0
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update the tqdm loop
        loop.set_postfix(loss=loss.item())

    for key in metrics:
        metrics[key] /= step_counter

    return metrics


def validation_fn(
    loader,
    model,
    num_classes: int = 1,
    from_logits: bool = True,
    thresh: float = 0.5,
    metrics: dict = {},
    metrics_fn: dict = {},
    device: str = "cuda",
):
    """
    does one validation step (used at the end of each epoch)
        simply, does (1) forward pass + (2) eval_metrics (for the val set)
    """

    # TODO: make it modular by taking the metrics and metrics_fn args
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    # TODO: why this line?
    # probab to set training=False, so do only forward
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
            preds = (preds > thresh).float()
            # calc #pixels in this img_batch that were classified correctly
            num_correct += (preds == y).sum()
            # calc the total #pixels in this img_batch
            num_pixels += torch.numel(preds)

            # calc dice score
            # method 1: (didn't work)
            # dice_score += (2 * (preds * y).sum()) / (preds + y).sum() + 1e-8
            # method 2
            dice_score += metrics_fn['dice'](preds, y)

    accu = num_correct / num_pixels
    dice = dice_score / len(loader)

    # TODO: why this line?
    # probab to set training=True (for future training)
    model.train()

    return accu, dice


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float = 0.001,
    device: str = "cuda:0",
    load_model: bool = False,
    checkpoint_path: str = None,
    metrics: tuple = (),
    metrics_fn: dict = {},
):
    """
    Do the training for several epoch (written in pure PyTorch)
    """
    # TODO: add the needed hyperparameters as args to this func (is more versatile)
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # load weights a pretrained model
    if load_model:
        load_checkpoint(torch.load(checkpoint_path), model)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        # create dict from metrics list/tuple
        metrics = dict.fromkeys(metrics, 0)
        metrics = train_fn(
            train_loader, model, optimizer, loss_fn, scaler, metrics, metrics_fn, device
        )

        # eval metrics (for training)
        print(f" epoch {epoch}'s metric(s) (training) ".center(79, "-"))
        for key, value in metrics.items():
            print(f"{key:<10} {value.item():>5.2f}")
        # print("-" * 79)

        if val_loader:
            # eval metrics (for validation)
            # TODO: use metrics dict (but with 'val_' prefix) to automate things
            val_accu, val_dice = validation_fn(val_loader, model, metrics_fn=metrics_fn, device=device)
            print(f" epoch {epoch}'s metric(s) (validation) ".center(79, "-"))
            print(f'{"val_accuracy:":<10} {val_accu.item():>5.2f}')
            print(f'{"val_dice:":<10} {val_dice.item():>5.2f}')
            print("-" * 79)

            # print some examples to a folder

    # save the trained model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    save_checkpoint(checkpoint, dirname="./models/")


# TODO: write a main fn for pytorch-lightning


###############################################################################
# For testing
###############################################################################
if __name__ == "__main__":
    pass
