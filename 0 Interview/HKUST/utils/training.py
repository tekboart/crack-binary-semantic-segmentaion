import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from typing import Union

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
    from_logits: bool,
    metrics: list,
    metrics_fn: dict,
    epoch: int,
    device: str = "cuda:0",
):
    """
    does one epoch of training
        includes (1) forwards pass, (2) calc loss, (3) metrics (if any), (4) backprop
    """
    loop = tqdm(loader)
    loop.set_description(f"Epoch {epoch+1}")
    # tqdm() returns an iterator so never access its content to avoid exhaustion
    # that's why we wrapped it in iter()
    # count the #iteration/steps in each epoch
    num_batches = sum(1 for _ in iter(loop))
    epoch_loss_cum = 0

    for batch_idx, (x, y) in enumerate(loop):
        # load our x:data, y:targets to device's MEM (e.g., GPU VRAM)
        x = x.float().to(device)
        y = y.float().to(device)

        # forward
        # we use float16 to reduce VRAM and MEM usage
        with torch.cuda.amp.autocast():
            yhat = model(x)
            assert yhat.shape == y.shape

            # calc the loss
            loss = loss_fn(yhat, y)

        # calc train metrics for this mini_batch
        for key in metrics:
            if eval_fn := metrics_fn.get(key):
                metrics[key] += eval_fn(yhat, y, from_logits).item()

        # backprop
        # init all grads az zero/0
        optimizer.zero_grad()
        # calc grads + use scaler to prevent underflow
        scaler.scale(loss).backward()
        # update weights + use scaler to prevent underflow
        scaler.step(optimizer)
        scaler.update()

        # save the loss (for this iteration/step/mini_batch)
        batch_loss = loss.item()
        epoch_loss_cum += batch_loss

        # update the tqdm loop
        loop.set_postfix(loss=batch_loss)

    # add the loss (of this epoch) to metrics
    metrics["loss"] = epoch_loss_cum

    for key in metrics:
        # divide all metrics by the #steps/batches
        metrics[key] /= num_batches

    return metrics


# TODO: This should work standalone (for evaluation of test data)
def validation_fn(
    loader,
    model,
    loss_fn,
    from_logits: bool,
    metrics: dict,
    metrics_fn: dict,
    device: str = "cuda",
):
    """
    does one validation step (used at the end of each epoch)
        simply, does (1) forward pass + (2) eval_metrics (for the val set)
    """
    num_val_batches = sum(1 for _, _ in loader)
    epoch_val_loss_cum = 0

    # TODO: why this line?
    # probab to set training=False, so do only forward
    model.eval()

    # TODO: do validation_fn in mini_batch style for more vectorization (it seems to predict one example at a time)
    # Don't cache values for backprop (ME)
    with torch.no_grad():
        for x, y in loader:
            # load our x:data, y:targets to device's MEM (e.g., GPU VRAM)
            x = x.float().to(device)
            y = y.float().to(device)

            # forward
            # we use float16 to reduce VRAM and MEM usage
            with torch.cuda.amp.autocast():
                yhat = model(x)
                assert yhat.shape == y.shape

                # calc the loss
                loss = loss_fn(yhat, y)

                # save the loss (for this iteration/step/mini_batch)
                batch_val_loss = loss.item()
                epoch_val_loss_cum += batch_val_loss

            # calc val metrics for this mini_batch
            for key in metrics:
                # used split() to remove 'val_' part of key
                if eval_fn := metrics_fn.get(key.split("_")[1]):
                    metrics[key] += eval_fn(yhat, y, from_logits).item()

    # add the val_loss (of this epoch) to metrics
    metrics["val_loss"] = epoch_val_loss_cum

    for key in metrics:
        # divide all metrics by the #steps/batches
        metrics[key] /= num_val_batches

    # TODO: why this line?
    # probab to set training=True (for future training)
    model.train()

    return metrics


# TODO: create a class with fit(), evaluate(), predict() methods
class BinarySegmentationModel(nn.Module):
    def __init__(
        self,
        model,
        train_loader,
        from_logits: bool,
        epochs: int = 5,
        val_loader=None,
        lr_rate: float = 0.001,
        device: str = "cuda:0",
        save_model: bool = False,
        save_checkpoint_path: str = None,
        save_checkpoint_name: str = None,
        load_model: bool = False,
        load_checkpoint_path: str = None,
        metrics: Union[tuple[str], list[str]] = (),
        metrics_fn: dict = {},
    ):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.from_logits = from_logits
        self.epochs = epochs
        self.val_loader = val_loader
        self.lr = lr_rate
        self.device = device
        self.save_model = save_model
        self.save_checkpoint_path = save_checkpoint_path
        self.save_checkpoint_name = save_checkpoint_name
        self.load_model = load_model
        self.load_checkpoint_path = load_checkpoint_path
        self.metrics = metrics
        self.metrics_fn = metrics_fn  # TODO: can make it universal (no need to input it to the class.__init__)


def train_model(
    model,
    train_loader,
    from_logits: bool,
    epochs: int = 5,
    val_loader=None,
    lr: float = 0.001,
    device: str = "cuda:0",
    save_model: bool = False,
    save_checkpoint_path: str = None,
    save_checkpoint_name: str = None,
    load_model: bool = False,
    load_checkpoint_path: str = None,
    metrics: Union[tuple[str], list[str]] = (),
    metrics_fn: dict = {},
):
    """
    Do the training for several epoch (written in pure PyTorch)
    """
    # TODO: add the needed hyperparameters as args to this func (is more versatile)
    model = model.to(device)

    # set the loss function
    if from_logits:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.BCELoss()

    # set the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # set learning schedualer
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # load weights a pretrained model
    if load_model:
        load_checkpoint(torch.load(load_checkpoint_path), model)

    # To prevent underflow in grads by scaling the loss
    # when precision lvl (e.g., Float16) cannot represent very small numbers
    scaler = torch.cuda.amp.GradScaler()

    # init the train metrcis (e.g., val_loss, val_dice, etc.)
    # we add 'loss' separately as it's not part of metrics
    history = {"loss": []}
    # add other metrics (if any)
    for key in metrics:
        history[key] = []

    # init the val metrcis (e.g., val_loss, val_dice, etc.)
    if val_loader:
        val_keys = []
        for key in history:
            val_keys.append(f"val_{key}")
        for key in val_keys:
            history[key] = []

    # loop over #epochs
    for epoch in range(epochs):
        print(f" epoch {epoch+1}/{epochs} ".center(79, "-"))
        # create/reset metrics values to 0 (for each epoch)
        train_metrics_init = dict.fromkeys(metrics, 0)

        # Start training iterations (for this epoch)
        # print(" Training Phase (In Progress) ".center(79, "-"))
        train_metrics = train_fn(
            train_loader,
            model,
            optimizer,
            loss_fn,
            scaler,
            from_logits,
            train_metrics_init,
            metrics_fn,
            epoch=epoch,
            device=device,
        )

        # plot the validation metrics + save to history (dict)
        # print(f" epoch {epoch}'s metric(s) (training) ".center(79, "."))
        print()
        # used reversed() to make 'loss' the first item
        for key, value in reversed(train_metrics.items()):
            print(f"{key+':':<20} {value:>5.2f}")
            history[key].append(value)
        # print(" Training Phase (Done) ".center(79, "-"))

        if val_loader:
            # print(" Validation Phase (In Progress) ".center(79, "-"))

            # add 'val_' to each metric
            # create/reset metrics values to 0 (for each epoch)
            val_metrics_init = dict.fromkeys([f"val_{metric}" for metric in metrics], 0)

            val_metrics = validation_fn(
                val_loader,
                model,
                loss_fn,
                from_logits,
                val_metrics_init,
                metrics_fn,
                device=device,
            )

            # plot the validation metrics
            # print(f" epoch {epoch}'s metric(s) (validation) ".center(79, "."))
            for key, value in reversed(val_metrics.items()):
                print(f"{key+':':<20} {value:>5.2f}")
                history[key].append(value)
            # print(f'{"val_accuracy:":<15} {val_accu.item():>5.2f}')
            # print(f'{"val_dice:":<15} {val_dice.item():>5.2f}')
            # history["val_accuracy"].append(val_accu.item())
            # history["val_dice"].append(val_dice.item())
            # print(" Validation Phase (Done) ".center(79, "-"))

            # print some examples to a folder

        # decay lr_rate (must be at the end of each epoch)
        scheduler.step()

    # save the trained model
    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        save_checkpoint(
            checkpoint, dirname=save_checkpoint_path, filename=save_checkpoint_name
        )

    return history


# TODO: write a main fn for pytorch-lightning


###############################################################################
# For testing
###############################################################################
if __name__ == "__main__":
    pass
