import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
from typing import Union


def save_checkpoint(
    state: dict,
    filename: str = None,
    utc_tz: bool = False,
    verbose: bool = False,
) -> None:
    """
    Save the trained model's state (aka checkpoint)

    Parameters
    ----------
    state: dict
        a dictionary of model.state, optimizer.stat, etc.
    filename: str
        the filename of the saved checkpoint.
        e.g., 2023.07.08@09-13-12-model_checkpoint.pth.tar
    utc_tz: bool
        Whether to use the UTC time (instead of the local TIMEZONE) when the filename is not provided.
    verbose: bool
        Whether print lines declaring that the model's checkpoint is being saved

    Examples
    --------
    >>> state_checkpoint = {"state_dict": model.state_dict(),"optimizer": optimizer.state_dict()}
    >>> save_checkpoint(state_checkpoint,dirname=os.path.join("models", "Oct"), filename="temp_model_checkpoint")
    """
    from datetime import datetime
    import os

    # Define the checkpoints filename
    if filename:
        filename = f"{filename}.pth.tar"
    elif not filename and not utc_tz:
        # get the date+time (of currect TimeZone)
        time = datetime.today().strftime("%Y.%m.%d@%H-%M-%S")
        filename = f"{time}@model_checkpoint.pth.tar"
    elif not filename and utc_tz:
        # get the date+time (of UTC TimeZone)
        time = datetime.utcnow().strftime("%Y-%m-%d %H-%M-%S")
        filename = f"{time}@model_checkpoint.pth.tar"

    if verbose:
        print(" Saving Checkpoint (In progress) ".center(79, "-"))

    torch.save(state, filename)
    if verbose:
        print(f"\nCheckpoint was saved as: {filename}\n")

        print(" Saving Checkpoint (Done) ".center(79, "-"))


def load_checkpoint(checkpoint: str, model, verbose: bool = False) -> None:
    """
    Load the weights from a the trained model's checkpoint to another model.

    Parameters
    ----------
    checkpoint: .pth | .pth.tar
        A previously saved model checkpoint (e.g., using torch.save())
        e.g., my_checkpoint.pth.tar
    model: torch.nn.Module
        the model into which we want to load the checkpoint.
    verbose: bool
        Whether print lines declaring that the model's checkpoint is being loaded.
    """
    if verbose:
        print(" Loading Checkpoint (In Progress) ".center(79, "-"))

    model.load_state_dict(checkpoint["state_dict"])

    if verbose:
        print(" Loading Checkpoint (Done) ".center(79, "-"))


def train_fn(
    loader,
    model,
    optimizer,
    loss_fn,
    scaler,
    metrics: dict,
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
                metrics[key] += eval_fn(yhat, y).item()

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
def evaluate_fn(
    loader,
    model,
    loss_fn,
    metrics: dict,
    metrics_fn: dict,
    device: str = "cuda",
):
    """
    does one validation step (used at the end of each epoch)
        simply, does (1) forward pass + (2) eval_metrics (for the val set)
    """
    num_batches = sum(1 for _, _ in loader)
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
                model = model.to(device)  # when call validation_fn alone (e.g., in model.evaluate()/predict())
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
                if eval_fn := metrics_fn.get(key):
                    metrics[key] += eval_fn(yhat, y).item()

    # add the val_loss (of this epoch) to metrics
    metrics["loss"] = epoch_val_loss_cum

    for key in metrics:
        # divide all metrics by the #steps/batches
        metrics[key] /= num_batches

    # TODO: why this line?
    # probab to set training=True (to continue training in next epoch)
    model.train()

    return metrics

# TODO: This should work standalone (for evaluation of test data)
def predict_fn(
    loader,
    model,
    thresh: float = 0.5,
    act_fn = None,
    device: str = "cuda",
):
    """
    #TODO: I made it a Generator to save MEM/Computation, does it work?

    does one inference step
        simply, does (1) forward pass

    Parameters
    ----------
    act_fn: Callable
        If the output layer of the model doesn't have an act_func (is from_logit),
        then provive a suitable act_func (e.g., torch.sigmoid for binary segmentation).
    """
    # training=False (for layers, dropout, batchnorm, etc.)
    model.eval()

    # Don't cache values for backprop (ME)
    with torch.no_grad():
        for x, y in loader:
            # load our x:data, y:targets to device's MEM (e.g., GPU VRAM)
            x = x.float().to(device)
            y = y.float().to(device)

            # forward
            # we use float16 to reduce VRAM and MEM usage
            with torch.cuda.amp.autocast():
                model = model.to(device)  # when call validation_fn alone (e.g., in model.evaluate()/predict())
                yhat = model(x)
                assert yhat.shape == y.shape

                # apply act func (if from_logit)
                yhat = act_fn(yhat)
                # make each pixel value of yhat binary (0 or 1)
                yhat = (yhat >= thresh).float()


            yield x, y, yhat

    # set training=True (to continue training in next epoch)
    model.train()

# TODO: create a class with fit(), evaluate(), predict() methods
class BinarySegmentationModel(nn.Module):
    """
    A Class model to do train, evaluation, & predict (like keras models).
    """

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
        # TODO: can make it universal (no need to input it to the class.__init__)
        # TODO: even better, can be the input of def compile(self, optimizer, losses=[], metrics=[]):
        self.metrics_fn = metrics_fn


def fit_fn(
    model,
    train_loader,
    optimizer,  # move these to <model Class>.compile method
    loss_fn,  # move these to <model Class>.compile method
    scheduler=None,  # move these to <model Class>.compile method
    metrics: dict = {},  # move these to <model Class>.compile method
    val_loader=None,
    epochs: int = 5,
    device: str = "cuda:0",
    save_model: bool = False,
    save_model_filename: str = None,
    save_model_temp: bool = False,
    save_model_temp_filename: str = None,
    load_model: bool = False,
    load_model_filename: str = None,
):
    """
    Do the training for several epoch (written in pure PyTorch)
    """
    # TODO: add the needed hyperparameters (e.g., lr_decay_step) as args to this func (is more versatile)
    model = model.to(device)

    # load weights a pretrained model (before further training)
    if load_model:
        load_checkpoint(torch.load(load_model_filename), model)

    # Create a list to store lr at each epoch (to plot it later)
    if scheduler:
        lr_list = [optimizer.param_groups[0]["lr"]]

    # To prevent underflow in grads by scaling the loss
    # when precision lvl (e.g., Float16) cannot represent very small numbers
    scaler = torch.cuda.amp.GradScaler()

    # TODO: add changin an setting of the metrics to <model Class>.compile()
    # init the train metrcis (e.g., val_loss, val_dice, etc.)
    # we add 'loss' separately as it's not part of metrics
    history = {"loss": []}
    # add other metrics (if any)
    for key in metrics:
        history[key] = []

    # init the val metrcis (e.g., val_loss, val_dice, etc.)
    if val_loader:
        val_keys = []
        # create metrics with 'val_' prefix (e.g., val_loss)
        for key in history:
            val_keys.append(f"val_{key}")
        # init val_<metric> in history:dict
        for key in val_keys:
            history[key] = []

    # loop over #epochs
    for epoch in range(epochs):
        print(f" epoch {epoch+1}/{epochs} ".center(79, "-"))
        # create/reset metrics values to 0 (for each epoch)
        train_metrics_init = dict.fromkeys(metrics, 0)

        # training phase
        # Start training iterations (for this epoch)
        train_metrics = train_fn(
            train_loader,
            model,
            optimizer,
            loss_fn,
            scaler,
            train_metrics_init,
            metrics,
            epoch,
            device,
        )

        # plot the validation metrics + save to history (dict)
        print()
        # used reversed() to make 'loss' the first item
        for key, value in reversed(train_metrics.items()):
            if key == "loss":
                print(f"{key+':':<20} {value:<10.6f}")
            else:
                print(f"{key+':':<20} {value:<10.2f}")
            history[key].append(value)

        # validation phase
        if val_loader:
            # add 'val_' to each metric
            # create/reset metrics values to 0 (for each epoch)
            # val_metrics_init = dict.fromkeys([f"val_{metric}" for metric in metrics], 0)
            val_metrics_init = dict.fromkeys(metrics, 0)

            val_metrics = evaluate_fn(
                val_loader,
                model,
                loss_fn,
                val_metrics_init,
                metrics,
                device,
            )

            # plot the validation metrics
            # add 'val_' prefix to the metrics
            temp_val_metric_list = {f"val_{key}":value for key, value in val_metrics.items()}
            # for key, value in reversed(val_metrics.items()):
            for key, value in reversed(temp_val_metric_list.items()):
                if key == "val_loss":
                    # plot loss with more decimal points
                    print(f"{key+':':<20} {value:<10.6f}")
                else:
                    print(f"{key+':':<20} {value:<10.2f}")
                history[key].append(value)

        # scheduler.step should be called after validation phase
        # to have access to the val set metrics (e.g., val_loss)
        if scheduler:
            if isinstance(scheduler, (StepLR)):
                # decay the lr_rate
                scheduler.step()
            elif isinstance(scheduler, (ReduceLROnPlateau)):
                # decay the lr_rate based on a val_<metric>
                scheduler.step(history['val_loss'][-1])
            # print the decayed lr_rate now
            # method 1: Caveman
            lr_now = optimizer.param_groups[0]["lr"]
            # method 2: simple but not all schedulers (e.g., ReduceLROnPlateau) has get_last_lr method
            # lr_now = scheduler.get_last_lr()
            lr_list.append(lr_now)
            # print if the lr has been changed/decayed
            if lr_list[-1] != lr_list[-2]:
                print(f'\n>>> lr_rate was decayed to: {lr_now:f}\n')

        #TODO: my method is caveman: use below link to use import tempdir
        # save the model checkpoint for this epoch (based on a criterion)
        # we overwrite the previous temp_checkpoints (in the interest of diskspace)
        if epoch + 1 == 1 and save_model_temp:
            # save the 1st epoch, nonetheless.
            checkpoint_temp = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }

            save_checkpoint(checkpoint_temp, filename=save_model_temp_filename)
            del checkpoint_temp
        elif (
            (epoch + 1 >= 2)
            and (history["val_loss"][epoch] < history["val_loss"][epoch - 1])
            and save_model_temp
        ):  # if we have prev epochs
            checkpoint_temp = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }

            save_checkpoint(checkpoint_temp, filename=save_model_temp_filename)
            del checkpoint_temp

    # save the trained model
    #TODO: Load the best performance model (through all epochs)
    # use pytorch Transfer learning tutorial for examples
    if save_model:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        save_checkpoint(
            checkpoint,
            filename=save_model_filename,
            verbose=True,
        )
        del checkpoint

    # save the lr_rate list (for all epochs + 1 (initial lr_rate) )
    plt.plot(lr_list)
    plt.yscale('log')
    plt.title('lr_rate Scheduling')
    plt.xlabel('Epochs')
    plt.ylabel('lr_rate')
    plt.show()
    #TODO: when converted training_loop as a nn.Module class, then use self.lr_list = lr_list

    return history


# TODO: write a BinarySegmentLightning class for pytorch-lightning train, eval, inference


###############################################################################
# For testing
###############################################################################
if __name__ == "__main__":
    pass
