import torch
import torch.nn as nn
import torchvision.transforms.v2 as TF


class DoubleConv(nn.Module):
    """
    A PyTorch NN Module to do a double 'same' 3x3CONV operation with arbitrary #filters

    ...

    Attributes
    ----------
    in_channels : int
        the #channels of the input (i.e., a 4DTensor)
    out_channels : int
        the #channels of the output (i.e., a 4DTensor)

    Methods
    -------
    forward(x)
        The forward pass of this NN (nn.Module)

    Examples
    --------
    x = DoubleConv(in_channels=64, out_channels=128)(x)
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        args:
            1. in_channels: the #channels of the input (i.e., a 4DTensor)
            1. out_channels: the #channels of the output (i.e., a 4DTensor)
        """
        # TODO: I think this class could be inside the main model (as a func): @staticmethod def double_conv(in_chennels, out_channels) = nn.Sequential(...)
        # !!!But it must be used in the __init__ of the model's Class so cannot use it before defining it right??
        super().__init__()
        self.conv = nn.Sequential(
            # padding = 1 means 'same' conv
            # bias=False, as it will get cancelled out in the following BN
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                bias=False,
            ),
            # BN was not used in the original UNET paper in 2015 (as BN was introduced in 2016), but BN never hurts right?
            # for FC layers we use BatchNorm1d(#hidden_units) but for Conv layers we use BatchNorm2d(#out_channels)
            nn.BatchNorm2d(num_features=out_channels),
            # used inplace=True, to change the input (i.e., BN output) directly, hence less MEM usage.
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of this NN (nn.Module)

        args:
            1. x: A channels_first Torch 4D-Tensor of shape (m, C, H, W)
        """
        return self.conv(x)


class UnetScratch(nn.Module):
    """
    A UNET v1 (original) model, built from scratch
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        num_channels: list = (64, 128, 256, 512),
    ):
        """

        args:
            1. in_channels: #channels of the input image to UNet (1: for grayscale, 3: for RGB images, etc.)
            1. num_classes: model output channels (number of classes in your dataset)
            #channels of the output Tensor of UNET, which is the same as the number classes (e.g., for binary segmentation == 1)
            1. num_channels: A list of channel sizes (== #filters) of the CONV layers.
            1. num_classes:
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # we must use nn.ModuleList or nn.ModuleDict as opposed to a normal list/dict (to use e.g., .eval())
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # specify the encoder part (of UNET)
        # we increase the #channels as we go down
        for channel_num in num_channels:
            self.encoder.append(DoubleConv(in_channels, channel_num))
            # the #input channels of each DoubleConv is the #output channels of the previous one
            in_channels = channel_num

        # specify the decoder part (of UNET)
        # using reversed() as with TransposeConv we decrease #channels (as we move up)
        for channel_num in reversed(num_channels):
            # used channel_num*2 as we have concatenated the corresponding CONV output (in encoder) via skip connection before doing the TransposeConv
            self.decoder.append(
                nn.ConvTranspose2d(
                    in_channels=channel_num * 2,
                    out_channels=channel_num,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.decoder.append(DoubleConv(channel_num * 2, channel_num))

        # specify the bottleneck part (at the bottom with 1024 #channels)
        self.bottle_neck = DoubleConv(num_channels[-1], num_channels[-1] * 2)

        # specify the ending 1x1CONV (#filters must match our #classes)
        # padding = 0 means 'valid' padding (the default)
        self.network_in_network = nn.Conv2d(
            in_channels=num_channels[0],
            out_channels=num_classes,
            kernel_size=1,
            padding=0,
        )

    def _encoder(self, x: torch.Tensor) -> list:
        """
        do the encoder part in Unet (the down part), minus the bottom section (aka bottleneck)


        Parameters
        ----------
        x : torch.Tensor
            A channels_first Torch 4D-Tensor of shape (m, C, H, W)

        Returns
        -------
        skip_connections: list
            a list of torch.Tensors, which are the output featuremaps of each enocder step (before applying MaxPool)
        """
        # a list to store skip_connections' inputs, which are the output featuremap of each enocder step (before applying MaxPool)
        skip_connections = []

        # do the encoder part
        for step in self.encoder:
            x = step(x)
            skip_connections.append(x)
            # we didn't add pool layers in the self.encoder as we wanted to store/cache the output of each step before applying the MaxPool
            x = self.pool(x)

        return x, skip_connections

    def _decoder(self, x: torch.Tensor, skip_connections: list[torch.Tensor]) -> None:
        """
        do the decoder part in Unet (the up part)

        Parameters
        ----------
        x : torch.Tensor
            A channels_first Torch 4D-Tensor of shape (m, C, H, W)
        skip_connections : list[torch.Tensor]
            a list of torch.Tensors, which are the output featuremaps of each enocder step (before applying MaxPool)

        Returns
        -------
        x : torch.Tensor

        Raises
        ------
        ValueError
            if skip_connections is empty
        """
        # used reversed as the last encoder's step gets concatenated to the 1st decoder's step (via skip connection)
        skip_connections_rev = reversed(skip_connections)

        for i, step in enumerate(self.decoder):
            x = step(x)

            # we wan't (i.e., the odd index steps in self.decoder)
            if i % 2 == 0:
                skip_connection = next(skip_connections_rev)

                # make sure the two tensors have the same H & W
                # note that we must change the HxW of the skip_connection (as seen in the UNET arch image) (Aladdin resized the decoder part)
                if x.shape != skip_connection.shape:
                    # method 1: using Center Cropping (from Torchvision) (might lose useful info)
                    # skip_connections = TF.CenterCrop(x.shape[2:])(skip_connection)
                    # method 2: use resizing
                    skip_connection = TF.Resize(x.shape[2:])(skip_connection)
                    # method 3: add padding

                # used dim=1 as the input is channel_first
                x = torch.cat((skip_connection, x), dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass for this Class

        args:
            1. x: A channels_first Torch 4D-Tensor of shape (m, C, H, W)
        """
        # do the encoder part
        x, skp = self._encoder(x)

        # do the bottleneck part
        x = self.bottle_neck(x)

        # do the decoder part
        x = self._decoder(x, skip_connections=skp)

        # do the 1x1Conv
        x = self.network_in_network(x)

        # TODO: add the act_fn for final layer
        # if you add a sigmoid/softmax here the loss_fn must not use from_logits
        # if self.num_classes == 1:
        #     # apply sigmoid
        #     pass
        # else:
        #     # apply softmax
        #     pass

        return x


###############################################################################
# For testing
###############################################################################
if __name__ == "__main__":

    def test(input_shape: tuple):
        """
        Test wether out model works correctly
        """
        # create a dummy RGB image batch
        # img = torch.randint(low=0, high=255, size=(10, 3, 512, 512))
        # img = torch.randn(size=(10, 3, 512, 512))
        # img = torch.randn(size=(32, 1, 572, 572))
        img = torch.randn(size=input_shape)

        model = UnetScratch(in_channels=3, num_classes=1)

        yhat = model(img)

        print(f'{"input image shape:":<20} {str(img.shape):>10}')
        print(f'{"output image shape:":<20} {str(yhat.shape):>10}')

    test((32, 3, 160, 160))
