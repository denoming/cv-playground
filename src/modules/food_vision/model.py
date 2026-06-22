from torch import nn

class ConvBlock(nn.Module):
    """
    Creates the convolution block architecture.

    Args:
        in_shape: An integer indicating number of input channels.
        hidden_units: An integer indicating number of hidden units between layers.
        out_shape: An integer indicating number of output units.
        conv_size: An integer indicating convolution kernel size.
        pool_size: An integer indicating pooling kernel size.
    """
    def __init__(self, in_shape: int, hidden_units: int, out_shape: int, conv_size: int, pool_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=hidden_units, kernel_size=conv_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=out_shape, kernel_size=conv_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size),
        )

    def forward(self, inputs):
        return self.block(inputs)

class ClassifierBlock(nn.Module):
    """
    Creates the classifier block architecture.

    Args:
        in_shape: An integer indicating number of input channels.
        out_shape: An integer indicating number of output channels.
    """
    def __init__(self, in_shape: int, out_shape: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_shape, out_features=out_shape),
        )

    def forward(self, inputs):
        return self.block(inputs)


class TinyVgg(nn.Module):
    """
    Creates the TinyVgg architecture.

    Replicates the TinyVgg architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/

    Args:
    in_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    out_shape: An integer indicating number of output units.
    """
    def __init__(self, in_shape: int, hidden_units: int, out_shape: int):
        super().__init__()
        self.block1 = ConvBlock(in_shape, hidden_units, hidden_units, 3, 2)
        self.block2 = ConvBlock(hidden_units, hidden_units, hidden_units, 3, 2)
        self.block3 = ClassifierBlock(hidden_units*13*13, out_shape)


    def forward(self, inputs):
        z = self.block1(inputs)
        z = self.block2(z)
        outputs = self.block3(z)
        return outputs