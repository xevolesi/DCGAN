"""Home for discriminator code."""
import torch
from torch import nn


def conv4x4(
        in_channels: int,
        out_channels: int,
        stride: int,
        padding: int
) -> nn.Conv2d:
    """Construct regular DCGAN discriminator 4x4 convolution."""
    return nn.Conv2d(in_channels, out_channels, 4, stride, padding)


class DiscriminatorBlock(nn.Module):
    """Stack of [CONV, NORM, ACT] layers."""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            padding: int,
            is_last_layer: bool
    ) -> None:
        super().__init__()
        self.conv = conv4x4(in_channels, out_channels, stride, padding)
        self.bn = None if is_last_layer else nn.BatchNorm2d(out_channels)
        self.act = nn.Sigmoid() if is_last_layer else nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.act(x)
        return x


class Discriminator(nn.Module):
    """DCGAN discriminator. Simple deep convolutional classifier."""
    def __init__(
            self,
            in_channels: int = 3,
            base_width: int = 64,
            num_layers: int = 5
    ) -> None:
        super().__init__()
        blocks = [DiscriminatorBlock(in_channels, base_width, 2, 1, False)]
        for i in range(num_layers-1):
            is_last_layer = i == (num_layers - 1) - 1
            in_channels = base_width * 2 ** i
            out_channels = 1 if is_last_layer else base_width * 2 ** (i + 1)
            stride = 1 if is_last_layer else 2
            padding = 0 if is_last_layer else 1
            blocks.append(
                DiscriminatorBlock(
                    in_channels,
                    out_channels,
                    stride,
                    padding,
                    is_last_layer
                )
            )
        self.block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
