"""Home for generator code."""
import torch
from torch import nn

from dcgan.utils import get_nearest_power_of_2


def transpose_conv4x4(
        in_features: int,
        out_features: int,
        stride: int = 2,
        padding: int = 1
) -> nn.ConvTranspose2d:
    """Construct regular DCGAN 4x4 transposed convolution."""
    return nn.ConvTranspose2d(in_features, out_features, 4, stride, padding)


class GeneratorBlock(nn.Module):
    """Stack of [CONV, NORM, ACT] layers."""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            padding: int,
            is_last_block: bool
    ) -> None:
        """
        :param in_channels: Number of input channels for transposed conv;
        :param out_channels: Number of output channels for transposed conv;
        :param stride: Stride for transposed conv;
        :param padding: Padding for transposed conv;
        :param is_last_block: Whether the block is the last one.
        """
        super().__init__()
        layers = [transpose_conv4x4(in_channels, out_channels, stride, padding)]
        if is_last_block:
            layers.append(nn.Tanh())
        else:
            layers.extend([nn.BatchNorm2d(out_channels), nn.ReLU()])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Generator(nn.Module):
    """DCGAN generator with Transposed convolutions."""
    def __init__(
            self,
            input_vector_dim: int = 100,
            output_channels: int = 3,
            number_of_layers: int = 5,
            final_depth: int = 128
    ) -> None:
        """
        :param input_vector_dim: The dimension of noise vector;
        :param output_channels: Number of channels in generated images;
        :param number_of_layers: Total number of transposed convolutional
            layer;
        :param final_depth: Number of channels before generated images.
        """
        super().__init__()
        self.final_depth = final_depth
        self.output_channels = output_channels
        self.input_vector_dim = input_vector_dim
        self.number_of_layers = number_of_layers
        self.G = self._build()

    def _build(self):
        """Build generator body. The DCGAN's generator has a good pattern for
        the number of input channels and the number of output channels. All
        these numbers are powers of 2. This function uses this pattern with
        some restrictions imposed on the last layer of the generator."""
        power_of_2 = get_nearest_power_of_2(
            self.final_depth,
            self.number_of_layers
        )
        layers = []
        for i in range(self.number_of_layers):
            ic = self.input_vector_dim if i == 0 else 2 ** (power_of_2 - i + 1)
            oc = 2 ** power_of_2 if i == 0 else 2 ** (power_of_2 - i)
            stride = 1 if i == 0 else 2
            padding = 0 if i == 0 else 1
            is_last_block = i == self.number_of_layers - 1
            layers.append(
                GeneratorBlock(
                    ic,
                    self.output_channels if is_last_block else oc,
                    stride,
                    padding,
                    is_last_block
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _ = x.size()
        x = x.view(batch_size, self.input_vector_dim, 1, 1)
        return self.G(x)
