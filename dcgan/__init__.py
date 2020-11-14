from typing import Tuple

from torch import nn

from .generator import Generator
from .discriminator import Discriminator


def paper_dcgan() -> Tuple[nn.Module, nn.Module]:
    g = Generator(100, 3, 5, 128)
    d = Discriminator(3, 64, 5)
    return g, d
