import torch
from dcgan.generator import Generator
from dcgan.discriminator import Discriminator
from dcgan import paper_dcgan

if __name__ == '__main__':
    g, d = paper_dcgan()
    noise = torch.randn((1, 100))
    gout = g(noise)
    dout = d(gout)