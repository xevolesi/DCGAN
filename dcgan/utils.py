import math


def get_nearest_power_of_2(final_depth: int, number_of_layers: int) -> int:
    """
    Compute nearest power of 2 for DCGAN block. This number will be used to
    determine how many input and output channels the block should has.

    :param final_depth: Number of channels in penultimate block;
    :param number_of_layers: Total number of layers in DCGAN generator.
    """
    power = int(math.ceil(math.log2(final_depth)) + number_of_layers - 2)
    return power
