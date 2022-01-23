"""

    Transforms

    References:
        * https://discuss.pytorch.org/t/permutate-mnist-help-needed/22901

"""
from math import prod

import torch
from numpy.random import RandomState
from torchvision.transforms import Lambda


def build_permute_transform(shape: tuple[int, ...], seed: int = 42) -> Lambda:
    """Generate a random permutation transform, conditioned on ``seed`.

    Args:
        shape (tuple[int, ...]): the shape of the input, e.g., ``[CHANNELS, HEIGHT, WIDTH]``
        seed (int): seed for random state

    Returns:
        transform (Lambda): permutation transform as a ``Lambda``

    """
    permutation = torch.as_tensor(
        RandomState(seed).permutation(prod(shape)),
        dtype=torch.int64,
    )
    return Lambda(lambda t: t.view(-1)[permutation].view(*shape))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import datasets
    from torchvision.transforms import Compose, ToTensor

    PLOT: bool = False

    image_shape = (1, 28, 28)

    ds = datasets.MNIST(
        "data",
        download=True,
        transform=Compose([ToTensor(), build_permute_transform(image_shape)]),
    )
    assert ds[0][0].shape == image_shape

    if PLOT:
        plt.imshow(ds[0][0][0].numpy(), cmap="gray")
        plt.show()
