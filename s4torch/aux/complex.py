"""

    Complex Number Handling

"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


def as_complex_layer(layer: nn.Module) -> nn.Module:
    """Decorator for constructing layers which support
    complex-valued inputs from layers build for real-valued
    inputs.

    This layer is implemented by applying the layer
    to the real and imaginary components independently
    (i.e., against the Cartesian form of the input).

    Args:
        layer (nn.Module): a layer intended for use with
            real-valued inputs

    Returns:
        ComplexWrapper (nn.Module): a wrapper for ``layer``

    """

    class ComplexWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.re_layer = deepcopy(layer)
            self.im_layer = deepcopy(layer)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (self.re_layer(x.real) + self.im_layer(x.imag).mul(1j)).type_as(x)

    return ComplexWrapper()


if __name__ == "__main__":
    x = torch.randn(1, 10, 128, dtype=torch.complex64)
    layer = as_complex_layer(nn.LayerNorm(128))

    assert layer(x).shape == x.shape
    assert layer(x).dtype == x.dtype
