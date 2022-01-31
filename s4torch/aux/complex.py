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

    Args:
        layer (nn.Module): a layer intended for use with
            real-valued inputs

    Returns:
        ComplexWrapper (nn.Module): a wrapper for ``layer``

    """

    class ComplexWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.real_layer = deepcopy(layer)
            self.imag_layer = deepcopy(layer)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.real_layer(x.real) + self.imag_layer(x.imag).mul(1j)

    return ComplexWrapper()


if __name__ == "__main__":
    x = torch.randn(1, 10, 128, dtype=torch.complex64)
    layer = as_complex_layer(nn.LayerNorm(128))

    assert layer(x).shape == x.shape
    assert layer(x).dtype == x.dtype
