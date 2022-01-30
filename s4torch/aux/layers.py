"""

    Layers

"""
import torch
from torch import nn
from torch.nn import functional as F


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # Use the initialization techniques for nn.Linear()
        real = nn.Linear(in_features, out_features, bias=bias)
        imag = nn.Linear(in_features, out_features, bias=bias)

        self.weight = real.weight + imag.weight.mul(1j)
        if bias:
            self.bias_tensor = real.bias + imag.bias.mul(1j)
        else:
            self.bias_tensor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input=x,
            weight=self.weight,
            bias=self.bias_tensor,
        )


if __name__ == "__main__":
    x = torch.randn(2, 512, dtype=torch.complex64)

    self = ComplexLinear(
        in_features=x.shape[-1],
        out_features=x.shape[-1],
        bias=True,
    )
    assert self(x).shape == x.shape
