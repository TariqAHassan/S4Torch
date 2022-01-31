"""

    Layers

    References:
        * https://stackoverflow.com/a/54170758

"""
import torch
from torch import nn
from torch.nn import functional as F


class ComplexDropout(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        if not 0 < p < 1:
            raise ValueError(f"`p` expected to be on (0, 1), got {p}")

        self._binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def _get_mask(self, x: torch.Tensor) -> torch.Tensor:
        mask = self._binomial.sample(x.shape)
        return (mask + mask.mul(1j)).type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = x * self._get_mask(x) * (1.0 / (1 - self.p))
        return x


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # Use the initialization techniques from `nn.Linear()`
        real = nn.Linear(in_features, out_features, bias=bias)
        imag = nn.Linear(in_features, out_features, bias=bias)

        self.weight = nn.Parameter(real.weight + imag.weight.mul(1j))
        if bias:
            self.bias_tensor = nn.Parameter(real.bias + imag.bias.mul(1j))
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

    for layer in (
        ComplexDropout(0.1),
        ComplexLinear(
            in_features=x.shape[-1],
            out_features=x.shape[-1],
            bias=True,
        ),
    ):
        assert layer(x).shape == x.shape
