"""

    Layers

    References:
        * https://stackoverflow.com/a/54170758

    ToDo: implement complex-valued batch norm

"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class ComplexDropout(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        if not 0 <= p < 1:
            raise ValueError(f"`p` expected to be on [0, 1), got {p}")

        self._binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def extra_repr(self) -> str:
        return f"p={self.p}"

    def _get_mask(self, x: torch.Tensor) -> torch.Tensor:
        mask = self._binomial.sample(x.shape)
        return (mask + mask.mul(1j)).type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p:
            x = x * self._get_mask(x) * (1.0 / (1 - self.p))
        return x


class ComplexLayerNorm1d(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: complex = 1e-05 + 1e-05j,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(1, 1, normalized_shape, dtype=torch.complex64).add(1j)
            )
            self.bias = nn.Parameter(
                torch.zeros(1, 1, normalized_shape, dtype=torch.complex64)
            )
        else:
            self.weight, self.bias = None, None

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x = [B, SEQ_LEN, DIM]
        num = x - x.mean(dim=-1, keepdim=True)
        denom = x.var(dim=-1, keepdim=True, unbiased=False).add(self.eps).sqrt()
        out = num / denom
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
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

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input=x,
            weight=self.weight,
            bias=self.bias_tensor,
        )


if __name__ == "__main__":
    x = torch.randn(2, 100, 512, dtype=torch.complex64)

    for layer in (
        ComplexDropout(0.1),
        ComplexLayerNorm1d(x.shape[-1]),
        ComplexLinear(
            in_features=x.shape[-1],
            out_features=x.shape[-1],
            bias=True,
        ),
    ):
        assert layer(x).shape == x.shape
