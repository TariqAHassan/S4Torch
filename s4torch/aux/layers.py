"""

    Layers

    References:
        * https://stackoverflow.com/a/54170758

"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import _verify_batch_size as verify_batch_size


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


class ComplexBatchNorm1d(nn.Module):
    weight: Optional[torch.Tensor]
    bias: Optional[torch.Tensor]

    running_mean: Optional[torch.Tensor]
    running_var: Optional[torch.Tensor]
    num_batches_tracked: Optional[torch.Tensor]

    def __init__(
        self,
        num_features: int,
        eps: complex = 1e-05 + 1e-05j,
        momentum: Optional[complex] = 0.1 + 0.1j,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(
                torch.ones(1, num_features, 1, dtype=torch.complex64).add(1j)
            )
            self.bias = nn.Parameter(
                torch.zeros(1, num_features, 1, dtype=torch.complex64)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer(
                "running_mean",
                torch.zeros(1, num_features, 1, dtype=torch.complex64),
            )
            self.register_buffer(
                "running_var",
                torch.ones(1, num_features, 1, dtype=torch.complex64).add(1j),
            )
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(0, dtype=torch.long),
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, "
            f"eps={self.eps}, "
            f"momentum={self.momentum}, "
            f"affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}"
        )

    def _get_exp_avg_factor(self) -> complex:
        if self.momentum is None:
            factor = 1.0 / float(self.num_batches_tracked)
            return factor + (factor * 1j)
        else:
            return self.momentum

    def _update_running_mean(self, new: torch.Tensor) -> None:
        factor = self._get_exp_avg_factor()
        if self.running_mean is None:
            self.running_mean = factor * new
        else:
            self.running_mean = ((1 - factor) * self.running_mean) + (factor * new)

    def _update_running_var(self, new: torch.Tensor, input_shape: torch.Size) -> None:
        factor = self._get_exp_avg_factor()
        if self.running_var is None:
            self.running_var = factor * new
        else:
            n = input_shape.numel() / input_shape[1]
            self.running_var = (factor * new * n / (n - 1)) + (
                1 - factor
            ) * self.running_var

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x = [B, SEQ_LEN, DIM]
        if self.training:
            verify_batch_size(x.size())

        mean = x.mean(dim=(0, 2), keepdim=True)
        var = x.var(dim=(0, 2), keepdim=True, unbiased=False)
        if self.track_running_stats:
            if self.training:
                self.num_batches_tracked += 1
                self._update_running_mean(mean)
                self._update_running_var(var, input_shape=x.shape)
                mean, var = self.running_mean, self.running_var
            elif self.num_batches_tracked.item() > 0:
                mean, var = self.running_mean, self.running_var

        out = (x - mean) / var.add(self.eps).sqrt()
        if self.affine:
            out = out * self.weight + self.bias
        return out


class ComplexLayerNorm1d(nn.Module):
    weight: Optional[torch.Tensor]
    bias: Optional[torch.Tensor]

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
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

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
    from torch.nn import functional as F

    x = torch.randn(2, 512, 100, dtype=torch.complex64)

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

    # LayerNorm
    actual_ln = ComplexLayerNorm1d(
        x.shape[-1],
        eps=1e-5,  # noqa
        elementwise_affine=False,
    )(x.real)
    expected_ln = nn.LayerNorm(x.shape[-1], elementwise_affine=False)(x.real)
    assert F.mse_loss(actual_ln, expected_ln).item() < 1e-5

    # BatchNorm
    x0 = torch.ones(2, 16, 10)
    x1 = torch.randn_like(x0).mul(10)
    x2 = torch.randn_like(x0).mul(15)

    sbnorm = nn.BatchNorm1d(x0.shape[1], eps=1e-5, momentum=0.1, affine=False)
    cbnorm = ComplexBatchNorm1d(x0.shape[1], eps=1e-5, momentum=0.1, affine=False)

    sbnorm(x0)
    cbnorm(x0)
    sbnorm(x1)
    cbnorm(x1)
    sbnorm(x2)
    cbnorm(x2)

    assert (
        torch.isclose(cbnorm.running_mean.real[0, :, 0], sbnorm.running_mean)
        .all()
        .item()
    )
    assert (
        torch.isclose(cbnorm.running_var.real[0, :, 0], sbnorm.running_var).all().item()
    )
