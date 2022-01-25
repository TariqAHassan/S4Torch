"""

    Residual

"""
from typing import Any

import torch
from torch import nn


class Residual(nn.Module):
    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return y + x


class GatedResidual(Residual):
    def __init__(
        self,
        features: int = 1,
        init_value: float = 0.0,  # sigmoid(0) = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.init_value = init_value

        self.gate = nn.Parameter(
            torch.fill_(torch.empty(1, 1, features), value=init_value),
        )

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate.sigmoid()
        return super().forward(gate * y, (1 - gate) * x)


class SequentialWithResidual(nn.Sequential):
    @staticmethod
    def _is_residual_module(obj: Any) -> bool:
        return isinstance(obj, Residual) or issubclass(type(obj), Residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for module in self:
            if self._is_residual_module(module):
                y = module(y, x=x)
            else:
                y = module(y)
        return y


if __name__ == "__main__":
    swr = SequentialWithResidual(nn.AvgPool1d(2), nn.ReLU())

    x = torch.randn(2, 512, 128)

    # Test `SequentialWithResidual()`
    output = swr(x)
    assert output.shape == (x.shape[0], x.shape[1], x.shape[-1] // 2)  # AvgPool1d
    assert output.min() >= 0  # ReLU

    # Test `SequentialWithResidual._is_residual_module()`
    assert not swr._is_residual_module(99)
    assert not swr._is_residual_module(None)
    assert not swr._is_residual_module(nn.ReLU())
    assert not swr._is_residual_module(nn.ReLU)
    assert not swr._is_residual_module(Residual)
    assert swr._is_residual_module(Residual())
    assert swr._is_residual_module(GatedResidual())

    # Test residual layers
    for layer in (Residual(), GatedResidual(x.shape[-1])):
        assert layer(x, x).shape == x.shape
