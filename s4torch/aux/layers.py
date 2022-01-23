"""

    Utils

"""
from __future__ import annotations

import re
from abc import ABC

import torch
from torch import nn


class TemporalBasePooling(ABC):
    kernel_size: int | tuple[int]

    @property
    def type(self) -> str:
        (type_,) = re.findall(r"Temporal(.*)Pooling", string=self.__class__.__name__)
        return type_.lower()


class TemporalAvgPooling(TemporalBasePooling, nn.AvgPool1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa
        return super().forward(input.transpose(-2, -1)).transpose(-2, -1)


class TemporalMaxPooling(TemporalBasePooling, nn.MaxPool1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa
        return super().forward(input.transpose(-2, -1)).transpose(-2, -1)


if __name__ == "__main__":
    x = torch.randn(1, 1024, 128)

    for pool in (TemporalMaxPooling(2), TemporalAvgPooling(2)):
        assert pool(x).shape == (x.shape[0], x.shape[1] // 2, x.shape[-1])
