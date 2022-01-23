"""

    Utils

"""
from __future__ import annotations
import re
from typing import Any

import torch
from torch import nn


class TemporalBasePooling:
    kernel_size: int | tuple[int]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @property
    def type(self) -> str:
        (type,) = re.findall(r"Temporal(.*)Pooling", string=self.__class__.__name__)
        return type


class TemporalMaxPooling(TemporalBasePooling, nn.MaxPool1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa
        return super().forward(input.transpose(-2, -1)).transpose(-2, -1)


class TemporalAvgPooling(TemporalBasePooling, nn.AvgPool1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa
        return super().forward(input.transpose(-2, -1)).transpose(-2, -1)
