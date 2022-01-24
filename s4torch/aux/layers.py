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


class TemporalBatchNorm1D(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa
        return super().forward(input.transpose(-2, -1)).transpose(-2, -1)


if __name__ == "__main__":
    BATCH_SIZE: int = 2
    SEQ_LEN: int = 1024
    HIDDEN_DIM: int = 128
    KERNEL_SIZE: int = 2
    EXPECTED_SHAPE = (BATCH_SIZE, SEQ_LEN // KERNEL_SIZE, HIDDEN_DIM)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    for pool in (TemporalMaxPooling, TemporalAvgPooling):
        assert pool(KERNEL_SIZE)(x).shape == EXPECTED_SHAPE
