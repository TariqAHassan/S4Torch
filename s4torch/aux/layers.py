"""

    Utils

"""
import torch
from torch import nn


class TemporalMaxPooling(nn.MaxPool1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa
        return super().forward(input.transpose(-2, -1)).transpose(-2, -1)


class TemporalAvgPooling(nn.AvgPool1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa
        return super().forward(input.transpose(-2, -1)).transpose(-2, -1)
