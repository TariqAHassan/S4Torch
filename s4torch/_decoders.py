"""

    Decoders

"""
import torch
from torch import nn

from s4torch.aux.layers import ComplexLinear


class ComplexDecoder(ComplexLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ToDo: explore using use output.real, which conserves sign information
        return super().forward(x).abs()
