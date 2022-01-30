"""

    Decoders

"""
import torch
from torch import nn
from s4torch.aux.layers import ComplexLinear


class StandardDecoder(nn.Linear):
    def __init__(self, d_model: int, d_output: int, bias: bool = True) -> None:
        super().__init__(in_features=d_model, out_features=d_output, bias=bias)
        self.d_model = d_model
        self.d_output = d_output


class ComplexDecoder(ComplexLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).abs()
