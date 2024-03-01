"""

    Encoders

"""
import torch
from torch import nn


class StandardEncoder(nn.Linear):
    def __init__(self, d_input: int, d_model: int, bias: bool = True) -> None:
        super().__init__(in_features=d_input, out_features=d_model, bias=bias)
        self.d_input = d_input
        self.d_model = d_model


if __name__ == "__main__":
    x = torch.randn(2, 2 ** 16, 1)

    tform = StandardEncoder(x.shape[-1], d_model=128)
    assert tform(x).shape == (*x.shape[:-1], tform.d_model)
