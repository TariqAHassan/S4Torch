"""

    Encoders

"""
import torch
from torch import nn

from s4torch.dsp.cwt import Cwt


class StandardEncoder(nn.Module):
    def __init__(self, d_input: int, d_model: int, bias: bool = True) -> None:
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.bias = bias

        self.linear = nn.Linear(d_input, out_features=d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class WaveletEncoder(nn.Module):
    def __init__(self, cwt: Cwt, d_model: int, bias: bool = True) -> None:
        super().__init__()
        self.cwt = cwt
        self.d_model = d_model
        self.bias = bias

        self.linear = nn.Linear(cwt.n_scales, out_features=d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        elif x.ndim != 2:
            raise IndexError(f"Expected x to be 2D or 3D with 1 feature")

        x = self.cwt(x).transpose(-2, -1).abs()
        return self.linear(x)


if __name__ == "__main__":
    x = torch.randn(2, 2 ** 16)

    wtform = Wavelet(Cwt(x.shape[-1]), d_model=128)
    assert wtform(x).shape == (*x.shape, wtform.d_model)
