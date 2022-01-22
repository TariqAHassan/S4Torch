"""

    S4 Block

"""
from typing import Any

import torch
from torch import nn

from s4torch import S4Layer


class S4Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n: int,
        l_max: int,
        p_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.l_max = l_max
        self.p_dropout = p_dropout

        self.pipeline = nn.Sequential(
            S4Layer(d_model, n=n, l_max=l_max, **kwargs),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.Dropout(p_dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        z = self.pipeline(u)
        return self.norm(z + u)


if __name__ == "__main__":
    N = 32
    d_input = 1
    d_model = 128
    d_output = 128
    l_max = 784

    u = torch.randn((1, l_max, d_model))

    s4block = S4Block(d_model, n=N, l_max=l_max)
    assert s4block(u).shape == u.shape
