"""

    S4 Model

"""
from typing import Any

import torch
from torch import nn

from s4torch.block import S4Block


class S4Model(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        n_layers: int,
        n: int,
        l_max: int,
        collapse: bool = False,
        p_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output
        self.n = n
        self.l_max = l_max
        self.collapse = collapse
        self.p_dropout = p_dropout

        self.encoder = nn.Linear(self.d_input, self.d_model)
        self.decoder = nn.Linear(self.d_model, self.d_output)
        self.layers = nn.ModuleList(
            [
                S4Block(
                    d_model=d_model,
                    n=n,
                    l_max=l_max,
                    p_dropout=p_dropout,
                    **kwargs,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        y = self.encoder(u)
        for layer in self.layers:
            y = layer(y)
        return self.decoder(y.mean(dim=1) if self.collapse else y)


if __name__ == "__main__":
    N = 32
    d_input = 1
    d_model = 128
    d_output = 128
    l_max = 784

    u = torch.randn((1, l_max, d_input))

    s4model = S4Model(
        d_input,
        d_model=d_model,
        d_output=d_output,
        n_layers=3,
        n=N,
        l_max=l_max,
        collapse=False,
    )
    assert s4model(u).shape == (*u.shape[:-1], s4model.d_output)
