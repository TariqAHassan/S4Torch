"""

    S4 Model

"""
from typing import Any

import torch
from torch import nn

from s4torch.layer import S4Layer


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

    u_block = torch.randn((1, l_max, d_model)).float()
    u_model = torch.randn((1, l_max, d_input)).float()

    s4block = S4Block(d_model, n=N, l_max=l_max)
    assert s4block(u_block).shape == u_block.shape

    s4model = S4Model(
        d_input,
        d_model=d_model,
        d_output=d_output,
        n_layers=3,
        n=N,
        l_max=l_max,
        collapse=False,
    )
    assert s4model(u_model).shape == (*u_model.shape[:-1], s4model.d_output)
