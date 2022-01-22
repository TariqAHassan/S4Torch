"""

    S4 Model

"""
from typing import Any

import torch
from torch import nn

from s4torch.block import S4Block


class S4Model(nn.Module):
    """S4 Model.

    High-level implementation of the S4 model which:

        1. encodes the input using a linear layer
        2. applies ``1..n_blocks`` S4 blocks
        3. decodes the output of 2. using another linear layer

    Args:
        d_input (int): number of input features
        d_model (int): number of internal features
        d_output (int): number of features to return
        n_blocks (int): number of S4 layers to construct
        n (int): dimensionality of the state representation
        l_max (int): length of input signal
        collapse (bool): if ``True`` average over time prior to
            decoding the result of the S4 block(s). (Useful for
            classification tasks.)
        p_dropout (float): probability of elements being set to zero
        **kwargs (Keyword Args): Keyword arguments to be passed to
            ``S4Block()``.

    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        n_blocks: int,
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
        self.n_blocks = n_blocks
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
                for _ in range(n_blocks)
            ]
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            u (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_INPUT]``

        Returns:
            y (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_OUTPUT]``

        """
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
        n_blocks=3,
        n=N,
        l_max=l_max,
        collapse=False,
    )
    assert s4model(u).shape == (*u.shape[:-1], s4model.d_output)
