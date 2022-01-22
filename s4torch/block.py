"""

    S4 Block

"""
from typing import Any, Type

import torch
from torch import nn

from s4torch.layer import S4Layer


class S4Block(nn.Module):
    """S4 Block.

    Applies an ``S4Layer()``, followed by an activation
    function, dropout, linear layer, skip connection and
    layer normalization.

    Args:
        d_model (int): number of internal features
        n (int): dimensionality of the state representation
        l_max (int): length of input signal
        p_dropout (float): probability of elements being set to zero
        activation (Type[nn.Module]): activation function to use after
            ``S4Layer()``.
        **kwargs (Keyword Args): Keyword arguments to be passed to
            ``S4Layer()``.

    """

    def __init__(
        self,
        d_model: int,
        n: int,
        l_max: int,
        p_dropout: float = 0.0,
        activation: Type[nn.Module] = nn.GELU,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.l_max = l_max
        self.activation = activation
        self.p_dropout = p_dropout

        self.pipeline = nn.Sequential(
            S4Layer(d_model, n=n, l_max=l_max, **kwargs),
            activation(),
            nn.Dropout(p_dropout),
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.Dropout(p_dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            u (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_INPUT]``

        Returns:
            y (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_OUTPUT]``

        """
        z = self.pipeline(u)
        return self.norm(z + u)


if __name__ == "__main__":
    N = 32
    d_input = 1
    d_model = 128
    d_output = 128
    l_max = 784

    u = torch.randn(1, l_max, d_model)

    s4block = S4Block(d_model, n=N, l_max=l_max)
    assert s4block(u).shape == u.shape
