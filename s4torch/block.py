"""

    S4 Block

"""
from typing import Optional, Type

import torch
from torch import nn

from s4torch.aux.norms import TemporalBatchNorm1D
from s4torch.layer import S4Layer


class S4Block(nn.Module):
    """S4 Block.

    Applies ``S4Layer()``, followed by an activation
    function, dropout, linear layer, skip connection and
    layer normalization.

    Args:
        d_model (int): number of internal features
        n (int): dimensionality of the state representation
        l_max (int): length of input signal
        p_dropout (float): probability of elements being set to zero
        activation (Type[nn.Module]): activation function to use after
            ``S4Layer()``.
        norm_type (str, optional): type of normalization to use.
            Options: ``batch``, ``layer``, ``None``.

    """

    def __init__(
        self,
        d_model: int,
        n: int,
        l_max: int,
        p_dropout: float = 0.0,
        activation: Type[nn.Module] = nn.GELU,
        norm_type: Optional[str] = "layer",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.l_max = l_max
        self.activation = activation
        self.norm_type = norm_type
        self.p_dropout = p_dropout

        self.pipeline = nn.Sequential(
            S4Layer(d_model, n=n, l_max=l_max),
            activation(),
            nn.Dropout(p_dropout),
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.Dropout(p_dropout),
        )

        if norm_type is None:
            self.norm = nn.Identity()
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == "batch":
            self.norm = TemporalBatchNorm1D(d_model)
        else:
            raise ValueError(f"Unsupported norm type '{norm_type}'")

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

    s4block = S4Block(d_model, n=N, l_max=l_max, norm_type="batch")
    assert s4block(u).shape == u.shape
