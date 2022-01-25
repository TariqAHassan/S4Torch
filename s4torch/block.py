"""

    S4 Block

"""
from typing import Optional, Type

import torch
from torch import nn

from s4torch.aux.norms import TemporalBatchNorm1D
from s4torch.aux.residual import Residual, SequentialWithResidual
from s4torch.layer import S4Layer


def _make_norm(d_model: int, norm_type: Optional[str]) -> nn.Module:
    if norm_type is None:
        return nn.Identity()
    elif norm_type == "layer":
        return nn.LayerNorm(d_model)
    elif norm_type == "batch":
        return TemporalBatchNorm1D(d_model)
    else:
        raise ValueError(f"Unsupported norm type '{norm_type}'")


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
        pre_norm (bool): if ``True`` apply normalization before ``S4Layer()``,
            otherwise apply prior to final dropout
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
        pre_norm: bool = False,
        norm_type: Optional[str] = "layer",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.l_max = l_max
        self.activation = activation
        self.norm_type = norm_type
        self.pre_norm = pre_norm
        self.p_dropout = p_dropout

        self.pipeline = SequentialWithResidual(
            _make_norm(d_model, norm_type=norm_type) if pre_norm else nn.Identity(),
            S4Layer(d_model, n=n, l_max=l_max),
            activation(),
            nn.Dropout(p_dropout),
            nn.Linear(d_model, d_model, bias=True),
            Residual(),
            nn.Identity() if pre_norm else _make_norm(d_model, norm_type=norm_type),
            nn.Dropout(p_dropout),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            u (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_INPUT]``

        Returns:
            y (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_OUTPUT]``

        """
        return self.pipeline(u)


if __name__ == "__main__":
    N = 32
    d_input = 1
    d_model = 128
    d_output = 128
    l_max = 784

    u = torch.randn(1, l_max, d_model)

    s4block = S4Block(d_model, n=N, l_max=l_max, norm_type="batch")
    assert s4block(u).shape == u.shape
