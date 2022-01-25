"""

    S4 Model

"""
from __future__ import annotations

from typing import Optional, Type

import torch
from torch import nn

from s4torch.block import S4Block


def _parse_pool_kernel(pool_kernel: Optional[int | tuple[int]]) -> int:
    if pool_kernel is None:
        return 1
    elif isinstance(pool_kernel, tuple):
        return pool_kernel[0]
    elif isinstance(pool_kernel, int):
        return pool_kernel
    else:
        raise TypeError(f"Unable to parse `pool_kernel`, got {pool_kernel}")


def _seq_length_schedule(
    n_blocks: int,
    l_max: int,
    pool_kernel: Optional[int | tuple[int]],
) -> list[tuple[int, int]]:
    ppk = _parse_pool_kernel(pool_kernel)

    schedule = list()
    for depth in range(n_blocks + 1):
        l_max_next = max(2, l_max // ppk)
        pool_ok = l_max_next > ppk
        schedule.append((l_max, pool_ok))
        l_max = l_max_next
    return schedule


class S4Model(nn.Module):
    """S4 Model.

    High-level implementation of the S4 model which:

        1. Encodes the input using a linear layer
        2. Applies ``1..n_blocks`` S4 blocks
        3. Decodes the output of step 2 using another linear layer

    Args:
        d_input (int): number of input features
        d_model (int): number of internal features
        d_output (int): number of features to return
        n_blocks (int): number of S4 blocks to construct
        n (int): dimensionality of the state representation
        l_max (int): length of input signal
        collapse (bool): if ``True`` average over time prior to
            decoding the result of the S4 block(s). (Useful for
            classification tasks.)
        p_dropout (float): probability of elements being set to zero
        activation (Type[nn.Module]): activation function to use after
            ``S4Layer()``.
        pre_norm (bool): if ``True`` apply normalization before ``S4Layer()``,
            otherwise apply prior to final dropout
        norm_type (str, optional): type of normalization to use.
            Options: ``batch``, ``layer``, ``None``.
        pooling (nn.AvgPool1d, nn.MaxPool1d, optional): pooling method to use
            following each ``S4Block()``.

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
        activation: Type[nn.Module] = nn.GELU,
        pre_norm: bool = False,
        norm_type: Optional[str] = "layer",
        pooling: Optional[nn.AvgPool1d | nn.MaxPool1d] = None,
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
        self.pre_norm = pre_norm
        self.norm_type = norm_type
        self.pooling = pooling

        *self.seq_len_schedule, (self.seq_len_out, _) = _seq_length_schedule(
            n_blocks=n_blocks,
            l_max=l_max,
            pool_kernel=None if self.pooling is None else self.pooling.kernel_size,
        )

        self.encoder = nn.Linear(self.d_input, self.d_model)
        self.decoder = nn.Linear(self.d_model, self.d_output)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    S4Block(
                        d_model=d_model,
                        n=n,
                        l_max=seq_len,
                        p_dropout=p_dropout,
                        activation=activation,
                        pre_norm=pre_norm,
                        norm_type=norm_type,
                        pooling=pooling if pooling and pool_ok else None,
                    ),
                )
                for (seq_len, pool_ok) in self.seq_len_schedule
            ]
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            u (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_INPUT]``

        Returns:
            y (torch.Tensor): a tensor of the form ``[BATCH, D_OUTPUT]`` if ``collapse``
                is ``True`` and ``[BATCH, SEQ_LEN // (POOL_KERNEL ** n_block), D_INPUT]``
                otherwise, where ``POOL_KERNEL`` is the kernel size of the ``pooling``
                layer. (If ``pooling`` is ``None``, ``POOL_KERNEL=1``.)

        """
        y = self.encoder(u)
        for block in self.blocks:
            y = block(y)
        return self.decoder(y.mean(dim=1) if self.collapse else y)


if __name__ == "__main__":
    N = 32
    d_input = 1
    d_model = 128
    d_output = 128
    n_blocks = 3
    l_max = 784

    u = torch.randn(1, l_max, d_input)

    s4model = S4Model(
        d_input,
        d_model=d_model,
        d_output=d_output,
        n_blocks=n_blocks,
        n=N,
        l_max=l_max,
        collapse=False,
    )
    assert s4model(u).shape == (u.shape[0], s4model.seq_len_out, s4model.d_output)
