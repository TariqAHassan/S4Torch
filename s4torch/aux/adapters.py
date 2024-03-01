"""

    Adapters

"""

import torch
from torch import nn


class TemporalAdapter(nn.Module):
    def __init__(self, wrapped: nn.Module) -> None:
        super().__init__()
        self.add_module("wrapped", wrapped)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [BATCH, SEQ_LEN, D_MODEL]
        return self.wrapped(x.transpose(-2, -1)).transpose(-2, -1)


if __name__ == "__main__":
    BATCH_SIZE: int = 2
    SEQ_LEN: int = 1024
    HIDDEN_DIM: int = 128
    KERNEL_SIZE: int = 2
    EXPECTED_SHAPE = (BATCH_SIZE, SEQ_LEN // KERNEL_SIZE, HIDDEN_DIM)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    for wrapped in (nn.AvgPool1d(2), nn.MaxPool1d(2)):
        assert TemporalAdapter(wrapped)(x).shape == EXPECTED_SHAPE
