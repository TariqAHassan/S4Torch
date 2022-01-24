"""

    Norms

"""
import torch
from torch import nn


class TemporalBatchNorm1D(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa
        return super().forward(input.transpose(-2, -1)).transpose(-2, -1)


if __name__ == "__main__":
    BATCH_SIZE: int = 2
    SEQ_LEN: int = 1024
    HIDDEN_DIM: int = 128
    EXPECTED_SHAPE = (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    assert TemporalBatchNorm1D(HIDDEN_DIM)(x).shape == EXPECTED_SHAPE
