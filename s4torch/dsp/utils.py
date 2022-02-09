"""

    DSP Utils

"""
from math import ceil, log2

import torch
from torch.nn import functional as F


def is_pow2(i: int) -> int:
    return log2(i) % 1 == 0


def next_pow2(i: int) -> int:
    return 2 ** ceil(log2(i))


def pow2pad(x: torch.Tensor) -> torch.Tensor:
    *_, t = x.shape
    if is_pow2(t):
        return x
    return F.pad(x, pad=(0, next_pow2(t) - t))
