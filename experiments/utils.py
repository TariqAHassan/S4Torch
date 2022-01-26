"""

    Utils

"""
from itertools import chain
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from s4torch.block import S4Block
from s4torch.layer import S4Layer


class OutputPaths:
    def __init__(self, output_dir: str, run_name: str) -> None:
        self.output_dir = Path(output_dir).expanduser().absolute()
        self.run_name = run_name

    def _make_dir(self, name: str) -> Path:
        path = self.output_dir.joinpath(name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logs(self) -> Path:
        return self._make_dir("logs")

    @property
    def checkpoints(self) -> Path:
        return self._make_dir(f"checkpoints/{self.run_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def to_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.ndim <= 1:
        raise IndexError(f"Input must be at least 2D, got {x.ndim}D")
    elif x.ndim == 2:
        return x.unsqueeze(-1)
    elif x.ndim == 3:
        return x
    elif x.ndim == 4:
        return x.flatten(2).transpose(-2, -1)
    else:
        raise IndexError(f"Expected 2D, 3D or 4D data, got {x.ndim}D")


def _parse_single_s4block(block: S4Block) -> tuple[S4Layer, list[nn.Module]]:
    keys = dict(block.named_modules(remove_duplicate=True)).keys()
    if all(k.startswith("pipeline") or k == "" for k in keys):
        pre, s4, *post = list(block.pipeline)
    else:
        raise KeyError("Unexpected modules found in block")
    return s4, [pre, *post]


def parse_params_in_s4blocks(
    blocks: list[S4Block],
) -> tuple[Iterable[nn.Parameter], Iterable[nn.Parameter]]:
    def chain_params(modules: list[nn.Module]) -> Iterable[nn.Parameter]:
        return chain(*map(lambda m: m.parameters(), modules))

    s4layers, others = list(), list()
    for block in blocks:
        s4, other = _parse_single_s4block(block)
        s4layers.append(s4)
        others.extend(other)
    return chain_params(s4layers), chain_params(others)
