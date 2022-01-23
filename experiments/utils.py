"""

    Utils

"""
from pathlib import Path

import torch

_DEFAULT_OUTPUT_DIR = Path("~/s4-output")


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


class OutputPath:
    def __init__(self, output_dir: str, run_name: str) -> None:
        self.output_dir = (
            (Path(output_dir) or _DEFAULT_OUTPUT_DIR).expanduser().absolute()
        )
        self.run_name = run_name

    @property
    def logs_path(self) -> Path:
        path = self.output_dir.joinpath("logs")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def checkpoints_path(self) -> Path:
        path = self.output_dir.joinpath(f"checkpoints/{self.run_name}")
        path.mkdir(parents=True, exist_ok=True)
        return path
