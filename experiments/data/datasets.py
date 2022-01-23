"""

    Datasets

"""
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torchaudio.datasets import SPEECHCOMMANDS as _SpeechCommands  # noqa


class SpeechCommands(_SpeechCommands):
    SEGMENT_SIZE: int = 16_000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.label_ids = {l: e for e, l in enumerate(self.classes)}
        self._walker = [i for i in self._walker if Path(i).parent.name in self.classes]

    @property
    def classes(self) -> list[str]:
        return [
            "bed",
            "cat",
            "down",
            "five",
            "forward",
            "go",
            "house",
            "left",
            "marvin",
            "no",
            "on",
            "right",
            "sheila",
            "tree",
            "up",
            "visual",
            "yes",
            "backward",
            "bird",
            "dog",
            "eight",
            "follow",
            "four",
            "happy",
            "learn",
            "nine",
            "off",
            "one",
            "seven",
            "six",
            "stop",
            "three",
            "two",
            "wow",
            "zero",
        ]

    def _pad(self, y: torch.Tensor) -> torch.Tensor:
        if y.shape[-1] == self.SEGMENT_SIZE:
            return y
        elif y.shape[-1] < self.SEGMENT_SIZE:
            return F.pad(y, pad=(0, self.SEGMENT_SIZE - y.shape[-1]))
        else:
            raise IndexError(f"Invalid shape {y.shape}")

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        y, _, label, *_ = super().__getitem__(item)
        return self._pad(y.squeeze(0)), self.label_ids[label]


class SpeechCommands10(SpeechCommands):
    @property
    def classes(self) -> list[str]:
        return ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
