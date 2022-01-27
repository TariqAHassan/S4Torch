"""

    Datasets

    References:
        * https://github.com/HazyResearch/state-spaces

"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.nn import functional as F
from torchaudio.datasets import SPEECHCOMMANDS as _SpeechCommands  # noqa
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, Lambda, ToTensor

from experiments.data.transforms import build_permute_transform


class SequenceDataset:
    NAME: Optional[str] = None
    SAVE_NAME: Optional[str] = None
    classes: list[str | int]

    def __init__(
        self,
        val_prop: float = 0.1,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(self.root_dir, **kwargs)
        self.val_prop = val_prop
        self.seed = seed

    @property
    def root_dir(self) -> Path:
        """Directory where data is stored."""
        name = self.SAVE_NAME or self.NAME
        if not isinstance(name, str):
            raise TypeError("`NAME` not set")

        path = Path("~/datasets").expanduser().joinpath(name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def n_classes(self) -> int:
        """Number of classes in the dataset."""
        return len(self.classes)

    @property
    def channels(self) -> int:
        """Channels in the data, as returned by the dataset."""
        raise NotImplementedError()

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the data in the dataset."""
        raise NotImplementedError()


class SMnistDataset(SequenceDataset, MNIST):
    NAME: str = "SMNIST"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            download=True,
            transform=Compose([ToTensor(), Lambda(lambda t: t.flatten())]),
            **kwargs,
        )

    @property
    def channels(self) -> int:
        return 0

    @property
    def shape(self) -> tuple[int, ...]:
        return (28 * 28,)  # noqa


class PMnistDataset(SequenceDataset, MNIST):
    NAME: str = "PMNIST"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            download=True,
            transform=Compose([ToTensor(), build_permute_transform((28 * 28,))]),
            **kwargs,
        )

    @property
    def channels(self) -> int:
        return 0

    @property
    def shape(self) -> tuple[int, ...]:
        return (28 * 28,)  # noqa


class SCIFAR10Dataset(SequenceDataset, CIFAR10):
    NAME: str = "SCIFAR10"
    classes = list(range(10))

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            download=True,
            transform=Compose(
                [
                    ToTensor(),
                    Lambda(lambda t: t.flatten(1).transpose(-2, -1)),
                ]
            ),
            **kwargs,
        )

    @property
    def channels(self) -> int:
        return 3

    @property
    def shape(self) -> tuple[int, ...]:
        return (32 * 32,)  # noqa


class SpeechCommands(SequenceDataset, _SpeechCommands):
    NAME: str = "SPEECH_COMMANDS"
    SEGMENT_SIZE: int = 16_000
    classes = [
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(download=True, **kwargs)

        self.label_ids = {l: e for e, l in enumerate(self.classes)}
        self._walker = [i for i in self._walker if Path(i).parent.name in self.classes]

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

    @property
    def channels(self) -> int:
        return 0

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.SEGMENT_SIZE,)  # noqa


class SpeechCommands10(SpeechCommands):
    NAME: str = "SPEECH_COMMANDS_10"
    SAVE_NAME = "SPEECH_COMMANDS"
    classes = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ]


class RepeatedSpeechCommands10(SpeechCommands10):
    NAME: str = "REPEATED_SPEECH_COMMANDS10"
    SAVE_NAME = "SPEECH_COMMANDS"
    N_REPEATS: int = 4
    classes: list[int] = list(range(N_REPEATS - 1))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        y, _ = super().__getitem__(item)

        hot_idx = np.random.uniform(size=self.N_REPEATS) >= 0.5
        if not hot_idx.any():  # ensure at least one
            hot_idx[np.random.choice(self.N_REPEATS - 1)] = True

        label = 0
        chunks = list()
        for use_y in hot_idx:
            chunks.append(y if use_y else torch.zeros_like(y))
            label += int(use_y)
        return torch.cat(chunks), label

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.N_REPEATS * self.SEGMENT_SIZE,)  # noqa


if __name__ == "__main__":
    smnist_wrapper = SMnistDataset()

    assert smnist_wrapper.NAME == "SMNIST"
    assert isinstance(smnist_wrapper.classes, list)
    assert smnist_wrapper.n_classes == 10
    assert smnist_wrapper.channels == 0
    assert smnist_wrapper.shape == (28 * 28,)
