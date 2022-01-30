"""

    Datasets

    References:
        * https://github.com/HazyResearch/state-spaces

"""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F
from torchaudio.datasets import SPEECHCOMMANDS as _SpeechCommands  # noqa
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, Lambda, ToTensor

from experiments.data._transforms import build_permute_transform
from experiments.data._utils import download, untar

_DATASETS_DIRECTORY = Path("~/datasets")


class SequenceDataset:
    NAME: Optional[str] = None
    SAVE_NAME: Optional[str] = None
    class_names: Optional[list[str | int]] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @property
    def root_dir(self) -> Path:
        """Directory where data is stored."""
        name = self.SAVE_NAME or self.NAME
        if not isinstance(name, str):
            raise TypeError("`NAME` not set")

        path = _DATASETS_DIRECTORY.expanduser().joinpath(name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def classes(self) -> list[str | int]:
        """Names of all classes in the dataset."""
        if self.class_names:
            return self.class_names
        else:
            raise AttributeError("Class names not set")

    @property
    def n_classes(self) -> int:
        """Number of class_names in the dataset."""
        return len(self.classes)

    @property
    def channels(self) -> int:
        """Channels in the data, as returned by the dataset."""
        raise NotImplementedError()

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the data in the dataset."""
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        raise NotImplementedError()


class SMnistDataset(SequenceDataset, MNIST):
    NAME: str = "SMNIST"
    class_names: list[int] = list(range(10))

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            root=self.root_dir,
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
            root=self.root_dir,
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
    class_names: list[int] = list(range(10))

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            root=self.root_dir,
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
    class_names: list[str] = [
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
        super().__init__(root=self.root_dir, download=True, **kwargs)

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
    class_names: list[int] = [
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
    N_REPEATS: int = 4

    @property
    def n_classes(self) -> int:
        return self.N_REPEATS

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        y, _ = super().__getitem__(item)

        hot_idx = np.random.uniform(size=self.N_REPEATS) >= 0.5
        if not hot_idx.any():  # ensure at least one
            hot_idx[np.random.choice(self.N_REPEATS - 1)] = True

        label = -1
        chunks = list()
        for use_y in hot_idx:
            chunks.append(y if use_y else torch.zeros_like(y))
            label += int(use_y)
        return torch.cat(chunks), label

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.N_REPEATS * self.SEGMENT_SIZE,)  # noqa


class NSynthDataset(SequenceDataset):
    NAME: str = "NSYNTH"
    SEGMENT_SIZE: int = 64_000
    URLS: dict[str, str] = {
        "train": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",  # noqa
        "valid": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",  # noqa
        "test": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",  # noqa
    }

    def __init__(self, download: bool = True, verbose: bool = True) -> None:
        super().__init__()
        self.download = download
        self.verbose = verbose

        if download:
            self.fetch_data()

    def fetch_data(self, force: bool = False) -> None:
        for url in self.URLS.values():
            dirname, *_ = Path(url).stem.split(".")
            if force or not self.root_dir.joinpath(dirname).is_dir():
                untar(
                    download(url, dst=self.root_dir, verbose=self.verbose),
                    dst=self.root_dir,
                    delete_src=True,
                    verbose=self.verbose,
                )

    @cached_property
    def metadata(self) -> dict[str, dict[str, Any]]:
        metadata = dict()
        for path in self.root_dir.rglob("*.json"):
            with path.open("r") as f:
                payload = json.load(f)
                for v in payload.values():
                    v["split"] = path.parent.name.split("-")[-1]
                metadata |= payload
        return metadata

    @cached_property
    def classes(self) -> list[str | int]:
        return sorted({v["instrument_family_str"] for v in self.metadata.values()})

    @cached_property
    def files(self) -> list[Path]:
        return list(self.root_dir.rglob("*.wav"))

    @property
    def channels(self) -> int:
        return 1

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.SEGMENT_SIZE,)  # noqa

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        path = self.files[item]
        y, _ = torchaudio.load(path, normalize=True, channels_first=False)  # noqa
        label = self.metadata[path.stem]["instrument_family"]
        return y[: self.SEGMENT_SIZE, ...], label


class NSynthDatasetShort(NSynthDataset):
    NAME: str = "NSYNTH_SHORT"
    SAVE_NAME: str = "NSYNTH"
    SEGMENT_SIZE: int = 64_000 // 2


if __name__ == "__main__":
    smnist_wrapper = SMnistDataset()

    assert smnist_wrapper.NAME == "SMNIST"
    assert isinstance(smnist_wrapper.classes, list)
    assert smnist_wrapper.n_classes == 10
    assert smnist_wrapper.channels == 0
    assert smnist_wrapper.shape == (28 * 28,)
