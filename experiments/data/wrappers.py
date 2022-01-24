"""

    Dataset Wrappers

"""
from __future__ import annotations

from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Optional, Type

import torch
from torch.cuda import is_available as cuda_available
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, Lambda, ToTensor

from experiments.data.datasets import SpeechCommands, SpeechCommands10
from experiments.data.transforms import build_permute_transform


def _train_val_split(
    dataset: Dataset,
    val_prop: float,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    if 0 < val_prop < 1:
        n_val = int(len(dataset) * val_prop)
    else:
        raise ValueError("`val_prop` expected to be on (0, 1)")
    return random_split(
        dataset=dataset,
        lengths=[len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )


class DatasetWrapper:
    """High-level dataset wrapper.

    Wrapper to standardize different datasets.

    Args:
        dataset (Type[Dataset]): a dataset with (data) `root` as the
            first argument
        val_prop (float): proportion of the data to use for validation
        seed (int): seed to use when splitting the data into training
            and validation datasets

    """

    NAME: Optional[str] = None

    def __init__(
        self,
        dataset: Type[Dataset],
        val_prop: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset(self.root_dir)  # noqa
        self.val_prop = val_prop
        self.seed = seed

        self.ds_train, self.ds_val = _train_val_split(
            self.dataset,
            val_prop=val_prop,
            seed=seed,
        )

    @property
    def root_dir(self) -> Path:
        """Directory where data is stored."""
        if not isinstance(self.NAME, str):
            raise TypeError("`NAME` not set")

        path = Path("~/datasets").expanduser().joinpath(self.NAME)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def classes(self) -> list[int | str]:
        """Name of each class in the dataset."""
        raise NotImplementedError()

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

    def make_dataloader(
        self,
        train: bool,
        batch_size: int,
        num_workers: int = max(1, cpu_count() - 1),
        pin_memory: Optional[bool] = True,
        **kwargs: Any,
    ) -> tuple[DataLoader, Dataset]:
        """Make a dataloaders.

        Args:
            train (bool): if ``True``, return the training dataloader.
                Otherwise, return the validation dataloader.
            batch_size (int): number of samples in each path
            num_workers (int): number of subprocesses to use for data loading
            pin_memory (bool, optional): if ``True`` tensors will be copied into
                CUDA pinned memory prior to being emitted. If ``None`` this will
                be determined automatically based on the availability of a device
                with CUDA support (GPU).
            **kwargs (Keyword Arguments): keyword arguments to pass to ``DataLoader()``

        Returns:
            dataloaders (tuple): a tuple of the form
                ``(train_dataloader, validation_dataloader)``

        """
        return DataLoader(
            dataset=self.ds_train if train else self.ds_val,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=cuda_available() if pin_memory is None else pin_memory,
            **kwargs,
        )


class SMnistWrapper(DatasetWrapper):
    """Sequential MNIST"""

    NAME: str = "SMNIST"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            partial(
                MNIST,
                download=True,
                transform=Compose([ToTensor(), Lambda(lambda t: t.flatten())]),
            ),
            **kwargs,
        )

    @property
    def classes(self) -> list[str]:
        return self.dataset.classes  # noqa

    @property
    def channels(self) -> int:
        return 1

    @property
    def shape(self) -> tuple[int, ...]:
        return (28 * 28,)  # noqa


class PMnistWrapper(DatasetWrapper):
    """Permuted MNIST."""

    NAME: str = "PMNIST"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            partial(
                MNIST,
                download=True,
                transform=Compose([ToTensor(), build_permute_transform((28 * 28,))]),
            ),
            **kwargs,
        )

    @property
    def classes(self) -> list[str]:
        return self.dataset.classes  # noqa

    @property
    def channels(self) -> int:
        return 0

    @property
    def shape(self) -> tuple[int, ...]:
        return (28 * 28,)  # noqa


class CIFAR10Wrapper(DatasetWrapper):
    """CIFAR10."""

    NAME: str = "CIFAR10"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            partial(CIFAR10, download=True, transform=ToTensor()),
            **kwargs,
        )

    @property
    def classes(self) -> list[int]:
        return list(range(10))

    @property
    def channels(self) -> int:
        return 3

    @property
    def shape(self) -> tuple[int, ...]:
        return 32, 32


class SpeechCommandWrapper(DatasetWrapper):
    """Speech Commands (Full)."""

    NAME: str = "SPEECHCOMMANDS"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            dataset=partial(SpeechCommands, download=True),
            **kwargs,
        )

    @property
    def classes(self) -> list[str]:
        return self.dataset.classes  # noqa

    @property
    def channels(self) -> int:
        return 0

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.dataset.SEGMENT_SIZE,)  # noqa


class SpeechCommand10Wrapper(DatasetWrapper):
    """Speech Commands (Subset)."""

    NAME: str = "SPEECHCOMMANDS10"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            dataset=partial(SpeechCommands10, download=True),
            **kwargs,
        )

    @property
    def classes(self) -> list[str]:
        return self.dataset.classes  # noqa

    @property
    def channels(self) -> int:
        return 0

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.dataset.SEGMENT_SIZE,)  # noqa


if __name__ == "__main__":
    smnist_wrapper = SMnistWrapper()
    train_dl = smnist_wrapper.make_dataloader(train=True, batch_size=1)
    val_dl = smnist_wrapper.make_dataloader(train=False, batch_size=1)

    assert smnist_wrapper.NAME == "SMNIST"
    assert isinstance(train_dl, DataLoader)
    assert isinstance(val_dl, DataLoader)
    assert isinstance(smnist_wrapper.classes, list)
    assert smnist_wrapper.n_classes == 10
    assert smnist_wrapper.channels == 1
    assert smnist_wrapper.shape == (28 * 28,)
