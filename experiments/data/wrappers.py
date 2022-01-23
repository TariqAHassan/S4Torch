"""

    Dataset Wrappers

"""
from __future__ import annotations

from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Optional, Type

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.cuda import is_available as cuda_available

from experiments.data.datasets import SpeechCommands, SpeechCommands10


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

        self.dataset_train, self.dataset_val = _train_val_split(
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

    def get_dataloaders(
        self,
        batch_size: int,
        num_workers: int = max(1, cpu_count() - 1),
        pin_memory: Optional[bool] = True,
        **kwargs: Any,
    ) -> tuple[DataLoader, Dataset]:
        """Get training and validation dataloaders.

        Args:
            batch_size (int): number of samples in each path
            num_workers (int): number of subprocesses to use for data loading
            pin_memory (bool, optional): if ``True`` tensors will be copied into
                CUDA pinned memory prior to being emitted. If ``None`` this will
                be determined automatically based on the availability of a device
                with CUDA support (GPU).
            **kwargs (Keyword Arguments): keyword arguments to pass
                to ``DataLoader()``

        Returns:
            dataloaders (tuple): a tuple of the form
                ``(train_dataloader, validation_dataloader)``

        """

        def make_dataloader(dataset: Dataset, shuffle: bool) -> DataLoader:
            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=cuda_available() if pin_memory is None else pin_memory,
                **kwargs,
            )

        return (
            make_dataloader(self.dataset_train, shuffle=True),
            make_dataloader(self.dataset_val, shuffle=False),
        )


class MnistWrapper(DatasetWrapper):
    NAME: str = "MNIST"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            partial(MNIST, download=True, transform=transforms.ToTensor()),
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
        return 28, 28


class CIFAR10Wrapper(DatasetWrapper):
    NAME: str = "CIFAR10"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            partial(CIFAR10, download=True, transform=transforms.ToTensor()),
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
    mnist_wrapper = MnistWrapper()
    train_dl, val_dl = mnist_wrapper.get_dataloaders(8)

    assert mnist_wrapper.NAME == "MNIST"
    assert isinstance(train_dl, DataLoader)
    assert isinstance(val_dl, DataLoader)
    assert isinstance(mnist_wrapper.classes, list)
    assert mnist_wrapper.n_classes == 10
    assert mnist_wrapper.channels == 1
    assert mnist_wrapper.shape == (28, 28)
