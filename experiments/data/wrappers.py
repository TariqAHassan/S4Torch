"""

    Dataset Wrappers

"""
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Type

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from experiments.datasets import SpeechCommandsDataset10


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
    def __init__(
        self,
        dataset: Type[Dataset],
        val_prop: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset(self.root_dir)
        self.val_prop = val_prop
        self.seed = seed

        self.dataset_train, self.dataset_val = _train_val_split(
            self.dataset,  # noqa
            val_prop=val_prop,
            seed=seed,
        )

    @property
    def name(self) -> str:
        return self.dataset.__class__.__name__

    @property
    def root_dir(self) -> Path:
        path = Path("~/datasets").expanduser().joinpath(self.name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def classes(self) -> list[str]:
        raise NotImplementedError()

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError()

    def get_dataloaders(
        self,
        batch_size: int,
        num_workers: int = max(1, cpu_count() - 1),
        pin_memory: bool = True,
        **kwargs: Any,
    ) -> tuple[DataLoader, Dataset]:
        def make_dataloader(dataset: Dataset, shuffle: bool) -> DataLoader:
            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory or torch.cuda.is_available(),
                **kwargs,
            )

        return (
            make_dataloader(self.dataset_train, shuffle=True),
            make_dataloader(self.dataset_val, shuffle=False),
        )


class MnistWrapper(DatasetWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            partial(MNIST, download=True, transform=transforms.ToTensor()),
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "MNIST"

    @property
    def classes(self) -> list[str]:
        return self.dataset.classes

    @property
    def shape(self) -> tuple[int, ...]:
        return 1, 28, 28


class SpeechCommand10Wrapper(DatasetWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            dataset=partial(SpeechCommandsDataset10, download=True),
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "SPEECHCOMMANDS"

    @property
    def classes(self) -> list[str]:
        return self.dataset.classes

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.dataset.SEGMENT_SIZE,)  # noqa


if __name__ == "__main__":
    mnist_wrapper = MnistWrapper()
    train_dl, val_dl = self.get_dataloaders(8)

    assert isinstance(train_dl, DataLoader)
    assert isinstance(val_dl, DataLoader)
    assert mnist_wrapper.name == "MNIST"
    assert isinstance(mnist_wrapper.classes, list)
    assert mnist_wrapper.n_classes == 10
    assert mnist_wrapper.shape == (1, 28, 28)
