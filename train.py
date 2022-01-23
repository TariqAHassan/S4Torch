"""

    S4 Training

"""
from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Type

import fire
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.seed import seed_everything
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from experiments.data.wrappers import DatasetWrapper
from s4torch import S4Model
from s4torch.aux.layers import TemporalAveragePooling, TemporalMaxPooling

_DATASETS = {d.NAME: d for d in DatasetWrapper.__subclasses__()}


def _get_dataset_wrapper(name: str) -> Type[DatasetWrapper]:
    try:
        return _DATASETS[name.upper()]
    except KeyError:
        raise KeyError(f"Unknown dataset '{name}'")


def _to_sequence(x: torch.Tensor) -> torch.Tensor:
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


def _compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == labels).float().mean()


def _parse_pooling(
    pooling: Optional[str],
) -> Optional[TemporalAveragePooling | TemporalMaxPooling]:
    if pooling is None:
        return None

    method, kernel_size = pooling.split("_")
    if method == "avg":
        return TemporalAveragePooling(int(kernel_size))
    elif method == "max":
        return TemporalMaxPooling(int(kernel_size))
    else:
        raise ValueError(f"Unsupported pooling method '{method}'")


class LighteningS4Model(pl.LightningModule):
    def __init__(
        self,
        model: S4Model,
        lr: float,
        lr_s4: float,
        weight_decay: float = 0.0,
        patience: int = 10,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_s4 = lr_s4
        self.weight_decay = weight_decay
        self.patience = patience

        self.loss = nn.CrossEntropyLoss()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.model(u)

    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        validation: bool,
    ) -> torch.Tensor:
        x, labels = batch
        logits = self.forward(_to_sequence(x))
        self.log(
            "val_acc" if validation else "acc",
            value=_compute_accuracy(logits.detach(), labels=labels),
            prog_bar=True,
        )
        loss = self.loss(logits, target=labels)
        return loss

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch, validation=False)

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = self._step(batch, validation=True)
        self.log("val_loss", value=loss)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.blocks.parameters(),
                    "lr": self.lr_s4,
                    "weight_decay": 0.0,
                },
                {"params": self.model.encoder.parameters()},
                {"params": self.model.decoder.parameters()},
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=self.patience),
                "monitor": "val_acc",
                "frequency": 1,
            },
        }


def main(
    dataset: str,
    batch_size: int,
    lr: float = 1e-2,
    lr_s4: float = 1e-3,
    weight_decay: float = 0.01,
    d_model: int = 128,
    n_blocks: int = 6,
    n: int = 64,
    p_dropout: float = 0.2,
    pooling: Optional[str] = None,
    norm_type: Optional[str] = "layer",
    swa: bool = False,
    accumulate_grad: int = 1,
    patience: int = 5,
    gpus: Optional[int] = None,
    val_prop: float = 0.1,
    seed: int = 1234,
) -> None:
    f"""Train S4 model.

    Args:
        dataset (str): datasets to train against. Available options:
            {', '.join([f"'{n}'" for n in sorted(_DATASETS)])}. Case-insensitive.
        batch_size (int): number of subprocesses to use for data loading
        lr (float): learning rate for parameters which do not belong to S4 blocks
        lr_s4 (float): learning rate for parameters which belong to S4 blocks
        weight_decay (float): weight decay to use with optimizer. (Ignored
            for parameters which belong to S4 blocks.)
        d_model (int): number of internal features
        n_blocks (int): number of S4 blocks to construct
        n (int): dimensionality of the state representation
        p_dropout (float): probability of elements being set to zero
        pooling (str, optional): pooling method to use. Options: ``None``, ``"max_KERNEL_SIZE"``,
            ``"avg_KERNEL_SIZE"``. Example: ``"avg_2"``.
        norm_type (str, optional): type of normalization to use.
            Options: ``batch``, ``layer``, ``None``.
        swa (bool): if ``True`` enable stochastic weight averaging.
        accumulate_grad (int): number of batches to accumulate gradient over.
        patience (int): number of epochs with no improvement to wait before
            reducing the learning rate
        gpus (int, optional): number of GPUs to use. If ``None``, use all available GPUs.
        val_prop (float): proportion of the data to use for validation
        seed (int): random seed for training

    Returns:
        None

    """
    seed_everything(seed, workers=True)
    dataset_wrapper = _get_dataset_wrapper(dataset.strip())(
        val_prop=val_prop,
        seed=seed,
    )  # noqa

    s4model = S4Model(
        d_input=max(1, dataset_wrapper.channels),
        d_model=d_model,
        d_output=dataset_wrapper.n_classes,
        n_blocks=n_blocks,
        n=n,
        l_max=math.prod(dataset_wrapper.shape),
        collapse=True,  # classification
        p_dropout=p_dropout,
        pooling=_parse_pooling(pooling),
        norm_type=norm_type,
    )

    pl_s4model = LighteningS4Model(
        s4model,
        lr=lr,
        lr_s4=lr_s4,
        weight_decay=weight_decay,
        patience=patience,
    )
    dl_train, dl_val = dataset_wrapper.get_dataloaders(batch_size)

    trainer = pl.Trainer(
        gpus=gpus or (torch.cuda.device_count() or None),
        stochastic_weight_avg=swa,
        accumulate_grad_batches=accumulate_grad,
    )
    trainer.fit(pl_s4model, dl_train, dl_val)


if __name__ == "__main__":
    fire.Fire(main)
