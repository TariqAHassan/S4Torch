"""

    S4 Training

"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional, Tuple, Type

import fire
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from experiments.data.wrappers import DatasetWrapper
from experiments.metrics import compute_accuracy
from experiments.utils import OutputPaths, to_sequence
from s4torch import S4Model
from s4torch.aux.layers import TemporalAvgPooling, TemporalMaxPooling

_DATASETS = {d.NAME: d for d in DatasetWrapper.__subclasses__()}


def _get_ds_wrapper(name: str) -> Type[DatasetWrapper]:
    try:
        return _DATASETS[name.upper()]
    except KeyError:
        raise KeyError(f"Unknown dataset '{name}'")


def _parse_pooling(
    pooling: Optional[str],
) -> Optional[TemporalAvgPooling | TemporalMaxPooling]:
    if pooling is None:
        return None
    elif pooling.count("_") != 1:
        raise ValueError(f"`pooling` expected one underscore, got '{pooling}'")

    method, digit = pooling.split("_")
    kernel_size = int(digit)
    if method == "avg":
        return TemporalAvgPooling(kernel_size)
    elif method == "max":
        return TemporalMaxPooling(kernel_size)
    else:
        raise ValueError(f"Unsupported pooling method '{method}'")


class LighteningS4Model(pl.LightningModule):
    def __init__(
        self,
        model: S4Model,
        lr: float,
        lr_s4: float,
        min_lr: float = 1e-6,
        weight_decay: float = 0.0,
        patience: int = 5,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_s4 = lr_s4
        self.min_lr = min_lr
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
        logits = self.forward(to_sequence(x))
        self.log(
            "val_acc" if validation else "acc",
            value=compute_accuracy(logits.detach(), labels=labels),
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
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=self.patience,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "frequency": 1,
            },
        }


def main(
    # Dataset
    dataset: str,
    batch_size: int,
    val_prop: float = 0.1,
    # Model
    d_model: int = 128,
    n_blocks: int = 6,
    s4_n: int = 64,
    p_dropout: float = 0.2,
    pooling: Optional[str] = None,
    norm_type: Optional[str] = "layer",
    # Training
    max_epochs: Optional[int] = None,
    lr: float = 1e-2,
    lr_s4: float = 1e-3,
    min_lr: float = 1e-6,
    weight_decay: float = 0.01,
    swa: bool = False,
    accumulate_grad: int = 1,
    patience: int = 5,
    gpus: int = -1,
    # Auxiliary
    output_dir: str = "~/s4-output",
    save_top_k: int = 0,
    seed: int = 1234,
) -> None:
    f"""Train a S4 model.

    Perform S4 model training using an ``AdamW`` optimizer and
    ``ReduceLROnPlateau`` learning rate scheduler.

    Args:
        dataset (str): datasets to train against. Available options:
            {', '.join([f"'{n}'" for n in sorted(_DATASETS)])}. Case-insensitive.
        batch_size (int): number of subprocesses to use for data loading
        val_prop (float): proportion of the data to use for validation
        d_model (int): number of internal features
        n_blocks (int): number of S4 blocks to construct
        s4_n (int): dimensionality of the state representation
        p_dropout (float): probability of elements being set to zero
        pooling (str, optional): pooling method to use. Options: ``None``, ``"max_KERNEL_SIZE"``,
            ``"avg_KERNEL_SIZE"``. Example: ``"avg_2"``.
        norm_type (str, optional): type of normalization to use.
            Options: ``batch``, ``layer``, ``None``.
        max_epochs (int, optional): maximum number of epochs to train for
        lr (float): learning rate for parameters which do not belong to S4 blocks
        lr_s4 (float): learning rate for parameters which belong to S4 blocks
        min_lr (float): minimum learning rate to permit ``ReduceLROnPlateau`` to use.
        weight_decay (float): weight decay to use with optimizer. (Ignored
            for parameters which belong to S4 blocks.)
        swa (bool): if ``True`` enable stochastic weight averaging.
        accumulate_grad (int): number of batches to accumulate gradient over.
        patience (int): number of epochs with no improvement to wait before
            reducing the learning rate
        gpus (int): number of GPUs to use. If ``-1``, use all available GPUs.
        output_dir (str): directory where output (logs and checkpoints) will be saved.
        save_top_k (int): save top k models, as determined by the ``"val_acc"``
            metric. (Defaults to ``0``, which disables model saving.)
        seed (int): random seed for training

    Returns:
        None

    """
    seed_everything(seed, workers=True)
    run_name = f"s4-model-{datetime.utcnow().isoformat()}"
    output_paths = OutputPaths(output_dir, run_name=run_name)

    ds_wrapper = _get_ds_wrapper(dataset.strip())(val_prop=val_prop, seed=seed)  # noqa
    dl_train, dl_val = ds_wrapper.get_dataloaders(batch_size)

    s4model = S4Model(
        d_input=max(1, ds_wrapper.channels),
        d_model=d_model,
        d_output=ds_wrapper.n_classes,
        n_blocks=n_blocks,
        n=s4_n,
        l_max=math.prod(ds_wrapper.shape),
        collapse=True,  # classification
        p_dropout=p_dropout,
        pooling=_parse_pooling(pooling),
        norm_type=norm_type,
    )

    pl.Trainer(
        max_epochs=max_epochs,
        gpus=(torch.cuda.device_count() if gpus == -1 else gpus) or None,
        stochastic_weight_avg=swa,
        accumulate_grad_batches=accumulate_grad,
        logger=TensorBoardLogger(output_paths.logs, name=run_name),
        callbacks=ModelCheckpoint(
            dirpath=output_paths.checkpoints,
            filename=run_name + "-{epoch:02d}-{val_acc:.2f}",
            monitor="val_acc",
            save_top_k=save_top_k,
        ),
    ).fit(
        LighteningS4Model(
            s4model,
            lr=lr,
            lr_s4=lr_s4,
            min_lr=min_lr,
            weight_decay=weight_decay,
            patience=patience,
        ),
        train_dataloaders=dl_train,
        val_dataloaders=dl_val,
    )


if __name__ == "__main__":
    fire.Fire(main)
