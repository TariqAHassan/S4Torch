"""

    S4 Training

"""
from __future__ import annotations

import math
from argparse import Namespace
from datetime import datetime
from typing import Any, Optional, Tuple, Type

import fire
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from experiments.data.wrappers import DatasetWrapper
from experiments.metrics import compute_accuracy
from experiments.utils import OutputPaths, parse_params_in_s4blocks, to_sequence
from s4torch import S4Model

_DATASET_WRAPPERS = {d.NAME: d for d in DatasetWrapper.__subclasses__()}


def _get_ds_wrapper(name: str) -> Type[DatasetWrapper]:
    try:
        return _DATASET_WRAPPERS[name.upper()]
    except KeyError:
        raise KeyError(f"Unknown dataset '{name}'")


def _parse_pooling(pooling: Optional[str]) -> Optional[nn.AvgPool1d | nn.MaxPool1d]:
    if pooling is None:
        return None
    elif pooling.count("_") != 1:
        raise ValueError(f"Expected one underscore, got '{pooling}'")

    method, digit = pooling.split("_")
    kernel_size = int(digit)
    if method == "avg":
        return nn.AvgPool1d(kernel_size)
    elif method == "max":
        return nn.MaxPool1d(kernel_size)
    else:
        raise ValueError(f"Unsupported pooling method '{method}'")


class LighteningS4Model(pl.LightningModule):
    def __init__(
        self,
        model: S4Model,
        hparams: Namespace,
        ds_wrapper: DatasetWrapper,
    ) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters(
            hparams,
            ignore=("model", "hparams", "ds_wrapper"),
        )
        self.ds_wrapper = ds_wrapper

        self.loss = nn.CrossEntropyLoss()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.model(u)

    def _step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, labels = batch
        logits = self.model(to_sequence(x))
        acc = compute_accuracy(logits.detach(), labels=labels)
        loss = self.loss(logits, target=labels)
        return loss, acc

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, acc = self._step(batch)
        self.log("loss", value=loss)
        self.log("acc", value=acc, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        losses, accs = map(torch.stack, zip(*outputs))
        self.log("val_loss", value=losses.mean())
        self.log("val_acc", value=accs.mean(), prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        s4layer_params, other_params = parse_params_in_s4blocks(self.model.blocks)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": s4layer_params,
                    "lr": self.hparams.lr_s4,
                    "weight_decay": 0.0,
                },
                {"params": other_params},
                {"params": self.model.encoder.parameters()},
                {"params": self.model.decoder.parameters()},
            ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=self.hparams.patience,
            min_lr=self.hparams.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_acc"},
        }

    def train_dataloader(self) -> DataLoader:
        return self.ds_wrapper.make_dataloader(
            train=True,
            batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        return self.ds_wrapper.make_dataloader(
            train=False,
            batch_size=self.hparams.batch_size,
        )


def main(
    # Dataset
    dataset: str,
    batch_size: int = -1,
    val_prop: float = 0.1,
    # Model
    d_model: int = 128,
    n_blocks: int = 6,
    s4_n: int = 64,
    p_dropout: float = 0.2,
    norm_strategy: str = "post",
    norm_type: Optional[str] = "layer",
    pooling: Optional[str] = None,
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
            {', '.join([f"'{n}'" for n in sorted(_DATASET_WRAPPERS)])}.
            Case-insensitive.
        batch_size (int): number of subprocesses to use for data loading.
            If ``batch_size=-1`` the largest possible batch size will be used.
        val_prop (float): proportion of the data to use for validation
        d_model (int): number of internal features
        n_blocks (int): number of S4 blocks to construct
        s4_n (int): dimensionality of the state representation
        p_dropout (float): probability of elements being set to zero
        norm_strategy (str): position of normalization relative to ``S4Layer()``.
            Must be "pre" (before ``S4Layer()``), "post" (after ``S4Layer()``)
            or "both" (before and after ``S4Layer()``).
        norm_type (str, optional): type of normalization to use.
            Options: ``batch``, ``layer``, ``None``.
        pooling (str, optional): pooling method to use. Options: ``None``,
            ``avg_KERNEL_SIZE``, ``max_KERNEL_SIZE``. Example: ``avg_2``.
        max_epochs (int, optional): maximum number of epochs to train for
        lr (float): learning rate for parameters which do not belong to S4 blocks
        lr_s4 (float): learning rate for parameters which belong to S4 blocks
        min_lr (float): minimum learning rate to permit ``ReduceLROnPlateau`` to use
        weight_decay (float): weight decay to use with optimizer. (Ignored
            for parameters which belong to S4 blocks.)
        swa (bool): if ``True`` enable stochastic weight averaging
        accumulate_grad (int): number of batches to accumulate gradient over
        patience (int): number of epochs with no improvement to wait before
            reducing the learning rate
        gpus (int): number of GPUs to use. If ``-1``, use all available GPUs.
        output_dir (str): directory where output (logs and checkpoints) will be saved
        save_top_k (int): save top k models, as determined by the ``val_acc``
            metric. (Defaults to ``0``, which disables model saving.)
        seed (int): random seed for training

    Returns:
        None

    """
    hparams = Namespace(**locals())
    seed_everything(seed, workers=True)
    run_name = f"s4-model-{datetime.utcnow().isoformat()}"
    output_paths = OutputPaths(output_dir, run_name=run_name)
    auto_scale_batch_size = batch_size == -1
    ds_wrapper = _get_ds_wrapper(dataset.strip())(val_prop=val_prop, seed=seed)  # noqa

    pl_model = LighteningS4Model(
        S4Model(
            d_input=max(1, ds_wrapper.channels),
            d_model=d_model,
            d_output=ds_wrapper.n_classes,
            n_blocks=n_blocks,
            n=s4_n,
            l_max=math.prod(ds_wrapper.shape),
            collapse=True,  # classification
            p_dropout=p_dropout,
            pooling=_parse_pooling(pooling),
            norm_strategy=norm_strategy,
            norm_type=norm_type,
        ),
        hparams=hparams,
        ds_wrapper=ds_wrapper,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=(torch.cuda.device_count() if gpus == -1 else gpus) or None,
        stochastic_weight_avg=swa,
        accumulate_grad_batches=accumulate_grad,
        auto_scale_batch_size=auto_scale_batch_size,
        logger=TensorBoardLogger(output_paths.logs, name=run_name),
        callbacks=ModelCheckpoint(
            dirpath=output_paths.checkpoints,
            filename=f"{run_name}-{'{epoch:02d}-{val_acc:.2f}'}",
            monitor="val_acc",
            save_top_k=save_top_k,
        ),
    )
    if auto_scale_batch_size:
        trainer.tune(pl_model)
    trainer.fit(pl_model)


if __name__ == "__main__":
    fire.Fire(main)
