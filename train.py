"""

    S4 Training

"""
import math
from typing import Optional, Tuple

import fire
import pytorch_lightning as pl
import torch
from torch import nn

from experiments.data.wrappers import DatasetWrapper
from s4torch import S4Model

_DATASETS = {d.NAME: d for d in DatasetWrapper.__subclasses__()}


def _get_dataset_wrapper(name: str) -> DatasetWrapper:
    if name in _DATASETS:
        return _DATASETS[name]
    else:
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


class LighteningS4Model(pl.LightningModule):
    def __init__(self, model: S4Model) -> None:
        super().__init__()
        self.model = model

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
            _compute_accuracy(logits.detach(), labels=labels),
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
        return self._step(batch, validation=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [
                {
                    "params": self.model.blocks.parameters(),
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                },
                {"params": self.model.encoder.parameters()},
                {"params": self.model.decoder.parameters()},
            ],
            lr=1e-3,
            weight_decay=0.01,
        )


def main(
    dataset: str,
    d_model: int = 128,
    n_blocks: int = 6,
    n: int = 64,
    p_dropout: float = 0.2,
    train_p: bool = False,
    train_q: bool = False,
    train_lambda: bool = False,
    gpus: Optional[int] = None,
) -> None:
    f"""Train S4 model.

    Args:
        dataset (str): datasets to train against. Options: {', '.join(sorted(_DATASETS))}.
        d_model (int): number of internal features
        n_blocks (int): number of S4 blocks to construct
        n (int): dimensionality of the state representation
        p_dropout (float): probability of elements being set to zero
        train_p (bool): if ``True`` train the ``p`` tensor in each S4 block
        train_q (bool): if ``True`` train the ``q`` tensor in each S4 block
        train_lambda (bool): if ``True`` train the ``lambda`` tensor in each S4 block
        gpus (int): number of GPUs to use. If ``None``, use all available GPUs.

    Returns:
        None

    """
    dataset_wrapper = _get_dataset_wrapper(dataset)

    s4model = S4Model(
        d_input=dataset_wrapper.channels,
        d_model=d_model,
        d_output=dataset_wrapper.n_classes,
        n_blocks=n_blocks,
        n=n,
        l_max=math.prod(dataset_wrapper.shape),
        collapse=True,  # classification
        p_dropout=p_dropout,
        train_p=train_p,
        train_q=train_q,
        train_lambda=train_lambda,
    )

    pl_s4_model = LighteningS4Model(s4model)
    trainer = pl.Trainer(gpus=gpus or (torch.cuda.device_count() or None))
    trainer.fit(pl_s4_model, dl_train, dl_val)


if __name__ == "__main__":
    fire.Fire(main)
