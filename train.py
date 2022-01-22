"""

    1D MNIST Classification

"""
from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import nn

from s4torch import S4Model


def _compute_acc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == labels).float().mean()


def _to_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        raise IndexError("1D input not supported")
    elif x.ndim == 2:
        return x.unsqueeze(-1)
    elif x.ndim == 3:
        return x
    elif x.ndim == 4:
        return x.flatten(2).transpose(-2, -1)
    else:
        raise IndexError(f"Expected 2D, 3D or 4D data, got {x.ndim}D")


class LighteningS4Model(pl.LightningModule):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes

        self.model = S4Model(
            d_input=1,
            d_model=128,
            d_output=n_classes,
            n_blocks=6,
            n=64,
            l_max=28 * 28,
            collapse=True,
            p_dropout=0.2,
        )
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
            _compute_acc(logits.detach(), labels),
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


if __name__ == "__main__":
    pass
