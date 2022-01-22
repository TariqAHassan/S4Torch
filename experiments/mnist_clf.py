"""

    1D MNIST Classification

"""
import pytorch_lightning as pl
import torch
from torch import nn
from torchvision.datasets import MNIST

from s4torch import S4Model


def _compute_acc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == labels).float().mean()


class LighteningS4Model(pl.LightningModule):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes

        self.model = S4Model(
            d_input=1,
            d_model=128,
            d_output=n_classes,
            n_blocks=3,  # ToDo: increase
            n=64,
            l_max=28 * 28,
            collapse=True,
            p_dropout=0.2,
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.model(u)

    def _step(self, batch: torch.Tensor, validation: bool) -> torch.Tensor:
        x, labels = batch
        logits = self.forward(x.flatten(1).unsqueeze(-1))
        self.log(
            "val_acc" if validation else "acc",
            _compute_acc(logits.detach(), labels),
            prog_bar=True,
        )
        loss = self.loss(logits, target=labels)
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._step(batch, validation=False)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
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
    from multiprocessing import cpu_count
    from pathlib import Path

    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import transforms

    DATA_DIRECTORY = Path("~/datasets").expanduser()
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    def make_dataloader(dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=16,
            shuffle=shuffle,
            num_workers=max(1, cpu_count() - 1),
            pin_memory=torch.cuda.is_available(),
        )

    # Dataset
    dataset = MNIST(
        str(DATA_DIRECTORY),
        download=True,
        transform=transforms.ToTensor(),
    )
    dataset_train, dataset_val = random_split(dataset, [55000, 5000])
    dl_train = make_dataloader(dataset_train, shuffle=True)
    dl_val = make_dataloader(dataset_val, shuffle=False)

    # Model
    pl_s4_model = LighteningS4Model(len(dataset.classes))

    # Trainer
    trainer = pl.Trainer(gpus=torch.cuda.device_count() or None)
    trainer.fit(pl_s4_model, dl_train, dl_val)
