"""

    Metrics

"""
import torch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == labels).float().mean()
