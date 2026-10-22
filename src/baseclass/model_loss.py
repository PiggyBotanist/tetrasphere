from __future__ import annotations

# Standard Import
from abc import ABC, abstractmethod
import torch.nn as nn
from torch import Tensor

# =================================================================
# Base Loss Function for Generic Training Loop
# =================================================================
class BaseLoss(nn.Module, ABC):
    """Abstract interface every training loss implements."""
    @abstractmethod
    def forward(self, output: dict[str, Tensor], batch: dict[str, Tensor], epoch: int) -> tuple[Tensor, dict[str, float]]:
        """Compute the training loss and per-metric scalars for one batch."""
        ...

    @property
    @abstractmethod
    def config(self) -> dict[str, object]:
        """Return this loss' configuration for logging and checkpoints."""
        ...
