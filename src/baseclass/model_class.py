from __future__ import annotations

# Standard Import
from abc import ABC, abstractmethod
import torch.nn as nn
from torch import Tensor

# =================================================================
# Base Model Structure for Generic Training Loop
# =================================================================
class BaseModel(nn.Module, ABC):
    """Abstract interface every trainable model implements."""
    @abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run one forward pass and return named output tensors."""
        ...

    @property
    @abstractmethod
    def config(self) -> dict[str, object]:
        """Return model configurations"""
        ...