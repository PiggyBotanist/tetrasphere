from __future__ import annotations

# Standard Import
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.amp.grad_scaler import GradScaler

# =================================================================
# Standardized Checkpoint Setup
# =================================================================
def find(folder: Path, epoch: int | None) -> Path:
    """Path to the checkpoint under folder, or the latest one saved if epoch is None."""
    # Select a specific epoch if specified
    if epoch is not None:
        path = folder / f"epoch_{epoch}.pt"
        if not path.exists():
            raise FileNotFoundError(f"no checkpoint at {path}")
        return path
    
    # Auto detect the latest epoch with sorting
    paths = sorted(folder.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise FileNotFoundError(f"no checkpoints under {folder}")
    return paths[-1]

def load(path: Path, model: nn.Module, optimizer: Optimizer, scaler: GradScaler, device: torch.device) -> int:
    """Restore model, optimizer, and scaler state from path; return the saved epoch."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt.get("epoch", 0)

def load_weights(path: Path, model: nn.Module, device: torch.device) -> int:
    """Restore only model weights from path (no optimizer/scaler); return the saved epoch."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    return ckpt.get("epoch", 0)

def load_config(path: Path) -> dict[str, object]:
    """Read a checkpoint's saved epoch, metrics, model_config, and loss_config without loading weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "epoch":        ckpt.get("epoch", 0),
        "metrics":      ckpt.get("metrics", {}),
        "model_config": ckpt.get("model_config", {}),
        "loss_config":  ckpt.get("loss_config", {}),
    }

def save(path: Path, model: nn.Module, optimizer: Optimizer, scaler: GradScaler, epoch: int, metrics: dict[str, float], loss_config: dict[str, object], model_config: dict[str, object]) -> None:
    """Write model, optimizer, scaler state, metrics, and configs to path."""
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "metrics":         metrics,
        "loss_config":     loss_config,
        "model_config":    model_config,
    }, path)
