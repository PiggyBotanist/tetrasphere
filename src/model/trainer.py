from __future__ import annotations

# Standard Import
import math
import wandb
import logging
from tqdm import tqdm
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

# Custom Import
from src.baseclass import BaseModel, BaseLoss
from src.model import checkpoint

# =================================================================
# Packaged Generic Trainer
# =================================================================
@dataclass
class TrainConfig:
    """Trainer's tunable parameters."""
    # Labels
    project: str = "random_project"
    label: str   = "random_label"
    
    # Parameters
    device:         str | None = None  
    epochs:         int = 20
    scheduler:      LRScheduler | None = None 
    precision:      str = "bf16"         # "fp32" | "fp16" | "bf16"
    grad_clip:      float | None = 1.0   # max gradient norm; None disables clipping
    model_config:           dict[str, object] = field(default_factory=dict)  # saved into every checkpoint
    
    # Checkpoints / Logging
    use_wandb:              bool = True
    checkpoint_dir:         Path | None = None
    save_every:             int | None = None
    
    # stepped once per epoch
    # Expect <model_output, batch_data, epoch #, logger>
    val_hook:       Callable[[dict[str, Tensor], dict[str, Tensor], int, logging.Logger], dict[str,float]] | None = None
    
class Trainer:
    """Generic trainer with BaseModel + BaseLoss."""
    def __init__(self, model: BaseModel, loss_fn: BaseLoss, optimizer: Optimizer, logger: logging.Logger, config: TrainConfig = TrainConfig()) -> None:
        self.config = config
        self.logger = logger
        
        # Model
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device      = torch.device(device)
        self.model        = model.to(device)
        self.loss_fn      = loss_fn
        self.optimizer    = optimizer

        # Automatic mixed precision (AMP)
        self.amp_enable = config.precision in ("fp16", "bf16") and self._device.type == "cuda"
        self.amp_dtype   = torch.float16 if config.precision == "fp16" else torch.bfloat16
        self.grad_scaler = GradScaler(enabled=(config.precision == "fp16" and self.amp_enable))

        # Initialize checkpoint folder
        self.folder_ckpt  = Path(config.checkpoint_dir) if config.checkpoint_dir else None
        if self.folder_ckpt is not None:
            self.folder_ckpt.mkdir(parents=True, exist_ok=True)
            
        # Wandb setup
        if self.config.use_wandb:
            print(self.config.project)
            print(self.config.label)
            wandb.init(project=self.config.project, name=self.config.label, reinit=True)

    # -----------------------------------------------------------------
    # Main function (training loop)
    # -----------------------------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict[str, list[float]]:
        """Train for self._epochs, validating and checkpointing each epoch; returns metric history."""
        history: dict[str, list[float]] = {}
        for epoch in range(self.config.epochs):
            # Train + Validation (optional)
            train_metrics = train_epoch(self.model, train_loader, self.loss_fn, self.optimizer, self.grad_scaler, self._device, self.amp_dtype, self.amp_enable, self.config.grad_clip, epoch, self.config.epochs)
            val_metrics   = eval_epoch(self.model, val_loader, self.loss_fn, self._device, self.amp_dtype, self.amp_enable, epoch, self.logger, self.config.val_hook) if val_loader else {}

            # Log per epoch result for train + validation
            lr = self.optimizer.param_groups[0]["lr"]
            log_epoch(self.logger, train_metrics, val_metrics, epoch, lr, self.config.use_wandb)

            # Append epoch metric to history
            for k, v in train_metrics.items():
                history.setdefault(f"train_{k}", []).append(v)
            for k, v in val_metrics.items():
                history.setdefault(f"val_{k}", []).append(v)

            # Step scheduler once per epoch
            if self.config.scheduler is not None:
                if isinstance(self.config.scheduler, ReduceLROnPlateau):
                    self.config.scheduler.step((val_metrics or train_metrics).get("loss", 0.0))
                else:
                    self.config.scheduler.step()

            # Save checkpoint
            if self.config.save_every and (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint_(epoch, {**train_metrics, **val_metrics}, f"epoch_{epoch + 1}")

        return history

    def load_checkpoint(self, path: Path) -> int:
        """Restore model/optimizer/scaler state from path; return the saved epoch number."""
        epoch = checkpoint.load(path, self.model, self.optimizer, self.grad_scaler, self._device)
        self.logger.info("Checkpoint loaded <- %s (epoch %d)", path, epoch)
        return epoch

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Run one no-grad pass over loader with the model's current weights; returns its metrics."""
        return eval_epoch(self.model, loader, self.loss_fn, self._device, self.amp_dtype, self.amp_enable, self.config.epochs, self.logger, self.config.val_hook)

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------
    def _save_checkpoint_(self, epoch: int, metrics: dict[str, float], tag: str) -> None:
        """Write a checkpoint under checkpoint_dir/{tag}.pt, if checkpointing is enabled."""
        if self.folder_ckpt is None:
            return
        path = self.folder_ckpt / f"{tag}.pt"
        checkpoint.save(path, self.model, self.optimizer, self.grad_scaler, epoch + 1, metrics, self.loss_fn.config, self.config.model_config)
        self.logger.info("Checkpoint -> %s", path)

# =================================================================
# Train / Validate per epoch
# =================================================================
def train_epoch(
    model:        BaseModel,
    loader:       DataLoader,
    loss_fn:      BaseLoss,
    optimizer:    Optimizer,
    scaler:       GradScaler,
    device:       torch.device,
    amp_dtype:    torch.dtype,
    amp_enabled:  bool,
    grad_clip:    float | None,
    epoch:        int,
    total_epochs: int,
) -> dict[str, float]:
    """Run one training epoch; returns each loss metric's batch-size-weighted mean."""
    model.train()
    accum:  dict[str, float] = {}
    counts: dict[str, int]   = {}

    pbar = tqdm(loader, desc=f"Train {epoch + 1}/{total_epochs}", leave=False)
    for batch in pbar:
        # Obtain batch (mount to GPU)
        batch = _to_device_(batch, device)
        bs = _batch_size_(batch)

        # Compute loss
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output        = model(batch)
            loss, metrics = loss_fn(output, batch, epoch)

        # Backward propagation
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Update cumulative history and update tqdm
        _accumulate_(accum, counts, metrics, bs)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, float)})
    return _finalize_(accum, counts)

def eval_epoch(
    model:       BaseModel,
    loader:      DataLoader,
    loss_fn:     BaseLoss,
    device:      torch.device,
    amp_dtype:   torch.dtype,
    amp_enabled: bool,
    epoch:       int,
    logger:      logging.Logger,
    val_hook:    Callable[[dict[str, Tensor], dict[str, Tensor], int, logging.Logger], dict[str,float]] | None = None
) -> dict[str, float]:
    """Run one no-grad validation epoch."""
    # Early return if validation dataset or procedure is not provided
    if loader is None or val_hook is None: return {}
    model.eval()
    accum:          dict[str, float]         = {}
    counts:         dict[str, int]           = {}

    with torch.no_grad():
        for batch in loader:
            # Obtain batch (mount to GPU)
            batch = _to_device_(batch, device)
            bs = _batch_size_(batch)

            # Forward pass
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(batch) 
            metrics = val_hook(output, batch, epoch, logger)
            _accumulate_(accum, counts, metrics, bs)

    return _finalize_(accum, counts)

def log_epoch(logger: logging.Logger, train_metrics: dict[str, float], val_metrics: dict[str, float], epoch: int, lr: float, use_wandb: bool) -> None:
    """Log one epoch's train/val metrics; also pushes to wandb if enabled and running."""
    parts  = [f"train/{k}={v:.4f}" for k, v in train_metrics.items()]
    parts += [f"val/{k}={v:.4f}"   for k, v in val_metrics.items()]
    parts.append(f"lr={lr:.2e}")
    logger.info("Epoch %4d | %s", epoch + 1, "  ".join(parts))

    if use_wandb:
        wandb.log({"epoch": epoch + 1, "lr":    lr, **{f"train/{k}": v for k, v in train_metrics.items()}, **{f"val/{k}":   v for k, v in val_metrics.items()}})
    return

# =================================================================
# Helper functions
# =================================================================
def _to_device_(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    """Move every tensor value in batch onto device; other values pass through."""
    return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

def _batch_size_(batch: dict[str, Tensor]) -> int:
    """Size of batch's leading dimension, from its first tensor value."""
    return next(v.size(0) for v in batch.values() if isinstance(v, Tensor))

def _accumulate_(accum: dict[str, float], counts: dict[str, int], metrics: dict[str, float], bs: int) -> None:
    """Add one batch's metrics into running batch-size-weighted sums."""
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not math.isnan(float(v)):
            accum[k]  = accum.get(k, 0.0) + v * bs
            counts[k] = counts.get(k, 0)  + bs

def _finalize_(accum: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    """Convert accumulated weighted sums into per-epoch means."""
    return {k: v / max(counts.get(k, 1), 1) for k, v in accum.items()}
