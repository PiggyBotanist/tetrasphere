from __future__ import annotations

# Standard Import
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Dataset

# =================================================================
# Blend schedule
# =================================================================
def blend_weights(epoch: int, total_epochs: int, n_lengths: int, alpha: float) -> np.ndarray:
    """
    Per-length sampling weight for one epoch.

    Illustration (example with alpha 3.0):
        ->          500k    100k   25k    5k
        - epoch 0:  0.950,  0.047, 0.002, 0.0001
        - midpoint: 0.25,   0.25,  0.25,  0.25
        - final:    0.0001, 0.002, 0.047, 0.950
        
    Parameters
    ----------
    epoch         : current epoch (0-indexed).
    total_epochs  : total epochs in this curriculum run.
    n_lengths     : number of fragment lengths in the curriculum.
    alpha         : schedule steepness at epoch 0 (and, negated, at the final epoch).
    """
    progress = epoch / max(total_epochs - 1, 1)
    alpha_progress = alpha * (1.0 - 2.0 * progress)
    ranks = np.arange(n_lengths, dtype=np.float64)
    logits = -alpha_progress * ranks
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()

def build_train_loader(train_datasets: list[Dataset], weights: np.ndarray, batch_size: int, workers: int, seed: int) -> DataLoader:
    """
    Build a unique combined loader that randomly samples by dataset with a weight

    Parameters
    ----------
    train_datasets  : a list of loaders (per length), NOTE!!! longest-first.
    weights         : this epoch's per-length blend weight, from blend_weights().
    batch_size      : loader batch size.
    workers         : loader worker count.
    seed            : seeds this epoch's WeightedRandomSampler.
    """
    concat = ConcatDataset(train_datasets)
    per_sample = np.concatenate([np.full(len(ds), w / max(len(ds), 1)) for ds, w in zip(train_datasets, weights)])
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.from_numpy(per_sample), num_samples=len(concat), replacement=True, generator=generator)
    return DataLoader(concat, batch_size=batch_size, sampler=sampler, num_workers=workers, drop_last=True)
