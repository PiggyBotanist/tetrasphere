"""
Function: loads trained fold checkpoints (from 03_train_genome.py) and computes full-dataset
validation metrics.

Usage
-----
    python scripts/tetrasphere/05_evaluate_genome.py --label ar53_4mer_clr --experiment <experiment> --split-file <split.hdf5> --epoch 50
"""
from __future__ import annotations

# Standard Import
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import hdf5
from src.system import create_logger
from src.model import checkpoint
from src.tetrasphere.model import Manifold, full_validation
from src.tetrasphere.dataloader import DataLoader_Manifold
from src.tetrasphere.metric import aggregate_metrics

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label",       required=True, type=str, help="training run label.")
    parser.add_argument("--experiment",  required=True, type=str, help="specific experiment (used for storing checkpoint).")
    parser.add_argument("--split-file",  required=True, type=Path, help="prepared split .hdf5.")
    parser.add_argument("--device",      default=None, type=str, help="e.g. 'cuda:0'; auto-selects cuda if unset.")

    # Which checkpoint to evaluate (batch-size only to prevent out of memory)
    parser.add_argument("--epoch",       default=None, type=int, help="checkpoint epoch to load (default to latest).")
    parser.add_argument("--batch-size",  default=4096, type=int)
    parser.add_argument("--worker",      default=4, type=int)
    return parser.parse_args()

# =================================================================
# Helper functions
# =================================================================
def validate_fold(fold_name: str, data: dict, eval_loader: DataLoader, ckpt_root: Path, args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> dict[str, object]:
    """Load this fold's checkpoint (--epoch, or the latest saved), then run full (pooled-metric)
    validation against its own val split and the shared eval holdout."""
    ckpt_path = checkpoint.find(ckpt_root / fold_name, args.epoch)
    cfg = checkpoint.load_config(ckpt_path)
    model = Manifold(**cfg["model_config"]).to(device)
    checkpoint.load_weights(ckpt_path, model, device)

    val_loader = DataLoader(DataLoader_Manifold(data[fold_name]["val"]), batch_size=args.batch_size, shuffle=False, num_workers=args.worker)
    val_metrics = full_validation(model, val_loader, device)
    eval_metrics = full_validation(model, eval_loader, device)
    return {"checkpoint": str(ckpt_path), "epoch": cfg["epoch"], "val": val_metrics, "eval": eval_metrics}

# Main
def main() -> None:
    args = parse_args()
    logger = create_logger(f"validate_{args.label}", output_log=config.root_log)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Step 1: Load validation fold + evaluation
    data = hdf5.load(args.split_file)
    fold_names = sorted((k for k in data if k.startswith("fold_")), key=lambda k: int(k.split("_")[1]))
    eval_loader = DataLoader(DataLoader_Manifold(data["eval"]), batch_size=args.batch_size, shuffle=False, num_workers=args.worker)

    # Step 2: Validate every fold's checkpoint against its own val split and the shared eval holdout
    ckpt_root = config.result_checkpoint / args.experiment / args.label
    summary_perFold = {}
    for fold_name in fold_names:
        logger.info(f"=== {fold_name} ===")
        summary_perFold[fold_name] = validate_fold(fold_name, data, eval_loader, ckpt_root, args, device, logger)

    # Step 3: compute across-fold mean and std (5-fold CV -- std wasn't meaningful with 1 fold)
    metric_names = ["pairwise_mse", "pairwise_mean_distortion", "pairwise_r_squared", "pairwise_pearson_r", "pairwise_spearman_r"]
    per_metric = {name: np.array([fold["eval"][name] for fold in summary_perFold.values()]) for name in metric_names}
    fold_epochs = np.array([fold["epoch"] for fold in summary_perFold.values()], dtype=float)
    row_mean = {"fold": "mean", "dataset": "eval", "epoch": float(fold_epochs.mean()), **{name: float(values.mean()) for name, values in per_metric.items()}}
    row_std  = {"fold": "std",  "dataset": "eval", "epoch": float(fold_epochs.std()),  **{name: float(values.std())  for name, values in per_metric.items()}}

    # Step 4: Export summary (one row per fold x split, plus "mean"/"std" summary rows over eval)
    rows = []
    for fold, result in summary_perFold.items():
        for split in ("val", "eval"):
            rows.append({"fold": fold, "dataset": split, "epoch": result["epoch"], **result[split]})
    rows.append(row_mean)
    rows.append(row_std)

    # Step 5: Save the .tsv
    output_path = config.result_table / args.experiment / f"{args.label}_validation.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)
    logger.info(f"Summary -> {output_path}")

if __name__ == "__main__":
    main()
