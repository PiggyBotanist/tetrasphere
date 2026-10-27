"""
Stage 1: trains the whole-genome Manifold (PhylogenyModel's encoder half) as a standalone
Siamese pair model. Its checkpoints are the frozen encoder that 04_train_fragment.py loads
(per fold) for stage 2.

Usage
-----
    python scripts/tetrasphere/03_train_genome.py --domain ar53 --split-file <split.hdf5>
"""
from __future__ import annotations

# Standard Import
import logging
import argparse
import numpy as np
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import hdf5
from src.system import create_logger
from src.model.trainer import Trainer, TrainConfig
from src.tetrasphere.model import Manifold
from src.tetrasphere.loss import Phylogeny_Loss, eval_phylogeny
from src.tetrasphere.dataloader import DataLoader_Manifold

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label",          required=True, type=Path, help="label given to this run.")
    parser.add_argument("--experiment",     required=True, type=str, help="specific experiment (used for storing checkpoint).")
    parser.add_argument("--split-file",     required=True, type=Path, help="prepared split .hdf5.")
    parser.add_argument("--device",         default=None, type=str, help="e.g. 'cuda:0'; auto-selects cuda if unset.")

    # Parameters
    parser.add_argument("--seed",           default=42, type=int)
    parser.add_argument("--worker",         default=4, type=int)
    parser.add_argument("--batch-size",     default=4096, type=int)
    parser.add_argument("--epoch",          default=50, type=int)
    parser.add_argument("--lr",             default=1e-3, type=float)
    parser.add_argument("--weight-decay",   default=0.0, type=float)
    parser.add_argument("--wandb",          action="store_true")

    # Model hyperparameters
    parser.add_argument("--hidden-dim",     default=256, type=int, help="Manifold residual block width.")
    parser.add_argument("--output-dim",     default=128, type=int, help="embedding space dimension.")
    parser.add_argument("--layers",         default=2,   type=int, help="number of Manifold residual blocks.")
    parser.add_argument("--dropout",        default=0.1, type=float)
    return parser.parse_args()

# =================================================================
# Train 1-fold
# =================================================================
def train_fold(fold_name: str, data: dict, eval_loader: DataLoader, args: argparse.Namespace, logger: logging.Logger) -> dict[str, object]:
    """Train one fold's model against its own CV split, then evaluate it against the shared eval holdout."""
    train_split = data[fold_name]["train"]
    val_split   = data[fold_name]["val"]

    train_loader = DataLoader(DataLoader_Manifold(train_split), batch_size=args.batch_size, shuffle=True,  num_workers=args.worker, drop_last=True)
    val_loader   = DataLoader(DataLoader_Manifold(val_split),   batch_size=args.batch_size, shuffle=False, num_workers=args.worker)

    dim_input = train_split["feature"].shape[1]
    model     = Manifold(dim_input, args.hidden_dim, args.output_dim, args.layers, args.dropout)
    loss_fn   = Phylogeny_Loss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_config = TrainConfig(
        project=        args.experiment,
        label=          f"{args.label}_{fold_name}",
        epochs=         args.epoch,
        device=         args.device,
        model_config=   model.config,
        checkpoint_dir= config.result_checkpoint / args.experiment / args.label / f"{fold_name}",
        save_every=     5,
        val_hook=       eval_phylogeny,
        use_wandb=      args.wandb
    )

    trainer = Trainer(model, loss_fn, optimizer, logger, train_config)
    result  = trainer.fit(train_loader, val_loader)

    # Step 1: Best CV epoch by lowest validation loss
    val_losses = result.get("val_loss", [])
    best_epoch = int(np.argmin(val_losses)) if val_losses else -1
    best_val   = {k.removeprefix("val_"): v[best_epoch] for k, v in result.items() if k.startswith("val_")}

    # Step 2: Evaluate this fold's final model against the shared eval holdout
    eval_metrics = trainer.evaluate(eval_loader)

    logger.info(f"{fold_name} summary: best_epoch={best_epoch + 1}  val={best_val}  eval={eval_metrics}")
    return {"best_epoch": best_epoch + 1, "val": best_val, "eval": eval_metrics}

# Main
def main() -> None:
    args = parse_args()
    logger = create_logger(f"train_{args.label}", output_log=config.root_log)
    torch.manual_seed(args.seed)

    # Step 1: Load the whole split once; the eval set is shared across every fold
    data = hdf5.load(args.split_file)
    fold_names = sorted((k for k in data if k.startswith("fold_")), key=lambda k: int(k.split("_")[1]))
    eval_loader = DataLoader(DataLoader_Manifold(data["eval"]), batch_size=args.batch_size, shuffle=False, num_workers=args.worker)

    # Step 2: Train and evaluate every fold
    summary = {}
    for fold_name in fold_names:
        logger.info(f"=== {fold_name} ===")
        summary[fold_name] = train_fold(fold_name, data, eval_loader, args, logger)


if __name__ == "__main__":
    main()
