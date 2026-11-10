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
import pandas as pd
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

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
from src.tetrasphere.preprocess import get_taxonomy, strip_genomePrefix

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label",          required=True, type=Path, help="label given to this run.")
    parser.add_argument("--experiment",     required=True, type=str, help="specific experiment (used for storing checkpoint).")
    parser.add_argument("--split-file",     required=True, type=Path, help="prepared split .hdf5.")
    parser.add_argument("--device",         default=None, type=str, help="e.g. 'cuda:0'; auto-selects cuda if unset.")
    parser.add_argument("--folds",          default=None, type=str, nargs="+", help="restrict to these fold names, e.g. fold_1; default: all folds in the split file.")

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

    # optional class-imbalance ablation
    parser.add_argument("--weighting",       default="none", choices=["none", "inverse_rank"], type=str)
    parser.add_argument("--weighting-rank",  default="phylum", choices=list(config.taxonomy_map.keys()), type=str, help="taxonomic rank inverse-weighted when --weighting inverse_rank.")
    parser.add_argument("--taxonomy-file",   default=None, type=Path, help="genome_id -> taxonomy lookup; required when --weighting inverse_rank.")
    return parser.parse_args()


def pair_weights_inverse_rank(leaf_id: np.ndarray, pairs: np.ndarray, taxonomy_file: Path, rank: str) -> np.ndarray:
    """Pair sampling weight."""
    df_taxonomy = pd.read_csv(taxonomy_file, sep="\t", compression="gzip", names=["id", "taxonomy"])
    stripped = strip_genomePrefix(leaf_id)
    df_rank = get_taxonomy(df_taxonomy, map=config.taxonomy_map, order=list(stripped), level=rank)

    values = df_rank[rank].to_numpy()
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    genome_weight = 1.0 / counts[inverse]
    genome_weight = genome_weight / genome_weight.mean()

    return np.sqrt(genome_weight[pairs[:, 0]] * genome_weight[pairs[:, 1]])

# =================================================================
# Train 1-fold
# =================================================================
def train_fold(fold_name: str, data: dict, eval_loader: DataLoader, args: argparse.Namespace, logger: logging.Logger) -> dict[str, object]:
    """Train one fold's model against its own CV split, then evaluate it against the shared eval holdout."""
    train_split = data[fold_name]["train"]
    val_split   = data[fold_name]["val"]

    train_dataset = DataLoader_Manifold(train_split)
    if args.weighting == "inverse_rank":
        weights = pair_weights_inverse_rank(train_split["leaf_id"], train_split["pair"], args.taxonomy_file, args.weighting_rank)
        sampler = WeightedRandomSampler(weights=torch.from_numpy(weights).double(), num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.worker, drop_last=True)
        logger.info(f"{fold_name}: inverse_{args.weighting_rank} weighting -- pair weight min={weights.min():.4f} max={weights.max():.4f} mean={weights.mean():.4f}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True)
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
    if args.weighting == "inverse_rank" and args.taxonomy_file is None:
        raise ValueError("taxonomy-file is required when --weighting inverse_rank")

    logger = create_logger(f"train_{args.label}", output_log=config.root_log)
    torch.manual_seed(args.seed)

    # Step 1: Load the whole split once; the eval set is shared across every fold
    data = hdf5.load(args.split_file)
    all_folds = sorted((k for k in data if k.startswith("fold_")), key=lambda k: int(k.split("_")[1]))
    fold_names = args.folds if args.folds else all_folds
    eval_loader = DataLoader(DataLoader_Manifold(data["eval"]), batch_size=args.batch_size, shuffle=False, num_workers=args.worker)

    # Step 2: Train and evaluate every fold
    summary = {}
    for fold_name in fold_names:
        logger.info(f"=== {fold_name} ===")
        summary[fold_name] = train_fold(fold_name, data, eval_loader, args, logger)


if __name__ == "__main__":
    main()
