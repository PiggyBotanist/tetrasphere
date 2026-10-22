"""
Stage 2: trains the Denoiser half of PhylogenyModel, per fold, against that SAME fold's own
stage-1 genome checkpoint (from 03_train_genome.py) loaded frozen into PhylogenyModel.manifold --
never a different fold's, which would leak across the CV split. Uses a length curriculum
(blend_weights/build_train_loader in src/tetrasphere/curriculum.py) that shifts sampling weight
from the longest fragment length toward the shortest over the course of training.

Usage
-----
    python scripts/tetrasphere/04_train_fragment.py \\
        --label ar53_4mer_clr_fragment --experiment ar53 \\
        --split-file-prefix <split_fragment_prefix> --lengths 50000 25000 10000 5000 2500 1000 \\
        --genome-split-file <whole_genome_split.hdf5> --genome-experiment ar53 --genome-label ar53_4mer_clr_genome
"""
from __future__ import annotations

# Standard Import
import logging
import argparse
import numpy as np
from pathlib import Path
import torch
import wandb
from torch.optim import AdamW
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, ConcatDataset

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.system import create_logger
from src.model import checkpoint
from src.model.trainer import train_epoch, eval_epoch, log_epoch
from src.tetrasphere.model import PhylogenyModel
from src.tetrasphere.loss import Phylogeny_Loss, eval_phylogeny
from src.tetrasphere.curriculum import blend_weights, build_train_loader
from src.tetrasphere.preprocess import build_parent_lookup, build_fragment_datasets

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label",              required=True, type=str, help="label given to this run.")
    parser.add_argument("--experiment",         required=True, type=str, help="specific experiment (used for storing checkpoint).")
    parser.add_argument("--split-file-prefix",  required=True, type=Path, help="--output-prefix used by fragment/00_precompute_split.py; resolves {prefix}_frag{length}.hdf5 per length.")
    parser.add_argument("--lengths",            required=True, type=int, nargs="+", help="space-separated fragment lengths in bp; sorted longest-first regardless of input order.")
    parser.add_argument("--genome-split-file",  required=True, type=Path, help="whole-genome split .hdf5 (03_train_genome.py's --split-file) -- source of each fragment's parent composition (the denoiser's reconstruction target).")
    parser.add_argument("--genome-experiment",  required=True, type=str, help="--experiment used by 03_train_genome.py, to locate its checkpoints.")
    parser.add_argument("--genome-label",       required=True, type=str, help="--label used by 03_train_genome.py, to locate its checkpoints.")
    parser.add_argument("--genome-epoch",       default=None, type=int, help="which genome-stage checkpoint epoch to freeze in, per fold (default latest).")
    parser.add_argument("--device",             default=None, type=str, help="e.g. 'cuda:0'; auto-selects cuda if unset.")

    # Parameters
    parser.add_argument("--seed",               default=42, type=int)
    parser.add_argument("--worker",              default=4, type=int)
    parser.add_argument("--batch-size",         default=4096, type=int)
    parser.add_argument("--epoch",              default=50, type=int)
    parser.add_argument("--lr",                 default=1e-3, type=float)
    parser.add_argument("--weight-decay",       default=0.0, type=float)
    parser.add_argument("--grad-clip",          default=1.0, type=float)
    parser.add_argument("--save-every",         default=5, type=int)
    parser.add_argument("--curriculum-alpha",   default=2.0, type=float, help="blend schedule steepness: epoch 0 favors the longest length by e^alpha per rank.")
    parser.add_argument("--max-pairs-per-length", default=None, type=int, help="subsample cap per length/split; default uses every precomputed pair.")

    # Denoiser hyperparameters (manifold's own dims come from the loaded genome checkpoint, not from here)
    parser.add_argument("--denoiser-hidden",    default=256, type=int)
    parser.add_argument("--denoiser-layers",    default=2,   type=int)
    parser.add_argument("--dropout",            default=0.1, type=float)

    # Loss weights -- defaults match the frozen-manifold+denoiser method validated above
    parser.add_argument("--lambda-recon",       default=1.0, type=float)
    parser.add_argument("--lambda-pairwise",    default=0.8, type=float)
    parser.add_argument("--lambda-root",        default=0.2, type=float)
    parser.add_argument("--log-eps",            default=0.01, type=float)

    parser.add_argument("--wandb",              action="store_true")
    return parser.parse_args()

# =================================================================
# Train 1-fold
# =================================================================
def train_fold(
    fold_name: str,
    lengths: list[int],
    train_datasets: list,
    val_datasets: list,
    eval_loader: DataLoader,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict[str, object]:
    """Load this fold's own frozen genome manifold, train its denoiser on the blended fragment
    curriculum, then evaluate against the shared eval holdout."""
    if args.wandb:
        wandb.init(project=args.experiment, name=f"{args.label}_{fold_name}", reinit=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16
    scaler = GradScaler(enabled=False)

    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.worker)

    # Step 1: This fold's own genome checkpoint (never a different fold's -- avoids CV leakage)
    genome_ckpt_dir = config.result_checkpoint / args.genome_experiment / args.genome_label / fold_name
    genome_ckpt_path = checkpoint.find(genome_ckpt_dir, args.genome_epoch)
    genome_cfg = checkpoint.load_config(genome_ckpt_path)["model_config"]
    logger.info(f"{fold_name}: freezing manifold <- {genome_ckpt_path}")

    dim_fragment = train_datasets[0][0]["feature_a"].shape[0]
    model = PhylogenyModel(
        dim_clean=genome_cfg["dim_input"],
        dim_fragment=dim_fragment,
        dim_output=genome_cfg["dim_output"],
        manifold_hidden=genome_cfg["dim_hidden"],
        manifold_layers=genome_cfg["layers"],
        denoiser_hidden=args.denoiser_hidden,
        denoiser_layers=args.denoiser_layers,
        dropout=args.dropout,
    ).to(device)
    checkpoint.load_weights(genome_ckpt_path, model.manifold, device)
    model.stage("fragment")

    loss_fn   = Phylogeny_Loss(args.lambda_recon, args.lambda_pairwise, args.lambda_root, args.log_eps)
    optimizer = AdamW(model.denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # frozen manifold has no grad anyway, but be explicit

    checkpoint_dir = config.result_checkpoint / args.experiment / args.label / fold_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {}
    for epoch in range(args.epoch):
        weights = blend_weights(epoch, args.epoch, len(lengths), args.curriculum_alpha)
        logger.info("Epoch %d blend: %s", epoch + 1, ", ".join(f"{length}bp={w:.3f}" for length, w in zip(lengths, weights)))

        train_loader = build_train_loader(train_datasets, weights, args.batch_size, args.worker, args.seed + epoch)
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, amp_dtype, amp_enabled, args.grad_clip, epoch, args.epoch)
        val_metrics   = eval_epoch(model, val_loader, loss_fn, device, amp_dtype, amp_enabled, epoch, logger, eval_phylogeny)
        log_epoch(logger, train_metrics, val_metrics, epoch, optimizer.param_groups[0]["lr"], args.wandb)

        for k, v in train_metrics.items(): history.setdefault(f"train_{k}", []).append(v)
        for k, v in val_metrics.items():   history.setdefault(f"val_{k}", []).append(v)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epoch - 1:
            path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
            checkpoint.save(path, model, optimizer, scaler, epoch + 1, {**train_metrics, **val_metrics}, loss_fn.config, model.config)
            logger.info("Checkpoint -> %s", path)

    # Step 1: Best CV epoch by lowest validation pairwise MSE
    val_mse = history.get("val_mse_pairwise", [])
    best_epoch = int(np.argmin(val_mse)) if val_mse else -1
    best_val   = {k.removeprefix("val_"): v[best_epoch] for k, v in history.items() if k.startswith("val_")}

    # Step 2: Evaluate this fold's final model against the shared eval holdout (all lengths pooled)
    model.eval()
    eval_metrics = eval_epoch(model, eval_loader, loss_fn, device, amp_dtype, amp_enabled, args.epoch, logger, eval_phylogeny)

    logger.info(f"{fold_name} summary: best_epoch={best_epoch + 1}  val={best_val}  eval={eval_metrics}")
    if args.wandb:
        wandb.finish()
    return {"best_epoch": best_epoch + 1, "val": best_val, "eval": eval_metrics}

# Main
def main() -> None:
    args = parse_args()
    logger = create_logger(f"train_{args.label}", output_log=config.root_log)
    torch.manual_seed(args.seed)

    # Step 1: Which folds to train -- exactly those with a trained genome-stage checkpoint
    genome_ckpt_root = config.result_checkpoint / args.genome_experiment / args.genome_label
    fold_names = sorted((d.name for d in genome_ckpt_root.iterdir() if d.is_dir() and d.name.startswith("fold_")), key=lambda k: int(k.split("_")[1]))
    if not fold_names:
        raise FileNotFoundError(f"no fold_* checkpoints under {genome_ckpt_root} -- run 03_train_genome.py first.")

    # Step 2: Parent (whole-genome) composition lookup, then every length's fragment datasets --
    # read once per length total (eval + every fold's train/val), not once per fold.
    lengths = sorted(set(args.lengths), reverse=True)
    parent_lookup = build_parent_lookup(args.genome_split_file)
    logger.info(f"Parent lookup: {len(parent_lookup)} genomes")
    eval_datasets, train_val_by_fold = build_fragment_datasets(
        args.split_file_prefix, lengths, fold_names, parent_lookup, args.max_pairs_per_length, args.seed, logger,
    )
    eval_loader = DataLoader(ConcatDataset(eval_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.worker)

    # Step 3: Train and evaluate every fold
    summary = {}
    for fold_name in fold_names:
        logger.info(f"=== {fold_name} ===")
        train_datasets, val_datasets = train_val_by_fold[fold_name]
        summary[fold_name] = train_fold(fold_name, lengths, train_datasets, val_datasets, eval_loader, args, logger)


if __name__ == "__main__":
    main()
