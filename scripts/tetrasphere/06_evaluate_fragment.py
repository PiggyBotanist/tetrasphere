"""
Function: loads trained fold checkpoints (from 04_train_fragment.py) and computes full-dataset
validation metrics, per fragment length and pooled across all lengths.

Usage
-----
    python scripts/tetrasphere/06_evaluate_fragment.py \\
        --label bac120_4mer_clr_fragment --experiment bac120 \\
        --split-file-prefix /bls2/data/gtdb/split/fragment/bac120_4mer_clr --lengths 50000 25000 10000 5000 2500 1000 \\
        --genome-split-file /bls2/data/gtdb/split/bac120_4mer_clr_genome.hdf5 --epoch 50
"""
from __future__ import annotations

# Standard Import
import logging
import argparse
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.system import create_logger
from src.model import checkpoint
from src.tetrasphere.model import PhylogenyModel, full_validation
from src.tetrasphere.preprocess import build_parent_lookup, build_fragment_datasets

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label",                required=True, type=str, help="training run label (04_train_fragment.py's --label).")
    parser.add_argument("--experiment",           required=True, type=str, help="specific experiment (used for storing checkpoint).")
    parser.add_argument("--split-file-prefix",    required=True, type=Path, help="--output-prefix used by 02_split_fragment.py; resolves {prefix}_frag{length}.hdf5 per length.")
    parser.add_argument("--lengths",              required=True, type=int, nargs="+", help="space-separated fragment lengths in bp to evaluate.")
    parser.add_argument("--genome-split-file",    required=True, type=Path, help="whole-genome split .hdf5 -- source of each fragment's parent composition (denoiser's reconstruction target).")
    parser.add_argument("--device",               default=None, type=str, help="e.g. 'cuda:0'; auto-selects cuda if unset.")

    # Which checkpoint to evaluate (batch-size only to prevent out of memory)
    parser.add_argument("--epoch",                default=None, type=int, help="checkpoint epoch to load (default: pick lowest pooled val pairwise_mse across all lengths).")
    parser.add_argument("--batch-size",           default=4096, type=int)
    parser.add_argument("--worker",               default=4, type=int)
    parser.add_argument("--max-pairs-per-length", default=None, type=int, help="subsample cap per length/split; default uses every precomputed pair.")
    parser.add_argument("--seed",                 default=42, type=int, help="subsampling seed, only relevant with --max-pairs-per-length.")
    return parser.parse_args()

# =================================================================
# Helper functions
# =================================================================
def load_model(ckpt_path: Path, device: torch.device) -> tuple[PhylogenyModel, dict]:
    """Build+load a PhylogenyModel from a fragment-stage checkpoint, staged for fragment inference."""
    cfg = checkpoint.load_config(ckpt_path)
    model = PhylogenyModel(**cfg["model_config"]).to(device)
    checkpoint.load_weights(ckpt_path, model, device)
    model.stage("fragment")
    return model, cfg

def select_best_epoch(fold_name: str, val_datasets: list[Dataset], ckpt_root: Path, args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> Path:
    """Return the checkpoint path with lowest pooled (all-lengths) val pairwise_mse for this fold."""
    ckpt_paths = sorted((ckpt_root / fold_name).glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpt_paths:
        raise FileNotFoundError(f"no checkpoints under {ckpt_root / fold_name}")

    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.worker)
    best_path, best_mse = None, float("inf")
    for ckpt_path in ckpt_paths:
        model, _ = load_model(ckpt_path, device)
        metrics = full_validation(model, val_loader, device)
        logger.info(f"  {ckpt_path.stem}: pooled_val_pairwise_mse={metrics['pairwise_mse']:.6f}")
        if metrics["pairwise_mse"] < best_mse:
            best_path, best_mse = ckpt_path, metrics["pairwise_mse"]

    logger.info(f"  -> best: {best_path.stem}")
    return best_path

def validate_fold(
    fold_name: str,
    lengths: list[int],
    eval_datasets: list[Dataset],
    train_val_by_fold: dict[str, tuple[list[Dataset], list[Dataset]]],
    ckpt_root: Path,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, object]:
    """Pick this fold's best checkpoint (by pooled val pairwise_mse, or --epoch if given), then run
    per-length validation against its own val split and the shared eval holdout, plus a pooled
    (all-lengths-combined) summary for both."""
    _, val_datasets = train_val_by_fold[fold_name]
    ckpt_path = checkpoint.find(ckpt_root / fold_name, args.epoch) if args.epoch is not None \
        else select_best_epoch(fold_name, val_datasets, ckpt_root, args, device, logger)
    model, cfg = load_model(ckpt_path, device)

    per_length = {}
    for length, val_ds, eval_ds in zip(lengths, val_datasets, eval_datasets):
        val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.worker)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.worker)
        per_length[length] = {"val": full_validation(model, val_loader, device), "eval": full_validation(model, eval_loader, device)}

    pooled_val_loader  = DataLoader(ConcatDataset(val_datasets),  batch_size=args.batch_size, shuffle=False, num_workers=args.worker)
    pooled_eval_loader = DataLoader(ConcatDataset(eval_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.worker)
    pooled = {"val": full_validation(model, pooled_val_loader, device), "eval": full_validation(model, pooled_eval_loader, device)}

    return {"checkpoint": str(ckpt_path), "epoch": cfg["epoch"], "per_length": per_length, "pooled": pooled}

# Main
def main() -> None:
    args = parse_args()
    logger = create_logger(f"fragment_validate_{args.label}", output_log=config.root_log)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Step 1: Which folds to evaluate -- exactly those with a trained fragment-stage checkpoint
    ckpt_root = config.result_checkpoint / args.experiment / args.label
    fold_names = sorted((d.name for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("fold_")), key=lambda k: int(k.split("_")[1]))
    if not fold_names:
        raise FileNotFoundError(f"no fold_* checkpoints under {ckpt_root} -- run 04_train_fragment.py first.")

    # Step 2: Parent (whole-genome) composition lookup, then every length's fragment datasets --
    # read once per length total (eval + every fold's train/val), not once per fold.
    lengths = sorted(set(args.lengths), reverse=True)
    parent_lookup = build_parent_lookup(args.genome_split_file)
    logger.info(f"Parent lookup: {len(parent_lookup)} genomes")
    eval_datasets, train_val_by_fold = build_fragment_datasets(
        args.split_file_prefix, lengths, fold_names, parent_lookup, args.max_pairs_per_length, args.seed, logger,
    )

    # Step 3: Validate every fold's checkpoint, per length + pooled across all lengths
    rows = []
    for fold_name in fold_names:
        logger.info(f"=== {fold_name} ===")
        result = validate_fold(fold_name, lengths, eval_datasets, train_val_by_fold, ckpt_root, args, device, logger)
        for length in lengths:
            for split in ("val", "eval"):
                rows.append({"fold": fold_name, "length": length, "dataset": split, "epoch": result["epoch"], **result["per_length"][length][split]})
        for split in ("val", "eval"):
            rows.append({"fold": fold_name, "length": "pooled", "dataset": split, "epoch": result["epoch"], **result["pooled"][split]})

    # Step 4: Average across folds, per length x split (mean/std over eval only -- mirrors 05_evaluate_genome.py)
    df = pd.DataFrame(rows)
    metric_names = ["pairwise_mse", "pairwise_mean_distortion", "pairwise_r_squared", "pairwise_pearson_r", "pairwise_spearman_r"]
    for length in [*lengths, "pooled"]:
        subset = df[(df["length"] == length) & (df["dataset"] == "eval")]
        if len(subset) == 0:
            continue
        rows.append({"fold": "mean", "length": length, "dataset": "eval", "epoch": float(subset["epoch"].mean()), **{name: float(subset[name].mean()) for name in metric_names}})
        rows.append({"fold": "std",  "length": length, "dataset": "eval", "epoch": float(subset["epoch"].std()),  **{name: float(subset[name].std())  for name in metric_names}})

    # Step 5: Save the .tsv
    output_path = config.result_table / args.experiment / f"{args.label}_fragment_validation.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)
    logger.info(f"Summary -> {output_path}")

if __name__ == "__main__":
    main()
