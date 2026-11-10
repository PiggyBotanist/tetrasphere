"""
Function: evaluates the trained kf2vec pipeline against ground-truth.
    1. For each fold, run classify() and query() function from original kf2vec.
    2. (note) kf2vec is not able to find distance that is not from the same subtree.
    3. (note) kf2vec's network is trained in sqrt-distance space, but its own query step already
       undoes this (torch.square() in its query.py) before writing its output file -- that file is
       already in true-distance units, so load_query_predictions() applies no further transform.
    4. (note) kf2vec's own get_distances runs on the raw (unnormalized) pruned tree, so its predicted
       distances come back in raw branch-length units (our model on the other hand trains on normalized
       distance).
    5. (note) please do note that kf2vec result here is not directly comparable with our model due
        some pairs being dropped.

Usage
-----
    python scripts/kf2vec/02_evaluate.py --domain ar53 --distance-file <ar53_distance.hdf5> --split-file <split.hdf5> --benchmark-folder <output_folder>
"""
from __future__ import annotations

# Standard Import
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import hdf5, nwk
from src.system import create_logger
from src.tetrasphere import kf2vec
from src.tetrasphere.preprocess import strip_genomePrefix, load_distance, lookup_distance
from src.tetrasphere.metric import aggregate_metrics

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain",             required=True, type=str, choices=["ar53", "bac120"], help="which domain to process.")
    parser.add_argument("--distance-file",      required=True, type=Path, help="precomputed pairwise distance .hdf5 (from 00_precompute_distance.py; every genome pair, not just within-split).")
    parser.add_argument("--split-file",         required=True, type=Path, help="the matching split file (from 01_split_genome.py (project/tetrasphere/main)), used only for its fold names.")
    parser.add_argument("--benchmark-folder",   required=True, type=Path, help="--output-folder used by 01_train.py.")
    parser.add_argument("--seed",               default=42, type=int, help="kf2vec classify/query random seed.")
    parser.add_argument("--conda",              default=config.env_kf2vec, type=str, help="which conda env to use.")
    return parser.parse_args()

# =================================================================
# Loads distance (uses file generated from precompute_distance.py)
# =================================================================
def evaluate_fold_split(fold: str, split: str, benchmark_folder: Path, map_id_idx: dict[str, int], sorted_keys: np.ndarray, sorted_dist: np.ndarray, n: int, norm_constant: float, args: argparse.Namespace, logger: logging.Logger) -> dict[str, float]:
    """
    Run kf2vec classify+query for one fold's val/eval genomes, then score query-to-backbone
    predictions against true distances.

    Parameters
    ----------
    fold              : fold name, e.g. "fold_1".
    split             : "val" or "eval" (matches the kf_val/kf_eval folders staged by 01_train.py).
    benchmark_folder  : --output-folder used by 01_train.py.
    map_id_idx, sorted_keys, sorted_dist, n : from build_true_distance_lookup().
    norm_constant     : --distance-file's normalization_constant attribute -- rescales kf2vec's raw-tree-unit
                        predictions into the same normalized units as true_dist (see module docstring note 4).
    args              : parsed CLI arguments (seed, conda).
    logger            : logs progress and the resulting metrics.
    """
    query_kf   = benchmark_folder / fold / f"kf_{split}"
    fold_model = benchmark_folder / fold
    output     = benchmark_folder / fold / f"query_{split}"

    kf2vec.classify(query_kf, fold_model, output, args.seed, logger, args.conda)
    kf2vec.query(query_kf, fold_model, output, output, args.seed, logger, args.conda)

    predictions = kf2vec.load_query_predictions(output)
    if len(predictions) == 0:
        logger.warning(f"{fold}/{split}: no predictions produced")
        return {"n_pairs": 0}

    true_dist = lookup_distance(predictions["query_id"].to_numpy(), predictions["backbone_id"].to_numpy(), map_id_idx, sorted_keys, sorted_dist, n, logger)
    valid = ~np.isnan(true_dist)
    pred_dist = predictions["pred_dist"].to_numpy()[valid] / norm_constant
    metrics = aggregate_metrics(pred_dist, true_dist[valid])
    logger.info(f"{fold}/{split}: {metrics}")
    return metrics

# Main
def main() -> None:
    args = parse_args()
    logger = create_logger(f"evaluate_kf2vec_{args.domain}", output_log=config.root_log)

    # Step 0: Load split from .hdf5 and the distance table
    data = hdf5.load(args.split_file)
    fold_names = sorted((k for k in data if k.startswith("fold_")), key=lambda k: int(k.split("_")[1]))
    map_id_idx, sorted_keys, sorted_dist, n = load_distance(args.distance_file, logger)
    norm_constant = float(hdf5.get_attr(args.distance_file, "normalization_constant"))
    logger.info(f"normalization_constant={norm_constant} (rescales kf2vec's raw-tree-unit predictions to match --distance-file's normalized pair_dist)")

    # Step 1: Evaluate every fold's val + eval predictions
    rows = []
    for fold in fold_names:
        for split in ("val", "eval"):
            logger.info(f"=== {fold}/{split} ===")
            metrics = evaluate_fold_split(fold, split, args.benchmark_folder, map_id_idx, sorted_keys, sorted_dist, n, norm_constant, args, logger)
            rows.append({"fold": fold, "dataset": split, **metrics})

    # Step 2: Average across folds, per split (only over folds that actually produced pairs)
    df = pd.DataFrame(rows)
    for split in ("val", "eval"):
        subset = df[(df["dataset"] == split) & (df["n_pairs"] > 0)]
        if len(subset) == 0:
            continue
        row_mean: dict[str, str | float] = {"fold": "mean", "dataset": split}
        for col in ("mse", "mean_distortion", "r_squared", "pearson_r", "spearman_r"):
            row_mean[col] = float(subset[col].mean())
        rows.append(row_mean)

    # Step 3: Save the .tsv
    output_path = config.result_table / "kf2vec" / f"{args.domain}_evaluation.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)
    logger.info(f"Summary -> {output_path}")

if __name__ == "__main__":
    main()
