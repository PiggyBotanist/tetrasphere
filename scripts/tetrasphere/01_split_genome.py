"""
Function: takes precomputed distance and kmer, generate a train-test split for genome

Usage
-----
    python scripts/tetrasphere/01_split_genome.py --domain ar53 --kmer-file <kmer.hdf5> --distance-file <distance.hdf5> --output-file <output_path>
"""
from __future__ import annotations

# Standard Import
import h5py
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import hdf5
from src.system import create_logger
from src.tetrasphere.preprocess import CombinedDataset, strip_genomePrefix, get_taxonomy

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Required fields
    parser.add_argument("--label",          required=True, help="help differentiate log.")
    parser.add_argument("--domain",         required=True, choices=["bac120", "ar53"], type=str, help="which GTDB domain this data is for.")
    parser.add_argument("--feature-file",   required=True, type=Path, help="precomputed k-mer feature .hdf5 (from 00_precompute_kmerGenome.py).")
    parser.add_argument("--distance-file",  required=True, type=Path, help="precomputed pairwise distance .hdf5 (from 00_precompute_distance.py).")
    parser.add_argument("--taxonomy-file",  required=True, type=Path, help="metadata for taxonomy-based fold holdout.")
    parser.add_argument("--output-file",    required=True, type=Path, help="output .hdf5 file path.")

    # Split parameters
    parser.add_argument("--seed",           default=42, type=int, help="random seed for the genome-level split.")
    parser.add_argument("--eval-frac",      default=0.2, type=float, help="fraction for evaluation.")
    parser.add_argument("--n-folds",        default=5,  type=int, help="number of cross-validation folds.")
    parser.add_argument("--holdout-level",  default="genome", choices=list(config.taxonomy_map.keys()) + ["genome"], type=str, help="hold out whole groups per fold; 'genome' make is regular splits instead.")
    parser.add_argument("--max-pairs",      default=2_000_000, type=int, help="cap pairs kept per split; unset keeps every within-split pair.")
    
    
    return parser.parse_args()

# =================================================================
# Helper functions
# =================================================================
def filter_pairs(pairs: np.ndarray, pair_dist: np.ndarray, remap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only pairs whose both endpoints are kept, remapped via `remap`.

    Parameters
    ----------
    pairs      : (n_pairs, 2) leaf-index pairs, indexed into whatever array `remap` maps from.
    pair_dist  : (n_pairs,) normalized pairwise distance.
    remap      : new row index for each original leaf position; -1 if dropped.
    """
    both_kept = (remap[pairs[:, 0]] >= 0) & (remap[pairs[:, 1]] >= 0)
    kept_pairs = remap[pairs[both_kept]].astype(np.int32)
    return kept_pairs, pair_dist[both_kept]

def subsample_pairs(pairs: np.ndarray, pair_dist: np.ndarray, max_pairs: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniformly subsample down to at most max_pairs pairs; a no-op if already smaller or max_pairs is None.

    Parameters
    ----------
    pairs      : (n_pairs, 2) leaf-index pairs.
    pair_dist  : (n_pairs,) normalized pairwise distance.
    max_pairs  : cap on pairs to keep; None keeps every pair.
    seed       : random seed.
    """
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs, pair_dist
    keep = np.random.default_rng(seed).choice(len(pairs), size=max_pairs, replace=False)
    return pairs[keep], pair_dist[keep]

# =================================================================
# Pipeline steps
# =================================================================
def join_dataset(kmer: dict, distance: dict, logger: logging.Logger) -> CombinedDataset:
    """
    Align k-mer features onto the distance file's leaf order, keeping only genomes present in both.

    Parameters
    ----------
    kmer      : loaded k-mer .hdf5 (id, feature).
    distance  : loaded distance .hdf5 (leaf_id, root_to_tip).
    logger    : logs information.
    """
    leafId_distance = strip_genomePrefix(distance["leaf_id"])
    map_kmerGenomeId_rowId = {gid: i for i, gid in enumerate(kmer["id"])}

    # Only keep genome ids that is both in distance and kmer .hdf5
    keep_mask = np.array([gid in map_kmerGenomeId_rowId for gid in leafId_distance], dtype=bool)
    kmer_row = np.array([map_kmerGenomeId_rowId[gid] for gid in leafId_distance[keep_mask]], dtype=np.int64)
    logger.info(f"Genomes in distance file:         {len(leafId_distance)}")
    logger.info(f"Genomes with matching k-mer rows: {int(keep_mask.sum())}")
    
    # Only keep distance pairs that is in kept genome
    remap = np.full(len(distance["leaf_id"]), -1, dtype=np.int64)
    remap[keep_mask] = np.arange(int(keep_mask.sum()))
    pairs, pair_dist = filter_pairs(distance["pairs"], distance["pair_dist"], remap)

    return CombinedDataset(
        leaf_id=        distance["leaf_id"][keep_mask],
        source_id=      np.flatnonzero(keep_mask),
        feature=        kmer["feature"][kmer_row],
        root_distance=  distance["midpoint_rooted_distance_normalized"][keep_mask],
        pair=           pairs,
        pair_dist=      pair_dist
    )
    
def shuffle_genomes(joined: CombinedDataset, seed: int) -> CombinedDataset:
    """
    Randomly reorder every genome-aligned array (ensure no ordering effect).

    Parameters
    ----------
    joined  : output of join_dataset().
    seed    : random seed.
    """
    perm = np.random.default_rng(seed).permutation(len(joined.leaf_id))
    return CombinedDataset(
        leaf_id=        joined.leaf_id[perm],
        source_id=      joined.source_id[perm],
        feature=        joined.feature[perm],
        root_distance=  joined.root_distance[perm],
        pair=           joined.pair,
        pair_dist=      joined.pair_dist, 
    )

def assign_split(leaf_ids: np.ndarray, groups: np.ndarray, n_folds: int, eval_frac: float, seed: int) -> dict[str, np.ndarray]:
    """
    Split genomes into:
        - n-folds (80% genomes total)
        - eval (20% held-out genomes)

    Parameters
    ----------
    leaf_ids          : genome id per genome.
    groups            : group label per genome (its taxonomy at some rank, or its own id for a plain per-genome split).
    n_folds           : number of cross-validation folds to create from the training pool.
    eval_frac         : fraction of groups held out for eval, e.g. 0.2 for roughly 20%.
    seed              : random seed.
    """
    # Obtain the evaluation set (the first fold)
    fold_eval = max(round(1 / eval_frac), 2)
    eval_mask = assign_folds_heldout(groups, fold_eval, seed) == 0
    pool_mask = ~eval_mask

    # The training set (exclude the eval set)
    ids_train = leaf_ids[pool_mask]
    fold_train = assign_folds_stratified(groups[pool_mask], n_folds, seed)

    split = {f"fold_{fold + 1}": ids_train[fold_train == fold] for fold in range(n_folds)}
    split["eval"] = leaf_ids[eval_mask]
    return split 

def assign_folds_heldout(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """
    Held-out sampling per fold (each fold contains unique group).

    Parameters
    ----------
    groups   : group label.
    n_folds  : number of buckets to create.
    seed     : random seed.
    """
    # Randomize groups (prevent ordering effect)
    rng = np.random.default_rng(seed)
    unique_groups, group_sizes = np.unique(groups, return_counts=True)

    # Random sorting (primarily for tie groups)
    shuffle = rng.permutation(len(unique_groups))
    unique_groups = unique_groups[shuffle]
    group_sizes = group_sizes[shuffle]

    # NOTE: this part causes issues with heldout experiment hence is removed
    # # Order groups in decesnding order
    # order = np.argsort(-group_sizes, kind="stable")
    # unique_groups, group_sizes = unique_groups[order], group_sizes[order]

    # initialize (group -> fold mapping) & sample size in each fold
    map_group_fold = {}
    fold_size = np.zeros(n_folds, dtype=np.int64)
    for n in range(0, len(unique_groups), n_folds):
        # Get first n groups (assign each to a fold)
        chunk_groups = unique_groups[n:n + n_folds]
        chunk_sizes  = group_sizes[n:n + n_folds]

        # Sort fold_id by size (ascending order)
        available_folds = list(np.argsort(fold_size, kind="stable"))
        for group, size in zip(chunk_groups, chunk_sizes):
            fold = available_folds.pop(0)
            map_group_fold[group] = fold
            fold_size[fold] += size

    return np.array([map_group_fold[g] for g in groups], dtype=np.int64)

def assign_folds_stratified(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """
    Stratified sampling per fold (even distribution per group).
    
    Parameters
    ----------
    groups   : group label.
    n_folds  : number of folds to create.
    seed     : random seed.
    """
    rng = np.random.default_rng(seed)
    map_id_fold = np.empty(len(groups), dtype=np.int64)
    for group in np.unique(groups):
        # Filter to the group index & random shuffle
        member_index = np.flatnonzero(groups == group)
        rng.shuffle(member_index)
        
        # Append starting at random fold index (prevent assigning last fold with low count consistently)
        offset = rng.integers(n_folds)
        map_id_fold[member_index] = (np.arange(len(member_index)) + offset) % n_folds
    return map_id_fold

def save_split(file: h5py.File, prefix: str, distance: dict, joined: CombinedDataset, mask: np.ndarray, max_pairs: int | None, seed: int, logger: logging.Logger) -> None:
    """
    Save one split's genomes and within-split pairs under `prefix/` in the output .hdf5.

    Parameters
    ----------
    file        : open output file, from hdf5.open().
    prefix      : group path to save under, e.g. "fold0/train" or "eval".
    distance    : loaded distance .hdf5 (leaf_id, pairs, pair_dist), for original-order pair indices.
    joined      : output of join_kmer_distance()/shuffle_genomes().
    mask        : boolean mask over joined's genome order, selecting this split's genomes.
    max_pairs   : cap on pairs to keep; None keeps every pair.
    seed        : random seed for pair subsampling.
    logger      : logs information.
    """
    remap = np.full(len(distance["leaf_id"]), -1, dtype=np.int64)
    remap[joined.source_id[mask]] = np.arange(int(mask.sum()))

    pairs, pair_dist = filter_pairs(distance["pairs"], distance["pair_dist"], remap)
    pairs, pair_dist = subsample_pairs(pairs, pair_dist, max_pairs, seed)
    logger.info(f"{prefix}: {int(mask.sum())} genomes, {len(pairs)} pairs")

    hdf5.add_col(file, f"{prefix}/leaf_id",         joined.leaf_id[mask])
    hdf5.add_col(file, f"{prefix}/feature",         joined.feature[mask])
    hdf5.add_col(file, f"{prefix}/root_distance",   joined.root_distance[mask])
    hdf5.add_col(file, f"{prefix}/pair",            pairs)
    hdf5.add_col(file, f"{prefix}/pair_dist",       pair_dist)
    
# Main
def main() -> None:
    args = parse_args()
    logger = create_logger(f"prepare_split_{args.label}", output_log=config.root_log)

    # Step 1: Load precomputed data
    kmer = hdf5.load(args.feature_file)
    distance = hdf5.load(args.distance_file)
    df_metadata = pd.read_csv(args.taxonomy_file, sep='\t', compression="gzip", names=['id', 'taxonomy'])
    
    # Step 2: Join k-mer features onto the distance file's genome set, then shuffle
    joined = join_dataset(kmer, distance, logger)
    joined = shuffle_genomes(joined, args.seed)

    # Step 3: Assign genomes to folds
    if args.holdout_level == "genome":
        groups = joined.leaf_id
    else:
        df_taxonomy = get_taxonomy(df_metadata, map=config.taxonomy_map, order=list(joined.leaf_id), level=args.holdout_level)
        groups = df_taxonomy[args.holdout_level].to_numpy()

    # Step 4: Split into fold_1..fold_n plus eval, each keyed by genome id
    split = assign_split(joined.leaf_id, groups, args.n_folds, args.eval_frac, args.seed)
    fold_names = [f"fold_{i + 1}" for i in range(args.n_folds)]
    for name in [*fold_names, "eval"]:
        logger.info(f"{name}: {len(split[name])} genomes")
    
    # Step 5: Save the eval set and each fold's train/val split
    file, parameters = hdf5.open(args.output_file)
    eval_mask = np.isin(joined.leaf_id, split["eval"])
    save_split(file, "eval", distance, joined, eval_mask, args.max_pairs, args.seed, logger)
    
    for i, name in enumerate(fold_names):
        val_mask = np.isin(joined.leaf_id, split[name])
        train_ids = np.concatenate([split[other] for other in fold_names if other != name])
        train_mask = np.isin(joined.leaf_id, train_ids)

        # Duplicated storage (but this allows us to obtain between validation pairs to train)
        save_split(file, f"{name}/train", distance, joined, train_mask, args.max_pairs, args.seed + i, logger)
        save_split(file, f"{name}/val",   distance, joined, val_mask,   args.max_pairs, args.seed + i, logger)
    
    hdf5.set_attr(file, "label", args.domain)
    hdf5.set_attr(file, "seed", args.seed)
    hdf5.set_attr(file, "n_folds", args.n_folds)
    hdf5.set_attr(file, "eval_frac", args.eval_frac)
    hdf5.set_attr(file, "holdout_level", args.holdout_level)
    hdf5.close(file, parameters)

if __name__ == "__main__":
    main()
