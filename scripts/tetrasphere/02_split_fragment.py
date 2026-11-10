"""
Function: create .hdf5 with different folds and associate membership of the genome for each.
    - writes 1 .hdf5 file per fragment length -genome split file).
    - pure array manipulation (no genome I/O, no vectorization) -- cheap to rerun with a different
      --seed/--max-pairs, unlike 00_precompute_kmerFragment.py's chopping+vectorization pass.

Usage
-----
    python scripts/tetrasphere/02_split_fragment.py --domain ar53 --split-file <split.hdf5> --distance-file <ar53_distance.hdf5> --feature-file-prefix <input_prefix> --lengths 50000 25000 10000 5000 2500 1000 --output-prefix <output_prefix>
"""
from __future__ import annotations

# Standard Import
import logging
import argparse
import numpy as np
from pathlib import Path

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import hdf5
from src.system import create_logger
from src.tetrasphere.dataloader import parent_id
from src.tetrasphere.preprocess import load_distance, lookup_distance, strip_genomePrefix

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain",              required=True, type=str, choices=["ar53", "bac120"], help="which GTDB domain to process.")
    parser.add_argument("--split-file",          required=True, type=Path, help="reference whole-genome split file (from 01_split_genome.py); defines fold/val/eval genome membership to inherit.")
    parser.add_argument("--distance-file",       required=True, type=Path, help="precomputed pairwise distance .hdf5 (from 00_precompute_distance.py).")
    parser.add_argument("--feature-file-prefix", required=True, type=Path, help="--output-prefix used by 00_precompute_kmerFragment.py; resolves {prefix}_frag{length}.hdf5 per length.")
    parser.add_argument("--lengths",             required=True, type=int, nargs="+", help="space-separated fragment lengths in bp, e.g. 50000 25000 10000 5000 2500 1000.")
    parser.add_argument("--output-prefix",       required=True, type=Path, help="output path prefix; writes {output-prefix}_frag{length}.hdf5 per length.")

    # Split parameters
    parser.add_argument("--max-pairs",           default=2_000_000, type=int, help="cap pairs kept per split (train_i/val_i/eval), per length.")
    parser.add_argument("--seed",                default=42, type=int, help="random seed for pair sampling.")

    return parser.parse_args()

# =================================================================
# Pair sampling
# =================================================================
def sample_pairs(n_items: int, max_pairs: int, seed: int) -> np.ndarray:
    """
    Randomly sample up to max_pairs distinct unordered (i, j) index pairs from range(n_items), i != j.

    Parameters
    ----------
    n_items   : number of fragments in this split.
    max_pairs : cap on pairs to sample; capped further at the actual number of possible pairs.
    seed      : random seed.
    """
    if n_items < 2:
        return np.empty((0, 2), dtype=np.int32)

    rng = np.random.default_rng(seed)
    target = min(max_pairs, n_items * (n_items - 1) // 2)
    seen: set[tuple[int, int]] = set()
    while len(seen) < target:
        batch = max(target - len(seen), 1)
        a = rng.integers(0, n_items, size=batch * 2)
        b = rng.integers(0, n_items, size=batch * 2)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        valid = lo != hi
        for x, y in zip(lo[valid].tolist(), hi[valid].tolist()):
            seen.add((x, y))
            if len(seen) >= target:
                break
    return np.array(sorted(seen), dtype=np.int32)

# =================================================================
# Pipeline steps
# =================================================================
def finalize_pairs(parents: np.ndarray, max_pairs: int, seed: int, map_id_idx: dict[str, int], sorted_keys: np.ndarray, sorted_dist: np.ndarray, n: int, logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample up to max_pairs fragment-index pairs and save it to .hdf5

    Parameters
    ----------
    parents                                  : parent genome id per fragment, in this split's write order.
    max_pairs, seed                          : passed to sample_pairs().
    map_id_idx, sorted_keys, sorted_dist, n  : from load_distance(), for true pair distances.
    logger                                   : logs information.
    """
    pairs = sample_pairs(len(parents), max_pairs, seed)
    if len(pairs) == 0:
        return pairs, np.empty(0, dtype=np.float32)

    parent_a, parent_b = parents[pairs[:, 0]], parents[pairs[:, 1]]
    same_parent = parent_a == parent_b

    dist = np.zeros(len(pairs), dtype=np.float64)
    cross = ~same_parent
    if cross.any():
        dist[cross] = lookup_distance(parent_a[cross], parent_b[cross], map_id_idx, sorted_keys, sorted_dist, n, logger)

    # Drop any cross-genome pair whose true distance genuinely couldn't be found
    valid = same_parent | ~np.isnan(dist)
    return pairs[valid], dist[valid].astype(np.float32)

def assign_fragments(fragment_ids: np.ndarray, splits_by_genome: dict[str, list[str]], all_splits: list[str]) -> dict[str, np.ndarray]:
    """
    Bucket fragment row-indices into every split their parent genome belongs to.

    Parameters
    ----------
    fragment_ids       : fragment id per row, in the precomputed feature file's order.
    splits_by_genome   : parent genome id -> list of split names it belongs to (e.g. ["eval"] or
                          ["fold_1/train", "fold_3/val"]), from 01_split_genome.py's membership.
    all_splits         : every split name, for a stable empty-split fallback.
    """
    parents = np.array([parent_id(fid) for fid in fragment_ids], dtype=object)
    indices_by_split: dict[str, list[int]] = {split: [] for split in all_splits}
    for i, parent in enumerate(parents):
        for split in splits_by_genome.get(parent, ()):
            indices_by_split[split].append(i)
    return {split: np.array(idx, dtype=np.int64) for split, idx in indices_by_split.items()}

# Main
def main() -> None:
    args = parse_args()
    logger = create_logger(f"split_fragment_{args.domain}", output_log=config.root_log)
    lengths = sorted(set(args.lengths), reverse=True)

    # Step 1: Load reference split (fold/val/eval genome membership) + distance table
    data = hdf5.load(args.split_file)
    fold_names = sorted((k for k in data if k.startswith("fold_")), key=lambda k: int(k.split("_")[1]))
    eval_leaf_ids = data["eval"]["leaf_id"]
    fold_leaf_ids = {name: {"train": data[name]["train"]["leaf_id"], "val": data[name]["val"]["leaf_id"]} for name in fold_names}

    eval_set = set(eval_leaf_ids)
    fold_train_sets = {fold: set(fold_leaf_ids[fold]["train"]) for fold in fold_names}
    fold_val_sets   = {fold: set(fold_leaf_ids[fold]["val"])   for fold in fold_names}

    map_id_idx, sorted_keys, sorted_dist, n = load_distance(args.distance_file, logger)
    distance_raw = hdf5.load(args.distance_file)
    root_distance_by_id = dict(zip(strip_genomePrefix(distance_raw["leaf_id"]), distance_raw["midpoint_rooted_distance_normalized"]))

    # Step 2: Every genome in the whole-genome split, mapped to the splits it belongs to -- mirrors
    # 01_split_genome.py's join_dataset() intersection filtering (drop genomes with no root distance).
    unique_genome_ids = np.unique(np.concatenate([eval_leaf_ids, *(ids for pair in fold_leaf_ids.values() for ids in pair.values())]))
    splits_by_genome: dict[str, list[str]] = {}
    n_dropped = 0
    for genome_id in unique_genome_ids:
        if genome_id not in root_distance_by_id:
            n_dropped += 1
            continue
        targets = ["eval"] if genome_id in eval_set else []
        for fold in fold_names:
            if genome_id in fold_train_sets[fold]:
                targets.append(f"{fold}/train")
            if genome_id in fold_val_sets[fold]:
                targets.append(f"{fold}/val")
        splits_by_genome[genome_id] = targets
    if n_dropped:
        logger.warning(f"{n_dropped}/{len(unique_genome_ids)} genomes have no root distance -- their fragments are skipped")

    all_splits = ["eval"] + [f"{fold}/{part}" for fold in fold_names for part in ("train", "val")]
    fold_seed = {fold: args.seed + i for i, fold in enumerate(fold_names)}
    split_seed = {"eval": args.seed, **{f"{fold}/{part}": fold_seed[fold] for fold in fold_names for part in ("train", "val")}}

    # Step 3: Per length, bucket the precomputed fragment features by parent's split membership
    for length in lengths:
        feature_path = Path(f"{args.feature_file_prefix}_frag{length}.hdf5")
        precomputed = hdf5.load(feature_path)
        fragment_ids, features = precomputed["id"], precomputed["feature"]

        indices_by_split = assign_fragments(fragment_ids, splits_by_genome, all_splits)
        total_fragments = sum(len(idx) for idx in indices_by_split.values())
        logger.info(f"Length {length}: {total_fragments} fragment-split assignments from {len(fragment_ids)} precomputed fragments")

        output_path = Path(f"{args.output_prefix}_frag{length}.hdf5")
        file = hdf5.create_file(output_path)
        try:
            for split in all_splits:
                idx = indices_by_split[split]
                split_leaf_ids = fragment_ids[idx]
                split_features = features[idx]
                split_parents  = np.array([parent_id(fid) for fid in split_leaf_ids], dtype=object)
                split_root     = np.array([root_distance_by_id[p] for p in split_parents], dtype=np.float32)

                hdf5.add_col(file, f"{split}/leaf_id", split_leaf_ids)
                hdf5.add_col(file, f"{split}/feature", split_features)
                hdf5.add_col(file, f"{split}/root_distance", split_root)

                pairs, pair_dist = finalize_pairs(split_parents, args.max_pairs, split_seed[split], map_id_idx, sorted_keys, sorted_dist, n, logger)
                hdf5.add_col(file, f"{split}/pair", pairs)
                hdf5.add_col(file, f"{split}/pair_dist", pair_dist)
                logger.info(f"{split}: {len(idx)} fragments, {len(pairs)} pairs")

            hdf5.set_attr(file, "label", args.domain)
            hdf5.set_attr(file, "length", length)
            hdf5.set_attr(file, "seed", args.seed)
        finally:
            file.close()
        logger.info(f"Saved -> {output_path}")

if __name__ == "__main__":
    main()
