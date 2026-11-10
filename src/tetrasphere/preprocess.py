from __future__ import annotations

# Standard Import
import logging
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset, Subset

# Custom Import
from src.io import hdf5
from src.tetrasphere.dataloader import DataLoader_Denoiser, parent_id

@dataclass
class CombinedDataset:
    """Genome-level features and distances, aligned to one shared order."""
    leaf_id:        np.ndarray
    source_id:      np.ndarray
    feature:        np.ndarray
    root_distance:  np.ndarray
    pair:           np.ndarray
    pair_dist:      np.ndarray
    
# =================================================================
# Manipulation
# =================================================================
def strip_genomePrefix(accessions: np.ndarray) -> np.ndarray:
    """Drop the additional source prefix, (GB_/RS_)."""
    return np.array([a.removeprefix("GB_").removeprefix("RS_") for a in accessions], dtype=object)

def get_taxonomy(df: pd.DataFrame, map: dict[str, str], order: list[str] = [], level: str = "") -> pd.DataFrame:
    """
    Parse a dataframe containing taxonomy.

    Parameters
    ----------
    df                  : dataframe with taxonomy (column name must be "id", "taxonomy")
    map                 : taxonomy mapping logic {"domain": "d__", "phylum": "p__", "class": "c__" ...}
    order               : (default: no ordering) whether to sort id by this order or not 
    level               : (default: return all) which taxonomic level to return as defined by `map`.
    """
    # Ensure input data is valid
    if "id" not in df.columns or "taxonomy" not in df.columns:
        raise ValueError("DataFrame must contain 'id' and 'taxonomy' columns.")
    if level and level not in map:
        raise ValueError(f"Unknown taxonomic level {level!r}. Expected one of: {list(map)}!")
    
    # Parse
    rows = []
    levels = {level: map[level]} if level else map
    for gid, taxonomy in zip(df["id"], df["taxonomy"]):
        ranks = taxonomy.split(";") if pd.notna(taxonomy) else []
        row = {"id": gid.removeprefix("RS_").removeprefix("GB_")}
        for name, prefix in levels.items():
            value = next((rank[len(prefix):] for rank in ranks if rank.startswith(prefix)), "")
            row[name] = value or f"unclassified_{name}"
        rows.append(row)

    # Sort by order
    parsed = pd.DataFrame(rows)
    if order:
        parsed = parsed.set_index("id").loc[list(order)].reset_index()
    return parsed

# =================================================================
# Filter
# =================================================================
def filter_genome(leaf_label: np.ndarray, df_metadata: pd.DataFrame, folder_representatives: Path, completness: float, contamination: float, logger: logging.Logger):
    """
    Filter genome ids to only those with known taxonomy and met quality threshold.

    Parameters
    ----------
    leaf_label              : tree leaf (genome ids).
    df_taxonomy             : dataframe for genome_id -> taxonomy mapping.
    df_metadata             : dataframe for genome_id -> completeness/contamination mapping.
    folder_representatives  : representative genomes
    completness             : completeness threshod (≥).
    contamination           : contamination threshold (≤).
    logger                  : logs information.
    """
    df = df_metadata.copy()
    
    # ensure numeric
    df["checkm2_completeness"] = pd.to_numeric(df["checkm2_completeness"], errors="coerce")
    df["checkm2_contamination"] = pd.to_numeric(df["checkm2_contamination"], errors="coerce")
    logger.info(f"Total Genome:                     {len(df)}")

    # filter (nodes present in tree)
    df = df[df["accession"].isin(leaf_label)]
    logger.info(f"Total Genome (tree filter):       {len(df)}")

    # filter (completeness /contamination)
    df = df[df['checkm2_completeness'] >= 100*completness]
    df = df[df['checkm2_contamination'] <= 100*contamination]
    logger.info(f"Total Genome (quality filter):    {len(df)}")
           
    # accession cleanup
    df["accession"] = strip_genomePrefix(df["accession"].to_numpy())
    
    # filter (node present in genome folder)
    genomes = [f.name.strip("_genomic.fna.gz") for f in folder_representatives.glob("*.fna.gz")]
    df = df[df["accession"].isin(genomes)]
    logger.info(f"Total Genome (folder matching):   {len(df)}")
    return list(df['accession'])

# =================================================================
# Pairwise distance lookup (using precomputed .hdf5)
# =================================================================
def load_distance(input_hdf5: Path, logger: logging.Logger) -> tuple[dict[str, int], np.ndarray, np.ndarray, int]:
    """
    Load the precomputed pairwise distance table and index it.

    Parameters
    ----------
    input_hdf5  : precomputed pairwise distance .hdf5, from project/tetrasphere/main/00_precompute_distance.py.
    logger      : logs the resulting genome/pair count.
    """
    distance = hdf5.load(input_hdf5)
    stripped_ids = strip_genomePrefix(distance["leaf_id"])
    map_id_idx = {gid: i for i, gid in enumerate(stripped_ids)}

    # Sort the pairs
    n = len(stripped_ids)
    i = distance["pairs"][:, 0].astype(np.int64)
    j = distance["pairs"][:, 1].astype(np.int64)
    
    # Obtain the key with highest/lowest value
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    keys = lo * n + hi

    order = np.argsort(keys)
    sorted_keys = keys[order]
    sorted_dist = distance["pair_dist"][order]

    logger.info(f"Loaded {len(sorted_keys)} pairwise distances for {n} genomes from {input_hdf5}")
    return map_id_idx, sorted_keys, sorted_dist, n

def lookup_distance(query_ids: np.ndarray, backbone_ids: np.ndarray, map_id_idx: dict[str, int], sorted_keys: np.ndarray, sorted_dist: np.ndarray, n: int, logger: logging.Logger) -> np.ndarray:
    """
    Lookup query to backbone distance for each genome (based on table generated from load distance)
        - NaN for any pair with a genome missing from the table, or whose pair wasn't stored.

    Parameters
    ----------
    query_ids                                   : query genome id per pair (stripped of GB_/RS_).
    backbone_ids                                : backbone genome id per pair (stripped of GB_/RS_).
    map_id_idx, sorted_keys, sorted_dist, n     : from build_true_distance_lookup().
    logger                                      : logs how many pairs were dropped.
    """
    idx_q = np.array([map_id_idx.get(q, -1) for q in query_ids], dtype=np.int64)
    idx_b = np.array([map_id_idx.get(b, -1) for b in backbone_ids], dtype=np.int64)
    valid_id = (idx_q >= 0) & (idx_b >= 0)

    lo, hi = np.minimum(idx_q, idx_b), np.maximum(idx_q, idx_b)
    keys = lo * n + hi

    pos = np.clip(np.searchsorted(sorted_keys, keys), 0, len(sorted_keys) - 1)
    found = valid_id & (sorted_keys[pos] == keys)

    result = np.full(len(query_ids), np.nan, dtype=np.float64)
    result[found] = sorted_dist[pos[found]]

    n_missing = len(query_ids) - int(found.sum())
    if n_missing:
        logger.warning(f"{n_missing}/{len(query_ids)} pairs had no true distance available (missing genome id or pair not stored) -- dropped")
    return result

# =================================================================
# Load 1 fold only for each .hdf5 (prevent reloading)
# =================================================================
def load_split_group(path: Path, *group_path: str) -> dict[str, np.ndarray]:
    """Read only 1 split group from a fragment .hdf5."""
    with h5py.File(path, "r") as f:
        node = f
        for part in group_path:
            node = node[part]
        return hdf5._recursive_load_(node)

# =================================================================
# Create fragment -> whole genome CLR composition lookup
# =================================================================
def build_parent_lookup(whole_genome_split_path: Path) -> dict[str, np.ndarray]:
    """genome_id -> whole-genome CLR composition."""
    data = hdf5.load(whole_genome_split_path)
    lookup: dict[str, np.ndarray] = {}
    leaf_groups = [group for group in data.values() if isinstance(group, dict) and "leaf_id" in group and "feature" in group]
    leaf_groups += [subgroup for group in data.values() if isinstance(group, dict) for subgroup in group.values() if isinstance(subgroup, dict) and "leaf_id" in subgroup and "feature" in subgroup]
    for group in leaf_groups:
        for leaf_id, feature in zip(group["leaf_id"], group["feature"]):
            lookup.setdefault(leaf_id, feature)
    return lookup

# =================================================================
# Per-length, per-fold fragment dataset builder (denoiser training)
# =================================================================
def capped_subset(dataset: DataLoader_Denoiser, max_pairs: int | None, seed: int) -> Dataset:
    """Randomly subsample dataset down to max_pairs pairs (replicable with seed)."""
    if max_pairs is None or len(dataset) <= max_pairs:
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_pairs, replace=False)
    return Subset(dataset, indices.tolist())

def build_fragment_datasets(
    fragment_prefix: Path,
    lengths: list[int],
    fold_names: list[str],
    parent_lookup: dict[str, np.ndarray],
    max_pairs_per_length: int | None,
    seed: int,
    logger: logging.Logger | None = None,
) -> tuple[list[Dataset], dict[str, tuple[list[Dataset], list[Dataset]]]]:
    """
    Build fragment-pair datasets for CV over every fold, reading each length's .hdf5 file once.

    Parameters
    ----------
    fragment_prefix         : output {prefix}_frag{length}.hdf5 per length.
    lengths                 : fragment lengths in bp, longest-first.
    fold_names              : e.g. ["fold_1", "fold_2", ...].
    parent_lookup           : from build_parent_lookup(), the denoiser's reconstruction targets.
    max_pairs_per_length    : subsample cap per length/split, or None to use every pair.
    seed                    : capped_subset()'s subsampling seed.
    logger                  : logs per-length dataset sizes.
    """
    eval_datasets: list[Dataset] = []
    train_val_by_fold: dict[str, tuple[list[Dataset], list[Dataset]]] = {fold: ([], []) for fold in fold_names}

    for length in lengths:
        path = Path(f"{fragment_prefix}_frag{length}.hdf5")

        eval_split = load_split_group(path, "eval")
        eval_ds = capped_subset(DataLoader_Denoiser(eval_split, parent_lookup), max_pairs_per_length, seed)
        eval_datasets.append(eval_ds)
        if logger:
            logger.info(f"length={length} eval: {len(eval_ds)} pairs")

        for fold in fold_names:
            train_split = load_split_group(path, fold, "train")
            val_split   = load_split_group(path, fold, "val")
            train_ds = capped_subset(DataLoader_Denoiser(train_split, parent_lookup), max_pairs_per_length, seed)
            val_ds   = capped_subset(DataLoader_Denoiser(val_split, parent_lookup), max_pairs_per_length, seed)
            train_val_by_fold[fold][0].append(train_ds)
            train_val_by_fold[fold][1].append(val_ds)
            if logger:
                logger.info(f"length={length} {fold}: train={len(train_ds)} val={len(val_ds)} pairs")

    return eval_datasets, train_val_by_fold
