"""
Function: precompute pairwise distance.

Leaf-level (one row per genome):
    - leaf_id: GTDB genome id (contains RS_/GB_<ncbi_accession>).
    - rooted_distance_raw: raw GTDB root-to-tip distance.
    - rooted_distance_normalized: "rooted_distance_raw" divided by the tree's max root-to-tip
      distance, so every genome falls in [0, 1].
    - midpoint_rooted_distance_raw: root-to-tip distance re-rooted at the tree's midpoint.
    - midpoint_rooted_distance_normalized: "midpoint_rooted_distance_raw", same [0, 1] normalization.

Pair-level (one row per genome pair -- every C(n_leaves, 2) combination, this is the expensive part):
    - pairs: (i, j) leaf indices into the leaf-level table above.
    - pair_dist: patristic (tree-path) distance between the pair, normalized by the same constant.

Attributes: label (domain), normalization_constant (the max root-to-tip distance used above).

Usage
-----
    python scripts/tetrasphere/00_precompute_distance.py --domain ar53 --output-file <output_path>
"""
from __future__ import annotations

# Standard Import
import argparse
import math
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import hdf5, nwk
from src.system import create_logger
from src.tetrasphere.preprocess import filter_genome, strip_genomePrefix

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Required fields
    parser.add_argument("--domain",         required=True, choices=["bac120", "ar53"], type=str, help="which GTDB tree to process.")
    parser.add_argument("--output-file",    required=True, type=Path, help="output .hdf5 file path.")
    parser.add_argument("--release",        default="232", type=str, help="GTDB release version, used to build file names.")
    parser.add_argument("--input-folder",   default=config.gtdb_download, help="folder for where representatives are (in .fasta).")
    
    # Filter procedure for GTDB representatives
    parser.add_argument("--min-completeness",   default=0.95, type=float, help="drop leaves with checkm2 completeness <= this value.")
    parser.add_argument("--max-contamination",  default=0.05, type=float, help="drop leaves with checkm2 contamination >= this value.")
    
    return parser.parse_args()


# Main
def main() -> None:
    args = parse_args()
    logger = create_logger("precompute_pairwise_distance")

    # Determine Paths
    file_tree               = args.input_folder / f"{args.domain}_r{args.release}.tree.gz"
    file_metadata           = args.input_folder / f"{args.domain}_metadata_r{args.release}.tsv.gz"
    folder_representatives  = args.input_folder / f"representative_genomes"

    # Step 1: Load data
    df_metadata = pd.read_csv(file_metadata, sep='\t', compression="gzip")
    index = nwk.build_tree_index(file_tree)

    # Step 2: Filtering
    eligible_genomes = filter_genome(index.leaf_ids, df_metadata, folder_representatives, args.min_completeness, args.max_contamination, logger)
    index.leaf_ids = strip_genomePrefix(index.leaf_ids) 
    keep_mask = np.isin(index.leaf_ids, np.array(eligible_genomes))
    
    # Step 3: Load constants (norm_constant stays over the full tree; everything else narrows to keep_mask)
    norm_constant = float(index.root_distance[index.leaf_node_index].max())
    leaf_ids = index.leaf_ids[keep_mask]
    leaf_node_index = index.leaf_node_index[keep_mask]
    root_to_tip_raw = index.root_distance[leaf_node_index]
    n_leaves = len(leaf_ids)

    # Step 4: Save leaf-level distances
    file, parameters = hdf5.open(args.output_file, chunk_size=1024)
    hdf5.add_col(file,  "leaf_id", leaf_ids)
    hdf5.add_col(file,  "rooted_distance_raw", root_to_tip_raw)
    hdf5.add_col(file,  "rooted_distance_normalized", root_to_tip_raw / norm_constant)
    hdf5.add_col(file,  "midpoint_rooted_distance_raw", index.midpoint_root_distance[keep_mask])
    hdf5.add_col(file,  "midpoint_rooted_distance_normalized", index.midpoint_root_distance[keep_mask] / norm_constant)
    hdf5.set_attr(file, "label", args.domain)
    hdf5.set_attr(file, "normalization_constant", norm_constant)

    # Step 4: Compute & Save Pairwise Distance
    try:
        pairs = combinations(range(n_leaves), 2)
        for i, j in tqdm(pairs, total=math.comb(n_leaves, 2), desc="Compute Pairwise Distance"):
            distance = nwk.patristic_distance(leaf_node_index[i], leaf_node_index[j], index.lift_table, index.topological_depth, index.root_distance)
            hdf5.add_row(file, parameters, pairs=np.array([i, j], dtype=np.int32), pair_dist=np.float32(distance / norm_constant))
    finally:
        hdf5.close(file, parameters)
    
if __name__ == "__main__":
    main()