"""
Function: precomputes k-mer composition.
    - id: genome ids
    - feature: k-mer compositions (and additional features)

Usage
-----
    python scripts/tetrasphere/00_precompute_kmerGenome.py --domain ar53 --k 4 --transform clr --output-file <output_path>
"""
from __future__ import annotations

# Standard Import
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import hdf5, nwk
from src.system import create_logger
from src.tetrasphere.kmer import Kmer
from src.tetrasphere.preprocess import filter_genome

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Required fields
    parser.add_argument("--label",          required=True, help="help differentiate log.")
    parser.add_argument("--domain",         required=True, choices=["bac120", "ar53"], type=str, help="which GTDB tree to process.")
    parser.add_argument("--output-file",    required=True, type=Path, help="output .hdf5 file path.")
    parser.add_argument("--release",        default="232", type=str, help="GTDB release version, used to build file names.")
    parser.add_argument("--input-folder",   default=config.gtdb_download, help="folder for where representatives are (in .fasta).")

    # Filter procedure for GTDB representatives
    parser.add_argument("--min-length",         default=1000, type=int, help="drop records shorter than this before vectorization.")
    parser.add_argument("--min-completeness",   default=0.95, type=float, help="drop leaves with checkm2 completeness <= this value.")
    parser.add_argument("--max-contamination",  default=0.05, type=float, help="drop leaves with checkm2 contamination >= this value.")

    # Parameters
    parser.add_argument("--k",                  default=4, type=int, help="k-mer size to generate.")
    parser.add_argument("--transform",          default='clr', choices=["frequency", "clr"], type=str, help="composition transform to apply; raw counts if omitted.")

    # Experimental parameters
    parser.add_argument("--length",             action="store_true", help="flag for adding genome length.")

    # Parallelism
    parser.add_argument("--workers",            default=8, type=int, help="worker processes (each genome is independent work).")

    return parser.parse_args()

# =================================================================
# Parallelization
# =================================================================
_worker_state: dict = {}
def _init_worker(k: int, folder_representatives: Path, transform: str | None, length: bool, min_length: int) -> None:
    """Per-worker-process setup: build one Kmer encoder and cache static config, once per process
    instead of once per genome. Single-threaded torch per worker -- with --workers processes
    already saturating cores, torch's own default thread pool would badly oversubscribe (and its
    sync overhead dominates for tensor ops this small anyway)."""
    torch.set_num_threads(1)
    _worker_state.update(
        kmer=Kmer(k=k, logger=logging.getLogger("kmer_genome_worker")),
        folder=folder_representatives,
        transform=transform,
        length=length,
        min_length=min_length,
    )

def _process_genome(genome: str) -> tuple[str, np.ndarray | None]:
    """Vectorize one genome's k-mer composition; None if its representative file is missing."""
    file_path = _worker_state["folder"] / f"{genome}_genomic.fna.gz"
    if not file_path.exists():
        return genome, None
    kmer: Kmer = _worker_state["kmer"]
    vector = kmer.vectorize_fasta(
        file_path,
        include_length=_worker_state["length"],
        transform_kmer=_worker_state["transform"],
        min_length=_worker_state["min_length"],
    )
    return genome, vector

# Main
def main() -> None:
    args = parse_args()
    logger = create_logger(f"precompute_kmer_{args.label}", output_log=config.root_log)

    # Determine Paths
    file_tree               = args.input_folder / f"{args.domain}_r{args.release}.tree.gz"
    file_metadata           = args.input_folder / f"{args.domain}_metadata_r{args.release}.tsv.gz"
    folder_representatives  = args.input_folder / f"representative_genomes"

    # Step 1: Load data
    tree = nwk.load(file_tree)
    df_metadata = pd.read_csv(file_metadata, sep='\t', compression="gzip")

    # Step 2: Filtering
    nodes = nwk.index_nodes(tree)
    leaf_label, leaf_ids = nwk.extract_leaves(nodes)
    eligible_genomes = filter_genome(leaf_label, df_metadata, folder_representatives, args.min_completeness, args.max_contamination, logger)

    # Step 3: Compute & Save Kmer (each genome is independent work -- parallelize across processes)
    file, parameters = hdf5.open(args.output_file)
    n_missing = 0
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.k, folder_representatives, args.transform, args.length, args.min_length),
        ) as pool:
            for genome, vector in tqdm(pool.map(_process_genome, eligible_genomes), total=len(eligible_genomes), desc="Compute k-mer Composition"):
                if vector is None:
                    n_missing += 1
                    logger.warning(f"Missing genome: {folder_representatives / f'{genome}_genomic.fna.gz'}")
                    continue
                hdf5.add_row(file, parameters, id=genome, feature=vector)
    finally:
        hdf5.close(file, parameters)
    if n_missing:
        logger.warning(f"{n_missing}/{len(eligible_genomes)} genomes had no representative file -- skipped")

if __name__ == "__main__":
    main()
