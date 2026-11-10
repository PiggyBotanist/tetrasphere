"""
Function: precomputes k-mer composition for fragments
    - fixed-length chunks with jitter
Attributes inside the .hdf5 file:
    - id: fragment ids ("{genome_id}__f{length}_{index:05d}").
    - feature: k-mer composition (+ log10-length column) per fragment.
Output file named as: {output-prefix}_frag{length}.hdf5
Usage
-----
    python scripts/tetrasphere/00_precompute_kmerFragment.py --domain ar53 --k 4 --transform clr --lengths 50000 25000 10000 5000 2500 1000 --output-prefix <output_prefix>
"""
from __future__ import annotations

# Standard Import
import math
import torch
import logging
import hashlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from concurrent.futures import ProcessPoolExecutor

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import hdf5, fasta, nwk
from src.system import create_logger
from src.tetrasphere.kmer import Kmer
from src.tetrasphere.preprocess import filter_genome

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Required fields
    parser.add_argument("--label",              required=True, help="help differentiate log.")
    parser.add_argument("--domain",              required=True, type=str, choices=["ar53", "bac120"], help="which GTDB tree to process.")
    parser.add_argument("--lengths",             required=True, type=int, nargs="+", help="space-separated fragment lengths in bp, e.g. 50000 25000 10000 5000 2500 1000.")
    parser.add_argument("--output-prefix",       required=True, type=Path, help="output path prefix; writes {output-prefix}_frag{length}.hdf5 per length.")
    parser.add_argument("--release",             default="232", type=str, help="GTDB release version, used to build file names.")
    parser.add_argument("--input-folder",        default=config.gtdb_download, type=Path, help="folder for where representatives are (in .fasta).")

    # Filter procedure for GTDB representatives (mirrors 00_precompute_kmerGenome.py)
    parser.add_argument("--min-length",          default=1000, type=int, help="drop records shorter than this before vectorization.")
    parser.add_argument("--min-completeness",    default=0.95, type=float, help="drop leaves with checkm2 completeness <= this value.")
    parser.add_argument("--max-contamination",   default=0.05, type=float, help="drop leaves with checkm2 contamination >= this value.")

    # k-mer parameters (mirrors 00_precompute_kmerGenome.py)
    parser.add_argument("--k",                   default=4, type=int, help="k-mer size to generate.")
    parser.add_argument("--transform",           default=None, choices=["frequency", "clr"], type=str, help="composition transform to apply; raw counts if omitted.")

    # Fragmentation parameters
    parser.add_argument("--jitter",              default=0.2, type=float, help="fractional bound on each fragment's own length around its target length (0.2 = +/-20%%).")
    parser.add_argument("--seed",                default=42, type=int, help="random seed (fragment length jitter + fragment subsampling).")

    # Per-genome fragment budget (sqrt-scaled -- see fragment_cap())
    parser.add_argument("--frag-cap-scale",      default=1.0, type=float, help="sqrt-scale multiplier on a genome's fragment budget: budget = ceil(scale * sqrt(genome_length // length)).")
    parser.add_argument("--frag-cap-max",        default=100, type=int, help="hard ceiling on fragments kept per genome per length, regardless of genome length.")

    # Parallelism
    parser.add_argument("--workers",             default=8, type=int, help="worker processes for fragmenting+vectorizing genomes (each genome is independent work).")

    # Optional: also stage each fragment's sequence as its own .fna, for the kf2vec benchmark at that length
    parser.add_argument("--stage-fasta",         action="store_true", help="also write each fragment's sequence as its own .fna.")
    parser.add_argument("--fasta-output-folder", default=None, type=Path, help="where to write staged fragment .fna files; required if --stage-fasta.")

    return parser.parse_args()

# =================================================================
# Per-genome fragment budget
# =================================================================
def fragment_cap(genome_length: int, target_length: int, scale: float, max_cap: int) -> int:
    """
    Sqrt-scaled fragment budget for one genome at one target length (prevent oversampling)
    """
    possible = max(genome_length // target_length, 1)
    return min(max_cap, max(1, math.ceil(scale * math.sqrt(possible))))

# =================================================================
# Parallelization
# =================================================================
_worker_state: dict = {}
def _init_worker(k: int, folder_representatives: Path, transform: str | None, jitter: float, cap_scale: float, cap_max: int, stage_fasta: bool, fasta_output_folder: Path | None) -> None:
    """Per-worker-process setup: build one Kmer encoder and cache static config, once per process instead of once per genome."""
    torch.set_num_threads(1)  # avoid oversubscribing CPUs across worker processes
    _worker_state.update(
        kmer=Kmer(k=k, logger=logging.getLogger("fragment_worker")),
        folder=folder_representatives,
        transform=transform,
        jitter=jitter,
        cap_scale=cap_scale,
        cap_max=cap_max,
        stage_fasta=stage_fasta,
        fasta_output_folder=fasta_output_folder,
    )

def _process_genome(payload: tuple[str, int, int]) -> tuple[str, list[tuple[str, np.ndarray]] | None]:
    """
    Chop one genome into a capped, subsampled set of fragments and vectorize each.

    Parameters
    ----------
    payload : (genome_id, length, base_seed) -- base_seed plus (length, genome_id) deterministically
              derive this genome's own RNG stream, so results don't depend on worker completion order.
    """
    genome_id, length, base_seed = payload
    source = _worker_state["folder"] / f"{genome_id}_genomic.fna.gz"
    if not source.exists():
        return genome_id, None

    records = fasta.load(source)
    genome_length = sum(len(r.seq) for r in records)
    cap = fragment_cap(genome_length, length, _worker_state["cap_scale"], _worker_state["cap_max"])

    seed = int(hashlib.sha256(f"{base_seed}-{length}-{genome_id}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    kmer: Kmer = _worker_state["kmer"]
    rows: list[tuple[str, np.ndarray]] = []
    for index, seq in fasta.fragment(records, length, rng, _worker_state["jitter"], max_fragments=cap):
        fragment_id = f"{genome_id}__f{length}_{index:05d}"
        vector = kmer.vectorize(seq, include_length=True, transform_kmer=_worker_state["transform"])
        rows.append((fragment_id, vector))
        if _worker_state["stage_fasta"]:
            fasta.save(_worker_state["fasta_output_folder"] / f"{fragment_id}.fna", [SeqRecord(Seq(seq), id=fragment_id, description="")])

    return genome_id, rows

# Main
def main() -> None:
    args = parse_args()
    logger = create_logger(f"precompute_kmerFragment_{args.label}", output_log=config.root_log)
    lengths = sorted(set(args.lengths), reverse=True)

    if args.stage_fasta and args.fasta_output_folder is None:
        raise ValueError("--fasta-output-folder is required when --stage-fasta is set.")
    if args.stage_fasta:
        args.fasta_output_folder.mkdir(parents=True, exist_ok=True)

    # Step 1: Same genome discovery as 00_precompute_kmerGenome.py -- independent of any split.
    file_tree              = args.input_folder / f"{args.domain}_r{args.release}.tree.gz"
    file_metadata           = args.input_folder / f"{args.domain}_metadata_r{args.release}.tsv.gz"
    folder_representatives  = args.input_folder / "representative_genomes"

    tree = nwk.load(file_tree)
    df_metadata = pd.read_csv(file_metadata, sep="\t", compression="gzip")

    nodes = nwk.index_nodes(tree)
    leaf_label, leaf_ids = nwk.extract_leaves(nodes)
    genome_ids = filter_genome(leaf_label, df_metadata, folder_representatives, args.min_completeness, args.max_contamination, logger)

    # Step 2: Chop + vectorize every eligible genome's fragments, per length
    for length in lengths:
        output_path = Path(f"{args.output_prefix}_frag{length}.hdf5")
        file, parameters = hdf5.open(output_path, flush_every=2048, chunk_size=256)

        payloads = [(genome_id, length, args.seed) for genome_id in genome_ids]
        total_fragments = 0
        try:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(args.k, folder_representatives, args.transform, args.jitter, args.frag_cap_scale, args.frag_cap_max, args.stage_fasta, args.fasta_output_folder),
            ) as pool:
                for genome_id, rows in tqdm(pool.map(_process_genome, payloads), total=len(payloads), desc=f"fragmenting genomes @ {length}bp"):
                    if rows is None:
                        logger.warning(f"Missing genome: {folder_representatives / f'{genome_id}_genomic.fna.gz'}")
                        continue
                    total_fragments += len(rows)
                    for fragment_id, vector in rows:
                        hdf5.add_row(file, parameters, id=fragment_id, feature=vector)
        finally:
            hdf5.close(file, parameters)

        logger.info(f"Length {length}: {total_fragments} fragments across {len(genome_ids)} genomes -> {output_path}")

if __name__ == "__main__":
    main()
