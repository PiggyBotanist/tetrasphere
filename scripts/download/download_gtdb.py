"""
Function: downloads 3 things from GTDB per type (archaea or bacteria).
    - The tree: contains nodes and edges with distance.
    - The metadata: contain genome information and accession link to NCBI.
    - The representative genome: contains all leaf representatives.

Usage
-----
    python project/download/gtdb.py --release 232 --type ar53 bac120
"""
from __future__ import annotations

# Standard Import
import logging
from pathlib import Path
import argparse

from tqdm import tqdm

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import tar, folder
from src.system import download_http, create_logger

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=str, default="232", help="which release of GTDB to download.")
    parser.add_argument("--type", nargs='+', type=str, default=['ar53', 'bac120'], help="which GTDB tree to download.")
    parser.add_argument("--output-folder", type=Path, default=config.gtdb_download, help="where to save the downloaded files.")
    return parser.parse_args()

# Main
def main() -> None:
    args = parse_args()
    logger = create_logger("download_gtdb", output_log=config.root_log)

    # Step 1: download files
    release = args.release
    gtdb_baseUrl = "https://data.gtdb.ecogenomic.org"
    gtdb_fileUrl = f"{gtdb_baseUrl}/releases/release{release}/{release}.0"
    files = [
        f"metadata_r{release}.tsv.gz",
        f"taxonomy_r{release}.tsv.gz",
        f"r{release}.tree.gz",
    ]
    for type in args.type:
        for f in files:
            download_http(f"{gtdb_fileUrl}/{type}_{f}", args.output_folder / f"{type}_{f}", logger)
        
    # Step 2: process representative genomes
    download_http(f"{gtdb_fileUrl}/genomic_files_reps/gtdb_genomes_reps_r{release}.tar.gz", args.output_folder / "representative_genomes.tar.gz", logger)
    tar.extract_archive(args.output_folder / "representative_genomes.tar.gz", args.output_folder / "representative_genomes", logger)
    folder.flatten(args.output_folder / "representative_genomes", "*.fna.gz", logger)
    folder.delete_subfolders(args.output_folder / "representative_genomes", logger, contain_files=True)

if __name__ == "__main__":
    main()
