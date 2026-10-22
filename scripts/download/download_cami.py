"""
Function: downloads CAMI simulated datasets:
    - toy: 1 low, 4 medium, 5 high
    - air-skin-urogenital: 28 samples
    - gastro-oral: 19 samples
    - marine: 9 samples
    - plant-associated: 20 samples
    - strain: 99 samples
For each file, download:
    - metadata <ground_truth, read-count>
    - sequence <contigs and its gold-standard assembly sequence>
    - fastq <original .fastq reads>
    - bam <directly converted from fastq, and fastq is dropped>

Note there are 3 separate format total so it is complex:
    - toy: multiple samples in one .tar.
    - bodysites: 1 .tar per sample.
    - environments: 1 .tar for metadata, 1 .tar for raw reads.

Usage
-----
    python project/download/cami.py --dataset all
"""
from __future__ import annotations

# Standard Import
import re
import time
import logging
import argparse
import requests
import pandas as pd
from pathlib import Path

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.system import create_logger, download_http, multithread
from src.io import tar, fastq, bam

# ===============================================================
# Config / URLs
# ===============================================================
URL_TOY = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100344/ToyTestDatasets.dir/"
URL_BODYSITE    = "https://frl.publisso.de/data/frl:6425518/"
URL_ENVIRONMENT = "https://frl.publisso.de/data/frl:6425521/"
BODYSITES       = {"airskinurogenital": 28, "gastrooral": 19}
ENVIRONMENTS    = {"marine/marmgCAMI2_": 9, "plant_associated/rhimgCAMI2_": 20, "strain/strmgCAMI2_": 99}
PREFIX = {'marine': 'marmgCAMI2_', 'plant_associated': "rhimgCAMI2_", "strain": "strmgCAMI2_"}

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Datasets to download
    parser.add_argument("--toy",         action="store_true", help="download the toy datasets.")
    parser.add_argument("--bodysite",    action="store_true", help="download the bodysite datasets.")
    parser.add_argument("--environment", action="store_true", help="download the environmental datasets.")
    
    # Where to store
    parser.add_argument("--metadata",    default=config.cami_metadata, type=Path, help="output folder for .tsv.gz metadata.")
    parser.add_argument("--sequence",    default=config.cami_sequence, type=Path, help="output folder for .fasta.gz sequences.")
    parser.add_argument("--bam",         default=config.cami_bam, type=Path, help="output folder for .bam alignments.")
    
    # Parameters
    parser.add_argument("--threads", default=8, type=int, help="threads for minimap2/samtools per sample.")
    parser.add_argument("--workers", default=1, type=int, help="concurrent samples to download/align.")
    parser.add_argument("--n-samples", default=None, type=int, help="only process the first N samples of each bodysite/environment type (toy is always processed in full).")
    parser.add_argument("--remove-tar", action="store_true", help="delete each downloaded tar once its samples are processed.")
    return parser.parse_args()

# ===============================================================
# TOY
# ===============================================================
def process_sample(file_tar: Path, name: str, member_fasta: str, member_metadata: str, member_reads: str, folder_metadata: Path, folder_sequence: Path, folder_bam: Path, logger, threads: int) -> None:
    """Extract one sample's fasta + metadata from an already-downloaded tar, then align its reads into a BAM."""
    file_fasta    = folder_sequence / f"{name}.fasta.gz"
    file_metadata = folder_metadata / f"{name}.tsv.gz"
    file_bam      = folder_bam / name / f"{name}.bam"

    if file_fasta.exists() and file_metadata.exists() and bam.verify(file_bam):
        logger.info("[%s] all outputs exist, skipping", name)
        return

    if not file_fasta.exists():
        tar.extract_file(file_tar, file_fasta, _member_(member_fasta), logger)
    if not file_metadata.exists():
        tar.extract_file(file_tar, file_metadata, _member_(member_metadata), logger)
    if not bam.verify(file_bam):
        file_fastq = folder_sequence / f"{name}.fastq.gz"
        tar.extract_file(file_tar, file_fastq, _member_(member_reads), logger)
        fastq.to_bam(file_fasta, file_fastq, file_bam, logger, threads=threads)
        file_fastq.unlink()
    logger.info("[%s] done", name)

def download_toy(folder_metadata: Path, folder_sequence: Path, folder_bam: Path, logger, threads: int, workers: int, remove_tar: bool = False) -> None:
    """Download and process the three CAMI toy datasets (low/medium/high complexity)."""
    medium = [("M1", 5000, "s001"), ("M1", 5000, "s002"), ("M2", 180, "s001"), ("M2", 180, "s002")]
    high   = ["s001", "s002", "s003", "s004", "s005"]

    # Samples sharing a tar, grouped by tar filename
    groups: dict[str, list[tuple[str, str, str, str]]] = {
        "30_genomes.tar": [(
            "cami_toy_low",
            "S_S001__genomes_30__insert_180_gsa_anonymous.fasta.gz",
            "S_S001__genomes_30__insert_180_gsa_mapping.tsv.gz",
            "S_S001__genomes_30__insert_180_reads_anonymous.fq.gz",
        )],
        "225_genomes.tar": [
            (
                f"cami_toy_medium_{prefix.lower()}_{sid}",
                f"{prefix}_S{sid[1:]}__insert_{insert}_gsa_anonymous.fasta.gz",
                f"{prefix}_S{sid[1:]}__insert_{insert}_gsa_mapping.tsv.gz",
                f"{prefix}_S{sid[1:]}__insert_{insert}_reads_anonymous.fq.gz",
            )
            for prefix, insert, sid in medium
        ],
        "450_genomes.tar": [
            (
                f"cami_toy_high_{sid}",
                f"H_S{sid[1:]}__insert_180_gsa_anonymous.fasta.gz",
                f"H_S{sid[1:]}__insert_180_gsa_mapping.tsv.gz",
                f"H_S{sid[1:]}__insert_180_reads_anonymous.fq.gz",
            )
            for sid in high
        ],
    }

    # Step 1: Download each shared tar once, skipping entirely if all its samples are already done
    args:  list[tuple] = []
    names: list[str]   = []
    for tar_name, samples in groups.items():
        if _all_done_([s[0] for s in samples], folder_sequence, folder_metadata, folder_bam):
            logger.info("[%s] all samples already done, skipping download", tar_name)
            continue
        file_tar = _ensure_tar_(URL_TOY + tar_name, config.cami_root / tar_name, logger)
        for name, member_fasta, member_metadata, member_reads in samples:
            args.append((file_tar, name, member_fasta, member_metadata, member_reads, folder_metadata, folder_sequence, folder_bam, logger, threads))
            names.append(name)

    # Step 2: Extract + align every sample, optionally in parallel
    multithread(process_sample, args, names, workers, logger)

    # Step 3: Optionally drop the shared tars once every sample in them is done
    if remove_tar:
        for tar_name, samples in groups.items():
            if _all_done_([s[0] for s in samples], folder_sequence, folder_metadata, folder_bam):
                _remove_tar_(config.cami_root / tar_name, logger)

# ===============================================================
# BODYSITE
# ===============================================================
def process_bodysite_sample(bodysite: str, i: int, folder_metadata: Path, folder_sequence: Path, folder_bam: Path, logger, threads: int, remove_tar: bool = False) -> None:
    """Download one bodysite sample's own tar, extract, and align."""
    name          = f"cami_{bodysite}_S{i:03d}"
    file_fasta    = folder_sequence / f"{name}.fasta.gz"
    file_metadata = folder_metadata / f"{name}.tsv.gz"
    file_bam      = folder_bam / name / f"{name}.bam"

    if file_fasta.exists() and file_metadata.exists() and bam.verify(file_bam):
        logger.info("[%s] all outputs exist, skipping", name)
        return

    file_tar = _ensure_tar_(f"{URL_BODYSITE}{bodysite}/sample_{i}.tar.gz", config.cami_root / f"{name}.tar.gz", logger)

    if not file_fasta.exists():
        tar.extract_file(file_tar, file_fasta, _member_("anonymous_gsa.fasta.gz"), logger)
    if not file_metadata.exists():
        tar.extract_file(file_tar, file_metadata, _member_("gsa_mapping.tsv.gz"), logger)
    if not bam.verify(file_bam):
        file_fastq = folder_sequence / f"{name}.fastq.gz"
        # Some bodysite tars (e.g. gastrooral sample_0) ship the reads uncompressed; minimap2 reads either transparently
        tar.extract_file(file_tar, file_fastq, r"(^|/)anonymous_reads\.fq(\.gz)?$", logger)
        fastq.to_bam(file_fasta, file_fastq, file_bam, logger, threads=threads)
        file_fastq.unlink()

    if remove_tar:
        _remove_tar_(file_tar, logger)
    logger.info("[%s] done", name)

def download_bodysite(folder_metadata: Path, folder_sequence: Path, folder_bam: Path, logger, threads: int, workers: int, remove_tar: bool = False, n_samples: int | None = None) -> None:
    """Download and process every CAMI bodysite sample (or the first n_samples of each)."""
    samples = [(bodysite, i) for bodysite, count in BODYSITES.items() for i in range(min(count, n_samples or count))]
    args    = [(bodysite, i, folder_metadata, folder_sequence, folder_bam, logger, threads, remove_tar) for bodysite, i in samples]
    names   = [f"cami_{bodysite}_S{i:03d}" for bodysite, i in samples]
    multithread(process_bodysite_sample, args, names, workers, logger)

# ===============================================================
# ENVIRONMENT
# ===============================================================
def process_environment_sample(env_path: str, env_prefix: str, i: int, folder_metadata: Path, folder_sequence: Path, folder_bam: Path, logger, threads: int, remove_tar: bool = False) -> None:
    """Download one environment sample's contigs + reads tars (separate archives), extract, and align."""
    name          = f"cami_{env_path}_S{i:03d}"
    file_fasta    = folder_sequence / f"{name}.fasta.gz"
    file_metadata = folder_metadata / f"{name}.tsv.gz"
    file_bam      = folder_bam / name / f"{name}.bam"

    if file_fasta.exists() and file_metadata.exists() and bam.verify(file_bam):
        logger.info("[%s] all outputs exist, skipping", name)
        return

    base_url = f"{URL_ENVIRONMENT}{env_path}/short_read/{env_prefix}sample_{i}"

    if not file_fasta.exists() or not file_metadata.exists():
        contigs_tar = _ensure_tar_(f"{base_url}_contigs.tar.gz", config.cami_root / f"{name}_contigs.tar.gz", logger)
        if not file_fasta.exists():
            tar.extract_file(contigs_tar, file_fasta, _member_("anonymous_gsa.fasta.gz"), logger)
        if not file_metadata.exists():
            tar.extract_file(contigs_tar, file_metadata, _member_("gsa_mapping.tsv.gz"), logger)
        if remove_tar:
            _remove_tar_(contigs_tar, logger)

    if not bam.verify(file_bam):
        reads_tar  = _ensure_tar_(f"{base_url}_reads.tar.gz", config.cami_root / f"{name}_reads.tar.gz", logger)
        file_fastq = folder_sequence / f"{name}.fastq.gz"
        tar.extract_file(reads_tar, file_fastq, _member_("anonymous_reads.fq.gz"), logger)
        fastq.to_bam(file_fasta, file_fastq, file_bam, logger, threads=threads)
        file_fastq.unlink()
        if remove_tar:
            _remove_tar_(reads_tar, logger)

    logger.info("[%s] done", name)

def download_environment(folder_metadata: Path, folder_sequence: Path, folder_bam: Path, logger, threads: int, workers: int, remove_tar: bool = False, n_samples: int | None = None) -> None:
    """Download and process every CAMI environmental sample (or the first n_samples of each)."""
    samples = [(env_path, env_prefix, i) for env_key, count in ENVIRONMENTS.items() for env_path, env_prefix in [env_key.split("/", 1)] for i in range(min(count, n_samples or count))]
    args    = [(env_path, env_prefix, i, folder_metadata, folder_sequence, folder_bam, logger, threads, remove_tar) for env_path, env_prefix, i in samples]
    names   = [f"cami_{env_path}_S{i:03d}" for env_path, _, i in samples]
    multithread(process_environment_sample, args, names, workers, logger)

# ===============================================================
# CLEANUP
# ===============================================================
def standardized_naming(folder: Path, logger) -> None:
    """
    Standardize CAMI metadata column-naming discrepancies:
        #anonymous_gsa_id     : used by most datasets
        #anonymous_contig_id  : used by toy datasets -> renamed to #anonymous_gsa_id
    """
    for f in sorted(folder.glob("*.tsv.gz")):
        logger.info(f"standardizing {f.name}...")
        df = pd.read_csv(f, sep="\t", compression="gzip")
        df = df.drop(columns=["start_position", "end_position"], errors="ignore")
        df = df.rename(columns={"#anonymous_contig_id": "#anonymous_gsa_id"})
        df.to_csv(f, sep="\t", index=False, compression="gzip")

# ===============================================================
# INTERNAL
# ===============================================================
def _all_done_(names: list[str], folder_sequence: Path, folder_metadata: Path, folder_bam: Path) -> bool:
    """Return True if every name already has a fasta, metadata, and verified bam."""
    return all(
        (folder_sequence / f"{n}.fasta.gz").exists()
        and (folder_metadata / f"{n}.tsv.gz").exists()
        and bam.verify(folder_bam / n / f"{n}.bam")
        for n in names
    )

def _ensure_tar_(url: str, file_tar: Path, logger, retries: int = 10) -> Path:
    """Download url to file_tar, resuming a partial file (download_http skips it if already complete); return the local path."""
    for attempt in range(1, retries + 1):
        try:
            # Some servers answer 200 (full resend) instead of 416 for a range at EOF, so check size first
            if file_tar.exists() and file_tar.stat().st_size == _remote_size_(url):
                logger.info("%s already fully downloaded, skipped.", file_tar.name)
                return file_tar
            download_http(url, file_tar, logger, show_tqdm=False)
            return file_tar
        except Exception as e:
            if attempt == retries:
                raise
            logger.warning("%s: download attempt %d/%d failed (%s); resuming in 60s", file_tar.name, attempt, retries, e)
            time.sleep(60)
    return file_tar

def _member_(name: str) -> str:
    """Regex matching a tar member by its exact basename."""
    return rf"(^|/){re.escape(name)}$"

def _remote_size_(url: str) -> int | None:
    """Content-Length of url (following redirects), or None if the server does not report it."""
    with requests.head(url, allow_redirects=True, timeout=60) as response:
        size = response.headers.get("Content-Length")
        return int(size) if size else None

def _remove_tar_(file_tar: Path, logger) -> None:
    """Delete a processed tar to free disk space."""
    if file_tar.exists():
        file_tar.unlink()
        logger.info("removed %s", file_tar.name)

# ===============================================================
# CLI
# ===============================================================
def main() -> None:
    args = parse_args()
    logger = create_logger("download cami dataset")

    config.cami_root.mkdir(parents=True, exist_ok=True)
    args.metadata.mkdir(parents=True, exist_ok=True)
    args.sequence.mkdir(parents=True, exist_ok=True)
    args.bam.mkdir(parents=True, exist_ok=True)

    run_all = not any([args.toy, args.bodysite, args.environment])

    if args.toy or run_all:
        download_toy(args.metadata, args.sequence, args.bam, logger, args.threads, args.workers, args.remove_tar)
    if args.bodysite or run_all:
        download_bodysite(args.metadata, args.sequence, args.bam, logger, args.threads, args.workers, args.remove_tar, args.n_samples)
    if args.environment or run_all:
        download_environment(args.metadata, args.sequence, args.bam, logger, args.threads, args.workers, args.remove_tar, args.n_samples)

    standardized_naming(args.metadata, logger)

if __name__ == "__main__":
    main()
