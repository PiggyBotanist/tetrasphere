"""
Central path configuration. Every path defaults to a layout under DATA_ROOT, overridable via
environment variables so this repo runs unmodified on a different machine.

    export TETRASPHERE_DATA_ROOT=/path/to/data
    export TETRASPHERE_RESULT_ROOT=/path/to/results

Defaults below point at this project's own data/results on the machine the paper was produced on;
override via env vars (or edit directly) elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path


class Config:
    def __init__(self) -> None:
        self.root_project = Path(__file__).resolve().parent
        self.data_root = Path(os.environ.get("TETRASPHERE_DATA_ROOT", "/bls2/data"))
        self.result_root = Path(os.environ.get("TETRASPHERE_RESULT_ROOT", str(self.root_project / "results")))

        self.root_log = self.root_project / "log"

        # Results
        self.result_figure = self.root_project / "figures"
        self.result_table = self.result_root / "table"
        self.result_checkpoint = self.root_project / "checkpoints"

        # GTDB
        self.gtdb_download = self.data_root / "gtdb/download"
        self.gtdb_feature = self.data_root / "gtdb/feature"
        self.gtdb_split = self.data_root / "gtdb/split"

        # CAMI
        self.cami_root = self.data_root / "cami"
        self.cami_sequence = self.cami_root / "download/sequence"
        self.cami_metadata = self.cami_root / "download/metadata"
        self.cami_bam = self.cami_root / "download/bam"
        self.cami_archive = self.cami_root / "archive"

        # kf2vec (external tool's own working directory)
        self.kf2vec_root = self.data_root / "gtdb/kf2vec"

        # RefSeq (gene-level annotations, used only by fig3c_marker_gene_mds.py)
        self.refseq_root = self.data_root / "refseq/download"
        self.refseq_manifest = self.refseq_root / "refseq_downloaded.tsv.gz"
        self.refseq_metadata = self.refseq_root / "metadata"
        self.refseq_sequence = self.refseq_root / "sequence"

        # Taxonomy rank prefixes used by GTDB's semicolon-delimited taxonomy strings
        self.taxonomy_map = {
            "domain": "d__", "phylum": "p__", "class": "c__", "order": "o__",
            "family": "f__", "genus": "g__", "species": "s__",
        }

        # External tool conda environments (override if yours differ)
        self.env_kf2vec = os.environ.get("TETRASPHERE_KF2VEC_ENV", "kf2vec")


config = Config()
