"""
Central configuration path. Either run in root or export this root path
> export TETRASPHERE_DATA_ROOT=</path/to/data>
> export TETRASPHERE_RESULT_ROOT=</path/to/results>
"""
from __future__ import annotations

from pathlib import Path
import os

class Config:
    def __init__(self) -> None:
        # Define root
        self.root_project = Path(__file__).resolve().parent
        self.data_root = Path(os.environ.get("TETRASPHERE_DATA_ROOT", str(self.root_project / "results")))
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

        # kf2vec
        self.kf2vec_root = self.data_root / "kf2vec"

        # (Optional) RefSeq: (gene-level annotations, used only by figure 3c)
        self.refseq_root = self.data_root / "refseq/download"
        self.refseq_manifest = self.refseq_root / "refseq_downloaded.tsv.gz"
        self.refseq_metadata = self.refseq_root / "metadata"
        self.refseq_sequence = self.refseq_root / "sequence"

        # Taxonomy rank prefixes used by GTDB's semicolon-delimited taxonomy strings
        self.taxonomy_map = {
            "domain": "d__", "phylum": "p__", "class": "c__", "order": "o__",
            "family": "f__", "genus": "g__", "species": "s__",
        }
        
        # External conda environments (kf2vec conda environment with their install)
        self.env_kf2vec = os.environ.get("TETRASPHERE_KF2VEC_ENV", "kf2vec")

config = Config()
