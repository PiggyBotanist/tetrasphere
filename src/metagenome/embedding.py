"""Load CAMI contigs + ground truth, and a trained Tetrasphere checkpoint to embed them with."""
from __future__ import annotations

# Standard Import
import logging
import numpy as np
import pandas as pd
import torch
from torch import Tensor

# Custom Import
from config import config
from src.io import fasta
from src.model import checkpoint
from src.tetrasphere.kmer import Kmer
from src.tetrasphere.model import Manifold, PhylogenyModel

# =================================================================
# Load
# =================================================================
def load_contig_features(sample: str, k: int, transform: str, min_contig_length: int, include_length: bool, logger: logging.Logger) -> dict[str, np.ndarray]:
    """
    Load a CAMI sample's contigs, gold-standard genome-of-origin, and per-contig k-mer composition.

    Parameters
    ----------
    sample             : CAMI sample name (no extension).
    k                  : k-mer length.
    transform          : k-mer transform (frequency/clr); raw counts if None.
    min_contig_length  : drop contigs shorter than this.
    include_length     : append log10(contig length) as a trailing feature.
    logger             : logs unmapped/kept contig counts.
    """
    # Load files
    path_fasta = config.cami_sequence / f"{sample}.fasta.gz"
    path_metadata = config.cami_metadata / f"{sample}.tsv.gz"

    # Load records
    records = fasta.load(path_fasta, length_filter=min_contig_length)
    df_meta = pd.read_csv(path_metadata, sep="\t", compression="gzip")
    map_id_genome = dict(zip(df_meta["#anonymous_gsa_id"], df_meta["genome_id"].astype(str)))

    # Parse into k-mer feature
    kmer = Kmer(k=k, logger=logger)
    ids, feature_list, true_genome, length = [], [], [], []
    n_unmapped = 0
    for record in records:
        genome = map_id_genome.get(record.id)
        if genome is None:
            n_unmapped += 1
            continue
        ids.append(record.id)
        feature_list.append(kmer.vectorize(str(record.seq), include_length=include_length, transform_kmer=transform))
        true_genome.append(genome)
        length.append(len(record.seq))

    # Return information
    logger.info(f"{sample}: {len(ids)} contigs kept ({n_unmapped} unmapped, dropped), {len(set(true_genome))} true source genomes")
    return {
        "ids":         np.array(ids, dtype=object),
        "features":    np.stack(feature_list),
        "true_genome": np.array([str(v) for v in true_genome], dtype=object),
        "length":      np.array(length, dtype=np.int64),
    }

def load_contig_truth(sample: str, min_contig_length: int) -> pd.DataFrame:
    """Load ground truth table (ids/lengths/labels), no composition features."""
    # Load files
    path_fasta = config.cami_sequence / f"{sample}.fasta.gz"
    path_metadata = config.cami_metadata / f"{sample}.tsv.gz"

    # Load records
    records = fasta.load(path_fasta, length_filter=min_contig_length)
    df_meta = pd.read_csv(path_metadata, sep="\t", compression="gzip")
    map_id_genome = dict(zip(df_meta["#anonymous_gsa_id"], df_meta["genome_id"].astype(str)))

    # Parse to dataframe
    rows = [{"ids": r.id, "lengths": len(r.seq), "labels": str(map_id_genome[r.id])} for r in records if r.id in map_id_genome]
    return pd.DataFrame(rows)

def load_pretrained_model(experiment: str, label: str, fold: int, epoch: int | None, device: torch.device) -> Manifold | PhylogenyModel:
    """
    Load a trained checkpoint.
        - A genome-scale checkpoint is a standalone Manifold (dim_input/dim_hidden/dim_output/layers)
        - A fragment-scale checkpoint is a PhylogenyModel (dim_clean/dim_fragment/manifold_hidden/denoiser_hidden/...).
    """
    ckpt_path = checkpoint.find(config.result_checkpoint / experiment / label / f"fold_{fold}", epoch)
    model_config = checkpoint.load_config(ckpt_path)["model_config"]
    model_cls = PhylogenyModel if "dim_clean" in model_config else Manifold
    model = model_cls(**model_config).to(device)
    checkpoint.load_weights(ckpt_path, model, device)
    model.eval()
    return model

def detect_include_length(model: Manifold | PhylogenyModel, k: int, logger: logging.Logger) -> bool:
    """
    Whether this checkpoint's raw input is expected to include the trailing length dimension --
    true for a PhylogenyModel (dim_fragment = kmer_dim + 1, consumed by its Denoiser), false for a
    standalone Manifold (dim_input = kmer_dim, pure composition).
    """
    kmer_dim = Kmer(k=k, logger=logger).dim
    if isinstance(model, PhylogenyModel):
        if model.dim_fragment != kmer_dim + 1:
            raise ValueError(f"checkpoint dim_fragment={model.dim_fragment} != k-mer dim ({kmer_dim}) + 1 -- check k.")
        return True
    if isinstance(model, Manifold):
        if model.dim_input != kmer_dim:
            raise ValueError(f"checkpoint dim_input={model.dim_input} != k-mer dim ({kmer_dim}) -- check k.")
        return False
    raise TypeError(f"unrecognized model type {type(model)!r}")

def embed_features(model: Manifold | PhylogenyModel, features: Tensor) -> Tensor:
    """
    Embed raw contig features through whichever checkpoint type was loaded: a PhylogenyModel runs
    its Denoiser first (features include the trailing length dim, see detect_include_length), then
    the Manifold's encode(); a standalone Manifold encodes directly (no length dim, no denoiser).
    """
    if isinstance(model, PhylogenyModel):
        return model.embed(features, fragment=True)
    return model.encode(features)
