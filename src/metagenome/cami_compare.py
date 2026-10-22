"""Load the four contig representations compared on CAMI samples: untrained 4-mer baseline, COMEBin's
own embedding, Tetrasphere whole-genome-stage embedding (no denoiser), Tetrasphere fragment-stage
embedding (with denoiser)."""
from __future__ import annotations

# Standard Import
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Custom Import
from config import config
from src.metagenome.embedding import load_contig_features, load_pretrained_model, detect_include_length, embed_features


def load_four_representations(
    sample: str,
    k: int,
    transform: str,
    min_contig_length: int,
    genome_experiment: str,
    genome_label: str,
    genome_fold: int,
    genome_epoch: int | None,
    fragment_experiment: str,
    fragment_label: str,
    fragment_fold: int,
    fragment_epoch: int | None,
    comebin_archive: Path | None,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, np.ndarray | list]:
    """
    Returns a dict with shared keys "ids"/"true_genome"/"length" (rows restricted to contigs present
    in every representation) and per-representation embeddings: "baseline", "comebin", "genome",
    "fragment".
    """
    comebin_archive = comebin_archive or (config.cami_archive / "comebin")

    genome_model = load_pretrained_model(genome_experiment, genome_label, genome_fold, genome_epoch, device)
    fragment_model = load_pretrained_model(fragment_experiment, fragment_label, fragment_fold, fragment_epoch, device)
    assert not detect_include_length(genome_model, k, logger), "genome checkpoint unexpectedly expects a length feature"
    assert detect_include_length(fragment_model, k, logger), "fragment checkpoint unexpectedly does not expect a length feature"

    data_noLength = load_contig_features(sample, k, transform, min_contig_length, False, logger)
    data_withLength = load_contig_features(sample, k, transform, min_contig_length, True, logger)
    assert list(data_noLength["ids"]) == list(data_withLength["ids"]), "contig order mismatch between the two feature variants"

    with torch.no_grad():
        embedding_genome = embed_features(genome_model, torch.from_numpy(data_noLength["features"]).float().to(device)).cpu().numpy()
        embedding_fragment = embed_features(fragment_model, torch.from_numpy(data_withLength["features"]).float().to(device)).cpu().numpy()

    path_embed = comebin_archive / sample / "comebin_res" / "embeddings.tsv.gz"
    df_comebin = pd.read_csv(path_embed, sep="\t", index_col=0)
    id_to_row = {cid: i for i, cid in enumerate(data_noLength["ids"])}
    shared_ids = [cid for cid in df_comebin.index if cid in id_to_row]
    rows = [id_to_row[cid] for cid in shared_ids]
    logger.info(f"{sample}: {len(shared_ids)} contigs shared across all four representations (>= {min_contig_length}bp)")

    return {
        "ids":         np.array(shared_ids, dtype=object),
        "true_genome": data_noLength["true_genome"][rows],
        "length":      data_noLength["length"][rows],
        "baseline":    data_noLength["features"][rows],
        "comebin":     df_comebin.loc[shared_ids].to_numpy(),
        "genome":      embedding_genome[rows],
        "fragment":    embedding_fragment[rows],
    }
