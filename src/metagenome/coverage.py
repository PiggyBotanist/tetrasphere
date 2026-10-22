from __future__ import annotations

# Standard Import
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# =================================================================
# Load
# =================================================================
def load_augmented_coverage(augmentation_folder: Path, n_views: int = 6, logger: logging.Logger | None = None) -> pd.Series:
    """
    Average per-contig coverage across COMEBin's own aug{0..n_views-1}_datacoverage_mean.tsv files
    (real, BAM-derived depth from COMEBin's own preprocessing, bootstrap-augmented into several
    views of the SAME single sample -- not a multi-sample abundance profile).

    Parameters
    ----------
    augmentation_folder  : e.g. settings.cami_archive / "comebin" / sample / "data_augmentation".
    n_views               : number of aug{i}_datacoverage_mean.tsv files to average (COMEBin writes 6).
    logger                 : if given, logs the median per-contig coefficient of variation across
                              views -- a sanity signal for how trustworthy the mean is, since these
                              are bootstrap views of one real sample, not independent replicates.
    """
    views = []
    for i in range(n_views):
        path = augmentation_folder / f"aug{i}_datacoverage_mean.tsv"
        series = pd.read_csv(path, sep="\t", skiprows=1, names=["ids", "coverage"], index_col="ids")["coverage"]
        views.append(series)

    table = pd.concat(views, axis=1)
    mean_coverage = table.mean(axis=1)

    if logger is not None:
        cv = (table.std(axis=1) / table.mean(axis=1).replace(0, np.nan)).median()
        logger.info(f"{augmentation_folder}: median per-contig coefficient of variation across {n_views} augmentation views = {cv:.3f}")

    return mean_coverage
