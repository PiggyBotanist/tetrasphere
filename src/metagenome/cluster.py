"""Embedding -> clusters -> scored-summary utility, with unsupervised (label-free) hyperparameter selection."""
from __future__ import annotations

# Standard Import
from collections.abc import Callable
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Custom Import
from src.metagenome import metric

# =================================================================
# Cluster
# =================================================================
def grid_search_dbscan(
    embedding: np.ndarray,
    true_genome: np.ndarray,
    length: np.ndarray,
    min_contig_length: int,
    eps_grid: list[float],
    min_samples_grid: list[int],
    feature_weights: np.ndarray | None = None,
    score_fn: Callable[..., pd.DataFrame] = metric.bin_summary,
) -> tuple[pd.DataFrame, dict[str, float], np.ndarray]:
    """
    Grid-search DBSCAN(eps, min_samples) on a standardized embedding.

    Hyperparameters are selected by maximizing silhouette score, computed only from the embedding
    and each candidate's own predicted labels -- true_genome is never used for selection, only to
    score the one winning configuration afterward (via score_fn). This avoids the label leakage of
    selecting hyperparameters directly against ground truth.

    Parameters
    ----------
    embedding          : (n_contigs, dim) model output to cluster.
    true_genome        : gold-standard genome-of-origin per contig (used only for final scoring).
    length             : contig length in bp.
    min_contig_length  : passed through to score_fn()'s recovery-eligibility filter.
    eps_grid           : DBSCAN eps values to try.
    min_samples_grid   : DBSCAN min_samples values to try.
    feature_weights    : (dim,) optional per-column weight applied AFTER standardization.
    score_fn            : scores one {labels, predictions, lengths} df -> one-row summary df.
                          Defaults to metric.bin_summary(df, metric.QUALITY_TIERS, min_length=min_contig_length).

    Returns
    -------
    (best_summary, best_params, best_labels) -- best_labels is the winning DBSCAN assignment
    (-1 = noise/unclassified, else a cluster id), same row order as embedding/true_genome/length.
    """
    scaled = StandardScaler().fit_transform(embedding)
    if feature_weights is not None:
        scaled = scaled * feature_weights

    best_params, best_labels, best_silhouette = None, None, -1.0
    for eps in eps_grid:
        for min_samples in min_samples_grid:
            pred_label = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
            n_clusters = len(set(pred_label) - {-1})
            n_assigned = (pred_label != -1).sum()
            if n_clusters < 2 or n_assigned < 2:
                continue

            sil = silhouette_score(scaled[pred_label != -1], pred_label[pred_label != -1])
            if sil > best_silhouette:
                best_silhouette, best_params, best_labels = sil, {"eps": eps, "min_samples": min_samples}, pred_label

    if best_labels is None:
        raise RuntimeError("DBSCAN produced fewer than 2 clusters for every (eps, min_samples) in the grid -- widen eps_grid.")

    df_eval = pd.DataFrame({"labels": true_genome, "predictions": best_labels, "lengths": length})
    best_summary = score_fn(df_eval, metric.QUALITY_TIERS, min_length=min_contig_length)
    best_params["silhouette"] = float(best_silhouette)
    return best_summary, best_params, best_labels
