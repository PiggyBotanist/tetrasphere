from __future__ import annotations
from typing import Any

# Standard Import
import numpy as np
import pandas as pd
from scipy import stats

def r_squared(pred: np.ndarray, label: np.ndarray, weight: np.ndarray | None = None) -> float:
    """(optionally weighted) Formula: R^2 = 1 - [(label - pred)^2 / (label - mean_pred)^2]"""
    # Default weight to 1 (uniform)
    w = np.ones(len(label), dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    weighted_mean = np.average(label, weights=w)
    ss_res = np.sum(w * (label - pred) ** 2)
    ss_tot = np.sum(w * (label - weighted_mean) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def pairwise_errors(pred: np.ndarray, label: np.ndarray, distortion_epsilon: float = 1e-6) -> pd.DataFrame:
    """Per-pair absolute error and relative distortion (epsilon-floored), flagging near-zero-true pairs."""
    abs_error  = np.abs(pred - label)
    distortion = abs_error / np.maximum(label, distortion_epsilon)
    return pd.DataFrame({"abs_error": abs_error, "distortion": distortion, "near_zero": label < distortion_epsilon})

def weighted_pearsonr(pred: np.ndarray, label: np.ndarray, weight: np.ndarray | None = None) -> float:
    """(optionally weighted) Formula: Pearson's r = Cov(x,y) / sqrt(var(x)*var(y))"""
    w = np.ones(len(label), dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    mean_pred, mean_label = np.average(pred, weights=w), np.average(label, weights=w)
    cov       = np.average((pred - mean_pred) * (label - mean_label), weights=w)
    var_pred  = np.average((pred - mean_pred) ** 2, weights=w)
    var_label = np.average((label - mean_label) ** 2, weights=w)
    denom = np.sqrt(var_pred * var_label)
    return float(cov / denom) if denom > 0 else float("nan")

def aggregate_metrics(pred: np.ndarray, true: np.ndarray, distortion_epsilon: float = 1e-6, weight: np.ndarray | None = None) -> dict[str, Any]:
    """(optionally weigthed) Aggregates important metric"""
    errors = pairwise_errors(pred, true, distortion_epsilon)
    w = np.ones(len(pred), dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)

    # NOTE: drops near zero pairs before averaging distortion
    keep_distortion = ~errors["near_zero"]
    distortion, w_distortion = errors.loc[keep_distortion, "distortion"].to_numpy(), w[keep_distortion.to_numpy()]

    if len(pred) < 2:
        pearson_r, spearman_r = float("nan"), float("nan")
    else:
        pearson_r     = weighted_pearsonr(pred, true, w)
        spearman_r, _ = stats.spearmanr(pred, true)

    return {
        "n_pairs":           len(pred),
        "n_effective":       float(np.sum(w) ** 2 / np.sum(w ** 2)),
        "n_near_zero":       int(errors["near_zero"].sum()),
        "mse":               float(np.average(errors["abs_error"] ** 2, weights=w)),
        "mean_abs_error":    float(np.average(errors["abs_error"], weights=w)),
        "median_abs_error":  weighted_quantile(errors["abs_error"].to_numpy(), w, 0.5),
        "mean_distortion":   float(np.average(distortion, weights=w_distortion)) if len(distortion) else float("nan"),
        "median_distortion": weighted_quantile(distortion, w_distortion, 0.5) if len(distortion) else float("nan"),
        "r_squared":         r_squared(pred, true, w),
        "pearson_r":         pearson_r,
        "spearman_r":        spearman_r,
    }

def weighted_quantile(values: np.ndarray, weight: np.ndarray, q: float) -> float:
    """Weighted quantile via cumulative-weight interpolation (Hazen-type); reduces to the ~usual median for q=0.5, uniform weight."""
    order = np.argsort(values)
    values, weight = values[order], weight[order]
    cum_weight = (np.cumsum(weight) - 0.5 * weight) / np.sum(weight)
    return float(np.interp(q, cum_weight, values))
