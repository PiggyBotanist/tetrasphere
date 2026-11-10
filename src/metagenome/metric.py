from __future__ import annotations

# Standard Import
import numpy as np
import pandas as pd
from pathlib import Path

# =================================================================
# Define Quality Filter (Metagenomic Binning)
# =================================================================
QUALITY_TIERS = {
    f"tier_{letter}": {"completeness": completeness_cutoff / 100, "contamination": contamination_cutoff / 100}
    for letter, completeness_cutoff, contamination_cutoff in zip("ABCDEF", range(95, 65, -5), range(5, 35, 5))
}

# =================================================================
# Build a Matrix (n, m)
# - n: number of ground truth bin
# - m: number of prediction bin
# - cell value: number of bp grouped into that region
# =================================================================
def bp_contingency(df: pd.DataFrame) -> pd.DataFrame:
    """Build the (truth bin x predicted bin) basepair contingency table."""
    df = df.copy()

    # Ensure unclassified is labelled with -1
    df["predictions"] = df["predictions"].fillna(-1)

    # Produce Contigency Table (Matrix)
    return (
        df.groupby(
            ["labels", "predictions"]
        )["lengths"]
        .sum()
        .unstack(fill_value=0)
    )

# =================================================================
# Compute Theoretical Upper Bound for Bins
# =================================================================
def perfect_assignment(df: pd.DataFrame, min_length: int = 1000) -> pd.DataFrame:
    """
    Theoretical upper bound (for a minimum contig length threshold bin may not be able
    to reach 100% completeness due to filter).

    Contigs >= min_length are assigned to their true label.
    Contigs < min_length are left unclassified.
    """
    oracle = df.copy()

    oracle["predictions"] = np.where(
        oracle["lengths"] >= min_length,
        oracle["labels"],   # perfect prediction
        -1                  # filtered out
    )

    return oracle

# =================================================================
# Per-truth-cluster contig counts
# =================================================================
def contig_counts(df: pd.DataFrame, label_to_bin: pd.Series, bin_to_dominant: pd.Series, min_length: int = 1000) -> pd.DataFrame:
    """Per-truth-cluster contig counts: total, length-pass, correctly binned, misclassified, unclassified."""
    df = df.copy()
    df["predictions"] = df["predictions"].fillna(-1)

    matched_bin     = df["labels"].map(label_to_bin)
    dominant_genome = df["predictions"].map(bin_to_dominant)
    unclassified    = df["predictions"] == -1
    correct         = (~unclassified) & (df["predictions"] == matched_bin) & (dominant_genome == df["labels"])
    misclassified   = (~unclassified) & ~correct

    g = df.assign(
        _pass=df["lengths"] >= min_length,
        _correct=correct, _mis=misclassified, _unc=unclassified,
    ).groupby("labels")

    out = pd.DataFrame({
        "total_contigs":            g.size(),
        "contigs_pass_filter":      g["_pass"].sum(),
        "contigs_binned_correctly": g["_correct"].sum(),
        "contigs_misclassified":    g["_mis"].sum(),
        "contigs_unclassified":     g["_unc"].sum(),
    })
    return out.reset_index(names="truth_cluster_id")

# =================================================================
# Completeness (basepair weighted)
# =================================================================
def completeness(table: pd.DataFrame) -> pd.DataFrame:
    """Basepair-weighted completeness of each truth cluster's best-matching predicted bin."""
    classified = table.drop(columns=-1, errors="ignore")

    truth_bp = table.sum(axis=1)              # total incl. unclassified
    recovered_bp = classified.max(axis=1)     # bp in the best (matched) bin
    unclassified_bp = table[-1] if -1 in table.columns else 0

    prediction = classified.idxmax(axis=1).where(recovered_bp > 0)
    
    result = pd.DataFrame({
        "truth_bp": truth_bp,
        "prediction": prediction,
        "recovered_bp": recovered_bp,
        "unclassified_bp": unclassified_bp,
    })
    # genome bp that ended up in some OTHER classified bin
    result["misclassified_bp"] = (
        result["truth_bp"] - result["recovered_bp"] - result["unclassified_bp"]
    )
    result["completeness"] = result["recovered_bp"] / result["truth_bp"]
    return result.reset_index(names="label")

# =================================================================
# Contamination (basepair weighted)
# =================================================================
def contamination(table: pd.DataFrame) -> pd.DataFrame:
    """Basepair-weighted contamination of each predicted bin (share of bp not from its dominant genome)."""
    # ignore unclassified
    table = table.drop(columns=-1, errors="ignore")

    bin_bp = table.sum(axis=0)

    result = pd.DataFrame({
        "bin_bp": bin_bp,
        "dominant_label": table.idxmax(axis=0),
        "dominant_bp": table.max(axis=0)
    })

    result["contamination"] = 1 - result["dominant_bp"] / result["bin_bp"]
    return result.reset_index(names="prediction")

# =================================================================
# Per Bin Report (Completeness/Contamination)
# =================================================================
def bin_quality(table: pd.DataFrame) -> pd.DataFrame:
    """Per-truth-cluster completeness + contamination report, joined onto its matched predicted bin."""
    comp = completeness(table)

    classified = table.drop(columns=-1, errors="ignore")
    bin_bp = classified.sum(axis=0)
    # sanity check: recovered_bp / total_bin_bp = 1 - contamination
    comp["total_bin_bp"] = comp["prediction"].map(bin_bp)
    comp["contamination"] = 1 - comp["recovered_bp"] / comp["total_bin_bp"]

    report = comp.rename(columns={
        "label": "truth_cluster_id",
        "prediction": "bin_cluster_id",
        "truth_bp": "total_truth_bp",
        "recovered_bp": "total_correctly_recovered_bp",
    })

    columns = [
        "truth_cluster_id",
        "bin_cluster_id",
        "completeness",
        "contamination",
        "total_truth_bp",
        "total_correctly_recovered_bp",
        "total_bin_bp",
        "misclassified_bp",
        "unclassified_bp",
    ]
    return report[columns]

def bin_metrics(df: pd.DataFrame, min_length: int = 1000) -> pd.DataFrame:
    """Full per-truth-cluster report: bin_quality() merged with contig_counts(), filtered/unclustered bins labeled."""
    table = bp_contingency(df)
    report = bin_quality(table)
    label_to_bin = report.set_index("truth_cluster_id")["bin_cluster_id"]
    bin_to_dominant = contamination(table).set_index("prediction")["dominant_label"]
    counts = contig_counts(df, label_to_bin, bin_to_dominant, min_length)
    merged = report.merge(counts, on="truth_cluster_id", how="left")
    not_recovered = merged["bin_cluster_id"].isna()
    filtered_out  = not_recovered & (merged["contigs_pass_filter"] == 0)
    unclustered   = not_recovered & (merged["contigs_pass_filter"] > 0)

    def _restore_int(v):
        if pd.isna(v):
            return v
        try:
            return int(v)
        except (TypeError, ValueError):
            return v
    bin_ids = [_restore_int(v) for v in merged["bin_cluster_id"]]
    merged["bin_cluster_id"] = pd.Series(bin_ids, index=merged.index, dtype=object)
    merged.loc[filtered_out, "bin_cluster_id"] = "filtered_out"
    merged.loc[filtered_out, "completeness"]   = np.nan   # never evaluable -> not a real 0%
    merged.loc[unclustered, "bin_cluster_id"]  = "unclustered"

    return merged

# =================================================================
# Assign a discrete quality tier to every row of a metrics table.
# =================================================================
def assign_quality(summary: pd.DataFrame, quality: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Assign each row the finest quality tier (strict -> loose) whose completeness/contamination cutoffs it clears."""
    summary = summary.copy()

    # Ensure strict -> loose ordering (highest completeness cutoff first).
    quality = dict(
        sorted(
            quality.items(),
            key=lambda x: x[1]["completeness"],
            reverse=True,
        )
    )

    def classify(row):
        # NaN completeness (e.g. genome with no classified bp) can't pass.
        if pd.isna(row["completeness"]) or pd.isna(row["contamination"]):
            return "fail"

        for name, cutoff in quality.items():
            if (
                row["completeness"] >= cutoff["completeness"]
                and row["contamination"] <= cutoff["contamination"]
            ):
                return name

        return "fail"

    summary["quality"] = summary.apply(classify, axis=1)
    return summary

# =================================================================
# Compile a Ready to Save Report
# =================================================================
def bin_summary(df: pd.DataFrame, quality: dict[str, dict[str, float]], min_length: int = 1000) -> pd.DataFrame:
    """One-row sample-level summary: bp recovered/misclassified/unclassified + per-tier bin counts."""
    metrics = assign_quality(bin_metrics(df, min_length), quality)

    classified = df[df.predictions != -1]

    table = bp_contingency(df).drop(columns=-1, errors="ignore")

    correct_bp = table.max(axis=0).sum()
    classified_bp = table.sum().sum()
    misclassified_bp = classified_bp - correct_bp

    total_bp = df.lengths.sum()
    summary = {
        "true_bins": df.labels.nunique(),
        "predicted_bins": classified.predictions.nunique(),
        "total_bp": total_bp,
        "classified_bp": classified_bp,
        "correct_bp": correct_bp,
        "misclassified_bp": misclassified_bp,
        "correct_fraction": correct_bp / classified_bp if classified_bp else np.nan,
        "classified_fraction": classified_bp / total_bp if total_bp else np.nan,
    }

    for q in quality:
        summary[f"{q}_bins"] = int((metrics.quality == q).sum())
    summary["failed_bins"] = int((metrics.quality == "fail").sum())

    # medians of contig counts across truth clusters
    summary["median_contigs_binned_correctly"] = metrics["contigs_binned_correctly"].median()
    summary["median_contigs_misclassified"]    = metrics["contigs_misclassified"].median()
    summary["median_contigs_unclassified"]     = metrics["contigs_unclassified"].median()

    return pd.DataFrame([summary])