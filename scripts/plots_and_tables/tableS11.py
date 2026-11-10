"""
Appendix Table S11: per-genome bin quality (average completeness, average contamination) and
tier-A (strict, >=95%/<=5%) recovery fraction, by method and CAMI sample type. Reads the
already-computed per-category x per-method quality stats (same DBSCAN selection -- euclidean
metric, SCG-count hyperparameter choice -- used by Figure 4B and Tables S9/S10).

Usage
-----
    python scripts/plots_and_tables/tableS11.py
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config

CATEGORY_ORDER = ["toy_low", "toy_medium", "toy_high", "airskin_urogenital", "gastro_oral", "marine", "plant_associated", "strain_madness"]
CATEGORY_LABELS = {
    "toy_low": "Toy low", "toy_medium": "Toy medium", "toy_high": "Toy high",
    "airskin_urogenital": "Airways/skin/urogenital", "gastro_oral": "Gastrointestinal/oral",
    "marine": "Marine", "plant_associated": "Plant-associated", "strain_madness": "Strain madness",
}
METHOD_ORDER = ["4mer_no_model", "comebin", "comebin_full_pipeline", "metabat2", "binny", "proposed_model_whole_genome", "proposed_model_fragment_refinement"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-category-table", default=None, type=Path, help="default: config.result_table/cami_binning/all_categories_quality_stats_matching_figure.tsv")
    parser.add_argument("--output", default=None, type=Path, help="default: config.result_table/tableS11.tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_category_table = args.per_category_table or (config.result_table / "cami_binning" / "all_categories_quality_stats_matching_figure.tsv")
    output = args.output or (config.result_table / "tableS11.tsv")

    df = pd.read_csv(per_category_table, sep="\t")
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    df["category"] = pd.Categorical(df["category"], categories=CATEGORY_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values(["category", "method"])

    out = df[["category", "method_label", "avg_completeness_pct", "avg_contamination_pct", "tier_A_fraction_of_evaluated"]].copy()
    out["category"] = out["category"].map(CATEGORY_LABELS)
    out = out.rename(columns={
        "category": "sample_type", "method_label": "method",
        "avg_completeness_pct": "avg_completeness_pct", "avg_contamination_pct": "avg_contamination_pct",
        "tier_A_fraction_of_evaluated": "tier_A_fraction",
    })
    out["avg_completeness_pct"] = out["avg_completeness_pct"].round(2)
    out["avg_contamination_pct"] = out["avg_contamination_pct"].round(2)
    out["tier_A_fraction"] = out["tier_A_fraction"].round(4)

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, sep="\t", index=False)
    print(out.to_string(index=False))
    print(f"\nSaved -> {output}")


if __name__ == "__main__":
    main()
