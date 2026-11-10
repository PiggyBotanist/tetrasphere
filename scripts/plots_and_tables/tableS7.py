"""
Appendix Table S7: CAMI sample statistics before and after filtering to contigs >5kb, aggregated
by sample type.

Usage
-----
    python scripts/plots_and_tables/tableS7.py
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config

CATEGORY_LABELS = [
    ("toy_low", "Toy low"), ("toy_medium", "Toy medium"), ("toy_high", "Toy high"),
    ("airskin_urogenital", "Airways/skin/urogenital"), ("gastro_oral", "Gastrointestinal/oral"),
    ("marine", "Marine"), ("plant_associated", "Plant-associated"), ("strain_madness", "Strain madness"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-sample-table", default=None, type=Path, help="default: config.result_table/cami_binning/table_sample_filter_stats.tsv")
    parser.add_argument("--output", default=None, type=Path, help="default: config.result_table/tableS7.tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_sample_table = args.per_sample_table or (config.result_table / "cami_binning" / "table_sample_filter_stats.tsv")
    output = args.output or (config.result_table / "tableS7.tsv")

    df = pd.read_csv(per_sample_table, sep="\t")
    rows = []
    for category, label in CATEGORY_LABELS:
        sub = df[df["category"] == category]
        rows.append({
            "sample_type": label,
            "n_samples": len(sub),
            "pre_filter_contigs": int(sub["pre_contigs"].sum()),
            "pre_filter_genomes": int(sub["pre_genomes"].sum()),
            "post_filter_contigs": int(sub["post_contigs"].sum()),
            "post_filter_genomes": int(sub["post_genomes"].sum()),
            "percent_retained": round(100 * sub["post_contigs"].sum() / sub["pre_contigs"].sum(), 2),
        })
    total = {
        "sample_type": "Total",
        "n_samples": len(df),
        "pre_filter_contigs": int(df["pre_contigs"].sum()),
        "pre_filter_genomes": int(df["pre_genomes"].sum()),
        "post_filter_contigs": int(df["post_contigs"].sum()),
        "post_filter_genomes": int(df["post_genomes"].sum()),
        "percent_retained": round(100 * df["post_contigs"].sum() / df["pre_contigs"].sum(), 2),
    }
    rows.append(total)

    out_df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, sep="\t", index=False)
    print(out_df.to_string(index=False))
    print(f"\nSaved -> {output}")


if __name__ == "__main__":
    main()
