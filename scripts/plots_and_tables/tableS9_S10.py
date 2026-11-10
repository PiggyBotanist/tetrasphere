"""
Appendix Tables S9 & S10: genomes recovered per sample type (%), for all 7 binning methods, same
underlying per-sample tables as Figure 4B. Both tables pivot the exact same summary -- only the
recovery-fraction column differs -- so one run writes both:
  - S9 (high-quality): completeness >=90%, contamination <=10% (= tier_A + tier_B fraction, since
    tier_B's own threshold is exactly 90/10).
  - S10 (low-quality): completeness >=70%, contamination <=30% (= recovered_fraction, the sum of
    tiers A-F, since tier_F's own threshold is exactly 70/30).

Usage
-----
    python scripts/plots_and_tables/tableS9_S10.py
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
import fig4b as f4b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-s9",  default=None, type=Path, help="default: config.result_table/tableS9.tsv")
    parser.add_argument("--output-s10", default=None, type=Path, help="default: config.result_table/tableS10.tsv")
    return parser.parse_args()


def build_pivot(summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = summary.pivot(index="category", columns="method", values=value_col) * 100
    pivot = pivot[[m for m, _, _ in f4b.METHODS]]
    pivot = pivot.rename(columns={m: label for m, label, _ in f4b.METHODS})
    pivot = pivot.reindex([c for c, _ in f4b.CATEGORIES])
    pivot.index = [label.replace("\n", " ") for _, label in f4b.CATEGORIES]
    pivot.loc["Overall mean"] = pivot.mean()
    return pivot.round(2)


def main() -> None:
    args = parse_args()
    output_s9 = args.output_s9 or (config.result_table / "tableS9.tsv")
    output_s10 = args.output_s10 or (config.result_table / "tableS10.tsv")

    binning_root = config.result_table / "cami_binning"
    df = f4b.load_rows(
        binning_root / "table_embedding_euclidean_dbscan_scg_per_sample.tsv",
        binning_root / "table_cami_comparison_external_per_sample.tsv",
        binning_root / "table_comebin_full_pipeline_per_sample.tsv",
    )
    summary = f4b.summarize(df)

    for label, value_col, output in [("S9 (HQ)", "hq_fraction_mean", output_s9), ("S10 (LQ)", "recovered_fraction_mean", output_s10)]:
        pivot = build_pivot(summary, value_col)
        output.parent.mkdir(parents=True, exist_ok=True)
        pivot.to_csv(output, sep="\t")
        print(f"=== Table {label} ===")
        print(pivot.to_string())
        print(f"Saved -> {output}\n")


if __name__ == "__main__":
    main()
