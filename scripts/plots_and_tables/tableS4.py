"""
Appendix Table S4: "Tetrasphere phylogenetic distance prediction on k in {2,3,4,5,6,7}".
Usage
-----
    python scripts/plots_and_tables/tableS4.py
"""
from __future__ import annotations

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config

KS = [2, 3, 4, 5, 6, 7]
METRICS = ["pairwise_mean_distortion", "pairwise_pearson_r", "pairwise_r_squared", "pairwise_spearman_r"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=None, type=Path, help="default: config.result_table/tableS4.tsv")
    return parser.parse_args()


def table_path(domain: str, k: int) -> Path:
    if k == 4:
        return config.result_table / domain / f"{domain}_4mer_clr_genome_validation.tsv"
    return config.result_table / "kmer_experiment" / f"{domain}_{k}mer_clr_validation.tsv"


def main() -> None:
    args = parse_args()
    rows = []
    for domain in ("ar53", "bac120"):
        for k in KS:
            path = table_path(domain, k)
            if not path.exists():
                print(f"MISSING (skipping): {path}")
                continue
            df = pd.read_csv(path, sep="\t")
            summary = df[(df["dataset"] == "eval") & (df["fold"].isin(["mean", "std"]))]
            if summary.empty:
                print(f"no mean/std summary rows in {path}, skipping")
                continue
            mean_row = summary[summary["fold"] == "mean"].iloc[0]
            std_row = summary[summary["fold"] == "std"].iloc[0]
            row = {"domain": domain, "k": k}
            for metric in METRICS:
                if metric in df.columns:
                    row[f"{metric}_mean"] = mean_row[metric]
                    row[f"{metric}_std"] = std_row[metric]
            rows.append(row)

    out = pd.DataFrame(rows)
    output_path = args.output or (config.result_table / "tableS4.tsv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, sep="\t", index=False, float_format="%.4f")
    print(out.to_string(index=False))
    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
