"""
Appendix Table S8: "Similarity of CAMI genomes to the Tetrasphere training set" -- for every
bacterial CAMI genome scored in the binning benchmark, ANI (skani) to its nearest bac120 fold-1
training genome, bucketed into >=99% / 95-99% / 80-95% / <80%-or-no-hit, by CAMI sample category
plus an "All" row. Archaeal and plasmid/virus genomes are excluded (the appendix text notes this:
skani's genome-sketch search here is effectively bacteria-only).

Usage
-----
    python scripts/plots_and_tables/tableS8.py
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-genome-table", default=HERE / "tableS8_ani_cache" / "cami_ani_nearest_training_per_genome.tsv", type=Path,
                         help="one row per CAMI genome with its nearest-training-genome ANI (from steps 1-2, or the shipped cache).")
    parser.add_argument("--domain", default="bacteria", choices=["bacteria", "archaea"], type=str,
                         help="appendix Table S8 reports bacteria only; archaea is excluded per the appendix text.")
    parser.add_argument("--evaluated-only", action="store_true", default=True, help="restrict to genomes actually scored in the binning benchmark (default: on).")
    parser.add_argument("--output", default=None, type=Path, help="default: config.result_table/tableS8.tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (config.result_table / "tableS8.tsv")
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.per_genome_table, sep="\t")
    df = df[df["domain"] == args.domain]
    if args.evaluated_only:
        df = df[df["evaluated"]]

    ani = df["ANI"]
    bucketed = df.assign(
        ge99=(ani >= 99).astype(int),
        ani95_99=((ani >= 95) & (ani < 99)).astype(int),
        ani80_95=((ani >= 80) & (ani < 95)).astype(int),
        lt80=(ani.isna() | (ani < 80)).astype(int),
    )

    def summarize(group: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_genomes": len(group),
            ">=99%": int(group["ge99"].sum()),
            "95-99%": int(group["ani95_99"].sum()),
            "80-95%": int(group["ani80_95"].sum()),
            "<80%": int(group["lt80"].sum()),
            "mean_ani": float(group["ANI"].mean()),
            "std_ani": float(group["ANI"].std()),
        })

    per_category = bucketed.groupby("category").apply(summarize, include_groups=False)
    all_row = summarize(bucketed)
    all_row.name = "All"
    table = pd.concat([per_category, all_row.to_frame().T]).reset_index(names="category")

    table.to_csv(output, sep="\t", index=False)
    print(table.to_string(index=False))
    print(f"\nSaved -> {output}")


if __name__ == "__main__":
    main()
