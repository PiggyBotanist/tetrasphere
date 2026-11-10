"""
Appendix Table S5: Phylogenetic distance prediction on kf2vec, Tetrasphere (without denoiser)/
Usage
-----
    python scripts/plots_and_tables/tableS5.py
"""
from __future__ import annotations

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config

FOLDS = [1, 2, 3, 4, 5]
SERIES = ["kf2vec", "no_denoiser", "with_denoiser"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=None, type=Path, help="default: config.result_table/tableS5.tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for domain in ("ar53", "bac120"):
        frames = []
        for fold in FOLDS:
            path = config.result_table / "fig2c" / f"{domain}_fold{fold}.tsv"
            if not path.exists():
                print(f"MISSING (skipping {domain}): {path}")
                frames = []
                break
            frames.append(pd.read_csv(path, sep="\t"))
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        for length, group in combined.groupby("length"):
            row = {"domain": domain, "length": int(length), "n_folds": len(group)}
            for series in SERIES:
                if series in group.columns:
                    row[f"{series}_mean"] = group[series].mean()
                    row[f"{series}_std"] = group[series].std()
            rows.append(row)

    out = pd.DataFrame(rows).sort_values(["domain", "length"], ascending=[True, False])
    output_path = args.output or (config.result_table / "tableS5.tsv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, sep="\t", index=False, float_format="%.4f")
    print(out.to_string(index=False))
    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
