"""
Appendix Figure S2: the full 5-fold class-weighting ablation.

Usage
-----
    python scripts/plots_and_tables/figS2.py --output figures/appendix_figureS2.svg
"""
from __future__ import annotations

# Standard Import
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config

# ar53 (Archaea) left, bac120 (Bacteria) right
DOMAINS = ["ar53", "bac120"]
DOMAIN_TITLES = {"ar53": "Archaea (ar53)", "bac120": "Bacteria (bac120)"}
BUCKET_NAMES = ["rarest", "rare", "mid", "common", "most common"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default=None, type=Path, help="default: config.result_table/figS2 (figS2_precomputeTable.py's --output-dir parent).")
    parser.add_argument("--output",    default=None, type=Path, help="default: config.result_figure/figS2.png.")
    return parser.parse_args()


def nice_round(v: float) -> int:
    """Round a boundary to a readable step size, scaled to its magnitude."""
    if v < 100:
        step = 10
    elif v < 1000:
        step = 50
    elif v < 10000:
        step = 500
    else:
        step = 1000
    return int(round(v / step) * step)


def bucket_thresholds(df: pd.DataFrame) -> list[str]:
    """<X / X-Y / >X style labels from the midpoint between adjacent buckets' observed
    min/max (rounded to a readable step), instead of each bucket's raw, fold-noisy min-max
    (which can overlap at the edges -- see rarity_bucket_summary_*_5fold.tsv)."""
    boundaries = [
        nice_round((df["rarity_max"].iloc[i] + df["rarity_min"].iloc[i + 1]) / 2)
        for i in range(len(df) - 1)
    ]
    labels = [f"<{boundaries[0]:,}"]
    for i in range(1, len(boundaries)):
        labels.append(f"{boundaries[i - 1]:,}–{boundaries[i]:,}")
    labels.append(f">{boundaries[-1]:,}")
    return labels


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir or (config.result_table / "figS2")
    output_path = args.output or (config.result_figure / "figS2.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, domain in zip(axes, DOMAINS):
        df = pd.read_csv(input_dir / domain / f"rarity_bucket_summary_{domain}_5fold.tsv", sep="\t")
        x = df["bucket"].to_numpy()
        width = 0.35

        ax.bar(x - width / 2, df["distortion_uniform_mean"], width, yerr=df["distortion_uniform_std"],
               label="uniform", color="tab:blue", capsize=3)
        ax.bar(x + width / 2, df["distortion_weighted_mean"], width, yerr=df["distortion_weighted_std"],
               label="inverse_phylum", color="tab:orange", capsize=3)

        thresholds = bucket_thresholds(df)
        labels = [f"{name}\n{thresh}" for name, thresh in zip(BUCKET_NAMES, thresholds)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("Phylum rarity bucket (genomes/phylum in train split, rarer side of pair)")
        ax.set_ylabel("Mean pairwise distortion (eval)")
        ax.set_title(DOMAIN_TITLES[domain])
        ax.legend()

    fig.suptitle("Inverse-phylum class weighting: 5-fold eval distortion by rarity bucket (Appendix Figure S2)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
