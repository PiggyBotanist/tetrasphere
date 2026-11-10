"""
Figure 2C: mean distortion vs. fragment length (log scale) for kf2vec (zero-shot).

Usage
-----
    python scripts/plots_and_tables/fig2c_precomputeTable.py --domain bac120   # once, first
    python scripts/plots_and_tables/fig2c.py --domain bac120 --fold 1 --output figures/figure2c.svg
"""
from __future__ import annotations

# Standard Import
import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Custom Import
from config import config
from src.system import create_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", required=True, choices=["ar53", "bac120"], type=str)
    parser.add_argument("--fold", default=1, type=int)
    parser.add_argument("--table", default=None, type=Path, help="default: config.result_table/fig2c/{domain}_fold{fold}.tsv (fig2c_precomputeTable.py's output).")
    parser.add_argument("--output", default=None, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = create_logger("fig2c", output_log=config.root_log)

    table_path = args.table or (config.result_table / "fig2c" / f"{args.domain}_fold{args.fold}.tsv")
    if not table_path.exists():
        raise FileNotFoundError(f"{table_path} not found -- run fig2c_precomputeTable.py --domain {args.domain} first.")
    df = pd.read_csv(table_path, sep="\t")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(df["length"], df["no_denoiser"], marker="s", color="tab:red", label="Tetrasphere (no denoiser)")
    ax.plot(df["length"], df["with_denoiser"], marker="o", color="tab:blue", label="Tetrasphere (with denoiser)")
    if "kf2vec" in df:
        ax.plot(df["length"], df["kf2vec"], marker="^", color="tab:orange", label="kf2vec (zero-shot)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fragment length (bp)")
    ax.set_ylabel("Mean distortion")
    ax.set_title(f"Fragment-length robustness ({args.domain}, fold_{args.fold})")
    ax.legend()
    fig.tight_layout()

    output_path = args.output or (config.result_figure / f"fig2c_{args.domain}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    logger.info(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
