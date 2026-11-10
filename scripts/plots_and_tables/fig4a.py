"""
Figure 4A: CAMI contig embedding comparison (t-SNE), across four representations.

Usage
-----
    python scripts/plots_and_tables/fig4a.py --domain bac120 --samples cami_toy_medium_m1_s001 --output-a figures/figure4a.svg --output-b figures/.figure4a_panelB_unused.png
"""
from __future__ import annotations

# Standard Import
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Custom Import
from config import config
from src.system import create_logger
from src.metagenome.cami_compare import load_four_representations
from src.metagenome.cluster import grid_search_dbscan
from src.metagenome import metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", required=True, type=str, nargs="+", help="CAMI sample names to score (one bin_summary row each).")
    parser.add_argument("--tsne-sample", default=None, type=str, help="which sample to render in panel A; default: first of --samples.")
    parser.add_argument("--k", default=4, type=int)
    parser.add_argument("--transform", default="clr", type=str)
    parser.add_argument("--min-contig-length", default=5000, type=int)
    parser.add_argument("--domain", default="bac120", choices=["ar53", "bac120"], type=str, help="which domain's checkpoints to embed with; main-text Figure 4 uses bac120.")
    parser.add_argument("--genome-experiment", default=None, type=str, help="default: --domain.")
    parser.add_argument("--genome-label", default=None, type=str, help="default: {domain}_4mer_clr_genome.")
    parser.add_argument("--genome-fold", default=1, type=int)
    parser.add_argument("--genome-epoch", default=None, type=int)
    parser.add_argument("--fragment-experiment", default=None, type=str, help="default: --domain.")
    parser.add_argument("--fragment-label", default=None, type=str, help="default: {domain}_4mer_clr_fragment.")
    parser.add_argument("--fragment-fold", default=1, type=int)
    parser.add_argument("--fragment-epoch", default=None, type=int)
    parser.add_argument("--eps-grid", default=[0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0], type=float, nargs="+")
    parser.add_argument("--min-samples-grid", default=[3, 5, 10], type=int, nargs="+")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output-a", default=None, type=Path, help="panel A output path; default: config.result_figure/fig4a_cami_embedding_{domain}.png")
    parser.add_argument("--output-b", default=None, type=Path, help="panel B output path; default: config.result_figure/fig4b_cami_recovery_{domain}.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = create_logger("fig4_cami_binning", output_log=config.root_log)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    genome_experiment = args.genome_experiment or args.domain
    genome_label = args.genome_label or f"{args.domain}_4mer_clr_genome"
    fragment_experiment = args.fragment_experiment or args.domain
    fragment_label = args.fragment_label or f"{args.domain}_4mer_clr_fragment"

    summary_rows = []
    panel_a_data = None
    for sample in args.samples:
        data = load_four_representations(
            sample, args.k, args.transform, args.min_contig_length,
            genome_experiment, genome_label, args.genome_fold, args.genome_epoch,
            fragment_experiment, fragment_label, args.fragment_fold, args.fragment_epoch,
            None, device, logger,
        )
        if sample == (args.tsne_sample or args.samples[0]):
            panel_a_data = data

        for repr_name in ("baseline", "comebin", "genome", "fragment"):
            summary, params, _ = grid_search_dbscan(data[repr_name], data["true_genome"], data["length"], args.min_contig_length, args.eps_grid, args.min_samples_grid)
            total_recovered = sum(int(summary[f"{tier}_bins"].iloc[0]) for tier in metric.QUALITY_TIERS)
            row = {"sample": sample, "representation": repr_name, "true_bins": int(summary["true_bins"].iloc[0]),
                   "total_recovered_fraction": total_recovered / int(summary["true_bins"].iloc[0]),
                   "silhouette": params["silhouette"], "eps": params["eps"], "min_samples": params["min_samples"]}
            for tier in metric.QUALITY_TIERS:
                row[f"{tier}_bins"] = int(summary[f"{tier}_bins"].iloc[0])
            summary_rows.append(row)
            logger.info(f"{sample}/{repr_name}: recovered {total_recovered}/{row['true_bins']} (silhouette={params['silhouette']:.3f}, eps={params['eps']}, min_samples={params['min_samples']})")

    df = pd.DataFrame(summary_rows)
    out_table = config.result_table / "fig4" / "cami_binning_summary.tsv"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_table, sep="\t", index=False)
    logger.info(f"Table -> {out_table}")

    # Panel A: t-SNE of one representative sample's four representations
    cmap = plt.get_cmap("tab20")
    genomes = sorted(set(panel_a_data["true_genome"]))
    color_map = {g: cmap(i % 20) for i, g in enumerate(genomes)}
    colors = [color_map[g] for g in panel_a_data["true_genome"]]
    panels = [("4-mer (Baseline)", "baseline"), ("COMEBin", "comebin"), ("Tetrasphere (Without Denoiser)", "genome"), ("Tetrasphere (With Denoiser)", "fragment")]

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, (title, key) in zip(axes.flat, panels):
        coords = TSNE(n_components=2, random_state=args.seed, init="pca", metric="euclidean" if key != "baseline" else "cosine").fit_transform(panel_a_data[key])
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=6, alpha=0.7, edgecolors="none")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{args.tsne_sample or args.samples[0]} (contigs >= {args.min_contig_length}bp, n={len(genomes)} true genomes)")
    fig.tight_layout()
    panel_a_path = args.output_a or (config.result_figure / f"fig4a_cami_embedding_{args.domain}.png")
    panel_a_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(panel_a_path, dpi=150)
    logger.info(f"Saved -> {panel_a_path}")

    # Panel B: stacked recovery-fraction bars, one per sample x representation
    fig, ax = plt.subplots(figsize=(10, 5))
    repr_names = ["baseline", "comebin", "genome", "fragment"]
    repr_colors = {"baseline": "gray", "comebin": "tab:blue", "genome": "tab:purple", "fragment": "tab:green"}
    x = np.arange(len(args.samples))
    width = 0.2
    for i, repr_name in enumerate(repr_names):
        values = [df[(df["sample"] == s) & (df["representation"] == repr_name)]["total_recovered_fraction"].iloc[0] for s in args.samples]
        ax.bar(x + (i - 1.5) * width, values, width, label=repr_name, color=repr_colors[repr_name])
    ax.set_xticks(x)
    ax.set_xticklabels(args.samples, rotation=30, ha="right")
    ax.set_ylabel("Fraction of true genomes recovered")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
    ax.legend()
    fig.tight_layout()
    panel_b_path = args.output_b or (config.result_figure / f"fig4b_cami_recovery_{args.domain}.png")
    panel_b_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(panel_b_path, dpi=150)
    logger.info(f"Saved -> {panel_b_path}")


if __name__ == "__main__":
    main()
