"""
Appendix Figure S3 (archaea) / Figure S4 (bacteria): 2D PCA of the whole-genome embedding space.

Usage
-----
    python scripts/plots_and_tables/figS3_S4.py --domain ar53   --output figures/appendix_figureS3.svg
    python scripts/plots_and_tables/figS3_S4.py --domain bac120 --output figures/appendix_figureS4.svg
"""
from __future__ import annotations

# Standard Import
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Custom Import
from config import config
from src.io import hdf5
from src.system import create_logger
from fig3a import load_embedding, load_taxonomy, stratified_subsample, project_2d, plot_with_group_labels

PANELS = ["phylum", "order", "genus", "split"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain",     required=True, type=str, choices=["ar53", "bac120"])
    parser.add_argument("--fold",       default=1, type=int, help="genome-stage checkpoint fold.")
    parser.add_argument("--epoch",      default=None, type=int, help="checkpoint epoch; default latest.")
    parser.add_argument("--method",     default="pca", type=str, choices=["tsne", "umap", "pca"])
    parser.add_argument("--top-n",      default=8, type=int, help="number of distinct groups to color+label per taxonomic panel; rest -> light gray \"other\".")
    parser.add_argument("--max-points", default=20000, type=int, help="stratified subsample cap, by phylum.")
    parser.add_argument("--seed",       default=0, type=int)
    parser.add_argument("--output",     default=None, type=Path, help="output path; default config.result_figure/<domain>_taxonomy_pca_4panel.png.")
    return parser.parse_args()


def split_labels(domain: str, genome_ids: np.ndarray) -> np.ndarray:
    """'evaluation' if the genome is in the shared held-out eval set, else 'training'."""
    split_file = config.gtdb_split / f"{domain}_4mer_clr_genome.hdf5"
    eval_ids = set(hdf5.load(split_file)["eval"]["leaf_id"])
    return np.where(np.isin(genome_ids, list(eval_ids)), "evaluation", "training")


def plot_split_panel(ax: plt.Axes, xy: np.ndarray, labels: np.ndarray) -> None:
    colors = {"training": "tab:blue", "evaluation": "tab:orange"}
    for name, color in colors.items():
        mask = labels == name
        ax.scatter(xy[mask, 0], xy[mask, 1], s=3, color=color, alpha=0.6, linewidths=0, zorder=2, rasterized=True, label=f"{name} (n={mask.sum()})")
    ax.legend(loc="best", fontsize=8, frameon=False, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    args = parse_args()
    logger = create_logger(f"figS_taxonomy_pca_{args.domain}", output_log=config.root_log)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref = load_embedding(args.domain, args.fold, args.epoch, device, logger)
    id_to_idx = {gid: i for i, gid in enumerate(ref["id"])}

    tax = load_taxonomy(args.domain)
    tax = tax[tax["id"].isin(id_to_idx)].copy()
    tax["embed_idx"] = tax["id"].map(id_to_idx)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, level in zip(axes.flat, PANELS):
        if level == "split":
            tax_split = tax.copy()
            tax_split["split"] = split_labels(args.domain, tax_split["id"].to_numpy())
            sub = stratified_subsample(tax_split, "split", args.max_points, args.seed)
            xy = project_2d(ref["embedding"][sub["embed_idx"].to_numpy()], args.method, args.seed)
            plot_split_panel(ax, xy, sub["split"].to_numpy())
            ax.set_title(f"{args.domain}: evaluation vs. training split")
            continue

        level_tax = tax[tax[level] != f"unclassified_{level}"]
        sub = stratified_subsample(level_tax, level, args.max_points, args.seed)
        top_groups = level_tax[level].value_counts().head(args.top_n).index.tolist()
        logger.info(f"[{level}] plotting {len(sub)} genomes, top groups = {top_groups}")
        xy = project_2d(ref["embedding"][sub["embed_idx"].to_numpy()], args.method, args.seed)
        plot_with_group_labels(ax, xy, sub[level], top_groups, label_boxes=True)
        ax.set_title(f"{args.domain}: colored by {level}")

    fig.suptitle(f"{args.domain}: embedding space by taxonomic rank and dataset split ({args.method})", fontsize=13)
    fig.tight_layout()

    output_path = args.output or (config.result_figure / f"{args.domain}_taxonomy_pca_4panel.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
