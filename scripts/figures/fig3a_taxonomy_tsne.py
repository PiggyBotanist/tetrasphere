"""
Function: 2D projection (t-SNE / UMAP / PCA) of the whole-genome embedding space, colored by one
    taxonomic rank (phylum/class/order/family/genus/species), with in-plot labels at each shown
    group's centroid -- the top-N groups by genome count get their own color + label, everything
    else is plotted light gray as "other".

    --restrict-level/--restrict-value zooms into one coarser-rank group first (e.g. one phylum),
    so --color-level can then show a finer rank within just that group (mirrors the reference
    image: phylum-wide view, or a zoom into one phylum colored by class).

    - (--domain):         ar53 or bac120.
    - (--fold/--epoch):   which genome-stage checkpoint (03_train_genome.py) to embed with.
    - (--method):         tsne, umap, or pca.
    - (--color-level):    taxonomic rank to color by.
    - (--top-n):          how many top groups (by count) get their own color + label.
    - (--restrict-level/--restrict-value): optional zoom-in filter, e.g. --restrict-level phylum
                          --restrict-value Bacillota.
    - (--max-points):     stratified subsample cap (projection quality/runtime).
    - (--label/--no-label): draw colored boxed labels at each group's centroid (default on).
    - (--inner-level):    optional finer rank shown WITHIN each --color-level group, in the same
                          plot: each --color-level group gets its own hue, each --inner-level
                          subgroup within it gets its own shade of that hue (same idea as "if it's
                          green, show different shades of green"), with its own smaller label.
                          Genomes in a shown --color-level group but not one of its top
                          --top-n-inner subgroups get a muted/desaturated version of that group's
                          hue (still visibly "part of" the group, just not individually labeled).
    - (--top-n-inner):    how many --inner-level subgroups per --color-level group get their own
                          shade + label.

Usage
-----
    python scripts/figures/fig3a_taxonomy_tsne.py --domain bac120 --color-level phylum
    python scripts/figures/fig3a_taxonomy_tsne.py --domain bac120 --method umap \\
        --restrict-level phylum --restrict-value Bacillota --color-level class
    python scripts/figures/fig3a_taxonomy_tsne.py --domain bac120 --color-level class --inner-level family
"""
from __future__ import annotations

# Standard Import
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from adjustText import adjust_text

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import hdf5
from src.system import create_logger
from src.model import checkpoint
from src.tetrasphere.model import Manifold
from src.tetrasphere.preprocess import get_taxonomy

RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain",          required=True, type=str, choices=["ar53", "bac120"])
    parser.add_argument("--fold",            default=1, type=int, help="genome-stage checkpoint fold.")
    parser.add_argument("--epoch",           default=None, type=int, help="checkpoint epoch; default latest.")
    parser.add_argument("--method",          default="tsne", type=str, choices=["tsne", "umap", "pca"])
    parser.add_argument("--color-level",     default="phylum", type=str, choices=RANKS)
    parser.add_argument("--top-n",           default=8, type=int, help="number of distinct groups to color+label; rest -> light gray \"other\".")
    parser.add_argument("--restrict-level",  default=None, type=str, choices=RANKS, help="optional: restrict to one group at a coarser rank first.")
    parser.add_argument("--restrict-value",  default=None, type=str, help="which value of --restrict-level to keep, e.g. Bacillota.")
    parser.add_argument("--max-points",      default=20000, type=int, help="stratified subsample cap, by --color-level.")
    parser.add_argument("--label",           dest="label", action="store_true", default=True)
    parser.add_argument("--no-label",        dest="label", action="store_false")
    parser.add_argument("--inner-level",     default=None, type=str, choices=RANKS, help="optional finer rank to shade+label within each --color-level group (must be finer than --color-level).")
    parser.add_argument("--top-n-inner",     default=4, type=int, help="number of --inner-level subgroups per --color-level group to shade+label.")
    parser.add_argument("--seed",            default=0, type=int)
    parser.add_argument("--output",          default=None, type=Path, help="output .png path; default result/figure/<domain>_taxonomy_<color-level>_<method>.png.")
    return parser.parse_args()

# =================================================================
# Pipeline steps
# =================================================================
def load_embedding(domain: str, fold: int, epoch: int | None, device: torch.device, logger: logging.Logger) -> dict[str, np.ndarray]:
    """Embed every GTDB representative genome through the frozen genome-stage Manifold."""
    data = hdf5.load(config.gtdb_feature / f"{domain}_4mer_clr.hdf5")
    ckpt_path = checkpoint.find(config.result_checkpoint / domain / f"{domain}_4mer_clr_genome" / f"fold_{fold}", epoch)
    cfg = checkpoint.load_config(ckpt_path)
    model = Manifold(**cfg["model_config"]).to(device)
    checkpoint.load_weights(ckpt_path, model, device)
    model.eval()
    logger.info(f"Loaded {ckpt_path}")

    with torch.no_grad():
        embedding = model.encode(torch.from_numpy(data["feature"].astype(np.float32)).to(device)).cpu().numpy()
    return {"id": data["id"], "embedding": embedding}

def load_taxonomy(domain: str) -> pd.DataFrame:
    path = config.gtdb_download / f"{domain}_taxonomy_r232.tsv.gz"
    df = pd.read_csv(path, sep="\t", compression="gzip", names=["id", "taxonomy"])
    return get_taxonomy(df, config.taxonomy_map)

def stratified_subsample(df: pd.DataFrame, group_col: str, max_points: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    rng = np.random.default_rng(seed)
    frac = max_points / len(df)
    parts = []
    for _, sub in df.groupby(group_col, sort=False):
        n = max(1, int(round(len(sub) * frac)))
        idx = rng.choice(sub.index, size=min(n, len(sub)), replace=False)
        parts.append(df.loc[idx])
    return pd.concat(parts)

def project_2d(embedding: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, metric="cosine", init="pca", random_state=seed, perplexity=30).fit_transform(embedding)
    if method == "umap":
        import umap
        return umap.UMAP(n_components=2, metric="cosine", random_state=seed).fit_transform(embedding)
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(embedding)
    raise ValueError(method)

def plot_with_group_labels(ax: plt.Axes, xy: np.ndarray, labels: pd.Series, top_labels: list[str], label_boxes: bool) -> None:
    # tab10 index 7 is itself a gray, which would be visually indistinguishable from the "other" points
    group_colors = [c for i, c in enumerate(plt.get_cmap("tab10").colors) if i != 7] + [plt.get_cmap("tab20").colors[i] for i in (2, 3)]
    is_top = labels.isin(top_labels).to_numpy()
    ax.scatter(xy[~is_top, 0], xy[~is_top, 1], s=3, c="lightgray", alpha=0.5, linewidths=0, zorder=1)

    texts = []
    for i, name in enumerate(top_labels):
        mask = (labels == name).to_numpy()
        color = group_colors[i % len(group_colors)]
        ax.scatter(xy[mask, 0], xy[mask, 1], s=5, color=color, alpha=0.8, linewidths=0, zorder=2)
        if label_boxes:
            cx, cy = np.median(xy[mask, 0]), np.median(xy[mask, 1])
            texts.append(ax.text(cx, cy, name, fontsize=11, fontweight="bold", color="black", zorder=5,
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85, edgecolor="none")))
    if label_boxes and texts:
        adjust_text(texts, ax=ax, expand=(1.2, 1.4), arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8))

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_with_nested_group_labels(ax: plt.Axes, xy: np.ndarray, outer: pd.Series, inner: pd.Series,
                                   top_outer: list[str], top_inner_by_outer: dict[str, list[str]], label_boxes: bool) -> dict[str, list[tuple]]:
    """Each outer (--color-level) group gets its own hue; each inner (--inner-level) subgroup within
    it gets its own shade of that hue (brightness varied, hue/saturation fixed) -- "different shades
    of green" for one outer group. Genomes in a shown outer group but not one of its labeled inner
    subgroups get a desaturated version of that hue, so they still read as "part of this group"
    without individually listing every single subgroup. Only the outer group gets an in-plot label
    (inner/family-level detail goes in the legend instead, see build_tabbed_legend() -- in-plot
    labels for every family got crowded/overlapping once there were ~30 of them).

    Returns {outer_name: [(inner_name, color), ...]} for build_tabbed_legend()."""
    import matplotlib.colors as mcolors

    is_shown_outer = outer.isin(top_outer).to_numpy()
    ax.scatter(xy[~is_shown_outer, 0], xy[~is_shown_outer, 1], s=3, c="lightgray", alpha=0.5, linewidths=0, zorder=1)

    outer_texts = []
    legend_entries: dict[str, list[tuple]] = {}
    hues = np.linspace(0, 1, len(top_outer), endpoint=False)
    for hue, outer_name in zip(hues, top_outer):
        outer_mask = (outer == outer_name).to_numpy()
        inner_names = top_inner_by_outer.get(outer_name, [])
        is_labeled_inner = outer_mask & inner.isin(inner_names).to_numpy()

        # Unlabeled genomes within this outer group: one muted (low-saturation) shade of its hue
        muted_mask = outer_mask & ~is_labeled_inner
        muted_color = mcolors.hsv_to_rgb((hue, 0.25, 0.85))
        ax.scatter(xy[muted_mask, 0], xy[muted_mask, 1], s=3, color=muted_color, alpha=0.6, linewidths=0, zorder=2)

        # Each labeled inner subgroup: full saturation, brightness varies by subgroup index
        legend_entries[outer_name] = []
        brightness = np.linspace(1.0, 0.5, max(len(inner_names), 1))
        for value, inner_name in zip(brightness, inner_names):
            mask = outer_mask & (inner == inner_name).to_numpy()
            color = mcolors.hsv_to_rgb((hue, 0.85, value))
            ax.scatter(xy[mask, 0], xy[mask, 1], s=6, color=color, alpha=0.85, linewidths=0, zorder=3)
            legend_entries[outer_name].append((inner_name, color))

        if label_boxes and outer_mask.any():
            cx, cy = np.median(xy[outer_mask, 0]), np.median(xy[outer_mask, 1])
            outer_color = mcolors.hsv_to_rgb((hue, 0.85, 0.85))
            outer_texts.append(ax.text(cx, cy, outer_name, fontsize=13, fontweight="bold", color="black", zorder=6,
                                        bbox=dict(boxstyle="round,pad=0.35", facecolor=outer_color, alpha=0.9, edgecolor="none")))

    if label_boxes and outer_texts:
        adjust_text(outer_texts, ax=ax, expand=(1.2, 1.4), arrowprops=dict(arrowstyle="-", color="0.3", lw=0.6))

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return legend_entries

def build_tabbed_legend(fig: plt.Figure, ax: plt.Axes, legend_entries: dict[str, list[tuple]], top_n_inner: int) -> None:
    """Single legend, outside the plot on the right: one "tab"/section per outer group -- a bold
    header (outer name, no marker) followed by its inner subgroups (colored square + name), with a
    blank spacer row between sections. One column (not one-column-per-class) so it reads as a tidy
    vertical list beside the plot instead of a wide strip under it."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    blank = Line2D([], [], linestyle="", label=" ")
    handles, labels, header_labels = [], [], set()
    for i, (outer_name, inner_list) in enumerate(legend_entries.items()):
        if i > 0:
            handles.append(blank); labels.append(" ")  # spacer between class sections
        handles.append(Line2D([], [], linestyle="", label=outer_name))  # header row, no marker
        labels.append(outer_name)
        header_labels.add(outer_name)
        for inner_name, color in inner_list:
            handles.append(Patch(facecolor=color, edgecolor="none"))
            labels.append(inner_name)

    legend = ax.legend(handles, labels, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5),
                        fontsize=8, frameon=False, handletextpad=0.6, labelspacing=0.4, handlelength=1.2)
    for text, label in zip(legend.get_texts(), labels):
        if label in header_labels:
            text.set_fontweight("bold")
            text.set_fontsize(9.5)

# Main
def main() -> None:
    args = parse_args()
    logger = create_logger(f"visualize_taxonomy_{args.domain}", output_log=config.root_log)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref = load_embedding(args.domain, args.fold, args.epoch, device, logger)
    id_to_idx = {gid: i for i, gid in enumerate(ref["id"])}

    tax = load_taxonomy(args.domain)  # every GTDB genome, not just the representatives embedded above
    tax = tax[tax["id"].isin(id_to_idx)].copy()  # restrict to genomes we actually have an embedding for
    tax["embed_idx"] = tax["id"].map(id_to_idx)
    tax = tax[tax[args.color_level] != f"unclassified_{args.color_level}"]

    if args.restrict_level:
        if not args.restrict_value:
            raise ValueError("--restrict-value is required with --restrict-level")
        tax = tax[tax[args.restrict_level] == args.restrict_value]
        if tax.empty:
            raise ValueError(f"no genomes with {args.restrict_level}=={args.restrict_value!r}")
    logger.info(f"{len(tax)} genomes with resolved {args.color_level!r}" + (f", restricted to {args.restrict_level}={args.restrict_value!r}" if args.restrict_level else ""))

    if args.inner_level and RANKS.index(args.inner_level) <= RANKS.index(args.color_level):
        raise ValueError(f"--inner-level {args.inner_level!r} must be finer than --color-level {args.color_level!r}")

    sub = stratified_subsample(tax, args.color_level, args.max_points, args.seed)
    top_groups = tax[args.color_level].value_counts().head(args.top_n).index.tolist()
    logger.info(f"Plotting {len(sub)} genomes, top {args.color_level} groups = {top_groups}")

    xy = project_2d(ref["embedding"][sub["embed_idx"].to_numpy()], args.method, args.seed)

    fig, ax = plt.subplots(figsize=(9, 7))
    if args.inner_level:
        top_inner_by_outer = {}
        for outer_name in top_groups:
            counts = tax[tax[args.color_level] == outer_name][args.inner_level].value_counts()
            counts = counts[counts.index != f"unclassified_{args.inner_level}"]
            top_inner_by_outer[outer_name] = counts.head(args.top_n_inner).index.tolist()
        logger.info(f"Top {args.inner_level} per {args.color_level}: {top_inner_by_outer}")
        legend_entries = plot_with_nested_group_labels(ax, xy, sub[args.color_level], sub[args.inner_level], top_groups, top_inner_by_outer, args.label)
        if args.label:
            build_tabbed_legend(fig, ax, legend_entries, args.top_n_inner)
    else:
        plot_with_group_labels(ax, xy, sub[args.color_level], top_groups, args.label)
    title = f"{args.domain}: embedding space colored by {args.color_level}" + (f" -> {args.inner_level}" if args.inner_level else "") + f" ({args.method})"
    if args.restrict_level:
        title += f"\n(restricted to {args.restrict_level}={args.restrict_value})"
    ax.set_title(title)
    fig.tight_layout()

    level_tag = f"{args.color_level}_{args.inner_level}" if args.inner_level else args.color_level
    output_path = args.output or (config.result_figure / f"{args.domain}_taxonomy_{level_tag}_{args.method}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved -> {output_path}")

if __name__ == "__main__":
    main()
