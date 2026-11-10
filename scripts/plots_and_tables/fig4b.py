"""
Figure 4B: CAMI binning benchmark across 7 methods.

Usage
-----
    python scripts/plots_and_tables/fig4b.py --output figures/figure4b.svg
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config

METHODS = [
    ("4mer_no_model",                      "4-mer (CLR)",                     "#8a8984"),
    ("comebin",                            "COMEBin embedding",               "#2a78d6"),
    ("comebin_full_pipeline",              "COMEBin (full pipeline)",         "#a3245c"),
    ("metabat2",                           "MetaBAT2",                        "#eb6834"),
    ("binny",                              "binny",                           "#1baf7a"),
    ("proposed_model_whole_genome",        "Tetrasphere (without denoiser)",  "#6250d6"),
    ("proposed_model_fragment_refinement", "Tetrasphere (with denoiser)",     "#e87ba4"),
]
CATEGORIES = [
    ("toy_low",            "Toy low"),
    ("toy_medium",         "Toy medium"),
    ("toy_high",           "Toy high"),
    ("airskin_urogenital", "Airways/skin/\nurogenital"),
    ("gastro_oral",        "Gastrointestinal/\noral"),
    ("marine",             "Marine"),
    ("plant_associated",   "Plant-associated"),
    ("strain_madness",     "Strain madness"),
]
TIERS = [("tier_A", "A  ≥95% / ≤5%"), ("tier_B", "B  ≥90% / ≤10%"), ("tier_C", "C  ≥85% / ≤15%"),
         ("tier_D", "D  ≥80% / ≤20%"), ("tier_E", "E  ≥75% / ≤25%"), ("tier_F", "F  ≥70% / ≤30%")]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def tier_shades(color: str, n: int = len(TIERS), dark: float = 0.25, light: float = 0.40) -> list[np.ndarray]:
    """n shades of one hue: darkened toward black for strict tiers, blended toward the surface for loose ones."""
    def rgb(h: str) -> np.ndarray:
        return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    base, surface = rgb(color), rgb(SURFACE)
    return [base * (1 + t) if t < 0 else base * (1 - t) + surface * t for t in np.linspace(-dark, light, n)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding-table",  default=None, type=Path, help="default: config.result_table/cami_binning/table_embedding_euclidean_dbscan_scg_per_sample.tsv")
    parser.add_argument("--external-table",   default=None, type=Path, help="default: config.result_table/cami_binning/table_cami_comparison_external_per_sample.tsv")
    parser.add_argument("--comebin-full-table", default=None, type=Path, help="default: config.result_table/cami_binning/table_comebin_full_pipeline_per_sample.tsv")
    parser.add_argument("--output",           default=None, type=Path, help="default: config.result_figure/figure4b.png")
    return parser.parse_args()


def load_rows(embedding_table: Path, external_table: Path, comebin_full_table: Path) -> pd.DataFrame:
    df = pd.concat([
        pd.read_csv(embedding_table, sep="\t"),
        pd.read_csv(external_table, sep="\t"),
        pd.read_csv(comebin_full_table, sep="\t"),
    ], ignore_index=True)
    df = df[df["method"].isin([m for m, _, _ in METHODS])].copy()
    df["hq_fraction"] = df["tier_A_fraction"] + df["tier_B_fraction"]
    df["recovered_fraction"] = df[[f"{tier}_fraction" for tier, _ in TIERS]].sum(axis=1)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["hq_fraction", "recovered_fraction"] + [f"{tier}_fraction" for tier, _ in TIERS]
    agg = df.groupby(["category", "method"])[metrics].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def plot(df: pd.DataFrame, output: Path) -> None:
    """Single panel: genomes recovered per method, stacked by quality tier A (dark) -> F (light)."""
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2})
    fig, ax = plt.subplots(figsize=(14, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    n = len(METHODS)
    group_width = 0.86
    bar_width = group_width / n
    x = np.arange(len(CATEGORIES))
    rng = np.random.default_rng(0)

    for i, (method, _, color) in enumerate(METHODS):
        offset = (i - (n - 1) / 2) * bar_width
        shades = tier_shades(color)
        for j, (category, _) in enumerate(CATEGORIES):
            rows = df.loc[(df["category"] == category) & (df["method"] == method)]
            if rows.empty:
                continue
            bottom = 0.0
            for (tier, _), shade in zip(TIERS, shades):
                height = rows[f"{tier}_fraction"].mean()
                ax.bar(x[j] + offset, height, bottom=bottom, width=bar_width, color=shade, edgecolor=SURFACE, linewidth=0.6, zorder=2)
                bottom += height
            values = rows["recovered_fraction"].to_numpy()
            jitter = rng.uniform(-bar_width * 0.18, bar_width * 0.18, len(values))
            ax.scatter(x[j] + offset + jitter, values, s=9, color=TEXT, alpha=0.5, linewidths=0, zorder=3)

    ax.set_ylim(0, 1.03)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    ax.set_ylabel("Genomes Recovered (%)", color=TEXT_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(x)
    xlabels = {"plant_associated": "Plant-\nassociated", "strain_madness": "Strain\nmadness"}
    ax.set_xticklabels([xlabels.get(category, label) for category, label in CATEGORIES], color=TEXT)

    method_handles = [Patch(facecolor=color, label=label) for _, label, color in METHODS]
    method_handles.append(plt.Line2D([], [], marker="o", linestyle="", color=TEXT, alpha=0.5, markersize=4, label="Replicate sample"))
    method_legend = ax.legend(handles=method_handles, title="Method", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                              labelcolor=TEXT, handlelength=1.2, alignment="left", title_fontproperties={"weight": "bold"})
    ax.add_artist(method_legend)
    tier_handles = [Patch(facecolor=shade, edgecolor=SURFACE, label=label) for (_, label), shade in zip(TIERS, tier_shades("#73726c"))]
    tier_legend = ax.legend(handles=tier_handles, title="Quality tier\n(completeness / contamination)", loc="upper left", bbox_to_anchor=(1.01, 0.48), frameon=False,
              labelcolor=TEXT, handlelength=1.2, alignment="left", title_fontproperties={"weight": "bold"})

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", bbox_extra_artists=[method_legend, tier_legend], facecolor=SURFACE)


def main() -> None:
    args = parse_args()
    binning_root = config.result_table / "cami_binning"
    embedding_table = args.embedding_table or (binning_root / "table_embedding_euclidean_dbscan_scg_per_sample.tsv")
    external_table = args.external_table or (binning_root / "table_cami_comparison_external_per_sample.tsv")
    comebin_full_table = args.comebin_full_table or (binning_root / "table_comebin_full_pipeline_per_sample.tsv")
    output = args.output or (config.result_figure / "figure4b.png")

    df = load_rows(embedding_table, external_table, comebin_full_table)
    plot(df, output)
    print(f"Saved -> {output}")
    summary = summarize(df)
    summary.to_csv(output.with_suffix(".tsv"), sep="\t", index=False)
    print(f"Saved plotted values -> {output.with_suffix('.tsv')}")


if __name__ == "__main__":
    main()
