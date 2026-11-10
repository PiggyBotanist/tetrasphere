"""
Function: scatter plot of the clade-holdout generalization sweep (heldout.sh).
    - (--domain):   ar53 or bac120 -- which holdout sweep to load.
    - (--x):        x column to plot.
    - (--y):        y column to plot, e.g. pairwise_r_squared.
    - (--series):   defines coloration legend
    - (--x-order):  ordering of x column (defaults to genome/species/genus/family/order/class)
    - (--x-log):    log the x axis if it is numerical
    - (--y-log):    log the y axis if it is numerical
    - (--drop):     drop column value

Usage
-----
    python scripts/plots_and_tables/fig2d.py --domain bac120 --x heldout --y pairwise_mean_distortion --series dataset --drop fold mean std --output figures/figure2d.svg
"""
from __future__ import annotations

# Standard Import
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config

MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*"]
LEVEL_ORDER = ["genome", "species", "genus", "family", "order", "class"]

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Which holdout sweep to load, and x-y values
    parser.add_argument("--domain",    required=True, type=str, choices=["ar53", "bac120"], help="which clade-holdout sweep to load (heldout.sh --domain).")
    parser.add_argument("--x",         required=True, type=str, help="x-axis column, e.g. heldout, epoch.")
    parser.add_argument("--y",         required=True, type=str, help="y-axis column (the metric), e.g. pairwise_r_squared.")
    parser.add_argument("--output",    default=None, type=Path, help="output .png path; default result/figure/<domain>_heldout_<y>.png.")
    parser.add_argument("--table-root", default=None, type=Path, help="root the holdout_experiment/ tables are read from; default config.result_table. Point this at e.g. old_result/table if the sweep predates the live result/ tree.")

    # Additional parameters
    parser.add_argument("--series",    default="dataset", type=str, help="column defining the coloration legend, e.g. dataset (val/eval).")
    parser.add_argument("--x-order",   default=None, type=str, nargs="+", help="ordering of the x column, if categorical (space-separated); default genome/species/genus/family/order/class, restricted to levels found on disk. Ignored if x is numeric.")
    parser.add_argument("--x-log",     action="store_true", help="log the x axis, if it is numerical.")
    parser.add_argument("--y-log",     action="store_true", help="log the y axis, if it is numerical.")
    parser.add_argument("--drop",      action="append", nargs="+", default=None, metavar=("COLUMN", "VALUES"), help='drop rows where COLUMN equals one of VALUES, e.g. --drop fold mean std. Repeatable: --drop fold mean --drop dataset other.')

    # Randomization for jitter only
    parser.add_argument("--jitter",    default=0.08, type=float, help="horizontal jitter (fraction of a category slot, or of x itself if numeric) so overlapping rows separate visually.")
    parser.add_argument("--seed",      default=42, type=int)
    return parser.parse_args()

# =================================================================
# Pipeline steps
# =================================================================
def load_heldout(domain: str, table_root: Path) -> pd.DataFrame:
    """Concatenate every raw per-level holdout_experiment/{domain}_4mer_clr_<level>_validation.tsv."""
    root = table_root / f"{domain}_heldout_experiment"
    prefix, suffix = f"{domain}_4mer_clr_", "_validation.tsv"
    paths = sorted(root.glob(f"{prefix}*{suffix}"))
    if not paths:
        raise FileNotFoundError(f"no {prefix}*{suffix} files under {root}")

    frames = []
    for path in paths:
        level = path.name.removeprefix(prefix).removesuffix(suffix)
        frame = pd.read_csv(path, sep="\t")
        frame.insert(0, "heldout", level)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def default_level_order(levels: pd.Series) -> list[str]:
    """restrict to genome/species/genus/family/order/class."""
    found = set(levels)
    order = [level for level in LEVEL_ORDER if level in found]
    order += sorted(found - set(order))
    return order

def is_numeric(values: pd.Series) -> bool:
    """Whether every value in this column can be interpreted as a number."""
    return bool(pd.to_numeric(values, errors="coerce").notna().all())

def drop_rows(df: pd.DataFrame, drop_specs: list[list[str]] | None) -> pd.DataFrame:
    """Apply one or more --drop COLUMN VALUES... filters, dropping matching rows."""
    if not drop_specs:
        return df
    for column, *values in drop_specs:
        df = df[~df[column].astype(str).isin(values)]
    return df

def resolve_categorical_order(values: pd.Series, x_order: list[str] | None) -> list:
    """Explicit order if given; else first-seen order (not alphabetical -- meaningless for e.g. rank names)."""
    return x_order if x_order is not None else list(dict.fromkeys(values))

def plot_series(ax: Axes, sub: pd.DataFrame, x_column: str, y_column: str, offset: float, color: str, marker: str, label: str, jitter: float, numeric_x: bool, x_pos: dict, rng: np.random.Generator) -> None:
    """One series' raw jittered points plus its mean trend line across x."""
    if numeric_x:
        # Multiplicative offset/jitter so both scale sensibly on a (possibly log) numeric axis
        base = sub[x_column].astype(float).to_numpy()
        x_raw = base * (1 + offset) * (1 + rng.uniform(-jitter, jitter, size=len(sub)))
        means = sub.groupby(x_column)[y_column].mean().sort_index()
        line_x = means.index.to_numpy(dtype=float) * (1 + offset)
    else:
        base = np.array([x_pos[v] for v in sub[x_column]], dtype=float)
        x_raw = base + offset + rng.uniform(-jitter, jitter, size=len(sub))
        means = sub.groupby(x_column)[y_column].mean().reindex(x_pos.keys())
        line_x = np.array([x_pos[v] for v in x_pos]) + offset

    ax.scatter(x_raw, sub[y_column], color=color, marker=marker, alpha=0.45, s=45, edgecolor="none")
    ax.plot(line_x, means, color=color, marker=marker, markersize=9, markeredgecolor="black", markeredgewidth=1, linewidth=2, label=label)

# Main
def main() -> None:
    args = parse_args()
    output_path = args.output or (config.result_figure / f"{args.domain}_heldout_{args.y}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_heldout(args.domain, args.table_root or config.result_table)
    df = drop_rows(df, args.drop)

    numeric_x = is_numeric(df[args.x])
    series_values = list(dict.fromkeys(df[args.series]))
    n_series = len(series_values)
    rng = np.random.default_rng(args.seed)

    if numeric_x:
        x_pos, x_order = {}, []  # unused in this branch -- empty, not None, so the shared type stays plain dict/list
        offsets = np.linspace(-0.05, 0.05, n_series) if n_series > 1 else [0.0]
    else:
        x_order = resolve_categorical_order(df[args.x], args.x_order or (default_level_order(df[args.x]) if args.x == "heldout" else None))
        x_pos = {v: i for i, v in enumerate(x_order)}
        offsets = np.linspace(-0.15, 0.15, n_series) if n_series > 1 else [0.0]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, series in enumerate(series_values):
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 10]
        marker = MARKER_CYCLE[i % len(MARKER_CYCLE)]
        plot_series(ax, df[df[args.series] == series], args.x, args.y, offsets[i], color, marker, str(series), args.jitter, numeric_x, x_pos, rng)

    if numeric_x:
        if args.x_log:
            ax.set_xscale("log")
    else:
        ax.set_xticks(range(len(x_order)))
        ax.set_xticklabels([str(v) for v in x_order])
        if args.x_log:
            print(f"--x-log ignored: {args.x!r} is categorical, not numeric")

    if args.y_log:
        ax.set_yscale("log")

    ax.set_xlabel(args.x)
    ax.set_ylabel("Mean Distortion")
    ax.set_title(f"{args.y} by {args.x} ({args.domain})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved -> {output_path}")

if __name__ == "__main__":
    main()
