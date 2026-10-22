"""
Function: bar chart of a k-mer-size sweep (kmer.sh), with error bars across CV folds, one k
    highlighted (bold outline) as the chosen value. Optionally appends extra one-off comparison
    bars from other experiments' validation .tsv (e.g. a frequency-transform run from
    transform.sh, to compare against the CLR sweep at the same k).

    Every input .tsv is a 03_train_genome.py/05_evaluate_genome.py-style *_validation.tsv: one row
    per (fold, dataset) plus trailing "mean"/"std" summary rows -- those summary rows are always
    dropped here (treating them as extra folds silently corrupts the aggregate, since they're
    already an aggregate themselves).

Usage
-----
    python scripts/figures/kmer_sweep.py --domain ar53 --metric pairwise_mean_distortion
"""
from __future__ import annotations

# Standard Import
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Sweep definition
    parser.add_argument("--domain",       required=True, type=str, choices=["ar53", "bac120"], help="which domain's sweep to load.")
    parser.add_argument("--experiment",   default="kmer_experiment", type=str, help="table-root subfolder the sweep's *_validation.tsv files live under (kmer.sh's --experiment).")
    parser.add_argument("--table-root",   default=None, type=Path, help="root folder tables are read from; default config.result_table. Point this at e.g. old_result/table for a sweep whose results predate the live result/ tree.")
    parser.add_argument("--kmer-min",     default=2, type=int, help="smallest k in the sweep.")
    parser.add_argument("--kmer-max",     default=7, type=int, help="largest k in the sweep.")
    parser.add_argument("--transform",    default="clr", type=str, help="transform suffix used in the sweep's table filenames, e.g. {domain}_{k}mer_{transform}_validation.tsv.")
    parser.add_argument("--best-k",       default=4, type=int, help="which k gets the bold outline (the value actually used elsewhere in the paper/pipeline).")

    # What to plot
    parser.add_argument("--metric",       default="pairwise_mean_distortion", type=str, help="column to plot, e.g. pairwise_mean_distortion, pairwise_mse, pairwise_pearson_r.")
    parser.add_argument("--dataset",      default="eval", type=str, help="which dataset split's rows to use (val/eval).")

    # Optional one-off comparison bars (e.g. a transform.sh run at a fixed k)
    parser.add_argument("--extra-label",  default=None, type=str, nargs="+", help="label(s) for extra comparison bar(s), e.g. \"freq. (k=4)\".")
    parser.add_argument("--extra-table",  default=None, type=str, nargs="+", help="matching result/table-relative path(s) for each --extra-label, e.g. transform_experiment/ar53_4mer_frequency_validation.tsv.")

    # Output
    parser.add_argument("--output",       default=None, type=Path, help="output .png path; default result/figure/<domain>_kmer_sweep_<metric>.png.")
    return parser.parse_args()

# =================================================================
# Pipeline steps
# =================================================================
def eval_mean_std(path: Path, metric: str, dataset: str) -> tuple[float, float]:
    """Mean/std of one metric column, across CV folds, for one dataset split -- trailing
    mean/std summary rows in the .tsv are excluded first, so they aren't double-counted as folds."""
    df = pd.read_csv(path, sep="\t")
    df = df[~df["fold"].isin(["mean", "std"])]
    df = df[df["dataset"] == dataset]
    if df.empty:
        raise ValueError(f"{path}: no rows left for dataset={dataset!r} after dropping mean/std fold rows")
    return float(df[metric].mean()), float(df[metric].std())

# Main
def main() -> None:
    args = parse_args()
    table_root = args.table_root or config.result_table
    sweep_dir = table_root / args.experiment

    labels, means, stds, colors, edgewidths = [], [], [], [], []
    for k in range(args.kmer_min, args.kmer_max + 1):
        path = sweep_dir / f"{args.domain}_{k}mer_{args.transform}_validation.tsv"
        mean, std = eval_mean_std(path, args.metric, args.dataset)
        labels.append(str(k))
        means.append(mean)
        stds.append(std)
        colors.append("tab:orange")
        edgewidths.append(2.5 if k == args.best_k else 0.8)

    if args.extra_label:
        if not args.extra_table or len(args.extra_table) != len(args.extra_label):
            raise ValueError("--extra-table must be given, with the same count as --extra-label.")
        for label, rel_path in zip(args.extra_label, args.extra_table):
            mean, std = eval_mean_std(table_root / rel_path, args.metric, args.dataset)
            labels.append(label)
            means.append(mean)
            stds.append(std)
            colors.append("tab:orange")
            edgewidths.append(0.8)

    fig, ax = plt.subplots(figsize=(5, 5.5))
    ax.bar(labels, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=edgewidths)
    ax.set_xlabel(f"$k$-mer")
    ax.set_ylabel(args.metric)
    ax.set_title(f"$k$-mer size sweep ({args.domain}, {args.dataset})")
    fig.tight_layout()

    output_path = args.output or (config.result_figure / f"{args.domain}_kmer_sweep_{args.metric}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved -> {output_path}")

    print(f"\n{'series':15s} {args.metric} (mean +/- std)")
    for label, mean, std in zip(labels, means, stds):
        print(f"{label:15s} {mean:.4f} +/- {std:.4f}")

if __name__ == "__main__":
    main()
