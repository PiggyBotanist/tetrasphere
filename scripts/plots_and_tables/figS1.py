"""
Appendix Figure S1: train/validation loss curves for both training phases.

Usage
-----
    python scripts/plots_and_tables/figS1.py --output figures/appendix_figureS1.svg
"""
from __future__ import annotations

# Standard Import
import re
import ast
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config

LAMBDA_PAIRWISE, LAMBDA_ROOT, LAMBDA_RECON = 0.8, 0.2, 1.0
N_FOLDS = 5

FOLD_SUMMARY_RE = re.compile(r"fold_(\d+)[^:]*summary: (\{.*\})\s*$")

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+train/loss_pairwise=([\d.eE+-]+)\s+train/loss_root=([\d.eE+-]+)\s+"
    r"train/cos_sim_mean=[\d.eE+-]+\s+train/norm_mean=[\d.eE+-]+\s+"
    r"(?:train/scale=[\d.eE+-]+\s+)?(?:train/loss_recon=([\d.eE+-]+)\s+)?"
    r"train/distortion_pairwise=[\d.eE+-]+\s+train/loss=([\d.eE+-]+)\s+"
    r"val/mse_pairwise=([\d.eE+-]+)\s+val/distortion_mean_pairwise=[\d.eE+-]+\s+"
    r"val/mse_root=([\d.eE+-]+)\s+val/distortion_mean_root=[\d.eE+-]+\s+"
    r"(?:val/scale=[\d.eE+-]+\s+)?(?:val/mse_recon=([\d.eE+-]+)\s+)?lr="
)

SUMMARY_RE = re.compile(r"fold_(\d+) summary: best_epoch=(\d+)\s+val=(\{.*?\})\s+eval=(\{.*?\})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", default=None, type=Path, help="default: config.root_log.")
    parser.add_argument("--output",  default=None, type=Path, help="default: config.result_figure/figS1.png.")
    return parser.parse_args()


def find_log(log_dir: Path, domain: str, phase: str) -> Path | None:
    """Most recent matching training log for (domain, phase); exact filename timestamps change
    every rerun, so this globs rather than hardcoding one."""
    matches = sorted(log_dir.glob(f"*_train_{domain}_4mer_clr_{phase}.log"))
    return matches[-1] if matches else None


def weighted_loss(d: dict) -> float:
    loss = LAMBDA_PAIRWISE * d["mse_pairwise"] + LAMBDA_ROOT * d["mse_root"]
    if "mse_recon" in d:
        loss += LAMBDA_RECON * d["mse_recon"]
    return loss


def fold_segments(text: str) -> list[str]:
    """Split the log into N_FOLDS segments at '=== fold_N ===' markers. Some logs (e.g. ar53/genome)
    have a duplicate back-to-back marker per fold (double-handler logging bug) -- for each fold
    number, use the text after its LAST marker occurrence, up to the next fold's first marker."""
    starts = []
    for i in range(1, N_FOLDS + 1):
        markers = [m.start() for m in re.finditer(rf"=== fold_{i} ===", text)]
        if not markers:
            raise ValueError(f"no '=== fold_{i} ===' marker found")
        starts.append(markers[-1])
    ends = starts[1:] + [len(text)]
    return [text[s:e] for s, e in zip(starts, ends)]


def parse_log(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    text = path.read_text()
    epoch_rows, summary_rows = [], []
    for fold_num, segment in enumerate(fold_segments(text), start=1):
        for m in EPOCH_RE.finditer(segment):
            epoch, train_pairwise, train_root, train_recon, train_loss, val_pairwise, val_root, val_recon = m.groups()
            val_loss = LAMBDA_PAIRWISE * float(val_pairwise) + LAMBDA_ROOT * float(val_root)
            if val_recon is not None:
                val_loss += LAMBDA_RECON * float(val_recon)
            epoch_rows.append({"fold": fold_num, "epoch": int(epoch), "train_loss": float(train_loss), "val_loss": val_loss})

        sm = SUMMARY_RE.search(segment)
        if sm is None:
            raise ValueError(f"no fold_{fold_num} summary line found in {path}")
        best_epoch = int(sm.group(2))
        eval_dict = ast.literal_eval(sm.group(4))
        summary_rows.append({"fold": fold_num, "epoch": best_epoch, "eval_loss": weighted_loss(eval_dict)})

    epoch_df = pd.DataFrame(epoch_rows)
    # Some logs (e.g. ar53/genome) have every epoch line duplicated by a double-handler logging
    # bug (identical content, ~6s apart) -- dedupe by (fold, epoch), keep the first occurrence.
    epoch_df = epoch_df.drop_duplicates(subset=["fold", "epoch"], keep="first").reset_index(drop=True)
    return epoch_df, pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir or config.root_log
    output_path = args.output or (config.result_figure / "figS1.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False)
    all_epoch, all_summary = {}, {}
    rows_order = ["ar53", "bac120"]
    cols_order = ["genome", "fragment"]
    for domain in rows_order:
        for phase in cols_order:
            path = find_log(log_dir, domain, phase)
            if path is None:
                print(f"MISSING: no *_train_{domain}_4mer_clr_{phase}.log under {log_dir}")
                continue
            epoch_df, summary_df = parse_log(path)
            all_epoch[(domain, phase)] = epoch_df
            all_summary[(domain, phase)] = summary_df
            print(f"[{domain}/{phase}] parsed {len(epoch_df)} epoch-rows across {epoch_df['fold'].nunique()} folds from {path.name}")
            print(f"[{domain}/{phase}] eval_loss by fold: {dict(zip(summary_df['fold'], summary_df['eval_loss'].round(4)))}")

    for i, domain in enumerate(rows_order):
        for j, phase in enumerate(cols_order):
            ax = axes[i, j]
            epoch_df = all_epoch.get((domain, phase))
            summary_df = all_summary.get((domain, phase))
            if epoch_df is None or epoch_df.empty:
                ax.set_title(f"{domain} / {phase} (no data)")
                continue
            for fold, fdf in epoch_df.groupby("fold"):
                fdf = fdf.sort_values("epoch")
                ax.plot(fdf["epoch"], fdf["train_loss"], color="tab:blue", alpha=0.5, linewidth=1.2)
                ax.plot(fdf["epoch"], fdf["val_loss"], color="tab:orange", alpha=0.5, linewidth=1.2)

            handles = [
                plt.Line2D([0], [0], color="tab:blue", alpha=0.7, label="train loss (5 folds)"),
                plt.Line2D([0], [0], color="tab:orange", alpha=0.7, label="val loss (5 folds, reconstructed)"),
            ]
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_ylabel("log loss")
            phase_num = "1" if phase == "genome" else "2"
            stage_label = "genome stage" if phase == "genome" else "fragment stage (denoiser)"
            ax.set_title(f"{domain} (Phase {phase_num}): {stage_label}", fontsize=10)
            ax.legend(handles=handles, fontsize=8)
            ax.grid(True, which="both", linewidth=0.4, color="0.85", zorder=0)
            ax.set_axisbelow(True)

    fig.suptitle("Train/val loss curves, all 5 folds, pub-repo production checkpoints (Appendix Figure S1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved -> {output_path}")

    # Also dump the raw parsed data for the appendix table / future replot
    combined_epoch, combined_summary = [], []
    for (domain, phase), df in all_epoch.items():
        d = df.copy(); d.insert(0, "phase", phase); d.insert(0, "domain", domain)
        combined_epoch.append(d)
    for (domain, phase), df in all_summary.items():
        d = df.copy(); d.insert(0, "phase", phase); d.insert(0, "domain", domain)
        combined_summary.append(d)
    table_dir = config.result_table / "figS1"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(combined_epoch, ignore_index=True).to_csv(table_dir / "loss_curves_epoch.tsv", sep="\t", index=False)
    pd.concat(combined_summary, ignore_index=True).to_csv(table_dir / "loss_curves_eval.tsv", sep="\t", index=False)
    print(f"Saved -> {table_dir / 'loss_curves_epoch.tsv'}")
    print(f"Saved -> {table_dir / 'loss_curves_eval.tsv'}")


if __name__ == "__main__":
    main()
