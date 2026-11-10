"""
Appendix Table S2: number of genomes/fragments used in each split (Train/Validation/Evaluation).

Usage
-----
    python scripts/plots_and_tables/tableS2.py
"""
from __future__ import annotations

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import hdf5

LENGTHS = [50000, 25000, 10000, 5000, 2500, 1000]
FOLDS = [1, 2, 3, 4, 5]

def count_group(group: dict) -> tuple[int, int]:
    """(n_items, n_pairs) for one split group (has 'leaf_id' and 'pair')."""
    return len(group["leaf_id"]), len(group["pair"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--genome-split-file-template", default=None, type=str, help="default: config.gtdb_split/{domain}_4mer_clr_genome.hdf5")
    parser.add_argument("--fragment-split-prefix-template", default=None, type=str, help="default: config.gtdb_split/fragment/{domain}_4mer_clr -- resolves {prefix}_frag{length}.hdf5")
    parser.add_argument("--output", default=None, type=Path, help="default: config.result_table/tableS2.tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_fold_rows = []

    for domain in ("ar53", "bac120"):
        genome_path = Path((args.genome_split_file_template or str(config.gtdb_split / "{domain}_4mer_clr_genome.hdf5")).format(domain=domain))
        data = hdf5.load(genome_path)
        n_eval, p_eval = count_group(data["eval"])
        for fold in FOLDS:
            n_train, p_train = count_group(data[f"fold_{fold}"]["train"])
            n_val, p_val = count_group(data[f"fold_{fold}"]["val"])
            per_fold_rows.append({"domain": domain, "stage": "genome", "length": "whole_genome", "fold": fold,
                                   "n_train": n_train, "n_val": n_val, "n_eval": n_eval})

        fragment_prefix = (args.fragment_split_prefix_template or str(config.gtdb_split / "fragment" / "{domain}_4mer_clr")).format(domain=domain)
        for length in LENGTHS:
            frag_path = Path(f"{fragment_prefix}_frag{length}.hdf5")
            data = hdf5.load(frag_path)
            n_eval, p_eval = count_group(data["eval"])
            for fold in FOLDS:
                n_train, p_train = count_group(data[f"fold_{fold}"]["train"])
                n_val, p_val = count_group(data[f"fold_{fold}"]["val"])
                per_fold_rows.append({"domain": domain, "stage": "fragment", "length": length, "fold": fold,
                                       "n_train": n_train, "n_val": n_val, "n_eval": n_eval})
        print(f"[{domain}] done")

    df = pd.DataFrame(per_fold_rows)

    # Aggregate across the 5 folds into min-max ranges; eval is identical across folds (shared holdout).
    summary_rows = []
    for (domain, stage, length), g in df.groupby(["domain", "stage", "length"], sort=False):
        summary_rows.append({
            "domain": domain, "stage": stage, "length": length,
            "n_train_min": g["n_train"].min(), "n_train_max": g["n_train"].max(),
            "n_val_min": g["n_val"].min(), "n_val_max": g["n_val"].max(),
            "n_eval": int(g["n_eval"].iloc[0]),
            "n_eval_const_across_folds": bool(g["n_eval"].nunique() == 1),
        })
    summary = pd.DataFrame(summary_rows)

    out_path = args.output or (config.result_table / "tableS2.tsv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
