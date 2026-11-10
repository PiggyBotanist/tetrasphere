"""
Appendix Table S1: average fragments generated per genome, by fragment length x domain
(mean +/- SD, min-max), read directly from the already-built fragment split .hdf5 files -- no
training, no model loading, just array shapes. Fragments are mapped back to their parent genome
via parent_id() and counted per genome.

Usage
-----
    python scripts/plots_and_tables/tableS1.py
"""
from __future__ import annotations

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import hdf5
from src.tetrasphere.dataloader import parent_id

LENGTHS = [50000, 25000, 10000, 5000, 2500, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-file-prefix", default=None, type=Path, help="default: config.gtdb_split/fragment/{domain}_4mer_clr -- resolves {prefix}_frag{length}.hdf5 per domain/length.")
    parser.add_argument("--output", default=None, type=Path, help="default: config.result_table/tableS1.tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for domain in ("ar53", "bac120"):
        prefix = args.split_file_prefix or (config.gtdb_split / "fragment" / f"{domain}_4mer_clr")
        for length in LENGTHS:
            path = Path(f"{prefix}_frag{length}.hdf5")
            data = hdf5.load(path)
            leaf_ids = list(data["fold_1"]["train"]["leaf_id"]) + list(data["fold_1"]["val"]["leaf_id"]) + list(data["eval"]["leaf_id"])
            counts = Counter(parent_id(lid) for lid in leaf_ids)
            values = np.array(list(counts.values()))
            rows.append({
                "domain": domain, "length": length,
                "n_genomes": len(values), "total_fragments": int(values.sum()),
                "mean_per_genome": float(values.mean()), "std_per_genome": float(values.std()),
                "min_per_genome": int(values.min()), "max_per_genome": int(values.max()),
            })
            print(f"[{domain}] length={length}: mean={values.mean():.2f} std={values.std():.2f} min={values.min()} max={values.max()}")

    df = pd.DataFrame(rows).sort_values(["domain", "length"], ascending=[True, False])
    out_path = args.output or (config.result_table / "tableS1.tsv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False, float_format="%.2f")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
