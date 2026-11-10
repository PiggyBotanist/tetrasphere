"""
Appendix Table S3: fragment-length curriculum in percentage across different epochs.

Usage
-----
    python scripts/plots_and_tables/tableS3.py
"""
from __future__ import annotations

import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.tetrasphere.curriculum import blend_weights

LENGTHS = [50000, 25000, 10000, 5000, 2500, 1000]
EPOCHS = [1, 5, 10, 15, 20, 25, 26, 30, 35, 40, 45, 50]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--total-epochs", default=50, type=int, help="04_train_fragment.py's --epoch default.")
    parser.add_argument("--alpha", default=2.0, type=float, help="04_train_fragment.py's --curriculum-alpha default.")
    parser.add_argument("--output", default=None, type=Path, help="default: config.result_table/tableS3.tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for epoch_1indexed in EPOCHS:
        weights = blend_weights(epoch_1indexed - 1, args.total_epochs, len(LENGTHS), args.alpha)
        row = {"epoch": epoch_1indexed}
        row.update({f"{length}bp_pct": 100.0 * w for length, w in zip(LENGTHS, weights)})
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = args.output or (config.result_table / "tableS3.tsv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False, float_format="%.2f")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
