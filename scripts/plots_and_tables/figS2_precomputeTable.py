"""
Specifically for appendix figure S2. 

NOTE: predicted pairwise distance is plain (1 - cos_sim).

Usage
-----
    python scripts/plots_and_tables/figS2_precomputeTable.py --domain ar53
    python scripts/plots_and_tables/figS2_precomputeTable.py --domain bac120
"""
from __future__ import annotations

import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import hdf5
from src.model import checkpoint
from src.tetrasphere.model import Manifold
from src.tetrasphere.metric import pairwise_errors
from src.tetrasphere.preprocess import get_taxonomy, strip_genomePrefix

N_BUCKETS = 5

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", required=True, choices=["ar53", "bac120"])
    parser.add_argument("--folds",  default=["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"], nargs="+")
    parser.add_argument("--split-file",    default=None, type=Path, help="default: config.gtdb_split/{domain}_4mer_clr_genome.hdf5.")
    parser.add_argument("--taxonomy-file", default=None, type=Path, help="default: config.gtdb_download/{domain}_taxonomy_r232.tsv.gz.")
    parser.add_argument("--output-dir",    default=None, type=Path, help="default: config.result_table/figS2/{domain}.")
    return parser.parse_args()


def load_model(ckpt_path: Path, dim_input: int, device: torch.device) -> Manifold:
    model = Manifold(dim_input, dim_hidden=256, dim_output=128, layers=2, dropout=0.1).to(device)
    checkpoint.load_weights(ckpt_path, model, device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    split_file = args.split_file or (config.gtdb_split / f"{args.domain}_4mer_clr_genome.hdf5")
    taxonomy_file = args.taxonomy_file or (config.gtdb_download / f"{args.domain}_taxonomy_r232.tsv.gz")
    out_dir = args.output_dir or (config.result_table / "figS2" / args.domain)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment = f"{args.domain}_class_weighting"
    label_uniform = f"{args.domain}_genome_none"
    label_weighted = f"{args.domain}_genome_inverse_phylum"

    data = hdf5.load(split_file)
    eval_split = data["eval"]
    df_taxonomy = pd.read_csv(taxonomy_file, sep="\t", compression="gzip", names=["id", "taxonomy"])

    eval_phylum = get_taxonomy(df_taxonomy, map=config.taxonomy_map, order=list(strip_genomePrefix(eval_split["leaf_id"])), level="phylum")["phylum"].to_numpy()
    pairs = eval_split["pair"]
    true_pairwise = eval_split["pair_dist"]
    near_zero = true_pairwise < 1e-6
    feature = torch.from_numpy(eval_split["feature"]).float().to(device)
    dim_input = feature.shape[1]

    per_fold_rows = []
    aggregate_rows = []
    for fold in args.folds:
        ckpt_uniform = checkpoint.find(config.result_checkpoint / experiment / label_uniform / fold, None)
        ckpt_weighted = checkpoint.find(config.result_checkpoint / experiment / label_weighted / fold, None)
        if not (ckpt_uniform.exists() and ckpt_weighted.exists()):
            print(f"[{args.domain} {fold}] missing checkpoint(s), skipping")
            continue

        # Rarity reference: this fold's train-split phylum counts
        train_split = data[fold]["train"]
        train_phylum = get_taxonomy(df_taxonomy, map=config.taxonomy_map, order=list(strip_genomePrefix(train_split["leaf_id"])), level="phylum")["phylum"].to_numpy()
        phylum_to_count = dict(zip(*np.unique(train_phylum, return_counts=True)))
        eval_genome_count = np.array([phylum_to_count.get(p, 0) for p in eval_phylum])
        rarity = np.minimum(eval_genome_count[pairs[:, 0]], eval_genome_count[pairs[:, 1]])
        quantile_edges = np.quantile(rarity[~near_zero], np.linspace(0, 1, N_BUCKETS + 1))
        bucket = np.clip(np.searchsorted(quantile_edges, rarity, side="right") - 1, 0, N_BUCKETS - 1)

        fold_results = {}
        for cond, ckpt in [("uniform", ckpt_uniform), ("inverse_phylum", ckpt_weighted)]:
            model = load_model(ckpt, dim_input, device)
            with torch.no_grad():
                embed = model.encode(feature)
            embed = embed.cpu().numpy()
            cos_sim = np.sum(embed[pairs[:, 0]] * embed[pairs[:, 1]], axis=1) / (
                np.linalg.norm(embed[pairs[:, 0]], axis=1) * np.linalg.norm(embed[pairs[:, 1]], axis=1)
            )
            pred_pairwise = 1.0 - cos_sim
            err = pairwise_errors(pred_pairwise, true_pairwise, 1e-6)
            fold_results[cond] = err["distortion"].to_numpy()
            overall = err.loc[~err["near_zero"], "distortion"].mean()
            print(f"[{args.domain} {fold}] {cond}: overall_distortion={overall:.4f}")
            aggregate_rows.append({"domain": args.domain, "fold": fold, "condition": cond, "overall_distortion": overall})

        for b in range(N_BUCKETS):
            mask = (bucket == b) & (~near_zero)
            if mask.sum() == 0:
                continue
            lo, hi = rarity[mask].min(), rarity[mask].max()
            d_uniform = fold_results["uniform"][mask].mean()
            d_weighted = fold_results["inverse_phylum"][mask].mean()
            per_fold_rows.append({
                "domain": args.domain, "fold": fold, "bucket": b, "rarity_min": lo, "rarity_max": hi,
                "n_pairs": int(mask.sum()), "distortion_uniform": d_uniform, "distortion_inverse_phylum": d_weighted,
                "delta": d_weighted - d_uniform,
            })

    df_bucket = pd.DataFrame(per_fold_rows)
    df_agg = pd.DataFrame(aggregate_rows)
    df_bucket.to_csv(out_dir / f"rarity_bucket_comparison_{args.domain}_5fold.tsv", sep="\t", index=False)
    df_agg.to_csv(out_dir / f"aggregate_comparison_{args.domain}_5fold.tsv", sep="\t", index=False)

    print(f"\n=== {args.domain}: aggregate eval distortion, mean +/- std across {df_agg['fold'].nunique()} folds ===")
    print(df_agg.groupby("condition")["overall_distortion"].agg(["mean", "std"]))

    print(f"\n=== {args.domain}: per-bucket distortion, mean +/- std across folds ===")
    summary = df_bucket.groupby("bucket").agg(
        rarity_min=("rarity_min", "min"), rarity_max=("rarity_max", "max"),
        n_pairs_total=("n_pairs", "sum"),
        distortion_uniform_mean=("distortion_uniform", "mean"), distortion_uniform_std=("distortion_uniform", "std"),
        distortion_weighted_mean=("distortion_inverse_phylum", "mean"), distortion_weighted_std=("distortion_inverse_phylum", "std"),
        delta_mean=("delta", "mean"), delta_std=("delta", "std"),
    )
    print(summary.to_string())
    summary.to_csv(out_dir / f"rarity_bucket_summary_{args.domain}_5fold.tsv", sep="\t")
    print(f"Summary saved to {out_dir}/rarity_bucket_comparison_{args.domain}_5fold.tsv, aggregate_comparison_{args.domain}_5fold.tsv, rarity_bucket_summary_{args.domain}_5fold.tsv")


if __name__ == "__main__":
    main()
