"""
Figure 2B: predicted-vs-true pairwise distance, Tetrasphere vs. kf2vec, restricted to exactly the
genome pairs kf2vec itself scored (same-subtree pairs only -- kf2vec cannot predict across
subtrees). Reuses kf2vec's own already-computed query output; does not rerun kf2vec.

Usage
-----
    python scripts/figures/fig2b_kf2vec_comparison.py --domain ar53 --fold 1
"""
from __future__ import annotations

# Standard Import
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Custom Import
from config import config
from src.system import create_logger
from src.io import hdf5
from src.model import checkpoint
from src.tetrasphere.model import Manifold
from src.tetrasphere.metric import aggregate_metrics
from src.tetrasphere.preprocess import strip_genomePrefix, load_distance, lookup_distance
from src.tetrasphere import kf2vec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", default="ar53", choices=["ar53", "bac120"], type=str)
    parser.add_argument("--fold", default=1, type=int)
    parser.add_argument("--split", default="eval", choices=["val", "eval"], type=str)
    parser.add_argument("--genome-experiment", default=None, type=str, help="default: --domain.")
    parser.add_argument("--genome-label", default=None, type=str, help="default: {domain}_4mer_clr_genome.")
    parser.add_argument("--genome-split-file", default=None, type=Path, help="default: config.gtdb_split/{domain}_4mer_clr_genome.hdf5.")
    parser.add_argument("--distance-file", default=None, type=Path, help="default: config.gtdb_feature/{domain}_distance.hdf5.")
    parser.add_argument("--kf2vec-benchmark-folder", default=None, type=Path, help="default: config.kf2vec_root/{domain}.")
    parser.add_argument("--n-plot-pairs", default=8000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output", default=None, type=Path, help="default: config.result_figure/fig2b_kf2vec_comparison_{domain}.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = create_logger("fig2b_kf2vec_comparison", output_log=config.root_log)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_name = f"fold_{args.fold}"

    genome_experiment = args.genome_experiment or args.domain
    genome_label = args.genome_label or f"{args.domain}_4mer_clr_genome"
    genome_split_file = args.genome_split_file or (config.gtdb_split / f"{args.domain}_4mer_clr_genome.hdf5")
    distance_file = args.distance_file or (config.gtdb_feature / f"{args.domain}_distance.hdf5")
    benchmark_folder = args.kf2vec_benchmark_folder or (config.kf2vec_root / args.domain)

    # Step 1: kf2vec's own scored pairs for this fold/split, exactly as in scripts/kf2vec/02_evaluate.py
    map_id_idx, sorted_keys, sorted_dist, n = load_distance(distance_file, logger)
    norm_constant = float(hdf5.get_attr(distance_file, "normalization_constant"))

    query_output = benchmark_folder / fold_name / f"query_{args.split}"
    predictions = kf2vec.load_query_predictions(query_output)
    true_dist = lookup_distance(predictions["query_id"].to_numpy(), predictions["backbone_id"].to_numpy(), map_id_idx, sorted_keys, sorted_dist, n, logger)
    valid = ~np.isnan(true_dist)
    predictions, true_dist = predictions[valid].reset_index(drop=True), true_dist[valid]
    kf_pred = predictions["pred_dist"].to_numpy() / norm_constant
    logger.info(f"kf2vec-comparable set: {len(predictions)} pairs")

    # Step 2: Tetrasphere's own prediction on the IDENTICAL genome pairs
    ckpt_path = checkpoint.find(config.result_checkpoint / genome_experiment / genome_label / fold_name, None)
    model_config = checkpoint.load_config(ckpt_path)["model_config"]
    model = Manifold(**model_config).to(device)
    checkpoint.load_weights(ckpt_path, model, device)
    model.eval()
    scale = float(model.scale().detach().cpu())

    split_data = hdf5.load(genome_split_file)
    feature_by_id: dict[str, np.ndarray] = {}
    for leaf_id, feature in zip(split_data[fold_name]["train"]["leaf_id"], split_data[fold_name]["train"]["feature"]):
        feature_by_id[leaf_id] = feature
    val_group = split_data["eval"] if args.split == "eval" else split_data[fold_name]["val"]
    for leaf_id, feature in zip(val_group["leaf_id"], val_group["feature"]):
        feature_by_id[leaf_id] = feature

    query_ids = strip_genomePrefix(predictions["query_id"].to_numpy())
    backbone_ids = strip_genomePrefix(predictions["backbone_id"].to_numpy())
    feature_a = np.stack([feature_by_id[g] for g in query_ids])
    feature_b = np.stack([feature_by_id[g] for g in backbone_ids])

    with torch.no_grad():
        za = model.encode(torch.from_numpy(feature_a).float().to(device))
        zb = model.encode(torch.from_numpy(feature_b).float().to(device))
        cos_sim = torch.nn.functional.cosine_similarity(za, zb, dim=-1).cpu().numpy()
    tetra_pred = (1.0 - cos_sim) / scale

    # Step 3: Metrics + scatter
    m_tetra = aggregate_metrics(tetra_pred, true_dist)
    m_kf = aggregate_metrics(kf_pred, true_dist)
    logger.info(f"Tetrasphere: n={len(true_dist)} mse={m_tetra['mse']:.4f} r_squared={m_tetra['r_squared']:.3f} pearson_r={m_tetra['pearson_r']:.3f} spearman_r={m_tetra['spearman_r']:.3f}")
    logger.info(f"kf2vec:      n={len(true_dist)} mse={m_kf['mse']:.4f} r_squared={m_kf['r_squared']:.3f} pearson_r={m_kf['pearson_r']:.3f} spearman_r={m_kf['spearman_r']:.3f}")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(true_dist), size=min(args.n_plot_pairs, len(true_dist)), replace=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_dist[idx], kf_pred[idx], s=4, alpha=0.15, color="tab:orange", edgecolors="none", rasterized=True)
    ax.scatter(true_dist[idx], tetra_pred[idx], s=4, alpha=0.15, color="tab:blue", edgecolors="none", rasterized=True)
    lim = 2.0
    ax.plot([0, lim], [0, lim], color="black", linewidth=1.2, linestyle="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("True distance")
    ax.set_ylabel("Predicted distance")
    ax.set_aspect("equal")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="tab:blue", markersize=8, label=f"Tetrasphere, r={m_tetra['pearson_r']:.3f}"),
        plt.Line2D([0], [0], marker="o", linestyle="", color="tab:orange", markersize=8, label=f"kf2vec, r={m_kf['pearson_r']:.3f}"),
        plt.Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="y = x"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    fig.tight_layout()

    output_path = args.output or (config.result_figure / f"fig2b_kf2vec_comparison_{args.domain}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    logger.info(f"Saved -> {output_path}")

    table_path = config.result_table / "fig2b" / f"{args.domain}_fold{args.fold}_{args.split}.tsv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"method": "tetrasphere", "n_pairs": len(true_dist), **m_tetra}, {"method": "kf2vec", "n_pairs": len(true_dist), **m_kf}]).to_csv(table_path, sep="\t", index=False)
    logger.info(f"Table -> {table_path}")


if __name__ == "__main__":
    main()
