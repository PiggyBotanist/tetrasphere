"""
Figure 2C: mean distortion vs. fragment length (log scale) for kf2vec (zero-shot), Tetrasphere
without the denoiser (zero-shot genome-stage Manifold applied directly to fragment composition),
and Tetrasphere with the denoiser (full two-phase model), fold 1 eval.

kf2vec has no fragment-length mode of its own; its zero-shot numbers come from a precomputed table
(results/table/kf2vec/{domain}_kf2vec_fragment_zeroshot.tsv) evaluated at 2000bp, while Tetrasphere's
own curriculum grid uses 2500bp at that point -- the two are not evaluated at an identical length at
this one grid point, only approximately comparable there.

Usage
-----
    python scripts/figures/fig2c_fragment_sweep.py --domain ar53 --fold 1
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
from src.tetrasphere.model import Manifold, PhylogenyModel
from src.tetrasphere.metric import aggregate_metrics

LENGTHS = [50000, 25000, 10000, 5000, 2500, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", default="ar53", choices=["ar53", "bac120"], type=str)
    parser.add_argument("--fold", default=1, type=int)
    parser.add_argument("--genome-label", default=None, type=str, help="default: {domain}_4mer_clr_genome.")
    parser.add_argument("--fragment-label", default=None, type=str, help="default: {domain}_4mer_clr_fragment.")
    parser.add_argument("--split-file-prefix", default=None, type=Path, help="default: config.gtdb_split/fragment/{domain}_4mer_clr.")
    parser.add_argument("--kf2vec-zeroshot-table", default=None, type=Path, help="default: config.result_table/kf2vec/{domain}_kf2vec_fragment_zeroshot.tsv.")
    parser.add_argument("--output", default=None, type=Path)
    return parser.parse_args()


def embed_no_denoiser(model: Manifold, feature_137: np.ndarray, device: torch.device) -> np.ndarray:
    """Strip the trailing log10(length) column (Manifold never saw one) and encode directly."""
    with torch.no_grad():
        return model.encode(torch.from_numpy(feature_137[:, :-1]).float().to(device)).cpu().numpy()


def pairwise_distortion(embed: np.ndarray, scale: float, pair: np.ndarray, pair_dist: np.ndarray) -> float:
    a, b = pair[:, 0], pair[:, 1]
    za, zb = embed[a], embed[b]
    cos_sim = np.sum(za * zb, axis=1) / (np.linalg.norm(za, axis=1) * np.linalg.norm(zb, axis=1))
    pred = (1.0 - cos_sim) / scale
    return aggregate_metrics(pred, pair_dist)["mean_distortion"]


def main() -> None:
    args = parse_args()
    logger = create_logger("fig2c_fragment_sweep", output_log=config.root_log)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_name = f"fold_{args.fold}"

    genome_label = args.genome_label or f"{args.domain}_4mer_clr_genome"
    fragment_label = args.fragment_label or f"{args.domain}_4mer_clr_fragment"
    split_prefix = args.split_file_prefix or (config.gtdb_split / "fragment" / f"{args.domain}_4mer_clr")
    kf2vec_table_path = args.kf2vec_zeroshot_table or (config.result_table / "kf2vec" / f"{args.domain}_kf2vec_fragment_zeroshot.tsv")

    genome_ckpt = checkpoint.find(config.result_checkpoint / args.domain / genome_label / fold_name, None)
    genome_model = Manifold(**checkpoint.load_config(genome_ckpt)["model_config"]).to(device)
    checkpoint.load_weights(genome_ckpt, genome_model, device)
    genome_model.eval()
    genome_scale = float(genome_model.scale().detach().cpu())

    fragment_ckpt = checkpoint.find(config.result_checkpoint / args.domain / fragment_label / fold_name, None)
    fragment_model = PhylogenyModel(**checkpoint.load_config(fragment_ckpt)["model_config"]).to(device)
    checkpoint.load_weights(fragment_ckpt, fragment_model, device)
    fragment_model.eval()
    fragment_model.stage("fragment")
    fragment_scale = float(fragment_model.scale().detach().cpu())

    rows = []
    for length in LENGTHS:
        path = Path(f"{split_prefix}_frag{length}.hdf5")
        ev = hdf5.load(path)["eval"]
        feature, pair, pair_dist = ev["feature"].astype(np.float32), ev["pair"], ev["pair_dist"]

        embed_noDenoiser = embed_no_denoiser(genome_model, feature, device)
        no_denoiser_distortion = pairwise_distortion(embed_noDenoiser, genome_scale, pair, pair_dist)

        with torch.no_grad():
            embed_denoiser = fragment_model.embed(torch.from_numpy(feature).to(device), fragment=True).cpu().numpy()
        denoiser_distortion = pairwise_distortion(embed_denoiser, fragment_scale, pair, pair_dist)

        logger.info(f"length={length}: no_denoiser_distortion={no_denoiser_distortion:.4f}  with_denoiser_distortion={denoiser_distortion:.4f}")
        rows.append({"length": length, "tetrasphere_no_denoiser": no_denoiser_distortion, "tetrasphere_with_denoiser": denoiser_distortion})

    df = pd.DataFrame(rows)
    if kf2vec_table_path.exists():
        kf_df = pd.read_csv(kf2vec_table_path, sep="\t")
        kf_by_length = dict(zip(kf_df["length"], kf_df["mean_distortion"]))
        df["kf2vec_zeroshot"] = df["length"].map(kf_by_length)
    else:
        logger.warning(f"{kf2vec_table_path} not found -- kf2vec series omitted")

    out_table = config.result_table / "fig2c" / f"{args.domain}_fold{args.fold}.tsv"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_table, sep="\t", index=False)
    logger.info(f"Table -> {out_table}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(df["length"], df["tetrasphere_no_denoiser"], marker="s", color="tab:red", label="Tetrasphere (no denoiser)")
    ax.plot(df["length"], df["tetrasphere_with_denoiser"], marker="o", color="tab:blue", label="Tetrasphere (with denoiser)")
    if "kf2vec_zeroshot" in df:
        ax.plot(df["length"], df["kf2vec_zeroshot"], marker="^", color="tab:orange", label="kf2vec (zero-shot)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fragment length (bp)")
    ax.set_ylabel("Mean distortion")
    ax.set_title(f"Fragment-length robustness ({args.domain}, {fold_name})")
    ax.legend()
    fig.tight_layout()

    output_path = args.output or (config.result_figure / f"fig2c_fragment_sweep_{args.domain}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    logger.info(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
