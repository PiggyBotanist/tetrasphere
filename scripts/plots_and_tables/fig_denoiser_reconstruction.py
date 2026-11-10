"""
Denoiser reconstruction fidelity by fragment length.

Usage
-----
    python scripts/plots_and_tables/fig_denoiser_reconstruction.py --domain ar53 --fold 1
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
from src.tetrasphere.model import PhylogenyModel
from src.tetrasphere.dataloader import parent_id
from src.tetrasphere.preprocess import build_parent_lookup

LENGTHS = [50000, 25000, 10000, 5000, 2500, 1000]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", required=True, choices=["ar53", "bac120"], type=str)
    parser.add_argument("--fold", default=1, type=int)
    parser.add_argument("--fragment-label", default=None, type=str, help="default: {domain}_4mer_clr_fragment.")
    parser.add_argument("--split-file-prefix", default=None, type=Path, help="default: config.gtdb_split/fragment/{domain}_4mer_clr.")
    parser.add_argument("--genome-split-file", default=None, type=Path, help="default: config.gtdb_split/{domain}_4mer_clr_genome.hdf5.")
    parser.add_argument("--output", default=None, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = create_logger("fig_denoiser_reconstruction", output_log=config.root_log)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_name = f"fold_{args.fold}"

    # Load data
    fragment_label = args.fragment_label or f"{args.domain}_4mer_clr_fragment"
    split_prefix = args.split_file_prefix or (config.gtdb_split / "fragment" / f"{args.domain}_4mer_clr")
    genome_split_file = args.genome_split_file or (config.gtdb_split / f"{args.domain}_4mer_clr_genome.hdf5")

    # Load checkpoint
    ckpt_path = checkpoint.find(config.result_checkpoint / args.domain / fragment_label / fold_name, None)
    model = PhylogenyModel(**checkpoint.load_config(ckpt_path)["model_config"]).to(device)
    checkpoint.load_weights(ckpt_path, model, device)
    model.eval()

    # Build lookup table
    parent_lookup = build_parent_lookup(genome_split_file)

    # Run evaluation for each length
    rows = []
    for length in LENGTHS:
        split_path = Path(f"{split_prefix}_frag{length}.hdf5")
        ev = hdf5.load(split_path)["eval"]
        feature, leaf_id = ev["feature"], ev["leaf_id"]
        target = np.stack([parent_lookup[parent_id(lid)] for lid in leaf_id]).astype(np.float32)

        with torch.no_grad():
            denoised = model.denoiser(torch.from_numpy(feature.astype(np.float32)).to(device)).cpu().numpy()

        mse = float(np.mean((denoised - target) ** 2))
        r = float(np.corrcoef(denoised.ravel(), target.ravel())[0, 1])
        logger.info(f"length={length}: n_fragments={len(leaf_id)}  recon_mse={mse:.5f}  pearson_r={r:.4f}")
        rows.append({"length": length, "n_fragments": len(leaf_id), "recon_mse": mse, "recon_pearson_r": r})

    # Compile a dataframe to save
    df = pd.DataFrame(rows)
    fig2c_table = config.result_table / "fig2c" / f"{args.domain}_fold{args.fold}.tsv"
    if fig2c_table.exists():
        distortion = pd.read_csv(fig2c_table, sep="\t").set_index("length")["with_denoiser"]
        df["distortion"] = df["length"].map(distortion)
        df["dist_mse_ratio"] = df["distortion"] / df["recon_mse"]
        logger.info(f"distortion/reconstruction-MSE ratio range: {df['dist_mse_ratio'].min():.2f} - {df['dist_mse_ratio'].max():.2f}")
    else:
        logger.warning(f"{fig2c_table} not found -- run fig2c_precomputeTable.py --domain {args.domain} first for the ratio column")
        
    out_table = config.result_table / "denoiser_reconstruction" / f"{args.domain}_fold{args.fold}.tsv"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_table, sep="\t", index=False)
    logger.info(f"Table Summary -> {out_table}")

    # Plot the figure
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.plot(df["length"], df["recon_mse"], marker="o", color="tab:purple", label="reconstruction MSE")
    ax1.set_xscale("log")
    ax1.set_xlabel("Fragment length (bp)")
    ax1.set_ylabel("Reconstruction MSE", color="tab:purple")
    ax2 = ax1.twinx()
    ax2.plot(df["length"], df["recon_pearson_r"], marker="s", color="tab:green", label="reconstruction Pearson r")
    ax2.set_ylabel("Reconstruction Pearson r", color="tab:green")
    ax1.set_title(f"Denoiser reconstruction fidelity ({args.domain}, {fold_name})")
    fig.tight_layout()

    output_path = args.output or (config.result_figure / f"denoiser_reconstruction_{args.domain}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    logger.info(f"Figure -> {output_path}")


if __name__ == "__main__":
    main()
