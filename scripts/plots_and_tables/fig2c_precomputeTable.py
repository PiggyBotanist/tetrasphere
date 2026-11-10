"""
Precompute step for Figure 2C.

Usage
-----
    python scripts/plots_and_tables/fig2c_precomputeTable.py --domain bac120
"""
from __future__ import annotations

# Standard Import
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Custom Import
from config import config
from src.system import create_logger
from src.io import hdf5
from src.model import checkpoint
from src.tetrasphere.model import Manifold, PhylogenyModel
from src.tetrasphere.metric import aggregate_metrics

LENGTHS = [50000, 25000, 10000, 5000, 2500, 1000]
FOLDS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", required=True, choices=["ar53", "bac120"], type=str)
    parser.add_argument("--folds", default=FOLDS, type=int, nargs="+")
    parser.add_argument("--genome-label", default=None, type=str, help="default: {domain}_4mer_clr_genome.")
    parser.add_argument("--fragment-label", default=None, type=str, help="default: {domain}_4mer_clr_fragment.")
    parser.add_argument("--split-file-prefix", default=None, type=Path, help="default: config.gtdb_split/fragment/{domain}_4mer_clr.")
    parser.add_argument("--kf2vec-zeroshot-table", default=None, type=Path, help="default: config.result_table/kf2vec/{domain}_kf2vec_fragment_zeroshot.tsv.")
    return parser.parse_args()


def embed_no_denoiser(model: Manifold, feature_137: np.ndarray, device: torch.device) -> np.ndarray:
    """Strip the trailing log10(length) column (Manifold never saw one) and encode directly."""
    with torch.no_grad():
        return model.encode(torch.from_numpy(feature_137[:, :-1]).float().to(device)).cpu().numpy()


def pairwise_distortion(embed: np.ndarray, pair: np.ndarray, pair_dist: np.ndarray) -> float:
    a, b = pair[:, 0], pair[:, 1]
    za, zb = embed[a], embed[b]
    cos_sim = np.sum(za * zb, axis=1) / (np.linalg.norm(za, axis=1) * np.linalg.norm(zb, axis=1))
    pred = 1.0 - cos_sim
    return aggregate_metrics(pred, pair_dist)["mean_distortion"]


def compute_fold(domain: str, fold: int, genome_label: str, fragment_label: str, split_prefix: Path,
                  kf2vec_table_path: Path, device: torch.device, logger) -> pd.DataFrame:
    fold_name = f"fold_{fold}"
    genome_ckpt = checkpoint.find(config.result_checkpoint / domain / genome_label / fold_name, None)
    genome_model = Manifold(**checkpoint.load_config(genome_ckpt)["model_config"]).to(device)
    checkpoint.load_weights(genome_ckpt, genome_model, device)
    genome_model.eval()

    fragment_ckpt = checkpoint.find(config.result_checkpoint / domain / fragment_label / fold_name, None)
    fragment_model = PhylogenyModel(**checkpoint.load_config(fragment_ckpt)["model_config"]).to(device)
    checkpoint.load_weights(fragment_ckpt, fragment_model, device)
    fragment_model.eval()
    fragment_model.stage("fragment")

    rows = []
    for length in LENGTHS:
        path = Path(f"{split_prefix}_frag{length}.hdf5")
        ev = hdf5.load(path)["eval"]
        feature, pair, pair_dist = ev["feature"].astype(np.float32), ev["pair"], ev["pair_dist"]

        embed_noDenoiser = embed_no_denoiser(genome_model, feature, device)
        no_denoiser = pairwise_distortion(embed_noDenoiser, pair, pair_dist)

        with torch.no_grad():
            embed_denoiser = fragment_model.embed(torch.from_numpy(feature).to(device), fragment=True).cpu().numpy()
        with_denoiser = pairwise_distortion(embed_denoiser, pair, pair_dist)

        logger.info(f"{fold_name} length={length}: no_denoiser={no_denoiser:.4f}  with_denoiser={with_denoiser:.4f}")
        rows.append({"length": length, "no_denoiser": no_denoiser, "with_denoiser": with_denoiser})

    df = pd.DataFrame(rows)
    if kf2vec_table_path.exists():
        kf_df = pd.read_csv(kf2vec_table_path, sep="\t")
        kf_by_length = dict(zip(kf_df["length"], kf_df["mean_distortion"]))
        df["kf2vec"] = df["length"].map(kf_by_length)
    else:
        logger.warning(f"{kf2vec_table_path} not found -- kf2vec series omitted")
    return df


def main() -> None:
    args = parse_args()
    logger = create_logger("fig2c_precomputeTable", output_log=config.root_log)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    genome_label = args.genome_label or f"{args.domain}_4mer_clr_genome"
    fragment_label = args.fragment_label or f"{args.domain}_4mer_clr_fragment"
    split_prefix = args.split_file_prefix or (config.gtdb_split / "fragment" / f"{args.domain}_4mer_clr")
    kf2vec_table_path = args.kf2vec_zeroshot_table or (config.result_table / "kf2vec" / f"{args.domain}_kf2vec_fragment_zeroshot.tsv")

    for fold in args.folds:
        df = compute_fold(args.domain, fold, genome_label, fragment_label, split_prefix, kf2vec_table_path, device, logger)
        out_table = config.result_table / "fig2c" / f"{args.domain}_fold{fold}.tsv"
        out_table.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_table, sep="\t", index=False)
        logger.info(f"Table -> {out_table}")


if __name__ == "__main__":
    main()
