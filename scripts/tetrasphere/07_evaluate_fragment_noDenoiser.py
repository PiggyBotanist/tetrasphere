"""
Function: zero-shot "no denoiser" fragment ablation -- loads the GENOME-stage Manifold checkpoint
    (from 03_train_genome.py, no Phase-2 Denoiser training at all) and applies it directly to raw
    fragment CLR composition (the length column stripped, since Manifold never saw one), for every
    fold and fragment length. This measures how much Phase-2 (04_train_fragment.py) actually buys
    over just reusing the genome-scale encoder zero-shot, the counterpart to
    06_evaluate_fragment.py's "with denoiser" per-length evaluation.

    Mirrors 05_evaluate_genome.py's structure (direct hdf5 access + aggregate_metrics), not
    06_evaluate_fragment.py's Dataset/DataLoader pipeline -- that pipeline is built around
    PhylogenyModel's denoiser-then-manifold forward pass and 137-dim (CLR + length) features; here
    the model is a bare Manifold expecting 136-dim input, so the simpler direct-hdf5 pattern fits
    better and avoids forcing an unrelated Dataset shape through it.

Usage
-----
    python scripts/tetrasphere/07_evaluate_fragment_noDenoiser.py \\
        --label bac120_4mer_clr_genome --experiment bac120 \\
        --split-file-prefix /bls2/data/gtdb/split/fragment/bac120_4mer_clr \\
        --lengths 50000 25000 10000 5000 2500 1000
"""
from __future__ import annotations

# Standard Import
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch

# Custom Import
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config
from src.io import hdf5
from src.system import create_logger
from src.model import checkpoint
from src.tetrasphere.model import Manifold
from src.tetrasphere.metric import aggregate_metrics

# Arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label",             required=True, type=str, help="genome-stage training run label (03_train_genome.py's --label).")
    parser.add_argument("--experiment",        required=True, type=str, help="genome-stage experiment subfolder (03_train_genome.py's --experiment).")
    parser.add_argument("--split-file-prefix", required=True, type=Path, help="--output-prefix used by 02_split_fragment.py; resolves {prefix}_frag{length}.hdf5 per length.")
    parser.add_argument("--lengths",           required=True, type=int, nargs="+", help="space-separated fragment lengths in bp to evaluate.")
    parser.add_argument("--device",            default=None, type=str, help="e.g. 'cuda:0'; auto-selects cuda if unset.")
    parser.add_argument("--epoch",             default=None, type=int, help="genome checkpoint epoch to load (default: latest).")
    return parser.parse_args()

# =================================================================
# Pipeline steps
# =================================================================
def embed_raw(model: Manifold, feature_137: np.ndarray, device: torch.device) -> np.ndarray:
    """Strip the trailing log10(length) column (Manifold never saw one) and encode directly, no denoiser."""
    with torch.no_grad():
        return model.encode(torch.from_numpy(feature_137[:, :-1]).float().to(device)).cpu().numpy()

def validate_fold_length(model: Manifold, scale: float, fragment_eval: dict, device: torch.device) -> dict[str, float]:
    feature, pair, pair_dist, root_distance = fragment_eval["feature"], fragment_eval["pair"], fragment_eval["pair_dist"], fragment_eval["root_distance"]
    embed = embed_raw(model, feature.astype(np.float32), device)
    a, b = pair[:, 0], pair[:, 1]
    za, zb = embed[a], embed[b]
    cos_sim = np.sum(za * zb, axis=1) / (np.linalg.norm(za, axis=1) * np.linalg.norm(zb, axis=1))
    pred_pairwise = (1.0 - cos_sim) / scale
    pred_root = np.concatenate([np.linalg.norm(za, axis=1), np.linalg.norm(zb, axis=1)]) / scale
    true_root = np.concatenate([root_distance[a], root_distance[b]])

    m_pair = aggregate_metrics(pred_pairwise, pair_dist)
    m_root = aggregate_metrics(pred_root, true_root)
    return {f"pairwise_{k}": v for k, v in m_pair.items()} | {f"root_{k}": v for k, v in m_root.items()}

# Main
def main() -> None:
    args = parse_args()
    logger = create_logger(f"evaluate_fragment_noDenoiser_{args.label}", output_log=config.root_log)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lengths = sorted(set(args.lengths), reverse=True)

    ckpt_root = config.result_checkpoint / args.experiment / args.label
    fold_names = sorted((d.name for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("fold_")), key=lambda k: int(k.split("_")[1]))
    if not fold_names:
        raise FileNotFoundError(f"no fold_* checkpoints under {ckpt_root} -- run 03_train_genome.py first.")

    rows = []
    for length in lengths:
        split_path = Path(f"{args.split_file_prefix}_frag{length}.hdf5")
        fragment_eval = hdf5.load(split_path)["eval"]
        logger.info(f"length={length}: {len(fragment_eval['pair'])} eval pairs from {split_path}")

        for fold_name in fold_names:
            ckpt_path = checkpoint.find(ckpt_root / fold_name, args.epoch)
            cfg = checkpoint.load_config(ckpt_path)
            model = Manifold(**cfg["model_config"]).to(device)
            checkpoint.load_weights(ckpt_path, model, device)
            model.eval()
            scale = float(model.scale().detach().cpu())

            metrics = validate_fold_length(model, scale, fragment_eval, device)
            logger.info(f"  {fold_name}: {metrics}")
            rows.append({"fold": fold_name, "length": length, "scale": scale, **metrics})

    df = pd.DataFrame(rows)
    output_path = config.result_table / args.experiment / f"{args.label}_noDenoiser_validation.tsv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved -> {output_path}")

if __name__ == "__main__":
    main()
