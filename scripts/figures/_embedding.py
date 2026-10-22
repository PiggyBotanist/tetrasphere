"""
Shared helper -- not a standalone CLI. Loads the domain's fragment-curriculum tetrasphere checkpoint
(the only trained model that's actually robust to short/variable-length input, since it was trained
on a length-curriculum blend of whole genomes and fragments -- see main/04_train_fragment.py) and
embeds raw sequences with it. Imported by scatterChimera.py and scatterGene.py.
"""
from __future__ import annotations

# Standard Import
import logging
import numpy as np
import torch
from pathlib import Path

# Custom Import
from config import config
from src.model import checkpoint
from src.tetrasphere.kmer import Kmer
from src.tetrasphere.model import PhylogenyModel

# The only trained checkpoint per domain that handles short/variable-length input (see module docstring).
FRAGMENT_CHECKPOINT = {
    "ar53":   {"experiment": "ar53",    "label": "ar53_4mer_clr_fragment"},
    "bac120": {"experiment": "bac120",  "label": "bac120_4mer_clr_fragment"},
}
K = 4
TRANSFORM = "clr"

def load_model(domain: str, fold: int, epoch: int | None, device: torch.device) -> tuple[PhylogenyModel, Kmer]:
    """Load the domain's fragment-curriculum checkpoint (frozen, eval mode) + a matching Kmer vectorizer (k=4, clr, +log length)."""
    paths = FRAGMENT_CHECKPOINT[domain]
    ckpt_root = config.result_checkpoint / paths["experiment"] / paths["label"] / f"fold_{fold}"
    ckpt_path = checkpoint.find(ckpt_root, epoch)
    cfg = checkpoint.load_config(ckpt_path)
    model = PhylogenyModel(**cfg["model_config"]).to(device)
    checkpoint.load_weights(ckpt_path, model, device)
    model.stage("fragment")
    model.eval()
    kmer = Kmer(k=K, logger=logging.getLogger("visualize_embedding"))
    return model, kmer

def embed_sequences(model: PhylogenyModel, kmer: Kmer, sequences: list[str], device: torch.device) -> np.ndarray:
    """Vectorize each raw sequence (k=4, clr, +log10 length -- matching the checkpoint's own training input) and embed it through the model's denoiser -> manifold."""
    vectors = np.stack([kmer.vectorize(seq, include_length=True, transform_kmer=TRANSFORM) for seq in sequences])
    with torch.no_grad():
        return model.embed(torch.from_numpy(vectors).float().to(device), fragment=True).cpu().numpy()
