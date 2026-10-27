from __future__ import annotations

# Standard Import
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

# Custom Import
from src.baseclass import BaseModel
from src.tetrasphere.metric import aggregate_metrics

# =================================================================
# Residual block
# =================================================================
class ResidualBlock(nn.Module):
    """One residual MLP block with LayerNorm and dropout, added back to its own input."""
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))

# =================================================================
# Manifold Embedding (Genome Model)
# =================================================================
class Manifold(BaseModel):
    """Phylogeny Manifold"""
    def __init__(self, dim_input: int, dim_hidden: int, dim_output: int, layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim_input  = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.layers      = layers
        self.dropout     = dropout

        self.input  = nn.Linear(dim_input, dim_hidden)
        self.blocks = nn.ModuleList([ResidualBlock(dim_hidden, dropout) for _ in range(layers)])
        self.output = nn.Linear(dim_hidden, dim_output)

    @property
    def config(self) -> dict[str, object]:
        return {
            "dim_input":  self.dim_input,
            "dim_hidden": self.dim_hidden,
            "dim_output": self.dim_output,
            "layers":     self.layers,
            "dropout":    self.dropout,
        }
        
    def encode(self, x: Tensor) -> Tensor:
        """Emebedding for feature vector"""
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)
        
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            "embed_a": self.encode(batch["feature_a"]),
            "embed_b": self.encode(batch["feature_b"]),
        }

# =================================================================
# Denoiser: (fragmented k-mer + log10 length) -> denoised (parent-like) composition
# =================================================================
class Denoiser(nn.Module):
    """Denoiser before feeding into Manifold"""
    def __init__(self, dim_input: int, dim_output: int, dim_hidden: int = 256, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.input  = nn.Linear(dim_input, dim_hidden)
        self.blocks = nn.ModuleList([ResidualBlock(dim_hidden, dropout) for _ in range(layers)])
        self.output = nn.Linear(dim_hidden, dim_output)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        out = self.output(x)
        # Force re-centering (so the result aligns with CLR by nature)
        return out - out.mean(dim=-1, keepdim=True)

# =================================================================
# Integrated genome + fragment model
# =================================================================
class PhylogenyModel(BaseModel):
    """Denoiser -> Manifold"""
    def __init__(
        self,
        dim_clean: int,                 # input dimension for {denoiser -> manifold}
        dim_fragment: int,              # input dimension for {denoiser}
        dim_output: int = 128,          # output dimension for embedding space
        manifold_hidden: int = 256,     # (manifold) hidden dimension width
        manifold_layers: int = 2,       # (manifold) manifold hidden layers   
        denoiser_hidden: int = 256,     # (denoiser) hidden dimension width
        denoiser_layers: int = 2,       # (denoiser) denoiser hidden layers
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim_clean       = dim_clean
        self.dim_fragment    = dim_fragment
        self.dim_output      = dim_output
        self.manifold_hidden = manifold_hidden
        self.manifold_layers = manifold_layers
        self.denoiser_hidden = denoiser_hidden
        self.denoiser_layers = denoiser_layers
        self.dropout         = dropout

        self.manifold = Manifold(dim_clean, manifold_hidden, dim_output, manifold_layers, dropout)
        self.denoiser = Denoiser(dim_fragment, dim_clean, denoiser_hidden, denoiser_layers, dropout)

        # Default training stage to "fragment" (flow through entire network).
        # Toggle between "genome" or "fragment"
        self._stage = "fragment"
        self.stage("fragment")

    def stage(self, name: str) -> "PhylogenyModel":
        """
        if "genome": freeze denoiser
        if "fragment": freeze manifold
        """
        if name not in ("genome", "fragment"):
            raise ValueError(f"stage must be 'genome' or 'fragment', got {name!r}")
        self._stage = name
        if name == "fragment":
            self.manifold.requires_grad_(False)
            self.denoiser.requires_grad_(True)
        else:
            self.manifold.requires_grad_(True)
            self.denoiser.requires_grad_(False)
        return self

    def train(self, mode: bool = True) -> "PhylogenyModel":
        """Freeze manifold if we are training on fragment, otherwise no need"""
        super().train(mode)
        if self._stage == "fragment":
            self.manifold.eval()
        return self

    def embed(self, x: Tensor, fragment: bool) -> Tensor:
        """Embed a single vector into embedding space"""
        if fragment:
            x = self.denoiser(x)
        return self.manifold.encode(x)

    @property
    def config(self) -> dict[str, object]:
        return {
            "dim_clean":       self.dim_clean,
            "dim_fragment":    self.dim_fragment,
            "dim_output":      self.dim_output,
            "manifold_hidden": self.manifold_hidden,
            "manifold_layers": self.manifold_layers,
            "denoiser_hidden": self.denoiser_hidden,
            "denoiser_layers": self.denoiser_layers,
            "dropout":         self.dropout,
        }

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Pair forward for training, at whichever stage is currently active. 'genome' stage
        expects feature_a/b at dim_clean width; 'fragment' stage expects dim_fragment width."""
        if self._stage == "fragment":
            denoised_a, denoised_b = self.denoiser(batch["feature_a"]), self.denoiser(batch["feature_b"])
            return {
                "denoised_a": denoised_a,
                "denoised_b": denoised_b,
                "embed_a":    self.manifold.encode(denoised_a),
                "embed_b":    self.manifold.encode(denoised_b),
            }
        return {
            "embed_a": self.manifold.encode(batch["feature_a"]),
            "embed_b": self.manifold.encode(batch["feature_b"]),
        }

# =================================================================
# Validation hook for training
# =================================================================
def full_validation(model: Manifold | PhylogenyModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Forward every batch, pool predictions/labels across the whole loader, then compute each metric once."""
    model.eval()
    pairwise_pred = []
    pairwise_label = []
    root_pred = []
    root_label = []

    with torch.no_grad():
        for batch in loader:
            # Model pass
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output = model(batch)
            embed_a, embed_b = output["embed_a"], output["embed_b"]

            # Root and Pairwise Distance
            cos_sim = F.cosine_similarity(embed_a, embed_b, dim=-1)
            pairwise_pred.append((1.0 - cos_sim).cpu().numpy())
            pairwise_label.append(batch["pairwise_distaince"].cpu().numpy())
            root_pred.append(torch.cat([embed_a.norm(dim=-1), embed_b.norm(dim=-1)]).cpu().numpy())
            root_label.append(torch.cat([batch["root_distance_a"], batch["root_distance_b"]]).cpu().numpy())

    pairwise_metrics    = aggregate_metrics(np.concatenate(pairwise_pred), np.concatenate(pairwise_label), 1e-6)
    root_metrics        = aggregate_metrics(np.concatenate(root_pred), np.concatenate(root_label))
    return {
        **{f"pairwise_{k}": float(v) for k, v in pairwise_metrics.items()},
        **{f"root_{k}": float(v) for k, v in root_metrics.items()},
    }
