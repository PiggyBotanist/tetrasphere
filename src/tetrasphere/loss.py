from __future__ import annotations

# Standard Import
import logging
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

# Custom Import
from src.baseclass import BaseLoss, BaseModel
from src.tetrasphere.metric import aggregate_metrics, pairwise_errors

# =================================================================
# Loss function
# =================================================================
class Phylogeny_Loss(BaseLoss):
    """Regress on """
    def __init__(self, lambda_recon: float = 1.0, lambda_pairwise: float = 0.8, lambda_root: float = 0.2, log_eps: float = 0.01) -> None:
        super().__init__()
        self.lambda_recon    = lambda_recon         # Weight for fragment-stage reconstruction (unused in genome stage)
        self.lambda_pairwise = lambda_pairwise      # Weight for pairwise distance
        self.lambda_root     = lambda_root          # Weight for root distance
        self.log_eps         = log_eps              # Floating point stability

    def forward(self, output: dict[str, Tensor], batch: dict[str, Tensor], epoch: int) -> tuple[Tensor, dict[str, float]]:
        """Compute the weighted pairwise-cosine and root-distance MSE losses for one batch, plus
        the reconstruction MSE if output is fragment-stage shaped."""
        loss_pairwise, cos_sim = loss_distance_pairwise(output["embed_a"], output["embed_b"], batch["pairwise_distaince"], self.log_eps)
        loss_root, norm_a, norm_b = loss_distance_root(output["embed_a"], output["embed_b"], batch["root_distance_a"], batch["root_distance_b"])
        loss = self.lambda_pairwise * loss_pairwise + self.lambda_root * loss_root

        metrics = {
            "loss_pairwise": loss_pairwise.item(),
            "loss_root":     loss_root.item(),
            "cos_sim_mean":  cos_sim.mean().item(),
            "norm_mean":     torch.cat([norm_a, norm_b]).mean().item(),
        }

        # Redefine loss if training on denoiser
        if "denoised_a" in output:
            loss_recon = 0.5 * (F.mse_loss(output["denoised_a"], batch["target_a"]) + F.mse_loss(output["denoised_b"], batch["target_b"]))
            loss = loss + self.lambda_recon * loss_recon
            metrics["loss_recon"] = loss_recon.item()

        # Compute error rate (metric)
        pred_pairwise = (1.0 - cos_sim.float()).detach().cpu().numpy()
        true_pairwise = batch["pairwise_distaince"].detach().cpu().numpy()
        pairwise      = pairwise_errors(pred_pairwise, true_pairwise, 1e-6)
        metrics["distortion_pairwise"] = float(pairwise.loc[~pairwise["near_zero"], "distortion"].mean())
        metrics["loss"] = loss.item()

        return loss, metrics

    @property
    def config(self) -> dict[str, object]:
        """Return this loss' lambda weights for logging and checkpoints."""
        return {"lambda_recon": self.lambda_recon, "lambda_pairwise": self.lambda_pairwise, "lambda_root": self.lambda_root, "log_eps": self.log_eps}

# =================================================================
# Loss terms
# =================================================================
def loss_distance_pairwise(embed_a: Tensor, embed_b: Tensor, distance: Tensor, log_eps: float = 0.01) -> tuple[Tensor, Tensor]:
    """MSE in log-space between (1 - cosine similarity) and the normalized distance."""
    cos_sim = F.cosine_similarity(embed_a.float(), embed_b.float(), dim=-1)
    pred, true = 1.0 - cos_sim, distance
    
    # NOTE: Log similarity seems to have better convergence.
    # loss = F.mse_loss(pred, true)
    loss = F.mse_loss(torch.log(pred + log_eps), torch.log(true + log_eps))
    return loss, cos_sim

def loss_distance_root(embed_a: Tensor, embed_b: Tensor, root_a: Tensor, root_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """MSE between each embedding's norm and its genome's normalized root-to-tip distance."""
    norm_a, norm_b = embed_a.norm(dim=-1), embed_b.norm(dim=-1)
    loss = 0.5 * (F.mse_loss(norm_a, root_a) + F.mse_loss(norm_b, root_b))
    return loss, norm_a, norm_b

# =================================================================
# Evaluation hook
# =================================================================
def eval_phylogeny(output: dict[str, Tensor], batch: dict[str, Tensor], epoch: int, logger: logging.Logger) -> dict[str, float]:
    """Log and return cos-sim/distance and norm/root-distance correlation for one validation
    epoch, for both training stages. Adds mse_recon when output is fragment-stage shaped (carries
    denoised_a/b) -- same auto-detection Loss_Manifold.forward() uses."""
    embed_a, embed_b = output['embed_a'].float(), output['embed_b'].float()

    # Step 1: Recover predicted pairwise/root distances in the same normalized units as the true labels
    cos_sim       = F.cosine_similarity(embed_a, embed_b, dim=-1)
    pred_pairwise = (1.0 - cos_sim).detach().cpu().numpy()
    true_pairwise = batch["pairwise_distaince"].detach().cpu().numpy()

    pred_root = torch.cat([embed_a.norm(dim=-1), embed_b.norm(dim=-1)]).detach().cpu().numpy()
    true_root = torch.cat([batch["root_distance_a"], batch["root_distance_b"]]).detach().cpu().numpy()

    # Step 2: MSE and mean distortion only -- for pairwise distance and root distance alike
    pairwise = pairwise_errors(pred_pairwise, true_pairwise, 1e-6)
    root     = pairwise_errors(pred_root, true_root, 1e-6)

    metrics = {
        "mse_pairwise":             float(np.mean(pairwise["abs_error"] ** 2)),
        "distortion_mean_pairwise": float(pairwise.loc[~pairwise["near_zero"], "distortion"].mean()),
        "mse_root":                 float(np.mean(root["abs_error"] ** 2)),
        "distortion_mean_root":     float(root.loc[~root["near_zero"], "distortion"].mean()),
    }

    if "denoised_a" in output:
        metrics["mse_recon"] = float(0.5 * (F.mse_loss(output["denoised_a"].float(), batch["target_a"].float()) + F.mse_loss(output["denoised_b"].float(), batch["target_b"].float())).item())

    return metrics
