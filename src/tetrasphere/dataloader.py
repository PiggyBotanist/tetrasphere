from __future__ import annotations

# Standard Import
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# =================================================================
# PAIR DATASET
# =================================================================
class DataLoader_Manifold(Dataset):
    """Load and train the manifold"""
    def __init__(self, split: dict[str, np.ndarray]) -> None:
        self.feature   = torch.from_numpy(split["feature"]).float()
        self.root      = torch.from_numpy(split["root_distance"]).float()
        self.pairs     = split["pair"]
        self.pair_dist = torch.from_numpy(split["pair_dist"]).float()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        a, b = self.pairs[index]
        return {
            "feature_a":            self.feature[a],
            "feature_b":            self.feature[b],
            "pairwise_distaince":   self.pair_dist[index],
            "root_distance_a":      self.root[a],
            "root_distance_b":      self.root[b],
        }

# =================================================================
# PAIR DATASET (fragment, with denoiser reconstruction targets)
# =================================================================
def parent_id(fragment_leaf_id: str) -> str:
    """'GCA_000012145.1__f5000_00000' -> 'GCA_000012145.1'"""
    return fragment_leaf_id.split("__f")[0]

class DataLoader_Denoiser(Dataset):
    """Load and train the denoiser"""
    def __init__(self, split: dict[str, np.ndarray], parent_lookup: dict[str, np.ndarray]) -> None:
        self.feature   = torch.from_numpy(split["feature"]).float()
        self.root      = torch.from_numpy(split["root_distance"]).float()
        self.pairs     = split["pair"]
        self.pair_dist = torch.from_numpy(split["pair_dist"]).float()

        # Find mapping for fragment to genome id (drop if not found)
        missing = {parent_id(lid) for lid in split["leaf_id"]} - parent_lookup.keys()
        if missing:
            raise KeyError(f"{len(missing)} fragment parent genome(s) missing from whole-genome lookup, e.g. {sorted(missing)[:3]}")
        target = np.stack([parent_lookup[parent_id(lid)] for lid in split["leaf_id"]])
        self.target = torch.from_numpy(target).float()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        a, b = self.pairs[index]
        return {
            "feature_a":          self.feature[a],
            "feature_b":          self.feature[b],
            "target_a":           self.target[a],
            "target_b":           self.target[b],
            "pairwise_distaince": self.pair_dist[index],
            "root_distance_a":    self.root[a],
            "root_distance_b":    self.root[b],
        }
