from __future__ import annotations

# Standard Import
import logging
from pathlib import Path
from Bio.SeqRecord import SeqRecord
import numpy as np
import torch

# Custom Import
from src.io import fasta

# =================================================================
# Kmer
# =================================================================
class Kmer:
    """Deterministic canonical k-mer composition encoder, with an optional length feature."""
    def __init__(self, k: int, logger: logging.Logger) -> None:
        self.k = k
        self.logger = logger
        self.alphabet = "ACGT"
        self.complement = np.array([3, 2, 1, 0])  # A <-> T, C <-> G

        # (str to ascii) mapping
        self.str_to_ascii = torch.full((256,), -1, dtype=torch.long)
        for i, ch in enumerate(self.alphabet): self.str_to_ascii[ord(ch)] = i

        # kmer position multiplier (e.g: [64, 16, 4, 1])
        self.weights = torch.tensor([len(self.alphabet) ** (self.k - 1 - j) for j in range(self.k)], dtype=torch.long)

        # (kmer to canonical kmer) mapping
        self.dim, self.kmer_to_ckmer = build_canonical_map(self.k, self.alphabet, self.complement)

    # -----------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------
    def vectorize(
        self, 
        seq:                    str, 
        include_length:         bool = False,
        include_dispersion:     bool = False,
        transform_kmer:         str = None, 
        transform_dispersion:   bool = False, 
    ) -> np.ndarray:
        """
        Vectorize a single sequence into a canonical k-mer composition vector.

        Parameters
        ----------
        seq                   : DNA sequence; any case; non-"ACGT" treated as invalid.
        include_length        : Append log10(sequence length) as a trailing dimension.
        include_dispersion    : Append phase-invariant positional dispersion per k-mer.
        transform_kmer        : Composition transform to apply (frequency/clr); raw counts if None.
        transform_dispersion  : Log-transform dispersion before appending; ignored if include_dispersion is False.
        """
        counts, dispersion, _ = sequence_stats(seq, self.k, self.str_to_ascii, self.weights, self.kmer_to_ckmer, self.dim, include_dispersion)
        vector = apply_transform(counts, transform_kmer)
        if include_dispersion:
            vector = concat_dispersion(vector, dispersion, transform_dispersion)
        if include_length:
            vector = concat_length(vector, len(seq))
        return vector

    def vectorize_fasta(
        self, 
        path_fasta:             Path, 
        include_length:         bool = False,
        include_dispersion:     bool = False,
        transform_kmer:         str | None = None, 
        transform_dispersion:   bool = False, 
        min_length:             int = 1000
    ) -> np.ndarray:
        """
        Vectorize every record in a FASTA file into one pooled k-mer composition vector.

        Composition is pooled by summing raw counts across records before the one
        transform, so a 2 Mb contig naturally outweighs a 2 kb one. Dispersion is
        pooled as a length-weighted average per canonical k-mer across records.

        Parameters
        ----------
        path_fasta            : Path to a FASTA file; transparently reads .gz.
        include_length        : Append log10(sequence length) as a trailing dimension.
        include_dispersion    : Append phase-invariant positional dispersion per k-mer.
        transform_kmer        : Composition transform to apply (frequency/clr); raw counts if None.
        transform_dispersion  : Log-transform dispersion before appending; ignored if include_dispersion is False.
        min_length            : Records shorter than this are dropped before vectorization.
        """
        records = fasta.load(path_fasta, length_filter=min_length)
        if not records:
            self.logger.warning(f"vectorize_fasta: no records >= min_length={min_length} in {path_fasta}")

        counts, dispersion = pool_stats(records, self.k, self.str_to_ascii, self.weights, self.kmer_to_ckmer, self.dim, include_dispersion)
        vector = apply_transform(counts, transform_kmer)
        if include_dispersion:
            vector = concat_dispersion(vector, dispersion, transform_dispersion)
        if include_length:
            vector = concat_length(vector, fasta.get_length(records, per_file=True))
        return vector

# =================================================================
# Kmer math
# =================================================================
def build_canonical_map(k: int, alphabet: str, complement: np.ndarray) -> tuple[int, torch.Tensor]:
    """Map each k-mer index to its canonical (lexicographically smaller strand) partner."""
    # Create a vector (|alphabet|^(k-1), |alphabet|^(k-2), ... |alphabet|^0)
    position_vector = len(alphabet) ** np.arange(k - 1, -1, -1, dtype=np.int64)

    # Allocate kmer vector (0,1,2,3,4 ... etc)
    kmer_index = np.arange(len(alphabet) ** k, dtype=np.int64)

    # Create kmer index map (e.g: AAAA -> [0,0,0,0])
    matrix_fwd = np.stack(
        [(kmer_index // len(alphabet) ** (k - 1 - j)) % len(alphabet) for j in range(k)],
        axis=1,
    )
    # Create reverse complement map
    matrix_rc = complement[matrix_fwd[:, ::-1]]
    kmer_index_rc = (matrix_rc * position_vector).sum(axis=1)

    # Find the minimum between forward and reverse complement index
    index = np.minimum(kmer_index, kmer_index_rc)

    # Create the mapping by index
    _, canonical_map = np.unique(index, return_inverse=True)
    return int(canonical_map.max()) + 1, torch.from_numpy(canonical_map.astype(np.int64))

def tokenize(seq: str, k: int, str_to_ascii: torch.Tensor, weights: torch.Tensor, kmer_to_ckmer: torch.Tensor) -> np.ndarray:
    """DNA string -> ordered canonical k-mer indices, one per valid all-ACGT window."""
    raw = torch.from_numpy(np.frombuffer(seq.upper().encode(), dtype=np.uint8).copy()).long()
    tokenized = str_to_ascii[raw]
    if len(tokenized) < k:
        return np.empty(0, dtype=np.int64)

    invalid = tokenized < 0
    windows = tokenized.unfold(0, k, 1)
    bad     = invalid.unfold(0, k, 1).any(dim=1)
    windows = windows[~bad]

    indices = (windows * weights).sum(dim=1)
    indices = kmer_to_ckmer[indices]
    return indices.numpy()

def kmer_dispersion(kmer_indices: np.ndarray, dim: int) -> np.ndarray:
    """
    Phase-invariant positional dispersion per canonical k-mer.

    Ratio of the variance to the squared mean of the gaps between consecutive
    occurrences of each k-mer: >1 clustered, ~1 uniform, <1 regular, NaN if fewer
    than 3 occurrences.

    Parameters
    ----------
    kmer_indices : Canonical k-mer index of each valid window, in sequence order.
    dim          : Number of distinct canonical k-mers.
    """
    dispersion = np.full(dim, np.nan, dtype=np.float64)
    for kappa in range(dim):
        pos = np.where(kmer_indices == kappa)[0]
        if len(pos) < 3:
            continue
        gaps = np.diff(pos)
        dispersion[kappa] = gaps.var() / (gaps.mean() ** 2)
    return dispersion

def sequence_stats(seq: str, k: int, str_to_ascii: torch.Tensor, weights: torch.Tensor, kmer_to_ckmer: torch.Tensor, dim: int, include_dispersion: bool) -> tuple[np.ndarray, np.ndarray, int]:
    """Tokenize once and return (counts, dispersion, window count) for one sequence; dispersion is None if not requested."""
    indices = tokenize(seq, k, str_to_ascii, weights, kmer_to_ckmer)
    counts = np.bincount(indices, minlength=dim).astype(np.float64)
    dispersion = kmer_dispersion(indices, dim) if include_dispersion else None
    return counts, dispersion, len(indices)

def pool_stats(records: list[SeqRecord], k: int, str_to_ascii: torch.Tensor, weights: torch.Tensor, kmer_to_ckmer: torch.Tensor, dim: int, include_dispersion: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum canonical k-mer counts across FASTA records, and length-weight-average dispersion per k-mer.

    Parameters
    ----------
    records            : FASTA records to pool.
    k                  : K-mer length.
    str_to_ascii       : Alphabet-to-index lookup table.
    weights            : Positional weight vector for k-mer index encoding.
    kmer_to_ckmer      : Raw k-mer index to canonical k-mer index map.
    dim                : Number of distinct canonical k-mers.
    include_dispersion : Also pool per-k-mer positional dispersion.
    """
    total_counts = np.zeros(dim, dtype=np.float64)
    weighted_sum = np.zeros(dim, dtype=np.float64) if include_dispersion else None
    weight_total = np.zeros(dim, dtype=np.float64) if include_dispersion else None

    for record in records:
        counts, dispersion, n_windows = sequence_stats(str(record.seq), k, str_to_ascii, weights, kmer_to_ckmer, dim, include_dispersion)
        total_counts += counts
        if include_dispersion and n_windows > 0:
            valid = ~np.isnan(dispersion)
            weighted_sum[valid] += dispersion[valid] * n_windows
            weight_total[valid] += n_windows

    pooled_dispersion = None
    if include_dispersion:
        pooled_dispersion = np.divide(
            weighted_sum, weight_total,
            out=np.full(dim, np.nan, dtype=np.float64), where=weight_total > 0,
        )
    return total_counts, pooled_dispersion

def apply_transform(counts: np.ndarray, transform: str | None = None) -> np.ndarray:
    """Convert raw k-mer counts into the composition vector named by transform (None=raw, frequency, or clr)."""
    if transform is None:
        return counts.copy()
    freqs = counts / max(counts.sum(), 1.0)
    if transform == "clr":
        log_x = np.log(freqs + 1e-6)
        return log_x - log_x.mean()
    return freqs

def concat_length(vector: np.ndarray, length: int) -> np.ndarray:
    """Append log10(length) to vector as a trailing dimension."""
    log_length = np.log10(max(length, 1))
    return np.concatenate([vector, np.array([log_length], dtype=vector.dtype)])

def concat_dispersion(vector: np.ndarray, dispersion: np.ndarray, transform_dispersion: bool = False) -> np.ndarray:
    """Append per-k-mer dispersion to vector, optionally log-transformed so clustered/regular are symmetric around zero."""
    if transform_dispersion:
        dispersion = np.log(dispersion + 1e-6)
        filled = np.nan_to_num(dispersion, nan=0.0)
    else:
        filled = np.nan_to_num(dispersion, nan=1.0)
    return np.concatenate([vector, filled.astype(vector.dtype)])