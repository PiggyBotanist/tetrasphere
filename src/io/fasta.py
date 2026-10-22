from __future__ import annotations

# Standard Import
import numpy as np
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections.abc import Iterator

# Custom Import
from src.io.file import smart_open

# =================================================================
# Basic functionality
# =================================================================
def load(input_fasta: Path, length_filter: int | None = None, to_dict: bool = False) -> dict[str, SeqRecord] | list[SeqRecord]:
    """
    Load all sequences from a FASTA file.

    Parameters
    ----------
    input_fasta     : path to .fasta/.fa/.fna (or .gz).
    length_filter   : (optional) drop records shorter than this many bp.
    to_dict         : return {record.id: record} instead of a list.
    """
    with smart_open(input_fasta) as f:
        records = list(SeqIO.parse(f, "fasta"))
    if length_filter is not None:
        records = [r for r in records if len(r.seq) >= length_filter]
    if to_dict:
        records = {r.id: r for r in records}
    return records

def save(output_fasta: Path, records: list[SeqRecord]) -> None:
    """
    Saves all sequences from records.

    Parameters
    ----------
    output_fasta    : destination path to .fasta/.fa/.fna (or .gz).
    records         : list of records to store (in order).
    """
    with smart_open(output_fasta, "w") as f:
        SeqIO.write(records, f, "fasta")

def stream(input_fasta: Path) -> Iterator[SeqRecord]:
    """
    Streams (for loop) sequences from source.

    Parameters
    ----------
    input_fasta     : path to .fasta/.fa/.fna (or .gz).
    """
    with smart_open(input_fasta) as f:
        yield from SeqIO.parse(f, "fasta")

def fragment(records: list[SeqRecord], length: int, rng: np.random.Generator, jitter: float = 0.2, max_fragments: int | None = None) -> Iterator[tuple[int, str]]:
    """
    Chop every record into consecutive, non-overlapping fragments (drawn independently and uniformly)

    Parameters
    ----------
    records        : FASTA records to chop (one genome's contigs).
    length         : target fragment length, in bp.
    rng            : random generator, for each fragment's length draw and (if capped) which tiles to keep.
    jitter         : fractional bound on each fragment's length around `length` (0.2 = +/-20%).
    max_fragments  : randomly keep fragments (exhaustive non-overlap otherwise).
    """
    lo, hi = int(length * (1 - jitter)), int(length * (1 + jitter))
    offsets: list[tuple[int, int, int]] = []
    for record_idx, record in enumerate(records):
        seq_len = len(record.seq)
        pos = 0
        while True:
            frag_length = int(rng.integers(lo, hi + 1))
            if pos + frag_length > seq_len:
                break
            offsets.append((record_idx, pos, frag_length))
            pos += frag_length

    if max_fragments is not None and len(offsets) > max_fragments:
        keep = np.sort(rng.choice(len(offsets), size=max_fragments, replace=False))
        offsets = [offsets[i] for i in keep]

    seq_cache: dict[int, str] = {}
    for index, (record_idx, pos, frag_length) in enumerate(offsets):
        if record_idx not in seq_cache:
            seq_cache[record_idx] = str(records[record_idx].seq)
        yield index, seq_cache[record_idx][pos:pos + frag_length]


# =================================================================
# Statistics
# =================================================================
def get_length(records: list[SeqRecord], per_file: bool = False) -> int | dict[str, int]:
    """
    Streams (for loop) sequences from source.

    Parameters
    ----------
    records        : list of records.
    per_file       : per-file(1)/per-contig(0).
    """
    result = {}
    total = 0
    for r in records:
        length = len(str(r.seq))
        result[str(r.id)] = length
        total += length

    if per_file:   
        return total
    return result

def get_n50(records: list[SeqRecord]) -> int:
    """
    Returns N50 score from records.

    Parameters
    ----------
    records        : list of records.
    """
    lengths = sorted((len(str(r.seq)) for r in records), reverse=True)
    half = sum(lengths) / 2
    cumulative = 0
    for length in lengths:
        cumulative += length
        if cumulative >= half:
            return length
    return 0

def get_l50(records: list[SeqRecord]) -> int:
    """
    Returns L50 score from records.

    Parameters
    ----------
    records        : list of records.
    """
    lengths = sorted((len(str(r.seq)) for r in records), reverse=True)
    half = sum(lengths) / 2
    cumulative = 0
    for i, length in enumerate(lengths, start=1):
        cumulative += length
        if cumulative >= half:
            return i
    return 0
