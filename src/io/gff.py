from __future__ import annotations

# Standard Import
from pathlib import Path
from urllib.parse import unquote
import pandas as pd
from Bio.SeqRecord import SeqRecord

# Custom Import
from src.io.file import smart_open

GFF_COLUMNS = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]

# =================================================================
# Basic functionality
# =================================================================
def load(input_gff: Path, feature_types: list[str] | None = None) -> pd.DataFrame:
    """
    Load a GFF3 file into a flat DataFrame, one row per feature. Attribute key=value pairs (gene,
    product, Name, ID, Parent, locus_tag, ...) are expanded into their own columns, using whichever
    keys are actually present in the file (missing ones are NaN).

    Parameters
    ----------
    input_gff       : path to .gff/.gff3 (or .gz).
    feature_types   : (optional) keep only rows whose `type` column is one of these, e.g. ["rRNA", "CDS"].
    """
    with smart_open(input_gff, "rt") as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip() and not line.startswith("#")]
    df = pd.DataFrame(rows, columns=GFF_COLUMNS)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    if feature_types is not None:
        df = df[df["type"].isin(feature_types)]

    attrs = pd.DataFrame([_parse_attributes(a) for a in df["attributes"]], index=df.index)
    return pd.concat([df.drop(columns="attributes"), attrs], axis=1)

def extract_sequence(row: pd.Series, records: dict[str, SeqRecord]) -> str:
    """
    Slice one feature's nucleotide sequence out of its parent contig (reverse-complemented if
    `strand` is "-").

    Parameters
    ----------
    row      : one row from load() -- needs seqid, start, end, strand (both 1-based, inclusive, per GFF3).
    records  : {seqid: SeqRecord}, e.g. fasta.load(path, to_dict=True).
    """
    seq = records[row["seqid"]].seq[row["start"] - 1: row["end"]]
    return str(seq.reverse_complement() if row["strand"] == "-" else seq)

# =================================================================
# Helper function
# =================================================================
def _parse_attributes(field: str) -> dict[str, str]:
    """"key=value;key2=value2" -> {"key": "value", "key2": "value2"} -- GFF3 percent-encodes reserved
    characters (","->"%2C", ";"->"%3B", "="->"%3D", etc.) inside values, so those are decoded too."""
    out = {}
    for kv in field.split(";"):
        if "=" in kv:
            key, value = kv.split("=", 1)
            out[key] = unquote(value)
    return out
