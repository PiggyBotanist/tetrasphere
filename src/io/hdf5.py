from __future__ import annotations
from typing import Any, cast

# Standard Import
import json
import h5py
import numpy as np
from pathlib import Path

# =================================================================
# Load .hdf5
# =================================================================
def load(file_path: Path) -> dict[str, Any]:
    """Loads .hdf5 with any information as dict."""
    with h5py.File(file_path, "r") as f:
        return _recursive_load_(f)

def _recursive_load_(h5_obj: h5py.File | h5py.Group) -> dict[str, Any]:
    """Loads .hdf5 internal file structure."""
    result = {}
    for key, item in h5_obj.items():
        # Datasets: Convert to NumPy array (string dtypes decode to str, not bytes)
        if isinstance(item, h5py.Dataset):
            result[key] = item.asstr()[()] if h5py.check_string_dtype(item.dtype) else item[()]
        
        # Groups: Recurse into the "folder"
        elif isinstance(item, h5py.Group):
            result[key] = _recursive_load_(item)
        
        # Datatypes: Store the definition (dtype) as a string or object
        elif isinstance(item, h5py.Datatype):
            result[key] = f"NamedDatatype({item.dtype})"  
    return result

# =================================================================
# Create/write .hdf5
# =================================================================
def open(output_file: Path, flush_every: int = 512, chunk_size: int = 1) -> tuple[h5py.File, dict]:
    """Open file_output for buffered incremental writing; returns a writer state dict."""
    return create_file(output_file), writer_state(flush_every, chunk_size)

def create_file(output_file: Path) -> h5py.File:
    """Open a fresh .hdf5 for writing, without an attached writer state (see writer_state())."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(output_file, "w")

def writer_state(flush_every: int = 512, chunk_size: int = 1) -> dict:
    """
    A fresh buffered-writer state dict for add_row()/flush()."""
    return {
        "bufs":        {},
        "n":           0,
        "buffered":    0,
        "flush_every": flush_every,
        "chunk_size": chunk_size,
    }
    
def add_row(file: h5py.File, parameters: dict, **row: Any) -> None:
    """Append one row. Keyword names become the dataset names."""
    bufs = parameters["bufs"]
 
    # First row: create one dataset per column from its value.
    if not bufs and parameters["n"] == 0:
        for name, value in row.items():
            _initialize_dataset_(file, parameters, name, value)
    
    # If keys does not match first input, raise error.
    if row.keys() != bufs.keys():
        raise KeyError(f"add_row expects columns {sorted(bufs)}; got {sorted(row)}")
    
    # Flush results
    for name, value in row.items():
        bufs[name].append(value)
    parameters["buffered"] += 1
    if parameters["buffered"] >= parameters["flush_every"]:
        _flush_(file, parameters)

def add_col(file: h5py.File, name: str, values: list | np.ndarray) -> None:
    """Write a whole fixed (non-streaming) dataset at once."""
    values = np.asarray(values)
    if values.dtype.kind in ("U", "S", "O"):
        file.create_dataset(name, data=values.astype(object), dtype=h5py.string_dtype("utf-8"))
    else:
        file.create_dataset(name, data=values)
 
def set_attr(file: h5py.File, key: str, value: Any) -> None:
    """Set a file-level attribute; non-primitives are JSON-encoded."""
    primitive = isinstance(value, (bool, int, float, str))
    file.attrs[key] = value if primitive else json.dumps(value)

def get_attr(input_hdf5: Path, key: str) -> Any:
    """Read a single file-level attribute without loading the rest of the file."""
    with h5py.File(input_hdf5, "r") as f:
        return f.attrs[key]

def close(file: h5py.File, parameters: dict) -> None:
    """Flush remaining buffered rows and close the file."""
    _flush_(file, parameters)
    file.close()

def flush(file: h5py.File, parameters: dict) -> None:
    """Flush any buffered rows without closing the file (for writer_state() instances sharing one file)."""
    _flush_(file, parameters)

# =================================================================
# Helper function
# =================================================================
def _initialize_dataset_(file: h5py.File, parameters: dict, name: str, value: Any) -> None:
    """Create a resizable per-row dataset inferred from the first value."""
    arr = np.asarray(value)
    # If it is a string, create a single list
    if arr.dtype.kind in ("U", "S", "O"):
        dtype = h5py.string_dtype("utf-8")
        row_shape = ()
    # If it is a number create a numpy array of that size
    else:
        dtype = arr.dtype
        row_shape = arr.shape
    chunk_rows = parameters["chunk_size"]
    file.create_dataset(
        name,
        shape=(0, *row_shape),
        maxshape=(None, *row_shape),
        dtype=dtype,
        compression="gzip",
        compression_opts=4,
        chunks=(chunk_rows, *row_shape),
    )
    parameters["bufs"][name] = []

def _flush_(file: h5py.File, parameters: dict) -> None:
    bufs = parameters["bufs"]
    if not bufs: return
    m = min(len(b) for b in bufs.values())
    if m == 0: return
    new_n = parameters["n"] + m
    for name, buf in bufs.items():
        dset = cast(h5py.Dataset, file[name])
        dset.resize(new_n, axis=0)
        dset[parameters["n"]:new_n] = np.asarray(buf[:m])
        del buf[:m]
    parameters["n"] = new_n
    parameters["buffered"] = max(0, parameters["buffered"] - m)

# =================================================================
# Loading data
# =================================================================
# def decode_strings(values: np.ndarray) -> np.ndarray:
#     """Decode a bytes-dtype string array (as loaded from HDF5) into a plain str object array."""
#     return np.array([v.decode() if isinstance(v, bytes) else v for v in values], dtype=object)
# 
# def read_attrs(file_path: Path) -> dict[str, object]:
#     """Read file-level attributes, JSON-decoding any that were JSON-encoded on write."""
#     with h5py.File(file_path, "r") as f:
#         attrs: dict[str, object] = {}
#         for k, v in f.attrs.items():
#             if isinstance(v, str):
#                 try:
#                     v = json.loads(v)
#                 except json.JSONDecodeError:
#                     pass
#             attrs[k] = v
#         return attrs
# 

