from __future__ import annotations

# Standard Import
import gzip
import shutil
from pathlib import Path
from typing import TextIO, Literal

def smart_open(input_file: Path, mode: str = "r") -> TextIO:
    """
    Open files with different format (.gz)

    Parameters
    ----------
    input_file      : source file path
    mode            : mode for default open()
    """
    if input_file.suffix == ".gz":
        return gzip.open(input_file, mode + "t")
    return open(input_file, mode)
