from __future__ import annotations

# Standard Import
import re 
import shutil
import logging
import tarfile
from tqdm import tqdm
from pathlib import Path

def flatten(input_folder: Path, pattern: str, logger: logging.Logger) -> None:
    """
    Flatten nested folder structure.

    Parameters
    ----------
    folder          : directory to flatten in place.
    pattern         : glob pattern (e.g. "*.fna.gz").
    logger          : logger for diagnostics.
    """
    # Recursively pull out files
    files = [p for p in input_folder.rglob(pattern) if p.parent != input_folder]
    logger.info(f"Flattening {len(files)} files into {input_folder}...")
    
    # Move
    for path in tqdm(files, desc=f"flatten {input_folder.name}", unit="files"):
        destination = input_folder / path.name
        if destination.exists():
            logger.warning(f"Skipping {path}: {destination} already exists")
            continue
        path.rename(destination)

def delete_subfolders(input_folder: Path, logger: logging.Logger, contain_files: bool = False) -> None:
    """
    Remove sub-folders (linux equivalent).

    Parameters
    ----------
    folder          : directory to flatten in place.
    pattern         : glob pattern (e.g. "*.fna.gz").
    logger          : logger for diagnostics.
    contain_files   : remove even when contain files (1) / only empty subfolder (0)
    """
    # Subfolder cleanup
    subdirs = sorted((p for p in input_folder.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True)
    removed = 0
    for subdir in subdirs:
        if contain_files:
            shutil.rmtree(subdir)
            removed += 1
        elif not any(subdir.iterdir()):
            subdir.rmdir()
            removed += 1
    logger.info(f"Removed {removed} subfolders")

def create_symlink(input_folder: Path, output_folder: Path, logger: logging.Logger, filenames: list[str] | None = None) -> None:
    """
    Create symlink mirror for a folder to another by filname.

    Parameters
    ----------
    input_folder    : original folder with actual files.
    output_folder   : folder to create symlink to.
    logger          : logger for diagnostics.
    filenames       : list of file to create symlink (default to all).
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    filenames = [f.name for f in input_folder.glob("*")] if filenames is None else filenames
    for filename in filenames:
        source = input_folder / filename
        target = output_folder / filename

        if not source.exists():
            logger.warning(f"Missing file: {source}")
            continue

        if target.exists() or target.is_symlink():
            logger.warning(f"File already exists, skipping: {target}")
            continue

        target.symlink_to(source.resolve())

    logger.info(f"Created symlinks for {len(filenames)} files -> {output_folder}")