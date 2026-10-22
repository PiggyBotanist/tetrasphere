# Standard Import
import re 
import shutil
import logging
import tarfile
from tqdm import tqdm
from pathlib import Path

# =================================================================
# Basic functionality
# =================================================================
def extract_archive(input_tar: Path, output_folder: Path, logger: logging.Logger, show_tqdm: bool = True) -> Path:
    """
    Extract an entire tar archive to a folder.

    Parameters
    ----------
    input_tar           : path to the .tar/.tar.gz archive.
    output_folder       : directory to extract into; created if missing.
    logger              : logger for diagnostics.
    show_tqdm           : whether to show tdqm or not.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {input_tar} -> {output_folder}")
    with tarfile.open(input_tar) as tf:
        members = tf.getmembers()
        with tqdm(total=len(members), desc=f"extract {input_tar.name}", unit="files", disable=not show_tqdm) as bar:
            for member in members:
                tf.extract(member, path=output_folder, filter="data")
                bar.update(1)
    return output_folder

def extract_file(input_tar: Path, output_file: Path, pattern: str, logger: logging.Logger) -> Path:
    """
    Extract the first tar member matching a regex pattern to a destination.

    Parameters
    ----------
    input_tar           : path to the .tar/.tar.gz archive.
    output_file         : file to extract into; created if missing.
    pattern             : regex matched against each member's full path (re.search).
    logger              : logger for diagnostics.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {output_file} from {input_tar} with {pattern} pattern.")
    with tarfile.open(input_tar) as tf:
        member = _find_member_(tf, pattern)
        logger.info("Extracting %s -> %s", member, output_file)
        with tf.extractfile(member) as src, open(output_file, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return output_file

def pack(input_folder: Path, output_tar: Path, logger: logging.Logger) -> Path:
    """
    Pack a folder into a tar archive (gzipped when file_tar ends in .gz).

    Parameters
    ----------
    input_folder        : directory to archive; stored under its own name in the tar.
    output_tar          : output archive path; parent directory is created if missing.
    logger              : logger for diagnostics.
    """
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    mode = "w:gz" if output_tar.suffix == ".gz" else "w"
    with tarfile.open(output_tar, mode) as tf:
        logger.info("Packing %s -> %s", input_folder, output_tar)
        tf.add(input_folder, arcname=input_folder.name)
    return output_tar

# =================================================================
# Helper function
# =================================================================
def _find_member_(input_tar: tarfile.TarFile, pattern: str) -> str:
    """
    Scan and return the first archive member whose path matches 
    pattern (re.search).

    Parameters
    ----------
    input_tar           : tar file.
    pattern             : re.search pattern
    """
    regex = re.compile(pattern)
    for name in input_tar.getnames():
        if regex.search(name):
            return name
    raise KeyError(f"No tar member matching pattern {pattern!r}")