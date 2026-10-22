from __future__ import annotations

# Standard Import
from pathlib import Path
import logging
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from ftplib import FTP

def download_http(url: str, output_file: Path, logger: logging.Logger, show_tqdm: bool = True) -> None:
    """
    Download url (http/https) object to local machine.
    
    Parameters
    ----------
    url         : HTTP URL to fetch.
    output_file : destination file path (include filename).
    logger      : logger to report status.
    show_tqdm   : whether to show tdqm or not.
    """
    logger.info(f"downloading: {url} -> {output_file}.")
    
    # Ensure folder exist
    (output_file.parent).mkdir(parents=True, exist_ok=True)
    
    # Resume download if file exist
    header: dict[str, str] = {}
    bytes_downloaded = output_file.stat().st_size if output_file.exists() else 0
    mode = "wb" if bytes_downloaded == 0 else "ab"
    header["Range"] = f"bytes={bytes_downloaded}-"
    
    with requests.get(url, headers=header, stream=True, timeout=60) as response:
        # Ensure server accepted the range request
        if response.status_code == 416:
            logger.info("already fully downloaded, skipped.")
            return
        elif response.status_code == 200:
            bytes_downloaded = 0
            mode = "wb"
        response.raise_for_status()
        
        # Save
        bytes_total = response.headers.get("Content-Length")
        bytes_total = int(bytes_total) + bytes_downloaded if bytes_total else None
        with open(output_file, mode) as f, tqdm(
            total=bytes_total, 
            initial=bytes_downloaded, 
            unit="B", 
            unit_scale=True,
            desc=f"download {output_file.name}",
            disable=not show_tqdm
        ) as bar:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))            
    logger.info(f"download complete.")


def download_tar(url: str, output_tar: Path, logger: logging.Logger) -> Path:
    """Download url to file_tar unless already present and non-empty; return the local path."""
    if output_tar.exists() and output_tar.stat().st_size == 0:
        logger.warning("%s is empty (previous download likely interrupted); re-downloading", output_tar.name)
        output_tar.unlink()
    if not output_tar.exists():
        download_http(url, output_tar, logger, log_runtime=True)
    return output_tar