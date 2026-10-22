from __future__ import annotations

# Standard Import
import sys
import logging
from pathlib import Path
from datetime import datetime

def create_logger(label: str, to_console: bool = True, output_log: Path | None = None) -> logging.Logger:
    """
    Create a named logger with optional console and file handlers.

    Parameters
    ----------
    label           : Logger name embedded in every log line.
    to_console      : (optional) Emit records to stdout.
    output_log      : (optional) Writes records to a timestamped file.
    """
    logger = logging.getLogger(label)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")

    if logger.hasHandlers():
        logger.handlers.clear()

    if to_console:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    if output_log is not None:
        output_log.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        handler = logging.FileHandler(output_log / f"{timestamp}_{label}.log", encoding="utf-8")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.propagate = False
    return logger
