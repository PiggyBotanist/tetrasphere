from __future__ import annotations

# Standard Import
import shlex
import logging
from pathlib import Path

# Custom Import
from src.system.shell import execute

def to_bam(input_fasta: Path, input_fastq: Path | list[Path], output_folder: Path, logger: logging.Logger, threads: int = 8, conda_env: str | None = None) -> None:
    """
    Convert fastq to bam with minimap + samtools

    Parameters
    ----------
    input_fasta         : reference FASTA to align against.
    input_fastq         : reads to align; a list of two paths is treated as a paired R1/R2 pair.
    output_folder       : output BAM path; parent directory is created if missing.
    logger              : logger for subprocess output.
    threads             : threads passed to minimap2 and samtools.
    conda_env           : conda environment name providing minimap2 and samtools (None = current env)
    """
    # Prepare folder and input
    output_folder.parent.mkdir(parents=True, exist_ok=True)
    fastq_files = [input_fastq] if isinstance(input_fastq, Path) else input_fastq

    # shlex.quote guards paths with spaces/specials
    fasta = shlex.quote(str(input_fasta))
    fastq = " ".join(shlex.quote(str(f)) for f in fastq_files)
    bam   = shlex.quote(str(output_folder))

    pipeline = (
        f"minimap2 -ax sr -t {threads} {fasta} {fastq} "
        f"| samtools sort -@ {threads} -o {bam} "
        f"&& samtools index {bam}"
    )

    # Run the whole pipeline inside the conda env via `bash -c`.
    # set -o pipefail so a minimap2 failure isn't masked by samtools succeeding.
    command = [
        "bash", "-c", f"set -o pipefail; {pipeline}",
    ]

    execute(command, logger, conda_env=conda_env)
    logger.info("BAM ready at %s", output_folder)
