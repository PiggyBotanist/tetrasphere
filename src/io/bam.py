from __future__ import annotations

# Standard Import
import subprocess
from collections.abc import Iterator
from pathlib import Path

def verify(input_bam: Path) -> bool:
    """
    Verify a BAM is usable (both .bam and its .bam.bai index exist).

    Parameters
    ----------
    input_bam           : file path to .bam (check .bam.bai also exist)
    """
    return input_bam.exists() and Path(f"{input_bam}.bai").exists()

def stream_depth(input_bam: Path, threads: int = 4, conda_env: str | None = None) -> Iterator[tuple[str, int, int]]:
    """
    Stream (contig_id, position, depth) triples from `samtools depth -a` over file_bam.

    Parameters
    ----------
    input_bam           : .bam used by samtools
    """
    cmd = ["samtools", "depth", "-a", "-@", str(threads), str(input_bam)]
    if conda_env is not None:
        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env] + cmd

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        contig_id, pos, depth = line.rstrip("\n").split("\t")
        yield contig_id, int(pos), int(depth)
    proc.stdout.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"samtools depth failed (exit {proc.returncode}): {input_bam}")