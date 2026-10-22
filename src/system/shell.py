from __future__ import annotations

# Standard Import
import os
import time
import psutil
import logging
import threading
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# =================================================================
# EXECUTE SHELL COMMAND
# =================================================================
def execute(
    cmd: list[str],
    logger: logging.Logger,
    log_cmd: bool = True,
    log_process_id: bool = True,
    log_runtime: bool = False,
    log_memory: bool = False,
    env: dict[str, str] | None = None,
    conda_env: str | None = None,
) -> subprocess.CompletedProcess:
    """
    Run a shell command, streaming stdout/stderr to the logger as it runs.

    Parameters
    ----------
    cmd             : argv list, e.g. ["ls", "-la", "/tmp"].
    logger          : logger receiving each output line as it arrives.
    env             : environment variable overrides; None inherits the parent env.
    log_cmd         : log the full command string before execution.
    log_process_id  : log the process id for reference
    log_runtime     : log wall-clock execution time after the command completes.
    log_memory      : log RSS memory delta of this process around the call.
    """
    # Whether to log the command ran or not
    if log_cmd:
        logger.info("$ %s", " ".join(cmd))

    # Whether to append conda environment
    if conda_env is not None:
        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env] + cmd

    # Keep track of process id, runtime, memory
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss if log_memory else 0
    t_start = time.perf_counter() if log_runtime else 0.0

    # Log the process Id
    if log_process_id:
        logger.info(f"Process ID: {proc.pid}")

    # Running process
    with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as p:
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        readers = [
            threading.Thread(target=_log_stream_, args=(p.stdout, logger, stdout_lines)),
            threading.Thread(target=_log_stream_, args=(p.stderr, logger, stderr_lines)),
        ]
        for t in readers:
            t.start()
        for t in readers:
            t.join()

        result = subprocess.CompletedProcess(cmd, p.wait(), "\n".join(stdout_lines), "\n".join(stderr_lines))

    # Report total runtime
    if log_runtime:
        elapsed = time.perf_counter() - t_start
        logger.info("Total Runtime: %.3f s", elapsed)

    # Report memory usage
    if log_memory:
        mem_after = proc.memory_info().rss
        logger.info("Total Memory Usage: %+.1f MB (RSS %.1f MB total)",
            (mem_after - mem_before) / 1024 ** 2,
            mem_after / 1024 ** 2
        )

    # Check run status and error code
    if result.returncode != 0:
        logger.error("Command failed (exit %d): %s", result.returncode, " ".join(cmd))
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\n"
            f"{result.stderr.strip()}"
        )
    return result

# =================================================================
# Multithread
# =================================================================
def multithread(fn: Callable, args: list[tuple], names: list[str], workers: int, logger: logging.Logger) -> None:
    """Run fn(*a) for each a in args, sequentially if workers==1 else across a thread pool."""
    if workers == 1:
        for a, name in zip(args, names):
            try:
                fn(*a)
            except Exception:
                logger.exception("[%s] failed", name)
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *a): name for a, name in zip(args, names)}
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception:
                logger.exception("[%s] failed", name)

# =================================================================
# INTERNAL
# =================================================================
def _log_stream_(stream, logger: logging.Logger, lines: list[str]) -> None:
    """Log and collect every line from stream until it hits EOF."""
    for line in iter(stream.readline, ""):
        logger.info(line.rstrip())
        lines.append(line.rstrip())
