"""
timing.py — lightweight timing utilities.

Provides `timed_solve()`, a context manager whose sole purpose is to capture
wall and CPU time around a solver call with no extra overhead inside the
with-block. Also provides `git_sha_or_none()` for provenance.
"""

from __future__ import annotations

import contextlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class SolveTimer:
    """Holds timing results. Fields are filled on __exit__."""
    wall_time_s: float = 0.0
    cpu_time_s: float = 0.0


@contextlib.contextmanager
def timed_solve() -> Iterator[SolveTimer]:
    """
    Context manager that times the wrapped block. Use as:

        with timed_solve() as t:
            result = solver.solve(...)
        # t.wall_time_s, t.cpu_time_s

    Intentionally does zero I/O or setup inside the with-block — the timing
    window is as tight as possible around the actual solve.
    """
    timer = SolveTimer()
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        yield timer
    finally:
        timer.wall_time_s = time.perf_counter() - wall0
        timer.cpu_time_s = time.process_time() - cpu0


def git_sha_or_none(project_root: Path | str | None = None) -> str | None:
    """
    Return the short git SHA (7 chars) for `project_root` or None if
    it is not a git repo / git is unavailable / anything else goes wrong.
    """
    if project_root is None:
        # Lazy import to avoid circular dep with results_io
        from .results_io import PROJECT_ROOT
        project_root = PROJECT_ROOT

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha if sha else None
