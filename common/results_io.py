"""
results_io.py — JSON read/write utilities for ResultRecord artifacts.

One authoritative `results_summary.json` lives at the project root. Every solver
call appends to it via `append_to_summary`. Writes are atomic (tmp + rename) so
concurrent subagents cannot half-corrupt the file if they race.

Dedup policy: records are keyed by (phase, case, method, parameters_hash).
If a new record shares a key with an existing one, the new one replaces the old —
"latest wins." This matches how re-runs of an instrumentation script are expected
to overwrite their own past entries rather than accumulate duplicates.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .results_schema import ResultRecord, parameters_hash


# Project root — resolve relative to this file so the module works from any cwd.
# common/ lives at <project_root>/common/, so parents[1] is the project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------
# Atomic JSON write
# ----------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Any) -> None:
    """Write `data` to `path` atomically via tmp file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # tempfile in same directory so os.replace is atomic on POSIX
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=False, default=str)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        # Clean up tmp on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ----------------------------------------------------------------------
# Save a single record to its own JSON file
# ----------------------------------------------------------------------

def save_result(record: ResultRecord, path: Path | str) -> None:
    """Atomically write a single ResultRecord to `path` as pretty-printed JSON."""
    record.validate()
    _atomic_write_json(Path(path), record.to_dict())


# ----------------------------------------------------------------------
# Append to authoritative summary
# ----------------------------------------------------------------------

def append_to_summary(
    record: ResultRecord,
    summary_path: Path | str | None = None,
) -> None:
    """
    Load existing summary (or empty list if missing), dedupe by
    (phase, case, method, parameters_hash), append, atomic write.
    """
    record.validate()

    summary_path = Path(summary_path) if summary_path else (PROJECT_ROOT / "results_summary.json")

    existing: list[dict] = []
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                raise ValueError(
                    f"{summary_path} must contain a JSON list, got {type(existing).__name__}"
                )
        except json.JSONDecodeError as e:
            raise ValueError(f"{summary_path} is not valid JSON: {e}") from e

    new_key = record.dedup_key()
    deduped = [
        r for r in existing
        if (r.get("phase"), r.get("case"), r.get("method"),
            parameters_hash(r.get("parameters", {}))) != new_key
    ]
    deduped.append(record.to_dict())

    _atomic_write_json(summary_path, deduped)


# ----------------------------------------------------------------------
# Load + filter
# ----------------------------------------------------------------------

def load_results(
    summary_path: Path | str | None = None,
    **filter_kwargs: Any,
) -> list[ResultRecord]:
    """
    Load all records from `summary_path`, filtering by field-equality on kwargs.

    Example:
        load_results(phase="1", method="segmented_bezier_slsqp")
    """
    summary_path = Path(summary_path) if summary_path else (PROJECT_ROOT / "results_summary.json")
    if not summary_path.exists():
        return []

    with open(summary_path) as f:
        data = json.load(f)

    records = [ResultRecord.from_dict(d) for d in data]

    if not filter_kwargs:
        return records

    def match(rec: ResultRecord) -> bool:
        for k, v in filter_kwargs.items():
            if not hasattr(rec, k):
                return False
            if getattr(rec, k) != v:
                return False
        return True

    return [r for r in records if match(r)]


# ----------------------------------------------------------------------
# DataFrame convenience
# ----------------------------------------------------------------------

def load_as_dataframe(summary_path: Path | str | None = None):
    """
    Load results as a pandas DataFrame (convenience for downstream plotting).
    Imports pandas lazily so the rest of this module has no hard pandas dep.
    """
    import pandas as pd

    summary_path = Path(summary_path) if summary_path else (PROJECT_ROOT / "results_summary.json")
    if not summary_path.exists():
        return pd.DataFrame()

    with open(summary_path) as f:
        data = json.load(f)

    return pd.DataFrame(data)
