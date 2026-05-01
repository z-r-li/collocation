"""
common/ — shared utilities for AAE 568 project instrumentation.

Provides:
  - ResultRecord schema (results_schema)
  - JSON IO + summary appender (results_io)
  - Timing context manager + git SHA helper (timing)
"""

from .results_schema import ResultRecord
from .results_io import (
    save_result,
    append_to_summary,
    load_results,
    load_as_dataframe,
    PROJECT_ROOT,
)
from .timing import timed_solve, git_sha_or_none

__all__ = [
    "ResultRecord",
    "save_result",
    "append_to_summary",
    "load_results",
    "load_as_dataframe",
    "PROJECT_ROOT",
    "timed_solve",
    "git_sha_or_none",
]
