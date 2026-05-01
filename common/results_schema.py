"""
results_schema.py — canonical record format for AAE 568 result artifacts.

Every solver run in the project emits a ResultRecord via `common.results_io.save_result`
or `append_to_summary`. The schema is intentionally flat-ish so it round-trips to JSON
cleanly and loads into a pandas DataFrame without pain.

Design notes
------------
- `parameters` is a free-form dict so different methods (shooting, Bézier, IPOPT) can
  stash their own hyperparameters without schema churn. It is the ONLY place method-specific
  knobs belong. Use it for {"N_segments": 16, "degree": 7, "n_collocation": 8}, etc.
- `cost`, `wall_time_s`, `converged` are required non-None — the validator enforces this.
- `residual` is "max constraint violation" for NLP solvers and "endpoint residual norm"
  for shooting (the two are semantically the same thing: how far from feasible).
- `convergence_history` is optional and holds a list of per-iteration dicts with
  {"iter": int, "obj": float, "constr_viol": float}. Populated for IPOPT when a callback
  is attached; typically None for fsolve/SLSQP unless we instrument further.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResultRecord:
    """Immutable record describing a single solver invocation."""

    # Identity / taxonomy
    phase: str                                        # "0" | "1" | "2"
    case: str                                         # e.g. "earth_mars_2body"
    method: str                                       # e.g. "indirect_shooting"
    parameters: dict[str, Any]                        # method-specific hyperparameters

    # Primary outcome
    cost: float                                       # objective value J
    converged: bool
    residual: float                                   # max constraint viol / endpoint error
    wall_time_s: float

    # Problem / solver size
    n_vars: int
    n_constraints: int

    # Solver diagnostics
    iterations: int | None = None
    nfev: int | None = None
    njev: int | None = None

    # Environment / provenance
    git_sha: str | None = None
    timestamp: str = ""                               # ISO-8601 UTC
    python_version: str = ""

    # Optional per-iteration trace
    convergence_history: list[dict] | None = None

    # Free-form notes
    notes: str = ""

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResultRecord":
        """Construct from a dict (e.g. read back from JSON)."""
        # Only pass through known fields; ignore any future extras silently.
        known = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """
        Assert required fields are non-None and types are plausible.
        Raises AssertionError on failure.
        """
        assert self.phase in ("0", "1", "2"), f"phase must be '0'/'1'/'2', got {self.phase!r}"
        assert isinstance(self.case, str) and self.case, "case must be a non-empty string"
        assert isinstance(self.method, str) and self.method, "method must be a non-empty string"
        assert isinstance(self.parameters, dict), "parameters must be a dict"
        assert self.cost is not None, "cost is required"
        assert isinstance(self.cost, (int, float)), "cost must be numeric"
        assert self.converged is not None, "converged is required"
        assert isinstance(self.converged, bool), "converged must be bool"
        assert self.residual is not None, "residual is required"
        assert isinstance(self.residual, (int, float)), "residual must be numeric"
        assert self.wall_time_s is not None, "wall_time_s is required"
        assert self.wall_time_s >= 0.0, "wall_time_s must be non-negative"
        assert self.n_vars is not None and self.n_vars >= 0, "n_vars required, non-negative"
        assert self.n_constraints is not None and self.n_constraints >= 0, \
            "n_constraints required, non-negative"

    # ------------------------------------------------------------------
    # Dedup helper
    # ------------------------------------------------------------------
    def dedup_key(self) -> tuple[str, str, str, str]:
        """
        Identity tuple used to deduplicate records in results_summary.json.
        Two records with the same key are considered the same run —
        the later one wins (overwrites the earlier).
        """
        return (self.phase, self.case, self.method, parameters_hash(self.parameters))


def parameters_hash(parameters: dict[str, Any]) -> str:
    """
    Stable hash of a parameters dict. Uses sorted-key JSON so logically-equal
    dicts produce the same hash regardless of key insertion order.
    """
    blob = json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]
