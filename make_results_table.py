"""
make_results_table.py — T4.1 consolidated results table.

Reads results_summary.json and emits:
  - results_table.tex : LaTeX tabular (booktabs-friendly), no surrounding float
  - results_table.md  : GitHub-flavored markdown with the same content

Row selection is canonical per REMEDIATION_PLAN.md §T4.1. No hard-coded numbers:
every cell is pulled from the JSON. If an expected record is missing, the script
raises with a name so the failure is loud.

Idempotent: safe to re-run after T2/T3 regenerate results_summary.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT: Path = Path(__file__).resolve().parent
SUMMARY_PATH: Path = PROJECT_ROOT / "results_summary.json"
TEX_PATH: Path = PROJECT_ROOT / "results_table.tex"
MD_PATH: Path = PROJECT_ROOT / "results_table.md"


# ---------------------------------------------------------------------------
# Row specification
# ---------------------------------------------------------------------------
# Each spec has:
#   - display_case   : Case column text (matches NARRATIVE phrasing)
#   - display_method : Method column text
#   - phase / case / method : keys matching results_summary.json record fields
#   - param_match    : callable(dict) -> bool to disambiguate when there are
#                      multiple records for the same (phase, case, method)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RowSpec:
    display_case: str
    display_method: str
    phase: str
    case: str
    method: str
    param_match: Callable[[dict], bool] = lambda p: True  # noqa: E731


ROW_SPECS: list[RowSpec] = [
    # 1. Earth-Mars indirect shooting
    RowSpec(
        display_case="Earth$\\to$Mars (2-body)",
        display_method="Indirect shooting",
        phase="0",
        case="earth_mars_2body",
        method="indirect_shooting",
    ),
    # 2. Earth-Mars global Bezier + IPOPT
    RowSpec(
        display_case="Earth$\\to$Mars (2-body)",
        display_method="Global B\\'ezier + IPOPT",
        phase="0",
        case="earth_mars_2body",
        method="global_bezier_ipopt",
    ),
    # 3. Planar CR3BP indirect shooting
    RowSpec(
        display_case="Planar CR3BP (L1$\\leftrightarrow$L2)",
        display_method="Indirect shooting",
        phase="1",
        case="planar_cr3bp_L1_L2_lyapunov",
        method="indirect_shooting",
    ),
    # 4. Planar CR3BP global Bezier + IPOPT
    RowSpec(
        display_case="Planar CR3BP (L1$\\leftrightarrow$L2)",
        display_method="Global B\\'ezier + IPOPT",
        phase="1",
        case="planar_cr3bp_L1_L2_lyapunov",
        method="global_bezier_ipopt",
    ),
    # 5. Planar CR3BP segmented Bezier SLSQP, N=16 only
    RowSpec(
        display_case="Planar CR3BP (L1$\\leftrightarrow$L2)",
        display_method="Segmented B\\'ezier + SLSQP ($N{=}16$)",
        phase="1",
        case="planar_cr3bp_L1_L2_lyapunov",
        method="segmented_bezier_slsqp",
        param_match=lambda p: p.get("N_segments") == 16,
    ),
    # 6. Artemis II Post-TLI shooting, best_of_15 rollup
    RowSpec(
        display_case="Artemis II Post-TLI",
        display_method="Indirect shooting (best of 15)",
        phase="2",
        case="artemis2_post_tli",
        method="indirect_shooting",
        param_match=lambda p: p.get("seed_strategy") == "best_of_15",
    ),
    # 7. Artemis II Post-TLI IPOPT multi-shooting
    RowSpec(
        display_case="Artemis II Post-TLI",
        display_method="IPOPT multi-shooting",
        phase="2",
        case="artemis2_post_tli",
        method="multi_shooting_ipopt",
    ),
    # 8. Artemis II Full Mission IPOPT multi-shooting
    RowSpec(
        display_case="Artemis II Full Mission",
        display_method="IPOPT multi-shooting",
        phase="2",
        case="artemis2_full_mission",
        method="multi_shooting_ipopt",
    ),
]


# ---------------------------------------------------------------------------
# Loading + lookup
# ---------------------------------------------------------------------------

def _parameters_hash(parameters: dict[str, Any]) -> str:
    """Stable short hash matching common.results_schema.parameters_hash."""
    import hashlib
    blob = json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def load_records(path: Path) -> list[dict]:
    """Load records, dedupe latest-wins by (phase, case, method, parameters_hash).

    `common.results_io.append_to_summary` already dedupes on write, but we
    re-apply the policy here so this script is defensive against hand-edits.
    Sort by timestamp ascending, then later records overwrite earlier ones.
    """
    with open(path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON list")

    # Sort ascending by timestamp so dict insertion order = latest-wins
    records = sorted(records, key=lambda r: r.get("timestamp") or "")
    keyed: dict[tuple, dict] = {}
    for r in records:
        key = (
            r.get("phase"),
            r.get("case"),
            r.get("method"),
            _parameters_hash(r.get("parameters", {})),
        )
        keyed[key] = r  # later timestamps overwrite earlier
    return list(keyed.values())


def find_record(records: list[dict], spec: RowSpec) -> dict:
    """Return the single record matching `spec`, or raise with a diagnostic."""
    candidates = [
        r for r in records
        if r.get("phase") == spec.phase
        and r.get("case") == spec.case
        and r.get("method") == spec.method
        and spec.param_match(r.get("parameters", {}))
    ]
    if not candidates:
        raise LookupError(
            f"No record matches row spec: "
            f"phase={spec.phase!r} case={spec.case!r} method={spec.method!r} "
            f"display=({spec.display_case} | {spec.display_method}). "
            f"Re-run T1-T3 or adjust the row spec."
        )
    if len(candidates) > 1:
        # Should not happen post-dedup; keep latest by timestamp as a final safety net.
        candidates.sort(key=lambda r: r.get("timestamp") or "")
    return candidates[-1]


# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------

def fmt_cost(J: float) -> str:
    """Bare number; scientific when |J|<1e-2 or |J|>1e3 (and nonzero)."""
    if J == 0.0:
        return "0"
    absJ = abs(J)
    if absJ < 1e-2 or absJ > 1e3:
        return f"{J:.3e}"
    return f"{J:.4f}"


def fmt_walltime(t: float) -> str:
    """2 significant figures; switch to scientific above 1000 s."""
    if t is None:
        return "--"
    if t == 0:
        return "0"
    if t >= 1000:
        return f"{t:.1e}"
    # 2 significant figures
    from math import floor, log10
    digits = 2
    magnitude = floor(log10(abs(t)))
    decimals = max(0, digits - 1 - magnitude)
    return f"{t:.{decimals}f}"


def fmt_int(n: Any) -> str:
    if n is None:
        return "--"
    return f"{int(n)}"


def iterations_cell(rec: dict) -> tuple[str, bool]:
    """Return (cell_text, used_nfev_fallback)."""
    iters = rec.get("iterations")
    if iters is not None:
        return fmt_int(iters), False
    nfev = rec.get("nfev")
    if nfev is not None:
        return f"{int(nfev)}$^{{\\dagger}}$", True
    return "--", False


# ---------------------------------------------------------------------------
# Table writers
# ---------------------------------------------------------------------------

LATEX_COLUMNS = r"l l r r r r"
LATEX_HEADER = [
    "Phase / Case",
    "Method",
    "NLP vars",
    "Iterations",
    "Wall time [s]",
    "Cost $J$",
]

MD_HEADER = [
    "Phase / Case",
    "Method",
    "NLP vars",
    "Iterations",
    "Wall time [s]",
    "Cost J",
]


def build_rows(records: list[dict]) -> tuple[list[list[str]], bool]:
    """Return (rows, any_fallback). Each row is a list of pre-formatted cells."""
    rows: list[list[str]] = []
    any_fallback = False
    for spec in ROW_SPECS:
        rec = find_record(records, spec)
        iters_cell, used_fallback = iterations_cell(rec)
        any_fallback = any_fallback or used_fallback
        rows.append([
            spec.display_case,
            spec.display_method,
            fmt_int(rec.get("n_vars")),
            iters_cell,
            fmt_walltime(rec.get("wall_time_s")),
            fmt_cost(rec.get("cost")),
        ])
    return rows, any_fallback


def write_latex(rows: list[list[str]], any_fallback: bool, path: Path) -> None:
    lines: list[str] = []
    lines.append("% Auto-generated by make_results_table.py. Do not edit by hand.")
    lines.append("% Wrap this in a table float (\\begin{table}...\\end{table}) at use-site.")
    lines.append(r"\begin{tabular}{" + LATEX_COLUMNS + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(LATEX_HEADER) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    # Caption + label for \input{} context
    caption_parts = [
        "Consolidated solver results across Phases 0--2. ",
        "Rows are ordered per NARRATIVE.md; cost $J$ is the Bolza min-energy ",
        "objective $\\int|u|^2\\,dt$. ",
    ]
    if any_fallback:
        caption_parts.append(
            r"$^{\dagger}$ Iteration count unavailable for this solver; "
            r"reported value is \texttt{nfev} (function evaluations). "
        )
    lines.append(r"\captionof{table}{" + "".join(caption_parts).rstrip() + "}")
    lines.append(r"\label{tab:results-consolidated}")
    path.write_text("\n".join(lines) + "\n")


def write_markdown(rows: list[list[str]], any_fallback: bool, path: Path) -> None:
    # Convert a few LaTeX-isms into markdown-readable form
    def demath(s: str) -> str:
        return (
            s.replace("$\\to$", "->")
             .replace("$\\leftrightarrow$", "<->")
             .replace("$N{=}16$", "N=16")
             .replace(r"\'e", "e")
             .replace(r"$^{\dagger}$", " (nfev)")
        )

    header = MD_HEADER
    md_rows = [[demath(c) for c in row] for row in rows]

    # Compute column widths
    all_rows = [header] + md_rows
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(header))]

    def fmt_row(r: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"

    lines: list[str] = []
    lines.append("<!-- Auto-generated by make_results_table.py. Do not edit by hand. -->")
    lines.append("")
    lines.append("**Consolidated solver results (Phases 0-2).** Cost J is the Bolza")
    lines.append("min-energy objective integral |u|^2 dt. Rows follow NARRATIVE.md order.")
    lines.append("")
    lines.append(fmt_row(header))
    lines.append(sep)
    for r in md_rows:
        lines.append(fmt_row(r))
    if any_fallback:
        lines.append("")
        lines.append("Note: cells marked `(nfev)` fall back to function-evaluation count")
        lines.append("because the solver (SciPy fsolve) does not expose a distinct iteration counter.")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found. Run T1-T3 instrumentation first."
        )
    records = load_records(SUMMARY_PATH)
    rows, any_fallback = build_rows(records)
    write_latex(rows, any_fallback, TEX_PATH)
    write_markdown(rows, any_fallback, MD_PATH)
    print(f"Wrote {TEX_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {MD_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Rows emitted: {len(rows)}")
    if any_fallback:
        print("Note: at least one row used nfev as iteration fallback.")


if __name__ == "__main__":
    main()
