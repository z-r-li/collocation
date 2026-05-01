"""
run_phase0_nsweep_cubic.py — h-vs-p study companion to run_phase0_nsweep.py

Same Earth-Mars two-body problem, same SLSQP, but with degree=3 (cubic) and
n_collocation=4 instead of degree=7 / n_collocation=8. The point is to test
whether segmented refinement at low polynomial order can recover what
degree-7 achieves at low N.

Theory: on smooth problems, p-refinement gives exponential convergence and
h-refinement (at fixed p) gives algebraic convergence. Cubic should need
many more segments to match degree-7. Empirically test by running both at
matched total-DOF points.

Records persist to results_summary.json (method=segmented_bezier_slsqp,
parameters.degree=3) so the deg-3 and deg-7 sweeps are deduped by their
parameter hashes and don't overwrite each other.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Reuse everything from run_phase0_nsweep
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import run_phase0_nsweep as p0  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    print("=" * 74)
    print("Phase 0 N-sweep — CUBIC (degree=3, n_collocation=4)")
    print("Tests h-refinement at low polynomial order vs degree-7 baseline")
    print("=" * 74)

    # Re-use the shooting baseline (idempotent — append_to_summary dedupes).
    shooting = p0.run_shooting()

    # Cubic sweep — push to N=32 to give h-refinement a fair chance.
    sweep = p0.run_sweep(
        shooting,
        N_list=(1, 2, 4, 8, 16, 32),
        bezier_degree=3,
        n_collocation=4,
        max_iter=600,
        ftol=1e-9,
    )

    # ---- comparison plot: cubic vs degree-7 ----
    # Read the degree-7 numbers back from results_summary.json so we don't
    # re-run them here.
    import json
    with open(p0._PROJECT_ROOT / "results_summary.json") as f:
        records = json.load(f)

    deg7 = sorted(
        [
            r for r in records
            if r["phase"] == "0"
            and r["method"] == "segmented_bezier_slsqp"
            and r["parameters"].get("degree") == 7
        ],
        key=lambda r: r["parameters"]["N_segments"],
    )
    deg3 = sorted(
        [
            r for r in records
            if r["phase"] == "0"
            and r["method"] == "segmented_bezier_slsqp"
            and r["parameters"].get("degree") == 3
        ],
        key=lambda r: r["parameters"]["N_segments"],
    )

    Jstar = shooting["cost"]
    t_shoot = shooting["wall_s"]

    p7_N = np.array([r["parameters"]["N_segments"] for r in deg7])
    p7_J = np.array([r["cost"] for r in deg7])
    p7_t = np.array([r["wall_time_s"] for r in deg7])
    p7_dof = np.array([r["n_vars"] for r in deg7])

    p3_N = np.array([r["parameters"]["N_segments"] for r in deg3])
    p3_J = np.array([r["cost"] for r in deg3])
    p3_t = np.array([r["wall_time_s"] for r in deg3])
    p3_dof = np.array([r["n_vars"] for r in deg3])

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), facecolor="white")

    # Panel 1: cost gap vs N
    ax = axes[0]
    ax.set_facecolor("white")
    ax.semilogy(p7_N, np.maximum(np.abs(p7_J - Jstar), 1e-16),
                "o-", color="#1f77b4", ms=9, lw=2,
                label="degree 7, n_colloc 8")
    ax.semilogy(p3_N, np.maximum(np.abs(p3_J - Jstar), 1e-16),
                "s--", color="#d62728", ms=9, lw=2,
                label="degree 3, n_colloc 4 (cubic)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(np.union1d(p7_N, p3_N))
    ax.set_xticklabels([str(n) for n in np.union1d(p7_N, p3_N)])
    ax.set_xlabel("N segments", color="black")
    ax.set_ylabel(r"$|J - J^*|$  (gap to PMP)", color="black")
    ax.set_title("Cost gap — h vs p refinement",
                 fontweight="bold", color="black")
    ax.grid(True, alpha=0.3, color="gray", which="both")
    ax.tick_params(colors="black")
    for s in ax.spines.values():
        s.set_color("black")
    ax.legend(loc="best", framealpha=0.9, facecolor="white",
              edgecolor="black", labelcolor="black", fontsize=9)

    # Panel 2: wall vs N
    ax = axes[1]
    ax.set_facecolor("white")
    ax.loglog(p7_N, p7_t, "o-", color="#1f77b4", ms=9, lw=2,
              label="degree 7")
    ax.loglog(p3_N, p3_t, "s--", color="#d62728", ms=9, lw=2,
              label="degree 3 (cubic)")
    ax.axhline(t_shoot, color="#2ca02c", ls=":", lw=1.5,
               label=f"Shooting = {t_shoot:.3f} s")
    ax.set_xscale("log", base=2)
    ax.set_xticks(np.union1d(p7_N, p3_N))
    ax.set_xticklabels([str(n) for n in np.union1d(p7_N, p3_N)])
    ax.set_xlabel("N segments", color="black")
    ax.set_ylabel("wall-clock time (s)", color="black")
    ax.set_title("Solve time — h vs p refinement",
                 fontweight="bold", color="black")
    ax.grid(True, alpha=0.3, color="gray", which="both")
    ax.tick_params(colors="black")
    for s in ax.spines.values():
        s.set_color("black")
    ax.legend(loc="best", framealpha=0.9, facecolor="white",
              edgecolor="black", labelcolor="black", fontsize=9)

    # Panel 3: cost gap vs total DOF (the cleanest h-vs-p comparison)
    ax = axes[2]
    ax.set_facecolor("white")
    ax.loglog(p7_dof, np.maximum(np.abs(p7_J - Jstar), 1e-16),
              "o-", color="#1f77b4", ms=9, lw=2,
              label="degree 7")
    ax.loglog(p3_dof, np.maximum(np.abs(p3_J - Jstar), 1e-16),
              "s--", color="#d62728", ms=9, lw=2,
              label="degree 3 (cubic)")
    ax.set_xlabel("total decision variables  $n_\\mathrm{vars}$", color="black")
    ax.set_ylabel(r"$|J - J^*|$  (gap to PMP)", color="black")
    ax.set_title("Cost gap vs total DOF\n(does p-refinement win per parameter?)",
                 fontweight="bold", color="black")
    ax.grid(True, alpha=0.3, color="gray", which="both")
    ax.tick_params(colors="black")
    for s in ax.spines.values():
        s.set_color("black")
    ax.legend(loc="best", framealpha=0.9, facecolor="white",
              edgecolor="black", labelcolor="black", fontsize=9)

    fig.suptitle(
        "h-refinement vs p-refinement on Earth-Mars 2-body — Bezier+SLSQP",
        fontsize=13, fontweight="bold", color="black", y=1.02,
    )
    plt.tight_layout()
    fname = p0.PLOT_DIR / "phase0_h_vs_p_refinement.png"
    plt.savefig(fname, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fname}")

    # Print a side-by-side table
    print("\n" + "=" * 74)
    print("  CUBIC SWEEP SUMMARY")
    print("=" * 74)
    print(p0.format_sweep_table(shooting, sweep))

    summary_path = _HERE / "phase0_nsweep_cubic_summary.txt"
    summary_path.write_text(
        "Cubic (degree=3) sweep on Earth-Mars 2-body\n"
        + p0.format_sweep_table(shooting, sweep) + "\n"
    )
    print(f"\nSaved: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
