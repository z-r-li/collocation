"""
make_pareto_plot.py — Phase 1 Pareto plot: |J − J*| vs wall time.

Loads `results_summary.json`, filters to phase="1" /
case="planar_cr3bp_L1_L2_lyapunov", and plots three series on log-log axes:

  - PMP anchor (re-run indirect shooting): single marker at (wall, |J-J*|=0)
    rendered at a small floor for log visibility.
  - Global Bézier + IPOPT: single marker at (wall, |J-J*|).
  - Segmented Bézier + SLSQP N-sweep: line joining (wall_N, |J-J*|_N),
    annotated with the N value at each node.

Also emits `pareto_data.csv` (the same rows in tabular form for the report).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common import load_results  # noqa: E402


CASE = "planar_cr3bp_L1_L2_lyapunov"
PHASE = "1"

# J* canonical value from NARRATIVE (what we annotate the plot with). Also
# accept whatever the re-run shooting returned, for an apples-to-apples
# error comparison across solvers that were all run in THIS session.
J_STAR_NARRATIVE = 0.04306


def main() -> int:
    records = load_results(phase=PHASE, case=CASE)
    if not records:
        print(f"No records found for phase={PHASE!r} case={CASE!r}", file=sys.stderr)
        return 1

    # Index by method
    by_method: dict[str, list] = {}
    for rec in records:
        by_method.setdefault(rec.method, []).append(rec)

    # --- PMP anchor ---
    shoot_recs = by_method.get("indirect_shooting", [])
    if not shoot_recs:
        print("Missing indirect_shooting record; aborting.", file=sys.stderr)
        return 1
    # Latest shooting record wins (append_to_summary deduplicates anyway)
    shoot = shoot_recs[-1]
    j_star_runtime = float(shoot.cost)
    shoot_wall = float(shoot.wall_time_s)

    # Treat J* := re-run shooting cost (most apples-to-apples for all other
    # methods that were re-run in the same session).
    j_star = j_star_runtime

    # --- IPOPT ---
    ipopt_recs = by_method.get("global_bezier_ipopt", [])
    if not ipopt_recs:
        print("Missing global_bezier_ipopt record; aborting.", file=sys.stderr)
        return 1
    ipopt = ipopt_recs[-1]

    # --- SLSQP N-sweep ---
    slsqp_recs = by_method.get("segmented_bezier_slsqp", [])
    slsqp_recs_sorted = sorted(
        slsqp_recs, key=lambda r: r.parameters.get("N_segments", 0)
    )

    # =================================================================
    # CSV dump
    # =================================================================
    csv_path = _HERE / "pareto_data.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "series", "label", "N_segments", "J", "abs_J_minus_Jstar",
            "wall_time_s", "iterations", "nfev", "njev", "converged",
        ])
        # PMP anchor row (|J-J*| = 0 exactly)
        w.writerow([
            "PMP_anchor",
            f"indirect_shooting  (J*={j_star:.6f})",
            "-",
            j_star,
            0.0,
            shoot_wall,
            shoot.iterations if shoot.iterations is not None else "-",
            shoot.nfev if shoot.nfev is not None else "-",
            shoot.njev if shoot.njev is not None else "-",
            shoot.converged,
        ])
        w.writerow([
            "IPOPT",
            "global_bezier_ipopt",
            ipopt.parameters.get("N_segments", "-"),
            ipopt.cost,
            abs(ipopt.cost - j_star),
            ipopt.wall_time_s,
            ipopt.iterations if ipopt.iterations is not None else "-",
            ipopt.nfev if ipopt.nfev is not None else "-",
            ipopt.njev if ipopt.njev is not None else "-",
            ipopt.converged,
        ])
        for rec in slsqp_recs_sorted:
            w.writerow([
                "SLSQP",
                f"segmented_bezier_slsqp N={rec.parameters['N_segments']}",
                rec.parameters["N_segments"],
                rec.cost,
                abs(rec.cost - j_star),
                rec.wall_time_s,
                rec.iterations if rec.iterations is not None else "-",
                rec.nfev if rec.nfev is not None else "-",
                rec.njev if rec.njev is not None else "-",
                rec.converged,
            ])
    print(f"Wrote {csv_path}")

    # =================================================================
    # Plot
    # =================================================================
    fig, ax = plt.subplots(figsize=(8.5, 6.2), facecolor='white')

    # A tiny floor so the PMP point is visible on a log axis (|J-J*| = 0
    # itself would be -inf).
    EPS = 1e-16

    # --- PMP anchor ---
    ax.scatter(
        [shoot_wall], [EPS],
        marker="*", s=260, color="#1f77b4", zorder=5,
        edgecolors="black", linewidths=0.8,
        label=f"PMP (shooting)  J*={j_star:.5f}",
    )
    ax.annotate(
        f"PMP (J*)  ", (shoot_wall, EPS),
        xytext=(-5, 8), textcoords="offset points",
        fontsize=9, color="#1f77b4", ha="right",
    )

    # --- IPOPT ---
    ipopt_err = max(abs(float(ipopt.cost) - j_star), EPS)
    ax.scatter(
        [ipopt.wall_time_s], [ipopt_err],
        marker="D", s=110, color="#2ca02c", zorder=4,
        edgecolors="black", linewidths=0.8,
        label="Global Bézier + IPOPT",
    )
    ax.annotate(
        f"IPOPT (N={ipopt.parameters.get('N_segments','?')})",
        (ipopt.wall_time_s, ipopt_err),
        xytext=(8, 2), textcoords="offset points",
        fontsize=9, color="#2ca02c",
    )

    # --- SLSQP sweep ---
    if slsqp_recs_sorted:
        Ns = [r.parameters["N_segments"] for r in slsqp_recs_sorted]
        walls = [r.wall_time_s for r in slsqp_recs_sorted]
        errs = [max(abs(r.cost - j_star), EPS) for r in slsqp_recs_sorted]

        ax.plot(
            walls, errs,
            "o-", color="#d62728", lw=1.8, ms=8, zorder=3,
            label="Segmented Bézier + SLSQP (N-sweep)",
            markeredgecolor="black", markeredgewidth=0.5,
        )
        for N_i, w_i, e_i, rec in zip(Ns, walls, errs, slsqp_recs_sorted):
            marker_ok = rec.converged
            ax.annotate(
                f"N={N_i}", (w_i, e_i),
                xytext=(7, 4), textcoords="offset points",
                fontsize=9, color="#d62728",
                weight="bold" if marker_ok else "normal",
            )

    # Axes styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wall time (s)", fontsize=11)
    ax.set_ylabel(r"$|J - J^*|$  (cost error vs PMP)", fontsize=11)
    ax.set_title("Phase 1 Pareto: accuracy vs wall time", fontsize=13, pad=10)
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)
    ax.minorticks_on()
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92, facecolor='white', edgecolor='black', labelcolor='black')

    fig.tight_layout()
    out_path = _HERE / "pareto_J_error_vs_wall_time.png"
    fig.savefig(out_path, dpi=180, facecolor='white', edgecolor='white')
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Short stdout summary
    print("\nPareto summary")
    print("-" * 60)
    print(f"{'series':<28} {'N':>3} {'J':>12} {'|J-J*|':>12} {'wall(s)':>10}")
    print("-" * 60)
    print(f"{'indirect_shooting':<28} {'-':>3} {j_star:>12.6f} "
          f"{0.0:>12.2e} {shoot_wall:>10.3f}")
    print(f"{'global_bezier_ipopt':<28} "
          f"{ipopt.parameters.get('N_segments','-'):>3} "
          f"{float(ipopt.cost):>12.6f} {abs(ipopt.cost-j_star):>12.2e} "
          f"{ipopt.wall_time_s:>10.3f}")
    for rec in slsqp_recs_sorted:
        print(f"{'segmented_bezier_slsqp':<28} {rec.parameters['N_segments']:>3} "
              f"{float(rec.cost):>12.6f} {abs(rec.cost-j_star):>12.2e} "
              f"{rec.wall_time_s:>10.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
