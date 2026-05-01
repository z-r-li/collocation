"""
make_jacobi_plot.py — T4.2 Jacobi-constant conservation plot for Phase 1.

Re-solves the three Phase 1 methods for the L1->L2 Lyapunov transfer:
  1. Indirect shooting (PMP-anchor trajectory)
  2. Global Bézier + IPOPT (warm-started from shooting)
  3. Segmented Bézier + SLSQP at N = 16 (warm-started through the N-sweep)

For each converged trajectory, evaluates the planar-CR3BP Jacobi constant

    r1 = sqrt((x + mu)^2 + y^2)
    r2 = sqrt((x - 1 + mu)^2 + y^2)
    Omega = 0.5*(x^2 + y^2) + (1 - mu)/r1 + mu/r2 + 0.5*mu*(1 - mu)
    C = 2*Omega - (vx^2 + vy^2)

and plots dC(t) = C(t) - C(0) for all three methods on the same axes.

Note: under thrust C is *not* a true constant — it drifts as the optimal
control u*(t) does work on the spacecraft. For the planar L1<->L2 Lyapunov
transfer that thrust-induced drift is O(1e-2), which dwarfs both the shooting
integrator tolerance (~1e-12) and the collocation defect residuals (~3e-11)
by ten orders of magnitude.

What this plot therefore shows is *cross-validation*: all three methods
converge to the same controlled trajectory, and so compute the same C(t)
along it. Exposing method-level dynamics error (e.g. collocation drift
between nodes) would require a residual-vs-reference check — forward-
propagating each method's control history with a high-tol integrator and
differencing against that baseline — not absolute conservation relative to
C(0). That finer diagnostic is out of scope for the course project.

Run:  python make_jacobi_plot.py
Output: Planer/jacobi_constant.png (150 dpi, tight_layout)

Idempotent — re-running overwrites the PNG cleanly.

Author: Zhuorui Li (AAE 568, Spring 2026)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for p in (_HERE, _PROJECT_ROOT, _PROJECT_ROOT / "Earth-Mars"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cr3bp_planar import MU  # noqa: E402
from cr3bp_transfer import (  # noqa: E402
    setup_transfer_problem,
    solve_shooting,
)
from ipopt_collocation import CR3BPBezierIPOPT  # noqa: E402
from bezier_segmented import run_n_sweep  # noqa: E402


# =============================================================================
# Jacobi constant — match the spec exactly (with the constant 0.5*mu*(1-mu))
# =============================================================================

def jacobi_constant(x, y, vx, vy, mu=MU):
    """Planar CR3BP Jacobi constant, synodic frame, nondimensional."""
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2)
    Omega = (0.5 * (x ** 2 + y ** 2)
             + (1.0 - mu) / r1
             + mu / r2
             + 0.5 * mu * (1.0 - mu))
    return 2.0 * Omega - (vx ** 2 + vy ** 2)


def jacobi_series(t, x, y, vx, vy, mu=MU):
    """Vector evaluation returning C(t) and dC(t) = C(t) - C(0)."""
    C = jacobi_constant(np.asarray(x), np.asarray(y),
                        np.asarray(vx), np.asarray(vy), mu=mu)
    return C, C - C[0]


# =============================================================================
# Method runners
# =============================================================================

def run_shooting_for_jacobi(x0, xf, t0, tf, mu=MU):
    """Re-solve the PMP TPBVP and return the dense controlled trajectory."""
    print("\n[1/3] Indirect shooting (PMP)...")
    lam0_sol, sol, info = solve_shooting(x0, xf, t0, tf, mu)
    residual = float(np.linalg.norm(info["fvec"]))
    print(f"       residual = {residual:.2e}")
    return {
        "label": "Shooting (PMP)",
        "t": sol.t,
        "x": sol.y[0],
        "y": sol.y[1],
        "vx": sol.y[2],
        "vy": sol.y[3],
    }, (sol.t, np.column_stack([sol.y[0], sol.y[1], sol.y[2], sol.y[3]]))


def run_ipopt_global_for_jacobi(x0, xf, t0, tf, warm_traj, mu=MU,
                                 n_eval=4000):
    """Re-solve the global-Bézier NLP and evaluate on a dense grid."""
    print("\n[2/3] Global Bézier + IPOPT (warm-started from shooting)...")
    # Match run_phase1.py's IPOPT level (N=16, deg=7, nc=12). This is the
    # single-rung solve flagged as "Global Bézier + IPOPT" in the NARRATIVE.
    solver = CR3BPBezierIPOPT(
        mu=mu, n_segments=16, bezier_degree=7, n_collocation=12,
    )
    sol = solver.solve(
        x0, xf, t0, tf,
        warm_traj=warm_traj,
        max_iter=3000, tol=1e-10, print_level=0,
    )

    # Re-evaluate on a dense grid so we can see inter-node drift clearly.
    # The solver already produced a sampled trajectory at default n_eval=500;
    # the internal `_evaluate_trajectory` accepts an n_eval arg.
    segments = sol.get("segments")
    if segments is not None:
        # Rebuild U array from solve output: u(t) on the dense grid is what
        # _evaluate_trajectory returns. Call it with a larger n_eval.
        # _evaluate_trajectory takes (segments, U, x0, xf, t0, tf, n_eval=500)
        # but U only shapes the control — for the Jacobi plot we need only
        # r, v. Pass None-shaped U via a pass-through.
        dense = solver._evaluate_trajectory(
            segments, None, x0, xf, t0, tf, n_eval=n_eval,
        )
        t_dense = dense["t"]
        r_dense = dense["r"]
        v_dense = dense["v"]
    else:
        # Fallback: use what solve() returned directly.
        t_dense = sol["t"]
        r_dense = sol["r"]
        v_dense = sol["v"]

    print(f"       cost     = {sol['cost']:.8f}")
    print(f"       converged= {sol.get('success')}")
    print(f"       n samples= {len(t_dense)}")
    return {
        "label": "Global Bezier + IPOPT",
        "t": np.asarray(t_dense),
        "x": np.asarray(r_dense[:, 0]),
        "y": np.asarray(r_dense[:, 1]),
        "vx": np.asarray(v_dense[:, 0]),
        "vy": np.asarray(v_dense[:, 1]),
    }


def run_segmented_N16_for_jacobi(x0, xf, t0, tf, warm_traj, mu=MU):
    """
    Run the SLSQP segmented-Bézier sweep through N = 16 with warm-start
    chain (same chain run_phase1.py uses), and return the N=16 trajectory.
    """
    print("\n[3/3] Segmented Bezier + SLSQP (N-sweep -> N = 16)...")
    sweep = run_n_sweep(
        x0, xf, t0, tf,
        N_list=(1, 2, 4, 8, 16),
        bezier_degree=7,
        n_collocation=8,
        mu=mu,
        warm_traj=warm_traj,
        max_iter=300,
        ftol=1e-9,
        verbose=True,
    )
    n16 = next((e for e in sweep if e["N"] == 16), None)
    if n16 is None or n16.get("t") is None:
        raise RuntimeError("SLSQP sweep did not produce an N=16 trajectory")
    if not n16["success"]:
        print(f"       WARNING: N=16 SLSQP did not converge ({n16.get('msg')})")
    print(f"       cost     = {n16['cost']:.8f}")
    print(f"       max def  = {n16['max_defect']:.2e}")
    print(f"       n samples= {len(n16['t'])}")

    # For a denser inter-node sample of C(t), the default n_eval=500 (~31 per
    # segment at N=16) is already adequate. If we want more density we can
    # recompute by calling the evaluator with a larger n_eval, but 500 points
    # is enough to see sawtooth / ratcheting without making the plot busy.
    return {
        "label": "Segmented Bezier + SLSQP (N=16)",
        "t": np.asarray(n16["t"]),
        "x": np.asarray(n16["r"][:, 0]),
        "y": np.asarray(n16["r"][:, 1]),
        "vx": np.asarray(n16["v"][:, 0]),
        "vy": np.asarray(n16["v"][:, 1]),
    }


# =============================================================================
# Plot
# =============================================================================

def make_plot(series_list, out_path, mu=MU):
    """
    Plot dC(t) = C(t) - C(0) for each method on a single set of axes.

    Chooses linear vs symlog Y-scale based on dynamic range.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='white')

    # Compute series first to decide on axis scaling.
    computed = []
    for s in series_list:
        C, dC = jacobi_series(s["t"], s["x"], s["y"], s["vx"], s["vy"], mu=mu)
        computed.append({
            "label": s["label"],
            "t": s["t"],
            "C": C,
            "dC": dC,
        })

    # Decide scale: if any |dC| exceeds 1e-4 but the smallest max is below 1e-8,
    # dynamic range spans ~4 decades -> use symlog. Otherwise linear.
    dC_maxs = np.array([float(np.max(np.abs(c["dC"]))) for c in computed])
    # Use symlog if ratio between largest and smallest max|dC| exceeds 1e3
    # (so three decades); linear otherwise.
    use_symlog = False
    if dC_maxs.min() > 0:
        ratio = dC_maxs.max() / dC_maxs.min()
        if ratio > 1e3:
            use_symlog = True

    colors = {
        "Shooting (PMP)": "#1f77b4",
        "Global Bezier + IPOPT": "#d62728",
        "Segmented Bezier + SLSQP (N=16)": "#2ca02c",
    }
    styles = {
        "Shooting (PMP)": "-",
        "Global Bezier + IPOPT": "--",
        "Segmented Bezier + SLSQP (N=16)": "-.",
    }

    for c in computed:
        label = c["label"]
        final_dC = float(c["dC"][-1])
        legend_label = f"{label} (final $\\Delta$C = {final_dC:+.2e})"
        ax.plot(c["t"], c["dC"],
                styles.get(label, "-"),
                color=colors.get(label, None),
                lw=1.7, label=legend_label)

    if use_symlog:
        # Linear threshold set to an order of magnitude below the smallest max.
        linthresh = max(1e-14, 0.1 * dC_maxs.min())
        ax.set_yscale("symlog", linthresh=linthresh)
    else:
        # Symmetric linear Y range around zero, padded 20%.
        ymax = 1.2 * dC_maxs.max()
        ax.set_ylim(-ymax, ymax)

    ax.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.5)
    ax.set_xlabel("Nondimensional time $t$ (synodic)")
    ax.set_ylabel(r"$\Delta C(t) = C(t) - C(0)$")
    ax.set_title("Jacobi constant along Phase 1 transfer — three-method cross-validation")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9, framealpha=0.9, facecolor='white', edgecolor='black', labelcolor='black')

    # Caption-style annotation inside the axes: makes explicit that the trace
    # is thrust-induced physics, not method error, so the figure is not
    # misread when pulled into the report.
    ax.text(
        0.02, 0.03,
        (r"$\Delta C(t)$ is dominated by work done by $u^*(t)$ along the "
         r"transfer ($\mathcal{O}(10^{-2})$);"
         "\n"
         r"method-level discretization error ($\lesssim 10^{-10}$) is not "
         r"resolvable at this scale."),
        transform=ax.transAxes,
        fontsize=8, color="#444444",
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor='white', edgecolor='white')
    plt.close(fig)

    return computed


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 72)
    print("T4.2 Jacobi-constant conservation check (Phase 1)")
    print("=" * 72)

    # Problem setup — identical to run_phase1.py / cr3bp_transfer_segmented.py
    x0, xf, t0, tf, _lyap = setup_transfer_problem(Ax_L1=0.02, Ax_L2=0.02)
    print(f"\n  mu (reused from cr3bp_planar.MU) = {MU}")
    print(f"  x0 = {x0}")
    print(f"  xf = {xf}")
    print(f"  t  in [{t0:.4f}, {tf:.6f}]  (tf = pi)")

    # 1. Shooting
    shoot_series, shoot_warm = run_shooting_for_jacobi(x0, xf, t0, tf)

    # 2. Global IPOPT (warm-start from shooting)
    ipopt_series = run_ipopt_global_for_jacobi(
        x0, xf, t0, tf, warm_traj=shoot_warm,
    )

    # 3. Segmented SLSQP N=16 (via N-sweep warm-start chain from shooting)
    seg_series = run_segmented_N16_for_jacobi(
        x0, xf, t0, tf, warm_traj=shoot_warm,
    )

    series_list = [shoot_series, ipopt_series, seg_series]

    out_path = _HERE / "jacobi_constant.png"
    computed = make_plot(series_list, str(out_path), mu=MU)

    print("\n" + "=" * 72)
    print("Final dC(tf) values:")
    for c in computed:
        print(f"  {c['label']:<38s}  dC = {float(c['dC'][-1]):+.3e}")
    print(f"\nSaved: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
