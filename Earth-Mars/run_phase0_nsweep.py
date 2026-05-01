"""
run_phase0_nsweep.py — Earth-Mars N-sweep, segmented Bezier + SLSQP.

Purpose (control sweep for the writeup)
---------------------------------------
The Phase 1 narrative leans on a segmented-Bezier + SLSQP N-sweep on the planar
CR3BP to motivate the IPOPT pivot ("87x slower at N=16"). That story only earns
its keep if the *easy* problem behaves differently. This script is the Phase 0
control: same solver class, same hyperparameters, easier dynamics — does
mesh refinement still buy you anything?

The headline question is whether N=1 (a single high-degree Bezier over the
whole trajectory) already matches the PMP optimum. If yes, that becomes the
Phase 0 punchline: "on near-Keplerian dynamics, segmented Bezier collapses to
global Bezier, and mesh refinement is wasted DOF." Phase 1 then earns the
mesh-refinement cascade as a *response to nonlinearity*, not a methodological
default.

Records flow into results_summary.json via the common harness, one per N.
"""

from __future__ import annotations

import datetime as _dt
import platform
import sys
import time as timer
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# Make common/ importable
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import (  # noqa: E402
    ResultRecord,
    append_to_summary,
    git_sha_or_none,
    timed_solve,
)
from shooting import propagate, solve_min_energy  # noqa: E402
from dynamics import two_body_state_costate_ode  # noqa: E402
from bezier import BezierCollocation  # noqa: E402


# =============================================================================
# Problem definition — matches run_phase0.py / compare_methods.py / pa_redo.mlx
# =============================================================================

MU = 1.0
A_EARTH = 1.0
A_MARS = 1.524
V_MARS_ANGLE = np.pi
VEL_EARTH = np.sqrt(MU / A_EARTH ** 3)
VEL_MARS = np.sqrt(MU / A_MARS ** 3)
T0 = 0.0
TF = 8.0

R0 = np.array([A_EARTH, 0.0])
V0 = np.array([0.0, A_EARTH * VEL_EARTH])
POS_MARS_F = A_MARS * np.array(
    [np.cos(V_MARS_ANGLE + VEL_MARS * TF), np.sin(V_MARS_ANGLE + VEL_MARS * TF)]
)
VEL_MARS_F = VEL_MARS * np.array([-POS_MARS_F[1], POS_MARS_F[0]])
X0_FULL = np.concatenate([R0, V0])
XF_FULL = np.concatenate([POS_MARS_F, VEL_MARS_F])


def gravity_2body(r, mu=MU):
    """Two-body gravity acceleration. Scalar-r safe (BezierCollocation calls
    this elementwise per collocation node)."""
    r = np.asarray(r, dtype=float)
    return -mu * r / np.linalg.norm(r) ** 3


# =============================================================================
# Helpers
# =============================================================================

def _now_iso_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


# =============================================================================
# Shooting reference (PMP anchor + warm-start source)
# =============================================================================

def run_shooting() -> dict:
    """
    Solve the indirect TPBVP, return a dict with the trajectory + cost
    (also persisted as a ResultRecord so the sweep figures can read it back
    from results_summary.json without re-running).
    """
    print("\n--- Indirect shooting (PMP) — baseline ---")
    lam0_guess = np.zeros(4)
    with timed_solve() as t:
        lam0_sol, info = solve_min_energy(
            R0, V0, POS_MARS_F, VEL_MARS_F, T0, TF,
            lam0_guess=lam0_guess, mu=MU,
        )
    residual = float(np.linalg.norm(info["fvec"]))
    converged = residual < 1e-6

    X0_full_aug = np.concatenate([R0, V0, lam0_sol])
    sol = propagate(two_body_state_costate_ode, X0_full_aug, [T0, TF],
                    n_steps=4000, mu=MU)
    t_traj = sol.t
    r_traj = sol.y[0:2, :].T
    v_traj = sol.y[2:4, :].T
    lam_v = sol.y[6:8, :].T
    u_traj = -0.5 * lam_v
    cost_J = float(np.trapezoid(np.sum(u_traj ** 2, axis=1), t_traj))

    rec = ResultRecord(
        phase="0",
        case="earth_mars_2body",
        method="indirect_shooting",
        parameters={
            "lam0_guess": lam0_guess.tolist(),
            "rtol": 1e-12,
            "atol": 1e-12,
            "n_propagation_steps": 4000,
            "tf": TF,
            "context": "phase0_nsweep_baseline",
        },
        cost=cost_J,
        converged=bool(converged),
        residual=residual,
        wall_time_s=float(t.wall_time_s),
        n_vars=4,
        n_constraints=4,
        iterations=None,
        nfev=int(info.get("nfev", 0)) if "nfev" in info else None,
        njev=None,
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        convergence_history=None,
        notes="PMP anchor for Phase 0 N-sweep (baseline cost J*).",
    )
    rec.validate()
    append_to_summary(rec)

    print(f"  J*       = {cost_J:.10f}")
    print(f"  residual = {residual:.2e}")
    print(f"  wall     = {t.wall_time_s:.3f} s")

    return {
        "cost": cost_J,
        "residual": residual,
        "wall_s": float(t.wall_time_s),
        "nfev": int(info.get("nfev", 0)) if "nfev" in info else None,
        "t": t_traj,
        "r": r_traj,
        "v": v_traj,
        "u": u_traj,
        "x": r_traj[:, 0],
        "y": r_traj[:, 1],
        "u_mag": np.linalg.norm(u_traj, axis=1),
    }


# =============================================================================
# Single segmented-Bezier + SLSQP solve
# =============================================================================

def solve_segmented_bezier(
    N: int,
    bezier_degree: int,
    n_collocation: int,
    warm_traj: tuple[np.ndarray, np.ndarray] | None,
    max_iter: int = 400,
    ftol: float = 1e-9,
):
    """
    Run BezierCollocation with N segments. Returns an entry dict shaped like
    the Phase 1 sweep plus a ResultRecord ready to append to results_summary.
    """
    solver = BezierCollocation(
        gravity_func=lambda r: gravity_2body(r, mu=MU),
        pos_dim=2,
        n_segments=N,
        bezier_degree=bezier_degree,
        n_collocation=n_collocation,
    )

    # Build initial guess: warm-start chain when available, else built-in.
    if warm_traj is not None:
        try:
            z_guess = solver._warm_start_from_trajectory(
                warm_traj[0], warm_traj[1], X0_FULL, XF_FULL, T0, TF,
            )
        except Exception as exc:
            print(f"    warm start failed ({exc}); falling back to built-in guess")
            z_guess = solver._initial_guess(X0_FULL, XF_FULL, T0, TF)
    else:
        z_guess = solver._initial_guess(X0_FULL, XF_FULL, T0, TF)

    dt_seg = (TF - T0) / N
    constraints = {
        "type": "eq",
        "fun": solver._defects,
        "args": (X0_FULL, XF_FULL, dt_seg),
    }

    t_start = timer.perf_counter()
    result = minimize(
        solver._objective, z_guess,
        args=(X0_FULL, XF_FULL, dt_seg),
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": ftol, "disp": False},
    )
    elapsed = timer.perf_counter() - t_start

    sol_dict = solver._evaluate(result.x, X0_FULL, XF_FULL, T0, TF)
    sol_dict["cost"] = float(result.fun)

    # Constraint violation
    defects = solver._defects(result.x, X0_FULL, XF_FULL, dt_seg)
    max_defect = float(np.max(np.abs(defects)))
    success = bool(result.success and max_defect < 1e-4)

    n_vars = solver._total_vars()
    # Constraints: defects (N * n_colloc * state_dim) — BCs are folded into
    # the parameterization via _unpack so they are not separate constraints.
    n_constraints = N * n_collocation * solver.state_dim

    entry = {
        "N": N,
        "success": success,
        "scipy_success": bool(result.success),
        "cost": float(result.fun),
        "time_s": float(elapsed),
        "nit": int(getattr(result, "nit", 0)),
        "nfev": int(getattr(result, "nfev", 0)),
        "max_defect": max_defect,
        "msg": str(getattr(result, "message", "")),
        "t": sol_dict["t"],
        "r": sol_dict["r"],
        "v": sol_dict["v"],
        "u": sol_dict["u"],
        "u_mag": np.linalg.norm(sol_dict["u"], axis=1),
        "segments": sol_dict.get("segments", []),
        "n_vars": int(n_vars),
        "n_constraints": int(n_constraints),
    }

    # ResultRecord
    rec = ResultRecord(
        phase="0",
        case="earth_mars_2body",
        method="segmented_bezier_slsqp",
        parameters={
            "N_segments": int(N),
            "degree": int(bezier_degree),
            "n_collocation": int(n_collocation),
            "max_iter": int(max_iter),
            "ftol": float(ftol),
            "solver": "scipy.optimize.minimize/SLSQP",
            "warm_start": warm_traj is not None,
            "warm_start_source": (
                "indirect_shooting" if (warm_traj is not None and N == 1)
                else ("prev_N" if warm_traj is not None else None)
            ),
            "tf": TF,
        },
        cost=float(result.fun),
        converged=success,
        residual=max_defect,
        wall_time_s=float(elapsed),
        iterations=int(getattr(result, "nit", 0)),
        nfev=int(getattr(result, "nfev", 0)),
        njev=int(getattr(result, "njev", 0)) if hasattr(result, "njev") else None,
        n_vars=int(n_vars),
        n_constraints=int(n_constraints),
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        convergence_history=None,
        notes=(
            "Phase 0 N-sweep control: literal segmented Bezier + SLSQP on "
            "Earth-Mars 2-body. Asks whether N=1 already matches the PMP optimum."
        ),
    )
    rec.validate()
    append_to_summary(rec)

    return entry


# =============================================================================
# Sweep driver
# =============================================================================

def run_sweep(
    shooting: dict,
    N_list=(1, 2, 4, 8, 16),
    bezier_degree: int = 7,
    n_collocation: int = 8,
    max_iter: int = 400,
    ftol: float = 1e-9,
):
    """
    Run the N-sweep with warm-start chain. The first N is warm-started from
    the shooting solution; later N's chain from the previous converged run.
    """
    sweep = []
    # First warm-start: shooting trajectory (in the same [r, v] state form)
    warm_traj = (
        shooting["t"],
        np.column_stack([shooting["r"], shooting["v"]]),
    )

    for N in N_list:
        print(f"\n--- N = {N} segments ---")
        print(f"    degree = {bezier_degree}, colloc = {n_collocation}")
        entry = solve_segmented_bezier(
            N=N,
            bezier_degree=bezier_degree,
            n_collocation=n_collocation,
            warm_traj=warm_traj,
            max_iter=max_iter,
            ftol=ftol,
        )

        flag = "OK" if entry["success"] else "FAIL"
        print(
            f"    [{flag}] cost = {entry['cost']:.10f}, "
            f"max|defect| = {entry['max_defect']:.2e}, "
            f"iters = {entry['nit']}, nfev = {entry['nfev']}, "
            f"time = {entry['time_s']:.2f} s"
        )
        if not entry["scipy_success"]:
            print(f"    msg: {entry['msg']}")

        # Chain to next N if usable
        if entry["success"]:
            warm_traj = (
                entry["t"],
                np.column_stack([entry["r"], entry["v"]]),
            )

        sweep.append(entry)

    return sweep


# =============================================================================
# Plots
# =============================================================================

PLOT_DIR = _HERE / "phase0_nsweep_figures"


def _ensure_plot_dir():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_convergence(shooting: dict, sweep: list):
    """Three-panel convergence summary: J vs N, time vs N, |defect| vs N."""
    _ensure_plot_dir()

    Ns = np.array([r["N"] for r in sweep])
    costs = np.array([r["cost"] for r in sweep])
    times = np.array([r["time_s"] for r in sweep])
    defects = np.array([r["max_defect"] for r in sweep])
    conv = np.array([r["success"] for r in sweep])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), facecolor="white")

    # Panel 1: cost vs N (with shooting baseline)
    ax = axes[0]
    ax.set_facecolor("white")
    ax.axhline(shooting["cost"], color="#1f77b4", ls="--", lw=1.5,
               label=f"Shooting (PMP)  J* = {shooting['cost']:.6f}")
    if conv.any():
        ax.plot(Ns[conv], costs[conv], "o-", color="#d62728", ms=9, lw=1.8,
                label="Segmented Bezier (conv.)")
    if (~conv).any():
        ax.plot(Ns[~conv], costs[~conv], "o", mfc="none", mec="#d62728",
                ms=9, mew=1.5, label="Segmented Bezier (not conv.)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("N segments", color="black")
    ax.set_ylabel(r"$J = \int |u|^2\,dt$", color="black")
    ax.set_title("Cost vs. mesh count", fontweight="bold", color="black")
    ax.grid(True, alpha=0.3, color="gray")
    ax.tick_params(colors="black")
    for s in ax.spines.values():
        s.set_color("black")
    ax.legend(loc="best", framealpha=0.9, facecolor="white",
              edgecolor="black", labelcolor="black", fontsize=9)

    # Panel 2: wall vs N
    ax = axes[1]
    ax.set_facecolor("white")
    ax.plot(Ns, times, "o-", color="#2ca02c", ms=9, lw=1.8,
            label="SLSQP wallclock")
    ax.axhline(shooting["wall_s"], color="#1f77b4", ls="--", lw=1.5,
               label=f"Shooting = {shooting['wall_s']:.3f} s")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("N segments", color="black")
    ax.set_ylabel("wall-clock time (s)", color="black")
    ax.set_title("Solve time vs. mesh count (log-log)",
                 fontweight="bold", color="black")
    ax.grid(True, alpha=0.3, color="gray", which="both")
    ax.tick_params(colors="black")
    for s in ax.spines.values():
        s.set_color("black")
    ax.legend(loc="best", framealpha=0.9, facecolor="white",
              edgecolor="black", labelcolor="black", fontsize=9)

    # Panel 3: |defect| vs N
    ax = axes[2]
    ax.set_facecolor("white")
    mask = defects > 0
    if mask.any():
        ax.semilogy(Ns[mask], defects[mask], "o-", color="#d62728",
                    ms=9, lw=1.8)
    ax.axhline(1e-4, color="#f39c12", ls=":", lw=1.5,
               label="convergence threshold 1e-4")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("N segments", color="black")
    ax.set_ylabel(r"$\max\,|\mathrm{defect}|$", color="black")
    ax.set_title("Constraint violation vs. mesh count",
                 fontweight="bold", color="black")
    ax.grid(True, alpha=0.3, color="gray", which="both")
    ax.tick_params(colors="black")
    for s in ax.spines.values():
        s.set_color("black")
    ax.legend(loc="best", framealpha=0.9, facecolor="white",
              edgecolor="black", labelcolor="black", fontsize=9)

    fig.suptitle(
        "Phase 0 control sweep — segmented Bezier + SLSQP on Earth-Mars 2-body",
        fontsize=13, fontweight="bold", color="black", y=1.02,
    )
    plt.tight_layout()
    fname = PLOT_DIR / "phase0_nsweep_convergence.png"
    plt.savefig(fname, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
    return fname


def plot_trajectories(shooting: dict, sweep: list):
    """Grid figure: trajectory for each N overlaid on shooting."""
    _ensure_plot_dir()
    n = len(sweep)
    n_cols = 2 if n <= 4 else 3
    n_rows = int(np.ceil(n / n_cols))

    fig = plt.figure(figsize=(5.5 * n_cols, 4.6 * n_rows), facecolor="white")
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.32, wspace=0.25,
                          left=0.06, right=0.97, top=0.92, bottom=0.06)

    fig.suptitle(
        "Phase 0 — Earth-Mars segmented Bezier N-sweep (overlaid on shooting)",
        fontsize=14, fontweight="bold", color="black", y=0.97,
    )

    earth_t = np.linspace(0, 2 * np.pi, 300)
    mars_t = np.linspace(0, 2 * np.pi / VEL_MARS, 300)

    for idx, r in enumerate(sweep):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        ax.set_facecolor("white")

        ax.plot(A_EARTH * np.cos(earth_t), A_EARTH * np.sin(earth_t),
                ":", color="#3498db", alpha=0.4, lw=0.8)
        ax.plot(A_MARS * np.cos(V_MARS_ANGLE + VEL_MARS * mars_t),
                A_MARS * np.sin(V_MARS_ANGLE + VEL_MARS * mars_t),
                ":", color="#d62728", alpha=0.4, lw=0.8)

        ax.plot(shooting["x"], shooting["y"], "-", color="#1f77b4", lw=2,
                alpha=0.85, label="Shooting (PMP)")
        if r["r"] is not None:
            ax.plot(r["r"][:, 0], r["r"][:, 1], "--", color="#d62728",
                    lw=2, label=f"Seg-Bezier N={r['N']}")
            # Segment boundaries
            if r.get("segments") and r["N"] > 1:
                eps = np.array([seg[0][:2] for seg in r["segments"]]
                               + [r["segments"][-1][-1][:2]])
                ax.plot(eps[:, 0], eps[:, 1], "o", color="#d62728",
                        ms=5, mec="black", mew=0.8, zorder=9,
                        label="Segment boundaries")

        ax.plot(0, 0, "o", color="#f39c12", ms=10, mec="#e67e22", mew=1)
        ax.plot(R0[0], R0[1], "o", color="#3498db", ms=7,
                mec="black", mew=0.8, zorder=10)
        ax.plot(POS_MARS_F[0], POS_MARS_F[1], "o", color="#d62728", ms=7,
                mec="black", mew=0.8, zorder=10)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, color="gray")
        flag = "converged" if r["success"] else "not conv."
        ax.set_title(
            f"N = {r['N']} — {flag}   "
            f"(J = {r['cost']:.4f},  max|def| = {r['max_defect']:.1e})",
            fontsize=10, fontweight="bold", color="black",
        )
        ax.set_xlabel("x (AU)", color="black", fontsize=9)
        ax.set_ylabel("y (AU)", color="black", fontsize=9)
        ax.tick_params(colors="black")
        for s in ax.spines.values():
            s.set_color("black")
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9,
                  facecolor="white", edgecolor="black", labelcolor="black")

    fname = PLOT_DIR / "phase0_nsweep_trajectories.png"
    plt.savefig(fname, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
    return fname


def plot_phase0_vs_phase1(shooting_phase0: dict, sweep: list):
    """
    Side-by-side: Phase 0 N-sweep vs Phase 1 N-sweep on the *same* axes
    (normalised by their respective shooting baselines). Reads Phase 1
    numbers from this memory:
        N |  time (s) |       cost J
        1 |     0.10  | 7.18e-01
        2 |     3.00  | 1.28e-01
        4 |    14.51  | 9.01e-02
        8 |    47.15  | 5.00e-02
       16 |   467.64  | 4.324e-02
       (J* = 0.04306, shoot = 5.33 s)
    """
    _ensure_plot_dir()

    # Phase 1 reference table (from memory project_aae568_nsweep)
    p1_N = np.array([1, 2, 4, 8, 16])
    p1_J = np.array([7.18e-01, 1.28e-01, 9.01e-02, 5.00e-02, 4.324e-02])
    p1_t = np.array([0.10, 3.00, 14.51, 47.15, 467.64])
    p1_Jstar = 0.04306
    p1_t_shoot = 5.33

    # Phase 0 from this run
    p0_N = np.array([r["N"] for r in sweep])
    p0_J = np.array([r["cost"] for r in sweep])
    p0_t = np.array([r["time_s"] for r in sweep])
    p0_Jstar = shooting_phase0["cost"]
    p0_t_shoot = shooting_phase0["wall_s"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor="white")

    # Left: J - J* (relative cost gap, log)
    ax = axes[0]
    ax.set_facecolor("white")
    p0_gap = np.maximum(np.abs(p0_J - p0_Jstar), 1e-16)
    p1_gap = np.maximum(np.abs(p1_J - p1_Jstar), 1e-16)
    ax.semilogy(p0_N, p0_gap, "o-", color="#1f77b4", ms=9, lw=2,
                label="Phase 0 (Earth-Mars 2-body)")
    ax.semilogy(p1_N, p1_gap, "s--", color="#d62728", ms=9, lw=2,
                label="Phase 1 (CR3BP L1↔L2 Lyapunov)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(np.union1d(p0_N, p1_N))
    ax.set_xticklabels([str(n) for n in np.union1d(p0_N, p1_N)])
    ax.set_xlabel("N segments", color="black")
    ax.set_ylabel(r"$|J - J^*|$  (gap to PMP optimum)", color="black")
    ax.set_title("Cost gap to PMP — Phase 0 vs Phase 1",
                 fontweight="bold", color="black")
    ax.grid(True, alpha=0.3, color="gray", which="both")
    ax.tick_params(colors="black")
    for s in ax.spines.values():
        s.set_color("black")
    ax.legend(loc="best", framealpha=0.9, facecolor="white",
              edgecolor="black", labelcolor="black", fontsize=9)

    # Right: solve time, log-log
    ax = axes[1]
    ax.set_facecolor("white")
    ax.loglog(p0_N, p0_t, "o-", color="#1f77b4", ms=9, lw=2,
              label="Phase 0 SLSQP")
    ax.loglog(p1_N, p1_t, "s--", color="#d62728", ms=9, lw=2,
             label="Phase 1 SLSQP")
    ax.axhline(p0_t_shoot, color="#1f77b4", ls=":", lw=1.5,
               label=f"P0 shooting = {p0_t_shoot:.3f}s")
    ax.axhline(p1_t_shoot, color="#d62728", ls=":", lw=1.5,
               label=f"P1 shooting = {p1_t_shoot:.2f}s")
    ax.set_xscale("log", base=2)
    ax.set_xticks(np.union1d(p0_N, p1_N))
    ax.set_xticklabels([str(n) for n in np.union1d(p0_N, p1_N)])
    ax.set_xlabel("N segments", color="black")
    ax.set_ylabel("wall-clock time (s)", color="black")
    ax.set_title("Solve time — Phase 0 vs Phase 1",
                 fontweight="bold", color="black")
    ax.grid(True, alpha=0.3, color="gray", which="both")
    ax.tick_params(colors="black")
    for s in ax.spines.values():
        s.set_color("black")
    ax.legend(loc="best", framealpha=0.9, facecolor="white",
              edgecolor="black", labelcolor="black", fontsize=9)

    fig.suptitle(
        "Mesh refinement earns its keep only in the nonlinear regime",
        fontsize=13, fontweight="bold", color="black", y=1.02,
    )
    plt.tight_layout()
    fname = PLOT_DIR / "phase0_vs_phase1_nsweep.png"
    plt.savefig(fname, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")
    return fname


# =============================================================================
# Summary table
# =============================================================================

def format_sweep_table(shooting: dict, sweep: list) -> str:
    lines = []
    lines.append("Phase 0 N-sweep summary")
    lines.append(f"  PMP shooting: J* = {shooting['cost']:.10f}, "
                 f"wall = {shooting['wall_s']:.3f} s")
    lines.append("")
    lines.append("    N | converged | iters |  nfev |    time (s) |"
                 "             cost |        |J - J*| |  max |defect|")
    lines.append("   ---+-----------+-------+-------+-------------+"
                 "------------------+----------------+----------------")
    for r in sweep:
        flag = "Y" if r["success"] else "N"
        gap = abs(r["cost"] - shooting["cost"])
        lines.append(
            f"   {r['N']:>2d} |     {flag}     | "
            f"{r['nit']:>5d} | {r['nfev']:>5d} | "
            f"{r['time_s']:>11.3f} | {r['cost']:>16.10f} | "
            f"{gap:>14.3e} | {r['max_defect']:>14.2e}"
        )
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    print("=" * 74)
    print("Phase 0 N-sweep — segmented Bezier + SLSQP on Earth-Mars 2-body")
    print("Headline question: does N=1 already match the PMP optimum?")
    print("=" * 74)
    print(f"  t  ∈ [{T0}, {TF}]")
    print(f"  x0 = {X0_FULL}")
    print(f"  xf = {XF_FULL}")

    shooting = run_shooting()

    print("\n" + "=" * 74)
    print("  N-sweep (degree=7, n_collocation=8, warm-start chain)")
    print("=" * 74)
    sweep = run_sweep(shooting)

    # Plots
    print("\n--- Generating figures ---")
    plot_convergence(shooting, sweep)
    plot_trajectories(shooting, sweep)
    plot_phase0_vs_phase1(shooting, sweep)

    # Summary table
    table = format_sweep_table(shooting, sweep)
    print("\n" + "=" * 74)
    print("  N-SWEEP SUMMARY")
    print("=" * 74)
    print(table)

    summary_path = _HERE / "phase0_nsweep_summary.txt"
    summary_path.write_text(table + "\n")
    print(f"\nSaved: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
