"""
run_phase1_nsweep_cubic.py — Phase 1 cubic Bézier headline sweep (planar CR3BP).

Per group decision (2026-04-26), cubic Bezier (degree=3, n_collocation=4) with
h-refinement is the chosen method for AAE 568. This script is the Phase 1
counterpart to run_phase0_nsweep_cubic_extended.py.

Setup
-----
- Problem: planar Earth-Moon CR3BP, L1 → L2 Lyapunov transfer (Ax = 0.02).
- Solver: SLSQP, finite-difference Jacobian, fresh shooting warm-start at
  every N (chained warm-start fails for cubic at low N — see Phase 0 findings).
- Sweep: N ∈ {1, 2, 4, 8, 16, 32}, --max-iter and --ftol tunable from CLI so
  the headline run can stay within the 30-min total budget.

Records persist to results_summary.json with method=segmented_bezier_slsqp,
phase=1, parameters.degree=3, parameters.warm_start_strategy=fresh_shooting_each_N.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_PROJECT_ROOT / "Earth-Mars"))

from common import ResultRecord, append_to_summary, git_sha_or_none, timed_solve  # noqa: E402

from cr3bp_planar import MU  # noqa: E402
from cr3bp_transfer import (  # noqa: E402
    setup_transfer_problem,
    solve_shooting,
    cr3bp_jacobi_planar,
)
from bezier_segmented import CR3BPSegmentedBezier  # noqa: E402
from scipy.optimize import minimize  # noqa: E402


def _now_iso_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


# =============================================================================
# Shooting baseline (PMP anchor)
# =============================================================================

def run_shooting(x0, xf, t0, tf, mu=MU) -> dict:
    print("\n--- Indirect shooting (PMP) — Phase 1 baseline ---", flush=True)
    with timed_solve() as t:
        lam0, sol, info = solve_shooting(x0, xf, t0, tf, mu)
    residual = float(np.linalg.norm(info["fvec"]))
    converged = residual < 1e-6

    lam_v = sol.y[6:8].T
    u = -0.5 * lam_v
    cost = float(np.trapezoid(np.sum(u ** 2, axis=1), sol.t))
    nfev = int(info.get("nfev", 0)) if "nfev" in info else None

    rec = ResultRecord(
        phase="1",
        case="cr3bp_l1_l2_lyapunov_planar",
        method="indirect_shooting",
        parameters={
            "lam0_guess": "physics_informed_plus_random",
            "rtol": 1e-12,
            "atol": 1e-12,
            "n_eval_pts": 1000,
            "tf": float(tf),
            "Ax_L1": 0.02,
            "Ax_L2": 0.02,
            "context": "phase1_cubic_nsweep_baseline",
        },
        cost=cost,
        converged=bool(converged),
        residual=residual,
        wall_time_s=float(t.wall_time_s),
        n_vars=4,
        n_constraints=4,
        iterations=None,
        nfev=nfev,
        njev=None,
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        convergence_history=None,
        notes="PMP anchor for Phase 1 cubic N-sweep (J*).",
    )
    rec.validate()
    append_to_summary(rec)

    print(f"  J*       = {cost:.10f}", flush=True)
    print(f"  residual = {residual:.2e}", flush=True)
    print(f"  wall     = {t.wall_time_s:.3f} s", flush=True)

    state = sol.y[:4].T  # (N, 4): [x, y, vx, vy]
    return {
        "cost": cost,
        "residual": residual,
        "wall_s": float(t.wall_time_s),
        "nfev": nfev,
        "t": sol.t,
        "x": sol.y[0], "y": sol.y[1],
        "vx": sol.y[2], "vy": sol.y[3],
        "u": u,
        "u_mag": np.linalg.norm(u, axis=1),
        "lam0": lam0,
        "state": state,
    }


# =============================================================================
# Cubic + SLSQP at one N (Phase 1)
# =============================================================================

def solve_cubic_p1(N, x0, xf, t0, tf, warm_traj, max_iter=400, ftol=1e-8,
                   bezier_degree=3, n_collocation=4, mu=MU,
                   wall_budget_s=None):
    """Solve cubic+SLSQP at N segments. wall_budget_s, if set, aborts SLSQP
    once total wall time exceeds the budget (uses scipy's callback abort
    feature, scipy >= 1.11)."""
    solver = CR3BPSegmentedBezier(
        mu=mu,
        n_segments=N,
        bezier_degree=bezier_degree,
        n_collocation=n_collocation,
    )
    z_guess = solver._warm_start_from_trajectory(
        warm_traj[0], warm_traj[1], x0, xf, t0, tf,
    )
    dt_seg = (tf - t0) / N

    t_start = time.perf_counter()

    # Wall-budget abort: scipy >= 1.11 SLSQP supports callback-based termination.
    # We capture the latest x via the callback's intermediate_result and raise
    # StopIteration when time exceeds budget. minimize returns the result with
    # status indicating the callback abort.
    last_x = [z_guess.copy()]
    last_iter = [0]

    cb = None
    if wall_budget_s is not None:
        def cb(intermediate_result):
            try:
                last_x[0] = intermediate_result.x.copy()
            except AttributeError:
                pass
            last_iter[0] += 1
            if time.perf_counter() - t_start > wall_budget_s:
                raise StopIteration("wall budget exceeded")
            return False

    try:
        result = minimize(
            solver._objective, z_guess,
            args=(x0, xf, dt_seg),
            method="SLSQP",
            constraints={"type": "eq", "fun": solver._defects,
                         "args": (x0, xf, dt_seg)},
            options={"maxiter": max_iter, "ftol": ftol, "disp": False},
            callback=cb,
        )
    except StopIteration:
        # Build a result-like object from last captured x
        from types import SimpleNamespace
        x_last = last_x[0]
        result = SimpleNamespace(
            x=x_last,
            fun=float(solver._objective(x_last, x0, xf, dt_seg)),
            success=False,
            nit=last_iter[0],
            nfev=0,
            njev=0,
            message="aborted: wall budget exceeded",
        )

    elapsed = time.perf_counter() - t_start

    sol = solver._evaluate(result.x, x0, xf, t0, tf)
    sol["cost"] = float(result.fun)
    defects = solver._defects(result.x, x0, xf, dt_seg)
    max_defect = float(np.max(np.abs(defects)))
    success = bool(result.success and max_defect < 1e-4)

    n_vars = solver._total_vars()
    n_constraints = N * n_collocation * solver.state_dim

    # Jacobi drift
    jac = np.array([
        cr3bp_jacobi_planar(
            np.array([sol["r"][k, 0], sol["r"][k, 1],
                      sol["v"][k, 0], sol["v"][k, 1]]),
            mu,
        )
        for k in range(len(sol["t"]))
    ])

    rec = ResultRecord(
        phase="1",
        case="cr3bp_l1_l2_lyapunov_planar",
        method="segmented_bezier_slsqp",
        parameters={
            "N_segments": int(N),
            "degree": int(bezier_degree),
            "n_collocation": int(n_collocation),
            "max_iter": int(max_iter),
            "ftol": float(ftol),
            "solver": "scipy.optimize.minimize/SLSQP",
            "warm_start": True,
            "warm_start_source": "indirect_shooting",
            "warm_start_strategy": "fresh_shooting_each_N",
            "tf": float(tf),
            "Ax_L1": 0.02,
            "Ax_L2": 0.02,
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
        notes=("Phase 1 cubic headline sweep (degree=3, fresh shooting WS each N). "
               "Group decision: cubic + h-refinement is the chosen method."),
    )
    rec.validate()
    append_to_summary(rec)

    return {
        "N": N,
        "success": success,
        "cost": float(result.fun),
        "time_s": float(elapsed),
        "nit": int(getattr(result, "nit", 0)),
        "nfev": int(getattr(result, "nfev", 0)),
        "max_defect": max_defect,
        "msg": str(getattr(result, "message", "")),
        "t": sol["t"],
        "r": sol["r"],
        "v": sol["v"],
        "u": sol["u"],
        "u_mag": np.linalg.norm(sol["u"], axis=1),
        "jacobi": jac,
        "segments": sol.get("segments", []),
        "n_vars": int(n_vars),
        "n_constraints": int(n_constraints),
    }


# =============================================================================
# Driver
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--ftol", type=float, default=1e-8)
    ap.add_argument("--per-N-budget-s", type=float, default=600.0,
                    help="Per-N wall-clock budget; SLSQP aborted via callback if exceeded.")
    args = ap.parse_args()

    print("=" * 74, flush=True)
    print(f"Phase 1 CUBIC headline sweep — N = {args.N}", flush=True)
    print(f"  degree = 3, n_collocation = 4 (group-locked method)", flush=True)
    print(f"  max_iter = {args.max_iter}, ftol = {args.ftol}", flush=True)
    print("=" * 74, flush=True)

    x0, xf, t0, tf, lyap_data = setup_transfer_problem(Ax_L1=0.02, Ax_L2=0.02)
    print(f"\nTransfer:", flush=True)
    print(f"  x0 = {x0}", flush=True)
    print(f"  xf = {xf}", flush=True)
    print(f"  t  = [{t0:.4f}, {tf:.4f}]", flush=True)

    shooting = run_shooting(x0, xf, t0, tf)
    warm = (shooting["t"], shooting["state"])

    sweep = []
    sweep_t0 = time.perf_counter()
    for N in args.N:
        elapsed = time.perf_counter() - sweep_t0
        print(f"\n--- N = {N} cubic, fresh shooting WS  "
              f"[total elapsed: {elapsed:.1f}s]  ---", flush=True)
        e = solve_cubic_p1(N, x0, xf, t0, tf, warm,
                           max_iter=args.max_iter, ftol=args.ftol,
                           wall_budget_s=args.per_N_budget_s)
        flag = "OK" if e["success"] else "FAIL"
        print(
            f"    [{flag}] cost = {e['cost']:.10f}, "
            f"max|def| = {e['max_defect']:.2e}, "
            f"iters = {e['nit']}, nfev = {e['nfev']}, "
            f"time = {e['time_s']:.2f} s",
            flush=True,
        )
        sweep.append(e)

    # Summary
    print("\n" + "=" * 74, flush=True)
    Jstar = shooting["cost"]
    print("    N | converged | iters |  nfev |    time (s) |"
          "             cost |        |J - J*| |  max |defect|")
    print("   ---+-----------+-------+-------+-------------+"
          "------------------+----------------+----------------")
    for r in sweep:
        flag = "Y" if r["success"] else "N"
        gap = abs(r["cost"] - Jstar)
        print(f"   {r['N']:>2d} |     {flag}     | "
              f"{r['nit']:>5d} | {r['nfev']:>5d} | "
              f"{r['time_s']:>11.3f} | {r['cost']:>16.10f} | "
              f"{gap:>14.3e} | {r['max_defect']:>14.2e}")

    out = _HERE / "phase1_nsweep_cubic_summary.txt"
    with open(out, "w") as f:
        f.write(f"Phase 1 cubic headline sweep (degree=3, n_colloc=4)\n")
        f.write(f"PMP shooting J* = {Jstar:.10f}, wall = {shooting['wall_s']:.3f}s\n\n")
        f.write("    N | converged | iters |  nfev |    time (s) |"
                "             cost |        |J - J*| |  max |defect|\n")
        f.write("   ---+-----------+-------+-------+-------------+"
                "------------------+----------------+----------------\n")
        for r in sweep:
            flag = "Y" if r["success"] else "N"
            gap = abs(r["cost"] - Jstar)
            f.write(f"   {r['N']:>2d} |     {flag}     | "
                    f"{r['nit']:>5d} | {r['nfev']:>5d} | "
                    f"{r['time_s']:>11.3f} | {r['cost']:>16.10f} | "
                    f"{gap:>14.3e} | {r['max_defect']:>14.2e}\n")
    print(f"\nSaved: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
