"""
run_phase1_psweep.py — Polynomial-degree p-sweep on Phase 1 (planar CR3BP).

Companion to run_phase0_psweep.py: same study, harder dynamics. For each
(degree, n_colloc=degree+1, N) it runs SLSQP with fresh shooting warm-start
and a wall-budget callback to avoid the infeasibility-chasing failure mode
that cubic exhibits at high iteration count.

Records: phase=1, method=segmented_bezier_slsqp, parameters.degree ∈ {4,5,6}.
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

from common import ResultRecord, append_to_summary, git_sha_or_none  # noqa: E402
from cr3bp_planar import MU, cr3bp_jacobi_planar  # noqa: E402
from cr3bp_transfer import setup_transfer_problem, solve_shooting  # noqa: E402
from bezier_segmented import CR3BPSegmentedBezier  # noqa: E402
from scipy.optimize import minimize  # noqa: E402


def _now_iso_utc():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _python_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def run_shooting(x0, xf, t0, tf):
    print("\n--- Phase 1 PMP shooting ---", flush=True)
    t_start = time.perf_counter()
    lam0, sol, info = solve_shooting(x0, xf, t0, tf, MU)
    elapsed = time.perf_counter() - t_start
    lam_v = sol.y[6:8].T
    u = -0.5 * lam_v
    cost = float(np.trapezoid(np.sum(u ** 2, axis=1), sol.t))
    print(f"  J* = {cost:.10f}, wall = {elapsed:.2f}s", flush=True)
    return {"cost": cost, "t": sol.t, "state": sol.y[:4].T,
            "wall_s": float(elapsed)}


def solve_p_p1(N, degree, n_colloc, x0, xf, t0, tf, warm_traj,
               max_iter, ftol, wall_budget_s):
    solver = CR3BPSegmentedBezier(
        mu=MU, n_segments=N, bezier_degree=degree, n_collocation=n_colloc,
    )
    z_guess = solver._warm_start_from_trajectory(
        warm_traj[0], warm_traj[1], x0, xf, t0, tf,
    )
    dt_seg = (tf - t0) / N
    t_start = time.perf_counter()

    last_x = [z_guess.copy()]
    last_iter = [0]

    def cb(intermediate_result):
        try:
            last_x[0] = intermediate_result.x.copy()
        except AttributeError:
            pass
        last_iter[0] += 1
        if time.perf_counter() - t_start > wall_budget_s:
            raise StopIteration

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
        from types import SimpleNamespace
        x_last = last_x[0]
        result = SimpleNamespace(
            x=x_last,
            fun=float(solver._objective(x_last, x0, xf, dt_seg)),
            success=False,
            nit=last_iter[0], nfev=0, njev=0,
            message="aborted: wall budget exceeded",
        )

    elapsed = time.perf_counter() - t_start
    sol = solver._evaluate(result.x, x0, xf, t0, tf)
    sol["cost"] = float(result.fun)
    defs = solver._defects(result.x, x0, xf, dt_seg)
    max_defect = float(np.max(np.abs(defs)))
    success = bool(result.success and max_defect < 1e-4)

    n_vars = solver._total_vars()
    n_constraints = N * n_colloc * solver.state_dim

    rec = ResultRecord(
        phase="1", case="cr3bp_l1_l2_lyapunov_planar",
        method="segmented_bezier_slsqp",
        parameters={
            "N_segments": int(N),
            "degree": int(degree),
            "n_collocation": int(n_colloc),
            "max_iter": int(max_iter), "ftol": float(ftol),
            "solver": "scipy.optimize.minimize/SLSQP",
            "warm_start": True, "warm_start_source": "indirect_shooting",
            "warm_start_strategy": "fresh_shooting_each_N",
            "wall_budget_s": float(wall_budget_s),
            "tf": float(tf), "Ax_L1": 0.02, "Ax_L2": 0.02,
        },
        cost=float(result.fun), converged=success, residual=max_defect,
        wall_time_s=float(elapsed),
        iterations=int(getattr(result, "nit", 0)),
        nfev=int(getattr(result, "nfev", 0)),
        njev=int(getattr(result, "njev", 0)) if hasattr(result, "njev") else None,
        n_vars=int(n_vars), n_constraints=int(n_constraints),
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        convergence_history=None,
        notes=f"P1 p-sweep degree={degree}, fresh shooting WS, wall-budget callback.",
    )
    rec.validate()
    append_to_summary(rec)
    return {
        "N": N, "degree": degree, "success": success,
        "cost": float(result.fun), "time_s": float(elapsed),
        "nit": int(getattr(result, "nit", 0)),
        "max_defect": max_defect,
        "n_vars": int(n_vars),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, required=True)
    ap.add_argument("--N", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--ftol", type=float, default=1e-7)
    ap.add_argument("--per-N-budget-s", type=float, default=20.0)
    args = ap.parse_args()

    n_colloc = args.degree + 1
    print("=" * 74, flush=True)
    print(f"P1 p-sweep — degree {args.degree}, n_colloc {n_colloc}, "
          f"N = {args.N}", flush=True)
    print(f"  max_iter = {args.max_iter}, ftol = {args.ftol}, "
          f"per-N budget = {args.per_N_budget_s}s", flush=True)
    print("=" * 74, flush=True)

    x0, xf, t0, tf, _ = setup_transfer_problem(Ax_L1=0.02, Ax_L2=0.02)
    sh = run_shooting(x0, xf, t0, tf)
    warm = (sh["t"], sh["state"])
    Jstar = sh["cost"]

    results = []
    for N in args.N:
        print(f"\n--- N = {N} (degree {args.degree}) ---", flush=True)
        e = solve_p_p1(N, args.degree, n_colloc, x0, xf, t0, tf, warm,
                       args.max_iter, args.ftol, args.per_N_budget_s)
        flag = "OK" if e["success"] else "FAIL"
        gap = abs(e["cost"] - Jstar)
        print(f"    [{flag}] cost = {e['cost']:.10f}, gap = {gap:.3e}, "
              f"defect = {e['max_defect']:.2e}, "
              f"iters = {e['nit']}, t = {e['time_s']:.2f}s", flush=True)
        results.append(e)

    print("\n" + "=" * 74)
    print(f"  P1 degree={args.degree} summary  (J* = {Jstar:.10f})")
    print("=" * 74)
    print(f"{'N':>4} {'cost':>16} {'gap':>14} {'defect':>10} {'iter':>5} {'t(s)':>8} feasible?")
    for e in results:
        gap = abs(e["cost"] - Jstar)
        feas = "Y" if e["success"] else ("borderline" if e["max_defect"] < 1e-3 else "NO")
        print(f"{e['N']:>4} {e['cost']:>16.10f} {gap:>14.3e} "
              f"{e['max_defect']:>10.2e} {e['nit']:>5d} {e['time_s']:>8.2f} {feas}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
