"""
run_phase0_psweep.py — Polynomial-degree (p) sweep on Phase 0 Earth-Mars.

Companion to the cubic-headline N-sweep, this script fills in the empirical
hp-collocation convergence map at degrees 4, 5, 6 (between the existing
cubic / degree-7 sweeps already in results_summary.json). For each (degree,
n_collocation = degree+1, N) it runs SLSQP with fresh shooting warm-start
and writes the record.

Theory: at fixed N, |J - J*| should drop as roughly N^-(p+1) for piecewise
degree-p approximations of a smooth solution; in the asymptotic regime
higher p means much faster convergence per N. The cubic vs degree-7
data already in JSON spans this — degrees 4-6 fill the gap.

Records: phase=0, method=segmented_bezier_slsqp, parameters.degree ∈ {4,5,6}.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import run_phase0_nsweep as p0  # noqa: E402
from common import ResultRecord, append_to_summary, git_sha_or_none  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from bezier import BezierCollocation  # noqa: E402


def solve_p_freshshoot(N, degree, n_colloc, warm_traj, max_iter, ftol):
    """One (degree, n_colloc, N) solve, fresh shooting warm-start."""
    solver = BezierCollocation(
        gravity_func=lambda r: p0.gravity_2body(r, mu=p0.MU),
        pos_dim=2, n_segments=N,
        bezier_degree=degree, n_collocation=n_colloc,
    )
    z_guess = solver._warm_start_from_trajectory(
        warm_traj[0], warm_traj[1], p0.X0_FULL, p0.XF_FULL, p0.T0, p0.TF,
    )
    dt_seg = (p0.TF - p0.T0) / N
    t_start = time.perf_counter()
    result = minimize(
        solver._objective, z_guess,
        args=(p0.X0_FULL, p0.XF_FULL, dt_seg),
        method="SLSQP",
        constraints={"type": "eq", "fun": solver._defects,
                     "args": (p0.X0_FULL, p0.XF_FULL, dt_seg)},
        options={"maxiter": max_iter, "ftol": ftol, "disp": False},
    )
    elapsed = time.perf_counter() - t_start
    sol = solver._evaluate(result.x, p0.X0_FULL, p0.XF_FULL, p0.T0, p0.TF)
    sol["cost"] = float(result.fun)
    defs = solver._defects(result.x, p0.X0_FULL, p0.XF_FULL, dt_seg)
    max_defect = float(np.max(np.abs(defs)))
    success = bool(result.success and max_defect < 1e-4)

    n_vars = solver._total_vars()
    n_constraints = N * n_colloc * solver.state_dim

    rec = ResultRecord(
        phase="0", case="earth_mars_2body", method="segmented_bezier_slsqp",
        parameters={
            "N_segments": int(N),
            "degree": int(degree),
            "n_collocation": int(n_colloc),
            "max_iter": int(max_iter), "ftol": float(ftol),
            "solver": "scipy.optimize.minimize/SLSQP",
            "warm_start": True, "warm_start_source": "indirect_shooting",
            "warm_start_strategy": "fresh_shooting_each_N",
            "tf": p0.TF,
        },
        cost=float(result.fun), converged=success, residual=max_defect,
        wall_time_s=float(elapsed),
        iterations=int(getattr(result, "nit", 0)),
        nfev=int(getattr(result, "nfev", 0)),
        njev=int(getattr(result, "njev", 0)) if hasattr(result, "njev") else None,
        n_vars=int(n_vars), n_constraints=int(n_constraints),
        git_sha=git_sha_or_none(),
        timestamp=p0._now_iso_utc(),
        python_version=p0._python_version(),
        convergence_history=None,
        notes=f"P0 p-sweep degree={degree}, fresh shooting WS each N.",
    )
    rec.validate()
    append_to_summary(rec)
    return {
        "N": N, "degree": degree, "success": success,
        "cost": float(result.fun), "time_s": float(elapsed),
        "nit": int(getattr(result, "nit", 0)),
        "nfev": int(getattr(result, "nfev", 0)),
        "max_defect": max_defect,
        "n_vars": int(n_vars),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, required=True,
                    help="Polynomial degree to sweep (n_colloc = degree+1)")
    ap.add_argument("--N", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--ftol", type=float, default=1e-9)
    args = ap.parse_args()

    n_colloc = args.degree + 1
    print("=" * 74, flush=True)
    print(f"P0 p-sweep — degree {args.degree}, n_colloc {n_colloc}, "
          f"N = {args.N}", flush=True)
    print("=" * 74, flush=True)

    shooting = p0.run_shooting()
    warm = (shooting["t"], np.column_stack([shooting["r"], shooting["v"]]))

    results = []
    for N in args.N:
        print(f"\n--- N = {N} (degree {args.degree}) ---", flush=True)
        e = solve_p_freshshoot(N, args.degree, n_colloc, warm,
                               args.max_iter, args.ftol)
        flag = "OK" if e["success"] else "FAIL"
        gap = abs(e["cost"] - shooting["cost"])
        print(f"    [{flag}] cost = {e['cost']:.10f}, gap = {gap:.3e}, "
              f"defect = {e['max_defect']:.2e}, "
              f"iters = {e['nit']}, t = {e['time_s']:.2f}s", flush=True)
        results.append(e)

    print("\n" + "=" * 74)
    print(f"  degree={args.degree} (n_colloc={n_colloc}) summary")
    print("=" * 74)
    print(f"{'N':>4} {'cost':>16} {'gap':>14} {'defect':>10} {'iter':>5} {'t(s)':>8}")
    for e in results:
        gap = abs(e["cost"] - shooting["cost"])
        print(f"{e['N']:>4} {e['cost']:>16.10f} {gap:>14.3e} "
              f"{e['max_defect']:>10.2e} {e['nit']:>5d} {e['time_s']:>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
