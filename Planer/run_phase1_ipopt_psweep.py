"""
run_phase1_ipopt_psweep.py — Phase 1 IPOPT polynomial-degree sweep.

Companion to run_phase1_psweep.py (the SLSQP version). Same problem, same
N values, same warm-start strategy — but driven through CR3BPBezierIPOPT
(CasADi + IPOPT + AD derivatives). The empirical question is whether
IPOPT exhibits the basin-multiplicity and infeasibility-chasing failure
modes that SLSQP did, or whether interior-point feasibility handling
makes the convergence clean across the (degree, N) plane.

Records: phase=1, method=segmented_bezier_ipopt, parameters.degree ∈ {3..7}.
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
from cr3bp_planar import MU, cr3bp_jacobi_planar  # noqa: E402
from cr3bp_transfer import setup_transfer_problem, solve_shooting  # noqa: E402
from ipopt_collocation import CR3BPBezierIPOPT, _compute_max_defect  # noqa: E402


def _now_iso_utc():
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _python_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def run_shooting(x0, xf, t0, tf):
    print("\n--- Phase 1 PMP shooting (anchor) ---", flush=True)
    t_start = time.perf_counter()
    lam0, sol, info = solve_shooting(x0, xf, t0, tf, MU)
    elapsed = time.perf_counter() - t_start
    lam_v = sol.y[6:8].T
    u = -0.5 * lam_v
    cost = float(np.trapezoid(np.sum(u ** 2, axis=1), sol.t))
    print(f"  J* = {cost:.10f}, wall = {elapsed:.2f}s", flush=True)
    return {
        "cost": cost,
        "t": sol.t,
        "state": sol.y[:4].T,
        "wall_s": float(elapsed),
    }


def solve_ipopt_p1(N, degree, n_colloc, x0, xf, t0, tf, warm_traj,
                    max_iter=3000, tol=1e-8, mu=MU):
    """One (degree, N) IPOPT solve, warm-started from shooting trajectory."""
    solver = CR3BPBezierIPOPT(
        mu=mu, n_segments=N, bezier_degree=degree, n_collocation=n_colloc,
    )

    result = solver.solve(
        x0, xf, t0, tf,
        warm_traj=warm_traj,
        max_iter=max_iter,
        tol=tol,
        print_level=0,
    )

    max_defect = float(_compute_max_defect(result, mu))
    success = bool(result.get("success", False)) and (max_defect < 1e-3)
    success = bool(success)  # ensure pure Python bool, not numpy.bool_

    stats = result.get("stats", {}) or {}
    iters = int(stats.get("iter_count", 0)) if stats.get("iter_count") is not None else None

    # n_vars / n_constraints — sized like the SLSQP runs but slightly different
    # because IPOPT keeps ALL CPs as variables and enforces BCs+continuity as
    # equality constraints. Approximate.
    n_vars_total = N * ((degree + 1) * 4 + n_colloc * 2)
    n_constraints = (4 + 4 + 4 * (N - 1) + 4 * N * n_colloc)  # BCs + C0 + defects

    rec = ResultRecord(
        phase="1", case="cr3bp_l1_l2_lyapunov_planar",
        method="segmented_bezier_ipopt",
        parameters={
            "N_segments": int(N),
            "degree": int(degree),
            "n_collocation": int(n_colloc),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "linear_solver": "mumps",
            "warm_start": warm_traj is not None,
            "warm_start_source": "indirect_shooting" if warm_traj is not None else None,
            "tf": float(tf),
            "Ax_L1": 0.02,
            "Ax_L2": 0.02,
        },
        cost=float(result["cost"]),
        converged=success,
        residual=float(max_defect),
        wall_time_s=float(result["solve_time"]),
        iterations=iters,
        nfev=None,
        njev=None,
        n_vars=int(n_vars_total),
        n_constraints=int(n_constraints),
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        convergence_history=None,
        notes=f"P1 IPOPT p-sweep degree={degree}, fresh shooting WS.",
    )
    rec.validate()
    append_to_summary(rec)

    return {
        "N": N, "degree": degree,
        "success": success,
        "cost": float(result["cost"]),
        "time_s": float(result["solve_time"]),
        "iters": iters,
        "max_defect": float(max_defect),
        "n_vars": int(n_vars_total),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--N", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args()

    print("=" * 74, flush=True)
    print(f"P1 IPOPT p-sweep — degrees {args.degree}, N = {args.N}", flush=True)
    print(f"  max_iter = {args.max_iter}, tol = {args.tol}", flush=True)
    print("=" * 74, flush=True)

    x0, xf, t0, tf, _ = setup_transfer_problem(Ax_L1=0.02, Ax_L2=0.02)
    sh = run_shooting(x0, xf, t0, tf)
    warm = (sh["t"], sh["state"])
    Jstar = sh["cost"]

    results = []
    for d in args.degree:
        n_colloc = d + 1
        for N in args.N:
            print(f"\n--- degree={d}, N={N}, n_colloc={n_colloc} ---", flush=True)
            try:
                e = solve_ipopt_p1(N, d, n_colloc, x0, xf, t0, tf, warm,
                                    max_iter=args.max_iter, tol=args.tol)
            except Exception as exc:
                print(f"    EXCEPTION: {exc!r}", flush=True)
                continue
            flag = "OK" if e["success"] else "FAIL"
            gap = abs(e["cost"] - Jstar)
            print(f"    [{flag}] cost = {e['cost']:.10f}, gap = {gap:.3e}, "
                  f"defect = {e['max_defect']:.2e}, "
                  f"iters = {e['iters']}, t = {e['time_s']:.2f}s", flush=True)
            results.append(e)

    # Summary
    print("\n" + "=" * 74)
    print(f"  P1 IPOPT p-sweep summary  (J* = {Jstar:.10f})")
    print("=" * 74)
    print(f"{'deg':>4} {'N':>4} {'cost':>16} {'gap':>12} {'defect':>10} "
          f"{'iter':>5} {'t(s)':>8} feas?")
    for e in results:
        gap = abs(e["cost"] - Jstar)
        feas = "Y" if e["success"] else "N"
        print(f"{e['degree']:>4} {e['N']:>4} {e['cost']:>16.10f} {gap:>12.3e} "
              f"{e['max_defect']:>10.2e} {str(e['iters']):>5} {e['time_s']:>8.2f} {feas}")

    out = _HERE / "phase1_ipopt_psweep_summary.txt"
    with open(out, "w") as f:
        f.write(f"Phase 1 IPOPT p-sweep summary\n")
        f.write(f"PMP shooting J* = {Jstar:.10f}, wall = {sh['wall_s']:.3f}s\n\n")
        f.write(f"{'deg':>4} {'N':>4} {'cost':>16} {'gap':>12} {'defect':>10} "
                f"{'iter':>5} {'t(s)':>8} feas?\n")
        for e in results:
            gap = abs(e["cost"] - Jstar)
            feas = "Y" if e["success"] else "N"
            f.write(f"{e['degree']:>4} {e['N']:>4} {e['cost']:>16.10f} "
                    f"{gap:>12.3e} {e['max_defect']:>10.2e} "
                    f"{str(e['iters']):>5} {e['time_s']:>8.2f} {feas}\n")
    print(f"\nSaved: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
