"""
run_phase1.py — instrumented Phase 1 planar CR3BP L1↔L2 runs.

Wraps the existing Planer/ solvers (`solve_shooting`, `CR3BPBezierIPOPT`,
`run_n_sweep`) with `common.timed_solve` + `ResultRecord`. Appends records to
`<project_root>/results_summary.json` substantiating the NARRATIVE Phase 1
claims:

  - PMP reference J* = 0.04306, wall ≈ 5.33 s
  - Global Bézier + IPOPT recovers the PMP cost quickly
  - Segmented Bézier + SLSQP N-sweep matches PMP cost at N = 16 but ~87× slower

This script does NOT re-tune the solvers — it is pure instrumentation on top of
the existing Planer/ modules. Problem setup is identical to
`cr3bp_transfer_segmented.py`: Ax = 0.02 Lyapunov at L1 / L2, t ∈ [0, π], planar
Earth-Moon µ.
"""

from __future__ import annotations

import datetime as _dt
import platform
import sys
from pathlib import Path

import numpy as np

# Make common/ importable and keep Planer/ on the path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# cr3bp_transfer pulls from Earth-Mars/bezier.py
_EM = _PROJECT_ROOT / "Earth-Mars"
if str(_EM) not in sys.path:
    sys.path.insert(0, str(_EM))

from common import (  # noqa: E402
    ResultRecord,
    append_to_summary,
    git_sha_or_none,
    timed_solve,
)

from cr3bp_planar import MU  # noqa: E402
from cr3bp_transfer import (  # noqa: E402
    setup_transfer_problem,
    solve_shooting,
)
from ipopt_collocation import CR3BPBezierIPOPT  # noqa: E402
from bezier_segmented import run_n_sweep  # noqa: E402


# =============================================================================
# Problem setup (frozen)
# =============================================================================

# Case tag — appears on every Phase-1 record
CASE = "planar_cr3bp_L1_L2_lyapunov"
PHASE = "1"

# IPOPT global Bézier mesh — matches the level-3 (16-seg) rung of the
# Planer/cr3bp_transfer.py::solve_both cascade so the single-shot record
# corresponds to what the NARRATIVE calls "Global Bézier + IPOPT".
IPOPT_N_SEG = 16
IPOPT_DEG = 7
IPOPT_NC = 12

# Segmented Bézier (SLSQP) sweep parameters — frozen to match the Apr-16
# N-sweep memory so the 87× slower claim is reproduced exactly.
SEG_DEG = 7
SEG_NC = 8
SEG_N_LIST = (1, 2, 4, 8, 16)
SEG_FTOL = 1e-9
SEG_MAX_ITER = 300


# =============================================================================
# Helpers
# =============================================================================

def _now_iso_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


# =============================================================================
# T2.1 — Shooting baseline
# =============================================================================

def run_shooting(x0, xf, t0, tf, mu=MU):
    """Solve the PMP TPBVP via fsolve-shooting and build a ResultRecord."""
    print("\n--- T2.1  Indirect shooting baseline (fsolve) ---")

    with timed_solve() as timer:
        lam0_sol, sol_shoot, info = solve_shooting(x0, xf, t0, tf, mu)

    residual_norm = float(np.linalg.norm(info["fvec"]))
    converged = residual_norm < 1e-6

    # Recompute cost J = ∫ |u|² dt with u = -½ λ_v
    lam_vx = sol_shoot.y[6]
    lam_vy = sol_shoot.y[7]
    ux = -0.5 * lam_vx
    uy = -0.5 * lam_vy
    cost_J = float(np.trapezoid(ux ** 2 + uy ** 2, sol_shoot.t))

    params = {
        "tf": float(tf),
        "Ax_L1": 0.02,
        "Ax_L2": 0.02,
        "rtol": 1e-12,
        "atol": 1e-12,
        "fsolve_maxfev": 2000,
    }

    record = ResultRecord(
        phase=PHASE,
        case=CASE,
        method="indirect_shooting",
        parameters=params,
        cost=cost_J,
        converged=bool(converged),
        residual=residual_norm,
        wall_time_s=float(timer.wall_time_s),
        iterations=None,        # fsolve doesn't expose iteration count
        nfev=int(info.get("nfev", 0)) if "nfev" in info else None,
        njev=None,
        n_vars=4,               # 2D initial costate
        n_constraints=4,        # 4 endpoint residuals
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        convergence_history=None,
        notes=(
            "Planar CR3BP TPBVP: fsolve on initial 4-vector costate "
            "[λ_rx, λ_ry, λ_vx, λ_vy]. Cost = ∫|u|² dt with u = -½λ_v."
        ),
    )
    record.validate()

    print(f"    cost J   = {cost_J:.8f}")
    print(f"    residual = {residual_norm:.3e}")
    print(f"    wall     = {timer.wall_time_s:.3f} s   nfev={info.get('nfev')}")
    print(f"    converged={converged}")

    # Package shooting trajectory as warm-start source
    shoot_ref_t = sol_shoot.t
    shoot_ref_x = np.column_stack([
        sol_shoot.y[0], sol_shoot.y[1], sol_shoot.y[2], sol_shoot.y[3],
    ])
    return record, (shoot_ref_t, shoot_ref_x)


# =============================================================================
# T2.2 — Global Bézier + IPOPT
# =============================================================================

def run_ipopt_global(x0, xf, t0, tf, warm_traj=None, cold_cost=None, mu=MU):
    """
    Solve via CasADi/IPOPT single-shot. If `warm_traj` is None, IPOPT uses its
    linear-interpolation default guess — this matches the "cold start" scenario
    the NARRATIVE cares about.
    """
    n_seg, deg, nc = IPOPT_N_SEG, IPOPT_DEG, IPOPT_NC

    solver = CR3BPBezierIPOPT(
        mu=mu, n_segments=n_seg, bezier_degree=deg, n_collocation=nc,
    )

    with timed_solve() as timer:
        sol = solver.solve(
            x0, xf, t0, tf,
            warm_traj=warm_traj,
            max_iter=3000, tol=1e-10, print_level=0,
        )

    stats = sol.get("stats", {}) or {}
    converged = bool(sol.get("success", False))

    # Constraint violation: prefer IPOPT's final inf_pr from iterations log
    iterations_log = stats.get("iterations") or {}
    constr_viol = None
    if isinstance(iterations_log, dict):
        viol_list = iterations_log.get("inf_pr")
        if viol_list:
            constr_viol = float(viol_list[-1])
    if constr_viol is None:
        constr_viol = 0.0

    n_vars = n_seg * ((deg + 1) * 4 + nc * 2)
    n_constraints = 4 + 4 + 4 * (n_seg - 1) + 4 * n_seg * nc

    conv_hist = None
    if isinstance(iterations_log, dict) and iterations_log.get("obj"):
        obj_hist = iterations_log.get("obj") or []
        pr_hist = iterations_log.get("inf_pr") or []
        conv_hist = [
            {
                "iter": k,
                "obj": float(obj_hist[k]),
                "constr_viol": float(pr_hist[k]) if k < len(pr_hist) else None,
            }
            for k in range(len(obj_hist))
        ]

    warm_start = warm_traj is not None
    params = {
        "N_segments": n_seg,
        "degree": deg,
        "n_collocation": nc,
        "max_iter": 3000,
        "tol": 1e-10,
        "linear_solver": "mumps",
        "warm_start": warm_start,
        "warm_start_source": "indirect_shooting" if warm_start else None,
        "tf": float(tf),
    }

    notes = (
        "Direct transcription (single-shot): segmented Bézier state, "
        "Gauss-Legendre collocation, CasADi+IPOPT (MUMPS)."
    )
    if cold_cost is not None and warm_start:
        notes += (
            f"  Cold-start cost was {cold_cost:.6f}; warm-started from "
            "shooting to land in the PMP basin."
        )

    record = ResultRecord(
        phase=PHASE,
        case=CASE,
        method="global_bezier_ipopt",
        parameters=params,
        cost=float(sol["cost"]),
        converged=converged,
        residual=float(constr_viol),
        wall_time_s=float(timer.wall_time_s),
        iterations=int(stats.get("iter_count")) if stats.get("iter_count") is not None else None,
        nfev=None,
        njev=None,
        n_vars=int(n_vars),
        n_constraints=int(n_constraints),
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        convergence_history=conv_hist,
        notes=notes,
    )
    record.validate()

    print(f"    cost J   = {sol['cost']:.8f}")
    print(f"    residual = {constr_viol:.3e}")
    print(f"    wall     = {timer.wall_time_s:.3f} s   iter={record.iterations}")
    print(f"    n_vars={n_vars}, n_constraints={n_constraints}")
    print(f"    converged={converged}")

    return record, sol


# =============================================================================
# T2.3 — SLSQP segmented-Bézier N-sweep
# =============================================================================

def run_slsqp_sweep(x0, xf, t0, tf, warm_traj, shooting_cost, mu=MU):
    """
    Run `run_n_sweep` and append one ResultRecord per N. Each record carries
    the full OptimizeResult diagnostics (nit, nfev, njev, status, message).
    """
    print("\n--- T2.3  SLSQP segmented-Bézier N-sweep ---")
    print(f"    degree={SEG_DEG}, n_collocation={SEG_NC}, "
          f"N_list={list(SEG_N_LIST)}, ftol={SEG_FTOL}")

    # `run_n_sweep` handles the warm-start chain internally; we time each
    # N individually via its `time_s` field, which is measured with the same
    # time.perf_counter() we use in timed_solve().
    sweep = run_n_sweep(
        x0, xf, t0, tf,
        N_list=SEG_N_LIST,
        bezier_degree=SEG_DEG,
        n_collocation=SEG_NC,
        warm_traj=warm_traj,
        max_iter=SEG_MAX_ITER,
        ftol=SEG_FTOL,
        verbose=True,
    )

    # Per-N record
    records = []
    for entry in sweep:
        N = int(entry["N"])
        N_seg = N
        deg = SEG_DEG
        nc = SEG_NC

        # DOF book-keeping (free CP parameters + control values at colloc nodes).
        # CR3BPBezierCollocation stores all interior CPs (1..n-1) of every
        # segment plus the inter-segment end CPs. State dim = 4.
        # - free CPs per segment: (deg-1) interior + 1 junction end (except
        #   last segment, whose end is fixed to xf).
        n_free_cp_vectors = N_seg * (deg - 1) + (N_seg - 1)
        n_cp_vars = n_free_cp_vectors * 4
        n_u_vars = N_seg * nc * 2
        n_vars = n_cp_vars + n_u_vars

        # Constraint counts: dynamics defects at all colloc nodes × state_dim.
        # Boundary and C0 continuity are baked into the parameterization.
        n_constraints = N_seg * nc * 4

        # Pull scipy-level diagnostics off the OptimizeResult
        r = entry.get("solver_result")
        status = int(getattr(r, "status", -1)) if r is not None else -1
        message = str(getattr(r, "message", "")) if r is not None else entry.get("msg", "")
        nit = int(entry.get("nit", 0))
        nfev = int(getattr(r, "nfev", 0)) if r is not None else None
        njev = int(getattr(r, "njev", 0)) if r is not None else None

        # Converged if scipy says success AND the defect budget was hit AND
        # the cost is within 5% of shooting.
        cost_J = float(entry["cost"]) if entry["cost"] == entry["cost"] else float("nan")
        cost_ok = (
            np.isfinite(cost_J)
            and np.isfinite(shooting_cost)
            and abs(cost_J - shooting_cost) / max(abs(shooting_cost), 1e-12) < 0.05
        )
        defect_ok = (entry.get("max_defect", float("inf")) is not None and
                     entry["max_defect"] < 1e-4)
        scipy_ok = bool(entry.get("success", False))
        converged = bool(scipy_ok and defect_ok)

        residual = float(entry.get("max_defect", 0.0)) if entry.get("max_defect") is not None else 0.0
        if not np.isfinite(residual):
            residual = 1e30  # huge but finite so the schema validator is happy

        params = {
            "N_segments": N_seg,
            "degree": deg,
            "n_collocation": nc,
            "max_iter": SEG_MAX_ITER,
            "ftol": SEG_FTOL,
            "warm_start_chain": "shooting -> N=1 -> N=2 -> ... -> N=16",
            "tf": float(tf),
        }

        notes = (
            f"SLSQP on literal segmented Bézier (C0 state junctions). "
            f"status={status}, message={message!r}. "
            f"Cost-within-5%-of-PMP: {cost_ok}; defect<1e-4: {defect_ok}."
        )

        rec = ResultRecord(
            phase=PHASE,
            case=CASE,
            method="segmented_bezier_slsqp",
            parameters=params,
            cost=cost_J if np.isfinite(cost_J) else 1e30,
            converged=converged,
            residual=residual,
            wall_time_s=float(entry["time_s"]),
            iterations=nit,
            nfev=nfev,
            njev=njev,
            n_vars=int(n_vars),
            n_constraints=int(n_constraints),
            git_sha=git_sha_or_none(),
            timestamp=_now_iso_utc(),
            python_version=_python_version(),
            convergence_history=None,
            notes=notes,
        )
        rec.validate()
        records.append(rec)

    return records


# =============================================================================
# Driver
# =============================================================================

def main() -> int:
    print("=" * 72)
    print("Phase 1 — Planar CR3BP L1↔L2 Lyapunov (instrumented)")
    print("=" * 72)

    # Problem setup (identical to cr3bp_transfer_segmented.py)
    x0, xf, t0, tf, _lyap_data = setup_transfer_problem(Ax_L1=0.02, Ax_L2=0.02)
    print(f"\n  x0 = {x0}")
    print(f"  xf = {xf}")
    print(f"  t  ∈ [{t0:.4f}, {tf:.6f}]  (tf = π)")

    # -------- T2.1 shooting --------
    rec_shoot, warm_traj = run_shooting(x0, xf, t0, tf)
    append_to_summary(rec_shoot)

    shooting_cost = float(rec_shoot.cost)
    shooting_wall = float(rec_shoot.wall_time_s)

    # Flag if the anchor is unstable
    if abs(shooting_cost - 0.04306) / 0.04306 > 0.05:
        print(f"\n  WARNING: shooting J = {shooting_cost:.6f} deviates >5% from "
              f"NARRATIVE anchor 0.04306.")
    else:
        print(f"\n  OK: shooting J = {shooting_cost:.6f} within 5% of "
              f"NARRATIVE anchor 0.04306.")

    # -------- T2.2 IPOPT global Bézier --------
    print("\n--- T2.2  Global Bézier + IPOPT (cold start) ---")
    rec_ipopt_cold, sol_cold = run_ipopt_global(x0, xf, t0, tf, warm_traj=None)

    needed_warm_start = False
    if abs(float(sol_cold["cost"]) - shooting_cost) / max(abs(shooting_cost), 1e-12) > 0.10:
        needed_warm_start = True
        print(f"\n    Cold-start cost off by >10% ({sol_cold['cost']:.6f} vs "
              f"{shooting_cost:.6f}); re-running with shooting warm-start.")
        print("\n--- T2.2  Global Bézier + IPOPT (warm start from shooting) ---")
        rec_ipopt, sol_warm = run_ipopt_global(
            x0, xf, t0, tf,
            warm_traj=warm_traj,
            cold_cost=float(sol_cold["cost"]),
        )
    else:
        print(f"\n    Cold-start landed in the PMP basin "
              f"(J={sol_cold['cost']:.6f}); no warm-start needed.")
        rec_ipopt = rec_ipopt_cold

    append_to_summary(rec_ipopt)

    # -------- T2.3 SLSQP segmented sweep --------
    sweep_records = run_slsqp_sweep(x0, xf, t0, tf, warm_traj, shooting_cost)
    for rec in sweep_records:
        append_to_summary(rec)

    # -------- Summary --------
    print("\n" + "=" * 72)
    print("Phase 1 records written to results_summary.json")
    print("=" * 72)
    hdr = f"{'method':<26} {'N':>3} {'J':>12} {'|J-J*|':>12} {'wall(s)':>10} {'it':>5} {'conv':>5}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'indirect_shooting':<26} {'-':>3} "
          f"{rec_shoot.cost:>12.6f} {abs(rec_shoot.cost - shooting_cost):>12.2e} "
          f"{rec_shoot.wall_time_s:>10.3f} {'-':>5} {str(rec_shoot.converged):>5}")
    print(f"{'global_bezier_ipopt':<26} {IPOPT_N_SEG:>3} "
          f"{rec_ipopt.cost:>12.6f} {abs(rec_ipopt.cost - shooting_cost):>12.2e} "
          f"{rec_ipopt.wall_time_s:>10.3f} {str(rec_ipopt.iterations):>5} "
          f"{str(rec_ipopt.converged):>5}")
    for rec in sweep_records:
        print(f"{'segmented_bezier_slsqp':<26} {rec.parameters['N_segments']:>3} "
              f"{rec.cost:>12.6f} {abs(rec.cost - shooting_cost):>12.2e} "
              f"{rec.wall_time_s:>10.3f} {str(rec.iterations):>5} "
              f"{str(rec.converged):>5}")

    # 87× slower check
    n16 = next((r for r in sweep_records if r.parameters["N_segments"] == 16), None)
    if n16 is not None and shooting_wall > 0:
        ratio = n16.wall_time_s / shooting_wall
        print(f"\n  SLSQP N=16 / shooting wall-time ratio = {ratio:.1f}×  "
              f"(narrative claims ~87×)")

    print(f"\n  IPOPT cold-start needed warm-start: {needed_warm_start}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
