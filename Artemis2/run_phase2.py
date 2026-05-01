"""
run_phase2.py — instrumented Phase 2 Artemis II runs.

Wraps `solve_shooting` and `solve_ipopt_collocation` (Post-TLI ephemeris case)
and `solve_ipopt` (Full mission case) with `common.timed_solve` +
`ResultRecord`, appending to `<project_root>/results_summary.json`.

Substantiates NARRATIVE Phase 2 claims:
  - "IPOPT multi-shooting converges with the same solver class carried over
    from P1" — captures iteration count, wall time, return status from
    CasADi solver.stats().
  - "15-seed sweep exposes initial-guess fragility against IPOPT's
    robustness" — writes ONE ResultRecord per seed (T3.2). A separate
    "best_of_15" summary record is also written.
  - "Deliverables: trajectory, control history, convergence history vs.
    OEM" — produces Artemis2/convergence_history.png from the IPOPT
    iteration callback output (T3.4).

This script is pure instrumentation — it calls the existing Artemis2 modules
in-place. No re-tuning of solver options.

Run order:
    python3 run_phase2.py               # runs Post-TLI only (default, safer)
    python3 run_phase2.py --full        # also attempts Full mission
    python3 run_phase2.py --skip-shooting   # skip the 15-seed sweep
    python3 run_phase2.py --skip-ipopt      # skip the IPOPT solve
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import io
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- path setup ---
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_EPHEM_FULL = _HERE / "Ephem_Full"

for p in (_PROJECT_ROOT, _EPHEM_FULL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import (  # noqa: E402
    ResultRecord,
    append_to_summary,
    git_sha_or_none,
    timed_solve,
)

# Importing artemis2_ephemeris prints a banner; silence it so the log stays clean
_saved_stdout = sys.stdout
sys.stdout = io.StringIO()
import artemis2_ephemeris as ae  # noqa: E402
import artemis2_full_mission as afm  # noqa: E402
sys.stdout = _saved_stdout


# =============================================================================
# Fixed paths / constants
# =============================================================================

OEM_FILE = str(_PROJECT_ROOT / "Artemis_II_OEM_2026_04_10_Post-ICPS-Sep-to-EI.asc")

# Post-TLI segment — matches Ephem_Full/artemis2_ephemeris.py::main()
POST_TLI_SEG_START_DAY = 1.0
POST_TLI_SEG_END_DAY = 8.5

CASE_POST_TLI = "artemis2_post_tli"
CASE_FULL = "artemis2_full_mission"
PHASE = "2"


# =============================================================================
# Helpers
# =============================================================================

def _now_iso_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


# =============================================================================
# Post-TLI case setup (reuses parse_oem + EphemerisCache from artemis2_ephemeris)
# =============================================================================

def setup_post_tli():
    """Reproduce the Post-TLI segment from artemis2_ephemeris.main."""
    times_utc, positions, velocities = ae.parse_oem(OEM_FILE)
    t0_utc = times_utc[0]
    day_offsets = np.array([(t - t0_utc).total_seconds() / 86400.0 for t in times_utc])

    mask = (day_offsets >= POST_TLI_SEG_START_DAY) & (day_offsets <= POST_TLI_SEG_END_DAY)
    seg_times_utc = [t for t, m in zip(times_utc, mask) if m]
    seg_pos = positions[mask]
    seg_vel = velocities[mask]

    t_seg_start = seg_times_utc[0]
    t_seg_end = seg_times_utc[-1]
    times_sec = np.array([(t - t_seg_start).total_seconds() for t in seg_times_utc])

    r0 = seg_pos[0]
    v0 = seg_vel[0]
    rf = seg_pos[-1]
    vf = seg_vel[-1]

    margin = 3600.0
    cache = ae.EphemerisCache(
        t_seg_start - timedelta(seconds=margin),
        t_seg_end + timedelta(seconds=margin),
        n_points=4000,  # 80s resolution for 7.5 days
    )
    t_offset = margin
    t_span_cache = [t_offset, times_sec[-1] + t_offset]
    times_sec_cache = times_sec + t_offset

    print(f"\n[Post-TLI] segment day {POST_TLI_SEG_START_DAY}-{POST_TLI_SEG_END_DAY}: "
          f"{len(seg_times_utc)} OEM points, duration {times_sec[-1]/86400:.2f} d")
    print(f"[Post-TLI] r0 norm = {np.linalg.norm(r0):.0f} km   "
          f"rf norm = {np.linalg.norm(rf):.0f} km")

    return dict(
        r0=r0, v0=v0, rf=rf, vf=vf,
        t_span=t_span_cache,
        ephem_cache=cache,
        nasa_t=times_sec_cache, nasa_pos=seg_pos, nasa_vel=seg_vel,
        duration_days=times_sec[-1] / 86400.0,
    )


# =============================================================================
# T3.2 — 15-seed sweep on Post-TLI shooting
# =============================================================================

def _seed_record_from_entry(entry: dict, best_cost_guess: float = None) -> ResultRecord:
    """Build a ResultRecord for a single shooting seed from its sweep entry.
    Shared by the incremental on_seed_done callback and the final rollup.
    """
    converged = bool(entry.get("converged", False))
    residual = entry.get("residual")
    if residual is None:
        residual = 1e30
    residual = float(residual)
    if not np.isfinite(residual):
        residual = 1e30

    cost = entry.get("cost")
    if cost is None or not np.isfinite(cost):
        cost = 0.0  # placeholder — best-of-15 rollup carries the real J

    wall = entry.get("wall_time_s")
    if wall is None:
        wall = 0.0
    nfev = entry.get("nfev") if entry.get("nfev") is not None else 0

    params = {
        "seed_index": int(entry["seed_index"]),
        "seed_strategy": entry["seed_strategy"],
        "seed_scale": float(entry["seed_scale"]),
        "lam0_guess_norm": float(np.linalg.norm(entry["lam0_guess"])),
        "fsolve_maxfev": 80,
        "fsolve_residual_early_stop": 1e-4,
        "segment_days": [POST_TLI_SEG_START_DAY, POST_TLI_SEG_END_DAY],
        "integrator": "DOP853",
        "rtol": 1e-11,
        "atol": 1e-13,
        "ephem": "astropy_builtin_DE405",
    }

    return ResultRecord(
        phase=PHASE,
        case=CASE_POST_TLI,
        method="indirect_shooting",
        parameters=params,
        cost=float(cost),
        converged=converged,
        residual=residual,
        wall_time_s=float(wall),
        iterations=None,
        nfev=int(nfev),
        njev=None,
        n_vars=6,
        n_constraints=6,
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        convergence_history=None,
        notes=(
            f"One seed of the 15-seed multi-start sweep. Strategy="
            f"{entry['seed_strategy']} scale={entry['seed_scale']:.1e}. "
            f"Per-guess maxfev=80. "
            f"{'exception: ' + entry['exception'] if entry.get('exception') else ''}"
        ).strip(),
    )


def run_shooting_sweep(ctx):
    """Run all 15 shooting seeds (no early-stop). Emits one ResultRecord per
    seed (persisted incrementally via on_seed_done so a mid-sweep timeout
    doesn't lose work) plus one "best_of_15" summary record.
    """
    print("\n" + "=" * 72)
    print("Phase 2 — Post-TLI indirect shooting (15-seed sweep)")
    print("=" * 72)

    seed_records: list[dict] = []

    def _persist_seed(entry: dict) -> None:
        """Callback: write this seed's record immediately."""
        try:
            rec = _seed_record_from_entry(entry)
            rec.validate()
            append_to_summary(rec)
            print(f"  [persist] wrote seed {entry['seed_index']:2d} record "
                  f"(conv={entry.get('converged')})", flush=True)
        except Exception as e:
            print(f"  [persist] failed for seed {entry.get('seed_index')}: {e}", flush=True)

    with timed_solve() as sweep_timer:
        result = ae.solve_shooting(
            ctx["r0"], ctx["v0"], ctx["rf"], ctx["vf"],
            ctx["t_span"], ctx["ephem_cache"],
            n_guesses=15,
            seed_records=seed_records,
            early_stop=False,  # run ALL 15 seeds for the fragility record
            on_seed_done=_persist_seed,
        )

    if result is None:
        print("  ALL seeds failed — no usable shooting solution.")
        best_cost = float("nan")
        best_residual = float("inf")
        sol_sh = None
        best_lam0 = None
    else:
        sol_sh, best_lam0, best_cost = result
        # find the winner's residual
        best_residual = float(min(
            (s["residual"] for s in seed_records
             if s.get("residual") is not None),
            default=float("inf"),
        ))

    # --- REWRITE the winning seed's per-seed record so its cost field is
    # populated with the integrated J from the best-costate forward run ---
    if best_lam0 is not None:
        for entry in seed_records:
            if entry.get("lam0_sol") is not None:
                if np.allclose(entry["lam0_sol"], best_lam0, rtol=1e-8, atol=1e-12):
                    entry["cost"] = float(best_cost)
                    rec = _seed_record_from_entry(entry)
                    rec.validate()
                    append_to_summary(rec)  # dedup key matches → overwrites
                    break

    # --- best-of-15 summary record ---
    n_converged = sum(1 for s in seed_records if s.get("converged"))
    summary_params = {
        "seed_strategy": "best_of_15",
        "n_seeds": 15,
        "n_converged_1e-4": int(n_converged),
        "fsolve_maxfev": 80,
        "segment_days": [POST_TLI_SEG_START_DAY, POST_TLI_SEG_END_DAY],
    }

    summary_rec = ResultRecord(
        phase=PHASE,
        case=CASE_POST_TLI,
        method="indirect_shooting",
        parameters=summary_params,
        cost=float(best_cost) if np.isfinite(best_cost) else 0.0,
        converged=bool(best_residual < 1e-4),
        residual=float(best_residual if np.isfinite(best_residual) else 1e30),
        wall_time_s=float(sweep_timer.wall_time_s),
        iterations=None,
        nfev=sum(int(s.get("nfev", 0) or 0) for s in seed_records),
        njev=None,
        n_vars=6,
        n_constraints=6,
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        convergence_history=None,
        notes=(
            f"Best-of-15 rollup over all seeds. {n_converged}/15 converged at "
            f"residual < 1e-4 (the arc-scaled early-stop threshold). Per-seed "
            f"records are written with seed_strategy in "
            f"{{velocity_aligned, position_aligned, random_normal}}."
        ),
    )
    summary_rec.validate()
    append_to_summary(summary_rec)

    print(f"\n  Converged seeds: {n_converged}/15")
    print(f"  Best residual:   {best_residual:.3e}")
    print(f"  Total sweep wall time: {sweep_timer.wall_time_s:.1f} s")

    return summary_rec, seed_records, sol_sh, best_lam0


# =============================================================================
# T3.1 + T3.4 — Post-TLI IPOPT multi-shooting + callback
# =============================================================================

def run_ipopt_post_tli(ctx, sol_shooting=None):
    print("\n" + "=" * 72)
    print("Phase 2 — Post-TLI IPOPT multi-shooting (with convergence callback)")
    print("=" * 72)

    stats_out: dict = {}

    # Match artemis2_ephemeris.main: n_seg=100, warm-start from NASA OEM.
    # If we have a shooting solution, use it preferentially.
    with timed_solve() as timer:
        result = ae.solve_ipopt_collocation(
            ctx["r0"], ctx["v0"], ctx["rf"], ctx["vf"],
            ctx["t_span"], ctx["ephem_cache"],
            n_seg=100,
            sol_shooting=((sol_shooting,) if sol_shooting is not None else None),
            nasa_warmstart=(ctx["nasa_t"], ctx["nasa_pos"], ctx["nasa_vel"]),
            stats_out=stats_out,
        )

    converged = bool(stats_out.get("success", False))
    iter_count = stats_out.get("iter_count")
    t_wall_internal = stats_out.get("t_wall_total")  # CasADi internal, may differ from perf_counter
    return_status = stats_out.get("return_status", "")
    conv_hist = stats_out.get("convergence_history") or []

    if result is not None:
        X_sol, U_sol, seg_times, J_val = result
        cost = float(J_val)
        rf_err = float(np.linalg.norm(X_sol[0:3, -1] - ctx["rf"]))
    else:
        cost = float("nan")
        rf_err = float("inf")

    # Final constraint violation from last iteration log entry
    final_constr = 0.0
    if conv_hist:
        final_constr = float(conv_hist[-1].get("constr_viol") or 0.0)

    n_seg = 100
    ns, nd = 6, 3
    n_vars = ns * (n_seg + 1) + nd * n_seg
    # Constraints: BC(12) + RK4 defects (6 * n_seg) + control bounds (2 * 3 * n_seg, box bounds)
    # We count equality constraints only: 12 BC + 6 * n_seg defects.
    n_constraints = 12 + 6 * n_seg

    params = {
        "n_segments": n_seg,
        "control_parameterization": "piecewise_constant",
        "rk4_substeps_per_segment": 4,
        "max_iter": 3000,
        "tol": 1e-8,
        "acceptable_tol": 1e-6,
        "linear_solver": "mumps",
        "warm_start_init_point": "yes",
        "warm_start_source": "NASA OEM" if sol_shooting is None else "shooting_then_OEM",
        "segment_days": [POST_TLI_SEG_START_DAY, POST_TLI_SEG_END_DAY],
    }

    record = ResultRecord(
        phase=PHASE,
        case=CASE_POST_TLI,
        method="multi_shooting_ipopt",
        parameters=params,
        cost=cost if np.isfinite(cost) else 1e30,
        converged=converged,
        residual=float(final_constr if np.isfinite(final_constr) else 1e30),
        wall_time_s=float(timer.wall_time_s),
        iterations=int(iter_count) if iter_count is not None else None,
        nfev=None,
        njev=None,
        n_vars=int(n_vars),
        n_constraints=int(n_constraints),
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        convergence_history=conv_hist if conv_hist else None,
        notes=(
            f"CasADi Opti()+IPOPT, RK4 multiple-shooting, ephemeris-driven N-body. "
            f"return_status={return_status}. Final position error to OEM endpoint: "
            f"{rf_err:.3e} km. Warm-started from NASA OEM states "
            f"(shooting unused here; OEM is a stronger prior)."
        ),
    )
    record.validate()
    append_to_summary(record)

    print(f"\n  IPOPT wall: {timer.wall_time_s:.2f} s   "
          f"(CasADi internal t_wall_total: {t_wall_internal})")
    print(f"  iter_count: {iter_count}   return_status: {return_status}")
    print(f"  cost J:     {cost:.6e}")
    print(f"  final inf_pr: {final_constr:.3e}")
    print(f"  convergence_history length: {len(conv_hist)}")

    return record, conv_hist


# =============================================================================
# Full-mission case (stretch goal)
# =============================================================================

def run_full_mission():
    print("\n" + "=" * 72)
    print("Phase 2 — Full mission IPOPT multi-shooting")
    print("=" * 72)

    times_utc, positions, velocities = afm.parse_oem(OEM_FILE)
    t0_utc = times_utc[0]
    day_offsets = np.array([(t - t0_utc).total_seconds() / 86400.0 for t in times_utc])
    burns = afm.detect_burns(times_utc, velocities, t0_utc)

    t_seg_start = times_utc[0]
    t_seg_end = times_utc[-1]
    times_sec = np.array([(t - t_seg_start).total_seconds() for t in times_utc])

    r0, v0 = positions[0], velocities[0]
    rf, vf = positions[-1], velocities[-1]

    margin = 3600.0
    cache = afm.EphemerisCache(
        t_seg_start - timedelta(seconds=margin),
        t_seg_end + timedelta(seconds=margin),
        n_points=5000,
    )
    t_offset = margin
    times_sec_cache = times_sec + t_offset
    t_span_cache = [t_offset, times_sec[-1] + t_offset]

    stats_out: dict = {}
    with timed_solve() as timer:
        result = afm.solve_ipopt(
            r0, v0, rf, vf, t_span_cache, cache,
            n_seg=120,
            nasa_data=(times_sec_cache, positions, velocities),
            burns=burns,
            stats_out=stats_out,
        )

    converged = bool(stats_out.get("success", False))
    iter_count = stats_out.get("iter_count")
    return_status = stats_out.get("return_status", "")
    conv_hist = stats_out.get("convergence_history") or []

    if result is not None:
        X_sol, U_sol, seg_times, J_val = result
        cost = float(J_val)
        rf_err = float(np.linalg.norm(X_sol[0:3, -1] - rf))
        n_seg = len(seg_times) - 1
    else:
        cost = float("nan")
        rf_err = float("inf")
        n_seg = 120

    final_constr = 0.0
    if conv_hist:
        final_constr = float(conv_hist[-1].get("constr_viol") or 0.0)

    ns, nd = 6, 3
    n_vars = ns * (n_seg + 1) + nd * n_seg
    n_constraints = 12 + 6 * n_seg  # BC + defects (waypoints not counted here)

    params = {
        "n_segments_nominal": 120,
        "n_segments_actual": int(n_seg),
        "control_parameterization": "piecewise_constant",
        "non_uniform_mesh": True,
        "burn_segment_dt_s": 120.0,
        "max_iter": 5000,
        "tol": 1e-6,
        "acceptable_tol": 1e-4,
        "linear_solver": "mumps",
        "warm_start_source": "NASA OEM",
        "waypoint_constraints": True,
        "burns_detected": [
            {"name": b["name"], "day_start": b["day_start"], "day_end": b["day_end"],
             "total_dv_km_s": float(b["total_dv"])}
            for b in burns
        ],
    }

    record = ResultRecord(
        phase=PHASE,
        case=CASE_FULL,
        method="multi_shooting_ipopt",
        parameters=params,
        cost=cost if np.isfinite(cost) else 1e30,
        converged=converged,
        residual=float(final_constr if np.isfinite(final_constr) else 1e30),
        wall_time_s=float(timer.wall_time_s),
        iterations=int(iter_count) if iter_count is not None else None,
        nfev=None,
        njev=None,
        n_vars=int(n_vars),
        n_constraints=int(n_constraints),
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        convergence_history=conv_hist if conv_hist else None,
        notes=(
            f"Full OEM arc (day 0 to 8.9) including TLI burn. "
            f"return_status={return_status}. Final position error: {rf_err:.3e} km. "
            f"Relaxed tolerances (tol=1e-6, acceptable_tol=1e-4) per RETUNING_AUDIT §3."
        ),
    )
    record.validate()
    append_to_summary(record)

    print(f"\n  IPOPT wall:   {timer.wall_time_s:.2f} s")
    print(f"  iter_count:   {iter_count}   return_status: {return_status}")
    print(f"  cost J:       {cost:.6e}")
    print(f"  final inf_pr: {final_constr:.3e}")

    return record, conv_hist


# =============================================================================
# T3.4 — Convergence-history plot
# =============================================================================

def plot_convergence_history(histories: dict, out_path: Path):
    """histories: {label: [ {iter, obj, constr_viol, dual_inf}, ... ]}"""
    fig, (ax_obj, ax_viol) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor='white')

    colors = {"artemis2_post_tli": "#2ca02c", "artemis2_full_mission": "#d62728"}

    for label, hist in histories.items():
        if not hist:
            print(f"  [plot] skipping {label}: empty history")
            continue
        iters = [h["iter"] for h in hist]
        obj = [max(abs(h["obj"]), 1e-30) for h in hist]
        vio = [max(abs(h.get("constr_viol") or 1e-30), 1e-30) for h in hist]

        c = colors.get(label, None)
        ax_obj.semilogy(iters, obj, label=label, linewidth=1.8, color=c)
        ax_viol.semilogy(iters, vio, label=label, linewidth=1.8, color=c)

    ax_obj.set_ylabel("|objective|  (∫ ‖u‖² dt, km²/s³)")
    ax_obj.set_title("IPOPT convergence history — Artemis II (Phase 2)")
    ax_obj.grid(True, which="both", alpha=0.3)
    ax_obj.legend(loc="best", facecolor='white', edgecolor='black', labelcolor='black')

    ax_viol.set_xlabel("IPOPT iteration")
    ax_viol.set_ylabel("constraint violation  (inf_pr, km or km/s)")
    ax_viol.grid(True, which="both", alpha=0.3)
    ax_viol.legend(loc="best", facecolor='white', edgecolor='black', labelcolor='black')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor='white', edgecolor='white')
    plt.close(fig)
    print(f"  Saved {out_path}")


# =============================================================================
# Driver
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Also run the Full-mission IPOPT (expensive, stretch goal).")
    parser.add_argument("--skip-shooting", action="store_true",
                        help="Skip the 15-seed Post-TLI shooting sweep.")
    parser.add_argument("--skip-ipopt", action="store_true",
                        help="Skip the Post-TLI IPOPT solve.")
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 2 instrumentation — Artemis II")
    print(f"  OEM file: {OEM_FILE}")
    print(f"  timestamp: {_now_iso_utc()}")
    print("=" * 72)

    post_tli_ctx = setup_post_tli()

    sol_sh = None
    conv_post = []

    if not args.skip_shooting:
        try:
            _, _, sol_sh, _ = run_shooting_sweep(post_tli_ctx)
        except Exception as e:
            print(f"\n  SHOOTING SWEEP FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    if not args.skip_ipopt:
        try:
            _, conv_post = run_ipopt_post_tli(post_tli_ctx, sol_shooting=sol_sh)
        except Exception as e:
            print(f"\n  POST-TLI IPOPT FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    conv_full = []
    if args.full:
        try:
            _, conv_full = run_full_mission()
        except Exception as e:
            print(f"\n  FULL MISSION FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    # ----- Convergence plot -----
    histories = {}
    if conv_post:
        histories["artemis2_post_tli"] = conv_post
    if conv_full:
        histories["artemis2_full_mission"] = conv_full

    out_path = _HERE / "convergence_history.png"
    if histories:
        plot_convergence_history(histories, out_path)
    else:
        print("  (no IPOPT convergence history captured — plot skipped)")

    print("\n" + "=" * 72)
    print("Phase 2 instrumentation complete")
    print("=" * 72)


if __name__ == "__main__":
    main()
