#!/usr/bin/env python3
"""
run_phase2_ipopt_psweep.py - Phase 2 Bezier/IPOPT polynomial-degree sweep.

Default run:
    python3 run_phase2_ipopt_psweep.py

Useful quick checks:
    python3 run_phase2_ipopt_psweep.py --dry-run
    python3 run_phase2_ipopt_psweep.py --case post_tli --degree 3 --N 4 --max-iter 5 --no-record --no-report
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import io
import sys
import traceback
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_ARTEMIS_DIR = _HERE.parent
_PROJECT_ROOT = _ARTEMIS_DIR.parent

for p in (_PROJECT_ROOT, _ARTEMIS_DIR, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import ResultRecord, append_to_summary, git_sha_or_none, timed_solve  # noqa: E402
from bezier_ipopt_3d import (  # noqa: E402
    Artemis3DBezierIPOPT,
    build_burn_aware_seg_times,
    build_control_bounds,
    trajectory_residuals_vs_oem,
)


# Importing the existing Artemis scripts prints banners; keep this runner's log
# focused on the sweep itself.
_saved_stdout = sys.stdout
sys.stdout = io.StringIO()
import artemis2_full_mission as afm  # noqa: E402
sys.stdout = _saved_stdout


OEM_FILE = _PROJECT_ROOT / "Artemis_II_OEM_2026_04_10_Post-ICPS-Sep-to-EI.asc"
PHASE = "2"
CASE_FULL = "artemis2_full_mission"
CASE_POST_TLI = "artemis2_post_tli"

POST_TLI_SEG_START_DAY = 1.0
POST_TLI_SEG_END_DAY = 8.5


@dataclass
class Phase2Case:
    case: str
    r0: np.ndarray
    v0: np.ndarray
    rf: np.ndarray
    vf: np.ndarray
    t_span: list[float]
    ephem_cache: object
    nasa_t: np.ndarray
    nasa_pos: np.ndarray
    nasa_vel: np.ndarray
    burns: list[dict]
    duration_days: float
    segment_days: list[float]

    @property
    def x0(self) -> np.ndarray:
        return np.concatenate([self.r0, self.v0])

    @property
    def xf(self) -> np.ndarray:
        return np.concatenate([self.rf, self.vf])

    @property
    def nasa_data(self):
        return self.nasa_t, self.nasa_pos, self.nasa_vel


def _now_iso_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def setup_case(case: str, ephem_points: int) -> Phase2Case:
    times_utc, positions, velocities = afm.parse_oem(str(OEM_FILE))
    mission_t0_utc = times_utc[0]
    day_offsets = np.array([(t - mission_t0_utc).total_seconds() / 86400.0 for t in times_utc])

    if case == CASE_FULL:
        mask = np.ones(len(times_utc), dtype=bool)
        burns = afm.detect_burns(times_utc, velocities, mission_t0_utc)
        segment_days = [0.0, float(day_offsets[-1])]
    elif case == CASE_POST_TLI:
        mask = (day_offsets >= POST_TLI_SEG_START_DAY) & (day_offsets <= POST_TLI_SEG_END_DAY)
        burns = []
        segment_days = [POST_TLI_SEG_START_DAY, POST_TLI_SEG_END_DAY]
    else:
        raise ValueError(f"unknown case {case!r}")

    seg_times_utc = [t for t, keep in zip(times_utc, mask) if keep]
    seg_pos = positions[mask]
    seg_vel = velocities[mask]
    if len(seg_times_utc) < 2:
        raise RuntimeError(f"{case}: not enough OEM points after segment selection")

    t_seg_start = seg_times_utc[0]
    t_seg_end = seg_times_utc[-1]
    times_sec = np.array([(t - t_seg_start).total_seconds() for t in seg_times_utc])

    margin = 3600.0
    ephem_cache = afm.EphemerisCache(
        t_seg_start - timedelta(seconds=margin),
        t_seg_end + timedelta(seconds=margin),
        n_points=int(ephem_points),
    )

    nasa_t = times_sec + margin
    t_span = [margin, float(times_sec[-1] + margin)]

    duration_days = float(times_sec[-1] / 86400.0)
    print(f"\n[{case}] OEM points: {len(seg_times_utc)}, duration: {duration_days:.2f} d")
    print(f"[{case}] segment days: {segment_days[0]:.2f} to {segment_days[1]:.2f}")
    print(f"[{case}] r0 norm = {np.linalg.norm(seg_pos[0]):.0f} km, "
          f"rf norm = {np.linalg.norm(seg_pos[-1]):.0f} km")

    return Phase2Case(
        case=case,
        r0=seg_pos[0],
        v0=seg_vel[0],
        rf=seg_pos[-1],
        vf=seg_vel[-1],
        t_span=t_span,
        ephem_cache=ephem_cache,
        nasa_t=nasa_t,
        nasa_pos=seg_pos,
        nasa_vel=seg_vel,
        burns=burns,
        duration_days=duration_days,
        segment_days=segment_days,
    )


def make_segment_times(ctx: Phase2Case, nominal_n_seg: int, burn_dt_s: float) -> np.ndarray:
    if ctx.case == CASE_FULL:
        return build_burn_aware_seg_times(
            ctx.t_span,
            nominal_n_seg=nominal_n_seg,
            burns=ctx.burns,
            burn_dt=burn_dt_s,
        )
    return np.linspace(ctx.t_span[0], ctx.t_span[1], int(nominal_n_seg) + 1)


def _final_solver_residual(stats: dict) -> float:
    it_log = (stats or {}).get("iterations") or {}
    if isinstance(it_log, dict):
        inf_pr = it_log.get("inf_pr") or []
        if inf_pr:
            try:
                return float(inf_pr[-1])
            except (TypeError, ValueError):
                pass
    return 0.0


def _iteration_count(stats: dict):
    value = (stats or {}).get("iter_count")
    return int(value) if value is not None else None


def _convergence_history(stats: dict):
    it_log = (stats or {}).get("iterations") or {}
    if not isinstance(it_log, dict) or not it_log.get("obj"):
        return None

    obj_hist = it_log.get("obj") or []
    pr_hist = it_log.get("inf_pr") or []
    du_hist = it_log.get("inf_du") or []
    hist = []
    for k in range(len(obj_hist)):
        hist.append({
            "iter": int(k),
            "obj": float(obj_hist[k]),
            "constr_viol": float(pr_hist[k]) if k < len(pr_hist) else None,
            "dual_inf": float(du_hist[k]) if k < len(du_hist) else None,
        })
    return hist


def run_one(
    ctx: Phase2Case,
    nominal_n_seg: int,
    degree: int,
    max_iter: int,
    tol: float,
    acceptable_tol: float,
    print_level: int,
    burn_dt_s: float,
    record: bool,
) -> dict:
    n_colloc = degree + 1
    seg_times = make_segment_times(ctx, nominal_n_seg, burn_dt_s)
    u_bounds = build_control_bounds(seg_times, burns=ctx.burns)

    solver = Artemis3DBezierIPOPT(
        seg_times=seg_times,
        ephem_cache=ctx.ephem_cache,
        bezier_degree=degree,
        n_collocation=n_colloc,
        u_bounds=u_bounds,
        burns=ctx.burns,
    )

    print(f"\n--- {ctx.case}: degree={degree}, nominal_N={nominal_n_seg}, "
          f"actual_N={solver.n_seg}, n_colloc={n_colloc} ---", flush=True)
    print(f"    Variables: {solver.n_seg * ((degree + 1) * 6 + n_colloc * 3)} approx")
    print(f"    Control bound range: {u_bounds.min():.3e} to {u_bounds.max():.3e} km/s^2")

    try:
        with timed_solve() as solve_timer:
            result = solver.solve(
                ctx.x0,
                ctx.xf,
                nasa_data=ctx.nasa_data,
                max_iter=max_iter,
                tol=tol,
                acceptable_tol=acceptable_tol,
                print_level=print_level,
            )
        stats = result.get("stats", {}) or {}
        residuals = trajectory_residuals_vs_oem(result, ctx.nasa_data)
        final_constr = _final_solver_residual(stats)
        converged = bool(result.get("success", False))
        status = stats.get("return_status")
        cost = float(result["cost"])
        wall_time = float(result.get("solve_time", solve_timer.wall_time_s))
        iterations = _iteration_count(stats)
        n_vars = int(result["n_vars"])
        n_constraints = int(result["n_constraints"])
        waypoint_stats = result["waypoint_stats"]
        conv_hist = _convergence_history(stats)
        notes = (
            f"3D ephemeris Bezier/IPOPT p-sweep. return_status={status}. "
            f"OEM max position residual={residuals['max_pos_km']:.3e} km, "
            f"max velocity residual={residuals['max_vel_km_s']:.3e} km/s. "
            f"Waypoints={waypoint_stats.count} "
            f"({waypoint_stats.full_state_count} full-state, "
            f"{waypoint_stats.position_only_count} position-only)."
        )
    except Exception as exc:
        traceback.print_exc()
        residuals = {
            "max_pos_km": 1e30,
            "rms_pos_km": 1e30,
            "max_vel_km_s": 1e30,
            "rms_vel_km_s": 1e30,
            "endpoint_pos_km": 1e30,
            "endpoint_vel_km_s": 1e30,
        }
        final_constr = 1e30
        converged = False
        status = f"EXCEPTION: {type(exc).__name__}"
        cost = 1e30
        wall_time = 0.0
        iterations = None
        n_vars = int(solver.n_seg * ((degree + 1) * 6 + n_colloc * 3))
        n_constraints = int(12 + 6 * (solver.n_seg - 1) + 6 * solver.n_seg * n_colloc)
        conv_hist = None
        notes = f"3D ephemeris Bezier/IPOPT p-sweep failed before producing a solution: {exc!r}"

    entry = {
        "case": ctx.case,
        "degree": int(degree),
        "nominal_N": int(nominal_n_seg),
        "actual_N": int(solver.n_seg),
        "n_collocation": int(n_colloc),
        "success": bool(converged),
        "cost": float(cost),
        "wall_time_s": float(wall_time),
        "iterations": iterations,
        "residual": float(final_constr),
        "status": status,
        "n_vars": int(n_vars),
        "n_constraints": int(n_constraints),
        **residuals,
    }

    if record:
        params = {
            "N_segments_nominal": int(nominal_n_seg),
            "N_segments_actual": int(solver.n_seg),
            "degree": int(degree),
            "n_collocation": int(n_colloc),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "acceptable_tol": float(acceptable_tol),
            "linear_solver": "mumps",
            "warm_start": True,
            "warm_start_source": "NASA OEM",
            "segment_days": ctx.segment_days,
            "non_uniform_mesh": bool(ctx.case == CASE_FULL),
            "burn_segment_dt_s": float(burn_dt_s) if ctx.case == CASE_FULL else None,
            "waypoint_constraints": True,
            "control_bounds": {
                "coast_bound_km_s2": float(np.min(u_bounds)),
                "max_bound_km_s2": float(np.max(u_bounds)),
            },
        }
        rec = ResultRecord(
            phase=PHASE,
            case=ctx.case,
            method="segmented_bezier_ipopt",
            parameters=params,
            cost=float(cost),
            converged=bool(converged),
            residual=float(final_constr),
            wall_time_s=float(wall_time),
            iterations=iterations,
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
        rec.validate()
        append_to_summary(rec)

    flag = "OK" if converged else "FAIL"
    print(f"    [{flag}] cost={cost:.6e}, constr={final_constr:.2e}, "
          f"OEM max pos={residuals['max_pos_km']:.3e} km, "
          f"iters={iterations}, wall={wall_time:.2f}s, status={status}")

    return entry


def dry_run(ctx: Phase2Case, nominal_ns: list[int], degrees: list[int], burn_dt_s: float):
    print("\nDry run only: no NLPs will be solved.")
    for nominal_n_seg in nominal_ns:
        seg_times = make_segment_times(ctx, nominal_n_seg, burn_dt_s)
        actual_n = len(seg_times) - 1
        for degree in degrees:
            n_colloc = degree + 1
            n_vars = actual_n * ((degree + 1) * 6 + n_colloc * 3)
            n_constraints = 12 + 6 * (actual_n - 1) + 6 * actual_n * n_colloc
            print(f"  {ctx.case:22s} degree={degree} nominal_N={nominal_n_seg} "
                  f"actual_N={actual_n} vars~{n_vars} constraints~{n_constraints}")


def write_validation_report(results: list[dict], out_path: Path):
    lines = [
        "# Phase 2 Bezier/IPOPT p-sweep results",
        "",
        f"Generated: {_now_iso_utc()}",
        "",
        "| case | degree | nominal N | actual N | converged | cost | IPOPT residual | max OEM pos err (km) | wall (s) | status |",
        "|---|---:|---:|---:|:---:|---:|---:|---:|---:|---|",
    ]
    for e in results:
        lines.append(
            f"| {e['case']} | {e['degree']} | {e['nominal_N']} | {e['actual_N']} "
            f"| {'Y' if e['success'] else 'N'} | {e['cost']:.6e} "
            f"| {e['residual']:.3e} | {e['max_pos_km']:.3e} "
            f"| {e['wall_time_s']:.2f} | {e['status']} |"
        )
    lines.extend([
        "",
        "Notes:",
        "- Cost is the Gauss-Legendre quadrature of squared control acceleration.",
        "- IPOPT residual is the last reported primal infeasibility when available.",
        "- OEM residual is measured on the evaluated Bezier trajectory, not only at pinned waypoints.",
    ])
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved validation report: {out_path}")


def plot_results(results: list[dict], out_path: Path):
    usable = [r for r in results if np.isfinite(r["cost"]) and r["cost"] < 1e20]
    if not usable:
        print("No finite p-sweep results to plot.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor="white")

    for case in sorted({r["case"] for r in usable}):
        sub = sorted([r for r in usable if r["case"] == case], key=lambda e: (e["nominal_N"], e["degree"]))
        for nominal_n in sorted({r["nominal_N"] for r in sub}):
            ss = [r for r in sub if r["nominal_N"] == nominal_n]
            label = f"{case}, N={nominal_n}"
            deg = [r["degree"] for r in ss]
            axes[0].semilogy(deg, [max(r["cost"], 1e-30) for r in ss], "o-", label=label)
            axes[1].plot(deg, [r["wall_time_s"] for r in ss], "o-", label=label)
            axes[2].semilogy(deg, [max(r["max_pos_km"], 1e-30) for r in ss], "o-", label=label)

    axes[0].set_title("Control cost")
    axes[0].set_xlabel("Bezier degree")
    axes[0].set_ylabel("J")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].set_title("Wall time")
    axes[1].set_xlabel("Bezier degree")
    axes[1].set_ylabel("seconds")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("OEM residual")
    axes[2].set_xlabel("Bezier degree")
    axes[2].set_ylabel("max position error (km)")
    axes[2].grid(True, which="both", alpha=0.3)

    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved p-sweep figure: {out_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["full", "post_tli", "both"], default="full",
                        help="Mission case to run. Default is the full mission.")
    parser.add_argument("--degree", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    parser.add_argument("--N", type=int, nargs="+", default=[120],
                        help="Nominal segment counts. Full mission uses burn-aware densification.")
    parser.add_argument("--ephem-points", type=int, default=None,
                        help="Ephemeris cache points. Defaults: 10000 full, 4000 post-TLI.")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--acceptable-tol", type=float, default=1e-4)
    parser.add_argument("--print-level", type=int, default=0)
    parser.add_argument("--burn-dt-s", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-record", action="store_true",
                        help="Do not append ResultRecord entries to results_summary.json.")
    parser.add_argument("--no-report", action="store_true",
                        help="Do not write phase2_bezier_validation.md.")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    case_names = []
    if args.case in ("full", "both"):
        case_names.append(CASE_FULL)
    if args.case in ("post_tli", "both"):
        case_names.append(CASE_POST_TLI)

    print("=" * 78)
    print("Phase 2 Bezier/IPOPT p-sweep")
    print(f"  OEM file: {OEM_FILE}")
    print(f"  cases: {', '.join(case_names)}")
    print(f"  degrees: {args.degree}")
    print(f"  nominal N values: {args.N}")
    print("=" * 78)

    all_results = []
    for case_name in case_names:
        default_ephem_points = 10000 if case_name == CASE_FULL else 4000
        ephem_points = int(args.ephem_points or default_ephem_points)
        ctx = setup_case(case_name, ephem_points=ephem_points)

        if args.dry_run:
            dry_run(ctx, args.N, args.degree, args.burn_dt_s)
            continue

        for degree in args.degree:
            for nominal_n_seg in args.N:
                entry = run_one(
                    ctx,
                    nominal_n_seg=nominal_n_seg,
                    degree=degree,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    acceptable_tol=args.acceptable_tol,
                    print_level=args.print_level,
                    burn_dt_s=args.burn_dt_s,
                    record=not args.no_record,
                )
                all_results.append(entry)

    if not args.dry_run and all_results and not args.no_report:
        report_path = _ARTEMIS_DIR / "phase2_bezier_validation.md"
        write_validation_report(all_results, report_path)

    if not args.dry_run and all_results and not args.no_plot:
        figure_path = _ARTEMIS_DIR / "phase0_nsweep_figures" / "phase2_ipopt_psweep.png"
        plot_results(all_results, figure_path)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
