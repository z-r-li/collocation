"""
run_constraint_remediation_cases.py

Setup/driver for the constraint-remediation runs in CONSTRAINT_REMEDIATION_PLAN.

What it does when executed:
  - computes P0/P1 unconstrained control diagnostics,
  - runs constrained P0 and P1 cases with optional ||u|| <= u_max bounds,
  - appends new constrained records to results_summary.json,
  - writes the saturation/control-envelope figures.

Phase 1 is intentionally held at degree 7. The constrained SLSQP sweep changes
only the segment count, N in {8, 16}, per T7.3.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
EARTH_MARS_DIR = PROJECT_ROOT / "Earth-Mars"
PLANER_DIR = PROJECT_ROOT / "Planer"

for _path in (PROJECT_ROOT, EARTH_MARS_DIR, PLANER_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import ResultRecord, append_to_summary, git_sha_or_none, load_results, timed_solve

import run_phase0 as p0
from dynamics import saturated_min_energy_control, two_body_state_costate_ode
from ipopt_collocation_2body import TwoBodyBezierIPOPT
from shooting import propagate, solve_min_energy

from cr3bp_planar import MU as P1_MU, saturated_min_energy_control as cr3bp_sat_control
from cr3bp_transfer import setup_transfer_problem, solve_shooting
from ipopt_collocation import CR3BPBezierIPOPT
from bezier_segmented import run_n_sweep


# Dimensional anchors for diagnostic reporting.
AU_KM = 149_597_870.7
SUN_MU_KM3_S2 = 132_712_440_018.0
P0_ACCEL_UNIT_KM_S2 = SUN_MU_KM3_S2 / AU_KM**2

EARTH_MOON_DISTANCE_KM = 384_400.0
EARTH_MOON_MU_KM3_S2 = 398_600.4418 + 4_902.800066
P1_ACCEL_UNIT_KM_S2 = EARTH_MOON_MU_KM3_S2 / EARTH_MOON_DISTANCE_KM**2

# Default constrained-case bounds. They are command-line overrideable because
# the final values should be locked after the T7.1 diagnostic is reviewed.
# P0 defaults to a non-binding electric-envelope value. P1 defaults to a
# binding demonstration value so the saturation-arc figure is meaningful.
P0_U_MAX_DIM_KM_S2 = 1.0e-5
P1_U_MAX_DIM_KM_S2 = 8.0e-7
P0_U_MAX = P0_U_MAX_DIM_KM_S2 / P0_ACCEL_UNIT_KM_S2
P1_U_MAX = P1_U_MAX_DIM_KM_S2 / P1_ACCEL_UNIT_KM_S2

P0_CASE_CONSTRAINED = "earth_mars_2body_constrained"
P1_CASE_CONSTRAINED = "planar_cr3bp_L1_L2_lyapunov_constrained"


def _now_iso_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _active_fraction(u: np.ndarray, u_max: float, tol: float = 0.99) -> float:
    if u.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(u, axis=1) >= tol * float(u_max)))


def _stats_from_ipopt(sol: dict) -> tuple[float, list[dict] | None]:
    stats = sol.get("stats", {}) or {}
    iterations_log = stats.get("iterations") or {}
    constr_viol = None
    conv_hist = None
    if isinstance(iterations_log, dict):
        viol_list = iterations_log.get("inf_pr")
        if viol_list:
            constr_viol = float(viol_list[-1])
        obj_hist = iterations_log.get("obj") or []
        pr_hist = iterations_log.get("inf_pr") or []
        if obj_hist:
            conv_hist = [
                {
                    "iter": k,
                    "obj": float(obj_hist[k]),
                    "constr_viol": float(pr_hist[k]) if k < len(pr_hist) else None,
                }
                for k in range(len(obj_hist))
            ]
    return (0.0 if constr_viol is None else constr_viol), conv_hist


def _write_diag(path: Path, title: str, lines: list[str]) -> None:
    path.write_text("# " + title + "\n\n" + "\n".join(lines) + "\n")


def _plot_control_envelope(
    title: str,
    out_path: Path,
    u_max: float,
    accel_unit: float,
    direct: dict | None = None,
    indirect: dict | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
    ax.set_facecolor("white")

    source_for_shade = direct or indirect
    if source_for_shade is not None:
        t = source_for_shade["t"]
        u_mag = np.linalg.norm(source_for_shade["u"], axis=1)
        active = u_mag >= 0.99 * u_max
        ax.fill_between(
            t, 0.0, u_max * accel_unit,
            where=active,
            color="#f2c14e",
            alpha=0.25,
            step=None,
            label="active envelope",
        )

    if direct is not None:
        ax.plot(
            direct["t"],
            np.linalg.norm(direct["u"], axis=1) * accel_unit,
            color="#d62728",
            lw=2.0,
            label="direct collocation",
        )
    if indirect is not None:
        ax.plot(
            indirect["t"],
            np.linalg.norm(indirect["u"], axis=1) * accel_unit,
            color="#1f77b4",
            lw=1.8,
            ls="--",
            label="indirect saturated PMP",
        )
        if "lam_v" in indirect:
            ax.plot(
                indirect["t"],
                0.5 * np.linalg.norm(indirect["lam_v"], axis=1) * accel_unit,
                color="#2ca02c",
                lw=1.2,
                ls=":",
                label=r"$0.5\|\lambda_v\|$",
            )

    ax.axhline(
        u_max * accel_unit,
        color="black",
        lw=1.3,
        ls="-.",
        label=r"$u_{\max}$",
    )
    ax.set_title(title, fontsize=12, fontweight="bold", color="black")
    ax.set_xlabel("time (canonical units)", color="black")
    ax.set_ylabel(r"control magnitude (km/s$^2$)", color="black")
    ax.grid(True, alpha=0.25, color="gray")
    ax.legend(framealpha=0.92, facecolor="white", edgecolor="black")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_cost_comparison(new_records: list[ResultRecord]) -> None:
    existing = load_results()
    rows = []

    def add_from_summary(label: str, phase: str, case: str, method: str) -> None:
        matches = [
            r for r in existing
            if r.phase == phase and r.case == case and r.method == method
        ]
        if matches:
            rows.append((label, matches[-1].cost))

    add_from_summary("P0 direct unbounded", "0", "earth_mars_2body", "global_bezier_ipopt")
    add_from_summary("P0 PMP unbounded", "0", "earth_mars_2body", "indirect_shooting")
    add_from_summary("P1 direct unbounded", "1", "planar_cr3bp_L1_L2_lyapunov", "global_bezier_ipopt")
    add_from_summary("P1 PMP unbounded", "1", "planar_cr3bp_L1_L2_lyapunov", "indirect_shooting")

    for rec in new_records:
        if rec.case == P0_CASE_CONSTRAINED:
            label = "P0 direct bounded" if "ipopt" in rec.method else "P0 PMP bounded"
            rows.append((label, rec.cost))
        if rec.case == P1_CASE_CONSTRAINED and rec.method in (
            "global_bezier_ipopt_bounded",
            "indirect_shooting_saturated",
        ):
            label = "P1 direct bounded" if "ipopt" in rec.method else "P1 PMP bounded"
            rows.append((label, rec.cost))

    if not rows:
        return

    labels, costs = zip(*rows)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor="white")
    colors = ["#4e79a7" if "PMP" in label else "#d62728" for label in labels]
    ax.bar(x, costs, color=colors, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"$J = \int \|u\|^2 dt$")
    ax.set_title("Constrained vs. Unconstrained Cost Comparison", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "constrained_cost_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_p0() -> tuple[list[ResultRecord], dict, dict]:
    records: list[ResultRecord] = []

    lam0_unc, _ = solve_min_energy(
        p0.R0, p0.V0, p0.POS_MARS_F, p0.VEL_MARS_F, p0.T0, p0.TF,
        lam0_guess=np.zeros(4), mu=p0.MU,
    )
    sol_unc = propagate(
        two_body_state_costate_ode,
        np.concatenate([p0.R0, p0.V0, lam0_unc]),
        [p0.T0, p0.TF],
        n_steps=4000,
        mu=p0.MU,
    )
    u_unc = -0.5 * sol_unc.y[6:8, :].T
    peak_unc = float(np.max(np.linalg.norm(u_unc, axis=1)))
    _write_diag(
        EARTH_MARS_DIR / "u_diagnostic.md",
        "Phase 0 Control-Magnitude Diagnostic",
        [
            f"- Unconstrained peak ||u|| = {peak_unc:.6e} canonical = "
            f"{peak_unc * P0_ACCEL_UNIT_KM_S2:.6e} km/s^2.",
            f"- Chosen constrained-case u_max = {P0_U_MAX:.6e} canonical = "
            f"{P0_U_MAX_DIM_KM_S2:.6e} km/s^2.",
            "- This bound is intentionally non-binding for P0; it documents that the "
            "reference Earth-Mars analog sits below electric-propulsion envelopes.",
        ],
    )

    with timed_solve() as timer:
        lam0, info = solve_min_energy(
            p0.R0, p0.V0, p0.POS_MARS_F, p0.VEL_MARS_F, p0.T0, p0.TF,
            lam0_guess=lam0_unc, mu=p0.MU, u_max=P0_U_MAX,
        )
    sol = propagate(
        two_body_state_costate_ode,
        np.concatenate([p0.R0, p0.V0, lam0]),
        [p0.T0, p0.TF],
        n_steps=4000,
        mu=p0.MU,
        u_max=P0_U_MAX,
    )
    lam_v = sol.y[6:8, :].T
    u = np.array([saturated_min_energy_control(row, P0_U_MAX) for row in lam_v])
    cost = float(np.trapezoid(np.sum(u * u, axis=1), sol.t))
    residual = float(np.linalg.norm(info["fvec"]))
    rec = ResultRecord(
        phase="0",
        case=P0_CASE_CONSTRAINED,
        method="indirect_shooting_saturated",
        parameters={
            "u_max": P0_U_MAX,
            "u_max_km_s2": P0_U_MAX_DIM_KM_S2,
            "tf": p0.TF,
            "rtol": 1e-12,
            "atol": 1e-12,
            "n_propagation_steps": 4000,
        },
        cost=cost,
        converged=residual < 1e-6,
        residual=residual,
        wall_time_s=float(timer.wall_time_s),
        iterations=None,
        nfev=int(info.get("nfev", 0)) if "nfev" in info else None,
        njev=None,
        n_vars=4,
        n_constraints=4,
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        notes="P0 bounded-control PMP variant with saturated min-energy law.",
    )
    append_to_summary(rec)
    records.append(rec)
    indirect = {
        "t": sol.t,
        "u": u,
        "lam_v": lam_v,
        "r": sol.y[0:2, :].T,
        "v": sol.y[2:4, :].T,
    }

    solver = TwoBodyBezierIPOPT(mu=p0.MU, n_segments=16, bezier_degree=7, n_collocation=12)
    warm_state = np.column_stack([sol.y[0:2, :].T, sol.y[2:4, :].T])
    with timed_solve() as timer:
        direct_sol = solver.solve(
            p0.X0_FULL, p0.XF_FULL, p0.T0, p0.TF,
            warm_traj=(sol.t, warm_state),
            max_iter=3000,
            tol=1e-10,
            print_level=0,
            u_max=P0_U_MAX,
        )
    constr_viol, conv_hist = _stats_from_ipopt(direct_sol)
    rec = ResultRecord(
        phase="0",
        case=P0_CASE_CONSTRAINED,
        method="global_bezier_ipopt_bounded",
        parameters={
            "N_segments": 16,
            "degree": 7,
            "n_collocation": 12,
            "max_iter": 3000,
            "tol": 1e-10,
            "linear_solver": "mumps",
            "warm_start": True,
            "warm_start_source": "bounded_indirect_shooting",
            "u_max": P0_U_MAX,
            "u_max_km_s2": P0_U_MAX_DIM_KM_S2,
            "tf": p0.TF,
        },
        cost=float(direct_sol["cost"]),
        converged=bool(direct_sol.get("success", False)),
        residual=float(constr_viol),
        wall_time_s=float(timer.wall_time_s),
        iterations=int(direct_sol["stats"].get("iter_count"))
        if direct_sol["stats"].get("iter_count") is not None else None,
        nfev=None,
        njev=None,
        n_vars=16 * ((7 + 1) * 4 + 12 * 2),
        n_constraints=4 + 4 + 4 * (16 - 1) + 4 * 16 * 12 + 16 * 12,
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        convergence_history=conv_hist,
        notes=(
            "P0 direct constrained variant. Inequality count includes one "
            "||u|| <= u_max path bound per collocation node. "
            f"Active fraction on evaluated path: {_active_fraction(direct_sol['u'], P0_U_MAX):.3f}."
        ),
    )
    append_to_summary(rec)
    records.append(rec)
    direct = {"t": direct_sol["t"], "u": direct_sol["u"], "r": direct_sol["r"], "v": direct_sol["v"]}
    return records, indirect, direct


def run_p1(skip_slsqp: bool = False) -> tuple[list[ResultRecord], dict | None, dict | None]:
    records: list[ResultRecord] = []
    x0, xf, t0, tf, _lyap = setup_transfer_problem(Ax_L1=0.02, Ax_L2=0.02)

    lam_unc, sol_unc, _ = solve_shooting(
        x0, xf, t0, tf, P1_MU,
        n_random=3,
        select_min_cost=False,
        fsolve_maxfev=600,
    )
    u_unc = -0.5 * sol_unc.y[6:8, :].T
    peak_unc = float(np.max(np.linalg.norm(u_unc, axis=1)))
    _write_diag(
        PLANER_DIR / "u_diagnostic.md",
        "Phase 1 Control-Magnitude Diagnostic",
        [
            f"- Unconstrained peak ||u|| = {peak_unc:.6e} canonical = "
            f"{peak_unc * P1_ACCEL_UNIT_KM_S2:.6e} km/s^2.",
            f"- Chosen constrained-case u_max = {P1_U_MAX:.6e} canonical = "
            f"{P1_U_MAX_DIM_KM_S2:.6e} km/s^2.",
            "- Phase 1 constrained runs keep degree=7 and use N={8,16} for "
            "the SLSQP segment-count sweep.",
        ],
    )

    indirect = None
    direct = None

    with timed_solve() as timer:
        lam, sol, info = solve_shooting(
            x0, xf, t0, tf, P1_MU,
            lam0_guess=lam_unc,
            n_random=3,
            u_max=P1_U_MAX,
            select_min_cost=True,
            fsolve_maxfev=600,
        )
    lam_v = sol.y[6:8, :].T
    u = np.array([cr3bp_sat_control(row, P1_U_MAX) for row in lam_v])
    cost = float(np.trapezoid(np.sum(u * u, axis=1), sol.t))
    residual = float(np.linalg.norm(info["fvec"]))
    rec = ResultRecord(
        phase="1",
        case=P1_CASE_CONSTRAINED,
        method="indirect_shooting_saturated",
        parameters={
            "tf": float(tf),
            "Ax_L1": 0.02,
            "Ax_L2": 0.02,
            "rtol": 1e-12,
            "atol": 1e-12,
            "fsolve_maxfev": 2000,
            "u_max": P1_U_MAX,
            "u_max_km_s2": P1_U_MAX_DIM_KM_S2,
            "select_min_cost_branch": False,
        },
        cost=cost,
        converged=residual < 1e-6,
        residual=residual,
        wall_time_s=float(timer.wall_time_s),
        iterations=None,
        nfev=int(info.get("nfev", 0)) if "nfev" in info else None,
        njev=None,
        n_vars=4,
        n_constraints=4,
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        notes=(
            "P1 bounded-control PMP variant with saturated min-energy law. "
            f"Active fraction on evaluated path: {_active_fraction(u, P1_U_MAX):.3f}."
        ),
    )
    append_to_summary(rec)
    records.append(rec)
    indirect = {
        "t": sol.t,
        "u": u,
        "lam_v": lam_v,
        "r": sol.y[0:2, :].T,
        "v": sol.y[2:4, :].T,
    }

    warm_state = np.column_stack([sol.y[0:2, :].T, sol.y[2:4, :].T])
    solver = CR3BPBezierIPOPT(mu=P1_MU, n_segments=16, bezier_degree=7, n_collocation=12)
    with timed_solve() as timer:
        direct_sol = solver.solve(
            x0, xf, t0, tf,
            warm_traj=(sol.t, warm_state),
            max_iter=3000,
            tol=1e-10,
            print_level=0,
            u_max=P1_U_MAX,
        )
    constr_viol, conv_hist = _stats_from_ipopt(direct_sol)
    rec = ResultRecord(
        phase="1",
        case=P1_CASE_CONSTRAINED,
        method="global_bezier_ipopt_bounded",
        parameters={
            "N_segments": 16,
            "degree": 7,
            "n_collocation": 12,
            "max_iter": 3000,
            "tol": 1e-10,
            "linear_solver": "mumps",
            "warm_start": True,
            "warm_start_source": "bounded_indirect_shooting",
            "u_max": P1_U_MAX,
            "u_max_km_s2": P1_U_MAX_DIM_KM_S2,
            "tf": float(tf),
            "Ax_L1": 0.02,
            "Ax_L2": 0.02,
        },
        cost=float(direct_sol["cost"]),
        converged=bool(direct_sol.get("success", False)),
        residual=float(constr_viol),
        wall_time_s=float(timer.wall_time_s),
        iterations=int(direct_sol["stats"].get("iter_count"))
        if direct_sol["stats"].get("iter_count") is not None else None,
        nfev=None,
        njev=None,
        n_vars=16 * ((7 + 1) * 4 + 12 * 2),
        n_constraints=4 + 4 + 4 * (16 - 1) + 4 * 16 * 12 + 16 * 12,
        git_sha=git_sha_or_none(),
        timestamp=_now_iso_utc(),
        python_version=_python_version(),
        convergence_history=conv_hist,
        notes=(
            "P1 direct constrained variant at degree=7. Inequality count "
            "includes one ||u|| <= u_max path bound per collocation node. "
            f"Active fraction on evaluated path: {_active_fraction(direct_sol['u'], P1_U_MAX):.3f}."
        ),
    )
    append_to_summary(rec)
    records.append(rec)
    direct = {"t": direct_sol["t"], "u": direct_sol["u"], "r": direct_sol["r"], "v": direct_sol["v"]}

    if not skip_slsqp:
        sweep = run_n_sweep(
            x0, xf, t0, tf,
            N_list=(8, 16),
            bezier_degree=7,
            n_collocation=8,
            warm_traj=(sol.t, warm_state),
            max_iter=300,
            ftol=1e-9,
            u_max=P1_U_MAX,
            verbose=True,
        )
        for entry in sweep:
            n_seg = int(entry["N"])
            r = entry.get("solver_result")
            residual = float(entry.get("max_defect", 0.0))
            if not np.isfinite(residual):
                residual = 1e30
            n_free_cp_vectors = n_seg * (7 - 1) + (n_seg - 1)
            n_vars = n_free_cp_vectors * 4 + n_seg * 8 * 2
            n_constraints = n_seg * 8 * 4 + n_seg * 8
            rec = ResultRecord(
                phase="1",
                case=P1_CASE_CONSTRAINED,
                method="segmented_bezier_slsqp_bounded",
                parameters={
                    "N_segments": n_seg,
                    "degree": 7,
                    "n_collocation": 8,
                    "max_iter": 300,
                    "ftol": 1e-9,
                    "warm_start_chain": "bounded shooting -> N=8 -> N=16",
                    "u_max": P1_U_MAX,
                    "u_max_km_s2": P1_U_MAX_DIM_KM_S2,
                    "tf": float(tf),
                    "Ax_L1": 0.02,
                    "Ax_L2": 0.02,
                },
                cost=float(entry["cost"]) if np.isfinite(entry["cost"]) else 1e30,
                converged=bool(entry.get("success", False) and residual < 1e-4),
                residual=residual,
                wall_time_s=float(entry["time_s"]),
                iterations=int(entry.get("nit", 0)),
                nfev=int(getattr(r, "nfev", 0)) if r is not None else None,
                njev=int(getattr(r, "njev", 0)) if r is not None else None,
                n_vars=int(n_vars),
                n_constraints=int(n_constraints),
                git_sha=git_sha_or_none(),
                timestamp=_now_iso_utc(),
                python_version=_python_version(),
                notes=(
                    "P1 constrained SLSQP segment-count case at degree=7. "
                    f"Active fraction on evaluated path: "
                    f"{_active_fraction(entry['u'], P1_U_MAX) if entry.get('u') is not None else 0.0:.3f}."
                ),
            )
            append_to_summary(rec)
            records.append(rec)

    return records, indirect, direct


def main(argv: list[str] | None = None) -> int:
    global P0_U_MAX_DIM_KM_S2, P1_U_MAX_DIM_KM_S2, P0_U_MAX, P1_U_MAX

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-slsqp", action="store_true",
                        help="skip the slower P1 constrained SLSQP N={8,16} sweep")
    parser.add_argument("--no-figures", action="store_true",
                        help="write records/diagnostics but skip figure generation")
    parser.add_argument("--p0-u-max-km-s2", type=float, default=P0_U_MAX_DIM_KM_S2,
                        help="Phase 0 dimensional acceleration bound")
    parser.add_argument("--p1-u-max-km-s2", type=float, default=P1_U_MAX_DIM_KM_S2,
                        help="Phase 1 dimensional acceleration bound")
    args = parser.parse_args(argv)

    P0_U_MAX_DIM_KM_S2 = float(args.p0_u_max_km_s2)
    P1_U_MAX_DIM_KM_S2 = float(args.p1_u_max_km_s2)
    if P0_U_MAX_DIM_KM_S2 <= 0.0 or P1_U_MAX_DIM_KM_S2 <= 0.0:
        raise ValueError("u_max values must be positive")
    P0_U_MAX = P0_U_MAX_DIM_KM_S2 / P0_ACCEL_UNIT_KM_S2
    P1_U_MAX = P1_U_MAX_DIM_KM_S2 / P1_ACCEL_UNIT_KM_S2

    new_records: list[ResultRecord] = []
    p0_records, p0_indirect, p0_direct = run_p0()
    new_records.extend(p0_records)
    p1_records, p1_indirect, p1_direct = run_p1(skip_slsqp=args.skip_slsqp)
    new_records.extend(p1_records)

    if not args.no_figures:
        _plot_control_envelope(
            "Phase 0 Control Profile Within Electric-Propulsion Envelope",
            EARTH_MARS_DIR / "p0_control_envelope.png",
            P0_U_MAX,
            P0_ACCEL_UNIT_KM_S2,
            direct=p0_direct,
            indirect=p0_indirect,
        )
        _plot_control_envelope(
            "Phase 1 Saturation Arcs, Degree 7",
            PLANER_DIR / "p1_saturation_arcs.png",
            P1_U_MAX,
            P1_ACCEL_UNIT_KM_S2,
            direct=p1_direct,
            indirect=p1_indirect,
        )
        _plot_cost_comparison(new_records)

    print("Constraint-remediation setup run complete.")
    print(f"Records written: {len(new_records)}")
    print(f"P0 u_max = {P0_U_MAX:.6e} canonical ({P0_U_MAX_DIM_KM_S2:.3e} km/s^2)")
    print(f"P1 u_max = {P1_U_MAX:.6e} canonical ({P1_U_MAX_DIM_KM_S2:.3e} km/s^2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
