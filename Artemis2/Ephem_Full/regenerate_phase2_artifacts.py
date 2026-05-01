#!/usr/bin/env python3
"""regenerate_phase2_artifacts.py

Regenerate the Phase 2 figure set from the locked Bezier+IPOPT transcription
(degree 7) so the slide deck no longer carries RK4-derived plots. Mirrors the
file names produced by the legacy artemis2_full_mission.py / artemis2_ephemeris.py
plotting routines so existing slide-rels keep working when these PNGs are
copied into the deck's media folder.

Static figures only (skips animations -- those remain on file from the legacy
run; the user can regenerate them later if needed).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
import numpy as np

_HERE = Path(__file__).resolve().parent
_ARTEMIS_DIR = _HERE.parent
_PROJECT_ROOT = _ARTEMIS_DIR.parent

for p in (_PROJECT_ROOT, _ARTEMIS_DIR, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Silence import-time banners from the legacy module
_saved_stdout = sys.stdout
sys.stdout = io.StringIO()
import run_phase2_ipopt_psweep as runner  # noqa: E402
import bezier_ipopt_3d as bez  # noqa: E402
sys.stdout = _saved_stdout

EPHEM_FULL = _ARTEMIS_DIR / "Ephem_Full"
EPHEM_POST_TLI = _ARTEMIS_DIR / "Ephem_Post_TLI"
CONVERGENCE = _ARTEMIS_DIR / "convergence_history.png"

EARTH_R = 6378.0  # km
MOON_R = 1737.0  # km
NAVY = "#1E2761"
GOLD = "#CFB991"
RED = "#A73131"


def solve_case(case: str, ephem_points: int) -> dict:
    ctx = runner.setup_case(case, ephem_points=ephem_points)
    seg_times = runner.make_segment_times(ctx, nominal_n_seg=120, burn_dt_s=120.0)
    u_bounds = bez.build_control_bounds(seg_times, burns=ctx.burns)
    solver = bez.Artemis3DBezierIPOPT(
        seg_times=seg_times,
        ephem_cache=ctx.ephem_cache,
        bezier_degree=7,
        n_collocation=8,
        u_bounds=u_bounds,
        burns=ctx.burns,
    )
    result = solver.solve(
        x0=ctx.x0,
        xf=ctx.xf,
        nasa_data=ctx.nasa_data,
        max_iter=5000,
        tol=1e-6,
        acceptable_tol=1e-4,
        print_level=0,
    )
    result["ctx"] = ctx
    result["seg_times"] = seg_times
    result["solver"] = solver
    return result


def _ballistic_propagation(ctx, t_eval):
    """Propagate from x0 with no thrust through the same dynamics. Cheap RK4."""
    cache = ctx.ephem_cache
    x = np.concatenate([ctx.r0, ctx.v0]).astype(float)
    out = np.zeros((len(t_eval), 6))
    out[0] = x
    for i in range(1, len(t_eval)):
        h = t_eval[i] - t_eval[i - 1]
        x = _rk4_step(x, t_eval[i - 1], h, cache)
        out[i] = x
    return out[:, 0:3], out[:, 3:6]


def _rk4_step(x, t, h, cache):
    def f(state, tt):
        r = state[0:3]
        rm, rs = cache.get_positions(tt)
        a = bez.artemis_gravity_numpy(r, rm, rs)
        return np.concatenate([state[3:6], a])

    k1 = f(x, t)
    k2 = f(x + 0.5 * h * k1, t + 0.5 * h)
    k3 = f(x + 0.5 * h * k2, t + 0.5 * h)
    k4 = f(x + h * k3, t + h)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _interp_oem(ctx, t_eval):
    pos = np.zeros((len(t_eval), 3))
    vel = np.zeros((len(t_eval), 3))
    for k in range(3):
        pos[:, k] = np.interp(t_eval, ctx.nasa_t, ctx.nasa_pos[:, k])
        vel[:, k] = np.interp(t_eval, ctx.nasa_t, ctx.nasa_vel[:, k])
    return pos, vel


def _draw_earth_moon(ax, ctx, sample_epochs=None, kind="3d"):
    # Earth at origin
    if kind == "3d":
        u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
        xs = EARTH_R * np.cos(u) * np.sin(v)
        ys = EARTH_R * np.sin(u) * np.sin(v)
        zs = EARTH_R * np.cos(v)
        ax.plot_surface(xs, ys, zs, color="#3C8DBC", alpha=0.45, linewidth=0)
    else:
        ax.scatter([0], [0], color="#3C8DBC", s=80, zorder=10, label="Earth")

    if sample_epochs is None:
        return
    for tt in sample_epochs:
        rm, _ = ctx.ephem_cache.get_positions(tt)
        if kind == "3d":
            ax.scatter(rm[0], rm[1], rm[2], color="gray", s=18, alpha=0.7)
        else:
            ax.scatter(rm[0], rm[1], color="gray", s=18, alpha=0.7)


def plot_3d_trajectory(result, *, title, out_path, prefix):
    ctx = result["ctx"]
    r = result["r"]
    t = result["t"]
    nasa_pos = ctx.nasa_pos

    sample_epochs = np.linspace(t[0], t[-1], 6)
    fig = plt.figure(figsize=(9, 7), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(nasa_pos[:, 0], nasa_pos[:, 1], nasa_pos[:, 2],
            color="#1F77B4", lw=2.0, label="NASA OEM")
    ax.plot(r[:, 0], r[:, 1], r[:, 2],
            color=NAVY, lw=1.8, label=f"{prefix} (Bézier+IPOPT, deg 7)")
    _draw_earth_moon(ax, ctx, sample_epochs=sample_epochs, kind="3d")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_2d_projections(result, *, title, out_path, prefix):
    ctx = result["ctx"]
    r = result["r"]
    nasa_pos = ctx.nasa_pos

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=130)
    pairs = [(0, 1, "X (km)", "Y (km)", "XY"),
             (0, 2, "X (km)", "Z (km)", "XZ"),
             (1, 2, "Y (km)", "Z (km)", "YZ")]
    for ax, (i, j, xl, yl, name) in zip(axes, pairs):
        ax.plot(nasa_pos[:, i], nasa_pos[:, j], color="#1F77B4", lw=1.8, label="NASA OEM")
        ax.plot(r[:, i], r[:, j], color=NAVY, lw=1.6,
                label=f"{prefix} (Bézier+IPOPT)")
        ax.scatter([0], [0], color="#3C8DBC", s=60, zorder=10, label="Earth" if name == "XY" else None)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_error_comparison(result, *, title, out_path, draw_burns=True):
    ctx = result["ctx"]
    t = result["t"]
    r = result["r"]
    v = result["v"]

    nasa_pos_eval, nasa_vel_eval = _interp_oem(ctx, t)
    pos_err = np.linalg.norm(r - nasa_pos_eval, axis=1)
    vel_err = np.linalg.norm(v - nasa_vel_eval, axis=1)

    bal_pos, bal_vel = _ballistic_propagation(ctx, t)
    bal_pos_err = np.linalg.norm(bal_pos - nasa_pos_eval, axis=1)
    bal_vel_err = np.linalg.norm(bal_vel - nasa_vel_eval, axis=1)

    days = (t - t[0]) / 86400.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), dpi=130, sharex=True)
    axes[0].semilogy(days, np.maximum(bal_pos_err, 1e-3), color=RED, lw=1.4, label="Ballistic")
    axes[0].semilogy(days, np.maximum(pos_err, 1e-3), color=NAVY, lw=1.7, label="Bézier + IPOPT (deg 7)")
    axes[0].set_ylabel("Position error (km)")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=9, loc="best")
    axes[0].set_title(title)

    axes[1].semilogy(days, np.maximum(bal_vel_err, 1e-9), color=RED, lw=1.4, label="Ballistic")
    axes[1].semilogy(days, np.maximum(vel_err, 1e-9), color=NAVY, lw=1.7, label="Bézier + IPOPT (deg 7)")
    axes[1].set_ylabel("Velocity error (km/s)")
    axes[1].set_xlabel("Time (days from segment start)")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=9, loc="best")

    if draw_burns:
        for burn in (ctx.burns or []):
            for ax in axes:
                ax.axvspan(burn["day_start"], burn["day_end"], color=GOLD, alpha=0.25, lw=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_control_profile(result, *, title, out_path, draw_burns=True):
    ctx = result["ctx"]
    t = result["t"]
    u = result["u"]
    days = (t - t[0]) / 86400.0
    u_mag = np.linalg.norm(u, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), dpi=130, sharex=True)
    axes[0].semilogy(days, np.maximum(u_mag, 1e-12), color=NAVY, lw=1.5)
    axes[0].set_ylabel("|u| (km/s²)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3, which="both")

    axes[1].plot(days, u[:, 0], color="#D62728", lw=1.0, label="$u_x$")
    axes[1].plot(days, u[:, 1], color="#2CA02C", lw=1.0, label="$u_y$")
    axes[1].plot(days, u[:, 2], color="#1F77B4", lw=1.0, label="$u_z$")
    axes[1].set_ylabel("Control (km/s²)")
    axes[1].set_xlabel("Time (days from mission start)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9, loc="best")

    if draw_burns:
        for burn in (ctx.burns or []):
            for ax in axes:
                ax.axvspan(burn["day_start"], burn["day_end"], color=GOLD, alpha=0.25, lw=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_summary_stats(result, *, title, out_path):
    ctx = result["ctx"]
    t = result["t"]
    r = result["r"]
    v = result["v"]
    nasa_pos_eval, nasa_vel_eval = _interp_oem(ctx, t)
    pos_err = np.linalg.norm(r - nasa_pos_eval, axis=1)
    vel_err = np.linalg.norm(v - nasa_vel_eval, axis=1)
    stats = result.get("stats") or {}
    iters = stats.get("iter_count") if isinstance(stats, dict) else None
    pr = stats.get("iterations", {}).get("inf_pr", []) if isinstance(stats, dict) else []
    last_pr = pr[-1] if pr else float("nan")

    lines = [
        f"Phase 2 Bézier + IPOPT (degree 7)  —  {ctx.case}",
        f"",
        f"Mesh:                 N={result['solver'].n_seg} segments",
        f"Wall time:            {result['solve_time']:.2f} s",
        f"IPOPT iterations:     {iters if iters is not None else '—'}",
        f"Status:               {'Solve_Succeeded' if result['success'] else 'NOT converged'}",
        f"Cost J:               {result['cost']:.6e}",
        f"Primal infeasibility: {last_pr:.3e}",
        f"",
        f"OEM position error (km):",
        f"   max  {pos_err.max():.2f}",
        f"   mean {pos_err.mean():.2f}",
        f"OEM velocity error (km/s):",
        f"   max  {vel_err.max():.4e}",
        f"   mean {vel_err.mean():.4e}",
    ]

    fig = plt.figure(figsize=(8, 5.5), dpi=130)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.96, title, ha="center", va="top", fontsize=14, fontweight="bold")
    ax.text(0.05, 0.88, "\n".join(lines), ha="left", va="top",
            fontsize=11, family="monospace")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_history(results: dict, out_path):
    """Combined IPOPT convergence panel (post-TLI + full mission)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), dpi=130, sharex=True)
    colors = {"post_tli": "#2CA02C", "full": "#D62728"}
    labels = {"post_tli": "artemis2_post_tli", "full": "artemis2_full_mission"}
    for case, result in results.items():
        stats = result.get("stats") or {}
        it = stats.get("iterations") or {}
        obj = it.get("obj") or []
        pr = it.get("inf_pr") or []
        if not obj:
            continue
        x = np.arange(len(obj))
        axes[0].semilogy(x, np.maximum(np.asarray(obj), 1e-30),
                         "-o", color=colors[case], lw=1.4, label=labels[case])
        axes[1].semilogy(x, np.maximum(np.asarray(pr), 1e-30),
                         "-o", color=colors[case], lw=1.4, label=labels[case])

    axes[0].set_ylabel("|objective| (∫ |u|² dt, km²/s³)")
    axes[0].set_title("IPOPT convergence history — Artemis II (Phase 2, Bézier deg 7)")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=9)

    axes[1].set_ylabel("constraint violation (inf_pr, km or km/s)")
    axes[1].set_xlabel("IPOPT iteration")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    print("[regen] solving full mission (Bézier degree 7)...", flush=True)
    full = solve_case("artemis2_full_mission", ephem_points=10000)
    print(f"[regen]   J={full['cost']:.4e}  wall={full['solve_time']:.2f}s  status={'OK' if full['success'] else 'FAIL'}",
          flush=True)

    print("[regen] solving post-TLI (Bézier degree 7)...", flush=True)
    post = solve_case("artemis2_post_tli", ephem_points=4000)
    print(f"[regen]   J={post['cost']:.4e}  wall={post['solve_time']:.2f}s  status={'OK' if post['success'] else 'FAIL'}",
          flush=True)

    print("[regen] plotting full mission...", flush=True)
    plot_3d_trajectory(full,
                       title="Artemis II — Full Mission (Bézier+IPOPT, deg 7)",
                       out_path=EPHEM_FULL / "full_3d_trajectory.png",
                       prefix="Full mission")
    plot_2d_projections(full,
                        title="Artemis II Full Mission — 2D projections (Bézier+IPOPT, deg 7)",
                        out_path=EPHEM_FULL / "full_2d_projections.png",
                        prefix="Full mission")
    plot_error_comparison(full,
                          title="Artemis II Full Mission — error vs OEM (Bézier+IPOPT, deg 7)",
                          out_path=EPHEM_FULL / "full_error_comparison.png",
                          draw_burns=True)
    plot_control_profile(full,
                         title="Artemis II Full Mission — Bézier-evaluated control (deg 7)",
                         out_path=EPHEM_FULL / "full_control_profile.png",
                         draw_burns=True)
    plot_summary_stats(full,
                       title="Artemis II Full Mission — summary (Bézier+IPOPT, deg 7)",
                       out_path=EPHEM_FULL / "full_summary_stats.png")

    print("[regen] plotting post-TLI...", flush=True)
    for outdir in (EPHEM_FULL, EPHEM_POST_TLI):
        outdir.mkdir(parents=True, exist_ok=True)
        plot_3d_trajectory(post,
                           title="Artemis II Post-TLI Coast (Bézier+IPOPT, deg 7)",
                           out_path=outdir / "ephem_3d_trajectory.png",
                           prefix="Post-TLI")
        plot_2d_projections(post,
                            title="Artemis II Post-TLI — 2D projections (Bézier+IPOPT, deg 7)",
                            out_path=outdir / "ephem_2d_projections.png",
                            prefix="Post-TLI")
        plot_error_comparison(post,
                              title="Artemis II Post-TLI — error vs OEM (Bézier+IPOPT, deg 7)",
                              out_path=outdir / "ephem_error_comparison.png",
                              draw_burns=False)
        plot_summary_stats(post,
                           title="Artemis II Post-TLI — summary (Bézier+IPOPT, deg 7)",
                           out_path=outdir / "ephem_summary_stats.png")

    print("[regen] plotting convergence history...", flush=True)
    plot_convergence_history({"post_tli": post, "full": full}, CONVERGENCE)

    print("[regen] done.", flush=True)


if __name__ == "__main__":
    main()
