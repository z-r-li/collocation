"""
cr3bp_transfer_segmented.py — N-sweep of literal segmented Bezier collocation

Companion to cr3bp_transfer.py. Where the main script compares shooting against
an IPOPT multi-shooting cascade with Bezier control grids, this script sticks
with a literal per-segment Bezier parameterization of the state [r, v] and
sweeps the mesh count N ∈ {1, 2, 4, 8}, analogous to h-refinement in CFD.

Everything but the Bezier solver is reused from cr3bp_transfer.py — problem
setup, shooting reference, Lyapunov orbit data, plotting palette.

Author: Zhuorui Li (AAE 568, Spring 2026)
"""

import sys
import os
import time as timer

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Path setup (mirrors cr3bp_transfer.py)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Earth-Mars'))

from cr3bp_planar import MU  # noqa: E402
from cr3bp_transfer import (  # noqa: E402
    setup_transfer_problem,
    solve_shooting,
    cr3bp_planar_controlled_ode,
    cr3bp_jacobi_planar,
)
from bezier_segmented import (  # noqa: E402
    CR3BPSegmentedBezier,
    run_n_sweep,
    format_sweep_table,
)
from scipy.integrate import solve_ivp


# =============================================================================
# Shooting baseline (re-used as warm-start and reference)
# =============================================================================

def solve_shooting_reference(x0, xf, t0, tf, mu=MU):
    """Wrap solve_shooting and package its output the same way solve_both does."""
    print("\n--- Indirect Shooting (Pontryagin) — baseline ---")
    t_start = timer.perf_counter()
    lam0_sol, sol_shoot, info = solve_shooting(x0, xf, t0, tf, mu)
    elapsed = timer.perf_counter() - t_start
    residual = float(np.linalg.norm(info['fvec']))

    lam_vx = sol_shoot.y[6]
    lam_vy = sol_shoot.y[7]
    ux = -0.5 * lam_vx
    uy = -0.5 * lam_vy
    cost = float(np.trapezoid(ux**2 + uy**2, sol_shoot.t))

    C = np.array([
        cr3bp_jacobi_planar(sol_shoot.y[:4, k], mu)
        for k in range(sol_shoot.y.shape[1])
    ])

    data = {
        'lam0': lam0_sol,
        'residual': residual,
        'time_s': elapsed,
        'cost': cost,
        'nfev': int(info['nfev']),
        't': sol_shoot.t,
        'x': sol_shoot.y[0], 'y': sol_shoot.y[1],
        'vx': sol_shoot.y[2], 'vy': sol_shoot.y[3],
        'ux': ux, 'uy': uy,
        'u_mag': np.sqrt(ux**2 + uy**2),
        'jacobi': C,
    }
    print(f"  Residual : {residual:.2e}")
    print(f"  Cost J   : {cost:.6f}")
    print(f"  Wallclock: {elapsed:.3f}s")
    return data


# =============================================================================
# Plots
# =============================================================================

def plot_sweep_trajectories(shooting, sweep, lyap_data,
                            save_prefix='cr3bp_transfer_segmented'):
    """Grid figure: trajectories for each N overlayed with shooting."""
    n_panels = len(sweep)
    # Choose a near-square grid (2-column-friendly)
    n_cols = 2 if n_panels <= 4 else 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig = plt.figure(figsize=(5.5 * n_cols, 4.6 * n_rows), facecolor='white')
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.32, wspace=0.25,
                          left=0.06, right=0.97, top=0.94, bottom=0.06)

    xL1 = lyap_data['L1']['xL']
    xL2 = lyap_data['L2']['xL']
    sol_L1 = lyap_data['L1']['sol']
    sol_L2 = lyap_data['L2']['sol']

    fig.suptitle(
        'Literal Segmented Bezier Collocation — N-sweep vs. Shooting',
        fontsize=14, fontweight='bold', color='black', y=0.98,
    )

    for idx, r in enumerate(sweep):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        ax.set_facecolor('white')

        ax.plot(sol_L1.y[0], sol_L1.y[1], '-', color='#2ca02c',
                alpha=0.7, lw=1.2, label='L1 Lyapunov')
        ax.plot(sol_L2.y[0], sol_L2.y[1], '-', color='#9467bd',
                alpha=0.7, lw=1.2, label='L2 Lyapunov')

        ax.plot(shooting['x'], shooting['y'], '-', color='#1f77b4',
                lw=2, label='Shooting', alpha=0.75)

        if r['success'] and r['r'] is not None:
            ax.plot(r['r'][:, 0], r['r'][:, 1], '--', color='#d62728',
                    lw=2, label=f"Seg-Bezier N={r['N']}")
            # Show segment endpoints
            if r.get('segments'):
                endpoints = np.array([seg[0][:2] for seg in r['segments']]
                                     + [r['segments'][-1][-1][:2]])
                ax.plot(endpoints[:, 0], endpoints[:, 1], 'o',
                        color='#d62728', ms=5, mec='black', mew=0.8,
                        zorder=9, label='Segment boundaries')

        ax.plot(1 - MU, 0, 'o', color='#95a5a6', ms=7,
                mec='black', mew=0.8, zorder=10)
        ax.plot(xL1, 0, 'D', color='#f39c12', ms=6,
                mec='black', mew=0.8, zorder=10)
        ax.plot(xL2, 0, 'D', color='#d62728', ms=6,
                mec='black', mew=0.8, zorder=10)

        pad = 0.04
        ax.set_xlim(xL1 - 0.04, xL2 + 0.04)
        y_ext = max(
            abs(sol_L1.y[1]).max(),
            abs(sol_L2.y[1]).max(),
            abs(shooting['y']).max(),
        ) + pad
        ax.set_ylim(-y_ext, y_ext)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color='gray')

        flag = 'converged' if (r['success'] and r['max_defect'] < 1e-4) \
            else 'not converged'
        ax.set_title(
            f"N = {r['N']} — {flag}   "
            f"(J = {r['cost']:.4f},  max|defect| = {r['max_defect']:.1e})",
            fontsize=10, color='black', fontweight='bold',
        )
        ax.set_xlabel('x (rotating frame)', fontsize=9, color='black')
        ax.set_ylabel('y (rotating frame)', fontsize=9, color='black')
        ax.tick_params(colors='black')
        for spine in ax.spines.values():
            spine.set_color('black')
        ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9,
                  facecolor='white', edgecolor='black',
                  labelcolor='black', handlelength=1.5,
                  borderpad=0.4, labelspacing=0.35)

    fname = f'{save_prefix}_trajectories.png'
    plt.savefig(fname, dpi=150, facecolor='white', edgecolor='white')
    print(f"\nSaved: {fname}")
    plt.close(fig)
    return fname


def plot_sweep_convergence(shooting, sweep,
                           save_prefix='cr3bp_transfer_segmented'):
    """
    Three-panel convergence summary:
      (1) cost vs N, with shooting baseline
      (2) wall-clock vs N
      (3) max |defect| vs N
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor='white')

    Ns = np.array([r['N'] for r in sweep])
    costs = np.array([r['cost'] for r in sweep])
    times = np.array([r['time_s'] for r in sweep])
    defects = np.array([r['max_defect'] for r in sweep])
    conv_mask = np.array([(r['success'] and r['max_defect'] < 1e-4)
                          for r in sweep])

    # Panel 1: Cost
    ax = axes[0]
    ax.set_facecolor('white')
    ax.axhline(shooting['cost'], color='#1f77b4', ls='--', lw=1.5,
               label=f"Shooting baseline = {shooting['cost']:.5f}")
    # Converged points
    if conv_mask.any():
        ax.plot(Ns[conv_mask], costs[conv_mask], 'o-',
                color='#d62728', ms=9, lw=1.8, label='Segmented Bezier (conv.)')
    # Non-converged points (show as hollow)
    if (~conv_mask).any():
        ax.plot(Ns[~conv_mask], costs[~conv_mask], 'o',
                mfc='none', mec='#d62728', ms=9, mew=1.5,
                label='Segmented Bezier (not conv.)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel('N segments', fontsize=10, color='black')
    ax.set_ylabel(r'$J = \int |u|^2 \, dt$', fontsize=10, color='black')
    ax.set_title('Cost vs. mesh count (log-log)', fontsize=11,
                 color='black', fontweight='bold')
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.legend(fontsize=8.5, loc='best', framealpha=0.9,
              facecolor='white', edgecolor='black', labelcolor='black')

    # Panel 2: Wall-clock
    ax = axes[1]
    ax.set_facecolor('white')
    ax.plot(Ns, times, 'o-', color='#2ca02c', ms=9, lw=1.8,
            label='SLSQP wallclock')
    ax.axhline(shooting['time_s'], color='#1f77b4', ls='--', lw=1.5,
               label=f"Shooting = {shooting['time_s']:.2f}s")
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel('N segments', fontsize=10, color='black')
    ax.set_ylabel('wall-clock time (s)', fontsize=10, color='black')
    ax.set_title('Solve time vs. mesh count (log-log)', fontsize=11,
                 color='black', fontweight='bold')
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.legend(fontsize=8.5, loc='best', framealpha=0.9,
              facecolor='white', edgecolor='black', labelcolor='black')

    # Panel 3: Max defect
    ax = axes[2]
    ax.set_facecolor('white')
    # Only plot positive defects (some may be zero for failed runs)
    mask = defects > 0
    ax.semilogy(Ns[mask], defects[mask], 'o-',
                color='#d62728', ms=9, lw=1.8)
    ax.axhline(1e-4, color='#f39c12', ls=':', lw=1.5,
               label='convergence threshold 1e-4')
    ax.set_xscale('log', base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel('N segments', fontsize=10, color='black')
    ax.set_ylabel(r'$\max |\mathrm{defect}|$', fontsize=10, color='black')
    ax.set_title('Constraint violation vs. mesh count', fontsize=11,
                 color='black', fontweight='bold')
    ax.grid(True, alpha=0.3, color='gray', which='both')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.legend(fontsize=8.5, loc='best', framealpha=0.9,
              facecolor='white', edgecolor='black', labelcolor='black')

    fig.suptitle(
        'Literal Segmented Bezier — mesh refinement cascade',
        fontsize=13, fontweight='bold', color='black', y=1.00,
    )
    plt.tight_layout()
    fname = f'{save_prefix}_convergence.png'
    plt.savefig(fname, dpi=150, facecolor='white', edgecolor='white', bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)
    return fname


# =============================================================================
# Forward-propagation sanity check for the finest converged Bezier solution
# =============================================================================

def validate_finest(sweep, x0, t0, tf, mu=MU):
    """
    For the finest converged Bezier in the sweep, forward-propagate the
    CR3BP with the Bezier-derived control history and report endpoint error
    and max deviation.
    """
    # Pick the finest converged run
    finest = None
    for r in sorted(sweep, key=lambda s: -s['N']):
        if r['success'] and r['max_defect'] < 1e-4 and r['r'] is not None:
            finest = r
            break
    if finest is None:
        print("\n--- Forward-prop validation: skipped (no converged run) ---")
        return None

    print(f"\n--- Forward-prop validation on finest N={finest['N']} ---")

    ux_interp = interp1d(finest['t'], finest['u'][:, 0],
                         kind='cubic', fill_value='extrapolate')
    uy_interp = interp1d(finest['t'], finest['u'][:, 1],
                         kind='cubic', fill_value='extrapolate')

    def ode(t, state):
        x, y, vx, vy = state
        r1 = np.sqrt((x + mu)**2 + y**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2)
        Ux = x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
        Uy = y - (1 - mu) * y / r1**3 - mu * y / r2**3
        return [vx, vy,
                2 * vy + Ux + float(ux_interp(t)),
                -2 * vx + Uy + float(uy_interp(t))]

    sol_fwd = solve_ivp(ode, [t0, tf], x0.tolist(),
                        method='RK45', rtol=1e-12, atol=1e-12,
                        t_eval=np.linspace(t0, tf, 500))

    xf_fwd = sol_fwd.y[:, -1]
    xf_bez = np.array([finest['r'][-1, 0], finest['r'][-1, 1],
                       finest['v'][-1, 0], finest['v'][-1, 1]])
    err_pos = float(np.linalg.norm(xf_fwd[:2] - xf_bez[:2]))
    err_vel = float(np.linalg.norm(xf_fwd[2:] - xf_bez[2:]))

    bx = interp1d(finest['t'], finest['r'][:, 0], kind='cubic')
    by = interp1d(finest['t'], finest['r'][:, 1], kind='cubic')
    dx = sol_fwd.y[0] - bx(sol_fwd.t)
    dy = sol_fwd.y[1] - by(sol_fwd.t)
    max_dev = float(np.max(np.sqrt(dx**2 + dy**2)))

    print(f"  Endpoint position error : {err_pos:.2e}")
    print(f"  Endpoint velocity error : {err_vel:.2e}")
    print(f"  Max position deviation  : {max_dev:.2e}")

    return {
        'N': finest['N'],
        'err_pos': err_pos, 'err_vel': err_vel, 'max_dev': max_dev,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  L1 -> L2 Lyapunov Transfer  |  Literal Segmented Bezier N-sweep")
    print("  Planar Earth-Moon CR3BP     |  Min-energy low-thrust")
    print("=" * 70)

    x0, xf, t0, tf, lyap_data = setup_transfer_problem(
        Ax_L1=0.02, Ax_L2=0.02,
    )
    print(f"\nTransfer: x0 = {x0}")
    print(f"          xf = {xf}")
    print(f"          t  = [{t0:.4f}, {tf:.4f}]")

    # 1. Shooting reference (gives us both a cost baseline and a warm-start
    #    trajectory for the first Bezier solve).
    shooting = solve_shooting_reference(x0, xf, t0, tf)
    shoot_ref_t = shooting['t']
    shoot_ref_x = np.column_stack([
        shooting['x'], shooting['y'], shooting['vx'], shooting['vy'],
    ])

    # 2. N-sweep, warm-started from shooting.
    print("\n" + "=" * 70)
    print("  Literal Segmented Bezier — N-sweep (SLSQP, C0-state junctions)")
    print("=" * 70)
    # n_collocation = deg+1 matches hp-pseudospectral sizing. Too few
    # collocation points (e.g., n_colloc=6 with deg=7) lets SLSQP find
    # "gaming" solutions that drive u -> 0 at the collocation nodes while
    # the Bezier polynomial oscillates violently in between, so the
    # forward-propagated endpoint drifts by megameters. Increasing to 8
    # densifies the defect constraints enough to kill that failure mode.
    sweep = run_n_sweep(
        x0, xf, t0, tf,
        N_list=(1, 2, 4, 8, 16, 32),
        bezier_degree=7,
        n_collocation=8,
        warm_traj=(shoot_ref_t, shoot_ref_x),
        max_iter=300,
        ftol=1e-9,
        verbose=True,
    )

    # 3. Forward-propagation sanity check on the finest converged run.
    validate_finest(sweep, x0, t0, tf)

    # 4. Plots.
    plot_sweep_trajectories(shooting, sweep, lyap_data)
    plot_sweep_convergence(shooting, sweep)

    # 5. Summary table (print to stdout and save as a plain-text file).
    print("\n" + "=" * 70)
    print("  N-SWEEP SUMMARY")
    print("=" * 70)
    table = format_sweep_table(sweep)
    print(table)

    summary_txt = (
        "Literal Segmented Bezier — N-sweep summary\n"
        f"Problem: L1 -> L2 Lyapunov transfer (planar Earth-Moon CR3BP)\n"
        f"Degree per segment: 7, Gauss-Legendre nodes per segment: 12\n"
        f"Shooting baseline cost: J = {shooting['cost']:.6f}\n\n"
        + table + "\n"
    )
    with open('cr3bp_transfer_segmented_summary.txt', 'w') as f:
        f.write(summary_txt)
    print("\nSaved: cr3bp_transfer_segmented_summary.txt")

    return shooting, sweep


if __name__ == '__main__':
    main()
