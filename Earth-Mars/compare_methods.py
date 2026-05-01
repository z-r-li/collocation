"""
compare_methods.py - Head-to-head: Shooting vs Bézier Collocation

Solves the same minimum-energy Earth-to-Mars transfer TPBVP using both
methods, compares results, and generates a side-by-side animation for
presentation.
"""

import time as timer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from dynamics import two_body_state_costate_ode, two_body_ode
from shooting import propagate, solve_min_energy
from bezier import BezierCollocation, bezier_eval, bezier_derivative, bezier_second_derivative


# =============================================================================
# Problem Setup
# =============================================================================

MU = 1.0
A_EARTH = 1.0
A_MARS = 1.524
V_MARS = np.pi
VEL_EARTH = np.sqrt(MU / A_EARTH**3)
VEL_MARS = np.sqrt(MU / A_MARS**3)

T0, TF = 0.0, 8.0

R0 = np.array([A_EARTH, 0.0])
V0 = np.array([0.0, A_EARTH * VEL_EARTH])

POS_MARS_F = A_MARS * np.array([np.cos(V_MARS + VEL_MARS * TF),
                                 np.sin(V_MARS + VEL_MARS * TF)])
VEL_MARS_F = VEL_MARS * np.array([-POS_MARS_F[1], POS_MARS_F[0]])

X0_STATE = np.concatenate([R0, V0])
XF_STATE = np.concatenate([POS_MARS_F, VEL_MARS_F])


def gravity_2body(r, mu=MU):
    """Two-body gravitational acceleration (scalar or vectorized)."""
    r = np.asarray(r)
    if r.ndim == 1:
        return -mu * r / np.linalg.norm(r)**3
    else:
        r_norms = np.linalg.norm(r, axis=1, keepdims=True)
        return -mu * r / r_norms**3


# =============================================================================
# Solve both methods
# =============================================================================

def solve_both():
    """Run both methods and return comparable trajectory data."""

    # --- Shooting Method (indirect, via Pontryagin's principle) ---
    print("Solving with shooting method...")
    t0_clock = timer.perf_counter()
    lam0_sol, info = solve_min_energy(
        R0, V0, POS_MARS_F, VEL_MARS_F, T0, TF,
        lam0_guess=np.zeros(4), mu=MU
    )
    shooting_time = timer.perf_counter() - t0_clock
    shooting_residual = np.linalg.norm(info['fvec'])

    X0_full = np.concatenate([R0, V0, lam0_sol])
    sol = propagate(two_body_state_costate_ode, X0_full, [T0, TF],
                    n_steps=500, mu=MU)

    # Compute shooting control: u* = -0.5 * lambda_v
    lam_vx = sol.y[6]
    lam_vy = sol.y[7]
    ux_s = -0.5 * lam_vx
    uy_s = -0.5 * lam_vy
    u_mag_s = np.sqrt(ux_s**2 + uy_s**2)
    # Shooting cost: ∫|u|² dt (trapezoid rule)
    shooting_cost = np.trapezoid(ux_s**2 + uy_s**2, sol.t)

    shooting_data = {
        'lam0': lam0_sol,
        'residual': shooting_residual,
        'time_s': shooting_time,
        'nfev': info['nfev'],
        'cost': shooting_cost,
        't': sol.t,
        'x': sol.y[0], 'y': sol.y[1],
        'vx': sol.y[2], 'vy': sol.y[3],
        'ux': ux_s, 'uy': uy_s, 'u_mag': u_mag_s,
    }
    print(f"  Done: residual={shooting_residual:.2e}, cost={shooting_cost:.4f}, "
          f"time={shooting_time:.3f}s, nfev={info['nfev']}")

    # --- Bézier Collocation (CasADi/IPOPT — dynamics as NLP constraints) ---
    print("Solving with Bézier collocation (CasADi/IPOPT)...")
    from ipopt_collocation_2body import TwoBodyBezierIPOPT

    x0_state = np.concatenate([R0, V0])
    xf_state = np.concatenate([POS_MARS_F, VEL_MARS_F])

    # Warm-start from the shooting solution
    shooting_ref_x = np.column_stack([
        shooting_data['x'], shooting_data['y'],
        shooting_data['vx'], shooting_data['vy'],
    ])
    warm_traj = (shooting_data['t'], shooting_ref_x)

    # When we have a good warm-start from shooting, skip coarse meshes
    # and go directly to a fine mesh to avoid drifting to a different
    # local minimum. Use mesh refinement only for the final polish.
    t0_clock = timer.perf_counter()

    prev_warm = warm_traj
    history = []
    levels = [
        (16, 7, 12),
        (32, 7, 12),
    ]
    for level_idx, (n_seg, deg, nc) in enumerate(levels):
        is_final = (level_idx == len(levels) - 1)
        n_total = n_seg * ((deg + 1) * 4 + nc * 2)
        print(f"\n  Level {level_idx+1}/{len(levels)}: "
              f"{n_seg} seg, deg {deg}, {nc} colloc ({n_total} vars)")

        solver_ipopt = TwoBodyBezierIPOPT(
            mu=MU, n_segments=n_seg, bezier_degree=deg,
            n_collocation=nc,
        )
        level_tol = 1e-8 if is_final else 1e-6
        level_result = solver_ipopt.solve(
            x0_state, xf_state, T0, TF,
            warm_traj=prev_warm,
            max_iter=3000,
            tol=level_tol,
            print_level=0,
        )
        level_result['level'] = (n_seg, deg, nc)
        history.append(level_result)

        print(f"    cost={level_result['cost']:.8f}, "
              f"time={level_result['solve_time']:.3f}s, "
              f"ok={level_result['success']}")

        if level_result['success']:
            prev_warm = (level_result['t'],
                         np.column_stack([level_result['r'],
                                          level_result['v']]))

    result_mr = history[-1] if history[-1]['success'] else history[-2]
    bezier_time = timer.perf_counter() - t0_clock

    sol_dict = result_mr
    max_defect = sol_dict.get('max_defect', 0.0)

    bezier_data = {
        'cost': sol_dict['cost'],
        'time_s': bezier_time,
        'nfev': sum(h['stats'].get('iter_count', 0) for h in history),
        'max_defect': max_defect,
        't': sol_dict['t'],
        'x': sol_dict['r'][:, 0], 'y': sol_dict['r'][:, 1],
        'vx': sol_dict['v'][:, 0], 'vy': sol_dict['v'][:, 1],
        'ux': sol_dict['u'][:, 0], 'uy': sol_dict['u'][:, 1],
        'u_mag': np.sqrt(sol_dict['u'][:, 0]**2 + sol_dict['u'][:, 1]**2),
        'segments': sol_dict.get('segments', []),
        'converged': sol_dict['success'],
        'mesh_history': history,
    }
    total_iters = bezier_data['nfev']
    print(f"\n  Converged: {sol_dict['success']}")
    print(f"  Cost J = ∫|u|²dt: {sol_dict['cost']:.6f}")
    print(f"  Solve time: {bezier_time:.3f}s (total across {len(history)} mesh levels)")
    print(f"  Total IPOPT iterations: {total_iters}")
    for i, h in enumerate(history):
        lvl = h['level']
        print(f"    Level {i+1} ({lvl[0]} seg, deg {lvl[1]}): "
              f"cost={h['cost']:.8f}, time={h['solve_time']:.3f}s")

    return shooting_data, bezier_data


# =============================================================================
# Static comparison plot
# =============================================================================

def plot_comparison(shooting, bezier):
    """Generate a 6-panel comparison figure."""
    from scipy.interpolate import interp1d

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.30,
                          left=0.06, right=0.97, top=0.92, bottom=0.07)

    fig.suptitle('Earth-Mars Transfer: Shooting vs Bézier (IPOPT)',
                 fontsize=15, fontweight='bold', y=0.97, color='black')

    # ---- Panel 1: Trajectories ----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor('white')
    # Full orbit arcs
    earth_orbit_t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(A_EARTH * np.cos(earth_orbit_t), A_EARTH * np.sin(earth_orbit_t),
            ':', color='#3498db', alpha=0.3, lw=0.8)
    t_mars = np.linspace(0, 2 * np.pi / VEL_MARS, 300)
    ax.plot(A_MARS * np.cos(V_MARS + VEL_MARS * t_mars),
            A_MARS * np.sin(V_MARS + VEL_MARS * t_mars),
            ':', color='#d62728', alpha=0.3, lw=0.8)

    ax.plot(shooting['x'], shooting['y'], '-', color='#1f77b4', lw=2.2,
            label='Shooting')
    ax.plot(bezier['x'], bezier['y'], '--', color='#d62728', lw=2,
            alpha=0.85, label='Bézier (IPOPT)')

    ax.plot(0, 0, 'o', color='#f39c12', ms=10, mec='#e67e22', mew=1,
            zorder=10)
    ax.annotate('Sun', (0, 0), textcoords='offset points',
                xytext=(8, 8), fontsize=8, color='#f39c12')
    ax.plot(R0[0], R0[1], 'o', color='#3498db', ms=7, mec='black', mew=0.8,
            zorder=10, label='Earth (dep)')
    ax.plot(POS_MARS_F[0], POS_MARS_F[1], 'o', color='#d62728', ms=7,
            mec='black', mew=0.8, zorder=10, label='Mars (arr)')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color='gray')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.set_xlabel('x (AU)', fontsize=10, color='black')
    ax.set_ylabel('y (AU)', fontsize=10, color='black')
    ax.set_title('Transfer Trajectory', fontsize=11, fontweight='bold', color='black')
    ax.tick_params(colors='black')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9,
              handlelength=1.5, borderpad=0.4, facecolor='white',
              edgecolor='black', labelcolor='black')

    # ---- Panel 2: Trajectory difference ----
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor('white')
    t_c = bezier['t']
    mask = (t_c >= T0 + 0.01) & (t_c <= TF - 0.01)
    t_c_m = t_c[mask]
    interp_sx = interp1d(shooting['t'], shooting['x'], kind='cubic')
    interp_sy = interp1d(shooting['t'], shooting['y'], kind='cubic')
    dx = bezier['x'][mask] - interp_sx(t_c_m)
    dy = bezier['y'][mask] - interp_sy(t_c_m)
    pos_diff = np.sqrt(dx**2 + dy**2)

    ax.semilogy(t_c_m, pos_diff, '-', color='#1f77b4', lw=1.5)
    ax.fill_between(t_c_m, pos_diff, alpha=0.15, color='#1f77b4')
    ax.set_xlabel('Time', fontsize=10, color='black')
    ax.set_ylabel('||Δr|| (AU)', fontsize=10, color='black')
    ax.set_title('Trajectory Difference', fontsize=11, fontweight='bold', color='black')
    ax.grid(True, alpha=0.3, color='gray')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.tick_params(colors='black')

    idx_max = np.argmax(pos_diff)
    ax.annotate(f'max = {pos_diff[idx_max]:.2e}',
                xy=(t_c_m[idx_max], pos_diff[idx_max]),
                xytext=(0.55, 0.85), textcoords='axes fraction',
                fontsize=9, color='#d62728',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1),
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='#d62728', alpha=0.8))

    # ---- Panel 3: Control magnitude ----
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor('white')
    ax.plot(shooting['t'], shooting['u_mag'], '-', color='#1f77b4', lw=1.8,
            label='Shooting', alpha=0.9)
    ax.plot(bezier['t'], bezier['u_mag'], '--', color='#d62728', lw=1.8,
            label='Bézier (IPOPT)', alpha=0.8)
    ax.set_xlabel('Time', fontsize=10, color='black')
    ax.set_ylabel('|u| (thrust magnitude)', fontsize=10, color='black')
    ax.set_title('Control Magnitude', fontsize=11, fontweight='bold', color='black')
    ax.legend(fontsize=9, loc='best', framealpha=0.9, facecolor='white',
              edgecolor='black', labelcolor='black')
    ax.grid(True, alpha=0.3, color='gray')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.tick_params(colors='black')

    # ---- Panel 4: Control components ----
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor('white')
    ax.plot(shooting['t'], shooting['ux'], '-', color='#1f77b4', lw=1.5,
            label='$u_x$ (shoot)')
    ax.plot(shooting['t'], shooting['uy'], '--', color='#1f77b4', lw=1.5,
            label='$u_y$ (shoot)')
    ax.plot(bezier['t'], bezier['ux'], '-', color='#d62728', lw=1.5,
            alpha=0.7, label='$u_x$ (Bézier)')
    ax.plot(bezier['t'], bezier['uy'], '--', color='#d62728', lw=1.5,
            alpha=0.7, label='$u_y$ (Bézier)')
    ax.set_xlabel('Time', fontsize=10, color='black')
    ax.set_ylabel('Control component', fontsize=10, color='black')
    ax.set_title('Control Components', fontsize=11, fontweight='bold', color='black')
    ax.legend(fontsize=8.5, ncol=2, loc='best', framealpha=0.9,
              columnspacing=1.0, handlelength=1.8, facecolor='white',
              edgecolor='black', labelcolor='black')
    ax.grid(True, alpha=0.3, color='gray')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.tick_params(colors='black')

    # ---- Panel 5: Bézier control points overlay ----
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor('white')
    mars_x = A_MARS * np.cos(V_MARS + VEL_MARS * np.linspace(T0, TF, 300))
    mars_y = A_MARS * np.sin(V_MARS + VEL_MARS * np.linspace(T0, TF, 300))
    ax.plot(mars_x, mars_y, ':', color='#d62728', alpha=0.3, lw=0.8)
    ax.plot(bezier['x'], bezier['y'], '-', color='#d62728', lw=1.5,
            alpha=0.6, label='Bézier trajectory')

    n_seg = len(bezier['segments'])
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_seg))
    for i, cp in enumerate(bezier['segments']):
        lbl = f'Seg {i+1}' if i < 3 else (f'... ({n_seg} total)' if i == 3 else None)
        ax.plot(cp[:, 0], cp[:, 1], 'o-', color=colors[i], ms=3,
                alpha=0.6, lw=0.5, label=lbl)

    ax.plot(R0[0], R0[1], 'o', color='#3498db', ms=7, mec='black', mew=0.8,
            zorder=10)
    ax.plot(POS_MARS_F[0], POS_MARS_F[1], 'o', color='#d62728', ms=7,
            mec='black', mew=0.8, zorder=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color='gray')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.set_xlabel('x (AU)', fontsize=10, color='black')
    ax.set_ylabel('y (AU)', fontsize=10, color='black')
    ax.set_title('Bézier Control Points', fontsize=11, fontweight='bold', color='black')
    ax.tick_params(colors='black')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9, facecolor='white',
              edgecolor='black', labelcolor='black')

    # ---- Panel 6: Stats table ----
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor('white')
    ax.axis('off')
    table_data = [
        ['Metric', 'Shooting', 'Bézier (IPOPT)'],
        ['Method', 'Indirect (PMP)', 'Direct NLP (CasADi)'],
        ['Converged', 'Yes', str(bezier.get('converged', 'N/A'))],
        ['Cost J = ∫|u|²dt', f"{shooting['cost']:.6f}",
         f"{bezier['cost']:.6f}"],
        ['Residual / Defect', f"{shooting['residual']:.2e}",
         f"{bezier.get('max_defect', 0):.2e}"],
        ['Solve time (s)', f"{shooting['time_s']:.3f}",
         f"{bezier['time_s']:.3f}"],
        ['Func evals / Iters', str(shooting['nfev']),
         str(bezier['nfev'])],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.38, 0.31, 0.31])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if i == 0:
            cell.set_facecolor('white')
            cell.set_text_props(color='black', fontweight='bold', fontsize=10)
        elif j == 0:
            cell.set_facecolor('white')
            cell.set_text_props(fontweight='bold', fontsize=9.5, color='black')
        elif i % 2 == 0:
            cell.set_facecolor('white')
            cell.set_text_props(color='black')
        else:
            cell.set_facecolor('white')
            cell.set_text_props(color='black')
    ax.set_title('Performance Comparison', pad=15, fontsize=11,
                 fontweight='bold', color='black')

    plt.savefig('comparison_shooting_vs_bezier.png', dpi=150, facecolor='white', edgecolor='white')
    print("Saved: comparison_shooting_vs_bezier.png")


# =============================================================================
# Side-by-side animation for presentation
# =============================================================================

def create_animation(shooting, bezier, fps=30, duration_s=8):
    """
    Create a side-by-side GIF animation showing both methods building
    their trajectories simultaneously with key metrics.
    """
    n_frames = fps * duration_s

    fig, (ax_s, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    for ax, title in [(ax_s, 'Shooting Method'), (ax_b, 'Bézier Collocation')]:
        ax.set_facecolor('white')
        ax.set_aspect('equal')
        ax.set_xlim(-2.0, 2.0); ax.set_ylim(-2.0, 2.0)
        ax.set_xlabel('x (AU)', color='black')
        ax.set_ylabel('y (AU)', color='black')
        ax.set_title(title, color='black', fontsize=14, fontweight='bold')
        ax.tick_params(colors='black')
        for spine in ax.spines.values():
            spine.set_color('black')
        ax.grid(True, alpha=0.3, color='gray')

    # Precompute Mars orbit
    t_mars = np.linspace(0, 2 * np.pi / VEL_MARS, 300)
    mars_orbit_x = A_MARS * np.cos(V_MARS + VEL_MARS * t_mars)
    mars_orbit_y = A_MARS * np.sin(V_MARS + VEL_MARS * t_mars)
    earth_orbit_x = A_EARTH * np.cos(np.linspace(0, 2*np.pi, 200))
    earth_orbit_y = A_EARTH * np.sin(np.linspace(0, 2*np.pi, 200))

    # Static elements
    for ax in [ax_s, ax_b]:
        ax.plot(earth_orbit_x, earth_orbit_y, ':', color='#3498db', alpha=0.3, lw=0.5)
        ax.plot(mars_orbit_x, mars_orbit_y, ':', color='#e74c3c', alpha=0.3, lw=0.5)
        ax.plot(0, 0, 'o', color='#f39c12', ms=10)  # Sun

    # Animated elements
    trail_s, = ax_s.plot([], [], '-', color='#00d2ff', lw=2)
    craft_s, = ax_s.plot([], [], 'o', color='#00d2ff', ms=8)
    earth_s, = ax_s.plot([], [], 'o', color='#3498db', ms=6)
    mars_s, = ax_s.plot([], [], 'o', color='#e74c3c', ms=6)
    text_s = ax_s.text(0.02, 0.02, '', transform=ax_s.transAxes,
                       color='#00d2ff', fontsize=9, family='monospace',
                       verticalalignment='bottom')

    trail_b, = ax_b.plot([], [], '-', color='#ff6b6b', lw=2)
    craft_b, = ax_b.plot([], [], 'o', color='#ff6b6b', ms=8)
    earth_b, = ax_b.plot([], [], 'o', color='#3498db', ms=6)
    mars_b, = ax_b.plot([], [], 'o', color='#e74c3c', ms=6)
    text_b = ax_b.text(0.02, 0.02, '', transform=ax_b.transAxes,
                       color='#ff6b6b', fontsize=9, family='monospace',
                       verticalalignment='bottom')

    # Also show Bézier control points fading in
    cp_plots = []
    colors_cp = plt.cm.plasma(np.linspace(0.2, 0.8, len(bezier['segments'])))
    for i, cp in enumerate(bezier['segments']):
        p, = ax_b.plot([], [], 's-', color=colors_cp[i], ms=4, alpha=0, lw=0.5)
        cp_plots.append((p, cp))

    # Time title
    time_text = fig.suptitle('', color='black', fontsize=12, y=0.98)

    # Index mappings
    n_s = len(shooting['t'])
    n_b = len(bezier['t'])

    def init():
        trail_s.set_data([], []); craft_s.set_data([], [])
        trail_b.set_data([], []); craft_b.set_data([], [])
        earth_s.set_data([], []); mars_s.set_data([], [])
        earth_b.set_data([], []); mars_b.set_data([], [])
        for p, _ in cp_plots:
            p.set_data([], [])
        return []

    def update(frame):
        frac = frame / n_frames
        t_now = T0 + frac * (TF - T0)

        # Shooting
        idx_s = min(int(frac * n_s), n_s - 1)
        trail_s.set_data(shooting['x'][:idx_s+1], shooting['y'][:idx_s+1])
        craft_s.set_data([shooting['x'][idx_s]], [shooting['y'][idx_s]])

        # Bézier
        idx_b = min(int(frac * n_b), n_b - 1)
        trail_b.set_data(bezier['x'][:idx_b+1], bezier['y'][:idx_b+1])
        craft_b.set_data([bezier['x'][idx_b]], [bezier['y'][idx_b]])

        # Planets
        ex = A_EARTH * np.cos(VEL_EARTH * t_now)
        ey = A_EARTH * np.sin(VEL_EARTH * t_now)
        mx = A_MARS * np.cos(V_MARS + VEL_MARS * t_now)
        my = A_MARS * np.sin(V_MARS + VEL_MARS * t_now)
        for e_plot in [earth_s, earth_b]:
            e_plot.set_data([ex], [ey])
        for m_plot in [mars_s, mars_b]:
            m_plot.set_data([mx], [my])

        # Control points fade in over time
        n_seg = len(bezier['segments'])
        for i, (p, cp) in enumerate(cp_plots):
            seg_start_frac = i / n_seg
            if frac > seg_start_frac:
                alpha = min(1.0, (frac - seg_start_frac) * n_seg)
                p.set_data(cp[:, 0], cp[:, 1])
                p.set_alpha(alpha * 0.6)

        # Stats text
        text_s.set_text(
            f"t = {t_now:.2f}\n"
            f"Cost: {shooting['cost']:.4f}\n"
            f"Solve: {shooting['time_s']:.3f}s"
        )
        text_b.set_text(
            f"t = {t_now:.2f}\n"
            f"Cost: {bezier['cost']:.4f}\n"
            f"Solve: {bezier['time_s']:.3f}s"
        )

        time_text.set_text(
            f"Earth-Mars Transfer  |  t = {t_now:.2f} / {TF:.1f}"
        )

        return []

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=n_frames, interval=1000/fps, blit=False)

    print("Saving animation (this takes a moment)...")
    writer = PillowWriter(fps=fps)
    anim.save('comparison_animation.gif', writer=writer, dpi=100)
    print("Saved: comparison_animation.gif")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Shooting Method vs Bézier Collocation")
    print("Min-Energy Earth-to-Mars Transfer (2D Two-Body)")
    print("=" * 60)

    shooting, bezier = solve_both()
    plot_comparison(shooting, bezier)
    create_animation(shooting, bezier, fps=20, duration_s=6)

    print("\nDone! Files created:")
    print("  - comparison_shooting_vs_bezier.png (static)")
    print("  - comparison_animation.gif (for presentation)")


if __name__ == '__main__':
    main()
