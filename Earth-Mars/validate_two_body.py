"""
validate_two_body.py - Validate Python port against MATLAB reference

Reproduces MATLAB pa_redo.mlx: minimum-energy Earth-to-Mars transfer
in the two-body problem using the shooting method.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynamics import two_body_state_costate_ode
from shooting import propagate, solve_min_energy


def main():
    # =========================================================================
    # Setup — matches MATLAB pa_redo.mlx exactly
    # =========================================================================
    mu = 1.0  # characteristic units

    # Orbital elements
    a_earth = 1.0       # AU
    v_earth = 0.0       # rad
    a_mars = 1.524      # AU
    v_mars = np.pi       # rad

    # Angular velocities
    vel_earth = np.sqrt(mu / a_earth**3)
    vel_mars = np.sqrt(mu / a_mars**3)

    # Planet position functions
    pos_earth = lambda t: a_earth * np.array([
        np.cos(v_earth + vel_earth * t),
        np.sin(v_earth + vel_earth * t)
    ])
    pos_mars = lambda t: a_mars * np.array([
        np.cos(v_mars + vel_mars * t),
        np.sin(v_mars + vel_mars * t)
    ])

    # Time
    t0 = 0.0
    tf = 8.0

    # Spacecraft initial conditions
    r0 = np.array([a_earth, 0.0])
    v0 = np.array([0.0, a_earth * vel_earth])

    # Target state at tf
    pos_mars_f = pos_mars(tf)
    vel_mars_f = vel_mars * np.array([-pos_mars_f[1], pos_mars_f[0]])

    # =========================================================================
    # Solve
    # =========================================================================
    print("Solving minimum-energy Earth-to-Mars transfer...")
    print(f"  t0={t0}, tf={tf}")
    print(f"  r0={r0}, v0={v0}")
    print(f"  Target pos: {pos_mars_f}")
    print(f"  Target vel: {vel_mars_f}")

    lam0_guess = np.zeros(4)
    lam0_sol, info = solve_min_energy(
        r0, v0, pos_mars_f, vel_mars_f, t0, tf,
        lam0_guess=lam0_guess, mu=mu
    )

    print(f"\nSolution lambda0: {lam0_sol}")
    residual_norm = np.linalg.norm(info['fvec'])
    print(f"Residual norm: {residual_norm:.2e}")

    # =========================================================================
    # Propagate full trajectory
    # =========================================================================
    X0 = np.concatenate([r0, v0, lam0_sol])
    time = np.linspace(t0, tf, 10000)
    sol = propagate(two_body_state_costate_ode, X0, [t0, tf], mu=mu)

    rx = sol.y[0]
    ry = sol.y[1]
    lam_vx = sol.y[6]
    lam_vy = sol.y[7]

    # Control
    ux = -0.5 * lam_vx
    uy = -0.5 * lam_vy
    u_mag = np.sqrt(ux**2 + uy**2)

    # Mars orbit
    t_plot = sol.t
    mars_x = a_mars * np.cos(v_mars + vel_mars * t_plot)
    mars_y = a_mars * np.sin(v_mars + vel_mars * t_plot)

    # Hamiltonian check
    H = np.zeros(len(t_plot))
    for i in range(len(t_plot)):
        r_i = sol.y[0:2, i]
        v_i = sol.y[2:4, i]
        lr_i = sol.y[4:6, i]
        lv_i = sol.y[6:8, i]
        u_i = -0.5 * lv_i
        H[i] = (u_i @ u_i + lr_i @ v_i
                + lv_i @ (-mu * r_i / np.linalg.norm(r_i)**3 + u_i))
    delta_H = H - H[0]

    # =========================================================================
    # Plots
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('white')

    # Trajectory
    ax = axes[0, 0]
    ax.set_facecolor('white')
    ax.plot(rx, ry, color='#1f77b4', label='Spacecraft', lw=2)
    ax.plot(mars_x, mars_y, color='#d62728', linestyle='--', alpha=0.5, label='Mars Orbit')
    ax.plot(r0[0], r0[1], 'o', color='#2ca02c', markersize=8, label='Earth (start)')
    ax.plot(pos_mars_f[0], pos_mars_f[1], 'x', color='#d62728', markersize=10,
            markeredgewidth=2, label='Mars (end)')
    ax.plot(0, 0, 'o', color='#f39c12', markersize=12, label='Sun')
    ax.set_aspect('equal')
    ax.set_xlabel('x (AU)', color='black')
    ax.set_ylabel('y (AU)', color='black')
    ax.set_title('Low-Thrust Min-Energy Transfer (Earth-Mars)', color='black')
    ax.legend(fontsize=8, facecolor='white', edgecolor='black', labelcolor='black')
    ax.grid(True, alpha=0.3, color='gray')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.tick_params(colors='black')

    # Control history
    ax = axes[0, 1]
    ax.set_facecolor('white')
    ax.plot(t_plot, ux, label='ux', color='#1f77b4')
    ax.plot(t_plot, uy, label='uy', color='#1f77b4')
    ax.plot(t_plot, u_mag, linestyle='--', label='|u|', color='#2ca02c', lw=1.5)
    ax.set_xlabel('Time (t)', color='black')
    ax.set_ylabel('Control (AU/t²)', color='black')
    ax.set_title('Control History', color='black')
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
    ax.grid(True, alpha=0.3, color='gray')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.tick_params(colors='black')

    # Costate history
    ax = axes[1, 0]
    ax.set_facecolor('white')
    ax.plot(t_plot, sol.y[4], label=r'$\lambda_{rx}$', color='#1f77b4')
    ax.plot(t_plot, sol.y[5], label=r'$\lambda_{ry}$', color='#1f77b4')
    ax.plot(t_plot, sol.y[6], label=r'$\lambda_{vx}$', color='#d62728')
    ax.plot(t_plot, sol.y[7], label=r'$\lambda_{vy}$', color='#d62728')
    ax.set_xlabel('Time (t)', color='black')
    ax.set_ylabel('Costate', color='black')
    ax.set_title('Costate History', color='black')
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
    ax.grid(True, alpha=0.3, color='gray')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.tick_params(colors='black')

    # Hamiltonian
    ax = axes[1, 1]
    ax.set_facecolor('white')
    ax.plot(t_plot, delta_H, color='#2ca02c', lw=1.5)
    ax.set_xlabel('Time (t)', color='black')
    ax.set_ylabel('H(t) - H(0)', color='black')
    ax.set_title('Hamiltonian Conservation Check', color='black')
    ax.grid(True, alpha=0.3, color='gray')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.tick_params(colors='black')

    plt.tight_layout()
    plt.savefig('validation_min_energy.png', dpi=150, facecolor='white', edgecolor='white')
    print(f"\nPlot saved to validation_min_energy.png")

    # =========================================================================
    # Validation summary
    # =========================================================================
    print("\n=== VALIDATION SUMMARY ===")
    print(f"  Residual norm:       {residual_norm:.2e}")
    print(f"  Max |delta H|:       {np.max(np.abs(delta_H)):.2e}")
    print(f"  Final pos error:     {np.linalg.norm(sol.y[0:2, -1] - pos_mars_f):.2e}")
    print(f"  Final vel error:     {np.linalg.norm(sol.y[2:4, -1] - vel_mars_f):.2e}")

    converged = residual_norm < 1e-6 and np.max(np.abs(delta_H)) < 1e-4
    if converged:
        print("  STATUS: PASS")
    else:
        print("  STATUS: CHECK NEEDED")

    return converged


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
