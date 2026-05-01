#!/usr/bin/env python3
"""
leo_to_nrho_cr3bp.py — LEO-to-NRHO Transfer in the 3D CR3BP

Minimum-energy transfer from 185 km LEO (28.5° inclination) to the
9:2 L2 Southern NRHO, solved with two methods:

  1. IPOPT Direct Multiple-Shooting (CasADi + RK4 defects)
     - Mesh refinement cascade: coarse → medium → fine
     - Warm-start from linear interpolation, then from previous solution

  2. Indirect Shooting (Pontryagin + scipy.fsolve)
     - 12D state+costate propagation
     - Warm-started from IPOPT control solution (u* ≈ -½λᵥ)

Both compared against ballistic propagation (no control) for reference.

Validated dynamics from cr3bp_3d.py (which was validated against NASA OEM).

Author: Zhuorui, AAE 568 Spring 2026
"""

import sys
import os
import numpy as np
import time as timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Import infrastructure from cr3bp_3d.py
from cr3bp_3d import (
    MU, L_STAR, T_STAR, V_STAR, R_MOON, R_EARTH,
    EARTH_POS, MOON_POS, NRHO_9_2, JPL_TUNIT,
    nrho_state, nrho_period,
    cr3bp_ode, cr3bp_controlled_ode,
    pseudo_potential_gradient, pseudo_potential_hessian,
    jacobi_constant, dist_from_moon, dist_from_earth,
    propagate, propagate_with_stm,
    leo_state, collinear_libration_points, validate_nrho,
)

# Check for CasADi
try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

# Output directory
OUTDIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# TRANSFER PROBLEM SETUP
# =============================================================================

def setup_transfer(departure_nu_deg=0.0, transfer_time_days=5.0, verbose=True):
    """
    Set up the LEO → NRHO transfer problem.

    Args:
        departure_nu_deg: true anomaly at LEO departure (degrees)
        transfer_time_days: total transfer time (days)
        verbose: print details

    Returns:
        x0: (6,) LEO departure state (nondim, rotating frame)
        xf: (6,) NRHO arrival state (nondim, rotating frame)
        tf: transfer time (nondim)
        info: dict with dimensional quantities
    """
    # LEO departure
    x0 = leo_state(altitude_km=185.0, inclination_deg=28.5,
                    true_anomaly_deg=departure_nu_deg)

    # NRHO arrival (apolune)
    xf = nrho_state()

    # Transfer time
    tf = transfer_time_days * 86400.0 / T_STAR

    # Dimensional info
    r0_km = dist_from_earth(x0) * L_STAR
    v0_kms = np.linalg.norm(x0[3:6]) * V_STAR
    rf_km = dist_from_moon(xf) * L_STAR

    info = {
        'departure_nu_deg': departure_nu_deg,
        'transfer_time_days': transfer_time_days,
        'leo_alt_km': 185.0,
        'leo_r_km': r0_km,
        'leo_v_kms': v0_kms,
        'nrho_r_moon_km': rf_km,
        'tf_nondim': tf,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("TRANSFER PROBLEM SETUP")
        print("=" * 70)
        print(f"\n  LEO Departure:")
        print(f"    Altitude     = 185 km circular")
        print(f"    Inclination  = 28.5°")
        print(f"    True anomaly = {departure_nu_deg}°")
        print(f"    r from Earth = {r0_km:.1f} km")
        print(f"    |v|          = {v0_kms:.3f} km/s")
        print(f"    Jacobi C     = {jacobi_constant(x0):.4f}")
        print(f"    State: {x0}")

        print(f"\n  NRHO Arrival (9:2 L2 Southern, apolune):")
        print(f"    r from Moon  = {rf_km:.1f} km")
        print(f"    Jacobi C     = {jacobi_constant(xf):.5f}")
        print(f"    State: {xf}")

        print(f"\n  Transfer:")
        print(f"    Duration     = {transfer_time_days:.1f} days = {tf:.6f} nondim")

    return x0, xf, tf, info


# =============================================================================
# BALLISTIC PROPAGATION (REFERENCE)
# =============================================================================

def propagate_ballistic(x0, tf, n_eval=2000):
    """Propagate uncontrolled from LEO for comparison."""
    sol = propagate(x0, (0, tf), max_step=tf/n_eval)
    t_eval = np.linspace(0, tf, n_eval)
    states = sol.sol(t_eval).T
    return t_eval, states


# =============================================================================
# METHOD 1: IPOPT DIRECT MULTIPLE-SHOOTING
# =============================================================================

def solve_ipopt_transfer(x0, xf, tf, n_seg=60, n_rk=4, warm_X=None, warm_U=None,
                         u_max=None, verbose=True):
    """
    Solve minimum-energy LEO→NRHO transfer via CasADi/IPOPT multiple-shooting.

    Dynamics: 3D CR3BP with low-thrust control u = [ux, uy, uz].
    Defects: RK4 integration within each segment.
    Objective: min ∫|u|² dt  (minimum energy)

    Args:
        x0, xf: initial and final states (6,)
        tf: transfer time (nondim)
        n_seg: number of shooting segments
        n_rk: RK4 sub-steps per segment
        warm_X: (6, n_seg+1) warm-start states
        warm_U: (3, n_seg) warm-start controls
        u_max: control magnitude bound (None = unbounded)
        verbose: print IPOPT output

    Returns:
        dict with solution data, or None if failed
    """
    if not HAS_CASADI:
        print("  CasADi not available — skipping IPOPT")
        return None

    t_start = timer.time()

    ns, nd = 6, 3  # state and control dimensions
    dt_seg = tf / n_seg

    # CasADi symbolic dynamics
    x_sym = ca.MX.sym('x', ns)
    u_sym = ca.MX.sym('u', nd)

    r1 = ca.sqrt((x_sym[0] + MU)**2 + x_sym[1]**2 + x_sym[2]**2)
    r2 = ca.sqrt((x_sym[0] - 1.0 + MU)**2 + x_sym[1]**2 + x_sym[2]**2)

    Ux = x_sym[0] - (1-MU)*(x_sym[0]+MU)/r1**3 - MU*(x_sym[0]-1+MU)/r2**3
    Uy = x_sym[1] - (1-MU)*x_sym[1]/r1**3 - MU*x_sym[1]/r2**3
    Uz = -(1-MU)*x_sym[2]/r1**3 - MU*x_sym[2]/r2**3

    xdot = ca.vertcat(
        x_sym[3], x_sym[4], x_sym[5],
        2*x_sym[4] + Ux + u_sym[0],
        -2*x_sym[3] + Uy + u_sym[1],
        Uz + u_sym[2]
    )

    f_dyn = ca.Function('f', [x_sym, u_sym], [xdot])

    # RK4 integrator for one segment
    def rk4_step(x, u, h):
        k1 = f_dyn(x, u)
        k2 = f_dyn(x + 0.5*h*k1, u)
        k3 = f_dyn(x + 0.5*h*k2, u)
        k4 = f_dyn(x + h*k3, u)
        return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # Build NLP
    opti = ca.Opti()
    X = opti.variable(ns, n_seg + 1)
    U = opti.variable(nd, n_seg)

    # Boundary conditions
    opti.subject_to(X[:, 0] == x0)
    opti.subject_to(X[:, -1] == xf)

    # Control bounds (if specified)
    if u_max is not None:
        for k in range(n_seg):
            for d in range(nd):
                opti.subject_to(opti.bounded(-u_max, U[d, k], u_max))

    # Objective: minimum energy
    J = 0
    for k in range(n_seg):
        J += ca.dot(U[:, k], U[:, k]) * dt_seg

    opti.minimize(J)

    # RK4 defect constraints
    h_rk = dt_seg / n_rk
    for k in range(n_seg):
        xk = X[:, k]
        uk = U[:, k]
        x_next = xk
        for s in range(n_rk):
            x_next = rk4_step(x_next, uk, h_rk)
        opti.subject_to(X[:, k+1] == x_next)

    # Initial guess
    if warm_X is not None:
        # Warm-start from previous solution (mesh refinement)
        if warm_X.shape[1] != n_seg + 1:
            # Interpolate to new mesh
            t_old = np.linspace(0, tf, warm_X.shape[1])
            t_new = np.linspace(0, tf, n_seg + 1)
            X_interp = interp1d(t_old, warm_X, axis=1, fill_value='extrapolate')
            warm_X_new = X_interp(t_new)
            for i in range(n_seg + 1):
                opti.set_initial(X[:, i], warm_X_new[:, i])
        else:
            for i in range(n_seg + 1):
                opti.set_initial(X[:, i], warm_X[:, i])

        if warm_U is not None:
            if warm_U.shape[1] != n_seg:
                t_old_u = np.linspace(0, tf, warm_U.shape[1])
                t_new_u = np.linspace(0, tf, n_seg)
                U_interp = interp1d(t_old_u, warm_U, axis=1, fill_value='extrapolate')
                warm_U_new = U_interp(t_new_u)
                for i in range(n_seg):
                    opti.set_initial(U[:, i], warm_U_new[:, i])
            else:
                for i in range(n_seg):
                    opti.set_initial(U[:, i], warm_U[:, i])
    else:
        # Linear interpolation
        for i in range(n_seg + 1):
            alpha = i / n_seg
            opti.set_initial(X[:, i], (1-alpha) * x0 + alpha * xf)
        opti.set_initial(U, 0)

    # Solver options
    ipopt_opts = {
        'print_level': 5 if verbose else 0,
        'max_iter': 500,
        'tol': 1e-6,
        'acceptable_tol': 1e-4,
        'acceptable_iter': 10,
        'linear_solver': 'mumps',
        'warm_start_init_point': 'yes' if warm_X is not None else 'no',
    }
    opts = {'ipopt': ipopt_opts, 'print_time': False}
    opti.solver('ipopt', opts)

    try:
        sol = opti.solve()
        X_sol = np.array(sol.value(X))
        U_sol = np.array(sol.value(U))
        J_val = float(sol.value(J))
        status = 'converged'
    except RuntimeError as e:
        # Try extracting the debug solution
        try:
            X_sol = np.array(opti.debug.value(X))
            U_sol = np.array(opti.debug.value(U))
            J_val = float(opti.debug.value(J))
            status = f'failed ({e})'
        except:
            return None

    elapsed = timer.time() - t_start

    # Compute diagnostics
    u_mag = np.linalg.norm(U_sol, axis=0)
    t_nodes = np.linspace(0, tf, n_seg + 1)
    t_ctrl = np.linspace(0, tf, n_seg)

    # Delta-v (approximate: sum of |u| * dt)
    dv_nondim = np.sum(u_mag) * dt_seg
    dv_kms = dv_nondim * V_STAR

    # Jacobi at each node
    J_nodes = np.array([jacobi_constant(X_sol[:, i]) for i in range(n_seg + 1)])

    result = {
        'X': X_sol,
        'U': U_sol,
        't_nodes': t_nodes,
        't_ctrl': t_ctrl,
        'J_cost': J_val,
        'dv_nondim': dv_nondim,
        'dv_kms': dv_kms,
        'u_max': u_mag.max(),
        'u_mean': u_mag.mean(),
        'jacobi': J_nodes,
        'n_seg': n_seg,
        'n_rk': n_rk,
        'status': status,
        'elapsed_s': elapsed,
    }

    if verbose:
        print(f"\n  IPOPT Result ({n_seg} segments, {n_rk} RK4 steps):")
        print(f"    Status:       {status}")
        print(f"    Objective J:  {J_val:.6e}")
        print(f"    Δv ≈ {dv_kms:.4f} km/s ({dv_nondim:.6f} nondim)")
        print(f"    Max |u|:      {u_mag.max():.6e} nondim = {u_mag.max()*V_STAR/T_STAR*1000:.4f} m/s²")
        print(f"    Wall time:    {elapsed:.1f} s")

    return result


def solve_ipopt_cascade(x0, xf, tf, verbose=True):
    """
    Mesh refinement cascade: coarse → medium → fine.

    Each stage warm-starts from the previous converged solution.
    """
    if not HAS_CASADI:
        print("  CasADi not available")
        return None

    print("\n" + "=" * 70)
    print("METHOD 1: IPOPT DIRECT MULTIPLE-SHOOTING (MESH CASCADE)")
    print("=" * 70)

    stages = [
        {'n_seg': 30,  'n_rk': 4, 'label': 'Coarse (30 seg, 4 RK4)'},
        {'n_seg': 60,  'n_rk': 4, 'label': 'Fine   (60 seg, 4 RK4)'},
    ]

    warm_X, warm_U = None, None
    result = None

    for i, stage in enumerate(stages):
        print(f"\n  --- Stage {i+1}: {stage['label']} ---")
        result = solve_ipopt_transfer(
            x0, xf, tf,
            n_seg=stage['n_seg'], n_rk=stage['n_rk'],
            warm_X=warm_X, warm_U=warm_U,
            verbose=verbose
        )

        if result is None:
            print(f"  Stage {i+1} failed — stopping cascade")
            break

        if 'converged' in result['status']:
            warm_X = result['X']
            warm_U = result['U']
            print(f"  Stage {i+1} converged — Δv = {result['dv_kms']:.4f} km/s")
        else:
            print(f"  Stage {i+1}: {result['status']} — using as warm start anyway")
            warm_X = result['X']
            warm_U = result['U']

    return result


# =============================================================================
# METHOD 2: INDIRECT SHOOTING
# =============================================================================

def shooting_residual(lam0, x0, xf, tf):
    """
    Compute shooting residual ||x(tf) - xf|| for 12D state+costate system.

    Uses cr3bp_controlled_ode from cr3bp_3d.py:
      state + costate ODE with u* = -½ λv

    Uses moderate integration accuracy for speed during fsolve.
    Large costate values can make the dynamics stiff, so we cap max_step.
    """
    X0 = np.concatenate([x0, lam0])

    try:
        sol = solve_ivp(
            cr3bp_controlled_ode, [0, tf], X0,
            method='RK45', rtol=1e-6, atol=1e-8,
            max_step=tf / 50, first_step=tf / 200
        )
        if sol.status != 0:
            return np.ones(6) * 1e6
        x_final = sol.y[:6, -1]
        res = x_final - xf
        # Check for NaN/Inf
        if not np.all(np.isfinite(res)):
            return np.ones(6) * 1e6
        return res
    except:
        return np.ones(6) * 1e6


def solve_indirect_shooting(x0, xf, tf, lam0_guesses=None, num_random=20, verbose=True,
                             max_time_s=180):
    """
    Solve LEO→NRHO transfer via indirect shooting.

    Unknowns: initial costate λ₀ (6 components)
    Target: x(tf) = xf  (6 equations, 6 unknowns)

    Initial guess strategies:
      1. From IPOPT solution (if provided): u* = -½λv → λv = -2u
      2. Physics-based: align costate with velocity difference
      3. Random perturbations

    NOTE: Indirect shooting on the full LEO→NRHO problem is extremely
    sensitive to initial costate guesses. Even with IPOPT warm-starting,
    convergence is not guaranteed — this is a known limitation of indirect
    methods for long-duration transfers in the CR3BP.

    Returns:
        dict with solution data, or None if failed
    """
    print("\n" + "=" * 70)
    print("METHOD 2: INDIRECT SHOOTING (PONTRYAGIN)")
    print("=" * 70)

    t_start = timer.time()

    best_sol = None
    best_residual = np.inf

    # Build guess list
    guesses = []

    # IPOPT-derived guesses (if provided)
    if lam0_guesses is not None:
        for g in lam0_guesses:
            guesses.append(('IPOPT-derived', g))

    # Physics-based guess: align velocity costate with Δv direction
    dv = xf[3:6] - x0[3:6]
    dv_norm = np.linalg.norm(dv) + 1e-10
    guesses.append(('physics (Δv-aligned)',
                     0.1 * np.concatenate([np.zeros(3), dv / dv_norm])))
    guesses.append(('physics (pos-aligned)',
                     0.01 * np.concatenate([(xf[:3] - x0[:3]) / np.linalg.norm(xf[:3] - x0[:3] + 1e-10), np.zeros(3)])))

    # Random guesses (limited — each fsolve call costs ~30-60s due to 12D propagation)
    for j in range(min(num_random, 5)):
        scale = 0.1 * (10 ** (j // 2))  # vary scale: 0.1, 1.0, 10
        guesses.append((f'random #{j+1}', np.random.randn(6) * scale))

    total = len(guesses)
    print(f"\n  Trying {total} initial costate guesses...")

    # Track evaluation count to limit total work per fsolve call
    _eval_count = [0]
    _max_evals_per_guess = 200  # total function calls including Jacobian

    def _counted_residual(lam):
        _eval_count[0] += 1
        if _eval_count[0] > _max_evals_per_guess:
            return np.ones(6) * 1e6  # force termination
        return shooting_residual(lam, x0, xf, tf)

    shooting_start = timer.time()
    for attempt, (label, lam0_guess) in enumerate(guesses):
        # Check time budget
        elapsed_total = timer.time() - shooting_start
        if elapsed_total > max_time_s:
            print(f"    Time budget exceeded ({elapsed_total:.0f}s > {max_time_s}s). "
                  f"Stopped after {attempt} of {total} guesses.")
            break

        t_guess_start = timer.time()
        _eval_count[0] = 0
        try:
            result = fsolve(
                _counted_residual,
                lam0_guess, full_output=True, maxfev=50
            )
            lam0_sol = result[0]
            info_dict = result[1]
            residual_norm = np.linalg.norm(info_dict['fvec'])
            t_guess_elapsed = timer.time() - t_guess_start

            print(f"    [{attempt+1:3d}/{total}] {label:25s}: residual = {residual_norm:.4e} "
                  f"({t_guess_elapsed:.1f}s, {_eval_count[0]} evals)")

            if residual_norm < best_residual:
                best_residual = residual_norm
                best_sol = {'lam0': lam0_sol, 'residual': residual_norm, 'label': label}

            # Early termination if we found a good solution
            if best_residual < 1e-6:
                print(f"    Converged! Stopping early.")
                break

        except Exception as e:
            t_guess_elapsed = timer.time() - t_guess_start
            print(f"    [{attempt+1:3d}/{total}] {label:25s}: FAILED ({t_guess_elapsed:.1f}s)")

    elapsed = timer.time() - t_start

    if best_sol is None or best_residual > 1e-2:
        print(f"\n  Shooting FAILED: best residual = {best_residual:.4e}")
        return None

    print(f"\n  Best result: {best_sol['label']}")
    print(f"    Residual = {best_residual:.4e}")

    # Propagate the converged solution for the full trajectory
    lam0_sol = best_sol['lam0']
    X0_full = np.concatenate([x0, lam0_sol])
    sol = solve_ivp(
        cr3bp_controlled_ode, [0, tf], X0_full,
        method='RK45', rtol=1e-10, atol=1e-12,
        max_step=tf / 2000, dense_output=True
    )

    n_eval = 2000
    t_eval = np.linspace(0, tf, n_eval)
    Y_eval = sol.sol(t_eval)
    states = Y_eval[:6, :].T    # (n_eval, 6)
    costates = Y_eval[6:12, :].T  # (n_eval, 6)

    # Control: u* = -½ λv
    controls = -0.5 * costates[:, 3:6]  # (n_eval, 3)
    u_mag = np.linalg.norm(controls, axis=1)

    # Delta-v
    dt = tf / n_eval
    dv_nondim = np.trapz(u_mag, t_eval)
    dv_kms = dv_nondim * V_STAR

    # Jacobi
    J_traj = np.array([jacobi_constant(states[i]) for i in range(n_eval)])

    result = {
        'states': states,
        'costates': costates,
        'controls': controls,
        't_eval': t_eval,
        'lam0': lam0_sol,
        'residual': best_residual,
        'dv_nondim': dv_nondim,
        'dv_kms': dv_kms,
        'u_max': u_mag.max(),
        'u_mean': u_mag.mean(),
        'jacobi': J_traj,
        'status': 'converged' if best_residual < 1e-3 else 'marginal',
        'elapsed_s': elapsed,
    }

    if verbose:
        print(f"\n  Shooting Result:")
        print(f"    Status:       {result['status']}")
        print(f"    Residual:     {best_residual:.4e}")
        print(f"    Δv ≈ {dv_kms:.4f} km/s ({dv_nondim:.6f} nondim)")
        print(f"    Max |u|:      {u_mag.max():.6e} nondim")
        print(f"    Wall time:    {elapsed:.1f} s")

    return result


# =============================================================================
# PLOTTING (ARTEMIS 2 STYLE)
# =============================================================================

def plot_nrho_orbit(ax, n_periods=2, color='green', alpha=0.3, label='NRHO'):
    """Plot the NRHO orbit on a 3D axis for context."""
    s0 = nrho_state()
    T = nrho_period()
    sol = propagate(s0, (0, n_periods * T), max_step=0.001)
    t_plot = np.linspace(0, n_periods * T, 5000)
    states = sol.sol(t_plot)
    ax.plot(states[0], states[1], states[2], color=color, alpha=alpha,
            linewidth=1.0, label=label)


def plot_results(x0, xf, tf, info, ballistic, ipopt_result, shooting_result):
    """
    Generate Artemis 2-style diagnostic figures.

    Figures:
      1. 3D trajectory in CR3BP rotating frame
      2. 2D projections (XY, XZ, YZ)
      3. Control profiles comparison
      4. Summary statistics table
    """
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    # Unpack data
    t_bal, x_bal = ballistic

    # Colors
    c_bal = '#888888'
    c_ipopt = '#2196F3'
    c_shoot = '#F44336'
    c_nrho = '#4CAF50'
    c_earth = '#1565C0'
    c_moon = '#9E9E9E'

    # ----- Figure 1: 3D Trajectory -----
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    # NRHO orbit (2 periods for context)
    plot_nrho_orbit(ax, n_periods=2, color=c_nrho, alpha=0.4)

    # Ballistic
    ax.plot(x_bal[:, 0], x_bal[:, 1], x_bal[:, 2],
            color=c_bal, alpha=0.4, linewidth=1, label='Ballistic (no control)')

    # IPOPT
    if ipopt_result and 'converged' in ipopt_result['status']:
        X = ipopt_result['X']
        ax.plot(X[0], X[1], X[2],
                color=c_ipopt, linewidth=2.0, label=f"IPOPT (Δv={ipopt_result['dv_kms']:.3f} km/s)")

    # Shooting
    if shooting_result and shooting_result['status'] in ('converged', 'marginal'):
        s = shooting_result['states']
        ax.plot(s[:, 0], s[:, 1], s[:, 2],
                color=c_shoot, linewidth=1.5, linestyle='--',
                label=f"Shooting (Δv={shooting_result['dv_kms']:.3f} km/s)")

    # Earth and Moon
    ax.scatter(*EARTH_POS, color=c_earth, s=100, marker='o', label='Earth', zorder=5)
    ax.scatter(*MOON_POS, color=c_moon, s=60, marker='o', label='Moon', zorder=5)

    # Start and end markers
    ax.scatter(*x0[:3], color='lime', s=80, marker='^', label='LEO departure', zorder=5)
    ax.scatter(*xf[:3], color='red', s=80, marker='*', label='NRHO arrival', zorder=5)

    # L2 point
    _, xL2, _ = collinear_libration_points()
    ax.scatter(xL2, 0, 0, color='purple', s=40, marker='x', label='L2', zorder=5)

    ax.set_xlabel('x (nondim)')
    ax.set_ylabel('y (nondim)')
    ax.set_zlabel('z (nondim)')
    ax.set_title('LEO → NRHO Transfer (3D CR3BP Rotating Frame)')
    ax.legend(loc='upper left', fontsize=10, facecolor='white', edgecolor='black', labelcolor='black')

    fig.savefig(os.path.join(OUTDIR, '3d_trajectory.png'), dpi=150, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close(fig)
    print("  Saved 3d_trajectory.png")

    # ----- Figure 2: 2D Projections -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    projections = [
        (0, 1, 'x', 'y', 'XY Projection (Earth-Moon plane)'),
        (0, 2, 'x', 'z', 'XZ Projection'),
        (1, 2, 'y', 'z', 'YZ Projection'),
    ]

    for ax, (i, j, xl, yl, title) in zip(axes, projections):
        # NRHO
        s0 = nrho_state()
        T = nrho_period()
        sol_n = propagate(s0, (0, 2*T), max_step=0.001)
        t_n = np.linspace(0, 2*T, 5000)
        sn = sol_n.sol(t_n)
        ax.plot(sn[i], sn[j], color=c_nrho, alpha=0.4, linewidth=1, label='NRHO')

        # Ballistic
        ax.plot(x_bal[:, i], x_bal[:, j], color=c_bal, alpha=0.3, linewidth=1, label='Ballistic')

        # IPOPT
        if ipopt_result and 'converged' in ipopt_result['status']:
            X = ipopt_result['X']
            ax.plot(X[i], X[j], color=c_ipopt, linewidth=2.0, label='IPOPT')

        # Shooting
        if shooting_result and shooting_result['status'] in ('converged', 'marginal'):
            s = shooting_result['states']
            ax.plot(s[:, i], s[:, j], color=c_shoot, linewidth=1.5, linestyle='--', label='Shooting')

        # Bodies
        ax.plot(EARTH_POS[i], EARTH_POS[j], 'o', color=c_earth, markersize=8)
        ax.plot(MOON_POS[i], MOON_POS[j], 'o', color=c_moon, markersize=6)
        ax.plot(x0[i], x0[j], '^', color='lime', markersize=8)
        ax.plot(xf[i], xf[j], '*', color='red', markersize=10)

        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True, facecolor='white', edgecolor='black', labelcolor='black', fontsize=9)

    fig.suptitle('LEO → NRHO Transfer — 2D Projections', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, '2d_projections.png'), dpi=150, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close(fig)
    print("  Saved 2d_projections.png")

    # ----- Figure 3: Control Profile -----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')

    t_scale = T_STAR / 86400.0  # nondim → days

    # Control magnitude
    ax = axes[0, 0]
    if ipopt_result and 'converged' in ipopt_result['status']:
        u_mag_ip = np.linalg.norm(ipopt_result['U'], axis=0)
        t_ctrl_days = ipopt_result['t_ctrl'] * t_scale
        ax.semilogy(t_ctrl_days, u_mag_ip * V_STAR * 1000 / T_STAR,
                     color=c_ipopt, linewidth=1.5, label='IPOPT')

    if shooting_result and shooting_result['status'] in ('converged', 'marginal'):
        u_mag_sh = np.linalg.norm(shooting_result['controls'], axis=1)
        t_sh_days = shooting_result['t_eval'] * t_scale
        ax.semilogy(t_sh_days, u_mag_sh * V_STAR * 1000 / T_STAR,
                     color=c_shoot, linewidth=1.0, linestyle='--', label='Shooting')

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('|u| (m/s²)')
    ax.set_title('Control Magnitude')
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
    ax.grid(True, alpha=0.3)

    # Control components (IPOPT)
    ax = axes[0, 1]
    if ipopt_result and 'converged' in ipopt_result['status']:
        U = ipopt_result['U']
        t_ctrl_days = ipopt_result['t_ctrl'] * t_scale
        for d, label in enumerate(['ux', 'uy', 'uz']):
            ax.plot(t_ctrl_days, U[d] * V_STAR * 1000 / T_STAR, linewidth=1.0, label=label)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Control (m/s²)')
    ax.set_title('IPOPT Control Components')
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
    ax.grid(True, alpha=0.3)

    # Jacobi constant
    ax = axes[1, 0]
    if ipopt_result and 'converged' in ipopt_result['status']:
        t_nodes_days = ipopt_result['t_nodes'] * t_scale
        ax.plot(t_nodes_days, ipopt_result['jacobi'], color=c_ipopt, label='IPOPT')
    if shooting_result and shooting_result['status'] in ('converged', 'marginal'):
        t_sh_days = shooting_result['t_eval'] * t_scale
        ax.plot(t_sh_days, shooting_result['jacobi'], color=c_shoot, linestyle='--', label='Shooting')
    ax.axhline(jacobi_constant(x0), color='lime', linestyle=':', alpha=0.5, label=f"LEO C={jacobi_constant(x0):.1f}")
    ax.axhline(jacobi_constant(xf), color='red', linestyle=':', alpha=0.5, label=f"NRHO C={jacobi_constant(xf):.4f}")
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Jacobi Constant C')
    ax.set_title('Jacobi Constant Along Transfer')
    ax.legend(fontsize=7, facecolor='white', edgecolor='black', labelcolor='black')
    ax.grid(True, alpha=0.3)

    # Distance from Moon
    ax = axes[1, 1]
    t_bal_days = t_bal * t_scale
    d_moon_bal = np.array([dist_from_moon(x_bal[i]) * L_STAR for i in range(len(t_bal))])
    ax.plot(t_bal_days, d_moon_bal, color=c_bal, alpha=0.4, label='Ballistic')
    if ipopt_result and 'converged' in ipopt_result['status']:
        X = ipopt_result['X']
        d_moon_ip = np.array([dist_from_moon(X[:, i]) * L_STAR for i in range(X.shape[1])])
        t_ip_days = ipopt_result['t_nodes'] * t_scale
        ax.plot(t_ip_days, d_moon_ip, color=c_ipopt, label='IPOPT')
    if shooting_result and shooting_result['status'] in ('converged', 'marginal'):
        s = shooting_result['states']
        d_moon_sh = np.array([dist_from_moon(s[i]) * L_STAR for i in range(len(s))])
        t_sh_days = shooting_result['t_eval'] * t_scale
        ax.plot(t_sh_days, d_moon_sh, color=c_shoot, linestyle='--', label='Shooting')
    ax.axhline(NRHO_9_2['apolune_km'], color=c_nrho, linestyle=':', alpha=0.5, label='NRHO apolune')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Distance from Moon (km)')
    ax.set_title('Distance from Moon')
    ax.legend(fontsize=7, facecolor='white', edgecolor='black', labelcolor='black')
    ax.grid(True, alpha=0.3)

    fig.suptitle('LEO → NRHO Transfer — Control and Diagnostics', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'control_profile.png'), dpi=150, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close(fig)
    print("  Saved control_profile.png")

    # ----- Figure 4: Summary Table -----
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.axis('off')

    if ipopt_result:
        ip_status = ipopt_result['status']
        ip_dv = f"{ipopt_result['dv_kms']:.4f} km/s"
        ip_J = f"{ipopt_result['J_cost']:.4e}"
        ip_time = f"{ipopt_result['elapsed_s']:.1f} s"
        ip_seg = f"{ipopt_result['n_seg']} seg × {ipopt_result['n_rk']} RK4"
    else:
        ip_status = ip_dv = ip_J = ip_time = ip_seg = 'N/A'

    if shooting_result:
        sh_status = shooting_result['status']
        sh_dv = f"{shooting_result['dv_kms']:.4f} km/s"
        sh_res = f"{shooting_result['residual']:.2e}"
        sh_time = f"{shooting_result['elapsed_s']:.1f} s"
    else:
        sh_status = 'FAILED (no convergence)'
        sh_dv = sh_res = sh_time = 'N/A'

    rows = [
        ['Parameter', 'IPOPT Collocation', 'Indirect Shooting'],
        ['Status', ip_status, sh_status],
        ['Δv', ip_dv, sh_dv],
        ['Objective / Residual', ip_J, sh_res],
        ['Wall Time', ip_time, sh_time],
        ['Discretization', ip_seg, '12D RK45 (continuous)'],
        ['', '', ''],
        ['Transfer Setup', 'Value', ''],
        ['LEO Altitude', f'{info["leo_alt_km"]:.0f} km', ''],
        ['LEO Inclination', '28.5°', ''],
        ['Transfer Time', f'{info["transfer_time_days"]:.1f} days', ''],
        ['NRHO Target', '9:2 L2 Southern', '(apolune insertion)'],
        ['NRHO Period', f'{nrho_period() * JPL_TUNIT / 86400:.2f} days', ''],
    ]

    table = ax.table(cellText=rows, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Header styling
    for i in [0, 7]:
        for j in range(3):
            cell = table[i, j]
            cell.set_facecolor('#E3F2FD')
            cell.set_text_props(weight='bold')

    ax.set_title('LEO → NRHO Transfer — Comparison Summary', fontsize=14, pad=20)
    fig.savefig(os.path.join(OUTDIR, 'summary_stats.png'), dpi=150, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close(fig)
    print("  Saved summary_stats.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("LEO → NRHO TRANSFER (3D CR3BP)")
    print("Bézier Collocation vs Indirect Shooting")
    print("=" * 70)
    print(f"\n  CasADi available: {HAS_CASADI}")
    print(f"  μ = {MU}, L* = {L_STAR} km, T* = {T_STAR:.2f} s")

    # --- NRHO validation (quick) ---
    sys.stdout.write("\n  Validating NRHO...")
    sys.stdout.flush()
    val = validate_nrho(verbose=False)
    print(f" Jacobi OK (|ΔC|={val['jacobi_error']:.1e}), "
          f"perilune={val['perilune_km']:.0f} km")

    # --- Setup transfer ---
    x0, xf, tf, info = setup_transfer(
        departure_nu_deg=0.0,
        transfer_time_days=5.0,
        verbose=True
    )

    # --- Ballistic reference ---
    sys.stdout.write("\n  Propagating ballistic reference...")
    sys.stdout.flush()
    _t0 = timer.time()
    t_bal, x_bal = propagate_ballistic(x0, tf, n_eval=500)
    print(f" done in {timer.time()-_t0:.1f}s (final dist from Moon = {dist_from_moon(x_bal[-1]) * L_STAR:.0f} km)")
    ballistic = (t_bal, x_bal)

    # --- Method 1: IPOPT ---
    ipopt_result = solve_ipopt_cascade(x0, xf, tf, verbose=True)

    # --- Method 2: Indirect Shooting ---
    # Build initial guesses from IPOPT if available
    lam0_guesses = []
    if ipopt_result is not None and 'converged' in ipopt_result['status']:
        # Extract approximate costate from IPOPT control: u = -½λv → λv = -2u
        U = ipopt_result['U']
        # Use control at t=0 and t=tf/2 as costate hints
        u_start = U[:, 0]
        u_mid = U[:, U.shape[1]//2]
        u_end = U[:, -1]

        # λv = -2u, λr is harder to estimate — use small values
        for u_sample in [u_start, u_mid]:
            lam_v = -2.0 * u_sample
            lam_r = 0.01 * np.random.randn(3)
            lam0_guesses.append(np.concatenate([lam_r, lam_v]))

    shooting_result = solve_indirect_shooting(
        x0, xf, tf,
        lam0_guesses=lam0_guesses if lam0_guesses else None,
        num_random=3,
        max_time_s=30,
        verbose=True
    )

    # --- Plots ---
    plot_results(x0, xf, tf, info, ballistic, ipopt_result, shooting_result)

    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Transfer: LEO (185 km, 28.5°) → 9:2 NRHO (apolune)")
    print(f"  Duration: {info['transfer_time_days']:.1f} days")

    if ipopt_result:
        print(f"\n  IPOPT:    Δv = {ipopt_result['dv_kms']:.4f} km/s  "
              f"({ipopt_result['status']}, {ipopt_result['elapsed_s']:.1f} s)")
    if shooting_result:
        print(f"  Shooting: Δv = {shooting_result['dv_kms']:.4f} km/s  "
              f"({shooting_result['status']}, {shooting_result['elapsed_s']:.1f} s)")

    print(f"\n  Output directory: {OUTDIR}")
    print("  Files: 3d_trajectory.png, 2d_projections.png, control_profile.png, summary_stats.png")

    return ipopt_result, shooting_result


if __name__ == '__main__':
    ipopt_result, shooting_result = main()
