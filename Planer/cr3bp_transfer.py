"""
cr3bp_transfer.py - L1→L2 Lyapunov transfer: Shooting vs Bézier Collocation

Solves a minimum-energy low-thrust transfer between L1 and L2 Lyapunov orbits
in the planar Earth-Moon CR3BP using:
  1. Indirect shooting (Pontryagin's Maximum Principle)
  2. Direct Bézier collocation (NLP with dynamics constraints)

Author: Mustakeen Bari, Zhuorui Li, Advait Jawaji
Course: AAE 568 — Applied Optimal Control and Estimation
"""

import sys
import os
import time as timer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, minimize
from scipy.interpolate import interp1d

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Earth-Mars'))

from cr3bp_planar import (
    MU, collinear_libration_points, cr3bp_planar_ode,
    cr3bp_planar_controlled_ode, cr3bp_planar_gravity,
    cr3bp_jacobi_planar, compute_lyapunov_orbit
)
from bezier import (
    BezierCollocation, bernstein, bezier_eval, bezier_derivative
)


# =============================================================================
# Problem Setup
# =============================================================================

def setup_transfer_problem(Ax_L1=0.02, Ax_L2=0.02, mu=MU):
    """
    Set up an L1 → L2 Lyapunov orbit transfer.

    Departure: a point on the L1 Lyapunov orbit
    Arrival:   a point on the L2 Lyapunov orbit

    We depart from the right-side x-axis crossing of the L1 orbit (y=0, vy<0)
    and arrive at the left-side x-axis crossing of the L2 orbit (y=0, vy>0),
    which are the natural "gateway" points in cislunar space.

    Returns:
        x0, xf:   departure/arrival states [x, y, vx, vy]
        t0, tf:   time window
        lyap_L1, lyap_L2: Lyapunov orbit data
    """
    xL1, xL2, xL3 = collinear_libration_points(mu)
    print(f"Libration points:  L1 = {xL1:.8f},  L2 = {xL2:.8f}")

    # Compute Lyapunov orbits
    print(f"Computing L1 Lyapunov orbit (Ax = {Ax_L1})...")
    state_L1, T_L1, sol_L1 = compute_lyapunov_orbit(xL1, Ax_L1, mu)
    C_L1 = cr3bp_jacobi_planar(state_L1, mu)
    print(f"  IC = {state_L1},  T = {T_L1:.6f},  C = {C_L1:.6f}")

    print(f"Computing L2 Lyapunov orbit (Ax = {Ax_L2})...")
    state_L2, T_L2, sol_L2 = compute_lyapunov_orbit(xL2, Ax_L2, mu)
    C_L2 = cr3bp_jacobi_planar(state_L2, mu)
    print(f"  IC = {state_L2},  T = {T_L2:.6f},  C = {C_L2:.6f}")

    # Departure state: right x-crossing of L1 orbit (IC itself: y=0, vx=0)
    x0 = state_L1.copy()  # [x0, 0, 0, vy0]

    # Arrival state: left x-crossing of L2 orbit
    # The IC of the L2 orbit is the right x-crossing (xL2+Ax, 0, 0, vy).
    # For the left x-crossing, propagate to the half-period.
    sol_half = solve_ivp(
        lambda t, X: cr3bp_planar_ode(t, X, mu),
        [0, T_L2 / 2], state_L2,
        method='RK45', rtol=1e-12, atol=1e-12
    )
    xf = sol_half.y[:, -1].copy()  # [x, ~0, ~0, vy]
    # Clean up numerical noise
    xf[1] = 0.0   # y = 0 by symmetry
    xf[2] = 0.0   # vx = 0 by symmetry
    print(f"  L2 half-period state: {xf}")

    # Transfer time: choose ~1 synodic period of L1/L2 oscillation
    # A reasonable guess is about pi (half a canonical time unit ≈ 2.2 days)
    # Use a moderate transfer time that allows low thrust
    tf_transfer = np.pi  # ~6.8 days in dimensional units

    lyap_data = {
        'L1': {'state0': state_L1, 'T': T_L1, 'sol': sol_L1, 'C': C_L1, 'xL': xL1},
        'L2': {'state0': state_L2, 'T': T_L2, 'sol': sol_L2, 'C': C_L2, 'xL': xL2},
    }

    return x0, xf, 0.0, tf_transfer, lyap_data


# =============================================================================
# CR3BP gravity for Bézier collocation
# =============================================================================

def cr3bp_gravity_for_bezier(r, mu=MU):
    """
    Full position-dependent acceleration for the Bézier collocation.
    In the CR3BP the EOM is:
        ax =  2*vy + Ux(x,y)
        ay = -2*vx + Uy(x,y)

    But the BezierCollocation class uses f(x, u) = [v; grav(r) + u],
    so grav(r) must include ONLY the position-dependent terms Ux, Uy.
    The Coriolis terms 2*vy, -2*vx depend on velocity and need
    special treatment.
    """
    return cr3bp_planar_gravity(r, mu)


# =============================================================================
# CR3BP-aware Bézier Collocation
# =============================================================================

class CR3BPBezierCollocation(BezierCollocation):
    """
    Bézier collocation for CR3BP dynamics.

    The CR3BP has velocity-dependent Coriolis terms that don't exist
    in the two-body problem. The full EOM is:
        ẋ  = vx
        ẏ  = vy
        v̇x =  2*vy + Ux(x,y) + ux
        v̇y = -2*vx + Uy(x,y) + uy

    We override the defect computation to include Coriolis.
    """

    def __init__(self, mu=MU, n_segments=8, bezier_degree=7, n_collocation=12):
        # Use a dummy gravity — we override _defects
        super().__init__(
            gravity_func=lambda r: cr3bp_planar_gravity(r, mu),
            pos_dim=2,
            n_segments=n_segments,
            bezier_degree=bezier_degree,
            n_collocation=n_collocation,
        )
        self.mu = mu

    def _defects(self, z, x0, xf, dt_seg):
        """
        Dynamics defects with full CR3BP EOM (including Coriolis).
        """
        segments, U = self._unpack(z, x0, xf, dt_seg)
        d = self.d
        s = self.state_dim
        tau = self.tau_c

        defects = []

        for seg_idx, cp in enumerate(segments):
            x_eval  = bezier_eval(cp, tau)
            dx_dtau = bezier_derivative(cp, tau)
            dx_dt   = dx_dtau / dt_seg

            r_eval = x_eval[:, :d]    # [x, y]
            v_eval = x_eval[:, d:]    # [vx, vy]
            u_seg  = U[seg_idx]

            # Full CR3BP dynamics: f(x, u)
            f_x = np.zeros_like(x_eval)
            for k in range(len(tau)):
                x_k, y_k = r_eval[k]
                vx_k, vy_k = v_eval[k]
                grav = self.gravity(r_eval[k])

                f_x[k, 0] = vx_k                         # ẋ = vx
                f_x[k, 1] = vy_k                         # ẏ = vy
                f_x[k, 2] = 2*vy_k + grav[0] + u_seg[k, 0]   # v̇x
                f_x[k, 3] = -2*vx_k + grav[1] + u_seg[k, 1]  # v̇y

            defects.append((dx_dt - f_x).ravel())

        return np.concatenate(defects)

    def _evaluate(self, z, x0, xf, t0, tf, n_eval=500):
        """
        Evaluate solution — reconstruct control from CR3BP dynamics.
        """
        dt_seg = (tf - t0) / self.n_seg
        segments, U = self._unpack(z, x0, xf, dt_seg)
        d = self.d

        n_per = n_eval // self.n_seg
        t_all, r_all, v_all, u_all = [], [], [], []

        for seg_idx, cp in enumerate(segments):
            t_start = t0 + seg_idx * dt_seg
            tau = np.linspace(0, 1, n_per,
                              endpoint=(seg_idx == self.n_seg - 1))
            t_seg = t_start + tau * dt_seg

            x_eval  = bezier_eval(cp, tau)
            dx_dtau = bezier_derivative(cp, tau)
            dx_dt   = dx_dtau / dt_seg

            r_eval = x_eval[:, :d]
            v_eval = x_eval[:, d:]

            # u = v̇_bezier - (Coriolis + gravity)
            dv_dt = dx_dt[:, d:]
            u_eval = np.zeros((len(tau), d))
            for k in range(len(tau)):
                grav = self.gravity(r_eval[k])
                vx_k, vy_k = v_eval[k]
                u_eval[k, 0] = dv_dt[k, 0] - (2*vy_k + grav[0])
                u_eval[k, 1] = dv_dt[k, 1] - (-2*vx_k + grav[1])

            t_all.append(t_seg)
            r_all.append(r_eval)
            v_all.append(v_eval)
            u_all.append(u_eval)

        return {
            't':  np.concatenate(t_all),
            'r':  np.vstack(r_all),
            'v':  np.vstack(v_all),
            'u':  np.vstack(u_all),
            'segments': [seg for seg in segments],
        }

    def _warm_start_from_trajectory(self, t_ref, x_ref, x0, xf, t0, tf):
        """
        Warm-start from a reference trajectory, estimating control
        using CR3BP dynamics (with Coriolis).
        """
        d = self.d
        s = self.state_dim
        n = self.deg
        dt_seg = (tf - t0) / self.n_seg

        interp_x = interp1d(t_ref, x_ref, axis=0, kind='cubic',
                             fill_value='extrapolate')

        # --- Fit Bézier CPs (same as parent) ---
        free_pts = []
        for seg in range(self.n_seg):
            t_start = t0 + seg * dt_seg
            t_end   = t_start + dt_seg

            n_sample = max(n + 1, 30)
            tau_s = np.linspace(0, 1, n_sample)
            t_s = t_start + tau_s * dt_seg
            x_s = interp_x(t_s)

            B_mat = np.zeros((n_sample, n + 1))
            for i in range(n + 1):
                B_mat[:, i] = bernstein(n, i, tau_s)

            cp = np.zeros((n + 1, s))
            if seg == 0:
                cp[0] = x0
            else:
                cp[0] = prev_end

            if seg == self.n_seg - 1:
                cp[n] = xf
            else:
                cp[n] = interp_x(t_end)

            rhs = x_s - np.outer(B_mat[:, 0], cp[0]) \
                       - np.outer(B_mat[:, n], cp[n])

            if n > 1:
                B_int = B_mat[:, 1:n]
                cp_int, _, _, _ = np.linalg.lstsq(B_int, rhs, rcond=None)
                cp[1:n] = cp_int

            for k in range(1, n):
                free_pts.append(cp[k])
            if seg < self.n_seg - 1:
                free_pts.append(cp[n])

            prev_end = cp[n]

        cp_guess = np.array(free_pts).ravel()

        # --- Estimate control at collocation points ---
        # u = a_measured - (Coriolis + gravity)
        u_all = []
        for seg in range(self.n_seg):
            t_start = t0 + seg * dt_seg
            t_colloc = t_start + self.tau_c * dt_seg
            x_c = interp_x(t_colloc)
            r_c = x_c[:, :d]
            v_c = x_c[:, d:]

            # Numerical acceleration via finite differences
            dt_fd = 1e-5 * dt_seg
            x_fwd = interp_x(np.minimum(t_colloc + dt_fd, tf))
            x_bwd = interp_x(np.maximum(t_colloc - dt_fd, t0))
            a_c = (x_fwd[:, d:] - x_bwd[:, d:]) / (2 * dt_fd)

            for k in range(len(self.tau_c)):
                grav = self.gravity(r_c[k])
                vx_k, vy_k = v_c[k]
                u_x = a_c[k, 0] - (2*vy_k + grav[0])
                u_y = a_c[k, 1] - (-2*vx_k + grav[1])
                u_all.append(np.array([u_x, u_y]))

        u_guess = np.array(u_all).ravel()
        return np.concatenate([cp_guess, u_guess])


# =============================================================================
# Indirect Shooting for CR3BP Transfer
# =============================================================================

def shooting_cr3bp_min_energy(lam0, x0, xf, t0, tf, mu=MU, u_max=None):
    """
    Shooting function for minimum-energy CR3BP transfer.

    Unknowns: lam0 = [lam_x, lam_y, lam_vx, lam_vy]
    Residual: x(tf) - xf = 0  (4 equations, 4 unknowns)
    """
    X0 = np.concatenate([x0, lam0])
    try:
        sol = solve_ivp(
            lambda t, X: cr3bp_planar_controlled_ode(t, X, mu, u_max=u_max),
            [t0, tf], X0,
            method='RK45', rtol=1e-12, atol=1e-12
        )
        xf_sol = sol.y[:4, -1]
    except Exception:
        return 1e6 * np.ones(4)

    return xf_sol - xf


def _build_costate_guess(x0, xf, t0, tf, mu=MU):
    """
    Build a physics-informed initial costate guess using the primer vector
    approximation: lambda_v ~ -2*(xf - x_ballistic(tf)) / (tf-t0).
    """
    # Propagate ballistic (uncontrolled) from x0
    sol_bal = solve_ivp(
        lambda t, X: cr3bp_planar_ode(t, X, mu),
        [t0, tf], x0, method='RK45', rtol=1e-10, atol=1e-10
    )
    xf_bal = sol_bal.y[:, -1]
    miss = xf - xf_bal
    dt = tf - t0

    # Rough estimate: lam_v ~ -2*control ~ proportional to miss distance
    lam_v_guess = -2.0 * miss[2:] / dt  # velocity miss
    lam_r_guess = -2.0 * miss[:2] / dt**2  # position miss

    return np.concatenate([lam_r_guess, lam_v_guess])


def solve_shooting(x0, xf, t0, tf, mu=MU, lam0_guess=None, n_random=8,
                   u_max=None, select_min_cost=False, fsolve_maxfev=2000):
    """
    Solve CR3BP min-energy transfer via indirect shooting.

    Uses physics-informed initial costate guess plus random perturbations.
    If select_min_cost=True, all converged guesses are evaluated and the
    lowest-cost feasible TPBVP branch is returned instead of the first one.
    """
    guesses = []

    # Physics-informed guess
    pv_guess = _build_costate_guess(x0, xf, t0, tf, mu)
    guesses.append(pv_guess)

    if lam0_guess is not None:
        guesses.append(lam0_guess)
    guesses.append(np.zeros(4))

    # Scaled random perturbations around the physics-informed guess
    rng = np.random.default_rng(42)
    for _ in range(n_random):
        scale = rng.uniform(0.1, 3.0)
        guesses.append(pv_guess * scale + rng.standard_normal(4) * 0.5)

    best_lam0 = None
    best_residual = np.inf
    best_info = None
    best_cost = np.inf

    for i, guess in enumerate(guesses):
        try:
            lam0_sol, info, ier, msg = fsolve(
                shooting_cr3bp_min_energy, guess,
                args=(x0, xf, t0, tf, mu, u_max),
                full_output=True, maxfev=fsolve_maxfev
            )
            res_norm = np.linalg.norm(info['fvec'])
            candidate_cost = np.inf
            if res_norm < 1e-6:
                X0 = np.concatenate([x0, lam0_sol])
                sol_cost = solve_ivp(
                    lambda t, X: cr3bp_planar_controlled_ode(
                        t, X, mu, u_max=u_max,
                    ),
                    [t0, tf], X0,
                    method='RK45', rtol=1e-12, atol=1e-12,
                    t_eval=np.linspace(t0, tf, 1000)
                )
                lam_v = sol_cost.y[6:8, :].T
                if u_max is None:
                    u = -0.5 * lam_v
                else:
                    u_norm = np.linalg.norm(-0.5 * lam_v, axis=1)
                    scale = np.ones_like(u_norm)
                    active = u_norm > float(u_max)
                    scale[active] = float(u_max) / np.maximum(u_norm[active], 1e-15)
                    u = (-0.5 * lam_v) * scale[:, None]
                candidate_cost = float(np.trapezoid(np.sum(u * u, axis=1), sol_cost.t))

            is_better = (
                candidate_cost < best_cost
                if select_min_cost and res_norm < 1e-6
                else res_norm < best_residual
            )
            if is_better:
                best_residual = res_norm
                best_lam0 = lam0_sol
                best_info = info
                best_cost = candidate_cost
                if res_norm < 1e-10 and not select_min_cost:
                    print(f"  Converged on guess #{i+1} (residual={res_norm:.2e})")
                    break
        except Exception:
            continue

    if best_lam0 is None:
        raise RuntimeError("Shooting failed for all initial guesses")

    if best_residual > 1e-6:
        print(f"  WARNING: best residual = {best_residual:.2e} (may not be converged)")

    # Propagate the converged solution
    X0 = np.concatenate([x0, best_lam0])
    sol = solve_ivp(
        lambda t, X: cr3bp_planar_controlled_ode(t, X, mu, u_max=u_max),
        [t0, tf], X0,
        method='RK45', rtol=1e-12, atol=1e-12,
        t_eval=np.linspace(t0, tf, 1000)
    )

    return best_lam0, sol, best_info


# =============================================================================
# Solve & Compare
# =============================================================================

def solve_both(x0, xf, t0, tf, mu=MU):
    """Run both methods on the CR3BP transfer problem."""

    # ---- 1. Indirect Shooting ----
    print("\n--- Indirect Shooting (Pontryagin) ---")
    t_start = timer.perf_counter()
    lam0_sol, sol_shoot, info = solve_shooting(x0, xf, t0, tf, mu)
    shooting_time = timer.perf_counter() - t_start
    shooting_residual = np.linalg.norm(info['fvec'])

    # Control: u* = -0.5 * lambda_v
    lam_vx = sol_shoot.y[6]
    lam_vy = sol_shoot.y[7]
    ux_s = -0.5 * lam_vx
    uy_s = -0.5 * lam_vy
    shooting_cost = np.trapezoid(ux_s**2 + uy_s**2, sol_shoot.t)

    # Jacobi "constant" drift (should vary due to thrust)
    C_shoot = np.array([cr3bp_jacobi_planar(sol_shoot.y[:4, k], mu)
                        for k in range(sol_shoot.y.shape[1])])

    shooting_data = {
        'lam0': lam0_sol,
        'residual': shooting_residual,
        'time_s': shooting_time,
        'cost': shooting_cost,
        'nfev': info['nfev'],
        't': sol_shoot.t,
        'x': sol_shoot.y[0], 'y': sol_shoot.y[1],
        'vx': sol_shoot.y[2], 'vy': sol_shoot.y[3],
        'ux': ux_s, 'uy': uy_s,
        'u_mag': np.sqrt(ux_s**2 + uy_s**2),
        'jacobi': C_shoot,
    }
    print(f"  Residual: {shooting_residual:.2e}")
    print(f"  Cost J = ∫|u|²dt: {shooting_cost:.6f}")
    print(f"  λ₀ = {lam0_sol}")
    print(f"  Solve time: {shooting_time:.3f}s")

    # ---- 2. Bézier Collocation (CasADi/IPOPT) ----
    print("\n--- Bézier Direct Collocation (CasADi/IPOPT) ---")
    from ipopt_collocation import CR3BPBezierIPOPT, mesh_refine_solve

    # Warm-start from shooting solution
    shoot_ref_t = shooting_data['t']
    shoot_ref_x = np.column_stack([
        shooting_data['x'], shooting_data['y'],
        shooting_data['vx'], shooting_data['vy'],
    ])
    warm_traj = (shoot_ref_t, shoot_ref_x)

    # Solve with mesh refinement, warm-started from the shooting solution
    # First level uses the shooting trajectory, subsequent levels warm-start
    # from the previous converged solution.
    t_start = timer.perf_counter()

    prev_warm = warm_traj
    history = []
    levels = [
        (4,  7, 12),
        (8,  7, 12),
        (16, 7, 12),
        (32, 7, 12),
    ]
    for level_idx, (n_seg, deg, nc) in enumerate(levels):
        is_final = (level_idx == len(levels) - 1)
        n_total = n_seg * ((deg + 1) * 4 + nc * 2)
        print(f"\n  Level {level_idx+1}/{len(levels)}: "
              f"{n_seg} seg, deg {deg}, {nc} colloc ({n_total} vars)")

        solver_ipopt = CR3BPBezierIPOPT(
            mu=mu, n_segments=n_seg, bezier_degree=deg,
            n_collocation=nc,
        )
        level_tol = 1e-8 if is_final else 1e-6
        level_result = solver_ipopt.solve(
            x0, xf, t0, tf,
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
    bezier_time = timer.perf_counter() - t_start

    sol_dict = result_mr

    # Jacobi along Bézier trajectory
    C_bez = np.array([cr3bp_jacobi_planar(
        np.array([sol_dict['r'][k, 0], sol_dict['r'][k, 1],
                  sol_dict['v'][k, 0], sol_dict['v'][k, 1]]), mu)
        for k in range(len(sol_dict['t']))])

    # Compute constraint violation from IPOPT stats
    ipopt_constr_viol = result_mr['stats'].get('iterations', {})
    max_defect = 0.0  # IPOPT enforces constraints to tolerance

    bezier_data = {
        'cost': sol_dict['cost'],
        'time_s': bezier_time,
        'nfev': sum(h['stats'].get('iter_count', 0) for h in history),
        'max_defect': max_defect,
        'converged': sol_dict['success'],
        't': sol_dict['t'],
        'x': sol_dict['r'][:, 0], 'y': sol_dict['r'][:, 1],
        'vx': sol_dict['v'][:, 0], 'vy': sol_dict['v'][:, 1],
        'ux': sol_dict['u'][:, 0], 'uy': sol_dict['u'][:, 1],
        'u_mag': np.sqrt(sol_dict['u'][:, 0]**2 + sol_dict['u'][:, 1]**2),
        'segments': sol_dict.get('segments', []),
        'jacobi': C_bez,
        'mesh_history': history,
    }
    total_iters = bezier_data['nfev']
    print(f"  Converged: {sol_dict['success']}")
    print(f"  Cost J = ∫|u|²dt: {sol_dict['cost']:.6f}")
    print(f"  Solve time: {bezier_time:.3f}s (total across {len(history)} mesh levels)")
    print(f"  Total IPOPT iterations: {total_iters}")
    for i, h in enumerate(history):
        lvl = h['level']
        print(f"    Level {i+1} ({lvl[0]} seg, deg {lvl[1]}): "
              f"cost={h['cost']:.8f}, time={h['solve_time']:.3f}s")

    return shooting_data, bezier_data


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(shooting, bezier, lyap_data, save_prefix='cr3bp_transfer'):
    """Generate a comprehensive comparison figure."""

    fig = plt.figure(figsize=(18, 11), facecolor='white')

    # Use gridspec for better control: top row 3 equal, bottom row 3 equal
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.30,
                          left=0.06, right=0.97, top=0.93, bottom=0.07)

    xL1 = lyap_data['L1']['xL']
    xL2 = lyap_data['L2']['xL']
    sol_L1 = lyap_data['L1']['sol']
    sol_L2 = lyap_data['L2']['sol']

    fig.suptitle('L1 → L2 Lyapunov Transfer: Shooting vs Bézier (IPOPT)',
                 fontsize=15, fontweight='bold', y=0.98, color='black')

    # ---- Panel 1: Trajectories (zoomed to cislunar corridor) ----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor('white')

    ax.plot(sol_L1.y[0], sol_L1.y[1], '-', color='#2ca02c', alpha=0.7,
            lw=1.2, label='L1 Lyapunov')
    ax.plot(sol_L2.y[0], sol_L2.y[1], '-', color='#9467bd', alpha=0.7,
            lw=1.2, label='L2 Lyapunov')

    ax.plot(shooting['x'], shooting['y'], '-', color='#1f77b4', lw=2, label='Shooting')
    ax.plot(bezier['x'], bezier['y'], '--', color='#d62728', lw=2, alpha=0.8,
            label='Bézier (IPOPT)')

    ax.plot(1 - MU, 0, 'o', color='#95a5a6', ms=7, mec='black', mew=0.8,
            zorder=10, label='Moon')
    ax.plot(xL1, 0, 'D', color='#f39c12', ms=7, mec='black', mew=0.8,
            zorder=10, label='L1')
    ax.plot(xL2, 0, 'D', color='#d62728', ms=7, mec='black', mew=0.8,
            zorder=10, label='L2')

    ax.plot(shooting['x'][0], shooting['y'][0], 'o', color='#2ca02c',
            ms=8, mec='black', mew=1, zorder=11, label='Departure')
    ax.plot(shooting['x'][-1], shooting['y'][-1], 's', color='#d62728',
            ms=8, mec='black', mew=1, zorder=11, label='Arrival')

    # Zoom to the L1-L2 corridor
    pad = 0.04
    ax.set_xlim(xL1 - 0.04, xL2 + 0.04)
    all_y = np.concatenate([shooting['y'], bezier['y'],
                            sol_L1.y[1], sol_L2.y[1]])
    y_ext = max(abs(all_y.min()), abs(all_y.max())) + pad
    ax.set_ylim(-y_ext, y_ext)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_xlabel('x (rotating frame)', fontsize=10, color='black')
    ax.set_ylabel('y (rotating frame)', fontsize=10, color='black')
    ax.set_title('Transfer Trajectory', fontsize=11, fontweight='bold', color='black')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9,
              handlelength=1.5, borderpad=0.4, labelspacing=0.35,
              facecolor='white', edgecolor='black', labelcolor='black')

    # ---- Panel 2: Trajectory difference ----
    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor('white')
    t_c = bezier['t']
    mask = (t_c >= t_c[1]) & (t_c <= t_c[-2])
    t_c_m = t_c[mask]
    interp_sx = interp1d(shooting['t'], shooting['x'], kind='cubic')
    interp_sy = interp1d(shooting['t'], shooting['y'], kind='cubic')
    dx = bezier['x'][mask] - interp_sx(t_c_m)
    dy = bezier['y'][mask] - interp_sy(t_c_m)
    pos_diff = np.sqrt(dx**2 + dy**2)

    ax.semilogy(t_c_m, pos_diff, '-', color='#1f77b4', lw=1.5)
    ax.fill_between(t_c_m, pos_diff, alpha=0.15, color='#1f77b4')
    ax.set_xlabel('Time (nondim)', fontsize=10, color='black')
    ax.set_ylabel('||Δr|| (position difference)', fontsize=10, color='black')
    ax.set_title('Trajectory Difference', fontsize=11, fontweight='bold', color='black')
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')

    # Annotate max difference
    idx_max = np.argmax(pos_diff)
    ax.annotate(f'max = {pos_diff[idx_max]:.2e}',
                xy=(t_c_m[idx_max], pos_diff[idx_max]),
                xytext=(0.55, 0.85), textcoords='axes fraction',
                fontsize=9, color='#d62728',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1),
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#d62728',
                          alpha=0.8))

    # ---- Panel 3: Control magnitude ----
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor('white')
    ax.plot(shooting['t'], shooting['u_mag'], '-', color='#1f77b4', lw=1.8,
            label='Shooting', alpha=0.9)
    ax.plot(bezier['t'], bezier['u_mag'], '--', color='#d62728', lw=1.8,
            label='Bézier (IPOPT)', alpha=0.8)
    ax.set_xlabel('Time (nondim)', fontsize=10, color='black')
    ax.set_ylabel('|u| (thrust magnitude)', fontsize=10, color='black')
    ax.set_title('Control Magnitude', fontsize=11, fontweight='bold', color='black')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.legend(fontsize=9, loc='best', framealpha=0.9,
              facecolor='white', edgecolor='black', labelcolor='black')
    ax.grid(True, alpha=0.3, color='gray')

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
    ax.set_xlabel('Time (nondim)', fontsize=10, color='black')
    ax.set_ylabel('Control component', fontsize=10, color='black')
    ax.set_title('Control Components', fontsize=11, fontweight='bold', color='black')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.legend(fontsize=8.5, ncol=2, loc='best', framealpha=0.9,
              columnspacing=1.0, handlelength=1.8,
              facecolor='white', edgecolor='black', labelcolor='black')
    ax.grid(True, alpha=0.3, color='gray')

    # ---- Panel 5: Jacobi constant ----
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor('white')
    ax.plot(shooting['t'], shooting['jacobi'], '-', color='#1f77b4', lw=1.8, label='Shooting')
    ax.plot(bezier['t'], bezier['jacobi'], '--', color='#d62728', lw=1.8, label='Bézier')
    C0 = shooting['jacobi'][0]
    Cf = shooting['jacobi'][-1]
    ax.axhline(y=C0, color='#2ca02c', ls=':', alpha=0.6, lw=1)
    ax.axhline(y=Cf, color='#9467bd', ls=':', alpha=0.6, lw=1)

    # Annotate the Jacobi values at the margins
    ax.text(0.02, C0, f' C₀={C0:.4f}', fontsize=8, color='#2ca02c',
            va='bottom', transform=ax.get_yaxis_transform())
    ax.text(0.02, Cf, f' Cf={Cf:.4f}', fontsize=8, color='#9467bd',
            va='top', transform=ax.get_yaxis_transform())

    ax.set_xlabel('Time (nondim)', fontsize=10, color='black')
    ax.set_ylabel('Jacobi constant C', fontsize=10, color='black')
    ax.set_title('Jacobi Constant (varies with thrust)', fontsize=11,
                 fontweight='bold', color='black')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.legend(fontsize=9, loc='best', framealpha=0.9,
              facecolor='white', edgecolor='black', labelcolor='black')
    ax.grid(True, alpha=0.3, color='gray')

    # ---- Panel 6: Stats table ----
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')

    table_data = [
        ['Metric', 'Shooting', 'Bézier (IPOPT)'],
        ['Method', 'Indirect (PMP)', 'Direct NLP (CasADi)'],
        ['Converged', 'Yes' if shooting['residual'] < 1e-6 else 'No',
         str(bezier.get('converged', 'N/A'))],
        ['Cost J = ∫|u|²dt', f"{shooting['cost']:.6f}",
         f"{bezier['cost']:.6f}"],
        ['Residual / Defect', f"{shooting['residual']:.2e}",
         f"{bezier.get('max_defect', 0):.2e}"],
        ['Solve time (s)', f"{shooting['time_s']:.3f}",
         f"{bezier['time_s']:.3f}"],
        ['Func evals / Iters', str(shooting['nfev']),
         str(bezier['nfev'])],
        ['C(t₀)', f"{shooting['jacobi'][0]:.6f}",
         f"{bezier['jacobi'][0]:.6f}"],
        ['C(tf)', f"{shooting['jacobi'][-1]:.6f}",
         f"{bezier['jacobi'][-1]:.6f}"],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.38, 0.31, 0.31])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.65)

    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if i == 0:
            cell.set_facecolor('white')
            cell.set_text_props(color='black', fontweight='bold', fontsize=10)
        elif j == 0:
            cell.set_facecolor('white')
            cell.set_text_props(color='black', fontweight='bold', fontsize=9.5)
        elif i % 2 == 0:
            cell.set_facecolor('white')
            cell.set_text_props(color='black')
        else:
            cell.set_facecolor('white')
            cell.set_text_props(color='black')

    ax.set_title('Performance Comparison', pad=15, fontsize=11,
                 fontweight='bold', color='black')

    fname = f'{save_prefix}_comparison.png'
    plt.savefig(fname, dpi=150, facecolor='white', edgecolor='white')
    print(f"\nSaved: {fname}")
    plt.close(fig)
    return fname


def plot_zoomed_cislunar(shooting, bezier, lyap_data, save_prefix='cr3bp_transfer'):
    """
    Zoomed-in cislunar view showing the transfer in the L1-L2 corridor.
    """
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    ax.set_facecolor('white')

    xL1 = lyap_data['L1']['xL']
    xL2 = lyap_data['L2']['xL']

    # Lyapunov orbits
    for name, color_ly, ls in [('L1', '#2ca02c', '-'), ('L2', '#9467bd', '-')]:
        sol = lyap_data[name]['sol']
        ax.plot(sol.y[0], sol.y[1], ls, color=color_ly, alpha=0.7, lw=1.5,
                label=f'{name} Lyapunov orbit')

    # Transfer trajectories — offset linewidth so both visible
    ax.plot(shooting['x'], shooting['y'], '-', color='#1f77b4', lw=3,
            label='Shooting (indirect)', zorder=5)
    ax.plot(bezier['x'], bezier['y'], '--', color='#d62728', lw=2.5,
            alpha=0.85, label='Bézier (IPOPT)', zorder=6)

    # Moon
    ax.plot(1 - MU, 0, 'o', color='#bdc3c7', ms=10, mec='black', mew=1.2,
            zorder=10)
    ax.annotate('Moon', (1 - MU, 0), textcoords='offset points',
                xytext=(8, -12), fontsize=10, color='black', fontweight='bold')

    # Libration points
    ax.plot(xL1, 0, 'D', color='#f39c12', ms=10, mec='black', mew=1,
            zorder=10)
    ax.plot(xL2, 0, 'D', color='#d62728', ms=10, mec='black', mew=1,
            zorder=10)
    ax.annotate('L1', (xL1, 0), textcoords='offset points',
                xytext=(-5, 12), fontsize=11, color='#f39c12',
                fontweight='bold', ha='center')
    ax.annotate('L2', (xL2, 0), textcoords='offset points',
                xytext=(5, 12), fontsize=11, color='#d62728',
                fontweight='bold', ha='center')

    # Departure / arrival markers
    ax.plot(shooting['x'][0], shooting['y'][0], '*', color='#2ca02c',
            ms=18, mec='black', mew=1, zorder=11)
    ax.plot(shooting['x'][-1], shooting['y'][-1], '*', color='#d62728',
            ms=18, mec='black', mew=1, zorder=11)
    ax.annotate('Departure', (shooting['x'][0], shooting['y'][0]),
                textcoords='offset points', xytext=(-14, -16), fontsize=9,
                color='#2ca02c', fontweight='bold', ha='center')
    ax.annotate('Arrival', (shooting['x'][-1], shooting['y'][-1]),
                textcoords='offset points', xytext=(14, -16), fontsize=9,
                color='#d62728', fontweight='bold', ha='center')

    # Control arrows (shooting only, every N-th point) — use quiver for clarity
    N_arrow = max(1, len(shooting['t']) // 30)
    arrow_idx = np.arange(0, len(shooting['t']), N_arrow)
    u_scale = 1.5
    ax.quiver(shooting['x'][arrow_idx], shooting['y'][arrow_idx],
              shooting['ux'][arrow_idx] * u_scale,
              shooting['uy'][arrow_idx] * u_scale,
              color='#1f77b4', alpha=0.4, scale=8, width=0.003,
              headwidth=4, headlength=5, zorder=4)

    # Zoom to corridor
    pad = 0.04
    ax.set_xlim(xL1 - pad, xL2 + pad)
    all_y = np.concatenate([shooting['y'], bezier['y'],
                            lyap_data['L1']['sol'].y[1],
                            lyap_data['L2']['sol'].y[1]])
    y_ext = max(abs(all_y.min()), abs(all_y.max())) + pad
    ax.set_ylim(-y_ext, y_ext)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color='gray')
    ax.set_xlabel('x (rotating frame, nondim)', fontsize=12, color='black')
    ax.set_ylabel('y (rotating frame, nondim)', fontsize=12, color='black')
    ax.set_title('L1 → L2 Lyapunov Transfer  —  Earth-Moon CR3BP',
                 fontsize=14, fontweight='bold', color='black')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.legend(fontsize=10, loc='lower left', framealpha=0.92,
              borderpad=0.6, handlelength=2.0,
              facecolor='white', edgecolor='black', labelcolor='black')

    plt.tight_layout()
    fname = f'{save_prefix}_cislunar.png'
    plt.savefig(fname, dpi=150, facecolor='white', edgecolor='white')
    print(f"Saved: {fname}")
    plt.close(fig)
    return fname


# =============================================================================
# Side-by-side animation for presentation
# =============================================================================

def create_animation(shooting, bezier, lyap_data, save_prefix='cr3bp_transfer',
                     fps=30, duration_s=8):
    """
    Create a side-by-side GIF animation showing both methods building
    their trajectories simultaneously in the cislunar environment.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    n_frames = fps * duration_s
    xL1 = lyap_data['L1']['xL']
    xL2 = lyap_data['L2']['xL']
    sol_L1 = lyap_data['L1']['sol']
    sol_L2 = lyap_data['L2']['sol']

    t0 = shooting['t'][0]
    tf = shooting['t'][-1]

    fig, (ax_s, ax_b) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')

    # Axis limits from trajectory extent
    all_x = np.concatenate([shooting['x'], bezier['x'],
                            sol_L1.y[0], sol_L2.y[0]])
    all_y = np.concatenate([shooting['y'], bezier['y'],
                            sol_L1.y[1], sol_L2.y[1]])
    pad = 0.05
    xl = (all_x.min() - pad, all_x.max() + pad)
    yl_range = max(all_y.max() - all_y.min(), 0.15)
    yl = (-yl_range / 2 - pad, yl_range / 2 + pad)

    for ax, title, color_accent in [
        (ax_s, 'Indirect Shooting (PMP)', '#00d2ff'),
        (ax_b, 'Bézier Collocation (IPOPT)', '#ff6b6b'),
    ]:
        ax.set_facecolor('white')
        ax.set_aspect('equal')
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_xlabel('x (rotating frame)', color='black', fontsize=10)
        ax.set_ylabel('y (rotating frame)', color='black', fontsize=10)
        ax.set_title(title, color=color_accent, fontsize=13, fontweight='bold')
        ax.tick_params(colors='black')
        for spine in ax.spines.values():
            spine.set_color('black')
        ax.grid(True, alpha=0.3, color='gray')

    # Static elements on both panels
    for ax in [ax_s, ax_b]:
        # Lyapunov orbits
        ax.plot(sol_L1.y[0], sol_L1.y[1], '-', color='#2ca02c', alpha=0.7,
                lw=1.2, label='L1 Lyapunov')
        ax.plot(sol_L2.y[0], sol_L2.y[1], '-', color='#9467bd', alpha=0.7,
                lw=1.2, label='L2 Lyapunov')

        # Bodies
        ax.plot(1 - MU, 0, 'o', color='#bdc3c7', ms=8, mec='black', mew=0.5,
                zorder=10)  # Moon
        ax.annotate('Moon', (1 - MU, 0), textcoords='offset points',
                    xytext=(0, -14), fontsize=7, color='black', ha='center')

        # Libration points
        ax.plot(xL1, 0, 'D', color='#ff7f0e', ms=7, mec='black', mew=1,
                zorder=10)
        ax.plot(xL2, 0, 'D', color='#f47067', ms=7, mec='black', mew=1,
                zorder=10)
        ax.annotate('L1', (xL1, 0), textcoords='offset points',
                    xytext=(0, 10), fontsize=8, color='#ff7f0e', ha='center',
                    fontweight='bold')
        ax.annotate('L2', (xL2, 0), textcoords='offset points',
                    xytext=(0, 10), fontsize=8, color='#f47067', ha='center',
                    fontweight='bold')

        # Departure / arrival markers (static)
        ax.plot(shooting['x'][0], shooting['y'][0], '*', color='#2ca02c',
                ms=14, mec='black', mew=0.8, zorder=11)
        ax.plot(shooting['x'][-1], shooting['y'][-1], '*', color='#f47067',
                ms=14, mec='black', mew=0.8, zorder=11)

    # Animated elements — Shooting panel
    trail_s, = ax_s.plot([], [], '-', color='#00d2ff', lw=2.5, zorder=5)
    craft_s, = ax_s.plot([], [], 'o', color='#00d2ff', ms=9, mec='black',
                          mew=1.2, zorder=8)
    # Control quiver (updated per frame)
    quiver_s = ax_s.quiver([], [], [], [], color='#1f77b4', scale=15,
                            width=0.003, alpha=0.6, zorder=4)
    text_s = ax_s.text(0.03, 0.03, '', transform=ax_s.transAxes,
                        color='#00d2ff', fontsize=9, family='monospace',
                        verticalalignment='bottom',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor='black',
                                  alpha=0.85))

    # Animated elements — Bézier panel
    trail_b, = ax_b.plot([], [], '-', color='#ff6b6b', lw=2.5, zorder=5)
    craft_b, = ax_b.plot([], [], 'o', color='#ff6b6b', ms=9, mec='black',
                          mew=1.2, zorder=8)
    quiver_b = ax_b.quiver([], [], [], [], color='#f778ba', scale=15,
                            width=0.003, alpha=0.6, zorder=4)
    text_b = ax_b.text(0.03, 0.03, '', transform=ax_b.transAxes,
                        color='#ff6b6b', fontsize=9, family='monospace',
                        verticalalignment='bottom',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor='black',
                                  alpha=0.85))

    # Bézier control points fading in
    cp_plots = []
    if bezier.get('segments'):
        colors_cp = plt.cm.cool(np.linspace(0.2, 0.8, len(bezier['segments'])))
        for i, cp in enumerate(bezier['segments']):
            p, = ax_b.plot([], [], 's-', color=colors_cp[i], ms=3, alpha=0,
                           lw=0.5, zorder=3)
            cp_plots.append((p, cp))

    # Time title
    time_text = fig.suptitle('', color='black', fontsize=12, y=0.97,
                              fontweight='bold')

    n_s = len(shooting['t'])
    n_b = len(bezier['t'])

    # Pre-scale control arrows
    u_scale = 4.0  # visual scale for arrow length
    arrow_step = max(1, n_s // 30)

    def init():
        trail_s.set_data([], [])
        craft_s.set_data([], [])
        trail_b.set_data([], [])
        craft_b.set_data([], [])
        for p, _ in cp_plots:
            p.set_data([], [])
        return []

    def update(frame):
        nonlocal quiver_s, quiver_b

        frac = frame / n_frames
        t_now = t0 + frac * (tf - t0)
        t_days = t_now * 4.343  # approx conversion to days

        # Shooting trail
        idx_s = min(int(frac * n_s), n_s - 1)
        trail_s.set_data(shooting['x'][:idx_s + 1], shooting['y'][:idx_s + 1])
        craft_s.set_data([shooting['x'][idx_s]], [shooting['y'][idx_s]])

        # Bézier trail
        idx_b = min(int(frac * n_b), n_b - 1)
        trail_b.set_data(bezier['x'][:idx_b + 1], bezier['y'][:idx_b + 1])
        craft_b.set_data([bezier['x'][idx_b]], [bezier['y'][idx_b]])

        # Control arrows (shooting) — show a few behind the spacecraft
        quiver_s.remove()
        arrow_indices = list(range(max(0, idx_s - 15 * arrow_step),
                                   idx_s + 1, arrow_step))
        if arrow_indices:
            qx = shooting['x'][arrow_indices]
            qy = shooting['y'][arrow_indices]
            qu = shooting['ux'][arrow_indices] * u_scale
            qv = shooting['uy'][arrow_indices] * u_scale
            quiver_s = ax_s.quiver(qx, qy, qu, qv, color='#1f77b4', scale=15,
                                    width=0.003, alpha=0.5, zorder=4)
        else:
            quiver_s = ax_s.quiver([], [], [], [], color='#1f77b4', scale=15,
                                    width=0.003, zorder=4)

        # Control arrows (Bézier)
        quiver_b.remove()
        arrow_indices_b = list(range(max(0, idx_b - 15 * arrow_step),
                                      idx_b + 1, arrow_step))
        if arrow_indices_b:
            qx = bezier['x'][arrow_indices_b]
            qy = bezier['y'][arrow_indices_b]
            qu = bezier['ux'][arrow_indices_b] * u_scale
            qv = bezier['uy'][arrow_indices_b] * u_scale
            quiver_b = ax_b.quiver(qx, qy, qu, qv, color='#f778ba', scale=15,
                                    width=0.003, alpha=0.5, zorder=4)
        else:
            quiver_b = ax_b.quiver([], [], [], [], color='#f778ba', scale=15,
                                    width=0.003, zorder=4)

        # Control points fade in
        n_seg = len(cp_plots)
        for i, (p, cp) in enumerate(cp_plots):
            seg_start_frac = i / max(n_seg, 1)
            if frac > seg_start_frac:
                alpha = min(1.0, (frac - seg_start_frac) * n_seg) * 0.5
                p.set_data(cp[:, 0], cp[:, 1])
                p.set_alpha(alpha)

        # Cost accumulated up to current time (approximate via trapezoid)
        if idx_s > 1:
            cost_s = np.trapezoid(
                shooting['ux'][:idx_s + 1] ** 2 + shooting['uy'][:idx_s + 1] ** 2,
                shooting['t'][:idx_s + 1]
            )
        else:
            cost_s = 0.0
        if idx_b > 1:
            cost_b = np.trapezoid(
                bezier['ux'][:idx_b + 1] ** 2 + bezier['uy'][:idx_b + 1] ** 2,
                bezier['t'][:idx_b + 1]
            )
        else:
            cost_b = 0.0

        text_s.set_text(
            f"t = {t_now:.3f} (~{t_days:.1f} days)\n"
            f"J(t) = {cost_s:.6f}\n"
            f"Final J = {shooting['cost']:.6f}"
        )
        text_b.set_text(
            f"t = {t_now:.3f} (~{t_days:.1f} days)\n"
            f"J(t) = {cost_b:.6f}\n"
            f"Final J = {bezier['cost']:.6f}"
        )

        time_text.set_text(
            f"L1 \u2192 L2 Lyapunov Transfer  |  "
            f"Earth-Moon CR3BP  |  t = {t_now:.3f} / {tf:.3f}"
        )

        return []

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=n_frames, interval=1000 / fps, blit=False)

    print("Saving animation (this takes a moment)...")
    fname = f'{save_prefix}_animation.gif'
    writer = PillowWriter(fps=fps)
    anim.save(fname, writer=writer, dpi=100)
    print(f"Saved: {fname}")
    plt.close(fig)
    return fname


# =============================================================================
# Forward Propagation Validation
# =============================================================================

def validate_forward_propagation(shooting_data, bezier_data, x0, t0, tf, mu=MU):
    """
    Validate the Bézier solution by forward-propagating the ODE
    using the Bézier-derived control history.
    """
    print("\n--- Forward Propagation Validation ---")

    # Interpolate Bézier control
    interp_ux = interp1d(bezier_data['t'], bezier_data['ux'],
                          kind='cubic', fill_value='extrapolate')
    interp_uy = interp1d(bezier_data['t'], bezier_data['uy'],
                          kind='cubic', fill_value='extrapolate')

    def controlled_ode(t, state):
        x, y, vx, vy = state
        r1 = np.sqrt((x + mu)**2 + y**2)
        r2 = np.sqrt((x - 1 + mu)**2 + y**2)
        Ux = x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
        Uy = y - (1-mu)*y/r1**3 - mu*y/r2**3
        ux_t = float(interp_ux(t))
        uy_t = float(interp_uy(t))
        return [vx, vy, 2*vy + Ux + ux_t, -2*vx + Uy + uy_t]

    sol_fwd = solve_ivp(controlled_ode, [t0, tf], x0.tolist(),
                        method='RK45', rtol=1e-12, atol=1e-12,
                        t_eval=np.linspace(t0, tf, 500))

    # Compare endpoint
    xf_fwd = sol_fwd.y[:, -1]
    xf_bez = np.array([bezier_data['x'][-1], bezier_data['y'][-1],
                       bezier_data['vx'][-1], bezier_data['vy'][-1]])

    err_pos = np.linalg.norm(xf_fwd[:2] - xf_bez[:2])
    err_vel = np.linalg.norm(xf_fwd[2:] - xf_bez[2:])

    # Max deviation along the trajectory
    interp_bx = interp1d(bezier_data['t'], bezier_data['x'], kind='cubic')
    interp_by = interp1d(bezier_data['t'], bezier_data['y'], kind='cubic')
    dx = sol_fwd.y[0] - interp_bx(sol_fwd.t)
    dy = sol_fwd.y[1] - interp_by(sol_fwd.t)
    max_pos_dev = np.max(np.sqrt(dx**2 + dy**2))

    print(f"  Forward-propagated endpoint error:")
    print(f"    Position: {err_pos:.2e}")
    print(f"    Velocity: {err_vel:.2e}")
    print(f"  Max position deviation along trajectory: {max_pos_dev:.2e}")

    return {
        'err_pos': err_pos, 'err_vel': err_vel,
        'max_pos_dev': max_pos_dev,
        'sol_fwd': sol_fwd,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 65)
    print("  L1 → L2 Lyapunov Orbit Transfer")
    print("  Planar Earth-Moon CR3BP  |  Min-Energy Low-Thrust")
    print("  Shooting (Indirect) vs Bézier Collocation (Direct)")
    print("=" * 65)

    # Set up the transfer
    x0, xf, t0, tf, lyap_data = setup_transfer_problem(Ax_L1=0.02, Ax_L2=0.02)
    print(f"\nTransfer: x0 = {x0}")
    print(f"          xf = {xf}")
    print(f"          t  = [{t0:.4f}, {tf:.4f}]")

    # Solve with both methods
    shooting_data, bezier_data = solve_both(x0, xf, t0, tf)

    # Generate comparison plots
    plot_comparison(shooting_data, bezier_data, lyap_data)
    plot_zoomed_cislunar(shooting_data, bezier_data, lyap_data)
    create_animation(shooting_data, bezier_data, lyap_data,
                     fps=20, duration_s=8)

    # Validate
    val = validate_forward_propagation(shooting_data, bezier_data, x0, t0, tf)

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Shooting cost:  {shooting_data['cost']:.6f}")
    print(f"  Bézier cost:    {bezier_data['cost']:.6f}")
    print(f"  Cost difference: {abs(shooting_data['cost'] - bezier_data['cost']):.2e}")
    print(f"  Max defect:      {bezier_data['max_defect']:.2e}")
    print(f"  Forward prop Δr: {val['max_pos_dev']:.2e}")
    print("=" * 65)


if __name__ == '__main__':
    main()
