"""
shooting.py - Shooting method solvers for TPBVPs

Ports of MATLAB pa_redo.mlx, pb.mlx, pc.mlx to Python.
Generalized to support both two-body and CR3BP dynamics.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


# =============================================================================
# Two-body shooting (2D) — direct port of MATLAB reference
# =============================================================================

def propagate(ode_func, X0, t_span, n_steps=2000, rtol=1e-12, atol=1e-12,
              **ode_kwargs):
    """Propagate an ODE and return the solution."""
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    sol = solve_ivp(
        lambda t, X: ode_func(t, X, **ode_kwargs),
        t_span, X0, t_eval=t_eval,
        method='RK45', rtol=rtol, atol=atol
    )
    return sol


def shooting_min_energy(lam0, r0, v0, pos_target_f, vel_target_f,
                        t0, tf, mu=1.0, u_max=None):
    """
    Shooting function for minimum-energy transfer (2D two-body).
    Matches MATLAB pa_redo.mlx.

    Args:
        lam0: initial costates [lam_rx, lam_ry, lam_vx, lam_vy]
        r0, v0: initial position and velocity (2D)
        pos_target_f: target position at tf
        vel_target_f: target velocity at tf
        t0, tf: initial and final time
        mu: gravitational parameter

    Returns:
        residual: [r_f - r_target; v_f - v_target]
    """
    from dynamics import two_body_state_costate_ode

    X0 = np.concatenate([r0, v0, lam0])
    sol = propagate(two_body_state_costate_ode, X0, [t0, tf],
                    mu=mu, u_max=u_max)

    rf = sol.y[0:2, -1]
    vf = sol.y[2:4, -1]

    residual = np.concatenate([rf - pos_target_f, vf - vel_target_f])
    return residual


def shooting_min_time(Z, r0, v0, pos_mars_func, vel_mars_angular, t0,
                      mu=1.0, u_max=0.1):
    """
    Shooting function for minimum-time transfer (2D two-body).
    Matches MATLAB pb.mlx. Free final time.

    Args:
        Z: [lam0 (4,), tf (1,)] — unknowns
    """
    from dynamics import two_body_min_time_ode

    lam0 = Z[0:4]
    tf = Z[4]

    if tf <= 0:
        return 1e6 * np.ones(5)

    X0 = np.concatenate([r0, v0, lam0])
    try:
        sol = propagate(two_body_min_time_ode, X0, [t0, tf],
                        mu=mu, u_max=u_max)
    except Exception:
        return 1e6 * np.ones(5)

    rf = sol.y[0:2, -1]
    vf = sol.y[2:4, -1]
    lam_rf = sol.y[4:6, -1]
    lam_vf = sol.y[6:8, -1]

    # Target state
    pos_mars_f = pos_mars_func(tf)
    vel_mars_f = vel_mars_angular * np.array([-pos_mars_f[1], pos_mars_f[0]])
    vel_mars_dot_f = -mu * pos_mars_f / np.linalg.norm(pos_mars_f)**3

    # Hamiltonian transversality for free tf
    mars_xdot_f = np.concatenate([vel_mars_f, vel_mars_dot_f])
    lam_f = np.concatenate([lam_rf, lam_vf])

    lam_v_norm = np.linalg.norm(lam_vf)
    if lam_v_norm < 1e-12:
        u_f = np.zeros(2)
    else:
        u_f = -u_max * lam_vf / lam_v_norm

    Hf = (1.0 + lam_rf @ vf
          + lam_vf @ (-mu * rf / np.linalg.norm(rf)**3 + u_f))
    Hres = lam_f @ mars_xdot_f

    residual = np.concatenate([rf - pos_mars_f, vf - vel_mars_f, [Hf - Hres]])
    return residual


def shooting_min_fuel(lam0, r0, v0, pos_target_f, vel_target_f,
                      t0, tf, mu=1.0, u_max=0.1, rho=1e-3):
    """
    Shooting function for minimum-fuel transfer (2D two-body).
    Matches MATLAB pc.mlx. Uses smoothed bang-bang (tanh).
    """
    from dynamics import two_body_min_fuel_ode

    X0 = np.concatenate([r0, v0, lam0])
    sol = propagate(two_body_min_fuel_ode, X0, [t0, tf],
                    mu=mu, u_max=u_max, rho=rho)

    rf = sol.y[0:2, -1]
    vf = sol.y[2:4, -1]

    residual = np.concatenate([rf - pos_target_f, vf - vel_target_f])
    return residual


# =============================================================================
# Solve wrappers
# =============================================================================

def solve_min_energy(r0, v0, pos_target_f, vel_target_f, t0, tf,
                     lam0_guess=None, mu=1.0, u_max=None):
    """Solve the minimum-energy TPBVP via shooting."""
    if lam0_guess is None:
        lam0_guess = np.zeros(4)

    lam0_sol, info, ier, msg = fsolve(
        shooting_min_energy, lam0_guess,
        args=(r0, v0, pos_target_f, vel_target_f, t0, tf, mu, u_max),
        full_output=True
    )

    if ier != 1:
        print(f"Warning: fsolve did not converge. Message: {msg}")

    return lam0_sol, info


def solve_min_time(r0, v0, pos_mars_func, vel_mars_angular, t0,
                   Z0_guess=None, mu=1.0, u_max=0.1):
    """Solve the minimum-time TPBVP via shooting."""
    if Z0_guess is None:
        Z0_guess = np.array([5.0, 2.0, 2.0, 7.0, 6.5])

    Z_sol, info, ier, msg = fsolve(
        shooting_min_time, Z0_guess,
        args=(r0, v0, pos_mars_func, vel_mars_angular, t0, mu, u_max),
        full_output=True
    )

    if ier != 1:
        print(f"Warning: fsolve did not converge. Message: {msg}")

    return Z_sol, info


def solve_min_fuel(r0, v0, pos_target_f, vel_target_f, t0, tf,
                   lam0_guess=None, mu=1.0, u_max=0.1,
                   rho_schedule=(1.0, 0.5, 0.1, 0.05, 1e-2, 1e-3)):
    """
    Solve the minimum-fuel TPBVP via shooting with continuation on rho.
    Matches MATLAB pc.mlx homotopy approach.
    """
    if lam0_guess is None:
        lam0_guess = np.array([0.8, 0.1, 0.2, 1.1])

    lam0_iter = lam0_guess.copy()
    history = {}

    for rho in rho_schedule:
        lam0_iter, info, ier, msg = fsolve(
            shooting_min_fuel, lam0_iter,
            args=(r0, v0, pos_target_f, vel_target_f, t0, tf, mu, u_max, rho),
            full_output=True
        )
        history[rho] = {'lam0': lam0_iter.copy(), 'converged': ier == 1}

        if ier != 1:
            print(f"Warning at rho={rho}: {msg}")

    return lam0_iter, history
