"""
dynamics.py - Orbital dynamics models

Contains:
  - Two-body problem (2D, for validation against MATLAB reference)
  - Circular Restricted Three-Body Problem (CR3BP, 3D)
"""

import numpy as np


# =============================================================================
# Optimal-control helpers
# =============================================================================

def saturated_min_energy_control(lam_v, u_max=None):
    """
    Minimum-energy control law with an optional magnitude bound.

    Unbounded: u* = -0.5 lambda_v.
    Bounded:   projection of that vector onto ||u|| <= u_max.
    """
    u = -0.5 * np.asarray(lam_v, dtype=float)
    if u_max is None:
        return u

    u_norm = np.linalg.norm(u)
    if u_norm <= float(u_max) or u_norm < 1e-15:
        return u
    return float(u_max) * u / u_norm


# =============================================================================
# Two-Body Problem (2D) — matches MATLAB pa_redo / pb / pc
# =============================================================================

def two_body_ode(t, X, mu=1.0):
    """
    Two-body equations of motion (2D inertial frame).
    State: X = [rx, ry, vx, vy]
    """
    r = X[0:2]
    v = X[2:4]
    r_norm = np.linalg.norm(r)

    r_dot = v
    v_dot = -mu * r / r_norm**3

    return np.concatenate([r_dot, v_dot])


def two_body_state_costate_ode(t, X, mu=1.0, u_max=None):
    """
    Two-body state + costate ODE for minimum-energy optimal control.
    State: X = [rx, ry, vx, vy, lam_rx, lam_ry, lam_vx, lam_vy]

    Optimal control: u* = -0.5 * lambda_v  (from dH/du = 0), optionally
    saturated to ||u|| <= u_max for the bounded-control min-energy variant.
    """
    r = X[0:2]
    v = X[2:4]
    lam_r = X[4:6]
    lam_v = X[6:8]

    r_norm = np.linalg.norm(r)

    # Optimal control (minimum energy), with optional saturation.
    u = saturated_min_energy_control(lam_v, u_max=u_max)

    # State derivatives
    r_dot = v
    v_dot = -mu * r / r_norm**3 + u

    # Costate derivatives
    # d(lam_r)/dt = -dH/dr = -mu * (3*r*r'/|r|^5 - I/|r|^3) * lam_v
    gravity_grad = mu * (3.0 * np.outer(r, r) / r_norm**5
                         - np.eye(2) / r_norm**3)
    lam_r_dot = -gravity_grad @ lam_v
    lam_v_dot = -lam_r

    return np.concatenate([r_dot, v_dot, lam_r_dot, lam_v_dot])


def two_body_min_time_ode(t, X, mu=1.0, u_max=0.1):
    """
    Two-body state + costate ODE for minimum-time with bounded thrust.
    Optimal control: u* = -u_max * lam_v / |lam_v|  (bang-bang)
    """
    r = X[0:2]
    v = X[2:4]
    lam_r = X[4:6]
    lam_v = X[6:8]

    r_norm = np.linalg.norm(r)
    lam_v_norm = np.linalg.norm(lam_v)

    eps = 1e-12
    if lam_v_norm < eps:
        u = np.zeros(2)
    else:
        u = -u_max * lam_v / lam_v_norm

    r_dot = v
    v_dot = -mu * r / r_norm**3 + u

    gravity_grad = mu * (3.0 * np.outer(r, r) / r_norm**5
                         - np.eye(2) / r_norm**3)
    lam_r_dot = -gravity_grad @ lam_v
    lam_v_dot = -lam_r

    return np.concatenate([r_dot, v_dot, lam_r_dot, lam_v_dot])


def two_body_min_fuel_ode(t, X, mu=1.0, u_max=0.1, rho=1e-3):
    """
    Two-body state + costate ODE for minimum-fuel with smoothed bang-bang.
    Switching function: S = |lam_v| - 1
    Thrust magnitude: T = (u_max/2) * (1 + tanh(S/rho))
    """
    r = X[0:2]
    v = X[2:4]
    lam_r = X[4:6]
    lam_v = X[6:8]

    r_norm = np.linalg.norm(r)
    lam_v_norm = np.linalg.norm(lam_v)

    S = lam_v_norm - 1.0
    thrust_mag = (u_max / 2.0) * (1.0 + np.tanh(S / rho))

    eps = 1e-12
    if lam_v_norm < eps:
        u = np.zeros(2)
    else:
        u = -thrust_mag * lam_v / lam_v_norm

    r_dot = v
    v_dot = -mu * r / r_norm**3 + u

    gravity_grad = mu * (3.0 * np.outer(r, r) / r_norm**5
                         - np.eye(2) / r_norm**3)
    lam_r_dot = -gravity_grad @ lam_v
    lam_v_dot = -lam_r

    return np.concatenate([r_dot, v_dot, lam_r_dot, lam_v_dot])


# =============================================================================
# Circular Restricted Three-Body Problem (CR3BP) — 3D
# =============================================================================

# Earth-Moon mass parameter
MU_EARTH_MOON = 0.012150585609624  # mu = m_Moon / (m_Earth + m_Moon)


def cr3bp_pseudo_potential(x, y, z, mu=MU_EARTH_MOON):
    """Pseudo-potential U-bar of the CR3BP."""
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)        # dist to larger body
    r2 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)  # dist to smaller body
    U = 0.5 * (x**2 + y**2) + (1.0 - mu) / r1 + mu / r2
    return U


def cr3bp_ode(t, X, mu=MU_EARTH_MOON):
    """
    CR3BP equations of motion in the rotating frame (uncontrolled).
    State: X = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = X

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)

    # Pseudo-potential partials
    Ux = x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x - 1.0 + mu) / r2**3
    Uy = y - (1.0 - mu) * y / r1**3 - mu * y / r2**3
    Uz = -(1.0 - mu) * z / r1**3 - mu * z / r2**3

    ax = 2.0 * vy + Ux
    ay = -2.0 * vx + Uy
    az = Uz

    return np.array([vx, vy, vz, ax, ay, az])


def cr3bp_controlled_ode(t, X, mu=MU_EARTH_MOON):
    """
    CR3BP with low-thrust control — state + costate (12D).
    State: X = [x, y, z, vx, vy, vz, lam_x, ..., lam_vz]

    Minimum-energy optimal control: u* = -0.5 * lambda_v
    """
    state = X[0:6]
    lam = X[6:12]

    x, y, z, vx, vy, vz = state
    lam_r = lam[0:3]   # position costates
    lam_v = lam[3:6]   # velocity costates

    # Optimal control
    u = -0.5 * lam_v

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)

    # State derivatives (CR3BP + control)
    Ux = x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x - 1.0 + mu) / r2**3
    Uy = y - (1.0 - mu) * y / r1**3 - mu * y / r2**3
    Uz = -(1.0 - mu) * z / r1**3 - mu * z / r2**3

    state_dot = np.array([
        vx, vy, vz,
        2.0 * vy + Ux + u[0],
        -2.0 * vx + Uy + u[1],
        Uz + u[2]
    ])

    # Gravity gradient tensor (second partials of pseudo-potential)
    # Uxx, Uxy, Uxz, Uyy, Uyz, Uzz
    Uxx = (1.0
           - (1.0 - mu) / r1**3 + 3.0 * (1.0 - mu) * (x + mu)**2 / r1**5
           - mu / r2**3 + 3.0 * mu * (x - 1.0 + mu)**2 / r2**5)
    Uyy = (1.0
           - (1.0 - mu) / r1**3 + 3.0 * (1.0 - mu) * y**2 / r1**5
           - mu / r2**3 + 3.0 * mu * y**2 / r2**5)
    Uzz = (-(1.0 - mu) / r1**3 + 3.0 * (1.0 - mu) * z**2 / r1**5
           - mu / r2**3 + 3.0 * mu * z**2 / r2**5)
    Uxy = (3.0 * (1.0 - mu) * (x + mu) * y / r1**5
           + 3.0 * mu * (x - 1.0 + mu) * y / r2**5)
    Uxz = (3.0 * (1.0 - mu) * (x + mu) * z / r1**5
           + 3.0 * mu * (x - 1.0 + mu) * z / r2**5)
    Uyz = (3.0 * (1.0 - mu) * y * z / r1**5
           + 3.0 * mu * y * z / r2**5)

    # Costate derivatives: lam_dot = -dH/dstate
    # dH/dr = U_grad_matrix @ lam_v + [0, 0, 0; 0, 0, 0; ...] terms from Coriolis
    # Full derivation from Hamiltonian:
    # H = |u|^2 + lam_r^T * v + lam_v^T * (f(x) + u)

    lam_x_dot = -(Uxx * lam_v[0] + Uxy * lam_v[1] + Uxz * lam_v[2])
    lam_y_dot = -(Uxy * lam_v[0] + Uyy * lam_v[1] + Uyz * lam_v[2])
    lam_z_dot = -(Uxz * lam_v[0] + Uyz * lam_v[1] + Uzz * lam_v[2])

    lam_vx_dot = -lam_r[0] + 2.0 * lam_v[1]
    lam_vy_dot = -lam_r[1] - 2.0 * lam_v[0]
    lam_vz_dot = -lam_r[2]

    lam_dot = np.array([lam_x_dot, lam_y_dot, lam_z_dot,
                        lam_vx_dot, lam_vy_dot, lam_vz_dot])

    return np.concatenate([state_dot, lam_dot])


def cr3bp_jacobi(state, mu=MU_EARTH_MOON):
    """Compute the Jacobi constant for a CR3BP state."""
    x, y, z, vx, vy, vz = state
    U = cr3bp_pseudo_potential(x, y, z, mu)
    v_sq = vx**2 + vy**2 + vz**2
    return 2.0 * U - v_sq
