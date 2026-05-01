"""
cr3bp_planar.py - Planar CR3BP utilities for cislunar trajectory design

Provides:
  - Collinear libration point computation (L1, L2, L3)
  - Planar Lyapunov orbit computation via differential correction
  - State Transition Matrix (STM) propagation
  - Planar CR3BP dynamics (with and without control)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# Earth-Moon mass parameter
MU = 0.012150585609624


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
# Libration Points
# =============================================================================

def collinear_libration_points(mu=MU):
    """
    Compute the three collinear libration points L1, L2, L3.

    Returns:
        L1, L2, L3: x-coordinates (y = 0 for all)
    """
    # The equilibrium condition on the x-axis (y=0) is:
    #   x - (1-mu)(x+mu)/|x+mu|^3 - mu(x-1+mu)/|x-1+mu|^3 = 0
    # which is the gradient of the pseudo-potential set to zero.

    def Ux(x):
        """Pseudo-potential x-derivative on the x-axis (y=0)."""
        r1 = abs(x + mu)
        r2 = abs(x - 1.0 + mu)
        return (x
                - (1 - mu) * (x + mu) / r1**3
                - mu * (x - 1.0 + mu) / r2**3)

    # L1: between the two bodies
    xL1 = brentq(Ux, 0.5, 1 - mu - 1e-6)
    # L2: beyond smaller body (Moon)
    xL2 = brentq(Ux, 1 - mu + 1e-6, 1.5)
    # L3: on the far side of the larger body (Earth)
    xL3 = brentq(Ux, -1.5, -mu - 1e-6)

    return xL1, xL2, xL3


# =============================================================================
# Planar CR3BP Dynamics
# =============================================================================

def cr3bp_planar_ode(t, X, mu=MU):
    """
    Planar CR3BP equations of motion (rotating frame, uncontrolled).
    State: X = [x, y, vx, vy]
    """
    x, y, vx, vy = X

    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1.0 + mu)**2 + y**2)

    Ux = x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    Uy = y - (1 - mu) * y / r1**3 - mu * y / r2**3

    ax = 2 * vy + Ux
    ay = -2 * vx + Uy

    return np.array([vx, vy, ax, ay])


def cr3bp_planar_controlled_ode(t, X, mu=MU, u_max=None):
    """
    Planar CR3BP with min-energy optimal control (8D: state + costate).
    State: X = [x, y, vx, vy, lam_x, lam_y, lam_vx, lam_vy]

    Optimal control: u* = -0.5 * [lam_vx, lam_vy], optionally saturated to
    ||u|| <= u_max for the bounded-control min-energy variant.
    """
    x, y, vx, vy = X[:4]
    lam_r = X[4:6]
    lam_v = X[6:8]

    u = saturated_min_energy_control(lam_v, u_max=u_max)

    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)

    Ux = x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    Uy = y - (1 - mu) * y / r1**3 - mu * y / r2**3

    # State
    x_dot = vx
    y_dot = vy
    vx_dot = 2 * vy + Ux + u[0]
    vy_dot = -2 * vx + Uy + u[1]

    # Gravity gradient (second partials of pseudo-potential)
    Uxx = (1 - (1 - mu) / r1**3 + 3 * (1 - mu) * (x + mu)**2 / r1**5
           - mu / r2**3 + 3 * mu * (x - 1 + mu)**2 / r2**5)
    Uyy = (1 - (1 - mu) / r1**3 + 3 * (1 - mu) * y**2 / r1**5
           - mu / r2**3 + 3 * mu * y**2 / r2**5)
    Uxy = (3 * (1 - mu) * (x + mu) * y / r1**5
           + 3 * mu * (x - 1 + mu) * y / r2**5)

    # Costate
    lam_x_dot = -(Uxx * lam_v[0] + Uxy * lam_v[1])
    lam_y_dot = -(Uxy * lam_v[0] + Uyy * lam_v[1])
    lam_vx_dot = -lam_r[0] + 2 * lam_v[1]
    lam_vy_dot = -lam_r[1] - 2 * lam_v[0]

    return np.array([x_dot, y_dot, vx_dot, vy_dot,
                     lam_x_dot, lam_y_dot, lam_vx_dot, lam_vy_dot])


def cr3bp_planar_gravity(r, mu=MU):
    """
    Planar CR3BP gravitational + centrifugal + Coriolis pseudo-acceleration.
    For use in Bézier collocation: a = f(r, v) split form.

    NOTE: In CR3BP the acceleration depends on BOTH r and v (Coriolis).
    This function returns just the position-dependent part.
    Full EOM: a = [2*vy + Ux, -2*vx + Uy]
    So:  a_pos(r) = [Ux, Uy],  a_vel(v) = [2*vy, -2*vx]
    """
    x, y = r
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)

    Ux = x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    Uy = y - (1 - mu) * y / r1**3 - mu * y / r2**3

    return np.array([Ux, Uy])


# =============================================================================
# STM Propagation
# =============================================================================

def cr3bp_planar_stm_ode(t, Y, mu=MU):
    """
    Propagate planar CR3BP state + STM (4 + 16 = 20 equations).
    Y = [x, y, vx, vy, Phi_11, Phi_12, ..., Phi_44]
    """
    state = Y[:4]
    Phi = Y[4:20].reshape(4, 4)

    x, y, vx, vy = state

    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)

    Ux = x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    Uy = y - (1 - mu) * y / r1**3 - mu * y / r2**3

    state_dot = np.array([vx, vy, 2 * vy + Ux, -2 * vx + Uy])

    # A-matrix (Jacobian of f w.r.t. state)
    Uxx = (1 - (1 - mu) / r1**3 + 3 * (1 - mu) * (x + mu)**2 / r1**5
           - mu / r2**3 + 3 * mu * (x - 1 + mu)**2 / r2**5)
    Uyy = (1 - (1 - mu) / r1**3 + 3 * (1 - mu) * y**2 / r1**5
           - mu / r2**3 + 3 * mu * y**2 / r2**5)
    Uxy = (3 * (1 - mu) * (x + mu) * y / r1**5
           + 3 * mu * (x - 1 + mu) * y / r2**5)

    A = np.array([
        [0,    0,   1,  0],
        [0,    0,   0,  1],
        [Uxx,  Uxy, 0,  2],
        [Uxy,  Uyy, -2, 0]
    ])

    Phi_dot = A @ Phi

    return np.concatenate([state_dot, Phi_dot.ravel()])


def propagate_with_stm(state0, t_span, mu=MU, n_steps=1000, events=None):
    """
    Propagate planar CR3BP state with STM.

    Returns:
        sol: ODE solution object
    """
    Phi0 = np.eye(4).ravel()
    Y0 = np.concatenate([state0, Phi0])

    sol = solve_ivp(
        cr3bp_planar_stm_ode, t_span, Y0,
        args=(mu,),
        method='RK45', rtol=1e-12, atol=1e-12,
        t_eval=np.linspace(t_span[0], t_span[1], n_steps),
        events=events, dense_output=True
    )
    return sol


# =============================================================================
# Lyapunov Orbit Computation (Differential Correction)
# =============================================================================

def compute_lyapunov_orbit(xL, Ax, mu=MU, max_iter=50, tol=1e-12):
    """
    Compute a planar Lyapunov orbit near a collinear libration point
    using single-shooting differential correction.

    Uses x-axis symmetry: if (x, y, vx, vy) is a solution at t,
    then (x, -y, -vx, vy) is a solution at -t.
    => Seek a half-period crossing: y(T/2) = 0, vx(T/2) = 0.

    Initial guess: x0 = xL + Ax, y0 = 0, vx0 = 0, vy0 free
    Correct (x0, vy0) to target y = 0, vx = 0 at the y=0 crossing.

    Args:
        xL: x-coordinate of the libration point
        Ax: initial x-amplitude offset (e.g., 0.01 for small Lyapunov)
        mu: mass parameter
        max_iter: maximum correction iterations
        tol: convergence tolerance

    Returns:
        state0: corrected initial state [x0, 0, 0, vy0]
        T: full orbital period
        sol: ODE solution for one full period
    """
    x0 = xL + Ax
    # Initial vy guess from linearized dynamics
    vy0 = _lyapunov_vy_guess(xL, Ax, mu)

    for iteration in range(max_iter):
        state0 = np.array([x0, 0.0, 0.0, vy0])

        # Propagate forward — detect y-axis crossings after a short delay
        # (to avoid triggering immediately at t=0 where y=0)
        t_min_cross = 0.05  # skip initial transient

        def y_crossing(t, Y, mu=mu):
            if t < t_min_cross:
                return -1.0  # keep negative to avoid terminal event
            return Y[1]
        y_crossing.terminal = True
        y_crossing.direction = 0  # any crossing direction

        sol = propagate_with_stm(state0, [0, 20.0], mu=mu, n_steps=5000,
                                  events=y_crossing)

        if len(sol.t_events[0]) == 0:
            raise RuntimeError(f"No y-axis crossing found (vy0={vy0:.6f})")

        t_half = sol.t_events[0][0]
        Y_half = sol.sol(t_half)

        state_half = Y_half[:4]
        Phi_half = Y_half[4:20].reshape(4, 4)

        y_err = state_half[1]      # should be 0
        vx_err = state_half[2]     # should be 0

        if abs(y_err) < tol and abs(vx_err) < tol:
            break

        # Correction using STM (2-variable: correct x0 and vy0)
        #
        # At t = T/2, we want y = 0 and vx = 0.
        # The event already approximately enforces y≈0, but we can improve.
        #
        # Variational equations with free half-period δT:
        #   δy(T/2)  = Φ21*δx0 + Φ24*δvy0 + ẏ_half*δT
        #   δvx(T/2) = Φ31*δx0 + Φ34*δvy0 + ax_half*δT
        #
        # Eliminate δT from the first equation:
        #   δT = -(Φ21*δx0 + Φ24*δvy0 + y_err) / vy_half
        #
        # Substitute into second:
        #   δvx = Φ31*δx0 + Φ34*δvy0 + ax_half*δT = -vx_err
        # Expand:
        #   (Φ31 - ax*Φ21/vy)*δx0 + (Φ34 - ax*Φ24/vy)*δvy0
        #        = -vx_err - ax*y_err/vy

        f_half = cr3bp_planar_ode(t_half, state_half, mu)
        vy_half = state_half[3]
        ax_half = f_half[2]

        if abs(vy_half) < 1e-14:
            raise RuntimeError("vy at half-period is zero — orbit may be degenerate")

        # Two-variable correction:
        # Row 1: target y_err → 0  (via δT elimination, constrains one DOF)
        # Row 2: target vx_err → 0
        # We have 2 free vars (δx0, δvy0) and 1 effective equation after
        # eliminating δT. Use 2×2 system:
        #   [A11  A12] [δx0 ]   [b1]
        #   [A21  A22] [δvy0] = [b2]

        A11 = Phi_half[1, 0]    # ∂y/∂x0 (before δT elimination)
        A12 = Phi_half[1, 3]    # ∂y/∂vy0
        A21 = Phi_half[2, 0] - ax_half * Phi_half[1, 0] / vy_half
        A22 = Phi_half[2, 3] - ax_half * Phi_half[1, 3] / vy_half

        b1 = -y_err
        b2 = -vx_err - ax_half * y_err / vy_half

        M = np.array([[A11, A12], [A21, A22]])
        b = np.array([b1, b2])

        # Use single-variable correction: fix x0, only adjust vy0.
        # This is more robust than 2-variable for keeping x0 near the LP.
        # Target: vx(T/2) = 0 only  (y≈0 enforced by event detection)
        if abs(A22) > 1e-14:
            dvy0 = b2 / A22
            # Dampen large corrections
            if abs(dvy0) > 0.1:
                dvy0 = 0.1 * np.sign(dvy0)
            vy0 += dvy0
        else:
            raise RuntimeError("Correction coefficient A22 is zero")

    else:
        print(f"Warning: differential correction did not converge after {max_iter} iter "
              f"(y_err={y_err:.2e}, vx_err={vx_err:.2e})")

    T = 2 * t_half
    state0 = np.array([x0, 0.0, 0.0, vy0])

    # Propagate full period
    sol_full = solve_ivp(
        lambda t, X: cr3bp_planar_ode(t, X, mu),
        [0, T], state0,
        method='RK45', rtol=1e-12, atol=1e-12,
        t_eval=np.linspace(0, T, 500)
    )

    return state0, T, sol_full


def _lyapunov_vy_guess(xL, Ax, mu):
    """
    Estimate initial vy for a Lyapunov orbit from linearized CR3BP dynamics.
    Uses the eigenvalue analysis at the libration point.
    """
    # Compute Uxx, Uyy at the libration point (y=0)
    r1 = abs(xL + mu)
    r2 = abs(xL - 1 + mu)

    Uxx = (1 - (1 - mu) / r1**3 + 3 * (1 - mu) * (xL + mu)**2 / r1**5
           - mu / r2**3 + 3 * mu * (xL - 1 + mu)**2 / r2**5)
    Uyy = (1 - (1 - mu) / r1**3 + 3 * (1 - mu) * 0 / r1**5
           - mu / r2**3 + 3 * mu * 0 / r2**5)

    # Characteristic equation: s^4 + (2 - Uxx - Uyy)*s^2 + (Uxx*Uyy - Uxy^2) = 0
    # For collinear points (y=0): Uxy = 0
    # In-plane eigenvalues: s^2 = -(2-c2) ± sqrt((2-c2)^2 - (Uxx*Uyy))
    # where c2 relates to the eigenvalue.

    # Simpler: just use the Jacobian eigenvalues
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [Uxx, 0, 0, 2],
        [0, Uyy, -2, 0]
    ])
    evals = np.linalg.eigvals(A)

    # Find the purely imaginary eigenvalues (in-plane oscillation)
    imag_evals = [ev for ev in evals if abs(ev.real) < 1e-6 and ev.imag > 0]

    if imag_evals:
        omega = imag_evals[0].imag
        # vy0 ≈ -omega * Ax * (correction factor)
        # From linearized solution: x(t) = Ax*cos(ωt), y(t) = -kAx*sin(ωt)
        # where k = (ω² + Uxx) / (2ω)
        k = (omega**2 + Uxx) / (2 * omega)
        return -k * Ax * omega  # vy0 = -k*Ax*ω (from ẏ(0))
    else:
        # Fallback
        return -0.5 * Ax


# =============================================================================
# Inertial-Frame Coordinate Transforms
# =============================================================================

def rotating_to_inertial(t, x_rot, y_rot, vx_rot, vy_rot):
    """
    Transform rotating-frame state to inertial-frame state.

    The rotating frame has angular velocity ω = 1 (nondimensional).
    Rotation matrix R(t) rotates position; velocity includes ω×r correction.

    Args:
        t:  time (scalar or array)
        x_rot, y_rot:   rotating-frame position
        vx_rot, vy_rot: rotating-frame velocity

    Returns:
        x_in, y_in, vx_in, vy_in: inertial-frame state
    """
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    # Position: r_inertial = R(t) @ r_rotating
    x_in = cos_t * x_rot - sin_t * y_rot
    y_in = sin_t * x_rot + cos_t * y_rot

    # Velocity: v_inertial = R(t) @ (v_rotating + ω × r_rotating)
    # ω × r = (-y, x)  in 2D (ω = 1)
    vx_rot_corr = vx_rot - y_rot
    vy_rot_corr = vy_rot + x_rot
    vx_in = cos_t * vx_rot_corr - sin_t * vy_rot_corr
    vy_in = sin_t * vx_rot_corr + cos_t * vy_rot_corr

    return x_in, y_in, vx_in, vy_in


def body_positions_inertial(t, mu=MU):
    """
    Positions of Earth and Moon in the inertial frame at time t.

    In the rotating frame:
        Earth: (-μ, 0)
        Moon:  (1-μ, 0)

    Args:
        t:  time (scalar or array)
        mu: mass parameter

    Returns:
        earth_xy: (x_E, y_E) tuple
        moon_xy:  (x_M, y_M) tuple
    """
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    # Earth at (-μ, 0) in rotating frame
    x_E = -mu * cos_t
    y_E = -mu * sin_t

    # Moon at (1-μ, 0) in rotating frame
    x_M = (1 - mu) * cos_t
    y_M = (1 - mu) * sin_t

    return (x_E, y_E), (x_M, y_M)


def trajectory_rotating_to_inertial(t_arr, states_rot, mu=MU):
    """
    Convert an entire trajectory from rotating to inertial frame.

    Args:
        t_arr:      time array, shape (N,)
        states_rot: rotating-frame states, shape (4, N) or (N, 4)
        mu:         mass parameter

    Returns:
        dict with keys 't', 'x', 'y', 'vx', 'vy' (inertial frame),
        plus 'earth_x', 'earth_y', 'moon_x', 'moon_y'
    """
    if states_rot.shape[0] == 4:
        x_r, y_r, vx_r, vy_r = states_rot
    else:
        x_r = states_rot[:, 0]
        y_r = states_rot[:, 1]
        vx_r = states_rot[:, 2]
        vy_r = states_rot[:, 3]

    x_in, y_in, vx_in, vy_in = rotating_to_inertial(
        t_arr, x_r, y_r, vx_r, vy_r
    )
    (x_E, y_E), (x_M, y_M) = body_positions_inertial(t_arr, mu)

    return {
        't': t_arr,
        'x': x_in, 'y': y_in,
        'vx': vx_in, 'vy': vy_in,
        'earth_x': x_E, 'earth_y': y_E,
        'moon_x': x_M, 'moon_y': y_M,
    }


# =============================================================================
# Jacobi Constant
# =============================================================================

def cr3bp_jacobi_planar(state, mu=MU):
    """Jacobi constant for planar CR3BP."""
    x, y, vx, vy = state
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    U = 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2
    return 2 * U - (vx**2 + vy**2)


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    print("Earth-Moon CR3BP (planar)")
    print(f"μ = {MU}")

    xL1, xL2, xL3 = collinear_libration_points()
    print(f"\nLibration points:")
    print(f"  L1: x = {xL1:.10f}")
    print(f"  L2: x = {xL2:.10f}")
    print(f"  L3: x = {xL3:.10f}")

    print("\nComputing L1 Lyapunov orbit (Ax = 0.02)...")
    state_L1, T_L1, sol_L1 = compute_lyapunov_orbit(xL1, Ax=0.02)
    C_L1 = cr3bp_jacobi_planar(state_L1)
    print(f"  IC: {state_L1}")
    print(f"  Period: {T_L1:.8f}")
    print(f"  Jacobi: {C_L1:.8f}")

    print("\nComputing L2 Lyapunov orbit (Ax = 0.02)...")
    state_L2, T_L2, sol_L2 = compute_lyapunov_orbit(xL2, Ax=0.02)
    C_L2 = cr3bp_jacobi_planar(state_L2)
    print(f"  IC: {state_L2}")
    print(f"  Period: {T_L2:.8f}")
    print(f"  Jacobi: {C_L2:.8f}")

    # Verify periodicity
    err_L1 = np.linalg.norm(sol_L1.y[:, -1] - state_L1)
    err_L2 = np.linalg.norm(sol_L2.y[:, -1] - state_L2)
    print(f"\n  L1 periodicity error: {err_L1:.2e}")
    print(f"  L2 periodicity error: {err_L2:.2e}")
