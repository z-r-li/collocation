#!/usr/bin/env python3
"""
cr3bp_3d.py — 3D CR3BP Infrastructure for LEO-to-NRHO Transfer Design

Core module providing:
  1. Earth-Moon CR3BP constants and nondimensionalization
  2. 3D equations of motion (uncontrolled, controlled, STM)
  3. Jacobi constant and energy diagnostics
  4. Halo orbit differential correction (3D periodic orbits)
  5. 9:2 NRHO validation from JPL Periodic Orbits Database
  6. LEO state construction in the rotating frame

Dynamics refactored from Artemis2/artemis2_3d.py (validated against NASA OEM).

Author: Zhuorui, AAE 568 Spring 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


# =============================================================================
# EARTH-MOON CR3BP CONSTANTS
# =============================================================================

MU = 0.012150585609624        # mass parameter: M_Moon / (M_Earth + M_Moon)
L_STAR = 384400.0             # km — mean Earth-Moon distance
MU_EARTH = 398600.4418        # km^3/s^2 — Earth gravitational parameter
MU_MOON  = 4902.800066        # km^3/s^2 — Moon gravitational parameter

# T_STAR from Kepler's third law for consistency with CR3BP dynamics:
#   T_STAR = sqrt(L*³ / (μ_E + μ_M))  so that the primary period = 2π in nondim time
T_STAR = np.sqrt(L_STAR**3 / (MU_EARTH + MU_MOON))  # seconds
V_STAR = L_STAR / T_STAR      # km/s — nondim velocity unit

# Note: The dimensional period of the primaries = 2π * T_STAR.
# This differs slightly from the actual Moon sidereal period (27.32 days)
# because the real Earth-Moon orbit is not perfectly circular at 384400 km.
# For dimensional comparisons with real ephemeris data, use the JPL units below.

R_MOON = 1737.4               # km — mean lunar radius
R_EARTH = 6371.0              # km — mean Earth radius

# Positions of primaries in rotating frame
EARTH_POS = np.array([-MU, 0.0, 0.0])
MOON_POS = np.array([1.0 - MU, 0.0, 0.0])

# JPL Periodic Orbits Database parameters (for reference / dimensional conversion)
# These differ slightly from L_STAR/T_STAR above but yield the same nondim dynamics
JPL_LUNIT = 389703.264829278  # km
JPL_TUNIT = 382981.289129055  # s

# 9:2 L2 Southern NRHO — from JPL Three-Body Periodic Orbits Database
# State at apolune (y=0 crossing), Earth-Moon rotating frame, nondimensional
NRHO_9_2 = {
    'x0':     1.0196625817,
    'y0':     0.0,
    'z0':    -0.1804191873,
    'vx0':    0.0,
    'vy0':   -0.0980598247,
    'vz0':    0.0,
    'period': 1.479980,        # nondim
    'jacobi': 3.04891,
    'stability': 1.255,
    'perilune_km': 2931.0,     # km from Moon center (propagation-verified)
    'apolune_km': 71395.0,     # km from Moon center
}


def nrho_state():
    """Return the 9:2 NRHO initial state as a (6,) array."""
    n = NRHO_9_2
    return np.array([n['x0'], n['y0'], n['z0'], n['vx0'], n['vy0'], n['vz0']])


def nrho_period():
    """Return the 9:2 NRHO period (nondimensional)."""
    return NRHO_9_2['period']


# =============================================================================
# 3D CR3BP EQUATIONS OF MOTION
# =============================================================================

def pseudo_potential_gradient(x, y, z, mu=MU):
    """
    Gradient of the CR3BP pseudo-potential Ω(x, y, z).

    Ω = ½(x² + y²) + (1-μ)/r₁ + μ/r₂

    Returns (Ux, Uy, Uz) — the partial derivatives.
    """
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)

    Ux = x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x - 1.0 + mu) / r2**3
    Uy = y - (1.0 - mu) * y / r1**3 - mu * y / r2**3
    Uz =   - (1.0 - mu) * z / r1**3 - mu * z / r2**3

    return Ux, Uy, Uz


def pseudo_potential_hessian(x, y, z, mu=MU):
    """
    Hessian (second partial derivatives) of the CR3BP pseudo-potential.

    Returns the 3x3 symmetric matrix:
        [[Uxx, Uxy, Uxz],
         [Uxy, Uyy, Uyz],
         [Uxz, Uyz, Uzz]]
    """
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)

    r1_3 = r1**3
    r1_5 = r1**5
    r2_3 = r2**3
    r2_5 = r2**5

    one_minus_mu = 1.0 - mu

    Uxx = (1.0
           - one_minus_mu / r1_3 + 3.0 * one_minus_mu * (x + mu)**2 / r1_5
           - mu / r2_3 + 3.0 * mu * (x - 1.0 + mu)**2 / r2_5)
    Uyy = (1.0
           - one_minus_mu / r1_3 + 3.0 * one_minus_mu * y**2 / r1_5
           - mu / r2_3 + 3.0 * mu * y**2 / r2_5)
    Uzz = (- one_minus_mu / r1_3 + 3.0 * one_minus_mu * z**2 / r1_5
           - mu / r2_3 + 3.0 * mu * z**2 / r2_5)
    Uxy = (3.0 * one_minus_mu * (x + mu) * y / r1_5
           + 3.0 * mu * (x - 1.0 + mu) * y / r2_5)
    Uxz = (3.0 * one_minus_mu * (x + mu) * z / r1_5
           + 3.0 * mu * (x - 1.0 + mu) * z / r2_5)
    Uyz = (3.0 * one_minus_mu * y * z / r1_5
           + 3.0 * mu * y * z / r2_5)

    return np.array([
        [Uxx, Uxy, Uxz],
        [Uxy, Uyy, Uyz],
        [Uxz, Uyz, Uzz]
    ])


def cr3bp_ode(t, X, mu=MU):
    """
    Uncontrolled 3D CR3BP equations of motion.

    State: X = [x, y, z, vx, vy, vz]  (6D)

    Returns dX/dt.
    """
    x, y, z, vx, vy, vz = X

    Ux, Uy, Uz = pseudo_potential_gradient(x, y, z, mu)

    return np.array([
        vx,
        vy,
        vz,
        2.0 * vy + Ux,
        -2.0 * vx + Uy,
        Uz
    ])


def cr3bp_controlled_ode(t, X, mu=MU):
    """
    Controlled 3D CR3BP with minimum-energy optimal control.

    State: X = [x, y, z, vx, vy, vz, λx, λy, λz, λvx, λvy, λvz]  (12D)

    Optimal control: u* = -½ λv  (from Pontryagin's principle, min ∫|u|² dt)

    Returns dX/dt including costate dynamics.
    """
    state = X[:6]
    lam = X[6:12]

    x, y, z, vx, vy, vz = state
    lam_r = lam[:3]    # position costates
    lam_v = lam[3:6]   # velocity costates

    # Optimal control
    u = -0.5 * lam_v

    # Pseudo-potential gradient
    Ux, Uy, Uz = pseudo_potential_gradient(x, y, z, mu)

    # State dynamics (controlled)
    state_dot = np.array([
        vx, vy, vz,
        2.0 * vy + Ux + u[0],
        -2.0 * vx + Uy + u[1],
        Uz + u[2]
    ])

    # Pseudo-potential Hessian for costate dynamics
    H = pseudo_potential_hessian(x, y, z, mu)

    # Costate dynamics: dλ/dt = -∂H/∂state
    # Position costates: dλr/dt = -H @ λv  (from gravity gradient)
    lam_r_dot = -H @ lam_v

    # Velocity costates: dλv/dt = -λr + Coriolis terms
    lam_vx_dot = -lam_r[0] + 2.0 * lam_v[1]
    lam_vy_dot = -lam_r[1] - 2.0 * lam_v[0]
    lam_vz_dot = -lam_r[2]

    lam_dot = np.concatenate([lam_r_dot, [lam_vx_dot, lam_vy_dot, lam_vz_dot]])

    return np.concatenate([state_dot, lam_dot])


# =============================================================================
# STATE TRANSITION MATRIX (6x6 STM)
# =============================================================================

def cr3bp_stm_ode(t, Y, mu=MU):
    """
    CR3BP equations of motion with 6x6 State Transition Matrix (STM).

    Y = [x, y, z, vx, vy, vz, Φ₁₁, Φ₁₂, ..., Φ₆₆]  (42D)

    The STM satisfies: dΦ/dt = A(t) @ Φ, where A is the 6x6 Jacobian
    of the CR3BP vector field evaluated along the trajectory.

    Returns dY/dt (42 components).
    """
    state = Y[:6]
    Phi = Y[6:42].reshape(6, 6)

    x, y, z = state[:3]

    # State derivatives (uncontrolled)
    state_dot = cr3bp_ode(t, state, mu)

    # Build the Jacobian A = df/dx (6x6)
    # A = [[0₃ₓ₃, I₃ₓ₃],
    #      [Ω + U_hess, 2J]]
    #
    # where Ω = [[0,2,0],[-2,0,0],[0,0,0]] (Coriolis)
    #       J = [[0,1,0],[-1,0,0],[0,0,0]]

    U_hess = pseudo_potential_hessian(x, y, z, mu)

    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)                  # ṙ = v
    A[3:6, 0:3] = U_hess                      # gravity gradient
    A[3, 4] = A[3, 4] + 2.0                   # Coriolis: 2vy term
    A[4, 3] = A[4, 3] - 2.0                   # Coriolis: -2vx term

    # STM derivative: dΦ/dt = A @ Φ
    Phi_dot = A @ Phi

    return np.concatenate([state_dot, Phi_dot.flatten()])


def propagate(state0, t_span, mu=MU, rtol=1e-12, atol=1e-14,
              max_step=0.001, dense_output=True, events=None):
    """
    Propagate the uncontrolled CR3BP from state0 over t_span.

    Args:
        state0: (6,) initial state [x, y, z, vx, vy, vz]
        t_span: (t0, tf) or array of times
        mu: mass parameter
        rtol, atol: integration tolerances
        max_step: maximum step size
        dense_output: whether to return dense output
        events: event functions for solve_ivp

    Returns:
        scipy OdeResult
    """
    return solve_ivp(
        cr3bp_ode, t_span, state0, args=(mu,),
        method='RK45', rtol=rtol, atol=atol,
        max_step=max_step, dense_output=dense_output, events=events
    )


def propagate_with_stm(state0, t_span, mu=MU, rtol=1e-12, atol=1e-14,
                        max_step=0.001, dense_output=True, events=None):
    """
    Propagate CR3BP state + 6x6 STM simultaneously.

    Args:
        state0: (6,) initial state
        t_span: (t0, tf)

    Returns:
        sol: OdeResult (Y has 42 components)
        To extract STM at time t: sol.sol(t)[6:42].reshape(6,6)
    """
    # Initial STM = identity
    Phi0 = np.eye(6).flatten()
    Y0 = np.concatenate([state0, Phi0])

    sol = solve_ivp(
        cr3bp_stm_ode, t_span, Y0, args=(mu,),
        method='RK45', rtol=rtol, atol=atol,
        max_step=max_step, dense_output=dense_output, events=events
    )

    return sol


# =============================================================================
# JACOBI CONSTANT
# =============================================================================

def jacobi_constant(state, mu=MU):
    """
    Compute the Jacobi constant C = 2Ω - v².

    This is the only integral of motion in the CR3BP.
    A conserved quantity for uncontrolled trajectories.
    """
    x, y, z, vx, vy, vz = state

    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)

    Omega = 0.5 * (x**2 + y**2) + (1.0 - mu) / r1 + mu / r2
    v_sq = vx**2 + vy**2 + vz**2

    return 2.0 * Omega - v_sq


def dist_from_moon(state, mu=MU):
    """Distance from the Moon (secondary body) in nondimensional units."""
    x, y, z = state[:3]
    return np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)


def dist_from_earth(state, mu=MU):
    """Distance from Earth (primary body) in nondimensional units."""
    x, y, z = state[:3]
    return np.sqrt((x + mu)**2 + y**2 + z**2)


# =============================================================================
# COLLINEAR LIBRATION POINTS
# =============================================================================

def collinear_libration_points(mu=MU):
    """
    Compute L1, L2, L3 positions on the x-axis.

    Returns (xL1, xL2, xL3) — x-coordinates in the rotating frame.
    """
    from scipy.optimize import brentq

    def dUdx(x):
        r1 = abs(x + mu)
        r2 = abs(x - 1.0 + mu)
        return x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x - 1.0 + mu) / r2**3

    # L1: between Earth and Moon
    xL1 = brentq(dUdx, -mu + 0.01, 1.0 - mu - 0.01)
    # L2: beyond Moon
    xL2 = brentq(dUdx, 1.0 - mu + 0.01, 1.0 - mu + 0.5)
    # L3: beyond Earth (opposite side)
    xL3 = brentq(dUdx, -2.0, -mu - 0.01)

    return xL1, xL2, xL3


# =============================================================================
# HALO ORBIT DIFFERENTIAL CORRECTION
# =============================================================================

def compute_halo_orbit(state_guess, T_guess=None, mu=MU, max_iter=100, tol=1e-8):
    """
    Full-period differential correction for 3D halo/NRHO orbits.

    Uses xz-plane symmetry: state₀ = [x0, 0, z0, 0, vy0, 0].
    Free variables: [x₀, z₀, ẏ₀, T]  (4 unknowns)
    Targets at T: x(T)=x₀, y(T)=0, z(T)=z₀, ẏ(T)=ẏ₀  (4 equations)

    This full-period approach is more robust than half-period methods
    for NRHOs, whose nearly-rectilinear geometry makes y=0 event
    detection at the half-period numerically fragile.

    Args:
        state_guess: (6,) initial guess [x0, 0, z0, 0, vy0, 0]
        T_guess: initial period estimate (default: use NRHO period)
        mu: mass parameter
        max_iter: maximum correction iterations
        tol: convergence tolerance on periodicity residual norm

    Returns:
        state0: (6,) corrected initial state
        T: full period (nondimensional)
        sol: OdeResult for one full period
    """
    state = state_guess.copy()
    if T_guess is None:
        T_guess = nrho_period()
    T = T_guess

    for iteration in range(max_iter):
        x0_val = state[0]
        z0_val = state[2]
        vy0_val = state[4]
        state_clean = np.array([x0_val, 0.0, z0_val, 0.0, vy0_val, 0.0])

        # Propagate with STM for one full period
        sol = propagate_with_stm(
            state_clean, (0, T), mu=mu,
            rtol=1e-12, atol=1e-14, max_step=0.005
        )

        Y_T = sol.sol(T)
        state_T = Y_T[:6]
        Phi_T = Y_T[6:42].reshape(6, 6)

        # Periodicity residual: we enforce symmetry conditions
        # x(T) = x0, y(T) = 0, z(T) = z0, vy(T) = vy0
        # (vx(T) = 0 and vz(T) = 0 follow from symmetry if the above hold)
        F = np.array([
            state_T[0] - x0_val,      # x(T) - x0
            state_T[1],                # y(T) - 0
            state_T[2] - z0_val,       # z(T) - z0
            state_T[4] - vy0_val,      # vy(T) - vy0
        ])

        error = np.linalg.norm(F)
        if error < tol:
            sol_full = propagate(state_clean, (0, T), mu=mu, max_step=0.001)
            return state_clean, T, sol_full

        # Jacobian: dF/d[x0, z0, vy0, T]  (4x4)
        # dF/dx0:    col = Φ[:,0] - e (where e accounts for the -x0 terms)
        # dF/dz0:    col = Φ[:,2] - e
        # dF/dvy0:   col = Φ[:,4] - e
        # dF/dT:     col = f(state_T) (the time derivative at t=T)

        f_T = cr3bp_ode(T, state_T, mu)

        J = np.zeros((4, 4))
        # Rows: [x(T)-x0, y(T), z(T)-z0, vy(T)-vy0]
        # dF[0]/dx0 = Φ[0,0] - 1,  dF[0]/dz0 = Φ[0,2],  dF[0]/dvy0 = Φ[0,4],  dF[0]/dT = f_T[0]
        # dF[1]/dx0 = Φ[1,0],       dF[1]/dz0 = Φ[1,2],  dF[1]/dvy0 = Φ[1,4],  dF[1]/dT = f_T[1]
        # dF[2]/dx0 = Φ[2,0],       dF[2]/dz0 = Φ[2,2]-1, dF[2]/dvy0 = Φ[2,4], dF[2]/dT = f_T[2]
        # dF[3]/dx0 = Φ[4,0],       dF[3]/dz0 = Φ[4,2],  dF[3]/dvy0 = Φ[4,4]-1, dF[3]/dT = f_T[4]

        rows = [0, 1, 2, 4]
        free_cols = [0, 2, 4]
        diag_offsets = {0: 0, 2: 2, 4: 3}  # which J column has the -1

        for i, row in enumerate(rows):
            for j, col in enumerate(free_cols):
                J[i, j] = Phi_T[row, col]
            J[i, 3] = f_T[row]   # dF/dT

        # Subtract identity contributions
        J[0, 0] -= 1.0  # d(x(T)-x0)/dx0 = Φ[0,0] - 1
        J[2, 1] -= 1.0  # d(z(T)-z0)/dz0 = Φ[2,2] - 1
        J[3, 2] -= 1.0  # d(vy(T)-vy0)/dvy0 = Φ[4,4] - 1

        # Newton update: dx = -J^{-1} F
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(J, -F, rcond=None)[0]

        # Damped update for large errors
        alpha = 1.0 if error < 1e-4 else 0.5
        state[0] += alpha * dx[0]   # x0
        state[2] += alpha * dx[1]   # z0
        state[4] += alpha * dx[2]   # vy0
        # Bound period update to ±10% to prevent family-jumping
        dT = np.clip(alpha * dx[3], -0.1 * T, 0.1 * T)
        T += dT

    raise RuntimeError(f"Halo orbit correction did not converge after {max_iter} iterations (error = {error:.2e})")


# =============================================================================
# LEO STATE IN ROTATING FRAME
# =============================================================================

def leo_state(altitude_km=185.0, inclination_deg=28.5, raan_deg=0.0,
              true_anomaly_deg=0.0, mu=MU):
    """
    Construct a circular LEO state in the CR3BP rotating frame.

    The LEO is centered on Earth at (-μ, 0, 0). The orbit plane is tilted
    by `inclination_deg` from the xy-plane (Earth-Moon orbital plane).

    In the rotating frame, the velocity includes the circular orbital velocity
    MINUS the rotating frame velocity (ω × r contribution).

    Args:
        altitude_km: LEO altitude above Earth surface (km)
        inclination_deg: orbital inclination w.r.t. Earth-Moon plane (deg)
        raan_deg: right ascension of ascending node in rotating frame (deg)
        true_anomaly_deg: true anomaly on LEO at departure (deg)
        mu: mass parameter

    Returns:
        state: (6,) nondimensional state [x, y, z, vx, vy, vz] in rotating frame
    """
    r_km = R_EARTH + altitude_km
    r_nd = r_km / L_STAR  # nondimensional radius

    # Circular velocity in DIMENSIONAL units, then nondimensionalize
    v_circ_km_s = np.sqrt(MU_EARTH / r_km)
    v_circ_nd = v_circ_km_s / V_STAR

    # Convert angles to radians
    inc = np.radians(inclination_deg)
    Omega = np.radians(raan_deg)
    nu = np.radians(true_anomaly_deg)

    # Position in the perifocal frame (circular orbit: r is constant)
    # argument of latitude u = omega + nu, for circular orbit omega is arbitrary
    u = nu  # argument of latitude

    # Rotation matrices: R3(-Ω) @ R1(-i) @ R3(-u)
    # Position in Earth-centered inertial-like frame (aligned with rotating frame at t=0)
    cos_u, sin_u = np.cos(u), np.sin(u)
    cos_O, sin_O = np.cos(Omega), np.sin(Omega)
    cos_i, sin_i = np.cos(inc), np.sin(inc)

    # Position unit vector in the orbital plane → 3D
    r_hat = np.array([
        cos_O * cos_u - sin_O * sin_u * cos_i,
        sin_O * cos_u + cos_O * sin_u * cos_i,
        sin_u * sin_i
    ])

    # Velocity direction (perpendicular to r_hat in the orbital plane, prograde)
    v_hat = np.array([
        -cos_O * sin_u - sin_O * cos_u * cos_i,
        -sin_O * sin_u + cos_O * cos_u * cos_i,
        cos_u * sin_i
    ])

    # Position and velocity in Earth-centered frame (nondimensional)
    r_earth_centered = r_nd * r_hat
    v_inertial = v_circ_nd * v_hat

    # Translate to barycentric rotating frame
    # Earth is at (-μ, 0, 0), so r_bary = r_earth + (-μ, 0, 0)
    r_rotating = r_earth_centered + EARTH_POS

    # Convert inertial velocity to rotating frame:
    # v_rotating = v_inertial - ω × r_rotating
    # where ω = [0, 0, 1] in nondimensional units
    omega_cross_r = np.array([-r_rotating[1], r_rotating[0], 0.0])
    v_rotating = v_inertial - omega_cross_r

    return np.concatenate([r_rotating, v_rotating])


# =============================================================================
# NRHO VALIDATION
# =============================================================================

def validate_nrho(verbose=True):
    """
    Validate the 9:2 NRHO by propagating one full period.

    Checks:
      1. Jacobi constant conservation (should be ~10⁻¹² or better)
      2. State periodicity |x(T) - x(0)|
      3. Perilune and apolune radii vs expected values
      4. 9:2 synodic resonance

    Returns:
        dict with validation results
    """
    state0 = nrho_state()
    T = nrho_period()

    if verbose:
        print("=" * 70)
        print("VALIDATING 9:2 L2 SOUTHERN NRHO")
        print("=" * 70)
        print(f"\n  Initial state (apolune, y=0 crossing):")
        print(f"    x₀  = {state0[0]:.10f}")
        print(f"    z₀  = {state0[2]:.10f}")
        print(f"    ẏ₀  = {state0[4]:.10f}")
        print(f"  Period = {T:.6f} nondim = {T * T_STAR / 86400:.3f} days")

    # Propagate one period
    sol = propagate(state0, (0, T), max_step=0.001)
    state_final = sol.sol(T)

    # Jacobi constant
    J0 = jacobi_constant(state0)
    Jf = jacobi_constant(state_final)

    # Periodicity
    periodicity_error = np.linalg.norm(state_final - state0)

    # Perilune and apolune
    t_eval = np.linspace(0, T, 10000)
    states = sol.sol(t_eval)
    d_moon = np.array([dist_from_moon(states[:, i]) for i in range(len(t_eval))])
    d_moon_km = d_moon * L_STAR

    perilune_km = d_moon_km.min()
    apolune_km = d_moon_km.max()
    perilune_alt_km = perilune_km - R_MOON
    apolune_alt_km = apolune_km - R_MOON

    # 9:2 resonance check
    # Use JPL dimensional units for the resonance comparison, since the 9:2
    # synodic resonance is defined relative to the real Moon's synodic period
    # (an ephemeris property, not a pure CR3BP property).
    T_days_jpl = T * JPL_TUNIT / 86400.0
    T_days = T * T_STAR / 86400.0  # CR3BP dimensional (slightly different)
    synodic_period = 29.53059  # days
    resonance_error = abs(9 * T_days_jpl - 2 * synodic_period)

    results = {
        'jacobi_initial': J0,
        'jacobi_final': Jf,
        'jacobi_error': abs(Jf - J0),
        'periodicity_error': periodicity_error,
        'perilune_km': perilune_km,
        'perilune_alt_km': perilune_alt_km,
        'apolune_km': apolune_km,
        'apolune_alt_km': apolune_alt_km,
        'period_days': T_days,
        'resonance_error_days': resonance_error,
        'sol': sol,
        'states': states,
        't_eval': t_eval,
    }

    if verbose:
        print(f"\n  --- Jacobi Constant ---")
        print(f"    C(0) = {J0:.10f}")
        print(f"    C(T) = {Jf:.10f}")
        print(f"    |ΔC| = {abs(Jf - J0):.2e}")

        print(f"\n  --- Periodicity ---")
        print(f"    |x(T) - x(0)| = {periodicity_error:.2e}")

        print(f"\n  --- Perilune ---")
        print(f"    Distance from Moon = {perilune_km:.1f} km")
        print(f"    Altitude           = {perilune_alt_km:.1f} km")

        print(f"\n  --- Apolune ---")
        print(f"    Distance from Moon = {apolune_km:.1f} km")
        print(f"    Altitude           = {apolune_alt_km:.1f} km")

        print(f"\n  --- 9:2 Synodic Resonance (JPL dimensional units) ---")
        print(f"    Period (JPL dim) = {T_days_jpl:.3f} days")
        print(f"    Period (CR3BP dim) = {T_days:.3f} days")
        print(f"    9 × T_orbit (JPL) = {9 * T_days_jpl:.3f} days")
        print(f"    2 × T_synodic = {2 * synodic_period:.3f} days")
        print(f"    Error = {resonance_error:.4f} days ({resonance_error / (2 * synodic_period) * 100:.3f}%)")
        print(f"    (Note: resonance is an ephemeris property; CR3BP matches approximately)")

        # Pass/fail summary
        print(f"\n  --- Validation Summary ---")
        checks = [
            ("Jacobi conservation", abs(Jf - J0) < 1e-8, f"|ΔC| = {abs(Jf-J0):.2e}"),
            ("Periodicity",         periodicity_error < 1e-4, f"|Δx| = {periodicity_error:.2e}"),
            ("Perilune ~2931 km",   abs(perilune_km - 2931) < 200, f"{perilune_km:.0f} km"),
            # Apolune in km differs from JPL because L_STAR=384400 vs JPL's 389703
            # Nondim apolune matches; dimensional ≈ 71395 * (384400/389703) ≈ 70422
            ("Apolune ~70420 km",   abs(apolune_km - 70420) < 500, f"{apolune_km:.0f} km (L*=384400)"),
            ("9:2 resonance",       resonance_error < 0.1, f"error = {resonance_error:.4f} days"),
        ]
        all_pass = True
        for name, passed, detail in checks:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"    [{status}] {name}: {detail}")

        print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    return results


# =============================================================================
# MAIN — run NRHO validation if executed directly
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("CR3BP 3D INFRASTRUCTURE MODULE")
    print("=" * 70)
    print(f"\n  Earth-Moon CR3BP parameters:")
    print(f"    μ  = {MU}")
    print(f"    L* = {L_STAR} km")
    print(f"    T* = {T_STAR:.2f} s = {T_STAR/86400:.4f} days")
    print(f"    V* = {V_STAR:.6f} km/s")

    # Validate NRHO
    results = validate_nrho(verbose=True)

    # Show LEO departure state
    print("\n" + "=" * 70)
    print("LEO DEPARTURE STATE (185 km, 28.5° inclination)")
    print("=" * 70)

    for nu_deg in [0, 90, 180, 270]:
        s = leo_state(true_anomaly_deg=nu_deg)
        r_earth_km = dist_from_earth(s) * L_STAR
        print(f"\n  ν = {nu_deg:3d}°: r_Earth = {r_earth_km:.1f} km")
        print(f"    state = [{s[0]:.8f}, {s[1]:.8f}, {s[2]:.8f},")
        print(f"             {s[3]:.8f}, {s[4]:.8f}, {s[5]:.8f}]")
        print(f"    Jacobi C = {jacobi_constant(s):.4f}")

    # Compute libration points
    xL1, xL2, xL3 = collinear_libration_points()
    print(f"\n  Libration points:")
    print(f"    L1: x = {xL1:.10f}  ({(xL1 - (1-MU)) * L_STAR:.0f} km from Moon)")
    print(f"    L2: x = {xL2:.10f}  ({(xL2 - (1-MU)) * L_STAR:.0f} km from Moon)")
    print(f"    L3: x = {xL3:.10f}")

    # Test halo orbit differential correction from a perturbed NRHO state
    print("\n" + "=" * 70)
    print("HALO ORBIT DIFFERENTIAL CORRECTION TEST")
    print("=" * 70)

    guess = nrho_state()
    # Perturb by 0.1% (NRHOs have a small basin of convergence)
    guess[0] *= 1.001
    guess[2] *= 1.001
    guess[4] *= 1.001
    print(f"  Perturbed initial guess (0.1% from JPL):")
    print(f"    x₀ = {guess[0]:.10f}, z₀ = {guess[2]:.10f}, ẏ₀ = {guess[4]:.10f}")

    try:
        state_corr, T_corr, sol_corr = compute_halo_orbit(guess)
        J_corr = jacobi_constant(state_corr)
        print(f"\n  Corrected state:")
        print(f"    x₀ = {state_corr[0]:.10f}, z₀ = {state_corr[2]:.10f}, ẏ₀ = {state_corr[4]:.10f}")
        print(f"    Period = {T_corr:.6f} ({T_corr * T_STAR / 86400:.3f} days)")
        print(f"    Jacobi = {J_corr:.6f}")
        print(f"    Periodicity = {np.linalg.norm(sol_corr.sol(T_corr) - state_corr):.2e}")
        print(f"  Differential correction: SUCCESS")
    except RuntimeError as e:
        print(f"  Differential correction: FAILED — {e}")
