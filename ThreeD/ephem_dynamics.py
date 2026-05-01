#!/usr/bin/env python3
"""
ephem_dynamics.py — Shared N-Body Ephemeris Dynamics (EME2000)

Newtonian 4-body propagator (Earth central + Moon + Sun) in the
Earth-centered EME2000 (J2000) inertial frame.

Factored out of artemis2_ephemeris.py so that the Artemis 2 modules
and the LEO→NRHO transfer (leo_to_nrho_ephem.py) share the same
dynamics and frame transforms.

Dynamics (units: km, s, km/s²):
    a = -μ_E·r / |r|³
      + μ_M·(r_M - r)/|r_M - r|³  -  μ_M·r_M/|r_M|³        (Moon 3rd body)
      + μ_S·(r_S - r)/|r_S - r|³  -  μ_S·r_S/|r_S|³        (Sun  3rd body)
      + u(t)                                                (control)

where r_M, r_S are Earth-centered EME2000 positions of the Moon and
Sun, queried from astropy's built-in JPL ephemeris.

Also provides:
  • CasADi-compatible bspline interpolants of Moon/Sun positions
    for use inside IPOPT-wrapped RK4 integrators
  • Rotating-frame (CR3BP nondim) ↔ EME2000 transform for seeding
    NRHO targets from cr3bp_3d.py states

Author: Zhuorui, AAE 568 Spring 2026
"""

import numpy as np
from scipy.interpolate import CubicSpline

from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time
import astropy.units as u

solar_system_ephemeris.set('builtin')

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


# =============================================================================
# CONSTANTS
# =============================================================================

MU_EARTH = 398600.4418          # km^3/s^2
MU_MOON  = 4902.800066          # km^3/s^2
MU_SUN   = 1.32712440018e11     # km^3/s^2

R_EARTH  = 6378.137             # km — equatorial radius
R_MOON   = 1737.4               # km

# CR3BP nondim reference units (must match cr3bp_3d.py)
CR3BP_L_STAR = 384400.0                                        # km
CR3BP_MU     = 0.012150585609624
CR3BP_T_STAR = np.sqrt(CR3BP_L_STAR**3 / (MU_EARTH + MU_MOON)) # s
CR3BP_V_STAR = CR3BP_L_STAR / CR3BP_T_STAR                     # km/s


# =============================================================================
# BODY POSITIONS FROM EPHEMERIS
# =============================================================================

def get_body_state_eme2000(body_name, epoch):
    """
    (r, v) of `body_name` in Earth-centered EME2000 at `epoch` (astropy Time).
    Returns km, km/s.
    """
    pos_b, vel_b = get_body_barycentric_posvel(body_name, epoch)
    pos_e, vel_e = get_body_barycentric_posvel('earth',    epoch)
    r = (pos_b - pos_e).xyz.to(u.km).value
    v = (vel_b - vel_e).xyz.to(u.km / u.s).value
    return np.asarray(r).reshape(3), np.asarray(v).reshape(3)


def body_positions_on_grid(epoch_0, t_sec_grid):
    """
    Precompute Moon and Sun positions (Earth-centered EME2000) on a time grid.

    Args:
        epoch_0: astropy Time — grid origin
        t_sec_grid: (N,) seconds after epoch_0

    Returns:
        moon_xyz: (N, 3) km
        sun_xyz:  (N, 3) km
    """
    epochs = epoch_0 + t_sec_grid * u.s
    mp, _ = get_body_barycentric_posvel('moon',  epochs)
    sp, _ = get_body_barycentric_posvel('sun',   epochs)
    ep, _ = get_body_barycentric_posvel('earth', epochs)
    # astropy CartesianRepresentation .xyz returns shape (3, N); transpose to (N, 3)
    moon_xyz = (mp - ep).xyz.to(u.km).value.T
    sun_xyz  = (sp - ep).xyz.to(u.km).value.T
    return moon_xyz, sun_xyz


# =============================================================================
# INTERPOLATORS (scipy + CasADi)
# =============================================================================

def build_scipy_interp(t_grid, xyz):
    """Return f(t) → (3,) via cubic spline. Accepts scalar or array t."""
    cs = CubicSpline(t_grid, xyz, axis=0)
    return lambda t: np.asarray(cs(t)).reshape(-1)


def build_casadi_interp(t_grid, xyz, name_prefix='body'):
    """
    CasADi bspline interpolant for a (N,3) time-series.
    Returns a CasADi Function(t) → (3,) SX.
    """
    if not HAS_CASADI:
        raise ImportError("CasADi not installed")
    t_list = [list(t_grid)]  # CasADi expects list-of-lists for grid points
    fx = ca.interpolant(f'{name_prefix}_x', 'bspline', t_list, list(xyz[:, 0]))
    fy = ca.interpolant(f'{name_prefix}_y', 'bspline', t_list, list(xyz[:, 1]))
    fz = ca.interpolant(f'{name_prefix}_z', 'bspline', t_list, list(xyz[:, 2]))
    t = ca.MX.sym('t')
    pos = ca.vertcat(fx(t), fy(t), fz(t))
    return ca.Function(f'{name_prefix}_pos', [t], [pos])


# =============================================================================
# EQUATIONS OF MOTION
# =============================================================================

def ephem_accel(r, r_moon, r_sun, u_ctrl=None):
    """
    Newtonian 4-body accel on a spacecraft at r (Earth-centered EME2000).

    Returns (3,) accel [km/s²].
    """
    a = -MU_EARTH * r / np.linalg.norm(r)**3

    d_m = r_moon - r
    a += MU_MOON * (d_m / np.linalg.norm(d_m)**3 - r_moon / np.linalg.norm(r_moon)**3)

    d_s = r_sun - r
    a += MU_SUN * (d_s / np.linalg.norm(d_s)**3 - r_sun / np.linalg.norm(r_sun)**3)

    if u_ctrl is not None:
        a += u_ctrl
    return a


def ephem_ode(t, x, moon_fn, sun_fn, u_fn=None):
    """
    6D state EOM for scipy.integrate.solve_ivp.

    Args:
        t: seconds from epoch_0
        x: (6,) [r (km), v (km/s)]
        moon_fn, sun_fn: callables t → (3,) [km]
        u_fn: callable t → (3,) [km/s²], or None

    Returns:
        (6,) dx/dt
    """
    r = x[:3]; v = x[3:]
    r_m = np.asarray(moon_fn(t)).reshape(3)
    r_s = np.asarray(sun_fn(t)).reshape(3)
    u_ctrl = None if u_fn is None else np.asarray(u_fn(t)).reshape(3)
    a = ephem_accel(r, r_m, r_s, u_ctrl)
    return np.concatenate([v, a])


def ephem_jacobian(r, r_moon, r_sun):
    """
    ∂a/∂r at (r, r_moon, r_sun). Needed for STM and costate propagation.

    For a point-mass body at position p with parameter μ, the accel on a
    satellite at r (taking r_rel = p - r) is  μ·(p - r)/|p - r|³  (+ indirect).
    Its derivative w.r.t. r is  -μ·(I/|p - r|³  -  3·(p - r)(p - r)ᵀ/|p - r|⁵).

    Earth-central term: a = -μ_E·r/|r|³, so ∂a/∂r = -μ_E·(I/|r|³ - 3·r·rᵀ/|r|⁵).
    """
    def J_third_body(mu, d):
        dn = np.linalg.norm(d)
        return -mu * (np.eye(3) / dn**3 - 3.0 * np.outer(d, d) / dn**5)

    rn = np.linalg.norm(r)
    A  = -MU_EARTH * (np.eye(3) / rn**3 - 3.0 * np.outer(r, r) / rn**5)
    A += J_third_body(MU_MOON, r_moon - r)
    A += J_third_body(MU_SUN,  r_sun  - r)
    return A


def ephem_ode_costate(t, Y, moon_fn, sun_fn):
    """
    12D state+costate EOM for indirect shooting (min-energy, u* = -½ λ_v).

    Y = [r (3), v (3), λ_r (3), λ_v (3)]

    EOMs (unconstrained control):
        dr/dt   =  v
        dv/dt   =  f(r, t) - ½ λ_v
        dλ_r/dt = -Aᵀ λ_v     (A = ∂f/∂r)
        dλ_v/dt = -λ_r
    """
    r = Y[0:3]; v = Y[3:6]
    lam_r = Y[6:9]; lam_v = Y[9:12]

    r_m = np.asarray(moon_fn(t)).reshape(3)
    r_s = np.asarray(sun_fn(t)).reshape(3)

    u_ctrl = -0.5 * lam_v
    a = ephem_accel(r, r_m, r_s, u_ctrl)
    A = ephem_jacobian(r, r_m, r_s)

    return np.concatenate([v, a, -(A.T @ lam_v), -lam_r])


# =============================================================================
# ROTATING-FRAME (CR3BP nondim) ↔ EME2000 TRANSFORMS
# =============================================================================

def rotating_basis_from_moon(r_moon, v_moon):
    """
    Instantaneous Earth-Moon rotating basis, expressed in EME2000.

        x̂ = r_M / |r_M|                 Earth→Moon line
        ẑ = (r_M × v_M) / |r_M × v_M|   orbit normal
        ŷ = ẑ × x̂

    Returns:
        R: (3, 3) rotation matrix (rotating → EME2000; columns are x̂, ŷ, ẑ)
        omega: (3,) angular velocity of rotating frame [rad/s]  (in EME2000)
        L_inst: instantaneous Earth-Moon distance [km]
    """
    L_inst = np.linalg.norm(r_moon)
    x_hat = r_moon / L_inst
    h = np.cross(r_moon, v_moon)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)
    R = np.column_stack([x_hat, y_hat, z_hat])
    # ω for circular motion: ω = (r × v) / |r|²  (in inertial frame)
    omega = h / L_inst**2
    return R, omega, L_inst


def rot_nondim_to_eme2000(x_rot_nondim, epoch):
    """
    Map CR3BP-nondim state (rotating frame, barycenter origin) → dimensional
    Earth-centered EME2000 at `epoch`.

    Conventions:
      • Input position/velocity are in CR3BP units with origin at the
        Earth-Moon barycenter and scales L* = CR3BP_L_STAR, T* = CR3BP_T_STAR.
      • Earth sits at (-μ, 0, 0) and Moon at (1-μ, 0, 0) in these units.
      • Output is Earth-centered EME2000, dimensional km and km/s.

    Note:
      This is an approximation — the CR3BP assumes a circular Moon orbit at
      a constant L*, while the ephemeris Moon has an eccentric orbit with
      time-varying distance. Use this as a starting target state; let IPOPT
      refine insertion conditions (phase, exact state) as free variables.
    """
    r_moon, v_moon = get_body_state_eme2000('moon', epoch)
    R, omega, L_inst = rotating_basis_from_moon(r_moon, v_moon)

    # Dimensionalize (use the reference CR3BP scales, NOT L_inst — those are
    # the scales under which x_rot_nondim was defined)
    r_rot = x_rot_nondim[:3] * CR3BP_L_STAR
    v_rot = x_rot_nondim[3:] * CR3BP_V_STAR

    # Shift origin from Earth-Moon barycenter to Earth center.
    # Earth sits at rotating-frame position (-μ · L*, 0, 0), so the vector
    # from Earth to the barycenter is (+μ · L*, 0, 0).
    earth_to_bary_rot = np.array([+CR3BP_MU * CR3BP_L_STAR, 0.0, 0.0])
    r_from_earth_rot  = r_rot + earth_to_bary_rot

    # Rotating → inertial:  r_I = R r_rot,   v_I = R v_rot + ω × r_I
    r_eme = R @ r_from_earth_rot
    v_eme = R @ v_rot + np.cross(omega, r_eme)

    return np.concatenate([r_eme, v_eme])


def eme2000_to_rot_nondim(x_eme, epoch):
    """Inverse of rot_nondim_to_eme2000 — useful for post-processing plots."""
    r_moon, v_moon = get_body_state_eme2000('moon', epoch)
    R, omega, L_inst = rotating_basis_from_moon(r_moon, v_moon)

    r_eme = x_eme[:3]; v_eme = x_eme[3:]
    r_rot_dim = R.T @ r_eme
    v_rot_dim = R.T @ (v_eme - np.cross(omega, r_eme))

    earth_to_bary_rot = np.array([+CR3BP_MU * CR3BP_L_STAR, 0.0, 0.0])
    r_rot_from_bary = r_rot_dim - earth_to_bary_rot

    return np.concatenate([r_rot_from_bary / CR3BP_L_STAR,
                           v_rot_dim / CR3BP_V_STAR])


# =============================================================================
# QUICK SELF-TEST
# =============================================================================

if __name__ == '__main__':
    print("ephem_dynamics.py self-test")
    epoch = Time('2027-12-01T00:00:00', scale='utc')
    r_m, v_m = get_body_state_eme2000('moon', epoch)
    r_s, v_s = get_body_state_eme2000('sun',  epoch)
    print(f"  Moon |r| at {epoch.iso}: {np.linalg.norm(r_m):.1f} km "
          f"(nominal ~384,400)")
    print(f"  Sun  |r| at {epoch.iso}: {np.linalg.norm(r_s)/1e6:.3f} Gm "
          f"(nominal ~149.6)")

    # Accel at ISS altitude, check that Earth term dominates
    r = np.array([R_EARTH + 400.0, 0.0, 0.0])
    a = ephem_accel(r, r_m, r_s)
    print(f"  |a| at ISS alt: {np.linalg.norm(a):.4f} km/s² "
          f"(nominal ~9.8e-3)")

    # Round-trip frame transform
    x_test = np.array([0.98, 0.01, 0.02, 0.0, 0.1, 0.05])
    x_eme  = rot_nondim_to_eme2000(x_test, epoch)
    x_back = eme2000_to_rot_nondim(x_eme, epoch)
    err = np.linalg.norm(x_back - x_test)
    print(f"  Rot→EME→Rot round-trip error: {err:.3e} (should be ~0)")
