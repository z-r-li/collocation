#!/usr/bin/env python3
"""
artemis2_ephemeris.py - Ephemeris-Driven Propagator for Artemis II

Instead of CR3BP (circular Moon orbit, no Sun), this uses astropy's JPL
ephemeris to get actual Moon and Sun positions at each integration step,
then computes gravitational accelerations from all three bodies.

Dynamics: Newtonian N-body in Earth-centered EME2000 (J2000) inertial frame
  a = -mu_E * r / |r|^3                     (Earth, central body)
    + mu_M * (r_M - r)/|r_M - r|^3          (Moon, 3rd body)
    - mu_M * r_M / |r_M|^3                  (indirect term)
    + mu_S * (r_S - r)/|r_S - r|^3          (Sun, 3rd body)
    - mu_S * r_S / |r_S|^3                  (indirect term)

Comparison:
  1. Ballistic propagation (no control) from OEM initial state
  2. Shooting method: find initial costate to match OEM endpoint
  3. CasADi/IPOPT Bézier collocation (minimum-energy optimal control)
  4. All compared against NASA OEM ephemeris

Author: Zhuorui, AAE 568 Spring 2026
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, minimize
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Astropy for ephemeris
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time
import astropy.units as u

solar_system_ephemeris.set('builtin')

# Optional: CasADi
try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

# ============================================================================
# PHYSICAL CONSTANTS (km, s)
# ============================================================================
MU_EARTH = 398600.4418        # km^3/s^2
MU_MOON  = 4902.800066        # km^3/s^2
MU_SUN   = 132712440041.93938 # km^3/s^2

# Nondimensionalization
L_STAR = 384400.0             # km (Earth-Moon distance)
T_STAR = 2360591.51           # seconds (Moon orbital period / 2pi)
V_STAR = L_STAR / T_STAR      # km/s

print("\n" + "=" * 80)
print("ARTEMIS II — EPHEMERIS-DRIVEN PROPAGATOR")
print("=" * 80)
print(f"\nGravitational parameters:")
print(f"  mu_Earth = {MU_EARTH:.4f} km^3/s^2")
print(f"  mu_Moon  = {MU_MOON:.6f} km^3/s^2")
print(f"  mu_Sun   = {MU_SUN:.5f} km^3/s^2")
print(f"  CasADi available: {HAS_CASADI}")


# ============================================================================
# SECTION 1: OEM PARSER (reused)
# ============================================================================

def parse_oem(filename):
    """Parse CCSDS OEM v2.0 file — returns datetimes, positions (km), velocities (km/s)."""
    print("\n" + "-" * 60)
    print("Parsing OEM file...")

    times_utc = []
    positions = []
    velocities = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() and line[0].isdigit() and 'T' in line:
            data_start = i
            break

    for line in lines[data_start:]:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            t_utc = datetime.fromisoformat(parts[0])
            pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            vel = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
            times_utc.append(t_utc)
            positions.append(pos)
            velocities.append(vel)
        except (ValueError, IndexError):
            continue

    positions = np.array(positions)
    velocities = np.array(velocities)

    print(f"  Parsed {len(times_utc)} state vectors")
    print(f"  Time span: {times_utc[0]} → {times_utc[-1]}")
    print(f"  Duration: {(times_utc[-1] - times_utc[0]).total_seconds()/86400:.2f} days")

    return times_utc, positions, velocities


# ============================================================================
# SECTION 2: ASTROPY EPHEMERIS LOOKUP
# ============================================================================

def get_moon_sun_eci(t_utc):
    """
    Get Moon and Sun positions in Earth-centered EME2000 (J2000) frame.

    Uses astropy's built-in ephemeris (low-precision DE405 fit).

    Args:
        t_utc: datetime object (UTC)

    Returns:
        r_moon: (3,) Moon position relative to Earth [km]
        r_sun:  (3,) Sun position relative to Earth [km]
    """
    t = Time(t_utc, scale='utc')

    # Barycentric positions
    earth_pos = get_body_barycentric_posvel('earth', t)[0].xyz.to(u.km).value
    moon_pos  = get_body_barycentric_posvel('moon', t)[0].xyz.to(u.km).value
    sun_pos   = get_body_barycentric_posvel('sun', t)[0].xyz.to(u.km).value

    r_moon = moon_pos - earth_pos  # Earth-centered
    r_sun  = sun_pos  - earth_pos  # Earth-centered

    return r_moon, r_sun


# Precompute and cache ephemeris over time grid for fast interpolation
class EphemerisCache:
    """
    Precompute Moon/Sun positions on a dense time grid, then interpolate.
    Uses vectorized astropy calls for speed (~0.1s for thousands of points).
    """

    def __init__(self, t_start_utc, t_end_utc, n_points=2000):
        print("  Building ephemeris cache...")
        dt_total = (t_end_utc - t_start_utc).total_seconds()

        # Time grid in seconds from t_start
        self.t_start = t_start_utc
        self.t_grid = np.linspace(0, dt_total, n_points)

        # Vectorized astropy call — build Time array all at once
        t_arr = Time([t_start_utc + timedelta(seconds=float(dt))
                      for dt in self.t_grid], scale='utc')

        earth_pos = get_body_barycentric_posvel('earth', t_arr)[0].xyz.to(u.km).value  # (3, N)
        moon_pos  = get_body_barycentric_posvel('moon', t_arr)[0].xyz.to(u.km).value
        sun_pos   = get_body_barycentric_posvel('sun', t_arr)[0].xyz.to(u.km).value

        r_moon_arr = (moon_pos - earth_pos).T  # (N, 3) Earth-centered
        r_sun_arr  = (sun_pos  - earth_pos).T

        # Build cubic interpolants
        self.moon_interp = interp1d(self.t_grid, r_moon_arr, axis=0, kind='cubic',
                                     fill_value='extrapolate')
        self.sun_interp  = interp1d(self.t_grid, r_sun_arr, axis=0, kind='cubic',
                                     fill_value='extrapolate')

        print(f"    Cached {n_points} ephemeris points over {dt_total/86400:.2f} days")
        print(f"    Moon distance range: "
              f"{np.linalg.norm(r_moon_arr, axis=1).min():.0f} – "
              f"{np.linalg.norm(r_moon_arr, axis=1).max():.0f} km")
        print(f"    Sun distance range: "
              f"{np.linalg.norm(r_sun_arr, axis=1).min():.0f} – "
              f"{np.linalg.norm(r_sun_arr, axis=1).max():.0f} km")

    def get_positions(self, t_sec):
        """Get Moon and Sun positions at t_sec seconds after t_start."""
        return self.moon_interp(t_sec), self.sun_interp(t_sec)


# ============================================================================
# SECTION 3: EPHEMERIS-DRIVEN DYNAMICS (ECI FRAME)
# ============================================================================

def dynamics_ephemeris(t_sec, state, ephem_cache, with_control=False):
    """
    Full N-body dynamics in Earth-centered inertial frame.

    state = [x, y, z, vx, vy, vz]          if with_control=False
    state = [x, y, z, vx, vy, vz, lam1..6] if with_control=True

    Accelerations:
      a_earth = -mu_E * r / |r|^3
      a_moon  = mu_M * [(r_M - r)/|r_M - r|^3 - r_M/|r_M|^3]
      a_sun   = mu_S * [(r_S - r)/|r_S - r|^3 - r_S/|r_S|^3]
    """
    r = state[0:3]
    v = state[3:6]

    r_norm = np.linalg.norm(r)

    # Earth central gravity
    a_earth = -MU_EARTH * r / r_norm**3

    # Moon and Sun from ephemeris
    r_moon, r_sun = ephem_cache.get_positions(t_sec)

    # Moon 3rd-body perturbation
    d_moon = r_moon - r
    d_moon_norm = np.linalg.norm(d_moon)
    r_moon_norm = np.linalg.norm(r_moon)
    a_moon = MU_MOON * (d_moon / d_moon_norm**3 - r_moon / r_moon_norm**3)

    # Sun 3rd-body perturbation
    d_sun = r_sun - r
    d_sun_norm = np.linalg.norm(d_sun)
    r_sun_norm = np.linalg.norm(r_sun)
    a_sun = MU_SUN * (d_sun / d_sun_norm**3 - r_sun / r_sun_norm**3)

    a_total = a_earth + a_moon + a_sun

    if not with_control:
        return np.concatenate([v, a_total])

    # With optimal control: u* = -0.5 * lambda_v (minimum-energy)
    lam = state[6:12]
    lam_v = lam[3:6]  # costate for velocity
    u_ctrl = -0.5 * lam_v  # optimal control
    a_total = a_total + u_ctrl

    # Costate dynamics: dlam/dt = -dH/dx
    # We need the gravity gradient tensor (Jacobian of acceleration w.r.t. position)
    # For Earth:
    I3 = np.eye(3)
    A_earth = -MU_EARTH * (I3 / r_norm**3 - 3.0 * np.outer(r, r) / r_norm**5)

    # For Moon:
    A_moon = -MU_MOON * (I3 / d_moon_norm**3 - 3.0 * np.outer(d_moon, d_moon) / d_moon_norm**5)

    # For Sun:
    A_sun = -MU_SUN * (I3 / d_sun_norm**3 - 3.0 * np.outer(d_sun, d_sun) / d_sun_norm**5)

    A_total = A_earth + A_moon + A_sun  # d(accel)/d(r)

    # Costate equations: dlam_r/dt = -A^T @ lam_v, dlam_v/dt = -lam_r
    lam_r = lam[0:3]
    dlam_r = -A_total.T @ lam_v
    dlam_v = -lam_r

    return np.concatenate([v, a_total, dlam_r, dlam_v])


# ============================================================================
# SECTION 4: BALLISTIC PROPAGATION (no control)
# ============================================================================

def propagate_ballistic(r0, v0, t_span_sec, ephem_cache, n_eval=500):
    """Propagate ballistic trajectory using ephemeris-driven dynamics."""
    print("\n" + "-" * 60)
    print("Ballistic propagation (no control)...")

    state0 = np.concatenate([r0, v0])
    t_eval = np.linspace(t_span_sec[0], t_span_sec[1], n_eval)

    sol = solve_ivp(
        lambda t, y: dynamics_ephemeris(t, y, ephem_cache, with_control=False),
        t_span_sec, state0, method='DOP853',
        rtol=1e-12, atol=1e-14, t_eval=t_eval, max_step=60.0
    )

    if not sol.success:
        print(f"  WARNING: Integration failed: {sol.message}")
    else:
        rf = sol.y[0:3, -1]
        print(f"  Final position: [{rf[0]:.2f}, {rf[1]:.2f}, {rf[2]:.2f}] km")
        print(f"  Integration steps: {len(sol.t)}")

    return sol


# ============================================================================
# SECTION 5: INDIRECT SHOOTING (minimum-energy, dimensional)
# ============================================================================

def shooting_residual(lam0, r0, v0, rf_target, vf_target, t_span_sec, ephem_cache):
    """
    Shoot from (r0, v0, lam0), integrate, return position+velocity mismatch at tf.
    """
    state0 = np.concatenate([r0, v0, lam0])

    sol = solve_ivp(
        lambda t, y: dynamics_ephemeris(t, y, ephem_cache, with_control=True),
        t_span_sec, state0, method='DOP853',
        rtol=1e-11, atol=1e-13, max_step=300.0
    )

    if not sol.success:
        return np.ones(6) * 1e10

    rf = sol.y[0:3, -1]
    vf = sol.y[3:6, -1]

    # Residual: position and velocity boundary conditions
    res = np.concatenate([rf - rf_target, vf - vf_target])
    return res


def solve_shooting(r0, v0, rf_target, vf_target, t_span_sec, ephem_cache, n_guesses=15,
                   seed_records=None, early_stop=True, on_seed_done=None):
    """
    Solve TPBVP via indirect shooting with multiple initial guesses.

    Parameters
    ----------
    seed_records : list or None
        If a list is passed, each seed's per-guess outcome is appended as a
        dict with keys {seed_index, seed_strategy, lam0_guess, lam0_sol,
        residual, nfev, wall_time_s, converged, exception, cost}. Used by the
        T3.2 instrumentation to write one ResultRecord per seed.
    early_stop : bool
        If True (default), breaks after first guess with residual < 1e-4.
        Set to False to run ALL 15 seeds to completion (needed for the
        "shooting fragility" sweep so every seed gets a record).
    on_seed_done : callable or None
        If set, called as `on_seed_done(entry)` immediately after each seed
        completes (before the next one starts). Used by the instrumented
        runner to persist per-seed records so a timeout mid-sweep still
        retains what's done.
    """
    import time as _time
    print("\n" + "-" * 60)
    print("Indirect shooting (minimum-energy, ephemeris dynamics)...")
    print(f"  Duration: {(t_span_sec[1]-t_span_sec[0])/86400:.2f} days, {n_guesses} guesses")

    best_res = np.inf
    best_lam0 = None
    best_sol = None

    # Initial guess strategies — label each one for the per-seed record
    guesses = []
    strategies = []  # parallel list of {"name": str, "scale": float}

    # Physics-based: align costate with velocity direction
    v_dir = (vf_target - v0) / np.linalg.norm(vf_target - v0)
    for scale in [1e-7, 1e-6, 1e-5, 1e-4]:
        lam0 = np.zeros(6)
        lam0[3:6] = scale * v_dir
        guesses.append(lam0)
        strategies.append({"name": "velocity_aligned", "scale": float(scale)})

    # Also try position-aligned
    r_dir = (rf_target - r0) / np.linalg.norm(rf_target - r0)
    for scale in [1e-8, 1e-7]:
        lam0 = np.zeros(6)
        lam0[0:3] = scale * r_dir
        guesses.append(lam0)
        strategies.append({"name": "position_aligned", "scale": float(scale)})

    # Random guesses
    rng = np.random.RandomState(42)
    for _ in range(n_guesses - len(guesses)):
        lam0 = rng.randn(6) * 1e-6
        guesses.append(lam0)
        strategies.append({"name": "random_normal", "scale": 1e-6})

    for i, lam0_guess in enumerate(guesses):
        strat = strategies[i]
        seed_t0 = _time.perf_counter()
        seed_entry = {
            "seed_index": i,
            "seed_strategy": strat["name"],
            "seed_scale": strat["scale"],
            "lam0_guess": lam0_guess.tolist(),
            "lam0_sol": None,
            "residual": None,
            "nfev": None,
            "wall_time_s": None,
            "converged": False,
            "exception": None,
            "cost": None,
        }
        try:
            lam0_sol, info, ier, msg = fsolve(
                shooting_residual, lam0_guess,
                args=(r0, v0, rf_target, vf_target, t_span_sec, ephem_cache),
                full_output=True, maxfev=80  # fewer evals per guess for speed
            )

            res_norm = float(np.linalg.norm(info['fvec']))
            seed_entry["lam0_sol"] = lam0_sol.tolist()
            seed_entry["residual"] = res_norm
            seed_entry["nfev"] = int(info.get("nfev", 0))
            seed_entry["converged"] = bool(res_norm < 1e-4)

            if res_norm < best_res:
                best_res = res_norm
                best_lam0 = lam0_sol

                status = "✓" if res_norm < 1.0 else "✗"
                print(f"  Guess {i+1:2d}: residual = {res_norm:.4e} {status}", flush=True)
            else:
                print(f"  Guess {i+1:2d}: residual = {res_norm:.4e} (not improved)", flush=True)

            if seed_records is not None:
                seed_entry["wall_time_s"] = float(_time.perf_counter() - seed_t0)
                seed_records.append(seed_entry)
            if on_seed_done is not None:
                try:
                    on_seed_done(dict(seed_entry))
                except Exception as cb_err:
                    print(f"  [on_seed_done raised {cb_err!r}, continuing]", flush=True)

            if early_stop and res_norm < 1e-4:
                # Fill in unrun seeds as 'not_attempted' so caller sees full list
                if seed_records is not None:
                    for j in range(i + 1, len(guesses)):
                        seed_records.append({
                            "seed_index": j,
                            "seed_strategy": strategies[j]["name"],
                            "seed_scale": strategies[j]["scale"],
                            "lam0_guess": guesses[j].tolist(),
                            "lam0_sol": None,
                            "residual": None,
                            "nfev": 0,
                            "wall_time_s": 0.0,
                            "converged": False,
                            "exception": "not_attempted_early_stop",
                            "cost": None,
                        })
                break  # good enough for 7.5-day arc
        except Exception as e:
            print(f"  Guess {i+1:2d}: exception ({type(e).__name__})", flush=True)
            seed_entry["exception"] = f"{type(e).__name__}: {e}"
            seed_entry["wall_time_s"] = float(_time.perf_counter() - seed_t0)
            if seed_records is not None:
                seed_records.append(seed_entry)
            if on_seed_done is not None:
                try:
                    on_seed_done(dict(seed_entry))
                except Exception as cb_err:
                    print(f"  [on_seed_done raised {cb_err!r}, continuing]", flush=True)
            continue

    if best_lam0 is None:
        print("  FAILED: No guess converged")
        return None

    print(f"\n  Best residual: {best_res:.4e}")

    # Full integration with best costate
    state0 = np.concatenate([r0, v0, best_lam0])
    duration = t_span_sec[1] - t_span_sec[0]
    n_eval_pts = max(500, int(duration / 120))  # at least one point per 2 min
    t_eval = np.linspace(t_span_sec[0], t_span_sec[1], n_eval_pts)

    sol = solve_ivp(
        lambda t, y: dynamics_ephemeris(t, y, ephem_cache, with_control=True),
        t_span_sec, state0, method='DOP853',
        rtol=1e-12, atol=1e-14, t_eval=t_eval, max_step=120.0
    )

    # Compute control history
    lam_v_hist = sol.y[9:12, :]  # costate for velocity
    u_hist = -0.5 * lam_v_hist   # optimal control
    u_mag = np.linalg.norm(u_hist, axis=0)
    cost = np.trapz(u_mag**2, sol.t)

    print(f"  Control cost (∫||u||² dt): {cost:.6e} km²/s⁴·s")
    print(f"  Max |u|: {u_mag.max():.6e} km/s²")

    # Record best-of-sweep cost onto the winning seed entry (if tracking)
    if seed_records is not None and best_lam0 is not None:
        for entry in seed_records:
            if (entry.get("lam0_sol") is not None
                    and np.allclose(entry["lam0_sol"], best_lam0, rtol=1e-8, atol=1e-12)):
                entry["cost"] = float(cost)
                break

    return sol, best_lam0, cost


# ============================================================================
# SECTION 6: IPOPT BÉZIER COLLOCATION (ephemeris dynamics)
# ============================================================================

def solve_ipopt_collocation(r0, v0, rf_target, vf_target, t_span_sec,
                            ephem_cache, n_seg=60, sol_shooting=None,
                            nasa_warmstart=None, stats_out=None):
    """
    Direct multiple-shooting using CasADi/IPOPT with ephemeris-driven dynamics.

    Key improvements over v1:
      - 60 segments (each ~48 min) instead of 12 (~4 hrs)
      - Explicit RK4 integration per segment (4 sub-steps)
      - Single control vector per segment (piecewise-constant)
      - Control magnitude bounded by 2× shooting max thrust
      - Warm-started from shooting solution if available

    Since ephemeris calls can't go through CasADi's AD, Moon/Sun positions
    are precomputed at each RK4 sub-step time and passed as numeric parameters.
    """
    if not HAS_CASADI:
        print("\n  CasADi not available — skipping IPOPT collocation")
        return None

    print("\n" + "-" * 60)
    print(f"IPOPT direct multiple-shooting (ephemeris, {n_seg} segments)...")

    t0, tf = t_span_sec
    seg_times = np.linspace(t0, tf, n_seg + 1)

    ns = 6   # state dim
    nd = 3   # control dim
    n_rk4 = 4  # RK4 sub-steps per segment

    # --- CasADi dynamics function ---
    # Build a CasADi function that takes (state, control, r_moon, r_sun) as inputs
    x_sym = ca.MX.sym('x', ns)
    u_sym = ca.MX.sym('u', nd)
    rm_sym = ca.MX.sym('rm', 3)
    rs_sym = ca.MX.sym('rs', 3)

    r_sc = x_sym[0:3]
    v_sc = x_sym[3:6]
    r_norm = ca.norm_2(r_sc)

    # Earth
    a_earth = -MU_EARTH * r_sc / r_norm**3

    # Moon
    d_moon = rm_sym - r_sc
    a_moon = MU_MOON * (d_moon / ca.norm_2(d_moon)**3 - rm_sym / ca.norm_2(rm_sym)**3)

    # Sun
    d_sun = rs_sym - r_sc
    a_sun = MU_SUN * (d_sun / ca.norm_2(d_sun)**3 - rs_sym / ca.norm_2(rs_sym)**3)

    xdot = ca.vertcat(v_sc, a_earth + a_moon + a_sun + u_sym)
    f_dyn = ca.Function('f_dyn', [x_sym, u_sym, rm_sym, rs_sym], [xdot])

    # --- Determine control bound from shooting ---
    u_max = 1e-4  # default: 0.1 mm/s² (very small)
    if sol_shooting is not None:
        lam_v_hist = sol_shooting[0].y[9:12, :]
        u_hist = -0.5 * lam_v_hist
        u_max_shooting = np.linalg.norm(u_hist, axis=0).max()
        u_max = 3.0 * u_max_shooting  # allow 3× headroom
        print(f"  Control bound: {u_max:.6e} km/s² (3× shooting max)")

    # --- Build NLP ---
    opti = ca.Opti()

    # Decision variables
    X = opti.variable(ns, n_seg + 1)  # state at each node
    U = opti.variable(nd, n_seg)      # piecewise-constant control per segment

    # Boundary conditions
    opti.subject_to(X[:, 0] == np.concatenate([r0, v0]))
    opti.subject_to(X[:, -1] == np.concatenate([rf_target, vf_target]))

    # Control bounds
    for k in range(n_seg):
        for d in range(nd):
            opti.subject_to(opti.bounded(-u_max, U[d, k], u_max))

    # Objective: minimize ∫||u||² dt
    J = 0
    for k in range(n_seg):
        h = seg_times[k + 1] - seg_times[k]
        J += ca.dot(U[:, k], U[:, k]) * h
    opti.minimize(J)

    # Dynamics constraints: RK4 integration per segment
    print(f"  Building RK4 defect constraints ({n_seg} segments × {n_rk4} sub-steps)...")
    for k in range(n_seg):
        t_k = seg_times[k]
        h_seg = seg_times[k + 1] - seg_times[k]
        h_rk = h_seg / n_rk4

        xk = X[:, k]
        uk = U[:, k]

        # Integrate with RK4 sub-steps
        x_curr = xk
        for s in range(n_rk4):
            t_s = t_k + s * h_rk

            # Precompute Moon/Sun at each RK4 evaluation point
            rm1, rs1 = ephem_cache.get_positions(t_s)
            rm2, rs2 = ephem_cache.get_positions(t_s + 0.5 * h_rk)
            rm3, rs3 = ephem_cache.get_positions(t_s + h_rk)

            k1 = f_dyn(x_curr, uk, rm1, rs1)
            k2 = f_dyn(x_curr + 0.5 * h_rk * k1, uk, rm2, rs2)
            k3 = f_dyn(x_curr + 0.5 * h_rk * k2, uk, rm2, rs2)
            k4 = f_dyn(x_curr + h_rk * k3, uk, rm3, rs3)

            x_curr = x_curr + (h_rk / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Continuity constraint
        opti.subject_to(X[:, k + 1] == x_curr)

    # --- Initial guess ---
    if sol_shooting is not None:
        print("  Warm-starting from shooting solution...")
        sol_sh = sol_shooting[0]
        sh_interp = interp1d(sol_sh.t, sol_sh.y[0:6, :], axis=1, fill_value='extrapolate')
        sh_u_interp = interp1d(sol_sh.t, -0.5 * sol_sh.y[9:12, :], axis=1, fill_value='extrapolate')

        for i in range(n_seg + 1):
            x_guess = sh_interp(seg_times[i])
            opti.set_initial(X[:, i], x_guess)

        for k in range(n_seg):
            t_mid = 0.5 * (seg_times[k] + seg_times[k + 1])
            u_guess = sh_u_interp(t_mid)
            opti.set_initial(U[:, k], u_guess)
    elif nasa_warmstart is not None:
        print("  Warm-starting from NASA OEM data...")
        nasa_t, nasa_pos, nasa_vel = nasa_warmstart
        # Build interpolants from NASA data
        nasa_pos_interp = interp1d(nasa_t, nasa_pos, axis=0, fill_value='extrapolate')
        nasa_vel_interp = interp1d(nasa_t, nasa_vel, axis=0, fill_value='extrapolate')

        for i in range(n_seg + 1):
            p = nasa_pos_interp(seg_times[i])
            v = nasa_vel_interp(seg_times[i])
            x_guess = np.concatenate([p, v])
            opti.set_initial(X[:, i], x_guess)

        opti.set_initial(U, 0)  # zero control — NASA trajectory is nearly ballistic
    else:
        # Linear interpolation fallback
        state0 = np.concatenate([r0, v0])
        statef = np.concatenate([rf_target, vf_target])
        for i in range(n_seg + 1):
            alpha = i / n_seg
            opti.set_initial(X[:, i], (1 - alpha) * state0 + alpha * statef)
        opti.set_initial(U, 0)

    # --- IPOPT options ---
    opts = {
        'ipopt.print_level': 3,
        'ipopt.max_iter': 3000,
        'ipopt.tol': 1e-8,
        'ipopt.acceptable_tol': 1e-6,
        'print_time': False,
        'ipopt.linear_solver': 'mumps',
        'ipopt.warm_start_init_point': 'yes',
    }
    opti.solver('ipopt', opts)

    # --- Iteration-history capture for T3.4 convergence plot ---
    # CasADi exposes per-iteration stats via opti.stats()["iterations"], but
    # only *after* the solve and only when the solver internally logs them.
    # Under CasADi 3.x the Ipopt interface populates iterations with keys
    # {"obj", "inf_pr", "inf_du", "iter"}. We snapshot that list into
    # stats_out["convergence_history"] for downstream plotting.
    #
    # We also record solver.stats() summary (iter_count, return_status,
    # t_wall_total) into stats_out.
    def _capture_stats(solve_ok, err_msg=None):
        if stats_out is None:
            return
        try:
            s = opti.stats()
        except Exception:
            s = {}
        stats_out["raw_stats"] = s
        stats_out["return_status"] = s.get("return_status")
        stats_out["iter_count"] = s.get("iter_count")
        stats_out["t_wall_total"] = s.get("t_wall_total")
        stats_out["success"] = bool(solve_ok)
        stats_out["error_message"] = err_msg
        conv = []
        it_log = s.get("iterations") or {}
        if isinstance(it_log, dict) and it_log.get("obj"):
            obj_hist = it_log.get("obj") or []
            pr_hist = it_log.get("inf_pr") or []
            du_hist = it_log.get("inf_du") or []
            for k in range(len(obj_hist)):
                conv.append({
                    "iter": int(k),
                    "obj": float(obj_hist[k]),
                    "constr_viol": float(pr_hist[k]) if k < len(pr_hist) else None,
                    "dual_inf": float(du_hist[k]) if k < len(du_hist) else None,
                })
        stats_out["convergence_history"] = conv

    try:
        sol = opti.solve()
        print("  IPOPT converged!")

        X_sol = sol.value(X)
        U_sol_mat = sol.value(U)  # (3, n_seg)
        J_val = sol.value(J)

        # Repack U into list-of-arrays format for compatibility
        U_sol = [U_sol_mat[:, k:k+1] for k in range(n_seg)]

        u_mag = np.linalg.norm(U_sol_mat, axis=0)
        print(f"  Objective (control cost): {J_val:.6e}")
        print(f"  Max |u|: {u_mag.max():.6e} km/s²")
        print(f"  Mean |u|: {u_mag.mean():.6e} km/s²")
        print(f"  Final position error: {np.linalg.norm(X_sol[0:3, -1] - rf_target):.4e} km")

        _capture_stats(solve_ok=True)
        return X_sol, U_sol, seg_times, J_val

    except RuntimeError as e:
        print(f"  IPOPT failed: {e}")
        _capture_stats(solve_ok=False, err_msg=str(e))
        return None


# ============================================================================
# SECTION 7: FORWARD PROPAGATION OF IPOPT SOLUTION
# ============================================================================

def propagate_ipopt_solution(X_sol, U_sol, seg_times, ephem_cache):
    """
    Forward-propagate the IPOPT solution using full ephemeris dynamics.
    Interpolate control from IPOPT node values.
    """
    print("\n  Forward-propagating IPOPT solution...")

    all_t = []
    all_states = []
    nc = U_sol[0].shape[1]

    for k in range(len(seg_times) - 1):
        t_k = seg_times[k]
        t_kp1 = seg_times[k + 1]
        h = t_kp1 - t_k

        # Average control for this segment
        u_seg = np.mean(U_sol[k], axis=1)

        # Initial state from IPOPT
        x0_seg = X_sol[:, k]

        t_eval = np.linspace(t_k, t_kp1, 50)

        sol = solve_ivp(
            lambda t, y: dynamics_ephemeris(t, y, ephem_cache, with_control=False) + np.concatenate([[0,0,0], u_seg]),
            [t_k, t_kp1], x0_seg, method='DOP853',
            rtol=1e-12, atol=1e-14, t_eval=t_eval, max_step=30.0
        )

        if sol.success:
            all_t.extend(sol.t.tolist())
            all_states.append(sol.y.T)

    all_t = np.array(all_t)
    all_states = np.vstack(all_states)

    return all_t, all_states


# ============================================================================
# SECTION 8: COMPARISON PLOTS
# ============================================================================

def plot_results(times_sec, nasa_pos, nasa_vel,
                 sol_ballistic, sol_shooting, ipopt_result,
                 ephem_cache, output_dir, rf_target=None, vf_target=None):
    """Generate comparison figures."""
    print("\n" + "-" * 60)
    print("Generating figures...")

    # NASA reference
    t_nasa = times_sec
    r_nasa = nasa_pos  # (N, 3) in km

    # --- Figure 1: 3D Trajectory ---
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # NASA
    ax.plot(r_nasa[:, 0], r_nasa[:, 1], r_nasa[:, 2],
            color='#1f77b4', linewidth=2, label='NASA (OEM)', alpha=0.9)

    # Ballistic
    if sol_ballistic is not None:
        ax.plot(sol_ballistic.y[0], sol_ballistic.y[1], sol_ballistic.y[2],
                color='#d62728', linewidth=1.5, linestyle='--', label='Ballistic', alpha=0.8)

    # Shooting
    if sol_shooting is not None:
        sol_sh = sol_shooting[0]
        ax.plot(sol_sh.y[0], sol_sh.y[1], sol_sh.y[2],
                color='#2ca02c', linewidth=2, linestyle='-.', label='Shooting (min-energy)', alpha=0.9)

    # IPOPT
    if ipopt_result is not None:
        X_sol = ipopt_result[0]
        ax.plot(X_sol[0, :], X_sol[1, :], X_sol[2, :],
                color='#2ca02c', marker='^', markersize=6, linewidth=1.5, label='IPOPT nodes', alpha=0.9)

        # Forward propagation
        t_ip, states_ip = propagate_ipopt_solution(*ipopt_result[:3], ephem_cache)
        ax.plot(states_ip[:, 0], states_ip[:, 1], states_ip[:, 2],
                color='#2ca02c', linewidth=1.5, linestyle=':', label='IPOPT (propagated)', alpha=0.7)

    # Earth
    ax.scatter([0], [0], [0], c='cyan', s=200, marker='o', edgecolors='black',
               linewidth=2, label='Earth', zorder=10)

    # Moon positions along orbit (multiple epochs)
    t_moon_sample = np.linspace(times_sec[0], times_sec[-1], 5)
    moon_pos_sample = []
    for t in t_moon_sample:
        rm, _ = ephem_cache.get_positions(t)
        moon_pos_sample.append(rm)
    moon_pos_sample = np.array(moon_pos_sample)
    ax.scatter(moon_pos_sample[:, 0], moon_pos_sample[:, 1], moon_pos_sample[:, 2],
               c='gray', s=80, marker='o', edgecolors='black', linewidth=1,
               label='Moon (orbit)', zorder=10)

    # Moon orbit trace (faint dashed line)
    ax.plot(moon_pos_sample[:, 0], moon_pos_sample[:, 1], moon_pos_sample[:, 2],
            color='gray', linewidth=0.8, linestyle='--', alpha=0.3)

    ax.set_xlabel('X (km)', fontsize=12, color='black')
    ax.set_ylabel('Y (km)', fontsize=12, color='black')
    ax.set_zlabel('Z (km)', fontsize=12, color='black')
    ax.tick_params(colors='black')
    ax.set_title('Artemis II — Ephemeris-Driven Propagator\n(Earth-Centered J2000)',
                 fontsize=14, color='black')
    ax.legend(loc='upper left', fontsize=10, facecolor='white', edgecolor='black',
             labelcolor='black')

    fig.savefig(os.path.join(output_dir, 'ephem_3d_trajectory.png'),
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close(fig)
    print("  Saved ephem_3d_trajectory.png")

    # --- Figure 2: 2D Projections ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor='white')

    projections = [
        (0, 1, 'X (km)', 'Y (km)', 'XY Projection'),
        (0, 2, 'X (km)', 'Z (km)', 'XZ Projection'),
        (1, 2, 'Y (km)', 'Z (km)', 'YZ Projection'),
    ]

    for ax, (i, j, xl, yl, title) in zip(axes, projections):
        ax.plot(r_nasa[:, i], r_nasa[:, j], 'b-', lw=2, label='NASA', alpha=0.9)

        if sol_ballistic is not None:
            ax.plot(sol_ballistic.y[i], sol_ballistic.y[j],
                    'r--', lw=1.5, label='Ballistic', alpha=0.7)

        if sol_shooting is not None:
            sol_sh = sol_shooting[0]
            ax.plot(sol_sh.y[i], sol_sh.y[j],
                    'g-.', lw=2, label='Shooting', alpha=0.8)

        if ipopt_result is not None:
            X_sol = ipopt_result[0]
            ax.plot(X_sol[i, :], X_sol[j, :], 'm^-', ms=5, lw=1.5,
                    label='IPOPT', alpha=0.8)

        ax.scatter([0], [0], c='cyan', s=100, marker='o', edgecolors='blue',
                   linewidth=1.5, zorder=10)
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel(yl, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, facecolor='white', edgecolor='black', labelcolor='black')
        ax.grid(True, alpha=0.3, color='gray')

    fig.suptitle('Artemis II — 2D Projections (ECI, km)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ephem_2d_projections.png'),
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close(fig)
    print("  Saved ephem_2d_projections.png")

    # --- Figure 3: Position/Velocity Error vs NASA ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, facecolor='white')

    t_days = (t_nasa - t_nasa[0]) / 86400.0

    if sol_ballistic is not None:
        # Interpolate ballistic to NASA times
        bal_interp = interp1d(sol_ballistic.t, sol_ballistic.y[0:3], axis=1,
                              fill_value='extrapolate')
        bal_v_interp = interp1d(sol_ballistic.t, sol_ballistic.y[3:6], axis=1,
                                fill_value='extrapolate')
        bal_pos = bal_interp(t_nasa).T
        bal_vel = bal_v_interp(t_nasa).T
        err_bal_pos = np.linalg.norm(bal_pos - r_nasa, axis=1)
        err_bal_vel = np.linalg.norm(bal_vel - nasa_vel, axis=1)
        ax1.semilogy(t_days, err_bal_pos, 'r-', lw=1.5, label='Ballistic', alpha=0.8)
        ax2.semilogy(t_days, err_bal_vel, 'r-', lw=1.5, label='Ballistic', alpha=0.8)

    if sol_shooting is not None:
        sol_sh = sol_shooting[0]
        sh_interp = interp1d(sol_sh.t, sol_sh.y[0:3], axis=1, fill_value='extrapolate')
        sh_v_interp = interp1d(sol_sh.t, sol_sh.y[3:6], axis=1, fill_value='extrapolate')
        sh_pos = sh_interp(t_nasa).T
        sh_vel = sh_v_interp(t_nasa).T
        err_sh_pos = np.linalg.norm(sh_pos - r_nasa, axis=1)
        err_sh_vel = np.linalg.norm(sh_vel - nasa_vel, axis=1)
        ax1.semilogy(t_days, err_sh_pos, 'g-', lw=2, label='Shooting', alpha=0.9)
        ax2.semilogy(t_days, err_sh_vel, 'g-', lw=2, label='Shooting', alpha=0.9)

    if ipopt_result is not None:
        # Interpolate IPOPT forward-propagated solution
        t_ip, states_ip = propagate_ipopt_solution(*ipopt_result[:3], ephem_cache)
        ip_interp = interp1d(t_ip, states_ip[:, 0:3].T, axis=1, fill_value='extrapolate')
        ip_v_interp = interp1d(t_ip, states_ip[:, 3:6].T, axis=1, fill_value='extrapolate')
        ip_pos = ip_interp(t_nasa).T
        ip_vel = ip_v_interp(t_nasa).T
        err_ip_pos = np.linalg.norm(ip_pos - r_nasa, axis=1)
        err_ip_vel = np.linalg.norm(ip_vel - nasa_vel, axis=1)
        ax1.semilogy(t_days, err_ip_pos, 'm-', lw=2, label='IPOPT', alpha=0.9)
        ax2.semilogy(t_days, err_ip_vel, 'm-', lw=2, label='IPOPT', alpha=0.9)

    ax1.set_ylabel('Position Error (km)', fontsize=12)
    ax1.set_title('Trajectory Error vs NASA Ephemeris', fontsize=14)
    ax1.legend(fontsize=11, facecolor='white', edgecolor='black', labelcolor='black')
    ax1.grid(True, alpha=0.3, color='gray')

    ax2.set_xlabel('Time (days from segment start)', fontsize=12)
    ax2.set_ylabel('Velocity Error (km/s)', fontsize=12)
    ax2.legend(fontsize=11, facecolor='white', edgecolor='black', labelcolor='black')
    ax2.grid(True, alpha=0.3, color='gray')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ephem_error_comparison.png'),
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close(fig)
    print("  Saved ephem_error_comparison.png")

    # --- Figure 4: Summary statistics ---
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.axis('off')

    lines = []
    lines.append("ARTEMIS II — EPHEMERIS-DRIVEN PROPAGATOR")
    lines.append("=" * 55)
    lines.append("")
    lines.append(f"Dynamics: Earth + Moon + Sun (astropy builtin ephemeris)")
    lines.append(f"Frame: Earth-centered EME2000 (J2000), dimensional (km, s)")
    lines.append(f"Segment duration: {(t_nasa[-1]-t_nasa[0])/86400:.2f} days")
    lines.append("")

    if sol_ballistic is not None:
        lines.append("BALLISTIC (no control):")
        lines.append(f"  Max position error: {err_bal_pos.max():.4f} km")
        lines.append(f"  Mean position error: {err_bal_pos.mean():.4f} km")
        lines.append(f"  Max velocity error: {err_bal_vel.max():.6f} km/s")
        lines.append("")

    if sol_shooting is not None:
        lines.append("INDIRECT SHOOTING (min-energy):")
        lines.append(f"  TPBVP residual: {np.linalg.norm(sol_shooting[0].y[0:3,-1] - rf_target):.4e} km")
        lines.append(f"  Control cost: {sol_shooting[2]:.6e}")
        lines.append(f"  Max position error: {err_sh_pos.max():.4f} km")
        lines.append(f"  Mean position error: {err_sh_pos.mean():.4f} km")
        lines.append(f"  Max velocity error: {err_sh_vel.max():.6f} km/s")
        lines.append("")

    if ipopt_result is not None:
        lines.append("IPOPT COLLOCATION:")
        lines.append(f"  Objective: {ipopt_result[3]:.6e}")
        lines.append(f"  Max position error: {err_ip_pos.max():.4f} km")
        lines.append(f"  Mean position error: {err_ip_pos.mean():.4f} km")
        lines.append(f"  Max velocity error: {err_ip_vel.max():.6f} km/s")
        lines.append("")

    lines.append("COMPARISON vs CR3BP:")
    lines.append("  CR3BP mean pos error ~ 100,000+ km (nondim ~0.26)")
    lines.append("  Ephemeris-driven errors should be MUCH smaller")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.9))

    fig.savefig(os.path.join(output_dir, 'ephem_summary_stats.png'),
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close(fig)
    print("  Saved ephem_summary_stats.png")


# ============================================================================
# ANIMATION
# ============================================================================

def create_animation(times_sec, nasa_pos, sol_ballistic, ipopt_result,
                      ephem_cache, output_dir, fps=15, duration=10):
    """
    Create animation of Artemis II trajectory with dark theme and moving Moon.

    Args:
        times_sec: Array of times in seconds (from segment start)
        nasa_pos: NASA OEM positions, shape (N, 3)
        sol_ballistic: Ballistic solution (scipy ODE result or None)
        ipopt_result: IPOPT result tuple (X_sol, U_sol, seg_times, J_val)
        ephem_cache: EphemerisCache instance
        output_dir: Output directory for animation
        fps: Frames per second (default 15)
        duration: Duration in seconds (default 10)
    """
    print("\n" + "-" * 60)
    print("Creating animation...")

    n_frames = int(fps * duration)
    t_min, t_max = times_sec[0], times_sec[-1]

    # Pre-compute Moon positions for all times
    t_moon_grid = np.linspace(t_min, t_max, n_frames * 2)
    moon_positions = []
    for t in t_moon_grid:
        rm, _ = ephem_cache.get_positions(t)
        moon_positions.append(rm)
    moon_positions = np.array(moon_positions)

    # Interpolate for animation frames
    moon_interp = interp1d(t_moon_grid, moon_positions, axis=0, kind='cubic')

    # Determine phase labels and closest Moon approach distance
    day_offsets = times_sec / 86400.0
    r_from_earth = np.linalg.norm(nasa_pos, axis=1)
    i_far = np.argmax(r_from_earth)
    # Compute closest Moon approach using interpolated Moon positions at NASA times
    moon_at_nasa = moon_interp(times_sec)
    r_min_to_moon = np.linalg.norm(nasa_pos - moon_at_nasa, axis=1).min()

    fig = plt.figure(figsize=(12, 9), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # Set text color to black for visibility on white background
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.zaxis.label.set_color('black')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')

    title_text = ax.text2D(0.5, 0.95, '', transform=ax.transAxes,
                           fontsize=13, ha='center', color='black', weight='bold')

    # Initialize line collections for building trajectories
    lines_data = {
        'nasa': {'color': '#1f77b4', 'segments': []},
        'ballistic': {'color': '#d62728', 'segments': []},
        'ipopt': {'color': '#2ca02c', 'segments': []}
    }

    # Initialize scatter plots
    earth_point = ax.scatter([0], [0], [0], c='cyan', s=150, marker='o',
                            edgecolors='black', linewidth=1.5, label='Earth', zorder=10)
    moon_point = ax.scatter([0], [0], [0], c='gray', s=80, marker='o',
                           edgecolors='black', linewidth=1, zorder=10)
    nasa_dot = ax.scatter([0], [0], [0], c='#1f77b4', s=60, marker='o', zorder=11)

    # Track indices for building trajectories
    n_nasa = len(nasa_pos)

    # Ballistic data
    if sol_ballistic is not None:
        bal_times = sol_ballistic.t
        bal_pos = sol_ballistic.y[0:3].T
        bal_interp_func = interp1d(bal_times, bal_pos, axis=0, fill_value='extrapolate')

    # IPOPT data
    if ipopt_result is not None:
        X_sol = ipopt_result[0]
        seg_times = ipopt_result[2]
        ipopt_pos = X_sol[0:3].T  # (n_nodes, 3)
        ipopt_times = seg_times

    # Moon orbit trace (faint dashed line)
    moon_trace, = ax.plot([], [], [], 'gray', linewidth=0.5, linestyle='--',
                          alpha=0.3, label='Moon orbit')

    ax.set_xlabel('X (km)', fontsize=11, color='black')
    ax.set_ylabel('Y (km)', fontsize=11, color='black')
    ax.set_zlabel('Z (km)', fontsize=11, color='black')
    ax.legend(loc='upper left', fontsize=9, facecolor='white', edgecolor='black',
             labelcolor='black')

    # Set axis limits
    all_pos = nasa_pos.copy()
    if sol_ballistic is not None:
        all_pos = np.vstack([all_pos, bal_pos])
    if ipopt_result is not None:
        all_pos = np.vstack([all_pos, ipopt_pos])

    lim = 1.1 * np.max(np.abs(all_pos))
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    # Store line artists for updating
    line_nasa = ax.plot([], [], [], color='#1f77b4', linewidth=2.5, alpha=0.9,
                       label='NASA (OEM)')[0]
    line_ballistic = ax.plot([], [], [], color='#d62728', linewidth=1.5, linestyle='--',
                            alpha=0.8, label='Ballistic')[0]
    line_ipopt = ax.plot([], [], [], color='#2ca02c', linewidth=2, alpha=0.9,
                        label='IPOPT')[0]

    def update_frame(frame_idx):
        # Map frame to time
        t_current = t_min + (frame_idx / (n_frames - 1)) * (t_max - t_min)
        day_current = t_current / 86400.0

        # Determine phase
        if day_current < 3.0:
            phase = "Translunar Coast"
        elif day_current < 4.5:
            phase = f"Lunar Flyby (r={r_min_to_moon:.0f}k km)"
        else:
            phase = "Free-Return Coast"

        title_text.set_text(f"Artemis II Ephemeris — Day {day_current:.2f} | {phase}")

        # Find indices up to current time
        mask_nasa = times_sec <= t_current
        n_current = np.sum(mask_nasa)

        # Update NASA trajectory
        if n_current > 0:
            line_nasa.set_data(nasa_pos[:n_current, 0], nasa_pos[:n_current, 1])
            line_nasa.set_3d_properties(nasa_pos[:n_current, 2])
            nasa_dot._offsets3d = (nasa_pos[n_current-1:n_current, 0],
                                  nasa_pos[n_current-1:n_current, 1],
                                  nasa_pos[n_current-1:n_current, 2])

        # Update ballistic trajectory
        if sol_ballistic is not None:
            mask_bal = bal_times <= t_current
            if np.sum(mask_bal) > 0:
                bal_current = bal_pos[mask_bal]
                line_ballistic.set_data(bal_current[:, 0], bal_current[:, 1])
                line_ballistic.set_3d_properties(bal_current[:, 2])

        # Update IPOPT trajectory
        if ipopt_result is not None:
            mask_ipopt = ipopt_times <= t_current
            if np.sum(mask_ipopt) > 0:
                ipopt_current = ipopt_pos[mask_ipopt]
                line_ipopt.set_data(ipopt_current[:, 0], ipopt_current[:, 1])
                line_ipopt.set_3d_properties(ipopt_current[:, 2])

        # Update Moon position
        rm_current = moon_interp(t_current)
        moon_point._offsets3d = ([rm_current[0]], [rm_current[1]], [rm_current[2]])

        # Update Moon orbit trace
        t_trace = np.linspace(max(t_min, t_current - 86400), t_current, 100)
        moon_trace_pts = moon_interp(t_trace)
        moon_trace.set_data(moon_trace_pts[:, 0], moon_trace_pts[:, 1])
        moon_trace.set_3d_properties(moon_trace_pts[:, 2])

        # Slowly rotating view
        azim = 45 + 360 * (frame_idx / n_frames)
        elev = 20
        ax.view_init(elev=elev, azim=azim)

        return [title_text, line_nasa, line_ballistic, line_ipopt, moon_point,
                nasa_dot, moon_trace]

    anim = FuncAnimation(fig, update_frame, frames=n_frames, interval=1000/fps,
                        blit=True, repeat=False)

    output_path = os.path.join(output_dir, 'ephem_animation.gif')
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)

    print(f"  Saved ephem_animation.gif ({n_frames} frames at {fps} fps)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # ../../ from Ephem_Full
    uploads_dir = '/sessions/nifty-pensive-volta/mnt/uploads'

    # Find OEM file: look in script dir, project root (where the .asc is checked in),
    # then a legacy sandbox uploads path for backward compat.
    oem_file = None
    for d in [script_dir, project_root, uploads_dir]:
        for f in os.listdir(d) if os.path.isdir(d) else []:
            if 'Artemis' in f and f.endswith('.asc'):
                oem_file = os.path.join(d, f)
                break
        if oem_file:
            break

    if oem_file is None:
        print("ERROR: OEM file not found!")
        sys.exit(1)

    print(f"\nOEM file: {oem_file}")

    # Parse OEM
    times_utc, positions, velocities = parse_oem(oem_file)

    # --- Segment selection ---
    # Full free-return: day 1.0 to 8.5 (after TLI burn, through lunar flyby, back)
    t0_utc = times_utc[0]
    day_offsets = np.array([(t - t0_utc).total_seconds() / 86400.0 for t in times_utc])

    seg_start_day = 1.0
    seg_end_day = 8.5
    mask = (day_offsets >= seg_start_day) & (day_offsets <= seg_end_day)

    seg_times_utc = [t for t, m in zip(times_utc, mask) if m]
    seg_pos = positions[mask]
    seg_vel = velocities[mask]

    print(f"\nSegment: day {seg_start_day} to {seg_end_day} (full free-return)")
    print(f"  {len(seg_times_utc)} data points")
    print(f"  Start: {seg_times_utc[0]}")
    print(f"  End: {seg_times_utc[-1]}")

    # Time in seconds from segment start
    t_seg_start = seg_times_utc[0]
    t_seg_end = seg_times_utc[-1]
    times_sec = np.array([(t - t_seg_start).total_seconds() for t in seg_times_utc])

    r0 = seg_pos[0]
    v0 = seg_vel[0]
    rf_target = seg_pos[-1]
    vf_target = seg_vel[-1]

    t_span_sec = [0.0, times_sec[-1]]

    print(f"\n  r0 = [{r0[0]:.2f}, {r0[1]:.2f}, {r0[2]:.2f}] km")
    print(f"  v0 = [{v0[0]:.6f}, {v0[1]:.6f}, {v0[2]:.6f}] km/s")
    print(f"  rf = [{rf_target[0]:.2f}, {rf_target[1]:.2f}, {rf_target[2]:.2f}] km")
    print(f"  Duration: {times_sec[-1]/3600:.2f} hours = {times_sec[-1]/86400:.2f} days")

    # Earth-Moon distance at closest approach
    r_earth = np.linalg.norm(seg_pos, axis=1)
    i_far = np.argmax(r_earth)
    print(f"\n  Max Earth distance: {r_earth[i_far]:.0f} km at day {day_offsets[mask][i_far]:.2f}")
    print(f"  (near lunar flyby)")

    # Build ephemeris cache — denser grid for 7.5 days
    margin = 3600.0  # 1 hour margin
    ephem_cache = EphemerisCache(
        t_seg_start - timedelta(seconds=margin),
        t_seg_end + timedelta(seconds=margin),
        n_points=8000  # ~80s resolution over 7.5 days
    )

    # Adjust times_sec since cache starts at t_seg_start - margin
    t_offset = margin
    times_sec_cache = times_sec + t_offset
    t_span_cache = [t_offset, times_sec[-1] + t_offset]

    # 1. Ballistic propagation
    sol_ballistic = propagate_ballistic(r0, v0, t_span_cache, ephem_cache, n_eval=2000)

    # 2. Indirect shooting — skip for long arcs (too slow, each eval = 7.5-day integration)
    sol_shooting = None
    print("\n" + "-" * 60)
    print("Skipping indirect shooting (too expensive for 7.5-day arc)")
    print("  Using NASA OEM as warm-start for IPOPT instead")

    # 3. IPOPT collocation — warm-started from NASA OEM data directly
    ipopt_result = solve_ipopt_collocation(r0, v0, rf_target, vf_target,
                                            t_span_cache, ephem_cache,
                                            n_seg=100, sol_shooting=sol_shooting,
                                            nasa_warmstart=(times_sec_cache, seg_pos, seg_vel))

    # 4. Plot results
    plot_results(times_sec_cache, seg_pos, seg_vel,
                 sol_ballistic, sol_shooting, ipopt_result,
                 ephem_cache, script_dir,
                 rf_target=rf_target, vf_target=vf_target)

    # 5. Animation
    create_animation(times_sec_cache, seg_pos, sol_ballistic, ipopt_result,
                     ephem_cache, script_dir)

    print("\n" + "=" * 80)
    print("DONE — All figures and animation saved to:", script_dir)
    print("=" * 80)


if __name__ == '__main__':
    main()
