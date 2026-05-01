#!/usr/bin/env python3
"""
artemis2_full_mission.py — Full Artemis II Mission: TLI Burn → Moon → Return

Covers the ENTIRE OEM arc (day 0 to 8.9):
  - Post-ICPS separation coast
  - TLI burn (day ~0.88-0.92, detected from OEM velocity jumps)
  - Translunar coast
  - Lunar flyby (day ~4.5-5.5)
  - Free-return coast back to Earth

Three comparisons:
  1. Ballistic propagation (no control) — shows what happens without TLI
  2. IPOPT direct multiple-shooting — finds the optimal control that
     reproduces NASA's trajectory; control concentrates at the TLI burn
  3. NASA OEM (ground truth)

Plus: 3D animation of the full mission with moving Moon.

Author: Zhuorui, AAE 568 Spring 2026
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time
import astropy.units as u
solar_system_ephemeris.set('builtin')

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

# ============================================================================
# CONSTANTS
# ============================================================================
MU_EARTH = 398600.4418
MU_MOON  = 4902.800066
MU_SUN   = 132712440041.93938

print("\n" + "=" * 80)
print("ARTEMIS II — FULL MISSION (TLI → Moon → Return)")
print("=" * 80)


# ============================================================================
# OEM PARSER
# ============================================================================
def parse_oem(filename):
    times_utc, positions, velocities = [], [], []
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
    return times_utc, np.array(positions), np.array(velocities)


# ============================================================================
# EPHEMERIS CACHE (vectorized astropy)
# ============================================================================
class EphemerisCache:
    def __init__(self, t_start_utc, t_end_utc, n_points=5000):
        print("  Building ephemeris cache...", flush=True)
        dt_total = (t_end_utc - t_start_utc).total_seconds()
        self.t_start = t_start_utc
        self.t_grid = np.linspace(0, dt_total, n_points)

        t_arr = Time([t_start_utc + timedelta(seconds=float(dt))
                      for dt in self.t_grid], scale='utc')
        earth_pos = get_body_barycentric_posvel('earth', t_arr)[0].xyz.to(u.km).value
        moon_pos  = get_body_barycentric_posvel('moon', t_arr)[0].xyz.to(u.km).value
        sun_pos   = get_body_barycentric_posvel('sun', t_arr)[0].xyz.to(u.km).value

        r_moon_arr = (moon_pos - earth_pos).T
        r_sun_arr  = (sun_pos  - earth_pos).T

        self.moon_interp = interp1d(self.t_grid, r_moon_arr, axis=0, kind='cubic',
                                     fill_value='extrapolate')
        self.sun_interp  = interp1d(self.t_grid, r_sun_arr, axis=0, kind='cubic',
                                     fill_value='extrapolate')

        # Store for animation
        self.moon_positions = r_moon_arr
        self.moon_times = self.t_grid

        print(f"    {n_points} points over {dt_total/86400:.2f} days")
        print(f"    Moon: {np.linalg.norm(r_moon_arr, axis=1).min():.0f}–"
              f"{np.linalg.norm(r_moon_arr, axis=1).max():.0f} km")

    def get_positions(self, t_sec):
        return self.moon_interp(t_sec), self.sun_interp(t_sec)

    def get_moon_at(self, t_sec):
        return self.moon_interp(t_sec)


# ============================================================================
# DYNAMICS
# ============================================================================
def dynamics_ephemeris(t_sec, state, ephem_cache):
    r = state[0:3]
    v = state[3:6]
    r_norm = np.linalg.norm(r)

    a_earth = -MU_EARTH * r / r_norm**3

    r_moon, r_sun = ephem_cache.get_positions(t_sec)
    d_moon = r_moon - r
    a_moon = MU_MOON * (d_moon / np.linalg.norm(d_moon)**3 - r_moon / np.linalg.norm(r_moon)**3)
    d_sun = r_sun - r
    a_sun = MU_SUN * (d_sun / np.linalg.norm(d_sun)**3 - r_sun / np.linalg.norm(r_sun)**3)

    return np.concatenate([v, a_earth + a_moon + a_sun])


# ============================================================================
# BURN DETECTION
# ============================================================================
def detect_burns(times_utc, velocities, t0_utc, threshold_accel=1e-3):
    """Detect engine burns from velocity discontinuities in OEM."""
    dt = np.array([(times_utc[i+1] - times_utc[i]).total_seconds()
                   for i in range(len(times_utc)-1)])
    dv = np.diff(velocities, axis=0)
    dv_mag = np.linalg.norm(dv, axis=1)
    accel = dv_mag / np.maximum(dt, 1e-6)

    burn_mask = accel > threshold_accel
    burns = []

    if np.any(burn_mask):
        # Group consecutive burn indices
        indices = np.where(burn_mask)[0]
        groups = np.split(indices, np.where(np.diff(indices) > 5)[0] + 1)

        for group in groups:
            if len(group) == 0:
                continue
            i_start = group[0]
            i_end = group[-1] + 1
            day_start = (times_utc[i_start] - t0_utc).total_seconds() / 86400
            day_end = (times_utc[i_end] - t0_utc).total_seconds() / 86400

            # Total delta-v
            total_dv = np.sum(dv_mag[group])
            max_accel = accel[group].max()

            # Name burns by epoch: early = TLI, late = entry correction
            if day_start < 2.0:
                burn_name = "TLI Burn"
            else:
                burn_name = "Entry Correction Burn"

            burns.append({
                'name': burn_name,
                'day_start': day_start,
                'day_end': day_end,
                'total_dv': total_dv,
                'max_accel': max_accel,
                'i_start': i_start,
                'i_end': i_end,
            })

            print(f"    Burn: day {day_start:.3f}–{day_end:.3f}, "
                  f"ΔV ≈ {total_dv:.2f} km/s, peak accel = {max_accel:.4e} km/s²")

    return burns


# ============================================================================
# BALLISTIC PROPAGATION
# ============================================================================
def propagate_ballistic(r0, v0, t_span_sec, ephem_cache, n_eval=3000):
    print("\n  Ballistic propagation...", flush=True)
    state0 = np.concatenate([r0, v0])
    t_eval = np.linspace(t_span_sec[0], t_span_sec[1], n_eval)

    sol = solve_ivp(
        lambda t, y: dynamics_ephemeris(t, y, ephem_cache),
        t_span_sec, state0, method='DOP853',
        rtol=1e-12, atol=1e-14, t_eval=t_eval, max_step=300.0
    )
    if sol.success:
        rf = sol.y[0:3, -1]
        print(f"    Final pos: [{rf[0]:.0f}, {rf[1]:.0f}, {rf[2]:.0f}] km")
    else:
        print(f"    WARNING: {sol.message}")
    return sol


# ============================================================================
# IPOPT DIRECT MULTIPLE-SHOOTING
# ============================================================================
def solve_ipopt(r0, v0, rf_target, vf_target, t_span_sec, ephem_cache,
                n_seg=120, nasa_data=None, burns=None, stats_out=None):
    if not HAS_CASADI:
        print("\n  CasADi not available")
        return None

    print(f"\n  IPOPT multiple-shooting (~{n_seg} segments)...", flush=True)

    t0, tf = t_span_sec
    total_dur = tf - t0

    # Non-uniform segment distribution: dense during burns, coarse during coast
    # Coast segments: ~2 hour spacing
    # Burn segments: ~2 minute spacing (to resolve thrust profile)
    seg_time_list = [t0]
    t_curr = t0

    coast_dt = total_dur / n_seg  # default coast spacing
    burn_dt = 120.0  # 2-minute spacing during burns

    while t_curr < tf - 1:
        curr_day = (t_curr - t0) / 86400.0
        in_burn = False
        if burns:
            for burn in burns:
                if burn['day_start'] - 0.05 <= curr_day <= burn['day_end'] + 0.05:
                    in_burn = True
                    break

        if in_burn:
            t_curr += burn_dt
        else:
            t_curr += coast_dt

        t_curr = min(t_curr, tf)
        seg_time_list.append(t_curr)

    seg_times = np.array(seg_time_list)
    n_seg = len(seg_times) - 1
    seg_days = (seg_times - t0) / 86400.0
    print(f"    Actual segments: {n_seg} ({n_seg - 120 + 120} total, dense during burns)")

    ns, nd = 6, 3
    n_rk4 = 4

    # CasADi dynamics
    x_sym = ca.MX.sym('x', ns)
    u_sym = ca.MX.sym('u', nd)
    rm_sym = ca.MX.sym('rm', 3)
    rs_sym = ca.MX.sym('rs', 3)

    r_sc = x_sym[0:3]; v_sc = x_sym[3:6]
    r_norm = ca.norm_2(r_sc)
    a_earth = -MU_EARTH * r_sc / r_norm**3

    d_moon = rm_sym - r_sc
    a_moon = MU_MOON * (d_moon / ca.norm_2(d_moon)**3 - rm_sym / ca.norm_2(rm_sym)**3)
    d_sun = rs_sym - r_sc
    a_sun = MU_SUN * (d_sun / ca.norm_2(d_sun)**3 - rs_sym / ca.norm_2(rs_sym)**3)

    xdot = ca.vertcat(v_sc, a_earth + a_moon + a_sun + u_sym)
    f_dyn = ca.Function('f_dyn', [x_sym, u_sym, rm_sym, rs_sym], [xdot])

    # Control bounds — vary by segment
    # During burn phases: allow large thrust; during coast: small bound
    u_bounds = np.ones(n_seg) * 1e-4  # default: 0.1 mm/s² coast
    if burns:
        for burn in burns:
            for k in range(n_seg):
                seg_day = 0.5 * (seg_days[k] + seg_days[k+1])
                # Use day offset relative to OEM start (burn days are relative to OEM start)
                if burn['day_start'] - 0.1 <= seg_day <= burn['day_end'] + 0.1:
                    u_bounds[k] = burn['max_accel'] * 1.5  # 1.5× headroom

    print(f"    Coast control bound: {u_bounds.min():.4e} km/s²")
    print(f"    Burn control bound:  {u_bounds.max():.4e} km/s²")

    # Build NLP
    opti = ca.Opti()
    X = opti.variable(ns, n_seg + 1)
    U = opti.variable(nd, n_seg)

    # BCs
    opti.subject_to(X[:, 0] == np.concatenate([r0, v0]))
    opti.subject_to(X[:, -1] == np.concatenate([rf_target, vf_target]))

    # Waypoint constraints — pin full state every N segments AND at burn boundaries
    # This forces IPOPT through the NASA trajectory, so it must use thrust at burns
    if nasa_data is not None:
        nasa_t, nasa_pos, nasa_vel = nasa_data
        pos_interp = interp1d(nasa_t, nasa_pos, axis=0, fill_value='extrapolate')
        vel_interp = interp1d(nasa_t, nasa_vel, axis=0, fill_value='extrapolate')

        # Waypoints: position-only during burns (every 5th burn node),
        # full state every ~8 hours during coast
        pin_interval_coast = 8.0 * 3600  # 8 hours
        last_pin_time = t0
        n_waypoints = 0
        burn_pin_every = 5  # every 5th burn node (~10 min)
        burn_node_count = 0

        for i in range(1, n_seg):  # skip 0 and n_seg (already BCs)
            seg_day_i = seg_days[i]
            dt_since_pin = seg_times[i] - last_pin_time

            in_burn = False
            if burns:
                for burn in burns:
                    if burn['day_start'] - 0.02 <= seg_day_i <= burn['day_end'] + 0.02:
                        in_burn = True
                        break

            if in_burn:
                burn_node_count += 1
                if burn_node_count % burn_pin_every == 0:
                    p = pos_interp(seg_times[i])
                    opti.subject_to(X[0:3, i] == p)  # position only
                    n_waypoints += 1
                    last_pin_time = seg_times[i]
            else:
                burn_node_count = 0
                if dt_since_pin >= pin_interval_coast:
                    p = pos_interp(seg_times[i])
                    v = vel_interp(seg_times[i])
                    opti.subject_to(X[0:3, i] == p)  # position
                    opti.subject_to(X[3:6, i] == v)  # velocity
                    n_waypoints += 1
                    last_pin_time = seg_times[i]

        print(f"    Pinned {n_waypoints} waypoint states from NASA OEM")

    # Control bounds per segment
    for k in range(n_seg):
        for d in range(nd):
            opti.subject_to(opti.bounded(-u_bounds[k], U[d, k], u_bounds[k]))

    # Objective
    J = 0
    for k in range(n_seg):
        h = seg_times[k+1] - seg_times[k]
        J += ca.dot(U[:, k], U[:, k]) * h
    opti.minimize(J)

    # RK4 defect constraints
    print(f"    Building RK4 constraints ({n_seg} segments)...", flush=True)
    for k in range(n_seg):
        t_k = seg_times[k]
        h_seg = seg_times[k+1] - seg_times[k]

        # Fewer RK4 sub-steps for short burn segments (already ~2 min)
        if h_seg < 300:  # < 5 min
            n_rk = 2
        else:
            n_rk = n_rk4

        h_rk = h_seg / n_rk
        xk = X[:, k]; uk = U[:, k]

        x_curr = xk
        for s in range(n_rk):
            t_s = t_k + s * h_rk
            rm1, rs1 = ephem_cache.get_positions(t_s)
            rm2, rs2 = ephem_cache.get_positions(t_s + 0.5 * h_rk)
            rm3, rs3 = ephem_cache.get_positions(t_s + h_rk)

            k1 = f_dyn(x_curr, uk, rm1, rs1)
            k2 = f_dyn(x_curr + 0.5*h_rk*k1, uk, rm2, rs2)
            k3 = f_dyn(x_curr + 0.5*h_rk*k2, uk, rm2, rs2)
            k4 = f_dyn(x_curr + h_rk*k3, uk, rm3, rs3)
            x_curr = x_curr + (h_rk/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        opti.subject_to(X[:, k+1] == x_curr)

    # Warm-start from NASA
    if nasa_data is not None:
        print("    Warm-starting from NASA OEM...", flush=True)
        nasa_t, nasa_pos, nasa_vel = nasa_data
        pos_interp = interp1d(nasa_t, nasa_pos, axis=0, fill_value='extrapolate')
        vel_interp = interp1d(nasa_t, nasa_vel, axis=0, fill_value='extrapolate')

        for i in range(n_seg + 1):
            p = pos_interp(seg_times[i])
            v = vel_interp(seg_times[i])
            opti.set_initial(X[:, i], np.concatenate([p, v]))
        opti.set_initial(U, 0)

    opts = {
        'ipopt.print_level': 3,
        'ipopt.max_iter': 5000,
        'ipopt.tol': 1e-6,
        'ipopt.acceptable_tol': 1e-4,
        'print_time': False,
        'ipopt.linear_solver': 'mumps',
    }
    opti.solver('ipopt', opts)

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
        X_sol = sol.value(X)
        U_sol = sol.value(U)
        J_val = sol.value(J)

        u_mag = np.linalg.norm(U_sol, axis=0)
        print(f"    IPOPT converged!")
        print(f"    Objective: {J_val:.6e}")
        print(f"    Max |u|: {u_mag.max():.6e} km/s²")
        print(f"    Coast max |u|: {u_mag[u_mag < 1e-3].max():.6e} km/s² (non-burn segments)")

        _capture_stats(solve_ok=True)
        return X_sol, U_sol, seg_times, J_val
    except RuntimeError as e:
        print(f"    IPOPT failed: {e}")
        _capture_stats(solve_ok=False, err_msg=str(e))
        return None


# ============================================================================
# FORWARD PROPAGATION OF IPOPT SOLUTION
# ============================================================================
def propagate_ipopt(X_sol, U_sol, seg_times, ephem_cache):
    all_t, all_states = [], []
    n_seg = len(seg_times) - 1

    for k in range(n_seg):
        t_k, t_kp1 = seg_times[k], seg_times[k+1]
        u_seg = U_sol[:, k]
        x0 = X_sol[:, k]
        t_eval = np.linspace(t_k, t_kp1, 30)

        sol = solve_ivp(
            lambda t, y: dynamics_ephemeris(t, y, ephem_cache) + np.concatenate([[0,0,0], u_seg]),
            [t_k, t_kp1], x0, method='DOP853',
            rtol=1e-12, atol=1e-14, t_eval=t_eval, max_step=60.0
        )
        if sol.success:
            all_t.extend(sol.t.tolist())
            all_states.append(sol.y.T)

    return np.array(all_t), np.vstack(all_states)


# ============================================================================
# STATIC PLOTS
# ============================================================================
def plot_static(times_sec, nasa_pos, nasa_vel, sol_ballistic, ipopt_result,
                ephem_cache, burns, output_dir, day_offsets_seg):
    print("\n  Generating static figures...", flush=True)

    t_nasa = times_sec
    r_nasa = nasa_pos
    t_days = (t_nasa - t_nasa[0]) / 86400.0

    # --- Figure 1: 3D trajectory ---
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(r_nasa[:, 0], r_nasa[:, 1], r_nasa[:, 2],
            'b-', lw=2.5, label='NASA (OEM)', alpha=0.9)

    if sol_ballistic is not None:
        ax.plot(sol_ballistic.y[0], sol_ballistic.y[1], sol_ballistic.y[2],
                'r--', lw=1.2, label='Ballistic (no TLI)', alpha=0.6)

    if ipopt_result is not None:
        X_sol = ipopt_result[0]
        ip_seg_times = ipopt_result[2]
        ip_days = (ip_seg_times - ip_seg_times[0]) / 86400.0
        ip_pos = X_sol[0:3, :].T  # (n+1, 3)

        # Color segments by burn/coast phase
        segments = [[ip_pos[j], ip_pos[j+1]] for j in range(len(ip_pos) - 1)]
        seg_colors = []
        seg_widths = []
        for j in range(len(ip_pos) - 1):
            mid_day = 0.5 * (ip_days[j] + ip_days[j+1])
            in_burn = False
            if burns:
                for burn in burns:
                    if burn['day_start'] - 0.02 <= mid_day <= burn['day_end'] + 0.02:
                        in_burn = True
                        break
            if in_burn:
                seg_colors.append('#ff7f0e')  # orange for burns
                seg_widths.append(3.5)
            else:
                seg_colors.append('#2ca02c')  # green for coast
                seg_widths.append(2.0)

        lc = Line3DCollection(segments, colors=seg_colors, linewidths=seg_widths, alpha=0.85)
        ax.add_collection3d(lc)
        # Dummy lines for legend
        ax.plot([], [], [], '-', color='#2ca02c', lw=2, label='IPOPT (coast)')
        ax.plot([], [], [], '-', color='#ff7f0e', lw=3.5, label='IPOPT (burn)')

    # Earth
    ax.scatter([0], [0], [0], c='cyan', s=250, marker='o', edgecolors='blue',
               lw=2, label='Earth', zorder=10)

    # Moon at several epochs
    for day_frac in [0, 2, 4.5, 6, 8]:
        t_cache = t_nasa[0] + day_frac * 86400
        try:
            rm = ephem_cache.get_moon_at(t_cache)
            label = 'Moon orbit' if day_frac == 0 else None
            ax.scatter([rm[0]], [rm[1]], [rm[2]], c='gray', s=50, alpha=0.5,
                       marker='o', label=label)
        except:
            pass

    # Moon at flyby
    r_earth_dist = np.linalg.norm(r_nasa, axis=1)
    i_flyby = np.argmax(r_earth_dist)
    t_flyby = t_nasa[i_flyby]
    rm_flyby = ephem_cache.get_moon_at(t_flyby)
    ax.scatter([rm_flyby[0]], [rm_flyby[1]], [rm_flyby[2]], c='silver', s=200,
               marker='o', edgecolors='black', lw=1.5, label='Moon (flyby)', zorder=10)

    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_zlabel('Z (km)', fontsize=12)
    ax.set_title('Artemis II — Full Mission\n(Earth-Centered J2000, Ephemeris-Driven)', fontsize=14)
    ax.legend(loc='upper left', fontsize=9, facecolor='white', edgecolor='black', labelcolor='black')

    fig.savefig(os.path.join(output_dir, 'full_3d_trajectory.png'), dpi=150, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close(fig)
    print("    Saved full_3d_trajectory.png")

    # --- Figure 2: Control profile ---
    if ipopt_result is not None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, facecolor='white')
        U_sol = ipopt_result[1]
        seg_times_arr = ipopt_result[2]
        seg_days_arr = (seg_times_arr[:-1] - seg_times_arr[0]) / 86400.0
        u_mag = np.linalg.norm(U_sol, axis=0)

        # Control magnitude
        axes[0].semilogy(seg_days_arr, u_mag, 'g-', lw=1.5)
        axes[0].fill_between(seg_days_arr, u_mag, alpha=0.3, color='green')
        axes[0].set_ylabel('|u| (km/s²)', fontsize=12)
        axes[0].set_title('IPOPT Control Profile — Reconstructed Thrust History', fontsize=14)
        axes[0].grid(True, alpha=0.3, color='gray')

        # Mark burns
        if burns:
            for burn in burns:
                axes[0].axvspan(burn['day_start'], burn['day_end'],
                                alpha=0.2, color='orange', label='Detected burn')
                axes[1].axvspan(burn['day_start'], burn['day_end'],
                                alpha=0.2, color='orange')

        axes[0].legend(fontsize=10, facecolor='white', edgecolor='black', labelcolor='black')

        # Control components
        axes[1].plot(seg_days_arr, U_sol[0, :], 'r-', lw=1, label='$u_x$', alpha=0.8)
        axes[1].plot(seg_days_arr, U_sol[1, :], 'g-', lw=1, label='$u_y$', alpha=0.8)
        axes[1].plot(seg_days_arr, U_sol[2, :], 'b-', lw=1, label='$u_z$', alpha=0.8)
        axes[1].set_xlabel('Time (days from mission start)', fontsize=12)
        axes[1].set_ylabel('Control (km/s²)', fontsize=12)
        axes[1].legend(fontsize=10, facecolor='white', edgecolor='black', labelcolor='black')
        axes[1].grid(True, alpha=0.3, color='gray')

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'full_control_profile.png'), dpi=150, bbox_inches='tight', facecolor='white', edgecolor='white')
        plt.close(fig)
        print("    Saved full_control_profile.png")

    # --- Figure 3: Error comparison ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, facecolor='white')

    if sol_ballistic is not None:
        bal_interp = interp1d(sol_ballistic.t, sol_ballistic.y[0:3], axis=1, fill_value='extrapolate')
        bal_pos = bal_interp(t_nasa).T
        err_bal = np.linalg.norm(bal_pos - r_nasa, axis=1)
        ax1.semilogy(t_days, err_bal, 'r-', lw=1.5, label='Ballistic', alpha=0.8)

    if ipopt_result is not None:
        t_ip, states_ip = propagate_ipopt(*ipopt_result[:3], ephem_cache)
        ip_interp = interp1d(t_ip, states_ip[:, 0:3].T, axis=1, fill_value='extrapolate')
        ip_v_interp = interp1d(t_ip, states_ip[:, 3:6].T, axis=1, fill_value='extrapolate')
        ip_pos = ip_interp(t_nasa).T
        ip_vel = ip_v_interp(t_nasa).T
        err_ip = np.linalg.norm(ip_pos - r_nasa, axis=1)
        err_ip_v = np.linalg.norm(ip_vel - nasa_vel, axis=1)
        ax1.semilogy(t_days, err_ip, 'm-', lw=2, label='IPOPT', alpha=0.9)
        ax2.semilogy(t_days, err_ip_v, 'm-', lw=2, label='IPOPT', alpha=0.9)

    # Mark burns
    if burns:
        for burn in burns:
            ax1.axvspan(burn['day_start'], burn['day_end'], alpha=0.15, color='orange')
            ax2.axvspan(burn['day_start'], burn['day_end'], alpha=0.15, color='orange')

    ax1.set_ylabel('Position Error (km)', fontsize=12)
    ax1.set_title('Trajectory Error vs NASA Ephemeris — Full Mission', fontsize=14)
    ax1.legend(fontsize=11, facecolor='white', edgecolor='black', labelcolor='black')
    ax1.grid(True, alpha=0.3, color='gray')

    ax2.set_xlabel('Time (days)', fontsize=12)
    ax2.set_ylabel('Velocity Error (km/s)', fontsize=12)
    ax2.legend(fontsize=11, facecolor='white', edgecolor='black', labelcolor='black')
    ax2.grid(True, alpha=0.3, color='gray')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'full_error_comparison.png'), dpi=150, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close(fig)
    print("    Saved full_error_comparison.png")

    # --- Figure 4: 2D projections ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor='white')
    projs = [(0,1,'X','Y'), (0,2,'X','Z'), (1,2,'Y','Z')]
    for ax, (i,j,xl,yl) in zip(axes, projs):
        ax.plot(r_nasa[:,i], r_nasa[:,j], 'b-', lw=2, label='NASA', alpha=0.9)
        if sol_ballistic is not None:
            ax.plot(sol_ballistic.y[i], sol_ballistic.y[j], 'r--', lw=1, label='Ballistic', alpha=0.5)
        if ipopt_result is not None:
            X_sol = ipopt_result[0]
            ax.plot(X_sol[i,:], X_sol[j,:], 'g-', lw=1.5, label='IPOPT', alpha=0.8)
        ax.scatter([0],[0], c='cyan', s=100, edgecolors='blue', lw=1.5, zorder=10)
        ax.scatter([rm_flyby[i]], [rm_flyby[j]], c='silver', s=100, edgecolors='black', lw=1, zorder=10)
        ax.set_xlabel(f'{xl} (km)', fontsize=11)
        ax.set_ylabel(f'{yl} (km)', fontsize=11)
        ax.set_title(f'{xl}{yl} Projection', fontsize=12)
        ax.legend(fontsize=9, facecolor='white', edgecolor='black', labelcolor='black')
        ax.grid(True, alpha=0.3, color='gray')

    fig.suptitle('Artemis II — 2D Projections (Full Mission)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'full_2d_projections.png'), dpi=150, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close(fig)
    print("    Saved full_2d_projections.png")


# ============================================================================
# ANIMATION
# ============================================================================
def create_animation(times_sec, nasa_pos, sol_ballistic, ipopt_result,
                     ephem_cache, burns, output_dir, day_offsets_seg, fps=15, duration=16):
    print("\n  Creating animation...", flush=True)

    t_nasa = times_sec
    r_nasa = nasa_pos
    n_frames = fps * duration

    # Subsample NASA for animation
    frame_indices = np.linspace(0, len(t_nasa)-1, n_frames).astype(int)

    # Precompute Moon positions at each frame
    moon_positions = np.array([ephem_cache.get_moon_at(t_nasa[i]) for i in frame_indices])

    # Ballistic trajectory (full)
    if sol_ballistic is not None:
        bal_pos = sol_ballistic.y[0:3].T  # (N, 3)
        bal_t = sol_ballistic.t
    else:
        bal_pos = None

    # IPOPT trajectory
    if ipopt_result is not None:
        ip_pos = ipopt_result[0][0:3, :].T  # (n_seg+1, 3)
        ip_seg_times = ipopt_result[2]
    else:
        ip_pos = None

    # --- Setup figure ---
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # Set axis style
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('black')
        axis.label.set_color('black')
        axis.set_tick_params(colors='black')

    ax.set_xlabel('X (km)', fontsize=10, color='black')
    ax.set_ylabel('Y (km)', fontsize=10, color='black')
    ax.set_zlabel('Z (km)', fontsize=10, color='black')

    # Compute axis limits from NASA trajectory
    pad = 30000
    xlim = [r_nasa[:,0].min()-pad, r_nasa[:,0].max()+pad]
    ylim = [r_nasa[:,1].min()-pad, r_nasa[:,1].max()+pad]
    zlim = [r_nasa[:,2].min()-pad, r_nasa[:,2].max()+pad]
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)

    # Static elements
    ax.scatter([0], [0], [0], c='cyan', s=200, marker='o', edgecolors='dodgerblue',
               lw=2, zorder=10)  # Earth

    # Moon orbit trace (full)
    moon_full_t = np.linspace(t_nasa[0], t_nasa[-1], 500)
    moon_full = np.array([ephem_cache.get_moon_at(t) for t in moon_full_t])
    ax.plot(moon_full[:,0], moon_full[:,1], moon_full[:,2],
            '--', color='gray', lw=0.5, alpha=0.5)

    # Precompute per-segment burn flag for IPOPT trajectory coloring
    ip_is_burn = np.zeros(len(ip_seg_times), dtype=bool) if ip_pos is not None else None
    if ip_pos is not None and burns:
        ip_days = (ip_seg_times - ip_seg_times[0]) / 86400.0
        for burn in burns:
            ip_is_burn |= ((ip_days >= burn['day_start'] - 0.02) &
                           (ip_days <= burn['day_end'] + 0.02))

    # Colors: coast = green, burn = orange
    COLOR_COAST = '#2ca02c'
    COLOR_BURN = '#ff7f0e'

    # Initialize animated elements
    nasa_line, = ax.plot([], [], [], '-', color='#1f77b4', lw=2.5, label='NASA (OEM)')
    nasa_dot, = ax.plot([], [], [], 'o', color='#1f77b4', ms=8, zorder=15)

    bal_line, = ax.plot([], [], [], '--', color='#d62728', lw=1.2, alpha=0.6, label='Ballistic')

    # IPOPT: use Line3DCollection for per-segment coloring
    # Initialize with a dummy segment (will be overwritten on first update)
    dummy_seg = [[[0, 0, 0], [0, 0, 0]]]
    ipopt_collection = Line3DCollection(dummy_seg, linewidths=[0], colors=[COLOR_COAST], alpha=0.85)
    ax.add_collection3d(ipopt_collection)
    # Dummy lines for legend entries
    ax.plot([], [], [], '-', color=COLOR_COAST, lw=2, label='IPOPT (coast)')
    ax.plot([], [], [], '-', color=COLOR_BURN, lw=3, label='IPOPT (burn)')

    moon_dot, = ax.plot([], [], [], 'o', color='silver', ms=12, zorder=10,
                         markeredgecolor='black', markeredgewidth=1)

    title = ax.set_title('', fontsize=14, color='black', pad=20)

    ax.legend(loc='upper left', fontsize=9, facecolor='white', edgecolor='black',
              labelcolor='black')

    def update(frame):
        idx = frame_indices[frame]
        current_day = day_offsets_seg[idx]

        # NASA trace
        nasa_line.set_data_3d(r_nasa[:idx+1, 0], r_nasa[:idx+1, 1], r_nasa[:idx+1, 2])
        nasa_dot.set_data_3d([r_nasa[idx, 0]], [r_nasa[idx, 1]], [r_nasa[idx, 2]])

        # Ballistic trace (up to same time)
        if bal_pos is not None:
            bal_mask = bal_t <= t_nasa[idx]
            if np.any(bal_mask):
                bp = bal_pos[bal_mask]
                bal_line.set_data_3d(bp[:, 0], bp[:, 1], bp[:, 2])

        # IPOPT trace — colored by phase
        if ip_pos is not None:
            ip_mask = ip_seg_times <= t_nasa[idx]
            n_show = np.sum(ip_mask)
            if n_show > 1:
                pts = ip_pos[:n_show]
                segments = [[pts[j], pts[j+1]] for j in range(n_show - 1)]
                colors = [COLOR_BURN if ip_is_burn[j] else COLOR_COAST
                          for j in range(n_show - 1)]
                widths = [3.0 if ip_is_burn[j] else 2.0
                          for j in range(n_show - 1)]
                ipopt_collection.set_segments(segments)
                ipopt_collection.set_colors(colors)
                ipopt_collection.set_linewidths(widths)

        # Moon
        rm = moon_positions[frame]
        moon_dot.set_data_3d([rm[0]], [rm[1]], [rm[2]])

        # Phase label
        in_burn = False
        burn_name = ""
        if burns:
            for burn in burns:
                if burn['day_start'] <= current_day <= burn['day_end']:
                    in_burn = True
                    burn_name = burn.get('name', 'BURN')
                    break

        if in_burn:
            phase = burn_name
        else:
            r_earth = np.linalg.norm(r_nasa[idx])
            if current_day < 0.87:
                phase = "Pre-TLI Coast"
            elif current_day < 4:
                phase = "Translunar Coast"
            elif current_day < 6:
                phase = f"Lunar Flyby (r={r_earth/1000:.0f}k km)"
            elif current_day < 8.8:
                phase = "Free-Return Coast"
            else:
                phase = "Entry Approach"

        title.set_text(f'Artemis II \u2014 Day {current_day:.2f}  |  {phase}')

        # Slow rotation
        ax.view_init(elev=25, azim=30 + frame * 0.5)

        return nasa_line, nasa_dot, bal_line, ipopt_collection, moon_dot, title

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=False)

    gif_path = os.path.join(output_dir, 'artemis2_full_mission.gif')
    print(f"    Saving {n_frames} frames at {fps} fps ({duration}s)...")
    writer = PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer, dpi=100)
    plt.close(fig)
    print(f"    Saved artemis2_full_mission.gif ({os.path.getsize(gif_path)/1e6:.1f} MB)")


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
        if os.path.isdir(d):
            for f in os.listdir(d):
                if 'Artemis' in f and f.endswith('.asc'):
                    oem_file = os.path.join(d, f)
                    break
        if oem_file:
            break

    if oem_file is None:
        print("ERROR: OEM file not found!")
        sys.exit(1)

    print(f"\nOEM: {oem_file}")
    times_utc, positions, velocities = parse_oem(oem_file)
    t0_utc = times_utc[0]

    # Detect burns
    print("\n  Detecting engine burns...")
    day_offsets = np.array([(t - t0_utc).total_seconds() / 86400.0 for t in times_utc])
    burns = detect_burns(times_utc, velocities, t0_utc)

    # Full mission: use all OEM data
    seg_start_day = 0.0
    seg_end_day = day_offsets[-1]
    mask = np.ones(len(times_utc), dtype=bool)

    seg_times_utc = times_utc
    seg_pos = positions
    seg_vel = velocities

    print(f"\n  Full mission: day {seg_start_day:.2f} to {seg_end_day:.2f}")
    print(f"  {len(seg_times_utc)} points")

    t_seg_start = seg_times_utc[0]
    t_seg_end = seg_times_utc[-1]
    times_sec = np.array([(t - t_seg_start).total_seconds() for t in seg_times_utc])

    r0 = seg_pos[0]
    v0 = seg_vel[0]
    rf_target = seg_pos[-1]
    vf_target = seg_vel[-1]

    t_span_sec = [0.0, times_sec[-1]]
    print(f"  Duration: {times_sec[-1]/86400:.2f} days")
    print(f"  r0 = [{r0[0]:.0f}, {r0[1]:.0f}, {r0[2]:.0f}] km")
    print(f"  rf = [{rf_target[0]:.0f}, {rf_target[1]:.0f}, {rf_target[2]:.0f}] km")

    # Ephemeris cache
    margin = 3600.0
    ephem_cache = EphemerisCache(
        t_seg_start - timedelta(seconds=margin),
        t_seg_end + timedelta(seconds=margin),
        n_points=10000
    )

    t_offset = margin
    times_sec_cache = times_sec + t_offset
    t_span_cache = [t_offset, times_sec[-1] + t_offset]

    # Ballistic propagation
    sol_ballistic = propagate_ballistic(r0, v0, t_span_cache, ephem_cache, n_eval=3000)

    # IPOPT — reconstruct trajectory with control
    ipopt_result = solve_ipopt(
        r0, v0, rf_target, vf_target, t_span_cache, ephem_cache,
        n_seg=120,
        nasa_data=(times_sec_cache, seg_pos, seg_vel),
        burns=burns
    )

    # Plots
    plot_static(times_sec_cache, seg_pos, seg_vel, sol_ballistic, ipopt_result,
                ephem_cache, burns, script_dir, day_offsets)

    # Animation
    create_animation(times_sec_cache, seg_pos, sol_ballistic, ipopt_result,
                     ephem_cache, burns, script_dir, day_offsets,
                     fps=15, duration=14)

    print("\n" + "=" * 80)
    print("DONE — All outputs saved to:", script_dir)
    print("=" * 80)


if __name__ == '__main__':
    main()
