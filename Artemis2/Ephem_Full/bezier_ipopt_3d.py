#!/usr/bin/env python3
"""
bezier_ipopt_3d.py - 3D ephemeris-driven Bezier collocation with CasADi/IPOPT.

This module is intentionally parallel to artemis2_full_mission.py's RK4
multiple-shooting implementation. It keeps the same dimensional units
(km, s, km/s) and the same ephemeris-cache pattern, but replaces the RK4
defects with segmented Bezier collocation defects.
"""

from __future__ import annotations

import time as timer
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import comb

try:
    import casadi as ca
except ImportError:  # Allows dry-run sizing in environments without CasADi.
    ca = None


MU_EARTH = 398600.4418
MU_MOON = 4902.800066
MU_SUN = 132712440041.93938


def bernstein_coeffs(n: int) -> np.ndarray:
    """Return binomial coefficients for a degree-n Bernstein basis."""
    return np.array([comb(n, i, exact=True) for i in range(n + 1)], dtype=float)


def bezier_eval_casadi(cp, tau: float, binom_coeffs: np.ndarray):
    """Evaluate a Bezier curve stored as an (n+1, state_dim) CasADi matrix."""
    n = cp.shape[0] - 1
    state_dim = cp.shape[1]
    result = ca.MX.zeros(state_dim)
    for i in range(n + 1):
        basis_i = binom_coeffs[i] * (1.0 - tau) ** (n - i) * tau ** i
        result += basis_i * cp[i, :].T
    return result


def bezier_deriv_casadi(cp, tau: float, binom_coeffs_deriv: np.ndarray):
    """Evaluate d/dtau of a Bezier curve."""
    n = cp.shape[0] - 1
    state_dim = cp.shape[1]
    if n < 1:
        return ca.MX.zeros(state_dim)

    q = ca.MX.zeros(n, state_dim)
    for i in range(n):
        q[i, :] = n * (cp[i + 1, :] - cp[i, :])
    return bezier_eval_casadi(q, tau, binom_coeffs_deriv)


def artemis_dynamics_casadi(state, control, r_moon, r_sun):
    """
    Earth-centered spacecraft dynamics with third-body Moon/Sun terms.

    Inputs are dimensional:
      state = [rx, ry, rz, vx, vy, vz] in km and km/s
      control = thrust acceleration in km/s^2
      r_moon, r_sun = Moon/Sun position relative to Earth in km
    """
    r_sc = state[0:3]
    v_sc = state[3:6]

    a_earth = -MU_EARTH * r_sc / ca.norm_2(r_sc) ** 3

    d_moon = r_moon - r_sc
    a_moon = MU_MOON * (
        d_moon / ca.norm_2(d_moon) ** 3
        - r_moon / ca.norm_2(r_moon) ** 3
    )

    d_sun = r_sun - r_sc
    a_sun = MU_SUN * (
        d_sun / ca.norm_2(d_sun) ** 3
        - r_sun / ca.norm_2(r_sun) ** 3
    )

    return ca.vertcat(v_sc, a_earth + a_moon + a_sun + control)


def artemis_gravity_numpy(position: np.ndarray, r_moon: np.ndarray, r_sun: np.ndarray) -> np.ndarray:
    """Dimensional third-body gravity acceleration without thrust."""
    r_sc = np.asarray(position, dtype=float)
    r_moon = np.asarray(r_moon, dtype=float)
    r_sun = np.asarray(r_sun, dtype=float)

    a_earth = -MU_EARTH * r_sc / np.linalg.norm(r_sc) ** 3
    d_moon = r_moon - r_sc
    a_moon = MU_MOON * (
        d_moon / np.linalg.norm(d_moon) ** 3
        - r_moon / np.linalg.norm(r_moon) ** 3
    )
    d_sun = r_sun - r_sc
    a_sun = MU_SUN * (
        d_sun / np.linalg.norm(d_sun) ** 3
        - r_sun / np.linalg.norm(r_sun) ** 3
    )
    return a_earth + a_moon + a_sun


def build_burn_aware_seg_times(
    t_span_sec,
    nominal_n_seg: int = 120,
    burns: list[dict] | None = None,
    burn_dt: float = 120.0,
    burn_padding_days: float = 0.05,
) -> np.ndarray:
    """Match the full-mission RK4 code's nonuniform burn/coast mesh."""
    t0, tf = float(t_span_sec[0]), float(t_span_sec[1])
    total_dur = tf - t0
    coast_dt = total_dur / nominal_n_seg

    seg_time_list = [t0]
    t_curr = t0
    while t_curr < tf - 1.0:
        curr_day = (t_curr - t0) / 86400.0
        in_burn = False
        if burns:
            for burn in burns:
                if burn["day_start"] - burn_padding_days <= curr_day <= burn["day_end"] + burn_padding_days:
                    in_burn = True
                    break

        t_curr += burn_dt if in_burn else coast_dt
        t_curr = min(t_curr, tf)
        seg_time_list.append(t_curr)

    return np.array(seg_time_list, dtype=float)


def build_control_bounds(
    seg_times: np.ndarray,
    burns: list[dict] | None = None,
    coast_bound: float = 1e-4,
    burn_headroom: float = 1.5,
) -> np.ndarray:
    """Per-segment scalar component bounds for control acceleration."""
    seg_times = np.asarray(seg_times, dtype=float)
    t0 = float(seg_times[0])
    seg_days = (seg_times - t0) / 86400.0
    u_bounds = np.ones(len(seg_times) - 1, dtype=float) * coast_bound

    if burns:
        for burn in burns:
            for k in range(len(u_bounds)):
                seg_day = 0.5 * (seg_days[k] + seg_days[k + 1])
                if burn["day_start"] - 0.1 <= seg_day <= burn["day_end"] + 0.1:
                    u_bounds[k] = max(u_bounds[k], float(burn["max_accel"]) * burn_headroom)

    return u_bounds


@dataclass(frozen=True)
class WaypointStats:
    count: int
    full_state_count: int
    position_only_count: int


class Artemis3DBezierIPOPT:
    """
    Segmented Bezier direct collocation for the 3D Artemis ephemeris problem.

    Decision variables per segment:
      - (degree + 1) six-dimensional state control points
      - n_collocation three-dimensional control vectors at GL nodes
    """

    state_dim = 6
    ctrl_dim = 3

    def __init__(
        self,
        seg_times,
        ephem_cache,
        bezier_degree: int = 5,
        n_collocation: int | None = None,
        u_bounds: np.ndarray | None = None,
        burns: list[dict] | None = None,
        waypoint_pin_interval_s: float = 8.0 * 3600.0,
        burn_pin_every: int = 5,
    ):
        if ca is None:
            raise ImportError(
                "CasADi is required to solve the Bezier/IPOPT NLP. "
                "On this machine, use: conda run -n cr3bp python ..."
            )

        self.seg_times = np.asarray(seg_times, dtype=float)
        if self.seg_times.ndim != 1 or len(self.seg_times) < 2:
            raise ValueError("seg_times must be a 1D array with at least two entries")
        if np.any(np.diff(self.seg_times) <= 0.0):
            raise ValueError("seg_times must be strictly increasing")

        self.ephem_cache = ephem_cache
        self.n_seg = len(self.seg_times) - 1
        self.deg = int(bezier_degree)
        self.n_colloc = int(n_collocation if n_collocation is not None else self.deg + 1)
        self.burns = burns or []
        self.waypoint_pin_interval_s = float(waypoint_pin_interval_s)
        self.burn_pin_every = int(burn_pin_every)

        if self.deg < 1:
            raise ValueError("bezier_degree must be >= 1")
        if self.n_colloc < 1:
            raise ValueError("n_collocation must be >= 1")

        from numpy.polynomial.legendre import leggauss

        pts, wts = leggauss(self.n_colloc)
        self.tau_c = 0.5 * (pts + 1.0)
        self.wts_c = 0.5 * wts

        self.binom_n = bernstein_coeffs(self.deg)
        self.binom_nm1 = bernstein_coeffs(max(self.deg - 1, 0))

        if u_bounds is None:
            self.u_bounds = np.ones(self.n_seg, dtype=float) * 1e-4
        else:
            self.u_bounds = np.asarray(u_bounds, dtype=float)
            if self.u_bounds.shape != (self.n_seg,):
                raise ValueError("u_bounds must have one scalar bound per segment")

    def solve(
        self,
        x0,
        xf,
        nasa_data=None,
        z_guess=None,
        max_iter: int = 5000,
        tol: float = 1e-6,
        acceptable_tol: float = 1e-4,
        print_level: int = 3,
    ) -> dict:
        """Build and solve the Bezier NLP."""
        x0 = np.asarray(x0, dtype=float)
        xf = np.asarray(xf, dtype=float)

        if x0.shape != (self.state_dim,) or xf.shape != (self.state_dim,):
            raise ValueError("x0 and xf must be length-6 state vectors")

        n = self.deg
        s = self.state_dim
        d = self.ctrl_dim
        nc = self.n_colloc

        cp_vars = []
        u_vars = []
        z_parts = []
        lbx_parts = []
        ubx_parts = []

        inf = np.inf
        for seg in range(self.n_seg):
            cp_seg = ca.MX.sym(f"cp_{seg}", n + 1, s)
            cp_vars.append(cp_seg)
            z_parts.append(ca.reshape(cp_seg, -1, 1))
            lbx_parts.append(np.full((n + 1) * s, -inf))
            ubx_parts.append(np.full((n + 1) * s, inf))

            u_seg = ca.MX.sym(f"u_{seg}", nc, d)
            u_vars.append(u_seg)
            z_parts.append(ca.reshape(u_seg, -1, 1))
            bound = float(self.u_bounds[seg])
            lbx_parts.append(np.full(nc * d, -bound))
            ubx_parts.append(np.full(nc * d, bound))

        z = ca.vertcat(*z_parts)
        lbx = np.concatenate(lbx_parts)
        ubx = np.concatenate(ubx_parts)

        g = []
        lbg = []
        ubg = []

        def add_eq(expr, size):
            g.append(ca.reshape(expr, -1, 1))
            lbg.extend([0.0] * size)
            ubg.extend([0.0] * size)

        add_eq(cp_vars[0][0, :].T - x0, s)
        add_eq(cp_vars[-1][n, :].T - xf, s)

        for seg in range(self.n_seg - 1):
            add_eq((cp_vars[seg][n, :] - cp_vars[seg + 1][0, :]).T, s)

        waypoint_stats = self._add_waypoint_constraints(cp_vars, g, lbg, ubg, nasa_data)

        for seg in range(self.n_seg):
            cp = cp_vars[seg]
            u_seg = u_vars[seg]
            h_seg = float(self.seg_times[seg + 1] - self.seg_times[seg])
            t_start = float(self.seg_times[seg])

            for j in range(nc):
                tau = float(self.tau_c[j])
                t_colloc = t_start + tau * h_seg
                r_moon, r_sun = self.ephem_cache.get_positions(t_colloc)

                x_j = bezier_eval_casadi(cp, tau, self.binom_n)
                dx_dtau_j = bezier_deriv_casadi(cp, tau, self.binom_nm1)
                dx_dt_j = dx_dtau_j / h_seg
                u_j = u_seg[j, :].T

                f_j = artemis_dynamics_casadi(x_j, u_j, r_moon, r_sun)
                add_eq(dx_dt_j - f_j, s)

        objective = 0.0
        for seg in range(self.n_seg):
            h_seg = float(self.seg_times[seg + 1] - self.seg_times[seg])
            for j in range(nc):
                u_j = u_vars[seg][j, :]
                objective += float(self.wts_c[j]) * ca.dot(u_j, u_j) * h_seg

        nlp = {"x": z, "f": objective, "g": ca.vertcat(*g)}
        opts = {
            "ipopt.max_iter": int(max_iter),
            "ipopt.tol": float(tol),
            "ipopt.acceptable_tol": float(acceptable_tol),
            "ipopt.constr_viol_tol": max(float(tol) * 0.1, 1e-10),
            "ipopt.print_level": int(print_level),
            "ipopt.linear_solver": "mumps",
            "ipopt.sb": "yes",
            "print_time": False,
        }
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        z0 = self._build_initial_guess(x0, xf, nasa_data=nasa_data, z_guess=z_guess)

        t_start = timer.perf_counter()
        sol = solver(
            x0=z0,
            lbx=lbx,
            ubx=ubx,
            lbg=np.asarray(lbg),
            ubg=np.asarray(ubg),
        )
        solve_time = timer.perf_counter() - t_start

        z_opt = np.asarray(sol["x"]).reshape(-1)
        stats = solver.stats()
        segments, controls = self.unpack_solution(z_opt)
        trajectory = self.evaluate_trajectory(segments, controls)

        return {
            **trajectory,
            "cost": float(sol["f"]),
            "solve_time": float(solve_time),
            "success": bool(stats.get("success", False)),
            "stats": stats,
            "z_opt": z_opt,
            "segments": segments,
            "controls": controls,
            "seg_times": self.seg_times.copy(),
            "u_bounds": self.u_bounds.copy(),
            "waypoint_stats": waypoint_stats,
            "n_vars": int(z.shape[0]),
            "n_constraints": int(len(lbg)),
        }

    def _add_waypoint_constraints(self, cp_vars, g, lbg, ubg, nasa_data) -> WaypointStats:
        """Pin Bezier endpoint control points to OEM waypoints."""
        if nasa_data is None:
            return WaypointStats(count=0, full_state_count=0, position_only_count=0)

        nasa_t, nasa_pos, nasa_vel = nasa_data
        pos_interp = interp1d(nasa_t, nasa_pos, axis=0, fill_value="extrapolate")
        vel_interp = interp1d(nasa_t, nasa_vel, axis=0, fill_value="extrapolate")

        seg_days = (self.seg_times - self.seg_times[0]) / 86400.0
        last_pin_time = float(self.seg_times[0])
        burn_node_count = 0
        full_state_count = 0
        position_only_count = 0

        def add_eq(expr, size):
            g.append(ca.reshape(expr, -1, 1))
            lbg.extend([0.0] * size)
            ubg.extend([0.0] * size)

        for i in range(1, self.n_seg):
            seg_day_i = float(seg_days[i])
            dt_since_pin = float(self.seg_times[i] - last_pin_time)

            in_burn = False
            for burn in self.burns:
                if burn["day_start"] - 0.02 <= seg_day_i <= burn["day_end"] + 0.02:
                    in_burn = True
                    break

            # Pin the right endpoint of segment i-1. C0 continuity makes this
            # equivalent to pinning the left endpoint of segment i.
            node_cp = cp_vars[i - 1][self.deg, :].T
            if in_burn:
                burn_node_count += 1
                if self.burn_pin_every > 0 and burn_node_count % self.burn_pin_every == 0:
                    p = np.asarray(pos_interp(self.seg_times[i]), dtype=float)
                    add_eq(node_cp[0:3] - p, 3)
                    position_only_count += 1
                    last_pin_time = float(self.seg_times[i])
            else:
                burn_node_count = 0
                if dt_since_pin >= self.waypoint_pin_interval_s:
                    p = np.asarray(pos_interp(self.seg_times[i]), dtype=float)
                    v = np.asarray(vel_interp(self.seg_times[i]), dtype=float)
                    add_eq(node_cp[0:3] - p, 3)
                    add_eq(node_cp[3:6] - v, 3)
                    full_state_count += 1
                    last_pin_time = float(self.seg_times[i])

        return WaypointStats(
            count=full_state_count + position_only_count,
            full_state_count=full_state_count,
            position_only_count=position_only_count,
        )

    def _build_initial_guess(self, x0, xf, nasa_data=None, z_guess=None) -> np.ndarray:
        if z_guess is not None:
            return np.asarray(z_guess, dtype=float)
        if nasa_data is not None:
            return self._warm_start_from_oem(nasa_data, x0, xf)
        return self._linear_guess(x0, xf)

    def _linear_guess(self, x0, xf) -> np.ndarray:
        z_parts = []
        t0 = float(self.seg_times[0])
        tf = float(self.seg_times[-1])
        for seg in range(self.n_seg):
            h_seg = float(self.seg_times[seg + 1] - self.seg_times[seg])
            t_start = float(self.seg_times[seg])
            cp = np.zeros((self.deg + 1, self.state_dim))
            for i in range(self.deg + 1):
                alpha = (t_start + (i / self.deg) * h_seg - t0) / (tf - t0)
                cp[i] = (1.0 - alpha) * x0 + alpha * xf
            z_parts.append(cp.ravel(order="F"))
            z_parts.append(np.zeros(self.n_colloc * self.ctrl_dim))
        return np.concatenate(z_parts)

    def _warm_start_from_oem(self, nasa_data, x0, xf) -> np.ndarray:
        nasa_t, nasa_pos, nasa_vel = nasa_data
        x_ref = np.column_stack([nasa_pos, nasa_vel])
        interp_x = interp1d(nasa_t, x_ref, axis=0, kind="linear", fill_value="extrapolate")

        z_parts = []
        for seg in range(self.n_seg):
            t_start = float(self.seg_times[seg])
            t_end = float(self.seg_times[seg + 1])
            h_seg = t_end - t_start

            cp = self._fit_segment_control_points(interp_x, t_start, h_seg)
            if seg == 0:
                cp[0] = x0
            else:
                cp[0] = interp_x(t_start)
            if seg == self.n_seg - 1:
                cp[self.deg] = xf
            else:
                cp[self.deg] = interp_x(t_end)
            z_parts.append(cp.ravel(order="F"))

            u_seg = self._estimate_segment_controls(interp_x, t_start, h_seg)
            bound = float(self.u_bounds[seg])
            u_seg = np.clip(u_seg, -0.8 * bound, 0.8 * bound)
            z_parts.append(u_seg.ravel(order="F"))

        return np.concatenate(z_parts)

    def _fit_segment_control_points(self, interp_x, t_start: float, h_seg: float) -> np.ndarray:
        n_sample = max(self.deg + 1, 30)
        tau_s = np.linspace(0.0, 1.0, n_sample)
        t_s = t_start + tau_s * h_seg
        x_s = interp_x(t_s)

        b_mat = np.zeros((n_sample, self.deg + 1))
        for i in range(self.deg + 1):
            b_mat[:, i] = (
                comb(self.deg, i, exact=True)
                * (1.0 - tau_s) ** (self.deg - i)
                * tau_s ** i
            )

        cp, _, _, _ = np.linalg.lstsq(b_mat, x_s, rcond=None)
        return cp

    def _estimate_segment_controls(self, interp_x, t_start: float, h_seg: float) -> np.ndarray:
        t_colloc = t_start + self.tau_c * h_seg
        x_c = interp_x(t_colloc)

        dt_fd = min(max(0.025 * h_seg, 1.0), 60.0)
        t_min = float(self.seg_times[0])
        t_max = float(self.seg_times[-1])
        x_fwd = interp_x(np.minimum(t_colloc + dt_fd, t_max))
        x_bwd = interp_x(np.maximum(t_colloc - dt_fd, t_min))
        denom = np.maximum(
            np.minimum(t_colloc + dt_fd, t_max) - np.maximum(t_colloc - dt_fd, t_min),
            1e-9,
        )
        a_ref = (x_fwd[:, 3:6] - x_bwd[:, 3:6]) / denom[:, None]

        u_seg = np.zeros((self.n_colloc, self.ctrl_dim))
        for j in range(self.n_colloc):
            r_moon, r_sun = self.ephem_cache.get_positions(t_colloc[j])
            gravity = artemis_gravity_numpy(x_c[j, 0:3], r_moon, r_sun)
            u_seg[j] = a_ref[j] - gravity
        return u_seg

    def unpack_solution(self, z: np.ndarray):
        """Unpack a flat NLP vector into segment state CPs and node controls."""
        segments = []
        controls = []
        idx = 0
        cp_size = (self.deg + 1) * self.state_dim
        u_size = self.n_colloc * self.ctrl_dim

        for _ in range(self.n_seg):
            cp = z[idx:idx + cp_size].reshape(self.deg + 1, self.state_dim, order="F")
            idx += cp_size
            u_seg = z[idx:idx + u_size].reshape(self.n_colloc, self.ctrl_dim, order="F")
            idx += u_size
            segments.append(cp)
            controls.append(u_seg)

        return segments, controls

    def evaluate_trajectory(self, segments, controls, n_per_segment: int = 20) -> dict:
        """Evaluate the solved state and linearly interpolated controls."""
        t_all = []
        x_all = []
        u_all = []
        control_times = []
        control_values = []

        for seg, cp in enumerate(segments):
            h_seg = float(self.seg_times[seg + 1] - self.seg_times[seg])
            t_start = float(self.seg_times[seg])
            endpoint = seg == self.n_seg - 1
            tau = np.linspace(0.0, 1.0, n_per_segment, endpoint=endpoint)
            if len(tau) == 0:
                continue
            t_seg = t_start + tau * h_seg
            x_seg = self._evaluate_state_numpy(cp, tau)

            u_nodes = controls[seg]
            u_seg = np.zeros((len(tau), self.ctrl_dim))
            for dim in range(self.ctrl_dim):
                u_seg[:, dim] = np.interp(tau, self.tau_c, u_nodes[:, dim])

            t_all.append(t_seg)
            x_all.append(x_seg)
            u_all.append(u_seg)

            control_times.append(t_start + self.tau_c * h_seg)
            control_values.append(u_nodes)

        x = np.vstack(x_all)
        return {
            "t": np.concatenate(t_all),
            "state": x,
            "r": x[:, 0:3],
            "v": x[:, 3:6],
            "u": np.vstack(u_all),
            "control_times": np.concatenate(control_times),
            "control_values": np.vstack(control_values),
            "junction_state": self.junction_states(segments),
        }

    def junction_states(self, segments) -> np.ndarray:
        """Return the segment-junction states implied by endpoint CPs."""
        states = [segments[0][0]]
        for seg in segments:
            states.append(seg[self.deg])
        return np.vstack(states)

    def _evaluate_state_numpy(self, cp: np.ndarray, tau: np.ndarray) -> np.ndarray:
        b_mat = np.zeros((len(tau), self.deg + 1))
        for i in range(self.deg + 1):
            b_mat[:, i] = (
                comb(self.deg, i, exact=True)
                * (1.0 - tau) ** (self.deg - i)
                * tau ** i
            )
        return b_mat @ cp


def trajectory_residuals_vs_oem(result: dict, nasa_data) -> dict:
    """Compute position/velocity residual summaries against OEM interpolants."""
    nasa_t, nasa_pos, nasa_vel = nasa_data
    pos_interp = interp1d(nasa_t, nasa_pos, axis=0, fill_value="extrapolate")
    vel_interp = interp1d(nasa_t, nasa_vel, axis=0, fill_value="extrapolate")

    t = result["t"]
    pos_err = np.linalg.norm(result["r"] - pos_interp(t), axis=1)
    vel_err = np.linalg.norm(result["v"] - vel_interp(t), axis=1)

    return {
        "max_pos_km": float(np.max(pos_err)),
        "rms_pos_km": float(np.sqrt(np.mean(pos_err ** 2))),
        "max_vel_km_s": float(np.max(vel_err)),
        "rms_vel_km_s": float(np.sqrt(np.mean(vel_err ** 2))),
        "endpoint_pos_km": float(np.linalg.norm(result["junction_state"][-1, 0:3] - nasa_pos[-1])),
        "endpoint_vel_km_s": float(np.linalg.norm(result["junction_state"][-1, 3:6] - nasa_vel[-1])),
    }
