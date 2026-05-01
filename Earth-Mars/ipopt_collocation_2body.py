"""
ipopt_collocation_2body.py - CasADi/IPOPT Bézier collocation for two-body transfers

Adapts the CR3BP IPOPT collocation framework for two-body (Keplerian) dynamics.
Key differences from the CR3BP version:
  - No Coriolis terms (inertial frame)
  - Gravity: a = -μ r / |r|³

Author: Zhuorui Li, Mustakeen Bari, Advait Jawaji
Course: AAE 568 — Applied Optimal Control and Estimation, Spring 2026
"""

import numpy as np
import casadi as ca
import time as timer
from scipy.special import comb
from scipy.interpolate import interp1d


# =============================================================================
# Bernstein / Bézier in CasADi symbolics
# =============================================================================

def bernstein_coeffs(n):
    """Precompute binomial coefficients for degree-n Bernstein basis."""
    return np.array([comb(n, i, exact=True) for i in range(n + 1)])


def bezier_eval_casadi(cp, tau, binom_coeffs):
    """
    Evaluate a Bézier curve at parameter tau using CasADi symbolics.

    Args:
        cp:           (n+1, s) CasADi MX matrix of control points
        tau:          scalar CasADi symbol in [0, 1]
        binom_coeffs: precomputed binomial coefficients (numpy array)

    Returns:
        (s,) CasADi MX vector — point on the curve
    """
    n = cp.shape[0] - 1
    s = cp.shape[1]
    result = ca.MX.zeros(s)
    for i in range(n + 1):
        B_i = binom_coeffs[i] * (1.0 - tau) ** (n - i) * tau ** i
        result += B_i * cp[i, :].T
    return result


def bezier_deriv_casadi(cp, tau, binom_coeffs_deriv):
    """
    Evaluate first derivative of a Bézier curve at tau.
    Derivative of degree-n Bézier = degree-(n-1) Bézier with CPs: Q_i = n*(P_{i+1} - P_i)
    """
    n = cp.shape[0] - 1
    s = cp.shape[1]
    if n < 1:
        return ca.MX.zeros(s)

    Q = ca.MX.zeros(n, s)
    for i in range(n):
        Q[i, :] = n * (cp[i + 1, :] - cp[i, :])

    return bezier_eval_casadi(Q, tau, binom_coeffs_deriv)


# =============================================================================
# Two-body dynamics in CasADi symbolics
# =============================================================================

def two_body_dynamics_casadi(state, u, mu=1.0):
    """
    2D two-body equations of motion with control, in CasADi symbolics.

    Args:
        state: (4,) CasADi vector [x, y, vx, vy]
        u:     (2,) CasADi vector [ux, uy] — thrust acceleration
        mu:    gravitational parameter

    Returns:
        (4,) CasADi vector [ẋ, ẏ, v̇x, v̇y]
    """
    x, y, vx, vy = state[0], state[1], state[2], state[3]

    r = ca.sqrt(x ** 2 + y ** 2)

    ax = -mu * x / r ** 3 + u[0]
    ay = -mu * y / r ** 3 + u[1]

    xdot = ca.vertcat(vx, vy, ax, ay)
    return xdot


# =============================================================================
# NLP Builder
# =============================================================================

class TwoBodyBezierIPOPT:
    """
    CasADi/IPOPT solver for minimum-energy two-body transfer via Bézier
    direct collocation.

    Decision variables:
      - Bézier control points for state x(t) = [x, y, vx, vy]
      - Control values u at Gauss-Legendre collocation points

    Constraints:
      - Dynamics defects at collocation points (equality)
      - C0 continuity at segment junctions
      - Boundary conditions

    Objective:
      min J = Σ_seg Σ_k  w_k |u_k|²  dt_seg
    """

    def __init__(self, mu=1.0, n_segments=8, bezier_degree=7,
                 n_collocation=12):
        self.mu = mu
        self.n_seg = n_segments
        self.deg = bezier_degree
        self.n_colloc = n_collocation
        self.state_dim = 4   # [x, y, vx, vy]
        self.ctrl_dim = 2    # [ux, uy]

        # Gauss-Legendre quadrature on [0, 1]
        from numpy.polynomial.legendre import leggauss
        pts, wts = leggauss(n_collocation)
        self.tau_c = 0.5 * (pts + 1.0)
        self.wts_c = 0.5 * wts

        # Precompute binomial coefficients
        self.binom_n = bernstein_coeffs(bezier_degree)
        self.binom_nm1 = bernstein_coeffs(max(bezier_degree - 1, 0))

    def solve(self, x0, xf, t0, tf, z_guess=None, warm_traj=None,
              max_iter=3000, tol=1e-8, print_level=5, u_max=None):
        """
        Build and solve the NLP.

        Args:
            x0, xf:     boundary states (numpy arrays, length 4)
            t0, tf:     time window
            z_guess:    optional initial guess (flat numpy array)
            warm_traj:  optional (t_ref, x_ref) for warm-start
            max_iter:   IPOPT iteration limit
            tol:        convergence tolerance
            print_level: IPOPT verbosity (0=silent, 5=default)
            u_max:      optional scalar path bound ||u_k|| <= u_max at every
                        collocation node

        Returns:
            result: dict with 't', 'r', 'v', 'u', 'cost', 'segments',
                    'solve_time', 'success', 'stats'
        """
        x0 = np.asarray(x0, dtype=float)
        xf = np.asarray(xf, dtype=float)
        if u_max is not None and float(u_max) <= 0.0:
            raise ValueError("u_max must be positive when provided")
        dt_seg = (tf - t0) / self.n_seg

        n = self.deg
        s = self.state_dim
        d = self.ctrl_dim
        nc = self.n_colloc
        n_seg = self.n_seg

        # ==============================================================
        # Decision variables
        # ==============================================================
        cp_vars = []
        u_vars = []
        all_decision = []

        for seg in range(n_seg):
            cp_seg = ca.MX.sym(f'cp_{seg}', n + 1, s)
            cp_vars.append(cp_seg)
            all_decision.append(ca.reshape(cp_seg, -1, 1))

            u_seg = ca.MX.sym(f'u_{seg}', nc, d)
            u_vars.append(u_seg)
            all_decision.append(ca.reshape(u_seg, -1, 1))

        z = ca.vertcat(*all_decision)
        n_vars = z.shape[0]

        # ==============================================================
        # Constraints
        # ==============================================================
        g = []
        lbg = []
        ubg = []

        # Boundary conditions
        g.append(ca.reshape(cp_vars[0][0, :].T - x0, -1, 1))
        lbg += [0.0] * s
        ubg += [0.0] * s

        g.append(ca.reshape(cp_vars[-1][n, :].T - xf, -1, 1))
        lbg += [0.0] * s
        ubg += [0.0] * s

        # C0 continuity at junctions
        for seg in range(n_seg - 1):
            junction_err = cp_vars[seg][n, :] - cp_vars[seg + 1][0, :]
            g.append(ca.reshape(junction_err.T, -1, 1))
            lbg += [0.0] * s
            ubg += [0.0] * s

        # Dynamics defects at collocation points
        for seg in range(n_seg):
            cp = cp_vars[seg]
            u_seg = u_vars[seg]

            for k in range(nc):
                tau_k = float(self.tau_c[k])

                x_k = bezier_eval_casadi(cp, tau_k, self.binom_n)
                dx_dtau_k = bezier_deriv_casadi(cp, tau_k, self.binom_nm1)
                dx_dt_k = dx_dtau_k / dt_seg

                u_k = u_seg[k, :].T
                f_k = two_body_dynamics_casadi(x_k, u_k, self.mu)

                defect_k = dx_dt_k - f_k
                g.append(defect_k)
                lbg += [0.0] * s
                ubg += [0.0] * s

        # Optional path inequality: ||u_k|| <= u_max at every collocation node.
        if u_max is not None:
            u_max_sq = float(u_max) ** 2
            for seg in range(n_seg):
                u_seg = u_vars[seg]
                for k in range(nc):
                    u_k = u_seg[k, :]
                    g.append(ca.dot(u_k, u_k))
                    lbg.append(0.0)
                    ubg.append(u_max_sq)

        g = ca.vertcat(*g)

        # ==============================================================
        # Objective: min Σ w_k |u_k|² dt_seg
        # ==============================================================
        J = 0.0
        for seg in range(n_seg):
            u_seg = u_vars[seg]
            for k in range(nc):
                u_k = u_seg[k, :]
                J += self.wts_c[k] * ca.dot(u_k, u_k) * dt_seg

        # ==============================================================
        # Build NLP and solve
        # ==============================================================
        nlp = {'x': z, 'f': J, 'g': g}
        opts = {
            'ipopt.max_iter': max_iter,
            'ipopt.tol': tol,
            'ipopt.constr_viol_tol': tol * 0.1,
            'ipopt.print_level': print_level,
            'print_time': False,
            'ipopt.sb': 'yes',
            'ipopt.linear_solver': 'mumps',
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # ==============================================================
        # Initial guess
        # ==============================================================
        z0 = self._build_initial_guess(x0, xf, t0, tf, z_guess, warm_traj,
                                        n, s, d, nc, n_seg, dt_seg)

        # ==============================================================
        # Solve
        # ==============================================================
        t_start = timer.perf_counter()
        sol = solver(
            x0=z0,
            lbg=np.array(lbg),
            ubg=np.array(ubg),
        )
        solve_time = timer.perf_counter() - t_start

        z_opt = np.array(sol['x']).flatten()
        stats = solver.stats()
        success = stats['success']
        cost_opt = float(sol['f'])

        # ==============================================================
        # Unpack solution
        # ==============================================================
        segments, U = self._unpack_solution(z_opt, n, s, d, nc, n_seg)
        sol_dict = self._evaluate_trajectory(segments, U, x0, xf, t0, tf)
        sol_dict['cost'] = cost_opt
        sol_dict['solve_time'] = solve_time
        sol_dict['success'] = success
        sol_dict['stats'] = stats
        sol_dict['z_opt'] = z_opt
        sol_dict['segments'] = segments

        return sol_dict

    # ------------------------------------------------------------------
    # Initial guess construction
    # ------------------------------------------------------------------

    def _build_initial_guess(self, x0, xf, t0, tf, z_guess, warm_traj,
                              n, s, d, nc, n_seg, dt_seg):
        if z_guess is not None:
            return z_guess

        if warm_traj is not None:
            t_ref, x_ref = warm_traj
            return self._warm_start(t_ref, x_ref, x0, xf, t0, tf,
                                     n, s, d, nc, n_seg, dt_seg)

        return self._linear_guess(x0, xf, t0, tf, n, s, d, nc, n_seg, dt_seg)

    def _linear_guess(self, x0, xf, t0, tf, n, s, d, nc, n_seg, dt_seg):
        z_parts = []
        for seg in range(n_seg):
            t_start = t0 + seg * dt_seg
            cp = np.zeros((n + 1, s))
            for i in range(n + 1):
                alpha = (t_start + (i / n) * dt_seg - t0) / (tf - t0)
                cp[i] = (1 - alpha) * x0 + alpha * xf
            # CasADi reshape is column-major (Fortran order)
            z_parts.append(cp.ravel(order='F'))
            z_parts.append(np.zeros(nc * d))
        return np.concatenate(z_parts)

    def _warm_start(self, t_ref, x_ref, x0, xf, t0, tf,
                     n, s, d, nc, n_seg, dt_seg):
        """Warm-start from a reference trajectory."""
        from scipy.special import comb as sp_comb

        interp_x = interp1d(t_ref, x_ref, axis=0, kind='cubic',
                             fill_value='extrapolate')

        z_parts = []
        for seg in range(n_seg):
            t_start = t0 + seg * dt_seg
            t_end = t_start + dt_seg

            # Fit CPs via least squares
            n_sample = max(n + 1, 30)
            tau_s = np.linspace(0, 1, n_sample)
            t_s = t_start + tau_s * dt_seg
            x_s = interp_x(t_s)

            B_mat = np.zeros((n_sample, n + 1))
            for i in range(n + 1):
                B_mat[:, i] = (sp_comb(n, i, exact=True)
                               * (1.0 - tau_s) ** (n - i) * tau_s ** i)

            cp, _, _, _ = np.linalg.lstsq(B_mat, x_s, rcond=None)

            if seg == 0:
                cp[0] = x0
            else:
                cp[0] = interp_x(t_start)
            if seg == n_seg - 1:
                cp[n] = xf
            else:
                cp[n] = interp_x(t_end)

            # CasADi reshape is column-major (Fortran order), so ravel must match
            z_parts.append(cp.ravel(order='F'))

            # Estimate control at collocation points
            t_colloc = t_start + self.tau_c * dt_seg
            x_c = interp_x(t_colloc)
            r_c = x_c[:, :2]
            v_c = x_c[:, 2:]

            dt_fd = 1e-5 * dt_seg
            x_fwd = interp_x(np.minimum(t_colloc + dt_fd, tf))
            x_bwd = interp_x(np.maximum(t_colloc - dt_fd, t0))
            a_c = (x_fwd[:, 2:] - x_bwd[:, 2:]) / (2 * dt_fd)

            u_seg = np.zeros((nc, d))
            for k in range(nc):
                rk = r_c[k]
                r_norm = np.sqrt(rk[0] ** 2 + rk[1] ** 2)
                grav = -self.mu * rk / r_norm ** 3
                u_seg[k] = a_c[k] - grav

            z_parts.append(u_seg.ravel(order='F'))

        return np.concatenate(z_parts)

    # ------------------------------------------------------------------
    # Unpack solution
    # ------------------------------------------------------------------

    def _unpack_solution(self, z, n, s, d, nc, n_seg):
        segments = []
        U = []
        idx = 0
        cp_size = (n + 1) * s
        u_size = nc * d

        for seg in range(n_seg):
            # CasADi uses column-major ordering
            cp = z[idx:idx + cp_size].reshape(n + 1, s, order='F')
            idx += cp_size
            u_seg = z[idx:idx + u_size].reshape(nc, d, order='F')
            idx += u_size
            segments.append(cp)
            U.append(u_seg)

        return segments, U

    # ------------------------------------------------------------------
    # Trajectory evaluation
    # ------------------------------------------------------------------

    def _evaluate_trajectory(self, segments, U, x0, xf, t0, tf, n_eval=500):
        from scipy.special import comb as sp_comb

        dt_seg = (tf - t0) / self.n_seg
        n = self.deg
        s = self.state_dim

        n_per = max(n_eval // self.n_seg, 20)
        t_all, r_all, v_all, u_all = [], [], [], []

        for seg_idx, cp in enumerate(segments):
            t_start = t0 + seg_idx * dt_seg
            is_last = (seg_idx == self.n_seg - 1)
            tau = np.linspace(0, 1, n_per, endpoint=is_last)
            t_seg = t_start + tau * dt_seg

            B_mat = np.zeros((len(tau), n + 1))
            for i in range(n + 1):
                B_mat[:, i] = (sp_comb(n, i, exact=True)
                               * (1.0 - tau) ** (n - i) * tau ** i)
            x_eval = B_mat @ cp

            # Derivative control points
            Q = np.zeros((n, s))
            for i in range(n):
                Q[i] = n * (cp[i + 1] - cp[i])

            B_mat_d = np.zeros((len(tau), n))
            for i in range(n):
                B_mat_d[:, i] = (sp_comb(n - 1, i, exact=True)
                                 * (1.0 - tau) ** (n - 1 - i) * tau ** i)
            dx_dtau = B_mat_d @ Q
            dx_dt = dx_dtau / dt_seg

            r_eval = x_eval[:, :2]
            v_eval = x_eval[:, 2:]
            dv_dt = dx_dt[:, 2:]

            # Reconstruct control: u = v̇_bezier - gravity
            u_eval = np.zeros((len(tau), 2))
            for k in range(len(tau)):
                rk = r_eval[k]
                r_norm = np.sqrt(rk[0] ** 2 + rk[1] ** 2)
                grav = -self.mu * rk / r_norm ** 3
                u_eval[k] = dv_dt[k] - grav

            t_all.append(t_seg)
            r_all.append(r_eval)
            v_all.append(v_eval)
            u_all.append(u_eval)

        return {
            't': np.concatenate(t_all),
            'r': np.vstack(r_all),
            'v': np.vstack(v_all),
            'u': np.vstack(u_all),
        }


# =============================================================================
# Mesh Refinement Cascade
# =============================================================================

def mesh_refine_solve(x0, xf, t0, tf, mu=1.0,
                      levels=None, print_level=3, tol=1e-8):
    """
    Coarse-to-fine mesh refinement cascade for two-body transfers.
    """
    if levels is None:
        levels = [
            (4,  7, 12),
            (8,  7, 12),
            (16, 7, 12),
            (32, 7, 12),
        ]

    history = []
    prev_sol = None

    for level_idx, (n_seg, deg, nc) in enumerate(levels):
        n_total_vars = n_seg * ((deg + 1) * 4 + nc * 2)
        print(f"\n{'='*60}")
        print(f"Mesh level {level_idx + 1}/{len(levels)}: "
              f"{n_seg} segments, degree {deg}, {nc} colloc pts "
              f"({n_total_vars} variables)")
        print(f"{'='*60}")

        solver = TwoBodyBezierIPOPT(
            mu=mu, n_segments=n_seg, bezier_degree=deg,
            n_collocation=nc,
        )

        warm_traj = None
        if prev_sol is not None:
            t_prev = prev_sol['t']
            x_prev = np.column_stack([prev_sol['r'], prev_sol['v']])
            warm_traj = (t_prev, x_prev)

        is_final = (level_idx == len(levels) - 1)
        level_tol = tol if is_final else max(tol * 100, 1e-6)
        level_iter = 3000 if is_final else 1500

        result = solver.solve(
            x0, xf, t0, tf,
            warm_traj=warm_traj,
            max_iter=level_iter,
            tol=level_tol,
            print_level=print_level,
        )

        max_defect = _compute_max_defect(result, mu)
        print(f"\n  Converged: {result['success']}")
        print(f"  Cost J = ∫|u|²dt: {result['cost']:.8f}")
        print(f"  Max dynamics defect: {max_defect:.2e}")
        print(f"  Solve time: {result['solve_time']:.3f}s")

        result['max_defect'] = max_defect
        result['level'] = (n_seg, deg, nc)
        history.append(result)
        prev_sol = result

        if not result['success'] and level_idx < len(levels) - 1:
            print("  WARNING: did not converge, continuing to next level anyway")

    return history[-1], history


def _compute_max_defect(sol, mu):
    """Compute max kinematic defect along the evaluated trajectory."""
    t = sol['t']
    r = sol['r']
    v = sol['v']

    max_def = 0.0
    for k in range(1, len(t) - 1):
        dt = t[k + 1] - t[k - 1]
        if dt < 1e-15:
            continue
        r_dot_num = (r[k + 1] - r[k - 1]) / dt
        kin_def = np.max(np.abs(r_dot_num - v[k]))
        max_def = max(max_def, kin_def)

    return max_def


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Two-Body Bézier Collocation with CasADi/IPOPT")
    print("=" * 60)

    MU = 1.0
    A_EARTH = 1.0
    A_MARS = 1.524
    V_MARS = np.pi
    VEL_EARTH = np.sqrt(MU / A_EARTH ** 3)
    VEL_MARS = np.sqrt(MU / A_MARS ** 3)

    T0, TF = 0.0, 8.0

    R0 = np.array([A_EARTH, 0.0])
    V0 = np.array([0.0, A_EARTH * VEL_EARTH])

    POS_MARS_F = A_MARS * np.array([np.cos(V_MARS + VEL_MARS * TF),
                                     np.sin(V_MARS + VEL_MARS * TF)])
    VEL_MARS_F = VEL_MARS * np.array([-POS_MARS_F[1], POS_MARS_F[0]])

    x0 = np.concatenate([R0, V0])
    xf = np.concatenate([POS_MARS_F, VEL_MARS_F])

    print(f"\nDeparture: {x0}")
    print(f"Arrival:   {xf}")
    print(f"Transfer time: {TF - T0:.4f}")

    # Mesh refinement cascade
    result, history = mesh_refine_solve(x0, xf, T0, TF, mu=MU, print_level=0)
    print(f"\nFinal result: cost={result['cost']:.8f}, "
          f"success={result['success']}")

    for i, h in enumerate(history):
        print(f"  Level {i+1} {h['level']}: cost={h['cost']:.8f}, "
              f"defect={h['max_defect']:.2e}, time={h['solve_time']:.3f}s")
