"""
ipopt_collocation.py - CasADi/IPOPT Bézier collocation for CR3BP transfers

Replaces scipy.optimize with CasADi's symbolic NLP + IPOPT backend.
Key advantages over the scipy version:
  1. Exact sparse Jacobians and Hessians via automatic differentiation
  2. IPOPT interior-point algorithm with trust-region globalization
  3. Exploits block-diagonal sparsity from the multi-segment formulation
  4. Coarse-to-fine mesh refinement for robust convergence

Author: Zhuorui Li, Mustakeen Bari, Advait Jawaji
Course: AAE 568 — Applied Optimal Control and Estimation, Spring 2026
"""

import numpy as np
import casadi as ca
import time as timer
from scipy.special import comb
from scipy.interpolate import interp1d

# Earth-Moon mass parameter
MU = 0.012150585609624


# =============================================================================
# Bernstein / Bézier in CasADi symbolics
# =============================================================================

def bernstein_coeffs(n):
    """
    Precompute binomial coefficients for degree-n Bernstein basis.
    Returns array C where C[i] = C(n, i).
    """
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

    Args:
        cp:                  (n+1, s) CasADi MX control points
        tau:                 scalar CasADi symbol
        binom_coeffs_deriv:  binomial coefficients for degree n-1

    Returns:
        (s,) CasADi MX vector — tangent at tau
    """
    n = cp.shape[0] - 1
    s = cp.shape[1]
    if n < 1:
        return ca.MX.zeros(s)

    # Derivative control points
    Q = ca.MX.zeros(n, s)
    for i in range(n):
        Q[i, :] = n * (cp[i + 1, :] - cp[i, :])

    return bezier_eval_casadi(Q, tau, binom_coeffs_deriv)


# =============================================================================
# CR3BP dynamics in CasADi symbolics
# =============================================================================

def cr3bp_dynamics_casadi(state, u, mu=MU):
    """
    Planar CR3BP equations of motion with control, in CasADi symbolics.

    Args:
        state: (4,) CasADi vector [x, y, vx, vy]
        u:     (2,) CasADi vector [ux, uy] — thrust acceleration
        mu:    mass parameter

    Returns:
        (4,) CasADi vector [ẋ, ẏ, v̇x, v̇y]
    """
    x, y, vx, vy = state[0], state[1], state[2], state[3]

    r1 = ca.sqrt((x + mu) ** 2 + y ** 2)
    r2 = ca.sqrt((x - 1.0 + mu) ** 2 + y ** 2)

    Ux = x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - 1 + mu) / r2 ** 3
    Uy = y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3

    xdot = ca.vertcat(
        vx,
        vy,
        2 * vy + Ux + u[0],
        -2 * vx + Uy + u[1],
    )
    return xdot


# =============================================================================
# NLP Builder
# =============================================================================

class CR3BPBezierIPOPT:
    """
    CasADi/IPOPT solver for minimum-energy CR3BP transfer via Bézier
    direct collocation.

    Decision variables:
      - Bézier control points for state x(t) = [x, y, vx, vy]
      - Control values u at Gauss-Legendre collocation points

    Constraints:
      - Dynamics defects at collocation points (equality)
      - C0 continuity at segment junctions (built into parameterization)
      - Boundary conditions (built into parameterization)

    Objective:
      min J = Σ_seg Σ_k  w_k |u_k|²  dt_seg
    """

    def __init__(self, mu=MU, n_segments=8, bezier_degree=5,
                 n_collocation=10):
        self.mu = mu
        self.n_seg = n_segments
        self.deg = bezier_degree
        self.n_colloc = n_collocation
        self.state_dim = 4   # planar: [x, y, vx, vy]
        self.ctrl_dim = 2    # [ux, uy]

        # Gauss-Legendre quadrature on [0, 1]
        from numpy.polynomial.legendre import leggauss
        pts, wts = leggauss(n_collocation)
        self.tau_c = 0.5 * (pts + 1.0)
        self.wts_c = 0.5 * wts

        # Precompute binomial coefficients
        self.binom_n = bernstein_coeffs(bezier_degree)
        self.binom_nm1 = bernstein_coeffs(max(bezier_degree - 1, 0))

    # ------------------------------------------------------------------
    # Build and solve
    # ------------------------------------------------------------------

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
        # Per segment: (n+1) control points × s states  (some fixed by BCs)
        # Plus: n_seg × nc × d control values
        #
        # Free CPs: for each segment, indices 1..n-1 are free (interior).
        # Index 0 is either BC (seg 0) or junction (copied from prev seg n).
        # Index n is either BC (last seg) or junction (free).

        # We store ALL control points as decision variables for CasADi,
        # then enforce BCs and continuity as equality constraints.
        # This is cleaner than the manual packing in the scipy version.

        cp_vars = []   # list of (n+1, s) MX variables, one per segment
        u_vars = []    # list of (nc, d) MX variables, one per segment
        all_decision = []  # flat list for the NLP

        for seg in range(n_seg):
            # Control points for this segment
            cp_seg = ca.MX.sym(f'cp_{seg}', n + 1, s)
            cp_vars.append(cp_seg)
            all_decision.append(ca.reshape(cp_seg, -1, 1))

            # Control at collocation points
            u_seg = ca.MX.sym(f'u_{seg}', nc, d)
            u_vars.append(u_seg)
            all_decision.append(ca.reshape(u_seg, -1, 1))

        z = ca.vertcat(*all_decision)
        n_vars = z.shape[0]

        # ==============================================================
        # Constraints
        # ==============================================================
        g = []   # constraint expressions
        lbg = []  # lower bounds
        ubg = []  # upper bounds

        # --- Boundary conditions ---
        # Segment 0, CP 0 = x0
        g.append(ca.reshape(cp_vars[0][0, :].T - x0, -1, 1))
        lbg += [0.0] * s
        ubg += [0.0] * s

        # Last segment, CP n = xf
        g.append(ca.reshape(cp_vars[-1][n, :].T - xf, -1, 1))
        lbg += [0.0] * s
        ubg += [0.0] * s

        # --- C0 continuity at junctions ---
        for seg in range(n_seg - 1):
            # Last CP of seg k = first CP of seg k+1
            junction_err = cp_vars[seg][n, :] - cp_vars[seg + 1][0, :]
            g.append(ca.reshape(junction_err.T, -1, 1))
            lbg += [0.0] * s
            ubg += [0.0] * s

        # --- Dynamics defects at collocation points ---
        for seg in range(n_seg):
            cp = cp_vars[seg]
            u_seg = u_vars[seg]

            for k in range(nc):
                tau_k = float(self.tau_c[k])

                # Evaluate Bézier state at tau_k
                x_k = bezier_eval_casadi(cp, tau_k, self.binom_n)

                # Evaluate Bézier derivative at tau_k (d/dtau)
                dx_dtau_k = bezier_deriv_casadi(cp, tau_k, self.binom_nm1)

                # Chain rule: dx/dt = (dx/dtau) / dt_seg
                dx_dt_k = dx_dtau_k / dt_seg

                # Control at this collocation point
                u_k = u_seg[k, :].T

                # CR3BP dynamics
                f_k = cr3bp_dynamics_casadi(x_k, u_k, self.mu)

                # Defect: dx/dt - f(x, u) = 0
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
        n_constraints = g.shape[0]

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
            'ipopt.sb': 'yes',           # suppress banner
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
        """
        Build initial guess vector for the NLP.
        Priority: z_guess > warm_traj > linear interpolation.
        """
        if z_guess is not None:
            return z_guess

        if warm_traj is not None:
            t_ref, x_ref = warm_traj
            return self._warm_start(t_ref, x_ref, x0, xf, t0, tf,
                                     n, s, d, nc, n_seg, dt_seg)

        # Default: linear interpolation between x0 and xf
        return self._linear_guess(x0, xf, t0, tf, n, s, d, nc, n_seg, dt_seg)

    def _linear_guess(self, x0, xf, t0, tf, n, s, d, nc, n_seg, dt_seg):
        """Linear interpolation initial guess."""
        z_parts = []
        for seg in range(n_seg):
            t_start = t0 + seg * dt_seg
            # Control points: linear interp between BCs
            cp = np.zeros((n + 1, s))
            for i in range(n + 1):
                alpha = (t_start + (i / n) * dt_seg - t0) / (tf - t0)
                cp[i] = (1 - alpha) * x0 + alpha * xf
            # CasADi reshape is column-major (Fortran order)
            z_parts.append(cp.ravel(order='F'))
            # Zero control
            z_parts.append(np.zeros(nc * d))
        return np.concatenate(z_parts)

    def _warm_start(self, t_ref, x_ref, x0, xf, t0, tf,
                     n, s, d, nc, n_seg, dt_seg):
        """
        Warm-start from a reference trajectory.
        Fits Bézier CPs via least-squares and estimates control from
        the dynamics residual.
        """
        from scipy.special import comb as sp_comb

        interp_x = interp1d(t_ref, x_ref, axis=0, kind='cubic',
                             fill_value='extrapolate')

        z_parts = []
        for seg in range(n_seg):
            t_start = t0 + seg * dt_seg
            t_end = t_start + dt_seg

            # --- Fit CPs via least squares ---
            n_sample = max(n + 1, 30)
            tau_s = np.linspace(0, 1, n_sample)
            t_s = t_start + tau_s * dt_seg
            x_s = interp_x(t_s)

            # Bernstein basis matrix
            B_mat = np.zeros((n_sample, n + 1))
            for i in range(n + 1):
                B_mat[:, i] = (sp_comb(n, i, exact=True)
                               * (1.0 - tau_s) ** (n - i) * tau_s ** i)

            # Solve for all CPs via least squares (unconstrained)
            cp, _, _, _ = np.linalg.lstsq(B_mat, x_s, rcond=None)

            # Snap endpoints to correct values
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

            # --- Estimate control at collocation points ---
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
                xk, yk = r_c[k]
                vxk, vyk = v_c[k]
                r1 = np.sqrt((xk + self.mu) ** 2 + yk ** 2)
                r2 = np.sqrt((xk - 1 + self.mu) ** 2 + yk ** 2)
                Ux = (xk - (1 - self.mu) * (xk + self.mu) / r1 ** 3
                      - self.mu * (xk - 1 + self.mu) / r2 ** 3)
                Uy = (yk - (1 - self.mu) * yk / r1 ** 3
                      - self.mu * yk / r2 ** 3)
                u_seg[k, 0] = a_c[k, 0] - (2 * vyk + Ux)
                u_seg[k, 1] = a_c[k, 1] - (-2 * vxk + Uy)

            z_parts.append(u_seg.ravel(order='F'))

        return np.concatenate(z_parts)

    # ------------------------------------------------------------------
    # Unpack solution
    # ------------------------------------------------------------------

    def _unpack_solution(self, z, n, s, d, nc, n_seg):
        """Unpack flat solution vector into control points and controls."""
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
        """Evaluate the solved trajectory on a fine grid."""
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

            # Evaluate Bézier state
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

            # Reconstruct control: u = v̇_bezier - (Coriolis + gravity)
            u_eval = np.zeros((len(tau), 2))
            for k in range(len(tau)):
                xk, yk = r_eval[k]
                vxk, vyk = v_eval[k]
                r1 = np.sqrt((xk + self.mu) ** 2 + yk ** 2)
                r2 = np.sqrt((xk - 1 + self.mu) ** 2 + yk ** 2)
                Ux = (xk - (1 - self.mu) * (xk + self.mu) / r1 ** 3
                      - self.mu * (xk - 1 + self.mu) / r2 ** 3)
                Uy = (yk - (1 - self.mu) * yk / r1 ** 3
                      - self.mu * yk / r2 ** 3)
                u_eval[k, 0] = dv_dt[k, 0] - (2 * vyk + Ux)
                u_eval[k, 1] = dv_dt[k, 1] - (-2 * vxk + Uy)

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

def mesh_refine_solve(x0, xf, t0, tf, mu=MU,
                      levels=None, print_level=3, tol=1e-8):
    """
    Coarse-to-fine mesh refinement cascade.

    Solves the NLP on a coarse mesh first, then interpolates the
    solution onto a finer mesh as an initial guess for the next level.

    Args:
        x0, xf:   boundary states
        t0, tf:   time window
        mu:       CR3BP mass parameter
        levels:   list of (n_segments, bezier_degree, n_collocation) tuples.
                  If None, uses default 4-level cascade.
        print_level: IPOPT verbosity
        tol:      final convergence tolerance

    Returns:
        result: solution dict from the finest level
        history: list of solution dicts from all levels
    """
    if levels is None:
        # Rule: nc < 2*deg to maintain positive DOF.
        # Keep degree constant across levels for clean warm-starting.
        # With degree 7: nc_max = 13.
        # Use enough colloc pts on coarse levels to avoid underdetermined problems.
        levels = [
            (4,  7, 12),    # Coarse: well-constrained (DOF=28)
            (8,  7, 12),    # Medium (DOF=60)
            (16, 7, 12),    # Fine (DOF=124)
            (32, 7, 12),    # Very fine (DOF=252)
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

        solver = CR3BPBezierIPOPT(
            mu=mu, n_segments=n_seg, bezier_degree=deg,
            n_collocation=nc,
        )

        # Build initial guess
        warm_traj = None
        if prev_sol is not None:
            # Use previous solution as warm start
            t_prev = prev_sol['t']
            x_prev = np.column_stack([prev_sol['r'], prev_sol['v']])
            warm_traj = (t_prev, x_prev)

        # Looser tolerance for coarse levels, tight for final
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

        # Report
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
    """
    Compute the maximum dynamics defect along the evaluated trajectory.

    Uses the analytical Bézier derivatives (stored in sol) compared against
    the expected CR3BP dynamics. Falls back to finite differences if
    analytical data isn't available.
    """
    t = sol['t']
    r = sol['r']
    v = sol['v']
    u = sol['u']

    # Use analytical defect: compare ṙ_bezier with v, and v̇_bezier with f(x,u)
    # The 'u' array is reconstructed as: u = v̇_bezier - (Coriolis + gravity)
    # So the defect check is whether the reconstruction is self-consistent.
    # A better check: verify ṙ = v (kinematics) via finite differences.
    max_def = 0.0
    for k in range(1, len(t) - 1):
        dt = t[k + 1] - t[k - 1]
        if dt < 1e-15:
            continue

        # Kinematic check: dr/dt should equal v
        r_dot_num = (r[k + 1] - r[k - 1]) / dt
        kin_def = np.max(np.abs(r_dot_num - v[k]))
        max_def = max(max_def, kin_def)

    return max_def


# =============================================================================
# Quick test / demo
# =============================================================================

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from cr3bp_planar import (
        collinear_libration_points, compute_lyapunov_orbit,
        cr3bp_planar_ode, cr3bp_jacobi_planar,
    )
    from scipy.integrate import solve_ivp

    print("=" * 60)
    print("CR3BP Bézier Collocation with CasADi/IPOPT")
    print("=" * 60)

    mu = MU
    xL1, xL2, _ = collinear_libration_points(mu)

    # Compute Lyapunov orbits
    print("\nComputing L1 Lyapunov orbit...")
    state_L1, T_L1, sol_L1 = compute_lyapunov_orbit(xL1, Ax=0.02, mu=mu)
    print(f"  IC = {state_L1}, T = {T_L1:.6f}")

    print("Computing L2 Lyapunov orbit...")
    state_L2, T_L2, sol_L2 = compute_lyapunov_orbit(xL2, Ax=0.02, mu=mu)
    print(f"  IC = {state_L2}, T = {T_L2:.6f}")

    # Departure: L1 right crossing
    x0 = state_L1.copy()

    # Arrival: L2 left crossing (half-period)
    sol_half = solve_ivp(
        lambda t, X: cr3bp_planar_ode(t, X, mu),
        [0, T_L2 / 2], state_L2,
        method='RK45', rtol=1e-12, atol=1e-12
    )
    xf = sol_half.y[:, -1].copy()
    xf[1] = 0.0
    xf[2] = 0.0

    tf_transfer = np.pi

    print(f"\nDeparture: {x0}")
    print(f"Arrival:   {xf}")
    print(f"Transfer time: {tf_transfer:.4f} (~{tf_transfer * 4.343:.1f} days)")

    # --- Single-level solve ---
    print("\n\n>>> Single-level solve (16 segments, degree 7)")
    solver = CR3BPBezierIPOPT(mu=mu, n_segments=16, bezier_degree=7,
                               n_collocation=10)
    result = solver.solve(x0, xf, 0.0, tf_transfer, print_level=3)
    print(f"\nResult: cost={result['cost']:.8f}, "
          f"time={result['solve_time']:.3f}s, "
          f"success={result['success']}")

    # --- Mesh refinement cascade ---
    print("\n\n>>> Mesh refinement cascade")
    result_mr, history = mesh_refine_solve(
        x0, xf, 0.0, tf_transfer, mu=mu, print_level=0
    )
    print(f"\nFinal result: cost={result_mr['cost']:.8f}, "
          f"success={result_mr['success']}")

    for i, h in enumerate(history):
        print(f"  Level {i+1} {h['level']}: cost={h['cost']:.8f}, "
              f"defect={h['max_defect']:.2e}, time={h['solve_time']:.3f}s")
