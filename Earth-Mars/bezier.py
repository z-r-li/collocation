"""
bezier.py - Bézier curve tools for trajectory collocation

Provides:
  - Bernstein polynomial evaluation
  - Single and composite Bézier curve construction
  - Collocation-based TPBVP solver (NLP formulation)
"""

import numpy as np
from scipy.special import comb
from scipy.optimize import minimize


# =============================================================================
# Bernstein Polynomials & Bézier Curves
# =============================================================================

def bernstein(n, i, t):
    """
    Evaluate the i-th Bernstein basis polynomial of degree n at parameter t.
    B_{i,n}(t) = C(n,i) * (1-t)^(n-i) * t^i
    """
    return comb(n, i, exact=True) * (1.0 - t)**(n - i) * t**i


def bezier_eval(control_points, t):
    """
    Evaluate a Bézier curve at parameter t in [0, 1].

    Args:
        control_points: (n+1, d) array of control points
        t: scalar or array of parameter values in [0, 1]

    Returns:
        points: (len(t), d) array of curve points
    """
    control_points = np.asarray(control_points)
    t = np.atleast_1d(t)
    n = len(control_points) - 1
    d = control_points.shape[1]

    points = np.zeros((len(t), d))
    for i in range(n + 1):
        B = bernstein(n, i, t)  # shape (len(t),)
        points += np.outer(B, control_points[i])

    return points


def bezier_derivative(control_points, t):
    """
    Evaluate the first derivative of a Bézier curve at parameter t.
    The derivative of a degree-n Bézier curve is a degree-(n-1) Bézier curve
    with control points: Q_i = n * (P_{i+1} - P_i)
    """
    control_points = np.asarray(control_points)
    n = len(control_points) - 1
    if n < 1:
        return np.zeros((len(np.atleast_1d(t)), control_points.shape[1]))

    # Derivative control points
    Q = n * np.diff(control_points, axis=0)
    return bezier_eval(Q, t)


def bezier_second_derivative(control_points, t):
    """
    Evaluate the second derivative of a Bézier curve at parameter t.
    Apply the derivative formula twice: first to get degree-(n-1) curve,
    then again to get degree-(n-2) curve.
    """
    control_points = np.asarray(control_points)
    n = len(control_points) - 1
    if n < 2:
        return np.zeros((len(np.atleast_1d(t)), control_points.shape[1]))

    Q = n * np.diff(control_points, axis=0)           # first derivative CPs
    R = (n - 1) * np.diff(Q, axis=0)                  # second derivative CPs
    return bezier_eval(R, t)


def composite_bezier_eval(segments, t_global, t_bounds):
    """
    Evaluate a composite Bézier curve (multiple segments joined together).

    Args:
        segments: list of (n+1, d) control point arrays, one per segment
        t_global: global parameter values in [t_bounds[0], t_bounds[-1]]
        t_bounds: array of segment boundaries, length = len(segments) + 1

    Returns:
        points: evaluated curve points
    """
    t_global = np.atleast_1d(t_global)
    n_seg = len(segments)
    d = segments[0].shape[1]
    points = np.zeros((len(t_global), d))

    for k, t_val in enumerate(t_global):
        # Find which segment this t belongs to
        seg_idx = np.searchsorted(t_bounds[1:], t_val, side='right')
        seg_idx = min(seg_idx, n_seg - 1)

        # Map to local parameter [0, 1]
        t_local = ((t_val - t_bounds[seg_idx])
                    / (t_bounds[seg_idx + 1] - t_bounds[seg_idx]))
        t_local = np.clip(t_local, 0.0, 1.0)

        points[k] = bezier_eval(segments[seg_idx], np.array([t_local]))[0]

    return points


# =============================================================================
# Cubic Bézier Chain — port of cubicChain.mlx
# =============================================================================

def cubic_bezier_chain(all_points, n_eval=50):
    """
    Build a chain of cubic Bézier segments from control points.
    Port of MATLAB cubicChain.mlx.

    Args:
        all_points: (N, d) array where every 4th point (0, 3, 6, ...) is
                    an interpolation knot, others are control points.
        n_eval: number of evaluation points per segment

    Returns:
        path: (M, d) array of evaluated curve points
        segments: list of (4, d) control point arrays
    """
    all_points = np.asarray(all_points)
    n_pts = len(all_points)
    d = all_points.shape[1]

    # Characteristic matrix for cubic Bézier
    bez_char = np.array([
        [1, 0, 0, 0],
        [-3, 3, 0, 0],
        [3, -6, 3, 0],
        [-1, 3, -3, 1]
    ], dtype=float)

    t_eval = np.linspace(0, 1, n_eval)
    t_vect = np.column_stack([np.ones_like(t_eval), t_eval,
                               t_eval**2, t_eval**3])

    segments = []
    path = []

    i = 0
    while i + 3 < n_pts:
        cp = all_points[i:i+4]
        segments.append(cp)

        seg_curve = t_vect @ bez_char @ cp
        path.append(seg_curve)
        i += 3

    if path:
        path = np.vstack(path)
    else:
        path = np.empty((0, d))

    return path, segments


# =============================================================================
# Bézier Direct Collocation for Optimal Control (NLP with dynamics constraints)
# =============================================================================

class BezierCollocation:
    """
    Solve a minimum-energy optimal control problem via Bézier direct
    collocation with dynamics enforced as hard NLP equality constraints.

    Formulation (standard direct collocation):

      Decision variables:
        - Bézier control points for the full state x(t) = [r, v]  (4D for 2D)
        - Control values u_k at each collocation point

      Equality constraints (dynamics defects) at N_c collocation points per
      segment:
        ẋ_bezier(τ_k) / dt_seg  =  f(x_bezier(τ_k), u_k)

      where f(x, u) = [v;  f_grav(r) + u]  (kinematic + dynamic eqs)

      Boundary conditions:
        x(t0) = [r0, v0],  x(tf) = [rf, vf]

      C0 continuity between segments:
        x_end(seg_i) = x_start(seg_{i+1})

      Objective:
        min J = Σ_k  w_k |u_k|²  dt_seg    (quadrature approx of ∫|u|²dt)

    This ensures the trajectory EXACTLY satisfies Newtonian dynamics (up to
    the collocation discretization), unlike a position-only parameterization.
    """

    def __init__(self, gravity_func, pos_dim=2, n_segments=6,
                 bezier_degree=5, n_collocation=10):
        """
        Args:
            gravity_func: callable(r) -> gravitational acceleration (scalar)
            pos_dim:       spatial dimension (2 for 2-D two-body)
            n_segments:    number of composite Bézier segments
            bezier_degree: polynomial degree per segment (>= 1)
            n_collocation: interior collocation points per segment
        """
        self.gravity = gravity_func
        self.d = pos_dim             # position dimension
        self.state_dim = 2 * pos_dim  # full state = [r, v]
        self.n_seg = n_segments
        self.deg = bezier_degree
        self.n_colloc = n_collocation

        # Pre-compute Gauss–Legendre nodes on [0,1]
        from numpy.polynomial.legendre import leggauss
        pts, wts = leggauss(n_collocation)
        self.tau_c = 0.5 * (pts + 1.0)     # (n_colloc,)
        self.wts_c = 0.5 * wts             # (n_colloc,)

    # ------------------------------------------------------------------
    # Variable packing / unpacking
    # ------------------------------------------------------------------

    def _count_free_cp(self):
        """Number of free state-CP scalars (after BCs + continuity)."""
        n = self.deg
        s = self.state_dim
        # Seg 0:  P0 fixed (BC) → n free interior + P_n free unless last
        # Middle: P0 is junction (fixed from prev) → same
        # Last:   P_n fixed (BC)
        # Junctions (n_seg-1) are shared: counted once as the last CP of
        # a non-final segment.
        count_pts = 0
        for seg in range(self.n_seg):
            # interior CPs: indices 1 .. n-1
            count_pts += (n - 1)
            # last CP free if not the final segment (junction or interior)
            if seg < self.n_seg - 1:
                count_pts += 1  # P_n = junction
        return count_pts * s

    def _count_control_vars(self):
        """Number of control scalars (u at each colloc pt, each segment)."""
        return self.n_seg * self.n_colloc * self.d

    def _total_vars(self):
        return self._count_free_cp() + self._count_control_vars()

    def _unpack(self, z, x0, xf, dt_seg):
        """
        Unpack decision vector z into:
          segments : list of (deg+1, state_dim) control-point arrays
          U        : (n_seg, n_colloc, d) control values
        """
        n = self.deg
        s = self.state_dim
        d = self.d

        n_cp_vars = self._count_free_cp()
        cp_flat = z[:n_cp_vars]
        u_flat  = z[n_cp_vars:]

        # --- Build segments ---
        free = cp_flat.reshape(-1, s)
        segments = []
        idx = 0
        for seg in range(self.n_seg):
            cp = np.zeros((n + 1, s))

            # First CP
            if seg == 0:
                cp[0] = x0
            else:
                cp[0] = segments[-1][n]  # C0 junction

            # Interior CPs: 1 .. n-1
            for k in range(1, n):
                cp[k] = free[idx]; idx += 1

            # Last CP
            if seg == self.n_seg - 1:
                cp[n] = xf
            else:
                cp[n] = free[idx]; idx += 1

            segments.append(cp)

        # --- Controls ---
        U = u_flat.reshape(self.n_seg, self.n_colloc, d)

        return segments, U

    # ------------------------------------------------------------------
    # Objective  J = Σ  w_k |u_k|² dt_seg
    # ------------------------------------------------------------------

    def _objective(self, z, x0, xf, dt_seg):
        n_cp_vars = self._count_free_cp()
        u_flat = z[n_cp_vars:]
        U = u_flat.reshape(self.n_seg, self.n_colloc, self.d)

        cost = 0.0
        for seg in range(self.n_seg):
            u_sq = np.sum(U[seg]**2, axis=1)          # (n_colloc,)
            cost += dt_seg * np.dot(self.wts_c, u_sq)
        return cost

    # ------------------------------------------------------------------
    # Dynamics defect constraints  (equality, must == 0)
    # ------------------------------------------------------------------

    def _defects(self, z, x0, xf, dt_seg):
        """
        Returns a 1-D array of defect values that must all be zero.

        For each segment, at each collocation point τ_k:
          defect = ẋ_bezier(τ_k) / dt_seg  -  f(x_bezier(τ_k), u_k)

        f(x, u) = [v ;  grav(r) + u]
        """
        segments, U = self._unpack(z, x0, xf, dt_seg)
        d = self.d
        s = self.state_dim
        tau = self.tau_c

        defects = []

        for seg_idx, cp in enumerate(segments):
            # Evaluate Bézier state and its τ-derivative
            x_eval  = bezier_eval(cp, tau)           # (nc, s)
            dx_dtau = bezier_derivative(cp, tau)      # (nc, s)
            dx_dt   = dx_dtau / dt_seg                # chain rule

            r_eval = x_eval[:, :d]
            v_eval = x_eval[:, d:]
            u_seg  = U[seg_idx]                       # (nc, d)

            # Compute f(x, u) = [v;  grav(r) + u]
            f_x = np.zeros_like(x_eval)
            f_x[:, :d] = v_eval
            for k in range(len(tau)):
                f_x[k, d:] = self.gravity(r_eval[k]) + u_seg[k]

            defects.append((dx_dt - f_x).ravel())

        return np.concatenate(defects)

    def _control_magnitude_margins(self, z, u_max):
        """
        SLSQP inequality margins for ||u|| <= u_max.

        SciPy treats inequality constraints as fun(z) >= 0, so each entry is
        u_max^2 - ||u_k||^2 at a collocation node.
        """
        if u_max is None:
            return np.array([])
        n_cp_vars = self._count_free_cp()
        u_flat = z[n_cp_vars:]
        U = u_flat.reshape(self.n_seg, self.n_colloc, self.d)
        return float(u_max) ** 2 - np.sum(U * U, axis=2).ravel()

    # ------------------------------------------------------------------
    # Initial guess
    # ------------------------------------------------------------------

    def _initial_guess(self, x0, xf, t0, tf):
        """
        Build an initial guess by propagating a ballistic trajectory from x0,
        blending linearly toward xf, and fitting Bézier CPs to each segment.
        Controls initialized to zero.
        """
        from scipy.integrate import solve_ivp
        from scipy.interpolate import interp1d

        d = self.d
        s = self.state_dim
        n = self.deg
        dt_seg = (tf - t0) / self.n_seg

        # Ballistic propagation
        def ode(t, X):
            r, v = X[:d], X[d:]
            return np.concatenate([v, self.gravity(r)])

        sol = solve_ivp(ode, [t0, tf], x0,
                        t_eval=np.linspace(t0, tf, 500),
                        method='RK45', rtol=1e-10, atol=1e-10)
        t_ref = sol.t
        x_ref = sol.y.T  # (N, s)

        # Blend toward xf
        alpha = np.linspace(0, 1, len(t_ref)).reshape(-1, 1)
        x_blend = x_ref * (1 - alpha) + xf * alpha
        x_blend[0]  = x0
        x_blend[-1] = xf

        interp_x = interp1d(t_ref, x_blend, axis=0, kind='cubic',
                             fill_value='extrapolate')

        # Fit each segment
        free_pts = []
        for seg in range(self.n_seg):
            t_start = t0 + seg * dt_seg
            t_end   = t_start + dt_seg

            n_sample = max(n + 1, 30)
            tau_s = np.linspace(0, 1, n_sample)
            t_s   = t_start + tau_s * dt_seg
            x_s   = interp_x(t_s)

            B_mat = np.zeros((n_sample, n + 1))
            for i in range(n + 1):
                B_mat[:, i] = bernstein(n, i, tau_s)

            cp = np.zeros((n + 1, s))
            if seg == 0:
                cp[0] = x0
            else:
                cp[0] = prev_end

            if seg == self.n_seg - 1:
                cp[n] = xf
            else:
                cp[n] = interp_x(t_end)

            # Subtract known endpoint contributions
            rhs = x_s - np.outer(B_mat[:, 0], cp[0]) - np.outer(B_mat[:, n], cp[n])

            if n > 1:
                B_int = B_mat[:, 1:n]
                cp_int, _, _, _ = np.linalg.lstsq(B_int, rhs, rcond=None)
                cp[1:n] = cp_int

            # Collect free CPs: interior (1..n-1) + junction (P_n if not last)
            for k in range(1, n):
                free_pts.append(cp[k])
            if seg < self.n_seg - 1:
                free_pts.append(cp[n])

            prev_end = cp[n]

        cp_guess = np.array(free_pts).ravel()
        u_guess  = np.zeros(self._count_control_vars())

        return np.concatenate([cp_guess, u_guess])

    def _warm_start_from_trajectory(self, t_ref, x_ref, x0, xf, t0, tf):
        """
        Build a warm-start decision vector from a known reference trajectory.

        This fits Bézier segments to the reference state history and estimates
        the control at collocation points from the dynamics residual of that
        trajectory.

        Args:
            t_ref: (N,) time array
            x_ref: (N, state_dim) state array [r, v]
            x0, xf: boundary states
            t0, tf: time window
        """
        from scipy.interpolate import interp1d

        d = self.d
        s = self.state_dim
        n = self.deg
        dt_seg = (tf - t0) / self.n_seg

        interp_x = interp1d(t_ref, x_ref, axis=0, kind='cubic',
                             fill_value='extrapolate')

        # --- Fit Bézier CPs ---
        free_pts = []
        for seg in range(self.n_seg):
            t_start = t0 + seg * dt_seg
            t_end   = t_start + dt_seg

            n_sample = max(n + 1, 30)
            tau_s = np.linspace(0, 1, n_sample)
            t_s = t_start + tau_s * dt_seg
            x_s = interp_x(t_s)

            B_mat = np.zeros((n_sample, n + 1))
            for i in range(n + 1):
                B_mat[:, i] = bernstein(n, i, tau_s)

            cp = np.zeros((n + 1, s))
            if seg == 0:
                cp[0] = x0
            else:
                cp[0] = prev_end

            if seg == self.n_seg - 1:
                cp[n] = xf
            else:
                cp[n] = interp_x(t_end)

            rhs = x_s - np.outer(B_mat[:, 0], cp[0]) \
                       - np.outer(B_mat[:, n], cp[n])

            if n > 1:
                B_int = B_mat[:, 1:n]
                cp_int, _, _, _ = np.linalg.lstsq(B_int, rhs, rcond=None)
                cp[1:n] = cp_int

            for k in range(1, n):
                free_pts.append(cp[k])
            if seg < self.n_seg - 1:
                free_pts.append(cp[n])

            prev_end = cp[n]

        cp_guess = np.array(free_pts).ravel()

        # --- Estimate control at collocation points ---
        # u_k ≈ v̇(t_k) - grav(r(t_k))
        # Estimate v̇ via finite differences on the reference
        from scipy.interpolate import UnivariateSpline

        u_all = []
        for seg in range(self.n_seg):
            t_start = t0 + seg * dt_seg
            t_colloc = t_start + self.tau_c * dt_seg
            x_c = interp_x(t_colloc)
            r_c = x_c[:, :d]

            # Numerical acceleration: finite-diff on velocity
            dt_fd = 1e-5 * dt_seg
            x_fwd = interp_x(np.minimum(t_colloc + dt_fd, tf))
            x_bwd = interp_x(np.maximum(t_colloc - dt_fd, t0))
            v_fwd = x_fwd[:, d:]
            v_bwd = x_bwd[:, d:]
            a_c = (v_fwd - v_bwd) / (2 * dt_fd)

            for k in range(len(self.tau_c)):
                u_k = a_c[k] - self.gravity(r_c[k])
                u_all.append(u_k)

        u_guess = np.array(u_all).ravel()

        return np.concatenate([cp_guess, u_guess])

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, x0, xf, t0, tf, z_guess=None, u_max=None):
        """
        Solve the minimum-energy transfer.

        Args:
            x0:  initial state [r0, v0]
            xf:  final   state [rf, vf]
            t0, tf: time window
            z_guess: optional initial guess for the full decision vector
            u_max: optional scalar path bound ||u|| <= u_max

        Returns:
            result:   scipy.optimize result
            sol_dict: dict with keys 't','r','v','u','cost'
        """
        x0 = np.asarray(x0, dtype=float)
        xf = np.asarray(xf, dtype=float)
        if u_max is not None and float(u_max) <= 0.0:
            raise ValueError("u_max must be positive when provided")
        dt_seg = (tf - t0) / self.n_seg

        if z_guess is None:
            z_guess = self._initial_guess(x0, xf, t0, tf)

        constraints = [{
            'type': 'eq',
            'fun': self._defects,
            'args': (x0, xf, dt_seg),
        }]
        if u_max is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, umax: self._control_magnitude_margins(z, umax),
                'args': (float(u_max),),
            })

        result = minimize(
            self._objective, z_guess,
            args=(x0, xf, dt_seg),
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False},
        )

        sol_dict = self._evaluate(result.x, x0, xf, t0, tf)
        sol_dict['cost'] = result.fun

        return result, sol_dict

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _evaluate(self, z, x0, xf, t0, tf, n_eval=500):
        """Evaluate the full trajectory on a fine grid."""
        dt_seg = (tf - t0) / self.n_seg
        segments, U = self._unpack(z, x0, xf, dt_seg)
        d = self.d

        n_per = n_eval // self.n_seg
        t_all, r_all, v_all, u_all = [], [], [], []

        for seg_idx, cp in enumerate(segments):
            t_start = t0 + seg_idx * dt_seg
            tau = np.linspace(0, 1, n_per,
                              endpoint=(seg_idx == self.n_seg - 1))
            t_seg = t_start + tau * dt_seg

            x_eval  = bezier_eval(cp, tau)
            dx_dtau = bezier_derivative(cp, tau)
            dx_dt   = dx_dtau / dt_seg

            r_eval = x_eval[:, :d]
            v_eval = x_eval[:, d:]

            # Reconstruct control from defect equation:
            # u = v̇_bezier - grav(r)
            dv_dt = dx_dt[:, d:]
            u_eval = np.zeros((len(tau), d))
            for k in range(len(tau)):
                u_eval[k] = dv_dt[k] - self.gravity(r_eval[k])

            t_all.append(t_seg)
            r_all.append(r_eval)
            v_all.append(v_eval)
            u_all.append(u_eval)

        return {
            't':  np.concatenate(t_all),
            'r':  np.vstack(r_all),
            'v':  np.vstack(v_all),
            'u':  np.vstack(u_all),
            'segments': [seg for seg in segments],
        }


# =============================================================================
# Bézier Collocation for TPBVPs (legacy — dynamics-residual formulation)
# =============================================================================

class BezierDirectCollocation:
    """
    Solve a TPBVP by parameterizing the trajectory with composite Bézier
    curves and enforcing dynamics at collocation points.

    The state trajectory is represented as a composite Bézier curve with
    free interior control points. Boundary conditions are enforced through
    the endpoint interpolation property. The dynamics residual is minimized
    at collocation points distributed along the trajectory.
    """

    def __init__(self, dynamics_func, state_dim, n_segments=4,
                 bezier_degree=3, n_collocation=20):
        """
        Args:
            dynamics_func: callable(t, state) -> state_dot
            state_dim: dimension of the state vector
            n_segments: number of Bézier segments
            bezier_degree: degree of each Bézier segment (default cubic)
            n_collocation: number of collocation points per segment
        """
        self.dynamics = dynamics_func
        self.state_dim = state_dim
        self.n_segments = n_segments
        self.degree = bezier_degree
        self.n_colloc = n_collocation

        # Number of control points per segment
        self.n_cp_per_seg = bezier_degree + 1
        # Total free control points (interior only; endpoints are fixed)
        # First segment: P0 fixed (BC), P1..P_{n-1} free, P_n shared
        # Middle segments: P0 shared, P1..P_{n-1} free, P_n shared
        # Last segment: P0 shared, P1..P_{n-1} free, P_n fixed (BC)
        self.n_free_interior = n_segments * (bezier_degree - 1)
        # Plus the junction points between segments (n_segments - 1)
        self.n_free_junctions = n_segments - 1
        self.n_free_total = self.n_free_interior + self.n_free_junctions
        self.n_vars = self.n_free_total * state_dim

    def _unpack_control_points(self, x_free, x0, xf):
        """
        Convert the free variable vector into a list of control point arrays.

        Returns:
            segments: list of n_segments arrays, each (degree+1, state_dim)
        """
        d = self.state_dim
        n = self.degree

        # Reshape free variables
        free_pts = x_free.reshape(-1, d)

        # Build ordered list: [P0_fixed, interior_1, ..., junction_1,
        #                      interior_2, ..., junction_2, ..., Pn_fixed]
        segments = []
        idx = 0  # index into free_pts

        for seg in range(self.n_segments):
            cp = np.zeros((n + 1, d))

            # First control point
            if seg == 0:
                cp[0] = x0
            else:
                # Junction from previous (already placed)
                cp[0] = segments[-1][-1]

            # Interior control points
            for k in range(1, n):
                cp[k] = free_pts[idx]
                idx += 1

            # Last control point
            if seg == self.n_segments - 1:
                cp[n] = xf
            else:
                cp[n] = free_pts[idx]
                idx += 1

            segments.append(cp)

        return segments

    def _collocation_residual(self, x_free, x0, xf, t0, tf):
        """
        Compute the dynamics residual at collocation points.
        """
        segments = self._unpack_control_points(x_free, x0, xf)

        dt_seg = (tf - t0) / self.n_segments
        residuals = []

        for seg_idx, cp in enumerate(segments):
            t_seg_start = t0 + seg_idx * dt_seg
            t_seg_end = t_seg_start + dt_seg

            # Collocation points (exclude endpoints to avoid redundancy)
            tau = np.linspace(0.0, 1.0, self.n_colloc + 2)[1:-1]
            t_colloc = t_seg_start + tau * dt_seg

            # Evaluate curve and derivative at collocation points
            x_eval = bezier_eval(cp, tau)
            dx_dtau = bezier_derivative(cp, tau)

            # Chain rule: dx/dt = dx/dtau * dtau/dt = dx/dtau / dt_seg
            dx_dt = dx_dtau / dt_seg

            for k in range(len(tau)):
                f_eval = self.dynamics(t_colloc[k], x_eval[k])
                residuals.append(dx_dt[k] - f_eval)

        return np.concatenate(residuals)

    def _guess_from_trajectory(self, traj_t, traj_x, x0, xf, t0, tf):
        """
        Build an initial guess for free control points by fitting Bézier
        segments to a reference trajectory (e.g., ballistic propagation).

        For each segment, sample the reference trajectory at the segment's
        parameter values and use least-squares to find best-fit control points.
        """
        from scipy.interpolate import interp1d
        d = self.state_dim
        n = self.degree

        # Interpolate reference trajectory onto uniform time grid
        interp = interp1d(traj_t, traj_x, axis=0, kind='cubic',
                          fill_value='extrapolate')

        dt_seg = (tf - t0) / self.n_segments
        free_pts = []

        prev_endpoint = None
        for seg in range(self.n_segments):
            t_start = t0 + seg * dt_seg
            t_end = t_start + dt_seg

            # Sample reference trajectory at Greville abscissae
            n_sample = max(n + 1, 20)
            tau_sample = np.linspace(0, 1, n_sample)
            t_sample = t_start + tau_sample * dt_seg
            x_sample = interp(t_sample)

            # Build Bernstein basis matrix
            B_mat = np.zeros((n_sample, n + 1))
            for i in range(n + 1):
                B_mat[:, i] = bernstein(n, i, tau_sample)

            # First and last control points are fixed
            cp = np.zeros((n + 1, d))
            if seg == 0:
                cp[0] = x0
            else:
                cp[0] = prev_endpoint

            if seg == self.n_segments - 1:
                cp[n] = xf
            else:
                # Endpoint from reference trajectory
                cp[n] = interp(t_end)

            # Solve for interior control points
            # x_sample ≈ B_mat @ cp
            # Subtract known contributions from endpoints
            rhs = x_sample - np.outer(B_mat[:, 0], cp[0]) - np.outer(B_mat[:, n], cp[n])

            if n > 1:
                B_interior = B_mat[:, 1:n]  # columns for interior CPs
                # Least squares: B_interior @ cp_interior = rhs
                cp_interior, _, _, _ = np.linalg.lstsq(B_interior, rhs, rcond=None)
                cp[1:n] = cp_interior

            # Collect free variables (interior + junction)
            for k in range(1, n):
                free_pts.append(cp[k])
            if seg < self.n_segments - 1:
                free_pts.append(cp[n])

            prev_endpoint = cp[n]

        return np.array(free_pts).ravel()

    def solve(self, x0, xf, t0, tf, x_free_guess=None, method='lm',
              reference_trajectory=None):
        """
        Solve the TPBVP using Bézier collocation.

        Args:
            x0: initial state (boundary condition)
            xf: final state (boundary condition)
            t0, tf: time span
            x_free_guess: initial guess for free control points.
                If None, a ballistic propagation is used as reference.
            method: optimization method ('lm' for Levenberg-Marquardt)
            reference_trajectory: tuple (t_ref, x_ref) for initial guess.
                If provided, Bézier segments are fitted to this trajectory.
                If None and x_free_guess is None, the dynamics are propagated
                from x0 as a ballistic (uncontrolled) trajectory.

        Returns:
            result: optimization result
            segments: list of control point arrays
        """
        from scipy.optimize import least_squares
        from scipy.integrate import solve_ivp

        x0 = np.asarray(x0)
        xf = np.asarray(xf)

        if x_free_guess is None:
            if reference_trajectory is not None:
                t_ref, x_ref = reference_trajectory
            else:
                # Propagate ballistic trajectory as initial guess
                sol = solve_ivp(
                    self.dynamics, [t0, tf], x0,
                    t_eval=np.linspace(t0, tf, 200),
                    method='RK45', rtol=1e-10, atol=1e-10
                )
                t_ref = sol.t
                x_ref = sol.y.T

            x_free_guess = self._guess_from_trajectory(
                t_ref, x_ref, x0, xf, t0, tf
            )

        result = least_squares(
            self._collocation_residual, x_free_guess,
            args=(x0, xf, t0, tf),
            method=method, ftol=1e-12, xtol=1e-12, gtol=1e-12,
            max_nfev=20000
        )

        segments = self._unpack_control_points(result.x, x0, xf)
        return result, segments

    def evaluate_solution(self, segments, t0, tf, n_eval=200):
        """Evaluate the full trajectory from solved segments."""
        dt_seg = (tf - t0) / self.n_segments
        t_all = []
        x_all = []

        for seg_idx, cp in enumerate(segments):
            t_start = t0 + seg_idx * dt_seg
            t_end = t_start + dt_seg
            tau = np.linspace(0, 1, n_eval // self.n_segments)
            t_seg = t_start + tau * dt_seg
            x_seg = bezier_eval(cp, tau)
            t_all.append(t_seg)
            x_all.append(x_seg)

        return np.concatenate(t_all), np.vstack(x_all)
