"""
bezier_segmented.py — Literal segmented Bézier collocation for CR3BP transfers

This module exists to isolate the "literal segmented Bézier" experiment from the
IPOPT/multi-shooting path used in the main comparison. The methodological
distinction matters for the AAE 568 writeup:

    Direct multi-shooting  (ipopt_collocation.CR3BPBezierIPOPT)
        Mesh of nodes; within each segment the trajectory is an ODE integral
        of a piecewise-constant/linear control; continuity is enforced by
        matching node states. Bézier is used only as a *control-grid shape*,
        not as a trajectory parameterization inside each segment.

    Literal segmented Bézier  (THIS MODULE)
        Each segment is a degree-n Bézier polynomial over the *state*
        [r, v]. Dynamics defects are imposed at Gauss–Legendre nodes within
        the segment; continuity is C0 of the state (which implies C1 of
        position at the junction). Control is an explicit decision variable
        at each collocation node. Solved with SLSQP.

Mechanically this is what cr3bp_transfer.CR3BPBezierCollocation already does;
this module just relabels it and adds an N-sweep helper so the analogy-to-CFD-
meshing argument can be tested empirically.

Author: Zhuorui Li (based on shared code by Mustakeen Bari, Zhuorui Li,
Advait Jawaji for AAE 568)
"""

import os
import sys
import time as timer

import numpy as np
from scipy.optimize import minimize

# Make sibling modules importable when this file is run or imported
_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_EM_DIR = os.path.join(_THIS_DIR, '..', 'Earth-Mars')
if _EM_DIR not in sys.path:
    sys.path.insert(0, _EM_DIR)

from cr3bp_planar import MU, cr3bp_jacobi_planar  # noqa: E402
from cr3bp_transfer import CR3BPBezierCollocation  # noqa: E402


# Re-export under a name that matches the storyline vocabulary. The underlying
# implementation is unchanged: literal per-segment Bézier, SLSQP, dynamics
# defects at Gauss–Legendre nodes.
CR3BPSegmentedBezier = CR3BPBezierCollocation


# =============================================================================
# N-sweep helper
# =============================================================================

def run_n_sweep(
    x0,
    xf,
    t0,
    tf,
    N_list=(1, 2, 4, 8),
    bezier_degree=7,
    n_collocation=6,
    mu=MU,
    warm_traj=None,
    max_iter=400,
    ftol=1e-9,
    u_max=None,
    verbose=True,
):
    """
    Run literal segmented Bézier collocation for each N in N_list.

    For each N, a fresh CR3BPSegmentedBezier is built with n_segments=N and
    asked to solve the minimum-energy transfer. The previous N's converged
    trajectory is used as a warm start for the next N (standard h-refinement
    warm-starting, analogous to a CFD mesh refinement cascade).

    Args:
        x0, xf         : boundary states [x, y, vx, vy]
        t0, tf         : time window (nondim)
        N_list         : tuple/list of segment counts to sweep
        bezier_degree  : polynomial degree per segment (kept constant across N)
        n_collocation  : interior Gauss-Legendre nodes per segment
        mu             : CR3BP mass parameter (default = Earth-Moon)
        warm_traj      : (t_ref, x_ref) tuple to warm-start the FIRST N;
                         later N's warm-start from previous solve.
        u_max          : optional scalar path bound ||u|| <= u_max at every
                         collocation node.
        verbose        : print per-level summary

    Returns:
        list of dicts, one per N, each with keys:
            N, success, cost, time_s, nit, max_defect, msg,
            t, r, v, u, u_mag, jacobi, segments, solver (result obj)
    """
    x0 = np.asarray(x0, dtype=float)
    xf = np.asarray(xf, dtype=float)
    if u_max is not None and float(u_max) <= 0.0:
        raise ValueError("u_max must be positive when provided")

    results = []
    prev_traj = warm_traj

    for N in N_list:
        if verbose:
            dof_cp = N * ((bezier_degree - 1) + (1 if N > 1 else 0))  # rough
            n_ctrl = N * n_collocation * 2
            print(f"\n--- N = {N} segments ---")
            print(f"    degree = {bezier_degree}, colloc = {n_collocation}, "
                  f"controls = {n_ctrl}")

        solver = CR3BPSegmentedBezier(
            mu=mu,
            n_segments=N,
            bezier_degree=bezier_degree,
            n_collocation=n_collocation,
        )

        # Warm-start: prefer most recent converged trajectory; fall back to
        # caller-provided one; fall back to built-in ballistic guess.
        if prev_traj is not None:
            t_ref, x_ref = prev_traj
            try:
                z_guess = solver._warm_start_from_trajectory(
                    t_ref, x_ref, x0, xf, t0, tf,
                )
            except Exception as exc:
                if verbose:
                    print(f"    warm start failed ({exc}); using built-in guess")
                z_guess = None
        else:
            z_guess = None

        t_start = timer.perf_counter()
        try:
            # Call SLSQP directly so we can dial the maxiter / ftol knobs
            # per-sweep-level (the class default is maxiter=2000 with ftol=1e-12
            # which is overkill for the larger-N levels).
            if z_guess is None:
                z_guess = solver._initial_guess(x0, xf, t0, tf)

            dt_seg = (tf - t0) / N
            constraints = [{
                'type': 'eq',
                'fun': solver._defects,
                'args': (x0, xf, dt_seg),
            }]
            if u_max is not None:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda z, umax: solver._control_magnitude_margins(z, umax),
                    'args': (float(u_max),),
                })
            result = minimize(
                solver._objective, z_guess,
                args=(x0, xf, dt_seg),
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': ftol, 'disp': False},
            )
            sol_dict = solver._evaluate(result.x, x0, xf, t0, tf)
            sol_dict['cost'] = float(result.fun)

            elapsed = timer.perf_counter() - t_start
            success = bool(result.success)
            msg = getattr(result, 'message', '')
            nit = int(getattr(result, 'nit', 0))

            defects = solver._defects(result.x, x0, xf, dt_seg)
            max_defect = float(np.max(np.abs(defects)))

            # Jacobi drift along the trajectory
            jac = np.array([
                cr3bp_jacobi_planar(
                    np.array([sol_dict['r'][k, 0], sol_dict['r'][k, 1],
                              sol_dict['v'][k, 0], sol_dict['v'][k, 1]]),
                    mu,
                )
                for k in range(len(sol_dict['t']))
            ])
            u_mag = np.linalg.norm(sol_dict['u'], axis=1)

            entry = {
                'N': N,
                'success': success,
                'cost': float(result.fun),
                'time_s': elapsed,
                'nit': nit,
                'max_defect': max_defect,
                'msg': msg,
                't': sol_dict['t'],
                'r': sol_dict['r'],
                'v': sol_dict['v'],
                'u': sol_dict['u'],
                'u_mag': u_mag,
                'jacobi': jac,
                'segments': sol_dict.get('segments', []),
                'solver_result': result,
                'u_max': float(u_max) if u_max is not None else None,
            }

            if success and max_defect < 1e-4:
                # Trajectory is usable as a warm start for the next N
                prev_traj = (
                    sol_dict['t'],
                    np.column_stack([sol_dict['r'], sol_dict['v']]),
                )

            if verbose:
                flag = 'OK' if (success and max_defect < 1e-4) else 'FAIL'
                print(f"    [{flag}] cost = {result.fun:.6f}, "
                      f"max |defect| = {max_defect:.2e}, "
                      f"iters = {nit}, time = {elapsed:.2f}s")
                if not success:
                    print(f"    message: {msg}")

        except Exception as exc:
            elapsed = timer.perf_counter() - t_start
            if verbose:
                print(f"    [EXC] {exc!r}")
            entry = {
                'N': N,
                'success': False,
                'cost': float('nan'),
                'time_s': elapsed,
                'nit': 0,
                'max_defect': float('nan'),
                'msg': repr(exc),
                't': None, 'r': None, 'v': None, 'u': None,
                'u_mag': None, 'jacobi': None, 'segments': [],
                'solver_result': None,
                'u_max': float(u_max) if u_max is not None else None,
            }

        results.append(entry)

    return results


def format_sweep_table(results, header=True):
    """
    Plain-text summary table for the N-sweep.

    Column layout matches what we'll want to paste into the course report:
        N  | converged | iterations | time (s) | cost | max |defect|
    """
    lines = []
    if header:
        lines.append(
            "    N | converged | iters |   time (s) |           cost | "
            " max |defect|"
        )
        lines.append(
            "   ---+-----------+-------+------------+----------------+-"
            "--------------"
        )
    for r in results:
        converged = 'Y' if (r['success'] and
                            r['max_defect'] < 1e-4) else 'N'
        lines.append(
            f"   {r['N']:>2d} |     {converged}     | "
            f"{r['nit']:>5d} | {r['time_s']:>10.3f} | "
            f"{r['cost']:>14.6e} | {r['max_defect']:>14.2e}"
        )
    return '\n'.join(lines)
