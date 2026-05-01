#!/usr/bin/env python3
"""
ephem_boundaries.py — LEO and NRHO Boundary States in EME2000

Builds the two endpoint states for the ephemeris LEO→NRHO transfer,
expressed in Earth-centered EME2000 (J2000) inertial coordinates
with km / km·s⁻¹ units.

  • LEO departure: classical Keplerian orbit (a, e=0, i, Ω, ω, ν).
    Free phasing via Ω (RAAN) and ν (true anomaly) — these get
    exposed as optimizer decision variables in leo_to_nrho_ephem.py.

  • NRHO arrival:  pull the 9:2 Southern NRHO state from cr3bp_3d.py
    at orbit phase φ ∈ [0, 1] (propagated forward from the reference
    initial state for φ · T_nrho), then map rotating-nondim → EME2000
    at the target epoch using instantaneous Moon geometry.

Author: Zhuorui, AAE 568 Spring 2026
"""

import os, sys
import numpy as np

# Local imports from the ThreeD/ folder
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ephem_dynamics import (
    MU_EARTH, R_EARTH,
    rot_nondim_to_eme2000,
)
from cr3bp_3d import nrho_state, nrho_period, propagate


# =============================================================================
# LEO DEPARTURE
# =============================================================================

def leo_departure_state_eme2000(
    altitude_km=185.0,
    inclination_deg=28.5,
    raan_deg=0.0,
    arg_perigee_deg=0.0,
    true_anomaly_deg=0.0,
):
    """
    LEO state in Earth-centered EME2000 from classical orbital elements
    (circular orbit; ω is kept as an argument for generality but has no
    dynamical effect when e = 0).

    Args:
        altitude_km:        altitude above Earth equatorial radius [km]
        inclination_deg:    inclination from equatorial plane     [deg]
        raan_deg:           right ascension of ascending node      [deg]
        arg_perigee_deg:    argument of perigee                   [deg]
        true_anomaly_deg:   true anomaly at departure              [deg]

    Returns:
        (6,) [r (km), v (km/s)]
    """
    a = R_EARTH + altitude_km
    e = 0.0
    i      = np.deg2rad(inclination_deg)
    Omega  = np.deg2rad(raan_deg)
    w      = np.deg2rad(arg_perigee_deg)
    nu     = np.deg2rad(true_anomaly_deg)

    # Perifocal position/velocity (circular)
    p     = a * (1 - e**2)
    r_mag = p / (1 + e * np.cos(nu))
    r_pf  = r_mag * np.array([np.cos(nu),        np.sin(nu),         0.0])
    v_mag = np.sqrt(MU_EARTH / p)
    v_pf  = v_mag   * np.array([-np.sin(nu),     e + np.cos(nu),     0.0])

    # Rotation: perifocal → EME2000  (R_3(-Ω) R_1(-i) R_3(-ω))
    cO, sO = np.cos(Omega), np.sin(Omega)
    cw, sw = np.cos(w),     np.sin(w)
    ci, si = np.cos(i),     np.sin(i)
    R = np.array([
        [ cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci,   sO*si],
        [ sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci,  -cO*si],
        [ sw*si,             cw*si,              ci   ],
    ])

    r_eci = R @ r_pf
    v_eci = R @ v_pf
    return np.concatenate([r_eci, v_eci])


# =============================================================================
# NRHO ARRIVAL
# =============================================================================

def nrho_arrival_state_eme2000(epoch_tf, phase_frac=0.0):
    """
    9:2 L2 Southern NRHO state in Earth-centered EME2000 at `epoch_tf`.

    Procedure:
      1. Take the reference NRHO state from cr3bp_3d.NRHO_9_2.
      2. Propagate forward by (phase_frac · T_nrho) in the CR3BP rotating
         frame to get the state at the desired orbital phase.
      3. Map rotating-nondim → dimensional Earth-centered EME2000 using
         the instantaneous Moon geometry at epoch_tf.

    Args:
        epoch_tf:    astropy Time — target arrival epoch
        phase_frac:  orbit phase ∈ [0, 1]; 0 → reference state (near apolune)

    Returns:
        (6,) [r (km), v (km/s)] — the NRHO target state in EME2000

    Caveats:
        The CR3BP NRHO is not exactly periodic in the ephemeris model; this
        state is an approximation (initial target). Leave NRHO insertion
        phase as an optimizer free variable so IPOPT can refine it.
    """
    if not (0.0 <= phase_frac < 1.0):
        phase_frac = float(phase_frac) % 1.0

    T = nrho_period()
    x_rot_nondim = nrho_state()

    if phase_frac > 1e-12:
        t_prop = phase_frac * T
        sol = propagate(x_rot_nondim, (0.0, t_prop), dense_output=False)
        x_rot_nondim = sol.y[:, -1]

    return rot_nondim_to_eme2000(x_rot_nondim, epoch_tf)


# =============================================================================
# QUICK SELF-TEST
# =============================================================================

if __name__ == '__main__':
    from astropy.time import Time

    print("ephem_boundaries.py self-test")

    x_leo = leo_departure_state_eme2000(
        altitude_km=185.0, inclination_deg=28.5, raan_deg=0.0, true_anomaly_deg=0.0
    )
    print(f"  LEO:  |r| = {np.linalg.norm(x_leo[:3]):.1f} km,  "
          f"|v| = {np.linalg.norm(x_leo[3:]):.4f} km/s "
          f"(nominal ~7.79 at 185 km)")

    epoch_tf = Time('2027-12-08T00:00:00', scale='utc')
    x_nrho_0 = nrho_arrival_state_eme2000(epoch_tf, phase_frac=0.0)
    x_nrho_h = nrho_arrival_state_eme2000(epoch_tf, phase_frac=0.5)
    print(f"  NRHO @ φ=0.0:  |r| = {np.linalg.norm(x_nrho_0[:3]):.1f} km")
    print(f"  NRHO @ φ=0.5:  |r| = {np.linalg.norm(x_nrho_h[:3]):.1f} km")
    print("  (9:2 NRHO should swing between apolune ~70,000 km and "
          "perilune ~3,000 km from Moon)")
