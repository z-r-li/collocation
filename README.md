# Bézier Collocation for Cislunar Trajectory Optimization

Course project for **AAE 568 — Applied Optimal Control and Estimation**, Purdue University, Spring 2026.
Authors: **Mustakeen Bari**, **Zhuorui Li**, **Advait Jawaji**.

We implement and compare two numerical methods for solving the minimum-energy
optimal-control problem on three progressively more challenging dynamical
regimes:

1. **Phase 0 — Earth-to-Mars two-body** (planar Keplerian, electric-propulsion analog)
2. **Phase 1 — Planar Earth-Moon CR3BP L1↔L2** (low-thrust cislunar)
3. **Phase 2 — Artemis II Post-TLI / Full-mission ephemeris** (chemical, real flight data)

The two methods compared at each phase are:

- **Indirect single-shooting** of the Pontryagin two-point BVP (`scipy.integrate` + `scipy.optimize.fsolve`).
- **Direct Bézier-curve collocation** transcribed into an NLP and solved with CasADi/IPOPT (interior-point) or SLSQP (segmented).

A 3D LEO-to-NRHO framework (`ThreeD/`) is in place but not exercised end-to-end.

The companion final report (PDF, ≤12 pp) lives outside this repository.

## Repository layout

```
common/               schema, IO, timing utilities used by all phases
Earth-Mars/           Phase 0 solvers, runs, figures
Planer/               Phase 1 solvers, runs, figures
Artemis2/             Phase 2 solvers, runs, figures (Ephem_Full, Post-TLI)
ThreeD/               3D framework + LEO-to-NRHO problem setup
results_summary.json  consolidated solver records (138 entries)
results_table.md/.tex tables generated from the JSON
make_results_table.py regenerates the tables
run_constraint_remediation_cases.py
                      generates bounded-control variants for P0/P1
BOUNDED_RESULTS.md    constrained-formulation findings (umax bounds, saturated-PMP law)
```

## Install

```bash
git clone https://github.com/z-r-li/collocation.git
cd collocation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CasADi pulls in IPOPT and the MUMPS sparse solver via a wheel; no separate
system install is needed on macOS or Linux.

## Reproducing the headline results

Each phase has a `run_phaseN.py` that runs both methods, appends records to
`results_summary.json`, and writes figures alongside the script. The runs are
pure instrumentation over the underlying solvers — no retuning, no random seeds
beyond what the scripts pass explicitly.

```bash
# Phase 0: Earth-Mars two-body
python Earth-Mars/run_phase0.py

# Phase 1: planar CR3BP L1↔L2 (Lyapunov-to-Lyapunov)
python Planer/run_phase1.py

# Phase 2: Artemis II
python Artemis2/run_phase2.py

# Bounded-control (umax) variants for P0 and P1
python run_constraint_remediation_cases.py

# Refresh the consolidated tables
python make_results_table.py
```

Auxiliary sweep / mesh-refinement scripts (`run_phaseN_nsweep*.py`,
`run_phaseN_psweep.py`) regenerate the figures in `Earth-Mars/phase0_nsweep_figures/`
and `Planer/`.

## Headline findings

- On the two-body Phase 0 problem, both methods converge to the same cost
  (`J ≈ 0.0149`) to machine tolerance. Shooting is faster (sub-second) but
  collocation matches its accuracy with substantially less initial-guess
  sensitivity.
- On the planar CR3BP Phase 1 problem, indirect shooting requires near-exact
  initial costates and frequently diverges in the strongly hyperbolic cislunar
  regime; direct Bézier collocation reliably converges from a straight-line
  state-space initial guess.
- The constrained variants (P0 with `u_max = 1e-5 km/s²`, P1 with `u_max =
  8e-7 km/s²`) demonstrate the saturated-PMP law
  `u* = sat(-½ R⁻¹ Bᵀλ)`. P0's bound is non-binding (cost matches
  unconstrained); P1's bound binds and produces visible saturation arcs.
  Detail in [`BOUNDED_RESULTS.md`](BOUNDED_RESULTS.md).

## External data

- **JPL 9:2 L₂ southern NRHO**: initial conditions taken from the [JPL Periodic
  Orbits Database](https://ssd-api.jpl.nasa.gov/doc/periodic_orbits.html);
  hard-coded in `ThreeD/leo_to_nrho_cr3bp.py`.
- **Artemis II flight ephemeris** (Post-ICPS-Sep through EI): Orion Mission
  Manager OEM file, cited and consumed by `Artemis2/Ephem_Full/artemis2_ephemeris.py`.
  The `.asc` file is not redistributed in this repo; obtain it from NASA's
  public Artemis II reference trajectory release.

## Contributing / contact

This is a one-shot academic submission, not an actively maintained library.
Issues and PRs are welcome but will be triaged best-effort.

## License

[MIT](LICENSE).
