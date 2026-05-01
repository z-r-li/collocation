# Bounded-control results

Generated 2026-04-29 from `run_constraint_remediation_cases.py`.
This document records the constrained min-energy formulation
$$\min_{u} \tfrac{1}{2}\int_{t_0}^{t_f} u^\top R\,u\;\mathrm{d}t \quad\text{s.t.}\quad \dot x = f(x) + B u,\;\; x(t_0)=x_0,\;\; x(t_f)=x_f,\;\; \|u\| \le u_{\max}.$$

`B` is the control-influence matrix `[0; I]` (4×2 in Phase 0/1, 6×3 in Phase 2/3D),
and `R = I` throughout. The unconstrained PMP law
$u^* = -\tfrac{1}{2} R^{-1} B^\top \lambda$
generalizes under the inequality bound to the saturation form
$u^* = \mathrm{sat}_{u_{\max}}\!\left(-\tfrac{1}{2} R^{-1} B^\top \lambda\right).$

## Constraint structure

| Class       | Members                                   | In our setup |
|-------------|-------------------------------------------|--------------|
| Equality    | dynamics defects + boundary conditions    | always present |
| Inequality  | $\|u\| \le u_{\max}$                      | optional; results below show with-and-without |

The unconstrained variant (no inequality) is the soft-regularized form. The
quadratic cost on `u` does not bound the optimal control magnitude — the
optimal $u$ can spike as large as the costate $\lambda_v$ demands. Adding
$\|u\| \le u_{\max}$ activates the saturated-PMP law and produces realistic
bang-off-bang or partially-saturated arcs depending on the bound.

## Phase 0 — Earth-Mars two-body (electric-propulsion analog)

Bound: `u_max = 1.0e-5 km/s²` (envelope for ion / Hall thrusters; cf. Dawn,
Hayabusa-2, Psyche).

| Method                              | Degree | N  | Cost J         | Residual    | Converged |
|-------------------------------------|:------:|:--:|---------------:|------------:|:---------:|
| Saturated PMP shooting              | —      | —  | 0.0149004478   | 3.05e-14    | yes       |
| Global Bézier + IPOPT bounded       | 7      | 16 | 0.0149004452   | 2.78e-15    | yes       |

Unconstrained peak control is `5.20e-7 km/s²`, well below the envelope, so the
bound is **non-binding**. Costs match across methods to ~7 digits — confirming
that the bounded direct-collocation machinery preserves the unconstrained
benchmark when the inequality is inactive.

Figure: `Earth-Mars/p0_control_envelope.png`.

## Phase 1 — Planar CR3BP L1↔L2 (electric-propulsion analog)

Bound: `u_max = 8.0e-7 km/s²` (cislunar low-thrust envelope; cf. SMART-1,
Lunar Gateway PPE), tuned so the bound binds.

| Method                                      | Degree | N  | Cost J         | Residual    | Converged |
|----------------------------------------------|:------:|:--:|---------------:|------------:|:---------:|
| Global Bézier + IPOPT bounded                | 7      | 16 | 0.0871808069   | 2.39e-08    | no¹       |
| Segmented Bézier + SLSQP bounded             | 7      | 8  | 0.0881961491   | 8.74e-10    | yes       |
| Segmented Bézier + SLSQP bounded             | 7      | 16 | 0.0881385271   | 9.49e-11    | yes       |
| Saturated PMP shooting                       | —      | —  | 0.1153461640   | 3.26e-03    | no²       |

¹ IPOPT bounded reaches the envelope (peak `‖u‖ = 8.001e-7`) but did not flag
solver success; trajectory is structurally correct but should not be cited as
a converged benchmark.
² Single-shooting did not converge to the constrained endpoint conditions;
report as an indirect-method fragility finding under the saturated PMP law,
not as a Phase 1 benchmark.

The two SLSQP-segmented records (N=8, N=16) are the clean constrained-formulation
demonstration: both converge with sub-`10⁻⁹` residual and sit on the same
bounded-control cost plateau (~0.088). Saturation arcs are visible in the
control magnitude trace.

Figure: `Planer/p1_saturation_arcs.png`.

## Cross-phase comparison

`constrained_cost_comparison.png` overlays the converged constrained costs
across both phases, showing the bound-binding contrast: P0's `J ≈ 0.0149`
matches the unconstrained reference (bound non-binding), while P1's
`J ≈ 0.0881` is elevated above the unconstrained `J* = 0.04306` because the
P1 bound is active.

## Phase 2 — Artemis II

Phase 2 was always formulated with `u_max` (Orion ESM chemical envelope) and
needs no remediation. See `Artemis2/RETUNING_AUDIT.md` and `Artemis2/phase2_bezier_validation.md`
for solver records.

## Reproducing

```
python run_constraint_remediation_cases.py
```

Records are appended to `results_summary.json`; figures are written next to
their phase folders. `make_results_table.py` regenerates `results_table.md`
and `results_table.tex` from the JSON.
