# M4 continuation stop report

The required escalation sequence was exhausted on the converted Exasim mesh.
No analytic-mesh run or M4 comparison gate was attempted after this stop.

## Attempt 1: prescribed four stages, PTC off

- Stage 1, `0.06*tanh(30d)`: converged in 9 Newton updates to
  `2.520326607699143e-09` (5 damped steps, minimum alpha `0.125`).
- Stage 2, `0.04*tanh(30d)`: the first full step reduced the residual from
  `0.1855352922744269` to `0.01130326815554138`, but minimum pressure became
  `-0.008986924625426027`; the next PETSc factorization failed.
- Archive: `continuation_history_attempt1_no_ptc.csv` and
  `continuation_summary_attempt1_no_ptc.csv`.

## Attempt 2: stage-2 PTC-SER, CFL_0 = 0.5

- Stage 1 reproduced the attempt-1 result.
- Stage 2 accepted residuals `0.1855352922744269`,
  `0.2223496078178542`, and `4.40028001901268`. Both accepted steps used the
  required line-search floor `alpha=0.0625`; density and pressure became
  negative, and the next PETSc factorization failed.
- Archive: `continuation_history_attempt2_stage2_ptc.csv` and
  `continuation_summary_attempt2_stage2_ptc.csv`.

## Attempt 3: YAML-only intermediate AV amplitude

The only schedule change was insertion of `0.05*tanh(30d)` between the
original `0.06` and `0.04` stages. The original `0.04` stage retained
PTC-SER with `CFL_0=0.5`.

- `0.06*tanh(30d)`: converged in 9 updates to
  `2.520326607699143e-09`.
- inserted `0.05*tanh(30d)`: converged without PTC in 3 updates to
  `1.053460075396740e-08`.
- original `0.04*tanh(30d)`: accepted residuals
  `0.1020744655918100`, `0.1153774817082495`, and
  `0.6014047442845043`, then hard-aborted at
  `1344797.248912052` as required by the Exasim abort rule.
- Archive: `continuation_history_attempt3_intermediate.csv` and
  `continuation_summary_attempt3_intermediate.csv`.

## Spatial localization of the catastrophic trial

- Minimum density: `-2189976.451880073`, element 5, at
  `(-0.08808920973046749, -2.550638809115203)`.
- Minimum pressure: `-2191588.301926927`, element 5, at
  `(-0.01805005636775617, -2.572022411443559)`.
- Largest element residual entry: `449449.0209627129`, element 622,
  equation 0, local dof 22, at
  `(-0.02425253861300005, 3.455838059309011)`.
- Largest trace residual entry: `2.493467611622236`, face 1238,
  equation 2, local dof 2, at
  `(-0.3958376340559311, 2.626211106263340)`.

The field failure is therefore concentrated along the two outflow-side
regions (`x` near zero), rather than nucleating at the cylinder wall or the
stagnation shock.

---

# Resumed diagnostic (2026-07-26): root cause identified

The blocked diagnostic was resumed by reproducing Exasim's own continuation
in a scratch copy of the reference run and comparing the first failing
Newton step across the two codes. Artifacts are archived under
`m4_diagnostic/` (logs, exported per-stage states, first-step debug dumps,
the floored model source, and the stage IC/AV writer).

## Scratch reproduction method

The run-dir text-mode app (`builtinmodelID=7`, `gendatain=0`) re-preprocesses
from `grid.bin`/`xdg.bin`/`udg.bin`/`vdg.bin` on every invocation, so the
continuation is driven by overwriting `udg.bin`/`vdg.bin` per stage
(`m4_diagnostic/write_stage.py`, IC verbatim from `pdeapp_ns.m:62-67`,
wall distance `r-1`, verified against `vdg.bin` at `2.3e-17`) and running
`exasimapp` at `mpiprocs=1` (identity element/face ordering). Pipeline
control: the baseline stage-4 restart reproduced its known signature
(4 updates, `0.343278 -> 3.72611e-7`, all `alpha=1`).

## The first failing Newton step, both codes

- Exasim floor-free stage 1 (`0.06*tanh(30d)` from the damped-freestream
  IC): 9 updates, 5 damped, minimum alpha `0.125`, final `2.69989e-9` —
  the same damping signature as our stage 1 (`2.520e-9`).
- Exasim floor-free stage 2 (`0.04*tanh(30d)` from its stage-1 state):
  **aborts on the first Newton step**. Initial residual `2.08434`
  (the per-stage restart re-initializes `uhat` to the one-sided trace,
  which alone produces O(1) residuals at a converged state); the
  `alpha=1` trial residual is NaN (`>1e6`/NaN abort rule) before any
  line search. Debug dump (`first_step_dump/`): trial minimum pressure
  `-0.008993212119465887` at `(-1.7515, 0.0000)` — the stagnation-line
  shock foot, 239/16275 nodes negative, minimum density `0.96508`.
- Our driver from the *same* exported stage-1 state and trace-init
  (`logs/e4_ours_s1_av004.log`): initial residual `2.7271` (side-selection
  convention differs in the trace init), `alpha=1` step accepted
  (`-> 0.014997`), trial minimum pressure `-0.0089912257614254351` — the
  same undershoot to 0.02%, and the next factorization fails. Identical
  root cause, surfacing one iteration later because our residual
  evaluation samples no negative-pressure quadrature point on the trial.
- Cross-code warm-restart parity at `(S1, av=0.06, trace re-init)`:
  Exasim `2.97915 -> 2 full steps -> 6.45e-7`; ours
  `3.8899 -> 2 full steps -> 7.54e-7`.

The attempt-1 first accepted step (min pressure `-0.008986924625426027`
from the carried-over converged trace) is the same structural undershoot:
the 0.06 -> 0.04 AV drop forces a first Newton step that drives the shock
foot ~0.009 below zero pressure regardless of the trace initialization.
The outflow-side localization recorded above for attempt 3 was a
downstream consequence of two accepted bad steps, not the nucleation site.

## Root cause

The exported model (run-dir `pdemodel.txt` = builtin model 7 = our
`ns_physics.hpp`) omits the density/pressure regularization present in
`pdemodel_ns.m`, which the original MATLAB-frontend continuation ran:

```
lmax(x,a) = x*(atan(a*x)/pi + 1/2) - atan(a)/pi + 1/2      (a = 1e3)
r <- rmin + lmax(r - rmin, a)                               (rmin = 1e-2)
p <- pmin + lmax(p - pmin, a)                               (pmin = 1e-3)
```

plus gradient sensors `dr`/`dp` scaling `(rx, ry)` and `(px, py)`.
The floors are inactive to ~1e-9 at every converged stage state — hence
invisible to all M3 parity gates (the floored stage-4 control matches the
floor-free control to 6 digits) — but load-bearing during the stage-2
transient, where the trial state must be evaluable at `p < 0` (without
floors the Sutherland `sqrt(T^3)` is NaN).

## Proof by reconstruction

The floored model was regenerated with `text2code` from
`m4_diagnostic/pdemodel_floored.txt` (scratch copy only; no repo code
changed) and the full 4-stage continuation rerun from the damped-freestream
IC with **no PTC and no extra stages**: 9/4/4/4 Newton updates, final
residuals `2.70e-9 / 2.74e-8 / 2.95e-7 / 3.73e-7`, every post-stage-1
alpha `1`. Its stage-3 state reproduces the original `udg.bin` to relative
L2 `[3.0e-6, 5.4e-6, 1.2e-5, 5.9e-6]` (components 0-3), and its stage-4
solve reproduces the baseline signature to 6 digits. The floored stage-2
first step accepts the same `p ~ -0.009` trial (`2.08434 -> 0.0213946`)
and converges 3 updates later.

## Consequence for M4

Attempts 1-3 failed for a model-level reason, not an implementation
defect: the exported floor-free physics cannot pass the `0.04` stage under
any globalization that must evaluate the flux at the first trial state
(PTC and inserted stages merely reshaped the same negative-pressure
failure). Unblocking M4-as-specified requires adopting the
`pdemodel_ns.m` regularized flux (with its analytic Jacobians) in
`ns_physics.hpp` — the physics the original continuation actually ran.
M3 parity would be unaffected (floors inactive at the reference state).
That model change is left as a decision for the plan owner.

---

# Resolution (2026-07-26): M4 unblocked and passing

The regularized flux was adopted (plan-owner approved) and M4 now passes
in full. Implementation:

- `ns_physics.hpp` gained `FluxRegularizedGenerated` /
  `FluxRegularizedJacobianGenerated`, mechanical transcriptions of the
  text2code-regenerated floored model
  (`m4_diagnostic/my_model_floored.hpp`), selected by
  `NSParams::regularized` / YAML `physics.regularization: floors`.
  Off by default — every M3 deck and recorded gate value is untouched
  (`test_m3_reference` reproduces `3.2999123144061386e-07` bit-for-bit).
- `test_physics` gained three gates: transcription parity vs the
  generated floored model over 1000 admissible + 1000 floor-straddling
  states (**exactly 0**), central FD of the floored Jacobian
  (`4.93e-7` vs a 1e-5 gate sized to the hinge curvature), and a
  properties check pinning floored/unfloored agreement at admissible
  states (`4.2e-7`) plus the M4 failure state itself (floored finite,
  unfloored NaN — the regression capture of this whole report).
- The continuation loop now re-initializes the trace
  (`InitializeTraceFromInterior`) at each stage start, mirroring
  Exasim's per-stage restart semantics.
- Both continuation decks were reverted to the prescribed
  `pdeapp_ns.m` schedule verbatim: four stages, **no PTC, no inserted
  stage**, with `regularization: floors`.

Results (converted Exasim mesh, `continuation_summary.csv`):

| Stage | av | Updates | Damped | Final residual |
|---|---|---:|---:|---:|
| 1 | `0.06*tanh(30d)` | 9 | 5 (min alpha 0.125) | `2.5212e-09` |
| 2 | `0.04*tanh(30d)` | 4 | 0 | `2.3049e-08` |
| 3 | `0.025*tanh(30d)` | 4 | 0 | `2.4918e-07` |
| 4 | `0.025*tanh(5d)` | 4 | 0 | `3.2999400258199408e-07` |

The 9/4/4/4 signature matches the floored Exasim reference runs; the
stage-2 histories track step for step (ours
`2.727 -> 0.0199 -> 0.0153 -> 1.3e-4 -> 2.3e-8`, Exasim's
`2.084 -> 0.0214 -> 0.0166 -> 1.6e-4 -> 2.7e-8`; the initial residuals
differ only by the trace-init side convention), and both accept the same
first-step trial with minimum pressure `~ -0.0090`. M4 gates
(`m4_comparison_report.md`): wall Fint max rel diff `3.43e-06` and Cp
`2.92e-06` vs M3 (gate 1e-2), shock-standoff sample identical (rel 0).
The analytic-mesh continuation passes with the same 9/4/4/4 signature
and wall parity `1.6e-07` against the converted-mesh run. The full
`make test` suite (M1-M4) is green.

One build-system fix rode along: the makefile's per-object dependency
lines were missing `ns_physics.hpp` for `hdg_ns_operator.o` /
`hdg_ns_driver.o` / `wall_post.o`, so changing `NSParams` without a
clean rebuild produced an ABI-mismatch heap crash. The dependencies are
now complete.
