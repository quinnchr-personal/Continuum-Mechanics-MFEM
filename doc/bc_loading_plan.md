# Implementation Plan — General Boundary Conditions and Load Scheduling (Gates L1–L5)

**Audience:** the implementing agent. Follow the gate order, meet each gate's acceptance
criteria before moving on, commit once per completed gate. This extends the framework of
`doc/hyperelasticity_implementation_plan.md`; its design decisions (total Lagrangian,
reference mesh never moves, thin `apps/`, `myapps/` untouched) still hold.

**Status (2026-09-06):** all five gates implemented; L1+L2 in one commit (the load
bookkeeping they share was written once, as `physics/loads.{hpp,cpp}`), L3+L4+L5 in a
second. Deviations from the text below: the pressure types and the follower kernel were
written with L1 (the shared `LoadSet` carries them) and tested in L4; the cylinder test
uses a curved second-order Gmsh mesh (`apps/mesh/annulus.geo`) so refinement follows the
arcs; in addition to L5, equibiaxial and pure-shear cube inputs for every incompressible
model were added under `apps/input/finite_elasticity/homogeneous/` and the comparison script derives the
free axis from the constrained faces. After review the `value`/`gradient` data form
(and the plain `body_force` list) was removed from the YAML schema at the user's request:
every entry is an `expression`, the default schedule is the ramp unless the expression
mentions `t`, and all inputs were rewritten (`["x", "-0.5*y"]` for an affine stretch).
`AffineVectorCoefficient` remains as a programmatic helper for the tests.

**Goal:** replace the single proportional load path (one `load_factor` scaling every
Dirichlet value, traction and body force together, in `load_steps` equal increments) with a
scheme where

1. each boundary condition and the body force has its own **schedule** in a pseudo-time
   `t in [0, 1]`,
2. Dirichlet conditions may constrain a **subset of components** (symmetry planes, rollers),
3. tractions may be given as a **normal pressure**, dead (reference area) or **follower**
   (current area),
4. the **step sequence** is user-controlled and **bisects** on Newton failure,
5. Dirichlet data, tractions and the body force may be given as **expressions**
   f(x, y, z, t) in the reference coordinates and the pseudo-time.

Every existing input must run unchanged and reproduce its frozen numbers.

---

## 1. Current state (verified 2026-09-06 — re-verify before starting)

- `BoundaryCondition` in `src/base/config.hpp:84` holds `attr`, `attr_names`, `value`,
  `gradient`; `BCConfig` has `dirichlet` and `traction` lists. Parsed in
  `ParseBCList`, `src/base/config.cpp:111`. `body_force` is a plain vector on `AppConfig`.
- `MakeBCCoefficient` (`src/base/coefficients.hpp:40`) returns a
  `VectorConstantCoefficient` or an `AffineVectorCoefficient` (value + gradient X).
- Both physics classes (`src/physics/solid_mechanics_tl.cpp`,
  `src/physics/mixed_solid_mechanics_tl.cpp`) keep `std::vector<BCEntry>{marker, coef}` for
  Dirichlet and traction. `Finalize()` unions the Dirichlet markers into one essential marker
  (all components), assembles **one** dead-load true vector `load_true_` from body force plus
  all tractions. `ApplyDirichlet` projects every Dirichlet coefficient into one grid function
  and writes `load_factor_ * g_true` on the essential true dofs. `Mult` subtracts
  `load_factor_ * load_true_` and zeroes the essential rows.
- `SolveQuasiStatic` (`src/solvers/quasi_static.cpp`) loops `step = 1..load_steps`,
  `lambda = step / load_steps`, calls `SetLoadFactor`, `ApplyDirichlet`, `DampedNewtonSolve`;
  stops at the first non-converged step.
- `QuasiStaticProblem` interface (`src/solvers/quasi_static.hpp`): `SetLoadFactor`,
  `LoadFactor`, `ApplyDirichlet`, `Comm`. `SolidProblem` adds `AddDirichlet(attrs, coef)`,
  `AddTraction(attrs, coef)`, `SetBodyForce(coef)`; tests use these programmatically
  (`tests/test_homogeneous.cpp:332`, `tests/test_mixed.cpp:227`, `tests/test_solid_mms.cpp:295`).
- Frozen regressions that must not move: `tests/test_benchmarks.cpp` (Cook corner values),
  `apps/homogeneous_compare.py` over `apps/input/finite_elasticity/homogeneous/*.yaml`, all `tests/test_*`.
- Follower loads are an explicit TODO seam at `src/physics/solid_mechanics_tl.cpp:171`.

## 2. Design decisions — do not re-litigate

1. **Pseudo-time replaces the load factor.** The stepper advances `t` from 0 to 1. Each load
   `i` carries a scalar schedule `s_i(t)`; its contribution is `s_i(t) * data_i`. The default
   schedule is `ramp` with `s(t) = t`, which reproduces today's behaviour exactly.
   `SetLoadFactor(lambda)` stays as the interface name (tests call it) but its argument is
   now `t`; document that in the header.
2. **Schedules are piecewise linear.** Kinds: `ramp` (0 to 1 over `[t0, t1]`, default
   `[0, 1]`, held at 1 after `t1`, 0 before `t0`), `constant` (1 for all `t > 0`, applied in
   full at the first step), `table` (`t: [...]`, `s: [...]`, linear interpolation, clamped).
   This covers prestress-then-stretch, load-then-unload, and staged loading; anything
   beyond that is written as an expression (decision 9).
3. **Component masks, not component BCs.** A Dirichlet entry gets an optional
   `components: [x, z]` (names or 0-based indices). The coefficient stays a full vector
   (value + gradient X); only the listed components become essential. Unlisted components
   on that boundary are free. Rollers and symmetry planes are then
   `{ attr: [bottom], value: [0, 0, 0], components: [y] }`.
4. **Overlaps are resolved by projection order, and conflicts are an error.** When two
   Dirichlet entries mark the same true dof and component, the later entry wins at
   projection, exactly as today. Add a check in `Finalize()`: if two entries share an
   attribute and a component, warn once on rank 0 (not an error; corner nodes of a clamped
   face and a roller face legitimately overlap and usually agree). Data disagreement is not
   detected; document that.
5. **Tractions get a `type`.** `vector` (default, today's nominal traction per reference
   area), `pressure` (scalar `value: p`; dead load `T = -p N` on the reference normal),
   `follower_pressure` (scalar; `T = -p J F^{-T} N` per current area, nonlinear, enters the
   nonlinear form with its own tangent). `pressure` and `follower_pressure` take a scalar
   `value` and no `gradient`.
6. **One assembled load vector per traction entry and one for the body force.** `Mult`
   subtracts `sum_i s_i(t) L_i`. Assembly cost is negligible; entries with fixed data are
   assembled once in `Finalize()`, time-dependent ones per step (decision 10).
7. **Step control lives in the stepper, not the physics.** `solver.load_steps: N` keeps its
   meaning (N equal increments of `t`). New alternative `solver.steps:` a list of
   `{ to: t_end, n: count }` segments. New `solver.substep: { on_failure: true,
   max_bisections: 4, min_dt: 1e-4 }`. On a non-converged Newton solve the stepper restores
   `x` from the last converged state, halves the increment, retries; after a successful
   step at reduced size it returns to the planned breakpoints (no automatic growth beyond
   the planned grid). Every attempt is recorded in `QuasiStaticReport`.
8. **Programmatic API stays additive.** `AddDirichlet(attrs, coef)` and
   `AddTraction(attrs, coef)` keep their signatures and mean "all components, ramp
   schedule, vector type". Add overloads taking a `BCOptions{components, schedule, type}`.
   Existing tests compile unchanged.
9. **Expressions are parsed by a hand-written evaluator, no dependency.** Grammar:
   numbers, variables `x y z t` (`z` is 0 in 2D), `pi`, binary `+ - * / ^`, unary minus,
   parentheses, functions `sin cos tan exp log sqrt abs min max` and `if(cond, a, b)` with
   comparisons `< <= > >= == !=` (evaluating to 1 or 0). Parsed once at config time to a
   flat postfix program, evaluated per point with no allocation. An entry is either
   `value` (+ `gradient`) or `expression`, never both; an expression entry defaults to the
   `constant` schedule (t enters through the function). If a schedule is given as well,
   the two multiply.
10. **Time enters through `SetTime`.** `SetLoadFactor(t)` calls `SetTime(t)` on every
    coefficient (MFEM `Coefficient::SetTime`), so a programmatic `VectorFunctionCoefficient`
    with a `(X, t, v)` callback works from C++ without further plumbing. Entries whose
    coefficient is time-dependent (expression mentioning `t`, or any programmatic
    coefficient, which cannot be inspected) have their dead-load vector reassembled at every
    step; affine and constant entries keep the one-time assembly.

## 3. YAML surface (final form; every key optional unless stated)

```yaml
bcs:
  dirichlet:
    - { attr: [left],  value: [0, 0, 0] }                                 # clamp, ramp (unchanged)
    - { attr: [bottom], value: [0, 0, 0], components: [y] }               # roller
    - { attr: [right], value: [0, 0, 0], gradient: [[1,0,0],[0,-0.29,0],[0,0,-0.29]],
        schedule: { type: ramp, from: 0.5, to: 1.0 } }                    # stretch in the second half
    - { attr: [top], expression: ["0.1*t*sin(pi*x)", "0", "0.05*t^2*(1-y)"] }   # f(X, t), constant schedule
  traction:
    - { attr: [top], value: [0, 0, -1.0], schedule: { type: ramp, to: 0.5 } }   # dead, applied first
    - { attr: [inner], type: follower_pressure, value: 0.2,
        schedule: { type: table, t: [0, 0.5, 1], s: [0, 0, 1] } }
    - { attr: [right], expression: ["0", "if(t < 0.5, 2*t, 1)*cos(pi*z)", "0"] }
    - { attr: [outer], type: pressure, expression: "0.3*t*(1 + 0.1*z)" }          # scalar for pressures
body_force: { value: [0, 0, -9.81], schedule: { type: constant } }   # plain list still accepted
body_force: { expression: ["0", "0", "-9.81*t"] }                    # alternative form
solver:
  load_steps: 8                          # or:
  steps: [ { to: 0.5, n: 2 }, { to: 1.0, n: 10 } ]
  substep: { on_failure: true, max_bisections: 4, min_dt: 1e-4 }
```

`components` accepts `x|y|z` or `0|1|2`. `schedule` on `body_force` requires the map form.
`expression` is a list of `dim` strings (a single string for the pressure types); a parse
error reports the key path, the offending string and the column.
Unknown keys, wrong types and inconsistent sizes raise `ConfigError` with the full key path,
as now.

## 4. Gates

### L1 — Schedules and step control (no new physics)

- `src/base/config.hpp`: `struct Schedule { kind; t0, t1; table_t, table_s; double Eval(double t) const; }`.
  Add `Schedule schedule` to `BoundaryCondition`; new `struct BodyForceConfig { value, schedule }`
  on `AppConfig` (parser accepts the old plain list). `SolverConfig` gains
  `std::vector<double> breakpoints` (built from `load_steps` or `steps`) and `SubstepConfig`.
- Physics (both classes): per-entry schedule stored in `BCEntry`; per-entry load vectors
  `std::vector<mfem::Vector> load_true_` plus `body_load_true_`; `SetLoadFactor(t)` stores
  `t`; `ApplyDirichlet` projects each entry into its own grid function scaled by `s_i(t)`
  (project, then overwrite the essential dofs of the running `g` with that entry's values so
  order semantics of decision 4 hold); `Mult` subtracts the scheduled sum.
  `ExternalLoad()` returns the scheduled sum at the current `t` (used by tests).
- Stepper: iterate breakpoints; bisection on failure per decision 7. `LoadStepReport`
  gains `t_begin`, `t_end`, `attempts`. Print `load step k: t = a -> b`.
- Tests (`tests/test_base.cpp` for parsing and `Schedule::Eval`; new `tests/test_loading.cpp`):
  - `ramp` default reproduces `test_solid_mms` and `test_mixed` bit-for-bit at `t = 0.7` / `0.6`.
  - Two-stage path on the 2D Cook membrane: traction ramped over `[0, 0.5]` then held, a
    second traction ramped over `[0.5, 1]`; final state equals the single-stage solve with
    both tractions to 1e-10 (path independence of hyperelasticity).
  - Load-unload `table` schedule `s = [0, 1, 0]` returns `|u| < 1e-9` at `t = 1`.
  - Bisection: an input whose planned step fails Newton (Cook with `load_steps: 1` and
    traction 5x) converges with `substep.on_failure: true` and the report lists the retries.
- **Acceptance:** all existing tests and `apps/homogeneous_compare.py` pass with unchanged
  numbers; README section "Boundary conditions and loading" written.

### L2 — Expression-valued data f(x, y, z, t)

- `src/base/expression.{hpp,cpp}`: `class Expression { static Expression Parse(const
  std::string &); double Eval(double x, double y, double z, double t) const; bool UsesTime()
  const; }` per decision 9. Parse errors throw `ConfigError` with the column. Evaluation is
  a loop over a `std::vector<Op>` with a small fixed-size stack; no virtual calls, no
  allocation, so it could later run inside `mfem::forall` if needed.
- `src/base/coefficients.hpp`: `ExpressionVectorCoefficient : mfem::VectorCoefficient`
  (one `Expression` per component; `Eval` transforms the integration point to reference
  coordinates `X` and evaluates at `GetTime()`), and `ExpressionCoefficient : mfem::Coefficient`
  for the scalar pressure types. `MakeBCCoefficient` dispatches on `expression`.
- Config: `BoundaryCondition` gains `std::vector<std::string> expression` (scalar entries
  stored as a one-element list); `value` and `expression` are mutually exclusive
  (`ConfigError` if both or neither). Same on `BodyForceConfig`. The default schedule of
  an expression entry is `constant`.
- Physics (both classes): `SetLoadFactor(t)` calls `SetTime(t)` on all coefficients.
  `BCEntry` gains `bool time_dependent` (from `UsesTime()` for expressions, `true` for
  programmatic coefficients); `Mult` reassembles the load vectors of time-dependent
  traction and body-force entries when `t` changed since the last assembly. Dirichlet
  projection already happens every step.
- Tests (`tests/test_base.cpp` for the parser, `tests/test_loading.cpp` for the rest):
  - Parser: precedence and associativity (`2^3^2`, `-x^2`, `1-2-3`), functions, `if` with
    every comparison, `pi`, error cases (unbalanced parentheses, unknown identifier, empty
    string, trailing garbage) each naming the column; round trip against a table of 20
    (expression, x, y, z, t, expected) rows evaluated by hand.
  - Programmatic `VectorFunctionCoefficient` with a `(X, t, v)` callback and constant
    schedule reproduces `SetLoadFactor(0.7)` in `test_solid_mms` when the callback scales by
    t, to round-off.
  - YAML MMS: the 2D plane-strain MMS of `test_solid_mms` written as a YAML input with the
    exact displacement as a Dirichlet `expression` and the MMS body force as a body-force
    `expression`; run through `LoadConfig` + `MakeSolidProblem` on two refinements and check
    the same convergence rate as the C++ test. This is the one test that exercises the whole
    path from string to residual.
  - Reassembly: a traction `expression` linear in t with 4 steps gives the same final state
    as the equivalent `value` + ramp to 1e-12; an expression not mentioning `t` is assembled
    once (assert via a counter exposed in a test hook or by timing-free inspection of
    `time_dependent`).
- **Acceptance:** above tests; README documents the grammar in full (it is the user-facing
  contract); `homogeneous_compare.py` refuses inputs with `expression` with a clear message
  (it has no closed form to compare against) rather than misreading them.

### L3 — Component-wise Dirichlet

- Config: `components` parsed to a `std::vector<int>` (validated against `dim`, no
  duplicates); empty means all.
- Physics: essential true dofs are the union over entries of
  `fes.GetEssentialTrueDofs(marker, list, component)` for each listed component. Keep a
  per-entry `ess_tdofs_i` so `ApplyDirichlet` writes only that entry's components. The
  `ParNonlinearForm` / `ParBlockNonlinearForm` essential list becomes the union. Overlap
  warning per decision 4.
- Tests (`tests/test_loading.cpp`):
  - Uniaxial cube by symmetry: eighth-symmetry model of `uniaxial_neo_hookean.yaml`
    (rollers on `X = 0`, `Y = 0`, `Z = 0`, affine stretch on `X = 1`, other faces free)
    reproduces the closed-form stress and `p` to 1e-8 (compare against the existing full-cube
    run through `homogeneous_compare.py`; add the input under `apps/input/finite_elasticity/homogeneous/`
    with a `symmetry_` prefix and teach the script's `load_case` to read `components`).
  - Rank-check: on a mesh with all faces rollered (each face one normal component) the
    tangent at `u = 0` has no null space (CG converges on a zero-traction problem, `|u| = 0`).
  - 2D plane-strain MMS from `test_solid_mms` with `components: [x]` on two edges and
    the exact vector on the others still converges at the expected rate (the free component
    is a natural BC with the MMS traction supplied through `AddTraction`).
- **Acceptance:** above tests; README documents `components`; `mpirun -np 4` on the
  symmetry cube gives the same probe values as serial.

### L4 — Pressure tractions, dead and follower

- Config: traction `type` and scalar `value` for the two pressure kinds.
- Dead pressure: `VectorBoundaryFluxLFIntegrator`-style term `-p N` on the reference
  normal, assembled into that entry's load vector like a vector traction.
- Follower pressure: new boundary kernel in `src/kernels/` computing on a boundary face
  `T = -p J F^{-T} N` with `F` from the adjacent element's displacement; residual
  contribution `-int T . delta u dA_R`, tangent by dual numbers over the face displacement
  dofs (same pattern as the plane-stress implicit derivative). Installed through a
  `ParNonlinearForm::AddBdrFaceIntegrator` (displacement) and the displacement block of the
  `ParBlockNonlinearForm` (mixed); scaled by `s_i(t)` inside the integrator (the integrator
  holds a pointer to the entry's current scale). The follower tangent is non-symmetric;
  `cg_amg` must refuse it with a `ConfigError` naming the entry.
- Tests:
  - Tangent consistency: finite-difference check of the follower integrator on a single
    distorted hex and quad (`tests/test_materials.cpp` style), relative error < 1e-6.
  - Dead vs follower at small load: on the Cook membrane with `p = 1e-3` the two agree to
    O(p^2) in the corner displacement.
  - Inflation of a thick-walled incompressible neo-Hookean cylinder under internal follower
    pressure (2D plane strain, quarter model with `components` rollers from L3): compare
    inner-radius stretch with the closed-form relation of Rivlin (Ogden, *Non-linear Elastic
    Deformations*, Sec. 5.3.1). Add `apps/input/finite_elasticity/cylinder_inflation.yaml` and the Gmsh `.geo`.
- **Acceptance:** above tests; `doc/solid_mechanics_forms.tex` gains the follower-pressure
  weak form and tangent; the TODO seam at `solid_mechanics_tl.cpp:171` is removed.

### L5 — Convenience and consistency (small, optional)

- `output.probes` print at every load step (currently only at the end) when
  `output.probe_every_step: true`, so schedules can be checked from the log.
- `apps/homogeneous_compare.py`: accept `schedule` (final state only) and `components`.
- `output.probe_every_step` prints `t` on each probe line, so f(X, t) data can be checked
  against the log.
- Per-step ParaView time stamp = `t` (already wired through `on_step`; confirm).

## 5. Out of scope (state in README as "not supported")

- Expressions depending on the solution (`u`, stresses) — those are follower-type loads and
  need tangents; only the follower pressure of L4 is provided.
- Expressions referencing the current (deformed) position; `x y z` are reference coordinates.
- Point loads and nodal constraints (use a small physical group instead).
- Multi-point constraints, periodic BCs, contact.
- Time-dependent (dynamic) loading; `t` is a pseudo-time only.
- Automatic step growth after a bisected step (only recovery to the planned grid).

## 6. Order of work and effort

| Gate | Touches | Estimate |
|------|---------|----------|
| L1 | config, both physics, stepper, tests, README | 1 day |
| L2 | expression parser + coefficients, config, both physics, tests, README | 1 day |
| L3 | config, both physics, tests, one new input, compare script | 0.5 day |
| L4 | new kernel, both physics, config, tests, doc tex, new input + mesh | 1.5 days |
| L5 | app, script | 0.25 day |

L1 first (it introduces the per-entry schedules and load vectors that L2 builds on). L2 and
L3 are independent of each other and of L4; L4 depends on L3 for the cylinder test and on
L2 only if its pressure is given as an expression. Commit per gate with the message prefix
`L1:`, `L2:`, ....
