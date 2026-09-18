# Implementation Plan — Inertia in the Solid Weak Forms: Implicit Elastodynamics (Gates DY1–DY5, optional DY6–DY7)

**Audience:** the implementing agent. Follow the gate order, meet each gate's acceptance
criteria before moving on, commit once per completed gate (on `main`, no attribution
trailers). This extends the framework of `doc/hyperelasticity_implementation_plan.md`,
`doc/bc_loading_plan.md` and `doc/linear_elasticity_plan.md`; their design decisions (reference
mesh never moves, thin `apps/`, `myapps/` untouched, materials are stateless value types, no
per-exercise drivers) still hold.

**Status (2026-09-18):** DY1 to DY4 complete.
DY1 complete: `tests/test_dynamics.cpp` (177 checks, 5.5 s, in `make check`); every existing
test line and the logs of `cook.yaml`, `bar_linear.yaml` and `cook_linear_incompressible.yaml`
are identical to those of the commit before. Measured: total mass `1.M.1 = sum rho_r V_r` to
2e-16 on perturbed quad / tri / hex / tet meshes, p = 1, 2, two densities (the reference
volumes need a rule that is exact for the trilinear Jacobian; `Mesh::GetElementVolume` is off
by 1e-4 on perturbed hexes). Free fall with two densities, `linear_elastic` and `neo_hookean`:
`a_0 = g` to 4e-14, `u = g t^2/2` at every node to 1.4e-13 over 50 steps. Temporal order
without spatial error (`newmark`, `hht` 0.1, `generalized_alpha` 0.8): 1.999-2.000 with
tractions, 2.05 -> 2.005 with prescribed motion, for `u` and `v`. Trapezoidal rule, linear free
vibration: energy constant to 3e-14 over 500 steps; at `w dt >> 1` the energy is annihilated
for `rho_inf = 0` (1e-32 after 10 steps), falls with slope -1.331 against `2 ln rho_inf =
-1.386` for 0.5, and is kept to 12 digits for 1. Step traction plus moving support: energy
balance to 1.6e-14 of the peak energy, global momentum balance to 5e-13. Neo-Hookean at
`max |u| = 0.75`: Newton order 1.90 with `K + c_M M`, self-convergence ratios 3.989 and 3.982;
space-time manufactured solution at the spatial rate 3.015. Linear problem, 200 steps: one
assembly of `K`, one of `K + c_M M`, one AMG setup, not a digit different from 200 of each; a
step that fails once is halved and the run equals the one with the half steps planned;
`t_final = 1e-3` and `1e3` end on `t_final` exactly.
Deviations from the text below. (a) The order study of item 3 runs on 2 x 2 elements with
steps from 2e-3 down: the error at a fixed time carries the free vibrations that the
truncation error excites, each with the phase of its numerical frequency, and the ratios are
only clean once `dt` resolves every mode of the mesh (on 4 x 4 elements from `dt = 0.04` they
scatter between 1.1 and 3.2 while the errors still fall as `dt^2`). For the same reason item
7 takes the order in `dt` of the nonlinear path from the self-convergence on such a mesh, and
uses the manufactured solution under joint refinement (the dynamic error equals the static
spatial one, 6.7e-6 on 8 x 8, and halving `dt` alone changes it by 2e-4). (b) Item 4's
"`E_n` non-increasing" is not a property of generalized-alpha: it dissipates in its own norm,
and the physical energy rose by 1.4e-5 `E_0` between two steps of the `rho_inf = 0.8` run; the
check is that it never exceeds `E_0` and is lower at the end. (c) Equal steps `t_final k / n`
differ in the last digits of `t_{k+1} - t_k`, which would give a new `c_M`, a new sum and a new
AMG setup at every step of a linear problem: the decorator keeps the previous increment when
the new one agrees with it to 1e-12. (d) The np 2 / 4 comparison needs inputs and moves to DY3.
DY2 complete: the `dynamics` block, schedules in physical time, `output.every` / `energy`, the
fields `velocity` and `acceleration`, the app branch; 85 new parser checks in `test_base` (every
error of Section 3) and a YAML-driven run in `test_dynamics` (189 checks). First input,
`apps/input/dynamics/bar_free_vibration.yaml`: the tip displacement after 20 steps of 0.02
equals `A cos(w~ t)` with the trapezoidal rule's own dispersion `tan(w~ dt/2) = w dt/2` to
1e-9 A (against the continuum frequency it is off by the predicted 3e-5 A); energy balance
1e-16 over the 2000 steps of the input; serial and np 4 agree to 12 digits. `make check` green,
the three reference logs unchanged. Deviations: `output.every` is honoured by both analyses
(default 1, so nothing changes); a `dt` that does not divide `t_final` is shortened to
`t_final / n` instead of being an error; `substep.min_dt` defaults to 1e-3 of the smallest
planned step under dynamics; the full static residual that `AcceptStep` evaluates (support
forces, `S_n`) is skipped when neither the external work is tracked (`output.energy`) nor the
scheme interpolates forces, and `Reactions()` then forms it on demand: a linear run is
residual-bound (16 ms per step on the 3075-dof bar, three residual evaluations against one
AMG-CG solve), so this is a third of the time.
DY3 complete: six inputs under `apps/input/dynamics/`, `tests/test_dynamic_verification.cpp`
(49 checks, 1 min 50 s, in `make test`; np 2 and 4 against serial to 5e-14 on 130 steps of the
two bar inputs), `apps/dynamics_compare.py` (`make dynamics`; with `--check` in `make test`, it
reads the executable's own per-step lines). Measured. Bar, first mode, five periods: tip within
6.1e-4 A of `A cos(w_1 t)`, energy to 7e-14; at `dt = 0.2` the period grows by 8.135e-3 against
`(w dt)^2/12 = 8.225e-3` and 8.171e-3 from the scheme's dispersion relation. Step load (new
mesh `bar.msh`, 100 elements): tip peak 1.9895 and mean 1.0001 of the static deflection, front
at mid-span at 0.5001, wall reaction 1.0000 p A on average. Cantilever: `w_1 = 0.101321`,
0.17 percent below Euler-Bernoulli. Manufactured solutions: rates 3.006, 3.002 (2D) and 3.012,
3.022 (3D) in `h`, self-convergence ratios 4.033, 4.010 in `dt`. Neo-Hookean block: ratios
3.921, 3.981; energy error of the trapezoidal rule falling by 4.005, 4.001; with
`rho_inf = 0.8` the energy never exceeds `E_0`.
Deviations. (a) The step load's "overshoot record" is not what separates the schemes: the first
overshoot of the wall reaction behind a front is made by the mesh and is 28 percent for both;
what `rho_inf = 0.8` removes is the ringing that follows (rms 0.18 percent of `2 p A` late in
the plateau against 1.62 percent), and that is what is checked. (b) The orders in `dt` are
taken from self-convergence with steps that resolve the mesh (2D manufactured solution on the
unrefined mesh with 800 steps and more: on the refined one the ratios are 3.74-3.78; block from
`dt = 5e-4` down: one level coarser gives 3.65), for the reason of DY1's deviation (a). (c) The
block input runs a cycle and a half (`t_final` 1.6) rather than the 0.16 of the convergence
study, so that its plot shows a vibration.
DY4 complete: the decorator on the block unknown needed no change; what was added is the
initial pressure of `u_0` at finite kappa (one solve with `K_pp`), the warning for a scheme
without dissipation, and the solver extension the measurement asked for. Outer FGMRES
iterations of the first Newton solve, quarter annulus, incompressible neo-Hookean (the
quasi-static inflation on the same mesh needs 18):

    dt                  0.125   0.02   0.0125   0.002
    refine 1, before      28     33      38      43
    refine 2, before      33     47      59      81
    refine 1, after       14      8       8       6
    refine 2, after       16      9       7       6

"After" is the Cahouet-Chabard sum `S~^{-1} = [scaled M_p]^{-1} + [Bt D^{-1} B / c_M + C]^{-1}`
in `SaddlePointSolver`, with `D` the diagonal of `M` scaled to the total mass and the second
inverse by CG + BoomerAMG; it is formed from the Jacobian's own blocks, which gives it the right
boundary conditions without a pressure Laplacian being assembled. It exists only when the
decorator hands the solver `c_M` (`SetInertia`): `test_mixed` and the mixed app logs are
unchanged to the last digit. Measured (`test_dynamics`, now 221 checks in 6.6 s): Herrmann
problem whose `u` (quadratic) and `p` (linear) the Taylor-Hood pair holds exactly, at
`nu = 0.4999` and incompressible: order 2.00 / 1.99 and 1.95 / 1.99 for `u`, 1.97 / 2.11 and
2.06 / 2.01 for `p`; the pressure mode: two runs that differ in `p_0` alone differ at every later
step by exactly `rho_inf` times the step before (0.6000 seven times for `rho_inf = 0.6`), which
is decision 13(a) to four digits; `p_0 = kappa div u_0` to 6e-14. Verification case
`knowles_tube_oscillation.yaml` (in `test_dynamic_verification`, now 54 checks in 2 min, and in
`dynamics_compare.py`): inner displacement peak 0.280754 against 0.280749 of Knowles' equation,
largest difference over two periods 2.3e-3 of it, period 4.2431 against 4.2428, and the pressure
unknown at mid-wall, which the inertia of the inner half carries, within 2.0e-3 of its largest
value after the start-up; np 2 and 4 against serial to 1e-15 (displacement, kinetic energy) and
2e-12 (pressure).
Deviations. (a) The order of `p` can only be seen where its temporal error stands above the
noise of the solves: the pressure balances `M a`, and `a = (u - u*) / (beta dt^2)` multiplies the
error of a solve by `c_M = O(1/dt^2)`. With `newton.rtol` 1e-10 that noise is 4e-5 on the unit
square and hides the order from 400 steps on; the study uses 50-400 steps and tight solves.
The same holds for a user: in a mixed dynamic analysis the pressure output carries the solver
tolerance times `c_M`. (b) "Mixed and displacement formulations agree to 1e-6" is dropped: they
are different discretisations of the same problem, and the exact-in-space order study says more.
(c) The pressure check of the tube is made at mid-wall, where inertia carries it, not at the
wall, where the free surface ties it to the kinematics.

**Goal:** an optional top-level `dynamics:` block turns the quasi-static problem into

    int rho_R u_tt . w dV_R  +  R(u; w)  =  0,        u(0) = u_0,  u_t(0) = v_0

for every existing material (finite and small strain), both formulations, 2D and 3D, with
every existing feature (schedules, components, expressions, pressures and follower pressures,
regions, reactions, probes, quadrature outputs, parallel) working in physical time. Without
the block nothing changes: every existing input, test number and log line is bit-identical.

**The one-paragraph answer to "what does adding the dynamic term amount to?"** In total
Lagrangian form the inertial term is integrated over the fixed reference mesh with the
reference density, so its matrix `M` is *constant*: assembled once, for finite and small
strain alike, independent of the material. With the displacement as the unknown of an implicit
one-step scheme, `a_{n+1}` is an affine function of `u_{n+1}`, and the equation of a time step is

    G(u) = S(u, t_{n+1}) + c_M M (u - u*) + h_n = 0,        dG/du = K(u) + c_M M,

where `S` is exactly the residual `SolidProblem::Mult` computes today and `K` its Jacobian:
the dynamic term is a linear spring `c_M M` plus a known history load. Inertia therefore
enters as a **decorator over `SolidProblem` that implements the stepper's interface**, not as
a new kernel or physics class; Newton, the line search, both linear solvers, Dirichlet data,
bisection, reactions and outputs are reused. What needs care is everything that silently
assumes a pseudo-time in [0, 1] (schedule validation and defaults, the stepper's time
tolerance), the density (one number today, even with regions), the loads at `t = 0` and the
initial acceleration, and, in the mixed formulation, the fact that the pressure carries no
inertia (a differential-algebraic system).

---

## 1. Current state (verified 2026-09-18 — re-verify before starting)

- Documentation already states the term and its absence: `doc/theory_manual.tex:351-354`
  ("With inertia the residual gains ... dynamics is not implemented"), `:147` ("inertia is
  absent"), `:188` ("Not implemented: dynamics"); `README.md:339` ("true dynamics"), `:438`.
- Static residual and Jacobian: `SolidMechanicsTL::Mult` (`src/physics/solid_mechanics_tl.cpp:138-145`,
  internal minus `loads_.ExternalLoad()`, essential rows zeroed), `GetGradient` (`:168-175`,
  eliminated `HypreParMatrix`, operator stamp, assembled once when `IsLinear()`); the mixed
  pair at `src/physics/mixed_solid_mechanics_tl.cpp:162-171` and `:194-201` (2x2
  `BlockOperator` of `HypreParMatrix`; the displacement block leads the block vector).
  The stamp is reachable only through `GradientAssemblies()` on the concrete classes
  (`solid_mechanics_tl.hpp:78`, `mixed_solid_mechanics_tl.hpp:61`).
- Stepper: `QuasiStaticProblem` (`src/solvers/quasi_static.hpp:18-33`: `SetLoadFactor`,
  `ApplyDirichlet`, `IsLinear`); `SolveQuasiStatic` (`quasi_static.cpp:75-147`) compares times
  with an **absolute** tolerance (`while (t < target - 1e-14)`, `:97`), prints
  `t = %.6f` (`:102`), bisects on failure and restores `x_last`.
- Time today is a pseudo-time: `LoadSet::time_ = 1.0` initially (`src/physics/loads.hpp:134`),
  `SetTime` evaluates schedules and calls `SetTime` on every coefficient
  (`loads.cpp:219-239`); expressions are `f(x, y, z, t)` (`src/base/expression.hpp:4-8`; no
  `sinh`/`cosh`, use `exp`). `ParseSchedule` rejects ramps and tables outside [0, 1]
  (`src/base/config.cpp:336`, `:351`); the default schedule is the ramp `s = t` unless the
  expression mentions `t` (`:113-120`); `Schedule::Constant` is `time > 0 ? 1 : 0` (`:284`);
  `substep.min_dt` must lie in (0, 1] (`:794`).
- Density: `material.rho0` (default 1) is read for the base **and for regions**
  (`config.cpp:609`, inside `ReadMaterialKeys`), but both physics keep only the base value
  (`solid_mechanics_tl.cpp:27`, `mixed_solid_mechanics_tl.cpp:20`) and hand it to
  `LoadSet::SetBodyForce`, which forms `ScalarVectorProductCoefficient(rho0, b)`
  (`loads.cpp:104-120`). A region's `rho0` is accepted and ignored. No input under
  `apps/input/` sets one, and none combines regions with a body force, so fixing this cannot
  move an existing number.
- Solvers: `LinearSolver::SetOperator` keeps its AMG hierarchy for the same operator and an
  unchanged stamp (`src/solvers/linear_solver.cpp:55-70`); `SaddlePointSolver` likewise
  (`saddle_point_solver.cpp:58-117`), with the Schur approximation
  `M_p (kappa + mu) / (kappa (mu + gamma))` (`:32-35`), a pure pressure-mass scaling. Newton's
  test is `|R| <= atol` or `<= rtol |R_0|` (`src/solvers/newton.cpp:68`), with the floor
  acceptance for linear problems (`:112-119`, `:142-147`).
- Output: `ParseOutputConfig` has a closed list of field names (`config.cpp:886-895`); the
  app writes every accepted step (`apps/solid_mechanics.cpp:110-121`) and already passes the
  step's `t` to `ParaViewWriter::Save` as the time.
- Tests drive YAML inputs through `LoadConfig` + `MakeSolidProblem`
  (`tests/test_linear_verification.cpp:67`, `:127`); `CHECK_TESTS` is `makefile:148`, the
  `test` target `:173`, the np 2/4 reference check `:183-185`.
- MFEM 4.8.1 (`~/MFEM/mfem`): `linalg/ode.hpp:704-918` has `NewmarkSolver`,
  `GeneralizedAlpha2Solver`, `HHTAlphaSolver`, `WBZAlphaSolver`; they advance in the
  **acceleration** form through `SecondOrderTimeDependentOperator::ImplicitSolve` and the
  generalized-alpha one solves at the alpha-level state (`linalg/ode.cpp:1229-1275`).
  `HypreParMatrix::EliminateBC(ess, DiagonalPolicy)` (`linalg/hypre.hpp:904`) and
  `mfem::Add(alpha, A, beta, B)` (`:1012`) exist.

## 2. Design decisions — do not re-litigate

1. **Opt-in by a `dynamics:` block.** Absent: the quasi-static path, untouched. Present:
   `t` is physical time. No flag inside the kernels, no change to `TotalLagrangianIntegrator`
   or `MixedTotalLagrangianIntegrator`.
2. **Constant, consistent mass matrix, assembled once.** `ParBilinearForm` on the displacement
   space with `VectorMassIntegrator(rho_R)`; `rho_R` is a `PWConstCoefficient` by element
   attribute built from `material.rho0` and the regions (the table convention of
   `MakeMaterialTable`). The **same coefficient** goes to the body force, which fixes the
   ignored region density of Section 1. Two copies are kept: `M` (nothing eliminated: residual,
   energies, reactions, and the coupling of a prescribed boundary acceleration into the free
   rows) and `M_e` (essential rows and columns zeroed, `DIAG_ZERO`: the Jacobian, so the
   unit diagonal of the eliminated `K` stays 1). In 2D the mass is per unit reference
   thickness, also under plane stress. Lumping exists only in the optional explicit gate.
3. **A decorator, not an integrator.** New `src/physics/dynamic_solid_problem.{hpp,cpp}`:
   `DynamicSolidProblem : QuasiStaticProblem` wraps any `SolidProblem&`. Rejected: an inertia
   integrator inside the nonlinear form (it would recompute a constant matrix at every
   residual, and `M` is needed as a global operator anyway for `a_0`, the energies and the
   reactions); MFEM's `SecondOrderODESolver` family (acceleration form, see Section 1:
   prescribed displacements, the Newton floor, reactions and the constant-Jacobian reuse are
   all written for the displacement as the unknown); a first-order `(u, v)` system with
   BE / DIRK (twice the unknowns, and BE is first order and strongly dissipative).
4. **One scheme family: generalized-alpha in displacement form with interpolated forces**
   (the form of Abaqus/Standard's HHT operator). With `S_n = S(u_n, t_n)`:

       predictors   u* = u_n + dt v_n + dt^2 (1/2 - beta) a_n,     v* = v_n + dt (1 - gamma) a_n
       unknown u    a(u) = (u - u*) / (beta dt^2),                 v(u) = v* + gamma dt a(u)
       balance      M [(1 - am) a(u) + am a_n] + (1 - af) S(u, t_{n+1}) + af S_n = 0

       newmark            am = af = 0; beta, gamma given (default 1/4, 1/2: trapezoidal rule)
       hht                am = 0, af = alpha in [0, 1/3], beta = (1 + alpha)^2/4, gamma = 1/2 + alpha
       generalized_alpha  am = (2r - 1)/(r + 1), af = r/(r + 1), beta = (1 - am + af)^2/4,
                          gamma = 1/2 - am + af,   r = rho_inf in [0, 1]

   All are second order; `rho_inf` is the spectral radius at infinite frequency (1: no
   dissipation, 0: asymptotic annihilation). Interpolating the forces rather than the state
   means `S` is only ever evaluated at `(u_{n+1}, t_{n+1})`: `LoadSet`, the Dirichlet data and
   the follower loads keep one time. Default `scheme: newmark`; the README recommends
   `generalized_alpha` with `rho_inf` 0.8-0.9 for finite-strain runs (the trapezoidal rule is
   not unconditionally stable for nonlinear problems).
5. **The step equation is divided by `1 - af`,** so the static operator enters with unit weight:

       G(u) = S(u, t_{n+1}) + c_M M (u - u*) + h_n,     dG/du = K(u) + c_M M_e
       c_M  = (1 - am) / ((1 - af) beta dt^2),          h_n = (am M a_n + af S_n) / (1 - af)

   Essential rows of `G` are zeroed like those of `S`. Consequences: `IsLinear()` forwards
   (a linear problem stays linear, the Newton floor logic applies unchanged); in the mixed
   formulation `M` acts on the displacement block only, `h_n` has a zero pressure block, the
   constraint row is `S_p(u_{n+1}, p_{n+1})` unweighted, and the off-diagonal blocks of the
   Jacobian are the static ones, so `SaddlePointSolver` sees the structure it knows.
6. **Initial state.** `u_0`, `v_0` from expressions `f(x, y, z)` (default zero). The Dirichlet
   data at `t = 0` overwrites `u_0` on the essential dofs, with a warning when that changes it
   by more than round-off. `a_0` solves `M a_0 = -S(u_0, 0)` on the free dofs (CG + Jacobi
   smoother, as the pressure-mass solve of the saddle-point solver) and is zero on the
   essential dofs unless given programmatically (`SetInitialAcceleration`, for MMS). **Loads
   at `t = 0` are right limits:** under dynamics a `constant` schedule is 1 for `t >= 0`, so a
   step load is on from the first instant and enters `a_0` (suggested mechanism:
   `Schedule::Eval(t, right_limit)`, `LoadSet` passing `true` once switched to physical time).
7. **Physical time replaces the pseudo-time.** `dynamics.dt` or `dynamics.steps` (segments
   `{to, n}` in physical time ending at `t_final`) give the breakpoints. Schedules are
   validated against `[0, t_final]`. An entry **without** a `schedule:` key is `constant`
   under dynamics (its data is the expression, with `t` the physical time): the quasi-static
   default ramp over [0, 1] has no meaning there. Because that is a default that depends on
   the analysis, the app echoes the resolved time dependence of every load and Dirichlet entry
   in its header. `solver.load_steps`, `solver.steps` and `solver.predictor` are configuration
   errors next to `dynamics:`; `solver.substep` stays (a failed step is halved; `min_dt` is
   then in physical time and only has to be positive); `newton` and `linear` are unchanged.
8. **The stepper is shared, its time tolerance becomes relative.** The loop of
   `SolveQuasiStatic` moves into an internal function with three parameters: the time
   tolerance, the step label, and an accept hook. `SolveQuasiStatic` calls it with today's
   values (absolute 1e-14, "load step", no hook) and stays bit-identical; the new
   `SolveDynamic` uses a tolerance of 1e-9 of the planned increment (an absolute 1e-14 is
   below the spacing of doubles once `t > 100`, and a missed breakpoint would produce a
   degenerate step with `c_M ~ 1/dt^2 ~ 1e26`), prints times with `%.9e`, and calls
   `DynamicSolidProblem::AcceptStep(x)` before the user callback. `SetLoadFactor(t)` of the
   decorator begins a step: `dt = t - t_n`, the constants, `u*`, `v*`, `h_n`, and the inner
   problem's time. Bisection needs nothing else, because the history `(u_n, v_n, a_n, S_n)`
   changes only in `AcceptStep`, which also evaluates `S_n` (one residual evaluation).
9. **Newton starts from the last converged state with the new Dirichlet data,** as today.
   No extrapolating start (`u_n + dt v_n`) in this plan: it makes `|R_0|` small, and
   `rtol |R_0|` then sits below the round-off floor of the residual — the trap that made
   `solver.predictor: tangent` an error for linear problems. Whoever adds one later must first
   give Newton a reference residual that does not depend on the start (for instance
   `max(|R_0|, |M a_n|)`).
10. **Dirichlet data** is prescribed on `u_{n+1}`; `v` and `a` on those dofs follow from the
    update formulas like everywhere else. Reactions are the full balance at `t_{n+1}`,
    `S_full(u_{n+1}, t_{n+1}) + M a_{n+1}` on the essential rows, so a support force includes
    the inertia it carries. `SolidProblem` therefore exposes the unconstrained static residual
    (the first half of both `Reactions` implementations, extracted) and its gradient stamp.
11. **A linear problem assembles and sets up once per run.** `K + c_M M_e` is constant while
    `dt` is: the decorator keeps the sum, owns its own operator stamp (bumped when the inner
    stamp or `c_M` changes) and re-points the solver made by `MakeLinearSolver` to it. For a
    nonlinear problem the sum is formed at every Newton iteration (`mfem::Add`; small next to
    the dual-number assembly of `K`). `cg_amg` stays valid whenever it is valid statically.
12. **Diagnostics.** Kinetic energy `v.Mv/2`, internal energy (`InternalEnergy`), external
    work accumulated by the trapezoidal rule over dead loads *and* moving supports
    (`(r_n + r_{n+1}).(u_{n+1} - u_n)/2` on the essential dofs), and their balance; nodal
    fields `velocity` and `acceleration` on the displacement space. For a linear problem the
    trapezoidal rule satisfies the balance exactly, which makes it an implementation test, not
    only an output.
13. **Mixed formulation: same decorator, two documented properties.** (a) The pressure force
    is interpolated with the rest of `S_u`, so an error in `p_n` propagates with the factor
    `-af/(1 - af) = -rho_inf`: a sign-alternating mode that decays only for `rho_inf < 1`.
    `formulation: mixed` with a non-dissipative scheme (`newmark` with `gamma = 1/2`,
    `rho_inf = 1`, `alpha = 0`) prints a warning that names this. (b) `p_0` is zero unless the
    volumetric law gives it (finite kappa: from `u_0`); the start-up error in the pressure
    decays with that same factor. No separate consistent initialisation of the index-3 system.
14. **Quasi-static numbers must not move.** No arithmetic on the static path is touched; the
    density coefficient replaces a scalar by a piecewise constant with the same value. Proof:
    `make check`, the frozen Cook and cantilever values, `test_parallel`, `test_homogeneous`.

## 3. YAML surface

```yaml
dynamics:                          # absent: quasi-static, exactly as today
  t_final: 2.0e-2
  dt: 1.0e-5                       # or steps: [ { to: 5.0e-3, n: 1000 }, { to: 2.0e-2, n: 300 } ]
  scheme: generalized_alpha        # newmark (default) | hht | generalized_alpha
  rho_inf: 0.8                     # generalized_alpha, in [0, 1] | hht: alpha in [0, 1/3]
                                   # | newmark: beta (0.25), gamma (0.5)
  initial:
    displacement: ["0", "0"]       # expressions f(x, y, z), default zero
    velocity: ["0", "1.5*x"]
material: { model: neo_hookean, E: 2.0e11, nu: 0.3, rho0: 7800.0,
            regions: [ { attr: [inclusion], rho0: 2700.0 } ] }   # density by region now honoured
bcs:
  traction: [ { attr: [right], expression: ["1e6*sin(200*t)", "0"] },        # t: physical time
              { attr: [top], type: pressure, expression: ["5e5"],
                schedule: { type: table, t: [0, 1e-3, 2e-3], s: [0, 1, 0] } } ]
output:
  fields: [displacement, velocity, acceleration, cauchy_stress]
  every: 10                        # ParaView stride; probe / reaction / energy lines stay per step
  energy: true                     # per step: kinetic, internal, external work, balance
```

Errors (each a `ConfigError` with the key path): `dynamics` without `t_final`, or with both or
neither of `dt` / `steps`; `steps` not increasing or not ending at `t_final`; a key that does
not belong to the chosen scheme (`rho_inf` with `newmark`, ...); `rho_inf` outside [0, 1],
`alpha` outside [0, 1/3], `beta <= 0`, `gamma < 1/2` (a warning, not an error, when
`2 beta < gamma`: conditionally stable); `solver.load_steps` / `steps` / `predictor` next to
`dynamics`; a schedule outside `[0, t_final]`; `initial.*` with the wrong number of
components or mentioning `t`; `velocity` / `acceleration` in `output.fields` or
`output.energy` without `dynamics`.

## 4. Gates

### DY1 — Mass, density by region, the step operator, the stepper (library and fast tests)

Files: new `src/physics/dynamic_solid_problem.{hpp,cpp}` (decisions 2-6, 10-12; a
programmatic `DynamicsConfig` struct in `config.hpp`, parsed in DY2); `solid_problem.hpp` and
both physics (`FullResidual`, `GradientStamp`, the density coefficient by attribute handed to
`LoadSet::SetBodyForce`, which takes a `Coefficient&` instead of a `double`);
`loads.{hpp,cpp}` (right-limit evaluation); `quasi_static.{hpp,cpp}` (decision 8,
`SolveDynamic`). New `tests/test_dynamics.cpp` in `CHECK_TESTS` (target: under 20 s).

1. Mass. `1.M.1 / dim = sum_r rho_r V_r` to 1e-13 on perturbed quad / tri / hex / tet meshes,
   p = 1, 2, two regions of different density; `M` symmetric; `M_e` has zero essential rows.
2. Free-fall patch test. No Dirichlet data, body force `g`, two regions of different density,
   perturbed mesh, `linear_elastic` and `neo_hookean`, trapezoidal rule: `u = g t^2/2` at
   every node and `a = g` to 1e-12 for 50 steps, stresses zero. The rule is exact for constant
   acceleration, so this holds on any mesh with any material, and it fails at once if the mass
   and the body force disagree on the density. (`K` is singular here; `K + c_M M` is not.)
3. Temporal order without spatial error. `linear_elastic`, affine mesh, p = 2,
   `u = sin(w t) U(X)` with `U` quadratic (in the space), analytic body force, homogeneous
   and time-dependent Dirichlet data (`a_0 = 0` for both): errors of `u` and `v` at `t_final`
   over four halvings of `dt` fall with rates 1.95-2.05 for `newmark`, `hht` (0.1) and
   `generalized_alpha` (0.8), with no spatial floor above 1e-9. A load taken at the wrong time
   level shows here as first order.
4. Energy and dissipation, linear free vibration from an initial displacement. Trapezoidal
   rule, 500 steps: `|E_n - E_0| <= 1e-10 E_0`. `rho_inf = 0.8`: `E_n` non-increasing.
   With `w_min dt >= 1e3` (every mode in the high-frequency limit): `rho_inf = 0` leaves less
   than 1e-6 `E_0` within 10 steps; `rho_inf = 0.5`: the slope of `ln E_n` over steps 20-60 is
   `2 ln rho_inf` within 10 percent (the roots at infinity are a defective `-rho_inf`, hence a
   slope, not a ratio). This checks the parameter map of decision 4.
5. Energy balance with loads: step traction and a moving support, trapezoidal rule, linear:
   kinetic + internal - external work constant to 1e-10 of the peak energy.
6. Global momentum balance with inertia: the sum of the reactions plus the external force
   equals `1.(M a)` per component to 1e-10, with a prescribed support motion.
7. Nonlinear path. `neo_hookean`, space-time MMS (`ManufacturedBodyForce` of
   `tests/test_solid_mms.cpp` plus `u_tt`): order 2 in `dt`; in one mid-run step the Newton
   contraction order is at least 1.8 (the Jacobian `K + c_M M_e` is consistent); probe
   self-convergence ratios 3.8-4.2 for a large-amplitude free vibration.
8. Reuse. Linear problem, 200 steps of constant `dt`: one Jacobian assembly, one AMG setup
   (`LinearSolver::Setups()`; two if the systems fallback triggers), results bit-identical
   with the reuse off; a forced bisection rebuilds once for the halved `dt` and once back.
9. Stepper. `t_final = 1e-3` and `t_final = 1e3`, `n = 1000`: exactly `n` accepted steps,
   final time exactly `t_final`, no degenerate last step.

Acceptance: `make check` green; every existing test, input and frozen number untouched
(decision 14); np 2 and 4 agree with serial to 1e-12 on one linear and one nonlinear
20-step run added to the `test_parallel` reference.

### DY2 — YAML, the app, outputs

`config.{hpp,cpp}`: the `dynamics` block, parsed **before** `bcs` and `body_force` so that
`ParseSchedule` gets its upper bound and defaulted schedules become `constant` (decision 7);
`output.every`, `output.energy`; `velocity` and `acceleration` in the field list, valid only
with `dynamics`. `solid_problem.{hpp,cpp}`: `MakeDynamicSolidProblem(problem, cfg)`.
`apps/solid_mechanics.cpp`: one branch (build the decorator, initial state, `SolveDynamic`),
the header echo of decision 7, per-step `energy` lines next to the probe and reaction lines
(same `step k t = ...` prefix, so the Python scripts parse them alike), the ParaView stride,
and the closing `result:` line with the time and the kinetic energy; the file's first
comment no longer says quasi-static only. The quasi-static branch prints what it prints today.

Tests (`test_dynamics`, `test_base` for the parser): every error of Section 3; the schedule
default and its echo; a quasi-static input is parsed to the same `AppConfig` as before;
fields registered and probed; the stride; an app run of a 20-step input whose `energy` lines
reproduce the library values of DY1 item 5.

Acceptance: `make check` green; log lines of the existing inputs unchanged (diff of the Cook
and elastic-bar logs against the ones of the current `main`).

### DY3 — Verification cases with reference solutions

Inputs under `apps/input/dynamics/`, histories compared by a new `apps/dynamics_compare.py`
(the pattern of `elastic_bar_compare.py`: parse the per-step probe / reaction / energy lines,
plot against the reference, print the measures below) and asserted in a new
`tests/test_dynamic_verification.cpp` (`make test`, target under 5 minutes).

| input | problem | reference | check |
|---|---|---|---|
| `bar_free_vibration.yaml` | fixed-free bar along x (`beam.msh`, the structured 10 x 1 x 1 hex box of `box.geo`; a finer one is a makefile rule), `linear_elastic`, `nu = 0` so that the clamp leaves the motion one-dimensional, initial displacement `A sin(pi X / 2L)` | `u = A sin(pi X / 2L) cos(w_1 t)`, `w_1 = (pi / 2L) sqrt(E / rho_R)` | tip history over five periods to 1e-3 A at the input's `dt`; the period elongation of the trapezoidal rule equals `(w_1 dt)^2 / 12` within 10 percent of that value at a coarse `dt` |
| `bar_step_load.yaml` | same bar, step traction `p` on the free end | d'Alembert: the tip displacement is a triangle wave of period `4L/c`, peak `2 p L / E`, mean `p L / E`; the wall stress jumps to `2p` at `t = L/c` | peak and mean within 2 percent; front arrival at mid-span within one element crossing time; record the overshoot of the wall stress for `newmark` and `rho_inf = 0.8` (the discontinuity is where the dissipation earns its place) |
| `cantilever_vibration.yaml` | `beam.msh`, `linear_elastic`, half-sine tip pulse, then free | Euler-Bernoulli `f_1 = (1.875104^2 / 2 pi) sqrt(E I / (rho_R A L^4))` | first frequency from the zero crossings within 2 percent |
| `mms_dynamic_2d.yaml`, `mms_dynamic_3d.yaml` | `linear_elastic`, manufactured `u(X, t)`, body force as an `expression` in `x, y, z, t` | manufactured | order p + 1 in `h` at a small fixed `dt`, order 2 in `dt` on the finest mesh |
| `neo_hookean_block_vibration.yaml` | finite-strain, large-amplitude free vibration, `rho_inf = 0.8` | none in closed form | probe self-convergence ratios 3.8-4.2 over three halvings of `dt`; total energy non-increasing; with the trapezoidal rule the energy balance error falls by 4 per halving |

Acceptance: `make test` red only at the pre-existing Cook ratio gate; np 4 runs of the two bar
inputs agree with serial to 1e-12.

### DY4 — Mixed u-p dynamics (near- and fully incompressible)

The decorator on the block vector (decision 5): `M` on the displacement block, the (0,0)
block of the Jacobian replaced by `K_uu + c_M M_e` in a `BlockOperator` the decorator owns,
`velocity` / `acceleration` of the displacement only, the warning of decision 13. **Measure
before changing the solver:** outer FGMRES iterations at `dt = h / c_s` and at a tenth of it,
against the quasi-static count of the same mesh. The Schur complement of a mass-dominated
displacement block tends to a pressure Laplacian, `B^T (c_M M)^{-1} B`, which a scaled
pressure mass does not represent; if the count more than doubles, extend the approximation in
the manner of Cahouet-Chabard, `S~^{-1} = (mu + gamma) kappa/(kappa + mu) M_p^{-1} + c_M rho_R L_p^{-1}`
with `L_p` a pressure Laplacian under AMG (H1 pressures make it available), and record both
counts. If it does not, record that and leave the solver alone.

Tests (`test_dynamics`, fast part; the rest in `test_dynamic_verification`): mixed MMS in time
at `nu = 0.4999` and `kappa = inf`, order 2 in `dt` for `u`, the rate of `p` reported and
gated at 1.8 after the first ten steps for `rho_inf = 0.8`; the pressure mode: perturb `p_0`,
successive norms of the pressure error in the ratio `rho_inf` within 5 percent; mixed and
displacement formulations at `nu = 0.3` agree to 1e-6 over 50 steps; np 2 / 4.

Verification case `knowles_tube_oscillation.yaml`: incompressible `iso_neo_hookean`, plane
strain, `annulus.msh` (the quarter model of the Lame case, rollers on `bottom` and `left`),
`inner` and `outer` traction-free, initial radial velocity `v_0 = k X / |X|^2` (isochoric; with
free surfaces no follower load is needed). Incompressibility
gives `r^2 = R^2 + c(t)`; integrating the radial momentum balance through the wall gives

    rho_R [ (c_tt / 2) ln(b / a) - (c_t^2 / 8) (1/a^2 - 1/b^2) ] = - int_a^b mu (lam^2 - lam^-2) / r dr,
    a^2 = A^2 + c,  b^2 = B^2 + c,  lam = r / R

(Knowles 1960; re-derive it in the verification manual rather than copying this line). The
compare script integrates it with `scipy.integrate.solve_ivp` at rtol 1e-10. Check: inner
radius over two periods within 0.5 percent, period within 0.5 percent, and the wall pressure
against the ODE's after the start-up steps.

Acceptance: as DY3; frozen mixed numbers (incompressible Cook corner `uy = 6.930412595013`)
unchanged.

### DY5 — Documentation and handback

- `doc/theory_manual.tex`: new section "Elastodynamics" after the small-strain section: strong
  and weak form with inertia, why the mass matrix is constant in total Lagrangian form, the
  semi-discrete system, the scheme family of decision 4 with its parameter table, the step
  equation and its Jacobian, initial acceleration and right-limit loads, prescribed motion and
  reactions, the energy balance, the mixed system as a DAE with the pressure mode of decision
  13, the linear case (one setup per run). Replace the sentences of Section 1 that say
  dynamics is absent; update "What is implemented", the pseudo-time subsection (physical time
  under `dynamics`), the solution-algorithms section (shared stepper), the seam section, and
  the symbols / input keys and source-file appendices. Notation: `rho_R`; `u_0`, `v_0`, `a_0`
  keep the subscript 0 (initial values, not referential quantities).
- `doc/verification_manual.tex`: `test_dynamics` and `test_dynamic_verification` subsections;
  a section "Dynamic cases" with problem / reference / check / tolerance / test for every DY3
  and DY4 input (with the derivation of the tube equation), the measured period elongation,
  overshoot and iteration counts, summary-table rows, references (Newmark 1959; Hilber,
  Hughes, Taylor 1977; Chung, Hulbert 1993; Knowles 1960; Cahouet, Chabard 1988 if used).
- `README.md`: schema block, a section "Dynamics" (choosing a scheme and `dt`: about 20 steps
  per period of interest for a 1 percent period error; consistent units are the user's job;
  the schedule default under dynamics), the case table, the "not supported" list of Section 5;
  remove "true dynamics" and "linear dynamics" from the existing lists. Schema comments in
  `src/base/config.hpp`.
- Build both manuals with `latexmk -pdf` in `doc/`, delete the auxiliary files, keep the PDFs.

### DY6 (optional, on demand) — Rayleigh damping

`dynamics.damping: { mass: a_M, stiffness: a_K }`, `C = a_M M + a_K K_0` with `K_0` the
Jacobian at `u_0` assembled once. The balance gains `C [(1 - af) v(u) + af v_n]`; since
`v(u)` is affine in `u` this adds `gamma / (beta dt) C` to the Jacobian and known terms to
`h_n`. Verification: logarithmic decrement of the first bar mode against
`zeta = a_M / (2 w_1) + a_K w_1 / 2` within 1 percent.

### DY7 (optional, on demand) — Explicit central differences

`scheme: central_difference`, displacement formulation only (an incompressible material has
an infinite wave speed): acceleration form with a lumped mass (HRZ scaling of the consistent
diagonal, positive on every element type and order, which row sums are not for p = 2
simplices), no Newton and no linear solver, a stable-`dt` estimate from a power iteration on
`M_L^{-1} K` with a safety factor, and an error when `dt` exceeds it. Verification: the two
bar cases of DY3; agreement with the implicit run at the same `dt`.

## 5. Out of scope (state in the README as "not supported")

- Energy-momentum conserving schemes (Simo-Tarnow, discrete gradients). The materials'
  `Energy(F)` makes an algorithmic stress feasible later; it is a kernel change, unlike this plan.
- Time-step adaptivity by an error estimate (only halving on a Newton failure).
- Static preload followed by a dynamic release: needs a load scale that is not the time, since
  the quasi-static stepper drives the same schedules with its pseudo-time. Until then a release
  test starts from an initial displacement or a pulse.
- Eigenfrequencies and mode shapes. With `M` from this plan, `HypreLOBPCG` on `(K, M)` is a short
  follow-up, and it would give DY3 a discrete-exact reference.
- Absorbing boundaries, contact and impact, viscoelastic (material) damping, checkpoint / restart.
- A lumped mass in the implicit path; time-dependent density; moving meshes.

## 6. Order of work and effort

DY1 (1.5 days) -> DY2 (0.5-1 day) -> DY3 (1 day) -> DY4 (1-2 days, the solver measurement
decides) -> DY5 (0.5-1 day); DY6 (0.5 day) and DY7 (1 day) on demand. DY1 alone is usable
from C++ but not from an input; do not hand over before DY2. DY3 does not depend on DY4. If
DY4's iteration counts call for the Laplacian term, commit the decorator with the measured
counts first and the solver change as its own commit.
