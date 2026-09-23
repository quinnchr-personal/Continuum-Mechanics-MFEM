# Implementation Plan — Scalar transport: the first-order flux/source physics of the kernel framework, and the convection–diffusion verification cases of `myapps/convection_diffusion` (Gates ST1–ST4)

**Audience:** the implementing agent. Follow the gate order, meet each gate's acceptance
criteria before moving on, commit once per completed gate (on `main`, no attribution
trailers). This extends the framework of `doc/hyperelasticity_implementation_plan.md`,
`doc/bc_loading_plan.md`, `doc/linear_elasticity_plan.md`, `doc/solid_dynamics_plan.md` and
`doc/thermoelasticity_plan.md`; their design decisions (reference mesh never moves, thin
`apps/`, `myapps/` untouched, models are value types templated on the scalar type, tangents by
dual numbers, no per-exercise drivers, both manuals kept current) still hold.

**Status (2026-09-23):** ST1–ST4 done the same day. ST1 commit fc9f8bbad4 (kernel, laws,
erf/erfc, scalar AMG), ST2 1db66c4fe8 (module, conditions, schema, executable, tests), then
ST3 (meshes, inputs, references, compare script, verification test) and ST4 (manuals, README).
Measured: the kernel's matrix equals the stock integrators to 1e-16, Jacobians to 1e-10,
steady MMS rates L2 2.02 / 2.99 / 3.99 and H1 1.00 / 2.01 / 3.00 at k = 1, 2, 3, first order in
dt with rates 0.999–1.000, the Kirchhoff case against its series with rates 0.94 / 0.96; the
cross-check against the myapps drivers on the identical triangulations: Pe = 1 3.8e-7, Pe = 10
3e-9, Pe = 100 9e-11, square 3e-9, disk 4.7e-7, transient MMS 4e-10 (L2) and 2e-9 (nodal
Linf), Kirchhoff 8e-9 (with the series initial condition and the rule 2p + 2), identical Newton
iteration counts; input rates: square L2 4.01 / H1 3.00, curved disk 2.96, 2.93 (k = 2) and
4.03, 3.94 (k = 3), transient MMS h-rate 2.00 with dt ∝ h².
Deviations from the plan as written: (1) the two transient linear drivers were rerun with
their Krylov tolerances tightened (rtol 1e-13, atol 1e-18) for the reference histories, since
with their own atol 1e-12 the Pe = 1 history drifted by 4e-4 (`reference/README.md`); (2) the
flow of a Dirichlet entry is +Σ r_j over its dofs (the sign that makes an inward flow positive;
item 6 wrote −Σ), asserted by the patch test; (3) the straight-polygon disk does not plateau
under refinement: the driver's boundary data is U projected on the polygon, so the polygon
problem has U as its exact solution and converges at rate 4; the curved meshes (lc = 0.1,
0.05, 0.025, not 0.05–0.0125: the finest would have been 14 MB) pose the problem on the exact
circle; (4) the first-order-in-dt check of case 5 runs at k = 2 on the input's mesh, since at
p = 1 the spatial error (2e-4) contaminates the finest step; (5) `ScalarFluxIntegrator` reads
the accepted state from a grid function and the kernel is not templated on the space
dimension (a runtime loop); the initial coefficient's time is reset to 0 in `InitialState`,
since a coefficient shared with the boundary data carries the conditions' time; (6) the
executable prints the Newton count on the step line rather than a Newton history CSV; (7)
`ScalarTransport` has a programmatic constructor (order, model, transient, quadrature order)
next to the YAML one, used by the tests.

## 0. What the cases need

`myapps/convection_diffusion/` holds five standalone MFEM drivers that verify a scalar
convection–diffusion solver against closed-form solutions (the other drivers of that
directory — ALE, ablation, two-domain coupling — are not part of this plan). All five are
continuous Galerkin on a scalar H1 space of order p, no stabilisation, backward Euler in
time, PETSc GMRES for the linear systems, and they print the L2 error against the exact
solution after every step. Reading them in full (2026-09-23) gives the following table; the
constants are those of the YAML inputs, which differ from some code defaults (driver 4).

| Case | myapps driver, input | Problem | Mesh, order, stepping | Exact solution, data | Recorded |
|------|----------------------|---------|-----------------------|----------------------|----------|
| 1 transient convection–diffusion, three Péclet numbers | `linear_convection_diffusion_1D.cpp`, `Input/input.yaml` (three uncoupled solves, Pe = 1, 10, 100) | ∂c/∂t + β·∇c − (1/Pe) Δc = 0 on (0,1)², β = (1, 0), non-conservative form (`ConvectionIntegrator`), c(x, 0) = 0; c = c_ex on x = 0 and x = 1 (nodal projection at t_{n+1}), natural on y = 0, 1 | `Mesh/unit_square.msh` (Gmsh, lc 0.05, 938 triangles, attributes bottom 1, right 2, top 3, left 4), p = 3, dt = 1e-3, 1000 steps to t = 1 | c(x, t) = ½ erfc((x − t)/(2√(t/Pe))) + √(t Pe/π) exp(−Pe (x − t)²/(4t)) − ½ (1 + Pe x + Pe t) e^{Pe x} erfc((x + t)/(2√(t/Pe))), c = 0 at t ≤ 0 (Homework 4-4, problem 3, part 1; uniform in y) | abs/rel L2 per step, `error_history.csv`; ParaView c, c_exact |
| 2 steady convection–diffusion–reaction, square | `linear_convection_diffusion_2D.cpp`, `Input/input_2d.yaml` | −κ Δu + c·∇u + s u = f on (0,1)², κ = 0.1, s = 1, c = (1, −2), u = 0 on Γ | `unit_square.msh`, p = 3, one solve | u = sin(nπx) sin(mπy), n = m = 3; f = κ(n² + m²)π² sin sin + c_x nπ cos(nπx) sin(mπy) + c_y mπ sin(nπx) cos(mπy) + s sin sin (course notes 16.930, eqs 7–12, table 1) | abs/rel L2, `error_history_2D.csv` |
| 3 steady convection–diffusion–reaction, disk | `linear_convection_diffusion_2D_circle.cpp`, `Input/input_2d_circle.yaml` | the same operator on the unit disk, κ = 1, s = 1, c = (1, 1); u = U on Γ (nodal projection on the polygonal boundary: nonzero at edge nodes inside the circle) | `Mesh/unit_circle.msh` (Gmsh, lc 0.05, 3056 straight triangles, one boundary attribute), p = 3 | U(r) = (r² − 1) cos(a r), a = 2π; f = −κ(U'' + U'/r) + (c·x) U'/r + s U, with U'' + U'/r → 2(2 + a²) at r = 0 (eq. 13, table 3) | abs/rel L2, `error_history_2D_circle.csv` |
| 4 nonlinear diffusion, Kirchhoff transform | `nonlinear_convection_diffusion_1D.cpp` (+ `newton_petsc_solver.hpp`), `Input/input_nonlinear_1d.yaml` (no convection despite the name) | m(u) ∂u/∂t − ∇·(a(u) ∇u) = 0 on (0, L)², L = 0.01, a = a0 + a1 (u − u_ref), m = m0 + m1 (u − u_ref), a0 = 10, a1 = 0.09, m0 = 4e6, m1 = 3.6e4, u_ref = 300; inward flux g = a(u) ∂u/∂n on x = 0 and x = L from the exact solution (which is q̄ = 7.5e5 at x = 0 and 0 at x = L, see below), natural on y; pure Neumann; u(x, 0) = the exact series at t = 0 (= 300 up to its truncation) | `Mesh/square_0p01.msh` (948 triangles), p = 3, dt = 0.1, 10 steps to t = 1; full Newton (no line search), atol 1e-10, rtol 1e-8 on max(1, ‖R_0‖), Jacobian by hand-written integrators with rule 2p + 2 | Kirchhoff transform of the linear problem (α = a0/m0 = 2.5e-6 since m1/m0 = a1/a0): θ = (q̄ L/κ1) [α t/L² + 1/3 − x/L + x²/(2L²) − (2/π²) Σ_{n=1}^{N} e^{−n²π²αt/L²} cos(nπx/L)/n²], u = T1 + (T2 − T1) (κ1/(κ2 − κ1)) [√(1 + γθ) − 1], γ = 2(κ2 − κ1)/((T2 − T1) κ1), κ1 = 10, κ2 = 100, T0 = T1 = 300, T2 = 1300, q̄ = 7.5e5, N = 1000 (`nonlinear_heat.m`; uniform in y) | abs/rel L2, Newton iterations and residuals per step, `error_history_nonlinear_1D.csv`, `newton_history_nonlinear_1D.csv` |
| 5 transient diffusion, manufactured solution | `diffusion_mms.cpp`, `Input/input_diffusion_mms.yaml` | ∂u/∂t − α Δu = f on (0,1)², α = 0.1; u = u_ex on Γ; u(x, 0) = u_ex(x, 0) = 0 | `unit_square.msh` refined once (3752 triangles), p = 1, dt = 0.01, 200 steps to t = 2 | u = sin t cos q, q = 2(x − ½)² + 2(y − ½)²; f = cos t cos q − α sin t [−16 r² cos q − 8 sin q], r² = (x − ½)² + (y − ½)² | L2 and nodal L∞ per step, `error_history.csv`; ParaView u, u_exact, error |

Facts that matter for reproducing them: the drivers are MPI (ParMesh), the L2 errors use
the quadrature rule of order max(2, 2p + 3), the relative error divides by ‖u_ex‖_L2, the
Dirichlet data is the nodal projection of the exact solution at t_{n+1}, t_n = n dt, the
stock integrators of drivers 1, 2, 3, 5 integrate their polynomial integrands exactly on the
affine triangles while the source term f uses `DomainLFIntegrator`'s rule of order 2p and the
nonlinear terms of driver 4 the rule 2p + 2 (the discrete problems are otherwise
quadrature-independent), and driver 4's flux datum evaluates to the constants q̄ and 0
because a(u_ex) ∂u_ex/∂x = −q̄ at x = 0 and ∂u_ex/∂x = 0 at x = L for its consistent
parameters (a0 = κ1, u_ref = T1, a1 = (κ2 − κ1)/(T2 − T1)). No driver has a refinement
loop; convergence rates were obtained by rerunning. Two names mislead: both "1D" drivers are
2D solves of y-uniform solutions, and driver 4 has no convection term.

The framework has no scalar unknown, no first-order-in-time physics, no source term inside a
kernel, no flux boundary condition on a scalar space outside the thermo module's private
entries, no error measure against an exact expression, and no `erfc` in its expressions. It
has everything else: the flux/source seam with dual tangents, the stepper in physical time
(`time` block), damped Newton with the linear-problem floor, the operator stamp, the Krylov
and direct solvers, Gmsh meshes with named groups, the field registry, ParaView output,
probes, and the test tiers.

## 1. Current state (verified 2026-09-23 — re-verify before starting)

- The seam (`README.md` "State of the seam", theory manual §Framework architecture) is
  implemented as the minimum the solid needs: `TotalLagrangianIntegrator<Material>`
  (`src/kernels/total_lagrangian.hpp`) hard-codes a vector H1 unknown and the flux P(F); the
  thermo kernel (`src/kernels/thermo_mixed_total_lagrangian.hpp`) is the only scalar-block
  kernel: `ThermoPointDensities<T>` templated on the scalar type, dual seeds on θ and on
  the dim components of Grad θ, each seed a rank-one update of the element tangent, the
  accepted θ_n held per quadrature point, implicit Euler over the step of the stepper,
  `HeatFluxIntegrator` as a hand-written boundary face integrator, `TemperatureEntry` /
  `FluxEntry` kept privately by the module on the temperature space.
- `QuasiStaticProblem` (`src/solvers/quasi_static.hpp`): `Mult`, `GetGradient`,
  `SetLoadFactor`, `ApplyDirichlet`, `IsLinear`, `AcceptStep`; `SolveQuasiStatic` (pseudo-time)
  and `SolveInTime` (the `time` block: targets, bisection, predictor). `SolidProblem` adds
  the solid API the app drives (`apps/solid_mechanics.cpp`): `Finalize`, `InitialState`,
  `RegisterFields` / `UpdateFields`, `MakeLinearSolver`, `FullResidual` + `ReactionsFrom`,
  `HasHistory` / `ResetHistory`, `GradientStamp`.
- `LoadSet` (`src/physics/loads.*`) is written for one vector H1 space (components,
  reactions as force and moment); `BCOptions{components, schedule, time_dependent, name,
  point}` and the nearest-node pin (`ResolvePoint`, global MINLOC) live there.
- Schema (`src/base/config.*`): `AppConfig` is the solid's; `MeshConfig`, `TimeConfig`,
  `SolverConfig` (`NewtonConfig`, `LinearSolverConfig`, `SubstepConfig`), `OutputConfig`,
  `Schedule`, `ProbeConfig` and their section parsers (`ParseMeshConfig`, `ParseTimeConfig`,
  `ParseSolverConfig`, `ParseOutputConfig`, `ParseSchedule`) are exposed "so tests and other
  physics can reuse them"; `NodeReader` tracks consumed keys and names unknown ones.
  `ParseOutputConfig` validates `fields` against a closed list of solid names.
- `Expression` (`src/base/expression.*`): x y z t, pi, + − * / ^, sin cos tan exp log sqrt
  abs pow min max, `if(cond, a, b)`; no erf/erfc; the untaken branch of `if` is evaluated
  (check that the selection is a select, not a blend, so a NaN there is harmless).
- `LinearSolver` (`src/solvers/linear_solver.*`): GMRES/CG + BoomerAMG with the
  `elasticity` or `systems` options (vector spaces); `DirectSolver` (MUMPS through PETSc)
  takes a `HypreParMatrix` or a block operator of them; both honour the operator stamp.
- `QuadratureFields` (`src/physics/quadrature_fields.*`) presents a quadrature-point
  quantity of any vdim at nodes / elements / point clouds; `FieldRegistry`, `ParaViewWriter`,
  `ProbeVector`, `ReactionWriter` are physics-agnostic.
- Meshes: `apps/mesh/*.geo` with named physical groups, `.msh` generated by `make meshes`
  and kept in the tree; `mesh_input.cpp` reads any MFEM format, keeps the physical names as
  attribute sets, refines serially and in parallel; the programmatic Cartesian box exists
  for the tests only.
- Tests: `make check` (serial, ~1 min) and `make test` (app runs, np = 2, 4, verification
  suites); the pattern of a YAML-driven verification test is `tests/test_linear_verification.cpp`
  (`ManufacturedTest`: refine the input's mesh, L2 error against the coded exact field,
  rate ≥ threshold) and of a Python cross-check `apps/elastic_bar_compare.py` (`--check`
  in `make test`).
- The five myapps drivers build with `make -C myapps/convection_diffusion <target>` against
  the same MFEM (with PETSc); their sources are unchanged since 2026-02-25 (the uncommitted
  changes of that directory are in the ablation drivers). None of the five binaries is
  currently built.

## 2. Design decisions — do not re-litigate

1. **A second physics, not a variant of the solid.** `ScalarTransport`
   (`src/physics/scalar_transport.*`) is a `QuasiStaticProblem` on a scalar H1 space of order
   k (`mesh.order`), the unknown u (a concentration, a temperature, a potential). It is not a
   `SolidProblem` (no displacement, no `LoadSet`); the app-facing API it needs (`Finalize`,
   `InitialState`, `RegisterFields` / `UpdateFields`, `MakeLinearSolver`, `FullResidual`,
   `Flows`, `Errors`, `GradientStamp`, `HasHistory` / `ResetHistory`, `Description`,
   `GlobalTrueVSize`) is declared on the class itself; a common abstract interface of the two
   physics is the coupling layer's job, not this plan's.
2. **A second thin executable,** `apps/scalar_transport.cpp` (`build/apps/scalar_transport`),
   with its own schema. The general solid executable cannot express a scalar transport
   problem (different unknown, laws, boundary conditions and outputs), which is the one
   condition under which new C++ in `apps/` is warranted (the no-per-exercise-drivers rule).
   It parses, wires and runs, as `solid_mechanics.cpp` does, and nothing else; the five
   cases are inputs of it plus one Python script. A single executable with a `physics:`
   switch is deferred to the coupling layer, which is when one executable will build
   several physics; the schema reserves the key `physics: scalar_transport` (required in
   the new executable's inputs, so that the two executables reject each other's files
   with a clear message).
3. **The kernel is the framework's first general scalar CG kernel with the flux/source
   contract,** `ScalarFluxIntegrator<Model>` (`src/kernels/scalar_flux.hpp`), a
   `NonlinearFormIntegrator` on the scalar space, templated on the space dimension (2, 3)
   like the solid kernels. Strong form on the fixed domain, per unit volume,
       c(u) ∂u/∂t − ∇·F(u, ∇u) + S(u, ∇u) = 0,
       F = κ(u) ∇u                  and S = β·∇u + s u − f      (non-conservative convection, default)
       F = κ(u) ∇u − β u            and S = s u − f              (conservative convection),
   so that F is the negative of the physical flux (the convention of the thermo kernel's
   `flux = −Q`) and the natural boundary condition is F·n = 0. Weak form with implicit Euler
   over the step t_n → t_{n+1} = t_n + Δt of the stepper (Δt = 0 drops the capacity term:
   the steady problem),
       R(u; q) = ∫ q c(u) (u − u_n)/Δt dV + ∫ ∇q·F(u, ∇u) dV + ∫ q S(u, ∇u) dV − Σ_i s_i(t) ∮ q g_i dA = 0,
   where g_i is the prescribed inward flux (g = F·n = κ ∂u/∂n for pure diffusion: positive
   into the domain, the `heat_flux` convention of the thermo module) of the i-th flux entry
   and s_i its schedule. The point contract is one templated function
   `ScalarDensities<T>(model, u, grad u, u_n, dt, x, t)` returning the scalar density
   r = c (u − u_n)/Δt + S and the vector F; the tangent comes from 1 + dim dual seeds (u and
   the components of ∇u), each a rank-one update of the element matrix (the thermo kernel's
   loop, reused as the template). Velocity β(x, t) and source f(x, t) are coefficients
   (expressions) evaluated per quadrature point by the integrator and passed in as values;
   they never enter the tangent. The old state u_n is read from the module's accepted-state
   grid function at the element's dofs (the same values as a quadrature-point history; no
   `HistoryField` is needed), and the quadrature rule is of order `transport.quadrature_order`
   (default 2k + 3, the framework's rule; the key exists for the cross-checks of gate ST3).
   The kernel does not derive from the solid kernels and does not modify them; factoring
   the three element loops into one CG kernel is a later refactor with two concrete
   instances in hand.
4. **The model is a value type of laws, not a material variant:** `ScalarTransportModel`
   (`src/kernels/materials/scalar_transport_model.hpp`) with the capacity c(u), the
   isotropic conductivity κ(u) and the reaction coefficient s as `AffineLaw{value, slope,
   reference}` (a(u) = value + slope (u − reference); a bare number in the input is slope 0),
   the convection form, and `Capacity<T>(u)`, `Conductivity<T>(u)` templated on the scalar
   type. No regions, no anisotropy, no tables: the five cases need constants and one affine
   pair; anything else is a later law (an expression in u, once `Expression` is templated on
   its scalar type, is the natural next one).
5. **Linearity is detected, not declared:** `IsLinear()` is true when both slopes are zero
   (β, s, f, g and the Dirichlet data are affine in u regardless of their dependence on x and
   t), so Newton accepts the linear problem at its round-off floor after one solve
   (`NewtonConfig::linear_problem`, the LE gates), and the Jacobian of a linear problem is
   assembled once and reused with its stamp until Δt changes, the boundary conditions change
   or the velocity expression mentions t (the LE6 mechanism, extended by the two extra
   invalidation conditions). Nonlinear laws use the damped Newton of the framework (line
   search on; the tangent predictor allowed).
6. **Boundary conditions of a scalar unknown live in `ScalarConditions`**
   (`src/physics/scalar_conditions.*`): Dirichlet entries (faces by attribute, or a
   nearest-node `point:`, one expression, a schedule, a name), flux entries (inward flux g
   per unit area, expression, schedule; assembled with `BoundaryLFIntegrator` into one true
   vector per entry, reassembled when the expression mentions t, summed with the schedules
   into the external vector), `Finalize`, `SetTime`, `SetPhysicalTime`, `ApplyDirichlet`,
   `EssentialTrueDofs`, overlap warnings, and `Flows(r)`: the flow through each Dirichlet
   entry from the full residual r (internal minus external terms, essential rows kept),
   Φ_i = −Σ_{its dofs} r_j, with the sign such that a flow into the domain is positive (the
   scalar analogue of the reactions; the patch test of ST2 asserts the sign). The thermo module's
   private entries are the ancestor of this class; migrating them is out of scope.
7. **Time integration is implicit Euler inside the kernel,** not a decorator: the capacity
   may depend on the state (case 4), which the seam note already identified as the reason a
   capacity term belongs in the nonlinear form. `SetLoadFactor(t)` sets Δt = t − t_accepted,
   `AcceptStep(x)` copies x into the accepted state and sets Δt = 0, `ResetHistory(t)` makes
   t the accepted time; `HasHistory()` is true under a `time` block. Without the block the
   problem is steady and the pseudo-time loop of `SolveQuasiStatic` applies (one load step
   by default; schedules ramp the data as for the solid). First order in time is a gate;
   higher-order schemes (a θ-scheme, BDF2, the first-order generalized-α of the seam note
   for a constant capacity) are out of scope.
8. **Errors against an exact expression are an output of the executable,** because all
   five cases are error histories: `output.exact` (an expression u_ex(x, y, z, t)) makes the
   module compute after every accepted step the L2 error with the rule of order 2k + 3, the
   relative L2 error (divided by ‖u_ex‖_L2, zero when that is below 1e-14) and the nodal L∞
   error max |u_i − Π u_ex,i| (the measure of driver 5), printed on the step line and written
   to `<paraview>/error_history.csv` (columns step, time, l2, rel_l2, linf_nodal), and
   registers the fields `<u>_exact` and `<u>_error`. H1 errors are computed in the tests from
   coded gradients (no symbolic differentiation of expressions).
9. **Outputs:** the nodal unknown under the name `transport.unknown` (default `u`), its
   exact and error fields, the quadrature quantity `flux` (the vector F, dim components,
   through `QuadratureFields` with the usual presentations), probes of every registered
   field, `output.flows` (the flow of every Dirichlet entry per step, log line
   `flow <name>: ...` and `<paraview>/flows.csv`), Newton iterations per step in the
   step line (driver 4's Newton history), the same `every` stride and `probe_every_step`.
10. **Solvers:** `LinearSolver` gains `amg: scalar` (BoomerAMG's defaults for a scalar
    operator; selected automatically when the space has vdim 1, and the only value the
    scalar app accepts), `gmres_amg` the default; `cg_amg` is an input error with a velocity
    (nonsymmetric); `direct` as is. The Newton tolerances follow the framework's
    conventions (relative to the first residual; the inputs of case 4 set `atol` above the
    round-off floor of rows of size m0 |Ω|/N).
11. **Meshes:** the three myapps meshes are copied into `apps/mesh/` unchanged
    (`square_tri.msh`, `square_0p01_tri.msh`, `disk_tri.msh`) with their `.geo` sources
    under the framework's header convention, so that the cross-checks run on the identical
    triangulations; the makefile rules regenerate them from the `.geo` files but another
    Gmsh version changes the triangulation, so the tracked `.msh` copies are the reference.
    Curved disks (`disk.geo` meshed with third-order geometry at lc = 0.05, 0.025 and
    0.0125, `disk_p3_{1,2,3}.msh`) serve the rate study of case 3: uniform refinement of the
    straight polygon never approaches the circle (the error plateaus at the polygon's
    geometric error), and MFEM's refinement of a curved mesh interpolates the coarse
    geometry, so the resolutions are generated, not refined (cross-check on the straight
    mesh, rates on the generated curved ones).
12. **The reference of the cross-check is the myapps output,** recorded once: the five
    drivers are built and run from their directory with their inputs (PETSc options as
    given), and their CSVs are copied to `apps/input/scalar_transport/reference/` with a
    README naming the commit, the commands and the date. On the same triangulation, order,
    steps, quadrature and error rule the discrete problems of cases 1, 2, 3 and 5 coincide,
    and case 4's as well with the series initial condition; so the error histories must agree
    to the linear-solver tolerance of the myapps runs (1e-10 relative on the solution: a
    few 1e-6 of the error). The inputs use the framework's defaults (rule 2k + 3, `initial:
    "300"` for case 4); the round-off agreement is a test with `quadrature_order` set to the
    drivers' rules (2p for the source terms of 2, 3, 5; 2p + 2 for case 4) and, for case 4,
    the series initial condition given programmatically.

## 3. YAML surface

```yaml
physics: scalar_transport             # required in the scalar executable's inputs
mesh: { file: apps/mesh/square_tri.msh, serial_refine: 0, order: 3 }   # as for the solid (perturb, parallel_refine)
transport:
  unknown: c                          # name of the nodal field (default u)
  capacity: 1.0                       # c(u): a number, or { value: 4.0e6, slope: 3.6e4, reference: 300.0 } = value + slope (u - reference)
  conductivity: 0.01                  # kappa(u): the same forms (1/Pe here); must be > 0 at the reference
  velocity: ["1", "0"]                # beta(x, y, z, t), one expression per space dimension; omit for none
  convection: nonconservative         # nonconservative (beta . grad u, default) | conservative (-div(beta u) in the flux)
  reaction: 0.0                       # s: the term s u (a number)
  source: "cos(t)*cos(2*(x-0.5)^2 + 2*(y-0.5)^2) - 0.1*sin(t)*(...)"   # f(x, y, z, t); omit for none
  quadrature_order: 9                 # rule of the kernel and of the source (default 2 k + 3)
initial: "300"                        # u(x, y, z) at t = 0 (default 0); transient only
time: { t_final: 1.0, dt: 1.0e-3 }    # optional: implicit Euler in physical time (dt or steps as for the solid); absent: steady
bcs:
  dirichlet:
    - { attr: [left, right], name: ends, expression: "if(t <= 0, 0, 0.5*erfc((x - t)/(2*sqrt(t/100))) + ...)" }
    - { point: [0.0, 0.0], expression: "0" }          # a node
  flux:
    - { attr: [left], name: heated, expression: "7.5e5" }   # inward flux g = kappa du/dn per unit area, positive into the domain
  # Every entry may add schedule: as for the solid; under time the default is constant and the expression carries t.
solver:
  predictor: none                     # none | tangent (nonlinear laws)
  newton:  { rtol: 1e-8, atol: 1e-10, max_it: 20, print_level: 1 }
  linear:  { type: gmres_amg, amg: scalar, rtol: 1e-12, max_it: 500 }   # gmres_amg | cg_amg (no velocity) | direct
output:
  paraview: out/peclet_100
  fields: [c, c_exact, c_error, flux]  # the unknown by its name, its exact and error fields (with exact), flux (quadrature quantity)
  quadrature_at: [nodes]              # presentations of flux, as for the solid
  exact: "if(t <= 0, 0, ...)"         # u_ex(x, y, z, t): L2, relative L2 and nodal Linf errors per step, error_history.csv, the two fields
  probes: [ { name: mid, point: [0.5, 0.5] } ]
  probe_every_step: false
  flows: true                         # the flow through every Dirichlet entry per step, flows.csv
  every: 10
```
Errors: a missing or different `physics`; an unknown key (the solid keys `formulation`,
`plane`, `material`, `body_force`, `dynamics` are unknown here); `velocity` with a
component count other than the space dimension; `conductivity` not positive at the
reference; `initial` without `time`; `cg_amg` with a velocity; `amg` other than `scalar`;
`fields` naming `<u>_exact` / `<u>_error` without `exact`; an expression
mentioning `u`; `point` and `attr` in one entry; a point farther than 1e-8 of the mesh
diameter from every node; the solid executable given a `physics` key. `Expression` gains
`erf` and `erfc`.

## 4. Gates

### ST1 — The scalar CG kernel and the model

- `ScalarTransportModel` with `AffineLaw`, the convection forms; `ScalarDensities<T>`;
  `ScalarFluxIntegrator<Model>` (residual, tangent by 1 + dim seeds, element energy not
  required); `erf` / `erfc` in `Expression` (and the `if` select check); no module yet.
- Tests (`tests/test_scalar_transport.cpp`, part 1, in `make check`; programmatic meshes):
  (a) for constant laws the assembled matrix of the kernel on a `ParNonlinearForm` equals
  `MassIntegrator(c/Δt) + DiffusionIntegrator(κ) + ConvectionIntegrator(β) + MassIntegrator(s)`
  of a `ParBilinearForm` to 1e-14 relative (both convection forms; the conservative one
  against `ConservativeConvectionIntegrator`, or against the transpose identity for a
  constant β), on quadrilaterals, triangles and hexahedra;
  (b) the assembled Jacobian against central differences of the residual (1e-6 relative)
  with affine capacity and conductivity, a velocity, a reaction and a source, in 2D and 3D;
  (c) the residual of the exact nodal state of a linear-in-x solution (the scalar patch
  test) is zero to round-off on a perturbed mesh, steady, both convection forms;
  (d) a Newton contraction of order ≥ 1.9 on the affine laws of case 4 (a 2 x 2 mesh,
  one step);
  (e) the two convection forms give the same solution for a constant β (1e-12).

### ST2 — The module, the conditions, the schema and the executable

- `ScalarConditions`; `ScalarTransport` (space, `ParNonlinearForm` with the kernel,
  accepted state, Δt protocol, `IsLinear`, the stamp with its invalidations, `InitialState`
  from `initial`, errors against `output.exact`, flows, fields, the linear solver of item
  10); `ScalarAppConfig` and `LoadScalarConfig` (`src/base/scalar_config.*`, reusing the
  section parsers and `NodeReader`; the `fields` list validated against the scalar names);
  `apps/scalar_transport.cpp`; the makefile builds it as it builds the solid app (`APP_SRC`
  wildcard).
- Tests (`tests/test_scalar_transport.cpp`, part 2, in `make check`; `mpirun -np 2` of it
  in `make test`):
  (a) steady patch test through the module from a YAML string: u = a + b·x prescribed on
  all faces, interior exact to 1e-13, the flows of two opposite faces equal and opposite and
  equal to κ b·n |face| (the sign convention of `Flows`), on a perturbed mesh;
  (b) transient exactness: a spatially constant state with a constant source, u = u_0 +
  f t/c, is reproduced by implicit Euler to round-off at every step (any dt), with a
  natural boundary; and the same with a prescribed inward flux g on one face of a bar,
  whose flow through the opposite Dirichlet face equals g times the face area at steady
  state;
  (c) manufactured solutions of the steady operator (the field of case 2 on the unit
  square, quadrilaterals) at k = 1, 2, 3 over three refinements: L2 rate ≥ k + 0.9, H1 rate
  ≥ k − 0.1 (coded gradient), one Newton iteration per solve (`IsLinear`), one Jacobian
  assembly and one solver setup per run (`GradientAssemblies`, `Setups`);
  (d) temporal order: the transient solution of case 5 on a fixed 16 x 16 k = 3 mesh (the
  spatial error, about 1e-5, stays below the temporal one) with dt = 0.02, 0.01, 0.005,
  0.0025: L2 error rate in dt ≥ 0.95 between consecutive pairs; and one
  Jacobian assembly for the whole run of the linear transient problem with constant dt,
  a reassembly when dt changes (segments) and when the velocity mentions t;
  (e) nonlinear laws: case 4 through the module on a 4 x 4 k = 3 mesh (L = 0.01) against
  the coded series (N = 1000) at t = 1: the L2 error consistent with first order in dt
  (dt = 0.1, 0.05, 0.025: rate ≥ 0.9), Newton converged in ≤ 4 iterations per step; and the
  Kirchhoff identity as a second, independent reference: with m1/m0 = a1/a0 the transformed
  unknown φ = a0 (u − u_ref) + ½ a1 (u − u_ref)² satisfies the linear heat equation
  φ_t = α Δφ with the flux datum q̄, so the inverse transform of the linear solve (constant
  laws, the same mesh and steps) approximates the same solution. The two discrete
  solutions are not the same problem in other variables, so they agree only to the
  discretisation error: assert that their difference is below the L2 error of either
  against the series, and record the measured value;
  (f) the `initial` expression, the point pin (pure Neumann steady problem made
  well-posed), the schedule of a flux entry, the errors and fields with `exact` (the error
  of the exact nodal state is the interpolation error only), `flows.csv` and
  `error_history.csv` written;
  (g) the schema errors of Section 3, and the solid executable rejecting a `physics` key;
  (h) np = 2 against serial: errors and flows to 1e-12.

### ST3 — The five cases

- Meshes: `apps/mesh/square_tri.{geo,msh}`, `square_0p01_tri.{geo,msh}`, `disk_tri.{geo,msh}`
  (copies of the myapps meshes, item 11), `disk.geo` with `disk_p3_{1,2,3}.msh` (third-order
  geometry at three resolutions), makefile rules and the `meshes` target.
- Inputs `apps/input/scalar_transport/`: `convection_diffusion_peclet_1.yaml`,
  `_10.yaml`, `_100.yaml` (case 1, one input per Péclet number), `steady_cdr_square_mms.yaml`
  (2), `steady_cdr_disk_mms.yaml` (3, straight mesh) and `steady_cdr_disk_mms_curved.yaml`
  (the curved mesh for the rates), `nonlinear_diffusion_kirchhoff.yaml` (4),
  `transient_diffusion_mms.yaml` (5); each with a header comment giving the problem, the
  source of the exact solution, and what the test checks; `reference/` (the myapps CSVs,
  `README.md`, `regenerate.sh`).
- `apps/scalar_transport_compare.py` (`make scalar_transport`): runs the inputs, parses
  the step lines / `error_history.csv`, overlays the error histories on the myapps CSVs
  (L2 against time per Péclet number; Newton iterations per step for case 4; one bar per
  steady case), prints the largest relative discrepancy per case, `--check` (in
  `make test`) asserts the tolerances below, `--no-plot`.
- `tests/test_scalar_verification.cpp` (in `make test`): (a) the cross-check of item 12
  for all five cases (case 4 with the series initial condition and `quadrature_order` 8;
  cases 2, 3, 5 with `quadrature_order` 2p): relative agreement of the error histories
  ≤ 1e-5 (the measured numbers go into the status note; if the myapps solver tolerance
  makes 1e-5 unreachable, record and set the gate at the measured level times 10);
  (b) the framework's own checks: case 1 at Pe = 1, the L2 error at t = 1 is first order
  in dt at the input's mesh (dt = 1e-3, 2e-3, 4e-3: rate ≥ 0.9; at p = 3 and dt = 1e-3 the
  temporal error dominates, so a spatial rate is not observable there), and at Pe = 10 and
  100 the errors are reported (the front of the Pe = 100 case is under-resolved on the
  given mesh, in both codes); case 2: the rates of ST2(c) on the tri mesh under uniform
  refinement; case 3: on the generated curved meshes L2 rate ≥ k + 0.9 at k = 2 and 3, and
  on the straight mesh the plateau at the polygon's geometric error under refinement
  (report only); case 5: the error at t = 2 is first order in dt at the fixed mesh and
  second order in h with dt ∝ h² (k = 1); case 4: the ST2(e) rates on the input's mesh.
- Acceptance: all of the above green serially and the compare script's `--check` green;
  `make test` extended with the two.

### ST4 — Documentation and handback

- Theory manual: a section "Scalar transport" (strong and weak forms, the flux/source
  contract and the two convection forms, implicit Euler and the accepted state, the tangent
  by seeds, the boundary conditions and flows, the error measure, linearity and the reuse
  of the operator, the executable), the seam section updated (two instances of the
  contract; scalar unknowns, sources and flux conditions now exist), the symbol and source
  tables.
- Verification manual: the `test_scalar_transport` subsection under the discretisation
  tests, a section "Scalar transport cases (`apps/input/scalar_transport/`)" with the five
  cases (problem, reference, check, tolerance, the test that runs it; the myapps
  cross-check and its measured agreement), the summary rows, the executable and the meshes
  subsections updated.
- README: the layout block, build and test lines, a top-level section "Scalar transport
  (`apps/scalar_transport`, `apps/input/scalar_transport/`)" with the schema of Section 3
  and the cases, the seam section.
- Memory file for the session; the makefile's stale `ale_validation_be*` targets in
  `myapps/convection_diffusion/makefile` are not this plan's concern (myapps untouched).

## 5. Out of scope (state in the README as "not supported")

SUPG or any stabilisation (the Pe = 100 case is under-resolved at the front on the given
mesh exactly as in myapps); discontinuous Galerkin; anisotropic or tensor conductivity;
laws other than affine in u (expressions in u, tables); Robin / convective boundary
conditions; regions with different laws; higher-order time integration; the ALE decorator
of the architecture (`diffusion_mms_ale.cpp` is its verification case and the natural next
plan); coupling with the solid (the temperature of the thermo module stays its own);
one-dimensional meshes (MFEM supports them; the kernels are written for dim = 2, 3 like
the solid's, and the two "1D" cases are 2D strips as in myapps); the two-domain coupled
Poisson drivers (a partitioned coupling of two `ScalarTransport` instances: the coupling
layer's first case); the ablation drivers.

## 6. Order of work and effort

ST1 (kernel, model, expression functions, kernel tests): about 30 %. ST2 (conditions,
module, schema, executable, solver mode, module tests): about 35 %. ST3 (meshes, inputs,
references from the myapps runs, script, verification test): about 25 %; build and run the
five myapps drivers first, so that the reference CSVs exist before the inputs are tuned.
ST4: the rest. All runs are small (at most 200 steps of 4k unknowns); nothing needs
detaching.
