# Continuum-Mechanics-MFEM
Applications that run continuum mechanics simulations using MFEM

## Flux-kernel framework (root `src/`, `apps/`, `tests/`)

A multiphysics framework on MFEM in which each physics is expressed as
fluxes and sources at quadrature points and the framework owns assembly,
Newton, and the linear solvers (architecture: `doc/flux_kernel_architecture.html`,
plan: `doc/hyperelasticity_implementation_plan.md`). The first physics is
quasi-static nonlinear solid mechanics (compressible hyperelasticity, total
Lagrangian, CG, weak form in `doc/theory_manual.tex`), with
small-strain linear elasticity as a material of the same kernels
(`model: linear_elastic`, plan and measured results in
`doc/linear_elasticity_plan.md`), and with inertia when the input carries a
`dynamics:` block (implicit elastodynamics of either formulation, plan and
measured results in `doc/solid_dynamics_plan.md`). The models and
methods of `src/` are described in `doc/theory_manual.tex`; every input and
test under `apps/` and `tests/`, with its reference solution and tolerance, in
`doc/verification_manual.tex`. The `myapps/` tree is legacy and separate.

### Layout

```
src/base/       tensor.hpp (fixed-size tensors), dual.hpp (forward-mode AD),
                config.{hpp,cpp} (YAML schema -> structs, key-path errors, load schedules),
                expression.{hpp,cpp} (f(x, y, z, t) parser/evaluator for boundary data),
                coefficients.hpp (constant, affine and expression coefficients of the YAML data),
                mesh_input (file or Cartesian box, corner map, jitter, refinement),
                fields.hpp (named field registry), output (ParaView), probes (point values)
src/kernels/    total_lagrangian.hpp: qpoint free functions + TotalLagrangianIntegrator<Material>;
                mixed_total_lagrangian.hpp: u-p qpoint functions + MixedTotalLagrangianIntegrator<Material>;
                follower_pressure.hpp: boundary face integrator T = -p J F^-T N (both forms)
src/kernels/materials/
                neo_hookean.hpp, st_venant_kirchhoff.hpp, gent_compressible_summit.hpp (coupled,
                displacement formulation; the last is SUMMIT's compressible Gent model);
                iso_neo_hookean.hpp, mooney_rivlin.hpp, yeoh.hpp, gent.hpp, arruda_boyce.hpp,
                ogden.hpp (isochoric-volumetric split, either formulation; isochoric.hpp shared
                I1bar pieces, volumetric.hpp the selectable volumetric laws U(J),
                spectral.hpp symmetric 3x3 eigen-solver for Ogden);
                linear_elastic.hpp (small strain: sigma(sym(F - I)), either formulation) and
                kinematics.hpp (finite or small strain as a trait of the material: Cauchy stress,
                volume ratio, strain);
                plane_stress.hpp (adapter: F33 = thickness stretch with P33 = 0, any base model);
                material_tangent.hpp (dual seeding), materials.{hpp,cpp} (variants, YAML factory, moduli)
src/physics/    solid_problem.{hpp,cpp} (common interface, factory by formulation, YAML load installer);
                loads.{hpp,cpp}: LoadSet, the boundary conditions and external loads of a
                displacement space (component masks, schedules of the pseudo-time t, per-entry
                dead-load vectors, follower-pressure scales), shared by both formulations;
                quadrature_fields.{hpp,cpp}: quadrature-point quantities and their nodal /
                element / point-cloud presentations (shared by both formulations);
                solid_mechanics_tl.{hpp,cpp}: displacement formulation (residual, assembled
                Jacobian, output fields);
                mixed_solid_mechanics_tl.{hpp,cpp}: u-p formulation on a Taylor-Hood pair;
                dynamic_solid_problem.{hpp,cpp}: inertia as a decorator over either of them
                (constant mass matrix, the step equation S(u) + c_M M (u - u*) + h_n = 0 of the
                time integrator, initial state, energies, reactions with inertia)
src/solvers/    newton (damped Newton, Armijo backtracking; a linear problem is accepted at the round-off
                floor of its residual), linear_solver (GMRES/CG + BoomerAMG),
                saddle_point_solver (augmented Lagrangian FGMRES for the u-p Jacobian; in a dynamic
                analysis with the inertial part of the Schur complement approximation),
                quasi_static (load stepping over the pseudo-time t in (0, 1] with bisection, and the
                same loop in physical time), time_integration (newmark | hht | generalized_alpha)
apps/           solid_mechanics.cpp (YAML parsing and wiring only), apps/mesh/*.geo (Gmsh sources of the
                example meshes, named physical groups) and the generated apps/mesh/*.msh (make meshes);
                apps/input/finite_elasticity/:
                  cooks_membrane/ (compressible, incompressible and nearly incompressible Cook's
                    membrane; literature benchmark, frozen regression values),
                  verification/ (cases checked against a reference in the test suite: euler_bernoulli_cantilever3d.yaml
                    vs Euler-Bernoulli in the small-load limit, rivlin_cylinder_inflation.yaml vs the Rivlin
                    inflation, homogeneous_deformations/*.yaml vs the closed forms of the incompressible
                    neo-Hookean model through apps/homogeneous_compare.py; any of the six models works in
                    those inputs and tests/test_homogeneous covers all six);
                apps/input/linear_elasticity/ (small strain, model: linear_elastic):
                  verification/ (Lame cylinder and sphere, Kirsch, cantilever, manufactured solutions,
                    two materials) and cooks_membrane/ (the benchmark as posed, and its incompressible
                    plane-strain variant), all checked by tests/test_linear_verification;
                apps/input/anand_coupled_theories/<chapter>/*.yaml (the examples of Anand's coupled-theories
                  book, from its FEniCSx companion codes; finite_elasticity so far);
                apps/input/elastic_bar/ (the exercise of myapps/elastic_bar as inputs of the general
                  executable, linear and nonlinear, with the exercise's own results in reference/) and
                  apps/elastic_bar_compare.py (force against displacement, compared);
                apps/input/plate_with_hole/ (the exercise of myapps/plate_with_hole: Kirsch's displacement
                  on the outer edges as expressions) and apps/plate_with_hole_compare.py (the computed
                  fields against the closed form);
                apps/input/dynamics/ (inputs with a dynamics block: bar vibration and d'Alembert's wave,
                  cantilever frequency, manufactured solutions in space and time, a neo-Hookean block,
                  Knowles' incompressible tube) and apps/dynamics_compare.py (time histories from the
                  per-step lines of the app, over the closed forms)
tests/          test_base, test_materials, test_solid_mms, test_mixed, test_homogeneous, test_loading,
                test_linear_elasticity, test_dynamics (make check); test_mixed --full, test_benchmarks,
                test_parallel, homogeneous compare, test_loading np=4, test_verification,
                test_linear_verification, test_dynamic_verification serial and np = 2, 4 (make test)
makefile        out-of-tree build under build/ (BUILD_DIR): build/libcmf.a (LIBNAME) from src/,
                then build/apps/* and build/tests/* linked against it
```

### Build and test

Requirements: MFEM 4.8 built with MPI, METIS, and HYPRE (the makefile finds
`~/MFEM/mfem/config/config.mk` automatically; set `MFEM_DIR` otherwise),
yaml-cpp via `pkg-config`, and an `mpirun`.

```
make            # build/libcmf.a, build/apps/solid_mechanics, build/tests/*
make meshes     # regenerate apps/mesh/*.msh from apps/mesh/*.geo with Gmsh (the .msh files are kept in the tree)
make check      # serial, ~45 s: tensor/dual/YAML units, materials, patch tests + MMS (both
                # formulations), homogeneous deformations vs closed forms (all incompressible models),
                # small-strain linear elasticity (operator vs MFEM's ElasticityIntegrator, outputs),
                # inertia and time integration (free fall, orders, energy, mixed u-p)
make homogeneous # the app on apps/input/finite_elasticity/verification/homogeneous_deformations/*.yaml, compared with the closed forms (python3 + yaml)
make elastic_bar # the elastic bar exercise: linear against Gent, force-displacement table and plot (python3 + yaml + matplotlib)
make plate_with_hole # the plate-with-a-hole exercise: errors against Kirsch's closed form and plot (python3 + yaml + pyvista + matplotlib)
make dynamics   # the dynamic cases: time histories over their closed forms, measures and plots in out/dynamics (python3 + yaml + matplotlib + scipy; ~2.5 min)
make test       # everything: app runs serial and np=4, np={2,4} consistency, benchmarks, homogeneous
make clean      # removes build/
```

All build products (objects, `.d` dependency fragments, the library, the
executables, and test scratch files) go under `build/`, mirroring the source
tree; set `BUILD_DIR=...` on the command line to put them elsewhere. Run the
targets from the repository root, since the inputs are referenced as
`apps/input/<set>/*.yaml`.

`make test` ends with `build/tests/test_benchmarks --cook-ratio-gate`, which asserts
the plan's requirement that the Cook's membrane corner displacement converge
with successive differences shrinking by at least 3x per uniform refinement.
That threshold is not met (measured ratios 2.34, 2.45, 2.31): uniform
refinement is limited by the singularity at the 108-degree clamped-free
corner, so the point value converges at roughly h^1.3. Everything before that
final step is green; the threshold is kept as written rather than relaxed.

### Running Cook's membrane

```
./build/apps/solid_mechanics -i apps/input/finite_elasticity/cooks_membrane/cook.yaml
mpirun -np 4 ./build/apps/solid_mechanics -i apps/input/finite_elasticity/cooks_membrane/cook.yaml
```

Plane strain, NeoHookean with E = 250, nu = 0.3, left edge clamped, uniform
upward shear traction of 3.75 per unit reference length on the right edge
(resultant 60). The run prints the Newton log, `|u|_L2`, the internal energy,
and the probe at the top-right corner (48, 60); the frozen regression value
on the 64x64 p = 2 mesh is uy = 4.905891700497 (30.7% of the 16 mm edge).
ParaView output goes to `out/cook/cook` (`displacement` for Warp by Vector,
`vonmises`, `jacobian`), one cycle per load step. The 3D cantilever of the
linear-limit test runs the same way from `apps/input/finite_elasticity/verification/euler_bernoulli_cantilever3d.yaml`.

The incompressible variant of the same benchmark (mixed u-p formulation,
isochoric neo-Hookean with mu = 80.194, resultant 100, the pressure a
Lagrange multiplier) is `apps/input/finite_elasticity/cooks_membrane/cook_incompressible.yaml`; its frozen
value on the 32x32 Q2-Q1 mesh is corner uy = 6.930412595013 (43.3% of the
edge), and `cook_nearly_incompressible.yaml` is the same problem with
nu = 0.4999 (kappa = 4.0e5). Both add a `pressure` output field.

### YAML schema

```yaml
formulation: displacement         # displacement (default) | mixed (u-p, needs mesh.order >= 2)
plane: strain                     # 2D only: strain (F33 = 1, default) | stress (F33 = thickness stretch with
                                  # sigma33 = 0; displacement formulation, incompressible models allowed)
mesh:
  file: apps/mesh/cook.msh        # Gmsh .msh (2.2 or 4.x, ASCII or binary) or any format MFEM reads;
                                  # physical groups become the element/boundary attributes and their
                                  # names can be used in bcs.*.attr (see "Meshes" below)
  perturb: 0.0                    # interior-vertex jitter of the base mesh, fraction of h, in [0, 0.25)
  serial_refine: 1                # uniform refinements of the file mesh
  parallel_refine: 0
  order: 2                        # H1 polynomial degree (independent of the geometry order of the file)
material: { model: neo_hookean, E: 250.0, nu: 0.3, rho0: 1.0 }
  # rho0: reference density (body force per unit mass times rho0; the mass matrix of a dynamic analysis).
  # neo_hookean, st_venant_kirchhoff: coupled compressible models, keys E, nu (nu < 0.5); displacement only
  # gent_compressible_summit: SUMMIT's compressible Gent model, keys mu, kappa, Jm; coupled, displacement only:
  #   W = -mu/2 (Jm ln(1 - (I1 - 3)/Jm) + 2 ln J) + kappa/2 ((J^2 - 1)/2 - ln J)^4. kappa scales a quartic
  #   penalty, not the small-strain bulk modulus (the linearised moduli are mu and lambda = 2 mu / Jm).
  # Decoupled (isochoric-volumetric) models, either formulation:
  #   iso_neo_hookean: mu (or E, nu)
  #   mooney_rivlin:   c1, c2                   (mu = 2 (c1 + c2))
  #   yeoh:            c10, [c20, c30]          (mu = 2 c10)
  #   gent:            mu, Jm                   (I1bar - 3 < Jm)
  #   arruda_boyce:    mu, N, [inverse_langevin] (mu = n k T; inverse_langevin: pade (default, Cohen's
  #                    L^-1(z) = z (3 - z^2)/(1 - z^2), z^2 = I1bar/(3N); N > 1, locks at I1bar = 3N,
  #                    small-strain modulus mu (3N - 1)/(3N - 3)) | series (five terms in I1bar/N,
  #                    small-strain modulus mu (1 + 3/(5N) + ...)))
  #   ogden:           mu_r: [..], alpha_r: [..] (up to 6 terms, mu_r alpha_r > 0; mu = 1/2 sum mu_r alpha_r)
  #   All take the bulk modulus from exactly one of kappa | nu (nu = 0.5 -> incompressible) |
  #   incompressible: true; finite kappa works in either formulation (penalty U(J) in the
  #   displacement formulation), kappa = inf needs formulation: mixed.
  #   volumetric: quadratic (default) | simo_taylor | logarithmic | j_log_j selects the volumetric
  #   law U(J) = kappa u(J) of a decoupled model at finite kappa: u = (J - 1)^2/2 |
  #   (J^2 - 1 - 2 ln J)/4 | (ln J)^2/2 (p = kappa ln J / J, Anand's FEniCSx codes) | J ln J - J + 1.
  #   Every law has u''(1) = 1, so kappa keeps its meaning; they differ at finite strain.
  #   Closed forms and homogeneous solutions: doc/verification_manual.tex, Appendix A.
  # linear_elastic: small-strain (geometrically linear) isotropic elasticity, either formulation;
  #   keys mu (or E, nu) and one of kappa | nu | incompressible, as for iso_neo_hookean, no volumetric
  #   law: sigma = 2 mu dev(eps) + kappa tr(eps) I with eps = sym(grad u). nu = 0.5 needs
  #   formulation: mixed or plane: stress. type: follower_pressure and solver.predictor: tangent are
  #   input errors for it. See "Small-strain linear elasticity" below.
  # regions: [ { attr: [inclusion, 3], mu: 2800.0, kappa: 2800000.0 } ]
  #   Other parameters of the same model by element attribute (physical-volume names and/or
  #   numbers); keys not given are inherited from the base, a region giving any of kappa | nu |
  #   incompressible replaces the base's bulk specification, every region must be incompressible
  #   or none, and no attribute may be covered twice. The base applies everywhere else. A region's
  #   rho0 is its density (body force and mass).
bcs:
  dirichlet: [ { attr: [left], expression: ["0", "0"] } ]      # attr: physical-group names and/or numbers;
  traction:  [ { attr: [right], expression: ["0", "3.75"] } ]  # expression: one string f(x, y, z, t) per
                                                               # component in the reference coordinates
  # (e.g. ["x", "-0.5*y"] for an affine stretch); nominal traction per unit reference area (dead load).
  # Dirichlet entries may add components: [x, z] (or 0-based indices) to prescribe a subset (rollers,
  # symmetry planes). Tractions may set type: pressure (dead, T = -p N; a single string) or
  # type: follower_pressure (T = -p J F^-T N per current area). Every entry may add
  # schedule: { type: ramp, from: 0.0, to: 1.0 } | { type: constant } | { type: table, t: [..], s: [..] }:
  # its data is schedule(t) * f over the pseudo-time t in [0, 1]; the default is the ramp unless f
  # mentions t (then constant). See "Boundary conditions and loading" below.
body_force: { expression: ["0", "0"], schedule: { type: ramp } }   # per unit mass; rho0 * b enters the
                                                                   # weak form; omit for none
dynamics:                         # optional. Absent: quasi-static, t is a pseudo-time in [0, 1]. Present: the
  t_final: 2.0e-2                 # inertial term joins the weak form and t is the physical time ("Dynamics" below)
  dt: 1.0e-5                      # or steps: [ { to: 5.0e-3, n: 1000 }, { to: 2.0e-2, n: 300 } ]; a dt that does
                                  # not divide t_final is shortened to t_final / n
  scheme: newmark                 # newmark (beta: 0.25, gamma: 0.5, the trapezoidal rule) | hht (alpha in
                                  # [0, 1/3]) | generalized_alpha (rho_inf in [0, 1]: 1 no dissipation, 0 annihilation)
  initial: { displacement: ["0", "0"], velocity: ["0", "1.5*x"] }   # expressions f(x, y, z); default zero
  # With this block: schedules live on [0, t_final], an entry without a schedule is constant (its data
  # is the expression; a step load if that does not mention t, on from t = 0), and solver.load_steps,
  # solver.steps and solver.predictor are errors (substep stays; min_dt is then a time).
solver:
  load_steps: 1                   # equal increments of t; or steps: [ { to: 0.5, n: 2 }, { to: 1.0, n: 4 } ]
  predictor: none                 # none | tangent: start each increment from a linear solve about the last
                                  # converged state with the Dirichlet increment imposed on the update
  substep: { on_failure: false, max_bisections: 4, min_dt: 1e-4 }   # halve a failed increment and retry
  newton:  { rtol: 1e-10, atol: 1e-12, max_it: 25, armijo_c: 1e-4, max_halvings: 8, print_level: 1 }
  linear:  { type: gmres_amg, amg: elasticity, rtol: 1e-12, atol: 0.0, max_it: 500, krylov_dim: 50, print_level: 0,
             inner_rtol: 1e-3, inner_max_it: 50, augmentation: 1.0 }
                                  # type: gmres_amg | cg_amg | direct; amg: elasticity | systems
                                  # direct: sparse LU (MUMPS through PETSc) of every Newton system, either
                                  # formulation, serial or parallel; the other keys of `linear` are then
                                  # unused. Much the fastest choice up to some 1e5 unknowns (the iterative
                                  # solvers are meant for problems too large to factor); needs an MFEM
                                  # built with MFEM_USE_PETSC and a PETSc with MUMPS
                                  # inner_*, augmentation: mixed formulation only (see below)
output:
  paraview: out/cook              # empty or absent -> no files
  fields: [displacement, vonmises, jacobian]   # nodal unknowns: displacement, pressure (mixed), and in a dynamic
                                  # analysis velocity, acceleration; quadrature
                                  # quantities: cauchy_stress (6: xx yy zz xy yz xz), pk1_stress (9, row-major),
                                  # deformation_gradient (9), strain (6: Green-Lagrange; the infinitesimal
                                  # strain for linear_elastic), jacobian, vonmises, energy_density,
                                  # thickness_stretch (plane stress); in 2D the out-of-plane terms are included
  quadrature_at: [nodes]          # presentations of the quadrature quantities: nodes (continuous field <name>),
                                  # elements (element average <name>_elem), quadrature_points (point cloud <name>_qp)
  nodal_projection: averaged      # averaged (element projection, mean at shared nodes) | projected (global L2)
  high_order: true
  probes: [ { name: top_right_corner, point: [48.0, 60.0] } ]   # every registered field printed at these points
  probe_every_step: false         # also after every load step, on lines prefixed "step k t = ..."
  reactions: false                # force and moment of every Dirichlet entry (bcs.*.name labels them) after every
                                  # step, in the log and, with paraview output, in <paraview>/reactions.csv; in a
                                  # dynamic analysis of the balance with inertia
  every: 1                        # ParaView stride: every n-th step (and always the last); per-step lines stay
  energy: false                   # dynamic analysis: per step, kinetic and internal energy, external work, balance
```

Unknown keys, missing required keys, and wrong types raise an error naming
the full key path (for example `key 'material.E' expected a number, got
'abc'`).

### Boundary conditions and loading

Every boundary condition and the body force is a load entry with its own
data and its own schedule in a pseudo-time `t` that the stepper advances
from 0 to 1 (`doc/bc_loading_plan.md` is the design record,
`doc/theory_manual.tex`, "Boundary conditions and loading", the formulation).

**Data.** An entry gives `expression`: one string per component,
`f(x, y, z, t)` in the reference coordinates and the pseudo-time (a single
string for pressures). Constants are strings too (`["0", "3.75"]`), and an
affine stretch is `["x", "-0.5*y"]`. The expression grammar is numbers, `x y z t` (`z` is
0 in 2D), `pi`, the operators `+ - * / ^` (`^` binds tightest and
associates to the right, so `2^3^2 = 512` and `-x^2 = -(x^2)`),
parentheses, the functions `sin cos tan exp log sqrt abs pow(a, b) min(a, b)
max(a, b)`, and `if(cond, a, b)` with the comparisons `< <= > >= == !=`
(1 or 0; comparisons do not chain). Parse errors name the entry and the
column. Expressions cannot refer to the solution or to the current
position; a load that depends on the deformation is a follower load, and
only the follower pressure below is provided.

**Schedules.** The data of entry `i` enters as `s_i(t) * data`. `schedule:
{ type: ramp, from: 0, to: 1 }` (the default) is 0 before `from`, 1 after
`to`, linear between; `{ type: constant }` is 1 for `t > 0`, i.e. the load
is applied in full at the first step; `{ type: table, t: [..], s: [..] }`
interpolates linearly and clamps outside. The default is the ramp, unless
the expression mentions `t`, in which case it is the constant schedule and
`t` enters through the function only; if a schedule is given as well the
two multiply. With the ramp on every entry, `t < 1` solves a proportionally
scaled problem and `t = 1` the problem of the weak form. A
prestress-then-stretch path is two entries with ramps over `[0, 0.5]` and
`[0.5, 1]`; load-unload is a table `s: [0, 1, 0]`. Two Dirichlet entries may
overlap on shared edges or corners: the later entry wins there, and the
physics prints a warning when two entries prescribe the same component on
the same attribute. Dead loads whose expression mentions `t` are
reassembled at every step; the others once.

**Components.** A Dirichlet entry with `components: [y]` (names or 0-based
indices) prescribes only those components; the others on that boundary are
free (natural). Rollers and symmetry planes are then one line each, e.g.
`apps/input/finite_elasticity/verification/homogeneous_deformations/symmetry_uniaxial_neo_hookean.yaml` (an octant of
the uniaxial cube with three rollers) and `apps/input/finite_elasticity/verification/rivlin_cylinder_inflation.yaml`
(a quarter annulus). The data is still a full vector; the unlisted
components are simply not applied.

**Pressures.** `type: pressure` is a dead normal pressure, `T = -p N` per
unit reference area on the reference outward normal. `type:
follower_pressure` is a pressure per unit current area, `T = -p J F^-T N`,
which depends on the displacement and enters the nonlinear form as a
boundary face term with its own (non-symmetric) tangent by dual numbers;
`solver.linear.type: cg_amg` is refused for such inputs. The two coincide
at `F = I` and differ at second order in `p`. `rivlin_cylinder_inflation.yaml`
reproduces the closed-form inflation of a thick-walled incompressible
cylinder (Rivlin) to 1e-6 on the 4 x 8 mesh with one refinement.

**Reactions.** `output.reactions: true` prints, after every step and at the
end, the resultant force and the moment about the origin (at the current
positions) that each Dirichlet entry exerts on the body, on lines
`reaction <name>: force = fx fy fz moment = mx my mz` (`name` is the
optional label of the entry, default `dirichlet[i]`). The reaction is the
residual on the entry's essential degrees of freedom, internal minus external
nodal forces, which is the exact discrete counterpart of the traction
integral over the constrained face: on the uniaxial cube it equals P_11 times
the area to round-off and on Cook's membrane the clamped edge carries the
applied resultant to 1e-14. Two entries sharing nodes both count the shared
nodal forces. With ParaView output the same reactions go to
`<output.paraview>/reactions.csv`, one row per accepted step from the initial
state on (columns `step, t, <name>_fx, _fy, _fz, _mx, _my, _mz` per entry), which
lines up with the cycles of the `.pvd` when `output.every` is 1 and is what
post-processing should read; a stress field integrated over the face, whether the
nodal `pk1_stress` of the output or the finite element stress at quadrature
points, is off by up to 10 percent where the face meets free faces or clamped
corners (see the finite elasticity examples below).

**Predictor.** By default an increment starts from the last converged state
with the new Dirichlet values written on the boundary, so the interior lags and
the first Newton linearisation is about a state whose boundary layer of elements
carries the whole increment (after that iterate J can range from 0.4 to 3 in a
sheared block). `solver.predictor: tangent` instead performs one linear solve
about the converged state with the increment imposed on the update,
`x = x_n + d - J(x_n)^-1 (R(x_n) + J(x_n) d)`, which is how codes that put the
boundary condition on the Newton update behave. The converged states are the
same; Newton typically needs half the iterations and no damped steps (Cook's
membrane with a prescribed edge: 18 -> 9). It matters most for nearly
incompressible materials and is required in practice for the logarithmic
volumetric law, whose pressure ln J / J is not monotone beyond J = e and whose
stiffness varies strongly over the J range of a lagging-interior iterate. It is
a no-op for traction-driven increments. `J(x_n) d` is formed by a forward
difference of the residual.

**Steps and recovery.** `solver.load_steps: N` takes N equal increments of
`t`; `solver.steps: [ { to: 0.5, n: 2 }, { to: 1.0, n: 4 } ]` takes 2
increments to `t = 0.5` and 4 more to 1 (the segments must end at 1).
`solver.substep: { on_failure: true, max_bisections: 4, min_dt: 1e-4 }`
restores the last converged state when a Newton solve fails, halves the
increment and retries (at most `max_bisections` times and never below
`min_dt`); after a converged reduced increment the stepper aims at the
planned breakpoint again and never grows beyond the planned grid. The
report lists the failed attempts and the bisection count; the app prints
`load step k/n: t = a -> b` per increment and `probe_every_step: true`
prints every probe after every step with its `t`.

**Programmatic use.** `AddDirichlet`, `AddTraction`, `AddPressure(attrs, p,
follower)` and `SetBodyForce` take any MFEM coefficient (the tests use
`AffineVectorCoefficient` and function coefficients) and an optional
`BCOptions{components, schedule, time_dependent}`; coefficients with a
`(X, t)` callback work because `SetLoadFactor(t)` calls `SetTime(t)` on every
coefficient (set `time_dependent` so that a dead load is reassembled per
step). The YAML schema itself has only the expression form. All of the
above lives in `physics/loads.{hpp,cpp}` (`LoadSet`), shared by both
formulations; `tests/test_loading.cpp` exercises every feature (staged
loading is path independent, load-unload returns to zero, bisection recovers
a capped Newton, a manufactured solution written entirely as YAML
expressions converges at third order, the symmetry cube reproduces the
closed form, the follower tangent matches finite differences, the cylinder
inflates as Rivlin says).

Not supported: point loads and nodal constraints (use a small physical
group), multi-point or periodic constraints, contact, automatic step growth
after a bisection. In a quasi-static analysis `t` is a pseudo-time; with a
`dynamics:` block it is the physical time (see "Dynamics" below).

### Verification cases (`apps/input/finite_elasticity/verification/`)

Every input here is compared with an independent reference by a test
(`tests/test_verification`, part of `make test`, or `tests/test_loading` and
`tests/test_benchmarks` where noted). The headers of the inputs state the
references and the formulas.

| Input | Reference | Check |
|-------|-----------|-------|
| `rivlin_torsion.yaml` | Rivlin's universal torsion of an incompressible neo-Hookean cylinder | displacement field to 2e-3, Cauchy stresses, pressure and von Mises on the mid-plane to 2-3% (curved P2 tetrahedra) |
| `rivlin_cylinder_inflation.yaml` | Rivlin's plane-strain inflation of a thick tube (test_loading) | inner and outer radius to 1e-6, error decreasing under refinement |
| `green_zerna_sphere_inflation.yaml` | Green-Zerna inflation of a thick incompressible sphere | inner and outer radius to 0.5%, spherical symmetry |
| `euler_bernoulli_cantilever3d.yaml` | Euler-Bernoulli beam in the small-load limit (test_benchmarks) | tip deflection to 0.15% |
| `euler_column_buckling.yaml` | Euler load of a clamped-clamped column, imperfect geometry | Southwell fit of the mid-height deflection recovers the critical strain to 5% |
| `kirsch_plate_with_hole.yaml` | Kirsch's stress concentration, small strain, W = 20 a | Cauchy stresses at the hole and on the axis to 2% |
| `manufactured_solutions/mms_2d_plane_strain.yaml` | manufactured solution, St. Venant-Kirchhoff (test_loading) | third-order L2 convergence for p = 2, body force as YAML expressions |
| `manufactured_solutions/mms_3d_{hex,tet}.yaml` | manufactured solution in 3D, St. Venant-Kirchhoff | third-order L2 convergence on hexahedra and on tetrahedra |
| `manufactured_solutions/mms_3d_mixed.yaml` | isochoric manufactured solution of the mixed formulation | third-order L2 convergence, exact pressure zero |
| `homogeneous_deformations/*_neo_hookean.yaml` | closed forms of doc/verification_manual.tex, Appendix A (apps/homogeneous_compare.py) | every probed quantity to 1e-8 |
| `homogeneous_deformations/compressible_uniaxial_*.yaml` | lateral stretch from P_22 = 0 with the material's own PK1 (mu = 0.5, nu = 0.45 throughout; one decoupled input per volumetric law); `apps/uniaxial_plots.py` and `apps/neo_hookean_compare.py` drive the uniaxial inputs to a stretch of 8 and plot P_11 on independently coded analytical curves | displacements, P_11, sigma_11 and J to 1e-7 (scripts: about 1e-12) |

The torsion input needs 20 increments: a larger first increment leaves the
elements under the rotated end face inverted before Newton starts (the
boundary layer caution of the plane-stress section), which bisection would
also recover from.

### Small-strain linear elasticity (`model: linear_elastic`, `apps/input/linear_elasticity/`)

It is not a new weak form. Because the stress is symmetric,
`int sigma(eps(u)) : eps(w) = int sigma : Grad w`, which is the total Lagrangian form
`int P(F) : Grad w` with the flux `P(F) = sigma(sym(F - I))`. `linear_elastic` is therefore a
material of the existing kernels: `TotalLagrangianIntegrator` assembles the classical `B^T C B`
operator (it equals `mfem::ElasticityIntegrator` to 5e-16, and the tangent at u = 0 of the
compressible hyperelastic models), the tangent is constant, and the first Newton step is the
exact linear solve.

What depends on the kinematics lies outside the flux and is switched by a compile-time trait of
the material (`materials/kinematics.hpp`; the plane-stress adapter inherits it from its base):

| | finite strain | small strain |
|---|---|---|
| `cauchy_stress`, `vonmises` | `J^-1 P F^T` | `P` (= `pk1_stress`, symmetric) |
| `jacobian` | `det F` | `1 + tr(eps)` |
| `strain` | Green-Lagrange `(F^T F - I)/2` | `eps = sym(F - I)` |
| mixed formulation | `P_iso + p J F^-T`, constraint `u'(J) - p/kappa` | Herrmann: `2 mu dev(eps) + p I`, constraint `tr(eps) - p/kappa` |
| plane stress, incompressible | `lambda_3 = 1/det F_2D` | `eps_33 = -(eps_11 + eps_22)` |
| follower pressure | a load of its own | the dead pressure; rejected as an input error |
| reaction moments | arms `X + u` | arms `X` |

Plane strain is the padded `F33 = 1`; under plane stress the compressible branch of the adapter
applies unchanged (its scalar Newton iteration is exact after one step) and returns
`E/(1 - nu^2)`, `E nu/(1 - nu^2)`. With `formulation: mixed` the same block kernel is Herrmann's
linear, symmetric saddle-point problem, valid up to `nu = 0.5`; at `nu = 0.4999` on the Lame
cylinder the displacement formulation is off by 50% with p = 1 and by 2.8e-4 with p = 2, the
mixed Q2-Q1 by 4e-6.

Solving a linear problem:

- One Newton iteration = one assembly, one AMG setup, one Krylov solve (the mixed formulation
  usually takes a second step, which refines the first). `cg_amg` is valid (SPD).
- Over a load path the Jacobian is assembled once and the solver's setup (the AMG hierarchy; for
  the mixed formulation also the augmented blocks) is built once: the matrix depends neither on
  the state nor on the pseudo-time, so `GetGradient` returns the same matrix until the boundary
  conditions change, and the physics shares an operator stamp with the solver it makes so that
  the solver knows. Bit-identical to reassembling; assembly + setup over 10 steps drops 7-9x
  (1.9 -> 0.2 s on the Lame sphere). The Krylov solves remain, and they dominate: about 1.5x
  overall for the displacement formulation, next to nothing for the mixed one.
- `newton.rtol` only accepts the linear solve; the accuracy is set by `linear.rtol`. The inputs use
  `1e-8` and `1e-13`. The reason is the round-off floor of the residual, `eps |K| |u| / |f|`
  relative to `|R0|`, which bending puts at 4e-11 (Cook), 8e-11 (the 10:1 cantilever) and 1.6e-9
  (a 20:1 beam), at or above the default `rtol` of 1e-10, however tight the Krylov tolerance. A
  nonlinear problem overshoots such a tolerance by quadratic convergence; a linear one lands on
  the floor and stays. A problem that declares itself linear (`QuasiStaticProblem::IsLinear`) is
  therefore accepted at the floor: Newton prints "the residual is at its round-off floor" instead of
  reporting a failed line search.
- `solver.predictor: tangent` is refused: after an exact predictor Newton would start at the floor.
- The solution scales with the load, so choose amplitudes with `|grad u| >= 1e-4` and scale the
  results: forming `F = I + H` and subtracting `I` again leaves a relative floor of
  `1e-16 / |grad u|` in the stress. (The hyperelastic "linear limit" inputs, which must keep
  `|grad u|` small, do not have that freedom.)

Cases, all checked by `tests/test_linear_verification` (`make test`, about 30 s):

| Input | Reference | Check |
|-------|-----------|-------|
| `verification/lame_cylinder.yaml` | Lame's thick-walled cylinder, plane strain, dead pressure | u_r to 1e-5 (the floor of the second-order arcs), wall stresses and strains converging with h^2 (ratios 3.90, 4.04, 4.02) |
| `verification/lame_cylinder_incompressible.yaml` | the same at nu = 0.5, mixed Q2-Q1 | u_r to 1e-5, the pressure unknown equal to the constant `p a^2/(b^2 - a^2)` to 1e-5 in the wall; locking record at nu = 0.4999 |
| `verification/lame_sphere.yaml` | Lame's thick-walled sphere | u_r to 3e-4, stresses to 1% |
| `verification/kirsch_plate_with_hole.yaml` | Kirsch, plane stress, sigma_0 = 1 | stresses to 2%; within 5e-4 of the neo-Hookean input run at a strain of 1e-4 |
| `verification/euler_bernoulli_cantilever3d.yaml` | Euler-Bernoulli at 1000 times the load of the finite-strain input | tip to 0.2%; compliance equal to the small-load neo-Hookean one to 1e-7 |
| `verification/manufactured_solutions/mms_{2d_plane_strain,3d_hex,3d_tet}.yaml` | exact field, body force `-[(lambda + mu) grad(div u) + mu lap(u)]` written out | third-order L2 convergence for p = 2 |
| `verification/spherical_inclusion.yaml` | two materials by `regions`: Reuss and Voigt bounds on the apparent modulus | bounds; reaction vs energy to 1e-9; regions with the base's parameters give the homogeneous field |
| `cooks_membrane/cook_linear.yaml` | Cook's membrane as posed (plane stress, E = 1, nu = 1/3, unit load): literature 23.96 at the mid-point (48, 52) of the free edge | 23.9650 (within 1%), corner 25.1640; both frozen to 1e-8 |
| `cooks_membrane/cook_linear_incompressible.yaml` | the incompressible plane-strain membrane, mixed Q2-Q1 | corner 19.4176 frozen; the displacement formulation at nu = 0.4999, p = 2 within 2% |

Not supported: anisotropic linear elasticity, thermal or shrinkage eigenstrains (materials cannot
read a field yet), modal analysis, linear buckling (needs a geometric
stiffness), small-strain plasticity or viscoelasticity (internal variables), and a factorisation
or an initial guess carried over the load steps of a linear problem (every step is a fresh Krylov
solve with the one hierarchy).

### Dynamics (`dynamics:` block, `apps/input/dynamics/`, `apps/dynamics_compare.py`)

A `dynamics:` block adds the inertial term to either weak form,

    int rho_R u_tt . w dV + R(u; w) = 0,     u(0) = u_0,  u_t(0) = v_0,

for every material, both formulations, 2D and 3D, with every load, region and output of the
quasi-static analysis. Without the block nothing changes, to the last digit. In total Lagrangian
form the term is integrated over the fixed reference mesh with the reference density, so the mass
matrix is constant: it is assembled once (consistent mass, `rho0` by region), and with the
displacement of the new time level as the unknown a time step solves

    S(u, t_{n+1}) + c_M M (u - u*) + h_n = 0,     Jacobian K(u) + c_M M,

where `S` and `K` are the residual and the Jacobian of the quasi-static problem: inertia is a
linear spring `c_M M` plus a known history load. It is therefore a decorator over the problem
(`physics/dynamic_solid_problem.hpp`), not a second set of kernels, and Newton, the line search,
the linear solvers, prescribed displacements, bisection, reactions and outputs are the ones
described above. A linear problem forms `K + c_M M` and builds its AMG hierarchy once per run.

| `scheme` | keys | what it is |
|---|---|---|
| `newmark` (default) | `beta: 0.25`, `gamma: 0.5` | the trapezoidal rule: second order, no numerical dissipation; conserves the energy of a linear problem exactly and lengthens a period by `(w dt)^2 / 12`. Not unconditionally stable for nonlinear problems, and what the mesh cannot resolve behind a wave front rings for ever |
| `hht` | `alpha` in [0, 1/3] | Hilber-Hughes-Taylor, `rho_inf = (1 - alpha) / (1 + alpha)` |
| `generalized_alpha` | `rho_inf` in [0, 1] | Chung-Hulbert: second order, with the spectral radius `rho_inf` at infinite frequency. 0.8 to 0.9 removes what the step does not resolve at little cost to what it does; the choice for finite strain and **required in practice for `formulation: mixed`** |

Choosing `dt`: about 20 steps per period of interest keep the period error near one percent
(`(w dt)^2 / 12`); for a wave, the time it needs from one node to the next. Units are the
user's: with `E` in Pa and `rho0` in kg/m^3, lengths are metres and `dt` seconds.

What differs from a quasi-static input:

- `t` is the physical time. Ramps and tables are given on `[0, t_final]`, and **an entry without
  a `schedule` is constant**: its data is the expression as it stands, a step load when the
  expression does not mention `t`, on from `t = 0` (it enters the initial acceleration). The
  quasi-static default, a ramp over the pseudo-time, has no meaning here; because this default
  depends on the analysis, the run header lists the time dependence of every entry:

      dynamics: generalized-alpha (rho_inf 0.8), t_final 8, 1600 time steps of 0.005
        dirichlet wall: constant in time (on from t = 0)
        traction end_load (vector): constant in time (on from t = 0)

- The initial state is `dynamics.initial` (zero by default); the initial acceleration is the
  consistent one, `M a_0 = -S(u_0, 0)`. Dirichlet data at `t = 0` overwrite `u_0` on their dofs.
- Reactions are those of the balance with inertia: a support force includes the inertia it carries.
- `output.energy: true` prints after every step `kinetic`, `internal`, `external_work` (dead loads
  and supports, trapezoidal rule) and their `balance`. For a linear problem and the trapezoidal
  rule the balance is zero to round-off; it is an identity of the scheme and a sharp test of an
  implementation. `velocity` and `acceleration` are nodal fields; `output.every: n` thins the
  ParaView output while the probe, reaction and energy lines stay per step (time histories are read
  from the log: `apps/dynamics_compare.py`).
- Mixed u-p: the pressure carries no inertia, a differential-algebraic system. An error in the
  pressure returns at every step with the factor `-rho_inf` (measured: 0.6000 for `rho_inf: 0.6`),
  so a scheme without dissipation keeps it for ever and the header warns. The pressure also carries
  the tolerance of the solves multiplied by `c_M = O(1 / dt^2)`: tighten `newton.rtol` and
  `linear.rtol` when the pressure matters. The saddle-point solver gains the inertial part of its
  Schur complement approximation (without it the outer iterations grow from 18 to 81 on a refined
  annulus as `dt` falls; with it they are 6 to 16).

| input | reference | checked (`tests/test_dynamic_verification`, `apps/dynamics_compare.py --check`) |
|---|---|---|
| `bar_free_vibration.yaml` | first axial mode `A sin(pi X / 2L) cos(w_1 t)`; dispersion of the trapezoidal rule | tip to 6e-4 A over five periods, energy to 7e-14; period elongation 8.135e-3 against 8.171e-3 |
| `bar_step_load.yaml` | d'Alembert: triangle wave of the tip between 0 and `2 p L / E`, square wave of the wall reaction | peak 1.9895, mean 1.0001 of the static deflection; front at mid-span at 0.5001; ringing 0.18 % of `2 p A` with `rho_inf = 0.8` against 1.62 % with the trapezoidal rule |
| `cantilever_vibration.yaml` | Euler-Bernoulli `w_1 = 1.875104^2 sqrt(E I / (rho A L^4))` | 0.17 % below it |
| `mms_dynamic_2d.yaml`, `mms_dynamic_3d.yaml` | manufactured `sin(w t) U(X)` | rates 3.00-3.02 in h; ratios 4.03, 4.01 in dt |
| `neo_hookean_block_vibration.yaml` | none: self-convergence, energy | ratios 3.92, 3.98; energy never above its initial value (`rho_inf = 0.8`), error falling by 4.00 per halving (trapezoidal rule) |
| `knowles_tube_oscillation.yaml` | Knowles' equation for the radial oscillation of an incompressible neo-Hookean tube, mixed u-p | inner radius to 2.3e-3 of its amplitude over two periods, period 4.2431 against 4.2428, pressure at mid-wall to 2.0e-3 |

`make dynamics` draws the histories over the closed forms into `out/dynamics/`.

Not supported: explicit time integration (central differences with a lumped mass), physical
(Rayleigh) damping, energy-momentum conserving schemes, time-step adaptivity by an error estimate
(a step is only halved when Newton fails), a static preload followed by a dynamic release (start
from `initial.displacement` or a pulse), eigenfrequencies and mode shapes, absorbing boundaries,
restart.

### The elastic bar exercise (`apps/input/elastic_bar/`, `apps/elastic_bar_compare.py`)

The exercise of `myapps/elastic_bar`: a 0.3 x 0.3 x 1 bar clamped on `z = 0`, the face `z = 1`
pulled to `u_z = 3` (a stretch of 4) with free lateral motion, the reaction force recorded
against the end displacement, once for linear elasticity and once for a compressible Gent
material (mu = 5e6, kappa = 1.5e9, Jm = 50). There it takes two drivers (a bilinear form with
MFEM's `ElasticityIntegrator`, a nonlinear form with a hyperelastic model). Here it takes no
driver at all: the two bars are inputs of `build/apps/solid_mechanics` that differ in one line,

```yaml
material: { model: linear_elastic, mu: 5.0e6, kappa: 1.5e9 }                       # bar_linear.yaml
material: { model: gent_compressible_summit, mu: 5.0e6, kappa: 1.5e9, Jm: 50.0 }   # bar_gent.yaml
material: { model: iso_neo_hookean, mu: 5.0e6, kappa: 1.5e9 }                      # bar_neo_hookean.yaml
```

(the third is the exercise's alternative model, MFEM's `NeoHookeanModel`). `output.reactions`
prints the force of the Dirichlet entry `front` after every load step and the probe
`pulled_face` its displacement. The mesh is the exercise's own file, so results compare degree
of freedom for degree of freedom.

```
python3 apps/elastic_bar_compare.py                      # or: make elastic_bar   (about a minute)
python3 apps/elastic_bar_compare.py --cases linear gent neo_hookean --paraview
python3 apps/elastic_bar_compare.py --order 2 --steps 20 --out out/elastic_bar_p2
```

runs the cases, writes `out/elastic_bar/<case>/force_displacement.csv` in the exercise's format
(`step,disp_z,total_Fz`; its `plot_force_displacement.py` reads them too), prints a table and
draws `out/elastic_bar/force_displacement.png`: the curves, the homogeneous uniaxial-stress
estimate of each model (lateral stretch from `P_22 = 0`), and the results of the exercise's own
drivers (`apps/input/elastic_bar/reference/`). Measured (order 1, 100 steps):

| | F/u, first step | F at u_z = 1 | u_z = 2 | u_z = 3 | against the exercise's driver |
|---|---|---|---|---|---|
| linear elastic | 1.5232e6 | 1.5232e6 | 3.0465e6 | 4.5697e6 | 2.6e-11 (its Krylov tolerance tightened from 1e-8 to 1e-14; 1.6e-5 as it stands) |
| compressible Gent | 0.9051e6 | 0.8163e6 | 1.5707e6 | 2.6332e6 | 9.9e-13 (its stored 500-step curve) |
| neo-Hookean | 1.4760e6 | 0.8641e6 | 1.4162e6 | 1.9138e6 | - |

Two things the comparison shows. The Gent curve starts with a lower slope than the linear bar
of the same mu and kappa, because kappa scales a quartic penalty there and does not enter the
small-strain response (lambda = 2 mu / Jm); the neo-Hookean curve, whose small-strain moduli
are the linear ones, leaves the origin along the linear curve. And the linear bar is 13 percent
stiffer than homogeneous uniaxial stress (`E A / H`): with `--order 2` it is 3.3 percent, so
10 of the 13 points are the volumetric locking of linear tetrahedra at nu = 0.4983 and 3 the
clamped end. `make test` runs the script with `--check`, which fails unless both curves agree
with the exercise's to 1e-8.

### The plate-with-a-hole exercise (`apps/input/plate_with_hole/`, `apps/plate_with_hole_compare.py`)

The exercise of `myapps/plate_with_hole`: Kirsch's problem posed as an exact-solution test. A
quarter plate `[0, 3R]^2` with a quarter hole of radius `R = 0.01` at the origin, plane strain,
linear elasticity with mu = 7e10, nu = 0.3; the displacement of the *infinite* plate under the
far-field stress `s = 0.01 mu` is prescribed on the four outer edges and the hole is traction
free, so the finite element solution is that field up to discretisation error. (The case
`linear_elasticity/verification/kirsch_plate_with_hole` pulls a finite 40 x 40 plate by a
traction instead and therefore carries a finite-width error of a few tenths of a percent.) The
exercise is a driver of its own; here it is one input of the general executable,
`apps/input/plate_with_hole/plate_linear.yaml`, whose Dirichlet data are the closed form written
as expressions in `x` and `y` (`cos 2theta = (x^2 - y^2)/r^2`, `sin 2theta = 2xy/r^2`; generated,
and checked against an independent implementation to 1e-13 by the test). The mesh is the
exercise's own file (10900 nodes, straight-sided triangles).

```
python3 apps/plate_with_hole_compare.py                   # or: make plate_with_hole   (a few seconds)
python3 apps/plate_with_hole_compare.py --order 2         # --refine N, --np N, --model neo_hookean
```

runs the input, reads the ParaView output back with pyvista, evaluates the closed form at every
node and prints the errors; it draws `out/plate_with_hole/plate_with_hole.png`, the stress
`sigma_xx` across the ligament `x = 0` and the hoop stress around the hole against Kirsch's
curves. Measured (root mean square over the nodes; stresses relative to `s`):

| | dofs | displacement | sigma_xx | concentration factor (3) |
|---|---|---|---|---|
| order 1 | 21800 | 5.5e-5 | 2.6e-3 | 2.991 |
| order 1, refined once | 86420 | 2.3e-5 | 9.1e-4 | 2.997 |
| order 2 | 86420 | 1.4e-5 | 2.1e-4 | 2.998 |
| order 1, `neo_hookean` with the same moduli | 21800 | 2.9e-4 | 3.5e-3 | 2.987 |

The displacement error levels off near 1e-5: the hole of this mesh is a polygon whose chords sag
by 1e-4 R, and uniform refinement keeps the polygon (a mesh generated with `gmsh -order 2` would
not have that floor). The last row is geometric nonlinearity: at these strains (0.4 to 1
percent) a finite-strain model differs from Kirsch's linear solution by five times the
discretisation error. `tests/test_linear_verification` holds the gates (displacement L2 error
below 1e-4 at order 1 and 3e-5 at order 2, probed stresses at the hole to 1 and 0.3 percent), and
`make test` runs the script with `--check`.

### Anand's coupled-theories examples (`apps/input/anand_coupled_theories/`)

Inputs after the FEniCSx companion codes of Lallit Anand's *Introduction to
coupled theories in solid mechanics* (Oxford University Press, 2025;
solidmechanicscoupledtheories.github.io, codes by Eric Stewart and Lallit
Anand), one subdirectory per chapter of the site. `finite_elasticity/` holds
the ten "1. Finite Elasticity" examples: Arruda-Boyce with
G0 = 280 kPa, lambda_L = 5.12 (`N = lambda_L^2`) and K = 1000 G0 in kPa and mm, the
reference's Pade approximation of the inverse Langevin function
(`inverse_langevin: pade`, the default, written out in the inputs), the reference's logarithmic volumetric law
p = K ln(J)/J (`volumetric: logarithmic`, with
`solver.predictor: tangent`, see "Predictor" above; `02_simple_shear` and
`08_column_buckling` keep the quadratic law, see below), mixed Q2-Q1 or
P2-P1, the sparse direct solver (`solver.linear.type: direct`, as the reference, which
factors with MUMPS; the largest input has 59 000 unknowns), the same geometry, boundary conditions, load histories and step counts.
Meshes come from `apps/mesh/*.geo` (`make meshes`); the curved ones are
second-order. Every input probes the points of the reference's plots after
every step (`probe_every_step`), so the curves (stress or force vs stretch,
pressure vs displacement) can be read from the log.

| Input | Reference | Notes | Mesh: this code / reference |
|-------|-----------|-------|-----------------------------|
| `01_uniaxial_tension` | 3D01 | 10 mm cube, stretch 7.75 in y, rollers on three planes | 4^3 hexahedra / 2^3 box, 48 tets |
| `02_simple_shear` | 3D02 | 1 mm cube, two sinusoidal cycles of shear strain 1 (`sin(4 pi t)`) | 8 x 8 x 4 hexahedra / the same box, 1536 tets |
| `03_cylinder_torsion` | 3D03 | R = 12.7, L = 25.4, top face rotated by 2.5 rad, `cylinder_torsion.geo` | 4045 curved tets / 3653 straight tets |
| `04_plate_with_hole` | 3D04 | quarter plate 15 x 10 x 1 with a 3 mm hole, stretch 3, `plate_hole.geo` | 1071 curved tets / 2904 straight tets |
| `05_cylinder_inflation` | 3D05 | quarter tube 10/11 x 5 mm, follower pressure to 50 kPa, `tube_quarter.geo` | 3969 curved tets / 600 straight tets |
| `06_sphere_inflation` | 3D06 | octant shell 10/11 mm, follower pressure to 35 kPa, `sphere_octant.geo` | 2606 curved tets / 2597 straight tets, both of size 0.75 |
| `07_cube_footing` | 3D07 | 50 mm cube, follower pressure 1500 kPa on a quarter of the top, `footing.geo` | 4766 tets / 10 x 10 x 6 box, 3600 tets |
| `08_column_buckling` | 3D08 | 1 x 1 x 20 column, imperfection by `perturb_column.py`, shortened by 2.5 mm | 4 x 4 x 100 hexahedra / 4 x 4 x 50 box, 4800 tets |
| `09_spherical_inclusion` | 3D09 | octant of a cube with a ten times stiffer spherical inclusion (`material.regions`), stretch 2, `inclusion.geo` | 1358 tets / 2039 tets |
| `10_column_twist` | 3D10 | 1 x 1 x 3 column, top face turned through 2 pi | 8 x 8 x 32 hexahedra, the same |

Both codes use Taylor-Hood elements (P2-P1 on tetrahedra, Q2-Q1 on hexahedra);
`create_box` of the reference cuts each cell of a box into six tetrahedra, and its
Gmsh meshes are first-order, so its curved boundaries are faceted where ours are
quadratic (`order: 2`). The reference integrates at degree 4, this code at 2p + 3 = 7.

`apps/anand_plots.py` reproduces the result plots of the reference pages
from the ParaView output of these runs (`python3 apps/anand_plots.py [case
...]` from the repository root, after the runs; every case with output by
default; the plots go to `out/anand_coupled_theories/finite_elasticity/plots/`).
It follows the `.pvd`, so it plots the steps of the last run. The probes are
the nodal fields interpolated at the probe points of the input. The reactions
are not in the output; the script integrates the traction `P N` of the nodal
`pk1_stress` over the faces of each Dirichlet entry (the faces of its physical
groups in the `.msh`, Gauss quadrature of the Lagrange interpolants, the
prescribed components only, the moment about the origin with the current
position, as the app's). That stress is recovered from the quadrature points,
and where the loaded face meets free faces or clamped corners the recovered force
is off: +9.7 percent for 04 and -8 percent for 08 at the end of the run, +3.5 for
02, 0.6 for 03, 0.1 for 09 and 10, round-off for the homogeneous cube. So the
script takes the reactions from `<paraview>/reactions.csv` when the run wrote it
(`output.reactions: true`), or else from the app's log beside the output
(`logs/<case>.log`, as `logs/run_set.sh` writes it); `--logs DIR` reads probes and
reactions from the logs alone (`DIR/<case>.log`; the inputs print both after every
step). The reference
overlays no analytical curves; the script adds one where a reference
exists: the homogeneous incompressible Arruda-Boyce response for the
uniaxial and shear blocks, Rivlin's universal torsion for torque
and axial force, the incompressible thick-walled cylinder and sphere
inflation by quadrature, the Euler load for the column, and the matrix-only
curve for the inclusion. `pip`-level dependencies: numpy, scipy and matplotlib,
and pyvista and PyYAML to read the ParaView output.
The reference's own results are overlaid too: `finite_elasticity/reference/<case>.csv`
are the histories the reference's notebooks record while they run (their `timeHist`
arrays: the probed displacement and the force or pressure of each plot, plus the
reaction of the loaded face, the residual summed over its dofs, which the notebooks do
not compute and `reference/scripts/*_rxn_run.py` add), from the notebooks run headless
with dolfinx 0.8.0 (`reference/README.md`; the patched notebooks, run scripts and
timings in `reference/scripts/`; 3D10 records none), so the comparison with the
reference is quantitative and compares reactions with reactions: the notebooks' own
forces are boundary integrals of their finite element stress, which the clamped
corners of 02 and 08 spoil by 2 to 9 percent on their tetrahedral meshes.
`--reference DIR` points elsewhere.
What the plots show, with the reference's values in parentheses: the uniaxial
cube lies on the nearly incompressible homogeneous solution at K = 1000 G and its
reaction equals the reference's to 13 digits at every step (6.030 MPa at the
stretch of 7.75; near locking the stiff Pade response makes the volume change
count, J = 1.057 there, where the incompressible limit is 15 percent above, 6.92
MPa); the sheared block carries 241 kPa at a shear strain of 1, 17 percent less than
homogeneous simple shear (290 kPa), because its lateral faces are free where simple
shear needs tractions, and its two cycles retrace one curve (elastic, no hysteresis;
the reference's reaction is 242.3 kPa at +1 and 240.2 at -1, its tetrahedral mesh
not being mirror-symmetric, while its plotted force, the traction integral of its
finite element stress, is 247 and 262; run on the reference's own mesh, `create_box`
8 x 8 x 4 tetrahedra exported by `reference/scripts/export_box.py`, with the
quadratic law, this code's reaction is +242.33 / -240.59 kPa and the reference's is
the same to five digits (`reference/scripts/3D02_same_mesh_check_run.py`); with
refinement this code converges to 240 kPa, 242.3, 240.9, 240.6, 240.4 over 4, 8, 12,
16 subdivisions); the torsion cylinder lies on Rivlin's curves for torque and
axial force, 1.167 N m and a compressive force of 57.4 N at 2.5 rad (the reference's
reactions 1.155 N m and 56.8 N, 1 percent below at every twist: its first-order mesh
is a faceted cylinder whose end face is a 31-gon with 0.993 of the circle's area and
0.986 of its polar moment); the plate with a
hole carries 1.639 MPa at a stretch of 3 (1.633); the tube goes through the plateau of the Pade model, 28.7 to
31 kPa while the inner wall moves from 3 to 30 mm, on the quadrature curve, and stops
at 40.0 kPa with the inner radius at six times its value (the reference at 37.5 kPa
and 5.7 times); the sphere stops at its limit pressure, 34.2 kPa with 4.3 mm of wall
displacement (the quadrature limit point 34.3 kPa at 4.4 mm; the reference 34.0 kPa
at 3.6 mm); the footing settles 37.97 mm at 1500 kPa (37.80); the buckling column
reaches 7.14 mN at 0.2 mm of shortening, 3 percent above the Euler load of the
clamped column, and rises slowly to 7.67 mN at 2.5 mm along the post-buckling path
(the reference's reaction is 7.16 and 7.70 mN, while its plotted traction integral is
6.83 and 8.00; on the reference's own tetrahedral column this code's reaction and the
reference's agree to four digits with either volumetric law, 7.696 against 7.696 mN
at 2.5 mm, and that value is converged: 4 x 4 x 50, 8 x 8 x 50 and 4 x 4 x 100
tetrahedra all give 7.69; Q2 hexahedra need the 100 elements along the axis, with 50
they give 7.49 and with 8 x 8 x 50, a worse aspect ratio, 7.45; the Euler value uses
E = 3 G and neglects the finite section); the cube with the inclusion carries 0.5791
MPa at a stretch of 2 (0.5792), 13 percent more than the matrix alone from an
inclusion of 6.5 percent of the volume; the twisted column needs a compressive axial
force of 78 mN to keep its length over a full turn (Poynting effect), as does the
torsion cylinder, with a torque of 94.4 mN mm; the reference notebook plots nothing
for this case, but its reactions, added to the notebook on its 8 x 8 x 32 hexahedra,
which are this code's mesh, agree with ours to 0.02 percent at every angle (77.95 mN
and 94.44 mN mm at 2 pi). With the direct solver the runs take 16 s (01) to 12 min (05) on
4 ranks; the reference's notebooks, serial, 3 s (3D01) to an hour (3D10).

Differences from the reference that change the numbers: the reference caps the
relative chain stretch lambda_bar / lambda_L of the Pade form at 0.95 and this code
does not (the cap is a chain stretch of 4.86; the homogeneous cube of 01, the most
stretched case, ends at 4.40); `02_simple_shear` and `08_column_buckling` use
p = K (J - 1) because the logarithmic law fails there, at t = 0.064 of the shear for
any increment size and at 0.8 mm of shortening of the column on either post-buckling
branch: both have clamped faces whose corners are stress singularities that this mesh
and quadrature resolve sharply, the Q1 pressure cannot follow them, J runs away there
(0.7 to 1.5 in the column with the pressure at 25 kPa, p / K = 1e-4, and without
bound in the sheared block), and the constraint ln J / J of that law loses its
definiteness (its tangent K (1 - ln J)/J^2 is zero at J = e), so Newton stalls; with
the quadratic law both converge in 3 iterations per step (see the header of
`08_column_buckling.yaml`, also for why the column needs the reference's 100 steps
with the tangent predictor); hexahedra replace tetrahedra on the boxes; and a failed
increment is bisected instead of ending the run (05 and 06 stop early in the
reference).

### Meshes: the Gmsh workflow

Meshes come from files. The intended workflow is: describe the geometry in
a Gmsh `.geo` file with named physical groups, generate the `.msh`, and
point the input at it with `mesh.file`; the boundary conditions then name
the physical groups they act on. MFEM reads Gmsh 2.2 and 4.x files and
turns every physical group into an attribute numbered by the group's tag,
keeping the names (`$PhysicalNames`) as attribute sets. `bcs.*.attr`
accepts those names next to plain attribute numbers; both are checked
against the mesh when the physics is built, and the error lists what the
mesh provides (`bcs.dirichlet[0]: the mesh has no boundary physical group
named 'lefft' (boundary attributes: 1 (bottom), 2 (right), 3 (top), 4
(left))`). The app prints the element and boundary attributes it found,
with their names, at startup.

The example geometries are in `apps/mesh/`: `square.geo` (unit square, n x n
quadrilaterals, groups bottom/right/top/left), `cook.geo` (Cook's membrane,
same groups; the transfinite mesh of the straight-sided quadrilateral is the
bilinear map the former cartesian + corners input used, so the frozen
benchmark values are unchanged), and `box.geo` (Lx x Ly x Lz hexahedra,
groups bottom/front/right/back/left/top; the unit cube and the 10 x 1 x 1
cantilever beam are the same file with different parameters). Their
physical tags follow MFEM's Cartesian numbering (2D bottom 1, right 2, top
3, left 4; 3D z=0 1, y=0 2, x=L 3, y=L 4, x=0 5, z=L 6), so numeric
attributes from older inputs still mean the same faces. `make meshes`
regenerates the `.msh` files with Gmsh (`gmsh -2/-3 -format msh22
-setnumber ...`); the generated files are kept in the tree so that running
the inputs and the tests does not need Gmsh.

Things to know when writing a `.geo`: once any physical group is defined,
Gmsh writes only the elements that belong to a group, so the domain needs a
surface or volume group and every boundary you will reference needs one
(boundaries without a group get no boundary elements, which is fine for
traction-free faces). A 2D mesh must be planar (z = 0). `serial_refine` and
`parallel_refine` refine the file mesh uniformly; a second-order Gmsh mesh
(`-order 2`) is read as a curved mesh and refines along the curved
geometry, while `mesh.order` stays the finite element order. Mixed
quad/tri or hex/tet meshes are accepted. The Cartesian box and the corner
map are no longer part of the YAML schema (the parser says so if an old
input still has them); the tests keep building boxes programmatically
through `MeshConfig::cartesian`. One MFEM detail the loader works around:
its move constructor and move assignment swap everything except the
attribute sets, so a mesh moved after loading would lose its physical
names; the loader constructs the mesh in place.

### Output: where quantities live and how they are presented

Two kinds of output fields exist (`src/physics/quadrature_fields.{hpp,cpp}`,
shared by both formulations):

- Nodal unknowns, `displacement` and (mixed) `pressure`, are the H1 fields
  the solver computes and are written as they are.
- Quadrature quantities, `cauchy_stress` (sigma = J^-1 P F^T as xx, yy, zz,
  xy, yz, xz; sigma = P for a small-strain material), `pk1_stress` (P
  row-major), `deformation_gradient` (F row-major), `strain` (the
  Green-Lagrange strain, or the infinitesimal strain for a small-strain
  material, in the order of the stress), `jacobian` (det F, or 1 + tr(eps)),
  `vonmises`, `energy_density` (the stored energy per
  unit reference volume, for the mixed formulation the integrand of its
  functional) and `thickness_stretch` (plane stress), are computed once per
  load step at the quadrature points of the kernels' rule (order 2p + 3)
  into a `QuadratureFunction`, the source of truth, and then presented where
  `quadrature_at` asks. Derived scalars such as von Mises are formed at the
  quadrature points and presented, never computed from a presented field.

The presentations are

- `nodes`: a continuous H1 field of the mesh order named `<name>`, derived
  from the quadrature data alone. `nodal_projection: averaged` projects the
  data onto each element's own polynomial space (element mass matrix) and
  takes the arithmetic mean of the element values at shared nodes, the
  classical extrapolate-and-average; `projected` solves the global L2
  projection (consistent mass matrix, CG to 1e-14), which is exact in the
  least-squares sense but can overshoot near stress concentrations.
- `elements`: `<name>_elem`, the quadrature-weighted volume average per
  element, an order-0 field that ParaView renders flat per element.
- `quadrature_points`: `<name>_qp`, the raw values. MFEM writes a
  `QuadratureFunction` as a point cloud of vertex cells, one
  `<name>_qp<rank>.vtu` per rank next to the mesh pieces plus a
  `<name>_qp.pvtu`, listed in the same `.pvd` as a part named `<name>_qp`
  at every load step. The collection then opens in ParaView as a multiblock
  with the mesh block and one block per point cloud (Multiblock Inspector
  or Extract Block; Point Gaussian representation for the clouds). A point
  cloud cannot share a file with the mesh: its values are samples that are
  never interpolated, whereas the mesh fields are interpolated on the
  refined visualization mesh.

Probes print the nodal unknowns and the `<name>` and `<name>_elem`
presentations. For a homogeneous state all presentations agree to machine
precision (`tests/test_homogeneous`); on a real problem the differences
between them are discretization error, not inconsistencies.

### Plane stress (thin sheets)

`plane: stress` (2D, displacement formulation) wraps the material in the
`PlaneStress` adapter of `src/kernels/materials/plane_stress.hpp`: the
kernels still pad F with F33 = 1, the adapter replaces F33 by the thickness
stretch lambda3 at which P33 = 0 (sigma33 = 0) and returns P at that state,
so the in-plane block the kernels use is the plane-stress response. For an
incompressible base (kappa = inf) lambda3 = 1/det F2D and the mean stress
p = -sigma_iso,33 replaces the Lagrange multiplier, P = P_iso + p F^-T: the
sheet problem is a pure displacement problem with no pressure unknown, no
inf-sup condition and no locking, which is why `plane: stress` is the one
way to run an incompressible model in the displacement formulation. For a
compressible base lambda3 solves P33(lambda3) = 0 by a scalar Newton
iteration at every quadrature point; with dual numbers the converged root is
refined by one Newton step in dual arithmetic, which carries the implicit
derivative d lambda3/dF, so the tangent stays consistent (checked against
finite differences in `test_materials`, together with P33 = 0, J = 1 for
incompressible bases, the plane-stress small-strain moduli E/(1 - nu^2) and
the thickness strain -nu/(1 - nu) tr eps). Every quadrature quantity uses
the completed F, and `thickness_stretch` becomes available as one more. The uniaxial, equibiaxial, pure-shear
and thin-sheet simple-shear states of `doc/verification_manual.tex`, Appendix A,
are all plane-stress states, so they run in 2D with the affine
displacement on the end faces (uniaxial) or on the whole boundary (the
others; well posed here, unlike plane strain). Two cautions: it is a thin-body
idealisation (stress uniform through the thickness, no bending), and because
each load step starts with the boundary moved and the interior lagging, shear
dominated sheet problems need increments that do not shear a boundary layer
of elements by more than a few percent (the energy grows like 1/det F2D^2),
e.g. `load_steps` such that the shear per step is about 0.05.

Linear solver note: the Krylov tolerance is relative to the AMG-preconditioned
residual. The `elasticity` options (rigid-body-mode interpolation) work well
in 2D but stall on slender 3D p = 2 meshes; the solver then falls back to the
`systems` options automatically, or select them directly with `amg: systems`.

### Mixed displacement-pressure formulation (near- and fully incompressible)

`formulation: mixed` solves the two-field problem

```
int [P_iso(F) + p J F^{-T}] : Grad w dV = int rho0 b . w dV + int T . w dA
int q (u'(J) - p / kappa) dV = 0            (kappa = inf: int q u'(J) dV = 0, i.e. J = 1)
```

with U(J) = kappa u(J) the material's volumetric law (`material.volumetric`; u' = J - 1 for
the default quadratic law, giving the familiar J - 1 - p / kappa). For a non-quadratic law the
block K_pu = u''(J) K_up^T is not the transpose of K_up; the FGMRES solver below accepts that.

on a Taylor-Hood pair (displacement H1 of order p, pressure H1 of order
p - 1, so `mesh.order >= 2`). Materials must provide the isochoric-volumetric
split (`iso_neo_hookean`, `mooney_rivlin`, `yeoh`, `gent`, `arruda_boyce`,
`ogden`); `nu: 0.5` or
`incompressible: true` makes the pressure a Lagrange multiplier. Dirichlet
conditions act on the displacement only; give the pressure a traction
boundary somewhere, since with the displacement prescribed on the whole
boundary an incompressible pressure is determined only up to a constant.

Each Newton system `[[K, B], [B^T, -M_p/kappa]]` is solved in its augmented
Lagrangian form (displacement block `K + gamma B diag(M_p)^{-1} B^T`,
`gamma = augmentation * mu`), which stays well conditioned when the pressure
is comparable to the shear modulus, by FGMRES with a block upper-triangular
preconditioner: the Schur complement is approximated by a scaled pressure
mass matrix (CG + Jacobi) and the displacement block by an inner GMRES +
BoomerAMG solve to `inner_rtol`. On the Cook and MMS problems this takes
about 20 outer iterations independent of the mesh. `augmentation: 0`
disables the transformation.

Checks (`tests/test_mixed`): affine patch tests with a constant pressure for
both materials at nu = 0.45 and kappa = inf (exact to 1e-14), block Jacobian
vs finite differences (4e-9), MMS with a volume-preserving manufactured
motion for the incompressible case (rates u 3.0, p 2.0) and a compressible
one at nu = 0.45 (rates u >= 3.4, p 2.0), and agreement between the mixed and
displacement formulations at finite kappa under refinement.

The incompressible constitutive models (neo-Hookean, Mooney-Rivlin, Yeoh,
Gent, Arruda-Boyce, Ogden) and the closed-form stresses and pressure of the
homogeneous states (uniaxial, equibiaxial, pure shear, simple shear, and
their plane-strain forms) are collected in
`doc/verification_manual.tex`, Appendix A. They are used as reference
solutions in two places:

- `apps/input/finite_elasticity/verification/homogeneous_deformations/plane_strain_neo_hookean.yaml` (unit square
  `apps/mesh/square.msh`, plane-strain extension to lambda = 2, affine
  displacement on the faces `left` and `right` as expressions (`["x", "-0.5*y"]`),
  lateral faces free), `plane_stress_neo_hookean.yaml` (the same
  sheet in plane stress: uniaxial tension, thickness stretch lambda^-1/2, no
  pressure unknown), `uniaxial_neo_hookean.yaml` (unit cube `apps/mesh/cube.msh`, uniaxial
  tension to lambda = 2), its symmetry model with rollers, and the equibiaxial and
  pure-shear cubes, all for the incompressible neo-Hookean model (swap the `material`
  line for any of the other five: the script's closed forms cover them all);
  `python3 apps/homogeneous_compare.py` runs
  them and prints the probed displacement, pressure or thickness stretch, von
  Mises stress, J and the stress components, in their nodal and element
  presentations, next to the closed forms (all agree to ~1e-13;
  `make homogeneous`).
- `tests/test_homogeneous` (`make check`): for every model the code's Cauchy
  stress at the material point vs the analytic principal stresses in uniaxial,
  equibiaxial, pure-shear, plane-strain and simple-shear states, and the mixed
  formulation's u and p on 4x4 (plane strain) and 2x2x2 (uniaxial) Q2/Q1
  meshes and the plane-stress displacement formulation's u for the sheet
  states uniaxial, equibiaxial, pure shear and simple shear on a 4x4 Q2
  mesh, together with every quadrature quantity in all three presentations
  (both nodal projections, element averages, raw quadrature values); all
  exact to solver tolerance with quadratic Newton convergence (the Ogden
  tangent is analytic, see below).

### Adding a material (NeoHookean as the template)

A material is a cheap-to-copy value type with its parameters as public
members and a `PK1` template over the scalar type; nothing else is required
for the displacement formulation. `Energy` is optional (used only for the
energy diagnostic). A material for the mixed formulation instead provides
`PK1Iso<T>(F)`, `EnergyIso<T>(F)`, `VolumetricPressure<T>(J)` and
`VolumetricEnergy<T>(J)` (U' and U of the selected law, `materials/volumetric.hpp`),
`NormalizedVolumetricPressure<T>(J)` and `NormalizedVolumetricModulus<T>(J)` (u', u''
for the mixed constraint), `kappa`, `law`, `Incompressible()`, and `ShearModulus()`
(the small-strain value, used to scale the saddle-point preconditioner; see
`iso_neo_hookean.hpp`); it can
also be used in the displacement formulation when `kappa` is finite. Models
that depend on I1bar alone (Yeoh, Gent, Arruda-Boyce) only implement
`DPsiDI1` and reuse `isochoric.hpp`. Ogden is the exception to the
"templated PK1" rule: eigenvectors are not differentiable at coincident
principal stretches, so it provides separate `PK1Iso` overloads for
`double` and `dual`, the latter forming the directional derivative
analytically from the spectral representation (limit branch at equal
eigenvalues); `test_materials` checks it against finite differences at
coincident stretches.

```cpp
// src/kernels/materials/neo_hookean.hpp
struct NeoHookean
{
  double mu = 1.0;
  double lambda = 1.0;

  template <typename T>
  tensor<T, 3, 3> PK1(const tensor<T, 3, 3> &F) const
  {
    const tensor<T, 3, 3> FinvT = transpose(inv(F));
    const T J = det(F);
    return mu * (F - FinvT) + (lambda * log(J)) * FinvT;
  }
};
```

1. Write the functor with `tensor<T, 3, 3>` in and out (2D problems arrive as
   plane strain with `F33 = 1`). Use only the operations in `base/tensor.hpp`
   and the functions in `base/dual.hpp` (`log`, `sqrt`, `pow`, ...) so that
   `T = dual` compiles; the tangent `A = dP/dF` is then produced by
   `MaterialTangent` with no further code.
2. Add the type to the `Material` (and, if decoupled, `MixedMaterial`)
   variant and `ModelName` in `src/kernels/materials/materials.hpp`, the
   factory and shear modulus in `materials.cpp`, and its keys to
   `MaterialConfig`/`ParseMaterialConfig`/`ValidateMaterialConfig` in
   `src/base/config.{hpp,cpp}`.
3. Add the model to `tests/test_materials.cpp` (stress-free reference state,
   AD tangent vs finite differences, objectivity, small-strain limit), to
   the patch test in `tests/test_solid_mms.cpp`, and, if decoupled, to the
   model list of `tests/test_homogeneous.cpp` together with its
   closed-form `Psi_1`, `Psi_2` (or principal form) in `Beta`.

The physics module instantiates `TotalLagrangianIntegrator<M>` once per
material type at setup through `std::visit`; no dispatch happens per
quadrature point.

### State of the seam (what is still solid-specific)

The plan's section 4.4 seam is implemented as the minimum the solid needs.
For the next physics the following will have to generalize:

- `TotalLagrangianIntegrator` hard-codes the flux `F = P(F)` contracted with
  `Grad w`, the unknown as a vector H1 field, and the tangent contraction
  `Grad w : A : Grad du`. The qpoint bodies (`DeformationGradient`,
  `QPointStress`, `QPointTangent`, `QPointCauchyStress`) are free functions
  on plain tensors and can be lifted into a general `F(u, grad u)`, `S(u)`
  contract; the element loops (`Residual`, `Tangent`) would become the
  framework's CG volume kernel taking that contract.
- The mixed u-p kernel is a second, separate block integrator with its own
  element loops. A multi-field CG kernel taking a list of unknown fields and
  a block flux/source contract would absorb both integrators; the block
  solver (`SaddlePointSolver`) is likewise specific to the 2x2 u-p structure
  and to the pressure-mass Schur complement approximation.
- Sources are not part of the integrator: the body force is a dead load on
  the linear-form side. A reaction or heat source `S(u)` needs a domain
  source term inside the nonlinear form with its own tangent.
- Boundary terms are stock MFEM linear-form integrators (dead loads) plus
  one hand-written boundary face integrator (the follower pressure).
  Robin/flux conditions and DG/HDG numerical fluxes have no home yet; a
  general boundary-flux contract would absorb the follower kernel.
- Dirichlet conditions prescribe all or some components of the vector
  unknown on an attribute; scalar unknowns are not yet expressible in the
  YAML `bcs` block (the `LoadSet` is written for one vector H1 space).
- The physics module owns its own space and, through `LoadSet`, its
  essential dofs and loads. A coupling layer will need these behind a common
  interface (`QuasiStaticProblem` is the current minimal one: residual,
  Jacobian, pseudo-time, Dirichlet application) plus field exchange by name
  through `FieldRegistry`.
- Time integration is a decorator over that interface, not a term of the
  kernels: with the unknown of the new time level as the variable, the
  second-order system adds `c_M M` and a history vector to the static
  residual (`DynamicSolidProblem`). A first-order physics (heat, species)
  would add `c C` and its history the same way, with the first-order
  generalized-alpha parameters and the same `rho_inf`, which is what a
  monolithic coupling needs. The mass matrix is a stock MFEM integrator
  because it is constant here; a state-dependent capacity would have to
  enter the nonlinear form as a source kernel.
- Materials are stateless and, per element attribute, one model with
  region-wise parameters (`material.regions`; the integrators hold a table
  indexed by attribute). Internal variables (`QuadratureFunction` state),
  temperature dependence, and hand-coded tangents are absent by design.
  The kinematics (finite or small strain) is a trait of the material, not of
  the kernel: the flux contract is shared, and what depends on the kinematics
  outside the flux goes through `materials/kinematics.hpp`.
- All kernels are CPU host code written as plain callables without
  allocation or virtual calls in the qpoint loops, so `MFEM_HOST_DEVICE` and
  `mfem::forall` can be added without restructuring; partial assembly is not
  implemented.
