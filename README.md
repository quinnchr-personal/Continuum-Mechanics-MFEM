# Continuum-Mechanics-MFEM
Applications that run continuum mechanics simulations using MFEM

## Flux-kernel framework (root `src/`, `apps/`, `tests/`)

A multiphysics framework on MFEM in which each physics is expressed as
fluxes and sources at quadrature points and the framework owns assembly,
Newton, and the linear solvers (architecture: `doc/flux_kernel_architecture.html`,
plan: `doc/hyperelasticity_implementation_plan.md`). The first physics is
quasi-static nonlinear solid mechanics (compressible hyperelasticity, total
Lagrangian, CG, weak form in `doc/solid_mechanics_forms.tex`). The models and
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
                neo_hookean.hpp, st_venant_kirchhoff.hpp (coupled, displacement formulation);
                iso_neo_hookean.hpp, mooney_rivlin.hpp, yeoh.hpp, gent.hpp, arruda_boyce.hpp,
                ogden.hpp (isochoric-volumetric split, either formulation; isochoric.hpp shared
                I1bar pieces, volumetric.hpp the selectable volumetric laws U(J),
                spectral.hpp symmetric 3x3 eigen-solver for Ogden);
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
                mixed_solid_mechanics_tl.{hpp,cpp}: u-p formulation on a Taylor-Hood pair
src/solvers/    newton (damped Newton, Armijo backtracking), linear_solver (GMRES/CG + BoomerAMG),
                saddle_point_solver (augmented Lagrangian FGMRES for the u-p Jacobian),
                quasi_static (load stepping over the pseudo-time t in (0, 1] with bisection)
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
                apps/input/anand_coupled_theories/<chapter>/*.yaml (the examples of Anand's coupled-theories
                  book, from its FEniCSx companion codes; finite_elasticity so far)
tests/          test_base, test_materials, test_solid_mms, test_mixed, test_homogeneous, test_loading
                (make check); test_mixed --full, test_benchmarks, test_parallel, homogeneous compare,
                test_loading np=4, test_verification (make test)
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
make check      # serial, ~20 s: tensor/dual/YAML units, materials, patch tests + MMS (both
                # formulations), homogeneous deformations vs closed forms (all incompressible models)
make homogeneous # the app on apps/input/finite_elasticity/verification/homogeneous_deformations/*.yaml, compared with the closed forms (python3 + yaml)
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
  # neo_hookean, st_venant_kirchhoff: coupled compressible models, keys E, nu (nu < 0.5); displacement only
  # Decoupled (isochoric-volumetric) models, either formulation:
  #   iso_neo_hookean: mu (or E, nu)
  #   mooney_rivlin:   c1, c2                   (mu = 2 (c1 + c2))
  #   yeoh:            c10, [c20, c30]          (mu = 2 c10)
  #   gent:            mu, Jm                   (I1bar - 3 < Jm)
  #   arruda_boyce:    mu, N                    (mu = n k T; small-strain modulus mu (1 + 3/(5N) + ...))
  #   ogden:           mu_r: [..], alpha_r: [..] (up to 6 terms, mu_r alpha_r > 0; mu = 1/2 sum mu_r alpha_r)
  #   All take the bulk modulus from exactly one of kappa | nu (nu = 0.5 -> incompressible) |
  #   incompressible: true; finite kappa works in either formulation (penalty U(J) in the
  #   displacement formulation), kappa = inf needs formulation: mixed.
  #   volumetric: quadratic (default) | simo_taylor | logarithmic | j_log_j selects the volumetric
  #   law U(J) = kappa u(J) of a decoupled model at finite kappa: u = (J - 1)^2/2 |
  #   (J^2 - 1 - 2 ln J)/4 | (ln J)^2/2 (p = kappa ln J / J, Anand's FEniCSx codes) | J ln J - J + 1.
  #   Every law has u''(1) = 1, so kappa keeps its meaning; they differ at finite strain.
  #   Closed forms and homogeneous solutions: doc/incompressible_hyperelasticity.tex.
  # regions: [ { attr: [inclusion, 3], mu: 2800.0, kappa: 2800000.0 } ]
  #   Other parameters of the same model by element attribute (physical-volume names and/or
  #   numbers); keys not given are inherited from the base, a region giving any of kappa | nu |
  #   incompressible replaces the base's bulk specification, every region must be incompressible
  #   or none, and no attribute may be covered twice. The base applies everywhere else.
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
solver:
  load_steps: 1                   # equal increments of t; or steps: [ { to: 0.5, n: 2 }, { to: 1.0, n: 4 } ]
  predictor: none                 # none | tangent: start each increment from a linear solve about the last
                                  # converged state with the Dirichlet increment imposed on the update
  substep: { on_failure: false, max_bisections: 4, min_dt: 1e-4 }   # halve a failed increment and retry
  newton:  { rtol: 1e-10, atol: 1e-12, max_it: 25, armijo_c: 1e-4, max_halvings: 8, print_level: 1 }
  linear:  { type: gmres_amg, amg: elasticity, rtol: 1e-12, atol: 0.0, max_it: 500, krylov_dim: 50, print_level: 0,
             inner_rtol: 1e-3, inner_max_it: 50, augmentation: 1.0 }
                                  # type: gmres_amg | cg_amg; amg: elasticity | systems
                                  # inner_*, augmentation: mixed formulation only (see below)
output:
  paraview: out/cook              # empty or absent -> no files
  fields: [displacement, vonmises, jacobian]   # nodal unknowns: displacement, pressure (mixed); quadrature
                                  # quantities: cauchy_stress (6: xx yy zz xy yz xz), pk1_stress (9, row-major),
                                  # deformation_gradient (9), jacobian, vonmises, energy_density,
                                  # thickness_stretch (plane stress); in 2D the out-of-plane terms are included
  quadrature_at: [nodes]          # presentations of the quadrature quantities: nodes (continuous field <name>),
                                  # elements (element average <name>_elem), quadrature_points (point cloud <name>_qp)
  nodal_projection: averaged      # averaged (element projection, mean at shared nodes) | projected (global L2)
  high_order: true
  probes: [ { name: top_right_corner, point: [48.0, 60.0] } ]   # every registered field printed at these points
  probe_every_step: false         # also after every load step, on lines prefixed "step k t = ..."
  reactions: false                # force and moment of every Dirichlet entry (bcs.*.name labels them)
```

Unknown keys, missing required keys, and wrong types raise an error naming
the full key path (for example `key 'material.E' expected a number, got
'abc'`).

### Boundary conditions and loading

Every boundary condition and the body force is a load entry with its own
data and its own schedule in a pseudo-time `t` that the stepper advances
from 0 to 1 (`doc/bc_loading_plan.md` is the design record,
`doc/solid_mechanics_forms.tex` Section 5 the formulation).

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
nodal forces.

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
group), multi-point or periodic constraints, contact, true dynamics (`t` is
a pseudo-time), automatic step growth after a bisection.

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
| `homogeneous_deformations/*_neo_hookean.yaml` | closed forms of doc/incompressible_hyperelasticity.tex (apps/homogeneous_compare.py) | every probed quantity to 1e-8 |
| `homogeneous_deformations/compressible_uniaxial_*.yaml` | lateral stretch from P_22 = 0 with the material's own PK1 (mu = 0.5, nu = 0.45 throughout; one decoupled input per volumetric law); `apps/uniaxial_plots.py` and `apps/neo_hookean_compare.py` drive the uniaxial inputs to a stretch of 8 and plot P_11 on independently coded analytical curves | displacements, P_11, sigma_11 and J to 1e-7 (scripts: about 1e-12) |

The torsion input needs 20 increments: a larger first increment leaves the
elements under the rotated end face inverted before Newton starts (the
boundary layer caution of the plane-stress section), which bisection would
also recover from.

### Anand's coupled-theories examples (`apps/input/anand_coupled_theories/`)

Inputs after the FEniCSx companion codes of Lallit Anand's *Introduction to
coupled theories in solid mechanics* (Oxford University Press, 2025;
solidmechanicscoupledtheories.github.io, codes by Eric Stewart and Lallit
Anand), one subdirectory per chapter of the site. `finite_elasticity/` holds
the ten "1. Finite Elasticity" examples: Arruda-Boyce with
G0 = 280 kPa, lambda_L = 5.12 and K = 1000 G0 in kPa and mm, the reference's
logarithmic volumetric law p = K ln(J)/J (`volumetric: logarithmic`, with
`solver.predictor: tangent`, see "Predictor" above; `02_simple_shear` keeps the
quadratic law, see below), mixed Q2-Q1 or
P2-P1, the same geometry, boundary conditions, load histories and step counts.
Meshes come from `apps/mesh/*.geo` (`make meshes`); the curved ones are
second-order. Every input probes the points of the reference's plots after
every step (`probe_every_step`), so the curves (stress or force vs stretch,
pressure vs displacement) can be read from the log.

| Input | Reference | Notes |
|-------|-----------|-------|
| `01_uniaxial_tension` | 3D01 | 10 mm cube, stretch 7.75 in y, rollers on three planes |
| `02_simple_shear` | 3D02 | 1 mm cube, two sinusoidal cycles of shear strain 1 (`sin(4 pi t)`) |
| `03_cylinder_torsion` | 3D03 | R = 12.7, L = 25.4, top face rotated by 2.5 rad, `cylinder_torsion.geo` |
| `04_plate_with_hole` | 3D04 | quarter plate 15 x 10 x 1 with a 3 mm hole, stretch 3, `plate_hole.geo` |
| `05_cylinder_inflation` | 3D05 | quarter tube 10/11 x 5 mm, follower pressure to 50 kPa, `tube_quarter.geo` |
| `06_sphere_inflation` | 3D06 | octant shell 10/11 mm, follower pressure to 35 kPa, `sphere_octant.geo` |
| `07_cube_footing` | 3D07 | 50 mm cube, follower pressure 1500 kPa on a quarter of the top, `footing.geo` |
| `08_column_buckling` | 3D08 | 1 x 1 x 20 column, imperfection by `perturb_column.py`, shortened by 2.5 mm |
| `09_spherical_inclusion` | 3D09 | octant of a cube with a ten times stiffer spherical inclusion (`material.regions`), stretch 2, `inclusion.geo` |
| `10_column_twist` | 3D10 | 1 x 1 x 3 column, top face turned through 2 pi |

`apps/anand_plots.py` reproduces the result plots of the reference pages
from the logs of these runs (the inputs print the probes and the reactions
after every step; save the stdout as `out/anand_coupled_theories/
finite_elasticity/logs/<case>.log`, or pass `--logs`). The reference
overlays no analytical curves; the script adds one where a reference
exists: the homogeneous incompressible Arruda-Boyce response for the
uniaxial and shear blocks (with both this code's series form and the
reference's Pade form of the model), Rivlin's universal torsion for torque
and axial force, the incompressible thick-walled cylinder and sphere
inflation by quadrature, the Euler load for the column, and the matrix-only
curve for the inclusion. `pip`-level dependencies: numpy and matplotlib.
What the plots show: the uniaxial cube, the torsion and both inflations lie
on their reference curves (the uniaxial one on the nearly incompressible
solution at K = 1000 G, with the incompressible limit a few percent above
at the largest stretch); the sheared block carries about 17 percent less
nominal shear stress than homogeneous simple shear at a shear strain of 1,
because its lateral faces are free where simple shear needs tractions, and
its two cycles retrace one curve (elastic, no hysteresis); the sphere stops
at its limit pressure of 34 kPa and the cylinder near 38 kPa, where the
reference also stopped; the buckling column reaches 7.3 mN at 0.2 mm of
shortening, 5 percent above the Euler load of the clamped column, and then
rises slowly to 7.5 mN at 2.5 mm, the hardening post-buckling path of the
elastica (the Euler value uses E = 3 G and neglects the finite section);
the twisted column needs a compressive axial force of 78 mN to keep its
length over a full turn (Poynting effect), as does the torsion cylinder.

Differences from the reference that change the numbers: the Arruda-Boyce
model here is the five-term series in I1/N (`N = lambda_L^2`) rather than the
Pade inverse Langevin, so it is softer near the locking stretch (visible in
01 above a stretch of about 3); `02_simple_shear` uses p = K (J - 1) because the
logarithmic law fails there at t = 0.064 for any increment size (its tangent bulk
modulus K (1 - ln J)/J^2 softens in dilatation, and J grows without bound at the
clamped corner singularities that this mesh resolves); hexahedra replace tetrahedra
on the boxes; and a failed
increment is bisected instead of ending the run (05 and 06 stop early in the
reference). Reaction forces and torques are not computed; the probes give
displacements, pressure and stresses at the reference's points.

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
  xy, yz, xz), `pk1_stress` (P row-major), `deformation_gradient` (F
  row-major), `jacobian`, `vonmises`, `energy_density` (the stored energy per
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
and thin-sheet simple-shear states of `doc/incompressible_hyperelasticity.tex`
Section 4 are all plane-stress states, so they run in 2D with the affine
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
`doc/incompressible_hyperelasticity.tex`. They are used as reference
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
- Materials are stateless and, per element attribute, one model with
  region-wise parameters (`material.regions`; the integrators hold a table
  indexed by attribute). Internal variables (`QuadratureFunction` state),
  temperature dependence, and hand-coded tangents are absent by design.
- All kernels are CPU host code written as plain callables without
  allocation or virtual calls in the qpoint loops, so `MFEM_HOST_DEVICE` and
  `mfem::forall` can be added without restructuring; partial assembly is not
  implemented.
