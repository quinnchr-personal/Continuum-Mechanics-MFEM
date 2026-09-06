# Continuum-Mechanics-MFEM
Applications that run continuum mechanics simulations using MFEM

## Flux-kernel framework (root `src/`, `apps/`, `tests/`)

A multiphysics framework on MFEM in which each physics is expressed as
fluxes and sources at quadrature points and the framework owns assembly,
Newton, and the linear solvers (architecture: `doc/flux_kernel_architecture.html`,
plan: `doc/hyperelasticity_implementation_plan.md`). The first physics is
quasi-static nonlinear solid mechanics (compressible hyperelasticity, total
Lagrangian, CG, weak form in `doc/solid_mechanics_forms.tex`). The `myapps/`
tree is legacy and separate.

### Layout

```
src/base/       tensor.hpp (fixed-size tensors), dual.hpp (forward-mode AD),
                config.{hpp,cpp} (YAML schema -> structs, key-path errors),
                mesh_input (file or Cartesian box, corner map, jitter, refinement),
                fields.hpp (named field registry), output (ParaView), probes (point values)
src/kernels/    total_lagrangian.hpp: qpoint free functions + TotalLagrangianIntegrator<Material>;
                mixed_total_lagrangian.hpp: u-p qpoint functions + MixedTotalLagrangianIntegrator<Material>
src/kernels/materials/
                neo_hookean.hpp, st_venant_kirchhoff.hpp (coupled, displacement formulation);
                iso_neo_hookean.hpp, mooney_rivlin.hpp, yeoh.hpp, gent.hpp, arruda_boyce.hpp,
                ogden.hpp (isochoric-volumetric split, either formulation; isochoric.hpp shared
                I1bar pieces, spectral.hpp symmetric 3x3 eigen-solver for Ogden);
                plane_stress.hpp (adapter: F33 = thickness stretch with P33 = 0, any base model);
                material_tangent.hpp (dual seeding), materials.{hpp,cpp} (variants, YAML factory, moduli)
src/physics/    solid_problem.hpp (common interface + factory by formulation);
                quadrature_fields.{hpp,cpp}: quadrature-point quantities and their nodal /
                element / point-cloud presentations (shared by both formulations);
                solid_mechanics_tl.{hpp,cpp}: displacement formulation (residual, assembled
                Jacobian, essential dofs, dead loads, load factor, output fields);
                mixed_solid_mechanics_tl.{hpp,cpp}: u-p formulation on a Taylor-Hood pair
src/solvers/    newton (damped Newton, Armijo backtracking), linear_solver (GMRES/CG + BoomerAMG),
                saddle_point_solver (augmented Lagrangian FGMRES for the u-p Jacobian),
                quasi_static (load stepping over lambda in (0, 1])
apps/           solid_mechanics.cpp (YAML parsing and wiring only), apps/input/*.yaml,
                apps/mesh/*.geo (Gmsh sources of the example meshes, named physical groups) and the
                generated apps/mesh/*.msh (make meshes),
                apps/input/homogeneous/*.yaml (homogeneous deformations of the six incompressible
                models: plane strain, plane stress, 3D) with apps/homogeneous_compare.py (runs them
                against the closed forms)
tests/          test_base, test_materials, test_solid_mms, test_mixed, test_homogeneous (make check);
                test_mixed --full, test_benchmarks, test_parallel, homogeneous compare (make test)
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
make homogeneous # the app on apps/input/homogeneous/*.yaml, compared with the closed forms (python3 + yaml)
make test       # everything: app runs serial and np=4, np={2,4} consistency, benchmarks, homogeneous
make clean      # removes build/
```

All build products (objects, `.d` dependency fragments, the library, the
executables, and test scratch files) go under `build/`, mirroring the source
tree; set `BUILD_DIR=...` on the command line to put them elsewhere. Run the
targets from the repository root, since the inputs are referenced as
`apps/input/*.yaml`.

`make test` ends with `build/tests/test_benchmarks --cook-ratio-gate`, which asserts
the plan's requirement that the Cook's membrane corner displacement converge
with successive differences shrinking by at least 3x per uniform refinement.
That threshold is not met (measured ratios 2.34, 2.45, 2.31): uniform
refinement is limited by the singularity at the 108-degree clamped-free
corner, so the point value converges at roughly h^1.3. Everything before that
final step is green; the threshold is kept as written rather than relaxed.

### Running Cook's membrane

```
./build/apps/solid_mechanics -i apps/input/cook.yaml
mpirun -np 4 ./build/apps/solid_mechanics -i apps/input/cook.yaml
```

Plane strain, NeoHookean with E = 250, nu = 0.3, left edge clamped, uniform
upward shear traction of 3.75 per unit reference length on the right edge
(resultant 60). The run prints the Newton log, `|u|_L2`, the internal energy,
and the probe at the top-right corner (48, 60); the frozen regression value
on the 64x64 p = 2 mesh is uy = 4.905891700497 (30.7% of the 16 mm edge).
ParaView output goes to `out/cook` (`displacement` for Warp by Vector,
`vonmises`, `jacobian`), one cycle per load step. The 3D cantilever of the
linear-limit test runs the same way from `apps/input/cantilever3d.yaml`.

The incompressible variant of the same benchmark (mixed u-p formulation,
isochoric neo-Hookean with mu = 80.194, resultant 100, the pressure a
Lagrange multiplier) is `apps/input/cook_incompressible.yaml`; its frozen
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
  #   incompressible: true; finite kappa works in either formulation (penalty U = kappa/2 (J - 1)^2
  #   in the displacement formulation), kappa = inf needs formulation: mixed.
  #   Closed forms and homogeneous solutions: doc/incompressible_hyperelasticity.tex.
bcs:
  dirichlet: [ { attr: [left], value: [0.0, 0.0] } ]   # attr: physical-group names and/or attribute numbers;
  traction:  [ { attr: [right], value: [0.0, 3.75] } ]  # all components prescribed / nominal traction per
                                                        # unit reference area (dead load)
  # Either entry may add gradient: [[..],[..]] (dim x dim): the data is then value + gradient X in the
  # reference coordinates (affine, e.g. the exact displacement of a homogeneous deformation).
body_force: [0.0, 0.0]            # per unit mass; rho0 * b enters the weak form
solver:
  load_steps: 1                   # loads and prescribed displacements scaled by k/load_steps
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
```

Unknown keys, missing required keys, and wrong types raise an error naming
the full key path (for example `key 'material.E' expected a number, got
'abc'`).

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
int q (J - 1 - p / kappa) dV = 0            (kappa = inf: int q (J - 1) dV = 0)
```

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

- `apps/input/homogeneous/plane_strain_<model>.yaml` (unit square
  `apps/mesh/square.msh`, plane-strain extension to lambda = 2, affine
  displacement on the faces `left` and `right` through the `gradient` key,
  lateral faces free), `plane_stress_<model>.yaml` (the same
  sheet in plane stress: uniaxial tension, thickness stretch lambda^-1/2, no
  pressure unknown) and `uniaxial_<model>.yaml` (unit cube `apps/mesh/cube.msh`, uniaxial
  tension to lambda = 2) for all six models; `python3 apps/homogeneous_compare.py` runs
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
`PK1Iso<T>(F)`, `EnergyIso<T>(F)`, `VolumetricPressure<T>(J)`, `kappa`,
`Incompressible()`, and `ShearModulus()` (the small-strain value, used to
scale the saddle-point preconditioner; see `iso_neo_hookean.hpp`); it can
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
- Boundary terms are stock MFEM linear-form integrators (dead loads only).
  Follower loads, Robin/flux conditions, and DG/HDG numerical fluxes have no
  home yet (the follower-load seam is a TODO in `SolidMechanicsTL::Finalize`).
- Dirichlet conditions prescribe all components of the vector unknown on an
  attribute; component-wise (symmetry) conditions and scalar unknowns are not
  expressible in the YAML `bcs` block.
- The physics module owns its own space, essential dofs, and loads. A
  coupling layer will need these behind a common interface
  (`QuasiStaticProblem` is the current minimal one: residual, Jacobian, load
  factor, Dirichlet application) plus field exchange by name through
  `FieldRegistry`.
- Materials are stateless. Internal variables (`QuadratureFunction` state),
  temperature dependence, and hand-coded tangents are absent by design.
- All kernels are CPU host code written as plain callables without
  allocation or virtual calls in the qpoint loops, so `MFEM_HOST_DEVICE` and
  `mfem::forall` can be added without restructuring; partial assembly is not
  implemented.
