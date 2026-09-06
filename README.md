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
src/materials/  neo_hookean.hpp, st_venant_kirchhoff.hpp (coupled, displacement formulation);
                iso_neo_hookean.hpp, mooney_rivlin.hpp (isochoric-volumetric split, either formulation);
                material_tangent.hpp (dual seeding), materials.{hpp,cpp} (variants, YAML factory, moduli)
src/physics/    solid_problem.hpp (common interface + factory by formulation);
                solid_mechanics_tl.{hpp,cpp}: displacement formulation (residual, assembled
                Jacobian, essential dofs, dead loads, load factor, output fields);
                mixed_solid_mechanics_tl.{hpp,cpp}: u-p formulation on a Taylor-Hood pair
src/solvers/    newton (damped Newton, Armijo backtracking), linear_solver (GMRES/CG + BoomerAMG),
                saddle_point_solver (augmented Lagrangian FGMRES for the u-p Jacobian),
                quasi_static (load stepping over lambda in (0, 1])
apps/           solid_mechanics.cpp (YAML parsing and wiring only) and apps/input/*.yaml
tests/          test_base, test_materials, test_solid_mms, test_mixed (make check);
                test_mixed --full, test_benchmarks, test_parallel (make test)
makefile        out-of-tree build under build/ (BUILD_DIR): build/libcmf.a (LIBNAME) from src/,
                then build/apps/* and build/tests/* linked against it
```

### Build and test

Requirements: MFEM 4.8 built with MPI, METIS, and HYPRE (the makefile finds
`~/MFEM/mfem/config/config.mk` automatically; set `MFEM_DIR` otherwise),
yaml-cpp via `pkg-config`, and an `mpirun`.

```
make            # build/libcmf.a, build/apps/solid_mechanics, build/tests/*
make check      # serial, ~12 s: tensor/dual/YAML units, materials, patch tests + MMS (both formulations)
make test       # everything: app runs serial and np=4, np={2,4} consistency, benchmarks
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
mesh:
  file: path.mesh                 # or a cartesian box (exactly one of the two)
  cartesian: { nx: 4, ny: 4, nz: 4, sx: 1.0, sy: 1.0, sz: 1.0, element: quad }
                                  # nz present -> 3D; element: quad|tri|hex|tet
  corners: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]   # optional bilinear image of the 2D box,
                                                  # counter-clockwise, mapped from (0,0),(sx,0),(sx,sy),(0,sy)
  perturb: 0.0                    # interior-vertex jitter of the base mesh, fraction of h, in [0, 0.25)
  serial_refine: 1
  parallel_refine: 0
  order: 2                        # H1 polynomial degree
material: { model: neo_hookean, E: 250.0, nu: 0.3, rho0: 1.0 }
  # neo_hookean, st_venant_kirchhoff: coupled compressible models, keys E, nu (nu < 0.5); displacement only
  # iso_neo_hookean: mu (or E, nu); mooney_rivlin: c1, c2 (mu = 2 (c1 + c2)). Both take the bulk
  #   modulus from exactly one of kappa | nu (nu = 0.5 -> incompressible) | incompressible: true;
  #   finite kappa works in either formulation (penalty U = kappa/2 (J - 1)^2 in the displacement
  #   formulation), kappa = inf needs formulation: mixed
bcs:
  dirichlet: [ { attr: [4], value: [0.0, 0.0] } ]  # all components prescribed on these boundary attributes
  traction:  [ { attr: [2], value: [0.0, 3.75] } ] # nominal traction per unit reference area (dead load)
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
  fields: [displacement, vonmises, jacobian]   # plus pressure in the mixed formulation
  high_order: true
  probes: [ { name: top_right_corner, point: [48.0, 60.0] } ]   # displacement printed at these points
```

Cartesian boundary attributes follow MFEM: 2D bottom 1, right 2, top 3,
left 4; 3D z=0 1, y=0 2, x=sx 3, y=sy 4, x=0 5, z=sz 6. Unknown keys,
missing required keys, and wrong types raise an error naming the full key
path (for example `key 'material.E' expected a number, got 'abc'`).

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
split (`iso_neo_hookean`, `mooney_rivlin`); `nu: 0.5` or
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

### Adding a material (NeoHookean as the template)

A material is a cheap-to-copy value type with its parameters as public
members and a `PK1` template over the scalar type; nothing else is required
for the displacement formulation. `Energy` is optional (used only for the
energy diagnostic). A material for the mixed formulation instead provides
`PK1Iso<T>(F)`, `EnergyIso<T>(F)`, `VolumetricPressure<T>(J)`, `kappa`, and
`Incompressible()` (see `iso_neo_hookean.hpp`); it can also be used in the
displacement formulation when `kappa` is finite.

```cpp
// src/materials/neo_hookean.hpp
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
   variant and the name to the factories in `src/materials/materials.cpp`,
   and its keys to `ValidateMaterialConfig` in `src/base/config.cpp`.
3. Add the model to `tests/test_materials.cpp` (stress-free reference state,
   AD tangent vs finite differences, objectivity, small-strain limit) and to
   the patch test in `tests/test_solid_mms.cpp`.

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
