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
src/kernels/    total_lagrangian.hpp: qpoint free functions + TotalLagrangianIntegrator<Material>
src/materials/  neo_hookean.hpp, st_venant_kirchhoff.hpp, material_tangent.hpp (dual seeding),
                materials.hpp (std::variant Material, YAML factory, E/nu -> mu/lambda)
src/physics/    solid_mechanics_tl.{hpp,cpp}: mfem::Operator (residual, assembled Jacobian,
                essential dofs, dead loads, load factor, output fields)
src/solvers/    newton (damped Newton, Armijo backtracking), linear_solver (GMRES/CG + BoomerAMG),
                quasi_static (load stepping over lambda in (0, 1])
apps/           solid_mechanics.cpp (YAML parsing and wiring only) and apps/input/*.yaml
tests/          test_base, test_materials, test_solid_mms (make check);
                test_benchmarks, test_parallel (make test)
makefile        builds lib$(LIBNAME).a (default libcmf.a) from src/, then apps/ and tests/
```

### Build and test

Requirements: MFEM 4.8 built with MPI, METIS, and HYPRE (the makefile finds
`~/MFEM/mfem/config/config.mk` automatically; set `MFEM_DIR` otherwise),
yaml-cpp via `pkg-config`, and an `mpirun`.

```
make            # libcmf.a, apps/solid_mechanics, tests/*
make check      # S1-S3 gates, serial, ~2 s: tensor/dual/YAML units, materials, patch test + MMS
make test       # everything: app runs serial and np=4, np={2,4} consistency, benchmarks
make clean
```

`make test` ends with `tests/test_benchmarks --cook-ratio-gate`, which asserts
the plan's requirement that the Cook's membrane corner displacement converge
with successive differences shrinking by at least 3x per uniform refinement.
That threshold is not met (measured ratios 2.34, 2.45, 2.31): uniform
refinement is limited by the singularity at the 108-degree clamped-free
corner, so the point value converges at roughly h^1.3. Everything before that
final step is green; the threshold is kept as written rather than relaxed.

### Running Cook's membrane

```
./apps/solid_mechanics -i apps/input/cook.yaml
mpirun -np 4 ./apps/solid_mechanics -i apps/input/cook.yaml
```

Plane strain, NeoHookean with E = 250, nu = 0.3, left edge clamped, uniform
upward shear traction of 3.75 per unit reference length on the right edge
(resultant 60). The run prints the Newton log, `|u|_L2`, the internal energy,
and the probe at the top-right corner (48, 60); the frozen regression value
on the 64x64 p = 2 mesh is uy = 4.905891700497 (30.7% of the 16 mm edge).
ParaView output goes to `out/cook` (`displacement` for Warp by Vector,
`vonmises`, `jacobian`), one cycle per load step. The 3D cantilever of the
linear-limit test runs the same way from `apps/input/cantilever3d.yaml`.

### YAML schema

```yaml
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
material: { model: neo_hookean, E: 250.0, nu: 0.3, rho0: 1.0 }   # model: neo_hookean | st_venant_kirchhoff
bcs:
  dirichlet: [ { attr: [4], value: [0.0, 0.0] } ]  # all components prescribed on these boundary attributes
  traction:  [ { attr: [2], value: [0.0, 3.75] } ] # nominal traction per unit reference area (dead load)
body_force: [0.0, 0.0]            # per unit mass; rho0 * b enters the weak form
solver:
  load_steps: 1                   # loads and prescribed displacements scaled by k/load_steps
  newton:  { rtol: 1e-10, atol: 1e-12, max_it: 25, armijo_c: 1e-4, max_halvings: 8, print_level: 1 }
  linear:  { type: gmres_amg, amg: elasticity, rtol: 1e-12, atol: 0.0, max_it: 500, krylov_dim: 50, print_level: 0 }
                                  # type: gmres_amg | cg_amg; amg: elasticity | systems
output:
  paraview: out/cook              # empty or absent -> no files
  fields: [displacement, vonmises, jacobian]
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

### Adding a material (NeoHookean as the template)

A material is a cheap-to-copy value type with its parameters as public
members and a `PK1` template over the scalar type; nothing else is required.
`Energy` is optional (used only for the energy diagnostic).

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
2. Add the type to the `Material` variant and the name to `MakeMaterial` in
   `src/materials/materials.hpp`, and to the allowed `material.model` values
   in `src/base/config.cpp`.
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
