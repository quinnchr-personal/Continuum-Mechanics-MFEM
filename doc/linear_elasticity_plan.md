# Implementation Plan — Small-Strain Linear Elasticity (Gates LE1–LE5, optional LE6)

**Audience:** the implementing agent. Follow the gate order, meet each gate's acceptance
criteria before moving on, commit once per completed gate (on `main`, no attribution
trailers). This extends the framework of `doc/hyperelasticity_implementation_plan.md` and
`doc/bc_loading_plan.md`; their design decisions (reference mesh never moves, thin `apps/`,
`myapps/` untouched, materials are stateless value types templated on the scalar) still hold.

**Status (2026-09-18):** not started.

**Goal:** `material: { model: linear_elastic, E: ..., nu: ... }` solves geometrically linear
(small-strain) isotropic elasticity in 2D (plane strain, plane stress) and 3D, in the
displacement formulation and, for `nu -> 0.5` and `nu = 0.5`, in the mixed u-p formulation,
with every existing feature (schedules, components, expressions, pressures, regions,
reactions, probes, quadrature outputs, parallel) working unchanged and every existing input
reproducing its frozen numbers.

**The one-paragraph answer to "is this a new weak form?"** No. The small-strain form is

    int sigma(eps(u)) : eps(w) dV  =  int sigma : Grad w dV        (sigma symmetric)

which is the total Lagrangian form `int P(F) : Grad w dV_R` with the flux
`P(F) := sigma(sym(F - I))`. `TotalLagrangianIntegrator` assembles `int P_ij N_a,j` and
`int N_a,j A_ijkl N_b,l` with `A = dP/dF` by dual numbers; for this flux `A = C` (constant,
with minor symmetries), so the element matrix is the textbook `B^T C B` stiffness and the
first Newton step is the exact linear solve. Small strain therefore enters as a **material
plus a kinematics trait**, not as a new integrator or physics class. What the trait must
switch is everything outside the flux that silently assumes finite kinematics: the stress and
volume measures of the outputs (`sigma = J^{-1} P F^T`, `J = det F`), loads that refer to the
deformed configuration (follower pressure, reaction moment arms), the volume constraint of
the mixed kernel (`J - 1`, `dJ/dF = J F^{-T}`), the incompressible branch of the plane-stress
adapter, and tests that assume objectivity.

---

## 1. Current state (verified 2026-09-18 — re-verify before starting)

- Flux contract: `QPointStress` / `QPointTangent` (`src/kernels/total_lagrangian.hpp:57-74`)
  call `material.PK1(DeformationGradient<dim>(H))`; 2D pads `F33 = 1` (plane strain). The
  integrator is otherwise material-agnostic. `QPointCauchyStress` (`:79-86`) hard-codes
  `J^{-1} P F^T`.
- Outputs: `PackQuantity` (`src/physics/quadrature_fields.cpp:25-46`) derives
  `jacobian = det(s.F)` (`:34`) and `sigma = J^{-1} P F^T` (`:37`) from `QPointState{F, P,
  energy}`; the evaluators live in `SolidMechanicsTL::UpdateFields`
  (`src/physics/solid_mechanics_tl.cpp:199-221`) and
  `MixedSolidMechanicsTL::UpdateFields` (`src/physics/mixed_solid_mechanics_tl.cpp:~238-247`).
- Mixed kernel (`src/kernels/mixed_total_lagrangian.hpp`): `MixedPK1 = PK1Iso(F) + p J F^{-T}`
  (`:29-35`), `QPointVolumeGradient = J F^{-T}` (`:73-80`), energy `p (J - 1)` (`:196`),
  constraint `u'(J) - p / kappa` (`:234-236`). Material interface it needs: `PK1Iso`,
  `EnergyIso`, `kappa`, `Incompressible()`, `NormalizedVolumetricPressure/Modulus`,
  `ComplementaryVolumetricEnergy`; the physics also reads `ShearModulus()`.
- Plane-stress adapter (`src/kernels/materials/plane_stress.hpp`): the compressible path
  (scalar Newton on `P33(l3) = 0`, dual refinement for the tangent) is kinematics-free; the
  incompressible path is finite strain (`l3 = 1 / det F2D` at `:56-59`, `P_iso + p F^{-T}` at
  `:92-98`).
- Materials: `Material` / `MixedMaterial` variants (`materials.hpp:31-43`); `ModelName` ends
  in `else return "ogden"` (`:101-114`) — a new type without its own branch is reported as
  ogden. `ResolveModuli` (`materials.cpp:41-84`) already resolves `mu | (E, nu)` plus exactly
  one of `kappa | nu | incompressible` for the decoupled models; `MakeDecoupled` sets
  `mat.law` on every `MixedMaterial` alternative (`:25-31`).
- Validation: `ValidateMaterialConfig` (`src/base/config.cpp:401-556`), known-model list
  and error text at `:604-613`, schema comment in `src/base/config.hpp:49-66`.
- Loads: follower pressure is wired in `AddPressure` of both physics
  (`solid_mechanics_tl.cpp:80-95`); reaction moments use current positions
  (`src/physics/loads.cpp:281`).
- Solver: Newton converges on `|R| <= atol` or `<= rtol |R0|` (`src/solvers/newton.cpp:68`);
  `TangentPredictor` (`src/solvers/quasi_static.cpp:41-71`); `LinearSolver::SetOperator`
  rebuilds BoomerAMG on every call (`src/solvers/linear_solver.cpp:54-66`).
- Tests: `TestMaterial` (`tests/test_materials.cpp:100-202`) checks objectivity (`:157-165`)
  and the small-strain limit (`:171-201`) for every model; `ManufacturedBodyForce`
  (`tests/test_solid_mms.cpp:92-98`) forms `Div P` by central differences of the material's
  own `PK1`, so it works for any material; `CHECK_TESTS` is `makefile:148`.
- The existing "linear limit" inputs (`kirsch_plate_with_hole.yaml`,
  `euler_bernoulli_cantilever3d.yaml`) use a hyperelastic model at strains 1e-4..5e-5 and
  carry comments about the round-off floor that forces `newton.rtol: 1e-6`.

## 2. Design decisions — do not re-litigate

1. **Material, not physics.** `LinearElastic` is a value type with `PK1<T>(F)` and
   `Energy<T>(F)` that reads `eps = sym(F - I)`. No `ParBilinearForm` /
   `mfem::ElasticityIntegrator` code path, no new `SolidProblem` subclass: LoadSet, regions,
   reactions, outputs and the stepper are reused as they are. (MFEM's integrator is used as a
   test oracle only, LE1.) `SolidMechanicsTL` keeps its name; at small strain total and
   updated Lagrangian coincide.
2. **Kinematics is a compile-time trait of the material.** `static constexpr bool
   small_strain = true;` in the struct, detected by `is_small_strain<M>` (false by default;
   `PlaneStress<B>` forwards to `B`). All kinematic switches are `if constexpr` on this trait
   inside two free helpers, `CauchyStress(material, F, P)` and `VolumeRatio(material, F)`,
   and inside the mixed kernel's volume measure. No virtual calls, no runtime flag in the
   quadrature loops. Rejected alternative: a top-level `kinematics: small_strain` key that
   linearises any model about `F = I` — every current model is isotropic, so they all
   collapse to the same two-constant law (see Out of scope for the anisotropic follow-up).
3. **One struct, parameters `(mu, kappa)`.** `PK1 = 2 mu dev(eps) + kappa tr(eps) I`,
   `Energy = mu dev(eps):dev(eps) + kappa/2 tr(eps)^2`, `Lambda() = kappa - 2 mu / 3`. The
   same struct later serves the mixed formulation (`kappa` possibly infinite), like
   `IsoNeoHookean`. It has no `law` member: the volumetric law is quadratic by construction.
4. **YAML keys follow `iso_neo_hookean`:** `mu`, or `E` and `nu`; the bulk modulus from
   exactly one of `kappa | nu | incompressible`. So `{ E, nu }`, `{ mu, kappa }`,
   `{ mu, nu }` are all valid; `volumetric:` is "not used by model 'linear_elastic'".
   `ResolveModuli` needs no new branch.
5. **The contract stays `PK1(F)`, not `Stress(H)`.** Forming `F = I + H` and subtracting `I`
   again costs an absolute error of 1e-16 in `eps`, i.e. a relative residual floor of
   `1e-16 / |Grad u|`. With a linear model there is never a reason to use tiny loads: inputs
   and tests use amplitudes with `|Grad u| >= 1e-4` and scale results afterwards.
6. **Follower pressure is a configuration error** with a small-strain material (`type:
   pressure` is the same load there); so is `solver.predictor: tangent` (the first Newton step
   already is the exact linear solve; after an exact predictor Newton starts at the round-off
   floor, `rtol |R0|` is unreachable and the line search fails unless `atol` happens to
   catch it). Both throw `ConfigError` naming the key, in the style of "key ... is not used by
   model ...".
7. **Newton stays the solver.** One iteration = one assembly, one AMG setup, one Krylov solve,
   two residual evaluations. Inputs keep `linear.rtol` at least 100x tighter than
   `newton.rtol` (as all existing inputs do), otherwise a second iteration starts at the floor.
   `cg_amg` is valid (SPD) and preferred in new inputs. Reuse of the constant tangent across
   load steps is a separate optional gate (LE6).
8. **Outputs under small strain:** `cauchy_stress = pk1_stress = sigma`, `jacobian =
   1 + tr(eps)`, `deformation_gradient = I + Grad u`, `thickness_stretch = 1 + eps_33`,
   `vonmises` from `sigma`; new quantity `strain` (6, VTK order): `eps` at small strain, the
   Green-Lagrange `E = (F^T F - I)/2` otherwise. Reaction moments use reference positions.
9. **Finite-strain numbers must not move.** LE2 refactors where `sigma` and `J` are formed;
   the finite-strain branch must be the same arithmetic, and `test_homogeneous` (all
   quantities, all presentations), the Cook and cantilever frozen values and
   `test_parallel` are the proof.

## 3. YAML surface

```yaml
plane: strain | stress                 # 2D; both supported
formulation: displacement | mixed      # mixed from LE4
material:
  model: linear_elastic
  E: 1000.0                            # or mu
  nu: 0.3                              # or kappa, or incompressible: true (mixed / plane stress, LE4)
  regions: [ { attr_names: [inclusion], E: 5000.0 } ]   # unchanged mechanism
solver:
  load_steps: 1
  newton: { rtol: 1e-10, atol: 1e-14 }
  linear: { type: cg_amg, rtol: 1e-13 }
output:
  fields: [displacement, strain, cauchy_stress, vonmises]
```

Errors (each a `ConfigError` with the key path): `volumetric` given; `mu` and `E` both given;
`bcs.traction[i].type: follower_pressure`; `solver.predictor: tangent`; before LE4 also
`incompressible: true` / `nu: 0.5` and `formulation: mixed`.

## 4. Gates

### LE1 — `LinearElastic` in the displacement formulation

Files: new `src/kernels/materials/kinematics.hpp` (trait + `CauchyStress`, `VolumeRatio`),
new `src/kernels/materials/linear_elastic.hpp`; `materials.hpp` (`LinearElastic` after
`Ogden`, `PlaneStress<LinearElastic>` after `PlaneStress<Ogden>`, `ModelName` branch **before**
the ogden fallthrough); `plane_stress.hpp` (trait forwarding); `materials.cpp`
(`MakeMaterial` branch; reject infinite kappa for this model until LE4, also under plane
stress, because the adapter's incompressible path is finite strain); `config.cpp` /
`config.hpp` (keys of decision 4, known-model list, comments); `solid_mechanics_tl.cpp`
(follower rejection, `Description()` suffix "(small strain)", reference-position moment arms
by passing a zero displacement to `LoadSet::Reactions`); `solid_problem.cpp` (predictor
rejection in `MakeSolidProblem`, where material and solver config meet).

Tests — extend `tests/test_materials.cpp`, new `tests/test_linear_elasticity.cpp` added to
`CHECK_TESTS`:

1. Material point. `TestMaterial` gains an `objective` flag (false here; every other check
   runs unchanged and the small-strain-limit check must now hold to 1e-13 instead of 1e-6).
   New: `P(I + W) = 0` to round-off for a skew `W` (invariance under infinitesimal rotation,
   the small-strain substitute for objectivity); the AD tangent equals the closed form
   `lambda d_ij d_kl + mu (d_ik d_jl + d_il d_jk)` to 1e-14 (lambda + 2 mu) at two random `F`
   (constant), hence minor and major symmetries; `PlaneStress<LinearElastic>` gives
   `E/(1 - nu^2)`, `E nu/(1 - nu^2)` and thickness strain `-nu/(1 - nu) (eps11 + eps22)` to
   1e-13 at strain 0.05 (not only in the limit).
2. Oracle. On affine Cartesian quad / tri / hex / tet meshes, p = 1, 2, the assembled
   Jacobian equals a `ParBilinearForm` with `mfem::ElasticityIntegrator(lambda, mu)` formed
   with the same essential dofs: `|K v - K_mfem v| <= 1e-12 |K_mfem v|` for a random `v`
   (affine meshes because the two integrators use different default quadrature orders; or
   hand MFEM's integrator the 2p + 3 rule). In 2D MFEM's integrator with the 3D `lambda, mu`
   is plane strain, which is what the padded `F33 = 1` gives.
3. Tangent of the verified nonlinear path: with matched moduli the assembled Jacobian of
   `neo_hookean`, `st_venant_kirchhoff` and `iso_neo_hookean` (finite kappa) at `u = 0`
   equals the `linear_elastic` Jacobian to 1e-12 (same random-vector check).
4. Linearity of the solve (perturbed 2D quad p = 2, 3D tet p = 1; traction + body force +
   nonzero Dirichlet data): Newton takes exactly one iteration from `x = 0` and from a random
   start and ends below `1e-10 |R0|`; at 1e3 times the load, still one iteration and
   `u` scales to 1e-10 (scale up, never down — decision 5); superposition
   `u(f1 + f2) = u(f1) + u(f2)` to 1e-10; Clapeyron `2 W_int = f_ext . u` to 1e-10 with
   homogeneous Dirichlet data (`InternalEnergy`, `ExternalLoad()`).
5. Patch test at amplitude 0.1 on perturbed quad / tri / hex / tet meshes: affine field
   reproduced to 1e-12 in one iteration (the nonlinear materials need amplitude <= 1e-5).
6. MMS with `ManufacturedBodyForce<LinearElastic>`: L2 rates >= 1.95 (p = 1), >= 2.95 (p = 2).
7. `st_venant_kirchhoff` vs `linear_elastic` on one traction problem at loads a, a/2, a/4:
   the relative difference falls by 1.8-2.2 per halving.
8. The `ConfigError`s of Section 3, next to the existing config-error tests.

Acceptance: `make check` green; no existing test, input or frozen number touched.

### LE2 — Outputs under small strain

`QPointState` gains `sigma` and `J`, filled by both evaluators through `CauchyStress` /
`VolumeRatio`; `PackQuantity` reads them and no longer forms `J^{-1} P F^T` itself;
`QPointCauchyStress` and `QPointMixedCauchyStress` use the helper; quantity `strain` added to
`Quantities()`, the `output.fields` validation and the `OutputConfig` comment.

Tests (in `test_linear_elasticity`): homogeneous states solved by FE — 3D uniaxial stress
with rollers, plane-strain and plane-stress uniaxial, pure shear, hydrostatic — at strain
**0.05**, so that a forgotten finite-strain push-forward is a 5 percent error, not a
round-off one. Displacement exact to 1e-12; every quantity in every presentation (nodes with
both projections, elements, quadrature points) equals Hooke's closed form to 1e-10:
`sigma_11 = E eps`, lateral strain `-nu eps`, `jacobian = 1 + (1 - 2 nu) eps`,
`energy_density = E eps^2 / 2`, `thickness_stretch = 1 - nu eps`, `cauchy_stress` equal to
`pk1_stress` componentwise, `strain` symmetric and equal to the prescribed one. Reaction
force equals `sigma A` and the moment balance closes with reference arms to 1e-10.

Acceptance: as above, plus bit-identical finite-strain outputs (decision 9): `make check`
and the `test_parallel` reference unchanged.

### LE3 — Closed-form verification cases and the linear Cook benchmark

Inputs under `apps/input/linear_elasticity/`, all `load_steps: 1`, `cg_amg`, order 2;
checked in `tests/test_verification.cpp` (second directory constant) unless noted.

| input | mesh, BCs | reference | check |
|---|---|---|---|
| `verification/lame_cylinder.yaml` | `annulus.msh`, plane strain, rollers on `bottom`/`left`, `pressure` on `inner` | `u_r = (1+nu)/E A [(1-2nu) r + b^2/r]`, `sigma_rr,tt = A (1 -/+ b^2/r^2)`, `A = p a^2/(b^2-a^2)` | `u_r` at a and b to 1e-6, stresses at three radii to 1e-4 |
| `verification/lame_sphere.yaml` | `sphere_octant.msh`, rollers on `xplane`/`yplane`/`zplane`, `pressure` on `inner` | `u_r = A/E [(1-2nu) r + (1+nu) b^3/(2 r^2)]`, `A = p a^3/(b^3-a^3)` | `u_r` to 1e-4, stresses to 1e-3 |
| `verification/kirsch_plate_with_hole.yaml` | copy of the finite-strain input, `linear_elastic`, far-field traction 1.0 | Kirsch: 3, -1, axis profile | 2 percent (finite width W = 20 a dominates), and within 1e-3 of the neo-Hookean run |
| `verification/timoshenko_cantilever3d.yaml` | `beam.msh`, traction 1000x the finite-strain input, `newton.rtol: 1e-10` | Euler-Bernoulli `P L^3/(3 E I)` | 0.2 percent; tip / load within 1e-4 of the small-load neo-Hookean value |
| `verification/mms_2d.yaml`, `mms_3d.yaml` | `square.msh`, `cube.msh`; body force written analytically as an `expression` (possible because the model is linear) | manufactured `u` | L2 rates p + 1 over `serial_refine` 0..3 |
| `cooks_membrane/cook_linear.yaml` | `cook.msh`, `plane: stress`, `E: 1`, `nu: 0.3333333333333333`, traction `["0", "0.0625"]` on `right` (resultant 1) | literature tip deflection about 23.9 (23.91-23.96; papers differ on mid-edge (48, 52) vs corner (48, 60)) | probe both points; literature as a 1 percent sanity band; freeze our converged 64x64 p = 2 value as the regression in `test_benchmarks`; report successive-difference ratios, gate them at >= 2.0 (the clamped corner singularity caps them near 2.5, as in the finite-strain Cook) |

Also: one `linear_elastic` case added to the `test_parallel` reference (np 1, 2, 4 agree to
1e-10); a two-material run on `inclusion.msh` through `material.regions` as a smoke test of
the region table (stress jump across the interface, continuity of traction to 1e-3).

Acceptance: `make test` red only at the pre-existing Cook ratio gate; added wall time under
two minutes (every case is one Newton iteration).

### LE4 — Nearly incompressible and incompressible: mixed u-p (Herrmann)

    R_u = int [2 mu dev(eps) + p I] : Grad w dV - loads,   R_p = int q (tr(eps) - p/kappa) dV

Kernel: inside `mixed_total_lagrangian.hpp` the volume measure becomes a trait switch —
`theta = 1 + tr H`, `G = d theta/dF = I` at small strain; `theta = J`, `G = J F^{-T}`
otherwise — used by `MixedPK1` (`P_iso + p G`), `QPointVolumeGradient`, the constraint
`u'(theta) - p/kappa` and the energy `p (theta - 1)`. The finite-strain branch is the present
arithmetic (decision 9). `LinearElastic` gains `PK1Iso = 2 mu dev(eps)`, `EnergyIso`,
`Incompressible()`, `ShearModulus()`, `NormalizedVolumetricPressure(theta) = theta - 1`,
`NormalizedVolumetricModulus = 1`, `ComplementaryVolumetricEnergy(p) = p^2/(2 kappa)`; it
joins `MixedMaterial` and `IsDecoupledModel`; the `mat.law = law` visit in `MakeDecoupled` is
guarded by the existing `has_volumetric_law`. `PlaneStress<LinearElastic>` gets the
small-strain incompressible branch (`eps_33 = -(eps_11 + eps_22)`, `p = -P_iso,33`,
`P = P_iso + p I`). `MixedSolidMechanicsTL`: follower rejection, outputs through the LE2
helpers, description. The LE1 rejection of infinite kappa is lifted. The saddle-point solver
is unchanged; `K_uu` has no pressure-weighted geometric term here and is positive
semi-definite, so expect outer iteration counts at or below the finite-strain ones (about 20,
mesh independent).

Tests (`test_mixed`, fast part in `make check`): patch test with constant pressure exact in
one iteration; MMS on Q2-Q1 at `nu = 0.3` with L2 rates 3 (u) and 2 (p); incompressible MMS
(`kappa = inf`, divergence-free `u` from a stream function, manufactured `p`) with the same
rates; mixed and displacement solutions at `nu = 0.3` converge to each other under
refinement; Lamé cylinder at `nu = 0.5`: `u_r = 3 p a^2 b^2 / (2 E (b^2 - a^2) r)` to 1e-6 and
the pressure unknown equal to the constant `A = p a^2/(b^2 - a^2)` to 1e-5 (the mean stress of
the incompressible Lamé field is uniform); locking record for the verification manual:
displacement p = 1 vs mixed at `nu = 0.4999` on the same mesh. Inputs
`verification/lame_cylinder_incompressible.yaml` and
`cooks_membrane/cook_linear_incompressible.yaml` (plane strain, mixed).

Acceptance: `make check` / `make test` as in LE3; frozen mixed numbers (incompressible Cook
corner `uy = 6.930412595013`) unchanged.

### LE5 — Documentation and handback

- `doc/theory_manual.tex`: new section "Small-strain elasticity" between the mixed
  formulation and the boundary conditions: linearisation of the total Lagrangian residual
  about `F = I` and the identity of the two weak forms; the model and its keys; a table
  *quantity | finite strain | small strain* for everything the trait switches; plane stress
  (the adapter recovers `lambda* = 2 lambda mu/(lambda + 2 mu)`); the Herrmann form; loads
  (dead = follower); solver remarks (one iteration, tolerances, predictor excluded, the
  `1e-16/|Grad u|` floor). Update "What is implemented", the seam section (kinematics is a
  material trait), the symbols / input keys and source-file appendices. Notation: subscript
  R for referential quantities, bold B; say once that at small strain the reference
  configuration is the domain of integration.
- `doc/verification_manual.tex`: `test_linear_elasticity` subsection; section "Verification
  cases (`apps/input/linear_elasticity/`)" with problem / reference / check / tolerance /
  test for every LE3-LE4 input; summary-table rows.
- `README.md`: model list and keys, a short small-strain paragraph, input table rows, the
  "not supported" list of Section 5. `src/base/config.hpp` schema comments.
- Build both manuals with `latexmk -pdf` in `doc/`, delete the auxiliary files, keep the PDFs.

### LE6 (optional, only when multi-step linear runs appear) — reuse the constant tangent

`SolidMechanicsTL::GetGradient` caches the assembled `HypreParMatrix` when the material is
small strain and no follower load exists (invalidate in `Finalize`, `ResetForm`,
`ClearBoundaryConditions`); `MakeLinearSolver` tells `LinearSolver` to skip the AMG rebuild
when `SetOperator` receives the same matrix object. Acceptance: a 10-step `table` schedule
run is bit-identical to the uncached run and at least 3x faster in assembly + setup time.

## 5. Out of scope (state in the README as "not supported")

- Anisotropic linear elasticity (needs material axes in the input). Natural route later: a
  `Linearized<M>` adapter holding `C = MaterialTangent(M, I)` computed once, `PK1(F) = C :
  (F - I)`; it inherits the minor symmetries because `P(I) = 0`.
- Thermal and shrinkage eigenstrains, `sigma = C : (eps - eps*)`: needs field-dependent
  materials (the seam section's "temperature dependence absent by design"); this plan keeps
  `eps` formation in one place (`LinearElastic::Strain`) so that is where it will enter.
- Linear dynamics, modal analysis, linear buckling (needs a geometric stiffness), corotational
  kinematics, small-strain plasticity / viscoelasticity (internal variables).
- A `ParBilinearForm` / partial-assembly path, device execution, renaming `SolidMechanicsTL`.

## 6. Order of work and effort

LE1 (0.5-1 day) -> LE2 (0.5 day) -> LE3 (1 day) -> LE4 (1 day) -> LE5 (0.5-1 day); LE6
(0.5 day) on demand. LE1 alone gives a usable displacement-formulation model with correct
displacements, reaction forces and energies; stresses in the ParaView output are only right
after LE2, so do not hand LE1 to a user without it. LE3 does not depend on LE4.
