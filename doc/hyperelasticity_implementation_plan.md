# Implementation Plan — Framework Bootstrap via Hyperelasticity (Gates S0–S5)

**Audience:** the implementing agent. This document is the contract for the work: follow the
gate order, meet each gate's acceptance criteria before moving on, and commit once per
completed gate. The architecture this implements is described in
`doc/flux_kernel_architecture.html` (Figure 1); read it first.

**Goal:** bootstrap the flux-kernel multiphysics framework in `src/` + `apps/`, with
**quasi-static nonlinear solid mechanics (hyperelasticity, total Lagrangian)** as the first
physics. The framework layers (base, kernels, materials, physics, solvers) are built only as
far as this physics needs them — but with the interfaces shaped so heat transfer, species
transport, and compressible flow slot in later without rework.

---

## 1. Environment facts (verified 2026-09-05 — re-verify before starting)

- Repo: `/home/quinnchr/dv/Continuum-Mechanics-MFEM`. Root `src/` exists and is **empty**;
  create `apps/` and `tests/` at the root alongside it.
- MFEM: `~/MFEM/mfem` (`config/config.mk` present). Built with
  `MFEM_USE_MPI=YES`, `MFEM_USE_METIS=YES`, `MFEM_USE_PETSC=YES` (HYPRE implied),
  `MFEM_USE_CUDA=NO`, `MFEM_USE_OPENMP=NO`, `MFEM_USE_SUNDIALS=NO`, `MFEM_USE_GSLIB=NO`.
  `MFEM_CXX=mpicxx`, `-O3 -std=c++17`.
  **Consequence:** everything in this plan runs on CPU. Write quadrature-point code as
  plain callables that *could* be `MFEM_HOST_DEVICE` (no virtual dispatch inside qpoint
  loops, no heap allocation per qpoint), but do not add device annotations or `mfem::forall`
  plumbing yet — that is a later gate, after a CUDA MFEM build exists.
- yaml-cpp 0.8.0 via `pkg-config`; Open MPI 4.1.6 (`mpirun -np 4` works).
- Makefile conventions: copy the MFEM `config.mk` discovery block from
  `myapps/hypersonic_cfd/makefile` (lines 4–34) — it finds `~/MFEM/mfem` automatically.
  Link with `$(MFEM_CXX) $(MFEM_FLAGS) ... $(MFEM_LIBS) $(YAML_LIBS) -lyaml-cpp`.
- **Do not touch** `myapps/` — `hdg_navierstokes` and `hypersonic_cfd` are frozen
  regression oracles for other work. `myapps/` is legacy and will be deprecated separately.

## 2. Target layout (create in this gate sequence, not all at once)

```
Continuum-Mechanics-MFEM/
├─ src/
│  ├─ base/        tensor.hpp, dual.hpp, input (YAML), field registry, output
│  ├─ kernels/     qfunction contract + CG assembly integrators
│  ├─ materials/   material functors (NeoHookean, StVenantKirchhoff)
│  ├─ physics/     solid_mechanics_tl.{hpp,cpp}
│  └─ solvers/     newton.{hpp,cpp}, linear solver factory
├─ apps/           solid_mechanics executable (thin: YAML + wiring only)
├─ tests/          unit + MMS + benchmark tests (link the library)
├─ makefile        builds libcmf.a from src/, then apps/ and tests/ against it
└─ doc/
```

Library name `libcmf.a` is a working placeholder; make it a single `LIBNAME` variable in the
makefile so renaming is one line.

**The thinness rule (enforced, not aspirational):** nothing in `apps/` contains physics,
assembly, or solver logic — if an app needs code beyond YAML parsing and wiring, that code
moves into `src/` first.

## 3. Design decisions already made — do not re-litigate

1. **Total Lagrangian on the reference mesh.** The mesh never moves. Unknown is displacement
   `u` in `[H¹(Ω₀)]^d`; residual uses the first Piola–Kirchhoff stress:
   `R(u)·w = ∫ P(F) : Grad w dV − ∫ ρ₀ b·w dV − ∫ T̄·w dA`, `F = I + Grad u`.
   The math is written out in `doc/solid_mechanics_forms.tex` (since merged into `doc/theory_manual.tex`, Section 3) — implement exactly that weak
   form (quasi-static: no inertia term).
2. **Materials are stateless functors templated on scalar type.** Tangents come from
   forward-mode dual numbers flowing through the same code path as the stress evaluation.
   Hand-coded tangents are a later optimization, not part of this plan.
3. **Full assembly (FA) via `mfem::ParNonlinearForm` + a custom
   `NonlinearFormIntegrator`.** Partial assembly / matrix-free is out of scope until GPU
   work starts. The qpoint contract is what keeps that door open.
4. **CG discretization only** for solids in this plan. DG/HDG for solids: out of scope.
5. **Dead loads only.** Follower (pressure) loads: out of scope, leave a TODO seam.
6. **Compressible hyperelasticity only.** Near-incompressibility (mixed u-p) is out of
   scope; pick ν ≤ 0.3 in all tests so pure-displacement CG is well behaved.

## 4. Core interfaces (signatures are the contract; implementation details are yours)

### 4.1 `src/base/tensor.hpp` — fixed-size tensor

Small stack tensor `tensor<T, m, n>` (and `tensor<T, n>`), with: `+ − *` (scalar and
contraction), `transpose`, `det`, `inv`, `dot`, `ddot`, `outer`, `I<n>()`. Templated on `T`
so duals flow through. No dynamic allocation. Unit-test `det`/`inv` for 2×2 and 3×3
against hand values.

### 4.2 `src/base/dual.hpp` — forward-mode dual number

`dual { double v; double d; }` with arithmetic and the functions materials need
(`log`, `sqrt`, `pow`). `tensor<dual,3,3>` must work. Unit-test: derivative of a scalar
composite function vs central finite differences to 1e-10.

### 4.3 `src/materials/` — material contract

```cpp
// Every hyperelastic material is a cheap-to-copy value type:
struct NeoHookean {
  double mu, lambda;                       // from YAML: E, nu -> (mu, lambda)
  template <typename T>
  tensor<T,3,3> PK1(const tensor<T,3,3>& F) const;   // P = mu(F - F^{-T}) + lambda ln(J) F^{-T}
};
struct StVenantKirchhoff { /* P = F (lambda tr(E) I + 2 mu E) */ };
```

2D problems run as plane strain: embed the 2×2 `F` into 3×3 with `F₃₃ = 1` and use the 3D
material everywhere (one code path, no 2D constitutive variants).

The qpoint tangent `A_ijkl = ∂P_ij/∂F_kl` is computed by seeding duals: 9 (3D) / 4 (2D
in-plane) evaluations of `PK1` per qpoint, each with one component of `F` perturbed. This
happens in the integrator, not in the material — materials never know about duals
explicitly, they are just templated.

### 4.4 `src/kernels/` — the flux/source seam (minimal version for this plan)

For this plan the seam is one custom integrator:

```cpp
// Consumes any material with the PK1<T>(F) signature (template, not virtual):
template <typename Material>
class TotalLagrangianIntegrator : public mfem::NonlinearFormIntegrator {
  // AssembleElementVector: loop qpoints, F = I + Grad u, accumulate P : Grad w  * weight
  // AssembleElementGrad:   loop qpoints, build A = dP/dF via duals, accumulate B^T A B
};
```

Body force and boundary traction use stock MFEM integrators
(`VectorDomainLFIntegrator`, `VectorBoundaryLFIntegrator`) on the RHS/linear form side.
Generalizing this to the full `F(u,∇u)·S(u)·F̂` contract of the architecture doc happens
when the second physics arrives — do not build speculative abstraction now, but keep the
qpoint loop body a free function so it can be lifted later.

### 4.5 `src/physics/solid_mechanics_tl.{hpp,cpp}`

`class SolidMechanicsTL : public mfem::Operator` owning: the `ParFiniteElementSpace`
(vector H¹, order p from YAML), essential BC dof list, the `ParNonlinearForm`, load
scaling factor for incremental loading. `Mult` = residual with essential dofs handled,
`GetGradient` = assembled `HypreParMatrix` with eliminated BCs. Constructor takes
(mesh, YAML node, material variant). Selection between the two materials by YAML string —
dispatch once at setup (e.g. instantiate the templated integrator per material type),
never per qpoint.

### 4.6 `src/solvers/`

- `newton.hpp`: damped Newton with backtracking line search (halve step while
  `||R|| > (1 − c·α)||R_old||`, c = 1e-4, max 8 halvings), absolute + relative tolerance,
  iteration log (it, ||R||, α). Crib the structure quality from
  `myapps/hypersonic_cfd/solvers/newton.hpp` but this one operates on a plain
  `mfem::Operator`.
- Linear solver factory (YAML-selected): default GMRES + `HypreBoomerAMG` with
  `SetElasticityOptions(fespace)` (falls back to `SetSystemsOptions(dim, byVDIM)` if
  elasticity options misbehave); optional direct-ish fallback for small tests (CG + AMG or
  SuperLU if available). PETSc paths: out of scope for this plan.
- Load stepping: outer loop scaling all loads by λ ∈ (0, 1] in `n_steps` increments, each
  solved by Newton warm-started from the previous step. In YAML: `load_steps: N`.

### 4.7 `src/base/` input + output

- YAML schema (keep flat and boring):

```yaml
mesh: { file: ..., serial_refine: 1, parallel_refine: 0, order: 2 }
material: { model: neo_hookean, E: 250.0, nu: 0.3, rho0: 1.0 }
bcs:
  dirichlet: [ { attr: [1], value: [0.0, 0.0] } ]
  traction:  [ { attr: [2], value: [0.0, 6.25] } ]
body_force: [0.0, 0.0]
solver: { load_steps: 1, newton: { rtol: 1e-10, atol: 1e-12, max_it: 25 },
          linear: { type: gmres_amg, rtol: 1e-12, max_it: 500 } }
output: { paraview: out/cook, fields: [displacement, vonmises] }
```

- ParaView output: displacement (as the "Warp by Vector" field), von Mises of Cauchy
  stress (compute σ = J⁻¹ P Fᵀ at qpoints, project to an L² space for output).

## 5. Gates

Each gate = tasks + acceptance criteria + a commit
(`Complete S<n>: <summary>` + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`).
Wire every gate's checks into the top-level makefile: `make check` runs S1–S3 unit/MMS
tests (fast, serial), `make test` runs everything including `mpirun -np 4` runs — mirror
the `check`/`test` pattern of `myapps/hypersonic_cfd/makefile`.

### S0 — Skeleton and build system

Tasks: repo layout of §2; top-level makefile (config.mk discovery block, `libcmf.a` from
`src/`, apps and tests link it); `apps/solid_mechanics.cpp` stub that parses YAML, loads a
mesh (`mfem::Mesh::MakeCartesian2D` fallback when no file given), builds a
`ParFiniteElementSpace`, writes ParaView output of a zero field.

Accept when:
- `make` builds clean (no warnings with `-Wall`) serial code path;
- `./apps/solid_mechanics -i <yaml>` and `mpirun -np 4 ./apps/solid_mechanics -i <yaml>`
  both run and produce ParaView output;
- `make clean && make` works from scratch.

### S1 — base utilities: tensor + dual + YAML plumbing

Tasks: §4.1, §4.2, YAML → config structs with validation errors that name the bad key.

Accept when `tests/test_base` passes:
- tensor algebra vs hand values (det/inv/ddot, 2×2 and 3×3);
- dual derivatives vs central FD ≤ 1e-10 relative;
- malformed YAML produces a clear error, not a crash.

### S2 — material library

Tasks: §4.3 with NeoHookean + StVenantKirchhoff; qpoint tangent builder (dual seeding)
as a free function `MaterialTangent(material, F) -> A` used later by the integrator.

Accept when `tests/test_materials` passes:
- `P(I) = 0` to machine precision, both materials;
- AD tangent vs central FD of `PK1` (step 1e-6) ≤ 1e-6 relative, at 5 random `F` with
  `det F > 0` (fixed RNG seed);
- objectivity: `P(QF) = Q P(F)` for random rotations `Q` ≤ 1e-12;
- small-strain limit: for `F = I + εH`, ε = 1e-7, both materials' stress matches the linear
  elasticity tensor `λ tr(ε̂) I + 2μ ε̂` to O(ε) — this catches λ/μ conversion bugs.

### S3 — solid physics module + Newton: patch test and MMS

Tasks: §4.4–§4.6 complete; the app now solves quasi-static problems end to end.

Accept when `tests/test_solid_mms` passes:
1. **Patch test:** affine manufactured displacement `u = A X + c` imposed as Dirichlet data
   on the whole boundary of an unstructured (perturbed-node) 2D quad mesh, zero body
   force. Discrete solution reproduces `u` to 1e-12 for p = 1 and p = 2, both materials,
   and Newton converges in ≤ 2 iterations (the problem is affine ⇒ one true Newton step).
2. **MMS convergence:** smooth manufactured displacement, e.g.
   `u = α (sin(πX)sin(πY), X²Y(1−Y))` with α sized so max|Grad u| ≈ 0.1 (genuinely
   nonlinear, far from singular `J`). Body force `b = −(1/ρ₀) Div P(X)` evaluated
   **numerically**: `P(X)` is a closed-form composition (manufactured `F(X)` → material),
   so compute its divergence by central finite differences in `X` with step 1e-5 at each
   quadrature point (FD error ~1e-10, far below discretization error — do not derive
   `Div P` by hand). Dirichlet = exact `u` on the whole boundary. Gate: L² displacement
   error rates on 3 uniform refinements ≥ p+0.9 for p = 1, 2, NeoHookean.
3. **Newton quality:** on the finest MMS mesh, the last two Newton steps show quadratic
   contraction (log ratio test, generous tolerance) — catches inconsistent Jacobians,
   which AD should preclude but wiring bugs (BC elimination, load scaling) do not.

### S4 — benchmarks, parallel consistency, regression freeze

Tasks: Cook's membrane and a 3D case as YAML-driven runs of the *same* app (no per-case
code); parallel consistency test; freeze regression values.

1. **Cook's membrane (compressible variant):** standard geometry — quadrilateral with
   corners (0,0), (48,44), (48,60), (0,44) [mm], left edge clamped, uniform upward shear
   traction on the right edge; plane strain, NeoHookean, `E = 250, ν = 0.3`, total traction
   resultant chosen to give a visibly nonlinear deflection (start with 6.25 per unit length
   and adjust so the top-right corner vertical displacement is ~25–35% of the 16 mm edge —
   record what you chose). Reference = self-convergence: p = 2, 4 uniform refinements;
   gate is monotone convergence of the corner displacement with successive differences
   shrinking by ≥ 3× per refinement. **Freeze** the finest value in the test as the
   regression oracle (assert within 1e-8 relative thereafter).
2. **Small-load linear limit:** 3D cantilever (Cartesian hex mesh, one end clamped, end
   shear traction) at a load scaled so max|Grad u| ≤ 1e-4; NeoHookean tip displacement vs
   the analytic Euler–Bernoulli estimate within 15% (sanity, geometry-dependent), and vs
   the StVK solution within 1e-6 relative (both must agree in the linear limit — a strong
   cross-check of two independent material codes).
3. **Parallel consistency:** serial run writes reference norms (`||u||_L2`, corner
   displacement); `mpirun -np 2` and `-np 4` reproduce them ≤ 1e-10 relative (same mesh,
   same solver tolerances tightened to 1e-14 for this test).
4. `make check` < 1 minute; `make test` runs everything green, serial and np=4.

### S5 — documentation and handback

Tasks:
- `README.md` at repo root section (or `doc/framework.md`): layout, build (`make`,
  `make check`, `make test`), how to add a material (walk through NeoHookean), how to run
  Cook's membrane, YAML schema reference.
- A short "state of the seam" note in the same doc: what of §4.4 is still
  solid-specific and what the second physics will need to generalize (this is the honest
  ledger for the next planning round).

Accept when a cold reader can build, run `make test`, and run Cook's membrane from the
README alone.

## 6. Out of scope (do not start these)

Dynamics/inertia; PA/matrix-free and any GPU/`mfem::forall` work; DG/HDG solids;
mixed/incompressible formulations; follower loads; plasticity or any material state
(`QuadratureFunction` internal variables); PETSc SNES; coupling layer; ALE; AMR.
If a gate seems to require one of these, stop and report instead of expanding scope.

## 7. Working agreements

- Verify the environment facts of §1 first; if any fail, stop and report.
- One commit per gate, message `Complete S<n>: ...`; intermediate WIP commits are fine
  (`S<n> in progress: ...`).
- If a gate's acceptance threshold cannot be met, do not weaken the threshold — report
  what was measured and why.
- Tests are the deliverable as much as the library: every acceptance bullet above must map
  to an executable check under `tests/`, wired into `make check`/`make test`.
- Keep files ASCII, match MFEM's code style loosely (2-space indent, `CamelCase` types,
  MFEM idioms for FEM objects), comments only where the code can't say it.
