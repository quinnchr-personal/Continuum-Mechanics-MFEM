# Implementation Plan — Finite Thermoelasticity: the coupled u–p–θ problem, axisymmetry, and the six examples of Anand's chapter 3 (Gates TE1–TE4)

**Audience:** the implementing agent. Follow the gate order, meet each gate's acceptance
criteria before moving on, commit once per completed gate (on `main`, no attribution
trailers). This extends the framework of `doc/hyperelasticity_implementation_plan.md`,
`doc/bc_loading_plan.md`, `doc/linear_elasticity_plan.md` and `doc/solid_dynamics_plan.md`
and the finite viscoelasticity of 2026-09-22 (quadrature-point histories, the `time` block,
the rigid-sphere contact); their design decisions (reference mesh never moves, thin `apps/`,
`myapps/` untouched, materials are value types templated on the scalar type, no per-exercise
drivers, both manuals kept current) still hold.

**Status (2026-09-22):** TE1–TE4 done. TE1 (axisymmetry) commit 259c0d7c66, TE2 (the coupled
module, point constraints, the schema, `test_thermoelastic`) 1233e8190c, TE3 (the six inputs,
meshes, reference histories, `anand_thermo_plots.py`) 128f64e392, TE4 the manuals and the README.
Measured agreement with the reference: the homogeneous stretch (02) to 1e-8, 01/03/04 to
0.01–0.3 %, the plate's late state to 0.1 % (its early transient is a through-thickness
resolution matter), the sail 3–4 % on hexahedra and 0.03 % on the reference's own tetrahedra.
Deviations from the plan as written: the tangent K_θθ is the exact derivative (the dual seed
differentiates θ M too); the optional heat-flow output of item 8 was not done (the plots do not
need it); the sail pins are the five nodes of each edge, since the reference's pins are the
whole edges.

## 0. What the examples need

The "3. Finite Thermoelasticity" pages of solidmechanicscoupledtheories.github.io (FEniCSx
codes `03_finite_thermoelasticity/TE01`–`TE06`, dolfinx 0.8; histories already recorded with
the notebooks run headless, see gate TE3) solve the coupled problem of Anand's chapter on
the thermoelasticity of elastomers with the unknowns displacement (P2), pressure (P1) and
temperature (P1) on one mixed element:

| Case | Geometry | Loading | Output plotted |
|------|----------|---------|----------------|
| TE01 constrained heating | plane strain 10 x 10 mm, 6 x 6 crossed triangles | rollers on x = 0, x = 10, y = 0; θ = θ0 + 50 (1 − e^(−t/20)) K on y = 10; 400 s, dt 1 | θ at (5, 10), (5, 5), (5, 0); pressure at (5, 0) |
| TE02 adiabatic stretch | axisymmetric cylinder R = 10, H = 10, 20 x 20 | u_z = 70 t/100 on z = 10 (stretch 8); insulated; 100 s, dt 1 | nominal stress (force/(π R²)) and Δθ against the stretch |
| TE03 heating-induced contraction | the same cylinder | step 1: dead traction 2 MPa on z = 10 in 50 s at θ0 on the outer faces; step 2: θ ramped +50 K over 50 s on r = R, z = 0, z = 10 and held to 300 s; dt 2 | stretch against stress and against surface temperature |
| TE04 bilayer actuator | plane strain 100 x 1 mm, two layers (`bilayer_beam.geo`), α = 180e-6 in the bottom layer and 0 in the top one | u_x = 0 on the left edge, the node (0, 0) pinned; θ ramped +50 K over 3600 s on bottom, right and top edges; dt 60 | tip deflection / L against surface temperature |
| TE05 circular plate, surface flux | axisymmetric plate R = 50, thickness 1, 20 x 2 | inward heat flux 1e4 μW/mm² per unit current area on the top face; θ = θ0 on the bottom face; the node (R, 0) pinned; 20 s, dt 0.2 | surface temperature and centre deflection against time |
| TE06 solar sail | 3D 100 x 100 x 1 mm, 10 x 10 x 2 box of tetrahedra | corners (0, 0, 0) and (L, L, 0) pinned, u_x = 0 on x = 0, u_y = 0 on y = 0; θ ramped +50 K over 100 s on z = 0, z = 1, x = L, y = L; follower pressure ramped to 10 Pa on the top face; θ0 = 273; 100 s, dt 1 | u_z at (70, 60) and (100, 0) against pressure |

Material of all six (kPa, mm, s, K, μJ): Arruda–Boyce with G0 = N_R k_B θ0 = 280 kPa,
λ_L = 5.12, K = 1000 G0, entropic shear modulus G(θ) = N_R k_B θ ζ (i.e. G0 θ/θ0 ζ),
thermal expansion α = 180e-6 /K in the pressure constraint p/K + [ln J − 3α(θ − θ0)]/J = 0,
heat capacity c_v = 1839 per unit reference volume, spatial conductivity k = 160 μW/(mm K),
θ0 = 298 K (273 in TE06). Weak forms of the reference (per unit reference volume, r-weighted
when axisymmetric):

    ∫ P : Grad w = ∫ t̄ · w                                  P = J^(−2/3) G(θ) (F − ⅓ tr C F^(−T)) − J p F^(−T)
    ∫ q [p/K + (ln J − 3α(θ − θ0))/J] = 0
    ∫ c_v (θ − θ_old) q − ½ θ M : (C − C_old) q − Δt Q · Grad q = −Δt ∮ q̄ q      Q = −J k C^(−1) Grad θ,  M = ∂S/∂θ

with M = J^(−2/3) N_R k_B ζ (I − ⅓ tr C C^(−1)) − 3Kα C^(−1) (the derivative of the
displacement-form second Piola–Kirchhoff stress, whose pressure is the constitutive one), the
Gough–Joule heating ½ θ M : Ċ, implicit Euler in the temperature, and the reference's
quadrature degree 2.

Three of the six are axisymmetric and three pin single nodes; the framework has neither, and
no temperature field. Everything else exists: the mixed u–p kernel and module, the `time`
block, the logarithmic law, follower pressures, regions, probes and reactions.

## 1. Current state (verified 2026-09-22 — re-verify before starting)

- Kernels take a `dim x dim` displacement gradient and embed it as plane strain,
  `DeformationGradient<dim>(H)` (`src/kernels/total_lagrangian.hpp`); the tangent loops seed
  the in-plane `dim x dim` block only (`material_tangent.hpp`, `QPointMixedTangent`). The
  quadrature weight is `ip.weight * Tr.Weight()`. The mixed kernel
  (`src/kernels/mixed_total_lagrangian.hpp`) evaluates `P_iso + p J F^-T`, the constraint
  `u'(J) - p/kappa` and the blocks `K_uu, K_up, K_pu, K_pp` explicitly; the face kernels
  (`follower_pressure.hpp`, `rigid_sphere_contact.hpp`) compute `F` from the element dofs
  and integrate with `CalcOrtho` of the face Jacobian.
- A history-dependent material binds per point through `AtPoint` / `HistoryBound`
  (`src/kernels/history_bound.hpp`); `HistoryField` (`src/kernels/history_field.hpp`) stores
  blocks of doubles per quadrature point of the rule of order 2k + 3, with `Dt()`; the
  physics sets `Dt` in `SetLoadFactor` and commits in `AcceptStep`.
- The mixed module `MixedSolidMechanicsTL` (`src/physics/mixed_solid_mechanics_tl.*`): a
  `ParBlockNonlinearForm` on `(fes_u, fes_p)`, block offsets, essential dofs on block 0
  only, `FullResidual`, reactions on the displacement block, `UpdateFields` with
  `QuadratureFields`, the direct solver through `DirectSolver` (`solvers/direct_solver.*`,
  any `mfem::Operator` with `HypreParMatrix` blocks) and the saddle-point solver.
- Loads: `LoadSet` (`src/physics/loads.*`) takes boundary attributes only; essential dofs
  come from `fes.GetEssentialTrueDofs(marker, list, component)`; the reference node
  coordinates are formed once (`EnsureCoords`); dead loads are assembled with MFEM's
  `VectorBoundaryLFIntegrator` / `VectorBoundaryFluxLFIntegrator` / `VectorDomainLFIntegrator`.
- Schema (`src/base/config.*`): `plane: strain | stress`, `MaterialConfig` with `regions`
  merging, `BCConfig{dirichlet, traction, contact}`, `TimeConfig`, `OutputConfig.fields` as a
  closed list (`ParseOutputConfig`). The app (`apps/solid_mechanics.cpp`) branches on
  `dynamics` / `time` and calls `physics.UpdateFields`, `Reactions`, probes of every
  registered field.
- Reference histories of TE01–TE06 are recorded (session scratchpad,
  `03_finite_thermoelasticity/hist/*.csv`; columns: TE01 t, θ(5,10), θ(5,5), θ(5,0),
  p(5,0); TE02 t, u_z top, 2π∫P22 r dA, θ(0,H), reaction of the top face; TE03 t, u_z top,
  force, θ; TE04 t, u_y tip, θ tip; TE05 t, u_z(0,H), force, θ(0,H); TE06 t, u_z(A), u_z(B)).
  Their run scripts (`TE*_run.py`, `patch_notebooks.py`, `run_all.sh`, `times.txt`) are to be
  copied into `apps/input/anand_coupled_theories/finite_thermoelasticity/reference/scripts/`
  at TE3, as the finite_viscoelasticity set did.

## 2. Design decisions — do not re-litigate

1. **Axisymmetry is a kinematics option of every kernel, not a new physics:**
   `plane: axisymmetric` on a 2D mesh with x = r and y = z. The deformation gradient is
   the 2 x 2 block plus F_33 = 1 + u_r / r (F_rr where r < 1e-12 of the mesh size, the
   limit on the axis), every integral carries the weight 2π r (per unit radian times 2π, so
   that reactions, forces, energies and the mass are the totals of the solid of
   revolution), and the test-function gradient carries the hoop term w_r / r in the
   (3, 3) slot. The kernels generalise their B-operator: per dof (a, i) the strain-like
   vector holds dF_ij = DS(a, j) for the in-plane entries and dF_33 = δ_ir N_a / r; the
   material tangent is seeded on the same entries. The face kernels weight by 2π r and use
   the completed F. `LoadSet` multiplies the dead-load, pressure and body-force coefficients
   by 2π r; the dynamics decorator's density coefficient is multiplied likewise
   (`ReferenceDensity()` of the physics returns the product under axisymmetry); the
   quadrature outputs use the completed F. Plane strain and 3D are untouched: the
   axisymmetric branch is a runtime flag of the kernels, checked once per element.
2. **Point constraints:** a Dirichlet entry with `point: [x, y(, z)]` instead of `attr`
   prescribes the components at the displacement node nearest to the point (global
   MINLOC over the owned true dofs; error if farther than 1e-6 of the mesh diameter). Its
   data is the expression at that point; schedule and reactions as for face entries.
   Nothing changes in the essential-dof machinery: the entry contributes its true dofs.
3. **The thermo-mechanical problem is a third module,** `ThermoSolidMechanicsTL`
   (`src/physics/thermo_solid_mechanics_tl.*`), unknown [u; p; θ] with u of order k and p, θ
   of order k − 1 (the reference's element), selected by `formulation: mixed` together with
   a `material.thermal` block. The displacement formulation with `thermal` is an input
   error (the constraint carries the thermal expansion; a penalty variant is out of scope).
   `dynamics` with `thermal` is an input error (out of scope); the `time` block is required.
4. **The material is an adapter** `Thermoelastic<Base>` over the six decoupled models
   (`src/kernels/materials/thermoelastic.hpp`), in its own variant `ThermoMaterial`
   consumed only by the thermo kernel (the `Material` / `MixedMaterial` variants are not
   touched). Parameters `theta0, alpha, c_v, k, entropic` (default true). Energy per unit
   reference volume
       ψ(F, θ) = (θ/θ0)^s Ψ_iso^base(F̄) + κ u(J / J_θ) − c_v [θ ln(θ/θ0) − (θ − θ0)],
       J_θ = exp(3α (θ − θ0)),  s = 1 (entropic) or 0,
   so that P_iso = (θ/θ0)^s P_iso^base, the constitutive pressure U'(J, θ) = κ u'(J_m)/J_θ
   with J_m = J/J_θ, the mixed constraint u'(J_m)/J_θ − p/κ = 0 and its tangent
   u''(J_m)/J_θ². For the logarithmic law this is exactly the reference: U' = K (ln J −
   3α Δθ)/J and ∂U'/∂θ = −3Kα/J. The thermal tangent M = F^-1 ∂P_disp/∂θ comes from a dual
   seed on θ of the displacement-form stress (`PK1<T>(F, θ)` with θ templated), the
   isotropic Fourier flux Q = −k J C^-1 Grad θ, and the heat capacity is c_v per unit
   reference volume (the reference's number). The heat equation is
       c_v θ̇ = ½ θ M : Ċ − Div Q,
   implicit Euler over the step of the `time` block with C_old, θ_old at the quadrature
   points held in a `HistoryField` (7 doubles per point: C_old in the 6-component order,
   θ_old) updated in `AcceptStep`; the tangent K_θθ takes ∂(θ M)/∂θ = M, dropping θ ∂M/∂θ
   (zero for the entropic model with the logarithmic law, O(α² κ) for the others: state it).
5. **Thermal boundary conditions** in `bcs`: `temperature` entries (Dirichlet on θ, one
   expression f(x, y, z, t), schedule as the others, default constant under `time`;
   `point:` allowed) and `heat_flux` entries (inward heat flux h per unit current area by
   default, `per_unit: current_area | reference_area`; per current area it is
   h |cof F N| and its tangent by duals over the face dofs, as the follower pressure). No
   convection, no volumetric heat source, no thermal contact. Adiabatic is the natural
   condition. The initial temperature is θ0 everywhere (`material.thermal.theta0`; regions
   may not change it: one reference temperature per problem).
6. **Kernel:** `ThermoMixedTotalLagrangianIntegrator` (`src/kernels/thermo_mixed_total_lagrangian.hpp`),
   a 3-block `BlockNonlinearFormIntegrator` with the residual densities
       r_u = P(F, p, θ) : Grad w,   r_p = q [u'(J_m)/J_θ − p/κ],
       r_θ = q [c_v (θ − θ_old) − ½ θ M(F, θ) : (C − C_old)] + Δt k J (C^-1 Grad θ) · Grad q,
   and the nine tangent blocks by dual seeds: F entries (K_uu, K_pu, K_θu, the last through
   M(F), C(F), J C^-1), p (K_up, K_pp; K_θp = 0), θ (K_uθ through P_iso(θ) and, with the
   constitutive pressure absent from r_u, nothing else; K_pθ through J_θ; K_θθ through
   c_v, ½ M and the flux), Grad θ (K_θθ conduction). Reuse the mixed kernel's element setup;
   do not derive from it (the block count differs).
7. **Solver:** the direct solver (MUMPS) for the 3-block Jacobian; `solver.linear.type`
   other than `direct` is an input error for the thermo module ("the saddle-point
   preconditioner has no temperature block"). Newton as is; the tangent predictor works.
8. **Outputs:** the nodal field `temperature` (registered like `pressure`), probes of it;
   the quadrature quantities as in the mixed module (P with the field pressure and the
   entropic modulus); reactions of the Dirichlet entries; plus, for every `temperature`
   entry, the heat flow into the body through its face (the θ-residual summed over its
   dofs, divided by Δt) printed and written as `heat <name>: flow = ...` in the log and as
   `<name>_q` in reactions.csv — optional, do it if the plots need it (they do not).

## 3. YAML surface

```yaml
plane: axisymmetric                # 2D only: strain (default) | stress | axisymmetric (x = r, y = z)
material:
  model: arruda_boyce
  mu: 280.0
  N: 26.2144
  kappa: 280000.0
  volumetric: logarithmic
  thermal: { theta0: 298.0, alpha: 180.0e-6, c_v: 1839.0, k: 160.0, entropic: true }
  regions: [ { attr: [top_layer], thermal: { alpha: 0.0 } } ]   # keys not given inherit the base's
time: { t_final: 400.0, dt: 1.0 }
bcs:
  dirichlet:
    - { attr: [left], expression: ["0", "0"], components: [x] }
    - { point: [0.0, 0.0], name: pin, expression: ["0", "0"] }          # a node
  temperature:
    - { attr: [top], name: heated, expression: "298 + 50*(1 - exp(-t/20))" }
    - { attr: [bottom], expression: "298" }
  heat_flux:
    - { attr: [top], expression: "1.0e4", per_unit: current_area }      # inward, μW/mm^2
  traction: [ { attr: [top], type: follower_pressure, expression: "0.01", schedule: { type: ramp, from: 0, to: 100 } } ]
output:
  fields: [displacement, pressure, temperature, cauchy_stress]
  probes: [ { name: top, point: [5.0, 10.0] } ]
```
Errors: `thermal` without `time`; `thermal` with `dynamics`; `thermal` with
`formulation: displacement` or `plane: stress`; `heat_flux` / `temperature` without
`thermal`; `plane: axisymmetric` on a 3D mesh or with `plane: stress`; a `point` entry
farther than the tolerance from every node; `point` and `attr` in one entry.

## 4. Gates

### TE1 — Axisymmetry in the existing kernels, loads, dynamics and outputs

- `plane: axisymmetric` in the schema and `AppConfig`; the kernels (displacement, mixed),
  the face kernels, `LoadSet` (2π r on coefficients), `MakeDensityTable`/`ReferenceDensity`
  (2π r product), `QuadratureFields` (completed F), the energy diagnostics.
- Tests (`tests/test_axisymmetric.cpp`, in `make check`):
  (a) patch test: u_r = ε r, u_z = ε z on all faces of an (r, z) rectangle away from the
  axis and one touching it: F = (1 + ε) I at every quadrature point, the residual of the
  free dofs zero to round-off, the reaction on z = H equal to P_zz π (R² − R_i²);
  (b) the thick-walled cylinder of `apps/input/finite_elasticity/verification/rivlin_*`
  (plane strain, 2D annulus) as an axisymmetric (r, z) strip with z rollers: the radii and
  the pressure to 1e-6 of the plane-strain run and of Rivlin's closed form;
  (c) the sphere inflation (Green–Zerna) as a quarter annulus in (r, z) against the 3D
  octant of `test_verification` to 1e-4 and the closed form to 0.5%;
  (d) the assembled Jacobian against finite differences in both formulations with a
  follower pressure and a contact term active;
  (e) mass of a solid of revolution (dynamics decorator) to 1e-14 of 2π ∫ ρ r dA;
  (f) np 2 and 4 against serial.
- Inputs: `apps/input/finite_elasticity/verification/rivlin_cylinder_axisymmetric.yaml`,
  `green_zerna_sphere_axisymmetric.yaml` (meshes from `.geo`, `make meshes`).

### TE2 — The thermo-mechanical module

- `Thermoelastic<Base>`, `ThermoMaterial`, the factory (`material.thermal`), the kernel,
  the module, thermal BCs, point constraints (for both modules: `LoadSet`), the app
  (nothing new: it drives a `SolidProblem`), the schema and its errors.
- Tests (`tests/test_thermoelastic.cpp`, in `make check`):
  (a) free thermal expansion of a block (all faces free but rollers), θ prescribed
  uniformly: J = exp(3α Δθ) at every point to 1e-12, zero stress, zero pressure, for two
  laws;
  (b) the material point: M against central differences of S in θ (1e-8), the mixed
  constraint's tangent against differences, objectivity of P(F, θ);
  (c) adiabatic homogeneous stretch of the cube (rollers, insulated): the temperature
  history against the material-point integration of c_v Δθ = ½ θ M : ΔC with the same
  steps and the solution's F (1e-10; the Gough–Joule rise is ~8 K at the stretch 8 in
  TE02);
  (d) transient conduction without deformation (a rigid-like block: the mechanical part
  disabled by a huge modulus is not available; instead θ prescribed on one face of a
  rollered bar with α = 0 and the entropic scaling off, so the mechanics stays at rest):
  the mid-point temperature against the series solution of the 1D heat equation to 1e-3
  at k = 2, dt refined once;
  (e) the assembled 3 x 3 block Jacobian against finite differences with the flux entry and
  the pins active, plane strain and axisymmetric;
  (f) point constraints: the pinned node's reaction equals the resultant of the loads;
  (g) the schema errors of Section 3.

### TE3 — The six examples

- Inputs `apps/input/anand_coupled_theories/finite_thermoelasticity/01_constrained_heating`
  … `06_solar_sail.yaml` (the reference's meshes are `create_rectangle` crossed triangles
  and two Gmsh files: use Q2-Q1-Q1 quadrilaterals of the same subdivisions for the
  rectangles, `bilayer_beam.geo` after the reference's with the two layers as physical
  surfaces, a 20 x 2 grid for the plate, `box.geo` 10 x 10 x 2 hexahedra for the sail),
  `run_set.sh`, `reference/` (CSVs, `README.md`, `scripts/`), `apps/anand_thermo_plots.py`
  (one figure per reference figure, the histories overlaid), and a same-mesh check where
  the mesh differs most (the sail on the reference's 10 x 10 x 2 box of tetrahedra via
  `export_box.py`).
- Acceptance: TE01–TE05 agree with the reference's histories to 1e-3 on the same
  subdivisions (the states are smooth); the sail's deflections within 5% (a coarse mesh of
  a wrinkling plate; the same-mesh check within 1e-3).

### TE4 — Documentation and handback

- Theory manual: a section "Finite thermoelasticity" (model, heat equation, time
  integration, the tangents, axisymmetry as a subsection of the kinematics or the FE
  section, point constraints and thermal BCs under loads), the symbols and source tables.
- Verification manual: the test subsections, the Anand section's third subsection with the
  table, differences and what the plots show, the summary rows.
- README: schema keys (`plane: axisymmetric`, `thermal`, `temperature`, `heat_flux`,
  `point`), the set's subsection, the build line.
- Memory file for the session.

## 5. Out of scope (state in the README as "not supported")

Dynamics with a temperature field; the displacement (penalty) formulation with thermal
expansion; convection and radiation boundary conditions; volumetric heat sources;
temperature-dependent conductivity or heat capacity; thermal contact; anisotropic
conductivity; axisymmetric problems with torsion (u_φ); plane stress with a temperature;
iterative solvers for the 3-block system.

## 6. Order of work and effort

TE1 (the cross-cutting change; two kernels, two face kernels, loads, mass, outputs, tests):
about a third. TE2 (new material, kernel, module, BCs, pins, tests): about half. TE3 and
TE4: the rest. Run the reference notebooks first (done), the inputs last on 4 ranks with
the direct solver.
