# Implementation Plan: Implicit DG Navier-Stokes Driver Replicating Exasim `nsmach8-baseline`

Target directory: `/home/quinnchr/dv/Continuum-Mechanics-MFEM/myapps/hdg_navierstokes`
This document is the **non-hybridized DG driver** plan. It shares the physics module (`ns_physics.*`) and the Exasim I/O layer (`exasim_io.*`, `exasim_mesh.*`) with the sibling HDG driver (`PLAN_HDG.md`), so those files are specified as method-agnostic; shared conventions are recorded in `README.md`.

---

## 1. Goal & scope

**Definition of "same run as nsmach8-baseline":**

| Aspect | Replicated exactly | Notes |
|---|---|---|
| Mesh | Yes | 651 curved quads (31 radial x 21 circumferential half-annulus), isoparametric p=4 geometry taken from `xdg.bin` via exact change of basis (not the analytic map re-sampled) |
| Physics | Yes | Nondimensional compressible NS, gam=1.4, Re=183500, Pr=0.71, Minf=8.03, Sutherland (Ts=110.4, Tref=124.49), fc = mu*gam/Pr acting on grad(e); **deck-literal rounded constants used verbatim** (§2.1) |
| AV | Yes | Frozen scalar field av(x) = 0.025*tanh(5*(r-1)) (== `vdg.bin` comp 0 to 2.8e-17), Laplacian term on all 4 equations |
| BCs | Yes (semantics) | attr 1 wall: rho extrapolated, zero momentum, rhoE = rho*TisoW; attr 2 (x=0): supersonic extrapolation; attr 3 (outer arc): freestream Dirichlet — imposed weakly through boundary states, mirroring Exasim's trace equations |
| Order/quadrature | Yes | p=4 (Q4, 25 dofs/elem/component), degree-8 quadrature = 5x5 Gauss-Legendre volume, 5-pt faces (identical to pgauss=8) |
| Steady solve | Yes (goal) | Steady residual driven to ||R|| < 1e-6, Newton endgame after PTC |
| Discretization | **No — by construction** | DG (Rusanov + BR2) is not HDG (tau-stabilized trace). The converged fields agree to discretization accuracy on this fixed mesh, not bitwise |

**Deliverables:** one executable `dg_navierstokes` + converters + a Python comparison tool producing surface Cp, wall heat flux (Exasim convention), shock standoff, and elementwise L2 field differences against `dataout/outudg_np*.bin`.

**Sign convention (the #1 trap):** Exasim stores q = **-grad(u)** and writes flux `f = F_inv + mu_terms(q) + av*q`. This driver uses physical gradients g = +grad(u) throughout, so the total flux is

```
F(u, g) = F_inv(u) - F_visc(u, g) - av(x) * g,      solve  div F = 0.
```

Every formula transcribed from `pdemodel.txt` / `my_model.hpp` gets the substitution q -> -g. Getting this wrong turns viscosity and AV into anti-diffusion and the Mach-8 solve will blow up; a dedicated unit test (Section 7, `test_physics`) locks this down against numerically evaluated `pdemodel.txt` expressions.

---

## 2. Physics module (`ns_physics.hpp/.cpp`, shared with the HDG driver)

Namespace `nsphys`. No MFEM types in the core API (plain `double*` / small fixed arrays) so both drivers and the Python-free unit tests can use it.

### 2.1 Parameters — deck literals verbatim, not re-derived

The run consumed the **rounded** `pdeapp.txt:49` literals (verified byte-identical in `datain/app.bin`), and the deck is internally inconsistent at the 4e-6 level: `0.5 + 0.027694 = 0.527694 != 0.52769`. Re-deriving "nicer" full-precision values changes the answer at the 1e-5 level and breaks parity with the reference. Rules:

```cpp
struct NSParams {
  // The 11 physicsparam entries, stored verbatim from pdeapp.txt (or read from the deck):
  // [gam, Re, Pr, Minf, rinf, ruinf, rvinf, rEinf, Tinf_param, Tref, Twall]
  // = [1.4, 183500, 0.71, 8.03, 1, 1, 0, 0.52769, 0.027694, 124.49, 294.44]
  double mu[11];
  double Ts = 110.4;                     // Sutherland constant, Kelvin
  double pinf()  const;                  // 1/(gam*Minf^2)         — full precision, from mu[0], mu[3]
  double TinfFlux() const;               // pinf/(gam-1)           — full precision; used by Sutherland ONLY
  double TisoW() const { return mu[8]*mu[10]/mu[9]; }  // 0.027694*294.44/124.49 = 0.06550101502128686
};
```

- The freestream Dirichlet state is `[1, 1, 0, 0.52769]` — the rounded `mu[4..7]`, verbatim.
- `mu[8] = 0.027694` (rounded) is used **only** in `TisoW` — exactly as the generated model does (`my_model.hpp` fbou_hdg ib=3: `mu8*mu10/mu9`).
- Sutherland never reads `mu[8]`: it forms `Tr = gam*Minf^2*p/r` algebraically (full precision).
- **No** `rEinf == 0.5 + Tinf` assert (it is false for the deck at 4e-6); at most a warning with 5e-6 tolerance.
- Sutherland sanity test: `mu*Re == 1` to 1e-14 holds at the **full-precision** freestream `[1, 1, 0, 0.5 + 1/(gam*(gam-1)*Minf^2)]`, *not* at the deck freestream (there it is off by ~1.3e-4 — expected, not a bug).

### 2.2 Exact formulas (matching `pdemodel.txt`, with g = +grad u)

State u = (r, ru, rv, rE); velocities uv = ru/r, vv = rv/r; pressure p = (gam-1)*(rE - 0.5*r*(uv^2+vv^2)); specific internal energy ("temperature") e = p/((gam-1)*r).

- **Sutherland:** `Tr = gam*Minf^2 * p/r` (= T/TinfFlux), `Tphys = Tref*Tr`, `mu = (1/Re) * Tr^1.5 * (Tref+Ts)/(Tphys+Ts)`, `fc = mu*gam/Pr`.
- **Viscous stress** (Stokes hypothesis, physical gradients): `txx = mu*(2/3)*(2*ux - vy)`, `txy = mu*(uy + vx)`, `tyy = mu*(2/3)*(2*vy - ux)` where ux etc. are velocity gradients recovered from conservative-variable gradients (`ux = (d(ru)/dx - uv*dr/dx)/r`, ...).
- **Fluxes** (x-column shown; y analogous):
  - inviscid: `(ru, ru*uv + p, rv*uv, uv*(rE + p))`
  - viscous: `(0, txx, txy, uv*txx + vv*txy + fc*ex)` with `ex = d(e)/dx` from the chain rule on conservative gradients
  - AV: `av * (dr/dx, d(ru)/dx, d(rv)/dx, d(rE)/dx)`
  - total: `F = F_inv - F_visc - F_av` (all three columns computed in one call).

### 2.3 API

```cpp
// F: 4x2, column d = flux in direction d
void EvalFlux(const NSParams&, const double u[4], const double g[8] /*du_c/dx_d, c fastest*/,
              double av, double F[8]);
// A[d]  = dF_d/du (4x4);  Bg[d][e] = dF_d/d(g_e-column) (4x4)
void EvalFluxJac(const NSParams&, const double u[4], const double g[8], double av,
                 double A[2][16], double Bg[2][2][16]);
// boundary states (Exasim trace semantics):
void FreestreamState(const NSParams&, double ub[4]);                  // [1, 1, 0, 0.52769] (deck literal)
void WallState(const NSParams&, const double u[4], double ub[4]);     // [u0, 0, 0, u0*TisoW]
double MaxWaveSpeed(const NSParams&, const double u[4], const double n[2]); // |v.n| + c
```

**Jacobian source:** transcribe `flux_jac_uq` from `my_model.hpp` (lines 222-531, exact symbolic 8x12) and apply the column sign flip on the 8 gradient columns (q -> -g). This reuses Exasim's already-validated symbolic differentiation instead of re-deriving dF_visc/du by hand. Note F_visc is **linear in g** (Bg is independent of g) — exploited by BR2 linearization below. Verify the transcription with central finite differences at 200 random admissible states (tolerance 1e-7 relative) — this is the `test_jacobian_fd` target.

### 2.4 AV coefficient

Two interchangeable `mfem::Coefficient`s, selected in YAML:
- `AnalyticAVCoefficient`: `lambda * tanh(c * (sqrt(x^2+y^2) - 1))` (defaults lambda=0.025, c=5). Exact wall distance for this geometry is r-1, so no distance solver is needed even for the continuation stages.
- `GridFunctionCoefficient` on a p=4 L2 GridFunction loaded from `vdg.bin` component 0 (of 2 — the file is (25, **2**, 651)).

The AV field is **data**: it never contributes state-derivative terms to the Jacobian (its av*g term does contribute through Bg, which is exact and linear).

---

## 3. Mesh strategy: converter from `grid.bin` + `xdg.bin` (recommended)

**Recommendation: convert, do not regenerate.** The analytic map (logdec grading + half-circle) sampled at *MFEM's* node locations produces a Q4 geometry that differs from Exasim's Q4 interpolant sampled at *Chebyshev* nodes at O(h^5) — small but nonzero, and it forfeits the machine-exact same-mesh comparison. The change-of-basis converter reproduces the identical geometric polynomial. Keep the analytic map in the code only as a cross-check and for optional mesh refinement studies (stretch).

### 3.1 File formats (verified)

All files: 3-double header (n1, n2, n3) then n1*n2*n3 little-endian float64, column-major.
- `grid.bin`: header [2, 704, 4, 651] (4 doubles here), then p as 2x704, then t as 4x651 **1-based** CCW quads.
- `xdg.bin`: [25, 2, 651], layout (node, component, element) — 25 x-coords of elem 1, then 25 y-coords, ...
- `udg.bin`: [25, 12, 651] = (u(4), qx(4), qy(4)) with q = -grad u. Only components 0-3 are used as the IC.
- `vdg.bin`: [25, 2, 651], comp 0 = av, comp 1 = 0.
- `dataout/outudg_np{0,1}.bin`: **full 12-component converged state**, headers [25, 12, 326] and [25, 12, 325] (this runmode forces `saveSolOpt=1`; trust the header, not an assumed u-only layout). Components 4-11 are the converged q = -grad(u) — used directly for the Exasim-side wall heat flux (§6.3). Within-rank element order is DMD-permuted; see §3.4.

### 3.2 Exasim element basis

npe=25 tensor-product Lagrange nodes at the 1D **Chebyshev-Gauss-Lobatto** set on [0,1]:
`{0, (3-sqrt5)/4 = 0.19098300562505, 1/2, (1+sqrt5)/4 = 0.80901699437495, 1}`, lexicographic with the first reference coordinate fastest. Implement `ChebyshevBasis1D` in `exasim_io.hpp`: node list + `Eval(x, phi[5])` by barycentric Lagrange.

### 3.3 Mesh construction (`exasim_mesh.cpp :: BuildExasimMesh`)

1. `mfem::Mesh mesh(2, 704, 651, /*nbdr*/104, 2)`; `AddVertex` from p; `AddQuad(v0..v3)` from t (subtract 1; CCW order matches MFEM's reference-square vertex order (0,0),(1,0),(1,1),(0,1)).
2. Boundary elements: collect edges with a single adjacent element; classify by edge midpoint using Exasim's ordered tests: `sqrt(x^2+y^2) < 1+1e-6` -> attr 1 (wall), else `x > -1e-7` -> attr 2 (outflow), else attr 3 (outer arc). Expected counts (assert): 21 / 62 / 21. The driver maps attribute -> bc type via `boundaryconditions: [3, 2, 1]` (README convention, mirroring `pdeapp.txt:52`).
3. `mesh.FinalizeQuadMesh(1, 0, true); mesh.SetCurvature(4, /*discont=*/true, 2, Ordering::byVDIM);`
4. Overwrite the Nodes GridFunction: for each element e, get MFEM's node reference points `nodes_fes->GetFE(e)->GetNodes()`; for each point (xi, eta) evaluate the tensor Chebyshev basis `phi_i(xi)*phi_j(eta)` (i = first-coordinate index fastest, matching Exasim ordering) and contract with the 25 (x, y) pairs of `xdg[:, :, e]`. **Orientation check first:** assert xdg corner nodes 0, 4, 24, 20 coincide with grid.bin vertices t[e][0..3] to 1e-12 — this pins the reference-coordinate correspondence; if it fails, insert the appropriate index permutation (one-time debug).
5. Validation: min detJ > 0 at the degree-8 rule; total area vs. analytic domain area; ParaView dump of the curved mesh.

### 3.4 Field transfer and reference reassembly (`exasim_mesh.cpp :: TransferExasimField`)

Same change-of-basis: for a target MFEM L2 GridFunction (vdim = nc, `Ordering::byVDIM`), evaluate the Chebyshev expansion of `udg[:, c, e]` at MFEM's L2 node reference points. This is an **exact** polynomial identity (both bases span Q4 on the same element with the same reference map) — no interpolation error, no `FindPoints`, no GSLIB (which is compiled out anyway). Used for: IC (udg comps 0-3), av (vdg comp 0), and the reference solution during validation.

**Reference reassembly — `partition.bin` is NOT sufficient.** The per-rank `outudg` files carry no element lists, and rank-local order is *not* ascending-original-filtered-by-rank: Exasim permutes each rank's elements via `dmd.elempart` = sorted groups [interior-minus-interface, interface subsets, exterior] (domaindecomposition.cpp:340-365) — interface elements go to the END of each rank's block. Naive concatenation silently scrambles them, and every M3 comparison metric would be computed against a scrambled reference. Recovery: parse `elempart` from `datain/mesh1.bin`/`mesh2.bin` (`ms.elempart`, buildstructs.hpp:439; rank-local element k -> original element elempart[k]; 326+325=651 confirms no ghosts written); validate geometrically against the per-rank xdg stored in the same mesh bins. Budget a day for the mesh*.bin layout. **Pragmatic alternative (first move): rerun the reference once with `mpiprocs=1`** — order becomes the identity and the mapping vanishes.

---

## 4. Discretization: implicit nodal DG, Rusanov + BR2 + frozen Laplacian AV

### 4.1 Chosen variant and justification

**Primary scheme:** nodal DG on Q4, solving for **u only (4 components, 65,100 dofs)**:
- **Inviscid numerical flux: Rusanov (local Lax-Friedrichs).** Justification for the Mach-8 bow shock: (i) *carbuncle immunity* — HLLC's exact contact/shear resolution is precisely what feeds the carbuncle / odd-even instability on shock-aligned structured quad meshes, and this 31x21 mesh is shock-aligned by design; Rusanov dissipates the shear/entropy waves and is carbuncle-free in practice without rotated-flux or entropy-fix machinery. (ii) *Closest match to the target:* Exasim's HDG stabilization is a constant tau = 1 on all four equations — effectively a global Lax-Friedrichs with unit wave speed (freestream |v|+c = 1 + 1/8.03 = 1.1245). Rusanov with the local max wave speed is the nearest DG analog, so dissipation levels are comparable by construction. (iii) The shock is captured by the frozen Laplacian AV (that is the Exasim baseline's philosophy); the Riemann solver's wave resolution inside the AV-smeared shock is second-order concern. (iv) In the boundary layer, face jumps are O(h^{p+1}) for the resolved solution, so Rusanov's extra dissipation is negligible against the physical viscous flux at the wall spacing here (first cell 0.0024, p=4). Keep an `HLLC` option behind the same interface for sensitivity studies, documented as carbuncle-prone here.
- **Viscous + AV discretization: BR2 (Bassi-Rebay 2).** Justification: compact stencil (face neighbors only) means the Jacobian has exactly the inviscid-DG block sparsity — <= 5 blocks of 100x100 per block row — which is what `mfem::BlockILU` (designed for elemental DG blocks, `linalg/solvers.hpp:1008`) and MUMPS both digest well. LDG would widen the stencil (or require an auxiliary variable, doubling unknowns); SIPG on the full nonlinear viscous flux needs the same lifting-free penalty scaled by mu(T), which is workable but BR2's local lifting gives parameter robustness (eta > n_faces suffices) and is the standard in implicit compressible DG codes.
- **AV term:** folded into the diffusive flux, `F_d(u, g) = F_visc(u, g) + av(x) * g`, and discretized by the *same* BR2 operator. Since the DG driver carries no gradient unknowns, **grad(u) is the broken gradient of u_h in volume terms, and the BR2-lifted gradient on faces**: at a face f, the diffusive numerical flux uses `g_face = grad(u_h) + eta_BR2 * r_f([[u_h]])` on each side, averaged. This is the explicit answer to "where does the AV get its gradient": lifting, not reconstruction — it keeps the stencil compact and the linearization exact (r_f is a linear operator).

**Weak residual** (per test function phi, all 4 components; this is the vector `NonlinearForm::Mult` returns, Newton solves R(u) = 0):

```
R(phi) = - Int_K  grad(phi) . F(u_h, grad u_h, av)  dx                        (volume)
         + Sum_interior_f  Int_f [[phi]] . Fhat . n  ds                       (faces)
         + Sum_bdr_f       Int_f phi . Fhat_b . n  ds                         (boundary)

Fhat.n     = 1/2 (F_inv(u-)+F_inv(u+)).n - 1/2 lam_max (u+ - u-)              (Rusanov)
             - { F_d(u, grad u_h + eta r_f([[u]])) } . n                      (BR2 primal)
             - symmetrizing term:  - Int_f [[u]] ox n : { (dF_d/dg)^T grad phi } ds
```

with lam_max = max over both sides of |v.n| + c, eta = 6 (> 4 faces), r_f the face-local lifting defined by `Int_K r_f . tau = -Int_f ([[u]] ox n) . {tau}`. Reference implementations to crib for jump/average and lifting bookkeeping: `fem/integ/bilininteg_br2.cpp` and `HyperbolicFormIntegrator::AssembleFaceGrad` in `fem/hyperbolic.cpp` (patterns only — both are unusable directly, see 4.4).

### 4.2 Boundary conditions (replicating Exasim trace semantics)

Per boundary attribute, define the boundary state u_b and evaluate a one-sided flux. Default mode `wall_flux: exasim` mimics Exasim's `Fhat = F(uhat, q).n + tau (u - uhat)` with uhat replaced by the BC-resolved state:

| attr | u_b | Fhat_b . n |
|---|---|---|
| 3 outer (freestream) | [1, 1, 0, 0.52769] (deck literal) | `F_inv(u_b).n + tau (u - u_b)` (supersonic inflow: equivalent to full upwinding) — no viscous term (matches Exasim: freestream row is pure Dirichlet on the trace, element eq. keeps fhat; in DG the one-sided flux at u_b is the analog) |
| 2 outflow (x = 0) | u_b = u_h (extrapolation) | `F_inv(u_h).n` — tau term vanishes identically; viscous flux with interior gradient |
| 1 wall (r = 1) | [rho_h, 0, 0, rho_h * TisoW], TisoW = 0.06550101502128686 | `F_inv(u_b).n - F_d(u_b, grad u_h + eta r_b(u_h - u_b)).n + tau (u_h - u_b)`, boundary lifting r_b uses the full jump (u_h - u_b) |

tau = 1 (YAML knob). Alternative mode `wall_flux: rusanov` uses Rusanov(u_h, u_b) at all boundaries — more conventional DG, available for robustness comparison. Note d(u_b)/d(u_h) is nonzero (wall: rows 0 and 3 depend on rho_h; outflow: identity) and must be chain-ruled in `AssembleFaceGrad`.

### 4.3 Spaces, quadrature, dof layout

- `DG_FECollection fec(4, 2, BasisType::GaussLobatto); FiniteElementSpace fes(&mesh, &fec, 4, Ordering::byVDIM);` — byVDIM makes each element's 100 **global** dofs contiguous, which is required for `BlockILU(block_size=100)` (scalar L2 element dofs are contiguous and vdof = 4*sdof + c).
- Quadrature pinned to Exasim: `IntRules.Get(Geometry::SQUARE, 8)` (5x5 GL) volume, `IntRules.Get(Geometry::SEGMENT, 8)` (5-pt GL) faces, set via `NonlinearFormIntegrator::SetIntRule`. Same deliberate under-integration of the quartic geometry as Exasim.

### 4.4 What MFEM provides vs. what is hand-written (honest inventory)

Provided and reused:
- `NonlinearForm` with `AddInteriorFaceIntegrator` / `AddBdrFaceIntegrator` (`fem/nonlinearform.hpp:145,154`) and the four virtuals `AssembleElementVector/Grad`, `AssembleFaceVector/Grad` (`fem/nonlininteg.hpp:64-86`). **Serial** `NonlinearForm::GetGradient` assembles face Jacobians into a `SparseMatrix` (`fem/nonlinearform.cpp:564-586`) — residual and Jacobian assembly loops come for free.
- `BlockILU` (block ILU(0), `linalg/solvers.hpp:1008`), `GMRESSolver`, `PetscLinearSolver` (PETSc 3.19 with MUMPS/SuperLU in its closure), `DenseMatrix/DenseTensor/LUFactors`.
- Mesh infrastructure: `SetCurvature`, face transformations, ParaView high-order output.

**Not provided — hand-written in this app:**
- All physics-bearing integrators (MFEM's `DGDiffusionIntegrator`/`BR2Integrator` are linear `BilinearFormIntegrator`s; cannot express G(u) grad u with Sutherland mu).
- Euler flux Jacobian (`EulerFlux` implements only `ComputeFlux/ComputeFluxDotN`; `ComputeFluxJacobian*` aborts — `fem/hyperbolic.hpp:159,175,1045`), so `RusanovFlux::Grad` is unusable for implicit work without a subclass.
- PTC / SER / positivity line search (no steady-state infrastructure exists).
- Shock capturing: none in MFEM — here supplied by the frozen AV field, no sensor needed for the baseline.
- **Parallel Jacobian:** `ParNonlinearForm::GetGradient` hits `MFEM_ABORT("TODO: shared face terms")` with face integrators (`fem/pnonlinearform.cpp:127`). Decision: **serial-first**. 65k dofs / 651 elements is trivially serial; keep the repo's `Mpi::Init` main() pattern but `MFEM_VERIFY(Mpi::WorldSize()==1, ...)`. Hand-assembled `HypreParMatrix` face Jacobians are a documented stretch item, not in scope.

**Build-on-ex18 decision: no — from scratch, with the library left in place.** ex18 is explicit RKDG Euler (ODE loop, per-element inverse mass) — everything it adds is discarded for a steady implicit solver; `HyperbolicFormIntegrator` covers only the inviscid half, imposes a du/dt = -div F sign convention, and its `EulerFlux` lacks Jacobians. Writing our own three integrators (a) lets one face loop evaluate inviscid + viscous + AV numerical fluxes with shared geometry/basis evaluations, (b) keeps all Jacobian blocks in one `AssembleFaceGrad`, and (c) reuses the *same* `ns_physics` kernels as the HDG driver — which we must write anyway. ex18.hpp and hyperbolic.cpp remain the reference for elmat block layout conventions (block (el1,el1), (el1,el2), (el2,el1), (el2,el2) ordering in the 200x200 face elmat).

### 4.5 Integrator classes (`dg_integrators.hpp/.cpp`)

```cpp
class BR2Lifting {            // precomputed once (geometry-only)
  // per element: 25x25 mass matrix inverse (DenseTensor, LU factors)
  // per face & side: B_f[q, j] = phi_j(x_q) * w_q * |J_f(q)|  and normal n(q)
  // exposes: LiftedGradAtFaceQP(face, side, jump_at_qps) -> g_corr[qp][2][4]
  //          and its (linear, exact) derivative w.r.t. the two elements' dofs
};

class NSVolumeIntegrator   : public NonlinearFormIntegrator;  // -Int grad(phi).F ; Grad via A[d], Bg[d][e]
                                                              // + optional analytic MMS source hook (M2)
class NSFaceIntegrator     : public NonlinearFormIntegrator;  // Rusanov + BR2 + AV, interior
class NSBdrFaceIntegrator  : public NonlinearFormIntegrator;  // per-attr boundary states (registered 3x with markers)
```

All integrators take `const NSParams&`, `Coefficient& av_coeff`, `const BR2Lifting&`. **Local element-vector layout (`elfun`): component-blocked, node-fastest — `u_c` at node i is `elfun[c*25 + i]`, and face elmats are 25x25 blocks per component pair `(c1*ndof+i, c2*ndof+j)`.** This is fixed by `DofsToVDofs` (`fem/fespace.cpp:31-59`: `dofs[i + size*vd]` for *both* orderings) — the global `Ordering::byVDIM` affects only global dof numbering, never the local elfun layout handed to integrators. Write a tiny `StateAtQP` helper computing (u, grad u) at quadrature points from elfun via the element's `DShape`/inverse Jacobian; this is the hot loop, keep it allocation-free (preallocated DenseMatrix workspace per integrator, single-threaded assembly).

**Jacobian decision: hand-coded analytic.** Rationale against alternatives for p=4 / 2D quads (100x100 diagonal blocks, ~3.3e7 nnz total):
- *FD coloring:* feasible in cost (~5-10 colors x 100 dofs) but the flux magnitudes span ~4 orders across the shock; FD step selection noise pollutes the Jacobian precisely when Newton must contract from 1e-4 to 1e-6. Exasim itself uses exact symbolic Jacobians; matching its convergence behavior needs the same.
- *JFNK:* BlockILU/MUMPS need an assembled matrix anyway, so matrix-free saves no assembly, and the FD matvec has the same noise problem.
- The analytic route is cheap here because `flux_jac_uq` is transcribed, not derived (Section 2.3), BR2 lifting is linear (exact chain rule), and the only approximation is **freezing lam_max in the Rusanov Jacobian** (standard practice; degrades quadratic to superlinear convergence only in shock-adjacent cells — acceptable, and an exact d(lam)/du term can be added later if the endgame stalls). Every `Assemble*Grad` is verified against central FD of the corresponding `Assemble*Vector` in `test_jacobian_fd` (random states, random curved elements, 1e-7 relative).

---

## 5. Nonlinear/linear solver strategy and globalization

### 5.1 PTC-Newton (`ptc_newton.hpp/.cpp`, bespoke ~250-line loop)

Steady Newton from freestream at Mach 8 **will not converge**; even from the near-converged `udg.bin` restart the DG residual will start at O(0.1-1) (it is HDG-converged, not DG-converged). Solve

```
( M_dt + dR/du ) du = -R(u),    M_dt = blockdiag( (1/dtau_e) * M_e ),   dtau_e = CFL * h_e / lam_e
```

- `M_e` = element mass matrices (assembled once via `VectorMassIntegrator` on the same fes, stored as DenseTensor; added into the assembled SparseMatrix per Newton step with `AddSubMatrix` — mass is block-diagonal so this is cheap; pass `skip_zeros=0` when building the graph). `h_e` = 2*|K|/perimeter, `lam_e` = max nodal |v|+c.
- **SER ramp:** `CFL_{k+1} = clamp(CFL_k * ||R_{k-1}|| / ||R_k||, CFL_k/4, 10*CFL_k)`, `CFL_0` from YAML (default 10 with the Exasim restart, 0.5 for freestream starts), `CFL_max = 1e12`.
- **Newton endgame:** when ||R||_2 < `newton_switch_tol` (default 1e-3), drop M_dt entirely — pure Newton, mirroring Exasim's damped Newton — and require ||R|| < 1e-6 (`NewtonTol` equivalent) within 30 iterations.
- **Positivity line search:** for candidate u + alpha*du, check rho > rho_floor (1e-8) and p > p_floor (1e-10) at *all element nodes and all volume/face quadrature points*; halve alpha (max 8 times); on failure reject the step and cut CFL by 4. In the pure-Newton phase additionally use Exasim's damping rule (exact semantics, nonlinearsolver.cpp:277-289): `while (||R_new|| > ||R_old|| && alpha > 0.1) { alpha /= 2; retry }` — accepted-step sequence 1, 0.5, 0.25, 0.125, 0.0625, and an increase at the last step is accepted; abort only on (increase AND ||R|| > 1e6) or NaN. Floors gate the line search only — they never enter the residual, so the converged solution is unmodified (Exasim's exported model likewise has no regularization).
- Not built on `mfem::NewtonSolver`: the dtau Jacobian shift and per-stage SER state don't fit its `GetGradient` contract cleanly; follow the repo's `newton_petsc_solver.hpp` structural conventions (config struct, result struct with timings, CSV history) instead.

### 5.2 Linear solver

- **Default for M3 (robustness first): direct.** `PetscLinearSolver` with `Input/petsc.opts`: `-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps` (MUMPS is in MFEM's PETSc link closure; native SuiteSparse is compiled out). At 65,100 dofs / ~3.3e7 nnz this factors in seconds and removes preconditioner quality from the debugging matrix entirely.
- **Scalable option: `GMRESSolver(kdim=100, maxit=100, rtol=1e-8)` + `BlockILU(100, Reordering::MINIMUM_DISCARDED_FILL)`** — matches the Exasim GMRES settings. Wrap `SetOperator` to re-factor each Newton step. If ILU(0) proves too weak in the shock region (plausible), fall back to MUMPS; do **not** attempt to replicate Exasim's harmonic-Ritz polynomial preconditioner — it is an efficiency device, not part of the discrete solution.
- YAML `linear.type: {mumps, gmres_bilu}`.

### 5.3 Globalization paths

- **Path A (baseline, replicates the Exasim run):** IC = `udg.bin` components 0-3 via exact transfer; av frozen = analytic 0.025*tanh(5(r-1)) (or vdg.bin). Single PTC-Newton solve to 1e-6. This is milestone M3.
- **Path B (stretch, self-contained):** reproduce `pdeapp_ns.m` lines 55-95. IC = freestream with `ru,rv *= tanh(10*(r-1))` and `rE = Tinf*((Twall/Tref-1)*exp(-10*(r-1)) + 1) + 0.5*(ru^2+rv^2)/r` (wall distance = r-1 exactly on this geometry — no distance solve). Four stages, av = {0.06*tanh(30 d), 0.04*tanh(30 d), 0.025*tanh(30 d), 0.025*tanh(5 d)}, each stage a full PTC-Newton solve restarting from the previous converged state, **SER CFL reset to CFL_0 at each stage** (the residual jumps when av changes). Stage table driven from YAML (`av.continuation: [{lambda, c}, ...]`). This is milestone M4. The adaptive dilatation-sensor/Helmholtz AV loop of `hypersoniccylinder_mach8/pdeapp.m` is explicitly out of scope (the baseline export predates it).

---

## 6. I/O & validation

### 6.1 YAML deck (repo convention: `LoadParams` throws on missing keys, rank-0 `PrintConfig`)

`Input/input_dg_ns_mach8.yaml`:
```yaml
case_name: dg_ns_mach8
exasim_run_dir: /home/quinnchr/dv/Exasim-apps/runs/nsmach8-baseline
initial_condition: udg          # udg | freestream_damped
av: { mode: analytic, lambda: 0.025, c: 5.0 }   # or mode: vdg | continuation: [...]
physics: { mu: [1.4, 183500, 0.71, 8.03, 1, 1, 0, 0.52769, 0.027694, 124.49, 294.44] }
bc: { boundaryconditions: [3, 2, 1] }           # attr (1=wall,2=outflow,3=outer) -> ib type
discretization: { order: 4, quad_order: 8, br2_eta: 6.0, tau: 1.0, wall_flux: exasim }
ptc: { cfl0: 10.0, cfl_max: 1.0e12, newton_switch_tol: 1.0e-3, tol: 1.0e-6, max_steps: 200 }
linear: { type: mumps }         # or gmres_bilu
output: { path: ParaView/dg_ns_mach8, every: 10, surface_csv: dg_wall_data.csv }
```
`Input/petsc.opts` per Section 5.2. main() follows the `diffusion_mms_ale.cpp` pattern (Mpi::Init, Hypre::Init, OptionsParser `-i`, MFEMInitializePetsc with graceful fallback, try/catch with rank-0 reporting).

### 6.2 ParaView

`ParaViewDataCollection("dg_ns_mach8")`, `SetPrefixPath`, `SetLevelsOfDetail(4)`, `SetHighOrderOutput(true)`, `VTKFormat::BINARY`. Registered fields: conservative `u` (vdim 4), derived scalars `rho, uvel, vvel, p` (p = 0.4*(rE - KE), matching Exasim's vis_scalars exactly), `Mach`, `av`, `resid_mag`. Saved every `output.every` PTC steps + final.

### 6.3 Quantitative comparison vs. Exasim `dataout`

In-driver (rank-0 CSV, setprecision(16)) + `tools/compare_exasim.py`:
1. **Surface data** at wall face quadrature points, vs. theta in [pi/2, 3pi/2]: `Cp = 2*(p - pinf)` (rho_inf = u_inf = 1) and wall heat flux in **Exasim's convention** `q_w = fc * grad(e).n` with the DG-side gradient taken as the discrete BR2 boundary flux gradient (`grad u_h + eta * r_b(u_h - u_b)`) — functional consistency with the residual matters for accuracy. The Exasim-side reference uses the **converged q stored in `outudg` components 4-11** (sign-flipped to physical gradients) — the gradient Exasim's own residual/QoI actually used — rather than re-differentiating the Chebyshev expansion.
2. **Shock standoff:** sample rho along the stagnation line y = 0, x in [-3, -1]; standoff = location of max |d rho/dx|; compare deltas.
3. **Field L2 differences:** same mesh, same element correspondence (after the §3.4 elempart mapping) => per-element quadrature at the 5x5 GL rule of `(u_DG - u_Exasim)^2` — exact, no point location. Report per-component relative L2.
4. **Acceptance targets** (DG != HDG on a fixed 651-element mesh; these are discretization-difference windows, not solver tolerances): Cp within 2% pointwise except within 2 elements of the outflow corners; q_w within 10% along the front half (theta near pi); standoff within one local cell (~0.05); L2 diffs reported (expect O(1e-2) relative near the AV-smeared shock, much smaller elsewhere).

---

## 7. File-by-file work breakdown and milestones

```
myapps/hdg_navierstokes/
  makefile                     # repo convention: locate $(HOME)/MFEM/mfem/config/config.mk, include it;
                               # per-TU .o rules (this app has ~7 TUs — avoid whole-program recompiles)
  README.md
  Input/input_dg_ns_mach8.yaml, input_mms.yaml, input_m45_sanity.yaml, petsc.opts
  exasim_io.hpp/.cpp           # BinFile reader (header+column-major), ChebyshevBasis1D, Grid/Xdg/Udg/Vdg structs,
                               # datain/mesh*.bin elempart parser
  exasim_mesh.hpp/.cpp         # BuildExasimMesh, boundary attrs, TransferExasimField, outudg reassembly
  ns_physics.hpp/.cpp          # NSParams, EvalFlux, EvalFluxJac (from my_model.hpp), boundary states, Sutherland, AV coeffs
  dg_integrators.hpp/.cpp      # BR2Lifting, NSVolumeIntegrator (+MMS source hook), NSFaceIntegrator, NSBdrFaceIntegrator
  ptc_newton.hpp/.cpp          # PTCNewtonSolver (SER, dtau shift, positivity line search, history CSV)
  dg_navierstokes.cpp          # main driver
  tests/test_physics.cpp       # mu*Re=1 at full-precision freestream; flux vs. transcribed pdemodel.txt; sign convention
  tests/test_jacobian_fd.cpp   # every Assemble*Grad vs central FD of Assemble*Vector
  tools/compare_exasim.py      # reads both solutions, produces Cp/q_w/standoff/L2 report + plots
  tools/plot_convergence.py
```

**M1 — infrastructure compiles, mesh and data verified (3 person-days).**
Deliver: makefile, exasim_io, exasim_mesh, skeleton driver dumping the mesh + transferred udg/vdg fields to ParaView.
Acceptance: corner-node/vertex consistency assert passes; boundary edge counts 21/62/21; min detJ > 0; transferred rho field range [1.000, 19.51]; av field max = 0.025; ParaView renders the curved half-annulus.

**M2 — physics + integrators verified on sanity problems (5-6 person-days).**
Deliver: ns_physics, dg_integrators, ptc_newton, both test executables.
Acceptance:
(a) `test_physics` and `test_jacobian_fd` green;
(b) **freestream preservation**: project u_inf on the curved Exasim mesh **with freestream Dirichlet applied on ALL boundary attributes** (YAML BC override — with the case's wall BC active, wall rows are O(1) at freestream by design and would mask the test), ||R||_inf < 1e-10. This exposes GCL/quadrature inconsistencies on the quartic geometry; the degree-8 rule is provably exact for the volume/face integrands here, so 1e-10 is attainable — if violated, diagnose the volume/face quadrature pairing before proceeding;
(c) **polynomial MMS with source (machine-zero gate)**: pick conservative fields in Q4 (constant-mu mode, YAML toggle), compute the analytic source div F(u_exact) offline (sympy), feed it through the `NSVolumeIntegrator` source hook; the exact solution lies in the discrete space, so Newton must converge to machine-zero residual against the projected exact solution. (Plane Couette is *not* exactly representable — viscous heating makes rho, ru, rE rational in y, so it is kept only as a physical sanity with discretization-level tolerances plus an h-refinement check, not a machine-zero gate.);
(d) **supersonic sanity on the actual mesh** (shared deck with the HDG sibling, `Input/input_m45_sanity.yaml`): M=4.5, Re=1e3, av = 0.05*tanh(5(r-1)), damped-freestream init: PTC converges to 1e-6, fields positive and y-symmetric. (M=4.5, not M=2: at M=2 the bow-shock standoff ~1.2R puts the shock wing across the outer arc where freestream Dirichlet is imposed, and a large subsonic pocket conflicts with the pure-extrapolation outflow — the gate could fail with a perfectly correct code. At M=4.5 the standoff is ~0.5R and the shock exits through the genuinely supersonic outflow plane.)

**M3 — Mach 8, frozen AV, Exasim IC (Path A) (5 person-days).**
Deliver: full deck, comparison tooling incl. §3.4 reference reassembly (or single-rank reference rerun).
Acceptance: PTC-Newton converges to ||R|| < 1e-6 (or a documented plateau with cause analysis); comparison report meets the Section 6.3 targets; convergence history CSV archived. Budget includes solver tuning (CFL_0, switch tolerance, eta) and the likely BlockILU-vs-MUMPS shakeout.

**M4 — self-contained continuation (Path B) (4 person-days).**
Deliver: continuation stage machinery, damped-freestream IC.
Acceptance: 4-stage run from freestream completes unattended; final state coincides with the M3 solution to Newton tolerance (identical final discrete problem => L2 difference < 1e-8), demonstrating the driver does not depend on Exasim restart data.

Total ~17-18 person-days plus ~20% contingency. The HDG sibling driver reuses exasim_io/exasim_mesh/ns_physics (and the PTC scaffolding pattern) — ~40% of this work is shared.

---

## 8. Risks and mitigations

1. **Gradient sign flip (q = -grad u).** Highest-severity, silent-until-blowup. Mitigation: single substitution point in `ns_physics.cpp`; `test_physics` compares against numerically evaluated `pdemodel.txt` expressions with q = -g; the MMS test would also catch it (anti-diffusion cannot converge to the manufactured solution).
2. **Freestream/GCL violation on the quartic mesh with degree-8 quadrature.** Would contaminate the smooth outer field. Mitigation: M2(b) gate (all-freestream BC configuration); if > 1e-10, raise the *geometric* quadrature (volume and face consistently) — a knowingly accepted deviation from pgauss=8, documented in the comparison report.
3. **BlockILU(0) too weak for the Mach-8 Jacobian** (only k=0 supported). Mitigation: MUMPS direct is the M3 default; GMRES+BlockILU is an option whose iteration counts get reported, not a dependency.
4. **Rusanov contact dissipation degrades wall heat flux.** Mitigation: quantify against Exasim (whose tau = 1 stabilization is comparably dissipative); HLLC toggle exists for sensitivity, with the carbuncle caveat on this shock-aligned mesh; the fixed 31x21 mesh caps what can be fixed — report, don't chase.
5. **DG vs. HDG discrepancy misread as a bug.** Mitigation: acceptance windows in 6.3 are defined up front; the exact same-mesh L2 machinery localizes differences (expect them concentrated in the AV-smeared shock); optional analytic-map mesh-refinement study (stretch) to show both converge to the same continuum solution.
6. **Positivity failures during PTC transients** (esp. Path B stage 1). Mitigation: line-search floors + CFL cut; if a stage still fails, insert an intermediate AV amplitude (YAML-driven schedule makes this a config change). **M4 amendment (2026-07-26): use the `pdemodel_ns.m` smoothed rho/p floors first** — the HDG M4 continuation proved the `0.06 -> 0.04` stage transient necessarily visits `p < 0` and is infeasible without them (see `M4_FAILURE_REPORT.md`); the shared `ns_physics.hpp` already provides the floored flux + Jacobians behind `physics.regularization: floors` (compose with the DG sign adapters as usual).
7. **my_model.hpp Jacobian transcription errors.** Mitigation: mechanical FD verification (`test_jacobian_fd`) over random admissible states including near-wall low-e and post-shock high-rho regimes.
8. **Serial-only Jacobian (ParNonlinearForm face-Grad abort).** Accepted: 651 elements is trivially serial; documented stretch item (hand-assembled HypreParMatrix, ~1 extra week) if larger meshes are ever needed.
9. **Weak-BC formulation difference at the wall** (DG boundary-state vs. Exasim trace-unknown enforcement; note udg's wall e ~ 0.075 vs TisoW = 0.0655 — the BC lives on the trace, not the volume state). Mitigation: `wall_flux: exasim` mode reproduces Exasim's fhat structure including the tau*(u - u_b) penalty; never impose strong Dirichlet dofs.
10. **Reference-data reassembly** (DMD elempart permutation; naive partition-order concatenation scrambles interface elements *silently*). Mitigation: §3.4 elempart parser + geometric validation against per-rank xdg; single-rank reference rerun as the recommended shortcut.
11. **Rounded deck constants** (0.52769 / 0.027694, internally inconsistent at 4e-6). Mitigation: §2.1 rules — literals verbatim, TisoW = 0.06550101502128686, no equality asserts on derived identities, sanity tests evaluated at full-precision states only.

## Alternative DG variant (brief contrast)

**Entropy-stable DGSEM (collocated LGL, flux-differencing with Chandrashekar/Ranocha two-point entropy-conservative fluxes) + subcell FV limiting (Trixi/FLEXI-style).** Prefer it when: (i) running *transient* shock-dominated problems where a precomputed AV field does not exist and provable semi-discrete entropy stability plus subcell positivity is the robustness mechanism; (ii) exploring parameter ranges (Mach, Re, geometry) far from the tuned continuation schedule. Do **not** prefer it here: the blending/limiting operators are non-smooth in the state, which cripples steady Newton convergence (residual stagnation at ~1e-3 is typical); the implementation effort is a multiple of this plan; and it would deliberately *not* reproduce the Exasim baseline, whose solution is defined by the frozen Laplacian AV. For this replication task — steady, frozen AV, one small fixed mesh — plain nodal DG + Rusanov + BR2 + AV is the right tool.
