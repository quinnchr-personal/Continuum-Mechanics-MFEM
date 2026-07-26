# Implementation Plan: MFEM HDG Driver Replicating Exasim `nsmach8-baseline`

Target directory: `/home/quinnchr/dv/Continuum-Mechanics-MFEM/myapps/hdg_navierstokes`
Reference run: `/home/quinnchr/dv/Exasim-apps/runs/nsmach8-baseline`
MFEM: `/home/quinnchr/MFEM/mfem` (4.8.1 dev, mpicxx, MPI+METIS+PETSc 3.19+hypre 2.26; no SuiteSparse, no LAPACK, no GSLIB).
Sibling document: `PLAN_DG.md` (shares `ns_physics`, `exasim_io`, `exasim_mesh`, tools — see `README.md` for the shared conventions).

---

## 1. Goal & scope

"Same run as nsmach8-baseline" means, precisely:

1. **Same mesh**: the 651-element (31 radial x 21 circumferential) curved quad half-annulus, with the *identical discrete Q4 geometry* taken from `xdg.bin` (not merely the same analytic surface), same element correspondence, same boundary partition (wall / outflow / freestream).
2. **Same physics**: 2D steady compressible NS, conservative variables, `mu[0..10] = [1.4, 183500, 0.71, 8.03, 1, 1, 0, 0.52769, 0.027694, 124.49, 294.44]` **verbatim from `pdeapp.txt:49`** — note `rEinf=0.52769` and `Tinf=0.027694` are the *rounded* values the run actually used; do not "fix" them to full precision. Sutherland viscosity (Ts=110.4), `fc = mu*gam/Pr` acting on `grad(e)`, frozen Laplacian AV `+av*grad(u)` on all 4 equations with `av` from `vdg.bin` (component 0 = `0.025*tanh(5*(r-1))`, component 1 = 0 and ignored).
   **Two different Tinf values coexist in the model — replicate this exactly.** The flux/Sutherland path never reads `mu[8]`: it forms `Tr = gam*Minf^2*p/r` algebraically, i.e. full-precision `Tinf = 1/(gam*Minf^2*(gam-1)) = 0.0276936935699453` derived from `mu[0], mu[3]` (`pdemodel.txt:25-26`; the generated `my_model.hpp:56` Sutherland uses only mu0,mu1,mu3,mu9). The wall BC, by contrast, uses the *rounded deck literal* `mu[8] = 0.027694` verbatim (`my_model.hpp:216`). Mixing these up costs 1.1e-5 relative — enough to fail the M3 parity gates.
3. **Same BCs** (weakly, on the trace): freestream Dirichlet-on-uhat (bc 1, outer arc), supersonic extrapolation `u - uhat` (bc 2, x=0 plane), isothermal no-slip wall `[u0-uh0, -uh1, -uh2, uh0*TisoW - uh3]` (bc 3, r=1), with
   `TisoW = mu[8]*mu[10]/mu[9] = 0.027694*294.44/124.49 = 0.06550101502128686`.
   Mesh boundary attributes follow the Exasim mesh indices (1=wall, 2=outflow, 3=outer arc); the driver maps attribute -> bc type via `boundaryconditions: [3, 2, 1]` exactly as `pdeapp.txt:52` (see README).
4. **Same discretization**: HDG, p=4 tensor quads (npe=25, npf=5), mixed first-order form with **Exasim's sign convention q = -grad(u)**, numerical flux `fhat = F(uhat, q_interior)·n + tau*(u - uhat)` with scalar `tau = 1` on all equations/faces, quadrature degree 8 (5-pt tensor Gauss-Legendre: 25 volume / 5 face points), static condensation to the trace.
5. **Same steady solve semantics**: damped Newton on the condensed trace system (max 30 iters, tol 1e-6 on `|Ru|+|Rh|`, alpha-halving line search — exact rules in §5), starting from the `udg.bin` restart + trace-of-u uhat initialization. The *linear* solver need not be bit-identical (we will not replicate the harmonic-Ritz/Leja polynomial preconditioner); with matching discretization the Newton fixed point is the same, so converged fields should agree to ~solver tolerance.
6. **Comparable outputs**: ParaView fields `[rho, u, v, p]` + `(ru, rv)`, surface Cp, wall heat flux via the exact `Fint` formula (`fc*(Tx*nx+Ty*ny) + tau*(rE - rEhat)` at trace), shock standoff, and quantitative field L2 diffs against `dataout/outudg_np*.bin` / `outuhat_np*.bin`.

Out of scope (explicitly): GPU, replication of GMRES iteration counts, the built-in `getavfield2d` model (inert in this run — trailing physicsparam `[10,0.5,0,4,0.02,2.5]` are read by nothing), the MATLAB adaptive dilatation-sensor AV loop (M4 implements only the 4-stage `tanh` continuation from `pdeapp_ns.m:55-95`).

Parallel posture: the repo convention is MPI-first, but 651 elements / a 27,080-dof condensed trace system is decisively serial territory. **Decision: write the driver with the standard `Mpi::Init`/`Hypre::Init` scaffold but support and validate np=1 only through M4; a hand-assembled `HypreParMatrix` trace system is a documented stretch item (M5).**

---

## 2. Physics module (`ns_physics.hpp`, shared by both drivers)

A single header of plain free functions on raw `double` arrays, no MFEM types, so both the HDG driver and the sibling (DG) driver can share it. **Design rule: the core routines use Exasim's conventions verbatim** — state vector `uq[12] = [r, ru, rv, rE, rx, rux, rvx, rEx, ry, ruy, rvy, rEy]` with `q = -grad(u)` (proved by `qequation.hpp` `M q = C u - E uhat` and the `postdiscretization.cpp:132` comment) — so that `pdemodel.txt` and the generated `my_model.hpp` can be transcribed without any sign surgery. This kills the single most dangerous bug class in this project. Thin adapter wrappers (`FluxPhysGrad`, negating the gradient slots on entry) are provided for the DG driver, which uses physical gradients.

Contents (all transcribed from the **run-dir** `pdemodel.txt`, *not* `pdemodel_ns.m`, which contains density/pressure floors that were stripped from the export):

> **M4 amendment (2026-07-26):** the floor-free export is correct for M3 only.
> The original continuation ran the `pdemodel_ns.m` floors (smoothed `lmax`
> clamps at `rmin=1e-2`/`pmin=1e-3`, `alpha=1e3`, plus `dr`/`dp` gradient
> sensors), and the `0.06 -> 0.04` stage is infeasible without them — the first
> Newton trial drives the shock foot to `p ~ -0.009` and the floor-free
> Sutherland `sqrt(T^3)` is NaN there (proved by rerunning Exasim's own solver
> both ways; see `M4_FAILURE_REPORT.md`). `ns_physics.hpp` therefore carries a
> second flux path transcribed from the regenerated floored model
> (`m4_diagnostic/my_model_floored.hpp`), selected by YAML
> `physics.regularization: floors`. Off by default: M3 decks/gates keep the
> export model bit-for-bit; the M4 continuation decks turn it on. The floors
> are inactive to ~1e-9 at converged states, so both models share the same
> fixed points.

```c++
struct NSParams {           // loaded from YAML; defaults = pdeapp.txt:49 verbatim
  double mu[11];            // gam, Re, Pr, Minf, rinf, ruinf, rvinf, rEinf, Tinf, Tref, Twall
  double tau = 1.0;         // scalar HDG stabilization, all equations, all faces
  // Wall BC ONLY — uses the rounded deck literal mu[8] verbatim:
  double TisoW() const { return mu[10]/mu[9]*mu[8]; }   // = 0.06550101502128686
  // Flux/Sutherland ONLY — full precision, derived from mu[0], mu[3]; never mu[8]:
  double TinfFlux() const { return 1.0/(mu[0]*(mu[0]-1.0)*mu[3]*mu[3]); }
};

// f[8] = (fx[4], fy[4]); optional dfduq[8*12] analytic Jacobian
void NSFlux(const double uq[12], double av, const NSParams&, double f[8], double* dfduq);
// ib in {1,2,3}; fb[4], dfbduq[4*12], dfbduh[4*4]
void NSFbouHdg(int ib, const double uq[12], const double uhat[4], const double n[2],
               const NSParams&, double fb[4], double* dfbduq, double* dfbduh);
void NSHeatFlux(const double uhat[4], const double uq[12], const NSParams&, double f[2]); // fc*Tx, fc*Ty
void NSVisScalars(const double u[4], double s[4]);   // rho, u, v, p = 0.4*(rE - ke)
double SutherlandMu(double p, double r, const NSParams&); // muphys
```

Exact formulas (from `pdemodel.txt` Flux, lines ~40-81; all gradients are model-signed):

- `p = gam1*(rE - r*ke)`, `ke = 0.5*(uv^2+vv^2)`, `T = p/(gam1*r)` (= specific internal energy `e`), `h = E + p/r`.
- Sutherland: `Tr = gam*Minf^2*p/r` (algebraic — this is `T/TinfFlux()`, full precision), `Tphys = Tref*Tr`, `muphys = (1/Re)*Tr^{3/2}*(Tref+110.4)/(Tphys+110.4)`, `fc = muphys*gam/Pr`. (Verified equal to the obfuscated `my_model.hpp:56` expression to 1.8e-16.)
- Derived model-gradients: `ux = (rux - rx*uv)/r`, `px = gam1*(rEx - rx*ke - r*(uv*ux+vv*vx))`, `Tx = (px*r - p*rx)/(gam1*r^2)`, and y-analogues.
- Stresses (Stokes hypothesis): `txx = muphys*(2/3)*(2ux - vy)`, `txy = muphys*(uy + vx)`, `tyy = muphys*(2/3)*(2vy - ux)`.
- Flux (note **+** signs — dissipative only because gradients are model-signed):
  `fx = [ru + av*rx, ru*uv + p + txx + av*rux, rv*uv + txy + av*rvx, ru*h + uv*txx + vv*txy + fc*Tx + av*rEx]`,
  `fy = [rv + av*ry, ru*vv + txy + av*ruy, rv*vv + p + tyy + av*rvy, rv*h + uv*txy + vv*tyy + fc*Ty + av*rEy]`.
- Source = 0. BC blocks exactly as `pdemodel.txt` FbouHdg: ib=1 `mu[4..7]-uhat`; ib=2 `u-uhat`; ib=3 `[u0-uh0, -uh1, -uh2, uh0*TisoW - uh3]`. **tau does not appear in fb** (confirmed: generated `fbou_hdg` never reads tau) — do not add `tau*(u-uhat)` to the boundary trace rows.

**Jacobians: transcribe the generated analytic Jacobians from `my_model.hpp`** (`flux_jac_uq` 8x12 at lines 222-531; `fbou_hdg_jac_uq` 4x12 and `fbou_hdg_jac_uh` 4x4 per ib at 587-805; source Jacobian is zero). Exasim uses exact symbolic Jacobians; hand-deriving them again is pure risk, and AD would add a dependency for no benefit. The transcription is mechanical (rename `x1..xN` temporaries), and the unit test cross-checks three ways: (a) `#include` a lightly adapted copy of `my_model.hpp` in `test_physics.cpp` and compare values at ~1000 random admissible states to 1e-13 relative; (b) central finite differences at eps=1e-6; (c) dissipativity sanity — contract the linearized viscous+AV flux with a gradient perturbation and assert negative-semidefinite entropy production (catches any residual sign error). Watch: the ib=3 energy row is *bilinear* in uhat — `d fb3/d uhat0 = +TisoW` must be present, not just the -1 diagonal.

**AV term**: because the viscous flux is already written in `q`, AV is literally three extra Jacobian entries per equation: `f_c += av*q_c` contributes `∂f_{c,d}/∂q_{c,d} += av` (identity blocks scaled by the frozen scalar). `av` is *data*: precomputed once at all volume/face quadrature points from `vdg.bin` component 0 (see §3), never differentiated, never updated inside Newton. This preserves Exasim's Jacobian exactly.

---

## 3. Mesh strategy: converter from `grid.bin` + `xdg.bin` (recommended), analytic generator as secondary

**Recommendation: converter.** The analytic map (`theta = 3pi/2 - pi*p2`, `d = 4.7 + 1.7*cos(theta)`, `r = d + logdec(p1,5)*(1-d)`, `logdec(x,5) = (1-e^{-5x})/(1-e^{-5})`) is *not polynomial*, so interpolating it at MFEM's node set produces Q4 elements that differ from Exasim's Chebyshev-interpolated Q4 elements at O(h^5) — enough to shift wall heat flux measurably at 31x21 resolution. But Exasim's *discrete* geometry **is** an elementwise Q4 polynomial (the `xdg.bin` map), and re-expressing a Q4 polynomial in a different nodal basis is **exact**. So:

1. **Topology from `grid.bin`** (format: 4 doubles header `[nd=2, np=704, nve=4, ne=651]`, then `p` as 2x704 column-major, then `t` as 4x651 one-based CCW quads). Build `mfem::Mesh mesh(2, 704, 651, nbe, 2)`; `AddVertex` x704; `AddQuad(t[e]-1, attr=1)` x651 preserving Exasim element numbering (element correspondence is then the identity — no centroid matching needed); boundary segments: enumerate boundary edges of `t`, classify by the run's own predicates (ordered tests on the edge midpoint, shared `exasim_mesh.cpp` convention — see README): `sqrt(x^2+y^2) < 1+1e-6` → **attribute 1 (wall, 21 faces)**, else `x > -1e-7` → **attribute 2 (outflow, the theta=pi/2 and 3pi/2 radial ends, 62 faces)**, else **attribute 3 (freestream outer arc, 21 faces)**. `FinalizeQuadMesh(1)`. Sanity: 1354 edges total, 104 boundary, Euler check `4*651 = 2*1250 + 104`.
2. **Geometry from `xdg.bin`** (`[25, 2, 651]` + column-major `(node, comp, elem)`; 3-double header then data, per `readmesh.cpp:653-667`). `mesh.SetCurvature(4, /*discont=*/false, 2, Ordering::byNODES)` (Exasim's dgnodes come from a globally continuous map, so H1 Q4 Nodes are correct). Then for each element `e` and each Nodes dof of that element: get its reference coordinate from `nodes_fes->GetFE(e)->GetNodes()`, evaluate **Exasim's tensor Lagrange basis** at that reference point — 1D nodes `[0, (3-sqrt5)/4, 1/2, (1+sqrt5)/4, 1]` (Chebyshev-Gauss-Lobatto, decoded from `masternodes.bin`; **not** GLL, not uniform), radial index fastest — and dot with the element's `xdg` columns. Write into the Nodes GridFunction via `GetElementVDofs`. Shared dofs get written twice with identical values (continuity of the source map) — assert agreement to 1e-13.
3. **Local orientation correspondence**: the 4 corners of Exasim's 5x5 tensor layout (local nodes 0, 4, 20, 24) coincide exactly with the `grid.bin` vertices of `t[e]` (both were mapped through the same transformation). Determine the dihedral map from MFEM's reference square to Exasim's parameter square per element by corner matching (expected: one uniform map for all elements given the structured generation); assert, and route all `xdg`/`udg`/`vdg` reference-coordinate evaluations through it.
4. **Analytic generator** (`exasim_mesh.cpp::BuildAnalyticMesh(nr, nc, order)`): ~60 lines, `Mesh::MakeCartesian2D(nr, nc, Element::QUADRILATERAL)` + `SetCurvature(order)` + `Mesh::Transform(std::function)` with the map above. Not for replication — for M2 sanity cases, M4 mesh-refinement checks, and MMS.

`udg.bin` (`[25,12,651]`) and `vdg.bin` (`[25,2,651]` — read the full 2-component layout, use comp 0) load through the same basis-evaluation path: for state I/O we build once the exact 25x25 change-of-basis matrix (tensor of the 5x5 1D Vandermonde solve) mapping Exasim nodal values to MFEM `L2_FECollection` nodal values — exact, no interpolation error, because both bases span Q4. For `av` we skip MFEM entirely and tabulate Exasim-basis evaluations directly at the fixed quadrature points (25 volume + 5x4 face per element), stored once.

---

## 4. HDG discretization core (the hand-written part)

### 4.1 What MFEM does and does not provide (honest inventory)

Provided and used: mesh/geometry (`IsoparametricTransformation`, `FaceElementTransformations` with `Loc1/Loc2`, `Mesh::GetElementEdges`, `GetFaceElementTransformations(f, 31)`), the face-only trace collection `DG_Interface_FECollection` (fe_coll.hpp:479), `L2_FECollection` for (u,q) storage/IO, `DenseMatrix`/`DenseTensor`/`LUFactors`/`DenseMatrixInverse` (works without LAPACK; MFEM's internal LU is fine at n=100), `SparseMatrix::AddSubMatrix`, `PetscLinearSolver`, `ParaViewDataCollection`, `IntRules`.

**Not provided — must be hand-written**: there is no HDG anywhere in MFEM (grep confirms zero hits); `mfem::Hybridization` is linear, `BilinearForm`-bound, fixed-constraint-integrator only; DPG `BlockStaticCondensation` is linear and lives outside libmfem; `HyperbolicFormIntegrator`/`EulerFlux` have no viscous terms and `EulerFlux` lacks flux Jacobians. **Decision: hand-roll the entire element+face assembly loop, the q-elimination, the static condensation, and the local recovery**, in one class. Justification: (a) the nonlinear, state-dependent trace flux evaluated at `(uhat, q_interior)` fits none of the linear scaffolding; (b) Exasim's structure (`qequation.hpp` / `uequation.hpp` / `matvec.hpp`) is a complete, line-referenced specification we can mirror block-for-block, which makes verification *against the reference implementation* tractable — bending `Hybridization` would obscure exactly the comparisons we need; (c) the problem is small enough (651 elements) that no MFEM fast-assembly path buys anything.

### 4.2 Weak forms (Exasim conventions, per element K, all spaces broken Q4)

- **q-relation (linear, geometry-only)**: per component c, direction d: `∫_K q φ_i = ∫_K u ∂φ_i/∂x_d - ∮_∂K uhat n_d φ_i`, i.e. per element `q_d = C_d u - E_d uhat` with `C_d = M^{-1}∫(∂φ/∂x_d)φ` (25x25, scalar, shared by all components) and `E_d = M^{-1}∮ φ φ_face n_d` (25x20, scattered by the face->element node map, the analogue of Exasim's `perm`). Matches `qequation.hpp:83-375` exactly. **q is never an independent unknown in the solver** — it is eliminated analytically from the start.
- **Element (u) equation**: `Ru_j = ∫_K ∇φ_j · f(u, q, av) dx - ∮_∂K φ_j fhat ds = 0`, with `fhat = f(uhat, q, av)·n + tau*(u - uhat)` — **the full (inviscid+viscous+AV) flux with the u-slot replaced by uhat and q kept from the element interior** (`drivers.hpp:442-506`, `AddStabilization1` kokkosimpl.h:303-313). This holds **on boundary faces too** — only the trace rows change at boundaries, never the element equation.
- **Trace equation**: interior face F: `∮_F μ (fhat|_{K1} + fhat|_{K2}) ds = 0` (flux continuity; normals opposite). Boundary face: rows replaced by `∮_F μ fb_ib(u, uhat, n) ds = 0` — **mass-weighted (weak), confirmed**: Exasim multiplies `fb` by the face surface Jacobian and applies `Gauss2Node(shapfgw)` (test-function values x weights) before `PutBoundaryNodes` (`uequation.hpp` boundary section), i.e. `Rh_bnd = -∮ φ fb ds` with the same 5-pt face rule. Residual-*norm* parity with Exasim is therefore attainable, not just solution parity.

### 4.3 Trace space representation

`DG_Interface_FECollection tr_fec(4, 2)` + `FiniteElementSpace tr_fes(&mesh, &tr_fec, /*vdim=*/4, Ordering::byVDIM)`. This gives single-valued L2 segment elements (5 nodes) on every mesh edge; `byVDIM` makes each face a contiguous run of `ncu*npf = 20` dofs, matching Exasim's `[ncu, npf, nf]` component-fastest layout and giving clean 20x20 face blocks for the optional block-Jacobi preconditioner. Global size `N = 4*5*1354 = 27,080`. Dof access: `tr_fes.GetFaceVDofs(f, vdofs)` (20 per face); element-side face list via `mesh.GetElementEdges(e, edges, ori)`. **Orientation**: never index trace dofs by element-local convention; always evaluate the trace basis at the face's own reference points and map to element reference coordinates through `FaceElementTransformations::Loc1/Loc2` (with the outward normal from Elem1, negated for Elem2). In 2D with segments this handles the flip exactly; add a unit test that projects a linear function onto uhat and checks both-side consistency on every interior face. Boundary faces carry genuine trace unknowns (BCs are weak, through fb rows) — nothing is eliminated strongly. uhat lives in a plain `Vector` (and optionally a GridFunction on `tr_fes` for checkpointing; note `ParaViewDataCollection` cannot render interface spaces — wall data goes to CSV instead).

### 4.4 Local solves: layout, cost, storage

The naive per-element block is `(u,q)` = (25 nodes)x(12 comps) = **300x300**. We do not build it: eliminating q first via the exact linear relation (as Exasim does in `uEquationSchurBlock`, uequation.hpp:576-795) reduces the LU to `n_u = ncu*npe = 100`, a **27x** flop reduction (100^3 vs 300^3) and ~9x memory reduction.

Newton linearization per element, written **consistently in the clean convention `J Δ = -R`** (with `Ru`, `Rh` as defined in §4.2 and `fhat = f(uhat,q)·n + tau*(u-uhat)`, so `∂fhat/∂u = +tau·I` and `∂fhat/∂uhat = ∂(f·n)/∂u|_(uhat,q) - tau·I`):

```
A ≡ ∂Ru/∂u    = ∫ ∇φ (∂f/∂u) dx  -  tau ∮ φ φᵀ ds                          100x100
B ≡ ∂Ru/∂q    = ∫ ∇φ (∂f/∂q) dx  -  ∮ φ (∂(f·n)/∂q)|_(uhat,q) ds           100x200
F ≡ ∂Ru/∂uhat = -∮ φ [ (∂(f·n)/∂u)|_(uhat,q) - tau·I ] μᵀ ds               100x80
K ≡ ∂Rh/∂u,  G ≡ ∂Rh/∂q,  H ≡ ∂Rh/∂uhat   (fhat sums on interior rows;      80x100, 80x200, 80x80
                                            ∂fb/∂u, 0, ∂fb/∂uhat on bdr rows)
Fold q (Δq_d = C_d Δu - E_d Δuhat):  A += Σ_d B_d C_d;  F -= Σ_d B_d E_d;
                                     K += Σ_d G_d C_d;  H -= Σ_d G_d E_d
Condense:  Hc = H - K·A⁻¹·F  (80x80);   rhs_c = K·A⁻¹·Ru - Rh
Solve      Hc_glob Δuhat = rhs_c_glob;  recover  Δu = -A⁻¹(Ru + F·Δuhat)
Then       q := C u - E uhat  (full recompute, as hdgGetQ)
```

**Debugging note (do not transcribe Exasim's block signs into the table above):** Exasim stores the u-equation blocks *negated* — `D = -A` is built with an explicit sign flip (`uequation.hpp:176-185`, face add at `:337`) and the recovery in `hdgGetDUDG` (`matvec.hpp:228-249`) uses `DinvF := -D⁻¹F`, so its `Δu = D⁻¹(Ru' - F·Δuhat)` is the same equation in negated variables. When diffing our per-element blocks against Exasim's arrays during M2/M3 debugging, compare against the negated forms; the FD check of the *global condensed* residual (M2b) is the sign arbiter.

Storage (per element, doubles x8B): persist between assembly and recovery within one Newton iteration only — `LUFactors(A)` 80 KB, `A⁻¹F` 64 KB, `A⁻¹Ru` 0.8 KB; geometry-constant `C_d` (2x625) 10 KB and `E_d` (2x500) 8 KB persist for the whole run. Total ≈ 651 x 163 KB ≈ **106 MB** peak — trivial. (The 300x300 route would be ~600 MB and slower; rejected.) B, G, K, H are formed in a per-element scratch and folded immediately, never stored globally. Dense kernels: `mfem::DenseMatrix::Mult/AddMult`, `LUFactors::Factor/Solve` (no LAPACK needed; 651 x (2/3)·100^3 ≈ 4e8 flops/Newton iteration — milliseconds).

### 4.5 Condensed Jacobian assembly

`SparseMatrix Hc_glob(N, N)`: for each element, concatenate its 4 faces' `GetFaceVDofs` into an 80-vector and `AddSubMatrix(vdofs, vdofs, Hc_e, /*skip_zeros=*/0)`; `Finalize()` once, then reuse the sparsity graph (`Hc_glob = 0.0;` + re-add) on subsequent Newton iterations. **`skip_zeros=0` is load-bearing**: `AddSubMatrix` defaults to `skip_zeros=1` (sparsemat.hpp:614) and silently drops exact zeros during the graph-building first assembly — and exact zeros are guaranteed here (near-diagonal constant boundary H blocks, `rv=0` symmetry-line states) — after which a later re-add at the missing position hits `MFEM_ABORT("Could not find entry...")` (sparsemat.hpp:943). Stencil: each face couples to at most 7 faces (itself + 3 siblings per adjacent element) → ≤140 nnz/row, ~3.8M nnz ≈ 30 MB. RHS `rhs_c_glob` scatter-added the same way (this *is* the flux-continuity sum — both adjacent elements contribute to shared face rows, mirroring `hdgAssembleRHS`). For the boundary rows, the element loop simply writes the fb-based `K, G, H` rows instead of the fhat-based ones for faces with bdr attribute; the element (A, B, F) rows are unchanged (asymmetry per `uequation.hpp:352-448`).

### 4.6 Quadrature and precomputed tables (constructor of `HDGNavierStokesOperator`)

- Volume: `IntRules.Get(Geometry::SQUARE, 8)` = 5x5 tensor Gauss-Legendre (25 pts); face: `IntRules.Get(Geometry::SEGMENT, 8)` (5 pts). M1 asserts the 1D points equal `[0.04691007703067, 0.23076534494716, 0.5, ...]` symmetric, matching Exasim's decoded `gaussnodes.bin`.
- Shape tables (analogue of `shapegt/shapegw`): L2 Q4 basis values (25x25) and reference gradients (25x25x2) at volume points; trace segment basis (5x5) at face points; element basis traced onto each local face side (25x5 per (face, orientation) variant), all computed once via `FiniteElement::CalcShape/CalcDShape`.
- Geometry factors: `detJ`, `adj(J)` at all volume points; face normal x surface-Jacobian at face points, from the mesh's Q4 `ElementTransformation` / `FaceElementTransformations` with the fixed rules — computed once (mesh never moves), stored in `DenseTensor`s. Because the Nodes were set exactly from `xdg`, these equal Exasim's `sol.elemg` values to roundoff.
- Frozen `av` at all volume/face quadrature points from `vdg.bin` (Exasim-basis evaluation, §3), stored once.

### 4.7 Wall heat flux (Fint-matching, superconvergent HDG flux)

Postprocessor `wall_post.cpp` iterates wall (attr 1) faces; at each face Gauss point evaluates, with **uhat in the u-slot and interior q in the gradient slots** exactly as `pdemodel.txt` Fint/HeatFlux:
`fb = fc*(Tx*nx + Ty*ny) + tau*(rE_interior - rEhat)`, plus `Cp = 2*(p(uhat) - pinf)`, `pinf = 1/(gam*Minf^2)`. Output CSV `(theta, x, y, Cp, q_wall)` at the 5 Gauss points per wall face, `setprecision(16)`. Note Exasim's "temperature" is `e = p/((gam-1)rho)` and `fc = mu*gam/Pr` — the CSV documents this convention so nobody double-normalizes when comparing to other codes. The identical formula is evaluated on the Exasim reference data by `tools/exasim_reference.py` (from `outudg`/`outuhat` + `vdg` — note `outudg` contains the converged q directly, §6.1), so the comparison is convention-proof.

---

## 5. Nonlinear/linear solver strategy and globalization

**Newton (bespoke, ~150 lines in `hdg_newton.hpp`, replicating `NewtonSolver` nonlinearsolver.cpp:150-303):** per iteration: full reassembly at current `(u, q, uhat)`; solve `Hc_glob Δuhat = rhs_c_glob` with x0=0; `uhat += α Δuhat`; local recovery `Δu` then `q := C u - E uhat`; reassemble residual only (cheap residual-only path, no Jacobian blocks); **line search, replicated exactly** (nonlinearsolver.cpp:277-289): `while (‖R_new‖ > ‖R_old‖ && alpha > 0.1) { alpha /= 2; retry }` — the guard tests alpha *before* halving, so the accepted-step sequence is `1, 0.5, 0.25, 0.125, 0.0625` (minimum accepted step 0.0625, and an increase at the final step is accepted — this quirk is load-bearing at Mach 8); **hard abort only when** `(‖R_new‖ > ‖R_old‖ && ‖R_new‖ > 1e6) || isnan(‖R_new‖)` (nonlinearsolver.cpp:262-268) — a large-but-decreasing residual does not abort; converged when `‖Ru‖ + ‖Rh‖ < 1e-6` (same stacked-norm definition); max 30 iterations. `matvecorder/matvectol` are LDG-only in Exasim — ignore.

**Linear solver — deliberate deviation:** primary = PETSc direct (`PetscLinearSolver`, KSPPREONLY + PCLU; MUMPS is in the PETSc link closure even though MFEM's native SuiteSparse is off), via `Input/petsc.opts`. At N=27,080 with 140 nnz/row this is instantaneous and removes the entire preconditioning question from the critical path. Optional mimic mode (YAML `linear.type: gmres_facejacobi`): MFEM `GMRESSolver` (restart/kdim 100, tol 1e-8) with a hand-rolled face-block-Jacobi `Solver` (invert each face's diagonal 20x20 block of `Hc_glob`, per `hdgBlockJacobi` matvec.hpp:197-226) — useful for studying solver parity, not needed for solution parity. The degree-20 harmonic-Ritz/Leja polynomial preconditioner is **not replicated** (documented deviation; affects iteration counts only, not the fixed point).

**Globalization — Mach 8 steady Newton from freestream WILL diverge.** Two paths:

- **Baseline (M3): the Exasim crutches.** Initialize **u only** from `udg.bin` components 0-3 (the converged continuation-Iter-3 state, rho up to 19.5); `av` from `vdg.bin`; `uhat` = trace of u taken from the side-1 (lower-index) element per face (matching `postdiscretization.cpp:119-124`); then **recompute `q := C u - E uhat`** — do *not* load the stored q slots. This mirrors Exasim exactly: `NewtonSolver` calls `hdgGetQ` before the first residual (nonlinearsolver.cpp:172-173), discarding the stored q (which corresponds to the Iter-3 uhat). Loading the stored q would change the first residual and the Newton path relative to the reference run. This starts Newton inside its basin; Exasim converges from here in a handful of iterations and so should we.
- **Stretch (M4): self-contained continuation** reproducing `pdeapp_ns.m:55-95` without any Exasim binaries: wall distance `d = r - 1` (exact for this geometry — verified `vdg.bin ≡ 0.025*tanh(5*(r-1))` to 2.8e-17); first-solve IC `u = [1, tanh(10d)*1, 0, Tinf*((Twall/Tref-1)e^{-10d}+1) + 0.5*tanh(10d)^2]`; four successive steady solves with frozen `av = {0.06*tanh(30d), 0.04*tanh(30d), 0.025*tanh(30d), 0.025*tanh(5d)}`, each warm-starting from the previous. Safety net if any stage's damped Newton still fails (Exasim's did not, but our linear solver differs): pseudo-transient continuation by adding `(1/Δτ_k)∫φ_j Δu` to A only (backward-Euler PTC with SER ramp `Δτ_{k+1} = Δτ_k·‖R_k‖/‖R_{k+1}‖`), switched off once `‖R‖ < 1e-2`. This is a ~30-line addition to the A-block assembly, off by default.

---

## 6. I/O & validation

**YAML deck** (repo convention: `Input/*.yaml`, loaded by a `DriverParams`+`LoadParams` pair that throws on missing keys, `PrintConfig` on rank 0):

```yaml
mesh:     { source: exasim, exasim_dir: /home/quinnchr/dv/Exasim-apps/runs/nsmach8-baseline }
bc:       { boundaryconditions: [3, 2, 1] }   # attr (1=wall face set,2=outflow,3=outer) -> ib type
physics:  { mu: [1.4, 183500, 0.71, 8.03, 1, 1, 0, 0.52769, 0.027694, 124.49, 294.44], tau: 1.0 }
av:       { mode: file }          # file | tanh {lambda, c} | none
init:     { mode: udg_file }      # udg_file (u comps only, q recomputed) | freestream | damped_freestream
newton:   { max_iters: 30, tol: 1.0e-6 }
linear:   { type: petsc_direct }  # petsc_direct | gmres_facejacobi {maxit,kdim,tol}
continuation: { enabled: false, stages: [...] }        # M4
output:   { path: ParaView/mach8_baseline, wall_csv: wall_data.csv, paraview_every: 0 }
```

**ParaView**: `ParaViewDataCollection` per repo convention (`SetPrefixPath`, `SetLevelsOfDetail(4)`, `SetHighOrderOutput(true)`, `VTKFormat::BINARY`); registered fields on the L2 vdim spaces: conservative `[r,ru,rv,rE]`, derived `rho,u,v,p` (matching Exasim's VisScalars incl. the hardcoded `0.4`), `av`, and optionally the 8 gradient components. Saved at Newton start/end plus `paraview_every` cadence over Newton iterations (diagnostic).

### 6.1 Reference-data reassembly (a real converter task — budget it)

The `dataout` files are **per-rank, in DMD-permuted order; `partition.bin` is NOT sufficient** (it gives element->rank only, not within-rank order):

- `outudg_np{0,1}.bin` hold the **full 12-component state** `[25, 12, 326]` / `[25, 12, 325]` (3-double header `[npe, nc, ne_rank]`; this runmode forces `saveSolOpt=1`, solutionwriter.cpp:40). Bonus: components 4-11 are the converged q — exactly what the Fint heat-flux reference needs, no re-differentiation.
- Within-rank element order is `dmd.elempart`: sorted groups [interior-minus-interface, interface subsets, exterior] (domaindecomposition.cpp:340-365) — interface elements are moved to the END of each rank's block. The permutation is stored in `datain/mesh1.bin`/`mesh2.bin` (`ms.elempart`, buildstructs.hpp:439); recover it there (primary), or by byte-matching each rank's xdg block in `datain/sol{1,2}.bin` against global `xdg.bin` (fallback; sol files store `select_columns(xdg, elempart)` verbatim). 326+325=651 confirms no ghost elements are written.
- `outuhat_np{0,1}.bin` are rank-local with **duplicated shared faces** (`8*(3+4*5*690)` and `8*(3+4*5*687)`; 690+687=1377 > 1354 global faces) and no partition-like file for the face map; recover rank-face->global-face from `datain/mesh{1,2}.bin` connectivity (f2e/elemcon).
- **Pragmatic alternative (recommended as the first move):** rerun the reference once with `mpiprocs=1` — then element and face order are the identity and all mappings vanish. Keep the elempart parser as the fallback if a single-rank rerun is not bit-comparable or not convenient.

**Quantitative comparison** (`tools/exasim_reference.py` + `tools/compare_fields.py`, plus an in-driver `--compare` mode):
1. **Field L2 diffs without interpolation error**: element correspondence is the identity (after elempart mapping) and both representations are Q4 on the same reference square, so evaluate both at the 25 Gauss points per element and form relative L2 per component.
2. **Wall quantities**: Cp(theta) and Fint heat flux (identical formula both sides, §4.7, Exasim-side q taken from `outudg` comps 4-11); compare pointwise at wall Gauss points.
3. **Shock standoff**: locate max |d(rho)/dx| along the stagnation line (theta=pi sample line, 1000 points via `GridFunction::GetValue`), both solutions.
4. **Residual cross-check (the strongest test, before any solving)**: load **Exasim's converged pair `(outudg, outuhat)`** into *our* assembler and evaluate `‖Ru‖+‖Rh‖` — if our discretization matches Exasim's, the residual of *their* converged solution in *our* code must be ≤ their final Newton tolerance (~1e-6). This isolates discretization parity from solver behavior and is the acceptance gate for the whole HDG core. (Do not pair `udg.bin` with `outuhat` — `udg.bin` is the Iter-3 *input*, and the mismatched pair gives an O(1) residual and a spurious gate failure.)

---

## 7. File-by-file work breakdown and milestones

All in `/home/quinnchr/dv/Continuum-Mechanics-MFEM/myapps/hdg_navierstokes/` (flat, per sibling-app convention). Makefile copied from `myapps/convection_diffusion/makefile` (config.mk discovery at `$(HOME)/MFEM/mfem/config/config.mk`, `$(MFEM_CXX) $(MFEM_FLAGS) ... $(MFEM_LIBS) -lyaml-cpp` — with per-object rules from the start since this app has ~6 TUs).

| File | Contents |
|---|---|
| `makefile` | targets: `hdg_ns_driver`, `test_physics`, `test_hdg_mms`, `mesh_convert_check` |
| `ns_physics.hpp` | §2: constants, fluxes, Sutherland, FbouHdg, HeatFlux + transcribed `my_model.hpp` Jacobians; +grad adapters for the DG sibling |
| `exasim_io.hpp/.cpp` | binary reader (3-double header + column-major), `grid/xdg/udg/vdg/outudg/outuhat` + `datain/mesh*.bin` elempart parser, Chebyshev-Lobatto basis evaluation, 25x25 change-of-basis, element-orientation map |
| `exasim_mesh.hpp/.cpp` | §3 converter (`BuildExasimMesh`) + analytic generator (`BuildAnalyticMesh`), boundary attribution |
| `hdg_ns_operator.hpp/.cpp` | §4: tables, `C/E`, assembly (residual, Jacobian blocks, fb rows), condensation, `SparseMatrix` scatter, local recovery, residual-only path |
| `hdg_newton.hpp` | §5 damped Newton + line search + optional PTC hook; linear-solver factory (PETSc direct / GMRES+face-Jacobi) |
| `wall_post.hpp/.cpp` | Cp, Fint heat flux CSV, shock standoff, field-comparison mode |
| `hdg_ns_driver.cpp` | `main` per repo pattern (Mpi::Init, OptionsParser `-i`, LoadParams, MFEMInitializePetsc, try/catch) |
| `test_physics.cpp` | flux/BC value + Jacobian checks vs adapted `my_model.hpp` and FD |
| `test_hdg_mms.cpp` | MMS convergence on analytic meshes |
| `Input/*.yaml`, `Input/petsc.opts` | decks per §6 |
| `tools/exasim_reference.py`, `tools/compare_fields.py` | reference wall data + field diffs |

**M1 — Foundations compile and match Exasim data structures (4-5 person-days).**
Deliver: makefile, `ns_physics`, `exasim_io`, `exasim_mesh`, `test_physics`; operator constructor (tables, geometry, C/E).
Acceptance: (a) `test_physics` passes: flux/fb values and all Jacobians match adapted `my_model.hpp` at 1000 random states to 1e-13 rel, FD-consistent, dissipativity sign test passes; (b) mesh converter: MFEM Q4 geometry evaluated at Exasim's 25 reference nodes reproduces `xdg.bin` to 1e-13; boundary faces count 21/62/21 with correct attributes; (c) quadrature points match Exasim's decoded 5-pt GL; (d) `udg`/`vdg` round-trip through change-of-basis to 1e-13; (e) freestream identities evaluated at the **full-precision** state `[1, 1, 0, 0.5 + 1/(gam*(gam-1)*Minf^2)]`: `p = pinf` and `muphys = 1/Re` to roundoff. (At the rounded deck state `rE=0.52769` these identities hold only to ~1.3e-4 — that discrepancy is real and expected; do not "fix" it.)

**M2 — HDG core verified on benign problems (5-6 person-days).**
Deliver: full assembly/condensation/recovery, Newton, PETSc solve, MMS test, ParaView out.
Acceptance: (a) trace-orientation unit test (linear-field projection consistent on all interior faces); (b) Jacobian check: directional FD of the *global condensed* residual matches `Hc_glob` action to 1e-6; (c) MMS (smooth manufactured solution with source, analytic meshes, 3 refinement levels): L2 order ≥ 4.8 for u; (d) freestream preservation: residual of exact freestream ~1e-10 *with freestream Dirichlet applied on all boundary attributes* (the wall BC rows are O(1) at freestream by design); (e) **supersonic viscous sanity case** on the actual geometry — M=4.5, Re=1e3, `av = 0.05*tanh(5(r-1))`, damped-freestream init (shared deck with the DG sibling, `Input/input_m45_sanity.yaml`): Newton (PTC permitted for startup) converges to 1e-6 with positive, y-symmetric fields. (M=4.5 keeps the run's BC types valid — outflow genuinely supersonic, bow shock standoff ~0.5R exits through the outflow plane, well inside the outer arc. A subsonic case would make the run's extrapolation/Dirichlet BCs ill-posed and is *not* a valid sanity test; M=2 puts the shock wing across the outer Dirichlet arc.)

**M3 — Mach 8 replication with frozen AV + Exasim IC (3-5 person-days).**
Deliver: `input_mach8_baseline.yaml` path end-to-end, wall postprocessing, comparison tooling incl. the §6.1 reassembly (or the single-rank reference rerun).
Acceptance (gated in order): (a) **residual cross-check**: `‖Ru‖+‖Rh‖` of Exasim's converged `(outudg, outuhat)` state in our assembler ≤ 1e-5 (target 1e-6) — this certifies discretization parity before any solving; (b) Newton from `udg.bin`-u + trace-init + recomputed q converges to 1e-6 within 30 iterations; (c) field rel-L2 vs Exasim converged solution ≤ 1e-5 per component (expected ~1e-6-1e-8 given (a)); (d) wall Cp max rel diff ≤ 1e-4, Fint heat flux ≤ 1e-3 (gradient quantity), shock standoff within one radial cell.

**M4 — Self-contained continuation (3-4 person-days).**
Deliver: 4-stage AV continuation with damped-freestream IC, optional SER PTC, analytic-mesh option.
Acceptance: full continuation from scratch (no Exasim binaries except mesh, or fully analytic mesh) reaches a converged final-stage solution whose wall heat flux and shock standoff match M3 within 1% (stage-path dependence is physical; exact match not expected unless stage-3 states coincide).

*(M5 stretch, unsized: parallel trace assembly into hand-built `HypreParMatrix` + hypre GMRES; not needed for the deliverable.)*

Total: **15-20 person-days**.

---

## 8. Risks and mitigations

1. **q-sign trap** (the #1 failure mode; wrong sign turns AV+viscosity into anti-diffusion and detonates at Mach 8). *Mitigation*: adopt Exasim's `q = -grad u` convention project-wide in the HDG driver; transcribe model code verbatim; dissipativity unit test; M3(a) residual cross-check would catch it immediately.
2. **fhat evaluated at `(uhat, q_interior)`, not `(u, q)`** — using u changes the scheme, its Jacobian structure, and stability. *Mitigation*: explicit in the operator API (`FluxAtTrace(uhat, q, av)`); covered by M3(a) since a wrong choice yields O(1) residual on Exasim's converged state.
3. **Boundary asymmetry** (element rows keep fhat with tau on boundary faces; only trace rows become fb; fb has no tau). *Mitigation*: single code path for element faces regardless of boundary status; fb-row substitution isolated; M3(a) again.
4. **Basis/node-set mismatch** (Exasim Chebyshev-Lobatto vs MFEM nodes): naive nodal copy is O(1e-3) wrong at p=4. *Mitigation*: exact polynomial change-of-basis everywhere (§3); round-trip tests at 1e-13. Galerkin residuals are basis-independent, so interior solves are unaffected by our choice of MFEM basis.
5. **Curved geometry fidelity**: straight-sided or analytically re-interpolated elements shift wall heat flux. *Mitigation*: Nodes set exactly from `xdg.bin` (§3); geometry acceptance test in M1.
6. **No HDG in MFEM — everything hand-rolled** (schedule risk, not feasibility risk). *Mitigation*: Exasim's `qequation/uequation/matvec` serve as a line-referenced spec; block-by-block FD verification in M2; scope kept serial.
7. **Steady Newton fragility at Mach 8**: converged-restart + frozen AV are load-bearing; even then our different linear solver perturbs the damped-Newton path. *Mitigation*: exact replication of Exasim's line-search rules incl. the accept-at-floor quirk (§5); PETSc *direct* solve removes inner-solve noise entirely (arguably more robust than Exasim's GMRES); PTC fallback wired but default-off.
8. **Reference-data reassembly** (per-rank DMD permutation, duplicated shared faces in outuhat). *Mitigation*: §6.1 elempart parser from `datain/mesh*.bin` with the single-rank reference rerun as the recommended shortcut; budget a day either way.
9. **Rounded constants**: `0.52769`/`0.027694` are what the run used, and the model mixes rounded (wall BC) with full-precision (flux) Tinf. *Mitigation*: YAML defaults copied verbatim from `pdeapp.txt:49`; `TinfFlux()`/`TisoW()` split in `NSParams` (§2); test asserts `TisoW = 0.06550101502128686`.
10. **PETSc direct solver availability** (MUMPS reachable only through PETSc's link closure; MFEM_EXT_LIBS is machine-specific and brittle). *Mitigation*: `petsc.opts`-driven, with the GMRES+face-block-Jacobi path as a pure-MFEM fallback that has no external dependencies; the 27k system is small enough that even undamped GMRES with block-Jacobi will converge.
11. **`vdg` layout**: must be read as `(25, 2, 651)`; component 2 exists and is zero. *Mitigation*: reader asserts header `[25,2,651]` and `max|comp1| < 1e-14`.
12. **Solver-parity expectations**: we will not match Exasim's GMRES iteration counts (no polynomial preconditioner) and possibly not its Newton iteration count. *Mitigation*: scope statement in §1 — parity target is the converged state and derived quantities, certified by the M3(a) residual cross-check, not the iteration history.
