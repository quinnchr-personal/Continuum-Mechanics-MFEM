# hdg_navierstokes — MFEM drivers replicating Exasim `nsmach8-baseline`

Two planned drivers reproducing the Exasim run at
`/home/quinnchr/dv/Exasim-apps/runs/nsmach8-baseline`:
**Mach 8.03 viscous flow over a half-cylinder** (Re=183,500, Pr=0.71, isothermal wall
Twall=294.44 K / Tref=124.49 K, Sutherland viscosity, frozen Laplacian artificial
viscosity `av = 0.025*tanh(5(r-1))`), steady, p=4 on 651 curved quads (31x21 graded
half-annulus), solved by damped Newton. The Exasim case originates from
`Exasim/examples/NavierStokes/nshtmach8/pdeapp_ns.m` (4-stage AV continuation;
the exported `udg.bin` is the converged stage-3 state, `vdg.bin` the stage-4 AV field).

| Plan | Method | Doc |
|---|---|---|
| A | HDG — replicates Exasim's numerics (mixed (u, q, uhat), tau=1 trace flux, static condensation, Newton on the 27,080-dof trace system); targets **discretization parity** with the reference (residual cross-check at 1e-6) | [PLAN_HDG.md](PLAN_HDG.md) |
| B | Implicit nodal DG — Rusanov + BR2 + the same frozen AV, PTC(SER)-Newton on 65,100 dofs; the robust "conventional hypersonic DG" counterpart; targets **physics parity** (Cp/heat-flux/standoff windows) | [PLAN_DG.md](PLAN_DG.md) |

~40% of the work is shared: `ns_physics` (fluxes, Sutherland, BC states, Jacobians
transcribed from the generated `my_model.hpp`), `exasim_io` (binary readers,
Chebyshev-Lobatto basis, change-of-basis), `exasim_mesh` (exact Q4 mesh converter
from `grid.bin`/`xdg.bin`), comparison tools, and the `input_m45_sanity.yaml` deck.
Combined effort estimate: ~17-20 person-days per driver, minus the shared 40% if both are built
(HDG-first order recommended, since its M3 residual cross-check certifies the shared
physics/mesh layers for the DG driver too).

## Shared conventions (both plans conform to these)

1. **Boundary attributes = Exasim mesh boundary indices**: 1 = cylinder wall (21 faces),
   2 = x=0 outflow plane (62), 3 = outer arc (21). Drivers map attribute -> BC type via
   `boundaryconditions: [3, 2, 1]` in YAML, mirroring `pdeapp.txt:52`
   (wall -> isothermal no-slip, outflow -> supersonic extrapolation, outer -> freestream).
2. **Deck constants verbatim** — the run used rounded literals (`rEinf=0.52769`,
   `Tinf=0.027694`, internally inconsistent at 4e-6). The wall temperature is
   `TisoW = 0.027694*294.44/124.49 = 0.06550101502128686` (rounded `mu[8]`, as the
   generated model does); the Sutherland law uses full-precision
   `Tinf = 1/(gam*(gam-1)*Minf^2)` and never reads `mu[8]`. Both plans encode this split.
3. **Gradient sign**: Exasim stores q = **-grad(u)**. The HDG driver adopts that
   convention verbatim (model code transcribes cleanly); the DG driver uses physical
   gradients through sign-flip adapters in `ns_physics`. Unit tests pin both.
4. **Reference-data caveat**: `dataout/*_np{0,1}.bin` are per-rank in DMD-permuted order
   (`dmd.elempart`; `partition.bin` is insufficient — interface elements are moved to the
   end of each rank's block) and `outudg` holds the full 12-component state `[25,12,ne]`.
   Recommended first move: **rerun the reference once with `mpiprocs=1`** so all orderings
   become the identity; otherwise parse `elempart` from `datain/mesh{1,2}.bin`.
5. **Serial-first**: 651 elements is decisively serial; MPI scaffolding kept, np=1
   validated, parallel assembly a stretch item in both plans.
6. **Linear solver**: PETSc direct (MUMPS via the PETSc closure) by default; iterative
   options (GMRES + face-block-Jacobi / BlockILU) provided for study. Exasim's degree-20
   polynomial preconditioner is deliberately not replicated (iteration counts out of scope).

## Decisions left open (defaults chosen, flag if you disagree)

- **Field-parity bar (HDG M3)**: 1e-5 relative L2 vs the Exasim converged state (backed by
  the residual cross-check). Relax to wall-quantity agreement only if the linear-solver
  deviation proves to matter.
- **DG heat-flux window**: 10% vs Exasim on the front half — a discretization difference,
  not an error bar; add the mesh-refinement consistency study if a tighter statement is needed.
- **Adaptive dilatation-sensor AV loop** (`hypersoniccylinder_mach8/pdeapp.m`, Helmholtz-
  smoothed sensor): out of scope in both plans (the baseline export predates it); natural M5 if wanted.
- **Wall-flux mode in the DG driver**: defaults to the Exasim-mimicking one-sided flux
  (`wall_flux: exasim`) with conventional mirrored-Rusanov as a toggle.
- An Exasim rerun with `nsurf` enabled would give an independent wall-heat-flux reference
  (the run wrote none; both plans self-compute it via the `Fint` formula on both datasets).
