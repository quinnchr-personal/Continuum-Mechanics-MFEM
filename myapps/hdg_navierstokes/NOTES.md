# Milestone notes

## M1 findings needed by M2

- The reference mesh has one uniform local orientation: MFEM corners map to
  Exasim corners as `0,1,2,3`. The converter still determines and checks this
  from corner coordinates on every element.
- `DG_Interface_FECollection(4, 2)` with four components and
  `Ordering::byVDIM` gives 20 vector dofs on every face and 27,080 total trace
  dofs on the 1,354-face mesh.
- `CalcOrtho(FaceElementTransformations::Jacobian())` supplies the
  surface-Jacobian-weighted normal for `Elem1`; constructor tables negate it
  for `Elem2`. M2 face assembly should reuse the stored side-outward normals.
- On the most strongly graded cells, the assembled double-precision
  `C*1-E*1` cancellation is `1.52e-10` in max norm even though all blocks are
  finite and the geometry Jacobians are positive. This is the observed
  geometry/mass-solve roundoff floor to keep visible during M2 residual
  diagnostics; no correction was applied.
- The actual `vdg.bin` header is `[25,2,651]`, its second component is exactly
  zero, and the first component reaches `0.025000000000000001` at the
  constructor quadrature points.
- Parsing rank-local `datain/mesh*.bin` permutations and `dataout/outuhat`
  face maps remains deferred to M3 as specified; the M1 generic 3-double
  array reader already covers the simple `outudg`/`outuhat` container layout.

## M2 findings needed by M3

- MFEM's `Ordering::byVDIM` makes global vector IDs component-fast, but
  `GetFaceVDofs` and `GetElementVDofs` return arrays grouped by component
  (`dof + scalar_dofs*component`). Transfers must explicitly distinguish the
  returned-list layout from the HDG local trace layout
  `component + 4*(face_dof + 5*local_face)`.
- Do not retain an `ElementTransformation*` across a call to
  `GetFaceElementTransformations`: MFEM reuses mutable transformation storage.
  Face physical coordinates (and analytic face AV) must be evaluated through
  `FaceElementTransformations::Face`. Using the stale element pointer left
  volume AV correct but corrupted face AV on two local sides; this produced a
  false asymmetric M=4.5 branch while all inviscid and stabilization terms
  remained reflection-equivariant.
- With that face-coordinate issue fixed, the M=4.5, Re=1000 actual-mesh case
  converges through the ordinary unrestricted condensed solve in 11 Newton
  iterations using SER-PTC with `initial_dt=15`. The M2 acceptance run reached
  residual `7.1986503333782181e-7`, minimum density
  `1.0000000013684818`, minimum pressure `0.035273368672036558`, and
  y-symmetry error `2.3489903626741273e-10`.
- The M2 output path writes high-order binary ParaView data with conservative
  and primitive fields at the initial and final Newton states. M3 can add wall
  and reference-comparison postprocessing without changing this writer.

## M3 findings needed by M4

- The two-rank `elempartpts` are `[305,21,21]` and `[304,21,21]`; therefore
  the owned output counts are 326 and 325, while each rank also carries 21
  ghost elements. Rank-local `xdg` from `datain/sol{1,2}.bin` byte-matches the
  corresponding global `xdg.bin` columns under `elempart`.
- `outuhat` has header `[ncu,npf,nf]=[4,5,690/687]` and component-fast
  storage `component + 4*(node + 5*face)`. Treating it as the generic
  `[node,component,element]` container scrambles the trace and gives an O(50)
  residual. The checked face reconstruction covers 1,377 local faces, maps
  to all 1,354 global faces, and finds 23 duplicated rank-interface faces
  whose values agree exactly.
- The converged reference pair has `||Ru||+||Rh|| =
  3.2998974152112905e-7`; its stored q and independently recomputed
  `C*u-E*uhat` differ by `1.3509212307697425e-12` relative.
- The Mach 8 baseline from `udg.bin` components 0-3, lower-index-side trace,
  and recomputed q converged in 4 Newton updates: residuals
  `0.40623030748514022, 0.013092347638725129,
  0.0064884343747586091, 0.00028592672769557588,
  3.2999123144061386e-7`. Every accepted alpha was 1.0; no line-search
  damping or accept-at-floor event occurred.
- On the final clean acceptance run, timed Newton work was 5.08423231 s in
  HDG assembly (including residual-only trials), 1.105271788 s in PETSc
  direct trace solves, and 6.2371649 s total. These are single-process wall-clock
  timings on the development host and are intended as an M4 comparison
  baseline, not a performance contract.
- Final-state relative L2 errors versus Exasim were
  `[5.350823864955018e-11, 2.192169270389216e-11,
  7.029717639862986e-11, 3.914696683623190e-11]`. Wall Cp and Fint maximum
  relative differences were `7.918626348422956e-10` and
  `5.010155960985892e-10`; the 1,000-point stagnation-line shock locations
  selected the same sample in both states.

## M4 findings (resumed diagnostic; details in M4_FAILURE_REPORT.md)

- The exported model (run-dir `pdemodel.txt` = builtin model 7 = our
  `ns_physics.hpp`) is **not** the model the original continuation ran.
  `pdemodel_ns.m` regularizes the flux with smoothed floors
  `lmax(x,a) = x*(atan(a*x)/pi + 1/2) - atan(a)/pi + 1/2`, `a=1e3`,
  `rmin=1e-2`, `pmin=1e-3`, plus `dr`/`dp` gradient sensors; these were
  stripped from the export. They are inactive to ~1e-9 at converged
  states (all M3 gates blind to them) but required to survive the
  `0.06 -> 0.04` stage transient, whose first Newton step drives the
  stagnation-line shock foot to `p ~ -0.009` (Sutherland `sqrt(T^3)`
  becomes NaN without floors). Exasim's own floor-free binary aborts on
  that first step; the floored rebuild runs all 4 stages in 9/4/4/4
  updates with no PTC and reproduces `udg.bin` to ~1e-5.
- The run-dir text-mode app re-preprocesses from
  `grid/xdg/udg/vdg.bin` each invocation (`gendatain=0`), so stage
  continuation is driven by overwriting `udg.bin`/`vdg.bin`
  (`m4_diagnostic/write_stage.py`) — no MATLAB needed. `writemeshsol=1`
  is required for `mpiprocs=1` runs (else `datain/mesh.bin` is missing);
  `debugmode=1` dumps the initial state and first Newton update then
  stops.
- Exasim's per-stage restart re-initializes `uhat` to the one-sided
  trace and recomputes q; this alone produces O(1) initial residuals at
  a stage's own converged state (2.98 at `(S1, av=0.06)`), and the side
  convention differs between np1 Exasim and our driver (ours gives 3.89
  at the same state; both re-converge in 2 full steps). Our M4 driver
  instead carries the converged trace across stages — a semantic
  difference from Exasim, but immaterial to the failure: both starts
  produce the same first-step trial (`min p -0.00899` both codes).
- Unblocking M4-as-specified means porting the `pdemodel_ns.m`
  regularized flux + analytic Jacobians into `ns_physics.hpp`
  (`m4_diagnostic/pdemodel_floored.txt` is the verified text2code
  source). M3 parity is unaffected (floored stage-4 control matches the
  floor-free control to 6 digits).

## M4 resolution (done; facts needed by M5/DG)

- `ns_physics.hpp` now has the floored flux + Jacobian transcribed from
  `m4_diagnostic/my_model_floored.hpp`, behind `NSParams::regularized`
  (YAML `physics.regularization: floors`, default off). Transcription
  parity vs the generated model is exactly 0 over 2000 states including
  negative-pressure ones; floored/unfloored agreement at admissible
  states is `4.2e-7` (dominated by the sensor tail at p near 0.08, so a
  1e-6 gate, not 1e-9).
- The continuation loop re-initializes the trace at each stage start
  (Exasim per-stage restart semantics); stage histories are then
  directly comparable to Exasim runs.
- M4 passes on the prescribed 4-stage schedule with no PTC:
  9/4/4/4 updates (converted and analytic mesh), final residual
  `3.2999400258199408e-07`, wall Fint/Cp vs M3 `3.4e-6`/`2.9e-6`,
  shock-standoff sample identical. Full `make test` (M1-M4) green;
  M3 recorded values reproduced bit-for-bit with floors off.
- FD-testing the floored Jacobian needs a ~1e-5 gate: the smoothed
  hinge carries `alpha^2` curvature, so central differences at h=1e-6
  bottom out near `5e-7`.
- Makefile per-object dependency lines must list `ns_physics.hpp` for
  every TU that includes it — a stale-object `NSParams` ABI mismatch
  crashes with `free(): invalid pointer` (hit once during this work;
  deps are fixed now).
