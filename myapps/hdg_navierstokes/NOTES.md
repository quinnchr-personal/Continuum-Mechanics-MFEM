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
