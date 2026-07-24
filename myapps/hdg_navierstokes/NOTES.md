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
