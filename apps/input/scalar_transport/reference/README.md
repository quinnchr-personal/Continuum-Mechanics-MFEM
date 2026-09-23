Error histories of the verification drivers of `myapps/convection_diffusion`, kept here so that
`apps/scalar_transport_compare.py` and `tests/test_scalar_verification.cpp` can lay the framework's
results over them in any clone (`myapps/` is the user's own tree). All were produced by
`scripts/regenerate.sh`, which builds the five drivers with the compile lines of that directory's
makefile (the binaries outside the tree) and runs each with its input as it stands there
(`Input/input.yaml`, `input_2d.yaml`, `input_2d_circle.yaml`, `input_nonlinear_1d.yaml`,
`input_diffusion_mms.yaml`: meshes `Mesh/unit_square.msh`, `square_0p01.msh`, `unit_circle.msh`,
copied to `apps/mesh/square_tri.msh`, `square_0p01_tri.msh`, `disk_tri.msh`), the outputs redirected
and the per-step ParaView files switched off. Recorded 2026-09-23 from the sources of commit
93908810ed (unchanged since 2026-02-25), MFEM 4.8.1 with PETSc 3.19.

- `myapps_convection_diffusion_peclet.csv`: `linear_convection_diffusion_1D` (Pe = 1, 10, 100 in one
  run, p = 3, dt = 1e-3 to t = 1, GMRES + Jacobi). Columns `step,time,abs_l2_pe1,rel_l2_pe1,
  abs_l2_pe2,rel_l2_pe2,abs_l2_pe3,rel_l2_pe3`, one row per step from 0. Run with the Krylov
  tolerances of `Input/petsc.opts` tightened from rtol 1e-10, atol 1e-12 to 1e-13, 1e-18
  (`petsc_tight.opts` written by the script): with the driver's own options the noise of a thousand
  solves against right-hand sides of order 1e-3 puts the Pe = 1 history 4e-4 off the framework's at
  t = 1, where its error is 1.6e-5; with the tightened ones the histories agree to the level the
  tests assert.
- `myapps_steady_cdr_square.csv`: `linear_convection_diffusion_2D` (p = 3, one solve). Columns
  `abs_l2,rel_l2`, one row.
- `myapps_steady_cdr_disk.csv`: `linear_convection_diffusion_2D_circle` (p = 3, one solve, GMRES +
  block Jacobi/ILU to 1e-10). Columns `abs_l2,rel_l2`, one row.
- `myapps_nonlinear_diffusion_kirchhoff.csv`: `nonlinear_convection_diffusion_1D` (p = 3, dt = 0.1 to
  t = 1, full Newton to 1e-8 relative). Columns `step,time,abs_l2,rel_l2,newton_iters,final_residual`;
  the errors are against the 1000-term series solution; the initial condition is that series at t = 0.
  `myapps_nonlinear_diffusion_kirchhoff_newton.csv` holds its Newton history, one row per iteration
  (`step,time,iter,residual,residual0,rel_residual,update_norm,update0,rel_update,converged`).
- `myapps_transient_diffusion_mms.csv`: `diffusion_mms` (p = 1 on the mesh refined once, dt = 0.01 to
  t = 2, the tightened Krylov tolerances as above). Columns `step,time,l2_error,linf_error` (the L∞
  error is the largest nodal error against the interpolant).

The drivers' L2 errors use the quadrature rule of order 2p + 3 and divide by the L2 norm of the exact
solution for the relative error; the framework does the same (`output.exact`). Their operators are
assembled by MFEM's stock integrators, whose rules integrate the polynomial integrands exactly, so
the discrete problems coincide with the framework's; the source terms use `DomainLFIntegrator`'s rule
of order 2p and the nonlinear terms of the Kirchhoff driver a rule of order 2p + 2, which the
comparisons reproduce through `transport.quadrature_order`.
