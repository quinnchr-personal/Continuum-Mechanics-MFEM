Force-displacement curves of the exercise's own drivers (`myapps/elastic_bar`), kept here so that
`apps/elastic_bar_compare.py` can lay the framework's results over them in any clone
(`myapps/elastic_bar` is not tracked). Format `step,disp_z,total_Fz`.

- `myapps_nonlinear_gent.csv`: `main_nonlinear` with `input.yaml` as it stands there
  (gent_compressible, mu = 5e6, kappa = 1.5e9, Jm = 50, order 1, 500 steps, no refinement); a copy of
  its `ParaViewNonLinearParallel4/force_displacement.csv`. The framework's `bar_gent.yaml` agrees with
  it to 1e-12 relative.
- `myapps_linear.csv`: `main` (MFEM's `ElasticityIntegrator`) on the same mesh with MU = 5e6,
  K = 1.5e9, order 1, 100 steps and the Krylov tolerance of `petsc.opts` tightened from 1e-8 to
  1e-14. The framework's `bar_linear.yaml` agrees with it to 3e-11 relative; with the exercise's own
  1e-8 the difference is 1.6e-5, which is that solver tolerance.
