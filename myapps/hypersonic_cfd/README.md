# hycfd — general-purpose hypersonic CFD on MFEM

An implicit HDG (hybridizable discontinuous Galerkin) compressible
Navier–Stokes solver seeded from the verified `myapps/hdg_navierstokes`
Exasim replica, which stays frozen as this code's regression oracle.

Roadmap (G-milestones; see the project plan):

- **G1 (this state)**: seeded and de-replicated. `q = +grad(u)` convention,
  full-precision `PerfectGasParams` (no rounded deck literals), Armijo line
  search with PTC step rejection. Still 2D / quads / p=4 / serial.
- **G2**: pluggable `PhysicsModel` interface (perfect gas now, sized for
  two-temperature thermochemical nonequilibrium later).
- **G3**: runtime order p=1..k, tri+quad meshes, general Gmsh/MFEM mesh
  input, boundary-attribute → BC mapping.
- **G4**: MPI (ParMesh, parallel DG_Interface trace space, PtAP condensed
  assembly, parallel MUMPS + GMRES/ASM options).
- **G5**: BC catalog (adiabatic wall, slip/symmetry, characteristic
  farfield, pressure outlet).
- **G6**: sensor-based artificial viscosity + generic staged continuation.
- **G7**: validation (Billig shock standoff, compressible Blasius,
  compression corner) and CI hardening.

Nodal-DG sibling driver: out of scope (HDG only); the `physics/` and
`solvers/` layers are discretization-agnostic by design.

## Layout

```
physics/         perfect-gas kernels + typed full-precision parameters
discretization/  HDG operator: gradient elimination, assembly, condensation
solvers/         damped Newton (Armijo + PTC/SER), condensed trace solve
io/              Exasim mesh/state readers (regression only), analytic mesh
post/            wall Cp / heat flux / shock standoff extraction
driver/          YAML-driven main (hycfd)
tests/           G1 gates
input/           YAML decks + petsc.opts
```

## Conventions (differences from the frozen replica)

- Auxiliary gradient `q = +grad(u)` (Exasim uses `q = -grad(u)`; the sign
  map lives inside `physics/perfect_gas.hpp` wrappers only).
- All nondimensional constants derive at full precision from
  `PerfectGasParams` (gamma, Re, Pr, M, T_inf_K, Twall_K); the rounded
  Exasim deck literals and their split usage are gone.
- Armijo backtracking line search; a failed search rejects the step and
  shrinks the PTC pseudo-time step instead of accepting a residual increase.

## Build and test

```
make            # all binaries
make check      # fast gates: physics parity/FD, HDG core MMS, PTC limit
make test       # + frozen Mach 8 regression and M4.5 sanity acceptance
```

Requires MFEM 4.8+ built with MPI/PETSc (config.mk auto-discovered),
yaml-cpp, and the Exasim run directory
(`EXASIM_RUN_DIR`, default `/home/quinnchr/dv/Exasim-apps/runs/nsmach8-baseline`)
for the regression mesh, oracle headers, and reference state.
