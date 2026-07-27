# hycfd — general-purpose hypersonic CFD on MFEM

An implicit HDG (hybridizable discontinuous Galerkin) compressible
Navier–Stokes solver seeded from the verified `myapps/hdg_navierstokes`
Exasim replica, which stays frozen as this code's regression oracle.

Milestones (see the project plan; G1–G6 complete):

- **G1**: seeded and de-replicated. `q = +grad(u)` convention,
  full-precision `PerfectGasParams` (no rounded deck literals), Armijo line
  search with PTC step rejection.
- **G2**: pluggable `PhysicsModel` interface (perfect gas now, sized for
  two-temperature thermochemical nonequilibrium later).
- **G3**: runtime order p=1..k, tri+quad meshes, general Gmsh/MFEM mesh
  input, boundary-attribute → BC mapping.
- **G4**: MPI (ParMesh, parallel DG_Interface trace space, PtAP condensed
  assembly, parallel MUMPS via PETSc).
- **G5**: BC catalog (adiabatic wall, slip/symmetry, characteristic
  farfield, pressure outlet).
- **G6**: dilatation-sensor artificial viscosity refreshed every Newton
  iteration (adaptive frozen-per-iteration AV: EMA under-relaxation,
  decaying bootstrap floor, permanent freeze near convergence) and
  generic staged continuation — stages are YAML override maps applied on
  top of the base deck (`physics`/`av`/`newton`/`ptc`), giving Mach/Re
  homotopy and AV-mode handovers for free. The end-to-end gate
  (`input/mach8_sensor.yaml`) reaches the frozen Mach 8 solution from a
  damped freestream with no hand-tuned AV profile.
- **G7 (in progress)**: physics validation against classical references
  in `tests/test_validation.cpp` — cylinder bow-shock standoff vs the
  Sinclair–Cui (2017) relation (Billig's cylinder fit underpredicts for
  M ≲ 6) and stagnation Cp vs Rayleigh pitot at M=3/5; M=6 flat plate vs
  a compressible-Blasius shooting solution (Sutherland C, cold wall);
  M=3 compression corner vs oblique-shock theory.

Nodal-DG sibling driver: out of scope (HDG only); the `physics/` and
`solvers/` layers are discretization-agnostic by design.

## Layout

```
physics/         PhysicsModel interface, perfect-gas kernels, BC registry
discretization/  HDG operator (assembly/condensation), dilatation-sensor AV
solvers/         damped Newton (Armijo + PTC/SER, NewtonPrepare hook)
io/              MFEM/Gmsh mesh input; Exasim readers (regression only)
post/            wall Cp / heat flux, shock standoff + density crossing
driver/          YAML-driven main (hycfd) with staged continuation
tests/           milestone gates + G7 validation suite
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
make check      # fast gates: physics parity/FD, HDG core MMS, PTC limit,
                # BC catalog, sensor AV units
make test       # + np={2,4} parallel fixtures, frozen Mach 8 regression
                # (serial and np=4), M4.5 sanity + sensor runs, Mach 8
                # sensor campaign, G7 validation suite
```

Requires MFEM 4.8+ built with MPI/PETSc (config.mk auto-discovered),
yaml-cpp, and the Exasim run directory
(`EXASIM_RUN_DIR`, default `/home/quinnchr/dv/Exasim-apps/runs/nsmach8-baseline`)
for the regression mesh, oracle headers, and reference state.
