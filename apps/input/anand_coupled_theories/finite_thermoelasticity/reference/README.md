# Reference results of the finite thermoelasticity examples

The histories that the reference's own FEniCSx notebooks record while they run, one
CSV per case, named after the input of this directory's parent.
`apps/anand_thermo_plots.py` overlays them on the plots of the inputs (`--reference DIR`,
this directory by default).

Source: the companion codes of Anand, Stewart and Chester, *Introduction to coupled
theories in solid mechanics*, `03_finite_thermoelasticity/TE*.ipynb` of
https://github.com/SolidMechanicsCoupledTheories/FEniCSx_codes at the commit in
`scripts/FEniCSx_codes_commit.txt`, run with dolfinx 0.8.0 (conda-forge, env
`fenicsx-0.8`, see `../../finite_elasticity/reference/README.md`), serial, on
2026-09-22 (`scripts/times_2026-09-22.txt`).

`scripts/patch_notebooks.py` turns the notebooks' code cells into the run scripts
`scripts/TE*_run.py`: headless (pyvista stubbed, the Agg matplotlib backend, the
IPython image display dropped), the `timeHist*` arrays written to `hist/<case>.csv`
right after the solve loop and, for the stretched cylinder (02), the reaction of the
loaded face added as a further column: the residual with no boundary conditions
applied, summed over the prescribed dofs of the face. `scripts/run_all.sh` runs them in
the notebooks' directory of a clone of the repository. The columns are the notebooks'
`timeHist0, timeHist1, ...` (units kPa, mm, s, K: forces in mN):

| case | columns |
|------|---------|
| 01 | time (s), temperature at (5, 10), (5, 5), (5, 0) (K), pressure at (5, 0) (kPa) |
| 02 | time (s), u_z at (0, 10) (mm), 2 pi int P_zz r dA over the top face (mN, the axial force), temperature at (0, 10) (K), reaction of the top face (the residual of the notebook's r-weighted form summed over its u_z dofs: 2 pi times it is the axial force) |
| 03 | time (s), u_z at (0, 10) (mm), 2 pi int P_zz r dA over the top face (mN), temperature at (0, 10) (K) |
| 04 | time (s), u_y at the tip (100, 0.5) (mm), temperature there (K) |
| 05 | time (s), u_z at (0, 1) (mm), 2 pi int P_zz r dA over the top face (mN; the free, heated face: a discretization residue, not plotted), temperature at (0, 1) (K) |
| 06 | time (s), u_z at A = (70, 60, 0) (mm), u_z at B = (100, 0, 0) (mm) |

The notebooks' meshes are dolfinx `create_rectangle` boxes of crossed triangles (01:
6 x 6; 02, 03: 20 x 20; 05: 20 x 2), the Gmsh bilayer of 200 x 2 triangles per layer
(04) and a `create_box` of 10 x 10 x 2 tetrahedra (06); the inputs use quadrilaterals
and hexahedra of the same subdivisions. `same_mesh/` holds this code's input of 06 on
the reference's own box (`apps/mesh/sail_box_ref.msh`, dolfinx's `create_box` written
as a Gmsh file by `../../finite_elasticity/reference/scripts/export_box.py`), so that
the two codes are compared on the same elements; 06 is the case whose mesh differs
most from the reference's and whose response (a membrane under pressure) is the most
mesh sensitive.

The reference pins "the corners" of the sail (TE06) with `locate_dofs_geometrical` on
x = y = 0 and x = y = 100 without a condition on z: every displacement dof of those two
edges is fixed, which the input reproduces with the five nodes of each edge.
