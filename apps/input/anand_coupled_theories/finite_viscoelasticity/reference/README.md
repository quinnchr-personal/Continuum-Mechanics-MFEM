# Reference results of the finite viscoelasticity examples

The histories that the reference's own FEniCSx notebooks record while they run, one
CSV per case, named after the input of this directory's parent, and the experimental
data the reference pages overlay (`exp_data/`, the notebooks' own files).
`apps/anand_visco_plots.py` overlays them on the plots of the inputs (`--reference DIR`,
this directory by default).

Source: the companion codes of Anand, Stewart and Chester, *Introduction to coupled
theories in solid mechanics*, `02_finite_viscoelasticity/FV*.ipynb` of
https://github.com/SolidMechanicsCoupledTheories/FEniCSx_codes at the commit in
`scripts/FEniCSx_codes_commit.txt`, run with dolfinx 0.8.0 (conda-forge, env
`fenicsx-0.8`, see `../../finite_elasticity/reference/README.md`), serial, on
2026-09-22 (`scripts/times_2026-09-22.txt`).

`scripts/patch_notebooks.py` turns the notebooks' code cells into the run scripts
`scripts/FV*_run.py`: headless (pyvista stubbed, the Agg matplotlib backend, the
IPython image display dropped), the `timeHist*` arrays written to `hist/<case>.csv`
right after the solve loop (inside the loop over the three rates for FV02, one file per
rate), and, where a plot uses a force, the reaction of the loaded face added as a
further column: the residual with no boundary conditions applied, summed over the
prescribed dofs of the face, which this code's reactions are. `scripts/run_all.sh`
runs them in the notebooks' directory of a clone of the repository. The columns are
the notebooks' `timeHist0, timeHist1, ...`:

| case | columns |
|------|---------|
| 01, 02, 03, 05 | time (s), u_y of the loaded face (mm), traction integral of the stress over that face (kPa), reaction (kPa) |
| 04 | time (s), applied traction (kPa), u_y at the centre of the loaded face (mm) |
| 06 | time (s), u_y of the loaded face (mm), traction integral over the 25 mm^2 face (mN), reaction (mN) |
| 07 | time (s), u_x of the top face (mm), traction integral (kPa), reaction (kPa) |
| 08 | time (s), pressure on the patch (kPa), u_z of the corner (0, 0, 50) (mm) |
| 09 | time (s), u_x of the top face (mm), traction integral (mN), reaction (mN) |
| 10 | time (s), shear traction (kPa), u_y of the tip (mm) |
| 11 | time (s), u_z of the top face (mm), traction integral (mN), reaction with inertia (mN) |
| 12 | time (s), applied depth (mm), u_z of the corner (0, 0, 50) (mm), traction integral over the top face (mN), the contact term alone summed over the z dofs of the top face (mN; the indenter force with the sign reversed) |

The unit cube of 01 to 05 has an area of 1 mm^2, so its forces are stresses.

`02_uniaxial_rate_*.csv` are the notebook as published; `02_uniaxial_rate_*_corrected.csv`
the same run with a typo of that notebook corrected (`scripts/FV02_fixed_run.py`): its
equilibrium Cauchy stress is written with `Fbar = J**(-1.3)*F` where the other eleven
notebooks have `J**(-1/3)`, which lowers the stress by the factor J^(-1.933) of the
volume ratio, 0.4 percent at the stretch 2.5 with K = 1000 G. This code implements the
model with the exponent -1/3; the plots show both.

`12_sphere_indentation.csv` is the coarse notebook (`FV12_NBR_sphere_indentation_coarse`,
a 10 x 10 x 6 box of tetrahedra); the reference's finer parallel script
(`FV12_NBR_sphere_indentation_MPI`, `scripts/run_fv12_mpi.sh`) was patched the same way.

`same_mesh/` holds this code's inputs of 07, 10 and 12 on the reference's own meshes
(`apps/mesh/*_box_ref.msh`, dolfinx's `create_box` written as Gmsh files by
`../../finite_elasticity/reference/scripts/export_box.py`), so that the two codes are
compared on the same elements.
