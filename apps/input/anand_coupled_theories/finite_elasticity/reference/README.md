# Reference results of the finite elasticity examples

The histories that the reference's own FEniCSx notebooks record while they run,
one CSV per case, named after the input of this directory's parent. `apps/anand_plots.py`
overlays them on the plots of the inputs (`--reference DIR`, this directory by default).

Source: the companion codes of Anand, Stewart and Chester, *Introduction to coupled
theories in solid mechanics*, `01_finite_elasticity/3D*_v0p8.ipynb` of
https://github.com/SolidMechanicsCoupledTheories/FEniCSx_codes at the commit in
`scripts/FEniCSx_codes_commit.txt`, run with dolfinx 0.8.0 (conda-forge, env
`fenicsx-0.8`: `conda create -n fenicsx-0.8 -c conda-forge python=3.11 fenics-dolfinx=0.8 mpich`),
serial, on 2026-09-21 (`scripts/times_2026-09-21.txt`).

`scripts/3D*_run.py` are the notebooks' code cells, unchanged except for running
headless (pyvista stubbed, the Agg matplotlib backend, the IPython image display
dropped) and for writing the `timeHist*` arrays to `hist/<case>.csv` right after the
solve loop. `scripts/run_all.sh` runs them in the notebooks' `meshes/` directory of a
clone of the repository. The columns are the notebooks' `timeHist0, timeHist1, ...`:

| case | columns |
|------|---------|
| 01, 09 | u_y of the loaded corner (mm), P22 on the loaded face (kPa), reaction / area (kPa) |
| 02 | u_x of the top corner (mm), P12 on the top face (kPa), reaction (kPa) |
| 03 | twist angle (rad), torque (N mm), axial force (mN), reaction torque about +x, reaction axial force (bottom face; opposite signs) |
| 04 | u_x of the loaded corner (mm), P11 integrated over the loaded face (kPa mm), reaction (kPa mm) |
| 05, 06 | time (s), pressure (kPa), radial displacement of the inner wall (mm) |
| 07 | time (s), pressure on the patch (kPa), u_z of the footing centre (mm) |
| 08 | time (s), u_z of the top (mm, negative), axial force (mN, negative), reaction (mN) |

The force columns of the notebooks are boundary integrals of the finite element stress
(`dot(Tmat, n) * ds`); the reaction columns, added by `scripts/*_rxn_run.py`, are the
residual (no boundary conditions applied) summed over the dofs of the loaded face, as
this code's reactions. `anand_plots.py` overlays the reaction where present.

Cases 05 and 06 end when the reference's Newton fails past the limit point (the
notebook's `Ended Early`); the rows after that are dropped. 3D10 records no history.

Where the loaded face has clamped corners (02, 08) the traction integral differs from
the reaction by several percent on the reference's tetrahedral meshes.
`3D02_same_mesh_check_run.py` and `3D08_same_mesh_check_run.py` are the 02 and 08
notebooks with the quadratic law and the reaction (the residual summed over the
loaded-face dofs) recorded as a last column; `export_box.py` writes dolfinx's `create_box` tetrahedral mesh as
a Gmsh file with `box.geo`'s face names, so that this code can run on the reference's
own mesh (`python export_box.py 8 8 4 shear_ref.msh` in the `fenicsx-0.8` env).
