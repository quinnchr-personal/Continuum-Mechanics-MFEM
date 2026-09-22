#!/bin/bash
# Reference FEniCSx runs, serial, one after another; wall time of each in times.txt.
E=/home/quinnchr/miniconda3/envs/fenicsx-0.8
export PATH=$E/bin:$PATH OMP_NUM_THREADS=1
: > times.txt
for c in 3D09_spherical_inclusion 3D01_uniaxial_tension 3D02_simple_shear 3D03_fixed_end_torsion 3D04_hole_in_plate 3D06_sphere_inflation 3D05_cylinder_inflation 3D07_cube_footing 3D08_beam_buckling 3D10_column_twist; do
  s=$(date +%s)
  $E/bin/python -u ${c}_run.py > $c.log 2>&1
  echo "$c exit $? wall $(( $(date +%s) - s )) s | $(grep 'Elapsed real time' $c.log)" >> times.txt
done
