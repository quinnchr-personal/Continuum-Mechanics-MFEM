#!/bin/bash
# The reference cases that failed in run_all.sh (vtk_mesh stub), after it has finished.
E=/home/quinnchr/miniconda3/envs/fenicsx-0.8
export PATH=$E/bin:$PATH OMP_NUM_THREADS=1
while pgrep -f "run_all.sh" > /dev/null; do sleep 5; done
for c in 3D06_sphere_inflation 3D05_cylinder_inflation 3D07_cube_footing; do
  s=$(date +%s)
  $E/bin/python -u ${c}_run.py > $c.log 2>&1
  echo "$c exit $? wall $(( $(date +%s) - s )) s | $(grep 'Elapsed real time' $c.log)" >> times.txt
done
