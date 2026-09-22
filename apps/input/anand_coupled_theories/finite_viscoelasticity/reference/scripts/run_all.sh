#!/bin/bash
# Reference FEniCSx runs, serial, one after another; wall time of each in times.txt; FV12 MPI on 4 ranks last.
E=/home/quinnchr/miniconda3/envs/fenicsx-0.8
export PATH=$E/bin:$PATH OMP_NUM_THREADS=1
: > times.txt
for c in FV01_VHB_uniaxial_tension_eq FV02_VHB_uniaxial_tension_neq FV03_VHB_stress_relaxation FV04_VHB_creep FV06_VHB_sinusoidal_tension FV07_VHB_sinusoidal_shear FV05_VHB_stretch_hold FV10_NBR_beam_impulse_oscillation FV08_NBR_cube_footing FV12_NBR_sphere_indentation_coarse FV09_NBR_bushing_shear FV11_NBR_beam_buckle; do
  s=$(date +%s)
  $E/bin/python -u ${c}_run.py > $c.log 2>&1
  echo "$c exit $? wall $(( $(date +%s) - s )) s | $(grep 'Elapsed real time' $c.log)" >> times.txt
done
s=$(date +%s)
$E/bin/mpirun -np 4 $E/bin/python -u FV12_NBR_sphere_indentation_MPI_run.py > FV12_NBR_sphere_indentation_MPI.log 2>&1
echo "FV12_NBR_sphere_indentation_MPI exit $? wall $(( $(date +%s) - s )) s | $(grep -m1 'Elapsed real time' FV12_NBR_sphere_indentation_MPI.log)" >> times.txt
echo ALL_DONE >> times.txt
