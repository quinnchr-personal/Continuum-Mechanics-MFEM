#!/bin/bash
E=/home/quinnchr/miniconda3/envs/fenicsx-0.8
export PATH=$E/bin:$PATH OMP_NUM_THREADS=1
s=$(date +%s)
$E/bin/mpirun -np 4 $E/bin/python -u FV12_NBR_sphere_indentation_MPI_run.py > FV12_NBR_sphere_indentation_MPI.log 2>&1
echo "FV12_NBR_sphere_indentation_MPI exit $? wall $(( $(date +%s) - s )) s | $(grep -m1 'Elapsed real time' FV12_NBR_sphere_indentation_MPI.log)" >> times.txt
echo MPI_DONE >> times.txt
