#!/bin/bash
E=/home/quinnchr/miniconda3/envs/fenicsx-0.8
export PATH=$E/bin:$PATH OMP_NUM_THREADS=1
: > times.txt
for c in TE01_pe_constrained_heating TE02_axi_thermoelas_stretch_adiabatic TE03_axi_thermoelas_steps TE04_pe_bilayer TE05_axi_circular_plate_flux TE06_solar_sail; do
  s=$(date +%s)
  $E/bin/python -u ${c}_run.py > $c.log 2>&1
  echo "$c exit $? wall $(( $(date +%s) - s )) s | $(grep 'Elapsed real time' $c.log | head -1)" >> times.txt
done
echo ALL_DONE >> times.txt
