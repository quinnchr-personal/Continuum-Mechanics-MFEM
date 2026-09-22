#!/bin/bash
# The six thermoelastic inputs one after another on 4 ranks (direct solver), logs and wall times under
# out/anand_coupled_theories/finite_thermoelasticity/logs (status.txt). Run from the repository root.
L=out/anand_coupled_theories/finite_thermoelasticity/logs
D=apps/input/anand_coupled_theories/finite_thermoelasticity
mkdir -p $L
: > $L/status.txt
for c in 01_constrained_heating 02_adiabatic_stretch 05_plate_flux 03_heating_contraction 04_bilayer_actuator 06_solar_sail; do
  s=$(date +%s)
  mpirun --bind-to none -np 4 ./build/apps/solid_mechanics -i $D/$c.yaml > $L/$c.log 2> $L/$c.err
  echo "$c exit $? wall $(( $(date +%s) - s )) s" >> $L/status.txt
done
echo ALL_DONE >> $L/status.txt
