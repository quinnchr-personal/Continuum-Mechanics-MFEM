#!/bin/bash
# The fourteen viscoelastic inputs one after another on 4 ranks (direct solver), logs and wall times under
# out/anand_coupled_theories/finite_viscoelasticity/logs (status.txt). Run from the repository root.
L=out/anand_coupled_theories/finite_viscoelasticity/logs
D=apps/input/anand_coupled_theories/finite_viscoelasticity
: > $L/status.txt
for c in 01_uniaxial_equilibrium 02_uniaxial_rate_0p01 02_uniaxial_rate_0p03 02_uniaxial_rate_0p05 03_stress_relaxation 04_creep 06_sinusoidal_tension 05_stretch_hold 10_beam_impulse 07_sinusoidal_shear 08_cube_footing 11_column_buckling 09_bushing_shear 12_sphere_indentation; do
  s=$(date +%s)
  mpirun --bind-to none -np 4 ./build/apps/solid_mechanics -i $D/$c.yaml > $L/$c.log 2> $L/$c.err
  echo "$c exit $? wall $(( $(date +%s) - s )) s" >> $L/status.txt
done
echo ALL_DONE >> $L/status.txt
