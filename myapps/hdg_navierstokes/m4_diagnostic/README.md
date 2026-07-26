# M4 resumed-diagnostic artifacts (2026-07-26)

Companion data for the "Resumed diagnostic" section of `../M4_FAILURE_REPORT.md`.
Everything here was produced in scratch copies of
`/home/quinnchr/dv/Exasim-apps/runs/nsmach8-baseline`; no repo code or
reference-run files were modified.

## Contents

- `write_stage.py` — overwrites `udg.bin`/`vdg.bin` in a run-dir copy for one
  continuation stage (`--ic` builds the `pdeapp_ns.m:62-67` damped-freestream
  IC; `--udg-from` restarts from a previous stage's `outudg_np0.bin`).
  Validated against the reference `vdg.bin` at `2.3e-17`.
- `pdemodel_floored.txt` — the run-dir `pdemodel.txt` with the
  `pdemodel_ns.m` density/pressure regularization restored in `Flux`
  (smoothed `lmax` floors, `dr`/`dp` sensors). Verified via a full
  4-stage continuation and a 6-digit match of the stage-4 control.
- `my_model_floored.hpp` — the text2code output generated from
  `pdemodel_floored.txt` (flux + symbolic Jacobians). This is the
  transcription source for the `regularized` path in `../ns_physics.hpp`
  and the cross-check reference for `../test_physics.cpp` (which pins
  the transcription at exactly 0 error).
- `Kokkos_Core.hpp` — empty stub so `test_physics` can include
  `my_model_floored.hpp` without a Kokkos installation.
- `states_floorfree/stage1_outudg.bin` — Exasim's converged stage-1 state
  (`0.06*tanh(30d)`, floor-free model, np=1, `[25,12,651]` + 3-double header).
  The starting state of the first failing Newton step.
- `states_floored/stage{1..4}_outudg.bin` — the reconstructed original
  continuation's intermediate states (floored model). `stage3` matches the
  reference `udg.bin` to rel-L2 `[3.0e-6, 5.4e-6, 1.2e-5, 5.9e-6]`.
- `first_step_dump/` — Exasim `debugmode=1` dump of the failing stage-2
  step (floor-free): `out0newton_uh.bin` (initial trace, 27080 flat),
  `out0newton_udg.bin` / `out1newton_udg.bin` (initial / alpha=1 trial
  state, 651*300 flat, headerless), `out1newton_x.bin` (first Newton
  trace update). Trial min pressure `-0.008993212119465887` at
  `(-1.7515, 0.0000)`.
- `logs/` — stdout of every run: floor-free control/stage1/stage2(+debug)/
  stage1-warm-restart, floored control/stage1-4, and our driver's runs from
  the exported stage-1 state (`e3_ours_s1_av006.log`, `e4_ours_s1_av004.log`).

## Reproduction recipe

1. Copy the reference run dir; in `pdeapp.txt` set `mpiprocs = 1`,
   `writemeshsol = 1` (required for np=1: else `datain/mesh.bin` is never
   written), `saveParaview = 0`. `gendatain = 0` already makes every app
   invocation re-preprocess from `grid/xdg/udg/vdg.bin`.
2. Per stage: `write_stage.py --dir . --lam L --c C [--ic|--udg-from PREV]`,
   then run `Exasim-apps/build/builtinlibrary/exasimapp pdeapp.txt`; collect
   `dataout/outudg_np0.bin`.
3. Floored variant: replace `pdemodel.txt` with `pdemodel_floored.txt`, set
   `builtinmodelID = 0`, `gencode = 1`, add
   `exasimpath = "/home/quinnchr/dv/Exasim";`, run
   `Exasim-install/bin/text2code pdeapp.txt --out-dir .` (then `./code2cpp`
   manually — the runner invokes it without `./`), build with
   `cmake -B build -DExasim_DIR=.../Exasim-install/lib/cmake/Exasim`,
   set `gencode = 0` back, and run `build/exasimapp pdeapp.txt`.
