#!/bin/bash
# Regenerates the reference error histories of apps/input/scalar_transport/reference from the
# drivers of myapps/convection_diffusion: builds the five drivers with the compile lines of that
# directory's makefile (binaries under $OUT/bin, the sources untouched), runs each with its input
# as it stands there (mesh and PETSc options relative to that directory) with the outputs
# redirected under $OUT and the per-step ParaView files switched off, then copies the CSVs.
#   apps/input/scalar_transport/reference/scripts/regenerate.sh [OUT]     (from the repository root)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$ROOT/myapps/convection_diffusion
OUT=${1:-$ROOT/out/scalar_transport/myapps_reference}
REF=$ROOT/apps/input/scalar_transport/reference
mkdir -p "$OUT/bin" "$OUT/inputs"
cd "$SRC"
DRIVERS="linear_convection_diffusion_1D linear_convection_diffusion_2D linear_convection_diffusion_2D_circle nonlinear_convection_diffusion_1D diffusion_mms"
echo "building the drivers into $OUT/bin"
make -n $DRIVERS | grep -v '^make' | sed "s# -o \([A-Za-z_0-9]*\) # -o $OUT/bin/\1 #" | bash
# The transient linear drivers accumulate the Krylov noise of every step in their error
# histories (with Input/petsc.opts, rtol 1e-10 and atol 1e-12 against right-hand sides of
# order 1e-3, the Pe = 1 history deviates from the framework's by 4e-4 at t = 1): they run
# with the tolerances tightened so that the histories measure the discretisation alone.
cat > "$OUT/inputs/petsc_tight.opts" <<'EOF'
# Input/petsc.opts of myapps/convection_diffusion with the Krylov tolerances tightened
# for the reference histories of apps/input/scalar_transport.
-ksp_type gmres
-ksp_rtol 1.0e-13
-ksp_atol 1.0e-18
-ksp_max_it 5000
-pc_type jacobi
EOF
run() {
  local driver=$1 input=$2 outdir=$3 opts=${4:-}
  mkdir -p "$OUT/$outdir"
  sed -e "s#^output_path:.*#output_path: $OUT/$outdir#" -e "s#^save_paraview:.*#save_paraview: false#" \
      "Input/$input" > "$OUT/inputs/$input"
  if [ -n "$opts" ]; then
    sed -i "s#^petsc_options_file:.*#petsc_options_file: $opts#" "$OUT/inputs/$input"
  fi
  echo "running $driver with $input"
  "$OUT/bin/$driver" -i "$OUT/inputs/$input" > "$OUT/$outdir.log" 2>&1
}
run linear_convection_diffusion_1D input.yaml convection_diffusion_peclet "$OUT/inputs/petsc_tight.opts"
run linear_convection_diffusion_2D input_2d.yaml steady_cdr_square
run linear_convection_diffusion_2D_circle input_2d_circle.yaml steady_cdr_disk
run nonlinear_convection_diffusion_1D input_nonlinear_1d.yaml nonlinear_diffusion_kirchhoff
run diffusion_mms input_diffusion_mms.yaml transient_diffusion_mms "$OUT/inputs/petsc_tight.opts"
cp "$OUT/convection_diffusion_peclet/error_history.csv" "$REF/myapps_convection_diffusion_peclet.csv"
cp "$OUT/steady_cdr_square/error_history_2D.csv" "$REF/myapps_steady_cdr_square.csv"
cp "$OUT/steady_cdr_disk/error_history_2D_circle.csv" "$REF/myapps_steady_cdr_disk.csv"
cp "$OUT/nonlinear_diffusion_kirchhoff/error_history_nonlinear_1D.csv" "$REF/myapps_nonlinear_diffusion_kirchhoff.csv"
cp "$OUT/nonlinear_diffusion_kirchhoff/newton_history_nonlinear_1D.csv" "$REF/myapps_nonlinear_diffusion_kirchhoff_newton.csv"
cp "$OUT/transient_diffusion_mms/error_history.csv" "$REF/myapps_transient_diffusion_mms.csv"
echo "reference CSVs copied to $REF"
