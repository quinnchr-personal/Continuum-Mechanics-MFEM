#!/usr/bin/env python3
"""Compare MFEM ablation case-2.2 temperature outputs against Amaryllis."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-convection_diffusion")
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compare_ablation_case2_2 as common


DEFAULT_OUTPUT_DIR = Path("ParaView/ablation_case2_2")
DEFAULT_INPUT = Path("Input/input_ablation_case2_2.yaml")
DEFAULT_AMARYLLIS_ENERGY = Path(
    "/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/"
    "data/ref/PATO/PATO_Energy_TestCase_2.2.txt"
)
DEFAULT_OUT_PREFIX = "ablation_case2_2"
TEMPERATURE_TOL_KEYS = (
    "temperature_rmse_max",
    "temperature_max_abs_max",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory containing temperature_probes.csv. If the CSV is missing, "
            "temperature histories are sampled from the MFEM ParaView output."
        ),
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input YAML file with acceptance tolerances and probe coordinates",
    )
    parser.add_argument(
        "--amaryllis-energy",
        default=str(DEFAULT_AMARYLLIS_ENERGY),
        help="Amaryllis energy reference file",
    )
    parser.add_argument(
        "--out-prefix",
        default=DEFAULT_OUT_PREFIX,
        help="Prefix for generated plot filenames",
    )
    parser.add_argument(
        "--max-time-steps",
        type=int,
        default=None,
        help=(
            "Maximum number of ParaView time steps to sample when falling back "
            "to MFEM .pvd output."
        ),
    )
    return parser


def write_tolerance_csv(path: Path, tol: dict[str, float]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["signal", "tolerance"])
        for key in TEMPERATURE_TOL_KEYS:
            writer.writerow([key, tol[key]])


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_time_steps is not None and args.max_time_steps < 1:
        parser.error("--max-time-steps must be >= 1")

    out_dir = Path(args.output_dir)
    probes_csv = out_dir / "temperature_probes.csv"
    tol = common.load_acceptance_from_yaml(Path(args.input))

    probes = common.load_named_csv(
        probes_csv, required=False, description="temperature_probes.csv"
    )
    if probes is None:
        print(
            f"temperature_probes.csv not found or empty in {out_dir}; "
            "sampling temperature probes from ParaView output instead."
        )
        probes = common.sample_temperature_probes_from_paraview(
            out_dir, Path(args.input), max_time_steps=args.max_time_steps
        )

    am_energy = common.ensure_2d(np.loadtxt(args.amaryllis_energy, skiprows=1))
    probe_depths = common.load_probe_depths_from_yaml(Path(args.input))

    mfem_by_depth = common.build_mfem_temperature_by_depth(probes, probe_depths)
    am_by_depth = common.build_amaryllis_temperature_by_depth(am_energy, probe_depths)
    n_common = min(len(mfem_by_depth), len(am_by_depth))
    if n_common == 0:
        raise RuntimeError("No temperature probes available for MFEM/Amaryllis comparison.")

    mfem_time = probes["time"]
    am_time = am_energy[:, 0]
    probe_pairs = list(zip(mfem_by_depth[:n_common], am_by_depth[:n_common]))
    temp_cmp_mask = common.overlap_mask_for_reference_time(
        am_time, [mfem_time], "temperature comparison"
    )

    temp_metrics: list[tuple[str, float, float, bool]] = []
    for (depth_mf, name_mf, sig_mf), (_, name_am, sig_am) in probe_pairs:
        valid = (sig_am > 1.0) & temp_cmp_mask
        if np.any(valid):
            mfem_interp = np.interp(am_time[valid], mfem_time, sig_mf)
            am_sig = sig_am[valid]
            rmse_val = common.rmse(mfem_interp, am_sig)
            max_val = common.max_abs(mfem_interp, am_sig)
            ok = (
                rmse_val <= tol["temperature_rmse_max"]
                and max_val <= tol["temperature_max_abs_max"]
            )
        else:
            rmse_val = float("nan")
            max_val = float("nan")
            ok = True
        sig_label = f"{name_mf}~{name_am}@depth={depth_mf:.6g}m"
        temp_metrics.append((sig_label, rmse_val, max_val, ok))

    wall_mf = np.interp(am_time, mfem_time, probe_pairs[0][0][2])
    wall_ref = probe_pairs[0][1][2]
    wall_valid = (wall_ref > 1.0) & temp_cmp_mask
    heat_rmse, heat_max = common.segmented_rmse_max(
        am_time, wall_mf, wall_ref, 0.1, 60.0, wall_valid
    )
    cool_rmse, cool_max = common.segmented_rmse_max(
        am_time, wall_mf, wall_ref, 60.1, 120.0, wall_valid
    )
    heat_pass = (
        (not np.isfinite(heat_rmse) and not np.isfinite(heat_max))
        or (
            heat_rmse <= tol["temperature_rmse_max"]
            and heat_max <= tol["temperature_max_abs_max"]
        )
    )
    cool_pass = (
        (not np.isfinite(cool_rmse) and not np.isfinite(cool_max))
        or (
            cool_rmse <= tol["temperature_rmse_max"]
            and cool_max <= tol["temperature_max_abs_max"]
        )
    )
    temperature_pass = all(metric[3] for metric in temp_metrics) and heat_pass and cool_pass

    metrics_csv = out_dir / "amaryllis_temperature_metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["category", "signal", "rmse", "max_abs", "metric", "value", "tolerance", "pass"]
        )
        for sig, rmse_val, max_val, ok in temp_metrics:
            writer.writerow(["temperature", sig, rmse_val, max_val, "", "", "", int(ok)])
        writer.writerow(
            [
                "temperature_segment",
                "wall_heating_0.1_60s",
                heat_rmse,
                heat_max,
                "",
                "",
                "",
                int(heat_pass),
            ]
        )
        writer.writerow(
            [
                "temperature_segment",
                "wall_cooling_60.1_120s",
                cool_rmse,
                cool_max,
                "",
                "",
                "",
                int(cool_pass),
            ]
        )
        writer.writerow(["summary", "temperature", "", "", "", "", "", int(temperature_pass)])

    tol_csv = out_dir / "amaryllis_temperature_tolerances.csv"
    write_tolerance_csv(tol_csv, tol)

    plot_path = out_dir / f"{args.out_prefix}_temperature_history.png"
    plt.figure(figsize=(14, 5))
    cmap = plt.get_cmap("tab10")
    for i, ((depth_mf, name_mf, sig_mf), (_, name_ref, sig_ref)) in enumerate(probe_pairs):
        color = "black" if i == 0 else cmap((i - 1) % 10)
        depth_label = f"{depth_mf:.4f} m"
        plt.plot(mfem_time, sig_mf, color=color, lw=2, label=f"MFEM {name_mf} ({depth_label})")
        plt.plot(
            am_time,
            sig_ref,
            color=color,
            lw=1.6,
            ls="--",
            label=f"Amaryllis {name_ref} ({depth_label})",
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.xlim(0.0, max(float(mfem_time[-1]), float(am_time[-1])))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=9)
    plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close()

    if len(mfem_by_depth) != len(am_by_depth):
        print("Probe-count mismatch: using nearest-to-surface shared count =", n_common)
        print("  MFEM probes:", len(mfem_by_depth), "Amaryllis probes:", len(am_by_depth))

    print(f"Wrote: {metrics_csv}")
    print(f"Wrote: {tol_csv}")
    print(f"Wrote: {plot_path}")
    print(f"Temperature PASS: {temperature_pass}")
    print(f"Wall heating RMSE/max: {heat_rmse:.6g} / {heat_max:.6g}")
    print(f"Wall cooling RMSE/max: {cool_rmse:.6g} / {cool_max:.6g}")

    if not temperature_pass:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"{Path(__file__).name}: {exc}", file=sys.stderr)
        sys.exit(1)
