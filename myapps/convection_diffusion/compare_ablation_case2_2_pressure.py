#!/usr/bin/env python3
"""Compare MFEM ablation case-2.2 pressure outputs against Amaryllis/PATO."""

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
DEFAULT_AMARYLLIS_PRESSURE = Path(
    "/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/"
    "data/ref/PATO/PATO_Pressure_TestCase_2.2.txt"
)
DEFAULT_OUT_PREFIX = "ablation_case2_2"
DEFAULT_PRESSURE_TOL = {
    "pressure_rmse_max": 100.0,
    "pressure_max_abs_max": 250.0,
}
PRESSURE_TOL_KEYS = (
    "pressure_rmse_max",
    "pressure_max_abs_max",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory containing pressure_probes.csv. If the CSV is missing, "
            "pressure histories are sampled from the MFEM ParaView output."
        ),
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input YAML file with acceptance tolerances and probe coordinates",
    )
    parser.add_argument(
        "--amaryllis-pressure",
        default=str(DEFAULT_AMARYLLIS_PRESSURE),
        help="Amaryllis/PATO pressure reference file",
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
        for key in PRESSURE_TOL_KEYS:
            writer.writerow([key, tol[key]])


def load_amaryllis_pressure(path: Path) -> np.ndarray:
    data = common.ensure_2d(np.loadtxt(path, skiprows=1))
    if data.shape[1] < 2:
        raise RuntimeError(
            f"Pressure reference {path} must contain time plus at least one signal."
        )
    return data


def build_amaryllis_pressure_by_depth(
    am_pressure: np.ndarray, probe_depths: list[float]
) -> list[tuple[float, str, np.ndarray]]:
    n_signals = int(am_pressure.shape[1]) - 1
    items: list[tuple[float, str, np.ndarray]] = []
    for i in range(n_signals):
        name = "Pw" if i == 0 else f"P{i + 1}"
        depth = probe_depths[i] if i < len(probe_depths) else float(i)
        items.append((depth, name, am_pressure[:, i + 1]))
    items.sort(key=lambda x: x[0])
    return items


def nonzero_plot_mask(values: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.isfinite(values) & (np.abs(values) > tol)


def sample_pressure_probes_from_paraview(
    output_dir: Path, input_yaml: Path, max_time_steps: int | None = None
) -> np.ndarray:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "pressure_probes.csv is missing and pyvista is not available "
            "for ParaView post-processing."
        ) from exc

    pvd_path = common.find_mfem_collection_pvd(output_dir, input_yaml)
    if pvd_path is None:
        raise FileNotFoundError(
            "pressure_probes.csv is missing and no MFEM ParaView .pvd "
            f"collection was found under {output_dir}"
        )

    probe_y = common.load_probe_y_from_yaml(input_yaml)
    if not probe_y:
        raise RuntimeError(f"No probe_y entries found in {input_yaml}")
    probe_x = common.load_probe_x_from_yaml(input_yaml)

    time_entries = common.filter_available_time_entries(
        common.load_pvd_time_entries(pvd_path), "mesh", str(pvd_path)
    )
    if max_time_steps is not None:
        time_entries = time_entries[:max_time_steps]
        if not time_entries:
            raise RuntimeError(
                "No ParaView time steps remain after applying --max-time-steps."
            )
    time_values = np.asarray([time_value for time_value, _ in time_entries], dtype=float)

    dtype = [("time", float), ("wall", float)]
    dtype.extend((f"TC{i}", float) for i in range(1, len(probe_y)))
    probes = np.zeros(time_values.size, dtype=dtype)
    probes["time"] = time_values
    print(
        f"Sampling pressure probes from ParaView over {len(time_entries)} time steps...",
        flush=True,
    )

    def read_dataset_at_path(mesh_path: Path):
        ds = pv.read(str(mesh_path))
        ds = common.select_dataset_with_point_fields(ds, "pressure")
        if ds is None:
            raise RuntimeError(f"Missing pressure field in {mesh_path}")
        return common.warp_dataset_points_by_vector_field(ds, "ale_displacement")

    def sample_pressure_at(mesh, x: float, y: float) -> float:
        query = np.array([[x, y, 0.0]], dtype=float)
        sampled = pv.PolyData(query).sample(mesh)
        mask = np.asarray(sampled.point_data.get("vtkValidPointMask", [1]), dtype=int)
        if mask.size == 0 or int(mask[0]) == 0:
            return float("nan")
        vals = np.asarray(sampled.point_data["pressure"], dtype=float)
        if vals.size == 0:
            return float("nan")
        return float(vals.reshape(-1)[0])

    warned_probe_x = False
    for i_time, (_, entry) in enumerate(time_entries):
        common.print_progress("Pressure probe sampling", i_time + 1, len(time_entries))
        mesh = read_dataset_at_path(entry["mesh"])
        x_min, x_max, y_bottom_live, y_top_live, _, _ = mesh.bounds
        x_span = max(1.0e-12, float(x_max - x_min))
        y_span = max(1.0e-12, float(y_top_live - y_bottom_live))
        x_inset = 1.0e-6 * x_span
        y_inset = 1.0e-6 * y_span
        y_min_sample = float(y_bottom_live + y_inset)
        y_max_sample = float(y_top_live - y_inset)

        if float(x_min) <= probe_x <= float(x_max):
            x_query = max(float(x_min + x_inset), min(float(x_max - x_inset), probe_x))
        else:
            x_query = 0.5 * (float(x_min) + float(x_max))
            if not warned_probe_x:
                print(
                    "Configured probe_x="
                    f"{probe_x:.6g} lies outside ParaView x-bounds "
                    f"[{float(x_min):.6g}, {float(x_max):.6g}]; "
                    f"sampling along centerline x={x_query:.6g} instead."
                )
                warned_probe_x = True

        wall_val = sample_pressure_at(mesh, x_query, y_max_sample)
        if not np.isfinite(wall_val):
            wall_val = sample_pressure_at(
                mesh, x_query, float(y_top_live - 10.0 * y_inset)
            )
        if not np.isfinite(wall_val):
            wall_val = 0.0
        probes["wall"][i_time] = wall_val

        for i_probe in range(1, len(probe_y)):
            y_fixed = float(probe_y[i_probe])
            if y_fixed < y_bottom_live or y_fixed > y_top_live:
                probes[f"TC{i_probe}"][i_time] = 0.0
                continue
            y_query = max(y_min_sample, min(y_max_sample, y_fixed))
            val = sample_pressure_at(mesh, x_query, y_query)
            probes[f"TC{i_probe}"][i_time] = val if np.isfinite(val) else 0.0

    return probes


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_time_steps is not None and args.max_time_steps < 1:
        parser.error("--max-time-steps must be >= 1")

    out_dir = Path(args.output_dir)
    probes_csv = out_dir / "pressure_probes.csv"
    tol = common.load_acceptance_from_yaml(Path(args.input))
    for key, value in DEFAULT_PRESSURE_TOL.items():
        tol.setdefault(key, value)

    probes = common.load_named_csv(
        probes_csv, required=False, description="pressure_probes.csv"
    )
    if probes is None:
        print(
            f"pressure_probes.csv not found or empty in {out_dir}; "
            "sampling pressure probes from ParaView output instead."
        )
        probes = sample_pressure_probes_from_paraview(
            out_dir, Path(args.input), max_time_steps=args.max_time_steps
        )

    am_pressure = load_amaryllis_pressure(Path(args.amaryllis_pressure))
    probe_depths = common.load_probe_depths_from_yaml(Path(args.input))

    mfem_by_depth = common.build_mfem_temperature_by_depth(probes, probe_depths)
    am_by_depth = build_amaryllis_pressure_by_depth(am_pressure, probe_depths)
    n_common = min(len(mfem_by_depth), len(am_by_depth))
    if n_common == 0:
        raise RuntimeError("No pressure probes available for MFEM/Amaryllis comparison.")

    mfem_time = probes["time"]
    am_time = am_pressure[:, 0]
    probe_pairs = list(zip(mfem_by_depth[:n_common], am_by_depth[:n_common]))
    pressure_cmp_mask = common.overlap_mask_for_reference_time(
        am_time, [mfem_time], "pressure comparison"
    )

    pressure_metrics: list[tuple[str, float, float, bool]] = []
    for (depth_mf, name_mf, sig_mf), (_, name_am, sig_am) in probe_pairs:
        if np.any(pressure_cmp_mask):
            mfem_interp = np.interp(am_time[pressure_cmp_mask], mfem_time, sig_mf)
            am_sig = sig_am[pressure_cmp_mask]
            rmse_val = common.rmse(mfem_interp, am_sig)
            max_val = common.max_abs(mfem_interp, am_sig)
            ok = (
                rmse_val <= tol["pressure_rmse_max"]
                and max_val <= tol["pressure_max_abs_max"]
            )
        else:
            rmse_val = float("nan")
            max_val = float("nan")
            ok = True
        sig_label = f"{name_mf}~{name_am}@depth={depth_mf:.6g}m"
        pressure_metrics.append((sig_label, rmse_val, max_val, ok))

    pressure_pass = all(metric[3] for metric in pressure_metrics)

    metrics_csv = out_dir / "amaryllis_pressure_metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["category", "signal", "rmse", "max_abs", "metric", "value", "tolerance", "pass"]
        )
        for sig, rmse_val, max_val, ok in pressure_metrics:
            writer.writerow(["pressure", sig, rmse_val, max_val, "", "", "", int(ok)])
        writer.writerow(["summary", "pressure", "", "", "", "", "", int(pressure_pass)])

    tol_csv = out_dir / "amaryllis_pressure_tolerances.csv"
    write_tolerance_csv(tol_csv, tol)

    plot_path = out_dir / f"{args.out_prefix}_pressure_history.png"
    plt.figure(figsize=(14, 5))
    cmap = plt.get_cmap("tab10")
    for i, ((depth_mf, name_mf, sig_mf), (_, name_ref, sig_ref)) in enumerate(probe_pairs):
        color = "black" if i == 0 else cmap((i - 1) % 10)
        depth_label = f"{depth_mf:.4f} m"
        mfem_mask = nonzero_plot_mask(sig_mf)
        am_mask = nonzero_plot_mask(sig_ref)
        plt.plot(
            mfem_time[mfem_mask],
            sig_mf[mfem_mask],
            color=color,
            lw=2,
            label=f"MFEM {name_mf} ({depth_label})",
        )
        plt.plot(
            am_time[am_mask],
            sig_ref[am_mask],
            color=color,
            lw=1.6,
            ls="--",
            label=f"Amaryllis {name_ref} ({depth_label})",
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (Pa)")
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
    print(f"Pressure PASS: {pressure_pass}")
    for sig, rmse_val, max_val, ok in pressure_metrics:
        print(
            f"  {sig}: RMSE/max = {rmse_val:.6g} / {max_val:.6g} Pa "
            f"(pass={ok})"
        )

    if not pressure_pass:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"{Path(__file__).name}: {exc}", file=sys.stderr)
        sys.exit(1)
