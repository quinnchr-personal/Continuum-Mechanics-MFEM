#!/usr/bin/env python3
"""Compare MFEM rho_s_qp density profiles against PATO reference data."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-convection_diffusion")
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

import matplotlib
import numpy as np
import pyvista as pv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_YAML = Path("Input/input_ablation_case2_2.yaml")
DEFAULT_PATO_DENSITY = Path(
    "/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/"
    "data/ref/PATO/PATO_DensityHistory_TestCase_2.2.csv"
)
DEFAULT_OUT_PREFIX = "ablation_case2_2_density"
TARGET_FIELD = "rho_s_qp"
PATO_ZERO_TOL = 1.0e-14


@dataclass(frozen=True)
class SnapshotSelection:
    index: int
    time: float


def load_top_level_value_from_yaml(path: Path, key: str) -> str | None:
    if not path.exists():
        return None

    key_prefix = f"{key}:"
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith(" "):
            continue
        if not stripped.startswith(key_prefix):
            continue

        value = stripped.split(":", 1)[1].strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value

    return None


def resolve_input_relative_path(input_yaml: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (input_yaml.parent / path).resolve()


def resolve_output_dir(input_yaml: Path, output_dir_override: Path | None) -> Path:
    if output_dir_override is not None:
        return output_dir_override.resolve()

    raw_output_dir = load_top_level_value_from_yaml(input_yaml, "output_path")
    if raw_output_dir is None:
        raise RuntimeError(f"output_path not found in {input_yaml}")
    return resolve_input_relative_path(input_yaml, raw_output_dir)


def load_probe_x_from_yaml(path: Path) -> float:
    raw = load_top_level_value_from_yaml(path, "probe_x")
    if raw is None:
        raise RuntimeError(f"probe_x not found in {path}")
    return float(raw)


def find_mfem_collection_pvd(output_dir: Path, input_yaml: Path) -> Path:
    collection_name = load_top_level_value_from_yaml(input_yaml, "collection_name")
    if collection_name:
        candidate = output_dir / collection_name / f"{collection_name}.pvd"
        if candidate.exists():
            return candidate

    pvd_files = sorted(output_dir.rglob("*.pvd"))
    if not pvd_files:
        raise FileNotFoundError(
            f"No MFEM ParaView .pvd collection found under {output_dir}"
        )
    return pvd_files[0]


def load_pvd_time_entries(pvd_path: Path) -> list[tuple[float, dict[str, Path]]]:
    try:
        root = ET.parse(pvd_path).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse ParaView collection {pvd_path}") from exc

    collection = root.find("Collection")
    if collection is None:
        raise RuntimeError(f"Missing Collection node in {pvd_path}")

    time_map: dict[float, dict[str, Path]] = {}
    for dataset in collection.findall("DataSet"):
        file_attr = dataset.attrib.get("file", "").strip()
        if not file_attr:
            continue

        time_value = float(dataset.attrib.get("timestep", "0"))
        name = dataset.attrib.get("name", "").strip()
        path = (pvd_path.parent / file_attr).resolve()
        entry = time_map.setdefault(time_value, {})
        if name:
            entry[name] = path
        if name == "mesh" or Path(file_attr).name == "data.pvtu":
            entry["mesh"] = path

    return [(time_value, time_map[time_value]) for time_value in sorted(time_map)]


def filter_available_time_entries(
    time_entries: list[tuple[float, dict[str, Path]]],
    required_keys: tuple[str, ...],
    context: str,
) -> list[tuple[float, dict[str, Path]]]:
    available: list[tuple[float, dict[str, Path]]] = []
    skipped = 0
    for time_value, entry in time_entries:
        if all(entry.get(key) is not None and entry[key].exists() for key in required_keys):
            available.append((time_value, entry))
        else:
            skipped += 1

    if not available:
        raise FileNotFoundError(
            f"No readable ParaView timesteps with {required_keys} were found in {context}"
        )

    if skipped:
        joined = ", ".join(required_keys)
        print(
            f"Skipping {skipped} ParaView time steps with missing {joined} files "
            f"while sampling from {context}.",
            file=sys.stderr,
        )

    return available


def iter_leaf_datasets(ds, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], object]]:
    if hasattr(ds, "n_blocks"):
        for i in range(ds.n_blocks):
            block = ds[i]
            if block is None:
                continue
            name = ""
            try:
                name = ds.get_block_name(i) or ""
            except Exception:
                name = ""
            yield from iter_leaf_datasets(block, path + (name,))
        return

    yield path, ds


def select_dataset_with_point_field(ds, field: str):
    matches = []
    for path, leaf in iter_leaf_datasets(ds):
        point_data = getattr(leaf, "point_data", None)
        if point_data is None or field not in point_data:
            continue
        matches.append((path, leaf))

    if not matches:
        return None

    for path, leaf in matches:
        if any(name == "mesh" for name in path):
            return leaf

    if len(matches) == 1:
        return matches[0][1]

    return pv.MultiBlock([leaf for _, leaf in matches]).combine()


def read_dataset_at_path(path: Path, field: str):
    ds = pv.read(str(path))
    selected = select_dataset_with_point_field(ds, field)
    if selected is None:
        raise RuntimeError(f"Field {field!r} not found in {path}")
    return selected


def select_snapshot(
    time_values: np.ndarray,
    *,
    time_index: int | None,
    time_value: float | None,
    label: str,
) -> SnapshotSelection:
    if (time_index is None) == (time_value is None):
        raise ValueError(f"Exactly one of time_index or time_value must be provided for {label}.")
    if time_values.size == 0:
        raise RuntimeError(f"No time values available for {label}.")

    if time_index is not None:
        if time_index < 0 or time_index >= int(time_values.size):
            raise IndexError(
                f"{label} time index {time_index} is out of range [0, {int(time_values.size) - 1}]."
            )
        idx = int(time_index)
    else:
        idx = int(np.argmin(np.abs(time_values - float(time_value))))

    return SnapshotSelection(index=idx, time=float(time_values[idx]))


def collapse_profile(y: np.ndarray, values: np.ndarray, tol: float = 1.0e-12) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if y.size != values.size:
        raise ValueError("Profile coordinate/value size mismatch.")
    if y.size == 0:
        return y, values

    order = np.argsort(y)
    y_sorted = y[order]
    values_sorted = values[order]

    y_out = [float(y_sorted[0])]
    v_out = [float(values_sorted[0])]
    counts = [1]

    for yi, vi in zip(y_sorted[1:], values_sorted[1:], strict=True):
        if abs(float(yi) - y_out[-1]) <= tol:
            v_out[-1] += float(vi)
            counts[-1] += 1
        else:
            y_out.append(float(yi))
            v_out.append(float(vi))
            counts.append(1)

    v_avg = np.array(
        [total / count for total, count in zip(v_out, counts, strict=True)],
        dtype=float,
    )
    return np.asarray(y_out, dtype=float), v_avg


def summarize_unique_coords(coords: np.ndarray, decimals: int = 12) -> np.ndarray:
    return np.unique(np.round(np.asarray(coords, dtype=float), decimals=decimals))


def format_values(values: np.ndarray, max_items: int = 8) -> str:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return "[]"
    shown = ", ".join(f"{val:.12g}" for val in values[:max_items])
    if values.size > max_items:
        shown += ", ..."
    return f"[{shown}]"


def load_mfem_density_profile(
    input_yaml: Path,
    output_dir: Path,
    *,
    time_index: int | None,
    time_value: float | None,
    probe_x: float,
    x_tol: float,
) -> tuple[SnapshotSelection, float, np.ndarray, np.ndarray]:
    pvd_path = find_mfem_collection_pvd(output_dir, input_yaml)
    time_entries = filter_available_time_entries(
        load_pvd_time_entries(pvd_path),
        ("rho_s_qp", "ale_displacement_qp"),
        str(pvd_path),
    )
    time_values = np.asarray([time_value for time_value, _ in time_entries], dtype=float)
    selection = select_snapshot(
        time_values, time_index=time_index, time_value=time_value, label="MFEM"
    )

    _, entry = time_entries[selection.index]
    qcloud = read_dataset_at_path(entry[TARGET_FIELD], TARGET_FIELD)
    disp_cloud = read_dataset_at_path(entry["ale_displacement_qp"], "ale_displacement_qp")

    points = np.asarray(qcloud.points, dtype=float)
    values = np.asarray(qcloud.point_data[TARGET_FIELD], dtype=float).reshape(-1)
    displacement = np.asarray(
        disp_cloud.point_data["ale_displacement_qp"], dtype=float
    )
    disp_points = np.asarray(disp_cloud.points, dtype=float)
    if points.ndim != 2 or points.shape[0] != values.size or points.shape[1] < 2:
        raise RuntimeError(
            f"Unexpected point/value shapes for {TARGET_FIELD}: points={points.shape}, "
            f"values={values.shape}"
        )
    if disp_points.shape != points.shape:
        raise RuntimeError(
            "rho_s_qp and ale_displacement_qp point shapes do not match: "
            f"{points.shape} vs {disp_points.shape}"
        )
    if not np.allclose(points, disp_points, atol=1.0e-12, rtol=0.0):
        raise RuntimeError(
            "rho_s_qp and ale_displacement_qp point coordinates do not align."
        )
    if displacement.ndim != 2 or displacement.shape[0] != points.shape[0]:
        raise RuntimeError(
            "Unexpected ale_displacement_qp array shape: "
            f"{displacement.shape}"
        )

    live_points = points.copy()
    ncomp = min(live_points.shape[1], displacement.shape[1])
    if ncomp <= 0:
        raise RuntimeError(
            "ale_displacement_qp has no spatial components to apply."
        )
    live_points[:, :ncomp] += displacement[:, :ncomp]

    close_mask = np.abs(live_points[:, 0] - probe_x) <= x_tol
    matched_x_candidates = summarize_unique_coords(live_points[close_mask, 0])
    if matched_x_candidates.size == 0:
        available_x = summarize_unique_coords(live_points[:, 0])
        raise RuntimeError(
            f"No {TARGET_FIELD} points matched probe_x={probe_x:.12g} within x_tol={x_tol:.12g}. "
            f"Available live x values: {format_values(available_x)}"
        )

    matched_x = float(matched_x_candidates[np.argmin(np.abs(matched_x_candidates - probe_x))])
    profile_mask = np.isclose(live_points[:, 0], matched_x, atol=x_tol, rtol=0.0)
    y_vals = live_points[profile_mask, 1]
    rho_vals = values[profile_mask]
    if y_vals.size == 0:
        raise RuntimeError(
            f"Matched x={matched_x:.12g} but found no profile points for {TARGET_FIELD}."
        )

    return selection, matched_x, *collapse_profile(y_vals, rho_vals)


def load_pato_time_values(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"PATO density reference not found: {path}")

    time_values: list[float] = []
    last_time: float | None = None
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"time_s", "x_m", "y_m", "rho_s_kg_m3"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"PATO density reference {path} is missing columns: {sorted(missing)}"
            )

        for row in reader:
            time_value = float(row["time_s"])
            if last_time is None or time_value != last_time:
                time_values.append(time_value)
                last_time = time_value

    if not time_values:
        raise RuntimeError(f"No time samples found in PATO density reference {path}")
    return np.asarray(time_values, dtype=float)


def load_pato_density_profile(
    path: Path,
    *,
    time_index: int | None,
    time_value: float | None,
    probe_x: float,
) -> tuple[SnapshotSelection, float, np.ndarray, np.ndarray]:
    time_values = load_pato_time_values(path)
    selection = select_snapshot(
        time_values, time_index=time_index, time_value=time_value, label="PATO"
    )

    rows: list[tuple[float, float, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_time = float(row["time_s"])
            if row_time < selection.time:
                continue
            if row_time > selection.time and rows:
                break
            if row_time != selection.time:
                continue

            rows.append(
                (
                    float(row["x_m"]),
                    float(row["y_m"]),
                    float(row["rho_s_kg_m3"]),
                )
            )

    if not rows:
        raise RuntimeError(
            f"No PATO density rows found at time {selection.time:.12g} in {path}"
        )

    x_values = summarize_unique_coords(np.array([row[0] for row in rows], dtype=float))
    matched_x = float(x_values[np.argmin(np.abs(x_values - probe_x))])
    selected_rows = [
        (y_val, rho_val)
        for x_val, y_val, rho_val in rows
        if abs(x_val - matched_x) <= 1.0e-12
    ]
    if not selected_rows:
        raise RuntimeError(
            f"No PATO density rows matched x={matched_x:.12g} at time {selection.time:.12g}."
        )

    filtered_rows = [
        (y_val, rho_val)
        for y_val, rho_val in selected_rows
        if abs(rho_val) > PATO_ZERO_TOL
    ]
    if not filtered_rows:
        raise RuntimeError(
            f"All PATO density rows at x={matched_x:.12g}, time={selection.time:.12g} "
            "were zero after filtering."
        )

    y_vals = np.array([row[0] for row in filtered_rows], dtype=float)
    rho_vals = np.array([row[1] for row in filtered_rows], dtype=float)
    return selection, matched_x, *collapse_profile(y_vals, rho_vals)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the MFEM rho_s_qp centerline density profile against the "
            "PATO density-history reference at a selected snapshot."
        )
    )
    parser.add_argument(
        "--input-yaml",
        type=Path,
        default=DEFAULT_INPUT_YAML,
        help="Input YAML used to resolve output_path, collection_name, and probe_x.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the ParaView output directory; defaults to output_path from the YAML.",
    )
    parser.add_argument(
        "--pato-density",
        type=Path,
        default=DEFAULT_PATO_DENSITY,
        help="PATO density-history CSV to compare against.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--time-index",
        type=int,
        default=None,
        help="Zero-based snapshot index to compare.",
    )
    group.add_argument(
        "--time",
        type=float,
        default=None,
        help="Target physical time (s); the nearest available snapshot is used.",
    )
    parser.add_argument(
        "--probe-x",
        type=float,
        default=None,
        help="Centerline x-location for the MFEM/PATO profile; defaults to probe_x in the YAML.",
    )
    parser.add_argument(
        "--x-tol",
        type=float,
        default=1.0e-8,
        help="Absolute tolerance used to select the MFEM centerline x column.",
    )
    parser.add_argument(
        "--out-prefix",
        default=DEFAULT_OUT_PREFIX,
        help="Prefix for the generated comparison CSV and plot.",
    )
    return parser


def write_comparison_csv(
    path: Path,
    pato_y: np.ndarray,
    pato_rho: np.ndarray,
    mfem_interp: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    abs_error = np.abs(mfem_interp - pato_rho)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "y_m",
                "pato_rho_s_kg_m3",
                "mfem_interp_rho_s_kg_m3",
                "abs_error_kg_m3",
            ]
        )
        for y_val, pato_val, mfem_val, err_val in zip(
            pato_y, pato_rho, mfem_interp, abs_error, strict=True
        ):
            writer.writerow(
                [
                    f"{float(y_val):.12g}",
                    f"{float(pato_val):.12g}",
                    f"{float(mfem_val):.12g}",
                    f"{float(err_val):.12g}",
                ]
            )


def save_profile_plot(
    path: Path,
    *,
    pato_y: np.ndarray,
    pato_rho: np.ndarray,
    mfem_y: np.ndarray,
    mfem_rho: np.ndarray,
    target_label: str,
    mfem_time: float,
    pato_time: float,
    probe_x: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    ax.plot(
        pato_rho,
        pato_y,
        color="tab:orange",
        lw=2.0,
        label="PATO",
    )
    ax.plot(
        mfem_rho,
        mfem_y,
        "--",
        color="tab:blue",
        lw=1.8,
        alpha=0.75,
        label="MFEM raw qpoints",
    )
    ax.set_xlabel("Solid Density (kg/m^3)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(200.0, 300.0)
    ax.set_ylim(0.035, 0.055)
    ax.set_title(
        f"Density profile at {target_label}\n"
        f"MFEM={mfem_time:.6g}s | PATO={pato_time:.6g}s | x={probe_x:.6g} m"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_yaml = args.input_yaml.resolve()
    if not input_yaml.exists():
        raise FileNotFoundError(f"Input YAML not found: {input_yaml}")

    output_dir = resolve_output_dir(input_yaml, args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    pato_density = args.pato_density.expanduser().resolve()
    probe_x = float(args.probe_x) if args.probe_x is not None else load_probe_x_from_yaml(input_yaml)

    mfem_selection, mfem_x, mfem_y, mfem_rho = load_mfem_density_profile(
        input_yaml,
        output_dir,
        time_index=args.time_index,
        time_value=args.time,
        probe_x=probe_x,
        x_tol=float(args.x_tol),
    )
    pato_selection, pato_x, pato_y, pato_rho = load_pato_density_profile(
        pato_density,
        time_index=args.time_index,
        time_value=args.time,
        probe_x=probe_x,
    )

    if mfem_y.size < 2:
        raise RuntimeError(
            f"MFEM density profile at x={mfem_x:.12g} has only {mfem_y.size} point(s); "
            "at least two are required for interpolation."
        )

    mfem_interp = np.interp(pato_y, mfem_y, mfem_rho)
    error = mfem_interp - pato_rho
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mean_abs = float(np.mean(np.abs(error)))
    max_abs = float(np.max(np.abs(error)))

    target_label = (
        f"time-index {args.time_index}"
        if args.time_index is not None
        else f"time {float(args.time):.6g}s"
    )
    stem = f"{args.out_prefix}_tidx{mfem_selection.index:05d}"
    csv_path = (output_dir / f"{stem}_density_profile_compare.csv").resolve()
    plot_path = (output_dir / f"{stem}_density_profile_compare.png").resolve()

    write_comparison_csv(csv_path, pato_y, pato_rho, mfem_interp)
    save_profile_plot(
        plot_path,
        pato_y=pato_y,
        pato_rho=pato_rho,
        mfem_y=mfem_y,
        mfem_rho=mfem_rho,
        target_label=target_label,
        mfem_time=mfem_selection.time,
        pato_time=pato_selection.time,
        probe_x=probe_x,
    )

    print("Density profile comparison")
    print(f"  target: {target_label}")
    print(
        f"  MFEM snapshot: index={mfem_selection.index} time={mfem_selection.time:.12g} s"
    )
    print(
        f"  PATO snapshot: index={pato_selection.index} time={pato_selection.time:.12g} s"
    )
    print(f"  probe_x target: {probe_x:.12g} m")
    print(f"  MFEM matched x: {mfem_x:.12g} m ({mfem_y.size} points)")
    print(f"  PATO matched x: {pato_x:.12g} m ({pato_y.size} points)")
    print(f"  RMSE: {rmse:.12g} kg/m^3")
    print(f"  Mean abs error: {mean_abs:.12g} kg/m^3")
    print(f"  Max abs error: {max_abs:.12g} kg/m^3")
    print(f"  Wrote CSV: {csv_path}")
    print(f"  Wrote plot: {plot_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"compare_ablation_case2_2_density.py: {exc}", file=sys.stderr)
        raise SystemExit(1)
