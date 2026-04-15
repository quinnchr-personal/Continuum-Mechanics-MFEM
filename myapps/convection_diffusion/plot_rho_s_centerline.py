#!/usr/bin/env python3
"""Plot MFEM centerline rho_s and rho_s_qp profiles from one ParaView snapshot."""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_OUT_PREFIX = "rho_s_centerline"
DEFAULT_MESH_NSAMPLES = 100000
MESH_FIELD = "rho_s"
MESH_DISPLACEMENT_FIELD = "ale_displacement"
QPOINT_FIELD = "rho_s_qp"
QPOINT_DISPLACEMENT_FIELD = "ale_displacement_qp"


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


def load_optional_probe_x_from_yaml(path: Path) -> float | None:
    raw = load_top_level_value_from_yaml(path, "probe_x")
    return None if raw is None else float(raw)


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


def iter_leaf_datasets(
    ds, path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], object]]:
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


def select_dataset_with_point_fields(ds, *fields: str):
    matches = []
    for path, leaf in iter_leaf_datasets(ds):
        point_data = getattr(leaf, "point_data", None)
        if point_data is None or not all(field in point_data for field in fields):
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


def read_dataset_at_path(path: Path, *fields: str):
    ds = pv.read(str(path))
    selected = select_dataset_with_point_fields(ds, *fields)
    if selected is None:
        joined = ", ".join(repr(field) for field in fields)
        raise RuntimeError(f"Fields {joined} not found in {path}")
    return selected


def warp_dataset_points_by_vector_field(ds, field_name: str):
    point_data = getattr(ds, "point_data", None)
    if point_data is None or field_name not in point_data:
        raise RuntimeError(f"Field {field_name!r} not found for ALE warp.")

    warped = ds.copy(deep=True)
    disp = np.asarray(warped.point_data[field_name], dtype=float)
    pts = np.asarray(warped.points, dtype=float).copy()
    if disp.ndim != 2 or pts.ndim != 2:
        raise RuntimeError(
            f"Unexpected point/displacement shapes for {field_name}: "
            f"points={pts.shape}, displacement={disp.shape}"
        )

    ncomp = min(pts.shape[1], disp.shape[1])
    if ncomp <= 0:
        raise RuntimeError(f"{field_name!r} has no spatial components to apply.")

    pts[:, :ncomp] += disp[:, :ncomp]
    warped.points = pts
    return warped


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


def collapse_profile(
    y: np.ndarray, values: np.ndarray, tol: float = 1.0e-12
) -> tuple[np.ndarray, np.ndarray]:
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


def load_qpoint_profile(
    entry: dict[str, Path],
    *,
    probe_x: float,
    x_tol: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    qcloud = read_dataset_at_path(entry[QPOINT_FIELD], QPOINT_FIELD)
    disp_cloud = read_dataset_at_path(entry[QPOINT_DISPLACEMENT_FIELD], QPOINT_DISPLACEMENT_FIELD)

    points = np.asarray(qcloud.points, dtype=float)
    values = np.asarray(qcloud.point_data[QPOINT_FIELD], dtype=float).reshape(-1)
    disp_points = np.asarray(disp_cloud.points, dtype=float)
    displacement = np.asarray(
        disp_cloud.point_data[QPOINT_DISPLACEMENT_FIELD], dtype=float
    )

    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] != values.size:
        raise RuntimeError(
            f"Unexpected point/value shapes for {QPOINT_FIELD}: "
            f"points={points.shape}, values={values.shape}"
        )
    if disp_points.shape != points.shape:
        raise RuntimeError(
            f"{QPOINT_FIELD} and {QPOINT_DISPLACEMENT_FIELD} point shapes do not match: "
            f"{points.shape} vs {disp_points.shape}"
        )
    if not np.allclose(points, disp_points, atol=1.0e-12, rtol=0.0):
        raise RuntimeError(
            f"{QPOINT_FIELD} and {QPOINT_DISPLACEMENT_FIELD} point coordinates do not align."
        )
    if displacement.ndim != 2 or displacement.shape[0] != points.shape[0]:
        raise RuntimeError(
            f"Unexpected {QPOINT_DISPLACEMENT_FIELD} array shape: {displacement.shape}"
        )

    live_points = points.copy()
    ncomp = min(live_points.shape[1], displacement.shape[1])
    if ncomp <= 0:
        raise RuntimeError(f"{QPOINT_DISPLACEMENT_FIELD} has no spatial components to apply.")
    live_points[:, :ncomp] += displacement[:, :ncomp]

    close_mask = np.abs(live_points[:, 0] - probe_x) <= x_tol
    matched_x_candidates = summarize_unique_coords(live_points[close_mask, 0])
    if matched_x_candidates.size == 0:
        available_x = summarize_unique_coords(live_points[:, 0])
        raise RuntimeError(
            f"No {QPOINT_FIELD} points matched probe_x={probe_x:.12g} within x_tol={x_tol:.12g}. "
            f"Available live x values: {format_values(available_x)}"
        )

    matched_x = float(
        matched_x_candidates[np.argmin(np.abs(matched_x_candidates - probe_x))]
    )
    profile_mask = np.isclose(live_points[:, 0], matched_x, atol=x_tol, rtol=0.0)
    y_vals = live_points[profile_mask, 1]
    rho_vals = values[profile_mask]
    if y_vals.size == 0:
        raise RuntimeError(
            f"Matched x={matched_x:.12g} but found no profile points for {QPOINT_FIELD}."
        )

    y_profile, rho_profile = collapse_profile(y_vals, rho_vals)
    return matched_x, y_profile, rho_profile


def load_mesh_profile(
    entry: dict[str, Path],
    *,
    x_value: float,
    nsamples: int,
) -> tuple[np.ndarray, np.ndarray]:
    mesh = read_dataset_at_path(entry["mesh"], MESH_FIELD, MESH_DISPLACEMENT_FIELD)
    mesh = warp_dataset_points_by_vector_field(mesh, MESH_DISPLACEMENT_FIELD)

    _, _, y_min, y_max, _, _ = mesh.bounds
    ys = np.linspace(float(y_min), float(y_max), nsamples, dtype=float)
    points = np.column_stack(
        [np.full_like(ys, x_value), ys, np.zeros_like(ys)]
    )
    sampled = pv.PolyData(points).sample(mesh)
    mask = np.asarray(
        sampled.point_data.get("vtkValidPointMask", np.ones(ys.size)),
        dtype=int,
    ).reshape(-1)
    values = np.asarray(sampled.point_data.get(MESH_FIELD, []), dtype=float).reshape(-1)
    if values.size != ys.size:
        raise RuntimeError(
            f"Unexpected sampled {MESH_FIELD} size at x={x_value:.12g}: "
            f"expected {ys.size}, got {values.size}"
        )

    valid = mask > 0
    if not np.any(valid):
        raise RuntimeError(
            f"No valid sampled {MESH_FIELD} points were found along x={x_value:.12g}."
        )

    y_profile, rho_profile = collapse_profile(ys[valid], values[valid])
    return y_profile, rho_profile


def determine_target_probe_x(
    cli_probe_x: float | None, yaml_probe_x: float | None, entry: dict[str, Path]
) -> float:
    if cli_probe_x is not None:
        return float(cli_probe_x)
    if yaml_probe_x is not None:
        return float(yaml_probe_x)

    mesh = read_dataset_at_path(entry["mesh"], MESH_FIELD, MESH_DISPLACEMENT_FIELD)
    mesh = warp_dataset_points_by_vector_field(mesh, MESH_DISPLACEMENT_FIELD)
    x_min, x_max, _, _, _, _ = mesh.bounds
    return 0.5 * (float(x_min) + float(x_max))


def write_profile_csv(
    path: Path,
    *,
    selection: SnapshotSelection,
    target_x: float,
    matched_x: float,
    mesh_y: np.ndarray,
    mesh_rho: np.ndarray,
    qpoint_y: np.ndarray,
    qpoint_rho: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "profile_kind",
                "time_s",
                "time_index",
                "target_x_m",
                "matched_x_m",
                "position_y_m",
                "rho_s_kg_m3",
            ],
        )
        writer.writeheader()

        for y_val, rho_val in zip(mesh_y, mesh_rho, strict=True):
            writer.writerow(
                {
                    "profile_kind": "rho_s_mesh",
                    "time_s": f"{selection.time:.16g}",
                    "time_index": selection.index,
                    "target_x_m": f"{target_x:.16g}",
                    "matched_x_m": f"{matched_x:.16g}",
                    "position_y_m": f"{float(y_val):.16g}",
                    "rho_s_kg_m3": f"{float(rho_val):.16g}",
                }
            )

        for y_val, rho_val in zip(qpoint_y, qpoint_rho, strict=True):
            writer.writerow(
                {
                    "profile_kind": "rho_s_qp",
                    "time_s": f"{selection.time:.16g}",
                    "time_index": selection.index,
                    "target_x_m": f"{target_x:.16g}",
                    "matched_x_m": f"{matched_x:.16g}",
                    "position_y_m": f"{float(y_val):.16g}",
                    "rho_s_kg_m3": f"{float(rho_val):.16g}",
                }
            )


def save_overlay_plot(
    path: Path,
    *,
    selection: SnapshotSelection,
    target_x: float,
    matched_x: float,
    mesh_y: np.ndarray,
    mesh_rho: np.ndarray,
    qpoint_y: np.ndarray,
    qpoint_rho: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.plot(mesh_y, mesh_rho, linewidth=2.0, label="rho_s")
    ax.plot(qpoint_y, qpoint_rho, linestyle="none", marker="o", markersize=4.5, label="rho_s_qp")
    ax.set_xlabel("y (m)")
    ax.set_ylabel(r"rho_s (kg/m$^3$)")
    ax.set_title(
        "MFEM centerline density\n"
        f"t={selection.time:.6g} s, tidx={selection.index}, "
        f"target x={target_x:.6g} m, matched x={matched_x:.6g} m"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_overlay_html(
    path: Path,
    *,
    selection: SnapshotSelection,
    target_x: float,
    matched_x: float,
    mesh_y: np.ndarray,
    mesh_rho: np.ndarray,
    qpoint_y: np.ndarray,
    qpoint_rho: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    title = (
        "MFEM centerline density<br>"
        f"t={selection.time:.6g} s, tidx={selection.index}, "
        f"target x={target_x:.6g} m, matched x={matched_x:.6g} m"
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MFEM centerline density</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: Helvetica, Arial, sans-serif;
      background: #f7f7f5;
      color: #202020;
    }}
    .page {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }}
    #plot {{
      width: 100%;
      height: 78vh;
      min-height: 560px;
      background: #ffffff;
      border: 1px solid #d8d8d3;
      border-radius: 10px;
    }}
    .note {{
      margin-top: 12px;
      font-size: 0.92rem;
      color: #555;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div id="plot"></div>
    <div class="note">
      This interactive plot uses Plotly loaded from a CDN when the HTML file is opened.
    </div>
  </div>
  <script>
    const traces = [
      {{
        x: {json.dumps(np.asarray(mesh_y, dtype=float).tolist())},
        y: {json.dumps(np.asarray(mesh_rho, dtype=float).tolist())},
        type: "scatter",
        mode: "lines",
        name: "rho_s",
        line: {{ width: 2, color: "#1f77b4" }},
        hovertemplate: "y=%{{x:.6g}} m<br>rho_s=%{{y:.6g}} kg/m^3<extra>rho_s</extra>"
      }},
      {{
        x: {json.dumps(np.asarray(qpoint_y, dtype=float).tolist())},
        y: {json.dumps(np.asarray(qpoint_rho, dtype=float).tolist())},
        type: "scatter",
        mode: "markers",
        name: "rho_s_qp",
        marker: {{ size: 7, color: "#d62728" }},
        hovertemplate: "y=%{{x:.6g}} m<br>rho_s_qp=%{{y:.6g}} kg/m^3<extra>rho_s_qp</extra>"
      }}
    ];
    const layout = {{
      template: "plotly_white",
      title: {json.dumps(title)},
      xaxis: {{ title: "y (m)" }},
      yaxis: {{ title: "rho_s (kg/m^3)" }},
      hovermode: "x unified",
      legend: {{ orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1.0 }},
      margin: {{ l: 80, r: 40, t: 90, b: 70 }}
    }};
    const config = {{
      responsive: true,
      displaylogo: false
    }};
    Plotly.newPlot("plot", traces, layout, config);
  </script>
</body>
</html>
"""
    path.write_text(html)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the MFEM rho_s mesh profile and rho_s_qp quadrature-point profile "
            "along one live centerline at a selected snapshot."
        )
    )
    parser.add_argument(
        "--input-yaml",
        type=Path,
        default=DEFAULT_INPUT_YAML,
        help="Input YAML used to resolve output_path, collection_name, and optional probe_x.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the ParaView output directory; defaults to output_path from the YAML.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--time-index",
        type=int,
        default=None,
        help="Zero-based snapshot index to plot.",
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
        help="Requested centerline x-location; defaults to probe_x in the YAML or the live mesh midpoint.",
    )
    parser.add_argument(
        "--x-tol",
        type=float,
        default=1.0e-8,
        help="Absolute tolerance used to match the qpoint centerline x column.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=DEFAULT_MESH_NSAMPLES,
        help=(
            "Number of points used to sample the rho_s mesh profile along the "
            f"matched centerline. Default: {DEFAULT_MESH_NSAMPLES}."
        ),
    )
    parser.add_argument(
        "--out-prefix",
        default=DEFAULT_OUT_PREFIX,
        help="Prefix for the generated centerline CSV and plot.",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also write an interactive Plotly HTML plot alongside the PNG and CSV.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.nsamples < 2:
        raise ValueError("--nsamples must be at least 2.")
    if args.x_tol <= 0.0:
        raise ValueError("--x-tol must be positive.")

    input_yaml = args.input_yaml.resolve()
    output_dir = resolve_output_dir(input_yaml, args.output_dir)
    pvd_path = find_mfem_collection_pvd(output_dir, input_yaml)
    time_entries = filter_available_time_entries(
        load_pvd_time_entries(pvd_path),
        ("mesh", QPOINT_FIELD, QPOINT_DISPLACEMENT_FIELD),
        str(pvd_path),
    )
    time_values = np.asarray([time_value for time_value, _ in time_entries], dtype=float)
    selection = select_snapshot(
        time_values,
        time_index=args.time_index,
        time_value=args.time,
        label="MFEM",
    )
    _, entry = time_entries[selection.index]

    yaml_probe_x = load_optional_probe_x_from_yaml(input_yaml)
    target_x = determine_target_probe_x(args.probe_x, yaml_probe_x, entry)
    matched_x, qpoint_y, qpoint_rho = load_qpoint_profile(
        entry,
        probe_x=target_x,
        x_tol=float(args.x_tol),
    )
    mesh_y, mesh_rho = load_mesh_profile(
        entry,
        x_value=matched_x,
        nsamples=int(args.nsamples),
    )

    stem = f"{args.out_prefix}_tidx{selection.index:05d}_rho_s_centerline"
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"
    html_path = output_dir / f"{stem}.html"
    write_profile_csv(
        csv_path,
        selection=selection,
        target_x=target_x,
        matched_x=matched_x,
        mesh_y=mesh_y,
        mesh_rho=mesh_rho,
        qpoint_y=qpoint_y,
        qpoint_rho=qpoint_rho,
    )
    save_overlay_plot(
        png_path,
        selection=selection,
        target_x=target_x,
        matched_x=matched_x,
        mesh_y=mesh_y,
        mesh_rho=mesh_rho,
        qpoint_y=qpoint_y,
        qpoint_rho=qpoint_rho,
    )
    if args.html:
        save_overlay_html(
            html_path,
            selection=selection,
            target_x=target_x,
            matched_x=matched_x,
            mesh_y=mesh_y,
            mesh_rho=mesh_rho,
            qpoint_y=qpoint_y,
            qpoint_rho=qpoint_rho,
        )

    print(f"Selected snapshot: tidx={selection.index}, time={selection.time:.16g} s")
    print(f"Target x: {target_x:.16g} m")
    print(f"Matched qpoint x: {matched_x:.16g} m")
    print(f"Mesh sample points: {mesh_y.size}")
    print(f"Quadrature points: {qpoint_y.size}")
    print(f"Wrote plot: {png_path}")
    if args.html:
        print(f"Wrote interactive HTML: {html_path}")
    print(f"Wrote CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
