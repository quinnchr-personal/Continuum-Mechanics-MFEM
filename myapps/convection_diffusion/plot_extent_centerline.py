#!/usr/bin/env python3
"""Plot MFEM centerline reaction-extent profiles from one ParaView snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

import plot_rho_s_centerline as common
import matplotlib.pyplot as plt


DEFAULT_INPUT_YAML = common.DEFAULT_INPUT_YAML
DEFAULT_OUT_PREFIX = "extent_centerline"
DEFAULT_MESH_NSAMPLES = common.DEFAULT_MESH_NSAMPLES
MESH_DISPLACEMENT_FIELD = "ale_displacement"
QPOINT_DISPLACEMENT_FIELD = "ale_displacement_qp"
MESH_FIELD_RE = re.compile(r"X(\d+)$")
QPOINT_FIELD_RE = re.compile(r"X(\d+)_qp$")
PLOT_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


@dataclass(frozen=True)
class ReactionProfile:
    reaction_id: int
    mesh_field_name: str
    qpoint_field_name: str
    mesh_y: np.ndarray
    mesh_extent: np.ndarray
    qpoint_y: np.ndarray
    qpoint_extent: np.ndarray
    pato_y: np.ndarray | None = None
    pato_extent: np.ndarray | None = None
    pato_column_name: str | None = None
    pato_time: float | None = None
    pato_matched_x: float | None = None


@dataclass(frozen=True)
class PatoSnapshot:
    time: float
    matched_x: float
    profiles_by_reaction: dict[int, tuple[str, np.ndarray, np.ndarray]]


def detect_available_reaction_ids(entry: dict[str, Path]) -> list[int]:
    mesh = common.read_dataset_at_path(entry["mesh"], MESH_DISPLACEMENT_FIELD)
    mesh_ids = {
        int(match.group(1))
        for field_name in mesh.point_data.keys()
        if (match := MESH_FIELD_RE.fullmatch(field_name))
    }
    qpoint_ids = {
        int(match.group(1))
        for field_name, path in entry.items()
        if path is not None
        and path.exists()
        and (match := QPOINT_FIELD_RE.fullmatch(field_name))
    }
    available = sorted(mesh_ids & qpoint_ids)
    if not available:
        raise RuntimeError(
            "No matching reaction extent fields Xk/Xk_qp were found in the selected snapshot."
        )
    return available


def resolve_requested_reaction_ids(
    requested_ids: list[int] | None, available_ids: list[int]
) -> list[int]:
    if not requested_ids:
        return available_ids

    unique_ids: list[int] = []
    seen: set[int] = set()
    for reaction_id in requested_ids:
        if reaction_id <= 0:
            raise ValueError("Reaction IDs must be positive integers.")
        if reaction_id in seen:
            continue
        unique_ids.append(reaction_id)
        seen.add(reaction_id)

    missing = [reaction_id for reaction_id in unique_ids if reaction_id not in available_ids]
    if missing:
        available = ", ".join(f"X{reaction_id}" for reaction_id in available_ids)
        requested = ", ".join(f"X{reaction_id}" for reaction_id in missing)
        raise RuntimeError(
            f"Requested reaction fields {requested} are unavailable. Available fields: {available}"
        )
    return unique_ids


def infer_pato_reaction_columns(fieldnames: list[str]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for field_name in fieldnames:
        match = re.fullmatch(r"Xsi(?:_\d+)?_(\d+)", field_name)
        if match is None:
            continue
        mapping[int(match.group(1))] = field_name
    return mapping


def load_pato_snapshot(
    path: Path,
    *,
    target_time: float,
    target_x: float,
    reaction_ids: list[int],
) -> PatoSnapshot:
    if not path.exists():
        raise FileNotFoundError(f"PATO extent history CSV not found: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise RuntimeError(f"PATO extent history CSV has no header: {path}")

        required = {"time_s", "x_m", "y_m"}
        missing_required = sorted(required - set(fieldnames))
        if missing_required:
            joined = ", ".join(missing_required)
            raise RuntimeError(
                f"PATO extent history CSV is missing required columns {joined}: {path}"
            )

        reaction_column_map = infer_pato_reaction_columns(fieldnames)
        missing_reactions = [
            reaction_id
            for reaction_id in reaction_ids
            if reaction_id not in reaction_column_map
        ]
        if missing_reactions:
            requested = ", ".join(f"X{reaction_id}" for reaction_id in missing_reactions)
            available = ", ".join(
                f"X{reaction_id}->{column}"
                for reaction_id, column in sorted(reaction_column_map.items())
            )
            raise RuntimeError(
                f"PATO extent history CSV is missing columns for {requested}. "
                f"Available reaction columns: {available}"
            )

        rows: list[dict[str, str]] = list(reader)

    if not rows:
        raise RuntimeError(f"PATO extent history CSV has no data rows: {path}")

    time_values = np.array([float(row["time_s"]) for row in rows], dtype=float)
    unique_times = np.unique(time_values)
    matched_time = float(unique_times[np.argmin(np.abs(unique_times - target_time))])
    rows_at_time = [row for row in rows if float(row["time_s"]) == matched_time]
    if not rows_at_time:
        raise RuntimeError(
            f"PATO extent history CSV had no rows at matched time {matched_time:.16g}: {path}"
        )

    x_values = np.array([float(row["x_m"]) for row in rows_at_time], dtype=float)
    unique_x = np.unique(np.round(x_values, decimals=12))
    matched_x = float(unique_x[np.argmin(np.abs(unique_x - target_x))])
    x_tol = max(1.0e-10, 1.0e-8 * max(1.0, abs(matched_x)))
    rows_at_x = [
        row for row in rows_at_time if abs(float(row["x_m"]) - matched_x) <= x_tol
    ]
    if not rows_at_x:
        raise RuntimeError(
            f"PATO extent history CSV had no rows at matched x {matched_x:.16g}: {path}"
        )

    y_vals = np.array([float(row["y_m"]) for row in rows_at_x], dtype=float)
    order = np.argsort(y_vals)
    y_sorted = y_vals[order]

    profiles_by_reaction: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    for reaction_id in reaction_ids:
        column_name = reaction_column_map[reaction_id]
        extent_vals = np.array([float(row[column_name]) for row in rows_at_x], dtype=float)
        profiles_by_reaction[reaction_id] = (
            column_name,
            y_sorted,
            extent_vals[order],
        )

    return PatoSnapshot(
        time=matched_time,
        matched_x=matched_x,
        profiles_by_reaction=profiles_by_reaction,
    )


def load_live_qpoint_cloud(
    entry: dict[str, Path], qpoint_field_name: str
) -> tuple[np.ndarray, np.ndarray]:
    qcloud = common.read_dataset_at_path(entry[qpoint_field_name], qpoint_field_name)
    disp_cloud = common.read_dataset_at_path(
        entry[QPOINT_DISPLACEMENT_FIELD], QPOINT_DISPLACEMENT_FIELD
    )

    points = np.asarray(qcloud.points, dtype=float)
    values = np.asarray(qcloud.point_data[qpoint_field_name], dtype=float).reshape(-1)
    disp_points = np.asarray(disp_cloud.points, dtype=float)
    displacement = np.asarray(
        disp_cloud.point_data[QPOINT_DISPLACEMENT_FIELD], dtype=float
    )

    if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] != values.size:
        raise RuntimeError(
            f"Unexpected point/value shapes for {qpoint_field_name}: "
            f"points={points.shape}, values={values.shape}"
        )
    if disp_points.shape != points.shape:
        raise RuntimeError(
            f"{qpoint_field_name} and {QPOINT_DISPLACEMENT_FIELD} point shapes do not match: "
            f"{points.shape} vs {disp_points.shape}"
        )
    if not np.allclose(points, disp_points, atol=1.0e-12, rtol=0.0):
        raise RuntimeError(
            f"{qpoint_field_name} and {QPOINT_DISPLACEMENT_FIELD} point coordinates do not align."
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
    return live_points, values


def load_qpoint_profile(
    entry: dict[str, Path],
    *,
    qpoint_field_name: str,
    probe_x: float | None = None,
    matched_x: float | None = None,
    x_tol: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    live_points, values = load_live_qpoint_cloud(entry, qpoint_field_name)

    if matched_x is None:
        if probe_x is None:
            raise ValueError("probe_x is required when matched_x is not provided.")
        close_mask = np.abs(live_points[:, 0] - probe_x) <= x_tol
        matched_x_candidates = common.summarize_unique_coords(live_points[close_mask, 0])
        if matched_x_candidates.size == 0:
            available_x = common.summarize_unique_coords(live_points[:, 0])
            raise RuntimeError(
                f"No {qpoint_field_name} points matched probe_x={probe_x:.12g} "
                f"within x_tol={x_tol:.12g}. Available live x values: "
                f"{common.format_values(available_x)}"
            )
        matched_x = float(
            matched_x_candidates[np.argmin(np.abs(matched_x_candidates - probe_x))]
        )

    profile_mask = np.isclose(live_points[:, 0], matched_x, atol=x_tol, rtol=0.0)
    y_vals = live_points[profile_mask, 1]
    extent_vals = values[profile_mask]
    if y_vals.size == 0:
        raise RuntimeError(
            f"Matched x={matched_x:.12g} but found no profile points for {qpoint_field_name}."
        )

    y_profile, extent_profile = common.collapse_profile(y_vals, extent_vals)
    return float(matched_x), y_profile, extent_profile


def load_mesh_profile(
    entry: dict[str, Path],
    *,
    mesh_field_name: str,
    x_value: float,
    nsamples: int,
) -> tuple[np.ndarray, np.ndarray]:
    mesh = common.read_dataset_at_path(
        entry["mesh"], mesh_field_name, MESH_DISPLACEMENT_FIELD
    )
    mesh = common.warp_dataset_points_by_vector_field(mesh, MESH_DISPLACEMENT_FIELD)

    _, _, y_min, y_max, _, _ = mesh.bounds
    ys = np.linspace(float(y_min), float(y_max), nsamples, dtype=float)
    points = np.column_stack([np.full_like(ys, x_value), ys, np.zeros_like(ys)])
    sampled = pv.PolyData(points).sample(mesh)
    mask = np.asarray(
        sampled.point_data.get("vtkValidPointMask", np.ones(ys.size)),
        dtype=int,
    ).reshape(-1)
    values = np.asarray(sampled.point_data.get(mesh_field_name, []), dtype=float).reshape(-1)
    if values.size != ys.size:
        raise RuntimeError(
            f"Unexpected sampled {mesh_field_name} size at x={x_value:.12g}: "
            f"expected {ys.size}, got {values.size}"
        )

    valid = mask > 0
    if not np.any(valid):
        raise RuntimeError(
            f"No valid sampled {mesh_field_name} points were found along x={x_value:.12g}."
        )

    y_profile, extent_profile = common.collapse_profile(ys[valid], values[valid])
    return y_profile, extent_profile


def determine_target_probe_x(
    cli_probe_x: float | None,
    yaml_probe_x: float | None,
    entry: dict[str, Path],
    reference_field_name: str,
) -> float:
    if cli_probe_x is not None:
        return float(cli_probe_x)
    if yaml_probe_x is not None:
        return float(yaml_probe_x)

    mesh = common.read_dataset_at_path(
        entry["mesh"], reference_field_name, MESH_DISPLACEMENT_FIELD
    )
    mesh = common.warp_dataset_points_by_vector_field(mesh, MESH_DISPLACEMENT_FIELD)
    x_min, x_max, _, _, _, _ = mesh.bounds
    return 0.5 * (float(x_min) + float(x_max))


def write_profile_csv(
    path: Path,
    *,
    selection: common.SnapshotSelection,
    target_x: float,
    matched_x: float,
    profiles: list[ReactionProfile],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "reaction_id",
                "profile_kind",
                "field_name",
                "time_s",
                "time_index",
                "target_x_m",
                "matched_x_m",
                "position_y_m",
                "extent_value",
            ],
        )
        writer.writeheader()

        for profile in profiles:
            for y_val, extent_val in zip(profile.mesh_y, profile.mesh_extent, strict=True):
                writer.writerow(
                    {
                        "reaction_id": profile.reaction_id,
                        "profile_kind": "mesh",
                        "field_name": profile.mesh_field_name,
                        "time_s": f"{selection.time:.16g}",
                        "time_index": selection.index,
                        "target_x_m": f"{target_x:.16g}",
                        "matched_x_m": f"{matched_x:.16g}",
                        "position_y_m": f"{float(y_val):.16g}",
                        "extent_value": f"{float(extent_val):.16g}",
                    }
                )

            for y_val, extent_val in zip(
                profile.qpoint_y, profile.qpoint_extent, strict=True
            ):
                writer.writerow(
                    {
                        "reaction_id": profile.reaction_id,
                        "profile_kind": "qpoint",
                        "field_name": profile.qpoint_field_name,
                        "time_s": f"{selection.time:.16g}",
                        "time_index": selection.index,
                        "target_x_m": f"{target_x:.16g}",
                        "matched_x_m": f"{matched_x:.16g}",
                        "position_y_m": f"{float(y_val):.16g}",
                        "extent_value": f"{float(extent_val):.16g}",
                    }
                )

            if profile.pato_y is not None and profile.pato_extent is not None:
                for y_val, extent_val in zip(
                    profile.pato_y, profile.pato_extent, strict=True
                ):
                    writer.writerow(
                        {
                            "reaction_id": profile.reaction_id,
                            "profile_kind": "pato",
                            "field_name": profile.pato_column_name,
                            "time_s": f"{float(profile.pato_time):.16g}",
                            "time_index": selection.index,
                            "target_x_m": f"{target_x:.16g}",
                            "matched_x_m": f"{float(profile.pato_matched_x):.16g}",
                            "position_y_m": f"{float(y_val):.16g}",
                            "extent_value": f"{float(extent_val):.16g}",
                        }
                    )


def save_overlay_plot(
    path: Path,
    *,
    selection: common.SnapshotSelection,
    target_x: float,
    matched_x: float,
    profiles: list[ReactionProfile],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.5), constrained_layout=True)
    has_pato = any(profile.pato_y is not None for profile in profiles)
    for idx, profile in enumerate(profiles):
        color = PLOT_COLORS[idx % len(PLOT_COLORS)]
        ax.plot(
            profile.mesh_y,
            profile.mesh_extent,
            linewidth=2.0,
            color=color,
            label=profile.mesh_field_name,
        )
        ax.plot(
            profile.qpoint_y,
            profile.qpoint_extent,
            linestyle="none",
            marker="o",
            markersize=4.5,
            markerfacecolor="none",
            markeredgewidth=1.25,
            color=color,
            label=profile.qpoint_field_name,
        )
        if profile.pato_y is not None and profile.pato_extent is not None:
            ax.plot(
                profile.pato_y,
                profile.pato_extent,
                linestyle="--",
                linewidth=1.6,
                marker="s",
                markersize=3.8,
                color=color,
                alpha=0.9,
                label=f"PATO {profile.pato_column_name}",
            )

    ax.set_xlabel("y (m)")
    ax.set_ylabel("Reaction extent (-)")
    ax.set_title(
        "MFEM centerline reaction extent\n"
        f"t={selection.time:.6g} s, tidx={selection.index}, "
        f"target x={target_x:.6g} m, matched x={matched_x:.6g} m"
    )
    if has_pato:
        ax.set_title(ax.get_title() + "\nwith PATO overlay")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_overlay_html(
    path: Path,
    *,
    selection: common.SnapshotSelection,
    target_x: float,
    matched_x: float,
    profiles: list[ReactionProfile],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    traces: list[dict[str, object]] = []
    has_pato = any(profile.pato_y is not None for profile in profiles)
    for idx, profile in enumerate(profiles):
        color = PLOT_COLORS[idx % len(PLOT_COLORS)]
        traces.append(
            {
                "x": np.asarray(profile.mesh_y, dtype=float).tolist(),
                "y": np.asarray(profile.mesh_extent, dtype=float).tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": profile.mesh_field_name,
                "line": {"width": 2, "color": color},
                "hovertemplate": (
                    f"y=%{{x:.6g}} m<br>{profile.mesh_field_name}=%{{y:.6g}}"
                    f"<extra>{profile.mesh_field_name}</extra>"
                ),
            }
        )
        traces.append(
            {
                "x": np.asarray(profile.qpoint_y, dtype=float).tolist(),
                "y": np.asarray(profile.qpoint_extent, dtype=float).tolist(),
                "type": "scatter",
                "mode": "markers",
                "name": profile.qpoint_field_name,
                "marker": {"size": 7, "color": color, "symbol": "circle-open"},
                "hovertemplate": (
                    f"y=%{{x:.6g}} m<br>{profile.qpoint_field_name}=%{{y:.6g}}"
                    f"<extra>{profile.qpoint_field_name}</extra>"
                ),
            }
        )
        if profile.pato_y is not None and profile.pato_extent is not None:
            traces.append(
                {
                    "x": np.asarray(profile.pato_y, dtype=float).tolist(),
                    "y": np.asarray(profile.pato_extent, dtype=float).tolist(),
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": f"PATO {profile.pato_column_name}",
                    "line": {"width": 1.6, "color": color, "dash": "dash"},
                    "marker": {"size": 6, "color": color, "symbol": "square"},
                    "hovertemplate": (
                        f"y=%{{x:.6g}} m<br>PATO {profile.pato_column_name}=%{{y:.6g}}"
                        f"<extra>PATO {profile.pato_column_name}</extra>"
                    ),
                }
            )

    title = (
        "MFEM centerline reaction extent<br>"
        f"t={selection.time:.6g} s, tidx={selection.index}, "
        f"target x={target_x:.6g} m, matched x={matched_x:.6g} m"
    )
    if has_pato:
        title += "<br>with PATO overlay"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MFEM centerline reaction extent</title>
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
    const traces = {json.dumps(traces)};
    const layout = {{
      template: "plotly_white",
      title: {json.dumps(title)},
      xaxis: {{ title: "y (m)" }},
      yaxis: {{ title: "Reaction extent (-)" }},
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
            "Plot MFEM centerline reaction-extent mesh profiles and quadrature-point "
            "profiles along one live centerline at a selected snapshot."
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
        "--reaction-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional 1-based reaction IDs to plot; defaults to all available Xk fields.",
    )
    parser.add_argument(
        "--pato-csv",
        type=Path,
        default=None,
        help=(
            "Optional PATO extent-history CSV to overlay. The nearest available "
            "PATO snapshot time and x-column are used."
        ),
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
            "Number of points used to sample each extent mesh profile along the "
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
    output_dir = common.resolve_output_dir(input_yaml, args.output_dir)
    pvd_path = common.find_mfem_collection_pvd(output_dir, input_yaml)
    time_entries = common.filter_available_time_entries(
        common.load_pvd_time_entries(pvd_path),
        ("mesh", QPOINT_DISPLACEMENT_FIELD),
        str(pvd_path),
    )
    time_values = np.asarray([time_value for time_value, _ in time_entries], dtype=float)
    selection = common.select_snapshot(
        time_values,
        time_index=args.time_index,
        time_value=args.time,
        label="MFEM",
    )
    _, entry = time_entries[selection.index]

    available_reaction_ids = detect_available_reaction_ids(entry)
    reaction_ids = resolve_requested_reaction_ids(args.reaction_ids, available_reaction_ids)
    reference_field_name = f"X{reaction_ids[0]}"

    yaml_probe_x = common.load_optional_probe_x_from_yaml(input_yaml)
    target_x = determine_target_probe_x(
        args.probe_x, yaml_probe_x, entry, reference_field_name
    )

    pato_snapshot: PatoSnapshot | None = None
    if args.pato_csv is not None:
        pato_snapshot = load_pato_snapshot(
            args.pato_csv.expanduser(),
            target_time=selection.time,
            target_x=target_x,
            reaction_ids=reaction_ids,
        )

    profiles: list[ReactionProfile] = []
    matched_x: float | None = None
    for reaction_id in reaction_ids:
        mesh_field_name = f"X{reaction_id}"
        qpoint_field_name = f"{mesh_field_name}_qp"
        matched_x, qpoint_y, qpoint_extent = load_qpoint_profile(
            entry,
            qpoint_field_name=qpoint_field_name,
            probe_x=target_x if matched_x is None else None,
            matched_x=matched_x,
            x_tol=float(args.x_tol),
        )
        mesh_y, mesh_extent = load_mesh_profile(
            entry,
            mesh_field_name=mesh_field_name,
            x_value=matched_x,
            nsamples=int(args.nsamples),
        )
        profiles.append(
            ReactionProfile(
                reaction_id=reaction_id,
                mesh_field_name=mesh_field_name,
                qpoint_field_name=qpoint_field_name,
                mesh_y=mesh_y,
                mesh_extent=mesh_extent,
                qpoint_y=qpoint_y,
                qpoint_extent=qpoint_extent,
                pato_y=(
                    pato_snapshot.profiles_by_reaction[reaction_id][1]
                    if pato_snapshot is not None
                    else None
                ),
                pato_extent=(
                    pato_snapshot.profiles_by_reaction[reaction_id][2]
                    if pato_snapshot is not None
                    else None
                ),
                pato_column_name=(
                    pato_snapshot.profiles_by_reaction[reaction_id][0]
                    if pato_snapshot is not None
                    else None
                ),
                pato_time=(pato_snapshot.time if pato_snapshot is not None else None),
                pato_matched_x=(
                    pato_snapshot.matched_x if pato_snapshot is not None else None
                ),
            )
        )

    reaction_tag = (
        "all"
        if reaction_ids == available_reaction_ids
        else "_".join(f"X{reaction_id}" for reaction_id in reaction_ids)
    )
    stem = f"{args.out_prefix}_{reaction_tag}_tidx{selection.index:05d}_extent_centerline"
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"
    html_path = output_dir / f"{stem}.html"

    write_profile_csv(
        csv_path,
        selection=selection,
        target_x=target_x,
        matched_x=matched_x,
        profiles=profiles,
    )
    save_overlay_plot(
        png_path,
        selection=selection,
        target_x=target_x,
        matched_x=matched_x,
        profiles=profiles,
    )
    if args.html:
        save_overlay_html(
            html_path,
            selection=selection,
            target_x=target_x,
            matched_x=matched_x,
            profiles=profiles,
        )

    reactions = ", ".join(f"X{reaction_id}" for reaction_id in reaction_ids)
    print(f"Selected snapshot: tidx={selection.index}, time={selection.time:.16g} s")
    print(f"Reaction fields: {reactions}")
    print(f"Target x: {target_x:.16g} m")
    print(f"Matched qpoint x: {matched_x:.16g} m")
    if pato_snapshot is not None:
        print(
            "Matched PATO snapshot: "
            f"time={pato_snapshot.time:.16g} s, x={pato_snapshot.matched_x:.16g} m"
        )
    print(f"Mesh sample points per reaction: {profiles[0].mesh_y.size}")
    print(f"Quadrature points per reaction: {profiles[0].qpoint_y.size}")
    print(f"Wrote plot: {png_path}")
    if args.html:
        print(f"Wrote interactive HTML: {html_path}")
    print(f"Wrote CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
