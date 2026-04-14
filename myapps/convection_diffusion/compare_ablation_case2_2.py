#!/usr/bin/env python3
"""Compare MFEM ablation case-2.2 outputs against Amaryllis reference data."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TOL = {
    "temperature_rmse_max": 300.0,
    "temperature_max_abs_max": 650.0,
    "m_dot_g_rmse_max": 0.025,
    "m_dot_g_max_abs_max": 0.08,
    "m_dot_g_peak_rel_error_max": 0.5,
    "m_dot_g_peak_time_error_max": 10.0,
    "front98_max_abs_max": 0.01,
    "front98_rmse_max": 0.01,
    "front2_max_abs_max": 0.01,
    "front2_rmse_max": 0.01,
    "m_dot_c_rmse_max": 0.01,
    "m_dot_c_peak_rel_error_max": 0.35,
    "recession_rmse_max": 0.0015,
    "recession_final_rel_error_max": 0.12,
}


def load_acceptance_from_yaml(path: Path) -> Dict[str, float]:
    vals = dict(DEFAULT_TOL)
    if not path.exists():
        return vals

    in_acceptance = False
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "acceptance:":
            in_acceptance = True
            continue
        if in_acceptance and not line.startswith(" "):
            break
        if in_acceptance and ":" in stripped:
            k, v = stripped.split(":", 1)
            try:
                vals[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return vals


def load_probe_y_from_yaml(path: Path) -> List[float]:
    if not path.exists():
        return []

    probe_y: List[float] = []
    in_probe_y = False
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "probe_y:":
            in_probe_y = True
            continue
        if in_probe_y:
            if line.startswith("  -"):
                try:
                    probe_y.append(float(line.split("-", 1)[1].strip()))
                except ValueError:
                    pass
                continue
            if not line.startswith(" "):
                break

    return probe_y


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
        if stripped.startswith(key_prefix):
            return stripped.split(":", 1)[1].strip()
    return None


def load_probe_x_from_yaml(path: Path) -> float:
    raw = load_top_level_value_from_yaml(path, "probe_x")
    if raw is None:
        raise RuntimeError(f"probe_x not found in {path}")
    return float(raw)


def load_probe_depths_from_yaml(path: Path) -> List[float]:
    probe_y = load_probe_y_from_yaml(path)
    if not probe_y:
        return []
    y_wall = probe_y[0]
    return [abs(y_wall - y) for y in probe_y]


def load_named_csv(
    path: Path, *, required: bool, description: str
) -> np.ndarray | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{description} not found: {path}")
        return None

    if path.stat().st_size == 0:
        if required:
            raise RuntimeError(
                f"{description} is empty: {path}. This usually means the "
                "driver run did not finish cleanly or the output file came "
                "from an interrupted run."
            )
        return None

    data = np.genfromtxt(path, delimiter=",", names=True)
    if getattr(data, "dtype", None) is None or data.dtype.names is None:
        if required:
            raise RuntimeError(f"{description} has no readable header: {path}")
        return None

    data = np.atleast_1d(data)
    if data.size == 0:
        if required:
            raise RuntimeError(
                f"{description} has a header but no data rows: {path}"
            )
        return None

    return data


def find_mfem_collection_pvd(output_dir: Path, input_yaml: Path) -> Path | None:
    collection_name = load_top_level_value_from_yaml(input_yaml, "collection_name")
    if collection_name:
        candidate = output_dir / collection_name / f"{collection_name}.pvd"
        if candidate.exists():
            return candidate

    pvd_files = sorted(output_dir.rglob("*.pvd"))
    if not pvd_files:
        return None
    return pvd_files[0]


def iter_leaf_datasets(ds, path: Tuple[str, ...] = ()):
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
    candidates = []
    for path, leaf in iter_leaf_datasets(ds):
        point_data = getattr(leaf, "point_data", None)
        if point_data is None:
            continue
        if all(field in point_data for field in fields):
            candidates.append((path, leaf))

    if not candidates:
        return None

    for path, leaf in candidates:
        if any(name == "mesh" for name in path):
            return leaf

    if len(candidates) == 1:
        return candidates[0][1]

    try:
        import pyvista as pv

        return pv.MultiBlock([leaf for _, leaf in candidates]).combine()
    except Exception:
        return candidates[0][1]


def warp_dataset_points_by_vector_field(ds, field_name: str):
    point_data = getattr(ds, "point_data", None)
    if point_data is None or field_name not in point_data:
        return ds

    warped = ds.copy(deep=True)
    disp = np.asarray(warped.point_data[field_name], dtype=float)
    pts = np.asarray(warped.points, dtype=float).copy()
    if disp.ndim != 2 or pts.ndim != 2:
        return warped

    ncomp = min(pts.shape[1], disp.shape[1])
    if ncomp <= 0:
        return warped

    pts[:, :ncomp] += disp[:, :ncomp]
    warped.points = pts
    return warped


def resolve_input_relative_path(input_yaml: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (input_yaml.parent / path).resolve()


def parse_inline_yaml_list(raw: str) -> List[float]:
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    vals: List[float] = []
    for tok in text.split(","):
        s = tok.strip()
        if not s:
            continue
        vals.append(float(s))
    return vals


def load_material_density_limits(input_yaml: Path) -> Tuple[float, float]:
    raw_material_path = load_top_level_value_from_yaml(input_yaml, "material_file")
    if raw_material_path is None:
        raise RuntimeError(f"material_file not found in {input_yaml}")

    material_path = resolve_input_relative_path(input_yaml, raw_material_path)
    if not material_path.exists():
        raise FileNotFoundError(f"Material YAML not found: {material_path}")

    rhoI: List[float] = []
    epsI: List[float] = []
    reactions: List[Dict[str, float]] = []
    current_reaction: Dict[str, float] | None = None
    in_reactions = False

    for raw in material_path.read_text().splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("rhoI:"):
            rhoI = parse_inline_yaml_list(stripped.split(":", 1)[1])
            continue
        if stripped.startswith("epsI:"):
            epsI = parse_inline_yaml_list(stripped.split(":", 1)[1])
            continue

        if stripped == "reactions:":
            if current_reaction is not None:
                reactions.append(current_reaction)
                current_reaction = None
            in_reactions = True
            continue

        if not line.startswith(" "):
            if in_reactions and current_reaction is not None:
                reactions.append(current_reaction)
                current_reaction = None
            in_reactions = False
            continue

        if not in_reactions:
            continue

        if stripped.startswith("- "):
            if current_reaction is not None:
                reactions.append(current_reaction)
            current_reaction = {"F": 0.0, "phase_index": 1.0}
            stripped = stripped[2:].strip()
            if not stripped:
                continue

        if current_reaction is None or ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "F":
            current_reaction["F"] = float(value)
        elif key == "phase_index":
            current_reaction["phase_index"] = float(value)

    if in_reactions and current_reaction is not None:
        reactions.append(current_reaction)

    if not rhoI or not epsI:
        raise RuntimeError(
            f"Could not parse rhoI/epsI from material file {material_path}"
        )
    if len(rhoI) != len(epsI):
        raise RuntimeError(
            f"rhoI/epsI length mismatch in material file {material_path}"
        )

    rho_eps0 = [rho * eps for rho, eps in zip(rhoI, epsI)]
    rho_v = float(sum(rho_eps0))
    nph = len(rho_eps0)
    rho_c = rho_v
    for reaction in reactions:
        phase_index = int(reaction.get("phase_index", 1.0))
        phase_index = max(0, min(nph - 1, phase_index))
        rho_c -= float(reaction.get("F", 0.0)) * rho_eps0[phase_index]

    return rho_v, max(rho_c, 1.0e-14)


def load_pvd_time_entries(pvd_path: Path) -> List[Tuple[float, Dict[str, Path]]]:
    try:
        root = ET.parse(pvd_path).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse ParaView collection {pvd_path}") from exc

    collection = root.find("Collection")
    if collection is None:
        raise RuntimeError(f"Missing Collection node in {pvd_path}")

    time_map: Dict[float, Dict[str, Path]] = {}
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
    time_entries: List[Tuple[float, Dict[str, Path]]],
    required_key: str,
    context: str,
) -> List[Tuple[float, Dict[str, Path]]]:
    available: List[Tuple[float, Dict[str, Path]]] = []
    skipped = 0
    for time_value, entry in time_entries:
        path = entry.get(required_key)
        if path is None or not path.exists():
            skipped += 1
            continue
        available.append((time_value, entry))

    if not available:
        raise FileNotFoundError(
            f"No readable ParaView timesteps with {required_key!r} were found in {context}"
        )

    if skipped:
        print(
            f"Skipping {skipped} ParaView time steps with missing {required_key} files "
            f"while sampling from {context}."
        )

    return available


def print_progress(label: str, current: int, total: int) -> None:
    if total <= 0:
        return
    stride = max(1, total // 20)
    if current == 1 or current == total or current % stride == 0:
        pct = 100.0 * float(current) / float(total)
        print(f"{label}: {current}/{total} ({pct:.1f}%)", flush=True)


def sample_temperature_probes_from_paraview(
    output_dir: Path, input_yaml: Path, max_time_steps: int | None = None
) -> np.ndarray:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "temperature_probes.csv is missing and pyvista is not available "
            "for ParaView post-processing."
        ) from exc

    pvd_path = find_mfem_collection_pvd(output_dir, input_yaml)
    if pvd_path is None:
        raise FileNotFoundError(
            "temperature_probes.csv is missing and no MFEM ParaView .pvd "
            f"collection was found under {output_dir}"
        )

    probe_y = load_probe_y_from_yaml(input_yaml)
    if not probe_y:
        raise RuntimeError(f"No probe_y entries found in {input_yaml}")
    probe_x = load_probe_x_from_yaml(input_yaml)

    time_entries = filter_available_time_entries(
        load_pvd_time_entries(pvd_path), "mesh", str(pvd_path)
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
        f"Sampling temperature probes from ParaView over {len(time_entries)} time steps...",
        flush=True,
    )

    def read_dataset_at_path(mesh_path: Path):
        ds = pv.read(mesh_path)
        ds = select_dataset_with_point_fields(ds, "temperature")
        if ds is None:
            raise RuntimeError(f"Missing temperature field in {mesh_path}")

        return warp_dataset_points_by_vector_field(ds, "ale_displacement")

    def sample_temperature_at(mesh, x: float, y: float) -> float:
        query = np.array([[x, y, 0.0]], dtype=float)
        sampled = pv.PolyData(query).sample(mesh)
        mask = np.asarray(sampled.point_data.get("vtkValidPointMask", [1]), dtype=int)
        if mask.size == 0 or int(mask[0]) == 0:
            return float("nan")
        vals = np.asarray(sampled.point_data["temperature"], dtype=float)
        if vals.size == 0:
            return float("nan")
        return float(vals.reshape(-1)[0])

    warned_probe_x = False
    for i_time, (_, entry) in enumerate(time_entries):
        print_progress("Temperature probe sampling", i_time + 1, len(time_entries))
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

        wall_val = sample_temperature_at(mesh, x_query, y_max_sample)
        if not np.isfinite(wall_val):
            wall_val = sample_temperature_at(
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
            val = sample_temperature_at(mesh, x_query, y_query)
            probes[f"TC{i_probe}"][i_time] = val if np.isfinite(val) else 0.0

    return probes


def sample_fronts_from_paraview(
    output_dir: Path, input_yaml: Path, max_time_steps: int | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "front columns are missing from mass_metrics.csv and pyvista is "
            "not available for ParaView post-processing."
        ) from exc

    pvd_path = find_mfem_collection_pvd(output_dir, input_yaml)
    if pvd_path is None:
        raise FileNotFoundError(
            "front columns are missing from mass_metrics.csv and no MFEM "
            f"ParaView .pvd collection was found under {output_dir}"
        )

    time_entries = filter_available_time_entries(
        load_pvd_time_entries(pvd_path), "mesh", str(pvd_path)
    )
    if max_time_steps is not None:
        time_entries = time_entries[:max_time_steps]
        if not time_entries:
            raise RuntimeError(
                "No ParaView time steps remain after applying --max-time-steps."
            )
    time_values = np.asarray([time_value for time_value, _ in time_entries], dtype=float)
    print(
        f"Sampling fronts from ParaView over {len(time_entries)} time steps...",
        flush=True,
    )

    def read_datasets_for_entry(entry: Dict[str, Path]):
        mesh = select_dataset_with_point_fields(pv.read(entry["mesh"]), "tau")
        if mesh is None:
            raise RuntimeError(f"Missing tau field in {entry['mesh']}")
        mesh = warp_dataset_points_by_vector_field(mesh, "ale_displacement")

        qcloud = None
        tau_qp_path = entry.get("tau_qp")
        if tau_qp_path is not None and tau_qp_path.exists():
            try:
                qcloud = select_dataset_with_point_fields(pv.read(tau_qp_path), "tau_qp")
                if qcloud is not None:
                    qcloud = warp_dataset_points_by_vector_field(
                        qcloud, "ale_displacement_qp"
                    )
            except Exception:
                qcloud = None
        return mesh, qcloud

    def compute_front_crossing_y_from_mesh(
        mesh, x: float, threshold: float
    ) -> Tuple[float | None, float]:
        ns = 250
        eps = 1.0e-9
        _, _, y_bottom, y_top, _, _ = mesh.bounds
        y0 = float(y_top - eps)
        y1 = float(y_bottom + eps)
        ys = np.linspace(y0, y1, ns + 1, dtype=float)
        pts = np.column_stack(
            [np.full_like(ys, x), ys, np.zeros_like(ys)]
        )
        sampled = pv.PolyData(pts).sample(mesh)
        vals = np.asarray(sampled.point_data.get("tau", []), dtype=float).reshape(-1)
        mask = np.asarray(
            sampled.point_data.get("vtkValidPointMask", np.ones(ys.size)),
            dtype=int,
        ).reshape(-1)
        if vals.size != ys.size:
            return None, float("nan")

        vals = np.where(mask > 0, vals, np.nan)
        surf_val = float(vals[0]) if vals.size else float("nan")
        for k in range(1, ys.size):
            vp = vals[k - 1]
            vc = vals[k]
            if np.isfinite(vp) and np.isfinite(vc) and vp < threshold and vc >= threshold:
                denom = vc - vp
                frac = 0.0
                if abs(denom) > 1.0e-14:
                    frac = (threshold - vp) / denom
                    frac = max(0.0, min(1.0, frac))
                y_cross = ys[k - 1] + frac * (ys[k] - ys[k - 1])
                return float(y_cross), surf_val
        return None, surf_val

    def compute_front_crossing_y_from_qcloud(
        qcloud,
        x: float,
        threshold: float,
    ) -> Tuple[float | None, float]:
        pts = np.asarray(qcloud.points, dtype=float)
        vals = np.asarray(qcloud.point_data.get("tau_qp", []), dtype=float).reshape(-1)
        if pts.ndim != 2 or pts.shape[1] < 2 or vals.size != pts.shape[0]:
            return None, float("nan")

        unique_x = np.unique(np.round(pts[:, 0], decimals=12))
        if unique_x.size == 0:
            return None, float("nan")

        x_sel = float(unique_x[np.argmin(np.abs(unique_x - x))])
        x_tol = max(1.0e-11, 1.0e-9 * max(1.0, abs(x_sel)))
        mask = np.isclose(pts[:, 0], x_sel, atol=x_tol, rtol=0.0)
        if not np.any(mask):
            return None, float("nan")

        ys = pts[mask, 1]
        line_vals = vals[mask]
        order = np.argsort(-ys)
        ys = ys[order]
        line_vals = line_vals[order]

        surf_val = float(line_vals[0]) if line_vals.size else float("nan")
        for k in range(1, ys.size):
            vp = line_vals[k - 1]
            vc = line_vals[k]
            if np.isfinite(vp) and np.isfinite(vc) and vp < threshold and vc >= threshold:
                denom = vc - vp
                frac = 0.0
                if abs(denom) > 1.0e-14:
                    frac = (threshold - vp) / denom
                    frac = max(0.0, min(1.0, frac))
                y_cross = ys[k - 1] + frac * (ys[k] - ys[k - 1])
                return float(y_cross), surf_val
        return None, surf_val

    front98 = np.zeros(time_values.size, dtype=float)
    front2 = np.zeros(time_values.size, dtype=float)

    initial_mesh, _ = read_datasets_for_entry(time_entries[0][1])
    _, _, _, initial_y_top, _, _ = initial_mesh.bounds
    xmid: float | None = None
    for i_time, (_, entry) in enumerate(time_entries):
        print_progress("Front sampling", i_time + 1, len(time_entries))
        mesh, qcloud = read_datasets_for_entry(entry)
        if xmid is None:
            xmin, xmax, _, _, _, _ = mesh.bounds
            xmid = 0.5 * (float(xmin) + float(xmax))
        xmin, xmax, y_bottom_live, y_top_live, _, _ = mesh.bounds
        recession = max(0.0, float(initial_y_top - y_top_live))
        if qcloud is not None:
            y98, tau_surface = compute_front_crossing_y_from_qcloud(
                qcloud, xmid, 0.98
            )
        else:
            y98, tau_surface = compute_front_crossing_y_from_mesh(
                mesh, xmid, 0.98
            )
        if np.isfinite(tau_surface) and tau_surface < 0.98:
            if y98 is None:
                front98[i_time] = max(0.0, float(initial_y_top - y_bottom_live))
            else:
                front98[i_time] = max(0.0, float(initial_y_top - y98))
        else:
            front98[i_time] = 0.0

        if np.isfinite(tau_surface) and tau_surface < 0.02:
            if qcloud is not None:
                y2, _ = compute_front_crossing_y_from_qcloud(
                    qcloud, xmid, 0.02
                )
            else:
                y2, _ = compute_front_crossing_y_from_mesh(
                    mesh, xmid, 0.02
                )
            if y2 is None:
                front2[i_time] = max(0.0, float(initial_y_top - y_bottom_live))
            else:
                front2[i_time] = max(0.0, float(initial_y_top - y2))
        else:
            front2[i_time] = 0.0

    return time_values, front98, front2


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def segmented_rmse_max(
    t: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    t0: float,
    t1: float,
    valid_mask: np.ndarray | None = None,
) -> Tuple[float, float]:
    mask = (t >= t0) & (t <= t1)
    if valid_mask is not None:
        mask = mask & valid_mask
    if not np.any(mask):
        return float("nan"), float("nan")
    return rmse(a[mask], b[mask]), max_abs(a[mask], b[mask])


def tc_index(name: str) -> int:
    if name.startswith("TC") and name[2:].isdigit():
        return int(name[2:])
    return -1


def build_mfem_temperature_by_depth(
    probes: np.ndarray, probe_depths: List[float]
) -> List[Tuple[float, str, np.ndarray]]:
    items: List[Tuple[float, str, np.ndarray]] = []
    for name in probes.dtype.names:
        if name == "time":
            continue
        if name == "wall":
            depth = 0.0
        else:
            idx = tc_index(name)
            if idx < 0:
                continue
            depth = probe_depths[idx] if idx < len(probe_depths) else float(idx)
        items.append((depth, name, probes[name]))
    items.sort(key=lambda x: x[0])
    return items


def build_amaryllis_temperature_by_depth(
    am_energy: np.ndarray, probe_depths: List[float]
) -> List[Tuple[float, str, np.ndarray]]:
    n_signals = int(am_energy.shape[1]) - 1
    items: List[Tuple[float, str, np.ndarray]] = []
    for i in range(n_signals):
        name = "wall" if i == 0 else f"TC{i}"
        depth = probe_depths[i] if i < len(probe_depths) else float(i)
        items.append((depth, name, am_energy[:, i + 1]))
    items.sort(key=lambda x: x[0])
    return items


def ensure_2d(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a


def time_derivative(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    if t.size != y.size:
        raise ValueError("time_derivative expects t and y with the same length")
    if t.size < 2:
        return np.zeros_like(y, dtype=float)
    edge_order = 2 if t.size >= 3 else 1
    return np.gradient(y, t, edge_order=edge_order)


def parse_csv_float_list(text: str) -> List[float]:
    vals: List[float] = []
    for tok in text.split(","):
        s = tok.strip()
        if not s:
            continue
        vals.append(float(s))
    return vals


def load_pato_point_plot(path: Path) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    with path.open("r") as f:
        header = f.readline().strip()
    y_vals = [
        float(m.group(1))
        for m in re.finditer(r"probe\d+\([^,]+,([^,]+),", header)
    ]
    data = ensure_2d(np.loadtxt(path, comments="/"))
    if data.shape[1] < 2:
        raise RuntimeError(f"Unexpected PATO point-plot format in {path}")
    time = data[:, 0]
    vals = data[:, 1:]
    if y_vals and len(y_vals) != vals.shape[1]:
        raise RuntimeError(
            f"PATO point-plot header/data column mismatch in {path}: "
            f"{len(y_vals)} probes in header vs {vals.shape[1]} data columns"
        )
    return time, vals, y_vals


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1.0e-30)
    out[mask] = num[mask] / den[mask]
    return out


def build_mfem_pressure_by_y(
    mfem_pressure_probes: np.ndarray, probe_y: List[float]
) -> Dict[float, Tuple[str, np.ndarray]]:
    out: Dict[float, Tuple[str, np.ndarray]] = {}
    names = list(mfem_pressure_probes.dtype.names or [])
    if "wall" in names and probe_y:
        out[probe_y[0]] = ("wall", mfem_pressure_probes["wall"])
    for name in names:
        if name in ("time", "wall"):
            continue
        idx = tc_index(name)
        if idx > 0 and idx < len(probe_y):
            out[probe_y[idx]] = (name, mfem_pressure_probes[name])
    return out


def match_pressure_probe_points(
    mfem_pressure_probes: np.ndarray,
    probe_y: List[float],
    pato_p_y: List[float],
    pato_tol: float = 1.0e-8,
) -> List[Tuple[float, str, int, np.ndarray]]:
    mfem_y_map = build_mfem_pressure_by_y(mfem_pressure_probes, probe_y)
    if not mfem_y_map:
        return []
    mfem_y_keys = list(mfem_y_map.keys())
    matched: List[Tuple[float, str, int, np.ndarray]] = []
    for j, y_pato in enumerate(pato_p_y):
        y_best = min(mfem_y_keys, key=lambda y: abs(y - y_pato))
        if abs(y_best - y_pato) <= pato_tol:
            name, series = mfem_y_map[y_best]
            matched.append((y_pato, name, j, series))
    matched.sort(key=lambda x: x[0], reverse=True)
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="ParaView/ablation_case2_2",
        help=(
            "Directory containing mass_metrics.csv and, optionally, "
            "temperature_probes.csv and front columns in mass_metrics.csv. "
            "If the temperature CSV or front columns are missing, the script "
            "samples them from the MFEM ParaView .pvd output."
        ),
    )
    parser.add_argument(
        "--input",
        default="Input/input_ablation_case2_2.yaml",
        help="Input YAML file with acceptance tolerances",
    )
    parser.add_argument(
        "--amaryllis-energy",
        default="/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/data/ref/PATO/PATO_Energy_TestCase_2.2.txt",
        help="Amaryllis energy reference file",
    )
    parser.add_argument(
        "--amaryllis-mass",
        default="/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/data/ref/PATO/PATO_Mass_TestCase_2.2.txt",
        help="Amaryllis mass reference file",
    )
    parser.add_argument(
        "--out-prefix",
        default="ablation_case2_2",
        help="Prefix for generated plot filenames",
    )
    parser.add_argument(
        "--pato-surface-diagnostics",
        default=(
            "/home/quinnchr/miniconda3/envs/pato/src/volume_pato/pato-3.1/"
            "tutorials/1D/AblationTestCase_2.x/output/porousMat/"
            "top_surface_diagnostics.csv"
        ),
        help="PATO top-surface diagnostics CSV (for snGradP/mobility comparison)",
    )
    parser.add_argument(
        "--pato-pressure-plot",
        default=(
            "/home/quinnchr/miniconda3/envs/pato/src/volume_pato/pato-3.1/"
            "tutorials/1D/AblationTestCase_2.x/output/porousMat/scalar/p_plot"
        ),
        help="PATO sampled point plot for pressure (system/porousMat/plotDict)",
    )
    parser.add_argument(
        "--pressure-profile-times",
        default="0.1,1,10,60",
        help="Comma-separated snapshot times (s) for centerline pressure profile comparison",
    )
    parser.add_argument(
        "--max-time-steps",
        type=int,
        default=None,
        help=(
            "Maximum number of ParaView time steps to sample when falling back "
            "to MFEM .pvd output for temperature probes or fronts."
        ),
    )
    args = parser.parse_args()
    if args.max_time_steps is not None and args.max_time_steps < 1:
        parser.error("--max-time-steps must be >= 1")

    out_dir = Path(args.output_dir)
    probes_csv = out_dir / "temperature_probes.csv"
    mass_csv = out_dir / "mass_metrics.csv"
    clamp_csv = out_dir / "bprime_clamp_stats.csv"
    boundary_diag_csv = out_dir / "boundary_diagnostics.csv"
    mfem_pressure_probes_csv = out_dir / "pressure_probes.csv"
    mfem_mesh_diag_csv = out_dir / "mesh_diagnostics.csv"
    mfem_mass_eq_probe_csv = out_dir / "mass_eq_probe_diagnostics.csv"
    pato_surface_diag_csv = Path(args.pato_surface_diagnostics).expanduser()
    pato_pressure_plot = Path(args.pato_pressure_plot).expanduser()

    if not mass_csv.exists():
        raise FileNotFoundError(f"Expected MFEM output not found: {mass_csv}")

    tol = load_acceptance_from_yaml(Path(args.input))

    probes = load_named_csv(
        probes_csv, required=False, description="temperature_probes.csv"
    )
    if probes is None:
        print(
            f"temperature_probes.csv not found or empty in {out_dir}; "
            "sampling temperature probes from ParaView output instead."
        )
        probes = sample_temperature_probes_from_paraview(
            out_dir, Path(args.input), max_time_steps=args.max_time_steps
        )

    mass = load_named_csv(
        mass_csv, required=True, description="mass_metrics.csv"
    )
    boundary_diag = (
        load_named_csv(
            boundary_diag_csv,
            required=False,
            description="boundary_diagnostics.csv",
        )
        if boundary_diag_csv.exists()
        else None
    )
    mfem_pressure_probes = (
        load_named_csv(
            mfem_pressure_probes_csv,
            required=False,
            description="pressure_probes.csv",
        )
        if mfem_pressure_probes_csv.exists()
        else None
    )
    mfem_mesh_diag = (
        load_named_csv(
            mfem_mesh_diag_csv,
            required=False,
            description="mesh_diagnostics.csv",
        )
        if mfem_mesh_diag_csv.exists()
        else None
    )
    mfem_mass_eq_probe = (
        load_named_csv(
            mfem_mass_eq_probe_csv,
            required=False,
            description="mass_eq_probe_diagnostics.csv",
        )
        if mfem_mass_eq_probe_csv.exists()
        else None
    )
    pato_surface_diag = (
        load_named_csv(
            pato_surface_diag_csv,
            required=False,
            description="surface_diags.dat",
        )
        if pato_surface_diag_csv.exists()
        else None
    )
    pato_p_time = None
    pato_p_vals = None
    pato_p_y: List[float] = []
    if pato_pressure_plot.exists():
        pato_p_time, pato_p_vals, pato_p_y = load_pato_point_plot(pato_pressure_plot)

    am_energy = ensure_2d(np.loadtxt(args.amaryllis_energy, skiprows=1))
    am_mass = ensure_2d(np.loadtxt(args.amaryllis_mass, skiprows=1))

    probe_y = load_probe_y_from_yaml(Path(args.input))
    probe_depths = load_probe_depths_from_yaml(Path(args.input))
    pressure_profile_times = parse_csv_float_list(args.pressure_profile_times)
    mfem_by_depth = build_mfem_temperature_by_depth(probes, probe_depths)
    am_by_depth = build_amaryllis_temperature_by_depth(am_energy, probe_depths)
    n_common = min(len(mfem_by_depth), len(am_by_depth))
    if n_common == 0:
        raise RuntimeError("No temperature probes available for MFEM/Amaryllis comparison.")

    mfem_time = probes["time"]
    am_time = am_energy[:, 0]
    probe_pairs = list(zip(mfem_by_depth[:n_common], am_by_depth[:n_common]))

    temp_metrics: List[Tuple[str, float, float, bool]] = []
    for (d_mf, name_mf, sig_mf), (_, name_am, sig_am) in probe_pairs:
        valid = sig_am > 1.0  # Ignore Amaryllis sentinel zeros.
        if np.any(valid):
            mfem_interp = np.interp(am_time[valid], mfem_time, sig_mf)
            am_sig = sig_am[valid]
            r = rmse(mfem_interp, am_sig)
            m = max_abs(mfem_interp, am_sig)
            ok = (r <= tol["temperature_rmse_max"]) and (
                m <= tol["temperature_max_abs_max"]
            )
        else:
            r = float("nan")
            m = float("nan")
            ok = True
        sig_label = f"{name_mf}~{name_am}@depth={d_mf:.6g}m"
        temp_metrics.append((sig_label, r, m, ok))

    # Segmented wall-temperature metrics for cooldown/heating analysis.
    wall_mf = np.interp(am_time, mfem_time, probe_pairs[0][0][2])
    wall_ref = probe_pairs[0][1][2]
    wall_valid = wall_ref > 1.0
    heat_rmse, heat_max = segmented_rmse_max(
        am_time, wall_mf, wall_ref, 0.1, 60.0, wall_valid
    )
    cool_rmse, cool_max = segmented_rmse_max(
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

    # Amaryllis mass columns: time, m_dot_g, m_dot_c, front98, front2, recession.
    t_ref = am_mass[:, 0]
    ref_mdot = am_mass[:, 1]
    ref_mdot_c = am_mass[:, 2]
    ref_front98 = am_mass[:, 3]
    ref_front2 = am_mass[:, 4]
    ref_recession = am_mass[:, 5]

    mfem_mass_t = mass["time"]
    mfem_mdot = mass["m_dot_g_surf"]
    mfem_mdot_centerline = (
        mass["m_dot_g_centerline"] if "m_dot_g_centerline" in mass.dtype.names else None
    )
    mfem_front_source = "mass_metrics.csv"
    if (
        "front_98_virgin" in mass.dtype.names
        and "front_2_char" in mass.dtype.names
    ):
        mfem_front_t = mfem_mass_t
        mfem_front98 = mass["front_98_virgin"]
        mfem_front2 = mass["front_2_char"]
    else:
        mfem_front_source = "ParaView fallback"
        print(
            "front columns not found in mass_metrics.csv; "
            "sampling fronts from ParaView output instead."
        )
        mfem_front_t, mfem_front98, mfem_front2 = sample_fronts_from_paraview(
            out_dir, Path(args.input), max_time_steps=args.max_time_steps
        )
    mfem_mdot_c = mass["m_dot_c"]
    mfem_recession = mass["recession"]
    mfem_recession_rate = time_derivative(mfem_mass_t, mfem_recession)
    ref_recession_rate = time_derivative(t_ref, ref_recession)

    mfem_mdot_i = np.interp(t_ref, mfem_mass_t, mfem_mdot)
    mfem_mdot_c_i = np.interp(t_ref, mfem_mass_t, mfem_mdot_c)
    mfem_front98_i = np.interp(t_ref, mfem_front_t, mfem_front98)
    mfem_front2_i = np.interp(t_ref, mfem_front_t, mfem_front2)
    mfem_recession_i = np.interp(t_ref, mfem_mass_t, mfem_recession)

    mdot_rmse = rmse(mfem_mdot_i, ref_mdot)
    mdot_max = max_abs(mfem_mdot_i, ref_mdot)

    i_mf = int(np.argmax(mfem_mdot))
    i_ref = int(np.argmax(ref_mdot))
    mf_peak = float(mfem_mdot[i_mf])
    ref_peak = float(ref_mdot[i_ref])
    mf_peak_t = float(mfem_mass_t[i_mf])
    ref_peak_t = float(t_ref[i_ref])

    peak_rel = abs(mf_peak - ref_peak) / max(abs(ref_peak), 1.0e-12)
    peak_time_err = abs(mf_peak_t - ref_peak_t)

    front98_rmse = rmse(mfem_front98_i, ref_front98)
    front98_max = max_abs(mfem_front98_i, ref_front98)
    front2_rmse = rmse(mfem_front2_i, ref_front2)
    front2_max = max_abs(mfem_front2_i, ref_front2)

    mdot_c_rmse = rmse(mfem_mdot_c_i, ref_mdot_c)
    i_mf_c = int(np.argmax(mfem_mdot_c))
    i_ref_c = int(np.argmax(ref_mdot_c))
    mf_peak_c = float(mfem_mdot_c[i_mf_c])
    ref_peak_c = float(ref_mdot_c[i_ref_c])
    mdot_c_peak_rel = abs(mf_peak_c - ref_peak_c) / max(abs(ref_peak_c), 1.0e-12)

    recession_rmse = rmse(mfem_recession_i, ref_recession)
    recession_final_rel = abs(mfem_recession_i[-1] - ref_recession[-1]) / max(
        abs(ref_recession[-1]), 1.0e-12
    )

    clamp_counts = {"pressure": np.nan, "BprimeG": np.nan, "temperature": np.nan}
    if clamp_csv.exists():
        with clamp_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                axis = str(row.get("axis", "")).strip()
                if axis in clamp_counts:
                    try:
                        clamp_counts[axis] = float(row.get("clamp_count", "nan"))
                    except ValueError:
                        clamp_counts[axis] = np.nan

    temp_pass = all(x[3] for x in temp_metrics)
    mdot_pass = (
        mdot_rmse <= tol["m_dot_g_rmse_max"]
        and mdot_max <= tol["m_dot_g_max_abs_max"]
        and peak_rel <= tol["m_dot_g_peak_rel_error_max"]
        and peak_time_err <= tol["m_dot_g_peak_time_error_max"]
    )
    temp_pass = temp_pass and heat_pass and cool_pass
    front98_pass = (
        front98_rmse <= tol["front98_rmse_max"]
        and front98_max <= tol["front98_max_abs_max"]
    )
    front2_pass = (
        front2_rmse <= tol["front2_rmse_max"]
        and front2_max <= tol["front2_max_abs_max"]
    )
    mdot_c_pass = (
        mdot_c_rmse <= tol["m_dot_c_rmse_max"]
        and mdot_c_peak_rel <= tol["m_dot_c_peak_rel_error_max"]
    )
    recession_pass = (
        recession_rmse <= tol["recession_rmse_max"]
        and recession_final_rel <= tol["recession_final_rel_error_max"]
    )

    overall_pass = (
        temp_pass
        and mdot_pass
        and front98_pass
        and front2_pass
        and mdot_c_pass
        and recession_pass
    )

    metrics_csv = out_dir / "amaryllis_error_metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "signal", "rmse", "max_abs", "metric", "value", "tolerance", "pass"])
        for sig, r, m, ok in temp_metrics:
            w.writerow(["temperature", sig, r, m, "", "", "", int(ok)])
        w.writerow(["temperature_segment", "wall_heating_0.1_60s", heat_rmse, heat_max,
                    "", "", "", int(heat_pass)])
        w.writerow(["temperature_segment", "wall_cooling_60.1_120s", cool_rmse, cool_max,
                    "", "", "", int(cool_pass)])

        w.writerow(["mass_flux", "m_dot_g", mdot_rmse, mdot_max, "rmse",
                    mdot_rmse, tol["m_dot_g_rmse_max"], int(mdot_rmse <= tol["m_dot_g_rmse_max"])])
        w.writerow(["mass_flux", "m_dot_g", mdot_rmse, mdot_max, "max_abs",
                    mdot_max, tol["m_dot_g_max_abs_max"], int(mdot_max <= tol["m_dot_g_max_abs_max"])])
        w.writerow(["mass_flux", "m_dot_g", "", "", "peak_rel_error",
                    peak_rel, tol["m_dot_g_peak_rel_error_max"], int(peak_rel <= tol["m_dot_g_peak_rel_error_max"])])
        w.writerow(["mass_flux", "m_dot_g", "", "", "peak_time_error",
                    peak_time_err, tol["m_dot_g_peak_time_error_max"], int(peak_time_err <= tol["m_dot_g_peak_time_error_max"])])

        w.writerow(["front", "front_98_virgin", front98_rmse, front98_max, "rmse",
                    front98_rmse, tol["front98_rmse_max"], int(front98_rmse <= tol["front98_rmse_max"])])
        w.writerow(["front", "front_98_virgin", front98_rmse, front98_max, "max_abs",
                    front98_max, tol["front98_max_abs_max"], int(front98_max <= tol["front98_max_abs_max"])])
        w.writerow(["front", "front_2_char", front2_rmse, front2_max, "rmse",
                    front2_rmse, tol["front2_rmse_max"], int(front2_rmse <= tol["front2_rmse_max"])])
        w.writerow(["front", "front_2_char", front2_rmse, front2_max, "max_abs",
                    front2_max, tol["front2_max_abs_max"], int(front2_max <= tol["front2_max_abs_max"])])

        w.writerow(["mass_flux", "m_dot_c", mdot_c_rmse, "", "rmse",
                    mdot_c_rmse, tol["m_dot_c_rmse_max"], int(mdot_c_rmse <= tol["m_dot_c_rmse_max"])])
        w.writerow(["mass_flux", "m_dot_c", "", "", "peak_rel_error",
                    mdot_c_peak_rel, tol["m_dot_c_peak_rel_error_max"], int(mdot_c_peak_rel <= tol["m_dot_c_peak_rel_error_max"])])
        w.writerow(["recession", "recession", recession_rmse, "", "rmse",
                    recession_rmse, tol["recession_rmse_max"], int(recession_rmse <= tol["recession_rmse_max"])])
        w.writerow(["recession", "recession", "", "", "final_rel_error",
                    recession_final_rel, tol["recession_final_rel_error_max"], int(recession_final_rel <= tol["recession_final_rel_error_max"])])

        for axis in ("pressure", "BprimeG", "temperature"):
            w.writerow(["bprime_clamp", axis, "", "", "count",
                        clamp_counts[axis], "", ""])

        w.writerow(["summary", "overall", "", "", "", "", "", int(overall_pass)])

    tol_csv = out_dir / "amaryllis_error_tolerances.csv"
    with tol_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal", "tolerance"])
        for k, v in DEFAULT_TOL.items():
            w.writerow([k, tol.get(k, v)])

    fronts_csv = out_dir / "amaryllis_front_comparison.csv"
    with fronts_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time",
                "mfem_front_98_virgin",
                "amaryllis_front_98_virgin",
                "front_98_virgin_error",
                "mfem_front_2_char",
                "amaryllis_front_2_char",
                "front_2_char_error",
                "mfem_front_source",
            ]
        )
        for (
            t_val,
            mfem_98,
            ref_98,
            mfem_2,
            ref_2,
        ) in zip(t_ref, mfem_front98_i, ref_front98, mfem_front2_i, ref_front2):
            w.writerow(
                [
                    t_val,
                    mfem_98,
                    ref_98,
                    mfem_98 - ref_98,
                    mfem_2,
                    ref_2,
                    mfem_2 - ref_2,
                    mfem_front_source,
                ]
            )

    # Plot 1: temperature history.
    plt.figure(figsize=(14, 5))
    cmap = plt.get_cmap("tab10")
    for i, ((d_mf, name_mf, sig_mf), (_, name_ref, sig_ref)) in enumerate(probe_pairs):
        col = "black" if i == 0 else cmap((i - 1) % 10)
        depth_label = f"{d_mf:.4f} m"
        plt.plot(mfem_time, sig_mf, color=col, lw=2,
                 label=f"MFEM {name_mf} ({depth_label})")
        plt.plot(am_time, sig_ref, color=col, lw=1.6, ls="--",
                 label=f"Amaryllis {name_ref} ({depth_label})")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.xlim(0.0, max(float(mfem_time[-1]), float(am_time[-1])))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=9)
    plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    plt.savefig(out_dir / f"{args.out_prefix}_temperature_history.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Plot 2: pyrolysis mass flux.
    plt.figure(figsize=(13, 4.8))
    plt.plot(mfem_mass_t, mfem_mdot, color="black", lw=2, label="MFEM area-avg")
    if mfem_mdot_centerline is not None:
        plt.plot(
            mfem_mass_t,
            mfem_mdot_centerline,
            color="tab:blue",
            lw=2,
            label="MFEM centerline",
        )
    plt.plot(t_ref, ref_mdot, color="black", ls="--", lw=2, label="Amaryllis")
    plt.xlabel("Time (s)")
    plt.ylabel("Pyrolysis mass flux (kg/m2/s)")
    plt.xlim(0.0, max(float(mfem_mass_t[-1]), float(t_ref[-1])))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
    plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    plt.savefig(out_dir / f"{args.out_prefix}_pyrolysis_mass_flux.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Plot 3: fronts.
    plt.figure(figsize=(13, 4.8))
    plt.plot(mfem_front_t, mfem_front98, color="black", lw=2, label="MFEM 98% virgin")
    plt.plot(mfem_front_t, mfem_front2, color="gray", lw=2, label="MFEM 2% char")
    plt.plot(t_ref, ref_front98, color="black", lw=2, ls="--", label="Amaryllis 98% virgin")
    plt.plot(t_ref, ref_front2, color="gray", lw=2, ls="--", label="Amaryllis 2% char")
    plt.xlabel("Time (s)")
    plt.ylabel("Depth (m)")
    plt.xlim(0.0, max(float(mfem_front_t[-1]), float(t_ref[-1])))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
    plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    plt.savefig(out_dir / f"{args.out_prefix}_fronts.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Plot 4: recession amount and recession rate comparison.
    fig, (ax_rec, ax_rate) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
    )
    ax_rec.plot(mfem_mass_t, mfem_recession, color="black", lw=2, label="MFEM")
    ax_rec.plot(t_ref, ref_recession, color="black", lw=2, ls="--", label="Amaryllis")
    ax_rec.set_ylabel("Recession (m)")
    ax_rec.grid(True, alpha=0.25)
    ax_rec.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    ax_rate.plot(mfem_mass_t, mfem_recession_rate, color="black", lw=2, label="MFEM")
    ax_rate.plot(t_ref, ref_recession_rate, color="black", lw=2, ls="--", label="Amaryllis")
    ax_rate.set_xlabel("Time (s)")
    ax_rate.set_ylabel("Recession rate (m/s)")
    ax_rate.grid(True, alpha=0.25)

    xmax_rec = max(float(mfem_mass_t[-1]), float(t_ref[-1]))
    ax_rec.set_xlim(0.0, xmax_rec)
    ax_rate.set_xlim(0.0, xmax_rec)

    fig.savefig(
        out_dir / f"{args.out_prefix}_recession_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Plot 5/6/7: MFEM vs PATO centerline surface diagnostics.
    pato_diag_plot = None
    pato_props_plot = None
    pato_flux_recon_plot = None
    pato_flux_ratio_plot = None
    pato_pressure_profile_plot = None
    pato_pressure_slope_plot = None
    mesh_diag_plot = None
    mass_eq_probe_plot = None
    mass_eq_wall_pato_plot = None
    if boundary_diag is not None and pato_surface_diag is not None:
        mfem_names = set(boundary_diag.dtype.names or ())
        pato_names = set(pato_surface_diag.dtype.names or ())
        mfem_needed = {"time", "gradp_n_centerline", "mobility_centerline"}
        pato_needed = {"time", "snGradP_centerline", "mobility_centerline"}
        if mfem_needed.issubset(mfem_names) and pato_needed.issubset(pato_names):
            fig, (ax_p, ax_mob) = plt.subplots(
                2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
            )

            ax_p.plot(
                boundary_diag["time"],
                boundary_diag["gradp_n_centerline"],
                color="tab:blue",
                lw=2,
                label="MFEM centerline",
            )
            ax_p.plot(
                pato_surface_diag["time"],
                pato_surface_diag["snGradP_centerline"],
                color="tab:blue",
                lw=2,
                ls="--",
                label="PATO centerline",
            )
            ax_p.set_ylabel("snGrad(p) (Pa/m)")
            ax_p.grid(True, alpha=0.25)
            ax_p.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            ax_mob.plot(
                boundary_diag["time"],
                boundary_diag["mobility_centerline"],
                color="tab:green",
                lw=2,
                label="MFEM centerline",
            )
            ax_mob.plot(
                pato_surface_diag["time"],
                pato_surface_diag["mobility_centerline"],
                color="tab:green",
                lw=2,
                ls="--",
                label="PATO centerline",
            )
            ax_mob.set_xlabel("Time (s)")
            ax_mob.set_ylabel(r"Mobility $\rho_g K / \mu$ (SI)")
            ax_mob.grid(True, alpha=0.25)
            ax_mob.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            xmax_diag = max(
                float(np.nanmax(boundary_diag["time"])),
                float(np.nanmax(pato_surface_diag["time"])),
            )
            ax_p.set_xlim(0.0, xmax_diag)
            ax_mob.set_xlim(0.0, xmax_diag)

            pato_diag_plot = out_dir / f"{args.out_prefix}_surface_diagnostics_compare.png"
            fig.savefig(pato_diag_plot, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            print(
                "Skipping PATO/MFEM surface diagnostics plot: missing columns in "
                f"{boundary_diag_csv} or {pato_surface_diag_csv}"
            )

        mfem_flux_needed = {
            "time",
            "m_dot_g_centerline",
            "gradp_n_centerline",
            "mobility_centerline",
        }
        pato_flux_needed = {
            "time",
            "mDotGw_centerline",
            "snGradP_centerline",
            "mobility_centerline",
        }
        if mfem_flux_needed.issubset(mfem_names) and pato_flux_needed.issubset(pato_names):
            mfem_flux_recon = (
                -boundary_diag["mobility_centerline"] * boundary_diag["gradp_n_centerline"]
            )
            pato_flux_recon = (
                -pato_surface_diag["mobility_centerline"] * pato_surface_diag["snGradP_centerline"]
            )

            fig, (ax_flux, ax_delta) = plt.subplots(
                2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
            )
            ax_flux.plot(
                boundary_diag["time"],
                boundary_diag["m_dot_g_centerline"],
                color="tab:blue",
                lw=2,
                label="MFEM direct",
            )
            ax_flux.plot(
                boundary_diag["time"],
                mfem_flux_recon,
                color="tab:blue",
                lw=1.8,
                ls=":",
                label="MFEM reconstructed",
            )
            ax_flux.plot(
                pato_surface_diag["time"],
                pato_surface_diag["mDotGw_centerline"],
                color="tab:orange",
                lw=2,
                ls="--",
                label="PATO direct",
            )
            ax_flux.plot(
                pato_surface_diag["time"],
                pato_flux_recon,
                color="tab:orange",
                lw=1.8,
                ls="-.",
                label="PATO reconstructed",
            )
            ax_flux.set_ylabel("m_dot_g (kg/m2/s)")
            ax_flux.grid(True, alpha=0.25)
            ax_flux.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            ax_delta.plot(
                boundary_diag["time"],
                mfem_flux_recon - boundary_diag["m_dot_g_centerline"],
                color="tab:blue",
                lw=2,
                label="MFEM recon - direct",
            )
            ax_delta.plot(
                pato_surface_diag["time"],
                pato_flux_recon - pato_surface_diag["mDotGw_centerline"],
                color="tab:orange",
                lw=2,
                ls="--",
                label="PATO recon - direct",
            )
            ax_delta.set_xlabel("Time (s)")
            ax_delta.set_ylabel("Flux residual")
            ax_delta.grid(True, alpha=0.25)
            ax_delta.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            xmax_flux = max(
                float(np.nanmax(boundary_diag["time"])),
                float(np.nanmax(pato_surface_diag["time"])),
            )
            ax_flux.set_xlim(0.0, xmax_flux)
            ax_delta.set_xlim(0.0, xmax_flux)

            pato_flux_recon_plot = out_dir / f"{args.out_prefix}_flux_reconstruction_compare.png"
            fig.savefig(pato_flux_recon_plot, dpi=180, bbox_inches="tight")
            plt.close(fig)

            # Ratio decomposition on a common time grid (MFEM time samples).
            t_mf = np.asarray(boundary_diag["time"], dtype=float)
            t_pa = np.asarray(pato_surface_diag["time"], dtype=float)
            t0 = max(float(np.nanmin(t_mf)), float(np.nanmin(t_pa)))
            t1 = min(float(np.nanmax(t_mf)), float(np.nanmax(t_pa)))
            common_mask = (t_mf >= t0) & (t_mf <= t1)
            t_common = t_mf[common_mask]
            if t_common.size >= 2:
                mfem_direct = np.asarray(boundary_diag["m_dot_g_centerline"], dtype=float)[common_mask]
                mfem_recon_common = np.asarray(mfem_flux_recon, dtype=float)[common_mask]
                mfem_mob_common = np.asarray(boundary_diag["mobility_centerline"], dtype=float)[common_mask]
                mfem_grad_common = np.asarray(boundary_diag["gradp_n_centerline"], dtype=float)[common_mask]

                pato_direct_i = np.interp(t_common, t_pa, pato_surface_diag["mDotGw_centerline"])
                pato_recon_i = np.interp(t_common, t_pa, pato_flux_recon)
                pato_mob_i = np.interp(t_common, t_pa, pato_surface_diag["mobility_centerline"])
                pato_grad_i = np.interp(t_common, t_pa, pato_surface_diag["snGradP_centerline"])

                ratio_direct = safe_divide(mfem_direct, pato_direct_i)
                ratio_recon = safe_divide(mfem_recon_common, pato_recon_i)
                ratio_mob = safe_divide(mfem_mob_common, pato_mob_i)
                ratio_grad_abs = safe_divide(np.abs(mfem_grad_common), np.abs(pato_grad_i))

                fig, (ax_fluxr, ax_mobr, ax_gradr) = plt.subplots(
                    3, 1, figsize=(13, 9.2), sharex=True, constrained_layout=True
                )
                for ax in (ax_fluxr, ax_mobr, ax_gradr):
                    ax.axhline(1.0, color="black", lw=1.2, ls=":")
                    ax.grid(True, alpha=0.25)

                ax_fluxr.plot(t_common, ratio_direct, lw=2, color="tab:blue", label="direct flux ratio")
                ax_fluxr.plot(
                    t_common, ratio_recon, lw=1.8, ls="--", color="tab:orange", label="reconstructed flux ratio"
                )
                ax_fluxr.set_ylabel("MFEM / PATO")
                ax_fluxr.set_title("Centerline mass-flux ratio decomposition")
                ax_fluxr.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

                ax_mobr.plot(t_common, ratio_mob, lw=2, color="tab:green", label="mobility ratio")
                ax_mobr.set_ylabel("MFEM / PATO")
                ax_mobr.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

                ax_gradr.plot(
                    t_common,
                    ratio_grad_abs,
                    lw=2,
                    color="tab:red",
                    label=r"$|\nabla p|$ ratio",
                )
                ax_gradr.set_xlabel("Time (s)")
                ax_gradr.set_ylabel("MFEM / PATO")
                ax_gradr.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

                ax_fluxr.set_xlim(float(t_common[0]), float(t_common[-1]))
                ax_mobr.set_xlim(float(t_common[0]), float(t_common[-1]))
                ax_gradr.set_xlim(float(t_common[0]), float(t_common[-1]))

                pato_flux_ratio_plot = out_dir / f"{args.out_prefix}_flux_ratio_decomposition.png"
                fig.savefig(pato_flux_ratio_plot, dpi=180, bbox_inches="tight")
                plt.close(fig)
        else:
            print(
                "Skipping PATO/MFEM flux reconstruction plot: missing columns in "
                f"{boundary_diag_csv} or {pato_surface_diag_csv}"
            )

        mfem_prop_needed = {"time", "rho_g_centerline", "mu_g_centerline", "K_centerline"}
        pato_prop_needed = {"time", "rho_g_centerline", "mu_g_centerline", "Knn_centerline"}
        if mfem_prop_needed.issubset(mfem_names) and pato_prop_needed.issubset(pato_names):
            fig, (ax_rho, ax_k, ax_mu) = plt.subplots(
                3, 1, figsize=(13, 10.0), sharex=True, constrained_layout=True
            )

            ax_rho.plot(
                boundary_diag["time"],
                boundary_diag["rho_g_centerline"],
                color="tab:orange",
                lw=2,
                label="MFEM centerline",
            )
            ax_rho.plot(
                pato_surface_diag["time"],
                pato_surface_diag["rho_g_centerline"],
                color="tab:orange",
                lw=2,
                ls="--",
                label="PATO centerline",
            )
            ax_rho.set_ylabel(r"$\rho_g$ (kg/m$^3$)")
            ax_rho.grid(True, alpha=0.25)
            ax_rho.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            ax_k.plot(
                boundary_diag["time"],
                boundary_diag["K_centerline"],
                color="tab:red",
                lw=2,
                label="MFEM centerline",
            )
            ax_k.plot(
                pato_surface_diag["time"],
                pato_surface_diag["Knn_centerline"],
                color="tab:red",
                lw=2,
                ls="--",
                label="PATO centerline",
            )
            ax_k.set_ylabel(r"$K_{nn}$ (m$^2$)")
            ax_k.grid(True, alpha=0.25)
            ax_k.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            ax_mu.plot(
                boundary_diag["time"],
                boundary_diag["mu_g_centerline"],
                color="tab:purple",
                lw=2,
                label="MFEM centerline",
            )
            ax_mu.plot(
                pato_surface_diag["time"],
                pato_surface_diag["mu_g_centerline"],
                color="tab:purple",
                lw=2,
                ls="--",
                label="PATO centerline",
            )
            ax_mu.set_xlabel("Time (s)")
            ax_mu.set_ylabel(r"$\mu_g$ (Pa·s)")
            ax_mu.grid(True, alpha=0.25)
            ax_mu.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            xmax_props = max(
                float(np.nanmax(boundary_diag["time"])),
                float(np.nanmax(pato_surface_diag["time"])),
            )
            ax_rho.set_xlim(0.0, xmax_props)
            ax_k.set_xlim(0.0, xmax_props)
            ax_mu.set_xlim(0.0, xmax_props)

            pato_props_plot = out_dir / f"{args.out_prefix}_surface_properties_compare.png"
            fig.savefig(pato_props_plot, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            print(
                "Skipping PATO/MFEM surface property plot: missing columns in "
                f"{boundary_diag_csv} or {pato_surface_diag_csv}"
            )
    elif boundary_diag is None or pato_surface_diag is None:
        print(
            "Skipping PATO/MFEM surface diagnostics plot: missing "
            f"{boundary_diag_csv if boundary_diag is None else pato_surface_diag_csv}"
        )

    # Plot 8: MFEM mesh diagnostics at selected probes (div w and w_y).
    if mfem_mesh_diag is not None:
        mesh_names = set(mfem_mesh_diag.dtype.names or ())
        if "time" not in mesh_names:
            print(f"Skipping mesh diagnostics plot: missing time column in {mfem_mesh_diag_csv}")
        else:
            tc_idxs = sorted(
                tc_index(n[len("divw_"):])
                for n in mesh_names
                if n.startswith("divw_TC") and tc_index(n[len("divw_"):]) > 0
            )
            tc_idxs = [i for i in tc_idxs if i > 0]
            if tc_idxs:
                chosen = [1, 2]
                deepest = tc_idxs[-1]
                if deepest not in chosen:
                    chosen.append(deepest)
                chosen = [i for i in chosen if i in tc_idxs]

                fig, (ax_div, ax_wy) = plt.subplots(
                    2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
                )
                colors = ["tab:blue", "tab:orange", "tab:red", "tab:green"]
                t_mesh = mfem_mesh_diag["time"]

                for k, idx in enumerate(chosen):
                    c = colors[k % len(colors)]
                    div_col = f"divw_TC{idx}"
                    wy_col = f"wy_TC{idx}"
                    if div_col in mesh_names:
                        ax_div.plot(
                            t_mesh,
                            mfem_mesh_diag[div_col],
                            lw=2,
                            color=c,
                            label=f"TC{idx}",
                        )
                    if wy_col in mesh_names:
                        ax_wy.plot(
                            t_mesh,
                            mfem_mesh_diag[wy_col],
                            lw=2,
                            color=c,
                            label=f"TC{idx}",
                        )

                if "divw_wall" in mesh_names:
                    ax_div.plot(
                        t_mesh,
                        mfem_mesh_diag["divw_wall"],
                        lw=1.5,
                        ls="--",
                        color="black",
                        label="wall",
                    )
                # For a strip with fixed bottom and receding top, the uniform-compression
                # kinematic prediction is div(w) ~= -v_rec / H(t).
                if probe_y and len(probe_y) >= 2:
                    H0 = float(max(probe_y) - min(probe_y))
                    if H0 > 0.0:
                        rec_t = mfem_mass_t
                        rec = mfem_recession
                        rec_rate = mfem_recession_rate
                        rec_i = np.interp(t_mesh, rec_t, rec)
                        rec_rate_i = np.interp(t_mesh, rec_t, rec_rate)
                        H_i = H0 - rec_i
                        with np.errstate(divide="ignore", invalid="ignore"):
                            divw_pred = -rec_rate_i / H_i
                        divw_pred = np.where(H_i > 1.0e-12, divw_pred, np.nan)
                        ax_div.plot(
                            t_mesh,
                            divw_pred,
                            lw=1.8,
                            ls=":",
                            color="black",
                            label=r"$-\,\dot r / (H_0-r)$",
                        )
                if "wy_wall" in mesh_names:
                    ax_wy.plot(
                        t_mesh,
                        mfem_mesh_diag["wy_wall"],
                        lw=1.5,
                        ls="--",
                        color="black",
                        label="wall",
                    )

                ax_div.set_ylabel(r"$\nabla \cdot w_{mesh}$ (1/s)")
                ax_div.grid(True, alpha=0.25)
                ax_div.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

                ax_wy.set_xlabel("Time (s)")
                ax_wy.set_ylabel(r"$w_y$ (m/s)")
                ax_wy.grid(True, alpha=0.25)
                ax_wy.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

                ax_div.set_xlim(0.0, float(np.nanmax(t_mesh)))
                ax_wy.set_xlim(0.0, float(np.nanmax(t_mesh)))

                mesh_diag_plot = out_dir / f"{args.out_prefix}_mesh_diagnostics.png"
                fig.savefig(mesh_diag_plot, dpi=180, bbox_inches="tight")
                plt.close(fig)
            else:
                print(
                    "Skipping mesh diagnostics plot: no divw_TC* columns found in "
                    f"{mfem_mesh_diag_csv}"
                )
    else:
        print(f"Skipping mesh diagnostics plot: missing {mfem_mesh_diag_csv}")

    # Plot 9: MFEM vs PATO pressure profiles p(y) at selected snapshot times.
    if (
        mfem_pressure_probes is not None
        and pato_p_time is not None
        and pato_p_vals is not None
        and len(pato_p_y) > 0
        and probe_y
        and pressure_profile_times
    ):
        mfem_p_names = list(mfem_pressure_probes.dtype.names or ())
        if "time" not in mfem_p_names:
            print(f"Skipping pressure profile plot: missing time column in {mfem_pressure_probes_csv}")
        else:
            mfem_y_to_series: Dict[float, np.ndarray] = {}
            for name in mfem_p_names:
                if name in ("time", "wall"):
                    continue
                idx = tc_index(name)
                if idx > 0 and idx < len(probe_y):
                    mfem_y_to_series[probe_y[idx]] = mfem_pressure_probes[name]

            matched_points: List[Tuple[float, int, np.ndarray]] = []
            if mfem_y_to_series:
                mfem_y_keys = list(mfem_y_to_series.keys())
                for j, y_pato in enumerate(pato_p_y):
                    k_best = min(mfem_y_keys, key=lambda y: abs(y - y_pato))
                    if abs(k_best - y_pato) <= 1.0e-8:
                        matched_points.append((y_pato, j, mfem_y_to_series[k_best]))

            if len(matched_points) < 2:
                print(
                    "Skipping pressure profile plot: insufficient matched probe points between "
                    f"{mfem_pressure_probes_csv} and {pato_pressure_plot}"
                )
            else:
                matched_points.sort(key=lambda x: x[0], reverse=True)
                n_snap = len(pressure_profile_times)
                ncols = 2 if n_snap > 1 else 1
                nrows = int(np.ceil(n_snap / ncols))
                fig, axes = plt.subplots(
                    nrows, ncols, figsize=(13, 4.2 * nrows), constrained_layout=True
                )
                axes_arr = np.atleast_1d(axes).ravel()
                mfem_ptime = mfem_pressure_probes["time"]

                for ax, t_target in zip(axes_arr, pressure_profile_times):
                    i_mf = int(np.argmin(np.abs(mfem_ptime - t_target)))
                    i_pato = int(np.argmin(np.abs(pato_p_time - t_target)))
                    y_vals = np.array([pt[0] for pt in matched_points], dtype=float)
                    mfem_vals = np.array([pt[2][i_mf] for pt in matched_points], dtype=float)
                    pato_vals = np.array([pato_p_vals[i_pato, pt[1]] for pt in matched_points], dtype=float)

                    ax.plot(mfem_vals, y_vals, "-o", color="tab:blue", lw=2, ms=4, label="MFEM")
                    ax.plot(pato_vals, y_vals, "--s", color="tab:orange", lw=2, ms=4, label="PATO")
                    ax.set_xlabel("Pressure (Pa)")
                    ax.set_ylabel("y (m)")
                    ax.grid(True, alpha=0.25)
                    ax.set_title(
                        f"target={t_target:g}s | MFEM={mfem_ptime[i_mf]:.3g}s | PATO={pato_p_time[i_pato]:.3g}s"
                    )
                    ax.legend(loc="best", fontsize=9)

                for ax in axes_arr[len(pressure_profile_times):]:
                    ax.axis("off")

                pato_pressure_profile_plot = (
                    out_dir / f"{args.out_prefix}_pressure_profiles_compare.png"
                )
                fig.savefig(pato_pressure_profile_plot, dpi=180, bbox_inches="tight")
                plt.close(fig)
    else:
        missing_parts = []
        if mfem_pressure_probes is None:
            missing_parts.append(str(mfem_pressure_probes_csv))
        if pato_p_time is None or pato_p_vals is None or len(pato_p_y) == 0:
            missing_parts.append(str(pato_pressure_plot))
        if missing_parts:
            print(
                "Skipping pressure profile plot: missing " + " and ".join(missing_parts)
            )

    # Plot 10: Common probe-stencil pressure slopes near the top (same discrete stencil in both codes).
    if (
        mfem_pressure_probes is not None
        and pato_p_time is not None
        and pato_p_vals is not None
        and len(pato_p_y) > 0
        and probe_y
    ):
        matched_p = match_pressure_probe_points(mfem_pressure_probes, probe_y, pato_p_y)
        n_pairs = min(3, max(0, len(matched_p) - 1))
        if n_pairs >= 1:
            t_mf_p = np.asarray(mfem_pressure_probes["time"], dtype=float)
            t_pa_p = np.asarray(pato_p_time, dtype=float)
            fig, axes = plt.subplots(
                n_pairs + 1,
                1,
                figsize=(13, 3.2 * (n_pairs + 1)),
                sharex=True,
                constrained_layout=True,
            )
            axes_arr = np.atleast_1d(axes).ravel()
            ratio_ax = axes_arr[-1]
            colors = ["tab:blue", "tab:orange", "tab:green"]

            t0 = max(float(np.nanmin(t_mf_p)), float(np.nanmin(t_pa_p)))
            t1 = min(float(np.nanmax(t_mf_p)), float(np.nanmax(t_pa_p)))
            common_mask = (t_mf_p >= t0) & (t_mf_p <= t1)
            t_common = t_mf_p[common_mask]

            for k in range(n_pairs):
                y_up, name_up, j_up, mfem_up = matched_p[k]
                y_dn, name_dn, j_dn, mfem_dn = matched_p[k + 1]
                dy = y_up - y_dn
                if abs(dy) <= 1.0e-14:
                    continue
                mfem_slope = (np.asarray(mfem_up, dtype=float) - np.asarray(mfem_dn, dtype=float)) / dy
                pato_slope = (np.asarray(pato_p_vals[:, j_up], dtype=float) - np.asarray(pato_p_vals[:, j_dn], dtype=float)) / dy

                ax = axes_arr[k]
                c = colors[k % len(colors)]
                ax.plot(t_mf_p, mfem_slope, lw=2, color=c, label=f"MFEM {name_up}-{name_dn}")
                ax.plot(t_pa_p, pato_slope, lw=2, ls="--", color=c, label="PATO")
                ax.set_ylabel("dp/dy (Pa/m)")
                ax.grid(True, alpha=0.25)
                depth_up = abs(matched_p[0][0] - y_up)
                depth_dn = abs(matched_p[0][0] - y_dn)
                ax.set_title(
                    f"Common stencil {name_up}-{name_dn} (depths {depth_up:.4g} m to {depth_dn:.4g} m)"
                )
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

                if t_common.size >= 2:
                    pato_slope_i = np.interp(t_common, t_pa_p, pato_slope)
                    mfem_slope_i = mfem_slope[common_mask]
                    ratio = safe_divide(mfem_slope_i, pato_slope_i)
                    ratio_ax.plot(
                        t_common,
                        ratio,
                        lw=2,
                        color=c,
                        label=f"{name_up}-{name_dn}",
                    )

            ratio_ax.axhline(1.0, color="black", lw=1.2, ls=":")
            ratio_ax.set_xlabel("Time (s)")
            ratio_ax.set_ylabel("MFEM / PATO")
            ratio_ax.set_title("Common probe-stencil pressure-slope ratio")
            ratio_ax.grid(True, alpha=0.25)
            ratio_ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            xmax_slope = max(float(np.nanmax(t_mf_p)), float(np.nanmax(t_pa_p)))
            for ax in axes_arr:
                ax.set_xlim(0.0, xmax_slope)

            pato_pressure_slope_plot = (
                out_dir / f"{args.out_prefix}_pressure_slope_stencils_compare.png"
            )
            fig.savefig(pato_pressure_slope_plot, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            print("Skipping pressure-slope stencil plot: insufficient matched pressure probes.")
    else:
        print("Skipping pressure-slope stencil plot: missing MFEM or PATO pressure probe data.")

    # Plot 11: MFEM mass-equation probe diagnostics (TC1/TC2/TC3).
    if mfem_mass_eq_probe is not None:
        mass_eq_names = set(mfem_mass_eq_probe.dtype.names or ())
        needed_cols = {"time"}
        for stem in ("pi_total", "gradp_y", "mflux_y"):
            for idx in (1, 2, 3):
                needed_cols.add(f"{stem}_TC{idx}")

        if needed_cols.issubset(mass_eq_names):
            t_meq = np.asarray(mfem_mass_eq_probe["time"], dtype=float)
            fig, axes = plt.subplots(
                3, 1, figsize=(13, 9.0), sharex=True, constrained_layout=True
            )
            colors = ["tab:blue", "tab:orange", "tab:green"]
            tc_ids = [1, 2, 3]

            series_specs = [
                ("pi_total", "pi_total", r"$\pi_{tot}$ (kg/m$^3$/s)"),
                ("gradp_y", "gradp_y", r"$\partial p / \partial y$ (Pa/m)"),
                ("mflux_y", "mflux_y", r"$m_{g,y}$ (kg/m$^2$/s)"),
            ]

            for ax, (key, title_key, ylabel) in zip(axes, series_specs):
                for c, idx in zip(colors, tc_ids):
                    col = f"{key}_TC{idx}"
                    ax.plot(
                        t_meq,
                        np.asarray(mfem_mass_eq_probe[col], dtype=float),
                        lw=2,
                        color=c,
                        label=f"TC{idx}",
                    )
                ax.set_ylabel(ylabel)
                ax.set_title(f"MFEM probe {title_key}: TC1/TC2/TC3")
                ax.grid(True, alpha=0.25)
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            axes[-1].set_xlabel("Time (s)")
            if t_meq.size:
                xmax = float(np.nanmax(t_meq))
                for ax in axes:
                    ax.set_xlim(0.0, xmax)

            mass_eq_probe_plot = out_dir / f"{args.out_prefix}_mass_eq_probe_terms.png"
            fig.savefig(mass_eq_probe_plot, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            missing = sorted(needed_cols - mass_eq_names)
            print(
                "Skipping MFEM mass-equation probe plot: missing columns in "
                f"{mfem_mass_eq_probe_csv}: {', '.join(missing)}"
            )
    else:
        print(f"Skipping MFEM mass-equation probe plot: missing {mfem_mass_eq_probe_csv}")

    # Plot 12: MFEM wall mass-equation probe terms vs PATO top-centerline diagnostics.
    if mfem_mass_eq_probe is not None and pato_surface_diag is not None:
        mfem_meq_names = set(mfem_mass_eq_probe.dtype.names or ())
        pato_names = set(pato_surface_diag.dtype.names or ())
        mfem_needed = {"time", "gradp_y_wall", "mflux_y_wall"}
        pato_needed = {"time", "snGradP_centerline", "mDotGw_centerline"}
        if mfem_needed.issubset(mfem_meq_names) and pato_needed.issubset(pato_names):
            t_mf = np.asarray(mfem_mass_eq_probe["time"], dtype=float)
            t_pa = np.asarray(pato_surface_diag["time"], dtype=float)
            fig, (ax_grad, ax_flux) = plt.subplots(
                2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
            )

            ax_grad.plot(
                t_mf,
                np.asarray(mfem_mass_eq_probe["gradp_y_wall"], dtype=float),
                lw=2,
                color="tab:blue",
                label="MFEM gradp_y_wall",
            )
            ax_grad.plot(
                t_pa,
                np.asarray(pato_surface_diag["snGradP_centerline"], dtype=float),
                lw=2,
                ls="--",
                color="tab:orange",
                label="PATO snGradP_centerline",
            )
            ax_grad.set_ylabel("Pressure gradient (Pa/m)")
            ax_grad.set_title("Wall pressure-gradient comparison (MFEM wall vs PATO top centerline)")
            ax_grad.grid(True, alpha=0.25)
            ax_grad.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            ax_flux.plot(
                t_mf,
                np.asarray(mfem_mass_eq_probe["mflux_y_wall"], dtype=float),
                lw=2,
                color="tab:blue",
                label="MFEM mflux_y_wall",
            )
            ax_flux.plot(
                t_pa,
                np.asarray(pato_surface_diag["mDotGw_centerline"], dtype=float),
                lw=2,
                ls="--",
                color="tab:orange",
                label="PATO mDotGw_centerline",
            )
            ax_flux.set_xlabel("Time (s)")
            ax_flux.set_ylabel(r"$m_g \cdot \hat{y}$ (kg/m$^2$/s)")
            ax_flux.set_title("Wall gas mass-flux comparison (MFEM wall vs PATO top centerline)")
            ax_flux.grid(True, alpha=0.25)
            ax_flux.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            xmax = max(float(np.nanmax(t_mf)), float(np.nanmax(t_pa)))
            ax_grad.set_xlim(0.0, xmax)
            ax_flux.set_xlim(0.0, xmax)

            mass_eq_wall_pato_plot = out_dir / f"{args.out_prefix}_wall_massflux_pressure_compare.png"
            fig.savefig(mass_eq_wall_pato_plot, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            missing_mf = sorted(mfem_needed - mfem_meq_names)
            missing_pa = sorted(pato_needed - pato_names)
            parts = []
            if missing_mf:
                parts.append(f"{mfem_mass_eq_probe_csv}: {', '.join(missing_mf)}")
            if missing_pa:
                parts.append(f"{pato_surface_diag_csv}: {', '.join(missing_pa)}")
            print("Skipping MFEM/PATO wall comparison plot: missing " + " | ".join(parts))
    elif mfem_mass_eq_probe is None or pato_surface_diag is None:
        missing_parts = []
        if mfem_mass_eq_probe is None:
            missing_parts.append(str(mfem_mass_eq_probe_csv))
        if pato_surface_diag is None:
            missing_parts.append(str(pato_surface_diag_csv))
        print("Skipping MFEM/PATO wall comparison plot: missing " + " and ".join(missing_parts))

    if len(mfem_by_depth) != len(am_by_depth):
        print("Probe-count mismatch: using nearest-to-surface shared count =", n_common)
        print("  MFEM probes:", len(mfem_by_depth), "Amaryllis probes:", len(am_by_depth))

    print(f"Wrote: {metrics_csv}")
    print(f"Wrote: {tol_csv}")
    print(f"Wrote: {fronts_csv}")
    print(f"Wrote: {out_dir / f'{args.out_prefix}_recession_comparison.png'}")
    if pato_diag_plot is not None:
        print(f"Wrote: {pato_diag_plot}")
    if pato_flux_recon_plot is not None:
        print(f"Wrote: {pato_flux_recon_plot}")
    if pato_flux_ratio_plot is not None:
        print(f"Wrote: {pato_flux_ratio_plot}")
    if pato_props_plot is not None:
        print(f"Wrote: {pato_props_plot}")
    if pato_pressure_profile_plot is not None:
        print(f"Wrote: {pato_pressure_profile_plot}")
    if pato_pressure_slope_plot is not None:
        print(f"Wrote: {pato_pressure_slope_plot}")
    if mesh_diag_plot is not None:
        print(f"Wrote: {mesh_diag_plot}")
    if mass_eq_probe_plot is not None:
        print(f"Wrote: {mass_eq_probe_plot}")
    if mass_eq_wall_pato_plot is not None:
        print(f"Wrote: {mass_eq_wall_pato_plot}")
    print(f"Overall PASS: {overall_pass}")
    print(f"Temperature PASS: {temp_pass}")
    print(f"Wall heating RMSE/max: {heat_rmse:.6g} / {heat_max:.6g}")
    print(f"Wall cooling RMSE/max: {cool_rmse:.6g} / {cool_max:.6g}")
    print(f"m_dot_g PASS: {mdot_pass}")
    print(f"Front 98 PASS: {front98_pass}")
    print(f"Front 2 PASS: {front2_pass}")
    print(f"m_dot_c PASS: {mdot_c_pass}")
    print(f"Recession PASS: {recession_pass}")
    if clamp_csv.exists():
        print(
            "B-prime clamp counts "
            f"(p, B'g, T): ({clamp_counts['pressure']:.0f}, "
            f"{clamp_counts['BprimeG']:.0f}, {clamp_counts['temperature']:.0f})"
        )

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"compare_ablation_case2_2.py: {exc}", file=sys.stderr)
        sys.exit(1)
