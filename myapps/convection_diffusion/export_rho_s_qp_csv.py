#!/usr/bin/env python3
"""Export quadrature-point rho_s diagnostics to a long-format CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pyvista as pv


DEFAULT_INPUT_YAML = Path("Input/input_ablation_case2_2.yaml")
DEFAULT_OUTPUT_NAME = "rho_s_qp_timeseries.csv"
TARGET_FIELD = "rho_s_qp"


def load_top_level_value_from_yaml(path: Path, key: str) -> str | None:
    if not path.exists():
        return None

    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or line.startswith(" "):
            continue
        if ":" not in stripped:
            continue
        k, v = stripped.split(":", 1)
        if k.strip() != key:
            continue
        value = v.strip()
        if not value:
            return None
        if value[0] == value[-1] and value[0] in {"'", '"'}:
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


def combine_point_field_datasets(ds, field: str):
    matches = []
    for _, leaf in iter_leaf_datasets(ds):
        point_data = getattr(leaf, "point_data", None)
        if point_data is None or field not in point_data:
            continue
        matches.append(leaf)

    if not matches:
        raise RuntimeError(f"Field {field!r} not found in the active ParaView dataset.")

    if len(matches) == 1:
        return matches[0]

    return pv.MultiBlock(matches).combine()


def read_qcloud_at_time(reader, time_value: float, has_time_values: bool):
    if has_time_values:
        reader.set_active_time_value(float(time_value))
    dataset = reader.read()
    return combine_point_field_datasets(dataset, TARGET_FIELD)


def resolve_output_csv(output_dir: Path, output_csv_override: Path | None) -> Path:
    if output_csv_override is not None:
        return output_csv_override.resolve()
    return (output_dir / DEFAULT_OUTPUT_NAME).resolve()


def export_rho_s_qp_csv(input_yaml: Path, output_dir: Path, output_csv: Path) -> tuple[int, int]:
    pvd_path = find_mfem_collection_pvd(output_dir, input_yaml)
    reader = pv.get_reader(str(pvd_path))
    raw_time_values = list(getattr(reader, "time_values", []))
    has_time_values = len(raw_time_values) > 0
    time_values = np.asarray(raw_time_values if has_time_values else [0.0], dtype=float)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "time_index", "qp_id", "x", "y", "z", TARGET_FIELD])

        for time_index, time_value in enumerate(time_values):
            qcloud = read_qcloud_at_time(reader, float(time_value), has_time_values)
            points = np.asarray(qcloud.points, dtype=float)
            values = np.asarray(qcloud.point_data[TARGET_FIELD], dtype=float).reshape(-1)

            if points.ndim != 2 or points.shape[1] != 3:
                raise RuntimeError(
                    f"Expected 3D point coordinates for {TARGET_FIELD}, got shape {points.shape}."
                )
            if values.size != points.shape[0]:
                raise RuntimeError(
                    f"Point/value size mismatch for {TARGET_FIELD} at time {time_value}: "
                    f"{points.shape[0]} points vs {values.size} values."
                )

            for qp_id, (coords, rho_s_qp) in enumerate(zip(points, values, strict=True)):
                writer.writerow(
                    [
                        float(time_value),
                        time_index,
                        qp_id,
                        float(coords[0]),
                        float(coords[1]),
                        float(coords[2]),
                        float(rho_s_qp),
                    ]
                )
                row_count += 1

    return time_values.size, row_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export the MFEM/ParaView quadrature-point rho_s diagnostic "
            "field (rho_s_qp) to a long-format CSV."
        )
    )
    parser.add_argument(
        "--input-yaml",
        type=Path,
        default=DEFAULT_INPUT_YAML,
        help="Input YAML used to resolve output_path and collection_name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the ParaView output directory; defaults to output_path from the YAML.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=f"Destination CSV path; defaults to <output_dir>/{DEFAULT_OUTPUT_NAME}.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_yaml = args.input_yaml.resolve()
    if not input_yaml.exists():
        raise FileNotFoundError(f"Input YAML not found: {input_yaml}")

    output_dir = resolve_output_dir(input_yaml, args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    output_csv = resolve_output_csv(output_dir, args.output_csv)
    step_count, row_count = export_rho_s_qp_csv(input_yaml, output_dir, output_csv)

    print(
        f"Wrote {row_count} rows across {step_count} time steps to {output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
