#!/usr/bin/env python3
"""Check the generated M3 comparison report and wall CSV products."""

from __future__ import annotations

import argparse
import csv
import pathlib
import re


def report_value(text: str, label: str) -> float:
    match = re.search(
        rf"^\| {re.escape(label)} \| ([^|]+) \|", text, re.MULTILINE
    )
    if not match:
        raise RuntimeError(f"missing report row: {label}")
    return float(match.group(1).strip())


def read_wall(path: pathlib.Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    if len(rows) != 105:
        raise RuntimeError(f"{path}: expected 105 wall samples, found {len(rows)}")
    return rows


def main() -> int:
    app = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", type=pathlib.Path, default=app / "comparison_report.md"
    )
    parser.add_argument(
        "--hdg-wall", type=pathlib.Path, default=app / "wall_data.csv"
    )
    parser.add_argument(
        "--exasim-wall",
        type=pathlib.Path,
        default=app / "wall_data_exasim.csv",
    )
    args = parser.parse_args()

    text = args.report.read_text()
    field_labels = [
        "rho relative L2",
        "rhou relative L2",
        "rhov relative L2",
        "rhoE relative L2",
    ]
    field_values = [report_value(text, label) for label in field_labels]
    cp = report_value(text, "wall Cp max relative difference")
    heat = report_value(text, "wall Fint heat-flux max relative difference")
    shock = report_value(text, "shock-standoff absolute difference")
    hdg_wall = read_wall(args.hdg_wall)
    exasim_wall = read_wall(args.exasim_wall)
    coordinate_error = max(
        abs(first[key] - second[key])
        for first, second in zip(hdg_wall, exasim_wall)
        for key in ("theta", "x", "y")
    )

    if max(field_values) > 1.0e-5:
        raise RuntimeError("field relative-L2 gate failed")
    if cp > 1.0e-4:
        raise RuntimeError("wall Cp gate failed")
    if heat > 1.0e-3:
        raise RuntimeError("wall heat-flux gate failed")
    if coordinate_error > 1.0e-13:
        raise RuntimeError("wall sample coordinates differ")
    print(
        "PASS generated M3 comparison:"
        f" max_field_L2={max(field_values):.17g}"
        f" Cp={cp:.17g} Fint={heat:.17g}"
        f" shock_abs_diff={shock:.17g}"
        f" wall_coordinate_error={coordinate_error:.17g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
