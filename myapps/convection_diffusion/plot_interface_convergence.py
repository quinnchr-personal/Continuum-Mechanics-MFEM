#!/usr/bin/env python3
"""Plot coupled-interface convergence criteria from interface_iteration_history.csv."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


MetricSeries = Tuple[str, str, List[int], List[float]]


def _parse_optional_yaml_tolerances(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}

    tolerances: Dict[str, float] = {}
    key_map = {
        "coupling_tol": "rel_change",
        "flux_tol": "flux_jump_l2_rel",
        "flux_change_tol": "rel_flux_change",
    }

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.split("#", 1)[0].strip()
        if key not in key_map:
            continue
        try:
            tolerances[key_map[key]] = float(val)
        except ValueError:
            continue

    return tolerances


def _safe_float(cell: str) -> float:
    return float(cell.strip())


def read_convergence_history(csv_path: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        if "iter" not in reader.fieldnames:
            raise ValueError(f"CSV missing required column 'iter'. Found: {reader.fieldnames}")

        metrics: Dict[str, List[float]] = {name: [] for name in reader.fieldnames if name != "iter"}
        iters: List[int] = []

        for row in reader:
            if not row.get("iter", "").strip():
                continue
            iters.append(int(float(row["iter"])))
            for name in metrics:
                cell = row.get(name, "").strip()
                metrics[name].append(_safe_float(cell) if cell else math.nan)

    if not iters:
        raise ValueError(f"No data rows found in {csv_path}")

    return iters, metrics


def select_metrics(iters: List[int], metrics: Dict[str, List[float]]) -> List[MetricSeries]:
    labels = [
        ("rel_change", "rel_change (temperature iterate)"),
        ("flux_jump_l2_rel", "flux_jump_l2_rel (interface balance)"),
        ("rel_flux_change", "rel_flux_change (iteration-to-iteration flux)"),
        ("temp_jump_avg", "temp_jump_avg"),
    ]

    series: List[MetricSeries] = []
    for key, label in labels:
        if key not in metrics:
            continue
        x_vals: List[int] = []
        y_vals: List[float] = []
        for x, y in zip(iters, metrics[key]):
            if not math.isfinite(y) or y <= 0.0:
                continue
            x_vals.append(x)
            y_vals.append(y)
        if y_vals:
            series.append((key, label, x_vals, y_vals))
    if not series:
        raise ValueError(
            "No plottable positive finite convergence columns found. "
            "Expected one or more of: rel_change, flux_jump_l2_rel, rel_flux_change, temp_jump_avg."
        )
    return series


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        default="ParaView/two_domain_poisson_coupled/interface_iteration_history.csv",
        help="Path to interface_iteration_history.csv.",
    )
    parser.add_argument(
        "-y",
        "--input-yaml",
        default="Input/input_two_domain_poisson_coupled.yaml",
        help="Optional YAML input file for tolerance lines.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ParaView/two_domain_poisson_coupled/convergence_criteria.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Interface Coupling Convergence Criteria",
        help="Plot title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot window.",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    yaml_path = Path(args.input_yaml) if args.input_yaml else None
    out_path = Path(args.output)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    iters, metrics = read_convergence_history(csv_path)
    series = select_metrics(iters, metrics)
    tolerances = _parse_optional_yaml_tolerances(yaml_path)

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for _, label, xs, ys in series:
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=label)

    for key, tol in tolerances.items():
        if key in {s[0] for s in series} and tol > 0.0:
            ax.axhline(tol, linestyle="--", linewidth=1.0, alpha=0.7, label=f"{key} tol")

    ax.set_yscale("log")
    ax.set_xlabel("Coupling iteration")
    ax.set_ylabel("Metric value (log scale)")
    ax.set_title(args.title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote plot: {out_path}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
