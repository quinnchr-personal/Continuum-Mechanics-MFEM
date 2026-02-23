#!/usr/bin/env python3
"""Plot Newton iteration counts per step from a Newton history CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_iterations_per_step(csv_path: Path):
    required = {"step", "time", "iter", "update_norm"}
    per_step = {}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            if not row["step"]:
                continue
            step = int(float(row["step"]))
            iter_idx = int(float(row["iter"]))
            time_val = float(row["time"])
            update_norm = float(row["update_norm"])

            data = per_step.setdefault(
                step, {"max_iter": -1, "conv_iter": None, "time": time_val}
            )
            data["time"] = time_val
            data["max_iter"] = max(data["max_iter"], iter_idx)

            # Converged rows are written with update_norm = 0 in this driver.
            if abs(update_norm) <= 1.0e-14:
                if data["conv_iter"] is None or iter_idx < data["conv_iter"]:
                    data["conv_iter"] = iter_idx

    if not per_step:
        raise ValueError(f"No data rows found in {csv_path}")

    steps = sorted(per_step.keys())
    iters = []
    for step in steps:
        data = per_step[step]
        if data["conv_iter"] is not None:
            iters.append(data["conv_iter"])
        else:
            # Fallback for incomplete logs without explicit converged row.
            iters.append(data["max_iter"] + 1)
    return steps, iters


def main():
    parser = argparse.ArgumentParser(
        description="Plot Newton iterations per step from CSV."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="ParaView/newton_history_nonlinear_1D.csv",
        help="Path to Newton history CSV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ParaView/newton_iterations_per_step.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Newton Iterations Per Step",
        help="Plot title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window interactively.",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    output_path = Path(args.output)

    steps, iters = read_iterations_per_step(csv_path)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(steps, iters, marker="o", linewidth=1.6, markersize=4)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Newton Iterations")
    ax.set_title(args.title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    y_max = max(iters)
    ax.set_ylim(bottom=0, top=max(1, y_max + 1))
    if y_max <= 20:
        ax.set_yticks(range(0, y_max + 2))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Wrote plot: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
