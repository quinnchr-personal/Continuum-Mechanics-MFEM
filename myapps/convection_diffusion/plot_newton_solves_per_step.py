#!/usr/bin/env python3
"""Plot Newton solver counts per time step from a Newton history CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _parse_int(cell: str) -> int:
    return int(float(cell))


def read_newton_solves_per_step(csv_path: Path) -> Tuple[List[int], List[int]]:
    required = {"step", "iter"}
    per_step: Dict[int, Dict[str, int]] = {}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        has_converged_col = "converged" in reader.fieldnames

        for row in reader:
            if not row["step"] or not row["iter"]:
                continue

            step = _parse_int(row["step"])
            iter_idx = _parse_int(row["iter"])
            state = per_step.setdefault(step, {"max_iter": -1, "conv_iter": -1})
            state["max_iter"] = max(state["max_iter"], iter_idx)

            if has_converged_col and row.get("converged", "").strip() == "1":
                # In this logger, converged iteration index equals Newton solve count.
                if state["conv_iter"] < 0:
                    state["conv_iter"] = iter_idx
                else:
                    state["conv_iter"] = min(state["conv_iter"], iter_idx)

    if not per_step:
        raise ValueError(f"No data rows found in {csv_path}")

    steps = sorted(per_step.keys())
    solves: List[int] = []
    for step in steps:
        state = per_step[step]
        if state["conv_iter"] >= 0:
            solves.append(state["conv_iter"])
        else:
            # No explicit converged row: all logged iterations represent solves.
            solves.append(state["max_iter"] + 1)

    return steps, solves


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Newton solver counts per time step from CSV."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="ParaView/ablation_case2_1/newton_history_ablation_case2_1_2D.csv",
        help="Path to Newton history CSV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ParaView/ablation_case2_1/newton_solves_per_step.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Newton Solver Count Per Step",
        help="Plot title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window interactively.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    steps, solves = read_newton_solves_per_step(input_path)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(steps, solves, marker="o", linewidth=1.6, markersize=3)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Newton Solves")
    ax.set_title(args.title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_ylim(bottom=0)

    y_max = max(solves)
    if y_max <= 25:
        ax.set_yticks(range(0, y_max + 1))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Wrote plot: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
