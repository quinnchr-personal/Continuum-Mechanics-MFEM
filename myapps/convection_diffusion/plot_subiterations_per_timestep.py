#!/usr/bin/env python3
"""Plot interface sub-iterations used per time step from time_step_history.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


def read_subiterations_per_step(
    csv_path: Path,
) -> Tuple[List[int], List[int], Optional[List[int]]]:
    required = {"step", "subiters_used"}
    steps: List[int] = []
    subiters: List[int] = []
    converged: List[int] = []

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

        has_converged_col = "step_converged" in reader.fieldnames

        for row in reader:
            if not row.get("step", "").strip():
                continue
            steps.append(int(float(row["step"])))
            subiters.append(int(float(row["subiters_used"])))
            if has_converged_col:
                converged.append(int(float(row["step_converged"])))

    if not steps:
        raise ValueError(f"No data rows found in {csv_path}")

    return steps, subiters, (converged if converged else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        default="ParaView/two_domain_poisson_coupled_transient/time_step_history.csv",
        help="Path to time_step_history.csv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ParaView/two_domain_poisson_coupled_transient/subiterations_per_timestep.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Interface Sub-Iterations Per Time Step",
        help="Plot title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot window interactively.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    steps, subiters, converged = read_subiterations_per_step(input_path)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(steps, subiters, marker="o", linewidth=1.6, markersize=4, label="subiters_used")

    if converged is not None:
        unconverged_x = [s for s, c in zip(steps, converged) if c == 0]
        unconverged_y = [n for n, c in zip(subiters, converged) if c == 0]
        if unconverged_x:
            ax.scatter(
                unconverged_x,
                unconverged_y,
                marker="x",
                s=64,
                linewidths=1.8,
                color="red",
                label="unconverged step",
            )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Sub-Iterations Used")
    ax.set_title(args.title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_ylim(bottom=0)

    y_max = max(subiters)
    if y_max <= 30:
        ax.set_yticks(range(0, y_max + 2))
    if converged is not None and any(c == 0 for c in converged):
        ax.legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Wrote plot: {output_path}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
