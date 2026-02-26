#!/usr/bin/env python3
"""Plot L2 error versus time from an error_history.csv file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def read_error_history(csv_path: Path) -> tuple[List[float], List[float]]:
    times: List[float] = []
    l2_vals: List[float] = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"time", "l2_error"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{csv_path} must contain columns {sorted(required)}; "
                f"got {reader.fieldnames}"
            )

        for row in reader:
            times.append(float(row["time"]))
            l2_vals.append(float(row["l2_error"]))

    if not times:
        raise ValueError(f"{csv_path} contains no data rows.")

    return times, l2_vals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "csv",
        nargs="?",
        default="ParaView/diffusion_mms_ale/error_history.csv",
        help="Path to error_history.csv (default: %(default)s)",
    )
    p.add_argument(
        "-o",
        "--out",
        help="Optional output image path (e.g. l2_vs_time.png). If omitted, shows interactively.",
    )
    p.add_argument(
        "--logy",
        action="store_true",
        help="Use a logarithmic y-axis for L2 error.",
    )
    p.add_argument(
        "--title",
        default="L2 Error vs Time",
        help="Plot title (default: %(default)s)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI when saving (default: %(default)s)",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    times, l2_vals = read_error_history(csv_path)

    if args.logy:
        filtered = [(t, e) for t, e in zip(times, l2_vals) if e > 0.0]
        if not filtered:
            raise ValueError("Cannot use --logy: no positive L2 error values found.")
        if len(filtered) != len(times):
            dropped = len(times) - len(filtered)
            print(f"Skipping {dropped} non-positive L2 values for log-scale plot.")
        times, l2_vals = map(list, zip(*filtered))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(times, l2_vals, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("L2 Error")
    ax.set_title(args.title)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    if args.logy:
        ax.set_yscale("log")

    fig.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
