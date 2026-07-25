#!/usr/bin/env python3
"""Validate and generate the checked M3 Exasim reference products.

The reassembly lives in exasim_reference.cpp so this front end uses the same
parser and Exasim unordered-map face ordering as the acceptance executable.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def run(command: list[str], cwd: pathlib.Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    app = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="/home/quinnchr/dv/Exasim-apps/runs/nsmach8-baseline",
        help="read-only Exasim reference run",
    )
    parser.add_argument(
        "--generate-wall",
        action="store_true",
        help="also solve the baseline and emit both wall CSVs",
    )
    args = parser.parse_args()

    run(["make", "test_m3_reference", "hdg_ns_driver"], app)
    run(["./test_m3_reference", args.run_dir], app)
    if args.generate_wall:
        run(
            [
                "./hdg_ns_driver",
                "-i",
                "Input/input_mach8_baseline.yaml",
                "--acceptance",
            ],
            app,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(f"reference tooling failed: {error}", file=sys.stderr)
        raise SystemExit(error.returncode)
