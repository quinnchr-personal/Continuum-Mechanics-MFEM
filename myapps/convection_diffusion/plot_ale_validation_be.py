#!/usr/bin/env python3
"""Generate paper-style plots from `ale_validation_be` CSV outputs.

This script plots the available Backward-Euler ALE validation data produced by
`ale_validation_be.cpp`. It creates:

- Fig. 7.1-like stability plot (L2 norm vs time for multiple dt values)
- Fig. 7.3-like convergence plot (L2 error vs dt in log-log scale)
- Fig. 7.5-like accuracy plot (fixed grid vs moving-grid maps A/B)

Note: the original paper compares multiple time integrators and SCL-violating
variants. This driver produces BE-only data, so the plots are "paper-style"
equivalents for the available BE results.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def load_stability(csv_path: Path) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    rows = read_csv_rows(csv_path)
    groups: Dict[float, List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if not row:
            continue
        dt = float(row["dt"])
        t = float(row["time"])
        l2 = float(row["l2_norm"])
        groups[dt].append((t, l2))

    out: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for dt, pts in groups.items():
        # Keep last value per time if the CSV somehow contains duplicates.
        latest: Dict[float, float] = {}
        for t, l2 in pts:
            latest[t] = l2
        times = np.array(sorted(latest.keys()), dtype=float)
        vals = np.array([latest[t] for t in times], dtype=float)
        out[dt] = (times, vals)
    return out


def load_convergence(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(csv_path)
    dt_vals: List[float] = []
    err_vals: List[float] = []
    for row in rows:
        dt = float(row["dt"])
        err = float(row["l2_error"])
        if math.isfinite(dt) and math.isfinite(err) and dt > 0.0 and err > 0.0:
            dt_vals.append(dt)
            err_vals.append(err)
    if not dt_vals:
        raise ValueError(f"No valid convergence rows in {csv_path}")

    order = np.argsort(np.array(dt_vals))[::-1]  # descending dt
    dt = np.array(dt_vals, dtype=float)[order]
    err = np.array(err_vals, dtype=float)[order]
    return dt, err


def load_accuracy(csv_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    rows = read_csv_rows(csv_path)
    groups: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        map_name = str(row["map"]).strip()
        dt = float(row["dt"])
        err = float(row["l2_error"])
        if math.isfinite(dt) and math.isfinite(err) and dt > 0.0 and err > 0.0:
            groups[map_name].append((dt, err))

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, pts in groups.items():
        # Last write wins for duplicate dt values.
        latest: Dict[float, float] = {}
        for dt, err in pts:
            latest[dt] = err
        dt_sorted = np.array(sorted(latest.keys(), reverse=True), dtype=float)
        err_sorted = np.array([latest[dt] for dt in dt_sorted], dtype=float)
        out[name] = (dt_sorted, err_sorted)
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slope_reference(x: np.ndarray, y: np.ndarray, slope: float) -> np.ndarray:
    """Return a slope reference line anchored at the smallest dt point."""
    if x.size == 0 or y.size == 0:
        return np.array([], dtype=float)
    idx = int(np.argmin(x))
    x0 = x[idx]
    y0 = y[idx]
    return y0 * (x / x0) ** slope


def finite_pairs(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    return x[mask], y[mask]


def save_fig(fig: plt.Figure, path: Path, dpi: int, show: bool) -> None:
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Wrote plot: {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_stability(stability: Dict[float, Tuple[np.ndarray, np.ndarray]],
                   out_path: Path,
                   dpi: int,
                   show: bool) -> None:
    if not stability:
        raise ValueError("No stability data available.")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    for dt in sorted(stability.keys(), reverse=True):
        t, l2 = stability[dt]
        if t.size == 0:
            continue
        label = rf"$\Delta t={dt:g}$"
        ax.plot(t, l2, marker="o", markersize=3.2, linewidth=1.6, label=label)

    ax.set_xlabel("Time t")
    ax.set_ylabel(r"$\|u_h\|_{L^2(\Omega(t))}$")
    ax.set_title("Fig. 7.1-like Stability Plot (BE, ALE scaling map)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax.legend(loc="best", fontsize=9)

    save_fig(fig, out_path, dpi=dpi, show=show)


def plot_convergence(dt: np.ndarray,
                     err: np.ndarray,
                     out_path: Path,
                     dpi: int,
                     show: bool) -> None:
    dt, err = finite_pairs(dt, err)
    if dt.size == 0:
        raise ValueError("No valid convergence data to plot.")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.loglog(dt, err, marker="o", linewidth=1.8, markersize=5, label="BE (ALE)")

    ref = slope_reference(dt, err, slope=1.0)
    if ref.size:
        ax.loglog(dt, ref, "k--", linewidth=1.2, label="slope 1")

    # Observed order from first/last points (global estimate).
    if dt.size >= 2:
        p = math.log(err[0] / err[-1]) / math.log(dt[0] / dt[-1])
        ax.text(
            0.03,
            0.05,
            f"observed slope ~ {p:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel(r"Time step $\Delta t$")
    ax.set_ylabel(r"$L^2$ error at final time")
    ax.set_title("Fig. 7.3-like Convergence Plot (BE)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.legend(loc="best")

    save_fig(fig, out_path, dpi=dpi, show=show)


def _plot_accuracy_panel(ax: plt.Axes,
                         data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                         moving_key: str,
                         title: str) -> None:
    if "fixed" not in data:
        raise ValueError("Accuracy CSV must contain a 'fixed' map curve.")
    if moving_key not in data:
        raise ValueError(f"Accuracy CSV must contain '{moving_key}' curve.")

    dt_fixed, err_fixed = finite_pairs(*data["fixed"])
    dt_mov, err_mov = finite_pairs(*data[moving_key])
    if dt_fixed.size == 0 or dt_mov.size == 0:
        raise ValueError(f"Insufficient data for accuracy panel '{moving_key}'.")

    ax.loglog(
        dt_fixed, err_fixed, marker="o", linewidth=1.8, markersize=4.5,
        label="Fixed grid (baseline)"
    )
    ax.loglog(
        dt_mov, err_mov, marker="s", linewidth=1.8, markersize=4.5,
        label=f"Moving grid ({moving_key})"
    )

    ref = slope_reference(dt_fixed, err_fixed, slope=1.0)
    if ref.size:
        ax.loglog(dt_fixed, ref, "k--", linewidth=1.0, label="slope 1")

    ax.set_title(title)
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$L^2$ error")
    ax.grid(True, which="both", linestyle="--", linewidth=0.55, alpha=0.5)


def plot_accuracy_panels(data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                         out_path: Path,
                         dpi: int,
                         show: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    _plot_accuracy_panel(axes[0], data, "map_A", "Fig. 7.5-like Accuracy (Map A)")
    _plot_accuracy_panel(axes[1], data, "map_B", "Fig. 7.5-like Accuracy (Map B)")

    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate labels if slope line appears in both panels.
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    fig.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=9,
    )

    save_fig(fig, out_path, dpi=dpi, show=show)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create paper-style plots from ale_validation_be CSV outputs."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default="ParaView/ale_validation_be",
        help="Directory containing ale_validation_be CSV outputs.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory for output images (default: input-dir).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving them.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    ensure_dir(output_dir)

    stability_csv = input_dir / "stability_l2_history.csv"
    convergence_csv = input_dir / "convergence_errors.csv"
    accuracy_csv = input_dir / "accuracy_errors.csv"

    wrote_any = False

    if stability_csv.exists():
        stability = load_stability(stability_csv)
        if stability:
            plot_stability(
                stability,
                output_dir / "ale_validation_fig7_1_like_stability_be.png",
                dpi=args.dpi,
                show=args.show,
            )
            wrote_any = True
    else:
        print(f"Skipping stability plot (missing {stability_csv}).")

    if convergence_csv.exists():
        dt, err = load_convergence(convergence_csv)
        plot_convergence(
            dt,
            err,
            output_dir / "ale_validation_fig7_3_like_convergence_be.png",
            dpi=args.dpi,
            show=args.show,
        )
        wrote_any = True
    else:
        print(f"Skipping convergence plot (missing {convergence_csv}).")

    if accuracy_csv.exists():
        accuracy = load_accuracy(accuracy_csv)
        if accuracy:
            plot_accuracy_panels(
                accuracy,
                output_dir / "ale_validation_fig7_5_like_accuracy_be.png",
                dpi=args.dpi,
                show=args.show,
            )
            wrote_any = True
    else:
        print(f"Skipping accuracy plot (missing {accuracy_csv}).")

    if not wrote_any:
        raise SystemExit(
            "No ALE validation CSVs found. Run `ale_validation_be` first to generate data."
        )


if __name__ == "__main__":
    main()

