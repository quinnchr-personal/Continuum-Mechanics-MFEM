#!/usr/bin/env python3
"""Compare numerical and analytical solutions for ablation_qstar_blowing_1D."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROFILE_REQUIRED_COLS = (
    "x_m",
    "T_numeric_K",
    "T_exact_table_K",
    "T_exact_numerical_s_K",
    "abs_err_table_K",
)

SUMMARY_COLS = (
    "sdot_table_m_s",
    "sdot_numerical_m_s",
    "sdot_rel_error",
)


def ensure_structured_1d(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        raise RuntimeError("CSV file is empty.")
    if a.shape == ():
        return a.reshape(1)
    return a


def load_structured_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = np.genfromtxt(path, delimiter=",", names=True)
    return ensure_structured_1d(data)


def check_columns(data: np.ndarray, cols: tuple[str, ...], label: str) -> None:
    names = data.dtype.names or ()
    missing = [c for c in cols if c not in names]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def l2_trap(x: np.ndarray, err: np.ndarray) -> float:
    return float(np.sqrt(np.trapezoid(err * err, x)))


def metrics(x: np.ndarray, y_num: np.ndarray, y_ref: np.ndarray) -> Dict[str, float]:
    err = y_num - y_ref
    span = float(np.max(y_ref) - np.min(y_ref))
    linf = float(np.max(np.abs(err)))
    return {
        "linf_K": linf,
        "l2_trap_K_sqrt_m": l2_trap(x, err),
        "rms_nodes_K": float(np.sqrt(np.mean(err * err))),
        "mean_err_K": float(np.mean(err)),
        "linf_rel_to_span": linf / span if span > 0.0 else float("nan"),
    }


def format_metrics(name: str, m: Dict[str, float]) -> str:
    return (
        f"{name}: "
        f"Linf={m['linf_K']:.6e} K, "
        f"L2_trap={m['l2_trap_K_sqrt_m']:.6e} K*sqrt(m), "
        f"RMS={m['rms_nodes_K']:.6e} K, "
        f"mean={m['mean_err_K']:.6e} K, "
        f"Linf/span={m['linf_rel_to_span']:.6e}"
    )


def plot_profiles(
    x: np.ndarray,
    t_num: np.ndarray,
    t_tab: np.ndarray,
    t_num_exact: np.ndarray,
    out_dir: Path,
    out_prefix: str,
) -> tuple[Path, Path]:
    err_tab = t_num - t_tab
    err_num_exact = t_num - t_num_exact

    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    ax.plot(x, t_num, lw=2.0, label="Numerical")
    ax.plot(x, t_tab, "--", lw=1.8, label="Analytical (Table-4 sdot)")
    ax.plot(x, t_num_exact, ":", lw=2.0, label="Analytical (numerical sdot)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Q* Ablation Blowing Verification: Temperature Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()
    profile_png = out_dir / f"{out_prefix}_temperature_profile.png"
    fig.savefig(profile_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    tiny = np.finfo(float).tiny
    ax.semilogy(x, np.maximum(np.abs(err_tab), tiny), lw=2.0,
                label="|Numerical - Analytical(Table-4 sdot)|")
    ax.semilogy(x, np.maximum(np.abs(err_num_exact), tiny), "--", lw=2.0,
                label="|Numerical - Analytical(numerical sdot)|")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Absolute Error [K]")
    ax.set_title("Q* Ablation Blowing Verification: Temperature Error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    err_png = out_dir / f"{out_prefix}_temperature_error.png"
    fig.savefig(err_png, dpi=180)
    plt.close(fig)

    return profile_png, err_png


def maybe_load_summary(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    data = load_structured_csv(path)
    check_columns(data, SUMMARY_COLS, "Summary CSV")
    row = data[-1]
    return {
        "sdot_table_m_s": float(row["sdot_table_m_s"]),
        "sdot_numerical_m_s": float(row["sdot_numerical_m_s"]),
        "sdot_rel_error": float(row["sdot_rel_error"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="ParaView/qstar_ablation_blowing_1D",
        help="Directory containing qstar_blowing_profile.csv",
    )
    parser.add_argument(
        "--profile",
        default="qstar_blowing_profile.csv",
        help="Profile CSV filename inside --output-dir",
    )
    parser.add_argument(
        "--summary",
        default="qstar_blowing_summary.csv",
        help="Summary CSV filename inside --output-dir (optional)",
    )
    parser.add_argument(
        "--out-prefix",
        default="qstar_blowing_compare",
        help="Prefix for generated plot files",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    profile_path = out_dir / args.profile
    summary_path = out_dir / args.summary

    prof = load_structured_csv(profile_path)
    check_columns(prof, PROFILE_REQUIRED_COLS, "Profile CSV")

    x = np.asarray(prof["x_m"], dtype=float)
    t_num = np.asarray(prof["T_numeric_K"], dtype=float)
    t_tab = np.asarray(prof["T_exact_table_K"], dtype=float)
    t_num_exact = np.asarray(prof["T_exact_numerical_s_K"], dtype=float)

    if x.ndim != 1:
      raise RuntimeError("Expected 1D profile arrays.")
    if np.any(np.diff(x) < 0.0):
        raise RuntimeError("x values must be nondecreasing.")

    m_tab = metrics(x, t_num, t_tab)
    m_num_exact = metrics(x, t_num, t_num_exact)

    summary = maybe_load_summary(summary_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    profile_png, err_png = plot_profiles(x, t_num, t_tab, t_num_exact, out_dir, args.out_prefix)

    print("Q* ablation blowing comparison")
    print(f"  Profile CSV: {profile_path}")
    print(f"  Number of nodes: {x.size}")
    print(f"  x-range [m]: [{x.min():.6e}, {x.max():.6e}]")
    print(f"  Temperature range [K]: [{t_num.min():.6e}, {t_num.max():.6e}]")
    print(f"  {format_metrics('Numerical vs analytical (Table-4 sdot)', m_tab)}")
    print(f"  {format_metrics('Numerical vs analytical (numerical sdot)', m_num_exact)}")

    if summary is not None:
        print("  Recession-rate summary (last row):")
        print(f"    sdot_table [m/s]    = {summary['sdot_table_m_s']:.12e}")
        print(f"    sdot_numerical [m/s]= {summary['sdot_numerical_m_s']:.12e}")
        print(f"    rel error           = {summary['sdot_rel_error']:.12e}")
    else:
        print(f"  Summary CSV not found (skipped): {summary_path}")

    print(f"  Wrote plot: {profile_png}")
    print(f"  Wrote plot: {err_png}")


if __name__ == "__main__":
    main()

