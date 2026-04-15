#!/usr/bin/env python3
"""Compare MFEM ablation case-2.2 recession and front outputs against Amaryllis."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-convection_diffusion")
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compare_ablation_case2_2 as common


DEFAULT_OUTPUT_DIR = Path("ParaView/ablation_case2_2")
DEFAULT_INPUT = Path("Input/input_ablation_case2_2.yaml")
DEFAULT_AMARYLLIS_MASS = Path(
    "/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/"
    "data/ref/PATO/PATO_Mass_TestCase_2.2.txt"
)
DEFAULT_OUT_PREFIX = "ablation_case2_2"
RECESSION_FRONT_TOL_KEYS = (
    "front98_rmse_max",
    "front98_max_abs_max",
    "front2_rmse_max",
    "front2_max_abs_max",
    "recession_rmse_max",
    "recession_final_rel_error_max",
)


def parse_yaml_bool(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory containing mass_metrics.csv and, optionally, front columns. "
            "If front columns are missing, the script samples fronts from the "
            "MFEM ParaView output."
        ),
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input YAML file with acceptance tolerances",
    )
    parser.add_argument(
        "--amaryllis-mass",
        default=str(DEFAULT_AMARYLLIS_MASS),
        help="Amaryllis mass reference file",
    )
    parser.add_argument(
        "--out-prefix",
        default=DEFAULT_OUT_PREFIX,
        help="Prefix for generated plot filenames",
    )
    parser.add_argument(
        "--max-time-steps",
        type=int,
        default=None,
        help=(
            "Maximum number of ParaView time steps to sample when falling back "
            "to MFEM .pvd output for fronts."
        ),
    )
    return parser


def write_tolerance_csv(path: Path, tol: dict[str, float]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["signal", "tolerance"])
        for key in RECESSION_FRONT_TOL_KEYS:
            writer.writerow([key, tol[key]])


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_time_steps is not None and args.max_time_steps < 1:
        parser.error("--max-time-steps must be >= 1")

    out_dir = Path(args.output_dir)
    mass_csv = out_dir / "mass_metrics.csv"
    if not mass_csv.exists():
        raise FileNotFoundError(f"Expected MFEM output not found: {mass_csv}")

    tol = common.load_acceptance_from_yaml(Path(args.input))
    mesh_is_moving = parse_yaml_bool(
        common.load_top_level_value_from_yaml(Path(args.input), "ale_enabled")
    )
    mass = common.load_named_csv(
        mass_csv, required=True, description="mass_metrics.csv"
    )
    am_mass = common.ensure_2d(np.loadtxt(args.amaryllis_mass, skiprows=1))

    t_ref = am_mass[:, 0]
    ref_front98 = am_mass[:, 3]
    ref_front2 = am_mass[:, 4]
    ref_recession = am_mass[:, 5]
    ref_recession_rate_full = common.time_derivative(t_ref, ref_recession)

    mfem_mass_t = mass["time"]
    mfem_recession = mass["recession"]
    mfem_recession_rate = common.time_derivative(mfem_mass_t, mfem_recession)

    mfem_front_source = "mass_metrics.csv"
    mfem_front_geometry: dict[str, np.ndarray | float] | None = None
    if (
        "front_98_virgin" in mass.dtype.names
        and "front_2_char" in mass.dtype.names
    ):
        mfem_front_t = mfem_mass_t
        mfem_front98 = mass["front_98_virgin"]
        mfem_front2 = mass["front_2_char"]
    else:
        mfem_front_source = "ParaView fallback"
        print(
            "front columns not found in mass_metrics.csv; "
            "sampling fronts from ParaView output instead."
        )
        (
            mfem_front_t,
            mfem_front98,
            mfem_front2,
            mfem_front_geometry,
        ) = common.sample_fronts_from_paraview(
            out_dir, Path(args.input), max_time_steps=args.max_time_steps
        )

    cmp_mask = common.overlap_mask_for_reference_time(
        t_ref, [mfem_mass_t, mfem_front_t], "recession/front comparison"
    )
    t_ref_cmp = t_ref[cmp_mask]
    ref_front98_cmp = ref_front98[cmp_mask]
    ref_front2_cmp = ref_front2[cmp_mask]
    ref_recession_cmp = ref_recession[cmp_mask]
    ref_recession_rate_cmp = ref_recession_rate_full[cmp_mask]

    mfem_mass_cmp_mask = (
        (mfem_mass_t >= (t_ref_cmp[0] - 1.0e-12))
        & (mfem_mass_t <= (t_ref_cmp[-1] + 1.0e-12))
    )
    mfem_front_cmp_mask = (
        (mfem_front_t >= (t_ref_cmp[0] - 1.0e-12))
        & (mfem_front_t <= (t_ref_cmp[-1] + 1.0e-12))
    )
    mfem_mass_t_cmp = mfem_mass_t[mfem_mass_cmp_mask]
    mfem_recession_cmp = mfem_recession[mfem_mass_cmp_mask]
    mfem_front_t_cmp = mfem_front_t[mfem_front_cmp_mask]
    mfem_front98_cmp = mfem_front98[mfem_front_cmp_mask]
    mfem_front2_cmp = mfem_front2[mfem_front_cmp_mask]

    mfem_front98_i = np.interp(t_ref_cmp, mfem_front_t_cmp, mfem_front98_cmp)
    mfem_front2_i = np.interp(t_ref_cmp, mfem_front_t_cmp, mfem_front2_cmp)
    mfem_recession_i = np.interp(t_ref_cmp, mfem_mass_t_cmp, mfem_recession_cmp)
    mfem_recession_rate_i = np.interp(t_ref_cmp, mfem_mass_t, mfem_recession_rate)

    front98_rmse = common.rmse(mfem_front98_i, ref_front98_cmp)
    front98_max = common.max_abs(mfem_front98_i, ref_front98_cmp)
    front2_rmse = common.rmse(mfem_front2_i, ref_front2_cmp)
    front2_max = common.max_abs(mfem_front2_i, ref_front2_cmp)
    recession_rmse = common.rmse(mfem_recession_i, ref_recession_cmp)
    recession_final_rel = abs(mfem_recession_i[-1] - ref_recession_cmp[-1]) / max(
        abs(ref_recession_cmp[-1]), 1.0e-12
    )

    front98_pass = (
        front98_rmse <= tol["front98_rmse_max"]
        and front98_max <= tol["front98_max_abs_max"]
    )
    front2_pass = (
        front2_rmse <= tol["front2_rmse_max"]
        and front2_max <= tol["front2_max_abs_max"]
    )
    recession_pass = (
        recession_rmse <= tol["recession_rmse_max"]
        and recession_final_rel <= tol["recession_final_rel_error_max"]
    )
    overall_pass = front98_pass and front2_pass and recession_pass

    metrics_csv = out_dir / "amaryllis_recession_front_metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["category", "signal", "rmse", "max_abs", "metric", "value", "tolerance", "pass"]
        )
        writer.writerow(
            [
                "front",
                "front_98_virgin",
                front98_rmse,
                front98_max,
                "rmse",
                front98_rmse,
                tol["front98_rmse_max"],
                int(front98_rmse <= tol["front98_rmse_max"]),
            ]
        )
        writer.writerow(
            [
                "front",
                "front_98_virgin",
                front98_rmse,
                front98_max,
                "max_abs",
                front98_max,
                tol["front98_max_abs_max"],
                int(front98_max <= tol["front98_max_abs_max"]),
            ]
        )
        writer.writerow(
            [
                "front",
                "front_2_char",
                front2_rmse,
                front2_max,
                "rmse",
                front2_rmse,
                tol["front2_rmse_max"],
                int(front2_rmse <= tol["front2_rmse_max"]),
            ]
        )
        writer.writerow(
            [
                "front",
                "front_2_char",
                front2_rmse,
                front2_max,
                "max_abs",
                front2_max,
                tol["front2_max_abs_max"],
                int(front2_max <= tol["front2_max_abs_max"]),
            ]
        )
        writer.writerow(
            [
                "recession",
                "recession",
                recession_rmse,
                "",
                "rmse",
                recession_rmse,
                tol["recession_rmse_max"],
                int(recession_rmse <= tol["recession_rmse_max"]),
            ]
        )
        writer.writerow(
            [
                "recession",
                "recession",
                "",
                "",
                "final_rel_error",
                recession_final_rel,
                tol["recession_final_rel_error_max"],
                int(recession_final_rel <= tol["recession_final_rel_error_max"]),
            ]
        )
        writer.writerow(["summary", "overall", "", "", "", "", "", int(overall_pass)])

    tol_csv = out_dir / "amaryllis_recession_front_tolerances.csv"
    write_tolerance_csv(tol_csv, tol)

    recession_front_label_mfem = (
        "MFEM recession front (2% char)" if mesh_is_moving else "MFEM 2% char"
    )
    recession_front_label_ref = (
        "Amaryllis recession front (2% char)"
        if mesh_is_moving
        else "Amaryllis 2% char"
    )

    fronts_csv = out_dir / "amaryllis_front_comparison.csv"
    with fronts_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time",
                "mfem_front_98_virgin",
                "amaryllis_front_98_virgin",
                "front_98_virgin_error",
                "mfem_front_2_char",
                "amaryllis_front_2_char",
                "front_2_char_error",
                "mfem_front_source",
            ]
        )
        for t_val, mfem_98, ref_98, mfem_2, ref_2 in zip(
            t_ref_cmp,
            mfem_front98_i,
            ref_front98_cmp,
            mfem_front2_i,
            ref_front2_cmp,
        ):
            writer.writerow(
                [
                    t_val,
                    mfem_98,
                    ref_98,
                    mfem_98 - ref_98,
                    mfem_2,
                    ref_2,
                    mfem_2 - ref_2,
                    mfem_front_source,
                ]
            )

    recession_csv = out_dir / "amaryllis_recession_comparison.csv"
    with recession_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time",
                "mfem_recession",
                "amaryllis_recession",
                "recession_error",
                "mfem_recession_rate",
                "amaryllis_recession_rate",
                "recession_rate_error",
            ]
        )
        for values in zip(
            t_ref_cmp,
            mfem_recession_i,
            ref_recession_cmp,
            mfem_recession_rate_i,
            ref_recession_rate_cmp,
        ):
            t_val, mfem_rec, ref_rec, mfem_rate, ref_rate = values
            writer.writerow(
                [
                    t_val,
                    mfem_rec,
                    ref_rec,
                    mfem_rec - ref_rec,
                    mfem_rate,
                    ref_rate,
                    mfem_rate - ref_rate,
                ]
            )

    front_geom_csv = None
    if mfem_front_geometry is not None:
        front_geom_csv = out_dir / "amaryllis_front_geometry.csv"
        with front_geom_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time",
                    "xmid",
                    "initial_y_top",
                    "y_top_live",
                    "y_bottom_live",
                    "y98",
                    "y2",
                    "rho_surface",
                    "front98_depth_initial",
                    "front2_depth_initial",
                    "front98_depth_live",
                    "front2_depth_live",
                ]
            )
            for i in range(len(mfem_front_geometry["time"])):
                writer.writerow(
                    [
                        mfem_front_geometry["time"][i],
                        mfem_front_geometry["xmid"],
                        mfem_front_geometry["initial_y_top"],
                        mfem_front_geometry["y_top_live"][i],
                        mfem_front_geometry["y_bottom_live"][i],
                        mfem_front_geometry["y98"][i],
                        mfem_front_geometry["y2"][i],
                        mfem_front_geometry["rho_surface"][i],
                        mfem_front_geometry["front98_depth_initial"][i],
                        mfem_front_geometry["front2_depth_initial"][i],
                        mfem_front_geometry["front98_depth_live"][i],
                        mfem_front_geometry["front2_depth_live"][i],
                    ]
                )

    fronts_plot = out_dir / f"{args.out_prefix}_fronts.png"
    plt.figure(figsize=(13, 4.8))
    plt.plot(mfem_front_t, mfem_front98, color="black", lw=2, label="MFEM 98% virgin")
    plt.plot(
        mfem_front_t,
        mfem_front2,
        color="gray",
        lw=2,
        label=recession_front_label_mfem,
    )
    plt.plot(t_ref, ref_front98, color="black", lw=2, ls="--", label="Amaryllis 98% virgin")
    plt.plot(
        t_ref,
        ref_front2,
        color="gray",
        lw=2,
        ls="--",
        label=recession_front_label_ref,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Depth (m)")
    plt.xlim(0.0, max(float(mfem_front_t[-1]), float(t_ref[-1])))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
    plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    plt.savefig(fronts_plot, dpi=180, bbox_inches="tight")
    plt.close()

    front_geom_plot = None
    if mfem_front_geometry is not None:
        geom_t = np.asarray(mfem_front_geometry["time"], dtype=float)
        y_top_live = np.asarray(mfem_front_geometry["y_top_live"], dtype=float)
        y_bottom_live = np.asarray(mfem_front_geometry["y_bottom_live"], dtype=float)
        y98 = np.asarray(mfem_front_geometry["y98"], dtype=float)
        y2 = np.asarray(mfem_front_geometry["y2"], dtype=float)
        initial_y_top = float(mfem_front_geometry["initial_y_top"])

        plt.figure(figsize=(13, 5.2))
        plt.plot(
            geom_t,
            np.full_like(geom_t, initial_y_top),
            color="black",
            lw=1.8,
            ls="--",
            label="Initial top surface",
        )
        plt.plot(geom_t, y_top_live, color="tab:blue", lw=2, label="Live top surface")
        plt.plot(
            geom_t,
            y_bottom_live,
            color="tab:blue",
            lw=1.4,
            ls=":",
            label="Live bottom surface",
        )
        plt.plot(geom_t, y98, color="tab:red", lw=2, label="y98 crossing")
        plt.plot(
            geom_t,
            y2,
            color="tab:orange",
            lw=2,
            label="Recession front (2% char)" if mesh_is_moving else "y2 crossing",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("y coordinate (m)")
        plt.xlim(0.0, float(geom_t[-1]))
        plt.grid(True, alpha=0.25)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
        plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
        front_geom_plot = out_dir / f"{args.out_prefix}_front_geometry.png"
        plt.savefig(front_geom_plot, dpi=180, bbox_inches="tight")
        plt.close()

    recession_plot = out_dir / f"{args.out_prefix}_recession_comparison.png"
    fig, (ax_rec, ax_rate) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
    )
    ax_rec.plot(mfem_mass_t, mfem_recession, color="black", lw=2, label="MFEM")
    ax_rec.plot(t_ref, ref_recession, color="black", lw=2, ls="--", label="Amaryllis")
    if mesh_is_moving:
        ax_rec.plot(
            mfem_front_t,
            mfem_front2,
            color="tab:orange",
            lw=1.8,
            label="MFEM recession front (2% char)",
        )
        ax_rec.plot(
            t_ref,
            ref_front2,
            color="tab:orange",
            lw=1.8,
            ls="--",
            label="Amaryllis recession front (2% char)",
        )
    ax_rec.set_ylabel("Recession (m)")
    ax_rec.grid(True, alpha=0.25)
    ax_rec.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    ax_rate.plot(mfem_mass_t, mfem_recession_rate, color="black", lw=2, label="MFEM")
    ax_rate.plot(t_ref, ref_recession_rate_full, color="black", lw=2, ls="--", label="Amaryllis")
    ax_rate.set_xlabel("Time (s)")
    ax_rate.set_ylabel("Recession rate (m/s)")
    ax_rate.grid(True, alpha=0.25)

    xmax = max(float(mfem_mass_t[-1]), float(t_ref[-1]))
    ax_rec.set_xlim(0.0, xmax)
    ax_rate.set_xlim(0.0, xmax)
    fig.savefig(recession_plot, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {metrics_csv}")
    print(f"Wrote: {tol_csv}")
    print(f"Wrote: {fronts_csv}")
    print(f"Wrote: {recession_csv}")
    if front_geom_csv is not None:
        print(f"Wrote: {front_geom_csv}")
    print(f"Wrote: {fronts_plot}")
    print(f"Wrote: {recession_plot}")
    if front_geom_plot is not None:
        print(f"Wrote: {front_geom_plot}")
    if mesh_is_moving:
        print("Using 2% char as the plotted recession-front curve because ale_enabled=true.")
    print(f"Front 98 PASS: {front98_pass}")
    print(f"Front 2 PASS: {front2_pass}")
    print(f"Recession PASS: {recession_pass}")
    print(f"Overall PASS: {overall_pass}")

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"{Path(__file__).name}: {exc}", file=sys.stderr)
        sys.exit(1)
