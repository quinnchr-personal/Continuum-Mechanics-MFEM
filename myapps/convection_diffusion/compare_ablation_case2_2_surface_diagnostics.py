#!/usr/bin/env python3
"""Compare MFEM and PATO surface diagnostics for ablation case 2.2."""

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
DEFAULT_PATO_SURFACE_DIAGNOSTICS = Path(
    "/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/output/porousMat/"
    "top_surface_diagnostics.csv"
)
DEFAULT_OUT_PREFIX = "ablation_case2_2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory used for default MFEM input and generated plots.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input YAML used to resolve the ParaView collection and gravity settings.",
    )
    parser.add_argument(
        "--mfem-boundary-diagnostics",
        default=None,
        help=(
            "Override the MFEM boundary diagnostics CSV. "
            "Defaults to <output-dir>/boundary_diagnostics.csv."
        ),
    )
    parser.add_argument(
        "--force-paraview",
        action="store_true",
        help="Ignore MFEM boundary_diagnostics.csv and reconstruct from VTU/PVTU files.",
    )
    parser.add_argument(
        "--pato-surface-diagnostics",
        default=str(DEFAULT_PATO_SURFACE_DIAGNOSTICS),
        help="PATO top_surface_diagnostics.csv file.",
    )
    parser.add_argument(
        "--out-prefix",
        default=DEFAULT_OUT_PREFIX,
        help="Prefix for generated plot filenames.",
    )
    parser.add_argument(
        "--max-time-steps",
        type=int,
        default=None,
        help="Maximum number of ParaView time steps to sample in VTU fallback mode.",
    )
    return parser


def finite_time_max(*arrays: np.ndarray) -> float:
    max_val = 0.0
    found = False
    for arr in arrays:
        vals = np.asarray(arr, dtype=float)
        mask = np.isfinite(vals)
        if np.any(mask):
            max_val = max(max_val, float(np.max(vals[mask])))
            found = True
    if not found:
        raise RuntimeError("No finite time values were found for plotting.")
    return max_val


def reconstruct_mfem_surface_diagnostics_from_paraview(
    output_dir: Path,
    input_yaml: Path,
    *,
    max_time_steps: int | None = None,
) -> np.ndarray:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "pyvista is not available for VTU/PVTU-based surface-diagnostics reconstruction."
        ) from exc

    pvd_path = common.find_mfem_collection_pvd(output_dir, input_yaml)
    if pvd_path is None:
        raise FileNotFoundError(
            f"No MFEM ParaView .pvd collection was found under {output_dir}"
        )

    cache_path = output_dir / "surface_diagnostics_paraview.csv"
    if max_time_steps is None and cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        input_mtime = input_yaml.stat().st_mtime if input_yaml.exists() else 0.0
        if cache_mtime >= max(pvd_path.stat().st_mtime, input_mtime):
            cached = common.load_named_csv(
                cache_path,
                required=False,
                description="surface_diagnostics_paraview.csv",
            )
            if cached is not None:
                required_cols = {
                    "time",
                    "m_dot_g_surf",
                    "m_dot_g_centerline",
                    "gradp_n_centerline",
                    "mobility_centerline",
                    "rho_g_centerline",
                }
                cached_names = set(cached.dtype.names or ())
                if required_cols.issubset(cached_names):
                    print(
                        f"Using cached VTU surface diagnostics from {cache_path}",
                        flush=True,
                    )
                    return cached

    time_entries = common.filter_available_time_entries(
        common.load_pvd_time_entries(pvd_path), "mesh", str(pvd_path)
    )
    if max_time_steps is not None:
        time_entries = time_entries[:max_time_steps]
        if not time_entries:
            raise RuntimeError(
                "No ParaView time steps remain after applying --max-time-steps."
            )

    gx, gy, _ = common.load_gravity_vector_from_yaml(input_yaml)
    if abs(gx) > 1.0e-14:
        print(
            "Warning: VTU reconstruction assumes the top-boundary normal is aligned "
            "with +y; gravity_x is ignored.",
            flush=True,
        )

    rows: list[tuple[float, float, float, float, float, float]] = []
    skipped = 0
    for i_time, (time_value, entry) in enumerate(time_entries):
        gas_path = entry.get("gas_density_qp")
        mob_path = entry.get("mobility_qp")
        flux_path = entry.get("m_dot_g_qp")
        if (
            gas_path is None
            or mob_path is None
            or flux_path is None
            or not gas_path.exists()
            or not mob_path.exists()
            or not flux_path.exists()
        ):
            skipped += 1
            continue

        common.print_progress(
            "Surface diagnostic reconstruction", i_time + 1, len(time_entries)
        )
        mesh = common.select_dataset_with_point_fields(pv.read(entry["mesh"]), "pressure")
        if mesh is None:
            raise RuntimeError(f"Missing pressure field in {entry['mesh']}")
        mesh = common.warp_dataset_points_by_vector_field(mesh, "ale_displacement")
        deriv = mesh.compute_derivative(scalars="pressure", gradient=True)

        gas_qcloud = common.read_qpoint_dataset(entry, "gas_density_qp", pv)
        mob_qcloud = common.read_qpoint_dataset(entry, "mobility_qp", pv)
        flux_qcloud = common.read_qpoint_dataset(entry, "m_dot_g_qp", pv)
        if gas_qcloud is None or mob_qcloud is None or flux_qcloud is None:
            skipped += 1
            continue

        gas_x, gas_y, gas_rho = common.extract_top_line_profile_from_qcloud(
            gas_qcloud, "gas_density_qp"
        )
        mob_x, mob_y, mobility_base = common.extract_top_line_profile_from_qcloud(
            mob_qcloud, "mobility_qp"
        )
        flux_x, flux_y, flux_direct_top = common.extract_top_line_profile_from_qcloud(
            flux_qcloud, "m_dot_g_qp"
        )

        x_min, x_max, _, y_top_live, _, _ = mesh.bounds
        x_min = float(x_min)
        x_max = float(x_max)
        y_top_live = float(y_top_live)
        xmid = 0.5 * (x_min + x_max)

        y_line = min(
            float(np.mean(gas_y)),
            float(np.mean(mob_y)),
            float(np.mean(flux_y)),
        )
        y_eps = 1.0e-9 * max(1.0, abs(y_top_live))
        y_line = min(y_line, y_top_live - y_eps)

        n_line = max(
            41,
            10 * max(len(gas_x), len(mob_x), len(flux_x)) + 1,
        )
        x_line = np.linspace(x_min, x_max, n_line, dtype=float)
        query_pts = np.column_stack(
            [x_line, np.full_like(x_line, y_line), np.zeros_like(x_line)]
        )
        sampled = pv.PolyData(query_pts).sample(deriv)

        valid = np.asarray(
            sampled.point_data.get("vtkValidPointMask", np.ones(x_line.size)),
            dtype=int,
        ).reshape(-1)
        grad = np.asarray(sampled.point_data.get("gradient", []), dtype=float)
        if grad.ndim != 2 or grad.shape[0] != x_line.size or grad.shape[1] < 2:
            raise RuntimeError(
                f"Could not sample pressure gradient from {entry['mesh']}"
            )

        gradp_n = np.full(x_line.size, np.nan, dtype=float)
        gradp_n[valid > 0] = grad[:, 1][valid > 0]

        rho_line = np.interp(
            x_line, gas_x, gas_rho, left=float(gas_rho[0]), right=float(gas_rho[-1])
        )
        mobility_line = np.interp(
            x_line,
            mob_x,
            mobility_base,
            left=float(mobility_base[0]),
            right=float(mobility_base[-1]),
        )
        direct_line = np.interp(
            x_line,
            flux_x,
            flux_direct_top,
            left=float(flux_direct_top[0]),
            right=float(flux_direct_top[-1]),
        )

        # mobility_qp stores K/mu; multiply by rho_g to recover rho_g K/mu.
        rho_darcy_line = rho_line * mobility_line
        recon_line = -rho_darcy_line * gradp_n + (rho_line * rho_darcy_line) * gy

        direct_valid = np.isfinite(direct_line)
        recon_valid = np.isfinite(recon_line)
        if np.count_nonzero(direct_valid) < 2 or np.count_nonzero(recon_valid) < 2:
            skipped += 1
            continue

        mdot_surf = float(
            np.trapezoid(direct_line[direct_valid], x_line[direct_valid])
            / max(x_line[direct_valid][-1] - x_line[direct_valid][0], 1.0e-30)
        )
        mdot_centerline = float(np.interp(xmid, x_line[direct_valid], direct_line[direct_valid]))
        grad_centerline = float(np.interp(xmid, x_line[recon_valid], gradp_n[recon_valid]))
        rho_darcy_centerline = float(np.interp(xmid, x_line, rho_darcy_line))
        rho_centerline = float(np.interp(xmid, x_line, rho_line))
        rows.append(
            (
                float(time_value),
                mdot_surf,
                mdot_centerline,
                grad_centerline,
                rho_darcy_centerline,
                rho_centerline,
            )
        )

    if not rows:
        raise RuntimeError(
            "No ParaView time steps with readable pressure/gas_density_qp/"
            "mobility_qp/m_dot_g_qp datasets were found for surface-diagnostics reconstruction."
        )

    if skipped:
        print(
            f"Skipping {skipped} ParaView time steps during surface-diagnostics "
            "reconstruction because required datasets were missing or unreadable.",
            flush=True,
        )

    data = np.zeros(
        len(rows),
        dtype=[
            ("time", float),
            ("m_dot_g_surf", float),
            ("m_dot_g_centerline", float),
            ("gradp_n_centerline", float),
            ("mobility_centerline", float),
            ("rho_g_centerline", float),
        ],
    )
    for i, row in enumerate(rows):
        data["time"][i] = row[0]
        data["m_dot_g_surf"][i] = row[1]
        data["m_dot_g_centerline"][i] = row[2]
        data["gradp_n_centerline"][i] = row[3]
        data["mobility_centerline"][i] = row[4]
        data["rho_g_centerline"][i] = row[5]

    if max_time_steps is None:
        with cache_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time",
                    "m_dot_g_surf",
                    "m_dot_g_centerline",
                    "gradp_n_centerline",
                    "mobility_centerline",
                    "rho_g_centerline",
                ]
            )
            for row in rows:
                writer.writerow(row)
        print(f"Wrote cached VTU surface diagnostics: {cache_path}", flush=True)

    return data


def save_surface_diagnostics_plot(
    mfem: np.ndarray, pato: np.ndarray, out_path: Path
) -> bool:
    mfem_names = set(mfem.dtype.names or ())
    pato_names = set(pato.dtype.names or ())
    mfem_needed = {"time", "gradp_n_centerline", "mobility_centerline"}
    pato_needed = {"time", "snGradP_centerline", "mobility_centerline"}
    if not (mfem_needed.issubset(mfem_names) and pato_needed.issubset(pato_names)):
        return False

    fig, (ax_p, ax_mob) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
    )

    ax_p.plot(
        mfem["time"],
        mfem["gradp_n_centerline"],
        color="tab:blue",
        lw=2,
        label="MFEM centerline",
    )
    ax_p.plot(
        pato["time"],
        pato["snGradP_centerline"],
        color="tab:blue",
        lw=2,
        ls="--",
        label="PATO centerline",
    )
    ax_p.set_ylabel("snGrad(p) (Pa/m)")
    ax_p.grid(True, alpha=0.25)
    ax_p.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    ax_mob.plot(
        mfem["time"],
        mfem["mobility_centerline"],
        color="tab:green",
        lw=2,
        label="MFEM centerline",
    )
    ax_mob.plot(
        pato["time"],
        pato["mobility_centerline"],
        color="tab:green",
        lw=2,
        ls="--",
        label="PATO centerline",
    )
    ax_mob.set_xlabel("Time (s)")
    ax_mob.set_ylabel(r"Mobility $\rho_g K / \mu$ (SI)")
    ax_mob.grid(True, alpha=0.25)
    ax_mob.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    xmax = finite_time_max(mfem["time"], pato["time"])
    ax_p.set_xlim(0.0, xmax)
    ax_mob.set_xlim(0.0, xmax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def save_flux_reconstruction_plot(
    mfem: np.ndarray, pato: np.ndarray, out_path: Path
) -> bool:
    mfem_names = set(mfem.dtype.names or ())
    pato_names = set(pato.dtype.names or ())
    mfem_needed = {
        "time",
        "m_dot_g_centerline",
        "gradp_n_centerline",
        "mobility_centerline",
    }
    pato_needed = {
        "time",
        "mDotGw_centerline",
        "snGradP_centerline",
        "mobility_centerline",
    }
    if not (mfem_needed.issubset(mfem_names) and pato_needed.issubset(pato_names)):
        return False

    mfem_flux_recon = -mfem["mobility_centerline"] * mfem["gradp_n_centerline"]
    pato_flux_recon = -pato["mobility_centerline"] * pato["snGradP_centerline"]

    fig, (ax_flux, ax_delta) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
    )
    ax_flux.plot(
        mfem["time"],
        mfem["m_dot_g_centerline"],
        color="tab:blue",
        lw=2,
        label="MFEM direct",
    )
    ax_flux.plot(
        mfem["time"],
        mfem_flux_recon,
        color="tab:blue",
        lw=1.8,
        ls=":",
        label="MFEM reconstructed",
    )
    ax_flux.plot(
        pato["time"],
        pato["mDotGw_centerline"],
        color="tab:orange",
        lw=2,
        ls="--",
        label="PATO direct",
    )
    ax_flux.plot(
        pato["time"],
        pato_flux_recon,
        color="tab:orange",
        lw=1.8,
        ls="-.",
        label="PATO reconstructed",
    )
    ax_flux.set_ylabel("m_dot_g (kg/m2/s)")
    ax_flux.grid(True, alpha=0.25)
    ax_flux.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    ax_delta.plot(
        mfem["time"],
        mfem_flux_recon - mfem["m_dot_g_centerline"],
        color="tab:blue",
        lw=2,
        label="MFEM recon - direct",
    )
    ax_delta.plot(
        pato["time"],
        pato_flux_recon - pato["mDotGw_centerline"],
        color="tab:orange",
        lw=2,
        ls="--",
        label="PATO recon - direct",
    )
    ax_delta.set_xlabel("Time (s)")
    ax_delta.set_ylabel("Flux residual")
    ax_delta.grid(True, alpha=0.25)
    ax_delta.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    xmax = finite_time_max(mfem["time"], pato["time"])
    ax_flux.set_xlim(0.0, xmax)
    ax_delta.set_xlim(0.0, xmax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def save_flux_ratio_plot(
    mfem: np.ndarray, pato: np.ndarray, out_path: Path
) -> bool:
    mfem_names = set(mfem.dtype.names or ())
    pato_names = set(pato.dtype.names or ())
    mfem_needed = {
        "time",
        "m_dot_g_centerline",
        "gradp_n_centerline",
        "mobility_centerline",
    }
    pato_needed = {
        "time",
        "mDotGw_centerline",
        "snGradP_centerline",
        "mobility_centerline",
    }
    if not (mfem_needed.issubset(mfem_names) and pato_needed.issubset(pato_names)):
        return False

    t_mf = np.asarray(mfem["time"], dtype=float)
    t_pa = np.asarray(pato["time"], dtype=float)
    t0 = max(float(np.nanmin(t_mf)), float(np.nanmin(t_pa)))
    t1 = min(float(np.nanmax(t_mf)), float(np.nanmax(t_pa)))
    common_mask = np.isfinite(t_mf) & (t_mf >= t0) & (t_mf <= t1)
    t_common = t_mf[common_mask]
    if t_common.size < 2:
        return False

    mfem_direct = np.asarray(mfem["m_dot_g_centerline"], dtype=float)[common_mask]
    mfem_mob = np.asarray(mfem["mobility_centerline"], dtype=float)[common_mask]
    mfem_grad = np.asarray(mfem["gradp_n_centerline"], dtype=float)[common_mask]
    mfem_recon = -mfem_mob * mfem_grad

    pato_direct = np.interp(t_common, t_pa, pato["mDotGw_centerline"])
    pato_mob = np.interp(t_common, t_pa, pato["mobility_centerline"])
    pato_grad = np.interp(t_common, t_pa, pato["snGradP_centerline"])
    pato_recon = -pato_mob * pato_grad

    ratio_direct = common.safe_divide(mfem_direct, pato_direct)
    ratio_recon = common.safe_divide(mfem_recon, pato_recon)
    ratio_mob = common.safe_divide(mfem_mob, pato_mob)
    ratio_grad_abs = common.safe_divide(np.abs(mfem_grad), np.abs(pato_grad))

    fig, (ax_flux, ax_mob, ax_grad) = plt.subplots(
        3, 1, figsize=(13, 9.2), sharex=True, constrained_layout=True
    )
    for ax in (ax_flux, ax_mob, ax_grad):
        ax.axhline(1.0, color="black", lw=1.2, ls=":")
        ax.grid(True, alpha=0.25)

    ax_flux.plot(
        t_common,
        ratio_direct,
        lw=2,
        color="tab:blue",
        label="direct flux ratio",
    )
    ax_flux.plot(
        t_common,
        ratio_recon,
        lw=1.8,
        ls="--",
        color="tab:orange",
        label="reconstructed flux ratio",
    )
    ax_flux.set_ylabel("MFEM / PATO")
    ax_flux.set_title("Centerline mass-flux ratio decomposition")
    ax_flux.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    ax_mob.plot(
        t_common,
        ratio_mob,
        lw=2,
        color="tab:green",
        label="mobility ratio",
    )
    ax_mob.set_ylabel("MFEM / PATO")
    ax_mob.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    ax_grad.plot(
        t_common,
        ratio_grad_abs,
        lw=2,
        color="tab:red",
        label=r"$|\nabla p|$ ratio",
    )
    ax_grad.set_xlabel("Time (s)")
    ax_grad.set_ylabel("MFEM / PATO")
    ax_grad.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    ax_flux.set_xlim(float(t_common[0]), float(t_common[-1]))
    ax_mob.set_xlim(float(t_common[0]), float(t_common[-1]))
    ax_grad.set_xlim(float(t_common[0]), float(t_common[-1]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def save_surface_properties_plot(
    mfem: np.ndarray, pato: np.ndarray, out_path: Path
) -> bool:
    mfem_names = set(mfem.dtype.names or ())
    pato_names = set(pato.dtype.names or ())
    panels: list[tuple[str, np.ndarray, np.ndarray, str, str]] = []

    if {"time", "rho_g_centerline"}.issubset(mfem_names) and {
        "time",
        "rho_g_centerline",
    }.issubset(pato_names):
        panels.append(
            (
                r"$\rho_g$ (kg/m$^3$)",
                mfem["rho_g_centerline"],
                pato["rho_g_centerline"],
                "tab:orange",
                "Density",
            )
        )

    has_k = {"time", "K_centerline"}.issubset(mfem_names) and {
        "time",
        "Knn_centerline",
    }.issubset(pato_names)
    if has_k:
        panels.append(
            (
                r"$K_{nn}$ (m$^2$)",
                mfem["K_centerline"],
                pato["Knn_centerline"],
                "tab:red",
                "Permeability",
            )
        )

    has_mu = {"time", "mu_g_centerline"}.issubset(mfem_names) and {
        "time",
        "mu_g_centerline",
    }.issubset(pato_names)
    if has_mu:
        panels.append(
            (
                r"$\mu_g$ (Pa·s)",
                mfem["mu_g_centerline"],
                pato["mu_g_centerline"],
                "tab:purple",
                "Viscosity",
            )
        )

    if (not has_k or not has_mu) and {"time", "mobility_centerline"}.issubset(
        mfem_names
    ) and {"time", "mobility_centerline"}.issubset(pato_names):
        panels.append(
            (
                r"Mobility $\rho_g K / \mu$ (SI)",
                mfem["mobility_centerline"],
                pato["mobility_centerline"],
                "tab:green",
                "Mobility",
            )
        )

    if not panels:
        return False

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(13, 3.2 * len(panels) + 0.8),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, (ylabel, mfem_vals, pato_vals, color, title) in zip(
        axes, panels, strict=True
    ):
        ax.plot(
            mfem["time"],
            mfem_vals,
            color=color,
            lw=2,
            label="MFEM centerline",
        )
        ax.plot(
            pato["time"],
            pato_vals,
            color=color,
            lw=2,
            ls="--",
            label="PATO centerline",
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
        ax.set_title(title)

    axes[-1].set_xlabel("Time (s)")

    xmax = finite_time_max(mfem["time"], pato["time"])
    for ax in axes:
        ax.set_xlim(0.0, xmax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_time_steps is not None and args.max_time_steps < 1:
        parser.error("--max-time-steps must be >= 1")

    out_dir = Path(args.output_dir)
    input_yaml = Path(args.input).expanduser().resolve()
    mfem_path = (
        Path(args.mfem_boundary_diagnostics).expanduser()
        if args.mfem_boundary_diagnostics is not None
        else out_dir / "boundary_diagnostics.csv"
    )
    pato_path = Path(args.pato_surface_diagnostics).expanduser()

    mfem = None
    mfem_source: str
    if not args.force_paraview:
        mfem = common.load_named_csv(
            mfem_path,
            required=False,
            description="MFEM boundary diagnostics CSV",
        )

    if mfem is not None:
        mfem_source = str(mfem_path)
    else:
        print(
            "MFEM boundary diagnostics CSV is unavailable; reconstructing "
            "surface diagnostics from VTU/PVTU output instead.",
            flush=True,
        )
        mfem = reconstruct_mfem_surface_diagnostics_from_paraview(
            out_dir,
            input_yaml,
            max_time_steps=args.max_time_steps,
        )
        mfem_source = f"VTU/PVTU reconstruction under {out_dir}"

    pato = common.load_named_csv(
        pato_path,
        required=True,
        description="PATO top surface diagnostics CSV",
    )

    generated: list[Path] = []

    surface_diag_plot = out_dir / f"{args.out_prefix}_surface_diagnostics_compare.png"
    if save_surface_diagnostics_plot(mfem, pato, surface_diag_plot):
        generated.append(surface_diag_plot)
    else:
        print("Skipping transport comparison: required centerline columns are missing.")

    flux_recon_plot = out_dir / f"{args.out_prefix}_flux_reconstruction_compare.png"
    if save_flux_reconstruction_plot(mfem, pato, flux_recon_plot):
        generated.append(flux_recon_plot)
    else:
        print("Skipping flux reconstruction plot: required columns are missing.")

    flux_ratio_plot = out_dir / f"{args.out_prefix}_flux_ratio_decomposition.png"
    if save_flux_ratio_plot(mfem, pato, flux_ratio_plot):
        generated.append(flux_ratio_plot)
    else:
        print("Skipping flux ratio plot: no overlapping direct/reconstructed flux data.")

    surface_props_plot = out_dir / f"{args.out_prefix}_surface_properties_compare.png"
    if save_surface_properties_plot(mfem, pato, surface_props_plot):
        generated.append(surface_props_plot)
    else:
        print("Skipping gas-property plot: required columns are missing.")

    if not generated:
        raise RuntimeError(
            "No comparable MFEM/PATO surface diagnostics were found in the supplied files."
        )

    print(f"MFEM diagnostics: {mfem_source}")
    print(f"PATO diagnostics: {pato_path}")
    for path in generated:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{Path(__file__).name}: {exc}", file=sys.stderr)
        raise SystemExit(1)
