#!/usr/bin/env python3
"""Compare MFEM ablation case-2.2 outputs against Amaryllis reference data."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TOL = {
    "temperature_rmse_max": 300.0,
    "temperature_max_abs_max": 650.0,
    "m_dot_g_rmse_max": 0.025,
    "m_dot_g_max_abs_max": 0.08,
    "m_dot_g_peak_rel_error_max": 0.5,
    "m_dot_g_peak_time_error_max": 10.0,
    "front98_max_abs_max": 0.01,
    "front98_rmse_max": 0.01,
    "front2_max_abs_max": 0.01,
    "front2_rmse_max": 0.01,
    "m_dot_c_rmse_max": 0.01,
    "m_dot_c_peak_rel_error_max": 0.35,
    "recession_rmse_max": 0.0015,
    "recession_final_rel_error_max": 0.12,
}


def load_acceptance_from_yaml(path: Path) -> Dict[str, float]:
    vals = dict(DEFAULT_TOL)
    if not path.exists():
        return vals

    in_acceptance = False
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "acceptance:":
            in_acceptance = True
            continue
        if in_acceptance and not line.startswith(" "):
            break
        if in_acceptance and ":" in stripped:
            k, v = stripped.split(":", 1)
            try:
                vals[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return vals


def load_probe_y_from_yaml(path: Path) -> List[float]:
    if not path.exists():
        return []

    probe_y: List[float] = []
    in_probe_y = False
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "probe_y:":
            in_probe_y = True
            continue
        if in_probe_y:
            if line.startswith("  -"):
                try:
                    probe_y.append(float(line.split("-", 1)[1].strip()))
                except ValueError:
                    pass
                continue
            if not line.startswith(" "):
                break

    return probe_y


def load_probe_depths_from_yaml(path: Path) -> List[float]:
    probe_y = load_probe_y_from_yaml(path)
    if not probe_y:
        return []
    y_wall = probe_y[0]
    return [abs(y_wall - y) for y in probe_y]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def segmented_rmse_max(
    t: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    t0: float,
    t1: float,
    valid_mask: np.ndarray | None = None,
) -> Tuple[float, float]:
    mask = (t >= t0) & (t <= t1)
    if valid_mask is not None:
        mask = mask & valid_mask
    if not np.any(mask):
        return float("nan"), float("nan")
    return rmse(a[mask], b[mask]), max_abs(a[mask], b[mask])


def tc_index(name: str) -> int:
    if name.startswith("TC") and name[2:].isdigit():
        return int(name[2:])
    return -1


def build_mfem_temperature_by_depth(
    probes: np.ndarray, probe_depths: List[float]
) -> List[Tuple[float, str, np.ndarray]]:
    items: List[Tuple[float, str, np.ndarray]] = []
    for name in probes.dtype.names:
        if name == "time":
            continue
        if name == "wall":
            depth = 0.0
        else:
            idx = tc_index(name)
            if idx < 0:
                continue
            depth = probe_depths[idx] if idx < len(probe_depths) else float(idx)
        items.append((depth, name, probes[name]))
    items.sort(key=lambda x: x[0])
    return items


def build_amaryllis_temperature_by_depth(
    am_energy: np.ndarray, probe_depths: List[float]
) -> List[Tuple[float, str, np.ndarray]]:
    n_signals = int(am_energy.shape[1]) - 1
    items: List[Tuple[float, str, np.ndarray]] = []
    for i in range(n_signals):
        name = "wall" if i == 0 else f"TC{i}"
        depth = probe_depths[i] if i < len(probe_depths) else float(i)
        items.append((depth, name, am_energy[:, i + 1]))
    items.sort(key=lambda x: x[0])
    return items


def ensure_2d(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a


def time_derivative(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    if t.size != y.size:
        raise ValueError("time_derivative expects t and y with the same length")
    if t.size < 2:
        return np.zeros_like(y, dtype=float)
    edge_order = 2 if t.size >= 3 else 1
    return np.gradient(y, t, edge_order=edge_order)


def parse_csv_float_list(text: str) -> List[float]:
    vals: List[float] = []
    for tok in text.split(","):
        s = tok.strip()
        if not s:
            continue
        vals.append(float(s))
    return vals


def load_pato_point_plot(path: Path) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    with path.open("r") as f:
        header = f.readline().strip()
    y_vals = [
        float(m.group(1))
        for m in re.finditer(r"probe\d+\([^,]+,([^,]+),", header)
    ]
    data = ensure_2d(np.loadtxt(path, comments="/"))
    if data.shape[1] < 2:
        raise RuntimeError(f"Unexpected PATO point-plot format in {path}")
    time = data[:, 0]
    vals = data[:, 1:]
    if y_vals and len(y_vals) != vals.shape[1]:
        raise RuntimeError(
            f"PATO point-plot header/data column mismatch in {path}: "
            f"{len(y_vals)} probes in header vs {vals.shape[1]} data columns"
        )
    return time, vals, y_vals


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1.0e-30)
    out[mask] = num[mask] / den[mask]
    return out


def build_mfem_pressure_by_y(
    mfem_pressure_probes: np.ndarray, probe_y: List[float]
) -> Dict[float, Tuple[str, np.ndarray]]:
    out: Dict[float, Tuple[str, np.ndarray]] = {}
    names = list(mfem_pressure_probes.dtype.names or [])
    if "wall" in names and probe_y:
        out[probe_y[0]] = ("wall", mfem_pressure_probes["wall"])
    for name in names:
        if name in ("time", "wall"):
            continue
        idx = tc_index(name)
        if idx > 0 and idx < len(probe_y):
            out[probe_y[idx]] = (name, mfem_pressure_probes[name])
    return out


def match_pressure_probe_points(
    mfem_pressure_probes: np.ndarray,
    probe_y: List[float],
    pato_p_y: List[float],
    pato_tol: float = 1.0e-8,
) -> List[Tuple[float, str, int, np.ndarray]]:
    mfem_y_map = build_mfem_pressure_by_y(mfem_pressure_probes, probe_y)
    if not mfem_y_map:
        return []
    mfem_y_keys = list(mfem_y_map.keys())
    matched: List[Tuple[float, str, int, np.ndarray]] = []
    for j, y_pato in enumerate(pato_p_y):
        y_best = min(mfem_y_keys, key=lambda y: abs(y - y_pato))
        if abs(y_best - y_pato) <= pato_tol:
            name, series = mfem_y_map[y_best]
            matched.append((y_pato, name, j, series))
    matched.sort(key=lambda x: x[0], reverse=True)
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="ParaView/ablation_case2_2",
        help="Directory containing temperature_probes.csv and mass_metrics.csv",
    )
    parser.add_argument(
        "--input",
        default="Input/input_ablation_case2_2.yaml",
        help="Input YAML file with acceptance tolerances",
    )
    parser.add_argument(
        "--amaryllis-energy",
        default="/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/data/ref/Amaryllis/Amaryllis_Energy_TestCase_2.2.txt",
        help="Amaryllis energy reference file",
    )
    parser.add_argument(
        "--amaryllis-mass",
        default="/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/data/ref/Amaryllis/Amaryllis_Mass_TestCase_2.2.txt",
        help="Amaryllis mass reference file",
    )
    parser.add_argument(
        "--out-prefix",
        default="ablation_case2_2",
        help="Prefix for generated plot filenames",
    )
    parser.add_argument(
        "--pato-surface-diagnostics",
        default=(
            "/home/quinnchr/miniconda3/envs/pato/src/volume_pato/pato-3.1/"
            "tutorials/1D/AblationTestCase_2.x/output/porousMat/"
            "top_surface_diagnostics.csv"
        ),
        help="PATO top-surface diagnostics CSV (for snGradP/mobility comparison)",
    )
    parser.add_argument(
        "--pato-pressure-plot",
        default=(
            "/home/quinnchr/miniconda3/envs/pato/src/volume_pato/pato-3.1/"
            "tutorials/1D/AblationTestCase_2.x/output/porousMat/scalar/p_plot"
        ),
        help="PATO sampled point plot for pressure (system/porousMat/plotDict)",
    )
    parser.add_argument(
        "--pressure-profile-times",
        default="0.1,1,10,60",
        help="Comma-separated snapshot times (s) for centerline pressure profile comparison",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    probes_csv = out_dir / "temperature_probes.csv"
    mass_csv = out_dir / "mass_metrics.csv"
    clamp_csv = out_dir / "bprime_clamp_stats.csv"
    boundary_diag_csv = out_dir / "boundary_diagnostics.csv"
    mfem_pressure_probes_csv = out_dir / "pressure_probes.csv"
    mfem_mesh_diag_csv = out_dir / "mesh_diagnostics.csv"
    mfem_mass_eq_probe_csv = out_dir / "mass_eq_probe_diagnostics.csv"
    pato_surface_diag_csv = Path(args.pato_surface_diagnostics).expanduser()
    pato_pressure_plot = Path(args.pato_pressure_plot).expanduser()

    if not probes_csv.exists() or not mass_csv.exists():
        raise FileNotFoundError(
            f"Expected MFEM outputs not found: {probes_csv} and {mass_csv}"
        )

    tol = load_acceptance_from_yaml(Path(args.input))

    probes = np.genfromtxt(probes_csv, delimiter=",", names=True)
    mass = np.genfromtxt(mass_csv, delimiter=",", names=True)
    boundary_diag = (
        np.genfromtxt(boundary_diag_csv, delimiter=",", names=True)
        if boundary_diag_csv.exists()
        else None
    )
    mfem_pressure_probes = (
        np.genfromtxt(mfem_pressure_probes_csv, delimiter=",", names=True)
        if mfem_pressure_probes_csv.exists()
        else None
    )
    mfem_mesh_diag = (
        np.genfromtxt(mfem_mesh_diag_csv, delimiter=",", names=True)
        if mfem_mesh_diag_csv.exists()
        else None
    )
    mfem_mass_eq_probe = (
        np.genfromtxt(mfem_mass_eq_probe_csv, delimiter=",", names=True)
        if mfem_mass_eq_probe_csv.exists()
        else None
    )
    pato_surface_diag = (
        np.genfromtxt(pato_surface_diag_csv, delimiter=",", names=True)
        if pato_surface_diag_csv.exists()
        else None
    )
    pato_p_time = None
    pato_p_vals = None
    pato_p_y: List[float] = []
    if pato_pressure_plot.exists():
        pato_p_time, pato_p_vals, pato_p_y = load_pato_point_plot(pato_pressure_plot)

    am_energy = ensure_2d(np.loadtxt(args.amaryllis_energy, skiprows=1))
    am_mass = ensure_2d(np.loadtxt(args.amaryllis_mass, skiprows=1))

    probe_y = load_probe_y_from_yaml(Path(args.input))
    probe_depths = load_probe_depths_from_yaml(Path(args.input))
    pressure_profile_times = parse_csv_float_list(args.pressure_profile_times)
    mfem_by_depth = build_mfem_temperature_by_depth(probes, probe_depths)
    am_by_depth = build_amaryllis_temperature_by_depth(am_energy, probe_depths)
    n_common = min(len(mfem_by_depth), len(am_by_depth))
    if n_common == 0:
        raise RuntimeError("No temperature probes available for MFEM/Amaryllis comparison.")

    mfem_time = probes["time"]
    am_time = am_energy[:, 0]
    probe_pairs = list(zip(mfem_by_depth[:n_common], am_by_depth[:n_common]))

    temp_metrics: List[Tuple[str, float, float, bool]] = []
    for (d_mf, name_mf, sig_mf), (_, name_am, sig_am) in probe_pairs:
        valid = sig_am > 1.0  # Ignore Amaryllis sentinel zeros.
        if np.any(valid):
            mfem_interp = np.interp(am_time[valid], mfem_time, sig_mf)
            am_sig = sig_am[valid]
            r = rmse(mfem_interp, am_sig)
            m = max_abs(mfem_interp, am_sig)
            ok = (r <= tol["temperature_rmse_max"]) and (
                m <= tol["temperature_max_abs_max"]
            )
        else:
            r = float("nan")
            m = float("nan")
            ok = True
        sig_label = f"{name_mf}~{name_am}@depth={d_mf:.6g}m"
        temp_metrics.append((sig_label, r, m, ok))

    # Segmented wall-temperature metrics for cooldown/heating analysis.
    wall_mf = np.interp(am_time, mfem_time, probe_pairs[0][0][2])
    wall_ref = probe_pairs[0][1][2]
    wall_valid = wall_ref > 1.0
    heat_rmse, heat_max = segmented_rmse_max(
        am_time, wall_mf, wall_ref, 0.1, 60.0, wall_valid
    )
    cool_rmse, cool_max = segmented_rmse_max(
        am_time, wall_mf, wall_ref, 60.1, 120.0, wall_valid
    )
    heat_pass = (
        (not np.isfinite(heat_rmse) and not np.isfinite(heat_max))
        or (
            heat_rmse <= tol["temperature_rmse_max"]
            and heat_max <= tol["temperature_max_abs_max"]
        )
    )
    cool_pass = (
        (not np.isfinite(cool_rmse) and not np.isfinite(cool_max))
        or (
            cool_rmse <= tol["temperature_rmse_max"]
            and cool_max <= tol["temperature_max_abs_max"]
        )
    )

    # Amaryllis mass columns: time, m_dot_g, m_dot_c, front98, front2, recession.
    t_ref = am_mass[:, 0]
    ref_mdot = am_mass[:, 1]
    ref_mdot_c = am_mass[:, 2]
    ref_front98 = am_mass[:, 3]
    ref_front2 = am_mass[:, 4]
    ref_recession = am_mass[:, 5]

    mfem_mass_t = mass["time"]
    mfem_mdot = mass["m_dot_g_surf"]
    mfem_mdot_centerline = (
        mass["m_dot_g_centerline"] if "m_dot_g_centerline" in mass.dtype.names else None
    )
    mfem_front98 = mass["front_98_virgin"]
    mfem_front2 = mass["front_2_char"]
    mfem_mdot_c = mass["m_dot_c"]
    mfem_recession = mass["recession"]
    mfem_recession_rate = time_derivative(mfem_mass_t, mfem_recession)
    ref_recession_rate = time_derivative(t_ref, ref_recession)

    mfem_mdot_i = np.interp(t_ref, mfem_mass_t, mfem_mdot)
    mfem_mdot_c_i = np.interp(t_ref, mfem_mass_t, mfem_mdot_c)
    mfem_front98_i = np.interp(t_ref, mfem_mass_t, mfem_front98)
    mfem_front2_i = np.interp(t_ref, mfem_mass_t, mfem_front2)
    mfem_recession_i = np.interp(t_ref, mfem_mass_t, mfem_recession)

    mdot_rmse = rmse(mfem_mdot_i, ref_mdot)
    mdot_max = max_abs(mfem_mdot_i, ref_mdot)

    i_mf = int(np.argmax(mfem_mdot))
    i_ref = int(np.argmax(ref_mdot))
    mf_peak = float(mfem_mdot[i_mf])
    ref_peak = float(ref_mdot[i_ref])
    mf_peak_t = float(mfem_mass_t[i_mf])
    ref_peak_t = float(t_ref[i_ref])

    peak_rel = abs(mf_peak - ref_peak) / max(abs(ref_peak), 1.0e-12)
    peak_time_err = abs(mf_peak_t - ref_peak_t)

    front98_rmse = rmse(mfem_front98_i, ref_front98)
    front98_max = max_abs(mfem_front98_i, ref_front98)
    front2_rmse = rmse(mfem_front2_i, ref_front2)
    front2_max = max_abs(mfem_front2_i, ref_front2)

    mdot_c_rmse = rmse(mfem_mdot_c_i, ref_mdot_c)
    i_mf_c = int(np.argmax(mfem_mdot_c))
    i_ref_c = int(np.argmax(ref_mdot_c))
    mf_peak_c = float(mfem_mdot_c[i_mf_c])
    ref_peak_c = float(ref_mdot_c[i_ref_c])
    mdot_c_peak_rel = abs(mf_peak_c - ref_peak_c) / max(abs(ref_peak_c), 1.0e-12)

    recession_rmse = rmse(mfem_recession_i, ref_recession)
    recession_final_rel = abs(mfem_recession_i[-1] - ref_recession[-1]) / max(
        abs(ref_recession[-1]), 1.0e-12
    )

    clamp_counts = {"pressure": np.nan, "BprimeG": np.nan, "temperature": np.nan}
    if clamp_csv.exists():
        with clamp_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                axis = str(row.get("axis", "")).strip()
                if axis in clamp_counts:
                    try:
                        clamp_counts[axis] = float(row.get("clamp_count", "nan"))
                    except ValueError:
                        clamp_counts[axis] = np.nan

    temp_pass = all(x[3] for x in temp_metrics)
    mdot_pass = (
        mdot_rmse <= tol["m_dot_g_rmse_max"]
        and mdot_max <= tol["m_dot_g_max_abs_max"]
        and peak_rel <= tol["m_dot_g_peak_rel_error_max"]
        and peak_time_err <= tol["m_dot_g_peak_time_error_max"]
    )
    temp_pass = temp_pass and heat_pass and cool_pass
    front98_pass = (
        front98_rmse <= tol["front98_rmse_max"]
        and front98_max <= tol["front98_max_abs_max"]
    )
    front2_pass = (
        front2_rmse <= tol["front2_rmse_max"]
        and front2_max <= tol["front2_max_abs_max"]
    )
    mdot_c_pass = (
        mdot_c_rmse <= tol["m_dot_c_rmse_max"]
        and mdot_c_peak_rel <= tol["m_dot_c_peak_rel_error_max"]
    )
    recession_pass = (
        recession_rmse <= tol["recession_rmse_max"]
        and recession_final_rel <= tol["recession_final_rel_error_max"]
    )

    overall_pass = (
        temp_pass
        and mdot_pass
        and front98_pass
        and front2_pass
        and mdot_c_pass
        and recession_pass
    )

    metrics_csv = out_dir / "amaryllis_error_metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "signal", "rmse", "max_abs", "metric", "value", "tolerance", "pass"])
        for sig, r, m, ok in temp_metrics:
            w.writerow(["temperature", sig, r, m, "", "", "", int(ok)])
        w.writerow(["temperature_segment", "wall_heating_0.1_60s", heat_rmse, heat_max,
                    "", "", "", int(heat_pass)])
        w.writerow(["temperature_segment", "wall_cooling_60.1_120s", cool_rmse, cool_max,
                    "", "", "", int(cool_pass)])

        w.writerow(["mass_flux", "m_dot_g", mdot_rmse, mdot_max, "rmse",
                    mdot_rmse, tol["m_dot_g_rmse_max"], int(mdot_rmse <= tol["m_dot_g_rmse_max"])])
        w.writerow(["mass_flux", "m_dot_g", mdot_rmse, mdot_max, "max_abs",
                    mdot_max, tol["m_dot_g_max_abs_max"], int(mdot_max <= tol["m_dot_g_max_abs_max"])])
        w.writerow(["mass_flux", "m_dot_g", "", "", "peak_rel_error",
                    peak_rel, tol["m_dot_g_peak_rel_error_max"], int(peak_rel <= tol["m_dot_g_peak_rel_error_max"])])
        w.writerow(["mass_flux", "m_dot_g", "", "", "peak_time_error",
                    peak_time_err, tol["m_dot_g_peak_time_error_max"], int(peak_time_err <= tol["m_dot_g_peak_time_error_max"])])

        w.writerow(["front", "front_98_virgin", front98_rmse, front98_max, "rmse",
                    front98_rmse, tol["front98_rmse_max"], int(front98_rmse <= tol["front98_rmse_max"])])
        w.writerow(["front", "front_98_virgin", front98_rmse, front98_max, "max_abs",
                    front98_max, tol["front98_max_abs_max"], int(front98_max <= tol["front98_max_abs_max"])])
        w.writerow(["front", "front_2_char", front2_rmse, front2_max, "rmse",
                    front2_rmse, tol["front2_rmse_max"], int(front2_rmse <= tol["front2_rmse_max"])])
        w.writerow(["front", "front_2_char", front2_rmse, front2_max, "max_abs",
                    front2_max, tol["front2_max_abs_max"], int(front2_max <= tol["front2_max_abs_max"])])

        w.writerow(["mass_flux", "m_dot_c", mdot_c_rmse, "", "rmse",
                    mdot_c_rmse, tol["m_dot_c_rmse_max"], int(mdot_c_rmse <= tol["m_dot_c_rmse_max"])])
        w.writerow(["mass_flux", "m_dot_c", "", "", "peak_rel_error",
                    mdot_c_peak_rel, tol["m_dot_c_peak_rel_error_max"], int(mdot_c_peak_rel <= tol["m_dot_c_peak_rel_error_max"])])
        w.writerow(["recession", "recession", recession_rmse, "", "rmse",
                    recession_rmse, tol["recession_rmse_max"], int(recession_rmse <= tol["recession_rmse_max"])])
        w.writerow(["recession", "recession", "", "", "final_rel_error",
                    recession_final_rel, tol["recession_final_rel_error_max"], int(recession_final_rel <= tol["recession_final_rel_error_max"])])

        for axis in ("pressure", "BprimeG", "temperature"):
            w.writerow(["bprime_clamp", axis, "", "", "count",
                        clamp_counts[axis], "", ""])

        w.writerow(["summary", "overall", "", "", "", "", "", int(overall_pass)])

    tol_csv = out_dir / "amaryllis_error_tolerances.csv"
    with tol_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal", "tolerance"])
        for k, v in DEFAULT_TOL.items():
            w.writerow([k, tol.get(k, v)])

    # Plot 1: temperature history.
    plt.figure(figsize=(14, 5))
    cmap = plt.get_cmap("tab10")
    for i, ((d_mf, name_mf, sig_mf), (_, name_ref, sig_ref)) in enumerate(probe_pairs):
        col = "black" if i == 0 else cmap((i - 1) % 10)
        depth_label = f"{d_mf:.4f} m"
        plt.plot(mfem_time, sig_mf, color=col, lw=2,
                 label=f"MFEM {name_mf} ({depth_label})")
        plt.plot(am_time, sig_ref, color=col, lw=1.6, ls="--",
                 label=f"Amaryllis {name_ref} ({depth_label})")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.xlim(0.0, max(float(mfem_time[-1]), float(am_time[-1])))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=9)
    plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    plt.savefig(out_dir / f"{args.out_prefix}_temperature_history.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Plot 2: pyrolysis mass flux.
    plt.figure(figsize=(13, 4.8))
    plt.plot(mfem_mass_t, mfem_mdot, color="black", lw=2, label="MFEM area-avg")
    if mfem_mdot_centerline is not None:
        plt.plot(
            mfem_mass_t,
            mfem_mdot_centerline,
            color="tab:blue",
            lw=2,
            label="MFEM centerline",
        )
    plt.plot(t_ref, ref_mdot, color="black", ls="--", lw=2, label="Amaryllis")
    plt.xlabel("Time (s)")
    plt.ylabel("Pyrolysis mass flux (kg/m2/s)")
    plt.xlim(0.0, max(float(mfem_mass_t[-1]), float(t_ref[-1])))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
    plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    plt.savefig(out_dir / f"{args.out_prefix}_pyrolysis_mass_flux.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Plot 3: fronts.
    plt.figure(figsize=(13, 4.8))
    plt.plot(mfem_mass_t, mfem_front98, color="black", lw=2, label="MFEM 98% virgin")
    plt.plot(mfem_mass_t, mfem_front2, color="gray", lw=2, label="MFEM 2% char")
    plt.plot(t_ref, ref_front98, color="black", lw=2, ls="--", label="Amaryllis 98% virgin")
    plt.plot(t_ref, ref_front2, color="gray", lw=2, ls="--", label="Amaryllis 2% char")
    plt.xlabel("Time (s)")
    plt.ylabel("Depth (m)")
    plt.xlim(0.0, max(float(mfem_mass_t[-1]), float(t_ref[-1])))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
    plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    plt.savefig(out_dir / f"{args.out_prefix}_fronts.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Plot 4: recession amount and recession rate comparison.
    fig, (ax_rec, ax_rate) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
    )
    ax_rec.plot(mfem_mass_t, mfem_recession, color="black", lw=2, label="MFEM")
    ax_rec.plot(t_ref, ref_recession, color="black", lw=2, ls="--", label="Amaryllis")
    ax_rec.set_ylabel("Recession (m)")
    ax_rec.grid(True, alpha=0.25)
    ax_rec.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    ax_rate.plot(mfem_mass_t, mfem_recession_rate, color="black", lw=2, label="MFEM")
    ax_rate.plot(t_ref, ref_recession_rate, color="black", lw=2, ls="--", label="Amaryllis")
    ax_rate.set_xlabel("Time (s)")
    ax_rate.set_ylabel("Recession rate (m/s)")
    ax_rate.grid(True, alpha=0.25)

    xmax_rec = max(float(mfem_mass_t[-1]), float(t_ref[-1]))
    ax_rec.set_xlim(0.0, xmax_rec)
    ax_rate.set_xlim(0.0, xmax_rec)

    fig.savefig(
        out_dir / f"{args.out_prefix}_recession_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Plot 5/6/7: MFEM vs PATO centerline surface diagnostics.
    pato_diag_plot = None
    pato_props_plot = None
    pato_flux_recon_plot = None
    pato_flux_ratio_plot = None
    pato_pressure_profile_plot = None
    pato_pressure_slope_plot = None
    mesh_diag_plot = None
    mass_eq_probe_plot = None
    mass_eq_wall_pato_plot = None
    if boundary_diag is not None and pato_surface_diag is not None:
        mfem_names = set(boundary_diag.dtype.names or ())
        pato_names = set(pato_surface_diag.dtype.names or ())
        mfem_needed = {"time", "gradp_n_centerline", "mobility_centerline"}
        pato_needed = {"time", "snGradP_centerline", "mobility_centerline"}
        if mfem_needed.issubset(mfem_names) and pato_needed.issubset(pato_names):
            fig, (ax_p, ax_mob) = plt.subplots(
                2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
            )

            ax_p.plot(
                boundary_diag["time"],
                boundary_diag["gradp_n_centerline"],
                color="tab:blue",
                lw=2,
                label="MFEM centerline",
            )
            ax_p.plot(
                pato_surface_diag["time"],
                pato_surface_diag["snGradP_centerline"],
                color="tab:blue",
                lw=2,
                ls="--",
                label="PATO centerline",
            )
            ax_p.set_ylabel("snGrad(p) (Pa/m)")
            ax_p.grid(True, alpha=0.25)
            ax_p.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            ax_mob.plot(
                boundary_diag["time"],
                boundary_diag["mobility_centerline"],
                color="tab:green",
                lw=2,
                label="MFEM centerline",
            )
            ax_mob.plot(
                pato_surface_diag["time"],
                pato_surface_diag["mobility_centerline"],
                color="tab:green",
                lw=2,
                ls="--",
                label="PATO centerline",
            )
            ax_mob.set_xlabel("Time (s)")
            ax_mob.set_ylabel(r"Mobility $\rho_g K / \mu$ (SI)")
            ax_mob.grid(True, alpha=0.25)
            ax_mob.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            xmax_diag = max(
                float(np.nanmax(boundary_diag["time"])),
                float(np.nanmax(pato_surface_diag["time"])),
            )
            ax_p.set_xlim(0.0, xmax_diag)
            ax_mob.set_xlim(0.0, xmax_diag)

            pato_diag_plot = out_dir / f"{args.out_prefix}_surface_diagnostics_compare.png"
            fig.savefig(pato_diag_plot, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            print(
                "Skipping PATO/MFEM surface diagnostics plot: missing columns in "
                f"{boundary_diag_csv} or {pato_surface_diag_csv}"
            )

        mfem_flux_needed = {
            "time",
            "m_dot_g_centerline",
            "gradp_n_centerline",
            "mobility_centerline",
        }
        pato_flux_needed = {
            "time",
            "mDotGw_centerline",
            "snGradP_centerline",
            "mobility_centerline",
        }
        if mfem_flux_needed.issubset(mfem_names) and pato_flux_needed.issubset(pato_names):
            mfem_flux_recon = (
                -boundary_diag["mobility_centerline"] * boundary_diag["gradp_n_centerline"]
            )
            pato_flux_recon = (
                -pato_surface_diag["mobility_centerline"] * pato_surface_diag["snGradP_centerline"]
            )

            fig, (ax_flux, ax_delta) = plt.subplots(
                2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
            )
            ax_flux.plot(
                boundary_diag["time"],
                boundary_diag["m_dot_g_centerline"],
                color="tab:blue",
                lw=2,
                label="MFEM direct",
            )
            ax_flux.plot(
                boundary_diag["time"],
                mfem_flux_recon,
                color="tab:blue",
                lw=1.8,
                ls=":",
                label="MFEM reconstructed",
            )
            ax_flux.plot(
                pato_surface_diag["time"],
                pato_surface_diag["mDotGw_centerline"],
                color="tab:orange",
                lw=2,
                ls="--",
                label="PATO direct",
            )
            ax_flux.plot(
                pato_surface_diag["time"],
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
                boundary_diag["time"],
                mfem_flux_recon - boundary_diag["m_dot_g_centerline"],
                color="tab:blue",
                lw=2,
                label="MFEM recon - direct",
            )
            ax_delta.plot(
                pato_surface_diag["time"],
                pato_flux_recon - pato_surface_diag["mDotGw_centerline"],
                color="tab:orange",
                lw=2,
                ls="--",
                label="PATO recon - direct",
            )
            ax_delta.set_xlabel("Time (s)")
            ax_delta.set_ylabel("Flux residual")
            ax_delta.grid(True, alpha=0.25)
            ax_delta.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            xmax_flux = max(
                float(np.nanmax(boundary_diag["time"])),
                float(np.nanmax(pato_surface_diag["time"])),
            )
            ax_flux.set_xlim(0.0, xmax_flux)
            ax_delta.set_xlim(0.0, xmax_flux)

            pato_flux_recon_plot = out_dir / f"{args.out_prefix}_flux_reconstruction_compare.png"
            fig.savefig(pato_flux_recon_plot, dpi=180, bbox_inches="tight")
            plt.close(fig)

            # Ratio decomposition on a common time grid (MFEM time samples).
            t_mf = np.asarray(boundary_diag["time"], dtype=float)
            t_pa = np.asarray(pato_surface_diag["time"], dtype=float)
            t0 = max(float(np.nanmin(t_mf)), float(np.nanmin(t_pa)))
            t1 = min(float(np.nanmax(t_mf)), float(np.nanmax(t_pa)))
            common_mask = (t_mf >= t0) & (t_mf <= t1)
            t_common = t_mf[common_mask]
            if t_common.size >= 2:
                mfem_direct = np.asarray(boundary_diag["m_dot_g_centerline"], dtype=float)[common_mask]
                mfem_recon_common = np.asarray(mfem_flux_recon, dtype=float)[common_mask]
                mfem_mob_common = np.asarray(boundary_diag["mobility_centerline"], dtype=float)[common_mask]
                mfem_grad_common = np.asarray(boundary_diag["gradp_n_centerline"], dtype=float)[common_mask]

                pato_direct_i = np.interp(t_common, t_pa, pato_surface_diag["mDotGw_centerline"])
                pato_recon_i = np.interp(t_common, t_pa, pato_flux_recon)
                pato_mob_i = np.interp(t_common, t_pa, pato_surface_diag["mobility_centerline"])
                pato_grad_i = np.interp(t_common, t_pa, pato_surface_diag["snGradP_centerline"])

                ratio_direct = safe_divide(mfem_direct, pato_direct_i)
                ratio_recon = safe_divide(mfem_recon_common, pato_recon_i)
                ratio_mob = safe_divide(mfem_mob_common, pato_mob_i)
                ratio_grad_abs = safe_divide(np.abs(mfem_grad_common), np.abs(pato_grad_i))

                fig, (ax_fluxr, ax_mobr, ax_gradr) = plt.subplots(
                    3, 1, figsize=(13, 9.2), sharex=True, constrained_layout=True
                )
                for ax in (ax_fluxr, ax_mobr, ax_gradr):
                    ax.axhline(1.0, color="black", lw=1.2, ls=":")
                    ax.grid(True, alpha=0.25)

                ax_fluxr.plot(t_common, ratio_direct, lw=2, color="tab:blue", label="direct flux ratio")
                ax_fluxr.plot(
                    t_common, ratio_recon, lw=1.8, ls="--", color="tab:orange", label="reconstructed flux ratio"
                )
                ax_fluxr.set_ylabel("MFEM / PATO")
                ax_fluxr.set_title("Centerline mass-flux ratio decomposition")
                ax_fluxr.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

                ax_mobr.plot(t_common, ratio_mob, lw=2, color="tab:green", label="mobility ratio")
                ax_mobr.set_ylabel("MFEM / PATO")
                ax_mobr.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

                ax_gradr.plot(
                    t_common,
                    ratio_grad_abs,
                    lw=2,
                    color="tab:red",
                    label=r"$|\nabla p|$ ratio",
                )
                ax_gradr.set_xlabel("Time (s)")
                ax_gradr.set_ylabel("MFEM / PATO")
                ax_gradr.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

                ax_fluxr.set_xlim(float(t_common[0]), float(t_common[-1]))
                ax_mobr.set_xlim(float(t_common[0]), float(t_common[-1]))
                ax_gradr.set_xlim(float(t_common[0]), float(t_common[-1]))

                pato_flux_ratio_plot = out_dir / f"{args.out_prefix}_flux_ratio_decomposition.png"
                fig.savefig(pato_flux_ratio_plot, dpi=180, bbox_inches="tight")
                plt.close(fig)
        else:
            print(
                "Skipping PATO/MFEM flux reconstruction plot: missing columns in "
                f"{boundary_diag_csv} or {pato_surface_diag_csv}"
            )

        mfem_prop_needed = {"time", "rho_g_centerline", "mu_g_centerline", "K_centerline"}
        pato_prop_needed = {"time", "rho_g_centerline", "mu_g_centerline", "Knn_centerline"}
        if mfem_prop_needed.issubset(mfem_names) and pato_prop_needed.issubset(pato_names):
            fig, (ax_rho, ax_k, ax_mu) = plt.subplots(
                3, 1, figsize=(13, 10.0), sharex=True, constrained_layout=True
            )

            ax_rho.plot(
                boundary_diag["time"],
                boundary_diag["rho_g_centerline"],
                color="tab:orange",
                lw=2,
                label="MFEM centerline",
            )
            ax_rho.plot(
                pato_surface_diag["time"],
                pato_surface_diag["rho_g_centerline"],
                color="tab:orange",
                lw=2,
                ls="--",
                label="PATO centerline",
            )
            ax_rho.set_ylabel(r"$\rho_g$ (kg/m$^3$)")
            ax_rho.grid(True, alpha=0.25)
            ax_rho.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            ax_k.plot(
                boundary_diag["time"],
                boundary_diag["K_centerline"],
                color="tab:red",
                lw=2,
                label="MFEM centerline",
            )
            ax_k.plot(
                pato_surface_diag["time"],
                pato_surface_diag["Knn_centerline"],
                color="tab:red",
                lw=2,
                ls="--",
                label="PATO centerline",
            )
            ax_k.set_ylabel(r"$K_{nn}$ (m$^2$)")
            ax_k.grid(True, alpha=0.25)
            ax_k.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            ax_mu.plot(
                boundary_diag["time"],
                boundary_diag["mu_g_centerline"],
                color="tab:purple",
                lw=2,
                label="MFEM centerline",
            )
            ax_mu.plot(
                pato_surface_diag["time"],
                pato_surface_diag["mu_g_centerline"],
                color="tab:purple",
                lw=2,
                ls="--",
                label="PATO centerline",
            )
            ax_mu.set_xlabel("Time (s)")
            ax_mu.set_ylabel(r"$\mu_g$ (Pa·s)")
            ax_mu.grid(True, alpha=0.25)
            ax_mu.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            xmax_props = max(
                float(np.nanmax(boundary_diag["time"])),
                float(np.nanmax(pato_surface_diag["time"])),
            )
            ax_rho.set_xlim(0.0, xmax_props)
            ax_k.set_xlim(0.0, xmax_props)
            ax_mu.set_xlim(0.0, xmax_props)

            pato_props_plot = out_dir / f"{args.out_prefix}_surface_properties_compare.png"
            fig.savefig(pato_props_plot, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            print(
                "Skipping PATO/MFEM surface property plot: missing columns in "
                f"{boundary_diag_csv} or {pato_surface_diag_csv}"
            )
    elif boundary_diag is None or pato_surface_diag is None:
        print(
            "Skipping PATO/MFEM surface diagnostics plot: missing "
            f"{boundary_diag_csv if boundary_diag is None else pato_surface_diag_csv}"
        )

    # Plot 8: MFEM mesh diagnostics at selected probes (div w and w_y).
    if mfem_mesh_diag is not None:
        mesh_names = set(mfem_mesh_diag.dtype.names or ())
        if "time" not in mesh_names:
            print(f"Skipping mesh diagnostics plot: missing time column in {mfem_mesh_diag_csv}")
        else:
            tc_idxs = sorted(
                tc_index(n[len("divw_"):])
                for n in mesh_names
                if n.startswith("divw_TC") and tc_index(n[len("divw_"):]) > 0
            )
            tc_idxs = [i for i in tc_idxs if i > 0]
            if tc_idxs:
                chosen = [1, 2]
                deepest = tc_idxs[-1]
                if deepest not in chosen:
                    chosen.append(deepest)
                chosen = [i for i in chosen if i in tc_idxs]

                fig, (ax_div, ax_wy) = plt.subplots(
                    2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
                )
                colors = ["tab:blue", "tab:orange", "tab:red", "tab:green"]
                t_mesh = mfem_mesh_diag["time"]

                for k, idx in enumerate(chosen):
                    c = colors[k % len(colors)]
                    div_col = f"divw_TC{idx}"
                    wy_col = f"wy_TC{idx}"
                    if div_col in mesh_names:
                        ax_div.plot(
                            t_mesh,
                            mfem_mesh_diag[div_col],
                            lw=2,
                            color=c,
                            label=f"TC{idx}",
                        )
                    if wy_col in mesh_names:
                        ax_wy.plot(
                            t_mesh,
                            mfem_mesh_diag[wy_col],
                            lw=2,
                            color=c,
                            label=f"TC{idx}",
                        )

                if "divw_wall" in mesh_names:
                    ax_div.plot(
                        t_mesh,
                        mfem_mesh_diag["divw_wall"],
                        lw=1.5,
                        ls="--",
                        color="black",
                        label="wall",
                    )
                # For a strip with fixed bottom and receding top, the uniform-compression
                # kinematic prediction is div(w) ~= -v_rec / H(t).
                if probe_y and len(probe_y) >= 2:
                    H0 = float(max(probe_y) - min(probe_y))
                    if H0 > 0.0:
                        rec_t = mfem_mass_t
                        rec = mfem_recession
                        rec_rate = mfem_recession_rate
                        rec_i = np.interp(t_mesh, rec_t, rec)
                        rec_rate_i = np.interp(t_mesh, rec_t, rec_rate)
                        H_i = H0 - rec_i
                        with np.errstate(divide="ignore", invalid="ignore"):
                            divw_pred = -rec_rate_i / H_i
                        divw_pred = np.where(H_i > 1.0e-12, divw_pred, np.nan)
                        ax_div.plot(
                            t_mesh,
                            divw_pred,
                            lw=1.8,
                            ls=":",
                            color="black",
                            label=r"$-\,\dot r / (H_0-r)$",
                        )
                if "wy_wall" in mesh_names:
                    ax_wy.plot(
                        t_mesh,
                        mfem_mesh_diag["wy_wall"],
                        lw=1.5,
                        ls="--",
                        color="black",
                        label="wall",
                    )

                ax_div.set_ylabel(r"$\nabla \cdot w_{mesh}$ (1/s)")
                ax_div.grid(True, alpha=0.25)
                ax_div.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

                ax_wy.set_xlabel("Time (s)")
                ax_wy.set_ylabel(r"$w_y$ (m/s)")
                ax_wy.grid(True, alpha=0.25)
                ax_wy.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

                ax_div.set_xlim(0.0, float(np.nanmax(t_mesh)))
                ax_wy.set_xlim(0.0, float(np.nanmax(t_mesh)))

                mesh_diag_plot = out_dir / f"{args.out_prefix}_mesh_diagnostics.png"
                fig.savefig(mesh_diag_plot, dpi=180, bbox_inches="tight")
                plt.close(fig)
            else:
                print(
                    "Skipping mesh diagnostics plot: no divw_TC* columns found in "
                    f"{mfem_mesh_diag_csv}"
                )
    else:
        print(f"Skipping mesh diagnostics plot: missing {mfem_mesh_diag_csv}")

    # Plot 9: MFEM vs PATO pressure profiles p(y) at selected snapshot times.
    if (
        mfem_pressure_probes is not None
        and pato_p_time is not None
        and pato_p_vals is not None
        and len(pato_p_y) > 0
        and probe_y
        and pressure_profile_times
    ):
        mfem_p_names = list(mfem_pressure_probes.dtype.names or ())
        if "time" not in mfem_p_names:
            print(f"Skipping pressure profile plot: missing time column in {mfem_pressure_probes_csv}")
        else:
            mfem_y_to_series: Dict[float, np.ndarray] = {}
            for name in mfem_p_names:
                if name in ("time", "wall"):
                    continue
                idx = tc_index(name)
                if idx > 0 and idx < len(probe_y):
                    mfem_y_to_series[probe_y[idx]] = mfem_pressure_probes[name]

            matched_points: List[Tuple[float, int, np.ndarray]] = []
            if mfem_y_to_series:
                mfem_y_keys = list(mfem_y_to_series.keys())
                for j, y_pato in enumerate(pato_p_y):
                    k_best = min(mfem_y_keys, key=lambda y: abs(y - y_pato))
                    if abs(k_best - y_pato) <= 1.0e-8:
                        matched_points.append((y_pato, j, mfem_y_to_series[k_best]))

            if len(matched_points) < 2:
                print(
                    "Skipping pressure profile plot: insufficient matched probe points between "
                    f"{mfem_pressure_probes_csv} and {pato_pressure_plot}"
                )
            else:
                matched_points.sort(key=lambda x: x[0], reverse=True)
                n_snap = len(pressure_profile_times)
                ncols = 2 if n_snap > 1 else 1
                nrows = int(np.ceil(n_snap / ncols))
                fig, axes = plt.subplots(
                    nrows, ncols, figsize=(13, 4.2 * nrows), constrained_layout=True
                )
                axes_arr = np.atleast_1d(axes).ravel()
                mfem_ptime = mfem_pressure_probes["time"]

                for ax, t_target in zip(axes_arr, pressure_profile_times):
                    i_mf = int(np.argmin(np.abs(mfem_ptime - t_target)))
                    i_pato = int(np.argmin(np.abs(pato_p_time - t_target)))
                    y_vals = np.array([pt[0] for pt in matched_points], dtype=float)
                    mfem_vals = np.array([pt[2][i_mf] for pt in matched_points], dtype=float)
                    pato_vals = np.array([pato_p_vals[i_pato, pt[1]] for pt in matched_points], dtype=float)

                    ax.plot(mfem_vals, y_vals, "-o", color="tab:blue", lw=2, ms=4, label="MFEM")
                    ax.plot(pato_vals, y_vals, "--s", color="tab:orange", lw=2, ms=4, label="PATO")
                    ax.set_xlabel("Pressure (Pa)")
                    ax.set_ylabel("y (m)")
                    ax.grid(True, alpha=0.25)
                    ax.set_title(
                        f"target={t_target:g}s | MFEM={mfem_ptime[i_mf]:.3g}s | PATO={pato_p_time[i_pato]:.3g}s"
                    )
                    ax.legend(loc="best", fontsize=9)

                for ax in axes_arr[len(pressure_profile_times):]:
                    ax.axis("off")

                pato_pressure_profile_plot = (
                    out_dir / f"{args.out_prefix}_pressure_profiles_compare.png"
                )
                fig.savefig(pato_pressure_profile_plot, dpi=180, bbox_inches="tight")
                plt.close(fig)
    else:
        missing_parts = []
        if mfem_pressure_probes is None:
            missing_parts.append(str(mfem_pressure_probes_csv))
        if pato_p_time is None or pato_p_vals is None or len(pato_p_y) == 0:
            missing_parts.append(str(pato_pressure_plot))
        if missing_parts:
            print(
                "Skipping pressure profile plot: missing " + " and ".join(missing_parts)
            )

    # Plot 10: Common probe-stencil pressure slopes near the top (same discrete stencil in both codes).
    if (
        mfem_pressure_probes is not None
        and pato_p_time is not None
        and pato_p_vals is not None
        and len(pato_p_y) > 0
        and probe_y
    ):
        matched_p = match_pressure_probe_points(mfem_pressure_probes, probe_y, pato_p_y)
        n_pairs = min(3, max(0, len(matched_p) - 1))
        if n_pairs >= 1:
            t_mf_p = np.asarray(mfem_pressure_probes["time"], dtype=float)
            t_pa_p = np.asarray(pato_p_time, dtype=float)
            fig, axes = plt.subplots(
                n_pairs + 1,
                1,
                figsize=(13, 3.2 * (n_pairs + 1)),
                sharex=True,
                constrained_layout=True,
            )
            axes_arr = np.atleast_1d(axes).ravel()
            ratio_ax = axes_arr[-1]
            colors = ["tab:blue", "tab:orange", "tab:green"]

            t0 = max(float(np.nanmin(t_mf_p)), float(np.nanmin(t_pa_p)))
            t1 = min(float(np.nanmax(t_mf_p)), float(np.nanmax(t_pa_p)))
            common_mask = (t_mf_p >= t0) & (t_mf_p <= t1)
            t_common = t_mf_p[common_mask]

            for k in range(n_pairs):
                y_up, name_up, j_up, mfem_up = matched_p[k]
                y_dn, name_dn, j_dn, mfem_dn = matched_p[k + 1]
                dy = y_up - y_dn
                if abs(dy) <= 1.0e-14:
                    continue
                mfem_slope = (np.asarray(mfem_up, dtype=float) - np.asarray(mfem_dn, dtype=float)) / dy
                pato_slope = (np.asarray(pato_p_vals[:, j_up], dtype=float) - np.asarray(pato_p_vals[:, j_dn], dtype=float)) / dy

                ax = axes_arr[k]
                c = colors[k % len(colors)]
                ax.plot(t_mf_p, mfem_slope, lw=2, color=c, label=f"MFEM {name_up}-{name_dn}")
                ax.plot(t_pa_p, pato_slope, lw=2, ls="--", color=c, label="PATO")
                ax.set_ylabel("dp/dy (Pa/m)")
                ax.grid(True, alpha=0.25)
                depth_up = abs(matched_p[0][0] - y_up)
                depth_dn = abs(matched_p[0][0] - y_dn)
                ax.set_title(
                    f"Common stencil {name_up}-{name_dn} (depths {depth_up:.4g} m to {depth_dn:.4g} m)"
                )
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

                if t_common.size >= 2:
                    pato_slope_i = np.interp(t_common, t_pa_p, pato_slope)
                    mfem_slope_i = mfem_slope[common_mask]
                    ratio = safe_divide(mfem_slope_i, pato_slope_i)
                    ratio_ax.plot(
                        t_common,
                        ratio,
                        lw=2,
                        color=c,
                        label=f"{name_up}-{name_dn}",
                    )

            ratio_ax.axhline(1.0, color="black", lw=1.2, ls=":")
            ratio_ax.set_xlabel("Time (s)")
            ratio_ax.set_ylabel("MFEM / PATO")
            ratio_ax.set_title("Common probe-stencil pressure-slope ratio")
            ratio_ax.grid(True, alpha=0.25)
            ratio_ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            xmax_slope = max(float(np.nanmax(t_mf_p)), float(np.nanmax(t_pa_p)))
            for ax in axes_arr:
                ax.set_xlim(0.0, xmax_slope)

            pato_pressure_slope_plot = (
                out_dir / f"{args.out_prefix}_pressure_slope_stencils_compare.png"
            )
            fig.savefig(pato_pressure_slope_plot, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            print("Skipping pressure-slope stencil plot: insufficient matched pressure probes.")
    else:
        print("Skipping pressure-slope stencil plot: missing MFEM or PATO pressure probe data.")

    # Plot 11: MFEM mass-equation probe diagnostics (TC1/TC2/TC3).
    if mfem_mass_eq_probe is not None:
        mass_eq_names = set(mfem_mass_eq_probe.dtype.names or ())
        needed_cols = {"time"}
        for stem in ("pi_total", "gradp_y", "mflux_y"):
            for idx in (1, 2, 3):
                needed_cols.add(f"{stem}_TC{idx}")

        if needed_cols.issubset(mass_eq_names):
            t_meq = np.asarray(mfem_mass_eq_probe["time"], dtype=float)
            fig, axes = plt.subplots(
                3, 1, figsize=(13, 9.0), sharex=True, constrained_layout=True
            )
            colors = ["tab:blue", "tab:orange", "tab:green"]
            tc_ids = [1, 2, 3]

            series_specs = [
                ("pi_total", "pi_total", r"$\pi_{tot}$ (kg/m$^3$/s)"),
                ("gradp_y", "gradp_y", r"$\partial p / \partial y$ (Pa/m)"),
                ("mflux_y", "mflux_y", r"$m_{g,y}$ (kg/m$^2$/s)"),
            ]

            for ax, (key, title_key, ylabel) in zip(axes, series_specs):
                for c, idx in zip(colors, tc_ids):
                    col = f"{key}_TC{idx}"
                    ax.plot(
                        t_meq,
                        np.asarray(mfem_mass_eq_probe[col], dtype=float),
                        lw=2,
                        color=c,
                        label=f"TC{idx}",
                    )
                ax.set_ylabel(ylabel)
                ax.set_title(f"MFEM probe {title_key}: TC1/TC2/TC3")
                ax.grid(True, alpha=0.25)
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            axes[-1].set_xlabel("Time (s)")
            if t_meq.size:
                xmax = float(np.nanmax(t_meq))
                for ax in axes:
                    ax.set_xlim(0.0, xmax)

            mass_eq_probe_plot = out_dir / f"{args.out_prefix}_mass_eq_probe_terms.png"
            fig.savefig(mass_eq_probe_plot, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            missing = sorted(needed_cols - mass_eq_names)
            print(
                "Skipping MFEM mass-equation probe plot: missing columns in "
                f"{mfem_mass_eq_probe_csv}: {', '.join(missing)}"
            )
    else:
        print(f"Skipping MFEM mass-equation probe plot: missing {mfem_mass_eq_probe_csv}")

    # Plot 12: MFEM wall mass-equation probe terms vs PATO top-centerline diagnostics.
    if mfem_mass_eq_probe is not None and pato_surface_diag is not None:
        mfem_meq_names = set(mfem_mass_eq_probe.dtype.names or ())
        pato_names = set(pato_surface_diag.dtype.names or ())
        mfem_needed = {"time", "gradp_y_wall", "mflux_y_wall"}
        pato_needed = {"time", "snGradP_centerline", "mDotGw_centerline"}
        if mfem_needed.issubset(mfem_meq_names) and pato_needed.issubset(pato_names):
            t_mf = np.asarray(mfem_mass_eq_probe["time"], dtype=float)
            t_pa = np.asarray(pato_surface_diag["time"], dtype=float)
            fig, (ax_grad, ax_flux) = plt.subplots(
                2, 1, figsize=(13, 7.5), sharex=True, constrained_layout=True
            )

            ax_grad.plot(
                t_mf,
                np.asarray(mfem_mass_eq_probe["gradp_y_wall"], dtype=float),
                lw=2,
                color="tab:blue",
                label="MFEM gradp_y_wall",
            )
            ax_grad.plot(
                t_pa,
                np.asarray(pato_surface_diag["snGradP_centerline"], dtype=float),
                lw=2,
                ls="--",
                color="tab:orange",
                label="PATO snGradP_centerline",
            )
            ax_grad.set_ylabel("Pressure gradient (Pa/m)")
            ax_grad.set_title("Wall pressure-gradient comparison (MFEM wall vs PATO top centerline)")
            ax_grad.grid(True, alpha=0.25)
            ax_grad.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            ax_flux.plot(
                t_mf,
                np.asarray(mfem_mass_eq_probe["mflux_y_wall"], dtype=float),
                lw=2,
                color="tab:blue",
                label="MFEM mflux_y_wall",
            )
            ax_flux.plot(
                t_pa,
                np.asarray(pato_surface_diag["mDotGw_centerline"], dtype=float),
                lw=2,
                ls="--",
                color="tab:orange",
                label="PATO mDotGw_centerline",
            )
            ax_flux.set_xlabel("Time (s)")
            ax_flux.set_ylabel(r"$m_g \cdot \hat{y}$ (kg/m$^2$/s)")
            ax_flux.set_title("Wall gas mass-flux comparison (MFEM wall vs PATO top centerline)")
            ax_flux.grid(True, alpha=0.25)
            ax_flux.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

            xmax = max(float(np.nanmax(t_mf)), float(np.nanmax(t_pa)))
            ax_grad.set_xlim(0.0, xmax)
            ax_flux.set_xlim(0.0, xmax)

            mass_eq_wall_pato_plot = out_dir / f"{args.out_prefix}_wall_massflux_pressure_compare.png"
            fig.savefig(mass_eq_wall_pato_plot, dpi=180, bbox_inches="tight")
            plt.close(fig)
        else:
            missing_mf = sorted(mfem_needed - mfem_meq_names)
            missing_pa = sorted(pato_needed - pato_names)
            parts = []
            if missing_mf:
                parts.append(f"{mfem_mass_eq_probe_csv}: {', '.join(missing_mf)}")
            if missing_pa:
                parts.append(f"{pato_surface_diag_csv}: {', '.join(missing_pa)}")
            print("Skipping MFEM/PATO wall comparison plot: missing " + " | ".join(parts))
    elif mfem_mass_eq_probe is None or pato_surface_diag is None:
        missing_parts = []
        if mfem_mass_eq_probe is None:
            missing_parts.append(str(mfem_mass_eq_probe_csv))
        if pato_surface_diag is None:
            missing_parts.append(str(pato_surface_diag_csv))
        print("Skipping MFEM/PATO wall comparison plot: missing " + " and ".join(missing_parts))

    if len(mfem_by_depth) != len(am_by_depth):
        print("Probe-count mismatch: using nearest-to-surface shared count =", n_common)
        print("  MFEM probes:", len(mfem_by_depth), "Amaryllis probes:", len(am_by_depth))

    print(f"Wrote: {metrics_csv}")
    print(f"Wrote: {tol_csv}")
    print(f"Wrote: {out_dir / f'{args.out_prefix}_recession_comparison.png'}")
    if pato_diag_plot is not None:
        print(f"Wrote: {pato_diag_plot}")
    if pato_flux_recon_plot is not None:
        print(f"Wrote: {pato_flux_recon_plot}")
    if pato_flux_ratio_plot is not None:
        print(f"Wrote: {pato_flux_ratio_plot}")
    if pato_props_plot is not None:
        print(f"Wrote: {pato_props_plot}")
    if pato_pressure_profile_plot is not None:
        print(f"Wrote: {pato_pressure_profile_plot}")
    if pato_pressure_slope_plot is not None:
        print(f"Wrote: {pato_pressure_slope_plot}")
    if mesh_diag_plot is not None:
        print(f"Wrote: {mesh_diag_plot}")
    if mass_eq_probe_plot is not None:
        print(f"Wrote: {mass_eq_probe_plot}")
    if mass_eq_wall_pato_plot is not None:
        print(f"Wrote: {mass_eq_wall_pato_plot}")
    print(f"Overall PASS: {overall_pass}")
    print(f"Temperature PASS: {temp_pass}")
    print(f"Wall heating RMSE/max: {heat_rmse:.6g} / {heat_max:.6g}")
    print(f"Wall cooling RMSE/max: {cool_rmse:.6g} / {cool_max:.6g}")
    print(f"m_dot_g PASS: {mdot_pass}")
    print(f"Front 98 PASS: {front98_pass}")
    print(f"Front 2 PASS: {front2_pass}")
    print(f"m_dot_c PASS: {mdot_c_pass}")
    print(f"Recession PASS: {recession_pass}")
    if clamp_csv.exists():
        print(
            "B-prime clamp counts "
            f"(p, B'g, T): ({clamp_counts['pressure']:.0f}, "
            f"{clamp_counts['BprimeG']:.0f}, {clamp_counts['temperature']:.0f})"
        )

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
