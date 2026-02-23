#!/usr/bin/env python3
"""Compare MFEM ablation case-2.1 outputs against Amaryllis reference data."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TOL = {
    "temperature_rmse_max": 250.0,
    "temperature_max_abs_max": 500.0,
    "m_dot_g_rmse_max": 0.02,
    "m_dot_g_max_abs_max": 0.06,
    "m_dot_g_peak_rel_error_max": 0.5,
    "m_dot_g_peak_time_error_max": 10.0,
    "front98_max_abs_max": 0.01,
    "front98_rmse_max": 0.01,
    "front2_max_abs_max": 0.01,
    "front2_rmse_max": 0.01,
    "m_dot_c_max_abs_max": 1.0e-8,
    "recession_max_abs_max": 1.0e-8,
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


def load_probe_depths_from_yaml(path: Path) -> List[float]:
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
    t: np.ndarray, a: np.ndarray, b: np.ndarray, t0: float, t1: float
) -> Tuple[float, float]:
    mask = (t >= t0) & (t <= t1)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="ParaView/ablation_case2_1",
        help="Directory containing temperature_probes.csv and mass_metrics.csv",
    )
    parser.add_argument(
        "--input",
        default="Input/input_ablation_case2_1.yaml",
        help="Input YAML file with acceptance tolerances",
    )
    parser.add_argument(
        "--amaryllis-energy",
        default="/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/data/ref/Amaryllis/Amaryllis_Energy_TestCase_2.1.txt",
        help="Amaryllis energy reference file",
    )
    parser.add_argument(
        "--amaryllis-mass",
        default="/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_2.x/data/ref/Amaryllis/Amaryllis_Mass_TestCase_2.1.txt",
        help="Amaryllis mass reference file",
    )
    parser.add_argument(
        "--out-prefix",
        default="ablation_case2_1",
        help="Prefix for generated plot filenames",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    probes_csv = out_dir / "temperature_probes.csv"
    mass_csv = out_dir / "mass_metrics.csv"
    clamp_csv = out_dir / "bprime_clamp_stats.csv"

    if not probes_csv.exists() or not mass_csv.exists():
        raise FileNotFoundError(
            f"Expected MFEM outputs not found: {probes_csv} and {mass_csv}"
        )

    tol = load_acceptance_from_yaml(Path(args.input))

    probes = np.genfromtxt(probes_csv, delimiter=",", names=True)
    mass = np.genfromtxt(mass_csv, delimiter=",", names=True)

    am_energy = ensure_2d(np.loadtxt(args.amaryllis_energy, skiprows=1))
    am_mass = ensure_2d(np.loadtxt(args.amaryllis_mass, skiprows=1))

    probe_depths = load_probe_depths_from_yaml(Path(args.input))
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
        mfem_interp = np.interp(am_time, mfem_time, sig_mf)
        r = rmse(mfem_interp, sig_am)
        m = max_abs(mfem_interp, sig_am)
        ok = (r <= tol["temperature_rmse_max"]) and (m <= tol["temperature_max_abs_max"])
        sig_label = f"{name_mf}~{name_am}@depth={d_mf:.6g}m"
        temp_metrics.append((sig_label, r, m, ok))

    # Segmented wall-temperature metrics for cooldown/heating analysis.
    wall_mf = np.interp(am_time, mfem_time, probe_pairs[0][0][2])
    wall_ref = probe_pairs[0][1][2]
    heat_rmse, heat_max = segmented_rmse_max(am_time, wall_mf, wall_ref, 0.1, 60.0)
    cool_rmse, cool_max = segmented_rmse_max(am_time, wall_mf, wall_ref, 60.1, 120.0)

    # Amaryllis mass columns: time, m_dot_g, m_dot_c, front98, front2, recession.
    t_ref = am_mass[:, 0]
    ref_mdot = am_mass[:, 1]
    ref_front98 = am_mass[:, 3]
    ref_front2 = am_mass[:, 4]

    mfem_mass_t = mass["time"]
    mfem_mdot = mass["m_dot_g_surf"]
    mfem_front98 = mass["front_98_virgin"]
    mfem_front2 = mass["front_2_char"]
    mfem_mdot_c = mass["m_dot_c"]
    mfem_recession = mass["recession"]

    mfem_mdot_i = np.interp(t_ref, mfem_mass_t, mfem_mdot)
    mfem_front98_i = np.interp(t_ref, mfem_mass_t, mfem_front98)
    mfem_front2_i = np.interp(t_ref, mfem_mass_t, mfem_front2)

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

    max_mdot_c = float(np.max(np.abs(mfem_mdot_c)))
    max_recession = float(np.max(np.abs(mfem_recession)))

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
    front98_pass = (
        front98_rmse <= tol["front98_rmse_max"]
        and front98_max <= tol["front98_max_abs_max"]
    )
    front2_pass = (
        front2_rmse <= tol["front2_rmse_max"]
        and front2_max <= tol["front2_max_abs_max"]
    )
    strict_pass = (
        max_mdot_c <= tol["m_dot_c_max_abs_max"]
        and max_recession <= tol["recession_max_abs_max"]
    )

    overall_pass = temp_pass and mdot_pass and front98_pass and front2_pass and strict_pass

    metrics_csv = out_dir / "amaryllis_error_metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "signal", "rmse", "max_abs", "metric", "value", "tolerance", "pass"])
        for sig, r, m, ok in temp_metrics:
            w.writerow(["temperature", sig, r, m, "", "", "", int(ok)])
        w.writerow(["temperature_segment", "wall_heating_0.1_60s", heat_rmse, heat_max,
                    "", "", "", int((heat_rmse <= tol["temperature_rmse_max"]) and (heat_max <= tol["temperature_max_abs_max"]))])
        w.writerow(["temperature_segment", "wall_cooling_60.1_120s", cool_rmse, cool_max,
                    "", "", "", int((cool_rmse <= tol["temperature_rmse_max"]) and (cool_max <= tol["temperature_max_abs_max"]))])

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

        w.writerow(["strict", "m_dot_c", "", "", "max_abs",
                    max_mdot_c, tol["m_dot_c_max_abs_max"], int(max_mdot_c <= tol["m_dot_c_max_abs_max"])])
        w.writerow(["strict", "recession", "", "", "max_abs",
                    max_recession, tol["recession_max_abs_max"], int(max_recession <= tol["recession_max_abs_max"])])

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
    plt.plot(mfem_mass_t, mfem_mdot, color="black", lw=2, label="MFEM")
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

    if len(mfem_by_depth) != len(am_by_depth):
        print("Probe-count mismatch: using nearest-to-surface shared count =", n_common)
        print("  MFEM probes:", len(mfem_by_depth), "Amaryllis probes:", len(am_by_depth))

    print(f"Wrote: {metrics_csv}")
    print(f"Wrote: {tol_csv}")
    print(f"Overall PASS: {overall_pass}")
    print(f"Temperature PASS: {temp_pass}")
    print(f"Wall heating RMSE/max: {heat_rmse:.6g} / {heat_max:.6g}")
    print(f"Wall cooling RMSE/max: {cool_rmse:.6g} / {cool_max:.6g}")
    print(f"m_dot_g PASS: {mdot_pass}")
    print(f"Front 98 PASS: {front98_pass}")
    print(f"Front 2 PASS: {front2_pass}")
    print(f"Strict 2.1 PASS: {strict_pass}")
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
