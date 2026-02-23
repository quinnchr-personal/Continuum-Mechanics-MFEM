#!/usr/bin/env python3
"""Compare MFEM ablation case-1 outputs against FIAT reference data."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_acceptance_from_yaml(path: Path) -> Dict[str, float]:
    vals: Dict[str, float] = {}
    defaults = {
        "temperature_rmse_max": 150.0,
        "temperature_max_abs_max": 300.0,
        "m_dot_g_peak_rel_error_max": 0.5,
        "m_dot_g_peak_time_error_max": 10.0,
        "front98_rmse_max": 0.01,
        "front2_rmse_max": 0.01,
    }
    vals.update(defaults)

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
            k = k.strip()
            v = v.strip()
            try:
                vals[k] = float(v)
            except ValueError:
                pass
    return vals


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def load_probe_depths_from_yaml(path: Path) -> List[float]:
    """Return probe depths measured from the hot surface (wall)."""
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
                val = line.split("-", 1)[1].strip()
                try:
                    probe_y.append(float(val))
                except ValueError:
                    pass
                continue
            if not line.startswith(" "):
                break

    if not probe_y:
        return []
    y_wall = probe_y[0]
    return [abs(y_wall - y) for y in probe_y]


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


def build_fiat_temperature_by_depth(
    fiat_T: np.ndarray, probe_depths: List[float]
) -> List[Tuple[float, str, np.ndarray]]:
    n_signals = int(fiat_T.shape[1]) - 1
    items: List[Tuple[float, str, np.ndarray]] = []
    for i in range(n_signals):
        name = "wall" if i == 0 else f"TC{i}"
        depth = probe_depths[i] if i < len(probe_depths) else float(i)
        items.append((depth, name, fiat_T[:, i + 1]))
    items.sort(key=lambda x: x[0])
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="ParaView/ablation_case1",
                        help="Directory containing temperature_probes.csv and mass_metrics.csv")
    parser.add_argument("--input", default="Input/input_ablation_case1.yaml",
                        help="Input YAML file with acceptance tolerances")
    parser.add_argument("--fiat-T", default="/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_1.0/data/ref/FIAT/T",
                        help="FIAT temperature file")
    parser.add_argument("--fiat-front", default="/home/quinnchr/Downloads/pato-3.1/tutorials/1D/AblationTestCase_1.0/data/ref/FIAT/pyrolysisFront",
                        help="FIAT pyrolysis-front file")
    parser.add_argument("--out-prefix", default="ablation_case1",
                        help="Prefix for generated plot filenames")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    probes_csv = out_dir / "temperature_probes.csv"
    mass_csv = out_dir / "mass_metrics.csv"

    if not probes_csv.exists() or not mass_csv.exists():
        raise FileNotFoundError(
            "Expected MFEM outputs not found: "
            f"{probes_csv} and {mass_csv}"
        )

    tol = load_acceptance_from_yaml(Path(args.input))

    probes = np.genfromtxt(probes_csv, delimiter=",", names=True)
    mass = np.genfromtxt(mass_csv, delimiter=",", names=True)

    fiat_T = np.loadtxt(args.fiat_T)
    fiat_front = np.loadtxt(args.fiat_front)

    t_fiat = fiat_T[:, 0]

    probe_depths = load_probe_depths_from_yaml(Path(args.input))
    mfem_by_depth = build_mfem_temperature_by_depth(probes, probe_depths)
    fiat_by_depth = build_fiat_temperature_by_depth(fiat_T, probe_depths)
    n_common = min(len(mfem_by_depth), len(fiat_by_depth))
    if n_common == 0:
        raise RuntimeError("No temperature probes available for MFEM/FIAT comparison.")

    mfem_time = probes["time"]
    probe_pairs = list(zip(mfem_by_depth[:n_common], fiat_by_depth[:n_common]))

    temp_metrics: List[Tuple[str, float, float, bool]] = []
    for (d_mf, name_mf, sig_mf), (d_fi, name_fi, sig_fi) in probe_pairs:
        mfem_interp = np.interp(t_fiat, mfem_time, sig_mf)
        r = rmse(mfem_interp, sig_fi)
        m = max_abs(mfem_interp, sig_fi)
        ok = (r <= tol["temperature_rmse_max"]) and (m <= tol["temperature_max_abs_max"])
        sig_label = f"{name_mf}~{name_fi}@depth={d_mf:.6g}m"
        temp_metrics.append((sig_label, r, m, ok))

    # Mass/front metrics from pyrolysisFront reference:
    # col2: pyrolysis mass flux, col7: 2% char front, col8: 98% virgin front
    t_front = fiat_front[:, 0]
    fiat_mdot = fiat_front[:, 2]
    fiat_front2 = fiat_front[:, 7]
    fiat_front98 = fiat_front[:, 8]

    mfem_mass_t = mass["time"]
    mfem_mdot = mass["m_dot_g_surf"]
    mfem_front98 = mass["front_98_virgin"]
    mfem_front2 = mass["front_2_char"]

    mfem_mdot_i = np.interp(t_front, mfem_mass_t, mfem_mdot)
    mfem_front98_i = np.interp(t_front, mfem_mass_t, mfem_front98)
    mfem_front2_i = np.interp(t_front, mfem_mass_t, mfem_front2)

    mdot_rmse = rmse(mfem_mdot_i, fiat_mdot)
    mdot_max = max_abs(mfem_mdot_i, fiat_mdot)

    i_mf = int(np.argmax(mfem_mdot))
    i_fi = int(np.argmax(fiat_mdot))
    mf_peak = float(mfem_mdot[i_mf])
    fi_peak = float(fiat_mdot[i_fi])
    mf_peak_t = float(mfem_mass_t[i_mf])
    fi_peak_t = float(t_front[i_fi])

    peak_rel = abs(mf_peak - fi_peak) / max(abs(fi_peak), 1.0e-12)
    peak_time_err = abs(mf_peak_t - fi_peak_t)

    front98_rmse = rmse(mfem_front98_i, fiat_front98)
    front98_max = max_abs(mfem_front98_i, fiat_front98)
    front2_rmse = rmse(mfem_front2_i, fiat_front2)
    front2_max = max_abs(mfem_front2_i, fiat_front2)

    mdot_pass = (
        peak_rel <= tol["m_dot_g_peak_rel_error_max"]
        and peak_time_err <= tol["m_dot_g_peak_time_error_max"]
    )
    front98_pass = front98_rmse <= tol["front98_rmse_max"]
    front2_pass = front2_rmse <= tol["front2_rmse_max"]
    temp_pass = all(x[3] for x in temp_metrics)

    overall_pass = temp_pass and mdot_pass and front98_pass and front2_pass

    out_csv = out_dir / "fiat_error_metrics.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "signal", "rmse", "max_abs", "metric", "value", "tolerance", "pass"])
        for sig, r, m, ok in temp_metrics:
            w.writerow(["temperature", sig, r, m, "", "", "", int(ok)])
        w.writerow(["mass_flux", "m_dot_g", mdot_rmse, mdot_max, "peak_rel_error",
                    peak_rel, tol["m_dot_g_peak_rel_error_max"], int(peak_rel <= tol["m_dot_g_peak_rel_error_max"])])
        w.writerow(["mass_flux", "m_dot_g", "", "", "peak_time_error",
                    peak_time_err, tol["m_dot_g_peak_time_error_max"], int(peak_time_err <= tol["m_dot_g_peak_time_error_max"])])
        w.writerow(["front", "front_98_virgin", front98_rmse, front98_max, "", "",
                    tol["front98_rmse_max"], int(front98_pass)])
        w.writerow(["front", "front_2_char", front2_rmse, front2_max, "", "",
                    tol["front2_rmse_max"], int(front2_pass)])
        w.writerow(["summary", "overall", "", "", "", "", "", int(overall_pass)])

    # Plot 1: temperature history.
    plt.figure(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    for i, ((d_mf, name_mf, sig_mf), (_, name_fi, sig_fi)) in enumerate(probe_pairs):
        col = "black" if i == 0 else cmap((i - 1) % 10)
        depth_label = f"{d_mf:.4f} m"
        plt.plot(mfem_time, sig_mf, color=col, lw=2,
                 label=f"MFEM {name_mf} ({depth_label})")
        plt.plot(t_fiat, sig_fi, color=col, lw=1.6, ls="--",
                 label=f"FIAT {name_fi} ({depth_label})")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.xlim(0.0, max(float(mfem_time[-1]), float(t_fiat[-1])))
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.out_prefix}_temperature_history.png", dpi=180)
    plt.close()

    # Plot 2: pyrolysis mass flux.
    plt.figure(figsize=(9, 4.8))
    plt.plot(mfem_mass_t, mfem_mdot, color="black", lw=2, label="MFEM")
    plt.plot(t_front, fiat_mdot, color="black", ls="--", lw=2, label="FIAT")
    plt.xlabel("Time (s)")
    plt.ylabel("Pyrolysis mass flux (kg/m2/s)")
    plt.xlim(0.0, max(float(mfem_mass_t[-1]), float(t_front[-1])))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.out_prefix}_pyrolysis_mass_flux.png", dpi=180)
    plt.close()

    # Plot 3: fronts.
    plt.figure(figsize=(9, 4.8))
    plt.plot(mfem_mass_t, mfem_front98, color="black", lw=2, label="MFEM 98% virgin")
    plt.plot(mfem_mass_t, mfem_front2, color="gray", lw=2, label="MFEM 2% char")
    plt.plot(t_front, fiat_front98, color="black", lw=2, ls="--", label="FIAT 98% virgin")
    plt.plot(t_front, fiat_front2, color="gray", lw=2, ls="--", label="FIAT 2% char")
    plt.xlabel("Time (s)")
    plt.ylabel("Depth (m)")
    plt.xlim(0.0, max(float(mfem_mass_t[-1]), float(t_front[-1])))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.out_prefix}_fronts.png", dpi=180)
    plt.close()

    if len(mfem_by_depth) != len(fiat_by_depth):
        print("Probe-count mismatch: using nearest-to-surface shared count =",
              n_common)
        print("  MFEM probes:", len(mfem_by_depth),
              "FIAT probes:", len(fiat_by_depth))

    print(f"Wrote: {out_csv}")
    print(f"Overall PASS: {overall_pass}")
    print(f"Temperature PASS: {temp_pass}")
    print(f"m_dot_g peak PASS: {mdot_pass}")
    print(f"Front 98 PASS: {front98_pass}")
    print(f"Front 2 PASS: {front2_pass}")


if __name__ == "__main__":
    main()
