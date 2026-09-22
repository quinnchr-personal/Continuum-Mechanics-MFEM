#!/usr/bin/env python3
"""Reproduce the result plots of the finite thermoelasticity examples of Anand's
Introduction to coupled theories in solid mechanics (FEniCSx companion codes,
solidmechanicscoupledtheories.github.io, section 3) from the runs of
apps/input/anand_coupled_theories/finite_thermoelasticity/*.yaml, overlaid with
the reference's own histories.

    python3 apps/anand_thermo_plots.py [--inputs DIR] [--logs DIR] [--out DIR] [--reference DIR] [figure ...]

Run it from the repository root, as the app. The data come from the log of a
run beside its ParaView output (logs/<case>.log, as run_set.sh writes it: the
inputs print the probes and the reactions after every step, where the ParaView
output holds every output.every-th step only), else from the ParaView output
(probes from the nodal fields, reactions from <collection>/reactions.csv), or
with --logs DIR from DIR/<case>.log alone; see apps/anand_plots.py, whose
readers this script uses. The figures, one per reference figure, go to plots/
beside the ParaView directories (or beside DIR):

  01_constrained_heating   temperature at the top, centre and bottom of the block, and the
                           pressure at the bottom, against time
  02_adiabatic_stretch     nominal stress and temperature rise against the stretch
  03_heating_contraction   stretch against the nominal stress and against the surface temperature
  04_bilayer_actuator      tip deflection over the length against the surface temperature
  05_plate_flux            top-centre temperature and deflection against time
  06_solar_sail            u_z at A = (70, 60, 0) and B = (100, 0, 0) against the pressure

The reference's histories are its notebooks' timeHist arrays (reference/<case>.csv,
reference/README.md for the columns); its axial force of 02 is the reaction of the
loaded face (a column added to its notebook), of 03 its traction integral.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anand_plots import load_data, paraview_steps, probe, reaction, REF_STYLE  # noqa: E402

REFERENCE_DIR = None
THETA0 = 298.0


def reference(name):
    """The columns timeHist0, timeHist1, ... of REFERENCE_DIR/<name>.csv, or None."""
    path = os.path.join(REFERENCE_DIR or "", name + ".csv")
    if REFERENCE_DIR is None or not os.path.exists(path):
        return None
    return np.atleast_2d(np.genfromtxt(path, delimiter=",", skip_header=1)).T


def scalar(data, name, field):
    """A scalar probe (temperature, pressure) as a flat array."""
    t, v = probe(data, name, field)
    return t, np.asarray(v).reshape(len(t), -1)[:, 0]


def plot_01(cases, fig):
    ax1, ax2 = fig.subplots(1, 2)
    data = cases["01_constrained_heating"]
    ref = reference("01_constrained_heating")
    for i, (name, label) in enumerate([("top", "top (5, 10)"), ("centre", "centre (5, 5)"), ("bottom", "bottom (5, 0)")]):
        t, th = scalar(data, name, "temperature")
        ax1.plot(t, th, "-", color=f"C{i}", label=label + ", this code")
        if ref is not None:
            ax1.plot(ref[0][::8], ref[1 + i][::8], **{**REF_STYLE, "label": label + ", reference (FEniCSx)" if i == 0 else None})
    # The reference's p is the mean pressure (P = ... - J p F^-T); this code's pressure unknown
    # is its negative (P = P_iso + p J F^-T).
    t, p = scalar(data, "bottom", "pressure")
    ax2.plot(t, -p, "-", label="this code (-p)")
    if ref is not None:
        ax2.plot(ref[0][::8], ref[4][::8], **REF_STYLE)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("temperature (K)")
    ax1.set_title("01 constrained heating: temperature", loc="left")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("mean pressure at (5, 0) (kPa)")
    ax2.set_title("01 constrained heating: pressure", loc="left")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)


def plot_02(cases, fig):
    ax1, ax2 = fig.subplots(1, 2)
    data = cases["02_adiabatic_stretch"]
    R, H = 10.0, 10.0
    t, u = probe(data, "top_axis", "displacement")
    _, th = scalar(data, "top_axis", "temperature")
    _, f, _ = reaction(data, "top")
    lam = 1.0 + u[:, 1] / H
    ax1.plot(lam, f[:, 1] / (math.pi * R * R) / 1e3, "-", label="this code (reaction / pi R^2)")
    ax2.plot(lam, th - THETA0, "-", label="this code")
    ref = reference("02_adiabatic_stretch")
    if ref is not None:
        lam_ref = 1.0 + ref[1] / H
        # The notebook's reaction column is the residual summed over the top dofs of its
        # r-weighted (not 2 pi r) form: 2 pi times it is the axial force.
        force = 2.0 * math.pi * ref[4] if ref.shape[0] > 4 and np.any(ref[4] != 0.0) else ref[2]
        ax1.plot(lam_ref[::4], force[::4] / (math.pi * R * R) / 1e3, **{**REF_STYLE, "label": "reference (FEniCSx, reaction)"})
        ax2.plot(lam_ref[::4], ref[3][::4] - THETA0, **REF_STYLE)
    ax1.set_xlabel("stretch")
    ax1.set_ylabel("nominal stress (MPa)")
    ax1.set_title("02 adiabatic stretch: stress", loc="left")
    ax2.set_xlabel("stretch")
    ax2.set_ylabel("temperature rise at the top of the axis (K)")
    ax2.set_title("02 adiabatic stretch: Gough-Joule heating", loc="left")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)


def plot_03(cases, fig):
    ax1, ax2 = fig.subplots(1, 2)
    data = cases["03_heating_contraction"]
    R, H = 10.0, 10.0
    t, u = probe(data, "top_axis", "displacement")
    _, th = scalar(data, "top_axis", "temperature")
    _, f, _ = reaction(data, "base")
    lam = 1.0 + u[:, 1] / H
    ax1.plot(-f[:, 1] / (math.pi * R * R) / 1e3, lam, "-", label="this code (base reaction / pi R^2)")
    ax2.plot(th, lam, "-", label="this code")
    ref = reference("03_heating_contraction")
    if ref is not None:
        lam_ref = 1.0 + ref[1] / H
        ax1.plot(ref[2][::4] / (math.pi * R * R) / 1e3, lam_ref[::4], **{**REF_STYLE, "label": "reference (FEniCSx, traction integral)"})
        ax2.plot(ref[3][::4], lam_ref[::4], **REF_STYLE)
    ax1.set_xlabel("nominal stress (MPa)")
    ax1.set_ylabel("stretch")
    ax1.set_title("03 heating-induced contraction: loading", loc="left")
    ax2.set_xlabel("surface temperature (K)")
    ax2.set_ylabel("stretch")
    ax2.set_title("03 heating-induced contraction: heating", loc="left")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)


def plot_04(cases, fig):
    ax = fig.subplots()
    data = cases["04_bilayer_actuator"]
    L = 100.0
    t, u = probe(data, "tip", "displacement")
    _, th = scalar(data, "tip", "temperature")
    ax.plot(th, u[:, 1] / L, "-", label="this code")
    ref = reference("04_bilayer_actuator")
    if ref is not None:
        ax.plot(ref[2][::2], ref[1][::2] / L, **REF_STYLE)
    ax.set_xlabel("temperature of the heated edges (K)")
    ax.set_ylabel("tip deflection / L")
    ax.set_title("04 bilayer actuator", loc="left")
    ax.legend(fontsize=8)


def plot_05(cases, fig):
    ax1, ax2 = fig.subplots(1, 2)
    data = cases["05_plate_flux"]
    t, u = probe(data, "centre_top", "displacement")
    _, th = scalar(data, "centre_top", "temperature")
    ax1.plot(t, th, "-", label="this code")
    ax2.plot(t, u[:, 1], "-", label="this code")
    ref = reference("05_plate_flux")
    if ref is not None:
        ax1.plot(ref[0][::4], ref[3][::4], **REF_STYLE)
        ax2.plot(ref[0][::4], ref[1][::4], **REF_STYLE)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("temperature at the top of the axis (K)")
    ax1.set_title("05 plate under a surface flux: temperature", loc="left")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("deflection u_z at the top of the axis (mm)")
    ax2.set_title("05 plate under a surface flux: deflection", loc="left")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)


def plot_06(cases, fig):
    ax = fig.subplots()
    data = cases["06_solar_sail"]
    t, uA = probe(data, "A", "displacement")
    _, uB = probe(data, "B", "displacement")
    p = 10.0 * np.asarray(t) / 100.0   # Pa
    ax.plot(p, uA[:, 2], "-", color="C0", label="A = (70, 60, 0), this code")
    ax.plot(p, uB[:, 2], "-", color="C3", label="B = (100, 0, 0), this code")
    ref = reference("06_solar_sail")
    if ref is not None:
        pr = 10.0 * ref[0] / 100.0
        ax.plot(pr[::4], ref[1][::4], **{**REF_STYLE, "label": "A, reference (FEniCSx)"})
        ax.plot(pr[::4], ref[2][::4], **{**REF_STYLE, "marker": "+", "label": "B, reference (FEniCSx)"})
    ax.set_xlabel("pressure (Pa)")
    ax.set_ylabel("u_z (mm)")
    ax.set_title("06 solar sail", loc="left")
    ax.legend(fontsize=8)


FIGURES = {
    "01_constrained_heating": (["01_constrained_heating"], plot_01, (11.0, 4.2)),
    "02_adiabatic_stretch": (["02_adiabatic_stretch"], plot_02, (11.0, 4.2)),
    "03_heating_contraction": (["03_heating_contraction"], plot_03, (11.0, 4.2)),
    "04_bilayer_actuator": (["04_bilayer_actuator"], plot_04, (6.0, 4.2)),
    "05_plate_flux": (["05_plate_flux"], plot_05, (11.0, 4.2)),
    "06_solar_sail": (["06_solar_sail"], plot_06, (6.0, 4.2)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*", help="figure names (default: every figure whose case has output)")
    ap.add_argument("--inputs", default="apps/input/anand_coupled_theories/finite_thermoelasticity",
                    help="directory of the inputs <case>.yaml")
    ap.add_argument("--logs", default=None, help="read DIR/<case>.log instead of the ParaView output")
    ap.add_argument("--out", default=None, help="plot directory (default: plots/ beside the ParaView directories)")
    ap.add_argument("--reference", default=None, help="directory of the reference's histories <case>.csv "
                    "(default: reference/ in the inputs directory)")
    args = ap.parse_args()
    global REFERENCE_DIR
    REFERENCE_DIR = args.reference or os.path.join(args.inputs, "reference")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml
    configs = {}
    for name, (needed, _, _) in FIGURES.items():
        for c in needed:
            path = os.path.join(args.inputs, c + ".yaml")
            if os.path.exists(path):
                configs[c] = yaml.safe_load(open(path))

    def log_dir(c):
        if args.logs is not None:
            return args.logs
        if c not in configs:
            return None
        d = os.path.join(os.path.dirname(configs[c]["output"]["paraview"].rstrip("/")), "logs")
        return d if os.path.exists(os.path.join(d, c + ".log")) else None

    def available(c):
        if log_dir(c) is not None:
            return True
        return args.logs is None and c in configs and paraview_steps(configs[c])[0]

    figures = args.figures or [f for f, (needed, _, _) in FIGURES.items() if any(available(c) for c in needed)]
    if not figures:
        raise SystemExit("no output of the inputs in %s found (run them first, from the repository root)" % args.inputs)
    status = 0
    loaded = {}
    for name in figures:
        if name not in FIGURES:
            print(f"{name}: unknown figure (one of {', '.join(FIGURES)})", file=sys.stderr)
            status = 1
            continue
        needed, plotter, size = FIGURES[name]
        try:
            cases = {}
            beside = None
            for c in needed:
                if not available(c):
                    continue
                if c not in loaded:
                    loaded[c] = load_data(c, configs.get(c), log_dir(c))
                    if log_dir(c) is not None and c in configs:
                        loaded[c] = (loaded[c][0], configs[c]["output"]["paraview"])
                cases[c], beside = loaded[c]
            if not cases:
                raise FileNotFoundError(f"no output of {', '.join(needed)}")
            out = args.out or os.path.join(os.path.dirname(beside.rstrip("/")), "plots")
            os.makedirs(out, exist_ok=True)
            fig = plt.figure(figsize=size)
            plotter(cases, fig)
            for ax in fig.axes:
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            png = os.path.join(out, name + ".png")
            fig.savefig(png, dpi=150)
            plt.close(fig)
            print(f"{name}: {png} ({', '.join(cases)})")
        except (KeyError, FileNotFoundError, ValueError) as e:
            status = 1
            print(f"{name}: cannot plot ({type(e).__name__}: {e})", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
