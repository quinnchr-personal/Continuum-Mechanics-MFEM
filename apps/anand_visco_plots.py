#!/usr/bin/env python3
"""Reproduce the result plots of the finite viscoelasticity examples of Anand's
Introduction to coupled theories in solid mechanics (FEniCSx companion codes,
solidmechanicscoupledtheories.github.io, section 2) from the runs of
apps/input/anand_coupled_theories/finite_viscoelasticity/*.yaml, overlaid with
the reference's own histories and, where the reference page shows them, the
experiments.

    python3 apps/anand_visco_plots.py [--inputs DIR] [--logs DIR] [--out DIR] [--reference DIR] [figure ...]

Run it from the repository root, as the app. The data come from the log of a
run beside its ParaView output (logs/<case>.log, as logs/run_set.sh writes it:
the inputs print the probes and the reactions after every step, where the
ParaView output holds every output.every-th step only), else from the ParaView
output (probes from the nodal fields, reactions and contact resultants from
<collection>/reactions.csv), or with --logs DIR from DIR/<case>.log alone; see
apps/anand_plots.py, whose readers this script uses. The figures, one per
reference figure, go to plots/ beside the ParaView directories (or beside DIR):

  01_uniaxial_equilibrium  nominal stress vs stretch at 1e-4 /s, the experiment of Wang et al. (2016)
  02_uniaxial_rates        the three rates of 02_uniaxial_rate_* vs stretch, the experiments of
                           Hossain et al. (2012); the reference's published histories and, dotted,
                           the same notebook with its exponent typo corrected (reference/README.md)
  03_stress_relaxation     stretch and nominal stress against time
  04_creep                 traction and stretch against time
  05_stretch_hold          nominal stress vs stretch over the equilibrium curve, and the histories
  06_sinusoidal_tension    nominal stress vs stretch, two cycles
  07_sinusoidal_shear      shear force per unit area vs shear strain, two cycles
  08_cube_footing          pressure against time, settlement of the corner against pressure
  09_bushing_shear         shear force vs shear strain u/L
  10_beam_impulse          traction pulse and tip deflection against time
  11_column_buckling       imposed shortening and axial force against time
  12_sphere_indentation    applied and measured depth, indenter force against time

The reference's histories are its notebooks' timeHist arrays (reference/<case>.csv,
reference/README.md for the columns); its forces are the residual summed over the
loaded face where that column was added, else its traction integral.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anand_plots import load_data, paraview_steps, probe, reaction, REF_STYLE  # noqa: E402

REFERENCE_DIR = None
EXP_STYLE = dict(marker="o", ms=4, ls="none", mfc="none", color="C3", label="experiment")


def reference(name):
    """The columns timeHist0, timeHist1, ... of REFERENCE_DIR/<name>.csv, or None."""
    path = os.path.join(REFERENCE_DIR or "", name + ".csv")
    if REFERENCE_DIR is None or not os.path.exists(path):
        return None
    return np.atleast_2d(np.genfromtxt(path, delimiter=",", skip_header=1)).T


def experiment(name):
    """REFERENCE_DIR/exp_data/<name>.csv as (x, y) columns, or None."""
    path = os.path.join(REFERENCE_DIR or "", "exp_data", name + ".csv")
    if REFERENCE_DIR is None or not os.path.exists(path):
        return None
    d = np.genfromtxt(path, delimiter=",")
    return d if d.shape[1] == 2 else d.T


def ref_force(ref, traction, rxn, scale=1.0):
    """The reference's force: its reaction column when the CSV has one, else its traction integral."""
    if ref.shape[0] > rxn and np.any(ref[rxn] != 0.0):
        return scale * ref[rxn], "reference (FEniCSx, reaction)"
    return scale * ref[traction], "reference (FEniCSx, traction integral)"


def stretch_stress(data, L, probe_name="corner", entry="loaded"):
    """Stretch of the loaded face and the nominal stress (reaction / reference area) of a block."""
    t, u = probe(data, probe_name, "displacement")
    tr, f, _ = reaction(data, entry)
    return t, 1.0 + u[:, 1] / L, f[:, 1] / (L * L)


def table(t, times, amps):
    return np.interp(t, times, amps)


# ----------------------------------------------------------------------- plots

def plot_01(cases, fig):
    ax = fig.subplots()
    t, lam, P = stretch_stress(cases["01_uniaxial_equilibrium"], 1.0)
    ax.plot(lam, P, "-", label="this code, rate 1e-4 /s")
    ref = reference("01_uniaxial_equilibrium")
    if ref is not None:
        f, label = ref_force(ref, 2, 3)
        ax.plot(1.0 + ref[1], f, **{**REF_STYLE, "label": label})
    exp = experiment("wang2016_eq_stretch")
    if exp is not None:
        ax.plot(exp[:, 0] - exp[0, 0] + 1.0, exp[:, 1] - exp[0, 1], **{**EXP_STYLE, "label": "experiment, Wang et al. (2016)"})
    ax.set_xlabel("stretch")
    ax.set_ylabel("nominal stress $T_{R22}$ (kPa)")
    ax.set_title("01 equilibrium uniaxial tension of VHB 4910")
    ax.legend(fontsize=8)


def plot_02(cases, fig):
    ax = fig.subplots()
    for k, rate in enumerate(["0p01", "0p03", "0p05"]):
        case = "02_uniaxial_rate_" + rate
        if case not in cases:
            continue
        t, lam, P = stretch_stress(cases[case], 1.0)
        ax.plot(lam, P, "-", color=f"C{k}", label=f"this code, {rate.replace('p', '.')} /s")
        ref = reference(case)
        if ref is not None:
            f, label = ref_force(ref, 2, 3)
            ax.plot(1.0 + ref[1], f, **{**REF_STYLE, "color": f"C{k}", "label": "reference (FEniCSx), as published" if k == 0 else None})
        fixed = reference(case + "_corrected")
        if fixed is not None:
            f, _ = ref_force(fixed, 2, 3)
            ax.plot(1.0 + fixed[1], f, ":", color="k", lw=1.0, label="reference, exponent typo corrected" if k == 0 else None)
        exp = experiment("Hossain2012_stretch2p5_rate" + rate)
        if exp is not None:
            ax.plot(exp[:, 0] - exp[0, 0] + 1.0, exp[:, 1] - exp[0, 1], **{**EXP_STYLE, "color": f"C{k}",
                    "label": "experiment, Hossain et al. (2012)" if k == 0 else None})
    ax.set_xlim(1.0, 2.6)
    ax.set_xlabel("stretch")
    ax.set_ylabel("nominal stress $T_{R22}$ (kPa)")
    ax.set_title("02 uniaxial tension of VHB 4910 at three stretch rates")
    ax.legend(fontsize=8)


def histories(fig, t, top, bottom, top_label, bottom_label, title, ref=None, ref_top=None, ref_bottom=None,
              ref_labels=("reference (FEniCSx)", "reference (FEniCSx)")):
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.plot(t, top, "-", label="this code")
    ax2.plot(t, bottom, "-", label="this code")
    if ref is not None:
        if ref_top is not None:
            ax1.plot(ref[0], ref_top, **{**REF_STYLE, "label": ref_labels[0]})
        if ref_bottom is not None:
            ax2.plot(ref[0], ref_bottom, **{**REF_STYLE, "label": ref_labels[1]})
    ax1.set_ylabel(top_label)
    ax2.set_ylabel(bottom_label)
    ax2.set_xlabel("time (s)")
    ax1.set_title(title)
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)


def plot_03(cases, fig):
    t, lam, P = stretch_stress(cases["03_stress_relaxation"], 1.0)
    ref = reference("03_stress_relaxation")
    f = label = None
    if ref is not None:
        f, label = ref_force(ref, 2, 3)
    histories(fig, t, lam, P, "stretch", "nominal stress $T_{R22}$ (kPa)", "03 stress relaxation of VHB 4910",
              ref, 1.0 + ref[1] if ref is not None else None, f, ("reference (FEniCSx)", label))


def plot_04(cases, fig):
    t, u = probe(cases["04_creep"], "face_center", "displacement")
    trac = 50.0 * np.minimum(t / 0.1, 1.0)
    ref = reference("04_creep")
    histories(fig, t, trac, 1.0 + u[:, 1], "nominal traction $T_{R22}$ (kPa)", "stretch", "04 creep of VHB 4910",
              ref, ref[1] if ref is not None else None, 1.0 + ref[2] if ref is not None else None)


def plot_05(cases, fig):
    ax = fig.subplots()
    t, lam, P = stretch_stress(cases["05_stretch_hold"], 1.0)
    ax.plot(lam, P, "-", label="this code")
    ref = reference("05_stretch_hold")
    if ref is not None:
        f, label = ref_force(ref, 2, 3)
        ax.plot(1.0 + ref[1], f, **{**REF_STYLE, "ms": 2, "label": label})
    eq = experiment("eq_stretch_stress_data")   # the reference's own equilibrium curve (its FV01 run)
    if eq is not None:
        ax.plot(eq[:, 0], eq[:, 1], "--", color="k", lw=1.0, label="equilibrium response (reference's 01 run)")
    ax.set_xlabel("stretch")
    ax.set_ylabel("nominal stress $T_{R22}$ (kPa)")
    ax.set_ylim(-30, 40)
    ax.set_title("05 stretch and hold of VHB 4910")
    ax.legend(fontsize=8)


def plot_05_time(cases, fig):
    t, lam, P = stretch_stress(cases["05_stretch_hold"], 1.0)
    ref = reference("05_stretch_hold")
    f = label = None
    if ref is not None:
        f, label = ref_force(ref, 2, 3)
    histories(fig, t, lam, P, "stretch", "nominal stress $T_{R22}$ (kPa)", "05 stretch and hold of VHB 4910, histories",
              ref, 1.0 + ref[1] if ref is not None else None, f, ("reference (FEniCSx)", label))


def plot_06(cases, fig):
    ax = fig.subplots()
    t, lam, P = stretch_stress(cases["06_sinusoidal_tension"], 5.0)
    ax.plot(lam, P, "-", label="this code")
    ref = reference("06_sinusoidal_tension")
    if ref is not None:
        f, label = ref_force(ref, 2, 3, 1.0 / 25.0)
        ax.plot(1.0 + ref[1] / 5.0, f, **{**REF_STYLE, "label": label})
    ax.set_xlabel("stretch")
    ax.set_ylabel("nominal stress $T_{R22}$ (kPa)")
    ax.set_title("06 sinusoidal tension of VHB 4910 at 1 Hz, two cycles")
    ax.legend(fontsize=8)


def plot_07(cases, fig):
    ax = fig.subplots()
    data = cases["07_sinusoidal_shear"]
    t, u = probe(data, "top_corner", "displacement")
    _, f, _ = reaction(data, "top")
    ax.plot(u[:, 0], f[:, 0], "-", label="this code (reaction / area)")
    ref = reference("07_sinusoidal_shear")
    if ref is not None:
        fr, label = ref_force(ref, 2, 3)
        ax.plot(ref[1], fr, **{**REF_STYLE, "label": label})
    ax.set_xlabel("shear strain $\\gamma$")
    ax.set_ylabel("shear stress $T_{R12}$ (kPa)")
    ax.set_title("07 sinusoidal shear of VHB 4910 at 1 Hz, two cycles")
    ax.legend(fontsize=8)


def plot_08(cases, fig):
    ax1, ax2 = fig.subplots(1, 2)
    t, u = probe(cases["08_cube_footing"], "footing_corner", "displacement")
    p = 1500.0 * table(t, [0.0, 30.0, 60.0, 460.0], [0.0, 1.0, 0.0, 0.0])
    ax1.plot(t, p, "o-", ms=3, label="this code")
    ax2.plot(p, u[:, 2], "o-", ms=3, label="this code")
    ref = reference("08_cube_footing")
    if ref is not None:
        ax1.plot(ref[0], ref[1], **REF_STYLE)
        ax2.plot(ref[1], ref[2], **REF_STYLE)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("pressure on the patch (kPa)")
    ax2.set_xlabel("pressure on the patch (kPa)")
    ax2.set_ylabel("displacement of the footing corner (mm)")
    ax1.set_title("08 viscoelastic cube footing (NBR)", loc="left")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)


def plot_09(cases, fig):
    ax = fig.subplots()
    data = cases["09_bushing_shear"]
    t, f, _ = reaction(data, "top")
    L = 19.3
    u = 10.0 * table(t, [0, 15, 45, 75, 105, 120], [0, 1, -1, 1, -1, 0])
    ax.plot(u / L, f[:, 0] / 1e3, "-", label="this code (reaction)")
    ref = reference("09_bushing_shear")
    if ref is not None:
        fr, label = ref_force(ref, 2, 3, 1e-3)
        ax.plot(ref[1] / L, fr, **{**REF_STYLE, "label": label})
    ax.set_xlabel("shear strain $u/L$")
    ax.set_ylabel("shear force (N)")
    ax.set_title("09 shear of an NBR bushing")
    ax.legend(fontsize=8)


def plot_10(cases, fig):
    t, u = probe(cases["10_beam_impulse"], "tip", "displacement")
    trac = 2.0 * table(t, [0.0, 0.004, 0.008, 0.4], [0.0, 1.0, 0.0, 0.0])
    ref = reference("10_beam_impulse")
    histories(fig, t, trac, u[:, 1], "shear traction (kPa)", "tip deflection (mm)",
              "10 impulse response of a viscoelastic NBR cantilever",
              ref, ref[1] if ref is not None else None, ref[2] if ref is not None else None)


def plot_11(cases, fig):
    data = cases["11_column_buckling"]
    t, f, _ = reaction(data, "top")
    s = np.minimum(t / 0.02, 1.0)
    shortening = 0.75 * (3.0 * s ** 2 - 2.0 * s ** 3)
    ref = reference("11_column_buckling")
    fr = label = None
    if ref is not None:
        fr, label = ref_force(ref, 2, 3, -1.0)
    histories(fig, t, shortening, -f[:, 2], "shortening (mm)", "axial force (mN)",
              "11 visco-dynamic buckling of an NBR column", ref, -ref[1] if ref is not None else None, fr,
              ("reference (FEniCSx)", label))


def plot_12(cases, fig):
    ax1, ax2 = fig.subplots(1, 2)
    data = cases["12_sphere_indentation"]
    t, u = probe(data, "corner", "displacement")
    ax1.plot(t, 10.0 * np.minimum(t / 60.0, 1.0), "-", color="C1", label="applied depth")
    ax1.plot(t, -u[:, 2], "o-", ms=3, label="this code, $-u_z$ under the indenter")
    tr, f, _ = reaction(data, "indenter")
    ax2.plot(tr, -f[:, 2] / 1e3, "o-", ms=3, label="this code (contact resultant)")
    ref = reference("12_sphere_indentation")
    if ref is not None:
        ax1.plot(ref[0], -ref[2], **{**REF_STYLE, "label": "reference (FEniCSx, coarse box)"})
        if ref.shape[0] > 4 and np.any(ref[4] != 0.0):
            ax2.plot(ref[0], ref[4] / 1e3, **{**REF_STYLE, "label": "reference (FEniCSx, coarse box, contact term)"})
        ax2.plot(ref[0], -ref[3] / 1e3, **{**REF_STYLE, "marker": "+", "label": "reference (FEniCSx, coarse box, traction integral)"})
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("indenter depth (mm)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("indenter force (N)")
    ax1.set_title("12 spherical indentation of an NBR block", loc="left")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)


# figure name -> (cases it needs, plot function, figure size)
FIGURES = {
    "01_uniaxial_equilibrium": (["01_uniaxial_equilibrium"], plot_01, (6.0, 4.2)),
    "02_uniaxial_rates": (["02_uniaxial_rate_0p01", "02_uniaxial_rate_0p03", "02_uniaxial_rate_0p05"], plot_02, (6.0, 4.2)),
    "03_stress_relaxation": (["03_stress_relaxation"], plot_03, (6.0, 5.4)),
    "04_creep": (["04_creep"], plot_04, (6.0, 5.4)),
    "05_stretch_hold": (["05_stretch_hold"], plot_05, (6.0, 4.2)),
    "05_stretch_hold_time": (["05_stretch_hold"], plot_05_time, (6.0, 5.4)),
    "06_sinusoidal_tension": (["06_sinusoidal_tension"], plot_06, (6.0, 4.2)),
    "07_sinusoidal_shear": (["07_sinusoidal_shear"], plot_07, (6.0, 4.2)),
    "08_cube_footing": (["08_cube_footing"], plot_08, (10.0, 4.2)),
    "09_bushing_shear": (["09_bushing_shear"], plot_09, (6.0, 4.2)),
    "10_beam_impulse": (["10_beam_impulse"], plot_10, (6.0, 5.4)),
    "11_column_buckling": (["11_column_buckling"], plot_11, (6.0, 5.4)),
    "12_sphere_indentation": (["12_sphere_indentation"], plot_12, (10.0, 4.2)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*", help="figure names (default: every figure whose cases have output)")
    ap.add_argument("--inputs", default="apps/input/anand_coupled_theories/finite_viscoelasticity",
                    help="directory of the inputs <case>.yaml")
    ap.add_argument("--logs", default=None, help="read DIR/<case>.log instead of the ParaView output")
    ap.add_argument("--out", default=None, help="plot directory (default: plots/ beside the ParaView directories)")
    ap.add_argument("--reference", default=None, help="directory of the reference's histories <case>.csv and "
                    "exp_data/ (default: reference/ in the inputs directory)")
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
        """The log directory beside the ParaView output of c, when its log is there."""
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
