#!/usr/bin/env python3
"""Overlay the uniaxial responses of every neo-Hookean input of
apps/input/finite_elasticity/verification/homogeneous_deformations on one
figure: the nominal stress P_11, its deviation from the incompressible limit
mu (lambda_1 - lambda_1^-2) of the same mu and the volume ratio J against the stretch lambda_1,
numerical markers on the analytical curves.

    python3 apps/neo_hookean_compare.py [--app build/apps/solid_mechanics] [--stretch 8]
                                        [--steps 70] [--np N] [--out DIR] [--rerun]
                                        [--set kappa=2000]

The runs are those of apps/uniaxial_plots.py (same generated inputs, same
ParaView post-processing and analytical curves, imported from it); a case is
run only if its series is missing from <out>/paraview, unless --rerun. The
inputs share mu and kappa (MPa), so the curves differ only through the energy
(coupled Bonet-Wood or decoupled isochoric), the volumetric law and
compressibility; the deviation panel divides each case by its own mu, so it
stays meaningful if --set changes mu. The three incompressible inputs collapse on
one curve, lambda_1 - lambda_1^-2, and share a colour. The figure goes to
<out>/neo_hookean_compare.png (default out/uniaxial_stretch). Run from the
repository root.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uniaxial_plots as up  # noqa: E402

# (input stem, response it belongs to); responses keep their colour whatever is plotted
CASES = [
    ("uniaxial_neo_hookean", "incompressible"),
    ("symmetry_uniaxial_neo_hookean", "incompressible"),
    ("incompressible_uniaxial_iso_neo_hookean", "incompressible"),
    ("compressible_uniaxial_iso_neo_hookean_mixed", "quadratic"),
    ("compressible_uniaxial_iso_neo_hookean_simo_taylor", "simo_taylor"),
    ("compressible_uniaxial_iso_neo_hookean_j_log_j", "j_log_j"),
    ("compressible_uniaxial_iso_neo_hookean_logarithmic", "logarithmic"),
    ("compressible_uniaxial_neo_hookean", "bonet_wood"),
]
STYLE = {  # colour (fixed categorical order), marker
    "incompressible": ("#2a78d6", "o"),
    "quadratic": ("#eb6834", "s"),
    "simo_taylor": ("#1baf7a", "^"),
    "j_log_j": ("#eda100", "D"),
    "logarithmic": ("#e87ba4", "v"),
    "bonet_wood": ("#008300", "P"),
}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"


def label(response, material):
    mu, _, kappa = up.lame(material)
    if response == "incompressible":
        return r"decoupled, incompressible ($\kappa = \infty$; 3 inputs)"
    nu = (3.0 * kappa - 2.0 * mu) / (2.0 * (3.0 * kappa + mu))
    ratio = rf"$\nu$ = {nu:.4g} ($\kappa/\mu$ = {kappa / mu:.4g})"
    if response == "bonet_wood":
        return f"coupled Bonet-Wood, {ratio}"
    return f"decoupled, {response} U(J), {ratio}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", default="build/apps/solid_mechanics")
    ap.add_argument("--stretch", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=70)
    ap.add_argument("--np", type=int, default=1, dest="np_")
    ap.add_argument("--out", default="out/uniaxial_stretch")
    ap.add_argument("--rerun", action="store_true", help="run every case even if its series exists")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                    help="material parameters to replace where the input has them (repeatable); implies --rerun")
    args = ap.parse_args()
    overrides = {k: float(v) for k, v in (item.split("=", 1) for item in args.overrides)}
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, dx, jx) = plt.subplots(3, 1, figsize=(7.6, 10.0), sharex=True, height_ratios=[3, 1.5, 1.5],
                                     constrained_layout=True)
    incompressible = lambda l: l - l ** -2.0  # P_11 / mu of the incompressible limit
    fine = np.linspace(1.0, args.stretch, 300)
    drawn, worst = set(), 0.0
    print(f"{'case':<52} {'kappa/mu':>9} {'P11 at end':>14} {'J at end':>10} {'max rel err P11':>16}")
    for k, (stem, response) in enumerate(CASES):
        path = os.path.join(up.INPUT_DIR, stem + ".yaml")
        if not os.path.exists(path):
            print(f"{stem:<52} missing input, skipped")
            continue
        new, cfg, stretch = up.make_input(path, args.out, args.stretch, args.steps, overrides)
        series = cfg["output"]["paraview"]
        if args.rerun or overrides or not glob.glob(os.path.join(series, "*.pvd")):
            cmd = [args.app, "-i", new]
            if args.np_ > 1:
                cmd = ["mpirun", "-np", str(args.np_)] + cmd
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            open(os.path.join(args.out, "logs", stem + ".log"), "w").write(proc.stdout)
        l1, l2, P11, _ = up.read_series(series)
        material = cfg["material"]
        mu, _, kappa = up.lame(material)
        exact = np.array([up.analytic(material, l) for l in l1])
        err = (np.abs(P11 - exact[:, 1]) / np.abs(exact[:, 1])).max()
        worst = max(worst, err)
        colour, marker = STYLE[response]
        if response not in drawn:  # one analytical curve and one legend entry per response
            curve = np.array([up.analytic(material, l) for l in fine])
            ax.plot(fine, curve[:, 1], color=colour, lw=2, zorder=2)
            jx.plot(fine, fine * curve[:, 0] ** 2, color=colour, lw=2, zorder=2)
            dx.plot(fine[1:], 100.0 * (curve[1:, 1] / mu / incompressible(fine[1:]) - 1.0), color=colour, lw=2, zorder=2)
            ax.plot([], [], color=colour, lw=2, marker=marker, ms=6, mec="white", mew=0.8,
                    label=label(response, material))
            drawn.add(response)
        # markers every 5th step, staggered so inputs sharing a curve stay visible
        every = (k % 5, 5)
        for a, y in ((ax, P11), (jx, l1 * l2 * l2), (dx, 100.0 * (P11 / mu / incompressible(l1) - 1.0))):
            a.plot(l1, y, ls="none", marker=marker, ms=6, mfc=colour, mec="white", mew=0.8,
                   markevery=every, zorder=3)
        print(f"{stem:<52} {kappa / mu:>9.4g} {P11[-1]:>14.6f} {l1[-1] * l2[-1] ** 2:>10.6f} {err:>16.2e}")

    ax.set_ylabel(r"nominal stress $P_{11}$ (MPa)")
    dx.set_ylabel("$P_{11}$ relative to\nincompressible (%)")
    jx.set_ylabel(r"volume ratio $J$")
    jx.set_xlabel(r"stretch $\lambda_1$")
    ax.set_title("Uniaxial tension, neo-Hookean inputs\nlines: analytical, markers: numerical (every 5th load step)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for a in (ax, dx, jx):
        a.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.tick_params(colors=MUTED)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(MUTED)
    out = os.path.join(args.out, "neo_hookean_compare.png")
    fig.savefig(out, dpi=160)
    print(f"largest relative error of P11 against the analytical curves: {worst:.2e}\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
