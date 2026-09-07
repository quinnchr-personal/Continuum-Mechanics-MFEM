#!/usr/bin/env python3
"""Reproduce the result plots of the finite elasticity examples of Anand's
Introduction to coupled theories in solid mechanics (FEniCSx companion codes,
solidmechanicscoupledtheories.github.io, section 1) from the logs of
apps/input/anand_coupled_theories/finite_elasticity/*.yaml, and overlay a
reference curve where one exists.

    python3 apps/anand_plots.py [--logs DIR] [--out DIR] [case ...]

A log is the stdout of `solid_mechanics -i <case>.yaml` (the inputs print the
probes and the reactions after every step); the default log directory is
out/anand_coupled_theories/finite_elasticity/logs and the plots go next to it.
The reference site overlays no analytical curves; the ones added here are:

  01 uniaxial tension     nominal stress vs stretch of the homogeneous
                          Arruda-Boyce block at K = 1000 G (lateral stretch
                          from sigma_22 = 0), with this code's five-term
                          series and the reference's Pade form, and the
                          incompressible limit
  02 simple shear         nominal shear stress vs shear strain, same models
  03 torsion              torque and axial force from Rivlin's universal
                          torsion with the Arruda-Boyce law (quadrature in r)
  05, 06 inflation        pressure vs inner-wall displacement of the
                          incompressible thick-walled cylinder / sphere
                          (quadrature in r), plus the limit point
  08 buckling             the Euler load 4 pi^2 E I / L^2 of the clamped column
  09 inclusion            the homogeneous curve of the matrix alone
  04 plate, 07 footing, 10 twist: numerical curves only.
"""
import argparse
import math
import os
import re
import sys

import numpy as np

G0, LAMBDA_L, KBULK = 280.0, 5.12, 280000.0     # kPa; the reference's Arruda-Boyce
N_CHAINS = LAMBDA_L ** 2
AB_C = [0.5, 1.0 / 20.0, 11.0 / 1050.0, 19.0 / 7000.0, 519.0 / 673750.0]

STEP = re.compile(r"^step (\d+) t = ([0-9.eE+-]+) (probe|reaction) (\S+?)(?: at \((.*?)\))?: (\w+) =(.*)$")


def psi1_series(I1):
    """Psi_1 = dPsi/dI1 of this code's Arruda-Boyce (five-term series in I1/N)."""
    I1 = np.asarray(I1, dtype=float)
    return G0 * sum((i + 1) * AB_C[i] * (I1 / N_CHAINS) ** i for i in range(5))


def psi1_pade(I1):
    """Psi_1 of the reference: Gshear/2 with the Pade inverse Langevin,
    Gshear = G0 (lambda_L / (3 lambda_bar)) L^-1(lambda_bar / lambda_L)."""
    lb = np.sqrt(np.asarray(I1, dtype=float) / 3.0)
    z = np.minimum(lb / LAMBDA_L, 0.95)
    beta = z * (3.0 - z * z) / (1.0 - z * z)
    return 0.5 * G0 * LAMBDA_L / (3.0 * lb) * beta


MODELS = [("series (this code)", psi1_series, "-"), ("Pade (reference)", psi1_pade, "--")]


def read_log(path):
    """{name: {"t": [...], field: [[...], ...]}} for probes and reactions."""
    data = {}
    for line in open(path):
        m = STEP.match(line.strip())
        if not m:
            continue
        step, t, kind, name, point, field, values = m.groups()
        key = (kind, name)
        entry = data.setdefault(key, {"t": []})
        vals = [float(v) for v in values.split()] if kind == "probe" else None
        if kind == "reaction":
            f, mom = values.split(" moment =")
            vals = {"force": [float(v) for v in f.split()], "moment": [float(v) for v in mom.split()]}
            if len(entry["t"]) == 0 or entry["t"][-1] != float(t):
                entry["t"].append(float(t))
            entry.setdefault("force", []).append(vals["force"])
            entry.setdefault("moment", []).append(vals["moment"])
        else:
            if field not in entry:
                entry[field] = []
            if len(entry[field]) == len(entry["t"]):
                entry["t"].append(float(t))
            entry[field].append(vals)
    return {k: {f: np.array(v) for f, v in d.items()} for k, d in data.items()}


def probe(data, name, field):
    d = data[("probe", name)]
    return d["t"], d[field]


def reaction(data, name):
    d = data[("reaction", name)]
    return d["t"], d["force"], d["moment"]


# ------------------------------------------------------------------ references

def uniaxial_nominal(lam, psi1, kappa=KBULK):
    """Nominal stress of homogeneous uniaxial tension of the decoupled model
    sigma = 2 Psi_1(I1bar) J^-1 dev(Bbar) + kappa (J - 1) I with the lateral
    stretch from sigma_22 = 0 (kappa = None: incompressible, 2 Psi_1 (lambda - lambda^-2))."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    if kappa is None:
        I1 = lam ** 2 + 2.0 / lam
        return 2.0 * psi1(I1) * (lam - lam ** -2)
    from scipy.optimize import brentq
    out = np.empty_like(lam)
    for k, l1 in enumerate(lam):
        def sigma22(l2):
            J = l1 * l2 * l2
            I1b = J ** (-2.0 / 3.0) * (l1 * l1 + 2.0 * l2 * l2)
            return 2.0 * float(psi1(I1b)) * J ** (-5.0 / 3.0) * (l2 * l2 - l1 * l1) / 3.0 + kappa * (J - 1.0)
        l2 = brentq(sigma22, 0.2, 1.5)
        J = l1 * l2 * l2
        I1b = J ** (-2.0 / 3.0) * (l1 * l1 + 2.0 * l2 * l2)
        s11 = 2.0 * float(psi1(I1b)) * J ** (-5.0 / 3.0) * 2.0 * (l1 * l1 - l2 * l2) / 3.0 + kappa * (J - 1.0)
        out[k] = s11 * J / l1
    return out


def shear_nominal(gamma, psi1):
    """Incompressible simple shear: P_12 = sigma_12 = 2 Psi_1 gamma."""
    return 2.0 * psi1(3.0 + gamma ** 2) * gamma


def torsion(psi, a, psi1, n=400):
    """Rivlin torsion of an incompressible cylinder: torque and axial force
    by quadrature; sigma_thetaz = 2 Psi_1 gamma, sigma_thetatheta - sigma_rr =
    2 Psi_1 gamma^2, gamma = psi r, sigma_rr(a) = 0, sigma_zz = sigma_rr."""
    r = np.linspace(0.0, a, n + 1)
    g = psi * r
    p1 = np.array([psi1(3.0 + gi * gi) for gi in g])
    torque = 2.0 * math.pi * np.trapz(2.0 * p1 * g * r * r, r)
    # sigma_rr(r) = -int_r^a 2 Psi_1 gamma^2 / rho drho
    integrand = 2.0 * p1 * g * g / np.where(r > 0, r, 1.0)
    integrand[0] = 0.0
    srr = -(np.trapz(integrand, r) - np.concatenate(([0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r)))))
    axial = 2.0 * math.pi * np.trapz(srr * r, r)
    return torque, axial


def inflation_pressure(a, A, B, psi1, sphere, n=200):
    """Incompressible thick-walled cylinder (plane strain) or sphere: the
    pressure that holds the inner radius at a, by quadrature in r."""
    if sphere:
        b = (B ** 3 - A ** 3 + a ** 3) ** (1.0 / 3.0)
        r = np.linspace(a, b, n + 1)
        R = (r ** 3 - a ** 3 + A ** 3) ** (1.0 / 3.0)
        lam = r / R
        I1 = 2.0 * lam ** 2 + lam ** -4
        diff = np.array([2.0 * psi1(i) for i in I1]) * (lam ** 2 - lam ** -4)
        return np.trapz(2.0 * diff / r, r)
    b = math.sqrt(B * B - A * A + a * a)
    r = np.linspace(a, b, n + 1)
    R = np.sqrt(r * r - a * a + A * A)
    lam = r / R
    I1 = lam ** 2 + lam ** -2 + 1.0
    diff = np.array([2.0 * psi1(i) for i in I1]) * (lam ** 2 - lam ** -2)
    return np.trapz(diff / r, r)


# ----------------------------------------------------------------------- plots

def plot_01(data, ax):
    t, u = probe(data, "corner", "displacement")
    _, f, _ = reaction(data, "loaded")
    lam = (10.0 + u[:, 1]) / 10.0
    ax.plot(lam, f[:, 1] / 100.0 / 1e3, "o-", ms=3, label="this code (reaction / area)")
    lam_ref = np.linspace(1.0, lam.max(), 200)
    for label, psi1, ls in MODELS:
        ax.plot(lam_ref, uniaxial_nominal(lam_ref, psi1) / 1e3, ls, label=f"homogeneous, K = 1000 G, {label}")
    ax.plot(lam_ref, uniaxial_nominal(lam_ref, psi1_series, None) / 1e3, ":", color="C1", alpha=0.7,
            label="homogeneous, incompressible, series")
    ax.set_xlabel("stretch")
    ax.set_ylabel("nominal stress P22 (MPa)")
    ax.set_title("01 uniaxial tension")


def plot_02(data, ax):
    t, u = probe(data, "top_corner", "displacement")
    _, f, _ = reaction(data, "top")
    gamma = u[:, 0] / 1.0
    ax.plot(gamma, f[:, 0] / 1.0, "o-", ms=3, label="this code (reaction / area)")
    g_ref = np.linspace(-1.0, 1.0, 200)
    for label, psi1, ls in MODELS:
        ax.plot(g_ref, shear_nominal(g_ref, psi1), ls, label=f"homogeneous, {label}")
    ax.set_xlabel("shear strain")
    ax.set_ylabel("nominal shear stress P12 (kPa)")
    ax.set_title("02 simple shear, two cycles")


def plot_03(data, ax):
    t, f, m = reaction(data, "top")
    L, a = 25.4, 12.7
    psi = 2.5 * t / L                       # rad/mm
    ax.plot(psi * 1e3, m[:, 2] * 1e-6, "o-", ms=3, label="torque, this code")
    ax2 = ax.twinx()
    ax2.plot(psi * 1e3, f[:, 2] * 1e-3, "s-", ms=3, color="C3", label="axial force, this code")
    psi_ref = np.linspace(0.0, psi.max(), 60)
    for label, psi1, ls in MODELS:
        tq = np.array([torsion(p, a, psi1)[0] for p in psi_ref])
        ax.plot(psi_ref * 1e3, tq * 1e-6, ls, color="C0", alpha=0.7, label=f"torque, Rivlin + {label}")
        fz = np.array([torsion(p, a, psi1)[1] for p in psi_ref])
        ax2.plot(psi_ref * 1e3, fz * 1e-3, ls, color="C3", alpha=0.7, label=f"axial force, Rivlin + {label}")
    ax.set_xlabel("twist per unit length (rad/m)")
    ax.set_ylabel("torque (N m)")
    ax2.set_ylabel("axial force (N)")
    ax.set_title("03 fixed-end torsion")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)
    return True


def plot_04(data, ax):
    t, u = probe(data, "end_corner", "displacement")
    _, f, _ = reaction(data, "loaded")
    L0, W0, t0 = 15.0, 10.0, 1.0
    lam = (L0 + u[:, 0]) / L0
    ax.plot(lam, 2.0 * f[:, 0] / (W0 * t0) / 1e3, "o-", ms=3, label="this code (reaction / area)")
    ax.set_xlabel("stretch")
    ax.set_ylabel("nominal stress P11 (MPa)")
    ax.set_title("04 plate with a hole")


def plot_inflation(data, ax, name, A, B, p_max, sphere):
    t, u = probe(data, "inner", "displacement")
    ax.plot(p_max * t, u[:, 0], "o-", ms=3, label="this code")
    a_ref = np.linspace(A, A + 1.15 * max(u[:, 0].max(), 0.1), 120)
    for label, psi1, ls in MODELS:
        p_ref = np.array([inflation_pressure(a, A, B, psi1, sphere) for a in a_ref])
        ax.plot(p_ref, a_ref - A, ls, label=f"incompressible, {label}")
    ax.set_xlabel("internal pressure (kPa)")
    ax.set_ylabel("displacement of the inner wall (mm)")
    ax.set_title(name)


def plot_05(data, ax):
    plot_inflation(data, ax, "05 cylinder inflation", 10.0, 11.0, 50.0, False)


def plot_06(data, ax):
    plot_inflation(data, ax, "06 sphere inflation", 10.0, 11.0, 35.0, True)


def plot_07(data, ax):
    t, u = probe(data, "footing_center", "displacement")
    ax.plot(1500.0 * t, -u[:, 2], "o-", ms=3, label="this code")
    ax.set_xlabel("pressure on the patch (kPa)")
    ax.set_ylabel("settlement of the footing centre (mm)")
    ax.set_title("07 cube footing")


def plot_08(data, ax):
    t, f, _ = reaction(data, "top")
    L, w = 20.0, 1.0
    shortening = 2.5 * t
    ax.plot(shortening, -f[:, 2], "o-", ms=3, label="this code (reaction)")
    E = 3.0 * G0                            # incompressible small-strain modulus
    p_cr = 4.0 * math.pi ** 2 * E * w ** 4 / 12.0 / L ** 2
    ax.axhline(p_cr, ls="--", color="C1", label=f"Euler load, clamped ends: {p_cr:.2f} mN")
    ax.set_xlabel("shortening (mm)")
    ax.set_ylabel("axial force (mN)")
    ax.set_title("08 column buckling")


def plot_09(data, ax):
    t, u = probe(data, "corner", "displacement")
    _, f, _ = reaction(data, "loaded")
    lam = (10.0 + u[:, 1]) / 10.0
    ax.plot(lam, f[:, 1] / 100.0 / 1e3, "o-", ms=3, label="cube with inclusion, this code")
    lam_ref = np.linspace(1.0, lam.max(), 200)
    ax.plot(lam_ref, uniaxial_nominal(lam_ref, psi1_series) / 1e3, "-", label="matrix alone, homogeneous (K = 1000 G)")
    ax.set_xlabel("stretch")
    ax.set_ylabel("nominal stress P22 (MPa)")
    ax.set_title("09 spherical inclusion")


def plot_10(data, ax):
    t, f, m = reaction(data, "top")
    theta = 2.0 * math.pi * t
    ax.plot(theta, m[:, 2], "o-", ms=3, label="torque, this code")
    ax2 = ax.twinx()
    ax2.plot(theta, f[:, 2], "s-", ms=3, color="C3", label="axial force, this code")
    ax.set_xlabel("twist angle (rad)")
    ax.set_ylabel("torque (mN mm)")
    ax2.set_ylabel("axial force (mN)")
    ax.set_title("10 column twist")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)
    return True


PLOTS = {"01_uniaxial_tension": plot_01, "02_simple_shear": plot_02, "03_cylinder_torsion": plot_03,
         "04_plate_with_hole": plot_04, "05_cylinder_inflation": plot_05, "06_sphere_inflation": plot_06,
         "07_cube_footing": plot_07, "08_column_buckling": plot_08, "09_spherical_inclusion": plot_09,
         "10_column_twist": plot_10}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help="case names (default: every case with a log)")
    ap.add_argument("--logs", default="out/anand_coupled_theories/finite_elasticity/logs")
    ap.add_argument("--out", default=None, help="plot directory (default: <logs>/../plots)")
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.logs.rstrip("/")), "plots")
    os.makedirs(out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = args.cases or [c for c in PLOTS if os.path.exists(os.path.join(args.logs, c + ".log"))]
    if not cases:
        raise SystemExit(f"no logs found in {args.logs}")
    status = 0
    for case in cases:
        path = os.path.join(args.logs, case + ".log")
        try:
            data = read_log(path)
            fig, ax = plt.subplots(figsize=(6.0, 4.2))
            own_legend = PLOTS[case](data, ax)
            if not own_legend:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            png = os.path.join(out, case + ".png")
            fig.savefig(png, dpi=150)
            plt.close(fig)
            print(f"{case}: {png}")
        except (KeyError, FileNotFoundError, ValueError) as e:
            status = 1
            print(f"{case}: cannot plot ({type(e).__name__}: {e}); the log needs the probes and "
                  f"reactions of the current input", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
