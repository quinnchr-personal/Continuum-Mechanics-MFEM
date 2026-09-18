#!/usr/bin/env python3
"""Drive every uniaxial input of
apps/input/finite_elasticity/verification/homogeneous_deformations to a large
stretch in many load steps and plot the computed first Piola-Kirchhoff stress
P_11 against the stretch lambda_1, with the analytical curve of the
homogeneous state F = diag(lambda_1, lambda_2, lambda_2).

    python3 apps/uniaxial_plots.py [--app build/apps/solid_mechanics] [--stretch 8]
                                   [--steps 70] [--np N] [--out DIR]
                                   [--set mu=0.4 --set kappa=2000] [case ...]

For each case the input is copied to <out>/inputs with the loading replaced by
a ramp to the target stretch (lambda_1 = 1 + (stretch - 1) t), the requested
number of equal load steps (bisection on failure) and pk1_stress among the
fields. Nothing is probed: the ParaView series the app writes after every step
(<out>/paraview/<case>) is read back with pyvista. lambda_1 and lambda_2 are the
mean u_x on X = 1 and u_y on Y = 1, P_11 the mean of the nodal pk1_stress, and
the table reports the largest deviation of any node from that mean (the state
must be homogeneous). The log goes to <out>/logs and the plot to <out>/<case>.png
(default out/uniaxial_stretch). Run from the repository root.

Analytical solutions (lambda_2 from P_22 = 0):
  incompressible neo-Hookean   lambda_2 = lambda_1^-1/2, P_11 = mu (lambda_1 - lambda_1^-2)
  St. Venant-Kirchhoff         lambda_2^2 = 1 - nu (lambda_1^2 - 1), P_11 = E lambda_1 (lambda_1^2 - 1) / 2;
                               the lateral stretch vanishes at lambda_1 = sqrt(1 + 1/nu), so this
                               case is driven to 0.95 of that limit instead of the target stretch
  neo-Hookean (Bonet-Wood)     P_aa = mu (lambda_a - 1/lambda_a) + lambda ln(J) / lambda_a
  gent_compressible_summit     P_aa = mu Jm / (Jm - I1 + 3) lambda_a + (2 kappa A^3 (J^2 - 1) - mu) / lambda_a,
                               A = (J^2 - 1)/2 - ln J (SUMMIT's compressible Gent; driven to at most
                               0.95 sqrt(Jm + 3), short of the locking stretch)
  decoupled neo-Hookean        P_aa = mu J^-2/3 (lambda_a - I1 / (3 lambda_a)) + U'(J) J / lambda_a,
                               U' = kappa (J - 1) (quadratic), kappa (J - 1/J) / 2 (simo_taylor),
                               kappa ln(J) (j_log_j) or kappa ln(J) / J (logarithmic)
with the scalar equation P_22(lambda_2) = 0 solved by bisection. In the full
affine incompressible case the end faces carry all three components, so the
lateral data follows lambda_1^-1/2 - 1 in t (a linear ramp would not be
isochoric between t = 0 and 1).
"""
import argparse
import copy
import glob
import math
import os
import subprocess
import sys

import numpy as np
import yaml

INPUT_DIR = "apps/input/finite_elasticity/verification/homogeneous_deformations"


def lame(material):
    """mu, lambda, kappa of a material block (E, nu or mu, kappa)."""
    if "E" in material:
        E, nu = material["E"], material["nu"]
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return mu, lam, lam + 2.0 * mu / 3.0
    mu, kappa = material["mu"], material.get("kappa", math.inf)
    return mu, kappa - 2.0 * mu / 3.0, kappa


def pk1(material, l1, l2):
    """(P_11, P_22) of F = diag(l1, l2, l2) for the compressible models."""
    model = material["model"]
    J = l1 * l2 * l2
    if model == "gent_compressible_summit":
        mu, kappa, Jm = material["mu"], material["kappa"], material["Jm"]
        I1 = l1 * l1 + 2.0 * l2 * l2
        A = 0.5 * (J * J - 1.0) - math.log(J)
        if I1 - 3.0 >= Jm:  # beyond the locking stretch
            return math.nan, math.nan
        return tuple(mu * Jm / (Jm - I1 + 3.0) * l + (2.0 * kappa * A ** 3 * (J * J - 1.0) - mu) / l for l in (l1, l2))
    mu, lam, kappa = lame(material)
    if model == "st_venant_kirchhoff":
        E1, E2 = 0.5 * (l1 * l1 - 1.0), 0.5 * (l2 * l2 - 1.0)
        tr = E1 + 2.0 * E2
        return l1 * (lam * tr + 2.0 * mu * E1), l2 * (lam * tr + 2.0 * mu * E2)
    if model == "neo_hookean":
        return tuple(mu * (l - 1.0 / l) + lam * math.log(J) / l for l in (l1, l2))
    if model == "iso_neo_hookean":
        I1 = l1 * l1 + 2.0 * l2 * l2
        law = material.get("volumetric", "quadratic")
        if law == "quadratic":
            dU = kappa * (J - 1.0)
        elif law == "logarithmic":
            dU = kappa * math.log(J) / J
        elif law == "simo_taylor":
            dU = 0.5 * kappa * (J - 1.0 / J)
        elif law == "j_log_j":
            dU = kappa * math.log(J)
        else:
            raise SystemExit(f"volumetric law '{law}' has no analytical curve here")
        return tuple(mu * J ** (-2.0 / 3.0) * (l - I1 / (3.0 * l)) + dU * J / l for l in (l1, l2))
    raise SystemExit(f"model '{model}' has no analytical curve here")


def analytic(material, l1):
    """(lambda_2, P_11) of uniaxial tension at the stretch l1."""
    if material.get("incompressible") or material.get("nu") == 0.5:
        if material["model"] != "iso_neo_hookean":
            raise SystemExit(f"model '{material['model']}' has no analytical curve here")
        return l1 ** -0.5, material["mu"] * (l1 - l1 ** -2.0)
    lo, hi = 1e-6, 2.0  # P_22 increases with lambda_2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if pk1(material, l1, mid)[1] > 0.0:
            hi = mid
        else:
            lo = mid
    l2 = 0.5 * (lo + hi)
    return l2, pk1(material, l1, l2)[0]


def stretch_limit(material, stretch):
    """The target stretch, reduced for St. Venant-Kirchhoff (lateral collapse)."""
    if material["model"] == "st_venant_kirchhoff":
        return min(stretch, 0.95 * math.sqrt(1.0 + 1.0 / material["nu"]))
    if material["model"] == "gent_compressible_summit":  # locking: I1 - 3 < Jm, I1 > lambda_1^2
        return min(stretch, 0.95 * math.sqrt(material["Jm"] + 3.0))
    return stretch


def make_input(path, out, stretch, steps, overrides):
    """Write the large-stretch copy of an input; returns (path, cfg, stretch)."""
    cfg = copy.deepcopy(yaml.safe_load(open(path)))
    for key, value in overrides.items():  # only keys the material already has
        if key in cfg["material"]:
            cfg["material"][key] = value
    name = os.path.splitext(os.path.basename(path))[0]
    stretch = stretch_limit(cfg["material"], stretch)
    d = f"{stretch - 1.0!r}"
    loaded = [bc for bc in cfg["bcs"]["dirichlet"] if any(e.strip() != "0" for e in bc["expression"])]
    if len(loaded) != 1:
        raise SystemExit(f"{path}: expected exactly one loaded Dirichlet entry")
    lateral = f"((1+{d}*t)^(-0.5)-1)"
    loaded[0]["expression"] = [f"{d}*t*x", f"{lateral}*y", f"{lateral}*z"]
    loaded[0].pop("schedule", None)  # the data mentions t: constant schedule
    cfg["solver"].pop("steps", None)
    cfg["solver"]["load_steps"] = steps
    cfg["solver"]["substep"] = {"on_failure": True, "max_bisections": 8, "min_dt": 1e-6}
    cfg["solver"]["newton"]["print_level"] = 0
    output = cfg["output"]
    output["paraview"] = os.path.join(out, "paraview", name)
    output["fields"] = sorted(set(output["fields"]) | {"displacement", "pk1_stress", "jacobian"})
    output["quadrature_at"] = ["nodes"]
    output.pop("probes", None)
    os.makedirs(os.path.join(out, "inputs"), exist_ok=True)
    new = os.path.join(out, "inputs", name + ".yaml")
    with open(new, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=None, sort_keys=False)
    return new, cfg, stretch


def read_series(directory):
    """lambda_1, lambda_2, mean P_11 and the largest nodal deviation of P_11
    from its mean, per saved step of a ParaView series (step 0 dropped)."""
    import pyvista as pv

    reader = pv.get_reader(glob.glob(os.path.join(directory, "*.pvd"))[0])
    rows = []
    for t in reader.time_values[1:]:
        reader.set_active_time_value(t)
        mesh = reader.read().combine()
        X, u = mesh.points, mesh.point_data["displacement"]
        P11 = mesh.point_data["pk1_stress"][:, 0]
        right, back = np.isclose(X[:, 0], X[:, 0].max()), np.isclose(X[:, 1], X[:, 1].max())
        l1 = 1.0 + u[right, 0].mean() / X[:, 0].max()
        l2 = 1.0 + u[back, 1].mean() / X[:, 1].max()
        rows.append((l1, l2, P11.mean(), np.abs(P11 - P11.mean()).max()))
    return np.array(rows).T


BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a63"


def plot(name, material, stretch, l1, P_num, l2_num, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fine = np.linspace(1.0, stretch, 400)
    P_fine = np.array([analytic(material, l)[1] for l in fine])
    exact = np.array([analytic(material, l) for l in l1])
    err = np.abs(P_num - exact[:, 1]) / np.maximum(np.abs(exact[:, 1]), 1e-300)
    err2 = np.abs(l2_num - exact[:, 0])

    fig, (ax, ex) = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True, height_ratios=[3, 1.3],
                                 constrained_layout=True)
    ax.plot(fine, P_fine, color=BLUE, lw=2, label="analytical")
    ax.plot(l1, P_num, ls="none", marker="o", ms=5, mfc=ORANGE, mec="white", mew=0.8,
            label=f"numerical ({len(l1)} load steps)")
    ax.set_ylabel(r"first Piola-Kirchhoff stress $P_{11}$")
    ax.legend(frameon=False, loc="upper left")
    params = ", ".join(f"{k} = {v}" for k, v in material.items() if k != "model")
    ax.set_title(f"{name}\n{material['model']} ({params})", fontsize=10, color=INK, loc="left")

    ex.semilogy(l1, np.maximum(err, 1e-17), color=ORANGE, lw=1.5, marker="o", ms=3, label=r"$P_{11}$ (relative)")
    ex.semilogy(l1, np.maximum(err2, 1e-17), color=BLUE, lw=1.5, marker="s", ms=3, label=r"$\lambda_2$ (absolute)")
    ex.set_xlabel(r"stretch $\lambda_1$")
    ex.set_ylabel("error")
    ex.legend(frameon=False, fontsize=8, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    for a in (ax, ex):
        a.grid(True, color="#e4e3dc", lw=0.6)
        a.set_axisbelow(True)
        a.tick_params(colors=MUTED)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(MUTED)
    path = os.path.join(out, name + ".png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, err.max(), err2.max()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help="input files (default: every *uniaxial*.yaml)")
    ap.add_argument("--app", default="build/apps/solid_mechanics")
    ap.add_argument("--stretch", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=70)
    ap.add_argument("--np", type=int, default=1, dest="np_")
    ap.add_argument("--out", default="out/uniaxial_stretch")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                    help="material parameters to replace where the input has them (repeatable: --set mu=0.4 --set kappa=2000)")
    args = ap.parse_args()
    cases = args.cases or sorted(glob.glob(os.path.join(INPUT_DIR, "*uniaxial*.yaml")))
    if not cases:
        raise SystemExit("no inputs found (run from the repository root)")
    if not os.path.exists(args.app):
        raise SystemExit(f"app '{args.app}' not found (make apps)")
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)

    overrides = {k: float(v) for k, v in (item.split("=", 1) for item in args.overrides)}
    failures = 0
    print(f"{'case':<52} {'stretch':>8} {'steps':>6} {'max rel err P11':>16} {'max err lambda2':>16} {'P11 nodal spread':>16}")
    for case in cases:
        new, cfg, stretch = make_input(case, args.out, args.stretch, args.steps, overrides)
        name = os.path.splitext(os.path.basename(new))[0]
        cmd = [args.app, "-i", new]
        if args.np_ > 1:
            cmd = ["mpirun", "-np", str(args.np_)] + cmd
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        open(os.path.join(args.out, "logs", name + ".log"), "w").write(proc.stdout)
        try:
            l1, l2, P11, spread = read_series(cfg["output"]["paraview"])
        except (IndexError, KeyError):
            l1 = []
        if len(l1) == 0:
            failures += 1
            print(f"{name:<52} RUN FAILED (exit {proc.returncode}), see {args.out}/logs/{name}.log")
            continue
        path, err, err2 = plot(name, cfg["material"], stretch, l1, P11, l2, args.out)
        inhom = (spread / np.abs(P11)).max()
        note = "" if proc.returncode == 0 and abs(l1[-1] - stretch) < 1e-9 else f"  <-- stopped at lambda_1 = {l1[-1]:.4g}"
        failures += bool(note)
        print(f"{name:<52} {l1[-1]:>8.4f} {len(l1):>6d} {err:>16.2e} {err2:>16.2e} {inhom:>16.2e}{note}\n    {path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
