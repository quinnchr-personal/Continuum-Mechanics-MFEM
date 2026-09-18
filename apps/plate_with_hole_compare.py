#!/usr/bin/env python3
"""The exercise of myapps/plate_with_hole on the framework: run the plate input of
apps/input/plate_with_hole with the general executable and compare the computed
displacement and stress with Kirsch's closed-form solution.

    python3 apps/plate_with_hole_compare.py [--app build/apps/solid_mechanics] [--np N]
                                            [--out out/plate_with_hole] [--order P] [--refine N]
                                            [--model MODEL] [--no-run] [--no-plot] [--check]

The exercise is a driver of its own (MFEM's ElasticityIntegrator, the analytic
displacement prescribed on the outer edges by a C++ function, numerical and analytic
fields written side by side). Here the problem is an input of the general executable
(the Dirichlet data are expressions in x and y) and the comparison is this script.
It copies the input to <out>/inputs with the overrides asked for (--order,
--refine = mesh.serial_refine, --model, e.g. neo_hookean with E, nu from mu, nu to
see what geometric nonlinearity does at these strains), runs it (log in <out>/logs),
reads the ParaView output back with pyvista and evaluates the closed form at every
node (plane strain, k = 3 - 4 nu, far-field stress s along x):

    u_r     =  s/(8 mu r) [(k-1) r^2 + 2 R^2 + 2 (R^2 (k+1) + r^2 - R^4/r^2) cos 2theta]
    u_theta = -s/(4 mu r) [r^2 + R^2 (k-1) + R^4/r^2] sin 2theta
    s_rr = s/2 [(1 - R^2/r^2) + (1 - 4 R^2/r^2 + 3 R^4/r^4) cos 2theta]
    s_tt = s/2 [(1 + R^2/r^2) - (1 + 3 R^4/r^4) cos 2theta]
    s_rt = -s/2 (1 + 2 R^2/r^2 - 3 R^4/r^4) sin 2theta,        s_zz = nu (s_rr + s_tt)

It prints the errors over all nodes (relative root mean square and largest), the
stress concentration factor at the hole, and draws <out>/plate_with_hole.png: the
stress s_xx along the ligament x = 0 and the hoop stress around the hole, computed
against closed form. With --check it fails unless the root-mean-square errors are
below 1e-4 (displacement) and 5e-3 (stresses) and the concentration factor within
0.03 of 3 (measured at order 1: 5.5e-5, 2.6e-3, 2.991). Run from the repository root.
"""
import argparse
import glob
import math
import os
import subprocess
import sys
import time

import numpy as np
import yaml

INPUT = "apps/input/plate_with_hole/plate_linear.yaml"
R, L, MU, NU, S = 0.01, 0.03, 7.0e10, 0.3, 7.0e8

# Slots 1 and 2 of the reference palette for the two stress components; ink tokens for text.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def kirsch(x, y):
    """Displacement (n, 2) and stress xx, yy, zz, xy (n, 4) of the infinite plate."""
    r2 = x * x + y * y
    r = np.sqrt(r2)
    c, s = x / r, y / r
    c2, s2 = (x * x - y * y) / r2, 2.0 * x * y / r2
    k, q, q2 = 3.0 - 4.0 * NU, R * R / r2, (R * R / r2) ** 2
    ur = S / (8.0 * MU * r) * ((k - 1.0) * r2 + 2.0 * R * R + 2.0 * (R * R * (k + 1.0) + r2 - R ** 4 / r2) * c2)
    ut = -S / (4.0 * MU * r) * (r2 + R * R * (k - 1.0) + R ** 4 / r2) * s2
    srr = 0.5 * S * ((1.0 - q) + (1.0 - 4.0 * q + 3.0 * q2) * c2)
    stt = 0.5 * S * ((1.0 + q) - (1.0 + 3.0 * q2) * c2)
    srt = -0.5 * S * (1.0 + 2.0 * q - 3.0 * q2) * s2
    u = np.column_stack([ur * c - ut * s, ur * s + ut * c])
    sig = np.column_stack([srr * c * c - 2.0 * srt * s * c + stt * s * s,
                           srr * s * s + 2.0 * srt * s * c + stt * c * c,
                           NU * (srr + stt),
                           (srr - stt) * s * c + srt * (c * c - s * s)])
    return u, sig


def prepare_input(args):
    with open(INPUT) as f:
        cfg = yaml.safe_load(f)
    if args.order:
        cfg["mesh"]["order"] = args.order
    if args.refine:
        cfg["mesh"]["serial_refine"] = args.refine
    if args.model:
        # the same small-strain moduli for a finite-strain model (E, nu or mu, nu by its keys)
        if args.model in ("neo_hookean", "st_venant_kirchhoff"):
            cfg["material"] = {"model": args.model, "E": 2.0 * MU * (1.0 + NU), "nu": NU}
        else:
            cfg["material"] = {"model": args.model, "mu": MU, "nu": NU}
        cfg["solver"]["newton"] = {"rtol": 1e-9, "atol": 1e-3, "max_it": 25}
        cfg["solver"]["linear"]["type"] = "gmres_amg"
    cfg["output"]["paraview"] = os.path.join(args.out, "paraview")
    cfg["output"]["fields"] = ["displacement", "cauchy_stress"]
    cfg["output"]["nodal_projection"] = "projected"
    cfg["output"].pop("probes", None)
    path = os.path.join(args.out, "inputs", "plate.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, width=10000)
    return path


def run(path, args):
    log = os.path.join(args.out, "logs", "plate.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    cmd = [args.app, "-i", path]
    if args.np > 1:
        cmd = ["mpirun", "-np", str(args.np)] + cmd
    t0 = time.time()
    with open(log, "w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    return code, time.time() - t0, log_summary(log), log


def log_summary(log):
    if not os.path.exists(log):
        return []
    return [l.strip() for l in open(log) if l.startswith(("mesh:", "result:"))]


def read_fields(directory):
    """Nodes, displacement (n, 2) and Cauchy stress xx, yy, zz, xy (n, 4) of the last cycle."""
    import pyvista as pv
    reader = pv.get_reader(glob.glob(os.path.join(directory, "*.pvd"))[0])
    reader.set_active_time_value(reader.time_values[-1])
    mesh = reader.read().combine()
    X = np.asarray(mesh.points)[:, :2]
    u = np.asarray(mesh.point_data["displacement"])[:, :2]
    sig = np.asarray(mesh.point_data["cauchy_stress"])[:, [0, 1, 2, 3]]   # xx yy zz xy (yz, xz vanish)
    # nodes shared by elements of the high-order output appear several times: keep one each
    _, first = np.unique(np.round(X / (1e-9 * L)).astype(np.int64), axis=0, return_index=True)
    return X[first], u[first], sig[first]


def errors(X, u, sig):
    u_ex, sig_ex = kirsch(X[:, 0], X[:, 1])
    rows = [("displacement", np.linalg.norm(u - u_ex, axis=1), np.linalg.norm(u_ex, axis=1))]
    for j, name in enumerate(("sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy")):
        rows.append((name, np.abs(sig[:, j] - sig_ex[:, j]), np.full(len(X), S)))
    out = {}
    for name, err, scale in rows:
        out[name] = (math.sqrt(np.mean(err ** 2)) / math.sqrt(np.mean(scale ** 2)), float(np.max(err / np.max(scale))))
    return out


def lines(X, sig):
    """Computed and closed-form s_xx on the ligament x = 0, and hoop stress on the hole."""
    tol = 1e-7 * L
    lig = np.where(np.abs(X[:, 0]) < tol)[0]
    lig = lig[np.argsort(X[lig, 1])]
    r = np.hypot(X[:, 0], X[:, 1])
    hole = np.where(np.abs(r - R) < 1e-6 * R)[0]
    theta = np.arctan2(X[hole, 1], X[hole, 0])
    hole, theta = hole[np.argsort(theta)], np.sort(theta)
    c, s = np.cos(theta), np.sin(theta)
    hoop = sig[hole, 0] * s * s + sig[hole, 1] * c * c - 2.0 * sig[hole, 3] * s * c
    return (X[lig, 1] / R, sig[lig, 0] / S), (np.degrees(theta), hoop / S)


def plot(ligament, hole, path, subtitle):
    import warnings
    warnings.filterwarnings("ignore", message="Unable to import Axes3D")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    y = np.linspace(1.0, L / R, 400)
    th = np.linspace(0.0, 90.0, 400)
    panels = [
        (axes[0], y, 1.0 + 0.5 / y ** 2 + 1.5 / y ** 4, ligament, BLUE, "distance from the centre  $y / R$  (on $x = 0$)",
         r"$\sigma_{xx} / \sigma_\infty$", "Stress across the ligament", (1.12, 2.93, "3 at the hole")),
        (axes[1], th, 1.0 - 2.0 * np.cos(2.0 * np.radians(th)), hole, ORANGE, r"angle from the load axis  $\theta$  (degrees)",
         r"$\sigma_{\theta\theta} / \sigma_\infty$", "Hoop stress around the hole", (90.0, -0.72, "3 at the top of the hole, -1 at its side")),
    ]
    for ax, xs, exact, (xn, fn), color, xlabel, ylabel, title, (px, py, note) in panels:
        ax.set_facecolor(SURFACE)
        ax.plot(xs, exact, color=INK_2, linewidth=1.4, zorder=2)
        every = max(1, len(xn) // 16)   # sparse markers, so the closed form stays visible between them
        ax.plot(xn[::every], fn[::every], linestyle="none", marker="o", markersize=5.5, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
        ax.text(px, py, note, ha="right" if ax is axes[1] else "left", va="center", fontsize=8, color=INK)
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        ax.legend(handles=[Line2D([], [], linestyle="none", marker="o", markersize=5.5, markerfacecolor=color,
                                  markeredgecolor=SURFACE, markeredgewidth=1.4, label="finite elements (nodes)"),
                           Line2D([], [], color=INK_2, linewidth=1.4, label="Kirsch, closed form")],
                  loc="upper right" if ax is axes[0] else "upper left", fontsize=8, frameon=False, labelcolor=INK_2)
    fig.suptitle("Plate with a hole: computed stress against Kirsch's solution", x=0.06, y=0.985, ha="left",
                 fontsize=11, color=INK)
    fig.text(0.06, 0.905, subtitle, fontsize=8, color=INK_2)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.13, wspace=0.22)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--app", default="build/apps/solid_mechanics")
    parser.add_argument("--np", type=int, default=1, help="MPI ranks")
    parser.add_argument("--out", default="out/plate_with_hole")
    parser.add_argument("--order", type=int, default=0, help="override mesh.order")
    parser.add_argument("--refine", type=int, default=0, help="mesh.serial_refine")
    parser.add_argument("--model", default="", help="another material model with the same mu, nu")
    parser.add_argument("--no-run", action="store_true", help="only re-read <out>/paraview")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--check", action="store_true", help="fail unless the errors are below the gates")
    args = parser.parse_args()

    failed = False
    info = log_summary(os.path.join(args.out, "logs", "plate.log"))
    if not args.no_run:
        path = prepare_input(args)
        print("running %s ..." % path, flush=True)
        code, seconds, info, log = run(path, args)
        for line in info:
            print("  " + line)
        print("  [%.1f s]" % seconds)
        if code != 0:
            print("  FAILED (exit %d); see %s" % (code, log))
            return 1
    X, u, sig = read_fields(os.path.join(args.out, "paraview"))
    err = errors(X, u, sig)
    print("\nErrors against Kirsch's closed form over %d nodes (stresses relative to the far-field stress):" % len(X))
    print("  %-14s %18s %14s" % ("", "root mean square", "largest"))
    for name, (rms, worst) in err.items():
        print("  %-14s %18.3e %14.3e" % (name, rms, worst))
    ligament, hole = lines(X, sig)
    print("  stress concentration: sigma_xx / s = %.4f at the top of the hole (3), hoop stress / s = %.4f at "
          "its side (-1)" % (ligament[1][0], hole[1][0]))

    if args.check:
        gates = {"displacement": 1e-4, "sigma_xx": 5e-3, "sigma_yy": 5e-3, "sigma_zz": 5e-3, "sigma_xy": 5e-3}
        for name, gate in gates.items():
            if not err[name][0] <= gate:
                print("--check: %s error %.3e > %.0e" % (name, err[name][0], gate))
                failed = True
        if abs(ligament[1][0] - 3.0) > 0.03:
            print("--check: stress concentration %.4f not within 0.03 of 3" % ligament[1][0])
            failed = True

    if not args.no_plot:
        mesh_line = next((l for l in info if l.startswith("mesh:")), "")
        dofs = mesh_line.split("true dofs ")[1].split(",")[0] if "true dofs " in mesh_line else "?"
        order = mesh_line.split("order ")[1].split(",")[0] if "order " in mesh_line else "?"
        subtitle = ("quarter plate 3R x 3R, R = 0.01, plane strain, mu = 7e10, nu = 0.3; Kirsch's displacement on the outer "
                    "edges, hole free; order %s, %s dofs" % (order, dofs))
        png = os.path.join(args.out, "plate_with_hole.png")
        plot(ligament, hole, png, subtitle)
        print("\nwrote %s" % png)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
