#!/usr/bin/env python3
"""The exercise of myapps/elastic_bar on the framework: run the bar inputs of
apps/input/elastic_bar with the general executable and compare the reaction force
against the end displacement, linear against nonlinear.

    python3 apps/elastic_bar_compare.py [--app build/apps/solid_mechanics] [--np N]
                                        [--out out/elastic_bar] [--cases linear gent neo_hookean]
                                        [--steps N] [--order P] [--paraview] [--no-run] [--no-plot]
                                        [--reference CSV] [--linear-reference CSV] [--check]

In the exercise the linear and the nonlinear bar are two drivers; here they are
inputs of one executable that differ in the material line (linear_elastic,
gent_compressible_summit, iso_neo_hookean). No driver of its own is needed:
output.reactions prints the force of the Dirichlet entry `front` after every load
step and the probe `pulled_face` the displacement of that face. For each case this
script copies the input to <out>/inputs with the overrides asked for (--steps,
--order; ParaView output only with --paraview), runs it (log in <out>/logs), and
writes <out>/<case>/force_displacement.csv in the exercise's own format,
`step,disp_z,total_Fz`, so the exercise's plot_force_displacement.py reads it too.

It then prints a table and draws <out>/force_displacement.png:
  - the curves of the cases;
  - the homogeneous uniaxial-stress estimate of each model, F = A P_11(lambda_1) with
    the lateral stretch from P_22 = 0 (no clamped end, no discretisation): the finite
    element force lies above it, by the constraint of the clamped end and by the
    volumetric locking of linear tetrahedra at nu = 0.4983 (try --order 2);
  - the results of the exercise's own drivers (--reference for main_nonlinear with the
    Gent model, --linear-reference for main; by default the copies kept in
    apps/input/elastic_bar/reference, see the README there). Same mesh, order and
    boundary conditions, so the codes agree to solver tolerance: 1e-12 and 3e-11. The
    comparison is meaningful at the inputs' order only, so it is skipped with --order.
    With --check the script fails unless every compared case agrees to 1e-8 (make test).

Run from the repository root.
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time

import yaml

INPUT_DIR = "apps/input/elastic_bar"
ENTRY = "front"          # the pulled Dirichlet entry of the inputs
PROBE = "pulled_face"    # the probe on that face
AREA, HEIGHT = 0.3 * 0.3, 1.0

# Categorical slots 1-3 of the reference palette (fixed order, validated for
# colour-vision deficiency in all pairings); text and grid wear ink tokens.
CASES = {
    "linear": {"label": "linear elastic", "color": "#2a78d6"},
    "gent": {"label": "compressible Gent", "color": "#eb6834"},
    "neo_hookean": {"label": "neo-Hookean", "color": "#1baf7a"},
}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


# ------------------------------------------------------------------ running

def prepare_input(case, args):
    """The case's input with the overrides, written to <out>/inputs."""
    with open(os.path.join(INPUT_DIR, "bar_%s.yaml" % case)) as f:
        cfg = yaml.safe_load(f)
    if args.steps:
        cfg["solver"]["load_steps"] = args.steps
    if args.order:
        cfg["mesh"]["order"] = args.order
    if args.paraview:
        cfg["output"]["paraview"] = os.path.join(args.out, case, "paraview")
    else:
        cfg["output"].pop("paraview", None)
    cfg["output"]["fields"] = ["displacement"]          # one probe line per step
    path = os.path.join(args.out, "inputs", "bar_%s.yaml" % case)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path, cfg


def run(case, path, args):
    log = os.path.join(args.out, "logs", "bar_%s.log" % case)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    cmd = [args.app, "-i", path]
    if args.np > 1:
        cmd = ["mpirun", "-np", str(args.np)] + cmd
    t0 = time.time()
    with open(log, "w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    return log, code, time.time() - t0


NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
STEP = r"^step (\d+) t = (%s) " % NUMBER


def parse_log(log):
    """(step, disp_z, F_z) after every load step, from the reaction and probe lines."""
    force = re.compile(STEP + r"reaction %s: force = (%s) (%s) (%s)" % (ENTRY, NUMBER, NUMBER, NUMBER))
    disp = re.compile(STEP + r"probe %s at \(.*\): displacement = (%s) (%s) (%s)" % (PROBE, NUMBER, NUMBER, NUMBER))
    fz, uz = {}, {}
    mesh = result = ""
    with open(log) as f:
        for line in f:
            m = force.match(line)
            if m:
                fz[int(m.group(1))] = float(m.group(5))
            m = disp.match(line)
            if m:
                uz[int(m.group(1))] = float(m.group(5))
            if line.startswith("mesh:"):
                mesh = line.strip()
            if line.startswith("result:"):
                result = line.strip()
    steps = sorted(set(fz) & set(uz))
    return [(0, 0.0, 0.0)] + [(k, uz[k], fz[k]) for k in steps], mesh, result


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "disp_z", "total_Fz"])
        for step, u, force in rows:
            w.writerow([step, repr(u), repr(force)])


def read_csv(path):
    with open(path, newline="") as f:
        return [(int(r["step"]), float(r["disp_z"]), float(r["total_Fz"])) for r in csv.DictReader(f) if r]


# ------------------------------------------- homogeneous uniaxial estimates

def uniaxial_force(material, u):
    """F = A P_11 of the homogeneous state F = diag(l1, l2, l2), l1 = 1 + u/H, P_22 = 0."""
    model, mu, kappa = material["model"], material["mu"], material["kappa"]
    if model == "linear_elastic":
        return AREA * 9.0 * kappa * mu / (3.0 * kappa + mu) * u / HEIGHT
    l1 = 1.0 + u / HEIGHT

    def stress(l2):
        J, I1 = l1 * l2 * l2, l1 * l1 + 2.0 * l2 * l2
        if model == "gent_compressible_summit":
            Jm = material["Jm"]
            a = 0.5 * (J * J - 1.0) - math.log(J)
            shear, vol = mu * Jm / (Jm - I1 + 3.0), 2.0 * kappa * a ** 3 * (J * J - 1.0) - mu
            return shear * l1 + vol / l1, shear * l2 + vol / l2
        if model == "iso_neo_hookean":                  # quadratic volumetric law
            s, p = mu * J ** (-2.0 / 3.0), kappa * (J - 1.0) * J
            return s * (l1 - I1 / (3.0 * l1)) + p / l1, s * (l2 - I1 / (3.0 * l2)) + p / l2
        raise SystemExit("no uniaxial estimate for model '%s'" % model)

    lo, hi = 0.05, 1.5                                   # P_22 increases with l2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if stress(mid)[1] < 0.0:
            lo = mid
        else:
            hi = mid
    return AREA * stress(0.5 * (lo + hi))[0]


def interpolate(rows, u):
    """Force at the displacement u, linear between the recorded steps."""
    for (_, u0, f0), (_, u1, f1) in zip(rows, rows[1:]):
        if u0 <= u <= u1 and u1 > u0:
            return f0 + (f1 - f0) * (u - u0) / (u1 - u0)
    return float("nan")


# ------------------------------------------------------------------- output

def report(results, references):
    points = [0.5, 1.0, 2.0, 3.0]
    print("\nReaction force F_z (MN) against the end displacement u_z; in brackets the ratio to the "
          "homogeneous\nuniaxial-stress estimate of the same model (no clamped end, no discretisation):")
    print("  %-18s %12s  " % ("case", "F/u, 1st step") + "  ".join("%16s" % ("u_z = %.1f" % u) for u in points))
    for case, r in results.items():
        rows, mat = r["rows"], r["material"]
        cells = []
        for u in points:
            f = interpolate(rows, u)
            cells.append("%16s" % ("-" if math.isnan(f) else "%7.4f (%5.3f)" % (f / 1e6, f / uniaxial_force(mat, u))))
        k0 = rows[1][2] / rows[1][1] if len(rows) > 1 and rows[1][1] > 0 else float("nan")
        print("  %-18s %9.4f e6  " % (CASES[case]["label"], k0 / 1e6) + "  ".join(cells))
    if "linear" in results:
        lin = results["linear"]["rows"]
        for case, r in results.items():
            if case == "linear":
                continue
            ratios = ", ".join("%.3f at %.1f" % (interpolate(r["rows"], u) / interpolate(lin, u), u)
                               for u in points if not math.isnan(interpolate(r["rows"], u)))
            print("  %s / linear elastic: %s" % (CASES[case]["label"], ratios))
    agreement = {}
    for case, (path, ref) in references.items():
        if case not in results:
            continue
        rows = results[case]["rows"]
        worst, at = 0.0, 0.0
        for _, u, f in rows[1:]:
            f_ref = interpolate(ref, u)
            if not math.isnan(f_ref) and abs(f_ref) > 0.0 and abs(f - f_ref) / abs(f_ref) > worst:
                worst, at = abs(f - f_ref) / abs(f_ref), u
        print("  %s against the exercise's %s (%d steps): largest relative difference %.2e (at u_z = %.3f)"
              % (CASES[case]["label"], path, len(ref) - 1, worst, at))
        agreement[case] = worst
    return agreement


def plot(results, references, path, subtitle):
    import warnings
    warnings.filterwarnings("ignore", message="Unable to import Axes3D")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    handles = []
    for case, r in results.items():
        rows, color = r["rows"], CASES[case]["color"]
        u = [row[1] for row in rows]
        # context first: the homogeneous estimate of the same model, a hairline in muted ink
        ax.plot(u, [uniaxial_force(r["material"], x) / 1e6 for x in u], color=MUTED, linewidth=1.0, zorder=2)
        ax.plot(u, [row[2] / 1e6 for row in rows], color=color, linewidth=2.0, solid_capstyle="round",
                solid_joinstyle="round", zorder=3)
        handles.append(Line2D([], [], color=color, linewidth=2.0, label=CASES[case]["label"]))
        # direct label at the end of the line, in ink (never in the series colour)
        ax.annotate("%s  %.2f MN" % (CASES[case]["label"], rows[-1][2] / 1e6), (u[-1], rows[-1][2] / 1e6),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    for case, (_, ref) in references.items():
        if case not in results:
            continue
        every = max(1, (len(ref) - 1) // 12)
        pts = ref[every::every]
        ax.plot([p[1] for p in pts], [p[2] / 1e6 for p in pts], linestyle="none", marker="o", markersize=6.5,
                markerfacecolor=CASES[case]["color"], markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=6.5,
                              markerfacecolor=CASES[case]["color"], markeredgecolor=SURFACE, markeredgewidth=1.6,
                              label="myapps/elastic_bar result (%s)" % CASES[case]["label"]))
    handles.append(Line2D([], [], color=MUTED, linewidth=1.0, label="homogeneous uniaxial stress (analytical)"))

    ax.set_xlabel("end displacement $u_z$  (bar length 1)", color=INK_2, fontsize=9)
    ax.set_ylabel("reaction force $F_z$  (MN)", color=INK_2, fontsize=9)
    fig.suptitle("Elastic bar: reaction force against end displacement", x=0.085, y=0.965, ha="left",
                 fontsize=11, color=INK)
    ax.set_title(subtitle, loc="left", fontsize=8, color=INK_2, pad=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    legend = ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False, labelcolor=INK_2)
    legend.set_zorder(5)
    fig.subplots_adjust(left=0.085, right=0.745, top=0.86, bottom=0.12)   # room for the end labels
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--app", default="build/apps/solid_mechanics")
    parser.add_argument("--np", type=int, default=1, help="MPI ranks")
    parser.add_argument("--out", default="out/elastic_bar")
    parser.add_argument("--cases", nargs="+", default=["linear", "gent"], choices=list(CASES))
    parser.add_argument("--steps", type=int, default=0, help="override solver.load_steps")
    parser.add_argument("--order", type=int, default=0, help="override mesh.order")
    parser.add_argument("--paraview", action="store_true", help="write ParaView output of every step")
    parser.add_argument("--no-run", action="store_true", help="only re-read <out>/<case>/force_displacement.csv")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="fail unless the compared cases agree with the exercise's results to 1e-8")
    parser.add_argument("--reference", default=os.path.join(INPUT_DIR, "reference/myapps_nonlinear_gent.csv"),
                        help="the exercise's nonlinear result, compared with the case gent")
    parser.add_argument("--linear-reference", default=os.path.join(INPUT_DIR, "reference/myapps_linear.csv"),
                        help="the exercise's linear result, compared with the case linear")
    args = parser.parse_args()

    results = {}
    failed = False
    for case in args.cases:
        path, cfg = prepare_input(case, args)
        csv_path = os.path.join(args.out, case, "force_displacement.csv")
        mesh = result = ""
        if not args.no_run:
            print("running %s ..." % case, flush=True)
            log, code, seconds = run(case, path, args)
            rows, mesh, result = parse_log(log)
            if code != 0 or len(rows) < 2:
                print("  FAILED (exit %d, %d steps recorded); see %s" % (code, len(rows) - 1, log))
                failed = True
            else:
                print("  %s\n  %s  [%.1f s]" % (mesh, result, seconds))
            write_csv(csv_path, rows)
            print("  wrote %s" % csv_path)
        elif not os.path.exists(csv_path):
            raise SystemExit("--no-run: %s does not exist" % csv_path)
        # PyYAML (YAML 1.1) reads 1.5e9 as a string, unlike the executable's yaml-cpp.
        material = {k: float(v) if isinstance(v, str) and re.fullmatch(NUMBER, v) else v
                    for k, v in cfg["material"].items()}
        results[case] = {"rows": read_csv(csv_path), "material": material, "mesh": mesh,
                         "order": cfg["mesh"]["order"]}

    references = {}
    for case, ref in (("gent", args.reference), ("linear", args.linear_reference)):
        if ref and os.path.exists(ref) and not args.order:
            references[case] = (ref, read_csv(ref))
    agreement = report(results, references)
    if args.check:
        if not agreement:
            print("--check: nothing was compared with the exercise's results")
            failed = True
        for case, worst in agreement.items():
            if not worst <= 1e-8:
                print("--check: %s differs from the exercise's result by %.2e > 1e-8" % (CASES[case]["label"], worst))
                failed = True

    if not args.no_plot:
        first = next(iter(results.values()))
        subtitle = "0.3 x 0.3 x 1 bar, clamped at z = 0, top face pulled with free lateral motion; mu = 5e6, kappa = 1.5e9; order %d" % first["order"]
        png = os.path.join(args.out, "force_displacement.png")
        plot(results, references, png, subtitle)
        print("\nwrote %s" % png)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
