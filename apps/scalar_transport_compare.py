#!/usr/bin/env python3
"""The convection-diffusion verification drivers of myapps/convection_diffusion on the
framework: run the inputs of apps/input/scalar_transport with the scalar transport
executable and lay their error histories over the drivers' own.

    python3 apps/scalar_transport_compare.py [--app build/apps/scalar_transport] [--np N]
                                             [--out out/scalar_transport] [--cases ...]
                                             [--paraview] [--no-run] [--no-plot] [--check]

The five drivers (transient convection-diffusion at Pe = 1, 10, 100; steady
convection-diffusion-reaction on the square and on the disk; nonlinear diffusion of the
Kirchhoff case; transient diffusion with a manufactured solution) are inputs here that
differ in the transport block and the data. For each case this script copies the input to
<out>/inputs with ParaView output under <out>/<case> (the collection carries
error_history.csv and flows.csv; the per-step files only with --paraview, output.every
set high otherwise) and the driver's quadrature rule for the source (the copies of the
inputs set transport.quadrature_order to 2p, the rule of MFEM's DomainLFIntegrator the
drivers use, so that the discrete problems coincide; the inputs themselves use the
framework's rule 2p + 3), runs it (log in <out>/logs), and reads the errors.

It then prints a table and draws <out>/error_histories.png:
  - the L2 error against time of the three Peclet numbers and of the transient
    manufactured solution, the framework's curves over the drivers' rows
    (apps/input/scalar_transport/reference/myapps_*.csv, see the README there);
  - the steady errors of the square and the disk as bars beside the drivers';
  - the Newton iterations per step of the Kirchhoff case beside the driver's (its error
    history against the series solution is compared in tests/test_scalar_verification.cpp,
    since the series is not an expression of the input).
On the same triangulation, order, steps and quadrature the discrete problems are those of
the drivers, so the error histories agree to the drivers' linear-solver tolerance (GMRES
to 1e-10): a few 1e-6 of the error. With --check the script fails unless every compared
error agrees to 1e-5 relative (make test).

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

INPUT_DIR = "apps/input/scalar_transport"
REF_DIR = os.path.join(INPUT_DIR, "reference")
CHECK_TOL = 1e-5

# Categorical slots of the reference palette (fixed order, validated for colour-vision
# deficiency in all pairings); text and grid wear ink tokens.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6", "#d63384"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# case: input, reference CSV, the reference columns of (abs, rel) L2, the driver's source rule.
CASES = {
    "peclet_1": {"input": "convection_diffusion_peclet_1.yaml", "ref": "myapps_convection_diffusion_peclet.csv",
                 "cols": ("abs_l2_pe1", "rel_l2_pe1"), "quadrature": 6, "label": "Pe = 1"},
    "peclet_10": {"input": "convection_diffusion_peclet_10.yaml", "ref": "myapps_convection_diffusion_peclet.csv",
                  "cols": ("abs_l2_pe2", "rel_l2_pe2"), "quadrature": 6, "label": "Pe = 10"},
    "peclet_100": {"input": "convection_diffusion_peclet_100.yaml", "ref": "myapps_convection_diffusion_peclet.csv",
                   "cols": ("abs_l2_pe3", "rel_l2_pe3"), "quadrature": 6, "label": "Pe = 100"},
    "steady_square": {"input": "steady_cdr_square_mms.yaml", "ref": "myapps_steady_cdr_square.csv",
                      "cols": ("abs_l2", "rel_l2"), "quadrature": 6, "label": "steady, square"},
    "steady_disk": {"input": "steady_cdr_disk_mms.yaml", "ref": "myapps_steady_cdr_disk.csv",
                    "cols": ("abs_l2", "rel_l2"), "quadrature": 6, "label": "steady, disk"},
    "kirchhoff": {"input": "nonlinear_diffusion_kirchhoff.yaml", "ref": "myapps_nonlinear_diffusion_kirchhoff.csv",
                  "cols": None, "quadrature": 8, "label": "nonlinear diffusion"},
    "transient_mms": {"input": "transient_diffusion_mms.yaml", "ref": "myapps_transient_diffusion_mms.csv",
                      "cols": ("l2_error", None), "quadrature": 2, "label": "transient MMS"},
}


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def run(case, args):
    spec = CASES[case]
    with open(os.path.join(INPUT_DIR, spec["input"])) as f:
        cfg = yaml.safe_load(f)
    out_dir = os.path.join(args.out, case)
    cfg["output"]["paraview"] = out_dir
    if not args.paraview:
        cfg["output"]["every"] = 1000000
    cfg["transport"]["quadrature_order"] = spec["quadrature"]
    os.makedirs(os.path.join(args.out, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    path = os.path.join(args.out, "inputs", spec["input"])
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, width=1000)
    cmd = [args.app, "-i", path]
    if args.np > 1:
        cmd = ["mpirun", "-np", str(args.np)] + cmd
    log = os.path.join(args.out, "logs", case + ".log")
    t0 = time.time()
    with open(log, "w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    print("  %-14s %6.1f s  exit %d  (%s)" % (case, time.time() - t0, code, log))
    if code != 0:
        sys.exit("run failed: see " + log)


def newton_iterations(log):
    """Newton iterations per accepted step from the app's step lines."""
    its = []
    with open(log) as f:
        for line in f:
            m = re.match(r"step (\d+) t = \S+ newton iterations = (\d+)", line)
            if m:
                its.append(int(m.group(2)))
    return its


def reference_newton_iterations(path):
    """The driver's Newton iterations per step: the row with update_norm = 0 closes a step."""
    counts = {}
    for row in read_csv(path):
        step = int(row["step"])
        counts[step] = max(counts.get(step, 0), int(row["iter"]))
    return [counts[s] for s in sorted(counts)]


def compare(case, args):
    """The framework's errors against the driver's; returns (rows, worst relative discrepancy)."""
    spec = CASES[case]
    ref = read_csv(os.path.join(REF_DIR, spec["ref"]))
    if spec["cols"] is None:
        ours = newton_iterations(os.path.join(args.out, "logs", case + ".log"))
        theirs = reference_newton_iterations(os.path.join(REF_DIR, "myapps_nonlinear_diffusion_kirchhoff_newton.csv"))
        return {"ours": ours, "theirs": theirs}, None
    mine = read_csv(os.path.join(args.out, case, "error_history.csv"))
    abs_col, rel_col = spec["cols"]
    worst = 0.0
    t, ours, theirs = [], [], []
    if len(ref) == 1:   # a steady case: one row without a time, the final state of the run
        ref_rows = [(1, ref[0])]
        last = max(mine, key=lambda r: int(r["step"]))
        mine_rows = [(1, last)]
    else:
        ref_rows = [(int(r["step"]), r) for r in ref]
        mine_rows = [(int(r["step"]), r) for r in mine]
    theirs_by_step = dict(ref_rows)
    for step, row in mine_rows:
        if step not in theirs_by_step:
            continue
        a = float(row["l2"])
        b = float(theirs_by_step[step][abs_col])
        t.append(float(row["t"]) if "t" in row else step)
        ours.append(a)
        theirs.append(b)
        if step == 0 or b < 1e-14:
            continue
        worst = max(worst, abs(a - b) / b)
        if rel_col:
            ra, rb = float(row["rel_l2"]), float(theirs_by_step[step][rel_col])
            if rb > 1e-14:
                worst = max(worst, abs(ra - rb) / rb)
    return {"t": t, "ours": ours, "theirs": theirs}, worst


def plot(results, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK_2)
    # Transient histories.
    ax = axes[0, 0]
    for k, case in enumerate(("peclet_1", "peclet_10", "peclet_100")):
        if case not in results:
            continue
        r = results[case]["data"]
        ax.plot(r["t"], r["ours"], color=COLORS[k], linewidth=1.8, label=CASES[case]["label"] + " (framework)")
        stride = max(1, len(r["t"]) // 25)
        ax.plot(r["t"][::stride], r["theirs"][::stride], "o", color=COLORS[k], markerfacecolor="none", markersize=5,
                label=CASES[case]["label"] + " (myapps)")
    ax.set_yscale("log")
    ax.set_xlabel("t", color=INK)
    ax.set_ylabel("L2 error", color=INK)
    ax.set_title("Transient convection-diffusion", color=INK)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[0, 1]
    if "transient_mms" in results:
        r = results["transient_mms"]["data"]
        ax.plot(r["t"], r["ours"], color=COLORS[3], linewidth=1.8, label="framework")
        stride = max(1, len(r["t"]) // 25)
        ax.plot(r["t"][::stride], r["theirs"][::stride], "o", color=COLORS[3], markerfacecolor="none", markersize=5,
                label="myapps")
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("t", color=INK)
    ax.set_ylabel("L2 error", color=INK)
    ax.set_title("Transient diffusion, manufactured solution", color=INK)
    # Steady bars.
    ax = axes[1, 0]
    names, ours, theirs = [], [], []
    for case in ("steady_square", "steady_disk"):
        if case in results:
            r = results[case]["data"]
            names.append(CASES[case]["label"])
            ours.append(r["ours"][-1])
            theirs.append(r["theirs"][-1])
    if names:
        x = range(len(names))
        ax.bar([i - 0.18 for i in x], ours, width=0.36, color=COLORS[0], label="framework")
        ax.bar([i + 0.18 for i in x], theirs, width=0.36, color=COLORS[1], label="myapps")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names)
        ax.set_yscale("log")
        ax.legend(fontsize=8, frameon=False)
    ax.set_ylabel("L2 error", color=INK)
    ax.set_title("Steady convection-diffusion-reaction", color=INK)
    # Newton iterations.
    ax = axes[1, 1]
    if "kirchhoff" in results:
        r = results["kirchhoff"]["data"]
        ax.plot(range(1, len(r["ours"]) + 1), r["ours"], "o-", color=COLORS[0], label="framework (damped Newton)")
        ax.plot(range(1, len(r["theirs"]) + 1), r["theirs"], "s--", color=COLORS[1], markerfacecolor="none",
                label="myapps (full Newton)")
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("time step", color=INK)
    ax.set_ylabel("Newton iterations", color=INK)
    ax.set_title("Nonlinear diffusion (Kirchhoff case)", color=INK)
    fig.tight_layout()
    path = os.path.join(args.out, "error_histories.png")
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    print("wrote " + path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", default="build/apps/scalar_transport")
    parser.add_argument("--np", type=int, default=1, help="MPI ranks")
    parser.add_argument("--out", default="out/scalar_transport")
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--paraview", action="store_true", help="write the ParaView files of every step")
    parser.add_argument("--no-run", action="store_true", help="only re-read <out>/<case>/error_history.csv")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="fail unless every compared error agrees with the driver's to %g" % CHECK_TOL)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not args.no_run:
        print("running %s" % args.app)
        for case in args.cases:
            run(case, args)
    results = {}
    print("%-14s %-22s %14s %14s %10s" % ("case", "", "framework", "myapps", "rel diff"))
    failed = []
    for case in args.cases:
        data, worst = compare(case, args)
        results[case] = {"data": data, "worst": worst}
        if worst is None:
            print("%-14s %-22s %14s %14s %10s" % (case, "Newton its per step", " ".join(map(str, data["ours"])),
                                                   " ".join(map(str, data["theirs"])), "-"))
            continue
        print("%-14s %-22s %14.6e %14.6e %10.2e" % (case, "L2 error at the end", data["ours"][-1], data["theirs"][-1], worst))
        if worst > CHECK_TOL:
            failed.append((case, worst))
    if not args.no_plot:
        plot(results, args)
    if args.check:
        for case, worst in failed:
            print("--check: %s differs from the driver's error history by %.2e > %g" % (case, worst, CHECK_TOL))
        if failed:
            sys.exit(1)
        print("--check: every compared error history agrees with the driver's to %g" % CHECK_TOL)


if __name__ == "__main__":
    main()
