#!/usr/bin/env python3
"""The dynamic inputs of apps/input/dynamics against their references: run them with the
general executable, read the time histories from the per-step lines of the logs, print the
measures and draw the histories over the closed forms.

    python3 apps/dynamics_compare.py [--app build/apps/solid_mechanics] [--np N] [--out out/dynamics]
                                     [--cases bar_free_vibration bar_step_load cantilever_vibration
                                              neo_hookean_block_vibration knowles_tube_oscillation]
                                     [--paraview] [--no-run] [--no-plot] [--check]

No driver of its own is needed: with output.probe_every_step, output.reactions and
output.energy the executable prints, after every time step, the registered fields at the
probes, the force of every Dirichlet entry (with the inertia it carries) and the energies, on
lines that begin with `step k t = ...`. For each case this script copies the input to
<out>/inputs (ParaView output only with --paraview), runs it (log in <out>/logs), writes
<out>/<case>/history.csv and draws <out>/<case>.png:

  bar_free_vibration           tip displacement over A cos(w_1 t), w_1 = pi c / (2 L)
  bar_step_load                d'Alembert's wave: the tip displacement over the triangle wave
                               between 0 and 2 p L / E, the wall reaction over the square wave
                               between 0 and -2 p A; run twice, with the input's
                               generalized-alpha (rho_inf = 0.8) and with the trapezoidal rule,
                               which keeps the ringing behind every front
  cantilever_vibration         tip deflection; first frequency from the zero crossings after
                               the pulse against Euler-Bernoulli
  neo_hookean_block_vibration  path of the free corner, and the energies: kinetic, internal
                               and their sum, which generalized-alpha never lets grow
  knowles_tube_oscillation     mixed u-p, incompressible: the inner radius and the pressure at
                               mid-wall over Knowles' ordinary differential equation, which
                               this script integrates with scipy

With --check the script fails unless the measures meet the tolerances of
tests/test_dynamic_verification.cpp (make test): this checks the executable's own output
lines, which that test, driving the library, does not see.

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

INPUT_DIR = "apps/input/dynamics"
CASES = ["bar_free_vibration", "bar_step_load", "cantilever_vibration", "neo_hookean_block_vibration",
         "knowles_tube_oscillation"]

# Categorical slots 1-3 of the reference palette (fixed order, validated for
# colour-vision deficiency in all pairings); text and grid wear ink tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

STEP = re.compile(r"^step (\d+) t = (\S+) (.*)$")
PROBE = re.compile(r"^probe (\S+) at \(.*?\): (\S+) =(.*)$")
REACTION = re.compile(r"^reaction (\S+): force =(.*) moment =(.*)$")
ENERGY = re.compile(r"^energy: kinetic = (\S+) internal = (\S+) external_work = (\S+) balance = (\S+)$")


def prepare_input(case, variant, overrides, args):
    with open(os.path.join(INPUT_DIR, case + ".yaml")) as f:
        cfg = yaml.safe_load(f)
    for section, values in overrides.items():
        cfg[section].update(values)
        for key in [k for k, v in values.items() if v is None]:
            del cfg[section][key]
    out = cfg.setdefault("output", {})
    if args.paraview:
        out["paraview"] = os.path.join(args.out, "paraview", variant)
    else:
        out.pop("paraview", None)
        out["fields"] = [f for f in out.get("fields", ["displacement"])
                         if f in ("displacement", "velocity", "acceleration", "pressure")]
    path = os.path.join(args.out, "inputs", variant + ".yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def run(variant, path, args):
    log = os.path.join(args.out, "logs", variant + ".log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    cmd = ([] if args.np == 1 else ["mpirun", "-np", str(args.np)]) + [args.app, "-i", path]
    start = time.time()
    with open(log, "w") as f:
        code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    print("  %s: %.0f s, exit code %d" % (variant, time.time() - start, code))
    if code != 0:
        sys.exit("run of %s failed, see %s" % (variant, log))
    return log


def parse_log(log):
    """{column: [values per step]} with t, probe <name> <field> <component>, reaction <name>
    <component>, and the four energies; step 0 carries the energies only."""
    rows = {}
    with open(log) as f:
        for line in f:
            m = STEP.match(line.rstrip("\n"))
            if not m:
                continue
            row = rows.setdefault(int(m.group(1)), {"t": float(m.group(2))})
            rest = m.group(3)
            p, r, e = PROBE.match(rest), REACTION.match(rest), ENERGY.match(rest)
            if p:
                for i, v in enumerate(p.group(3).split()):
                    row["%s %s %d" % (p.group(1), p.group(2), i)] = float(v)
            elif r:
                for i, v in enumerate(r.group(2).split()):
                    row["reaction %s %d" % (r.group(1), i)] = float(v)
            elif e:
                for k, v in zip(("kinetic", "internal", "external_work", "balance"), e.groups()):
                    row[k] = float(v)
    steps = sorted(k for k in rows if k > 0)
    columns = sorted({c for k in steps for c in rows[k]} - {"t"})
    history = {"t": [rows[k]["t"] for k in steps]}
    for c in columns:
        history[c] = [rows[k].get(c, float("nan")) for k in steps]
    history["initial_energy"] = rows.get(0, {}).get("kinetic", 0.0) + rows.get(0, {}).get("internal", 0.0)
    return history


def write_csv(path, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = ["t"] + [c for c in history if c not in ("t", "initial_energy")]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for k in range(len(history["t"])):
            w.writerow(["%.12e" % history[c][k] for c in columns])


def upward_crossings(t, y, start):
    return [t[k] + (t[k + 1] - t[k]) * (-y[k]) / (y[k + 1] - y[k])
            for k in range(len(t) - 1) if t[k] >= start and y[k] < 0.0 <= y[k + 1]]


def triangle_wave(t, period):
    s = (t / period) % 1.0
    return 2.0 * s if s < 0.5 else 2.0 - 2.0 * s      # 0 -> 1 -> 0


# --- the figures --------------------------------------------------------------------------------
def new_figure(n_panels, title, subtitle, height=None):
    import warnings
    warnings.filterwarnings("ignore", message="Unable to import Axes3D")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_panels, 1, figsize=(7.6, height or 2.6 * n_panels + 1.0), dpi=200, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes[:, 0]:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
    # title and a subtitle of at most two lines, left-aligned with the plot area
    fig.suptitle(title, x=0.105, y=0.975, ha="left", fontsize=11, color=INK)
    fig.text(0.105, 0.925 if n_panels == 1 else 0.945, subtitle, ha="left", va="top", fontsize=8, color=INK_2,
             linespacing=1.35)
    return plt, fig, list(axes[:, 0])


def finish(plt, fig, axes, path, n_panels):
    top = 0.78 if n_panels == 1 else 0.875
    fig.subplots_adjust(left=0.105, right=0.97, top=top, bottom=0.14 if n_panels == 1 else 0.085, hspace=0.34)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("  wrote %s" % path)


def series(ax, t, y, color, label, linewidth=2.0):
    ax.plot(t, y, color=color, linewidth=linewidth, solid_capstyle="round", solid_joinstyle="round", zorder=3,
            label=label)


def reference(ax, t, y, label):
    """The closed form: context, a hairline in muted ink behind the computed series."""
    ax.plot(t, y, color=MUTED, linewidth=1.0, zorder=2, label=label)


def legend(ax, loc="upper right", ncol=1):
    ax.legend(loc=loc, fontsize=8, frameon=False, labelcolor=INK_2, ncol=ncol).set_zorder(5)


# --- the cases ----------------------------------------------------------------------------------
def bar_free_vibration(args, failures):
    case = "bar_free_vibration"
    history = args.histories(case, case, {})
    A, L, c = 0.01, 10.0, 10.0
    w = 0.5 * math.pi * c / L
    t, u = history["t"], history["tip displacement 0"]
    exact = [A * math.cos(w * x) for x in t]
    worst = max(abs(a - b) for a, b in zip(u, exact)) / A
    balance = max(abs(b) for b in history["balance"]) / history["initial_energy"]
    print("  tip against A cos(w_1 t) over %.0f periods: max difference %.2e A; energy balance %.1e E_0"
          % (t[-1] * w / (2.0 * math.pi), worst, balance))
    if worst > 1e-3:
        failures.append(case + ": tip history off by more than 1e-3 A")
    if balance > 1e-10:
        failures.append(case + ": the trapezoidal rule does not conserve the energy")
    if args.no_plot:
        return
    plt, fig, axes = new_figure(1, "Fixed-free bar, first axial mode",
                                "Tip displacement over five periods, trapezoidal rule with 400 steps per period.\n"
                                "The closed form A cos(w1 t) lies under the computed line: largest difference %.1e A."
                                % worst)
    reference(axes[0], t, exact, "A cos(w1 t)")
    series(axes[0], t, u, SERIES[0], "finite elements")
    axes[0].set_xlabel("time  (period 4 L / c = 4)", color=INK_2, fontsize=9)
    axes[0].set_ylabel("tip displacement", color=INK_2, fontsize=9)
    legend(axes[0], ncol=2)
    axes[0].set_ylim(-1.35 * A, 1.6 * A)
    finish(plt, fig, axes, os.path.join(args.out, case + ".png"), 1)


def bar_step_load(args, failures):
    case = "bar_step_load"
    p, L, E, c, area = 0.25, 10.0, 250.0, 10.0, 1.0
    period, u_static = 4.0 * L / c, p * L / E
    variants = [("generalized-alpha, rho_inf = 0.8", args.histories(case, case, {})),
                ("trapezoidal rule", args.histories(case, case + "_trapezoidal",
                                                    {"dynamics": {"scheme": "newmark", "rho_inf": None}}))]
    measures = []
    for label, h in variants:
        t, tip, wall = h["t"], h["tip displacement 0"], h["reaction wall 0"]
        one = [k for k in range(len(t)) if t[k] <= period + 1e-9]
        mean = sum(0.5 * (tip[k] + tip[k - 1]) * (t[k] - t[k - 1]) for k in one[1:]) / (t[one[-1]] - t[one[0]])
        peak = max(tip[k] for k in one)
        over = [-wall[k] / (2.0 * p * area) - 1.0 for k in one]
        late = [over[i] for i, k in enumerate(one) if 2.0 < t[k] < 2.9]
        ringing = math.sqrt(sum(x * x for x in late) / len(late))
        measures.append((label, peak / u_static, mean / u_static, max(over), ringing))
        print("  %-33s tip peak %.4f, mean %.4f of the static deflection; wall reaction overshoot %.1f %%, "
              "rms ringing for 2 < t < 2.9 %.2f %%" % (label + ":", peak / u_static, mean / u_static,
                                                        100.0 * max(over), 100.0 * ringing))
        if abs(peak / u_static - 2.0) > 0.04 or abs(mean / u_static - 1.0) > 0.03:
            failures.append(case + " (" + label + "): peak or mean of the tip displacement")
    if measures[0][4] > 0.5 * measures[1][4]:
        failures.append(case + ": numerical dissipation does not remove the ringing")
    if args.no_plot:
        return
    plt, fig, axes = new_figure(2, "Step load on a fixed-free bar: d'Alembert's wave",
                                "Tip displacement (top) and force of the wall on the bar (bottom); closed form of "
                                "period 4 L / c = 4 in grey.\nBoth schemes overshoot the first "
                                "front alike; the dissipation of generalized-alpha removes the ringing that follows.")
    t = variants[0][1]["t"]
    reference(axes[0], t, [2.0 * u_static * triangle_wave(x, period) for x in t], "closed form")
    reference(axes[1], t, [-2.0 * p * area if 1.0 < (x % period) < 3.0 else 0.0 for x in t], "closed form")
    # the scheme without dissipation first, so that the one of the input lies on top
    for (label, h), color in reversed(list(zip(variants, SERIES))):
        series(axes[0], h["t"], h["tip displacement 0"], color, label)
        series(axes[1], h["t"], h["reaction wall 0"], color, label, linewidth=1.2)   # a dense signal: thinner
    axes[0].set_ylabel("tip displacement", color=INK_2, fontsize=9)
    axes[1].set_ylabel("wall reaction", color=INK_2, fontsize=9)
    axes[1].set_xlabel("time", color=INK_2, fontsize=9)
    axes[0].set_ylim(-0.1 * u_static, 2.75 * u_static)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(v[0]) for v in variants] + [labels.index("closed form")]
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right", fontsize=8,
                   frameon=False, labelcolor=INK_2, ncol=3).set_zorder(5)
    axes[1].annotate("rms ringing for 2 < t < 2.9: %.2f %% of 2 p A with generalized-alpha, %.2f %% with the "
                     "trapezoidal rule" % (100.0 * measures[0][4], 100.0 * measures[1][4]), (0.0, 1.03),
                     xycoords="axes fraction", fontsize=8, color=INK_2)
    finish(plt, fig, axes, os.path.join(args.out, case + ".png"), 2)


def cantilever_vibration(args, failures):
    case = "cantilever_vibration"
    history = args.histories(case, case, {})
    E, I, rho_a, L = 250.0, 1.0 / 12.0, 2.5, 10.0
    w_eb = 1.875104 ** 2 * math.sqrt(E * I / (rho_a * L ** 4))
    t, uz = history["t"], history["tip displacement 2"]
    crossings = upward_crossings(t, uz, 30.0)
    if len(crossings) < 3:
        failures.append(case + ": fewer than three zero crossings after the pulse")
        return
    w = 2.0 * math.pi * (len(crossings) - 1) / (crossings[-1] - crossings[0])
    print("  first bending frequency from %d periods: w_1 = %.6f, Euler-Bernoulli %.6f (%.2f %%)"
          % (len(crossings) - 1, w, w_eb, 100.0 * (w / w_eb - 1.0)))
    if abs(w / w_eb - 1.0) > 0.02:
        failures.append(case + ": first frequency more than 2 percent from Euler-Bernoulli")
    if args.no_plot:
        return
    plt, fig, axes = new_figure(1, "Cantilever: first bending frequency",
                                "Tip deflection under a half-sine pulse of the tip load until t = 30, free afterwards.\n"
                                "w1 = %.5f from the zero crossings; Euler-Bernoulli %.5f (%+.2f %%: shear and rotary "
                                "inertia lower it)." % (w, w_eb, 100.0 * (w / w_eb - 1.0)))
    series(axes[0], t, uz, SERIES[0], "tip deflection")
    axes[0].axvline(30.0, color=MUTED, linewidth=1.0, zorder=2)
    axes[0].plot(crossings, [0.0] * len(crossings), linestyle="none", marker="o", markersize=6.5,
                 markerfacecolor=SERIES[0], markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4,
                 label="upward zero crossings, one period apart")
    # head room above the curve for the legend and the note on the pulse
    amplitude = max(abs(v) for v in uz)
    axes[0].set_ylim(-1.15 * amplitude, 1.75 * amplitude)
    axes[0].annotate("pulse ends", (30.0, 0.96), xycoords=("data", "axes fraction"), xytext=(-5, 0),
                     textcoords="offset points", fontsize=8, color=INK_2, va="top", ha="right")
    legend(axes[0], ncol=2)
    axes[0].set_xlabel("time  (Euler-Bernoulli period %.1f)" % (2.0 * math.pi / w_eb), color=INK_2, fontsize=9)
    axes[0].set_ylabel("tip deflection $u_z$", color=INK_2, fontsize=9)
    finish(plt, fig, axes, os.path.join(args.out, case + ".png"), 1)


def neo_hookean_block_vibration(args, failures):
    case = "neo_hookean_block_vibration"
    history = args.histories(case, case, {})
    t, e0 = history["t"], history["initial_energy"]
    total = [k + i for k, i in zip(history["kinetic"], history["internal"])]
    highest, end = max(total) / e0, total[-1] / e0
    travel = max(abs(v) for v in history["corner displacement 1"])
    print("  free corner travels %.3f of the block's size; energy: max E / E_0 = %.9f, E_end / E_0 = %.6f"
          % (travel, highest, end))
    if highest > 1.0 + 1e-10 or end >= 1.0:
        failures.append(case + ": the energy exceeds its initial value, or does not end below it")
    if travel < 0.3:
        failures.append(case + ": the motion is not of large amplitude")
    if args.no_plot:
        return
    plt, fig, axes = new_figure(2, "Neo-Hookean block thrown upwards: large-amplitude free vibration",
                                "Displacement of the free corner of the unit block (top) and the energies (bottom).\n"
                                "Generalized-alpha with rho_inf = 0.8: the energy never exceeds its initial value, "
                                "E_end / E_0 = %.6f." % end)
    series(axes[0], t, history["corner displacement 0"], SERIES[0], "$u_x$")
    series(axes[0], t, history["corner displacement 1"], SERIES[1], "$u_y$")
    axes[0].set_ylabel("corner displacement", color=INK_2, fontsize=9)
    axes[0].set_ylim(-1.0, 1.15)     # head room for the legend
    legend(axes[0], loc="upper right", ncol=2)
    series(axes[1], t, history["kinetic"], SERIES[0], "kinetic")
    series(axes[1], t, history["internal"], SERIES[1], "internal")
    series(axes[1], t, total, SERIES[2], "kinetic + internal")
    axes[1].set_ylabel("energy", color=INK_2, fontsize=9)
    axes[1].set_xlabel("time", color=INK_2, fontsize=9)
    axes[1].set_ylim(0.0, 1.35 * e0)
    legend(axes[1], loc="upper right", ncol=3)
    finish(plt, fig, axes, os.path.join(args.out, case + ".png"), 2)


def knowles_tube_oscillation(args, failures):
    """r^2 = R^2 + c(t); rho [(c''/2) ln(b/a) - (c'^2/8)(1/a^2 - 1/b^2)]
    = -(mu/2) [ln(B^2 a^2 / (A^2 b^2)) + c (1/a^2 - 1/b^2)], a^2 = A^2 + c, b^2 = B^2 + c."""
    from scipy.integrate import solve_ivp

    case = "knowles_tube_oscillation"
    history = args.histories(case, case, {})
    A, B, mu, rho, k, R_mid = 1.0, 2.0, 1.0, 1.0, 0.4, 1.5

    def acceleration(c, cd):
        a2, b2 = A * A + c, B * B + c
        elastic = 0.5 * mu * (math.log(B * B * a2 / (A * A * b2)) + c * (1.0 / a2 - 1.0 / b2))
        return (-2.0 * elastic / rho + 0.25 * cd * cd * (1.0 / a2 - 1.0 / b2)) / (0.5 * math.log(b2 / a2))

    def pressure(R, c, cd):
        a2, r2 = A * A + c, R * R + c
        sigma_rr = (rho * (0.25 * acceleration(c, cd) * math.log(r2 / a2) + 0.125 * cd * cd * (1.0 / r2 - 1.0 / a2))
                    + 0.5 * mu * (math.log((r2 - c) * a2 / (r2 * A * A)) - c / r2 + c / a2))
        return sigma_rr - mu * (2.0 * R * R / r2 - r2 / (R * R) - 1.0) / 3.0

    t = history["t"]
    sol = solve_ivp(lambda _, y: [y[1], acceleration(y[0], y[1])], [0.0, t[-1]], [0.0, 2.0 * k], t_eval=t,
                    rtol=1e-11, atol=1e-13)
    a_exact = [math.sqrt(A * A + c) - A for c in sol.y[0]]
    p_exact = [pressure(R_mid, c, cd) for c, cd in zip(sol.y[0], sol.y[1])]
    u, p = history["inner displacement 0"], history["mid pressure 0"]
    amplitude, p_scale = max(abs(v) for v in a_exact), max(abs(v) for v in p_exact)
    worst = max(abs(x - y) for x, y in zip(u, a_exact)) / amplitude
    p_worst = max(abs(x - y) for x, y, time in zip(p, p_exact, t) if time > 0.5) / p_scale
    print("  inner radius against Knowles' equation: max difference %.2e of the amplitude %.4f; mid-wall pressure "
          "within %.2e of its maximum %.4f (after t = 0.5)" % (worst, amplitude, p_worst, p_scale))
    if worst > 5e-3:
        failures.append(case + ": inner radius more than 0.5 percent from Knowles' equation")
    if p_worst > 0.02:
        failures.append(case + ": mid-wall pressure more than 2 percent from Knowles' equation")
    if args.no_plot:
        return
    plt, fig, axes = new_figure(2, "Radial oscillation of an incompressible tube (mixed u-p)",
                                "Displacement of the inner wall (top) and pressure unknown at mid-wall (bottom) over "
                                "Knowles' equation.\nLargest differences: %.1e of the amplitude; %.1e of the largest "
                                "pressure, which the inertia of the inner half carries." % (worst, p_worst))
    reference(axes[0], t, a_exact, "Knowles' equation")
    series(axes[0], t, u, SERIES[0], "finite elements")
    reference(axes[1], t, p_exact, "Knowles' equation")
    series(axes[1], t, p, SERIES[0], "finite elements")
    axes[0].set_ylabel("inner wall displacement", color=INK_2, fontsize=9)
    axes[1].set_ylabel("pressure at R = 1.5", color=INK_2, fontsize=9)
    axes[1].set_xlabel("time", color=INK_2, fontsize=9)
    axes[0].set_ylim(min(a_exact) - 0.03, max(a_exact) + 0.14)      # head room for the legend
    axes[1].set_ylim(min(p_exact) - 0.04, max(p_exact) + 0.18)
    legend(axes[0], ncol=2)
    legend(axes[1], ncol=2)
    finish(plt, fig, axes, os.path.join(args.out, case + ".png"), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--app", default="build/apps/solid_mechanics")
    parser.add_argument("--np", type=int, default=1, help="MPI ranks")
    parser.add_argument("--out", default="out/dynamics")
    parser.add_argument("--cases", nargs="+", default=CASES, choices=CASES)
    parser.add_argument("--paraview", action="store_true", help="keep the ParaView output of the inputs")
    parser.add_argument("--no-run", action="store_true", help="only re-read <out>/logs")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="fail unless the measures meet the tolerances of tests/test_dynamic_verification.cpp")
    args = parser.parse_args()

    def histories(case, variant, overrides):
        log = os.path.join(args.out, "logs", variant + ".log")
        if not args.no_run:
            log = run(variant, prepare_input(case, variant, overrides, args), args)
        history = parse_log(log)
        if not history["t"]:
            sys.exit("no per-step lines in %s" % log)
        write_csv(os.path.join(args.out, variant, "history.csv"), history)
        return history

    args.histories = histories
    failures = []
    for case in args.cases:
        print(case)
        globals()[case](args, failures)
    for f in failures:
        print("FAIL " + f)
    if args.check and failures:
        sys.exit(1)
    if args.check:
        print("dynamics_compare: all measures within tolerance")


if __name__ == "__main__":
    main()
