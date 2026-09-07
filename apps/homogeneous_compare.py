#!/usr/bin/env python3
"""Run the homogeneous-deformation inputs and compare the probed displacement,
pressure, von Mises stress and Jacobian with the closed-form solutions of
doc/incompressible_hyperelasticity.tex.

    python3 apps/homogeneous_compare.py [--app build/apps/solid_mechanics]
                                        [--tol 1e-8] [--np N] [inputs.yaml ...]

Without inputs every apps/input/homogeneous/*.yaml is run: plane-strain
extension and plane-stress sheet tension in 2D, uniaxial, equibiaxial and
pure-shear tension in 3D (full affine data on the loaded faces, or the
symmetry models with rollers through `components`). The quadrature
quantities are compared in their nodal presentation and, where the input
writes it, in their element-average presentation (rows tagged elem). The
sigma and P rows report the largest error over the six Cauchy / nine first
Piola-Kirchhoff components of the nodal stress fields, with the xx component
shown. Each input is a box whose constrained faces carry an affine
displacement u = G X written as expressions (exactly one Dirichlet entry is
nonzero; the others are zero rollers) and whose remaining faces are free,
so the principal stretches are the diagonal of I + G (2D plane strain: the
out-of-plane stretch is 1) and the principal stress of the first free axis
vanishes (2D plane stress with every edge constrained: the thickness). The
expressions are evaluated here at the origin and the unit points, and must
be affine and independent of t. Exit status 1 if a run fails or any
relative error exceeds the tolerance.
"""
import argparse
import glob
import math
import os
import re
import subprocess
import sys

import yaml

AB_C = [0.5, 1.0 / 20.0, 11.0 / 1050.0, 19.0 / 7000.0, 519.0 / 673750.0]


def beta(material, lam):
    """lambda_a dPsi/dlambda_a up to an isotropic term (doc, Sec. 2.2)."""
    model = material["model"]
    if model == "ogden":
        return [sum(m * l ** a for m, a in zip(material["mu_r"], material["alpha_r"])) for l in lam]
    I1 = sum(l * l for l in lam)
    psi1, psi2 = 0.0, 0.0
    if model == "iso_neo_hookean":
        mu = material.get("mu")
        if mu is None:
            mu = material["E"] / (2.0 * (1.0 + material["nu"]))
        psi1 = 0.5 * mu
    elif model == "mooney_rivlin":
        psi1, psi2 = material["c1"], material["c2"]
    elif model == "yeoh":
        x = I1 - 3.0
        psi1 = material["c10"] + 2.0 * material.get("c20", 0.0) * x + 3.0 * material.get("c30", 0.0) * x * x
    elif model == "gent":
        psi1 = 0.5 * material["mu"] * material["Jm"] / (material["Jm"] - (I1 - 3.0))
    elif model == "arruda_boyce":
        N = material["N"]
        psi1 = material["mu"] * sum((i + 1) * AB_C[i] * I1 ** i / N ** i for i in range(5))
    else:
        raise SystemExit(f"model '{model}' has no closed form here")
    return [2.0 * psi1 * l * l - 2.0 * psi2 / (l * l) for l in lam]


def analytic(material, lam, free_axis=1):
    """Principal Cauchy stresses, mean stress p and von Mises stress for
    principal stretches lam with sigma_free_axis = 0."""
    b = beta(material, lam)
    sigma = [bi - b[free_axis] for bi in b]
    p = sum(sigma) / 3.0
    vm = math.sqrt(1.5 * sum((s - p) ** 2 for s in sigma))
    return sigma, p, vm


def evaluate(expr, x, y, z, t=1.0):
    """Evaluate a data expression of the input schema (README grammar) in Python."""
    env = {"x": x, "y": y, "z": z, "t": t, "pi": math.pi, "sin": math.sin, "cos": math.cos, "tan": math.tan,
           "exp": math.exp, "log": math.log, "sqrt": math.sqrt, "abs": abs, "pow": math.pow, "min": min,
           "max": max, "if": lambda c, a, b: a if c else b}
    return float(eval(expr.replace("^", "**").replace("if(", "if_("), {"__builtins__": {}}, dict(env, if_=env["if"])))


def affine_data(expression, dim):
    """value and G of u = value + G X from expression strings; None if not affine."""
    unit = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    value = [evaluate(e, 0.0, 0.0, 0.0) for e in expression]
    G = [[evaluate(expression[i], *unit[j]) - value[i] for j in range(dim)] for i in range(dim)]
    # Affinity check at a third point and independence of t.
    for i, e in enumerate(expression):
        pt = [0.3, -0.7, 0.4][:dim] + [0.0] * (3 - dim)
        want = value[i] + sum(G[i][j] * pt[j] for j in range(dim))
        if abs(evaluate(e, *pt) - want) > 1e-12 or abs(evaluate(e, *pt, t=0.25) - want) > 1e-12:
            return None
    return value, G


# Faces of apps/mesh/square.geo and box.geo by axis: (name at X_a = 0, name at X_a = L).
FACES = {2: [("left", "right"), ("bottom", "top")],
         3: [("left", "right"), ("front", "back"), ("bottom", "top")]}


def load_case(path):
    cfg = yaml.safe_load(open(path))
    dirichlet = cfg["bcs"]["dirichlet"]
    if cfg["bcs"].get("traction"):
        raise SystemExit(f"{path}: homogeneous inputs have no tractions")
    loaded = []
    for bc in dirichlet:
        dim = len(bc["expression"])
        data = affine_data(bc["expression"], dim)
        if data is None:
            raise SystemExit(f"{path}: Dirichlet data must be affine in X and independent of t")
        if any(abs(g) > 0.0 for row in data[1] for g in row) or any(abs(v) > 0.0 for v in data[0]):
            loaded.append(data)
    if len(loaded) != 1:
        raise SystemExit(f"{path}: expected exactly one nonzero Dirichlet entry (the affine stretch)")
    value, G = loaded[0]
    dim = len(G)
    lam = [G[i][i] + 1.0 for i in range(dim)]
    if dim == 2:
        # plane strain: lambda3 = 1; plane stress: the thickness stretch of an
        # incompressible sheet, lambda3 = 1 / (lambda1 lambda2)
        lam.append(1.0 / (lam[0] * lam[1]) if cfg.get("plane", "strain") == "stress" else 1.0)
    if abs(lam[0] * lam[1] * lam[2] - 1.0) > 1e-12:
        raise SystemExit(f"{path}: the prescribed stretches are not isochoric")
    # The free axis: the first axis with a face that is not constrained in the
    # axis' own component (that face is traction free, so the principal stress
    # along the axis vanishes; a roller on X = 0 constrains x only). With every
    # edge of a plane-stress sheet constrained the free axis is the thickness.
    constrained = set()
    for bc in dirichlet:
        comps = bc.get("components")
        comps = list(range(dim)) if comps is None else [{"x": 0, "y": 1, "z": 2}.get(str(c), c) for c in comps]
        for name in bc["attr"]:
            constrained.update((name, int(c)) for c in comps)
    free = [a for a, faces in enumerate(FACES[dim]) if any((f, a) not in constrained for f in faces)]
    if free:
        free_axis = free[0]
    elif dim == 2 and cfg.get("plane", "strain") == "stress":
        free_axis = 2
    else:
        raise SystemExit(f"{path}: no free axis (every face is constrained in its normal direction)")
    return cfg, dim, G, value, lam, free_axis


PROBE = re.compile(r"probe (\S+) at \((.*?)\): (\w+) =(.*)")


def run(app, path, np_):
    cmd = [app, "-i", path]
    if np_ > 1:
        cmd = ["mpirun", "-np", str(np_)] + cmd
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    probes = {}
    for line in proc.stdout.splitlines():
        m = PROBE.match(line)
        if m:
            name, point, field, values = m.groups()
            probes.setdefault(name, {"point": [float(x) for x in point.split(",")]})
            probes[name][field] = [float(v) for v in values.split()]
    return proc.returncode, probes, proc.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--app", default="build/apps/solid_mechanics")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--np", type=int, default=1, dest="np_")
    args = ap.parse_args()
    inputs = args.inputs or sorted(glob.glob("apps/input/homogeneous/*.yaml"))
    if not inputs:
        raise SystemExit("no inputs found (run from the repository root)")
    if not os.path.exists(args.app):
        raise SystemExit(f"app '{args.app}' not found (make apps)")

    header = f"{'input':<36} {'model':<15} {'quantity':<22} {'numerical':>16} {'analytic':>16} {'rel. error':>11}"
    print(header)
    print("-" * len(header))
    failures = 0
    for path in inputs:
        cfg, dim, G, value, lam, free_axis = load_case(path)
        material = cfg["material"]
        sigma, p, vm = analytic(material, lam, free_axis)
        code, probes, out = run(args.app, path, args.np_)
        label = os.path.basename(path)
        if code != 0 or not probes:
            failures += 1
            print(f"{label:<36} {material['model']:<15} RUN FAILED (exit {code})")
            print(out[-2000:])
            continue
        rows = []
        for name, pr in probes.items():
            X = pr["point"]
            u_exact = [value[i] + sum(G[i][j] * X[j] for j in range(dim)) for i in range(dim)]
            u_num = pr["displacement"]
            scale = max(1.0, max(abs(x) for x in u_exact))
            rows.append((f"u({name})", max(abs(a - b) for a, b in zip(u_num, u_exact)) / scale,
                         math.sqrt(sum(x * x for x in u_num)), math.sqrt(sum(x * x for x in u_exact))))
            sscale = max(1.0, vm)
            if "pressure" in pr:  # mixed formulation only
                rows.append((f"p({name})", abs(pr["pressure"][0] - p) / sscale, pr["pressure"][0], p))
            if "thickness_stretch" in pr:  # plane stress only
                rows.append((f"lambda3({name})", abs(pr["thickness_stretch"][0] - lam[2]), pr["thickness_stretch"][0], lam[2]))
            rows.append((f"vonmises({name})", abs(pr["vonmises"][0] - vm) / sscale, pr["vonmises"][0], vm))
            rows.append((f"J({name})", abs(pr["jacobian"][0] - 1.0), pr["jacobian"][0], 1.0))
            for suffix, tag in (("", ""), ("_elem", " elem")):
                if "cauchy_stress" + suffix in pr:  # xx, yy, zz, xy, yz, xz
                    exact = sigma + [0.0, 0.0, 0.0]
                    got = pr["cauchy_stress" + suffix]
                    err = max(abs(a - b) for a, b in zip(got, exact)) / sscale
                    rows.append((f"sigma{tag}({name})", err, got[0], exact[0]))
                if "pk1_stress" + suffix in pr:  # row-major, P_aa = sigma_a / lambda_a
                    exact = [0.0] * 9
                    for i in range(3):
                        exact[4 * i] = sigma[i] / lam[i]
                    got = pr["pk1_stress" + suffix]
                    err = max(abs(a - b) for a, b in zip(got, exact)) / sscale
                    rows.append((f"P{tag}({name})", err, got[0], exact[0]))
                if "jacobian" + suffix in pr and suffix:
                    rows.append((f"J{tag}({name})", abs(pr["jacobian_elem"][0] - 1.0), pr["jacobian_elem"][0], 1.0))
        for quantity, err, num, exact in rows:
            flag = "" if err <= args.tol else "  <-- FAIL"
            if flag:
                failures += 1
            print(f"{label:<36} {material['model']:<15} {quantity:<22} {num:>16.10f} {exact:>16.10f} {err:>11.2e}{flag}")
        print(f"{'':<36} {'':<15} principal Cauchy stresses sigma = ({sigma[0]:.6f}, {sigma[1]:.6f}, {sigma[2]:.6f}), "
              f"stretches ({lam[0]:.4f}, {lam[1]:.4f}, {lam[2]:.4f}), free axis {'xyz'[free_axis]}")
    print("-" * len(header))
    if failures:
        print(f"FAIL: {failures} quantities exceed tol {args.tol:g} or runs failed")
        return 1
    print(f"PASS: every probed quantity within tol {args.tol:g} of the closed-form solution")
    return 0


if __name__ == "__main__":
    sys.exit(main())
