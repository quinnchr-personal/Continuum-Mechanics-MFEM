#!/usr/bin/env python3
"""Reproduce the result plots of the finite elasticity examples of Anand's
Introduction to coupled theories in solid mechanics (FEniCSx companion codes,
solidmechanicscoupledtheories.github.io, section 1) from the runs of
apps/input/anand_coupled_theories/finite_elasticity/*.yaml, and overlay a
reference curve where one exists.

    python3 apps/anand_plots.py [--inputs DIR] [--logs DIR] [--out DIR] [case ...]

Run it from the repository root, as the app. The data come from the ParaView
output of a run (output.paraview of <case>.yaml; every case that has one by
default): the probes are the nodal fields at the probe points of the input, and
the reactions are the traction of the nodal pk1_stress integrated over the faces
of the Dirichlet entries (see "ParaView output" below; needs pyvista, PyYAML and
scipy). With --logs DIR they come from DIR/<case>.log instead, the saved stdout
of `solid_mechanics -i <case>.yaml` (the inputs print the probes and the
reactions after every step); those reactions are the app's own, the residual
summed over the prescribed dofs. The plots go to plots/ beside the ParaView
directories, or beside DIR.
The reference site overlays no analytical curves; the ones added here are:

  01 uniaxial tension     nominal stress vs stretch of the homogeneous
                          Arruda-Boyce block at K = 1000 G (lateral stretch
                          from sigma_22 = 0), with the Pade inverse Langevin
                          of the inputs and the reference, and the
                          incompressible limit
  02 simple shear         nominal shear stress vs shear strain, same model
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

STEP = re.compile(r"^step (\d+) t = ([0-9.eE+-]+) (probe|reaction) (\S+?)(?: at \((.*?)\))?: (\w+) =(.*)$")


def psi1_pade(I1):
    """Psi_1 of the inputs (inverse_langevin: pade, the default) and the reference: Gshear/2
    with the Pade inverse Langevin, Gshear = G0 (lambda_L / (3 lambda_bar))
    L^-1(lambda_bar / lambda_L). The reference caps lambda_bar / lambda_L at
    0.95 (the cube of 01 ends at 0.86); this code does not."""
    lb = np.sqrt(np.asarray(I1, dtype=float) / 3.0)
    z = np.minimum(lb / LAMBDA_L, 0.95)
    beta = z * (3.0 - z * z) / (1.0 - z * z)
    return 0.5 * G0 * LAMBDA_L / (3.0 * lb) * beta




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


def read_reactions_csv(path):
    """{("reaction", name): {"t", "force", "moment"}} from the app's <collection>/reactions.csv
    (output.reactions with ParaView output: step, t, then <name>_fx .. _mz per entry)."""
    with open(path) as f:
        header = f.readline().strip().split(",")
    raw = np.atleast_2d(np.genfromtxt(path, delimiter=",", skip_header=1))
    data = {}
    for k in range(2, len(header), 6):
        name = header[k][:-3]
        data[("reaction", name)] = {"t": raw[:, 1], "force": raw[:, k:k + 3], "moment": raw[:, k + 3:k + 6]}
    return data


def probe(data, name, field):
    d = data[("probe", name)]
    return d["t"], d[field]


def reaction(data, name):
    d = data[("reaction", name)]
    return d["t"], d["force"], d["moment"]


# ------------------------------------------------------------ ParaView output
#
# The same data from the ParaView output of a run. The .pvd lists the steps of the last run and
# their t; the points of the output are the reference coordinates (the mesh never moves). A probe
# is the interpolant of the nodal fields at its point. The reactions are not in the output: the
# force and the moment of a Dirichlet entry are those of the traction P N of the nodal pk1_stress
# on the faces of its boundary groups,
#     F = int P N dA,   M = int (X + u) x (P N) dA   (about the origin, as the app's),
# restricted to the prescribed components as the app's are. Since that stress is recovered from the
# quadrature points, the force differs from the app's reaction where the loaded face meets free
# faces or clamped corners: +9.7 percent for 04 and -8 percent for 08 at the end, +3.5 for 02, 0.6
# for 03, 0.1 for 09 and 10, round-off for 01 (measured 2026-09-22). main() therefore takes the
# reactions from the log when it is beside the output. The faces are those of the groups in
# the Gmsh file, found in the output by their corner nodes, and the integrals are Gauss quadrature
# of the Lagrange interpolants with VTK's own shape functions, so that the quadrature adds no error
# to the nodal stress. That stress is recovered from the quadrature points (output.nodal_projection),
# so the force equals the app's reaction, the sum of the residual over the entry's dofs, only up to
# the error of the recovery: exactly in a homogeneous state, within the discretization error otherwise.

GMSH_FACE_CORNERS = {2: 3, 9: 3, 3: 4, 10: 4, 16: 4}    # triangles of 3, 6 and quadrilaterals of 4, 9, 8 nodes
VTK_FACE_CORNERS = {69: 3, 70: 4}                        # VTK_LAGRANGE_TRIANGLE, VTK_LAGRANGE_QUADRILATERAL


def read_msh_faces(path):
    """Nodes (n, 3), {group name: tag} and {tag: faces} of the boundary groups of a 3D Gmsh 2.2
    ASCII mesh; a face is the frozenset of the indices of its corner nodes."""
    lines = [line.strip() for line in open(path)]

    def section(name):
        if "$" + name not in lines:
            return []
        return lines[lines.index("$" + name) + 1:lines.index("$End" + name)]

    fmt = section("MeshFormat")[0].split()
    if not fmt[0].startswith("2") or fmt[1] != "0":
        raise ValueError(f"{path}: expected a Gmsh 2.2 ASCII mesh (make meshes), found format {' '.join(fmt)}")
    names = {}
    for line in section("PhysicalNames")[1:]:
        dim, tag, name = line.split(None, 2)
        if dim == "2":
            names[name.strip('"')] = int(tag)
    rows = [line.split() for line in section("Nodes")[1:]]
    index = {int(r[0]): k for k, r in enumerate(rows)}
    X = np.array([[float(v) for v in r[1:4]] for r in rows])
    faces = {}
    for line in section("Elements")[1:]:
        e = line.split()
        corners = GMSH_FACE_CORNERS.get(int(e[1]))
        if corners is not None:                          # tags: physical group, then elementary entity
            first = 3 + int(e[2])
            faces.setdefault(int(e[3]), []).append(frozenset(index[int(v)] for v in e[first:first + corners]))
    return X, names, faces


def face_quadrature(corners, m=6):
    """Points (r, s) and weights of VTK's parametric quadrilateral [0, 1]^2 or triangle r, s >= 0,
    r + s <= 1 (the Gauss points of the square collapsed onto it)."""
    x, w = np.polynomial.legendre.leggauss(m)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    r, s = np.meshgrid(x, x, indexing="ij")
    ww = np.outer(w, w)
    if corners == 3:
        s, ww = s * (1.0 - r), ww * (1.0 - r)
    return r.ravel(), s.ravel(), ww.ravel()


def shape_tables(face, corners):
    """Weights (q), shape functions (q, a) and their parametric derivatives at the quadrature points
    of a VTK face, from the cell itself: the node ordering and the order are then VTK's."""
    n = face.GetNumberOfPoints()
    r, s, w = face_quadrature(corners)

    def shape(rq, sq):
        N = np.zeros(n)
        face.InterpolateFunctions((rq, sq, 0.0), N)
        return N

    # VTK lays the derivatives out per node for some cells and per direction for others
    # (Lagrange quadrilateral and triangle, VTK 9): take the layout that matches a difference.
    h, d = 1e-6, np.zeros(2 * n)
    face.InterpolateDerivs((r[0], s[0], 0.0), d)
    fd = (shape(r[0] + h, s[0]) - shape(r[0] - h, s[0])) / (2.0 * h)
    layouts = [lambda v: (v[0::2], v[1::2]), lambda v: (v[:n], v[n:])]
    layout = [f for f in layouts if np.allclose(f(d)[0], fd, atol=1e-5)]
    if len(layout) != 1:
        raise RuntimeError(f"{face.GetClassName()}: unknown layout of InterpolateDerivs")
    N, dr, ds = (np.zeros((len(w), n)) for _ in range(3))
    for q in range(len(w)):
        N[q] = shape(r[q], s[q])
        face.InterpolateDerivs((r[q], s[q], 0.0), d)
        dr[q], ds[q] = layout[0](d)
    return w, N, dr, ds


def face_rules(grid, node_of_point, faces, what):
    """Quadrature of the faces of the output mesh that make up a boundary group:
    [(point ids (f, a), H (f, a, b, 3))] per kind of face, H_abj = int N_a N_b N_j dA with the
    outward reference normal N, so that int x_i P_lj N_j dA = x_ai P_blj H_abj."""
    pts = np.asarray(grid.points)
    conn, off = np.asarray(grid.cell_connectivity), np.asarray(grid.offset)
    on_group = np.zeros(node_of_point.max() + 2, dtype=bool)
    on_group[list(set().union(*faces))] = True
    touching = np.add.reduceat(on_group[node_of_point][conn].astype(int), off[:-1]) >= 3
    found, tables = {}, {}
    for c in np.nonzero(touching)[0]:
        cell = grid.GetCell(int(c))
        center = pts[conn[off[c]:off[c + 1]]].mean(axis=0)
        for j in range(cell.GetNumberOfFaces()):
            face = cell.GetFace(j)                       # owned by the cell: use it before the next call
            corners = VTK_FACE_CORNERS.get(face.GetCellType())
            if corners is None:
                raise ValueError(f"{what}: cells of {face.GetClassName()}; the high-order ParaView "
                                 f"output is needed (output.high_order: true, the default)")
            ids = [face.GetPointId(k) for k in range(face.GetNumberOfPoints())]
            if frozenset(node_of_point[ids[:corners]]) not in faces:
                continue
            key = (corners, len(ids))
            if key not in tables:
                tables[key] = shape_tables(face, corners)
            found.setdefault(key, []).append((ids, center))
    matched = sum(len(v) for v in found.values())
    if matched != len(faces):
        raise ValueError(f"{what}: {matched} of the {len(faces)} faces of the group found in the output mesh")
    rules = []
    for key, items in found.items():
        w, N, dr, ds = tables[key]
        ids = np.array([i for i, _ in items])
        X = pts[ids]
        normal = np.cross(np.einsum("qa,fai->fqi", dr, X), np.einsum("qa,fai->fqi", ds, X))   # N dA / (dr ds)
        H = np.einsum("q,qa,qb,fqj->fabj", w, N, N, normal, optimize=True)
        inward = np.einsum("fj,fj->f", H.sum(axis=(1, 2)), X[:, :key[0]].mean(axis=1) - np.array([c for _, c in items])) < 0.0
        H[inward] *= -1.0
        rules.append((ids, H))
    return rules


def resultant(rules, x, P, mask):
    """Force and moment about the origin of the components `mask` of the traction P N on the faces."""
    F, A = np.zeros(3), np.zeros((3, 3))
    for ids, H in rules:
        Pm = P[ids].reshape(ids.shape + (3, 3)) * mask[None, None, :, None]      # pk1_stress is row-major
        F += np.einsum("fblj,fbj->l", Pm, H.sum(axis=1), optimize=True)
        A += np.einsum("fai,fblj,fabj->il", x[ids], Pm, H, optimize=True)        # int x_i T_l dA
    return F, np.array([A[1, 2] - A[2, 1], A[2, 0] - A[0, 2], A[0, 1] - A[1, 0]])


def probe_weights(grid, point, tol):
    """Point ids and weights of the interpolant at a point: the node there, else the shape functions
    of the cell that holds it (None outside the mesh)."""
    import vtk
    pts = np.asarray(grid.points)
    d = np.linalg.norm(pts - point, axis=1)
    if d.min() < tol:
        return np.array([int(np.argmin(d))]), np.array([1.0])
    sub, pcoords, w = vtk.reference(0), [0.0] * 3, [0.0] * grid.GetMaxCellSize()
    c = grid.FindCell([float(v) for v in point], None, 0, tol * tol, sub, pcoords, w)
    if c < 0:
        return None
    ids = np.asarray(grid.cell_connectivity)[grid.offset[c]:grid.offset[c + 1]]
    return ids, np.array(w[:len(ids)])


def paraview_steps(cfg):
    """[(t, file)] of the .pvd of an input's ParaView output (None without one) and the directory."""
    import glob
    import xml.etree.ElementTree as ET
    base = cfg["output"]["paraview"]
    pvd = glob.glob(os.path.join(base, "*.pvd"))
    if not pvd:
        return None, base
    sets = ET.parse(pvd[0]).getroot().iter("DataSet")
    return [(float(s.get("timestep")), os.path.join(base, s.get("file"))) for s in sets], base


def read_paraview(cfg):
    """The data of read_log from the ParaView output of the input cfg (run from the repository root,
    as the app is: the paths of the input are relative to it)."""
    import glob
    import pyvista as pv
    from scipy.spatial import cKDTree
    steps, base = paraview_steps(cfg)
    if not steps:
        raise FileNotFoundError(f"no .pvd with steps in {base}: run the case first, from the repository root")
    stale = len(glob.glob(os.path.join(base, "Cycle*"))) - len(steps)
    if stale > 0:
        print(f"note: the .pvd of {base} lists {len(steps)} steps, to t = {steps[-1][0]:g}; {stale} more Cycle "
              f"directories are left from an earlier, longer run and are not read", file=sys.stderr)
    grid = pv.read(steps[0][1])
    pts = np.asarray(grid.points)
    tol = 1e-8 * np.linalg.norm(np.ptp(pts, axis=0))
    probes = {}
    for p in cfg["output"].get("probes", []):
        found = probe_weights(grid, np.array(p["point"], dtype=float), tol)
        if found is not None:
            probes[p["name"]] = found
    entries = []
    dirichlet = cfg.get("bcs", {}).get("dirichlet", [])
    if dirichlet:
        Xm, names, faces = read_msh_faces(cfg["mesh"]["file"])
        d, node_of_point = cKDTree(Xm).query(pts, distance_upper_bound=tol)       # len(Xm) where there is none
        for i, e in enumerate(dirichlet):
            name = e.get("name", f"dirichlet[{i}]")
            tags = [names[a] if isinstance(a, str) else int(a) for a in e["attr"]]
            group = set().union(*(faces.get(tag, []) for tag in tags))
            mask = np.zeros(3)
            mask[[{"x": 0, "y": 1, "z": 2}.get(c, c) for c in e.get("components", [0, 1, 2])]] = 1.0
            entries.append((name, face_rules(grid, node_of_point, group, f"{base}, {name}"), mask))
    data = {}
    for t, path in steps:
        g = grid if path == steps[0][1] else pv.read(path)
        if g.n_points != grid.n_points:
            raise ValueError(f"{path}: the mesh differs from that of the first step")
        fields = {f: np.asarray(g.point_data[f]) for f in g.point_data.keys()}
        for name, (ids, w) in probes.items():
            entry = data.setdefault(("probe", name), {"t": []})
            entry["t"].append(t)
            for f, v in fields.items():
                entry.setdefault(f, []).append(np.atleast_1d(w @ v[ids]))
        if "pk1_stress" in fields and "displacement" in fields:
            for name, rules, mask in entries:
                entry = data.setdefault(("reaction", name), {"t": [], "force": [], "moment": []})
                F, M = resultant(rules, pts + fields["displacement"], fields["pk1_stress"], mask)
                entry["t"].append(t)
                entry["force"].append(F)
                entry["moment"].append(M)
    return {k: {f: np.array(v) for f, v in d.items()} for k, d in data.items()}


# ------------------------------------------------------------------ references

def uniaxial_nominal(lam, psi1, kappa=KBULK):
    """Nominal stress of homogeneous uniaxial tension of the decoupled model
    sigma = 2 Psi_1(I1bar) J^-1 dev(Bbar) + kappa ln(J)/J I (the logarithmic volumetric
    law of the inputs and the reference) with the lateral stretch from sigma_22 = 0
    (kappa = None: incompressible, 2 Psi_1 (lambda - lambda^-2))."""
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
            return 2.0 * float(psi1(I1b)) * J ** (-5.0 / 3.0) * (l2 * l2 - l1 * l1) / 3.0 + kappa * math.log(J) / J
        l2 = brentq(sigma22, 0.2, 1.5)
        J = l1 * l2 * l2
        I1b = J ** (-2.0 / 3.0) * (l1 * l1 + 2.0 * l2 * l2)
        s11 = 2.0 * float(psi1(I1b)) * J ** (-5.0 / 3.0) * 2.0 * (l1 * l1 - l2 * l2) / 3.0 + kappa * math.log(J) / J
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

REFERENCE_DIR = None
REF_STYLE = dict(marker="x", ms=4, ls="none", color="k", label="reference (FEniCSx)")


def ref_force(ref, traction, reaction, scale=1.0):
    """The reference's force: its reaction (the residual summed over the loaded face, a
    column added to its notebooks, reference/README.md) when the CSV has that column,
    else its plotted traction integral of the finite element stress; with the label."""
    if ref.shape[0] > reaction and np.any(ref[reaction] != 0.0):
        return scale * ref[reaction], "reference (FEniCSx, reaction)"
    return scale * ref[traction], "reference (FEniCSx, traction integral)"


def reference(case):
    """The histories of the reference's own run, REFERENCE_DIR/<case>.csv with
    the columns timeHist0, timeHist1, ... of its notebook (reference/README.md),
    or None."""
    path = os.path.join(REFERENCE_DIR or "", case + ".csv")
    if REFERENCE_DIR is None or not os.path.exists(path):
        return None
    return np.atleast_2d(np.genfromtxt(path, delimiter=",", skip_header=1)).T


def plot_01(data, ax):
    t, u = probe(data, "corner", "displacement")
    _, f, _ = reaction(data, "loaded")
    lam = (10.0 + u[:, 1]) / 10.0
    ax.plot(lam, f[:, 1] / 100.0 / 1e3, "o-", ms=3, label="this code (reaction / area)")
    lam_ref = np.linspace(1.0, lam.max(), 200)
    ax.plot(lam_ref, uniaxial_nominal(lam_ref, psi1_pade) / 1e3, "-", label="homogeneous, K = 1000 G")
    ax.plot(lam_ref, uniaxial_nominal(lam_ref, psi1_pade, None) / 1e3, ":", color="C1", alpha=0.7,
            label="homogeneous, incompressible")
    ref = reference("01_uniaxial_tension")
    if ref is not None:
        f, label = ref_force(ref, 1, 2, 1e-3)
        ax.plot((10.0 + ref[0]) / 10.0, f, **{**REF_STYLE, "label": label})
    ax.set_xlabel("stretch")
    ax.set_ylabel("nominal stress P22 (MPa)")
    ax.set_title("01 uniaxial tension")


def plot_02(data, ax):
    t, u = probe(data, "top_corner", "displacement")
    _, f, _ = reaction(data, "top")
    gamma = u[:, 0] / 1.0
    ax.plot(gamma, f[:, 0] / 1.0, "o-", ms=3, label="this code (reaction / area)")
    g_ref = np.linspace(-1.0, 1.0, 200)
    ax.plot(g_ref, shear_nominal(g_ref, psi1_pade), "-", label="homogeneous simple shear")
    ref = reference("02_simple_shear")
    if ref is not None:
        f, label = ref_force(ref, 1, 2)
        ax.plot(ref[0] / 1.0, f, **{**REF_STYLE, "label": label})
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
    tq = np.array([torsion(p, a, psi1_pade)[0] for p in psi_ref])
    ax.plot(psi_ref * 1e3, tq * 1e-6, "--", color="C0", alpha=0.7, label="torque, Rivlin")
    fz = np.array([torsion(p, a, psi1_pade)[1] for p in psi_ref])
    ax2.plot(psi_ref * 1e3, fz * 1e-3, "--", color="C3", alpha=0.7, label="axial force, Rivlin")
    ref = reference("03_cylinder_torsion")
    if ref is not None:
        # The reaction columns are the torque about +x and the axial force on the bottom face;
        # the notebook's own integrals use the outward normal there, hence the signs.
        tq, label = ref_force(ref, 1, 3, 1e-6)
        fz, _ = ref_force(ref, 2, 4, 1e-3)
        sgn = -1.0 if "reaction" in label else 1.0
        ax.plot(ref[0] / L * 1e3, sgn * tq, **{**REF_STYLE, "label": "torque, " + label})
        ax2.plot(ref[0] / L * 1e3, sgn * fz, **{**REF_STYLE, "marker": "+", "label": "axial force, " + label})
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
    ref = reference("04_plate_with_hole")
    if ref is not None:
        f, label = ref_force(ref, 1, 2, 2.0 / (W0 * t0) / 1e3)
        ax.plot((L0 + ref[0]) / L0, f, **{**REF_STYLE, "label": label})
    ax.set_xlabel("stretch")
    ax.set_ylabel("nominal stress P11 (MPa)")
    ax.set_title("04 plate with a hole")


def plot_inflation(data, ax, name, A, B, p_max, sphere, case):
    t, u = probe(data, "inner", "displacement")
    ax.plot(p_max * t, u[:, 0], "o-", ms=3, label="this code")
    a_ref = np.linspace(A, A + 1.15 * max(u[:, 0].max(), 0.1), 120)
    p_ref = np.array([inflation_pressure(a, A, B, psi1_pade, sphere) for a in a_ref])
    ax.plot(p_ref, a_ref - A, "-", label="incompressible, quadrature in r")
    ref = reference(case)
    if ref is not None:
        ax.plot(ref[1], ref[2], **REF_STYLE)
    ax.set_xlabel("internal pressure (kPa)")
    ax.set_ylabel("displacement of the inner wall (mm)")
    ax.set_title(name)


def plot_05(data, ax):
    plot_inflation(data, ax, "05 cylinder inflation", 10.0, 11.0, 50.0, False, "05_cylinder_inflation")


def plot_06(data, ax):
    plot_inflation(data, ax, "06 sphere inflation", 10.0, 11.0, 35.0, True, "06_sphere_inflation")


def plot_07(data, ax):
    t, u = probe(data, "footing_center", "displacement")
    ax.plot(1500.0 * t, -u[:, 2], "o-", ms=3, label="this code")
    ref = reference("07_cube_footing")
    if ref is not None:
        ax.plot(ref[1], -ref[2], **REF_STYLE)
    ax.set_xlabel("pressure on the patch (kPa)")
    ax.set_ylabel("settlement of the footing centre (mm)")
    ax.set_title("07 cube footing")


def plot_08(data, ax):
    t, f, _ = reaction(data, "top")
    L, w = 20.0, 1.0
    shortening = 2.5 * t
    ax.plot(shortening, -f[:, 2], "o-", ms=3, label="this code (reaction)")
    ref = reference("08_column_buckling")
    if ref is not None:
        f, label = ref_force(ref, 2, 3, -1.0)
        ax.plot(-ref[1], f, **{**REF_STYLE, "label": label})
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
    ref = reference("09_spherical_inclusion")
    if ref is not None:
        f, label = ref_force(ref, 1, 2, 1e-3)
        ax.plot((10.0 + ref[0]) / 10.0, f, **{**REF_STYLE, "label": "cube with inclusion, " + label})
    lam_ref = np.linspace(1.0, lam.max(), 200)
    ax.plot(lam_ref, uniaxial_nominal(lam_ref, psi1_pade) / 1e3, "-", label="matrix alone, homogeneous (K = 1000 G)")
    ax.set_xlabel("stretch")
    ax.set_ylabel("nominal stress P22 (MPa)")
    ax.set_title("09 spherical inclusion")


def plot_10(data, ax):
    t, f, m = reaction(data, "top")
    theta = 2.0 * math.pi * t
    # The reference's rotation matrix turns the face clockwise about z, so
    # the torque about +z is negative; its magnitude is plotted.
    ax.plot(theta, -m[:, 2], "o-", ms=3, label="torque (magnitude), this code")
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
    ap.add_argument("cases", nargs="*", help="case names (default: every case with output, or with a log)")
    ap.add_argument("--inputs", default="apps/input/anand_coupled_theories/finite_elasticity",
                    help="directory of the inputs <case>.yaml")
    ap.add_argument("--logs", default=None, help="read DIR/<case>.log, the saved stdout of the runs, "
                    "instead of their ParaView output")
    ap.add_argument("--out", default=None, help="plot directory (default: plots/ beside the ParaView "
                    "directories, or beside --logs)")
    ap.add_argument("--reference", default=None, help="directory of the reference's own histories <case>.csv "
                    "(default: reference/ in the inputs directory); overlaid where present")
    args = ap.parse_args()
    global REFERENCE_DIR
    REFERENCE_DIR = args.reference or os.path.join(args.inputs, "reference")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configs = {}
    if args.logs is None:
        import yaml
        for c in args.cases or PLOTS:
            path = os.path.join(args.inputs, c + ".yaml")
            if os.path.exists(path):
                configs[c] = yaml.safe_load(open(path))
        cases = args.cases or [c for c in configs if paraview_steps(configs[c])[0]]
        missing = f"no ParaView output of the inputs in {args.inputs} found (run from the repository root)"
        needs = ("the input needs the probes and the named Dirichlet entries of the plot, and displacement and "
                 "pk1_stress among its output.fields")
    else:
        cases = args.cases or [c for c in PLOTS if os.path.exists(os.path.join(args.logs, c + ".log"))]
        missing = f"no logs found in {args.logs}"
        needs = "the log needs the probes and reactions of the current input"
    if not cases:
        raise SystemExit(missing)
    status = 0
    for case in cases:
        try:
            if args.logs is None:
                if case not in configs:
                    raise FileNotFoundError(os.path.join(args.inputs, case + ".yaml"))
                data = read_paraview(configs[case])
                beside = configs[case]["output"]["paraview"]
                # The app's own reactions, from <collection>/reactions.csv (output.reactions) or
                # else from its log beside the output (logs/<case>.log, as run_set.sh writes them),
                # replace the ones recovered from the nodal stress: that recovery is off by up to
                # 10 percent where the loaded face meets free faces or clamped corners (04, 08,
                # 02); see "ParaView output" above.
                csv = os.path.join(beside.rstrip("/"), "reactions.csv")
                log = os.path.join(os.path.dirname(beside.rstrip("/")), "logs", case + ".log")
                if os.path.exists(csv) or os.path.exists(log):
                    logged = read_reactions_csv(csv) if os.path.exists(csv) else read_log(log)
                    for key, entry in logged.items():
                        if key[0] != "reaction" or key not in data:
                            continue
                        # the output has the initial state too (zero reaction), the log has not
                        tl = np.asarray(entry["t"])
                        rows = [np.flatnonzero(np.abs(tl - t) < 1e-9) for t in data[key]["t"]]
                        if all(len(r) == 1 or abs(t) < 1e-12 for r, t in zip(rows, data[key]["t"])):
                            for field in ("force", "moment"):
                                src = np.asarray(entry[field])
                                data[key][field] = np.array([src[r[0]] if len(r) == 1 else np.zeros(src.shape[1])
                                                             for r in rows])
                            data["reactions_from"] = "reactions.csv" if os.path.exists(csv) else "log"
            else:
                data = read_log(os.path.join(args.logs, case + ".log"))
                beside = args.logs
            out = args.out or os.path.join(os.path.dirname(beside.rstrip("/")), "plots")
            os.makedirs(out, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6.0, 4.2))
            own_legend = PLOTS[case](data, ax)
            if not own_legend:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            png = os.path.join(out, case + ".png")
            fig.savefig(png, dpi=150)
            plt.close(fig)
            t = next(v for v in data.values() if isinstance(v, dict))["t"]
            print(f"{case}: {png} ({len(t)} steps to t = {t[-1]:g})")
        except (KeyError, FileNotFoundError, ValueError) as e:
            status = 1
            hint = "" if isinstance(e, FileNotFoundError) else f"; {needs}"
            print(f"{case}: cannot plot ({type(e).__name__}: {e}){hint}", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
