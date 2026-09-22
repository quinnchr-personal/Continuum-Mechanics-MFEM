"""Write dolfinx's create_box tetrahedral mesh as Gmsh 2.2 with box.geo's face names."""
import sys, numpy as np
from mpi4py import MPI
from dolfinx import mesh
nx, ny, nz, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
L = [float(v) for v in sys.argv[5:8]] if len(sys.argv) > 7 else [1.0, 1.0, 1.0]
dom = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], L], [nx, ny, nz], mesh.CellType.tetrahedron)
X = dom.geometry.x; cells = dom.geometry.dofmap
tets = []
for c in cells:
    v = X[c]; d = np.linalg.det(np.array([v[1]-v[0], v[2]-v[0], v[3]-v[0]]))
    tets.append([c[0], c[2], c[1], c[3]] if d < 0 else list(c))
faces = {"bottom": (2, 0.0), "top": (2, L[2]), "front": (1, 0.0), "back": (1, L[1]), "left": (0, 0.0), "right": (0, L[0])}
tris = []
ftags = {}
for i, (name, (ax, val)) in enumerate(faces.items()):
    ftags[name] = i + 1
    for t in tets:
        for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            ids = [t[k] for k in f]
            if all(abs(X[j, ax] - val) < 1e-12 for j in ids): tris.append((i + 1, ids))
with open(out, "w") as f:
    f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n%d\n" % (len(faces) + 1))
    for name, tag in ftags.items(): f.write('2 %d "%s"\n' % (tag, name))
    f.write('3 7 "domain"\n$EndPhysicalNames\n$Nodes\n%d\n' % len(X))
    for i, p in enumerate(X): f.write("%d %.17g %.17g %.17g\n" % (i + 1, *p))
    f.write("$EndNodes\n$Elements\n%d\n" % (len(tris) + len(tets)))
    k = 1
    for tag, ids in tris: f.write("%d 2 2 %d %d %d %d %d\n" % (k, tag, tag, *(j + 1 for j in ids))); k += 1
    for t in tets: f.write("%d 4 2 7 7 %d %d %d %d\n" % (k, *(j + 1 for j in t))); k += 1
    f.write("$EndElements\n")
print(out, len(X), "nodes", len(tets), "tets", len(tris), "boundary triangles")
