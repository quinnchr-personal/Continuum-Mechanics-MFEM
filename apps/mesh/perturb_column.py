#!/usr/bin/env python3
"""Apply the geometric imperfection of the column-buckling example to a Gmsh
2.2 ASCII mesh of the box [0, w]^2 x [0, L]: every node moves by
  x -= (imperf/2) (1 - cos(2 pi z / L)),  y -= (imperf/2) (1 - cos(2 pi z / L)),
which seeds the first buckling mode (solidmechanicscoupledtheories.github.io,
3D08). Usage: perturb_column.py in.msh out.msh L imperf
"""
import math
import sys

src, dst, L, imperf = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
lines = open(src).read().split("\n")
out = []
i = 0
while i < len(lines):
    out.append(lines[i])
    if lines[i].strip() == "$Nodes":
        n = int(lines[i + 1])
        out.append(lines[i + 1])
        for k in range(n):
            idx, x, y, z = lines[i + 2 + k].split()[:4]
            x, y, z = float(x), float(y), float(z)
            shift = 0.5 * imperf * (1.0 - math.cos(2.0 * math.pi * z / L))
            out.append(f"{idx} {x - shift:.16g} {y - shift:.16g} {z:.16g}")
        i += 2 + n
        continue
    i += 1
open(dst, "w").write("\n".join(out))
