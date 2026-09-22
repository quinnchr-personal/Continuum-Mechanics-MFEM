"""Turn the TE notebooks into headless run scripts that save their timeHist arrays (plus, for the
stretched cylinder, the reaction of the loaded face from the residual) to hist/<case>.csv."""
import json, glob, os, re

HEADER = '''# Patched from the reference notebook: headless (pyvista/IPython stubbed), histories saved to hist/<case>.csv.
import sys
from unittest.mock import MagicMock
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
sys.modules["pyvista"] = MagicMock()
'''
SAVE = '''# ---- histories (added)
import numpy as _np
_h = {k: v for k, v in sorted(globals().items()) if k.startswith("timeHist") and hasattr(v, "shape")}
if MPI.COMM_WORLD.rank == 0:
    _n = min(min(len(v) for v in _h.values()), ii + 1)   # ii: last increment stored (runs that end early)
    _np.savetxt("hist/%s.csv", _np.column_stack([v[:_n] for v in _h.values()]), delimiter=",", header=",".join(_h), comments="")

'''
REACTIONS = {"TE02": ("timeHist4", "yTop_u2_dofs")}

def patch(code, case):
    lines = code.split("\n")
    out = []
    key = case[:4]
    extra = REACTIONS.get(key)
    last_decl = max((i for i, l in enumerate(lines) if re.match(r"^\s*timeHist\d+\s*=\s*np\.zeros", l)), default=None)
    last_while = max((i for i, l in enumerate(lines) if l.startswith("while (round(t + dt, 9)")), default=None)
    saved = False
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("from IPython.display import Image"):
            out.append(line.replace(s, "Image = lambda *a, **k: None")); continue
        # the save block at the first cell separator after the last time loop
        if not saved and last_while is not None and i > last_while and line.startswith("# %% ----"):
            out.append(SAVE % case); saved = True
        out.append(line)
        ind = re.match(r"^(\s*)", line).group(1)
        if extra and i == last_decl:
            out.append(ind + f"{extra[0]} = np.zeros(shape=[totSteps])")
        if extra and re.match(r"^\s*timeHist0\[ii\]\s*=\s*t\b", line):
            out.append(ind + "# added: the reaction of the loaded face, the residual (no boundary conditions applied) summed over its dofs")
            out.append(ind + "from dolfinx.fem.petsc import assemble_vector as _av")
            out.append(ind + "_Rv = _av(fem.form(Res)); _Rv.assemble()")
            out.append(ind + f"{extra[0]}[ii] = domain.comm.allreduce(float(np.sum(_Rv.array[{extra[1]}[{extra[1]} < _Rv.array.size]])), op=MPI.SUM)")
    if not saved:
        out.append(SAVE % case)
    return "\n".join(out)

for f in sorted(glob.glob("TE*.ipynb")):
    case = os.path.basename(f)[:-6]
    nb = json.load(open(f))
    code = "\n\n# %% ----\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    open(case + "_run.py", "w").write(HEADER + patch(code, case))
    print("wrote", case + "_run.py")
