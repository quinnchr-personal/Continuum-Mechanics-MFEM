"""Turn the FV notebooks into headless run scripts that save their timeHist arrays (and added
reaction / contact-force columns) to hist/<case>.csv, as the finite-elasticity set was run."""
import json, glob, os, re

HEADER = '''# Patched from the reference notebook: headless (pyvista/IPython stubbed), histories saved to hist/<case>.csv.
import sys
from unittest.mock import MagicMock
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
sys.modules["pyvista"] = MagicMock()
'''
SAVE = '''{ind}# ---- histories (added)
{ind}import numpy as _np
{ind}_h = {{k: v for k, v in sorted(globals().items()) if k.startswith("timeHist") and hasattr(v, "shape")}}
{ind}if MPI.COMM_WORLD.rank == 0:
{ind}    _n = min(min(len(v) for v in _h.values()), ii + 1)   # ii: last increment stored (runs that end early)
{ind}    _np.savetxt({name}, _np.column_stack([v[:_n] for v in _h.values()]), delimiter=",", header=",".join(_h), comments="")
'''
# added columns: (history name, dof list expression) summed from the residual without boundary conditions
REACTIONS = {
    "FV01": [("timeHist3", "yTop_u2_dofs")], "FV02": [("timeHist3", "yTop_u2_dofs")],
    "FV03": [("timeHist3", "yTop_u2_dofs")], "FV05": [("timeHist3", "yTop_u2_dofs")],
    "FV06": [("timeHist3", "yTop_u2_dofs")], "FV07": [("timeHist3", "yTop_u1_dofs")],
    "FV09": [("timeHist3", "yTop_u1_dofs")], "FV11": [("timeHist3", "zTop_u3_dofs")],
    "FV12": [("timeHist4", "zTop_all_u3_dofs")],
}

def patch(code, case, key):
    lines = code.split("\n")
    out = []
    n_hist = 0
    for line in lines:
        s = line.strip()
        if s.startswith("from IPython.display import Image"):
            out.append(line.replace(s, "Image = lambda *a, **k: None")); continue
        out.append(line)
        m = re.match(r"^(\s*)timeHist(\d+)\s*=\s*np\.zeros", line)
        if m:
            n_hist = max(n_hist, int(m.group(2)) + 1)
    lines, out = out, []
    # declarations of the added histories after the last existing one; the residual sums after timeHist0[ii] = t
    extra = REACTIONS.get(key, [])
    last_decl = max((i for i, l in enumerate(lines) if re.match(r"^\s*timeHist\d+\s*=\s*np\.zeros", l)), default=None)
    for i, line in enumerate(lines):
        out.append(line)
        ind = re.match(r"^(\s*)", line).group(1)
        if key == "FV12" and line.strip().startswith("load_u2_dofs ="):
            out.append(ind + "zTop_all_u3_dofs = fem.locate_dofs_topological(ME.sub(0).sub(2), facet_tags.dim, facet_tags.find(6))")
        if i == last_decl:
            for name, _ in extra:
                out.append(ind + f"{name} = np.zeros(shape=[totSteps])")
        if re.match(r"^\s*timeHist0\[ii\]\s*=\s*t\b", line):
            for name, dofs in extra:
                out.append(ind + "# added: the reaction of the loaded face, the residual (no boundary conditions applied) summed over its dofs")
                out.append(ind + "from dolfinx.fem.petsc import assemble_vector as _av")
                if key == "FV12":
                    out.append(ind + "_Cv = _av(fem.form(-dot(contact_traction, u_test)*ds)); _Cv.assemble()   # the contact term alone")
                    out.append(ind + f"{name}[ii] = domain.comm.allreduce(float(np.sum(_Cv.array[{dofs}])), op=MPI.SUM)")
                else:
                    out.append(ind + "_Rv = _av(fem.form(Res)); _Rv.assemble()")
                    out.append(ind + f"{name}[ii] = domain.comm.allreduce(float(np.sum(_Rv.array[{dofs}])), op=MPI.SUM)")
        if 'print("End computation")' in line:
            name = f'"hist/{case}.csv"' if "for rate in" not in code else f'"hist/{case}_rate%g.csv" % rate'
            out.append(SAVE.format(ind=ind, name=name))
    return "\n".join(out)

for f in sorted(glob.glob("FV*.ipynb")):
    if f.startswith("FV00"): continue
    case = os.path.basename(f)[:-6]
    key = case[:4]
    nb = json.load(open(f))
    code = "\n\n# %% ----\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    open(case + "_run.py", "w").write(HEADER + patch(code, case, key))
    print("wrote", case + "_run.py")
# the parallel version of FV12 (a script already): the same patch
case = "FV12_NBR_sphere_indentation_MPI"
open(case + "_run.py", "w").write(HEADER + patch(open(case + ".py").read(), case, "FV12"))
print("wrote", case + "_run.py")
