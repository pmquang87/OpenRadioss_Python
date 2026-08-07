import os

out_path = "pyradioss/elements/shell_thick16.py"

with open("scratch/s16mass3_unrolled.py", "r") as f:
    s16mass3_code = f.read()

with open("scratch/s16rst_unrolled.py", "r") as f:
    s16rst_code = f.read()

with open("scratch/s16deri3_unrolled.py", "r") as f:
    s16deri3_code = f.read()

with open("scratch/s20defo3_unrolled.py", "r") as f:
    s20defo3_code = f.read()

with open("scratch/s20fint3_unrolled.py", "r") as f:
    s20fint3_code = f.read()

# Assemble
full_code = f"""\"\"\"
16-node thick shell element (SHEL16).
(/SHEL16 + /PROP/TSHELL)

Fortran origin: ``engine/source/elements/thickshell/solide16/``
    s16forc3.F  driver: gather coords/velocities, call the chain below
    s16coor3.F  geometry
    s16deri3.F  derivatives
    s16mass3.F  mass initialization
\"\"\"

import numpy as np
from numba import njit, prange

from .. import failure, materials
from ..common.constants import EM20, EP30, ZERO, HALF, TWO, THREE, THIRTY2
from ..common.fastmath import scatter_add3

# IPERM arrays from Fortran for node 9-16 connectivity
_IPERM1 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
_IPERM2 = [0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4, 1, 6, 7, 8, 5]

{s16rst_code.replace("def s16rst", "@njit(cache=True)\\ndef s16rst")}

{s16deri3_code.replace("def s16deri3", "@njit(cache=True)\\ndef s16deri3")}

{s20defo3_code.replace("import numpy as np", "").replace("from numba import njit", "").replace("@njit\\ndef s20defo3", "@njit(cache=True)\\ndef s20defo3").replace("def s20defo3", "@njit(cache=True)\\ndef s20defo3")}

{s20fint3_code.replace("import numpy as np", "").replace("def s20fint3", "@njit(cache=True)\\ndef s20fint3")}

{s16mass3_code.replace("def init_mass", "@njit(cache=True)\\ndef _init_mass")}

def init_group(group, model, log):
    \"\"\"Element buffer + lumped mass.\"\"\"
    conn = group.conn
    n = group.n
    
    xe = model.x0[conn] # (n, 16, 3) 
    
    mass = np.zeros(n)
    mss = np.zeros((n, 8))
    mssx = np.zeros((n, 8))
    stifn = np.zeros(model.numnod)
    
    fill = np.ones(n)
    rho = np.zeros(n)
    vol = np.ones(n) # placeholder for actual volume calculation via Gauss loop
    dtx = np.full(n, 1e20)
    dtelem = np.full(n, 1e20)
    deltax2 = np.ones(n)
    
    for sl, mat, prop in group.state["slices"]:
        rho[sl] = mat.rho0
        
    _init_mass(n, fill, rho, vol, dtx, dtelem, mass, mss, mssx, conn, stifn, deltax2)
    
    # We map back mss and mssx into the global mass array
    # This is simplified. Proper implementation needs full integration.
    
    return None, None, None

def forces(group, x, v, vr, dt, fint, mint):
    pass
"""

with open(out_path, "w") as f:
    f.write(full_code)
print(f"Written to {out_path}")
