import sys
import os

geom = open("scratch/dkt18_unrolled_geometry.py").read()
deri = open("scratch/dkt18_unrolled_derivatives.py").read()
rates = open("scratch/dkt18_unrolled_rates.py").read()
assem = open("scratch/dkt18_unrolled_assembly.py").read()

def extract_funcs(code):
    lines = code.split('\n')
    funcs = []
    current = []
    for line in lines:
        if line.startswith("import"):
            continue
        if line.startswith("def "):
            if current:
                funcs.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        funcs.append("\n".join(current))
    return funcs

all_funcs = []
all_funcs.extend(extract_funcs(geom))
all_funcs.extend(extract_funcs(deri))
all_funcs.extend(extract_funcs(rates))
all_funcs.extend(extract_funcs(assem))

header = """\"\"\"DKT18 shell unrolled scalar loops for Numba JIT.\"\"\"

import numpy as np

"""

# Add @njit(cache=True) to each function
numba_funcs = []
for f in all_funcs:
    if f.strip():
        numba_funcs.append("@njit(cache=True)\n" + f)

out_code = header + "\n\n".join(numba_funcs)

with open("pyradioss/accel/jit_kernels/shells_dkt18.py", "w") as fh:
    fh.write(out_code)

print("Created pyradioss/accel/jit_kernels/shells_dkt18.py")
