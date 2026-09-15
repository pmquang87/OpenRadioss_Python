import numpy as np
import time
from pyradioss.materials.law27_brittle import shell_update
from pyradioss.model.entities import Material
import os
os.environ["PYRADIOSS_BACKEND"] = "numba"

print("Compiling shell_update...")
mat = Material(1, law=27, rho0=7.8e-9, params={
    "E": 210000.0, "nu": 0.3,
    "eps_t1": 0.1, "eps_m1": 0.2, "dmax1": 0.9, "eps_f1": 1.0,
    "eps_t2": 0.1, "eps_m2": 0.2, "dmax2": 0.9, "eps_f2": 1.0,
    "A": 200.0, "B": 300.0, "n": 0.5, "sig_max": 1000.0
})

from numba.typed import Dict
from numba import types

d = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
for k, v in mat.params.items():
    d[k] = v
mat.params = d

n = 95
eps27 = np.zeros((n, 3))
crk27 = np.zeros(n)
ang27 = np.zeros(n)
dmg27 = np.zeros((n, 2))
layfail = np.ones(n)
extra = {"eps27": eps27, "crk27": crk27, "ang27": ang27, "dmg27": dmg27, "layfail": layfail}

sig = np.zeros((n, 3))
deps = np.zeros((n, 3))
epsp = np.zeros(n)

t0 = time.time()
sig, epsp = shell_update(mat, sig, deps, epsp, dt=1e-5, extra=extra)
t1 = time.time()

print(f"Compilation took {t1 - t0:.2f} seconds!")
