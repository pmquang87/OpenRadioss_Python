import time
import numpy as np
from pyradioss.model import Material
from pyradioss.materials.law27_brittle import shell_update

mat = Material(1, law=27, rho0=7.8e-9, params={
    "E": 210000.0, "nu": 0.3,
    "eps_t1": 0.1, "eps_m1": 0.2, "dmax1": 0.9, "eps_f1": 1.0,
    "eps_t2": 0.1, "eps_m2": 0.2, "dmax2": 0.9, "eps_f2": 1.0,
    "A": 200.0, "B": 300.0, "n": 0.5, "sig_max": 1000.0
})

N = 800
sig = np.zeros((N, 3))
deps = np.zeros((N, 3))
deps[:, 0] = 0.001
epsp = np.zeros(N)
extra = {
    "eps27": np.zeros((N, 3)),
    "crk27": np.zeros((N, 2)),
    "ang27": np.zeros(N),
    "dmg27": np.zeros((N, 2)),
    "layfail": np.ones(N)
}

# Warmup
shell_update(mat, sig, deps, epsp, dt=1e-5, extra=extra)

t0 = time.time()
for _ in range(100):
    sig, epsp = shell_update(mat, sig, deps, epsp, dt=1e-5, extra=extra)
t1 = time.time()
print(f"Time for 100 calls: {(t1 - t0)*1000:.2f} ms")
