import time
import numpy as np
from pyradioss.model import Material
from pyradioss.materials.law02_johnson_cook import shell_update

mat = Material(1, law=2, rho0=7.8e-9, params={
    "E": 210000.0, "nu": 0.3,
    "A": 200.0, "B": 300.0, "n": 0.5, "sig_max": 1000.0
})

N = 800
sig = np.zeros((N, 3))
deps = np.zeros((N, 3))
deps[:, 0] = 0.001
epsp = np.zeros(N)

# Warmup
shell_update(mat, sig, deps, epsp, dt=1e-5)

t0 = time.time()
for _ in range(100):
    sig, epsp = shell_update(mat, sig, deps, epsp, dt=1e-5)
t1 = time.time()
print(f"Time for 100 calls of law02: {(t1 - t0)*1000:.2f} ms")
