import numpy as np
import time
from pyradioss.accel.jit_kernels import scatter3_colored

n = 1549
nelem = 95
npe = 4
m = nelem * npe
target = np.zeros((n, 3))
idx = np.random.randint(0, n, size=m)
values = np.random.randn(m, 3)
color_indices = np.arange(nelem)
color_offsets = np.array([0, nelem])

print("Compiling...")
scatter3_colored(target, idx, values, color_indices, color_offsets, npe)

print("Running...")
t0 = time.time()
for _ in range(1000):
    scatter3_colored(target, idx, values, color_indices, color_offsets, npe)
t1 = time.time()

print(f"Time for 1000 calls: {t1 - t0:.4f} seconds")

print(f"Time for 1000 calls: {t1 - t0:.4f} seconds")
