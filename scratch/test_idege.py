import numpy as np

def compute_idege(conn):
    eq = conn[:, :, None] == conn[:, None, :]
    eq[:, np.arange(8), np.arange(8)] = False
    has_dup = eq.any(axis=2)
    idege = has_dup.sum(axis=1) // 2
    return idege

# Hex: no duplicates
hex = [1, 2, 3, 4, 5, 6, 7, 8]
# Wedge (2 nodes duplicated once)
wedge = [1, 2, 3, 4, 5, 6, 6, 5]
# Pyramid (1 node duplicated 4 times)
pyr = [1, 2, 3, 4, 5, 5, 5, 5]
# Tetra (1 node duplicated twice, 1 node duplicated 4 times)
tet = [1, 2, 3, 3, 4, 4, 4, 4]

conn = np.array([hex, wedge, pyr, tet])
idege = compute_idege(conn)
print("IDEGE:", idege)

# FAC logic
fac = np.ones(4)
fac[idege > 2] = 1.0 / 9.0
fac[(idege > 1) & (idege <= 2)] = 0.25
fac_sqrt = np.sqrt(fac)
# DELTAX scales as 1 / sqrt(fac)
scale = 1.0 / fac_sqrt
print("Scale:", scale)
