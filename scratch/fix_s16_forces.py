
import re

with open("scratch/patch_shel16.py", "r") as f:
    patch_code = f.read()

# Fix mass in init_group
patch_code = patch_code.replace(
    "vol=vol,",
    "vol=vol,\n        mass=mass,"
)

with open("scratch/patch_shel16.py", "w") as f:
    f.write(patch_code)

with open("scratch/s16_forces.py", "r") as f:
    forces_code = f.read()

forces_code = forces_code.replace(
    "if mat.is_law1:\n                from pyradioss.materials.law1 import law1_solid_step\n                law1_solid_step(mat.params, sig_k, deps)\n            elif mat.is_law2:\n                from pyradioss.materials.law2 import law2_solid_step\n                epsp_k = st[\"epsp\"][sl, k]\n                law2_solid_step(mat.params, sig_k, deps, epsp_k, np.ones(n_sl))",
    "if mat.law == 1:\n                from pyradioss.materials.law01_elastic import solid_update\n                solid_update(mat, sig_k, deps)\n            elif mat.law == 2:\n                from pyradioss.materials.law02_johnson_cook import solid_update\n                epsp_k = st[\"epsp\"][sl, k]\n                extra = st.get(\"mat_extra\", {})\n                solid_update(mat, sig_k, deps, epsp_k, np.ones(n_sl), extra)"
)

forces_code = forces_code.replace(
    "from pyradioss.elements.solid_hexa8 import _scatter_add3\n    _scatter_add3(fint, conn, fint_e)",
    "from pyradioss.common.fastmath import scatter_add3\n    conn_flat = conn.reshape(-1)\n    fint_e_flat = fint_e.reshape(-1, 3)\n    valid = conn_flat >= 0\n    scatter_add3(fint, conn_flat[valid], fint_e_flat[valid])"
)

with open("scratch/s16_forces.py", "w") as f:
    f.write(forces_code)
