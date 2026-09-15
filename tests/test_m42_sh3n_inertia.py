import numpy as np
from pyradioss.starter.starter import run_starter
import contextlib
import io
import tempfile
import os

def _starter(text):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M42_0000.rad")
    with open(sp, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(sp)

def test_sh3n_rotational_inertia():
    # A single unit-thick, right-angled triangle.
    # Nodes: (0,0,0), (10,0,0), (0,10,0) -> Area = 50
    # Thickness = 1.0, rho = 7.8e-6
    rho = 7.8e-6
    thick = 1.0
    area = 50.0
    mass = rho * area * thick
    deck = f"""#RADIOSS STARTER
/BEGIN
SH3N_INERTIA
/NODE
         1                 0.0   0.0   0.0
         2                10.0   0.0   0.0
         3                 0.0  10.0   0.0
/SH3N/1
         1         1         2         3
/PART/1
p
         1         1
/MAT/LAW1/1
s
   {rho}
     210.0       0.3
/PROP/SHELL/1
sh
       {thick}         3      0.833
/END
"""
    m = _starter(deck)
    group = m.sh3n
    dt_iner = group.state["dt_iner"][0]
    
    # Expected from c3inmas.F: INS = EM * (AREA/4.5 + THK**2/12)
    em = mass / 3.0
    expected = em * (area / 4.5 + thick**2 / 12.0)
    
    assert np.allclose(dt_iner, expected, rtol=1e-7), f"Expected {{expected}}, got {{dt_iner}}"
