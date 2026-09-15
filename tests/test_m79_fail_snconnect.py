"""M79: FAIL/SNCONNECT failure model.

Validates the FAIL/SNCONNECT failure model against the reference OpenRadioss Fortran solver logic.
"""

import numpy as np
import pytest
from pyradioss.model.entities import FailureModel
from pyradioss.failure.snconnect import solid_step

def test_snconnect_solid_step():
    # Mock parameters
    params = {
        "a2": 0.0,
        "b2": 1.0,
        "a3": 0.0,
        "b3": 1.0,
        "isym": 0,
        "xscale0": 1.0,
        "xscalef": 1.0
    }
    fail = FailureModel(type="SNCONNECT", params=params)
    
    # 2 elements
    dama = np.zeros(2, dtype=np.float64)
    # create a dummy base array to simulate engine behavior
    dama_base = np.zeros(10, dtype=np.float64)
    dama = dama_base[5:7]
    
    dt = 1e-4
    
    # Cycle 1: epsp = 0.5 (d_epsp = 0.5)
    d_epsp = np.array([0.5, 0.5])
    # sig: [xx, yy, zz, xy, yz, zx]
    # signzz = sig[:, 2], signyz = sig[:, 4], signzx = sig[:, 5]
    # Let's apply pure normal stress to element 0, and pure shear stress to element 1
    sig = np.zeros((2, 6))
    sig[0, 2] = 100.0 # pure normal
    sig[1, 4] = 100.0 # pure shear
    
    # Since a2=0, b2=1, fun2n=1, phi for el 0 is pi/2, so sphi=1, cphi=0
    # t1 = 1, t2 = 0
    # TTN = 1 * epsp = 0.5, TTS = 0
    # fct = 0.5 (no damage start)
    
    # For el 1, phi=0, sphi=0, cphi=1
    # t1 = 0, t2 = 1
    # TTN = 0, TTS = 1 * epsp = 0.5
    # fct = 0.5 (no damage start)
    
    broken = solid_step(fail, sig, d_epsp, None, dt, dama)
    assert not np.any(broken)
    assert np.allclose(dama, 0.0)
    
    # Cycle 2: d_epsp = 0.6 (total epsp = 1.1)
    d_epsp = np.array([0.6, 0.6])
    broken = solid_step(fail, sig, d_epsp, None, dt, dama)
    
    # fct = 1.1 > 1.0 -> damage starts!
    # pla1 freezes at 1.0
    # new pla2 is computed. Since a3=0, b3=1, t1_3=1, pla2 = 1.0
    # d_val = (epsp - pla1) / max(tiny, pla2 - pla1) -> (1.1 - 1.0) / tiny -> > 1.0 -> 1.0
    assert np.all(broken)
    assert np.allclose(dama, 1.0)
