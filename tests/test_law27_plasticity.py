import numpy as np
import pytest

from pyradioss.materials.law27_brittle import shell_update
from pyradioss.model.entities import Material

def test_law27_plasticity_yields_before_cracking():
    """Verify that LAW27 applies Johnson-Cook plasticity to the trial stress."""
    # A=200, B=300, n=0.5
    mat = Material(1, law=27, rho0=7.8e-9, params={
        "E": 210000.0, "nu": 0.3,
        "eps_t1": 0.1, "eps_m1": 0.2, "dmax1": 0.9, "eps_f1": 1.0,
        "eps_t2": 0.1, "eps_m2": 0.2, "dmax2": 0.9, "eps_f2": 1.0,
        "A": 200.0, "B": 300.0, "n": 0.5, "sig_max": 1000.0
    })
    
    # 1 integration point
    eps27 = np.zeros((1, 3))
    crk27 = np.zeros(1)
    ang27 = np.zeros(1)
    dmg27 = np.zeros((1, 2))
    layfail = np.ones(1)
    extra = {"eps27": eps27, "crk27": crk27, "ang27": ang27, "dmg27": dmg27, "layfail": layfail}
    
    sig = np.zeros((1, 3))
    # Apply a small strain increment that will cause yielding but NOT cracking 
    # (eps_t1 is 0.1, so we apply 0.001)
    deps = np.array([[0.001, 0.0, 0.0]])
    epsp = np.zeros(1)
    
    # Elastic prediction: sxx = E/(1-v^2) * 0.001 = 210000/0.91 * 0.001 = 230.7 MPa
    # Yield stress = 200 MPa. It should yield and scale back.
    sig, epsp = shell_update(mat, sig, deps, epsp, dt=1e-5, extra=extra)
    
    # Check von Mises is capped at yield stress
    sxx, syy, sxy = sig[0, 0], sig[0, 1], sig[0, 2]
    vm = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)
    
    # Since it yielded, epsp > 0
    print(f"DEBUG: s1_trial={sig[0,0]}, sig_eq={vm}, epsp={epsp[0]}")
    assert epsp[0] > 0.0
    
    # sy = 200 + 300 * epsp^0.5
    sy = 200.0 + 300.0 * epsp[0]**0.5
    assert vm == pytest.approx(sy, rel=5e-3)
    
    # Check it didn't crack because strain is 0.05 < 0.1
    assert crk27[0] == 0.0
