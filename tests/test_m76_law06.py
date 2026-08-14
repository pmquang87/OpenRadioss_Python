import numpy as np
import pytest

from pyradioss.materials.law06_hyd_visc import solid_update
from pyradioss.model.entities import Material

def test_law06_viscous_stress():
    """Verify the Newtonian viscous stress deviator calculation."""
    # Create a mock material
    mat = Material(id=1, law=6, rho0=1000.0, title="Water", params={"visc": 1.0})
    
    # Strain increment over 0.1s
    # D1, D2, D3 = 0.1, 0.2, -0.3 (volumetric rate is 0)
    # D4, D5, D6 = 0.1, -0.1, 0.0
    deps = np.array([
        [0.1, 0.2, -0.3, 0.1, -0.1, 0.0]
    ])
    dt = 0.1
    
    # extra dict provides current density
    extra = {"rho": np.array([1200.0])}
    
    # deps_rate will be deps / dt -> [1.0, 2.0, -3.0, 1.0, -1.0, 0.0]
    # volumetric strain rate dav = -(1.0 + 2.0 - 3.0)/3.0 = 0.0
    
    # Expected viscosity: visc = mat.params["visc"] * rho = 1.0 * 1200.0 = 1200.0
    # vis2 = 2.0 * visc = 2400.0
    
    # sig1 = vis2 * (1.0 + 0.0) = 2400.0
    # sig2 = vis2 * (2.0 + 0.0) = 4800.0
    # sig3 = vis2 * (-3.0 + 0.0) = -7200.0
    # sig4 = visc * 1.0 = 1200.0
    # sig5 = visc * -1.0 = -1200.0
    # sig6 = visc * 0.0 = 0.0
    
    sig = np.zeros((1, 6))
    epsp = None
    
    sig_out, epsp_out, c_out = solid_update(mat, sig, deps, epsp, dt, extra)
    
    expected_sig = np.array([
        [2400.0, 4800.0, -7200.0, 1200.0, -1200.0, 0.0]
    ])
    
    np.testing.assert_allclose(sig_out, expected_sig)
    assert epsp_out is None
    assert c_out is None
