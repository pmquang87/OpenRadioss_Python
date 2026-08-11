import numpy as np
import pytest

from pyradioss.model.model import Model, ElementGroup
from pyradioss.model.entities import NodeGroup
from pyradioss.elements.solid_tetra4 import init_group, pre_forces, forces

def test_itetra3_sfem_smoothing():
    """Verify that Itetra=3 (SFEM) properly smooths volumes across elements."""
    model = Model()
    
    # We will create two tetrahedra sharing a single triangular face.
    # The shared face is at z=0 (nodes 0, 1, 2).
    # Element 1 has its 4th node at z = 1 (node 3).
    # Element 2 has its 4th node at z = -1 (node 4).
    # Total volume is constant if we push node 3 down by dz and node 4 down by dz,
    # i.e., compressing Element 1 and expanding Element 2 exactly the same amount.
    model.x0 = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        [0.0, 0.0,-1.0]   # 4
    ])
    
    # Both use the same property with itetra4=3
    class MockProp:
        def __init__(self):
            self.params = {"itetra4": 3, "qa": 1.0, "qb": 1.0}
            
    class MockMat:
        def __init__(self):
            self.law = 1
            self.fail = None
            self.eos = None
            self.radioss_mat_id = 1
            self.params = {}
            self.rho0 = 1.0
            self.E = 1.0
            self.G = 1.0
            self.K = 1.0
            def sound_speed_solid(self):
                return 1.0
            self.sound_speed_solid = sound_speed_solid.__get__(self, MockMat)
            
    prop = MockProp()
    mat = MockMat()
    
    # Node winding must be positive V_std (since V_std = -V_Radioss, and Radioss takes positive).
    # We'll just provide nodes.
    conn = np.array([
        [0, 1, 2, 3],
        [0, 2, 1, 4]  # reversed base to have positive volume
    ])
    
    class DummyGroup:
        def __init__(self, conn):
            self.conn = conn
            self.n = len(conn)
            self.ids = np.array([1, 2])
            self.state = {
                "slices": [(slice(0, 2), mat, prop)],
                "mat_extra": {},
                "chk_fail": False
            }
            
    group = DummyGroup(conn)
    
    # Initialize group, which should populate model.nodal_vol_0
    init_group(group, model, log=None)
    
    assert hasattr(model, "nodal_vol_0")
    
    # V0 of each element should be 1/6
    # Let's verify model.nodal_vol_0
    v0_elem = 1.0 / 6.0
    # Node 0, 1, 2 are shared -> volume should be 2 * (1/6) = 1/3
    # Node 3, 4 are unique -> volume should be 1/6
    assert np.allclose(model.nodal_vol_0[0], 2.0 / 6.0)
    assert np.allclose(model.nodal_vol_0[1], 2.0 / 6.0)
    assert np.allclose(model.nodal_vol_0[2], 2.0 / 6.0)
    assert np.allclose(model.nodal_vol_0[3], 1.0 / 6.0)
    assert np.allclose(model.nodal_vol_0[4], 1.0 / 6.0)
    
    # Cycle 1: Displace nodes to compress elem 1 and expand elem 2
    # Node 3 moves down (compression)
    # Node 4 moves down (expansion)
    x = model.x0.copy()
    x[3, 2] = 0.5   # V1 is now 1/12
    x[4, 2] = -1.5  # V2 is now 3/12
    
    # Nodal total volume for shared nodes should be exactly the same!
    # V1 + V2 = 4/12 = 1/3.
    
    model.nodal_vol_t[:] = 0.0
    pre_forces(group, model, x, dt=0.0)
    
    # Shared nodes should have exactly V=1/3 still!
    assert np.allclose(model.nodal_vol_t[0], 2.0 / 6.0)
    
    # Node 3 volume is now 1/12
    assert np.allclose(model.nodal_vol_t[3], 1.0 / 12.0)
    
    # Node 4 volume is now 3/12
    assert np.allclose(model.nodal_vol_t[4], 3.0 / 12.0)
    
    # Run forces
    v = np.zeros_like(x)
    vr = np.zeros_like(x)
    fint = np.zeros_like(x)
    mint = np.zeros_like(x)
    
    deps_passed = []
    def mock_solid_update(mat, sig, deps, epsp, dt, extra):
        deps_passed.append(deps.copy())
        sig[:, :] = 1.0
        return sig, epsp, np.zeros(len(sig))
    
    import pyradioss.materials as materials
    
    # Store original and mock
    orig_solid_update = materials.solid_update
    materials.solid_update = mock_solid_update
    
    try:
        # Needs some dummy arrays
        group.state["sig"] = np.zeros((2, 6))
        group.state["epsp"] = np.zeros(2)
        group.state["off"] = np.ones(2)
        group.state["mass"] = np.ones(2)
        group.state["vol0"] = np.array([1.0/6.0, 1.0/6.0])
        
        forces(group, x, v, vr, dt=1e-4, fint=fint, mint=mint)
    finally:
        materials.solid_update = orig_solid_update
    
    # Check that tr(deps) is corrected
    # The volumetric strain correction `corr` is applied to deps
    tr_deps_1 = deps_passed[0][0, 0] + deps_passed[0][0, 1] + deps_passed[0][0, 2]
    tr_deps_2 = deps_passed[0][1, 0] + deps_passed[0][1, 1] + deps_passed[0][1, 2]
    
    amu_1 = 1.0 / 0.875 - 1.0
    amu_2 = 1.0 / 1.125 - 1.0
    # Since amu0 = 0, divde = -amu
    assert np.allclose(tr_deps_1, -amu_1)
    assert np.allclose(tr_deps_2, -amu_2)
