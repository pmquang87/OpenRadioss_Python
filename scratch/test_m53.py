import numpy as np
import pytest
from pyradioss.model.model import Model
from pyradioss.model.entities import MonitoredVolume, Surface
from pyradioss.materials.mat_gas import GasMaterial
from pyradioss.engine.airbag import update_airbag_volume, apply_airbag_forces, update_airbag_thermodynamics

def test_airbag_structural_coupling():
    model = Model()
    
    # 8 nodes of a unit box [0, 1]^3
    # N0: 0,0,0
    # N1: 1,0,0
    # N2: 1,1,0
    # N3: 0,1,0
    # N4: 0,0,1
    # N5: 1,0,1
    # N6: 1,1,1
    # N7: 0,1,1
    model.x0 = np.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1]
    ], dtype=float)
    model.x = model.x0.copy()
    
    # Surface with 6 quad segments (normals pointing OUT)
    surf = Surface(id=1, title="Box")
    
    # 0-based indices for segments
    # Bottom: 0-3-2-1 (normal -Z)
    # Top: 4-5-6-7 (normal +Z)
    # Front: 0-1-5-4 (normal -Y)
    # Back: 2-3-7-6 (normal +Y)
    # Left: 3-0-4-7 (normal -X)
    # Right: 1-2-6-5 (normal +X)
    surf.segments = np.array([
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
        [1, 2, 6, 5]
    ], dtype=int)
    
    model.surfaces[1] = surf
    
    cp = 1004.0; cv = 717.0
    r_spec = cp - cv
    mat = GasMaterial(id=1, law=999, rho0=1.0, params={'CPA': cp, 'R_igc': r_spec, 'MW': 1.0})
    model.materials[1] = mat
    
    mv = MonitoredVolume(
        id=1, matid=1, pext=1e5, t_initial=300.0, iequil=1, surf_id=1
    )
    model.monitored_volumes[1] = mv
    
    # 1. Init volume
    update_airbag_volume(mv, model, model.x)
    assert np.isclose(mv.volume, 1.0)
    
    # 2. Init thermo
    update_airbag_thermodynamics(mv, model, 0.0, 0.0)
    assert np.isclose(mv.pressure, 1e5)
    
    # 3. Apply forces
    fext = np.zeros((8, 3))
    # Let's increase pressure manually to create expansion
    mv.pressure = 2e5
    apply_airbag_forces(mv, model, model.x, fext)
    
    # p_eff = 2e5 - 1e5 = 1e5
    # Force on top face (Z normal): Area = 1.0, Total force = 1e5 in +Z.
    # Distributed to nodes 4,5,6,7 -> 25000 each in Z
    assert np.isclose(fext[4, 2], 25000.0)
    assert np.isclose(fext[5, 2], 25000.0)
    
    # Node 4 gets forces from Top, Front, Left.
    # Top normal: +Z. Front normal: -Y. Left normal: -X.
    # Total fext[4] should be [-25000, -25000, 25000]
    assert np.allclose(fext[4], [-25000.0, -25000.0, 25000.0])

if __name__ == '__main__':
    test_airbag_structural_coupling()
    print("Passed M53 unit test")
