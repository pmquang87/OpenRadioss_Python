import numpy as np
import pytest

from pyradioss.materials.law24_concrete import solid_update
from pyradioss.model.entities import Material

def test_law24_steel_reinforcement():
    """Verify that LAW24 adds 1D steel reinforcement via Rule of Mixtures."""
    # concrete params
    params = {
        "MAT_E": 30000.0, "MAT_NU": 0.2, "RHO0": 2e-9, "Iflag": 0,
        "MAT_SIGY": 30.0, "MAT_FtFc": 3.0, "MAT_FbFc": 36.0, "MAT_F2Fc": 120.0, "MAT_SoFc": 1.25,
        "MAT_ETAN": -30000.0, "MAT_DAMAGE": 0.99, "MAT_EPS": 1e20, "MAT_BETA": 0.5,
        "MAT_PPRES": 10.0, "MAT_YPRES": -10.0, "MAT_BPMOD": 0.0, "MAT_ETC": 0.0,
        "MAT_DIL_Y": -0.2, "MAT_DIL_F": -0.1, "MAT_COMPAC": -0.35,
        "MAT_CAP_BEG": -10.0, "MAT_CAP_END": -24.0, "MAT_TPMOD": 6000.0,
        
        # steel params
        "MAT_E2": 210000.0, "MAT_SSIG": 500.0, "MAT_SETAN": 2100.0,
        "MAT_PDIR1": 0.1, "MAT_PDIR2": 0.0, "MAT_PDIR3": 0.0
    }
    
    # We must build it properly to get all the derived constants
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    from types import SimpleNamespace
    rec = SimpleNamespace(id=1, title="mat", density=2e-9, params=params)
    mat = MAT_PHYSICS_REGISTRY["LAW24"](rec)
    
    # states
    nip = 1
    sig = np.zeros((nip, 6))
    deps = np.zeros((nip, 6))
    deps[0, 0] = 0.01  # 1% strain in x
    
    extra = {
        "strain24": np.zeros((nip, 6)),
        "sigc24": np.zeros((nip, 6)),
        "crak24": np.zeros((nip, 3)),
        "dam24": np.zeros((nip, 3)),
        "ang24": np.zeros((nip, 6)),
        "epsf24": np.full((nip, 3), -1.0),
        "siga24": np.zeros((nip, 3)),
        "epsa24": np.zeros((nip, 3)),
        "vk024": np.zeros(nip),
        "vk24": np.zeros(nip),
        "rob24": np.zeros(nip),
        "off24": np.ones(nip),
        "ini24": np.zeros(nip)
    }
    
    sig_out, epsp_out, c_out = solid_update(mat, sig, deps, None, dt=1e-5, extra=extra)
    
    # Concrete is likely cracked/yielded, but steel should yield at 500 MPa.
    # Trial stress in steel: 210000 * 0.01 = 2100 MPa > 500 MPa
    # Yield stress with hardening: H = 210000*2100/(210000-2100) = 2121.2
    # dp = (2100 - 500) / (210000 + 2121.2) = 1600 / 212121.2 = 0.00754
    # Final steel stress = 500 + 2121.2 * 0.00754 = 516 MPa
    
    siga = extra["siga24"][0, 0]
    epsa = extra["epsa24"][0, 0]
    
    assert epsa > 0.0
    assert siga == pytest.approx(516.0, rel=1e-2)
    
    # Global stress in x should be sig_c * 0.9 + 516.0 * 0.1
    assert sig_out[0, 0] == pytest.approx(extra["sigc24"][0, 0] * 0.9 + siga * 0.1, rel=1e-5)
