"""Unit tests for OpenRadioss LAW14 (Composite Solid).

References:
  - starter/source/materials/mat/mat014/hm_read_mat14.F:151-158
  - engine/source/materials/mat/mat014/m14law.F (solid 3D formulation)
  - engine/source/materials/mat/mat014/m14ama.F, m14gtf.F, m14ftg.F
  - engine/source/materials/mat_share/mulawc.F90:1125-1307 (no LAW14 shell kernel)
"""

import numpy as np
import pytest

from pyradioss.materials import law14_compso


@pytest.fixture
def standard_law14_mat():
    """Fixture returning standard orthotropic composite properties."""
    return law14_compso.build_law14(
        E11=140000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.3,
        nu23=0.35,
        nu31=0.02,
        G12=5000.0,
        G23=3000.0,
        G31=4500.0,
        rho0=1.6e-9,
        sigt1=1500.0,
        sigt2=50.0,
        delta=0.05,
        sigyt1=1200.0,
        sigyc1=1000.0,
        sigyt2=40.0,
        sigyc2=150.0,
        sigyt12=70.0,
        sigyc12=70.0,
        cb=200.0,
        cn=0.5,
        fmax=2000.0,
        wplaref=1.0,
    )


def test_law14_tangent_dispatcher(standard_law14_mat):
    """Verify tangent(group) dispatcher for solids and rejected shells."""
    mat = standard_law14_mat

    class ShellGroup:
        def __init__(self, m):
            self.mat = m
            self.elem_type = "shell"

    class SolidGroup:
        def __init__(self, m):
            self.mat = m
            self.elem_type = "solid"

    sh_grp = ShellGroup(mat)
    T_sh = law14_compso.tangent(sh_grp)
    assert T_sh is None

    so_grp = SolidGroup(mat)
    T_so = law14_compso.tangent(so_grp)
    assert T_so is not None
    assert T_so.shape == (6, 6)


def test_law14_solid_update_regression(standard_law14_mat):
    """Verify that existing 3D solid element update functionality is fully preserved."""
    mat = standard_law14_mat

    sig6 = np.zeros(6)
    deps6 = np.array([1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_out, ep_out, ssp = law14_compso.solid_update(mat, sig6, deps6)

    assert sig_out.shape == (6,)
    assert sig_out[0] == pytest.approx(mat.params["D11"] * 1.0e-3, rel=1e-4)
    assert ssp > 0.0
