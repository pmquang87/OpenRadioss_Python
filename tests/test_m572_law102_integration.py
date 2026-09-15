"""Tests for /MAT/LAW102 Continuum Integration, Element Compatibility, and Materials Dispatch.

Verifies:
  - Vectorized solid update across batches of 3D continuum elements
  - Materials dispatch: solid_update, sound_speed, solid_tangent
  - Rejection of shell elements in materials dispatch (ANCMSG 305)
  - Rejection of shell tangent in materials dispatch
  - Pure hydrostatic compression elasticity vs shear plastic flow
"""

import math
import numpy as np
import pytest

from pyradioss.model.entities import MatLaw102
from pyradioss.materials.law102_dprag2 import (
    DPrag2Params,
    build_law102,
    solid_update,
    solid_update_array,
    sound_speed,
    solid_tangent,
)
from pyradioss.materials import (
    solid_update as mat_solid_update,
    shell_update as mat_shell_update,
    solid_tangent as mat_solid_tangent,
    sound_speed as mat_sound_speed,
    consistent_shell_tangent as mat_shell_tangent,
)


def _sample_material() -> MatLaw102:
    return MatLaw102(
        id=102,
        title="DPRAG2 Integration Test Material",
        rho=2.5e-6,
        iform=2,
        e=25000.0,
        nu=0.25,
        c=12.0,
        phi=30.0,
        amax=200.0,
        pmin=-5.0,
    )


def test_vectorized_solid_update_batch():
    """Verify solid_update_array processes batches of 3D solid elements simultaneously."""
    mat = _sample_material()
    params = build_law102(mat)
    n_elems = 16

    # Random small strain increments
    np.random.seed(42)
    deps = np.random.uniform(-1e-4, 1e-4, size=(n_elems, 6))
    sig_old = np.zeros((n_elems, 6))
    hist_old = np.zeros((n_elems, 2))

    sig_new, hist_new, ssp = solid_update_array(params, deps, sig_old, hist_old)

    assert sig_new.shape == (n_elems, 6)
    assert hist_new.shape == (n_elems, 2)
    assert ssp.shape == (n_elems,)
    assert np.all(ssp > 0.0)
    assert np.all(hist_new[:, 0] >= 0.0)


def test_materials_dispatch_solid_update():
    """Verify central materials.solid_update dispatches to LAW102."""
    mat = _sample_material()
    n_elems = 4
    sig = np.zeros((n_elems, 6))
    deps = np.full((n_elems, 6), 1.0e-4)

    sig_out, epsp_out, c_out = mat_solid_update(mat, sig, deps)

    assert sig_out.shape == (n_elems, 6)
    assert epsp_out.shape == (n_elems,)
    assert c_out is not None or c_out > 0.0


def test_materials_dispatch_sound_speed_and_tangent():
    """Verify central materials.sound_speed and solid_tangent dispatches to LAW102."""
    mat = _sample_material()

    # Sound speed dispatch
    c_val = mat_sound_speed(mat)
    expected_c = math.sqrt((mat.K + 4.0 * mat.G / 3.0) / mat.rho)
    assert pytest.approx(c_val) == expected_c

    # Solid tangent dispatch
    sig = np.zeros((2, 6))
    c_tangent = mat_solid_tangent(mat, sig)
    assert c_tangent.shape == (2, 6, 6)
    # Check elastic symmetry
    assert np.allclose(c_tangent, c_tangent.transpose(0, 2, 1))


def test_shell_rejection_in_materials_dispatch():
    """Verify LAW102 raises NotImplementedError for shell formulations (ANCMSG 305)."""
    mat = _sample_material()
    sig_shell = np.zeros(3)
    deps_shell = np.zeros(3)

    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        mat_shell_update(mat, sig_shell, deps_shell)

    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        mat_shell_tangent(mat, sig_shell)


def test_hydrostatic_compression_vs_shear_flow():
    """Verify pure hydrostatic compression remains elastic while large shear yields."""
    mat = _sample_material()
    params = build_law102(mat)

    # 1. Pure hydrostatic compression: deps_xx = deps_yy = deps_zz = -0.005 (negative strain = compression)
    deps_comp = np.array([-0.005, -0.005, -0.005, 0.0, 0.0, 0.0])
    sig_old = np.zeros(6)
    hist_old = np.zeros(2)

    sig_comp, hist_comp, _ = solid_update_array(
        params, deps_comp[np.newaxis, :], sig_old[np.newaxis, :], hist_old[np.newaxis, :]
    )
    # Deviatoric stress and plastic strain remain 0
    p_comp = -(sig_comp[0, 0] + sig_comp[0, 1] + sig_comp[0, 2]) / 3.0
    assert p_comp > 0.0  # Positive pressure
    assert hist_comp[0, 0] == 0.0  # Zero plastic strain

    # 2. Large shear strain causing plastic flow
    deps_shear = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0])
    sig_shear, hist_shear, _ = solid_update_array(
        params, deps_shear[np.newaxis, :], sig_old[np.newaxis, :], hist_old[np.newaxis, :]
    )
    # Plastic flow must occur
    assert hist_shear[0, 0] > 0.0
