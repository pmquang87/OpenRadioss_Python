"""Unit tests for LAW37 biphasic liquid-gas material model.

Upstream Fortran reference:
- Constitutive Engine Kernel: engine/source/materials/mat/mat037/sigeps37.F
- Starter Card Reader: starter/source/materials/mat/mat037/hm_read_mat37.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials import (
    Law37Params,
    build_law37,
    solid_update,
    solid_tangent,
    sound_speed,
    MATERIAL_SOLID_DISPATCH,
    LAW_DISPATCH_METADATA,
)
from pyradioss.materials.law37_biphas import (
    init_uv37,
    consistent_solid_tangent,
    tangent,
    solid_step,
)


def test_law37_params_and_build():
    """Test Law37Params and build_law37 default initialization and attributes."""
    params = Law37Params(
        rho_l0=1000.0,
        c_l=2.2e9,
        alpha1=0.8,
        rho_g0=1.2,
        gamma=1.4,
        p0=1.01325e5,
        isolver=1,
    )
    mat = build_law37(params)
    assert mat.law == 37
    assert mat.params["c_l"] == 2.2e9
    assert mat.params["rho_l0"] == 1000.0
    assert mat.params["rho_g0"] == 1.2
    assert mat.params["alpha1"] == 0.8
    assert mat.params["gamma"] == 1.4
    assert mat.params["isolver"] == 1
    # Check expected rho0 = alpha1*rho_l0 + (1-alpha1)*rho_g0
    expected_rho0 = 0.8 * 1000.0 + 0.2 * 1.2
    assert math.isclose(mat.rho0, expected_rho0, rel_tol=1e-6)


def test_law37_init_uv37():
    """Test initial state variables uv37 array generation (sigeps37.F lines 179-197)."""
    p = Law37Params(alpha1=0.5, rho_l0=1000.0, rho_g0=1.0)
    mat = build_law37(p)
    nel = 4
    rho = np.full(nel, mat.rho0)
    sig = np.zeros((nel, 6))

    uv = init_uv37(mat, nel=nel, rho=rho, sig=sig, pshift=-1.01325e5)
    assert uv.shape == (nel, 5)
    # Volumetric fraction of liquid + gas = 1.0
    np.testing.assert_allclose(uv[:, 3] + uv[:, 4], 1.0, rtol=1e-5)
    # Liquid density and gas density are positive
    assert np.all(uv[:, 1] > 0.0)
    assert np.all(uv[:, 2] > 0.0)


def test_law37_solid_update_isolver1():
    """Test constitutive update with legacy solver (ISOLVER=1, sigeps37.F lines 340-392)."""
    p = Law37Params(alpha1=0.9, isolver=1, nu_l=1e-3, nu_g=1e-5)
    mat = build_law37(p)
    nel = 2
    sig = np.zeros((nel, 6))
    deps = np.full((nel, 6), 1e-4)
    dt = 1e-6
    extra = {"rho": np.full(nel, mat.rho0 * 1.001)}

    sign, epsp, c = solid_update(mat, sig, deps, dt=dt, extra=extra)
    assert sign.shape == (nel, 6)
    assert c is not None
    assert np.all(c > 0.0)
    # Compressive volumetric strain should produce negative normal stresses (-pressure)
    assert np.all(sign[:, 0] < 0.0)
    assert np.all(sign[:, 1] < 0.0)
    assert np.all(sign[:, 2] < 0.0)


def test_law37_solid_update_isolver2_newton():
    """Test constitutive update with 2D Newton solver (ISOLVER=2, sigeps37.F lines 234-336)."""
    p = Law37Params(alpha1=0.7, isolver=2, nu_l=1e-3, nu_g=1e-5)
    mat = build_law37(p)
    nel = 3
    sig = np.zeros((nel, 6))
    deps = np.zeros((nel, 6))
    deps[:, 0] = -1e-4  # compression
    deps[:, 1] = -1e-4
    deps[:, 2] = -1e-4
    dt = 1e-6
    extra = {"rho": np.full(nel, mat.rho0 * 1.002)}

    sign, epsp, c = solid_update(mat, sig, deps, dt=dt, extra=extra)
    assert sign.shape == (nel, 6)
    assert c is not None
    assert np.all(c > 0.0)
    # Verify uv37 history was updated
    assert "uv37" in extra
    uv37 = extra["uv37"]
    assert uv37.shape == (nel, 5)
    # Check volume fractions sum to 1.0
    np.testing.assert_allclose(uv37[:, 3] + uv37[:, 4], 1.0, rtol=1e-5)


def test_law37_tangent_and_consistent_stiffness():
    """Test consistent algorithmic tangent matrix (6x6)."""
    p = Law37Params(alpha1=0.6, nu_l=1e-3, nu_vol_l=2e-3)
    mat = build_law37(p)
    extra = {"rho": np.full(1, mat.rho0), "dt": 1e-6}

    D = solid_tangent(mat, extra=extra)
    assert D.shape == (1, 6, 6)
    # Bulk components must be positive
    assert D[0, 0, 0] > 0.0
    assert D[0, 1, 1] > 0.0
    assert D[0, 2, 2] > 0.0
    # Shear components must be non-negative
    assert D[0, 3, 3] >= 0.0
    assert D[0, 4, 4] >= 0.0
    assert D[0, 5, 5] >= 0.0

    # Also test generic template interface tangent()
    D_tmpl = tangent(mat, extra=extra)
    np.testing.assert_allclose(D, D_tmpl)


def test_law37_dispatcher_registration():
    """Verify registration in MATERIAL_SOLID_DISPATCH and LAW_DISPATCH_METADATA."""
    assert 37 in MATERIAL_SOLID_DISPATCH
    assert "LAW37" in MATERIAL_SOLID_DISPATCH
    assert "BIPHAS" in MATERIAL_SOLID_DISPATCH
    assert "BIPHASIC" in MATERIAL_SOLID_DISPATCH

    assert 37 in LAW_DISPATCH_METADATA
    assert LAW_DISPATCH_METADATA[37]["solid"] is True
    assert LAW_DISPATCH_METADATA[37]["shell"] is False
    assert LAW_DISPATCH_METADATA[37]["plane_stress"] is False


def test_law37_shell_rejection():
    """LAW37 is formulated for 3D solids/SPH only and must reject shells."""
    p = Law37Params()
    mat = build_law37(p)
    from pyradioss.materials import shell_update

    with pytest.raises(NotImplementedError, match="3D solid and SPH"):
        shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)))
