"""Unit tests for LAW24 Peric concrete scalar damage model.

Upstream Fortran reference:
- Engine constitutive routines: engine/source/materials/mat/mat024/sigeps24.F
  (m24law.F, conc24.F, elas24.F, dama24.F, plas24.F)
- Starter reader: starter/source/materials/mat/mat024/hm_read_mat24.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials import (
    Law24Params,
    build_law24,
    solid_update,
    solid_tangent,
    sound_speed,
    MATERIAL_SOLID_DISPATCH,
    LAW_DISPATCH_METADATA,
)
from pyradioss.materials.law24_concrete import (
    peric_damage_update,
    consistent_solid_tangent,
    tangent,
    solid_step,
    shell_update,
)


def test_law24_params_and_build():
    """Test Law24Params initialization and build_law24 constructor."""
    params = Law24Params(
        E=30000.0,
        nu=0.2,
        rho0=2400.0,
        fc=30.0,
        ft=3.0,
        damage_max=0.95,
        damage_model="scalar",
    )
    mat = build_law24(params)
    assert mat.law == 24
    assert mat.params["E"] == 30000.0
    assert mat.params["nu"] == 0.2
    assert mat.params["FC"] == 30.0
    assert mat.params["FT"] == 3.0
    assert mat.params["DSUP"] == 0.95
    assert mat.params["DAMAGE_MODEL"] == "scalar"
    assert math.isclose(mat.rho0, 2400.0)


def test_law24_linear_elastic_below_initiation():
    """Verify linear elastic response prior to damage initiation (r <= eps_0)."""
    E = 30000.0
    nu = 0.2
    ft = 3.0
    eps_0 = ft / E  # 1e-4
    params = Law24Params(E=E, nu=nu, fc=30.0, ft=ft, damage_model="scalar")
    mat = build_law24(params)

    # Apply small uniaxial strain below eps_0 (e.g. 0.5 * eps_0)
    nel = 1
    sig = np.zeros((nel, 6))
    deps = np.zeros((nel, 6))
    deps[0, 0] = 0.5 * eps_0
    extra = {}

    sign, epsp, c = solid_update(mat, sig, deps, dt=1e-5, extra=extra)

    # Damage must be 0
    dam_scalar = extra.get("dam24_scalar")
    assert dam_scalar is not None
    assert dam_scalar[0] == 0.0

    # Stress must match 3D isotropic elasticity: sigma_xx = (lam + 2G) * eps_xx
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    g = E / (2.0 * (1.0 + nu))
    expected_sxx = (lam + 2.0 * g) * deps[0, 0]
    expected_syy = lam * deps[0, 0]

    assert math.isclose(sign[0, 0], expected_sxx, rel_tol=1e-4)
    assert math.isclose(sign[0, 1], expected_syy, rel_tol=1e-4)
    assert c is not None and c[0] > 0.0


def test_law24_scalar_damage_evolution():
    """Verify scalar damage growth when equivalent tensile strain exceeds eps_0."""
    E = 30000.0
    nu = 0.2
    ft = 3.0
    eps_0 = ft / E  # 1e-4
    params = Law24Params(E=E, nu=nu, fc=30.0, ft=ft, damage_model="scalar")
    mat = build_law24(params)

    # Strain exceeding threshold
    nel = 1
    sig = np.zeros((nel, 6))
    deps = np.zeros((nel, 6))
    deps[0, 0] = 3.0 * eps_0
    extra = {}

    sign, epsp, c = solid_update(mat, sig, deps, dt=1e-5, extra=extra)

    # Damage D in (0, 1)
    D = extra["dam24_scalar"][0]
    assert 0.0 < D < 1.0

    # Effective stress: sigma = (1 - D) * C * eps
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    g = E / (2.0 * (1.0 + nu))
    expected_sxx = (1.0 - D) * (lam + 2.0 * g) * deps[0, 0]
    assert math.isclose(sign[0, 0], expected_sxx, rel_tol=1e-4)


def test_law24_unloading_damage_irreversibility():
    """Verify damage does not decrease during unloading (irreversibility of damage)."""
    E = 30000.0
    nu = 0.2
    ft = 3.0
    eps_0 = ft / E
    params = Law24Params(E=E, nu=nu, fc=30.0, ft=ft, damage_model="scalar")
    mat = build_law24(params)

    extra = {}
    sig = np.zeros((1, 6))

    # Step 1: Loading to 3 * eps_0
    deps1 = np.zeros((1, 6))
    deps1[0, 0] = 3.0 * eps_0
    sig1, _, _ = solid_update(mat, sig, deps1, dt=1e-5, extra=extra)
    D1 = float(extra["dam24_scalar"][0])
    assert D1 > 0.0

    # Step 2: Unloading by reducing strain to 1.5 * eps_0 (deps = -1.5 * eps_0)
    deps2 = np.zeros((1, 6))
    deps2[0, 0] = -1.5 * eps_0
    sig2, _, _ = solid_update(mat, sig1, deps2, dt=1e-5, extra=extra)
    D2 = float(extra["dam24_scalar"][0])

    # Damage must be unchanged
    assert math.isclose(D1, D2, rel_tol=1e-6)

    # Stress at 1.5 * eps_0 must be proportional to (1 - D1) * C * (1.5 * eps_0)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    g = E / (2.0 * (1.0 + nu))
    expected_sxx = (1.0 - D2) * (lam + 2.0 * g) * (1.5 * eps_0)
    assert math.isclose(sig2[0, 0], expected_sxx, rel_tol=1e-4)


def test_law24_tangent_and_sound_speed():
    """Verify consistent tangent degradation by (1 - D) and sound speed calculation."""
    E = 30000.0
    nu = 0.2
    params = Law24Params(E=E, nu=nu, fc=30.0, ft=3.0, damage_model="scalar")
    mat = build_law24(params)

    # Intact tangent
    D0 = solid_tangent(mat)
    assert D0.shape == (1, 6, 6)
    assert D0[0, 0, 0] > 0.0

    # Tangent with damage
    extra = {"dam24_scalar": np.array([0.5]), "scalar_damage": True}
    D_dam = solid_tangent(mat, extra=extra)
    np.testing.assert_allclose(D_dam[0], 0.5 * D0[0], rtol=1e-5)

    # Verify sound speed
    c = sound_speed(mat)
    assert c > 0.0


def test_law24_dispatcher_and_shell_rejection():
    """Verify registration in dispatchers and rejection of shell elements."""
    assert 24 in MATERIAL_SOLID_DISPATCH
    assert "LAW24" in MATERIAL_SOLID_DISPATCH
    assert "CONC" in MATERIAL_SOLID_DISPATCH

    assert 24 in LAW_DISPATCH_METADATA
    assert LAW_DISPATCH_METADATA[24]["solid"] is True
    assert LAW_DISPATCH_METADATA[24]["shell"] is False

    p = Law24Params()
    mat = build_law24(p)
    with pytest.raises(NotImplementedError, match="solid elements only"):
        shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)))
