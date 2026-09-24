"""Unit tests for LAW78 (Composite material with progressive damage stub).

OpenRadioss Fortran reference:
  C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\materials\\mat\\mat078\\sigeps78.F
  C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\engine\\source\\materials\\mat\\mat078\\sigeps78c.F
  C:\\OpenRadioss\\source\\OpenRadioss-latest-20260520\\starter\\source\\materials\\mat\\mat078\\hm_read_mat78.F
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.materials import law78_composite_dmg
from pyradioss.materials.law78_composite_dmg import (
    Law78Params,
    build_law78,
    extra_shapes,
    init_state,
    needs_defgrad,
    resolve,
    shell_tangent,
    shell_update,
    solid_tangent,
    solid_update,
    sound_speed,
    tangent,
)


def test_law78_params_defaults():
    """Test default values of Law78Params and derived moduli."""
    p = Law78Params()
    assert p.law == 78
    assert p.law_name == "LAW78"
    assert p.young == 210000.0
    assert p.nu == 0.3
    assert p.rho0 == 7.8e-3
    assert p.yield_stress == 200.0
    assert p.bsat == 300.0
    assert p.dmax == 0.99

    # Check derived moduli
    expected_bulk = 210000.0 / (3.0 * (1.0 - 2.0 * 0.3))
    expected_g = 0.5 * 210000.0 / (1.0 + 0.3)
    assert math.isclose(p.bulk, expected_bulk, rel_tol=1e-5)
    assert math.isclose(p.g, expected_g, rel_tol=1e-5)

    expected_c_solid = math.sqrt((expected_bulk + 4.0 / 3.0 * expected_g) / 7.8e-3)
    assert math.isclose(p.c_solid, expected_c_solid, rel_tol=1e-5)

    expected_a11 = 210000.0 / (1.0 - 0.3 * 0.3)
    expected_c_shell = math.sqrt(expected_a11 / 7.8e-3)
    assert math.isclose(p.c_shell, expected_c_shell, rel_tol=1e-5)


def test_law78_build_from_dict_and_material():
    """Test build_law78 with various input dictionaries and attributes."""
    data = {
        "id": 42,
        "title": "Carbon_Epoxy_Ply",
        "MAT_RHO": 1.6e-3,
        "MAT_E": 140000.0,
        "MAT_NU": 0.28,
        "EA": 140000.0,
        "EB": 10000.0,
        "EC": 10000.0,
        "G12": 5000.0,
        "MAT_SIGY": 250.0,
        "MAT_BSAT": 400.0,
        "DMAX": 0.95,
        "DC0": 0.015,
        "DT0": 0.01,
    }
    p = build_law78(data)
    assert p.id == 42
    assert p.title == "Carbon_Epoxy_Ply"
    assert p.rho0 == 1.6e-3
    assert p.young == 140000.0
    assert p.nu == 0.28
    assert p.ea == 140000.0
    assert p.eb == 10000.0
    assert p.g12 == 5000.0
    assert p.yield_stress == 250.0
    assert p.bsat == 400.0
    assert p.dmax == 0.95
    assert p.dc0 == 0.015
    assert p.dt0 == 0.01

    # Test resolution caching
    mat = SimpleNamespace(params=data, law=78, law_name="LAW78")
    p_res1 = resolve(mat)
    p_res2 = resolve(mat)
    assert p_res1 is p_res2
    assert p_res1.young == 140000.0


def test_law78_defgrad_and_extra_shapes():
    """Verify defgrad requirement and history variable shapes."""
    assert needs_defgrad() is False

    shapes_1ip = extra_shapes(nip=1)
    assert shapes_1ip["uvar78"] == (6,)
    assert shapes_1ip["uvar"] == (6,)
    assert shapes_1ip["alpha"] == (6,)
    assert shapes_1ip["beta"] == (6,)
    assert shapes_1ip["q"] == (6,)
    assert shapes_1ip["damage"] == (6,)

    shapes_5ip = extra_shapes(nip=5)
    assert shapes_5ip["uvar78"] == (5, 6)
    assert shapes_5ip["damage"] == (5, 6)


def test_law78_init_state():
    """Verify group.state initialization matching sigeps78.F."""
    p = Law78Params(yield_stress=220.0, bsat=320.0)
    group = SimpleNamespace(nel=4)

    state = init_state(group=group, n_elements=4, nip=2, mat=p)
    assert hasattr(group, "state")
    assert group.state is state
    assert state["uvar"].shape == (4, 2, 6)
    assert state["damage"].shape == (4, 2, 6)
    assert state["alpha"].shape == (4, 2, 6)
    assert state["beta"].shape == (4, 2, 6)
    assert state["q"].shape == (4, 2, 6)

    # Initial values: UVAR[2] = B_sat - Y, UVAR[3] = Y
    np.testing.assert_allclose(state["uvar"][:, :, 2], 100.0)
    np.testing.assert_allclose(state["uvar"][:, :, 3], 220.0)
    np.testing.assert_allclose(state["damage"], 0.0)


def test_law78_sound_speed():
    """Test acoustic sound speed for solids and shells."""
    p = Law78Params(young=210000.0, nu=0.3, rho0=7.8e-3)
    c_sol = sound_speed(p, is_shell=False)
    c_sh = sound_speed(p, is_shell=True)

    assert c_sol > 0.0
    assert c_sh > 0.0
    assert math.isclose(c_sol, p.c_solid, rel_tol=1e-6)
    assert math.isclose(c_sh, p.c_shell, rel_tol=1e-6)

    # Vectorized check
    eps_batch = np.zeros((5, 6))
    c_batch = sound_speed(p, eps=eps_batch, is_shell=False)
    assert isinstance(c_batch, np.ndarray)
    assert len(c_batch) == 5
    assert np.allclose(c_batch, p.c_solid)


def test_law78_solid_update_elastic_placeholder():
    """Verify 3D solid update computes exact Hooke's law without damage degradation."""
    p = Law78Params(young=200000.0, nu=0.25, rho0=8.0e-3)
    # Bulk = 200000 / (3 * 0.5) = 133333.333
    # G = 200000 / (2 * 1.25) = 80000.0
    # Lame = 80000.0

    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.001, 0.0, 0.0, 0.002, 0.0, 0.0], dtype=np.float64)
    extra = {}

    sig_new, epsp, c = solid_update(p, sig0, deps, epsp=0.0, dt=1.0e-6, extra=extra)

    # Analytical Hooke's law:
    # sig_xx = (lambda + 2G) * deps_xx = (80000 + 160000) * 0.001 = 240.0
    # sig_yy = lambda * deps_xx = 80000 * 0.001 = 80.0
    # sig_zz = lambda * deps_xx = 80000 * 0.001 = 80.0
    # sig_xy = G * deps_xy = 80000 * 0.002 = 160.0
    assert math.isclose(sig_new[0], 240.0, rel_tol=1e-5)
    assert math.isclose(sig_new[1], 80.0, rel_tol=1e-5)
    assert math.isclose(sig_new[2], 80.0, rel_tol=1e-5)
    assert math.isclose(sig_new[3], 160.0, rel_tol=1e-5)
    assert math.isclose(sig_new[4], 0.0, abs_tol=1e-8)
    assert math.isclose(sig_new[5], 0.0, abs_tol=1e-8)

    # In stub mode: epsp remains 0 (elastic placeholder)
    assert epsp == 0.0
    assert math.isclose(c, p.c_solid, rel_tol=1e-6)

    # History variables saved in extra
    assert "uvar78" in extra
    assert "damage" in extra


def test_law78_solid_update_vectorized():
    """Verify solid_update works on batch of elements."""
    p = Law78Params(young=210000.0, nu=0.3, rho0=7.8e-3)
    nel = 3
    sig_batch = np.zeros((nel, 6), dtype=np.float64)
    deps_batch = np.array([
        [0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.002, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0015, 0.0, 0.0],
    ], dtype=np.float64)

    sig_out, epsp_out, c_out = solid_update(p, sig_batch, deps_batch)
    assert sig_out.shape == (3, 6)
    assert len(epsp_out) == 3
    assert len(c_out) == 3
    assert sig_out[0, 0] > 0.0
    assert sig_out[1, 1] > 0.0
    assert sig_out[2, 3] > 0.0


def test_law78_shell_update_elastic():
    """Verify 2D shell plane-stress update."""
    p = Law78Params(young=210000.0, nu=0.3, rho0=7.8e-3)
    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.001, 0.0, 0.002], dtype=np.float64)

    sig_out, epsp_out = shell_update(p, sig_sh, deps_sh)
    # sig_xx = a11 * deps_xx = 210000 / (1 - 0.09) * 0.001 = 230769.23 * 0.001 = 230.769
    expected_xx = p.a11_2d * 0.001
    expected_yy = p.a12_2d * 0.001
    expected_xy = p.g * 0.002
    assert math.isclose(sig_out[0], expected_xx, rel_tol=1e-4)
    assert math.isclose(sig_out[1], expected_yy, rel_tol=1e-4)
    assert math.isclose(sig_out[2], expected_xy, rel_tol=1e-4)


def test_law78_tangents():
    """Verify 3D and 2D tangent stiffness matrices."""
    p = Law78Params(young=210000.0, nu=0.3)
    C_3d = solid_tangent(p)
    assert C_3d.shape == (1, 6, 6)
    # Check symmetry
    np.testing.assert_allclose(C_3d[0], C_3d[0].T)
    # Check eigenvalues positive definite
    eigvals = np.linalg.eigvalsh(C_3d[0])
    assert np.all(eigvals > 0.0)

    C_2d = shell_tangent(p)
    assert C_2d.shape == (1, 3, 3)
    np.testing.assert_allclose(C_2d[0], C_2d[0].T)
    eigvals_2d = np.linalg.eigvalsh(C_2d[0])
    assert np.all(eigvals_2d > 0.0)


def test_law78_element_group_compliance():
    """Test group-level calling convention compliance."""
    group = SimpleNamespace(
        nel=2,
        mat=Law78Params(),
    )
    # solid_update with group argument
    res = solid_update(group, x=None, u=None, ur=None, dt=1.0e-5, fint=np.zeros(12))
    assert hasattr(group, "state")
    assert "uvar" in group.state
    assert res is not None

    # tangent with group argument
    tang = tangent(group)
    assert tang is not None
    assert tang.shape == (1, 6, 6)


def test_law78_materials_export():
    """Verify law78_composite_dmg is exported from pyradioss.materials."""
    import pyradioss.materials as mats
    assert hasattr(mats, "law78_composite_dmg")
    assert mats.law78_composite_dmg is law78_composite_dmg
