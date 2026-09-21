"""Unit tests for LAW77 (Thermoplastic Viscoplastic Polymer / Foam-Air) ported from sigeps77.F."""

import numpy as np
import pytest

from pyradioss.materials import law77_polymer
from pyradioss.materials.law77_polymer import (
    Law77Params,
    build_law77,
    extra_shapes,
    needs_defgrad,
    resolve,
    shell_tangent,
    shell_update,
    solid_tangent,
    solid_update,
    sound_speed,
    tangent,
)
from pyradioss.model.entities import Material


def test_law77_params_from_dict():
    """Verify parameter parsing and derived constants from dictionary."""
    mat_data = {
        "id": 77,
        "title": "TEST_POLYMER",
        "MAT_RHO": 0.05,
        "MAT_E": 10.0,
        "MAT_NU": 0.25,
        "E_Max": 100.0,
        "MAT_EPS": 0.8,
        "MAT_P0": 0.1013,
        "GAMMA": 1.4,
        "MAT_POROS": 0.85,
    }
    p = build_law77(mat_data)
    assert isinstance(p, Law77Params)
    assert p.id == 77
    assert p.e0 == 10.0
    assert p.nu == 0.25
    assert p.emax == 100.0
    assert p.epsmax == 0.8
    assert p.p0 == 0.1013
    assert p.gamma == 1.4
    assert p.frac == 0.85

    # Check derived moduli
    assert p.g > 0.0
    assert p.bulk > 0.0
    assert p.aa1 > p.bulk
    assert p.c_solid > 0.0
    assert p.c_shell > 0.0
    assert needs_defgrad(p) is False


def test_law77_params_from_material_entity():
    """Verify parameter parsing from pyradioss Material entity."""
    mat = Material(
        id=5,
        law=77,
        law_name="LAW77",
        params={
            "MAT_RHO": 0.08,
            "MAT_E": 12.5,
            "MAT_NU": 0.2,
            "E_Max": 75.0,
        },
    )
    p = resolve(mat)
    assert isinstance(p, Law77Params)
    assert p.id == 5
    assert p.e0 == 12.5
    assert p.nu == 0.2
    assert p.emax == 75.0
    assert p.rho0 == 0.08


def test_law77_extra_shapes():
    """Verify history variable allocation returns 23 state variables."""
    shapes = extra_shapes(None, nip=1)
    assert "uvar77" in shapes
    assert shapes["uvar77"] == (23,)
    assert shapes["uvar"] == (23,)

    shapes_multi = extra_shapes(None, nip=4)
    assert shapes_multi["uvar77"] == (4, 23)


def test_law77_solid_update_single_element():
    """Verify 3D solid continuum update on a single element under compression."""
    p = build_law77(e0=10.0, nu=0.2, emax=80.0, rho0=0.05)
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([-0.02, -0.005, -0.005, 0.0, 0.0, 0.0], dtype=np.float64)
    extra = {}

    sig_out, epsp_out, c_out = solid_update(
        p, sig0, deps, dt=1.0e-6, extra=extra, return_sound_speed=True
    )

    assert sig_out.shape == (6,)
    # Compressive strain in x produces compressive stress in x (negative)
    assert sig_out[0] < 0.0
    assert sig_out[1] < 0.0
    assert sig_out[2] < 0.0
    assert c_out > 0.0

    # Verify history variables were initialized and updated
    assert "uvar77" in extra
    uvar = extra["uvar77"]
    assert uvar.shape == (1, 23)
    assert uvar[0, 11] == p.e0  # UVAR(12) = current Young's modulus
    assert uvar[0, 12] > 0.0  # UVAR(13) = EPST equivalent strain
    assert uvar[0, 13] == 1.0  # UVAR(14) = ILOAD (+1 loading)


def test_law77_solid_update_vectorized():
    """Verify multi-element vectorized batch solid update."""
    p = build_law77(e0=20.0, nu=0.3, emax=120.0, rho0=0.1)
    nel = 5
    sig0 = np.zeros((nel, 6), dtype=np.float64)
    deps = np.full((nel, 6), -0.01, dtype=np.float64)
    deps[:, 3:] = 0.002
    extra = {}

    sig_out, epsp_out, c_out = solid_update(
        p, sig0, deps, dt=1.0e-5, extra=extra, return_sound_speed=True
    )

    assert sig_out.shape == (nel, 6)
    assert len(c_out) == nel
    assert np.all(sig_out[:, 0] < 0.0)
    assert extra["uvar77"].shape == (nel, 23)


def test_law77_shell_update():
    """Verify 2D plane-stress shell constitutive update."""
    p = build_law77(e0=15.0, nu=0.25, rho0=0.04)
    sig0 = np.zeros(3, dtype=np.float64)
    deps = np.array([-0.01, -0.002, 0.005], dtype=np.float64)
    extra = {}

    sig_out, epsp_out, c_out = shell_update(
        p, sig0, deps, dt=1.0e-6, extra=extra
    )

    assert sig_out.shape == (3,)
    assert sig_out[0] < 0.0
    assert c_out > 0.0
    assert extra["uvar77"].shape == (1, 23)


def test_law77_tangents():
    """Verify consistent solid and shell tangent stiffness matrices."""
    p = build_law77(e0=50.0, nu=0.3, rho0=0.08)

    # Solid tangent
    c_sol = solid_tangent(p)
    assert c_sol.shape == (1, 6, 6)
    # Check symmetry
    np.testing.assert_allclose(c_sol[0], c_sol[0].T, rtol=1e-12)
    # Diagonal components must be positive
    assert np.all(np.diag(c_sol[0]) > 0.0)

    # Shell tangent
    c_sh = shell_tangent(p)
    assert c_sh.shape == (1, 3, 3)
    np.testing.assert_allclose(c_sh[0], c_sh[0].T, rtol=1e-12)
    assert np.all(np.diag(c_sh[0]) > 0.0)

    # Elemental template tangent interface
    t_res = tangent(p)
    assert t_res is not None
    assert t_res.shape == (1, 6, 6)


def test_law77_sound_speed():
    """Verify acoustic sound speed calculation."""
    p = build_law77(e0=100.0, nu=0.2, rho0=0.05)
    c_sol = sound_speed(p, is_shell=False)
    c_sh = sound_speed(p, is_shell=True)
    assert c_sol > 0.0
    assert c_sh > 0.0
    assert isinstance(c_sol, float)


def test_law77_elemental_template_signature():
    """Verify template signature compliance: solid_update(group, x, u, ur, dt, fint, mint)."""
    class MockGroup:
        elements = [1, 2, 3]

    grp = MockGroup()
    x = np.zeros((3, 3))
    u = np.zeros((3, 3))
    ur = np.zeros((3, 3))
    dt = 1.0e-6
    fint = np.zeros((3, 3))
    mint = np.zeros((3, 3))

    res = solid_update(grp, x, u, ur, dt, fint, mint)
    assert res is None

    t_res = tangent(grp)
    assert t_res is None or isinstance(t_res, np.ndarray)


def test_law77_dispatcher():
    """Verify dispatch through pyradioss.materials central dispatcher."""
    import pyradioss.materials as pmat

    mat = Material(
        id=77,
        law=77,
        law_name="LAW77",
        params={
            "MAT_RHO": 0.05,
            "MAT_E": 10.0,
            "MAT_NU": 0.2,
        },
    )

    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([-0.01, -0.002, -0.002, 0.0, 0.0, 0.0], dtype=np.float64)

    sig_out, epsp_out, c_out = pmat.solid_update(mat, sig0, deps, dt=1.0e-6)
    assert sig_out[0] < 0.0
    assert c_out > 0.0

