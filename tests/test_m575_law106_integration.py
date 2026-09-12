"""
Integration tests for /MAT/LAW106 (/MAT/JCOOK_ALM) central material dispatch (Milestone M575).
"""

import math
import numpy as np
import pytest

from pyradioss.model.entities import MaterialLaw106
from pyradioss.model.model import Model
import pyradioss.materials as pm
from pyradioss.materials.law106_jcook_alm import build_law106, JCookAlmParams


def test_central_dispatch_solid_update_elastic():
    """Verify pm.solid_update correctly routes to LAW106 solid update for elastic loading."""
    mat = MaterialLaw106(
        id=1,
        title="Ti6Al4V_Solid_Dispatch",
        rho0=4.4e-6,
        young=110000.0,
        nu=0.34,
        sigy=850.0,
        beta=400.0,
        hard_n=0.45,
        vp=2,
        cjc=0.015,
        law=106,
        law_name="LAW106",
    )
    sig = np.zeros(6, dtype=float)
    deps = np.array([1.0e-4, -3.4e-5, -3.4e-5, 0.0, 0.0, 0.0], dtype=float)
    epsp = np.array([0.0], dtype=float)
    dt = 1.0e-6

    sig_out, epsp_out, c = pm.solid_update(mat, sig, deps, epsp=epsp, dt=dt)

    assert sig_out[0] > 0.0
    assert float(np.asarray(epsp_out).ravel()[0]) == 0.0
    assert c is not None
    assert c > 0.0


def test_central_dispatch_solid_update_plastic():
    """Verify pm.solid_update triggers plastic yield and return mapping."""
    mat = MaterialLaw106(
        id=2,
        title="Alloy_Plastic_Dispatch",
        rho0=7.8e-6,
        young=200000.0,
        nu=0.3,
        sigy=300.0,
        beta=250.0,
        hard_n=0.5,
        vp=2,
        cjc=0.01,
        law=106,
        law_name="JCOOK_ALM",
    )
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.01, -0.003, -0.003, 0.0, 0.0, 0.0], dtype=float)
    epsp = np.array([0.0], dtype=float)
    dt = 1.0e-5

    sig_out, epsp_out, c = pm.solid_update(mat, sig, deps, epsp=epsp, dt=dt)

    assert float(np.asarray(epsp_out).ravel()[0]) > 0.0
    s_dev = sig_out[:3] - np.mean(sig_out[:3])
    von_mises = math.sqrt(1.5 * np.sum(s_dev ** 2))
    assert von_mises > 300.0
    assert von_mises < 200000.0 * 0.01


def test_central_dispatch_shell_update():
    """Verify pm.shell_update routes to LAW106 shell plane-stress update."""
    mat = MaterialLaw106(
        id=3,
        title="Sheet_Shell_Dispatch",
        rho0=2.7e-6,
        young=70000.0,
        nu=0.33,
        sigy=200.0,
        beta=150.0,
        hard_n=0.35,
        vp=2,
        law=106,
        law_name="LAW106",
    )
    sig = np.zeros(3, dtype=float)
    deps = np.array([0.005, -0.00165, 0.0], dtype=float)
    epsp = np.array([0.0], dtype=float)
    extra = {"thk": 1.5, "thkly": 1.0, "off": 1.0}

    sig_out, epsp_out = pm.shell_update(mat, sig, deps, epsp=epsp, dt=1.0e-6, extra=extra)

    assert len(sig_out) == 3
    assert float(np.asarray(epsp_out).ravel()[0]) > 0.0
    assert extra["thk"] < 1.5


def test_central_dispatch_sound_speed():
    """Verify pm.sound_speed produces expected dilatational and plane-stress acoustic speeds."""
    rho0 = 7.8e-6
    young = 210000.0
    nu = 0.3
    mat = MaterialLaw106(
        id=4,
        rho0=rho0,
        young=young,
        nu=nu,
        law=106,
        law_name="LAW106",
    )

    k_mod = young / (3.0 * (1.0 - 2.0 * nu))
    g_mod = young / (2.0 * (1.0 + nu))
    c_solid_expected = math.sqrt((k_mod + (4.0 / 3.0) * g_mod) / rho0)
    c_shell_expected = math.sqrt(young / ((1.0 - nu ** 2) * rho0))

    c_solid = pm.sound_speed(mat, is_shell=False)
    c_shell = pm.sound_speed(mat, is_shell=True)

    assert pytest.approx(c_solid, rel=1e-4) == c_solid_expected
    assert pytest.approx(c_shell, rel=1e-4) == c_shell_expected


def test_central_dispatch_tangents():
    """Verify solid and shell tangent operators return correct matrix dimensions."""
    mat = MaterialLaw106(
        id=5,
        rho0=7.8e-6,
        young=210000.0,
        nu=0.3,
        sigy=400.0,
        beta=300.0,
        hard_n=0.5,
        law=106,
        law_name="LAW106",
    )
    sig_solid = np.zeros(6, dtype=float)
    c_solid = pm.solid_tangent(mat, sig_solid)
    assert c_solid.shape == (6, 6)
    assert c_solid[0, 0] > 0.0

    sig_shell = np.zeros(3, dtype=float)
    c_shell = pm.consistent_shell_tangent(mat, sig_shell)
    assert c_shell.shape == (3, 3)
    assert c_shell[0, 0] > 0.0


def test_central_dispatch_resolve_curves():
    """Verify resolve_curves hooks into model.curves/functions."""
    mat = MaterialLaw106(
        id=6,
        rho0=4.5e-6,
        young=100000.0,
        nu=0.3,
        fct_id1=10,
        fct_id2=20,
        fct_id3=30,
        law=106,
        law_name="LAW106",
    )
    model = Model()
    model.functions[10] = [(293.0, 100000.0), (1000.0, 50000.0)]
    model.functions[20] = [(293.0, 95000.0), (1000.0, 48000.0)]
    model.functions[30] = [(293.0, 0.3), (1000.0, 0.35)]

    pm.resolve_curves(mat, model)

    assert mat.params.get("fct1") is not None
    assert mat.params.get("fct2") is not None
    assert mat.params.get("fct3") is not None
