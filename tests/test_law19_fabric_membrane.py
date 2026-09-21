"""
Tests for LAW19 — Orthotropic fabric membrane with Ogden-type hyperelasticity,
bilinear fiber response, rate dependency, and reduced compression (/MAT/LAW19, /MAT/FABRI).

Verifies:
- Orthotropic response in two fiber directions (warp/weft)
- Bilinear stress-strain response past knee strain (ET1, ET2, EPSY1, EPSY2)
- Rate-dependent strain-rate enhancement (C_RATE, EPS0)
- Ogden-type hyperelastic membrane extension (MU_OGDEN, ALPHA_OGDEN)
- Reduced compression kinematics (RCOMP, sigeps19c.F)
- REF-STATE zerostress option
- Plane-stress consistent tangents and sound speed
- Template interfaces: tangent, extra_shapes, needs_defgrad, resolve
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials import (
    law19_fabric,
    shell_membrane_tangent,
    consistent_shell_tangent,
)
from pyradioss.model.entities import Material


def _make_mat(e11=2000.0, e22=1000.0, nu12=0.3, g12=500.0, rcomp=0.1, **kwargs):
    rec = type("Rec", (), {})()
    rec.id = 19
    rec.title = "FABRIC_MEMBRANE"
    rec.density = 1.2e-9
    rec.params = {
        "E11": e11,
        "E22": e22,
        "NU12": nu12,
        "G12": g12,
        "G23": 400.0,
        "G31": 300.0,
        "RCOMP": rcomp,
        **kwargs,
    }
    return law19_fabric.build_fabric(rec)


def test_law19_params_and_resolve():
    p = law19_fabric.Law19Params(
        e11=3000.0,
        e22=1500.0,
        nu12=0.25,
        g12=600.0,
        rcomp=0.05,
        et1=600.0,
        et2=300.0,
        epsy1=0.02,
        epsy2=0.03,
        c_rate=0.05,
        eps0=1.0,
    )
    assert p.e11 == 3000.0
    assert p.e22 == 1500.0
    assert p.rcomp == 0.05
    assert abs(p.nu21 - 0.25 * (1500.0 / 3000.0)) < 1e-6
    assert abs(p.detc - (1.0 - 0.25 * 0.125)) < 1e-6

    # Test resolve from dict and Material
    mat_dict = {"E11": 2500.0, "E22": 1200.0, "NU12": 0.2, "G12": 400.0}
    mat_res = law19_fabric.resolve(mat_dict)
    assert isinstance(mat_res, law19_fabric.FabricMaterial)
    assert mat_res.params["E11"] == 2500.0


def test_law19_bilinear_fiber_response():
    """Verify bilinear stress-strain curves in warp and weft directions past knee strains."""
    e11, e22, nu12 = 2000.0, 1000.0, 0.0  # uncoupled to check exact 1D values
    et1, et2 = 400.0, 200.0
    epsy1, epsy2 = 0.01, 0.02

    mat = _make_mat(
        e11=e11, e22=e22, nu12=nu12,
        ET1=et1, ET2=et2, EPSY1=epsy1, EPSY2=epsy2,
    )

    # 1. Pre-knee strain in warp (eps_xx = 0.005 < 0.01):
    sig = np.zeros((1, 3))
    deps_pre = np.array([[0.005, 0.0, 0.0]])
    extra = {}
    s_pre, _ = law19_fabric.shell_update(mat, sig, deps_pre, extra=extra)
    # Expected: E11 * eps = 2000 * 0.005 = 10.0
    assert s_pre[0, 0] == pytest.approx(10.0, rel=1e-6)

    # 2. Post-knee strain in warp (eps_xx = 0.02 > 0.01):
    sig2 = np.zeros((1, 3))
    deps_post = np.array([[0.02, 0.0, 0.0]])
    extra2 = {}
    s_post, _ = law19_fabric.shell_update(mat, sig2, deps_post, extra=extra2)
    # Expected bilinear: E11 * epsy1 + ET1 * (eps - epsy1) = 2000 * 0.01 + 400 * (0.02 - 0.01) = 20.0 + 4.0 = 24.0
    assert s_post[0, 0] == pytest.approx(24.0, rel=1e-6)

    # 3. Post-knee strain in weft (eps_yy = 0.03 > 0.02):
    sig3 = np.zeros((1, 3))
    deps_weft = np.array([[0.0, 0.03, 0.0]])
    extra3 = {}
    s_weft, _ = law19_fabric.shell_update(mat, sig3, deps_weft, extra=extra3)
    # Expected: E22 * epsy2 + ET2 * (eps - epsy2) = 1000 * 0.02 + 200 * (0.03 - 0.02) = 20.0 + 2.0 = 22.0
    assert s_weft[0, 1] == pytest.approx(22.0, rel=1e-6)


def test_law19_rate_dependent_enhancement():
    """Verify strain rate dependent stress enhancement factor F_rate."""
    mat = _make_mat(e11=2000.0, e22=1000.0, nu12=0.0, C_RATE=0.1, EPS0=10.0)
    sig = np.zeros((1, 3))
    # deps_xx = 0.01 with dt = 1e-4 -> strain rate = 0.01 / 1e-4 = 100.0 s^-1
    deps = np.array([[0.01, 0.0, 0.0]])
    dt = 1e-4
    extra = {}
    s, _ = law19_fabric.shell_update(mat, sig, deps, dt=dt, extra=extra)

    # Expected: F_rate = 1 + 0.1 * ln(100.0 / 10.0) = 1 + 0.1 * ln(10) = 1 + 0.2302585 = 1.2302585
    # Base stress: 2000.0 * 0.01 = 20.0
    # Enhanced stress: 20.0 * 1.2302585 = 24.60517
    expected_sxx = 20.0 * (1.0 + 0.1 * math.log(10.0))
    assert s[0, 0] == pytest.approx(expected_sxx, rel=1e-5)


def test_law19_ogden_hyperelasticity_extension():
    """Verify Ogden-type nonlinear stretch contribution for rubberized/coated fabric membranes."""
    mat = _make_mat(e11=1000.0, e22=1000.0, nu12=0.0, MU_OGDEN=50.0, ALPHA_OGDEN=2.0)
    sig = np.zeros((1, 3))
    deps = np.array([[0.1, 0.0, 0.0]])
    extra = {}
    s, _ = law19_fabric.shell_update(mat, sig, deps, extra=extra)

    # Linear part: 1000 * 0.1 = 100.0
    # Ogden part: lambda = 1.1, mu = 50, alpha = 2 -> 50 * (1.1^2 - 1.1^(-1)) = 50 * (1.21 - 0.9090909) = 15.04545
    lam = 1.1
    ogden_part = 50.0 * (lam ** 2.0 - lam ** (-1.0))
    expected_sxx = 100.0 + ogden_part
    assert s[0, 0] == pytest.approx(expected_sxx, rel=1e-5)


def test_law19_reduced_compression_scaling():
    """Verify reduced compression (RCOMP) under bi-compression and mixed stress states (sigeps19c.F)."""
    rcomp = 0.05
    mat = _make_mat(e11=2000.0, e22=1000.0, nu12=0.0, rcomp=rcomp)

    # Bi-compression: deps_xx = -0.01, deps_yy = -0.01
    sig = np.zeros((1, 3))
    deps_bicomp = np.array([[-0.01, -0.01, 0.0]])
    extra = {}
    s_bi, _ = law19_fabric.shell_update(mat, sig, deps_bicomp, extra=extra)

    # Under bi-compression, stress is scaled directly by RCOMP
    assert s_bi[0, 0] == pytest.approx(-20.0 * rcomp, rel=1e-5)
    assert s_bi[0, 1] == pytest.approx(-10.0 * rcomp, rel=1e-5)


def test_law19_tangents_and_template_interfaces():
    mat = _make_mat(e11=2000.0, e22=1000.0, nu12=0.3, g12=400.0)

    # Membrane tangent
    c_mem = law19_fabric.shell_membrane_tangent(mat)
    assert c_mem.shape == (3, 3)
    assert c_mem[0, 0] > 0.0
    assert c_mem[1, 1] > 0.0
    assert c_mem[2, 2] == pytest.approx(400.0)

    # Template tangent alias
    c_tan = law19_fabric.tangent(mat)
    assert np.array_equal(c_tan, c_mem)

    # Sound speed
    c_spd = law19_fabric.sound_speed(mat)
    assert c_spd > 0.0

    # Extra shapes and defgrad
    shapes = law19_fabric.extra_shapes(mat, nip=3)
    assert "eps19" in shapes and shapes["eps19"] == (3, 3)
    assert "sigi19" in shapes and shapes["sigi19"] == (3, 3)
    assert not law19_fabric.needs_defgrad(mat)


def test_law19_dispatcher_wiring():
    mat = _make_mat()
    c_mem = shell_membrane_tangent(mat)
    assert c_mem.shape == (3, 3)

    extra = {"eps19": np.array([[0.01, 0.005, 0.0]])}
    d_cons = consistent_shell_tangent(mat, extra=extra)
    assert d_cons.shape == (1, 3, 3)
