"""
Tests for Milestone M568: /MAT/LAW93 (/MAT/ORTH_HILL) Orthotropic Hill 1948 model.

Verifies:
- Parameter initialization, Hill coefficients, symmetric Poisson ratios, compliance matrices.
- Isotropic limit check: when R_ij = 1.0, Hill yield surface and equivalent stress match von Mises.
- Voce continuous 2-term hardening law.
- Orthotropic 3D continuum solid update: elastic predictor and cutting-plane Newton return mapping.
- Orthotropic 2D plane-stress shell update: elastic predictor, return mapping, and thickness update.
- Algorithmic consistent tangents (6x6 for solid, 3x3 for shell).
- Sound speeds (solid and shell acoustic limits).
"""

import numpy as np
import pytest

from pyradioss.materials.law93_orth_hill import (
    OrthHillParams,
    build_law93,
    eval_yield_stress,
    solid_update,
    shell_update,
    sound_speed,
    sound_speed_shell,
    consistent_tangent,
    extra_shapes,
)


def test_law93_params_isotropic_limit():
    """When all Hill ratios R_ij = 1.0, Hill coefficients match von Mises isotropic plasticity."""
    p = build_law93(
        rho0=7.8e-6,
        e11=210000.0,
        e22=210000.0,
        e33=210000.0,
        g12=80769.23,
        nu12=0.3,
        nu13=0.3,
        nu23=0.3,
        sigma_y=300.0,
        r11=1.0,
        r22=1.0,
        r33=1.0,
        r12=1.0,
        r13=1.0,
        r23=1.0,
    )
    # A1 = A2 = A3 = 1
    assert p.a1 == pytest.approx(1.0)
    assert p.a2 == pytest.approx(1.0)
    assert p.a3 == pytest.approx(1.0)
    # F = G = H = 0.5
    assert p.ff == pytest.approx(0.5)
    assert p.gg == pytest.approx(0.5)
    assert p.hh == pytest.approx(0.5)
    # L = M = N = 1.5
    assert p.ll == pytest.approx(1.5)
    assert p.mm == pytest.approx(1.5)
    assert p.nn == pytest.approx(1.5)


def test_law93_voce_hardening():
    """Verify Voce 2-term continuous hardening: sigma_y(p) = sigma_y0 + Q1(1-exp(-C1*p)) + Q2(1-exp(-C2*p))."""
    p = build_law93(
        rho0=7.8e-6,
        sigma_y=250.0,
        qr1=100.0,
        cr1=20.0,
        qr2=50.0,
        cr2=5.0,
    )
    # At p = 0:
    sy0, dsy0 = eval_yield_stress(p, 0.0)
    assert sy0 == pytest.approx(250.0)
    assert dsy0 == pytest.approx(100.0 * 20.0 + 50.0 * 5.0)

    # At p = 0.05:
    sy_05, dsy_05 = eval_yield_stress(p, 0.05)
    expected_sy = 250.0 + 100.0 * (1.0 - np.exp(-20.0 * 0.05)) + 50.0 * (1.0 - np.exp(-5.0 * 0.05))
    expected_dsy = 100.0 * 20.0 * np.exp(-20.0 * 0.05) + 50.0 * 5.0 * np.exp(-5.0 * 0.05)
    assert sy_05 == pytest.approx(expected_sy, rel=1e-5)
    assert dsy_05 == pytest.approx(expected_dsy, rel=1e-5)

    # Saturated yield stress as p -> infinity:
    sy_inf, _ = eval_yield_stress(p, 10.0)
    assert sy_inf == pytest.approx(250.0 + 100.0 + 50.0, rel=1e-4)


def test_law93_solid_elastic_step():
    """Under yield stress, solid_update responds strictly elastically with zero plastic strain."""
    p = build_law93(
        rho0=7.8e-6,
        e11=210000.0,
        e22=210000.0,
        e33=210000.0,
        g12=80769.23,
        nu12=0.3,
        nu13=0.3,
        nu23=0.3,
        sigma_y=300.0,
    )
    # Small uniaxial strain eps_xx = 5e-4 -> sigma_xx ~ 210000 * 5e-4 = 105 MPa < 300 MPa
    deps = np.array([5e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_old = np.zeros(6)
    extra_dict = {}
    sig_new, pla_new, _ = solid_update(p, deps=deps, sig_old=sig_old, pla=0.0, extra=extra_dict)
    assert extra_dict.get("dpla", 0.0) == 0.0
    assert pla_new == 0.0
    # sigma_xx = D11 * deps_xx
    assert sig_new[0] == pytest.approx(p.d11 * 5e-4, rel=1e-4)
    assert sig_new[1] == pytest.approx(p.d12 * 5e-4, rel=1e-4)
    assert sig_new[2] == pytest.approx(p.d13 * 5e-4, rel=1e-4)


def test_law93_solid_plastic_step_isotropic():
    """In isotropic limit, solid_update Hill return mapping yields von Mises equivalent stress = yield stress."""
    p = build_law93(
        rho0=7.8e-6,
        e11=200000.0,
        e22=200000.0,
        e33=200000.0,
        g12=76923.08,
        nu12=0.3,
        nu13=0.3,
        nu23=0.3,
        sigma_y=300.0,
        qr1=0.0,
        cr1=0.0,
        r11=1.0,
        r22=1.0,
        r33=1.0,
        r12=1.0,
        r13=1.0,
        r23=1.0,
    )
    # Large uniaxial strain eps_xx = 0.01 -> well into plastic regime
    deps = np.array([0.01, -0.003, -0.003, 0.0, 0.0, 0.0])
    sig_old = np.zeros(6)
    extra_dict = {}
    sig_new, pla_new, _ = solid_update(p, deps=deps, sig_old=sig_old, pla=0.0, extra=extra_dict)

    dpla = extra_dict.get("dpla", 0.0)
    assert dpla > 0.0
    assert pla_new == pytest.approx(dpla)

    # Compute von Mises stress:
    sxx, syy, szz, sxy, syz, szx = sig_new
    vm = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 6.0 * (sxy**2 + syz**2 + szx**2)))
    assert vm == pytest.approx(300.0, rel=1e-3)


def test_law93_shell_elastic_step():
    """Under 2D plane stress, elastic trial keeps sigma_zz = 0 and computes Poisson thinning."""
    p = build_law93(
        rho0=7.8e-6,
        e11=210000.0,
        e22=210000.0,
        e33=210000.0,
        g12=80769.23,
        nu12=0.3,
        nu13=0.3,
        nu23=0.3,
        sigma_y=350.0,
    )
    # deps = [deps_xx, deps_yy, deps_xy]
    deps = np.array([5e-4, 0.0, 0.0])
    sig_old = np.zeros(3)
    extra_dict = {"thick": 1.0}
    sig_new, pla_new, _ = shell_update(p, deps=deps, sig_old=sig_old, pla=0.0, extra=extra_dict)

    assert extra_dict.get("dpla", 0.0) == 0.0
    assert pla_new == 0.0
    # sigma_xx = A11 * deps_xx, sigma_yy = A12 * deps_xx
    assert sig_new[0] == pytest.approx(p.a11 * 5e-4, rel=1e-4)
    assert sig_new[1] == pytest.approx(p.a12 * 5e-4, rel=1e-4)
    assert sig_new[2] == pytest.approx(0.0, abs=1e-8)

    # Thickness changes due to elastic transverse strain
    new_thick = extra_dict["thick"]
    assert new_thick < 1.0


def test_law93_shell_plastic_step_anisotropic():
    """Verify anisotropic plane stress return mapping with Voce hardening."""
    # Strong planar anisotropy: R11 = 1.2, R22 = 0.8, R12 = 1.0
    p = build_law93(
        rho0=7.8e-6,
        e11=200000.0,
        e22=200000.0,
        e33=200000.0,
        g12=76923.08,
        nu12=0.3,
        sigma_y=250.0,
        qr1=80.0,
        cr1=15.0,
        r11=1.2,
        r22=0.8,
        r12=1.0,
    )
    # Tensile strain along y (transverse direction, weaker yield stress R22=0.8)
    deps = np.array([0.0, 0.008, 0.0])
    sig_old = np.zeros(3)
    extra_dict = {"thick": 1.0}
    sig_new, pla_new, _ = shell_update(p, deps=deps, sig_old=sig_old, pla=0.0, extra=extra_dict)

    dpla = extra_dict.get("dpla", 0.0)
    assert dpla > 0.0
    assert pla_new == pytest.approx(dpla)
    assert extra_dict["thick"] < 1.0

    # Plane stress Hill yield function at converged stress should match Voce yield stress
    s11, s22, s12 = sig_new
    hill_plane = np.sqrt((p.gg + p.hh) * s11**2 + (p.ff + p.hh) * s22**2 - 2.0 * p.hh * s11 * s22 + 2.0 * p.nn * s12**2)
    sy_exp, _ = eval_yield_stress(p, pla_new)
    assert hill_plane == pytest.approx(sy_exp, rel=1e-3)


def test_law93_sound_speeds():
    """Verify acoustic sound speeds for 3D solid and 2D shell."""
    rho = 7.8e-6
    p = build_law93(
        rho0=rho,
        e11=210000.0,
        e22=210000.0,
        e33=210000.0,
        g12=80769.23,
        nu12=0.3,
        nu13=0.3,
        nu23=0.3,
    )
    c_s = sound_speed(p, rho=rho)
    c_sh = sound_speed_shell(p, rho=rho)
    assert c_s == pytest.approx(np.sqrt(p.d11 / rho), rel=1e-5)
    assert c_sh == pytest.approx(np.sqrt(p.a11 / rho), rel=1e-5)
    assert c_s > c_sh > 0.0


def test_law93_consistent_tangent():
    """Verify algorithmic consistent tangent shapes and positive definiteness."""
    p = build_law93(
        rho0=7.8e-6,
        e11=210000.0,
        e22=210000.0,
        e33=210000.0,
        g12=80769.23,
        nu12=0.3,
        sigma_y=300.0,
    )
    # Elastic tangent solid (6x6)
    c_el_solid = consistent_tangent(p, is_shell=False, sig=np.zeros(6), pla=0.0)
    assert c_el_solid.shape == (6, 6)
    assert np.all(np.linalg.eigvals(c_el_solid) > 0.0)

    # Elastic tangent shell (3x3)
    c_el_shell = consistent_tangent(p, is_shell=True, sig=np.zeros(3), pla=0.0)
    assert c_el_shell.shape == (3, 3)
    assert np.all(np.linalg.eigvals(c_el_shell) > 0.0)


def test_law93_extra_shapes():
    """Verify history variable shapes for element integration."""
    shapes = extra_shapes(None)
    assert "uvar" in shapes
    assert shapes["uvar"] == (1,)
    assert "dpla" in shapes
    assert shapes["dpla"] == (1,)
