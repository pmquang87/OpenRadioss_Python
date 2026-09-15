"""Tests for /MAT/LAW104 Drucker-Voce-Johnson-Cook material law kernel (M574).

Fortran references:
- starter/source/materials/mat/mat104/hm_read_mat104.F
- engine/source/materials/mat/mat104/sigeps104.F
- engine/source/materials/mat/mat104/mat104_nodam_nice.F
- engine/source/materials/mat/mat104/sigeps104c.F
- engine/source/materials/mat/mat104/mat104c_nodam_nice.F
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law104_drucker import (
    DruckerParams,
    DruckerConstants,
    build_law104,
    compute_drucker_constants,
    compute_drucker_equivalent_stress,
    compute_drucker_yield_stress,
    solid_update,
    shell_update,
    sound_speed,
    sound_speed_solid,
    sound_speed_shell,
    solid_tangent,
    consistent_shell_tangent,
)


def test_drucker_constants_and_convexity():
    """Verify Drucker scaling constant Kdr and convexity clamping."""
    # When Cdr = 0, Drucker collapses to von Mises: Kdr = sqrt(3)
    c0 = compute_drucker_constants(young=210000.0, nu=0.3, cdr=0.0)
    assert math.isclose(c0.kdr, math.sqrt(3.0), rel_tol=1e-12)
    assert c0.cdr == 0.0

    # Upper convexity limit: Cdr = 2.25
    c_upper = compute_drucker_constants(young=210000.0, nu=0.3, cdr=3.0)
    assert c_upper.cdr == 2.25

    # Lower convexity limit: Cdr = -27/8 = -3.375
    c_lower = compute_drucker_constants(young=210000.0, nu=0.3, cdr=-5.0)
    assert c_lower.cdr == -3.375

    # Elastic moduli
    expected_g = 210000.0 / (2.0 * 1.3)
    expected_bulk = 210000.0 / (3.0 * (1.0 - 0.6))
    assert math.isclose(c0.g, expected_g, rel_tol=1e-12)
    assert math.isclose(c0.bulk, expected_bulk, rel_tol=1e-12)


def test_drucker_equivalent_stress_uniaxial_tension():
    """In uniaxial tension, sigma_dr MUST equal sigma_xx for any Cdr."""
    young = 200000.0
    nu = 0.3
    sigma_xx = 450.0
    # Uniaxial stress tensor in Voigt [xx, yy, zz, xy, yz, zx]
    sig_1d = np.array([sigma_xx, 0.0, 0.0, 0.0, 0.0, 0.0])

    for cdr in [-3.375, -2.0, -1.0, 0.0, 0.5, 1.5, 2.25]:
        consts = compute_drucker_constants(young=young, nu=nu, cdr=cdr)
        sig_dr, s, j2, j3 = compute_drucker_equivalent_stress(sig_1d, consts.kdr, consts.cdr)
        assert math.isclose(sig_dr, sigma_xx, rel_tol=1e-9), f"Failed for Cdr={cdr}: got {sig_dr} vs {sigma_xx}"


def test_drucker_equivalent_stress_pure_shear():
    """Test pure shear state against analytical formula."""
    young = 200000.0
    nu = 0.3
    tau = 250.0
    # Pure shear: sig_xy = tau, all other components 0
    sig_shear = np.array([0.0, 0.0, 0.0, tau, 0.0, 0.0])

    # For pure shear: s_xy = tau, s_xx = s_yy = s_zz = 0 -> J3 = 0, J2 = tau^2
    # Fdr = J2^3 - Cdr * 0 = tau^6
    # sig_dr = Kdr * (tau^6)^(1/6) = Kdr * tau
    for cdr in [-3.375, 0.0, 2.25]:
        consts = compute_drucker_constants(young=young, nu=nu, cdr=cdr)
        sig_dr, s, j2, j3 = compute_drucker_equivalent_stress(sig_shear, consts.kdr, consts.cdr)
        assert math.isclose(j3, 0.0, abs_tol=1e-12)
        assert math.isclose(sig_dr, consts.kdr * tau, rel_tol=1e-9)


def test_voce_hardening():
    """Test Voce exponential saturation hardening curve."""
    params = DruckerParams(
        sigma_r=300.0,
        h=500.0,
        qv=150.0,
        bv=20.0,
        cjc=0.0,
        tss=0.0,
    )
    # At epsp = 0 -> sigma_y = sigma_r = 300
    y0, d_epsp = compute_drucker_yield_stress(0.0, 0.0, 0.0, params)
    assert math.isclose(y0, 300.0, rel_tol=1e-12)

    # At epsp = 0.05:
    # linear: H * epsp = 500 * 0.05 = 25
    # voce: Qv * (1 - exp(-Bv * epsp)) = 150 * (1 - exp(-1)) = 150 * (1 - 0.36787944117) = 94.818
    # y = 300 + 25 + 94.818 = 419.818
    epsp = 0.05
    y_expected = 300.0 + 500.0 * epsp + 150.0 * (1.0 - math.exp(-20.0 * epsp))
    y_calc, d_epsp = compute_drucker_yield_stress(epsp, 0.0, 0.0, params)
    assert math.isclose(y_calc, y_expected, rel_tol=1e-9)


def test_johnson_cook_rate_sensitivity():
    """Test Johnson-Cook logarithmic strain rate enhancement."""
    params = DruckerParams(
        sigma_r=400.0,
        h=0.0,
        qv=0.0,
        bv=0.0,
        cjc=0.05,
        eps0=0.001,
        tss=0.0,
    )
    # Strain rate 100 /s: ratio = 100 / 0.001 = 1e5
    # JC factor = 1 + 0.05 * ln(1e5) = 1 + 0.05 * 11.512925 = 1.575646
    rate = 100.0
    y_calc, _ = compute_drucker_yield_stress(0.0, rate, 0.0, params)
    jc_factor = 1.0 + 0.05 * math.log(rate / 0.001)
    assert math.isclose(y_calc, 400.0 * jc_factor, rel_tol=1e-9)

    # Rate below eps0: factor should clamp to 1.0
    y_slow, _ = compute_drucker_yield_stress(0.0, 0.0001, 0.0, params)
    assert math.isclose(y_slow, 400.0, rel_tol=1e-9)


def test_thermal_softening_and_self_heating():
    """Test linear thermal softening and cubic self-heating polynomial."""
    params = DruckerParams(
        sigma_r=500.0,
        tss=0.002,   # mu
        tref=293.15,
        tini=293.15,
        eta=0.9,
        cp=500.0,
        rho=7800.0,
        eps_iso=1.0,
        eps_ad=100.0,
    )
    # At T = 393.15 (delta T = 100):
    # factor = 1 - 0.002 * 100 = 0.8
    y_warm, _ = compute_drucker_yield_stress(0.0, 0.0, 393.15, params)
    assert math.isclose(y_warm, 500.0 * 0.8, rel_tol=1e-9)


def test_sound_speeds():
    """Verify solid and shell sound speeds."""
    params = DruckerParams(
        rho=7800.0,
        young=210000.0e6,
        nu=0.3,
    )
    # Solid longitudinal wave speed: c = sqrt((K + 4/3 G) / rho)
    c_solid = sound_speed_solid(params)
    k = 210000.0e6 / (3.0 * (1.0 - 0.6))
    g = 210000.0e6 / (2.0 * 1.3)
    c_expected_solid = math.sqrt((k + 4.0 / 3.0 * g) / 7800.0)
    assert math.isclose(c_solid, c_expected_solid, rel_tol=1e-9)

    # Shell membrane wave speed: c = sqrt((E / (1 - nu^2)) / rho)
    c_shell = sound_speed_shell(params)
    c_expected_shell = math.sqrt((210000.0e6 / (1.0 - 0.09)) / 7800.0)
    assert math.isclose(c_shell, c_expected_shell, rel_tol=1e-9)
