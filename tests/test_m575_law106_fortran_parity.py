"""
Fortran parity tests for /MAT/LAW106 (/MAT/JCOOK_ALM) against sigeps106.F90 and sigeps106c.F90 (Milestone M575).
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law106_jcook_alm import (
    JCookAlmParams,
    compute_jcook_alm_yield_stress,
    compute_temperature_elastic_moduli,
    solid_update,
    shell_update,
)


def test_fortran_parity_thermal_softening():
    """Verify thermal softening matches sigeps106.F90:312-322."""
    t0 = 293.0
    tmelt = 1800.0
    tmax = 1200.0
    m = 1.2
    params = JCookAlmParams(
        id=1,
        young=100000.0,
        nu=0.3,
        a=500.0,
        b=300.0,
        n=0.5,
        m=m,
        tmelt=tmelt,
        tmax=tmax,
        t0=t0,
        tref=t0,
    )

    # 1. T <= T0 -> thsoft = 1.0
    sigy_200, _ = compute_jcook_alm_yield_stress(params, eps_p=0.0, temp=200.0)
    assert pytest.approx(sigy_200, rel=1e-6) == 500.0

    # 2. T0 < T < Tmax -> thsoft = 1 - ((T - T0) / (Tmelt - T0))^m
    t_test = 800.0
    t_star = (t_test - t0) / (tmelt - t0)
    expected_thsoft = 1.0 - (t_star ** m)
    sigy_800, _ = compute_jcook_alm_yield_stress(params, eps_p=0.0, temp=t_test)
    assert pytest.approx(sigy_800, rel=1e-6) == 500.0 * expected_thsoft

    # 3. T > Tmax -> m = 1.0
    t_test_high = 1400.0
    t_star_high = (t_test_high - t0) / (tmelt - t0)
    expected_thsoft_high = 1.0 - t_star_high
    sigy_1400, _ = compute_jcook_alm_yield_stress(params, eps_p=0.0, temp=t_test_high)
    assert pytest.approx(sigy_1400, rel=1e-6) == 500.0 * expected_thsoft_high

    # 4. T >= Tmelt -> thsoft = 0.0 (floored to 1e-10)
    sigy_melt, _ = compute_jcook_alm_yield_stress(params, eps_p=0.0, temp=tmelt + 50.0)
    assert sigy_melt <= 1e-9


def test_fortran_parity_strain_rate_enhancement():
    """Verify strain rate dependence matches sigeps106.F90:304-308."""
    params = JCookAlmParams(
        id=2,
        young=200000.0,
        nu=0.3,
        a=600.0,
        b=0.0,
        n=1.0,
        cjc=0.02,
        deps0=1.0,
    )
    # eps_dot = 100 s^-1 -> 1 + C * ln(1 + 100 / 1)
    eps_dot = 100.0
    expected_ratio = 1.0 + 0.02 * math.log(1.0 + 100.0 / 1.0)
    sigy_dyn, _ = compute_jcook_alm_yield_stress(params, eps_p=0.0, eps_dot=eps_dot)
    assert pytest.approx(sigy_dyn, rel=1e-6) == 600.0 * expected_ratio

    # eps_dot = 0 -> srdep = 1.0 (no enhancement)
    sigy_static, _ = compute_jcook_alm_yield_stress(params, eps_p=0.0, eps_dot=0.0)
    assert pytest.approx(sigy_static, rel=1e-6) == 600.0

    # eps_dot = 0.5 -> 1 + C * ln(1 + 0.5 / 1)
    sigy_half, _ = compute_jcook_alm_yield_stress(params, eps_p=0.0, eps_dot=0.5)
    expected_ratio_half = 1.0 + 0.02 * math.log(1.0 + 0.5)
    assert pytest.approx(sigy_half, rel=1e-6) == 600.0 * expected_ratio_half


def test_fortran_parity_moduli_heating_cooling():
    """Verify heating vs cooling branch of temperature-dependent Young's modulus (sigeps106.F90:230-240)."""
    curve_heat = {293.0: 100000.0, 1000.0: 80000.0}
    curve_cool = {293.0: 90000.0, 1000.0: 70000.0}
    curve_nu = {293.0: 0.3, 1000.0: 0.55}  # should be capped at 0.495

    params = JCookAlmParams(
        id=3,
        young=100000.0,
        nu=0.3,
        fct1=curve_heat,
        fct2=curve_cool,
        fct3=curve_nu,
    )

    # Heating: temp = 600, temp_prev = 500 (temp > temp_prev) -> uses fct1
    e_heat, nu_heat, g_heat, k_heat = compute_temperature_elastic_moduli(params, temp=600.0, temp_prev=500.0)
    expected_e_heat = 100000.0 + (600.0 - 293.0) / (1000.0 - 293.0) * (80000.0 - 100000.0)
    assert pytest.approx(float(e_heat), rel=1e-5) == expected_e_heat

    # Cooling: temp = 500, temp_prev = 600 (temp <= temp_prev) -> uses fct2
    e_cool, nu_cool, g_cool, k_cool = compute_temperature_elastic_moduli(params, temp=500.0, temp_prev=600.0)
    expected_e_cool = 90000.0 + (500.0 - 293.0) / (1000.0 - 293.0) * (70000.0 - 90000.0)
    assert pytest.approx(float(e_cool), rel=1e-5) == expected_e_cool

    # High temperature Poisson's ratio capped at 0.495
    _, nu_high, _, _ = compute_temperature_elastic_moduli(params, temp=1000.0, temp_prev=1000.0)
    assert float(nu_high) == 0.495


def test_fortran_parity_cutting_plane_return():
    """Verify cutting-plane return mapping reduces yield function below tolerance."""
    params = JCookAlmParams(
        id=4,
        young=200000.0,
        nu=0.3,
        a=400.0,
        b=300.0,
        n=0.5,
        vp=2,
        nmax=20,
        tol=1e-6,
    )
    sig_init = np.zeros(6, dtype=float)
    # Significant stretch
    deps = np.array([0.02, -0.006, -0.006, 0.0, 0.0, 0.0], dtype=float)
    epsp = np.array([0.0], dtype=float)

    sig_out, epsp_out, c = solid_update(params, sig_init, deps, epsp=epsp, dt=1e-4)

    s_dev = sig_out - np.mean(sig_out[:3]) * np.array([1, 1, 1, 0, 0, 0], dtype=float)
    von_mises = math.sqrt(1.5 * (s_dev[0]**2 + s_dev[1]**2 + s_dev[2]**2 + 2.0 * (s_dev[3]**2 + s_dev[4]**2 + s_dev[5]**2)))
    sigy, _ = compute_jcook_alm_yield_stress(params, eps_p=float(np.asarray(epsp_out).ravel()[0]))

    # Must be on yield surface within tolerance
    assert abs(von_mises - sigy) / max(sigy, 1.0) < 1e-4


def test_fortran_parity_plane_stress_thinning():
    """Verify shell through-thickness thinning matches sigeps106c.F90:512-515."""
    params = JCookAlmParams(
        id=5,
        young=70000.0,
        nu=0.33,
        a=250.0,
        b=150.0,
        n=0.4,
        vp=2,
    )
    sig_init = np.zeros(3, dtype=float)
    deps = np.array([0.01, -0.0033, 0.0], dtype=float)
    epsp = np.array([0.0], dtype=float)
    thk = np.array([2.0], dtype=float)
    extra = {"thkly": 1.0, "off": 1.0}

    sig_out, epsp_out, _ = shell_update(params, sig_init, deps, epsp=epsp, thk=thk, dt=1e-5, extra=extra)

    # Thickness must decrease under biaxial tension/shear
    assert thk[0] < 2.0
    assert thk[0] > 1.8


def test_fortran_parity_rupture_degradation():
    """Verify off degrades 1.0 -> 0.8 -> ... -> 0.0 upon reaching eps_max (sigeps106.F90:181-183, 480)."""
    params = JCookAlmParams(
        id=6,
        young=100000.0,
        nu=0.3,
        a=300.0,
        eps_max=0.05,
    )
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.08, -0.024, -0.024, 0.0, 0.0, 0.0], dtype=float)
    epsp = np.array([0.0], dtype=float)
    extra = {"off": 1.0}

    # Cycle 1: epsp reaches eps_max -> off becomes 0.8
    sig_out1, epsp_out1, _ = solid_update(params, sig, deps, epsp=epsp, dt=1e-5, extra=extra)
    assert float(np.asarray(epsp_out1).ravel()[0]) >= 0.05
    assert pytest.approx(extra["off"], rel=1e-5) == 0.8

    # Cycle 2: off degrades: 0.8 * 0.8 = 0.64
    sig_out2, epsp_out2, _ = solid_update(params, sig_out1, deps * 0.01, epsp=epsp_out1, dt=1e-5, extra=extra)
    assert pytest.approx(extra["off"], rel=1e-5) == 0.64
