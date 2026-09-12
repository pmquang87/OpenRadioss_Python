"""
Tests for /MAT/LAW106 (/MAT/JCOOK_ALM) constitutive physics (Milestone M575).

Validates:
  1. Temperature-dependent elastic moduli E(T) during heating (fct_ID1) and cooling (fct_ID2).
  2. Temperature-dependent Poisson's ratio nu(T) (fct_ID3), bounded by nu <= 0.495.
  3. Yield stress evaluation:
     - Johnson-Cook power hardening (A + B * eps_p^n), capped at sig_max.
     - Viscoplastic and dynamic rate sensitivity (VP=1, 2, 3) with Fcut filter.
     - Thermal softening (1 - (T*)^m) with T* = (T - Tref)/(Tmelt - Tref) and m=1 above Tmax.
  4. Taylor-Quinney adiabatic self-heating: delta_T = (eta / (rho * Cp)) * sigma_vm * delta_epsp.
  5. Cutting-plane return mapping for 3D solids (sigeps106.F90) and 2D plane-stress shells (sigeps106c.F90).
  6. Shell through-thickness thinning: h_new = h_old * (1 + delta_eps_zz * thkly * off).
  7. Element rupture deletion when eps_p >= eps_max (off -> 0.8 -> 0.0, zero stress).
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law106_jcook_alm import (
    JCookAlmParams,
    build_law106,
    compute_temperature_elastic_moduli,
    compute_jcook_alm_yield_stress,
    solid_update,
    shell_update,
    sound_speed,
    sound_speed_solid,
    sound_speed_shell,
    solid_tangent,
    consistent_shell_tangent,
)


def test_temperature_dependent_moduli_heating_cooling():
    """Verify E(T) uses fct_ID1 when dT >= 0 (heating) and fct_ID2 when dT < 0 (cooling)."""
    fct1 = {300.0: 200000.0, 600.0: 150000.0}
    fct2 = {300.0: 210000.0, 600.0: 180000.0}
    fct3 = {300.0: 0.30, 600.0: 0.52}

    p = JCookAlmParams(
        young=200000.0,
        nu=0.3,
        rho=7.8e-6,
        fct_id1=1,
        fct_id2=2,
        fct_id3=3,
        fct1=fct1,
        fct2=fct2,
        fct3=fct3,
        tref=300.0,
        t0=300.0,
    )

    # Heating: T goes from 300 to 450 (dT = +150 > 0)
    e_heat, nu_heat, g_heat, k_heat = compute_temperature_elastic_moduli(p, 450.0, 300.0)
    assert np.isclose(e_heat, 175000.0, rtol=1e-4)

    # Cooling: T goes from 450 to 300 (dT = -150 < 0)
    e_cool, nu_cool, g_cool, k_cool = compute_temperature_elastic_moduli(p, 300.0, 450.0)
    assert np.isclose(e_cool, 210000.0, rtol=1e-4)

    # Clamping test for Poisson's ratio: at T=600, fct3 is 0.52, must be clamped to 0.495
    e_high, nu_high, g_high, k_high = compute_temperature_elastic_moduli(p, 600.0, 600.0)
    assert np.isclose(nu_high, 0.495, atol=1e-5)
    assert nu_high <= 0.495


def test_jcook_alm_yield_stress_and_hardening():
    """Verify flow stress calculation with hardening, rate sensitivity, and thermal softening."""
    p = JCookAlmParams(
        young=200000.0,
        nu=0.3,
        rho=7.8e-6,
        a=400.0,
        b=500.0,
        n=0.5,
        sigma_max=800.0,
        cjc=0.05,
        deps0=1.0,
        m=1.2,
        tmelt=1800.0,
        tref=300.0,
        t0=300.0,
        vp=2,
    )

    # Static room temperature (T = 300, epsp = 0.04, rate = 0.0)
    sig_y, h_mod = compute_jcook_alm_yield_stress(p, epsp=0.04, eps_rate=0.0, temp=300.0)
    assert np.isclose(sig_y, 500.0, rtol=1e-4)
    assert np.isclose(h_mod, 1250.0, rtol=1e-4)

    # Rate sensitivity test: rate = 100.0 s^-1 (sigeps106.F90:304: log(1 + epsd/deps0))
    sig_rate, _ = compute_jcook_alm_yield_stress(p, epsp=0.04, eps_rate=100.0, temp=300.0)
    assert np.isclose(sig_rate, 500.0 * (1.0 + 0.05 * math.log(1.0 + 100.0 / 1.0)), rtol=1e-4)

    # Thermal softening test: T = 1050 K
    sig_therm, _ = compute_jcook_alm_yield_stress(p, epsp=0.04, eps_rate=0.0, temp=1050.0)
    assert np.isclose(sig_therm, 500.0 * (1.0 - 0.5**1.2), rtol=1e-4)

    # Sig_max capping test
    sig_capped, h_capped = compute_jcook_alm_yield_stress(p, epsp=10.0, eps_rate=0.0, temp=300.0)
    assert np.isclose(sig_capped, 800.0, rtol=1e-4)
    assert np.isclose(h_capped, 0.0, atol=1e-4)


def test_solid_update_elastic_and_plastic():
    """Verify 3D solid continuum update: elastic loading and plastic yield return."""
    p = JCookAlmParams(
        young=200000.0,
        nu=0.3,
        rho=7.8e-6,
        a=400.0,
        b=300.0,
        n=0.5,
        sigma_max=1000.0,
        cjc=0.0,
        deps0=1.0,
        m=1.0,
        tmelt=1800.0,
        tref=300.0,
        t0=300.0,
        vp=2,
        nmax=20,
    )

    n_elem = 2
    sig = np.zeros((n_elem, 6), dtype=np.float64)
    deps_elastic = np.array([[0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]])
    epsp = np.zeros(n_elem, dtype=np.float64)
    extra = {"temp": np.full(n_elem, 300.0), "temp_prev": np.full(n_elem, 300.0)}

    bulk = p.bulk
    g = p.g
    c11 = bulk + (4.0 / 3.0) * g
    sig_out, epsp_out, c_out = solid_update(p, sig, deps_elastic, epsp=epsp, dt=1e-5, extra=extra)

    assert np.allclose(epsp_out, 0.0)
    assert np.isclose(sig_out[0, 0], c11 * 0.001, rtol=1e-3)
    assert c_out is not None

    deps_plastic = np.array([[0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0],
                             [0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0]])
    sig_p, epsp_p, _ = solid_update(p, sig_out, deps_plastic, epsp=epsp_out, dt=1e-5, extra=extra)

    assert np.all(epsp_p > 0.0)
    s_dev = sig_p[0, :3] - np.mean(sig_p[0, :3])
    vm = math.sqrt(1.5 * np.sum(s_dev**2))
    expected_flow, _ = compute_jcook_alm_yield_stress(p, epsp=epsp_p[0], eps_rate=0.0, temp=300.0)
    assert np.isclose(vm, expected_flow, rtol=1e-2)


def test_taylor_quinney_adiabatic_heating():
    """Verify adiabatic temperature rise from plastic dissipation in solid update."""
    rho_val = 7.8e-6
    cs_val = 460.0 * 1e6
    p = JCookAlmParams(
        young=200000.0,
        nu=0.3,
        rho=rho_val,
        a=400.0,
        b=0.0,
        n=1.0,
        cs=cs_val,
        eta=0.9,
        tref=300.0,
        t0=300.0,
    )

    sig = np.zeros((1, 6), dtype=np.float64)
    deps = np.array([[0.02, -0.006, -0.006, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1, dtype=np.float64)
    extra = {"temp": np.array([300.0]), "temp_prev": np.array([300.0])}

    sig_out, epsp_out, _ = solid_update(p, sig, deps, epsp=epsp, dt=1e-4, extra=extra)

    t_new = extra["temp"][0]
    assert t_new > 300.0
    delta_t = t_new - 300.0

    s_dev = sig_out[0, :3] - np.mean(sig_out[0, :3])
    vm = math.sqrt(1.5 * np.sum(s_dev**2))
    expected_dt = (0.9 * vm * epsp_out[0]) / cs_val
    assert np.isclose(delta_t, expected_dt, rtol=1e-2)


def test_shell_update_plane_stress_and_thinning():
    """Verify 2D shell plane-stress return and through-thickness thinning."""
    p = JCookAlmParams(
        young=200000.0,
        nu=0.3,
        rho=7.8e-6,
        a=350.0,
        b=400.0,
        n=0.3,
        sigma_max=800.0,
        tref=300.0,
        t0=300.0,
        nmax=30,
    )

    sig = np.zeros((1, 3), dtype=np.float64)
    deps = np.array([[0.003, 0.0, 0.0]], dtype=np.float64)
    epsp = np.zeros(1, dtype=np.float64)
    thick = np.array([2.0])
    extra = {
        "temp": np.array([300.0]),
        "temp_prev": np.array([300.0]),
        "thick": thick,
        "thkly": 1.0,
    }

    sig_out, epsp_out, c_out = shell_update(p, sig, deps, epsp=epsp, dt=1e-5, extra=extra)

    assert sig_out[0, 0] > 0.0
    vm = math.sqrt(sig_out[0, 0]**2 + sig_out[0, 1]**2 - sig_out[0, 0]*sig_out[0, 1] + 3.0*sig_out[0, 2]**2)
    expected_flow, _ = compute_jcook_alm_yield_stress(p, epsp=epsp_out[0], eps_rate=0.0, temp=300.0)
    assert np.isclose(vm, expected_flow, rtol=1e-2)
    assert thick[0] < 2.0


def test_element_deletion_at_eps_max():
    """Verify element rupture deletion when plastic strain exceeds eps_max."""
    p = JCookAlmParams(
        young=200000.0,
        nu=0.3,
        rho=7.8e-6,
        a=300.0,
        b=100.0,
        n=1.0,
        eps_max=0.05,
    )

    sig = np.zeros((1, 6), dtype=np.float64)
    deps = np.array([[0.10, -0.03, -0.03, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1, dtype=np.float64)
    off = np.array([1.0])
    extra = {"temp": np.array([300.0]), "temp_prev": np.array([300.0]), "off": off}

    # Step 1: Rupture occurs (eps_p >= 0.05) -> off drops from 1.0 to 0.8
    sig_out, epsp_out, _ = solid_update(p, sig, deps, epsp=epsp, dt=1e-5, extra=extra)
    assert extra["off"][0] == 0.8
    assert not np.allclose(sig_out, 0.0)

    # Step 2: In subsequent steps, off degrades (0.8 -> 0.64 -> ... -> 0.0)
    for _ in range(12):
        sig_out, epsp_out, _ = solid_update(p, sig_out, np.zeros_like(deps), epsp=epsp_out, dt=1e-5, extra=extra)
    assert extra["off"][0] == 0.0
    assert np.allclose(sig_out, 0.0)


def test_sound_speeds():
    """Verify sound speed calculations for solids and shells."""
    p = JCookAlmParams(
        young=210000.0,
        nu=0.3,
        rho=7.8e-6,
    )
    c_solid = sound_speed_solid(p)
    c_shell = sound_speed_shell(p)

    expected_solid = math.sqrt((p.bulk + (4.0 / 3.0) * p.g) / p.rho)
    expected_shell = math.sqrt(p.young / (p.rho * (1.0 - p.nu**2)))

    assert np.isclose(c_solid, expected_solid, rtol=1e-6)
    assert np.isclose(c_shell, expected_shell, rtol=1e-6)
    assert c_solid > c_shell
