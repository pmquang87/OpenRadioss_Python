"""Tests for LAW103 Hensel-Spittel hot metal forming model (M573).

Verifies:
- Flow stress computation across strain, strain rate, and temperature.
- Logarithmic analytical hardening modulus vs finite difference.
- Single element update: elastic step, plastic yield, radial return.
- Temperature rise from adiabatic plastic work.
- Vectorized array update.
- Algorithmic tangent and acoustic sound speed.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law103_hensel_spittel import (
    HenselSpittelParams,
    build_law103,
    compute_hensel_spittel_flow_stress,
    compute_hensel_spittel_hardening_modulus,
    init_history,
    solid_update_single,
    solid_update_array,
    solid_update,
    sound_speed,
    solid_tangent,
)


def test_flow_stress_basic():
    """Test Hensel-Spittel flow stress with baseline parameters."""
    # A0 = 200 MPa, m1 = -0.001 (thermal softening), m2 = 0.2 (strain hardening)
    # m3 = 0.05 (rate sensitivity), m4 = -0.001, m5 = -0.0001, m7 = 0.05
    a0 = 200.0
    m1 = -0.002
    m2 = 0.15
    m3 = 0.04
    m4 = -0.001
    m5 = -0.0001
    m7 = 0.02

    eps = 0.1
    eps_dot = 10.0
    temp = 1073.15  # 800 C
    theta = temp - 273.15  # 800 C

    sigma_y = compute_hensel_spittel_flow_stress(
        a0, m1, m2, m3, m4, m5, m7, eps, eps_dot=eps_dot, temp=temp
    )

    # Expected factors:
    # yld_h = eps^m2 * exp(m4/eps) * exp(m7*eps)
    yld_h = (eps ** m2) * math.exp(m4 / eps) * math.exp(m7 * eps)
    # yld_sr = eps_dot^m3
    yld_sr = eps_dot ** m3
    # yld_t = exp(m1*theta) * (1+eps)^(m5*theta)
    yld_t = math.exp(m1 * theta) * ((1.0 + eps) ** (m5 * theta))
    expected = a0 * yld_h * yld_sr * yld_t

    assert math.isclose(sigma_y, expected, rel_tol=1.0e-12)
    assert sigma_y > 0.0


def test_thermal_softening():
    """Higher temperatures should yield lower flow stress with m1 < 0."""
    a0 = 300.0
    m1 = -0.003  # thermal softening
    m2 = 0.2
    m3 = 0.05
    m4 = 0.0
    m5 = 0.0
    m7 = 0.0

    eps = 0.2
    eps_dot = 1.0

    # Cold (room temp 20 C = 293.15 K) vs Hot (900 C = 1173.15 K)
    sig_cold = compute_hensel_spittel_flow_stress(
        a0, m1, m2, m3, m4, m5, m7, eps, eps_dot=eps_dot, temp=293.15
    )
    sig_hot = compute_hensel_spittel_flow_stress(
        a0, m1, m2, m3, m4, m5, m7, eps, eps_dot=eps_dot, temp=1173.15
    )

    assert sig_cold > sig_hot
    ratio = sig_hot / sig_cold
    expected_ratio = math.exp(m1 * (900.0 - 20.0))
    assert math.isclose(ratio, expected_ratio, rel_tol=1.0e-6)


def test_strain_rate_sensitivity():
    """Higher strain rates should increase flow stress with m3 > 0."""
    a0 = 250.0
    m1 = 0.0
    m2 = 0.1
    m3 = 0.08  # strain rate sensitivity
    m4 = 0.0
    m5 = 0.0
    m7 = 0.0

    eps = 0.1
    sig_slow = compute_hensel_spittel_flow_stress(
        a0, m1, m2, m3, m4, m5, m7, eps, eps_dot=0.1, temp=293.15
    )
    sig_fast = compute_hensel_spittel_flow_stress(
        a0, m1, m2, m3, m4, m5, m7, eps, eps_dot=100.0, temp=293.15
    )

    assert sig_fast > sig_slow
    expected_ratio = (100.0 / 0.1) ** m3
    assert math.isclose(sig_fast / sig_slow, expected_ratio, rel_tol=1.0e-6)


def test_hardening_modulus_numerical_derivative():
    """Analytical hardening modulus Hm matches sigeps103.F lines 201-206 and numerical derivative when m5=0."""
    a0 = 400.0
    m1 = -0.001
    m2 = 0.25
    m3 = 0.05
    m4 = -0.002
    m5 = 0.0  # sigeps103.F evaluates Hm from YLD_H terms (m7, m2, m4)
    m7 = 0.03

    eps = 0.15
    eps_dot = 5.0
    temp = 773.15  # 500 C

    hm_analytical = compute_hensel_spittel_hardening_modulus(
        a0, m1, m2, m3, m4, m5, m7, eps, eps_dot=eps_dot, temp=temp
    )

    # Central difference
    deps = 1.0e-6
    sig_plus = compute_hensel_spittel_flow_stress(
        a0, m1, m2, m3, m4, m5, m7, eps + deps, eps_dot=eps_dot, temp=temp
    )
    sig_minus = compute_hensel_spittel_flow_stress(
        a0, m1, m2, m3, m4, m5, m7, eps - deps, eps_dot=eps_dot, temp=temp
    )
    hm_numerical = (sig_plus - sig_minus) / (2.0 * deps)

    assert math.isclose(hm_analytical, hm_numerical, rel_tol=1.0e-6)

    # When m5 != 0, sigeps103.F evaluates Hm directly as HM = M7*YLD + YLD*(M2 - M4/EPS)/EPS
    m5_val = -0.0002
    yld_with_m5 = compute_hensel_spittel_flow_stress(
        a0, m1, m2, m3, m4, m5_val, m7, eps, eps_dot=eps_dot, temp=temp
    )
    hm_fortran_formula = m7 * yld_with_m5 + yld_with_m5 * (m2 - m4 / eps) / eps
    hm_calc = compute_hensel_spittel_hardening_modulus(
        a0, m1, m2, m3, m4, m5_val, m7, eps, eps_dot=eps_dot, temp=temp
    )
    assert math.isclose(hm_calc, hm_fortran_formula, rel_tol=1.0e-12)


def test_solid_update_elastic_step():
    """Under small strain within the yield surface, response is purely elastic."""
    params = build_law103(
        rho=7800.0, e=200.0e9, nu=0.3, a0=400.0e6,
        m1=0.0, m2=0.0, m3=0.0, m4=0.0, m5=0.0, m7=0.0,
        t0=293.15, eps0=0.001
    )
    history = init_history(1, t0=293.15)[0]
    sig_old = np.zeros(6, dtype=np.float64)

    # Apply small tension below yield: deps_xx = 5.0e-4 -> sig_xx approx E * deps_xx = 100 MPa < 400 MPa
    deps = np.array([5.0e-4, -0.3 * 5.0e-4, -0.3 * 5.0e-4, 0.0, 0.0, 0.0], dtype=np.float64)

    sig_new, hist_new, ssp = solid_update_single(params, deps, sig_old, history)

    # Plastic strain should remain 0
    assert hist_new[0] == 0.0
    # Temperature should remain unchanged
    assert math.isclose(hist_new[1], 293.15, abs_tol=1.0e-9)

    # Stress should match linear elasticity
    vm = math.sqrt(0.5 * ((sig_new[0] - sig_new[1])**2 + (sig_new[1] - sig_new[2])**2 + (sig_new[2] - sig_new[0])**2))
    assert vm < 400.0e6


def test_solid_update_plastic_step_and_heating():
    """Under large strain exceeding yield, radial return projects stress and updates temperature."""
    rho = 7800.0
    cp = 500.0  # specific heat
    rhocp = rho * cp  # 3.9e6 J/(m^3 K)
    eta = 0.9
    yield_val = 300.0e6

    params = build_law103(
        rho=rho, e=210.0e9, nu=0.3, a0=yield_val,
        m1=0.0, m2=0.0, m3=0.0, m4=0.0, m5=0.0, m7=0.0,
        rcp=rhocp, eta=eta, t0=293.15, eps0=0.001
    )
    history = init_history(1, t0=293.15)[0]
    sig_old = np.zeros(6, dtype=np.float64)

    # Apply large tensile strain increment deps_xx = 0.01 (1%)
    deps = np.array([0.01, -0.003, -0.003, 0.0, 0.0, 0.0], dtype=np.float64)

    sig_new, hist_new, ssp = solid_update_single(params, deps, sig_old, history)

    # Plastic strain must have accumulated
    dpla = hist_new[0]
    assert dpla > 0.0

    # Temperature must have risen due to plastic work
    t_new = hist_new[1]
    assert t_new > 293.15
    delta_t = t_new - 293.15

    # Check that temperature increase matches delta_T = eta * sigma_y * dpla / rhocp
    expected_delta_t = eta * yield_val * dpla / rhocp
    assert math.isclose(delta_t, expected_delta_t, rel_tol=1.0e-3)


def test_solid_update_array_vectorization():
    """Vectorized solid_update_array matches single core execution exactly."""
    params = build_law103(
        rho=7800.0, e=200.0e9, nu=0.28, a0=350.0e6,
        m1=-0.001, m2=0.15, m3=0.02, m4=0.0, m5=0.0, m7=0.01,
        rcp=3.5e6, eta=0.85, t0=300.0, eps0=0.005
    )
    nel = 4
    history = init_history(nel, t0=300.0)
    sig_old = np.zeros((nel, 6), dtype=np.float64)

    deps = np.array([
        [0.0005, -0.00015, -0.00015, 0.0, 0.0, 0.0],  # elastic
        [0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0],    # moderate plastic
        [0.02, -0.006, -0.006, 0.005, 0.0, 0.0],     # heavy plastic with shear
        [-0.01, 0.003, 0.003, 0.0, 0.002, 0.0],      # compressive
    ], dtype=np.float64)

    sig_arr, hist_arr, ssp_arr = solid_update_array(params, deps, sig_old, history)

    for i in range(nel):
        s_single, h_single, ssp_single = solid_update_single(
            params, deps[i], sig_old[i], history[i]
        )
        np.testing.assert_allclose(sig_arr[i], s_single, rtol=1.0e-12)
        np.testing.assert_allclose(hist_arr[i], h_single, rtol=1.0e-12)
        assert math.isclose(ssp_arr[i], ssp_single, rel_tol=1.0e-12)


def test_sound_speed_and_tangent():
    """Test acoustic wave speed and algorithmic tangent matrix."""
    rho = 7850.0
    e = 210.0e9
    nu = 0.3
    params = build_law103(rho=rho, e=e, nu=nu, a0=400.0e6)

    # Dilatational acoustic sound speed: sqrt((K + 4/3 G) / rho)
    c_expected = math.sqrt((params.bulk + (4.0 / 3.0) * params.g) / rho)
    assert math.isclose(params.sound_speed, c_expected, rel_tol=1.0e-9)
    assert math.isclose(sound_speed(params), c_expected, rel_tol=1.0e-9)

    # Elastic tangent when stress is zero
    d_e = solid_tangent(params)
    assert d_e.shape == (6, 6)
    # Symmetric
    np.testing.assert_allclose(d_e, d_e.T, rtol=1.0e-12)
    # Diagonal components
    c11 = params.bulk + (4.0 / 3.0) * params.g
    assert math.isclose(d_e[0, 0], c11, rel_tol=1.0e-9)
    assert math.isclose(d_e[3, 3], params.g, rel_tol=1.0e-9)
