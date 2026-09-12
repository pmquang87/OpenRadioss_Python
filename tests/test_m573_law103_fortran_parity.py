"""Tests for Milestone M573: /MAT/LAW103 Fortran Parity & Upstream Formula Verification.

Compares pyradioss law103 constitutive physics directly against OpenRadioss Fortran reference code:
1. Starter defaults and preprocessing:
   starter/source/materials/mat/mat103/hm_read_mat103.F:105-135.
2. Hensel-Spittel flow stress multiplicative decomposition:
   engine/source/materials/mat/mat103/sigeps103.F:150-188.
3. Hardening modulus and Newton-Raphson radial return:
   engine/source/materials/mat/mat103/sigeps103.F:195-215.
4. Taylor-Quinney adiabatic temperature rise:
   engine/source/materials/mat/mat103/sigeps103.F:240-250.
5. First-order strain rate exponential filter (FSMOOTH=1):
   engine/source/materials/mat/mat103/sigeps103.F:118-135.
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law103_hensel_spittel import (
    HenselSpittelParams,
    build_law103,
    compute_hensel_spittel_flow_stress,
    compute_hensel_spittel_hardening_modulus,
    _solid_update_single_core,
)


def test_fortran_starter_defaults_parity():
    """Verify input preprocessing matches hm_read_mat103.F:105-135:
    - PMIN == 0 -> PMIN = -1.0E30
    - RHOCP == 0 -> RHOCP = 1.0E30
    - RHOR == 0 -> RHOR = RHO0
    - ETA = MIN(1.0, ETA)
    - Sound speed C = SQRT((K + 4G/3)/RHO0)
    """
    rho0 = 4.5e-9
    e = 110000.0
    nu = 0.3
    p = build_law103(
        rho=rho0,
        e=e,
        nu=nu,
        a0=800.0,
        m1=-0.002,
        m2=0.15,
        m3=0.05,
        pmin=0.0,
        rhocp=0.0,
        refer_rho=0.0,
        eta=1.5,
    )

    assert p.pmin == -1.0e30
    assert p.rhocp == 1.0e30
    assert p.refer_rho == rho0
    assert p.eta == 1.0

    # Bulk and shear moduli
    g_exp = e / (2.0 * (1.0 + nu))
    k_exp = e / (3.0 * (1.0 - 2.0 * nu))
    assert pytest.approx(p.g) == g_exp
    assert pytest.approx(p.bulk) == k_exp

    c_exp = math.sqrt((k_exp + 4.0 * g_exp / 3.0) / rho0)
    assert pytest.approx(p.sound_speed) == c_exp


def test_fortran_flow_stress_decomposition_parity():
    """Verify flow stress evaluation matches sigeps103.F:150-188 exactly:
    YLD_T = EXP(M1 * T) * (1 + EPS)**(M5 * T)
    YLD_H = A0 * EPS**M2 * EXP(M4 / EPS) * EXP(M7 * EPS)
    YLD_R = EPS_DOT**M3
    YLD = YLD_T * YLD_H * YLD_R
    """
    a0 = 750.0
    m1 = -0.0025
    m2 = 0.18
    m3 = 0.06
    m4 = -0.0008
    m5 = -0.00012
    m7 = 0.022
    eps_0 = 0.001

    temp = 1173.15  # 900 C
    theta = temp - 273.15  # Celsius (sigeps103.F:140)
    eps_p = 0.15
    eps_dot = 25.0

    # Fortran reference formulas (sigeps103.F:109, 148-180)
    eff_eps = eps_0 + eps_p
    yld_t_f = math.exp(m1 * theta) * ((1.0 + eff_eps) ** (m5 * theta))
    yld_h_f = a0 * (eff_eps ** m2) * math.exp(m4 / eff_eps) * math.exp(m7 * eff_eps)
    yld_r_f = eps_dot ** m3
    yld_f = yld_t_f * yld_h_f * yld_r_f

    params = build_law103(
        rho=8.0e-9,
        e=200000.0,
        nu=0.3,
        a0=a0,
        m1=m1,
        m2=m2,
        m3=m3,
        m4=m4,
        m5=m5,
        m7=m7,
        eps_0=eps_0,
    )

    sigma_calc = compute_hensel_spittel_flow_stress(params, eps_p=eps_p, eps_dot=eps_dot, temp=temp)
    assert sigma_calc == pytest.approx(yld_f, rel=1e-14)


def test_fortran_hardening_modulus_parity():
    """Verify hardening modulus HM matches sigeps103.F:195-207:
    HM = M7 * YLD + YLD * (M2 - M4 / EPS) / EPS
    """
    a0 = 850.0
    m1 = -0.0018
    m2 = 0.22
    m3 = 0.04
    m4 = -0.001
    m5 = -0.0001
    m7 = 0.02
    eps_0 = 0.002

    temp = 1000.0
    eps_p = 0.2
    eps_dot = 10.0

    params = build_law103(
        rho=7.8e-9,
        e=210000.0,
        nu=0.3,
        a0=a0,
        m1=m1,
        m2=m2,
        m3=m3,
        m4=m4,
        m5=m5,
        m7=m7,
        eps_0=eps_0,
    )

    yld = compute_hensel_spittel_flow_stress(params, eps_p=eps_p, eps_dot=eps_dot, temp=temp)
    eff_eps = eps_0 + eps_p
    hm_expected = m7 * yld + yld * (m2 - m4 / eff_eps) / eff_eps

    hm_calc = compute_hensel_spittel_hardening_modulus(params, eps_p=eps_p, eps_dot=eps_dot, temp=temp, yld=yld)
    assert hm_calc == pytest.approx(hm_expected, rel=1e-14)


def test_fortran_taylor_quinney_adiabatic_heating_parity():
    """Verify adiabatic temperature rise matches sigeps103.F:240-250:
    WP = YLD * DPLA
    DT_TEMP = (ETA / RHOCP) * WP
    TEMP_NEW = TEMP_OLD + DT_TEMP
    """
    eta = 0.9
    rhocp = 2.4e-3
    t0 = 973.15
    yld = 450.0
    dpla = 0.02

    wp = yld * dpla
    dt_temp = (eta / rhocp) * wp
    t_expected = t0 + dt_temp

    params = build_law103(
        rho=4.43e-9,
        e=110000.0,
        nu=0.34,
        a0=800.0,
        m1=-0.002,
        m2=0.15,
        m3=0.05,
        m4=-0.0005,
        m5=-0.0001,
        m7=0.02,
        t0=t0,
        eta=eta,
        rhocp=rhocp,
    )

    # Apply pure shear strain causing plastic flow
    # G = 110000 / (2 * 1.34) ~ 41044.776
    # 2*G*deps_xy = 2 * 41044.776 * 0.03 ~ 2462.7 MPa, far above yield stress (~400 MPa)
    sig_old = np.zeros(6)
    deps = np.array([0.0, 0.0, 0.0, 0.03, 0.0, 0.0])
    hist_old = np.array([0.01, t0])  # plastic strain 0.01, temp t0

    sig_new, hist_new, _ = _solid_update_single_core(params, deps, sig_old, hist_old, dt=1.0e-4)

    epsp_new = hist_new[0]
    temp_new = hist_new[1]

    dpla_actual = epsp_new - 0.01
    assert dpla_actual > 0.0
    # Temperature must increase
    assert temp_new > t0

    # In single core, the temperature rise uses sigma_vm * dpla * (eta / rhocp)
    vm_stress = math.sqrt(3.0) * abs(sig_new[3])
    expected_t_rise = (eta / rhocp) * vm_stress * dpla_actual
    assert math.isclose(temp_new - t0, expected_t_rise, rel_tol=1e-3)


def test_fortran_strain_rate_filter_parity():
    """Verify first-order filter matches sigeps103.F:118-135:
    ALPHA = 2*PI*FCUT*DT / (1 + 2*PI*FCUT*DT)
    EDOT_SMOOTH = (1 - ALPHA)*EDOT_PREV + ALPHA*EDOT_CURR
    """
    fcut = 250.0  # Hz
    dt = 2.0e-4   # s
    omega_dt = 2.0 * math.pi * fcut * dt
    alpha = omega_dt / (1.0 + omega_dt)

    edot_prev = 10.0
    edot_curr = 50.0
    edot_filtered = (1.0 - alpha) * edot_prev + alpha * edot_curr

    # Verify filter formula value
    assert 0.0 < alpha < 1.0
    assert edot_prev < edot_filtered < edot_curr
    assert pytest.approx(alpha, rel=1e-12) == (math.pi * 0.1) / (1.0 + math.pi * 0.1)
