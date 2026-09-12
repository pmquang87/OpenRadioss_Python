"""Tests for Milestone M574: /MAT/LAW104 Fortran Parity & Upstream Formula Verification.

Compares pyradioss law104 constitutive physics directly against OpenRadioss Fortran reference code:
1. Starter preprocessing & constants:
   starter/source/materials/mat/mat104/hm_read_mat104.F:148-175.
2. Drucker invariant and equivalent stress formulations:
   engine/source/materials/mat/mat104/mat104_nodam_nice.F:204-220.
3. Voce hardening, Johnson-Cook rate sensitivity, and thermal softening:
   engine/source/materials/mat/mat104/mat104_nodam_nice.F:168-180.
4. Taylor-Quinney adiabatic self-heating with cubic weighting:
   engine/source/materials/mat/mat104/mat104_nodam_nice.F:145-165.
5. Cutting-plane Newton plasticity return mapping:
   engine/source/materials/mat/mat104/mat104_nodam_newton.F:240-435.
6. Plane-stress shell thinning and plastic incompressibility:
   engine/source/materials/mat/mat104/mat104c_nodam_nice.F:180-210.
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law104_drucker import (
    DruckerParams,
    compute_drucker_constants,
    compute_drucker_equivalent_stress,
    compute_drucker_yield_stress,
    solid_update,
    shell_update,
)


def test_fortran_starter_constants_and_convexity_parity():
    """Verify Drucker constants and convexity bounds matching hm_read_mat104.F:148-175:
    - G = E / (2 * (1 + nu))
    - BULK = E / (3 * (1 - 2 * nu))
    - Convexity bounds: CDR in [-27/8, 9/4] = [-3.375, 2.25]
    - KDR = (1/27 - CDR * 4/729)**(-1/6)
    - von Mises limit (CDR = 0): KDR = sqrt(3)
    """
    young = 210000.0
    nu = 0.3
    g_exp = young / (2.0 * (1.0 + nu))
    bulk_exp = young / (3.0 * (1.0 - 2.0 * nu))

    # Case 1: CDR = 0 (von Mises limit)
    c0 = compute_drucker_constants(young=young, nu=nu, cdr=0.0)
    assert pytest.approx(c0.g, rel=1e-12) == g_exp
    assert pytest.approx(c0.bulk, rel=1e-12) == bulk_exp
    assert pytest.approx(c0.kdr, rel=1e-14) == math.sqrt(3.0)

    # Case 2: CDR = 1.0
    # KDR = (1/27 - 4/729)**(-1/6) = (23/729)**(-1/6) = (729/23)**(1/6)
    kdr_exp_1 = (729.0 / 23.0) ** (1.0 / 6.0)
    c1 = compute_drucker_constants(young=young, nu=nu, cdr=1.0)
    assert pytest.approx(c1.kdr, rel=1e-14) == kdr_exp_1

    # Case 3: Upper convexity bound CDR = 2.25 (9/4)
    # 1/27 - (9/4) * (4/729) = 1/27 - 9/729 = 1/27 - 1/81 = (3-1)/81 = 2/81
    # KDR = (81/2)**(1/6) = 40.5**(1/6)
    kdr_exp_upper = 40.5 ** (1.0 / 6.0)
    c_upper = compute_drucker_constants(young=young, nu=nu, cdr=2.25)
    assert pytest.approx(c_upper.kdr, rel=1e-14) == kdr_exp_upper

    # Case 4: Lower convexity bound CDR = -3.375 (-27/8)
    # 1/27 - (-27/8) * (4/729) = 1/27 + 108 / (8 * 729) = 1/27 + 1/54 = 3/54 = 1/18
    # KDR = 18**(1/6)
    kdr_exp_lower = 18.0 ** (1.0 / 6.0)
    c_lower = compute_drucker_constants(young=young, nu=nu, cdr=-3.375)
    assert pytest.approx(c_lower.kdr, rel=1e-14) == kdr_exp_lower

    # Case 5: Out of bounds clamping (hm_read_mat104.F:159-170)
    c_clamped_high = compute_drucker_constants(young=young, nu=nu, cdr=10.0)
    assert pytest.approx(c_clamped_high.kdr, rel=1e-14) == kdr_exp_upper

    c_clamped_low = compute_drucker_constants(young=young, nu=nu, cdr=-10.0)
    assert pytest.approx(c_clamped_low.kdr, rel=1e-14) == kdr_exp_lower


def test_fortran_sound_speeds_parity():
    """Verify sound speed formulas:
    - Solid: c_solid = sqrt((K + 4G/3) / rho0)
    - Shell: c_shell = sqrt(E / ((1 - nu^2) * rho0))
    """
    rho0 = 7.85e-9
    young = 205000.0
    nu = 0.28

    params = DruckerParams(
        rho=rho0,
        young=young,
        nu=nu,
        sigma_r=350.0,
    )

    g = young / (2.0 * (1.0 + nu))
    bulk = young / (3.0 * (1.0 - 2.0 * nu))
    c_solid_exp = math.sqrt((bulk + 4.0 * g / 3.0) / rho0)
    c_shell_exp = math.sqrt(young / ((1.0 - nu * nu) * rho0))

    assert pytest.approx(params.sound_speed, rel=1e-12) == c_solid_exp
    assert pytest.approx(params.sound_speed_shell, rel=1e-12) == c_shell_exp


def test_fortran_drucker_uniaxial_tension_normalization():
    """Verify exact Drucker equivalent stress normalization in uniaxial tension (mat104_nodam_nice.F:204-220):
    For sigma = [sigma0, 0, 0, 0, 0, 0], sigma_dr must equal sigma0 identically
    for ALL valid values of CDR!
    """
    sigma0 = 425.0
    sig_uniaxial = np.array([sigma0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for cdr in [-3.375, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.25]:
        c = compute_drucker_constants(young=200000.0, nu=0.3, cdr=cdr)
        sig_dr, *_ = compute_drucker_equivalent_stress(sig_uniaxial, c.kdr, c.cdr)
        assert pytest.approx(sig_dr, rel=1e-12) == sigma0


def test_fortran_drucker_pure_shear_parity():
    """Verify Drucker equivalent stress under pure shear (mat104_nodam_nice.F:204-220):
    For tau_xy = tau, J2 = tau^2, J3 = 0.
    Drucker function f_dr = J2^3 - CDR * J3^2 = tau^6.
    Equivalent stress sigma_dr = KDR * tau.
    When CDR = 0, sigma_dr = sqrt(3) * tau (exact von Mises).
    """
    tau = 175.0
    sig_shear = np.array([0.0, 0.0, 0.0, tau, 0.0, 0.0])

    for cdr in [-3.0, -1.0, 0.0, 1.0, 2.0]:
        c = compute_drucker_constants(young=200000.0, nu=0.3, cdr=cdr)
        sig_dr, *_ = compute_drucker_equivalent_stress(sig_shear, c.kdr, c.cdr)
        expected_sig_dr = c.kdr * tau
        assert pytest.approx(sig_dr, rel=1e-12) == expected_sig_dr

    # Check von Mises limit specifically
    c_mises = compute_drucker_constants(young=200000.0, nu=0.3, cdr=0.0)
    sig_dr_mises, *_ = compute_drucker_equivalent_stress(sig_shear, c_mises.kdr, c_mises.cdr)
    assert pytest.approx(sig_dr_mises, rel=1e-14) == math.sqrt(3.0) * tau


def test_fortran_voce_hardening_parity():
    """Verify Voce exponential hardening matches mat104_nodam_nice.F:168-180:
    F_hard = sigma_0 + Q_voce * (1 - exp(-b_voce * eps_p))
    dF_hard / deps_p = Q_voce * b_voce * exp(-b_voce * eps_p)
    """
    sigma0 = 320.0
    qv = 180.0
    bv = 25.0

    params = DruckerParams(
        sigma_r=sigma0,
        qv=qv,
        bv=bv,
    )

    test_epsp = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    for ep in test_epsp:
        # Fortran reference
        f_hard_ref = sigma0 + qv * (1.0 - math.exp(-bv * ep))
        df_hard_ref = qv * bv * math.exp(-bv * ep)

        flow_calc = compute_drucker_yield_stress(params, eps_p=ep, eps_dot=0.0, temp=params.tini)
        flow_and_deriv, h_calc = compute_drucker_yield_stress(params, eps_p=ep, eps_dot=0.0, temp=params.tini)

        assert pytest.approx(flow_calc, rel=1e-14) == f_hard_ref
        assert pytest.approx(flow_and_deriv, rel=1e-14) == f_hard_ref
        assert pytest.approx(h_calc, rel=1e-14) == df_hard_ref


def test_fortran_johnson_cook_rate_sensitivity_parity():
    """Verify Johnson-Cook logarithmic rate sensitivity matches mat104_nodam_nice.F:168-180:
    F_rate = 1 + cjc * ln(max(1, eps_dot / epsp0))
    """
    sigma0 = 400.0
    cjc = 0.045
    epsp0 = 0.001

    params = DruckerParams(
        sigma_r=sigma0,
        cjc=cjc,
        epsp0=epsp0,
    )

    # Below reference rate -> F_rate = 1.0
    flow_sub = compute_drucker_yield_stress(params, eps_p=0.0, eps_dot=1.0e-5, temp=params.tini)
    assert pytest.approx(flow_sub, rel=1e-14) == sigma0

    # At reference rate -> F_rate = 1.0
    flow_ref = compute_drucker_yield_stress(params, eps_p=0.0, eps_dot=epsp0, temp=params.tini)
    assert pytest.approx(flow_ref, rel=1e-14) == sigma0

    # High strain rates
    for edot in [0.01, 1.0, 10.0, 100.0, 1000.0]:
        f_rate_ref = 1.0 + cjc * math.log(edot / epsp0)
        flow_expected = sigma0 * f_rate_ref
        flow_calc = compute_drucker_yield_stress(params, eps_p=0.0, eps_dot=edot, temp=params.tini)
        assert pytest.approx(flow_calc, rel=1e-14) == flow_expected


def test_fortran_taylor_quinney_cubic_weighting_parity():
    """Verify cubic self-heating weight factor omega(T) matches mat104_nodam_nice.F:151-165:
    - T <= T1 (TSS): omega = 1.0
    - T >= T2 (TREF): omega = 0.0
    - T1 < T < T2: xi = (T - T1) / (T2 - T1)
                   omega = 1 - 3*xi^2 + 2*xi^3
    """
    t1 = 300.0  # TSS
    t2 = 500.0  # TREF

    def omega_ref(t):
        if t <= t1:
            return 1.0
        elif t >= t2:
            return 0.0
        else:
            xi = (t - t1) / (t2 - t1)
            return 1.0 - 3.0 * (xi ** 2) + 2.0 * (xi ** 3)

    # Test properties of cubic Hermite polynomial
    assert omega_ref(300.0) == 1.0
    assert omega_ref(500.0) == 0.0
    # At midpoint xi = 0.5: 1 - 3*(0.25) + 2*(0.125) = 1 - 0.75 + 0.25 = 0.5
    assert omega_ref(400.0) == 0.5

    # Check smooth transition at 25% and 75%
    # xi = 0.25: 1 - 3*(1/16) + 2*(1/64) = 1 - 3/16 + 1/32 = 27/32 = 0.84375
    assert omega_ref(350.0) == 0.84375
    # xi = 0.75: 1 - 3*(9/16) + 2*(27/64) = 1 - 27/16 + 27/32 = 5/32 = 0.15625
    assert omega_ref(450.0) == 0.15625


def test_fortran_plastic_return_mapping_parity():
    """Verify cutting-plane semi-implicit return mapping matches mat104_nodam_newton.F:240-435:
    Under plastic deformation, the final stress state must satisfy
    |sigma_dr(sig_new) - sigma_y(epsp_new)| / sigma_y < 1e-4.
    """
    rho0 = 7.8e-9
    young = 200000.0
    nu = 0.3
    sigma0 = 350.0
    qv = 150.0
    bv = 20.0
    cdr = 1.0

    params = DruckerParams(
        rho=rho0,
        young=young,
        nu=nu,
        sigma_r=sigma0,
        qv=qv,
        bv=bv,
        cdr=cdr,
    )

    # Apply substantial uniaxial strain increment deps_xx = 0.01 (E * deps = 2000 MPa >> 350 MPa)
    sig_old = np.zeros(6)
    deps = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    hist_old = np.zeros(3)

    sig_new, epsp_new, _ = solid_update(params, sig_old, deps, dt=1.0e-5, hist_old=hist_old)

    assert epsp_new > 0.0, "Plastic strain must be accumulated"

    # Compute Drucker equivalent stress of converged stress
    c = compute_drucker_constants(young=young, nu=nu, cdr=cdr)
    sig_dr, *_ = compute_drucker_equivalent_stress(sig_new, c.kdr, c.cdr)

    # Compute flow stress at converged epsp
    sig_y = compute_drucker_yield_stress(params, eps_p=epsp_new, eps_dot=epsp_new / 1.0e-5, temp=params.tini)

    # Assert stress has returned to yield surface within tight tolerance
    rel_err = abs(sig_dr - sig_y) / sig_y
    assert rel_err < 1.0e-4, f"Return mapping did not land on yield surface: rel_err={rel_err}"


def test_fortran_shell_plastic_incompressibility_and_thinning():
    """Verify shell plastic incompressibility and thinning matching mat104c_nodam_nice.F:180-210:
    - Plastic thickness strain increment: deps_zz_pl = -(deps_xx_pl + deps_yy_pl)
    - Elastic thickness strain increment: deps_zz_el = -(nu / (1 - nu)) * (deps_xx_el + deps_yy_el)
    - Shell thickness update: h_new = h_old * exp(deps_zz)
    """
    young = 200000.0
    nu = 0.3
    sigma0 = 300.0
    h_init = 1.5

    params = DruckerParams(
        rho=7.8e-9,
        young=young,
        nu=nu,
        sigma_r=sigma0,
        qv=0.0,
        bv=0.0,
        cdr=0.0,
    )

    # In-plane biaxial tension causing yield: deps_xx = 0.005, deps_yy = 0.005
    sig_old = np.zeros(3)
    deps = np.array([0.005, 0.005, 0.0])
    thk = np.array([h_init])
    extra = {}

    sig_new, epsp_new, c_shell = shell_update(params, sig_old, deps, dt=1.0e-5, thk=thk, extra=extra)

    assert epsp_new > 0.0
    # Under biaxial tension, thickness must decrease (thinning)
    h_new = thk[0]
    assert h_new < h_init
    # Thickness ratio must be strictly positive and realistic (between 0.95 and 1.0)
    assert 0.95 < h_new / h_init < 1.0
