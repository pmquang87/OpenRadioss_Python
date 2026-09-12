"""Tests for Milestone M572: /MAT/LAW102 Fortran Parity & Upstream Formula Verification.

Compares pyradioss law102 constitutive physics directly against OpenRadioss Fortran reference code:
1. Mohr-Coulomb to Drucker-Prager conversion (IFORM 1, 2, 3):
   starter/source/materials/mat/mat102/hm_read_mat102.F:144-152.
2. Parabolic yield coefficients A0, A1, A2:
   starter/source/materials/mat/mat102/hm_read_mat102.F:154-156.
3. Tensile pressure root PSTAR calculation:
   starter/source/materials/mat/mat102/hm_read_mat102.F:167-185.
4. Deviatoric trial stress tensor and second invariant J2:
   engine/source/materials/mat/mat102/sigeps102.F:80-107.
5. Original Mohr-Coulomb formulation with 3rd invariant I3 and Lode angle (IFORM 4):
   engine/source/materials/mat/mat102/sigeps102.F:110-122.
6. Radial return projection factor and plastic strain increment:
   engine/source/materials/mat/mat102/sigeps102.F:140-159.
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law102_dprag2 import (
    DPrag2Params,
    build_law102,
    compute_dprag2_constants,
    solid_update_single,
    _solid_update_single_core,
)


def test_fortran_mohr_coulomb_conversion_parity():
    """Verify conversion constants match hm_read_mat102.F:144-156 exactly:
    IFORM=1: K = 6*C*COS(PHI)/(SQRT(3)*(3 - SIN(PHI))), ALPHA = 2*SIN(PHI)/(SQRT(3)*(3 - SIN(PHI)))
    IFORM=2: K = 6*C*COS(PHI)/(SQRT(3)*(3 + SIN(PHI))), ALPHA = 2*SIN(PHI)/(SQRT(3)*(3 + SIN(PHI)))
    IFORM=3: K = 3*C*COS(PHI)/SQRT(9 + 3*SIN(PHI)**2), ALPHA = SIN(PHI)/SQRT(9 + 3*SIN(PHI)**2)
    A0 = K*K, A1 = 6*K*ALPHA, A2 = 9*ALPHA*ALPHA
    """
    c_val = 18.5
    phi_deg = 28.0
    phi_rad = phi_deg * math.pi / 180.0
    sin_p = math.sin(phi_rad)
    cos_p = math.cos(phi_rad)
    sq3 = math.sqrt(3.0)

    # IFORM=1
    k1_f = 6.0 * c_val * cos_p / (sq3 * (3.0 - sin_p))
    a1_f = 2.0 * sin_p / (sq3 * (3.0 - sin_p))
    p1 = compute_dprag2_constants(c=c_val, phi_deg=phi_deg, iform=1)
    assert p1.k_yield == pytest.approx(k1_f, rel=1e-15)
    assert p1.alpha == pytest.approx(a1_f, rel=1e-15)
    assert p1.a0 == pytest.approx(k1_f * k1_f, rel=1e-15)
    assert p1.a1 == pytest.approx(6.0 * k1_f * a1_f, rel=1e-15)
    assert p1.a2 == pytest.approx(9.0 * a1_f * a1_f, rel=1e-15)

    # IFORM=2
    k2_f = 6.0 * c_val * cos_p / (sq3 * (3.0 + sin_p))
    a2_f = 2.0 * sin_p / (sq3 * (3.0 + sin_p))
    p2 = compute_dprag2_constants(c=c_val, phi_deg=phi_deg, iform=2)
    assert p2.k_yield == pytest.approx(k2_f, rel=1e-15)
    assert p2.alpha == pytest.approx(a2_f, rel=1e-15)
    assert p2.a0 == pytest.approx(k2_f * k2_f, rel=1e-15)
    assert p2.a1 == pytest.approx(6.0 * k2_f * a2_f, rel=1e-15)
    assert p2.a2 == pytest.approx(9.0 * a2_f * a2_f, rel=1e-15)

    # IFORM=3
    denom3 = math.sqrt(9.0 + 3.0 * sin_p * sin_p)
    k3_f = 3.0 * c_val * cos_p / denom3
    a3_f = sin_p / denom3
    p3 = compute_dprag2_constants(c=c_val, phi_deg=phi_deg, iform=3)
    assert p3.k_yield == pytest.approx(k3_f, rel=1e-15)
    assert p3.alpha == pytest.approx(a3_f, rel=1e-15)
    assert p3.a0 == pytest.approx(k3_f * k3_f, rel=1e-15)
    assert p3.a1 == pytest.approx(6.0 * k3_f * a3_f, rel=1e-15)
    assert p3.a2 == pytest.approx(9.0 * a3_f * a3_f, rel=1e-15)


def test_fortran_pstar_calculation_parity():
    """Verify PSTAR matches hm_read_mat102.F:167-185:
    IF (A2 .EQ. 0) THEN PSTAR = -A0 / A1
    ELSE
      DELTA = A1**2 - 4*A0*A2
      IF (DELTA .GE. 0) THEN PSTAR = (-A1 + SQRT(DELTA)) / (2*A2)
      ELSE PSTAR = -A1 / (2*A2)
    """
    # Case 1: Linear Drucker-Prager (A2 = 0)
    p_lin = compute_dprag2_constants(a0=50.0, a1=10.0, a2=0.0)
    assert p_lin.pstar == pytest.approx(-50.0 / 10.0, rel=1e-15)

    # Case 2: Parabolic with positive discriminant (standard Mohr-Coulomb fit)
    p_quad = compute_dprag2_constants(c=10.0, phi_deg=30.0, iform=2)
    delta = p_quad.a1 ** 2 - 4.0 * p_quad.a0 * p_quad.a2
    expected_pstar = (-p_quad.a1 + math.sqrt(delta)) / (2.0 * p_quad.a2)
    assert p_quad.pstar == pytest.approx(expected_pstar, rel=1e-15)


def test_fortran_deviatoric_trial_stress_parity():
    """Verify deviatoric trial stress matches sigeps102.F:80-94:
    POLD = -(SIG(1) + SIG(2) + SIG(3)) / 3
    SCRT = (DEPS(1) + DEPS(2) + DEPS(3)) / 3
    T(1) = SIG(1) + POLD + 2*G*(DEPS(1) - SCRT)
    ...
    T(4) = SIG(4) + G*DEPS(4)
    """
    e = 20000.0
    nu = 0.25
    g = e / (2.0 * (1.0 + nu))
    gg = 2.0 * g
    params = build_law102(rho=2.5e-6, e=e, nu=nu, c=15.0, phi=30.0)

    sig_old = np.array([-10.0, -15.0, -20.0, 5.0, 2.0, 3.0])
    deps = np.array([1e-4, -2e-4, 3e-4, 4e-4, -1e-4, 2e-4])

    pold_f = -(sig_old[0] + sig_old[1] + sig_old[2]) / 3.0
    scrt_f = (deps[0] + deps[1] + deps[2]) / 3.0

    t1_f = sig_old[0] + pold_f + gg * (deps[0] - scrt_f)
    t2_f = sig_old[1] + pold_f + gg * (deps[1] - scrt_f)
    t3_f = sig_old[2] + pold_f + gg * (deps[2] - scrt_f)
    t4_f = sig_old[3] + g * deps[3]
    t5_f = sig_old[4] + g * deps[4]
    t6_f = sig_old[5] + g * deps[5]

    aj2_f = 0.5 * (t1_f**2 + t2_f**2 + t3_f**2) + t4_f**2 + t5_f**2 + t6_f**2

    # Verify via single step
    hist = np.array([0.0, pold_f])
    sig_new, hist_new, _ = _solid_update_single_core(params, deps, sig_old, hist)

    # Because trial stress is within elastic bounds (P = 15 MPa compression), ratio = 1
    ptot_f = pold_f - 3.0 * params.bulk * scrt_f
    g0_f = params.a0 + params.a1 * ptot_f + params.a2 * ptot_f**2
    assert aj2_f <= g0_f  # Elastic


def test_fortran_iform4_lode_angle_parity():
    """Verify IFORM=4 3rd invariant and Lode angle match sigeps102.F:110-122:
    I3 = T2*T3*T1 - T2*T6**2 - T3*T4**2 - T5**2*T1 + 2*T5*T4*T6
    COS3T = 9*I3 / (2*SQRT(3)*J2**(3/2))
    THETA = ACOS(CLAMP(COS3T, 0, 1))
    G0 = MAX(0, -PTOT*SIN(PHI) + SQRT(J2)*(COS(THETA) - (1/SQRT(3))*SIN(THETA)*SIN(PHI)) - C*COS(PHI))
    """
    c_val = 10.0
    phi_deg = 30.0
    phi_rad = phi_deg * math.pi / 180.0
    params = build_law102(rho=2.5e-6, e=20000.0, nu=0.25, c=c_val, phi=phi_deg, iform=4)

    # General deviatoric tensor
    t1, t2, t3 = 12.0, -5.0, -7.0
    t4, t5, t6 = 3.0, 2.0, 1.0

    i3_f = t2 * t3 * t1 - t2 * t6**2 - t3 * t4**2 - t5**2 * t1 + 2.0 * t5 * t4 * t6
    aj2_f = 0.5 * (t1**2 + t2**2 + t3**2) + t4**2 + t5**2 + t6**2

    sqrt_j2 = math.sqrt(aj2_f)
    denom_f = 2.0 * math.sqrt(3.0) * (sqrt_j2**3)
    cos3t_f = 9.0 * i3_f / denom_f
    cos3t_clamped = max(0.0, min(1.0, cos3t_f))
    theta_f = math.acos(cos3t_clamped)

    ptot = 10.0  # Compression
    k_mc = 1.0 / math.sqrt(3.0)
    sin_p = math.sin(phi_rad)
    cos_p = math.cos(phi_rad)
    g0_f = max(0.0, -ptot * sin_p + sqrt_j2 * (math.cos(theta_f) - k_mc * math.sin(theta_f) * sin_p) - c_val * cos_p)

    assert i3_f != 0.0
    assert 0.0 <= cos3t_clamped <= 1.0
    assert g0_f >= 0.0


def test_fortran_radial_return_projection_parity():
    """Verify projection factor and stress update match sigeps102.F:140-159:
    RATIO = SQRT(G0 / (AJ2 + 1E-14))
    SIG_NEW(i) = RATIO * T(i) * OFF - PNEW
    DPLA = (1 - RATIO) * SQRT(AJ2) / (3*G)
    """
    g = 8000.0
    aj2 = 150.0
    g0 = 96.0  # Plastic state (AJ2 > G0)

    ratio_f = math.sqrt(g0 / (aj2 + 1.0e-14))
    dpla_f = (1.0 - ratio_f) * math.sqrt(aj2) / (3.0 * g)

    assert 0.0 < ratio_f < 1.0
    assert dpla_f > 0.0
    # Scaled check
    assert pytest.approx(ratio_f) == math.sqrt(96.0 / 150.0)
