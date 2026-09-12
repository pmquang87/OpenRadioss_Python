"""
Tests for Milestone M570: /MAT/LAW100 (/MAT/VISC_HYP, /MAT/MNF) Multi-Network Visco-Hyperelastic Model.

Constitutive physics kernel:
- Left Cauchy-Green tensor B and deformation gradient Fe (calc_mat_b)
- Hyperelastic kernels:
  - Polynomial (poly_stress)
  - Arruda-Boyce (arruda_boyce_stress)
  - Neo-Hookean with temperature dependency (neo_hook_t_stress)
- Viscoelastic flow rules:
  - Flag_visc=1: Bergstrom-Boyce (visc_bb)
  - Flag_visc=2: Hyperbolic sine (visc_sinh)
  - Flag_visc=3: Power law (visc_power)
- Equilibrium creep / plasticity
- Multi-network state update, energy balance, sound speed, tangent matrix.
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law100_multi_network import (
    MultiNetworkParams,
    SecondaryNetworkParams,
    build_law100,
    calc_mat_b,
    poly_stress,
    arruda_boyce_stress,
    neo_hook_t_stress,
    compute_he_stress,
    visc_bb,
    visc_sinh,
    visc_power,
    solid_update,
    sound_speed,
    solid_tangent,
)


# ============================================================================
# 1. Kinematics: calc_mat_b & Invariants
# ============================================================================

def test_calc_mat_b_identity():
    """Identity deformation F = I gives B = I, J = 1."""
    F = np.eye(3)
    b, fe = calc_mat_b(F)
    np.testing.assert_allclose(b, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(fe, np.eye(3), atol=1e-12)
    I1 = np.trace(b)
    J = math.sqrt(np.linalg.det(b))
    assert I1 == pytest.approx(3.0, abs=1e-12)
    assert J == pytest.approx(1.0, abs=1e-12)


def test_calc_mat_b_uniaxial():
    """Uniaxial stretch lambda1 = 2, lambda2 = lambda3 = 1/sqrt(2) (isochoric J = 1)."""
    lam1 = 2.0
    lam2 = 1.0 / math.sqrt(2.0)
    F = np.diag([lam1, lam2, lam2])
    b, fe = calc_mat_b(F)
    J = math.sqrt(np.linalg.det(b))
    assert J == pytest.approx(1.0, abs=1e-12)
    I1 = np.trace(b)
    expected_I1 = lam1**2 + 2 * lam2**2  # 4 + 1 = 5
    assert I1 == pytest.approx(expected_I1, rel=1e-6)


def test_calc_mat_b_shear():
    """Simple shear F = [[1, gamma, 0], [0, 1, 0], [0, 0, 1]], J = 1."""
    gamma = 0.5
    F = np.array([[1.0, gamma, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    b, fe = calc_mat_b(F)
    J = math.sqrt(np.linalg.det(b))
    assert J == pytest.approx(1.0, abs=1e-12)
    I1 = np.trace(b)
    expected_I1 = 3.0 + gamma**2
    assert I1 == pytest.approx(expected_I1, rel=1e-6)


# ============================================================================
# 2. Hyperelastic Kernels
# ============================================================================

def test_poly_stress_zero_at_identity():
    """At F = I, stress is zero."""
    F = np.eye(3)
    b, _ = calc_mat_b(F)
    sigma, sb = poly_stress(
        b, c10=1.0e6, c01=2.0e5, c20=0.0, c11=0.0, c02=0.0,
        c30=0.0, c21=0.0, c12=0.0, c03=0.0,
        d1=1.0e7, d2=0.0, d3=0.0, rbulk=2.0e7, iform=1,
    )
    np.testing.assert_allclose(sigma, np.zeros((3, 3)), atol=1e-10)


def test_poly_stress_pure_hydrostatic():
    """Pure volumetric compression F = (1 - eps)*I -> only pressure develops."""
    J_target = 0.95
    lam = J_target ** (1.0 / 3.0)
    F = np.eye(3) * lam
    b, _ = calc_mat_b(F)
    d1 = 1.0e7
    sigma, sb = poly_stress(
        b, c10=1.0e6, c01=0.0, c20=0.0, c11=0.0, c02=0.0,
        c30=0.0, c21=0.0, c12=0.0, c03=0.0,
        d1=d1, d2=0.0, d3=0.0, rbulk=2.0e7, iform=1,
    )
    p_mean = np.trace(sigma) / 3.0
    dev_sigma = sigma - p_mean * np.eye(3)
    np.testing.assert_allclose(dev_sigma, np.zeros((3, 3)), atol=1e-6)
    assert p_mean < 0.0  # compressive stress


def test_arruda_boyce_stress_identity():
    """Arruda-Boyce stress at F = I is zero."""
    F = np.eye(3)
    b, _ = calc_mat_b(F)
    sigma, sb = arruda_boyce_stress(b, mu=1.0e6, lambda_m=7.0, d_inv=1.0e7)
    np.testing.assert_allclose(sigma, np.zeros((3, 3)), atol=1e-10)


def test_arruda_boyce_extensibility_stiffening():
    """Arruda-Boyce stress stiffens dramatically as stretch approaches lambda_m."""
    mu = 1.0e6
    lambda_m = 3.0
    d_inv = 1.0e7

    # Stretch 1: moderate lambda = 1.3
    lam1 = 1.3
    F1 = np.diag([lam1, 1.0 / math.sqrt(lam1), 1.0 / math.sqrt(lam1)])
    b1, _ = calc_mat_b(F1)
    sig1, _ = arruda_boyce_stress(b1, mu=mu, lambda_m=lambda_m, d_inv=d_inv)

    # Stretch 2: close to locking lambda = 2.7
    lam2 = 2.7
    F2 = np.diag([lam2, 1.0 / math.sqrt(lam2), 1.0 / math.sqrt(lam2)])
    b2, _ = calc_mat_b(F2)
    sig2, _ = arruda_boyce_stress(b2, mu=mu, lambda_m=lambda_m, d_inv=d_inv)

    # Stress ratio should exceed linear ratio showing non-linear upturn
    stress_ratio = sig2[0, 0] / sig1[0, 0]
    strain_ratio = (lam2 - 1.0) / (lam1 - 1.0)
    assert stress_ratio > strain_ratio * 1.5


def test_neo_hook_t_stress():
    """Basic/thermal Neo-Hookean stress."""
    F = np.diag([1.2, 1.0, 1.0])
    b, _ = calc_mat_b(F)
    sig1, _ = neo_hook_t_stress(b, mu=1.0e6, d_bulk=1.0e7)
    sig2, _ = neo_hook_t_stress(b, mu=2.0e6, d_bulk=2.0e7)
    np.testing.assert_allclose(sig2, 2.0 * sig1, rtol=1e-5)


# ============================================================================
# 3. Viscoelastic Relaxation Flow Rules
# ============================================================================

def test_visc_bb_flow():
    """Bergstrom-Boyce viscbb flow rate positive and monotonically increasing with stress."""
    fp = np.eye(3)
    rate1 = visc_bb(fp=fp, tbnorm=1.0e5, a1=0.01, expc=-0.7, expm=2.0, ksi=0.01, tauref=1.0e5)
    rate2 = visc_bb(fp=fp, tbnorm=2.0e5, a1=0.01, expc=-0.7, expm=2.0, ksi=0.01, tauref=1.0e5)
    assert rate1 > 0.0
    assert rate2 > rate1
    # For expm=2, doubling tbnorm quadruples the stress ratio
    assert rate2 == pytest.approx(4.0 * rate1, rel=1e-5)


def test_visc_sinh_flow():
    """Hyperbolic sine flow rate visc_sinh."""
    rate1 = visc_sinh(tbnorm=1.0e5, a1=0.01, b0=1.0e-5, expn=1.0)
    rate2 = visc_sinh(tbnorm=2.0e5, a1=0.01, b0=1.0e-5, expn=1.0)
    assert rate1 > 0.0
    assert rate2 > rate1
    # sinh(2) / sinh(1) = 3.62686 / 1.17520 ~ 3.086
    expected_ratio = math.sinh(2.0) / math.sinh(1.0)
    assert rate2 / rate1 == pytest.approx(expected_ratio, rel=1e-5)


def test_visc_power_flow():
    """Power-law flow rate visc_power."""
    rate1 = visc_power(tbnorm=1.0e5, a1=1.0e-8, expm=0.0, expn=2.0, gammaold=1.0e-6)
    rate2 = visc_power(tbnorm=2.0e5, a1=1.0e-8, expm=0.0, expn=2.0, gammaold=1.0e-6)
    assert rate1 > 0.0
    assert rate2 > rate1
    assert rate2 == pytest.approx(4.0 * rate1, rel=1e-5)


# ============================================================================
# 4. Multi-Network Assembly & Stress Relaxation
# ============================================================================

def test_solid_update_pure_elastic():
    """LAW100 with N_net = 0 behaves as pure hyperelastic model."""
    params = build_law100(
        id=1, rho0=1000.0, flag_he=1, c10=1.0e6, c01=2.0e5, d1=1.0e-7, n_net=0
    )
    F = np.diag([1.1, 1.0 / math.sqrt(1.1), 1.0 / math.sqrt(1.1)])
    extra = {"F": F}
    sig1, epsp1, c_sound = solid_update(params, dt=1e-4, extra=extra)
    # Step 2 at same deformation: stress should not relax because N_net = 0
    sig2, epsp2, _ = solid_update(params, dt=1e-4, extra=extra)
    np.testing.assert_allclose(sig1, sig2, atol=1e-10)


def test_solid_update_stress_relaxation():
    """LAW100 with secondary network relaxes stress monotonically under constant deformation."""
    sec = SecondaryNetworkParams(network_id=1, flag_visc=1, stiffness=1.0, a=0.1, expc=-0.7, expm=1.0, ksi=0.01, tauref=1.0e5)
    params = build_law100(
        id=1, rho0=1000.0, flag_he=1, c10=1.0e6, c01=0.0, d1=1.0e-7,
        n_net=1, networks=[sec]
    )
    eps = np.array([0.1, -0.05, -0.05, 0.0, 0.0, 0.0])
    extra = {}
    dt = 1.0e-4

    stress_history = []
    # Hold strain constant for 10 steps
    for _ in range(10):
        sig_step, _, _ = solid_update(params, eps=eps, dt=dt, extra=extra)
        stress_history.append(sig_step[0])

    # Stresses must strictly decrease over time due to viscoelastic relaxation
    for i in range(len(stress_history) - 1):
        assert stress_history[i] > stress_history[i + 1]


# ============================================================================
# 5. Acoustic Wave Speed and Consistent Tangent
# ============================================================================

def test_law100_sound_speed():
    """Dilatational sound speed c = sqrt((K + 4/3 G) / rho)."""
    c10 = 1.0e6
    c01 = 2.0e5
    d1 = 1.0e-7
    rho0 = 1000.0
    params = build_law100(id=1, rho0=rho0, flag_he=1, c10=c10, c01=c01, d1=d1)
    c = sound_speed(params)
    assert c > 0.0
    G = 2.0 * (c10 + c01)
    # Bulk modulus from params.rbulk
    K = params.rbulk
    expected_c = math.sqrt((K + 4.0 / 3.0 * G) / rho0)
    assert c == pytest.approx(expected_c, rel=1e-5)


def test_law100_tangent_positive_definite():
    """Material tangent matrix C_ijkl is positive-definite at F = I."""
    params = build_law100(id=1, rho0=1000.0, flag_he=1, c10=1.0e6, c01=2.0e5, d1=1.0e-7)
    C = solid_tangent(params, np.eye(3))
    assert C.shape == (6, 6)
    # Check eigenvalues of 6x6 Voigt tangent matrix
    eigvals = np.linalg.eigvalsh(C)
    assert np.all(eigvals > 0.0)
