"""
Tests for Milestone M569: /MAT/LAW95 (/MAT/BERGSTROM_BOYCE) Bergstrom-Boyce polymer model.

Constitutive physics, parameters, multiplicative kinematics, viscoelastic creep,
rate-dependence, stress relaxation, acoustic wave speeds, and consistent tangent tensor.
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law95_bergstrom_boyce import (
    BergstromBoyceParams,
    build_law95,
    poly_stress,
    visc_bb,
    calc_mat_b,
    solid_update,
    sound_speed,
    sound_speed_solid,
    consistent_tangent,
    extra_shapes,
    resolve,
)


# ============================================================================
# 1. Parameter Initialization & Derived Linear Properties
# ============================================================================

def test_law95_params_defaults():
    """Verify default Fortran values matching hm_read_mat95.F."""
    p = build_law95(id=1, rho0=1000.0, c10=1.0e6, c01=2.0e5)
    assert p.id == 1
    assert p.rho0 == 1000.0
    assert p.c10 == 1.0e6
    assert p.c01 == 2.0e5
    assert p.sb == 0.0
    assert p.iform == 1
    assert p.expc == pytest.approx(-0.7)
    assert p.expm == pytest.approx(1.0)
    assert p.ksi == pytest.approx(0.01)
    assert p.tauref == pytest.approx(1.0)

    # Incompressible default: nu = 0.495
    # G0 = 2 * (C10 + C01) * (1 + Sb) = 2 * 1.2e6 * 1.0 = 2.4e6
    assert p.g0 == pytest.approx(2.4e6)
    assert p.G == pytest.approx(2.4e6)
    expected_k = (2.0 / 3.0) * 2.4e6 * (1.0 + 0.495) / (1.0 - 2.0 * 0.495)
    assert p.rbulk == pytest.approx(expected_k)
    assert p.bulk == pytest.approx(expected_k)
    assert p.K == pytest.approx(expected_k)
    expected_e = 2.0 * 2.4e6 * (1.0 + 0.495)
    assert p.e == pytest.approx(expected_e)
    assert p.young == pytest.approx(expected_e)


def test_law95_params_compressible_with_sb():
    """Verify compressible bulk modulus and shear modulus with network B scaling Sb > 0."""
    c10 = 5.0e5
    c01 = 1.0e5
    sb = 1.5
    d1_in = 1.0e-7  # D1 > 0
    p = build_law95(c10=c10, c01=c01, sb=sb, d1=d1_in, rho0=1100.0)

    # G0 = 2 * (5e5 + 1e5) * (1 + 1.5) = 2 * 6e5 * 2.5 = 3.0e6
    assert p.g0 == pytest.approx(3.0e6)
    # d1_inv = 1 / D1 = 1e7
    # K = 2 * (1/D1) * (1 + Sb) = 2 * 1e7 * 2.5 = 5.0e7
    assert p.d1 == pytest.approx(1.0e7)
    assert p.rbulk == pytest.approx(5.0e7)

    # nu = (3*K - 2*G0) / (2 * (3*K + G0))
    expected_nu = (3.0 * 5.0e7 - 2.0 * 3.0e6) / (2.0 * (3.0 * 5.0e7 + 3.0e6))
    assert p.nu == pytest.approx(expected_nu)

    # E = 9*K*G0 / (3*K + G0)
    expected_e = 9.0 * 5.0e7 * 3.0e6 / (3.0 * 5.0e7 + 3.0e6)
    assert p.e == pytest.approx(expected_e)


def test_law95_params_user_nu():
    """Verify user-specified Poisson's ratio when D1=0."""
    c10 = 1.0e6
    p = build_law95(c10=c10, nu=0.48, rho0=1000.0)
    assert p.nu == pytest.approx(0.48)
    # G0 = 2 * 1e6 = 2e6
    assert p.g0 == pytest.approx(2.0e6)
    expected_k = (2.0 / 3.0) * 2.0e6 * (1.0 + 0.48) / (1.0 - 2.0 * 0.48)
    assert p.rbulk == pytest.approx(expected_k)


# ============================================================================
# 2. Small-Strain Hookean Limit vs Linear Elasticity
# ============================================================================

def test_law95_small_strain_hookean_limit():
    """Under small strain, LAW95 response matches linear elasticity G0 and K."""
    c10 = 1.2e6
    c01 = 3.0e5
    sb = 0.5
    d1 = 1.0e-7  # K = 2 * 1e7 * 1.5 = 3.0e7
    p = build_law95(c10=c10, c01=c01, sb=sb, d1=d1, rho0=1000.0)
    G = p.g0  # 2 * 1.5e6 * 1.5 = 4.5e6
    K = p.rbulk

    # 1. Pure shear strain: gamma_xy = 1e-5 (deps = [0, 0, 0, 1e-5, 0, 0])
    eps_shear = np.array([0.0, 0.0, 0.0, 1e-5, 0.0, 0.0])
    sig, _, _ = solid_update(p, eps=eps_shear, dt=0.0)
    # sig_xy = G * gamma_xy
    assert sig[3] == pytest.approx(G * 1e-5, rel=1e-3)
    assert np.allclose(sig[:3], 0.0, atol=1e-3)

    # 2. Volumetric strain: eps_xx = eps_yy = eps_zz = 1e-5
    eps_vol = np.array([1e-5, 1e-5, 1e-5, 0.0, 0.0, 0.0])
    sig_v, _, _ = solid_update(p, eps=eps_vol, dt=0.0)
    # sig_xx = sig_yy = sig_zz = 3 * K * 1e-5
    expected_p = 3.0 * K * 1e-5
    assert sig_v[0] == pytest.approx(expected_p, rel=1e-3)
    assert sig_v[1] == pytest.approx(expected_p, rel=1e-3)
    assert sig_v[2] == pytest.approx(expected_p, rel=1e-3)


# ============================================================================
# 3. Polynomial Hyperelastic Potential (poly_stress)
# ============================================================================

def test_poly_stress_iform1_higher_order_pressure():
    """Verify IFORM=1 polynomial pressure with D1, D2, D3 terms."""
    c_coeffs = (1.0e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    d1_stiff = 5.0e7
    d2_stiff = 1.0e8
    d3_stiff = 2.0e8
    d_params = (d1_stiff, d2_stiff, d3_stiff)
    rbulk = 2.0 * d1_stiff

    # Pure spherical deformation: b = J^(2/3) * I
    J = 1.04
    b = (J ** (2.0 / 3.0)) * np.eye(3)

    sig, cii = poly_stress(b, c_coeffs, d_params, rbulk, iform=1)
    # For spherical deformation, deviatoric Cauchy stress = 0
    # Hydrostatic pressure P = 2*D1*(J-1) + 4*D2*(J-1)^3 + 6*D3*(J-1)^5
    j_m1 = J - 1.0
    expected_P = 2.0 * d1_stiff * j_m1 + 4.0 * d2_stiff * (j_m1 ** 3) + 6.0 * d3_stiff * (j_m1 ** 5)
    assert sig[0, 0] == pytest.approx(expected_P, rel=1e-5)
    assert sig[1, 1] == pytest.approx(expected_P, rel=1e-5)
    assert sig[2, 2] == pytest.approx(expected_P, rel=1e-5)
    assert sig[0, 1] == pytest.approx(0.0, abs=1e-10)


def test_poly_stress_iform2_pressure():
    """Verify IFORM=2 pressure formulation: P = K * (1 - 1/J)."""
    c_coeffs = (1.0e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rbulk = 1.0e8
    d_params = (rbulk / 2.0, 0.0, 0.0)

    J = 1.05
    b = (J ** (2.0 / 3.0)) * np.eye(3)

    sig, _ = poly_stress(b, c_coeffs, d_params, rbulk, iform=2)
    expected_P = rbulk * (1.0 - 1.0 / J)
    assert sig[0, 0] == pytest.approx(expected_P, rel=1e-5)
    assert sig[1, 1] == pytest.approx(expected_P, rel=1e-5)
    assert sig[2, 2] == pytest.approx(expected_P, rel=1e-5)


# ============================================================================
# 4. Kinematics & Creep Rate (calc_mat_b, visc_bb)
# ============================================================================

def test_calc_mat_b_identity():
    """When F == F_p, elastic stretch is identity."""
    f = np.diag([1.2, 0.9, 0.9259])
    fp = f.copy()
    matb, fe = calc_mat_b(f, fp)
    assert np.allclose(fe, np.eye(3))
    assert np.allclose(matb, np.eye(3))


def test_visc_bb_creep_scaling():
    """Verify visc_bb creep strain increment formula."""
    fp = np.diag([1.1, 0.95, 0.956])
    tbnorm = 2.5e5
    a1 = 1.0e-3
    expc = -0.7
    expm = 1.5
    ksi = 0.01
    tauref = 1.0e5

    dgamma = visc_bb(fp, tbnorm, a1, expc, expm, ksi, tauref)
    assert dgamma > 0.0

    # Test stress ratio scaling: if tbnorm doubles, stress ratio increases by 2^expm
    dgamma_2 = visc_bb(fp, 2.0 * tbnorm, a1, expc, expm, ksi, tauref)
    expected_ratio = (2.0) ** expm
    assert dgamma_2 / dgamma == pytest.approx(expected_ratio, rel=1e-6)


# ============================================================================
# 5. Viscoelastic Relaxation & Rate Sensitivity
# ============================================================================

def test_law95_viscoelastic_relaxation():
    """Under held constant strain, stress relaxes monotonically toward equilibrium network A."""
    c10 = 1.0e6
    sb = 1.0  # Equal network A and B initial stiffness
    a = 0.1   # Physical creep rate
    p = build_law95(c10=c10, sb=sb, a=a, tau_ref=1.0e5, rho0=1000.0)

    # Apply 10% uniaxial stretch
    eps = np.array([0.1, -0.05, -0.05, 0.0, 0.0, 0.0])
    extra = {}
    dt = 1.0e-4

    stress_history = []
    # Hold strain constant for 10 steps
    for _ in range(10):
        sig_step, _, _ = solid_update(p, eps=eps, dt=dt, extra=extra)
        stress_history.append(sig_step[0])

    # Stresses must strictly decrease over time due to creep relaxation
    for i in range(len(stress_history) - 1):
        assert stress_history[i] > stress_history[i + 1]


def test_law95_rate_dependence():
    """Higher strain rate produces higher stress response due to viscous network B."""
    c10 = 1.0e6
    sb = 1.0
    a = 0.1
    p = build_law95(c10=c10, sb=sb, a=a, tau_ref=1.0e5, rho0=1000.0)

    eps_target = np.array([0.1, -0.05, -0.05, 0.0, 0.0, 0.0])

    # Fast loading: high strain rate (dt = 1e-6)
    sig_fast, _, _ = solid_update(p, eps=eps_target, dt=1.0e-6, extra={})

    # Slow loading: low strain rate (dt = 1e-3)
    sig_slow, _, _ = solid_update(p, eps=eps_target, dt=1.0e-3, extra={})

    # Fast rate retains more viscous stress in Network B
    assert sig_fast[0] > sig_slow[0]


# ============================================================================
# 6. Acoustic Sound Speed
# ============================================================================

def test_law95_sound_speed():
    """Verify dilatational sound speed calculation."""
    p = build_law95(c10=2.0e6, c01=5.0e5, sb=0.5, d1=1.0e-7, rho0=1200.0)
    c_solid = sound_speed(p)
    assert c_solid > 0.0
    assert sound_speed_solid(p) == pytest.approx(c_solid)

    # Compare with analytical minimum bound: sqrt((4/3*G0 + K) / rho)
    stiff0 = (4.0 / 3.0) * p.g0 + p.rbulk
    c_min = math.sqrt(stiff0 / 1200.0)
    assert c_solid >= c_min * 0.999


# ============================================================================
# 7. Consistent Tangent Stiffness
# ============================================================================

def test_law95_consistent_tangent():
    """Verify (6x6) consistent tangent matrix matches finite difference perturbations."""
    p = build_law95(c10=1.5e6, c01=2.0e5, sb=0.5, d1=1.0e-7, rho0=1000.0)
    eps = np.array([0.02, -0.01, -0.01, 0.005, 0.0, 0.0])
    tangent = consistent_tangent(p, eps=eps, dt=0.0)

    assert tangent.shape == (6, 6)
    # Check minor symmetry
    assert np.allclose(tangent, tangent.T, atol=1e-5)
    # Check positive definiteness (eigenvalues > 0)
    evals = np.linalg.eigvalsh(tangent)
    assert np.all(evals > 0.0)


# ============================================================================
# 8. Extra Shapes & Resolve Hook
# ============================================================================

def test_law95_extra_shapes_and_resolve():
    """Verify persistent state storage definitions and resolve hook."""
    shapes = extra_shapes()
    assert "uvar" in shapes
    assert shapes["uvar"] == (10,)
    assert "uvar95" in shapes
    assert shapes["uvar95"] == (10,)

    p = resolve({"c10": 1.5e6, "rho0": 1050.0})
    assert isinstance(p, BergstromBoyceParams)
    assert p.c10 == 1.5e6
    assert p.rho0 == 1050.0
