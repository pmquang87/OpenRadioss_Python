"""Unit tests for LAW80: Ramberg-Osgood Elastoplastic Constitutive Model.

Tests:
1. Verification of elastic modulus E at small strains (epsilon << 1, Hooke's limit).
2. Verification of power-law hardening sigma = K * eps_p^n at large strains.
3. Bidirectional analytical inversion between ramberg_osgood_strain and ramberg_osgood_stress.
4. 3D solid continuum constitutive update (solid_update) matching sigeps80.F.
5. 2D plane-stress shell update (shell_update) matching sigeps80c.F.
6. Consistent tangent stiffness operators (solid_tangent, shell_tangent).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.materials import law80_ramberg
from pyradioss.model.entities import Material


# ============================================================================
# 1D Ramberg-Osgood Analytical & Newton Iteration Tests
# ============================================================================

def test_ramberg_osgood_small_strain_elastic_modulus():
    """Verify that at small strains (epsilon << 1), the stress response is linear elastic
    with slope equal to Young's modulus E (Hooke's limit).
    """
    E = 200000.0  # MPa
    K = 1000.0    # MPa
    n = 0.2       # Hardening exponent

    small_strains = [1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5]

    for eps in small_strains:
        # Compute stress via Newton-Raphson inversion of Ramberg-Osgood
        sig = law80_ramberg.ramberg_osgood_stress(eps, E, K, n)
        # Ratio sigma / eps must equal E
        assert pytest.approx(sig / eps, rel=1e-5) == E

        # Tangent modulus dsigma/deps must equal E at small strains
        tan = law80_ramberg.ramberg_osgood_tangent(sig, E, K, n)
        assert pytest.approx(tan, rel=1e-5) == E

    # Negative strain (compression symmetry)
    eps_neg = -1.0e-6
    sig_neg = law80_ramberg.ramberg_osgood_stress(eps_neg, E, K, n)
    assert pytest.approx(sig_neg / eps_neg, rel=1e-5) == E


def test_ramberg_osgood_large_strain_power_law_hardening():
    """Verify that at large strains, plastic deformation dominates and the stress
    follows the power-law hardening relation: sigma = K * (eps_p)^n.
    """
    E = 210000.0  # MPa
    K = 1200.0    # MPa
    n = 0.22      # Hardening exponent

    large_strains = [0.02, 0.05, 0.10, 0.20]

    for eps in large_strains:
        sig = law80_ramberg.ramberg_osgood_stress(eps, E, K, n)

        # Decompose into elastic and plastic components
        eps_e = sig / E
        eps_p = eps - eps_e

        # Verify power-law hardening: sigma = K * (eps_p)^n
        sig_expected = K * (eps_p ** n)
        assert pytest.approx(sig, rel=1e-5) == sig_expected

        # Tangent modulus dsigma/deps should match elastoplastic tangent: 1 / (1/E + 1/(n*K)*(sig/K)^(1/n-1))
        tan = law80_ramberg.ramberg_osgood_tangent(sig, E, K, n)
        expected_tan = 1.0 / (1.0 / E + (1.0 / (n * K)) * ((sig / K) ** (1.0 / n - 1.0)))
        assert pytest.approx(tan, rel=1e-4) == expected_tan


def test_ramberg_osgood_roundtrip_inversion():
    """Verify exact roundtrip consistency between strain(sigma) and stress(epsilon)."""
    E = 205000.0
    K = 950.0
    n = 0.18

    # Sweep from very small to very large stress
    stresses = [1.0, 50.0, 200.0, 500.0, 800.0, 1200.0]
    for s_in in stresses:
        eps = law80_ramberg.ramberg_osgood_strain(s_in, E, K, n)
        s_out = law80_ramberg.ramberg_osgood_stress(eps, E, K, n)
        assert pytest.approx(s_out, rel=1e-6) == s_in

    # Cyclic Masing curve roundtrip
    delta_sigmas = [10.0, 200.0, 600.0, 1000.0]
    for ds_in in delta_sigmas:
        de = law80_ramberg.ramberg_osgood_cyclic_strain(ds_in, E, K, n)
        ds_out = law80_ramberg.ramberg_osgood_cyclic_stress(de, E, K, n)
        assert pytest.approx(ds_out, rel=1e-6) == ds_in


# ============================================================================
# 3D Solid Constitutive Update Tests (sigeps80.F)
# ============================================================================

def test_solid_update_small_strain_elastic_modulus():
    """Verify that solid_update reproduces the linear elastic modulus E at small strains."""
    p = law80_ramberg.build_law80(
        young=200000.0,
        nu=0.3,
        k_strength=1000.0,
        n_exp=0.2,
        sigy0=0.0,  # Pure Ramberg-Osgood
    )

    sig0 = np.zeros(6, dtype=np.float64)
    # Apply small uniaxial strain increment: eps_11 = 1e-6, eps_22 = eps_33 = -nu * eps_11
    eps_11 = 1.0e-6
    deps = np.array([eps_11, -0.3 * eps_11, -0.3 * eps_11, 0.0, 0.0, 0.0], dtype=np.float64)

    sig_new, epsp_new, c = law80_ramberg.solid_update(p, sig0, deps, epsp=0.0, dt=1.0e-6)

    # In uniaxial state with Poisson contraction: sigma_11 = E * eps_11
    assert pytest.approx(sig_new[0] / eps_11, rel=1e-4) == 200000.0
    # Transverse stresses should be nearly zero in uniaxial tension
    assert abs(sig_new[1]) < 1.0e-3
    assert abs(sig_new[2]) < 1.0e-3
    # Plastic strain at this small strain should be essentially zero
    assert epsp_new < 1.0e-10
    # Sound speed should match longitudinal wave speed
    assert c > 0.0


def test_solid_update_large_strain_power_law_hardening():
    """Verify that solid_update accumulates plastic strain and follows power-law hardening
    at large strain increments.
    """
    p = law80_ramberg.build_law80(
        young=200000.0,
        nu=0.3,
        k_strength=1000.0,
        n_exp=0.2,
        sigy0=0.0,
    )

    sig0 = np.zeros(6, dtype=np.float64)
    # Apply large strain increment
    deps = np.array([0.05, -0.02, -0.02, 0.0, 0.0, 0.0], dtype=np.float64)

    sig_new, epsp_new, _ = law80_ramberg.solid_update(p, sig0, deps, epsp=0.0, dt=1.0e-6)

    # Plastic strain must be positive
    assert epsp_new > 0.01

    # Compute von Mises stress
    pres = np.sum(sig_new[:3]) / 3.0
    s_dev = sig_new.copy()
    s_dev[:3] -= pres
    j2 = 0.5 * np.sum(s_dev[:3] ** 2) + np.sum(s_dev[3:] ** 2)
    svm = np.sqrt(3.0 * j2)

    # Check power-law hardening: svm = K * (epsp)^n
    svm_expected = p.k_strength * (epsp_new ** p.n_exp)
    assert pytest.approx(svm, rel=1e-4) == svm_expected


def test_solid_update_plastic_unloading():
    """Verify elastic unloading from a plastically deformed state."""
    p = law80_ramberg.build_law80(
        young=210000.0,
        nu=0.3,
        k_strength=1100.0,
        n_exp=0.2,
        sigy0=0.0,
    )

    # Step 1: Load into plastic regime
    sig0 = np.zeros(6, dtype=np.float64)
    deps_load = np.array([0.02, -0.008, -0.008, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_1, epsp_1, _ = law80_ramberg.solid_update(p, sig0, deps_load, epsp=0.0, dt=1.0e-6)
    assert epsp_1 > 0.0

    # Step 2: Small elastic unloading increment
    sig_1_before = sig_1.copy()
    d_unload = -1.0e-5
    deps_unload = np.array([d_unload, -0.3 * d_unload, -0.3 * d_unload, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_2, epsp_2, _ = law80_ramberg.solid_update(p, sig_1, deps_unload, epsp=epsp_1, dt=1.0e-6)

    # Plastic strain must NOT increase during elastic unloading
    assert pytest.approx(epsp_2, abs=1e-12) == epsp_1

    # Stress delta in direction 1 must correspond to elastic slope E * d_unload
    dsig_11 = sig_2[0] - sig_1_before[0]
    assert pytest.approx(dsig_11 / d_unload, rel=1e-4) == 210000.0


# ============================================================================
# 2D Plane-Stress Shell Update Tests (sigeps80c.F)
# ============================================================================

def test_shell_update_small_and_large_strain():
    """Verify 2D plane-stress shell constitutive update."""
    p = law80_ramberg.build_law80(
        young=200000.0,
        nu=0.3,
        k_strength=1000.0,
        n_exp=0.2,
        sigy0=0.0,
    )

    # 1. Small strain: plane-stress Hooke's law
    sig_sh0 = np.zeros(3, dtype=np.float64)
    deps_small = np.array([1.0e-6, 0.0, 0.0], dtype=np.float64)
    sig_sh1, epsp_1, _ = law80_ramberg.shell_update(p, sig_sh0, deps_small, epsp=0.0, dt=1.0e-6)
    # sigma_xx = E / (1 - nu^2) * deps_xx
    expected_sig_xx = (200000.0 / (1.0 - 0.3 ** 2)) * 1.0e-6
    assert pytest.approx(sig_sh1[0], rel=1e-4) == expected_sig_xx
    assert epsp_1 < 1.0e-10

    # 2. Large strain: power-law hardening
    deps_large = np.array([0.04, -0.015, 0.0], dtype=np.float64)
    sig_sh2, epsp_2, _ = law80_ramberg.shell_update(p, sig_sh0, deps_large, epsp=0.0, dt=1.0e-6)
    assert epsp_2 > 0.01
    svm_sh = np.sqrt(sig_sh2[0] ** 2 + sig_sh2[1] ** 2 - sig_sh2[0] * sig_sh2[1] + 3.0 * sig_sh2[2] ** 2)
    assert pytest.approx(svm_sh, rel=1e-4) == p.k_strength * (epsp_2 ** p.n_exp)


# ============================================================================
# Tangent Stiffness Operators & Parameter Resolution Tests
# ============================================================================

def test_tangent_stiffness_operators():
    """Verify solid and shell tangent stiffness matrices."""
    p = law80_ramberg.build_law80(young=210000.0, nu=0.3)

    c_solid = law80_ramberg.solid_tangent(p)
    assert c_solid.shape == (1, 6, 6)
    # Check elastic coefficients: C11 = bulk + 4/3*G, C12 = bulk - 2/3*G, C44 = G
    c11 = p.bulk + 4.0 / 3.0 * p.g
    c12 = p.bulk - 2.0 / 3.0 * p.g
    assert pytest.approx(c_solid[0, 0, 0]) == c11
    assert pytest.approx(c_solid[0, 0, 1]) == c12
    assert pytest.approx(c_solid[0, 3, 3]) == p.g

    c_shell = law80_ramberg.shell_tangent(p)
    assert c_shell.shape == (1, 3, 3)
    assert pytest.approx(c_shell[0, 0, 0]) == p.a11_2d
    assert pytest.approx(c_shell[0, 0, 1]) == p.a12_2d
    assert pytest.approx(c_shell[0, 2, 2]) == p.g


def test_build_law80_parameters():
    """Verify parameter extraction from Material object with K and n."""
    mat = Material(
        id=80,
        law=80,
        rho0=7.85e-3,
        params={
            "E": 205000.0,
            "nu": 0.29,
            "K": 1050.0,
            "n": 0.21,
        },
    )
    p = law80_ramberg.build_law80(mat)
    assert p.young == 205000.0
    assert p.nu == 0.29
    assert p.k_strength == 1050.0
    assert p.n_exp == 0.21
    assert pytest.approx(p.n_ramberg) == 1.0 / 0.21
    assert p.E == 205000.0
    assert p.K == 1050.0
    assert p.n == 0.21
    assert law80_ramberg.needs_defgrad(mat) is False
    assert "uvar80" in law80_ramberg.extra_shapes(mat)
