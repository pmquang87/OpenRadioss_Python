"""
Unit tests for LAW82 — Ogden hyperelastic material (/MAT/LAW82, /MAT/OGDEN).

Tests cover:
- Parameter initialization (N=1, 2, 3 terms, nu clamping, D1 and bulk modulus)
- Pure uniaxial tension (analytical comparison for N=1, 2, 3)
- Equibiaxial tension
- Pure shear / planar tension
- Near-incompressibility and volumetric response
- Small-strain limit matching classical linear elasticity
- Consistent algorithmic tangents for solids and shells (FD check < 1e-5)
- Nonlinear sound speed for solids and shells
- Multi-element vectorization
- Shell out-of-plane stretch lambda_3 solve and thickness update
"""

import numpy as np
import pytest

from pyradioss.materials.law82_ogden import (
    OgdenParams,
    build_law82,
    solid_update,
    shell_update,
    solid_sound_speed,
    shell_sound_speed,
    consistent_solid_tangent,
    consistent_shell_tangent,
)


# ============================================================================
# 1. Parameter Initialization Tests
# ============================================================================


def test_param_initialization_defaults():
    """Test default Ogden parameters from build_law82."""
    mat = build_law82()
    assert mat.id == 1
    assert mat.rho0 == 1.0
    assert mat.rhor == 1.0
    assert mat.nu == 0.475
    assert mat.nordre == 1
    assert np.isclose(mat.g0, 1.0)
    assert np.isclose(mat.rbulk, 2.0 / mat.d[0])
    assert np.isclose(mat.K, mat.rbulk)
    assert np.isclose(mat.G, mat.g0)


def test_param_initialization_nu_clamp():
    """Test clamping of nu = 0.5 to 0.495 as per hm_read_mat82.F."""
    mat = build_law82(nu=0.5, mu=[20.0], alpha=[2.0])
    assert np.isclose(mat.nu, 0.495)
    expected_d1 = 3.0 * (1.0 - 2.0 * 0.495) / (20.0 * (1.0 + 0.495))
    assert np.isclose(mat.d[0], expected_d1)
    assert np.isclose(mat.rbulk, 2.0 / expected_d1)


def test_param_initialization_nu_zero_with_d():
    """Test nu0 = 0 with D1 > 0 computing nu from bulk modulus."""
    mat = build_law82(nu=0.0, mu=[10.0], alpha=[2.0], d=[0.01])
    expected_nu = (3.0 * 200.0 - 20.0) / (6.0 * 200.0 + 20.0)
    assert np.isclose(mat.nu, expected_nu)
    assert np.isclose(mat.rbulk, 200.0)


def test_param_initialization_nu_zero_without_d():
    """Test nu0 = 0 with D1 <= 0 defaulting nu to 0.495."""
    mat = build_law82(nu=0.0, mu=[15.0], alpha=[2.0], d=[0.0])
    assert np.isclose(mat.nu, 0.495)
    expected_d1 = 3.0 * (1.0 - 2.0 * 0.495) / (15.0 * (1.0 + 0.495))
    assert np.isclose(mat.d[0], expected_d1)
    assert np.isclose(mat.rbulk, 2.0 / expected_d1)


def test_param_initialization_n_terms():
    """Test N=1, N=2, N=3 Ogden series initialization."""
    mu_3 = [6.3e5, 1.2e3, -1.0e4]
    alpha_3 = [1.3, 5.0, -2.0]
    mat = build_law82(nordre=3, mu=mu_3, alpha=alpha_3, nu=0.495)
    assert mat.nordre == 3
    assert np.allclose(mat.mu, mu_3)
    assert np.allclose(mat.alpha, alpha_3)
    assert np.isclose(mat.g0, np.sum(mu_3))


# ============================================================================
# 2. Pure Uniaxial Tension Tests
# ============================================================================


@pytest.mark.parametrize("n_order,mu,alpha", [
    (1, [10.0], [2.0]),
    (2, [8.0, 2.0], [2.0, -2.0]),
    (3, [6.0, 3.0, 1.0], [1.3, 4.0, -2.0]),
])
def test_solid_pure_uniaxial_tension(n_order, mu, alpha):
    """Test 3D solid Ogden model in pure uniaxial tension against analytical formula.

    For incompressible uniaxial tension (lambda_1 = lambda, lambda_2 = lambda_3 = lambda^(-1/2)):
        sigma_1 - sigma_2 = sum_{k=1}^N (2 mu_k / alpha_k) * (lambda^alpha_k - lambda^(-alpha_k / 2))
    """
    mat = build_law82(nordre=n_order, mu=mu, alpha=alpha, nu=0.499)

    stretch = 1.25
    eps_1 = np.log(stretch)
    eps_2 = -0.5 * eps_1
    eps_3 = eps_2

    eps = np.array([[eps_1, eps_2, eps_3, 0.0, 0.0, 0.0]])
    sig = np.zeros_like(eps)
    deps = np.zeros_like(eps)

    sig_new, _ = solid_update(mat, sig, deps, eps, dt=1e-3, ismstr=0)

    s_diff_analytical = 0.0
    for m_k, a_k in zip(mu, alpha):
        s_diff_analytical += (2.0 * m_k / a_k) * (stretch ** a_k - stretch ** (-0.5 * a_k))

    s_diff_computed = sig_new[0, 0] - sig_new[0, 1]
    assert np.isclose(s_diff_computed, s_diff_analytical, rtol=1e-4)


# ============================================================================
# 3. Equibiaxial Tension Tests
# ============================================================================


@pytest.mark.parametrize("n_order,mu,alpha", [
    (1, [12.0], [2.0]),
    (2, [7.0, 5.0], [2.0, -1.0]),
    (3, [5.0, 4.0, 3.0], [2.0, 4.0, -2.0]),
])
def test_solid_equibiaxial_tension(n_order, mu, alpha):
    """Test 3D solid Ogden model in equibiaxial tension against analytical formula.

    For equibiaxial tension (lambda_1 = lambda_2 = lambda, lambda_3 = lambda^(-2)):
        sigma_1 - sigma_3 = sum_{k=1}^N (2 mu_k / alpha_k) * (lambda^alpha_k - lambda^(-2 alpha_k))
    """
    mat = build_law82(nordre=n_order, mu=mu, alpha=alpha, nu=0.499)

    stretch = 1.2
    eps_1 = np.log(stretch)
    eps_2 = eps_1
    eps_3 = -2.0 * eps_1

    eps = np.array([[eps_1, eps_2, eps_3, 0.0, 0.0, 0.0]])
    sig = np.zeros_like(eps)
    deps = np.zeros_like(eps)

    sig_new, _ = solid_update(mat, sig, deps, eps, dt=1e-3, ismstr=0)

    s_diff_analytical = 0.0
    for m_k, a_k in zip(mu, alpha):
        s_diff_analytical += (2.0 * m_k / a_k) * (stretch ** a_k - stretch ** (-2.0 * a_k))

    s_diff_computed = sig_new[0, 0] - sig_new[0, 2]
    assert np.isclose(s_diff_computed, s_diff_analytical, rtol=1e-4)
    assert np.isclose(sig_new[0, 0], sig_new[0, 1], rtol=1e-6)


# ============================================================================
# 4. Pure Shear / Planar Tension Tests
# ============================================================================


def test_solid_planar_tension_pure_shear():
    """Test 3D solid Ogden model in planar tension (pure shear state).

    Planar tension: lambda_1 = lambda, lambda_2 = 1, lambda_3 = lambda^(-1).
        sigma_1 - sigma_3 = sum_{k=1}^N (2 mu_k / alpha_k) * (lambda^alpha_k - lambda^(-alpha_k))
    """
    mu = [8.0, 4.0]
    alpha = [2.0, -2.0]
    mat = build_law82(nordre=2, mu=mu, alpha=alpha, nu=0.499)

    stretch = 1.3
    eps_1 = np.log(stretch)
    eps_2 = 0.0
    eps_3 = -eps_1

    eps = np.array([[eps_1, eps_2, eps_3, 0.0, 0.0, 0.0]])
    sig = np.zeros_like(eps)
    deps = np.zeros_like(eps)

    sig_new, _ = solid_update(mat, sig, deps, eps, dt=1e-3, ismstr=0)

    s_diff_analytical = 0.0
    for m_k, a_k in zip(mu, alpha):
        s_diff_analytical += (2.0 * m_k / a_k) * (stretch ** a_k - stretch ** (-a_k))

    s_diff_computed = sig_new[0, 0] - sig_new[0, 2]
    assert np.isclose(s_diff_computed, s_diff_analytical, rtol=1e-4)


def test_solid_simple_shear():
    """Test simple shear kinematics: shear strain gamma_xy produces tau_xy."""
    mat = build_law82(nordre=1, mu=[10.0], alpha=[2.0], nu=0.475)
    gamma = 0.1
    eps = np.array([[0.0, 0.0, 0.0, gamma, 0.0, 0.0]])
    sig = np.zeros_like(eps)
    sig_new, _ = solid_update(mat, sig, None, eps, dt=1e-3, ismstr=0)

    assert np.isclose(sig_new[0, 3], 1.0, rtol=1e-2)


# ============================================================================
# 5. Near-Incompressibility and Volumetric Response Tests
# ============================================================================


def test_solid_volumetric_response():
    """Test that pure volumetric strain produces hydrostatic pressure P = K * (J - 1)."""
    mu = [10.0]
    alpha = [2.0]
    mat = build_law82(nordre=1, mu=mu, alpha=alpha, nu=0.475)

    eps_0 = 0.001
    eps = np.array([[eps_0, eps_0, eps_0, 0.0, 0.0, 0.0]])
    sig = np.zeros_like(eps)

    sig_new, _ = solid_update(mat, sig, None, eps, dt=1e-3, ismstr=0)

    J = np.exp(3.0 * eps_0)
    P_expected = 2.0 / mat.d[0] * (J - 1.0)
    assert np.isclose(sig_new[0, 0], P_expected, rtol=1e-3)
    assert np.isclose(sig_new[0, 1], P_expected, rtol=1e-3)
    assert np.isclose(sig_new[0, 2], P_expected, rtol=1e-3)
    assert np.isclose(sig_new[0, 3], 0.0, atol=1e-12)


# ============================================================================
# 6. Small-Strain Linear Elasticity Limit
# ============================================================================


def test_small_strain_linear_elasticity_limit():
    """Test that tangent at eps = 0 matches classical isotropic linear elasticity.

    Solid:
        C_11 = K + 4/3 G
        C_12 = K - 2/3 G
        C_44 = G
    Shell:
        C_11 = E / (1 - nu^2)
        C_12 = nu * E / (1 - nu^2)
        C_33 = G
    """
    mu = [10.0]
    alpha = [2.0]
    nu = 0.475
    mat = build_law82(mu=mu, alpha=alpha, nu=nu)

    G = mat.g0
    K = mat.rbulk
    E = 2.0 * G * (1.0 + mat.nu)

    # Solid tangent
    eps_solid = np.zeros((1, 6))
    C_solid = consistent_solid_tangent(mat, eps_solid)[0]

    assert np.isclose(C_solid[0, 0], K + (4.0 / 3.0) * G, rtol=1e-5)
    assert np.isclose(C_solid[0, 1], K - (2.0 / 3.0) * G, rtol=1e-5)
    assert np.isclose(C_solid[3, 3], G, rtol=1e-5)
    assert np.isclose(C_solid[4, 4], G, rtol=1e-5)
    assert np.isclose(C_solid[5, 5], G, rtol=1e-5)
    # Symmetric at zero strain
    assert np.max(np.abs(C_solid - C_solid.T)) < 1e-6

    # Shell tangent
    eps_shell = np.zeros((1, 3))
    C_shell = consistent_shell_tangent(mat, eps_shell)[0]

    C11_expected = E / (1.0 - mat.nu ** 2)
    C12_expected = mat.nu * C11_expected

    assert np.isclose(C_shell[0, 0], C11_expected, rtol=1e-5)
    assert np.isclose(C_shell[0, 1], C12_expected, rtol=1e-5)
    assert np.isclose(C_shell[1, 1], C11_expected, rtol=1e-5)
    assert np.isclose(C_shell[2, 2], G, rtol=1e-5)
    assert np.max(np.abs(C_shell - C_shell.T)) < 1e-6


# ============================================================================
# 7. Consistent Algorithmic Tangent Tests (FD Verification < 1e-5)
# ============================================================================


def test_consistent_solid_tangent_fd():
    """Verify consistent_solid_tangent at general non-zero strain state against central FD."""
    mat = build_law82(nordre=2, mu=[8.0, 3.0], alpha=[2.0, -2.0], nu=0.45)
    eps = np.array([[0.05, -0.02, 0.01, 0.03, -0.01, 0.02]])

    C_analytic = consistent_solid_tangent(mat, eps)[0]

    # Independent central finite difference verification with different step size
    h = 2e-7
    C_fd = np.zeros((6, 6))
    for j in range(6):
        ep = eps.copy()
        ep[0, j] += h
        em = eps.copy()
        em[0, j] -= h
        sp, _ = solid_update(mat, np.zeros_like(ep), None, ep, dt=1e-3)
        sm, _ = solid_update(mat, np.zeros_like(em), None, em, dt=1e-3)
        C_fd[:, j] = (sp[0] - sm[0]) / (2.0 * h)

    # Compare to < 1e-5
    diff = np.max(np.abs(C_analytic - C_fd))
    rel_diff = diff / np.max(np.abs(C_analytic))
    assert rel_diff < 1e-5

    # Verify nonlinear continuum mechanics identity: C_12 - C_21 = sigma_yy - sigma_xx
    sig, _ = solid_update(mat, np.zeros_like(eps), None, eps, dt=1e-3)
    assert np.isclose(C_analytic[0, 1] - C_analytic[1, 0], sig[0, 1] - sig[0, 0], rtol=1e-4)


def test_consistent_shell_tangent_fd():
    """Verify consistent_shell_tangent at general non-zero in-plane strain state against central FD."""
    mat = build_law82(nordre=2, mu=[10.0, 2.0], alpha=[2.0, -1.0], nu=0.45)
    eps = np.array([[0.04, -0.01, 0.02]])

    C_tangent = consistent_shell_tangent(mat, eps)[0]

    h = 2e-7
    C_fd = np.zeros((3, 3))
    for j in range(3):
        ep = eps.copy()
        ep[0, j] += h
        em = eps.copy()
        em[0, j] -= h
        sp, _ = shell_update(mat, np.zeros_like(ep), None, ep, dt=1e-3)
        sm, _ = shell_update(mat, np.zeros_like(em), None, em, dt=1e-3)
        C_fd[:, j] = (sp[0, :3] - sm[0, :3]) / (2.0 * h)

    diff = np.max(np.abs(C_tangent - C_fd))
    rel_diff = diff / np.max(np.abs(C_tangent))
    assert rel_diff < 1e-5

    # At zero strain, verify exact symmetry and recovery of plane stress linear elasticity:
    C_zero = consistent_shell_tangent(mat, np.zeros((1, 3)))[0]
    assert np.allclose(C_zero, C_zero.T, atol=1e-6)
    E_mod = 2.0 * mat.g0 * (1.0 + mat.nu)
    c11_plane = E_mod / (1.0 - mat.nu ** 2)
    c12_plane = mat.nu * c11_plane
    assert np.isclose(C_zero[0, 0], c11_plane, rtol=1e-5)
    assert np.isclose(C_zero[1, 1], c11_plane, rtol=1e-5)
    assert np.isclose(C_zero[0, 1], c12_plane, rtol=1e-5)
    assert np.isclose(C_zero[2, 2], mat.g0, rtol=1e-5)


# ============================================================================
# 8. Sound Speed Tests
# ============================================================================


def test_solid_and_shell_sound_speed():
    """Test nonlinear sound speed calculations for solids and shells."""
    mat = build_law82(nordre=1, mu=[15.0], alpha=[2.0], nu=0.475, rho0=2.0)
    rho = 2.0

    # Ground state sound speed
    c_solid_0 = solid_sound_speed(mat, rho=rho)
    c_solid_expected = np.sqrt(((4.0 / 3.0) * mat.g0 + mat.rbulk) / rho)
    assert np.isclose(c_solid_0, c_solid_expected)

    c_shell_0 = shell_sound_speed(mat, rho=rho)
    c_shell_expected = np.sqrt(((2.0 / 3.0) * mat.g0 + mat.rbulk) / rho)
    assert np.isclose(c_shell_0, c_shell_expected)

    # Large stretch stiffening for solid
    eps_large = np.array([[np.log(2.0), -0.5 * np.log(2.0), -0.5 * np.log(2.0), 0.0, 0.0, 0.0]])
    c_solid_stretched = solid_sound_speed(mat, rho=rho, eps=eps_large)
    assert c_solid_stretched > c_solid_0


# ============================================================================
# 9. Shell Newton-Raphson and Thickness Update Tests
# ============================================================================


def test_shell_plane_stress_and_thickness_update():
    """Test that shell_update zeroes T3 (sigma_zz = 0) and updates thickness."""
    mat = build_law82(nordre=1, mu=[10.0], alpha=[2.0], nu=0.475)

    eps = np.array([[0.05, -0.02, 0.01]])
    deps = np.array([[0.01, -0.005, 0.002]])
    sig = np.zeros_like(eps)

    extra = {"uvar": np.ones((1, 1)), "thkn": 1.0, "thk0": 1.0}

    sig_new, epsp_new = shell_update(mat, sig, deps, eps, dt=1e-3, extra=extra)

    lam3 = extra["uvar"][0, 0]
    assert 0.5 < lam3 < 1.5

    nu = mat.nu
    dezz_expected = -nu / (1.0 - nu) * (deps[0, 0] + deps[0, 1])
    thk_expected = 1.0 + dezz_expected * 1.0
    assert np.isclose(extra["thkn"], thk_expected)


def test_shell_transverse_shear():
    """Test transverse shear stress update for (n, 5) shell tensors."""
    mat = build_law82(mu=[10.0], alpha=[2.0])
    sig = np.array([[0.0, 0.0, 0.0, 2.0, 3.0]])
    deps = np.array([[0.0, 0.0, 0.0, 0.1, -0.2]])
    eps = np.array([[0.01, 0.01, 0.0, 0.1, -0.2]])

    sig_new, _ = shell_update(mat, sig, deps, eps, dt=1e-3)

    assert np.isclose(sig_new[0, 3], 3.0)
    assert np.isclose(sig_new[0, 4], 1.0)


# ============================================================================
# 10. Multi-Element Vectorization Tests
# ============================================================================


def test_vectorization():
    """Test batch processing of multiple elements in solid and shell updates."""
    mat = build_law82(nordre=2, mu=[10.0, 5.0], alpha=[2.0, -2.0], nu=0.475)

    n_elem = 5
    eps_solid = np.random.RandomState(42).randn(n_elem, 6) * 0.02
    sig_solid = np.zeros((n_elem, 6))

    sig_batch, _ = solid_update(mat, sig_solid, None, eps_solid, dt=1e-3)

    for i in range(n_elem):
        sig_single, _ = solid_update(mat, sig_solid[i:i+1], None, eps_solid[i:i+1], dt=1e-3)
        assert np.allclose(sig_batch[i], sig_single[0], atol=1e-12)

    eps_shell = np.random.RandomState(43).randn(n_elem, 3) * 0.02
    sig_shell = np.zeros((n_elem, 3))
    sig_shell_batch, _ = shell_update(mat, sig_shell, None, eps_shell, dt=1e-3)

    for i in range(n_elem):
        sig_single_s, _ = shell_update(mat, sig_shell[i:i+1], None, eps_shell[i:i+1], dt=1e-3)
        assert np.allclose(sig_shell_batch[i], sig_single_s[0], atol=1e-12)
