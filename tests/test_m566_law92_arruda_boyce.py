"""
Tests for Milestone M566: /MAT/LAW92 (/MAT/ARRUDA_BOYCE) Arruda-Boyce Hyperelastic Model
Core Constitutive Kernel: Solid, Shell, Sound Speed, Tangent Moduli & Levenberg-Marquardt Fit.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law92_arruda_boyce import (
    C_LANGEVIN,
    initial_shear_modulus,
    bulk_modulus,
    solid_update,
    shell_update,
    sound_speed,
    sound_speed_shell,
    consistent_tangent,
    fit_curve_arruda_boyce,
    arruda_boyce_analytical_stress,
)


# ============================================================================
# 1. Langevin Coefficients & Initial Elastic Moduli
# ============================================================================

def test_inverse_langevin_coefficients():
    """Verify coefficients C1..C5 match Fortran hm_read_mat92.F and sigeps92.F."""
    assert math.isclose(C_LANGEVIN[0], 0.5)
    assert math.isclose(C_LANGEVIN[1], 0.05)
    assert math.isclose(C_LANGEVIN[2], 11.0 / 1050.0)
    assert math.isclose(C_LANGEVIN[3], 19.0 / 7000.0)
    assert math.isclose(C_LANGEVIN[4], 519.0 / 673750.0)


def test_initial_shear_and_bulk_modulus():
    """Verify G0 and K calculations citing constant_mod.F EIGHTY19 = 99.0."""
    mu = 1.0e6
    lam = 5.0
    beta = 1.0 / (lam * lam)
    # Fortran: G = MU*(1 + 3/5*beta + 99/175*beta^2 + 513/875*beta^3 + 42039/67375*beta^4)
    expected_poly = (
        1.0
        + 0.6 * beta
        + (99.0 / 175.0) * (beta ** 2)
        + (513.0 / 875.0) * (beta ** 3)
        + (42039.0 / 67375.0) * (beta ** 4)
    )
    expected_g0 = mu * expected_poly
    g0 = initial_shear_modulus(mu, lam)
    assert math.isclose(g0, expected_g0, rel_tol=1e-12)

    # Bulk modulus from D: K = 2 / D
    d = 1.0e-7
    k = bulk_modulus(d=d)
    assert math.isclose(k, 2.0e7, rel_tol=1e-12)

    # Bulk modulus from Poisson ratio: K = 2/3 * (1 + nu)/(1 - 2*nu) * G0
    nu = 0.49
    k_nu = bulk_modulus(d=0.0, nu=nu, g0=g0)
    expected_k_nu = (2.0 / 3.0) * (1.0 + nu) * g0 / (1.0 - 2.0 * nu)
    assert math.isclose(k_nu, expected_k_nu, rel_tol=1e-12)


# ============================================================================
# 2. Solid Update: Small Deformation Limit Matches Linear Elasticity
# ============================================================================

def test_solid_update_small_strain_hooke_limit():
    """At tiny strains, Arruda-Boyce response must converge to linear isotropic elasticity."""
    mu = 1.5e6
    lam = 7.0
    nu = 0.3
    g0 = initial_shear_modulus(mu, lam)
    k = bulk_modulus(d=0.0, nu=nu, g0=g0)
    rho = 1000.0

    # Small uniaxial strain: eps_11 = 1e-5
    eps_11 = 1e-5
    # For isotropic uniaxial strain (constrained lateral): sigma_11 = (K + 4/3*G)*eps_11
    eps_tens = np.array([eps_11, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig, history, c_sound = solid_update(eps_tens, mu=mu, d=0.0, lam=lam, nu=nu, rho=rho)

    expected_sig11 = (k + (4.0 / 3.0) * g0) * eps_11
    expected_sig22 = (k - (2.0 / 3.0) * g0) * eps_11

    assert math.isclose(sig[0], expected_sig11, rel_tol=1e-3)
    assert math.isclose(sig[1], expected_sig22, rel_tol=1e-3)
    assert math.isclose(sig[2], expected_sig22, rel_tol=1e-3)
    assert abs(sig[3]) < 1e-8
    assert abs(sig[4]) < 1e-8
    assert abs(sig[5]) < 1e-8


# ============================================================================
# 3. Pure Volumetric Deformation
# ============================================================================

def test_solid_update_pure_volumetric_compression():
    """Pure hydrostatic strain produces isotropic pressure p = -K * ln(J) / J."""
    mu = 2.0e6
    lam = 6.0
    d = 1.0e-8  # K = 2e8
    k = 2.0 / d
    rho = 1200.0

    eps_vol = -0.01  # 1% compression on each axis -> eps = [-0.01, -0.01, -0.01, 0, 0, 0]
    eps_tens = np.array([eps_vol, eps_vol, eps_vol, 0.0, 0.0, 0.0], dtype=float)
    sig, history, c_sound = solid_update(eps_tens, mu=mu, d=d, lam=lam, rho=rho)

    # In pure volumetric deformation, deviatoric isochoric stretch is 1.0, dev stress is 0
    # Cauchy stress is (K / 2) * (1 - 1/J^2) per sigeps92.F:245-248
    j = math.exp(3.0 * eps_vol)
    expected_stress = (k / 2.0) * (1.0 - 1.0 / (j * j))
    assert math.isclose(sig[0], expected_stress, rel_tol=1e-5)
    assert math.isclose(sig[1], expected_stress, rel_tol=1e-5)
    assert math.isclose(sig[2], expected_stress, rel_tol=1e-5)
    assert abs(sig[3]) < 1e-8
    assert abs(sig[4]) < 1e-8
    assert abs(sig[5]) < 1e-8


# ============================================================================
# 4. Large Uniaxial Deformation & Non-linear Locking
# ============================================================================

def test_solid_update_uniaxial_monotonicity_and_locking():
    """Stress increases monotonically with stretch and stiffens dramatically near lambda_m."""
    mu = 1.0e6
    lam = 4.0  # limiting chain stretch lambda_m = 4.0
    nu = 0.495
    rho = 1000.0

    stretches = [1.1, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5]
    stresses = []
    sound_speeds = []

    for s in stretches:
        # Incompressible uniaxial extension: lambda_1 = s, lambda_2 = lambda_3 = 1/sqrt(s)
        e1 = math.log(s)
        e2 = math.log(1.0 / math.sqrt(s))
        eps_tens = np.array([e1, e2, e2, 0.0, 0.0, 0.0], dtype=float)
        sig, history, c_sound = solid_update(eps_tens, mu=mu, d=0.0, lam=lam, nu=nu, rho=rho)
        # Net axial tension: sigma_11 - sigma_22
        axial_stress = sig[0] - sig[1]
        stresses.append(axial_stress)
        sound_speeds.append(c_sound)

    # Monotonicity check
    for i in range(1, len(stresses)):
        assert stresses[i] > stresses[i - 1], f"Stress not monotonic at stretch {stretches[i]}"

    # Stiffening check: tangent slope at high stretch is much larger than at low stretch
    slope_low = (stresses[1] - stresses[0]) / (stretches[1] - stretches[0])
    slope_high = (stresses[-1] - stresses[-2]) / (stretches[-1] - stretches[-2])
    assert slope_high > 2.5 * slope_low, "Arruda-Boyce locking effect not observed"


# ============================================================================
# 5. Mullins Hyperelastic Strain Energy Accounting
# ============================================================================

def test_mullins_strain_energy():
    """Strain energy W is 0 at zero strain and positive definite under deformation."""
    mu = 2.0e6
    lam = 5.0
    nu = 0.45
    rho = 1000.0

    # 1. Zero deformation
    zero_eps = np.zeros(6, dtype=float)
    _, hist0, _ = solid_update(zero_eps, mu=mu, d=0.0, lam=lam, nu=nu, rho=rho)
    assert abs(hist0.get("w_mullins", 0.0)) < 1e-12

    # 2. Deformed state
    eps = np.array([0.1, -0.05, -0.05, 0.0, 0.0, 0.0], dtype=float)
    _, hist_def, _ = solid_update(eps, mu=mu, d=0.0, lam=lam, nu=nu, rho=rho)
    w_def = hist_def.get("w_mullins", 0.0)
    assert w_def > 0.0, f"Expected positive strain energy, got {w_def}"


# ============================================================================
# 6. Plane-Stress Shell Update
# ============================================================================

def test_shell_update_plane_stress_condition():
    """Shell update enforces sigma_zz = 0 via Newton-Raphson out-of-plane stretch search."""
    mu = 1.0e6
    lam = 5.0
    nu = 0.49
    rho = 1000.0

    # Apply in-plane strains: eps_xx = 0.15, eps_yy = -0.05, gamma_xy = 0.08
    eps_inplane = np.array([0.15, -0.05, 0.08], dtype=float)
    sig_shell, eps_zz, hist, c_shell = shell_update(
        eps_inplane, mu=mu, d=0.0, lam=lam, nu=nu, rho=rho
    )

    assert len(sig_shell) == 3
    # Check that out-of-plane stretch was determined (negative for net in-plane expansion)
    assert eps_zz < 0.0
    assert c_shell > 0.0

    # Reconstruct 3D strain and verify sigma_zz == 0
    eps_full = np.array([0.15, -0.05, eps_zz, 0.08, 0.0, 0.0], dtype=float)
    sig_3d, _, _ = solid_update(eps_full, mu=mu, d=0.0, lam=lam, nu=nu, rho=rho)
    assert abs(sig_3d[2]) < 1e-4, f"Plane-stress condition violated: sigma_zz = {sig_3d[2]}"
    assert math.isclose(sig_shell[0], sig_3d[0], rel_tol=1e-4)
    assert math.isclose(sig_shell[1], sig_3d[1], rel_tol=1e-4)
    assert math.isclose(sig_shell[2], sig_3d[3], rel_tol=1e-4)


# ============================================================================
# 7. Tangent Stiffness & Sound Speed
# ============================================================================

def test_consistent_tangent_and_sound_speed():
    """Consistent tangent matrix has correct symmetry and eigenvalues."""
    mu = 1.5e6
    lam = 6.0
    nu = 0.49
    rho = 1000.0

    eps = np.array([0.05, -0.02, -0.02, 0.01, 0.0, 0.0], dtype=float)
    tangent = consistent_tangent(eps, mu=mu, d=0.0, lam=lam, nu=nu, rho=rho)

    assert tangent.shape == (6, 6)
    # Tangent symmetry within finite-strain Cauchy rate tolerance
    diff = tangent - tangent.T
    assert np.all(np.abs(diff) < 5e-3 * np.max(np.abs(tangent)))

    # Symmetrized tangent must be positive definite
    sym_tangent = 0.5 * (tangent + tangent.T)
    eigvals = np.linalg.eigvalsh(sym_tangent)
    assert np.all(eigvals > 0.0)

    # Sound speed must be strictly positive
    c = sound_speed(eps, mu=mu, d=0.0, lam=lam, nu=nu, rho=rho)
    assert c > 100.0


# ============================================================================
# 8. Experimental Curve Fitting (Levenberg-Marquardt)
# ============================================================================

def test_fit_curve_arruda_boyce_uniaxial():
    """Fit synthetic uniaxial data and accurately recover mu and lambda_m."""
    true_mu = 1.2e6
    true_lam = 5.0

    # Generate synthetic uniaxial data points using Fortran ARRUDA_BOYCE (law92_nlsqf.F90:462-472)
    stretches = np.array([1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.3, 2.6])
    stresses = np.array([
        arruda_boyce_analytical_stress(s, true_mu, true_lam, itype=1)
        for s in stretches
    ])

    fitted_mu, fitted_lam = fit_curve_arruda_boyce(stretches, stresses, itest=1)
    assert math.isclose(fitted_mu, true_mu, rel_tol=0.05)
    assert math.isclose(fitted_lam, true_lam, rel_tol=0.05)


def test_fit_curve_arruda_boyce_equibiaxial():
    """Fit synthetic equibiaxial data (itest=2)."""
    true_mu = 8.0e5
    true_lam = 4.5
    stretches = np.array([1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.9])
    stresses = np.array([
        arruda_boyce_analytical_stress(s, true_mu, true_lam, itype=2)
        for s in stretches
    ])

    fitted_mu, fitted_lam = fit_curve_arruda_boyce(stretches, stresses, itest=2)
    assert math.isclose(fitted_mu, true_mu, rel_tol=0.05)
    assert math.isclose(fitted_lam, true_lam, rel_tol=0.05)
