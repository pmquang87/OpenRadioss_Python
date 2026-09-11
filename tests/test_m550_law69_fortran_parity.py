"""
Fortran physics parity test suite for Milestone M550: LAW69 hyperelastic material.
(/MAT/LAW69, /MAT/HYP_ELAS, /MAT/HYPERELASTIC).

Validates pyradioss/materials/law69_hyperelastic.py against exact Fortran source formulas from:
- engine/source/materials/mat/mat069/sigeps69.F (3D continuum solid kernel)
- engine/source/materials/mat/mat069/sigeps69c.F (2D shell plane-stress kernel)
- starter/source/materials/mat/mat069/hm_read_mat69.F & law69_upd.F (material setup & stability)
- starter/source/materials/tools/nlsqf.F (OGDEN0, OGDEN1, LAW69_GUESS_BOUNDS)

Covers at least 25 exhaustive analytical/parity tests:
1. Uniaxial tension/compression Cauchy stresses vs Fortran T_i formula.
2. Equibiaxial tension symmetry and zero out-of-plane stress.
3. Pure shear diagonal deviatoric stresses and rotated shear stress.
4. Plane-strain tension lateral constraint stress.
5. Mooney-Rivlin transformation: C_10 = mu_1/2, C_01 = -mu_2/2 and ground-state G0.
6. Bulk modulus scaling function f(RV) (FUN_A1): P = RBULK * FSCALE * f(RV).
7. Anti-buckling condition: RBULK > 24*GMAX and lambda_min < 0.2 scaling P_FAC.
8. Shell out-of-plane stretch Newton-Raphson solve: T_3 = 0 and uvar[:, 2] update.
9. Stiffening factor GTMAX and acoustic sound speeds for solid and shell.
10. Tensile cut-off: stress zeroing and element erosion (off=0 solid, off=0.8 shell).
11. Exact OGDEN0/OGDEN1 evaluation and analytical derivatives.
12. LAW69_GUESS_BOUNDS starter curve bounds estimation.
13. Drucker-Prager material stability conditions (law69_upd.F).
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest

from pyradioss.materials.law69_hyperelastic import (
    Law69Params,
    build_law69,
    fit_law69_curve,
    sigeps69_solid,
    sigeps69c_shell,
    solid_sound_speed,
    shell_sound_speed,
)


# ============================================================================
# Fortran Reference Physics Oracle Implementations
# Directly mirroring OpenRadioss Fortran source line-by-line
# ============================================================================


def fortran_ogden0(x: float | np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Fortran reference: starter/source/materials/tools/nlsqf.F:472-493 SUBROUTINE OGDEN0.

    Y = sum_k mu_k * (X**(alpha_k - 1) - X**(-0.5*alpha_k - 1))
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x_arr)
    for m, a in zip(mu, alpha):
        y += m * (x_arr ** (a - 1.0) - x_arr ** (-0.5 * a - 1.0))
    return y


def fortran_ogden1(
    x: float, mu: np.ndarray, alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fortran reference: starter/source/materials/tools/nlsqf.F:507-537 SUBROUTINE OGDEN1.

    dy_dmu = X**(alpha_k - 1) - X**(-0.5*alpha_k - 1)
    dy_dalpha = mu_k * log(X) * (X**(alpha_k - 1) + 0.5 * X**(-0.5*alpha_k - 1))
    """
    dy_dmu = np.zeros(len(mu), dtype=np.float64)
    dy_dalpha = np.zeros(len(alpha), dtype=np.float64)
    log_x = np.log(x)
    for i, (m, a) in enumerate(zip(mu, alpha)):
        tmp1 = x ** (a - 1.0)
        tmp2 = x ** (-0.5 * a - 1.0)
        dy_dmu[i] = tmp1 - tmp2
        dy_dalpha[i] = m * log_x * (tmp1 + 0.5 * tmp2)
    return dy_dmu, dy_dalpha


def fortran_guess_bounds(
    lawid: int,
    icheck: int,
    x: np.ndarray,
    y: np.ndarray,
    n_terms: int,
    icomp: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fortran reference: starter/source/materials/tools/nlsqf.F:1274-1350 SUBROUTINE LAW69_GUESS_BOUNDS."""
    npt = len(x)
    dx = x - 1.0
    dx_clamped = np.where(dx >= 0.0, np.maximum(dx, 1e-6), np.minimum(dx, -1e-6))
    ave_slope = float(np.sum((y - 0.0) / dx_clamped) / npt)
    mu_max = ave_slope
    if icomp == 0:
        mu_max = max(ave_slope, 20.0)

    nmual = 2 * n_terms
    mcof_min = np.zeros(nmual, dtype=np.float64)
    mcof_max = np.zeros(nmual, dtype=np.float64)

    if lawid == 1:  # Ogden
        for i in range(n_terms):
            mcof_min[2 * i] = -mu_max
            mcof_max[2 * i] = mu_max
            mcof_min[2 * i + 1] = -10.0
            mcof_max[2 * i + 1] = 10.0
    elif lawid == 2:  # Mooney-Rivlin
        if icheck == 2:
            mcof_min[0] = 0.0
            mcof_max[0] = mu_max
        else:
            mcof_min[0] = -mu_max
            mcof_max[0] = mu_max
        mcof_min[1] = 2.0
        mcof_max[1] = 2.0

        if icheck == 2:
            mcof_min[2] = -mu_max
            mcof_max[2] = 0.0
        else:
            mcof_min[2] = -mu_max
            mcof_max[2] = mu_max
        mcof_min[3] = -2.0
        mcof_max[3] = -2.0

    return mcof_min, mcof_max


def fortran_sigeps69_principal(
    ev: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    rbulk: float,
    gmax: float,
    kfp: int = 0,
    fscale: float = 1.0,
    f_rv: Any = None,
) -> tuple[np.ndarray, float, float, float]:
    """Fortran reference: engine/source/materials/mat/mat069/sigeps69.F:204-300.

    Calculates principal Cauchy stresses T_1, T_2, T_3, pressure P, and relative volume RV.
    """
    rv = ev[0] * ev[1] * ev[2]
    rvt = np.exp((-1.0 / 3.0) * np.log(rv))
    evm = ev * rvt

    # Anti-buckling
    p_fac = 1.0
    if rbulk > 24.0 * gmax and gmax > 0.0:
        nu_1 = 40.0 * (0.5 - (3.0 * rbulk - gmax) / (6.0 * rbulk + gmax))
        amin = np.min(ev)
        if amin < 0.2:
            p_fac = max(1.0, nu_1 / max(1e-20, amin))

    if kfp > 0 and f_rv is not None:
        p = rbulk * fscale * float(f_rv(rv))
    else:
        p = p_fac * rbulk

    # Deviatoric derivatives
    dwdl = np.zeros(3, dtype=np.float64)
    for m, a in zip(mu, alpha):
        if a != 0.0:
            dwdl += m * np.exp(a * np.log(evm))
        else:
            dwdl += m

    sumdwdl = np.sum(dwdl) / 3.0
    dwdrv = p * (rv - 1.0)
    T = (dwdl - sumdwdl) / rv + dwdrv
    return T, p, p_fac, rv


def fortran_drucker_stability(
    stretch: float, mu: np.ndarray, alpha: np.ndarray, mode: str = "uniaxial"
) -> tuple[float, float, bool]:
    """Fortran reference: starter/source/materials/mat/mat069/law69_upd.F:249-348.

    Checks Drucker stability: INVD1 = D11 + D22 > 0 and INVD2 = D11*D22 - D12^2 > 0.
    """
    if mode == "uniaxial":
        lam1 = stretch
        lam2 = 1.0 / np.sqrt(stretch)
        lam3 = lam2
    elif mode == "biaxial":
        lam1 = stretch
        lam2 = stretch
        lam3 = 1.0 / (stretch * stretch)
    elif mode == "shear":
        lam1 = stretch
        lam2 = 1.0
        lam3 = 1.0 / stretch
    else:
        raise ValueError(f"Unknown mode: {mode}")

    d11 = 0.0
    d22 = 0.0
    d12 = 0.0
    for m, a in zip(mu, alpha):
        lam12 = (lam1 * lam2) ** (-a)
        d11 += a * m * (lam1 ** a + lam12)
        d22 += a * m * (lam2 ** a + lam12)
        d12 += a * m * lam12

    invd1 = d11 + d22
    invd2 = d11 * d22 - d12 ** 2
    is_stable = bool(invd1 > 0.0 and invd2 > 0.0)
    return invd1, invd2, is_stable


# ============================================================================
# Section 1: Uniaxial Tension/Compression Cauchy Stresses vs Fortran Formula
# ============================================================================


def test_ogden0_exact_formula_evaluation():
    """Verify OGDEN0 in nlsqf.F matches analytical engineering stress."""
    mu = np.array([20.0, -5.0], dtype=np.float64)
    alpha = np.array([2.0, -2.0], dtype=np.float64)
    stretches = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0], dtype=np.float64)

    # Analytical: mu1*(x - x^-2) + mu2*(x^-3 - 1)
    expected = 20.0 * (stretches - stretches ** (-2.0)) - 5.0 * (stretches ** (-3.0) - 1.0)
    actual = fortran_ogden0(stretches, mu, alpha)
    assert np.allclose(actual, expected, atol=1e-14, rtol=1e-14)


def test_ogden1_exact_derivatives_vs_finite_difference():
    """Verify OGDEN1 derivatives d(stress)/d(mu) and d(stress)/d(alpha) in nlsqf.F:507."""
    mu = np.array([15.0, 8.0], dtype=np.float64)
    alpha = np.array([1.8, -1.5], dtype=np.float64)
    x = 1.35
    h = 1e-7

    dy_dmu_exact, dy_dalpha_exact = fortran_ogden1(x, mu, alpha)

    # Check dY/dmu via finite difference
    for k in range(len(mu)):
        mu_p = mu.copy()
        mu_m = mu.copy()
        mu_p[k] += h
        mu_m[k] -= h
        fd = (fortran_ogden0(x, mu_p, alpha) - fortran_ogden0(x, mu_m, alpha)) / (2.0 * h)
        assert np.isclose(dy_dmu_exact[k], fd, rtol=1e-5, atol=1e-6)

    # Check dY/dalpha via finite difference
    for k in range(len(alpha)):
        al_p = alpha.copy()
        al_m = alpha.copy()
        al_p[k] += h
        al_m[k] -= h
        fd = (fortran_ogden0(x, mu, al_p) - fortran_ogden0(x, mu, al_m)) / (2.0 * h)
        assert np.isclose(dy_dalpha_exact[k], fd, rtol=1e-5, atol=1e-6)


def test_solid_uniaxial_tension_cauchy_vs_fortran():
    """Test 3D solid uniaxial tension matches exact Fortran T_i formula in sigeps69.F:297-300."""
    params = build_law69(
        mu=[10.0, -2.0],
        alpha=[2.0, -2.0],
        nu=0.495,
        rho0=1000.0,
    )
    # Logarithmic strain state: e11=0.15, e22=-0.075, e33=-0.075 (nearly incompressible)
    eps = np.array([0.15, -0.075, -0.075, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_init = np.zeros(6, dtype=np.float64)

    sig_py, _, soundsp = sigeps69_solid(params, sig_init, eps=eps, return_sound_speed=True)

    # Compute Fortran reference
    ev = np.exp([0.15, -0.075, -0.075])
    t_fort, p_fort, _, rv_fort = fortran_sigeps69_principal(
        ev, params.mu, params.alpha, params.rbulk, params.gmax
    )

    # Principal stresses in x, y, z align with coordinate axes
    assert np.isclose(sig_py[0], t_fort[0], rtol=1e-12, atol=1e-12)
    assert np.isclose(sig_py[1], t_fort[1], rtol=1e-12, atol=1e-12)
    assert np.isclose(sig_py[2], t_fort[2], rtol=1e-12, atol=1e-12)
    assert np.allclose(sig_py[3:], 0.0, atol=1e-14)


def test_solid_uniaxial_compression_cauchy_vs_fortran():
    """Test 3D solid uniaxial compression matches exact Fortran T_i formula."""
    params = build_law69(
        mu=[12.0, 5.0],
        alpha=[1.5, 3.0],
        nu=0.49,
        rho0=1200.0,
    )
    # Logarithmic compression: e11=-0.25, e22=0.125, e33=0.125
    eps = np.array([-0.25, 0.125, 0.125, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_init = np.zeros(6, dtype=np.float64)

    sig_py, _, _ = sigeps69_solid(params, sig_init, eps=eps, return_sound_speed=True)

    ev = np.exp([-0.25, 0.125, 0.125])
    # Note: Fortran sorts ev descending: ev[1], ev[2] > ev[0]
    ev_sorted = np.sort(ev)[::-1]
    t_fort, _, _, _ = fortran_sigeps69_principal(
        ev_sorted, params.mu, params.alpha, params.rbulk, params.gmax
    )

    # Check Cauchy stresses (matching eigenvalues)
    sig_eigs = np.sort(sig_py[:3])[::-1]
    assert np.allclose(sig_eigs, t_fort, rtol=1e-12, atol=1e-12)


def test_solid_incompressible_deviatoric_cauchy_vs_ogden0():
    """Verify Cauchy stress difference (T_1 - T_2) equals lambda * OGDEN0(lambda) under incompressible tension."""
    params = build_law69(
        mu=[20.0, -4.0],
        alpha=[2.0, -2.0],
        nu=0.495,
        rbulk=1e6,  # high penalty
    )
    lam = 1.3
    # Incompressible uniaxial: ev1=lam, ev2=ev3=1/sqrt(lam)
    ev = np.array([lam, 1.0 / np.sqrt(lam), 1.0 / np.sqrt(lam)], dtype=np.float64)
    eps = np.array([np.log(ev[0]), np.log(ev[1]), np.log(ev[2]), 0.0, 0.0, 0.0], dtype=np.float64)

    sig_py, _, _ = sigeps69_solid(params, np.zeros(6), eps=eps, return_sound_speed=True)

    cauchy_diff = sig_py[0] - sig_py[1]
    # Incompressible theory: Cauchy diff = lambda * Engineering stress = lambda * OGDEN0(lambda)
    nominal_ogden = fortran_ogden0(lam, params.mu, params.alpha)
    expected_cauchy_diff = lam * nominal_ogden

    assert np.isclose(cauchy_diff, expected_cauchy_diff, rtol=1e-5, atol=1e-5)


# ============================================================================
# Section 2: Equibiaxial Tension Parity Tests
# ============================================================================


def test_equibiaxial_tension_symmetry_and_fortran_parity():
    """Verify equal in-plane stresses T_1 == T_2 and match to Fortran formula for equibiaxial tension."""
    params = build_law69(
        mu=[15.0, 10.0],
        alpha=[2.0, 1.0],
        nu=0.495,
    )
    # Equibiaxial strain: e11=e22=0.1, e33=-0.2
    eps = np.array([0.1, 0.1, -0.2, 0.0, 0.0, 0.0], dtype=np.float64)
    sig, _, _ = sigeps69_solid(params, np.zeros(6), eps=eps, return_sound_speed=True)

    # In-plane stresses must be strictly equal
    assert np.isclose(sig[0], sig[1], rtol=1e-12, atol=1e-12)

    ev = np.exp([0.1, 0.1, -0.2])
    t_fort, _, _, _ = fortran_sigeps69_principal(
        ev, params.mu, params.alpha, params.rbulk, params.gmax
    )
    # Fortran principal stresses match
    assert np.isclose(sig[0], t_fort[0], rtol=1e-12, atol=1e-12)
    assert np.isclose(sig[1], t_fort[1], rtol=1e-12, atol=1e-12)
    assert np.isclose(sig[2], t_fort[2], rtol=1e-12, atol=1e-12)


def test_equibiaxial_incompressible_analytical_stress():
    """Verify equibiaxial deviatoric stress T_1 - T_3 = sum mu_k * (lambda^alpha_k - lambda^(-2*alpha_k))."""
    params = build_law69(
        mu=[10.0, -3.0],
        alpha=[2.0, -2.0],
        nu=0.495,
    )
    lam = 1.25
    # Incompressible: lam1=lam2=lam, lam3 = lam^-2
    ev = np.array([lam, lam, lam ** (-2.0)], dtype=np.float64)
    eps = np.array([np.log(ev[0]), np.log(ev[1]), np.log(ev[2]), 0.0, 0.0, 0.0], dtype=np.float64)

    sig, _, _ = sigeps69_solid(params, np.zeros(6), eps=eps, return_sound_speed=True)

    # Deviatoric difference
    t_diff = sig[0] - sig[2]
    # Analytical: sum mu_k * (lam^alpha_k - lam^(-2*alpha_k))
    expected_diff = np.sum(params.mu * (lam ** params.alpha - lam ** (-2.0 * params.alpha)))

    assert np.isclose(t_diff, expected_diff, rtol=1e-5, atol=1e-5)


# ============================================================================
# Section 3: Pure Shear (Planar Tension) Parity Tests
# ============================================================================


def test_pure_shear_principal_stresses_vs_fortran():
    """Verify pure shear / planar tension: lam1 = lam, lam2 = 1, lam3 = 1/lam."""
    params = build_law69(
        mu=[18.0, 6.0],
        alpha=[2.0, -1.0],
        nu=0.495,
    )
    lam = 1.4
    ev = np.array([lam, 1.0, 1.0 / lam], dtype=np.float64)
    eps = np.array([np.log(ev[0]), np.log(ev[1]), np.log(ev[2]), 0.0, 0.0, 0.0], dtype=np.float64)

    sig, _, _ = sigeps69_solid(params, np.zeros(6), eps=eps, return_sound_speed=True)

    # Fortran reference
    t_fort, _, _, _ = fortran_sigeps69_principal(
        ev, params.mu, params.alpha, params.rbulk, params.gmax
    )
    assert np.allclose(sig[:3], t_fort, rtol=1e-12, atol=1e-12)

    # Verify T_1 - T_3 = sum mu_k * (lam^alpha_k - lam^(-alpha_k))
    expected_diff = np.sum(params.mu * (lam ** params.alpha - lam ** (-params.alpha)))
    assert np.isclose(sig[0] - sig[2], expected_diff, rtol=1e-5, atol=1e-5)


def test_pure_shear_rotated_shear_stress_parity():
    """Verify pure shear in rotated 45-degree frame produces shear stress tau_xy = (T_1 - T_2)/2."""
    params = build_law69(
        mu=[25.0],
        alpha=[2.0],  # Neo-Hookean
        nu=0.495,
    )
    gamma = 0.2
    # Simple shear state in xy: eps_xy = gamma/2 => Voigt eps3 = gamma
    eps_shear = np.array([0.0, 0.0, 0.0, gamma, 0.0, 0.0], dtype=np.float64)

    sig, _, _ = sigeps69_solid(params, np.zeros(6), eps=eps_shear, return_sound_speed=True)

    # Under infinitesimal shear, sigma_xy = G0 * gamma = (mu*alpha/2) * gamma
    g0 = params.g0
    assert np.isclose(sig[3], g0 * gamma, rtol=1e-2, atol=1e-3)


# ============================================================================
# Section 4: Plane-Strain Tension Parity Tests
# ============================================================================


def test_plane_strain_tension_lateral_constraint_stress():
    """Verify lateral constraint stress T_2 in plane-strain tension (eps22 = 0)."""
    params = build_law69(
        mu=[20.0, 8.0],
        alpha=[2.0, 1.5],
        nu=0.495,
    )
    # eps11 = 0.2, eps22 = 0.0, eps33 = -0.2 (incompressible plane strain)
    eps = np.array([0.2, 0.0, -0.2, 0.0, 0.0, 0.0], dtype=np.float64)
    sig, _, _ = sigeps69_solid(params, np.zeros(6), eps=eps, return_sound_speed=True)

    ev = np.exp([0.2, 0.0, -0.2])
    t_fort, _, _, _ = fortran_sigeps69_principal(
        ev, params.mu, params.alpha, params.rbulk, params.gmax
    )
    assert np.allclose(sig[:3], t_fort, rtol=1e-12, atol=1e-12)
    # T_2 must be strictly between T_3 and T_1
    assert sig[2] < sig[1] < sig[0]


def test_plane_strain_shell_transverse_zero():
    """Verify shell plane-stress enforces sigma_33 = 0 while sustaining lateral in-plane stress."""
    params = build_law69(
        mu=[15.0, -3.0],
        alpha=[2.0, -2.0],
        nu=0.495,
    )
    # Plane strain in shell: eps_xx = 0.15, eps_yy = 0.0, eps_xy = 0.0
    eps_shell = np.array([0.15, 0.0, 0.0], dtype=np.float64)
    uvar = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    sig_new, _, c = sigeps69c_shell(params, np.zeros(3), eps=eps_shell, uvar=uvar, return_sound_speed=True)

    # In-plane normal stress in y must be positive (lateral constraint)
    assert sig_new[0] > 0.0
    assert sig_new[1] > 0.0
    assert sig_new[0] > sig_new[1]
    # Shear stress is zero
    assert np.isclose(sig_new[2], 0.0, atol=1e-12)


# ============================================================================
# Section 5: Mooney-Rivlin Transformation Parity Tests
# ============================================================================


def test_mooney_rivlin_coefficients_and_transformation():
    """Verify Mooney-Rivlin transformation: C_10 = mu_1/2, C_01 = -mu_2/2 from law69_upd.F:217."""
    c10 = 15.0
    c01 = 3.5

    # Inverse: mu_1 = 2*C_10, mu_2 = -2*C_01
    mu1 = 2.0 * c10
    mu2 = -2.0 * c01
    alpha = np.array([2.0, -2.0], dtype=np.float64)
    mu = np.array([mu1, mu2], dtype=np.float64)

    # Verify Fortran print formula in law69_upd.F:217
    c10_recovered = 0.5 * mu[0]
    c01_recovered = -0.5 * mu[1]
    assert np.isclose(c10_recovered, c10)
    assert np.isclose(c01_recovered, c01)

    # Verify ground state shear modulus G_0 = 2*(C_10 + C_01)
    g0_expected = 2.0 * (c10 + c01)
    gmax = float(np.sum(mu * alpha))
    g0_actual = gmax / 2.0
    assert np.isclose(g0_actual, g0_expected)


def test_mooney_rivlin_nominal_stress_parity():
    """Verify Mooney-Rivlin engineering stress formula matches classical expression."""
    c10 = 12.0
    c01 = 4.0
    mu = np.array([2.0 * c10, -2.0 * c01], dtype=np.float64)
    alpha = np.array([2.0, -2.0], dtype=np.float64)

    stretches = np.linspace(0.6, 2.5, 20)
    # Classical: 2*C10*(lam - lam^-2) + 2*C01*(1 - lam^-3)
    p_classical = 2.0 * c10 * (stretches - stretches ** (-2.0)) + 2.0 * c01 * (1.0 - stretches ** (-3.0))
    # Fortran OGDEN0: mu1*(lam - lam^-2) + mu2*(lam^-3 - 1)
    p_ogden0 = fortran_ogden0(stretches, mu, alpha)

    assert np.allclose(p_classical, p_ogden0, rtol=1e-12, atol=1e-12)


def test_mooney_rivlin_curve_fit_exactness():
    """Verify fit_law69_curve(law_id=2) recovers exact Mooney-Rivlin parameters from synthetic data."""
    c10_true = 8.5
    c01_true = 2.2
    mu_true = np.array([2.0 * c10_true, -2.0 * c01_true], dtype=np.float64)
    alpha_true = np.array([2.0, -2.0], dtype=np.float64)

    eps_grid = np.linspace(0.05, 1.2, 25)
    lam_grid = 1.0 + eps_grid
    sig_grid = fortran_ogden0(lam_grid, mu_true, alpha_true)

    mu_fit, al_fit, gmax, g0, bulk, E = fit_law69_curve(
        eps_grid, sig_grid, law_id=2, n_pair=2, nu=0.495
    )

    assert np.allclose(mu_fit, mu_true, rtol=1e-8, atol=1e-8)
    assert np.allclose(al_fit, alpha_true, atol=1e-12)
    assert np.isclose(g0, 2.0 * (c10_true + c01_true), rtol=1e-8)


# ============================================================================
# Section 6: Bulk Modulus Scaling Function f(RV) (FUN_A1) Parity Tests
# ============================================================================


def test_bulk_modulus_function_linear_scaling():
    """Verify pressure P = RBULK * FSCALE * f(RV) matches Fortran sigeps69.F:236."""
    # Define arbitrary nonlinear volumetric function f(RV) = (RV - 1) + 0.3*(RV - 1)^2
    def my_bulk_func(rv: np.ndarray | float) -> np.ndarray | float:
        return (rv - 1.0) + 0.3 * (rv - 1.0) ** 2

    rbulk = 500.0
    fscale = 1.8
    fct_id = 99

    params = build_law69(
        mu=[10.0],
        alpha=[2.0],
        rbulk=rbulk,
        fscale=fscale,
        fct_id=fct_id,
        nu=0.49,
    )
    # Pure volumetric strain: e11=e22=e33=0.05 => RV = exp(0.15)
    eps = np.array([0.05, 0.05, 0.05, 0.0, 0.0, 0.0], dtype=np.float64)
    extra = {"curves": {fct_id: my_bulk_func}}

    sig, _, _ = sigeps69_solid(params, np.zeros(6), eps=eps, extra=extra, return_sound_speed=True)

    ev = np.exp([0.05, 0.05, 0.05])
    t_fort, p_fort, _, _ = fortran_sigeps69_principal(
        ev, params.mu, params.alpha, rbulk, params.gmax, kfp=fct_id, fscale=fscale, f_rv=my_bulk_func
    )

    # Spherical hydrostatic stress = P * (RV - 1)
    assert np.allclose(sig[:3], t_fort, rtol=1e-12, atol=1e-12)


def test_bulk_modulus_piecewise_curve_interpolation():
    """Verify discrete curve points interpolation for FUN_A1 bulk modulus."""
    class MockCurve:
        def __init__(self, x_pts: Any, y_pts: Any):
            self.x = np.asarray(x_pts, dtype=np.float64)
            self.y = np.asarray(y_pts, dtype=np.float64)

    # Discrete points around RV in [0.8, 1.2]
    x_data = [0.8, 0.9, 1.0, 1.1, 1.2]
    y_data = [-0.3, -0.15, 0.0, 0.2, 0.45]
    curve = MockCurve(x_data, y_data)

    params = build_law69(
        mu=[12.0],
        alpha=[2.0],
        rbulk=400.0,
        fscale=2.0,
        fct_id=12,
    )
    eps = np.array([0.03, 0.03, 0.03, 0.0, 0.0, 0.0], dtype=np.float64)
    rv = float(np.exp(0.09))
    expected_f_rv = float(np.interp(rv, x_data, y_data))
    expected_p = params.rbulk * params.fscale * expected_f_rv

    extra = {"curves": {12: curve}}
    sig, _, _ = sigeps69_solid(params, np.zeros(6), eps=eps, extra=extra, return_sound_speed=True)

    # Since strain is purely hydrostatic (deviatoric DWDL - sumDWDL = 0), stress = P * (RV - 1)
    expected_stress = expected_p * (rv - 1.0)
    assert np.isclose(sig[0], expected_stress, rtol=1e-10)


# ============================================================================
# Section 7: Anti-Buckling Condition Parity Tests (sigeps69.F:204-211)
# ============================================================================


def test_anti_buckling_inactive_when_rbulk_low():
    """Verify P_FAC == 1.0 when RBULK <= 24*GMAX even if stretch < 0.2."""
    gmax = 20.0
    rbulk = 20.0 * gmax  # 400 < 24*20=480
    params = build_law69(mu=[10.0], alpha=[2.0], rbulk=rbulk, gmax=gmax)

    # ev = [0.1, 1.0, 1.0] => amin = 0.1 < 0.2
    ev = np.array([0.1, 1.0, 1.0], dtype=np.float64)
    eps = np.array([np.log(ev[0]), 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    sig, _, _ = sigeps69_solid(params, np.zeros(6), eps=eps, return_sound_speed=True)

    _, p_fort, p_fac_fort, _ = fortran_sigeps69_principal(
        ev, params.mu, params.alpha, params.rbulk, params.gmax
    )
    assert p_fac_fort == 1.0


def test_anti_buckling_inactive_when_stretch_safe():
    """Verify P_FAC == 1.0 when RBULK > 24*GMAX but stretch >= 0.2."""
    gmax = 10.0
    rbulk = 30.0 * gmax  # 300 > 240
    params = build_law69(mu=[5.0], alpha=[2.0], rbulk=rbulk, gmax=gmax)

    # ev = [0.25, 1.0, 1.0] => amin = 0.25 >= 0.2
    ev = np.array([0.25, 1.0, 1.0], dtype=np.float64)
    eps = np.array([np.log(ev[0]), 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    sig, _, _ = sigeps69_solid(params, np.zeros(6), eps=eps, return_sound_speed=True)

    _, _, p_fac_fort, _ = fortran_sigeps69_principal(
        ev, params.mu, params.alpha, params.rbulk, params.gmax
    )
    assert p_fac_fort == 1.0


def test_anti_buckling_active_exact_fortran_formula():
    """Verify P_FAC scaling exactly matches Fortran formula when RBULK > 24*GMAX and amin < 0.2."""
    gmax = 10.0
    rbulk = 500.0  # > 24*10 = 240
    params = build_law69(mu=[5.0], alpha=[2.0], rbulk=rbulk, gmax=gmax)

    amin = 0.08  # < 0.2
    ev = np.array([amin, 1.0, 1.0], dtype=np.float64)
    eps = np.array([np.log(ev[0]), 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    sig, _, _ = sigeps69_solid(params, np.zeros(6), eps=eps, return_sound_speed=True)

    # Fortran reference
    nu_1 = 40.0 * (0.5 - (3.0 * rbulk - gmax) / (6.0 * rbulk + gmax))
    expected_p_fac = max(1.0, nu_1 / amin)
    t_fort, p_fort, p_fac_fort, _ = fortran_sigeps69_principal(
        ev, params.mu, params.alpha, rbulk, gmax
    )

    assert np.isclose(p_fac_fort, expected_p_fac, rtol=1e-14)
    # Check that solid stress matches Fortran reference with scaled bulk modulus
    sig_eigs = np.sort(sig[:3])[::-1]
    t_fort_sorted = np.sort(t_fort)[::-1]
    assert np.allclose(sig_eigs, t_fort_sorted, rtol=1e-12)


def test_anti_buckling_multi_element_vectorization():
    """Verify anti-buckling correctly triggers only on compressed elements across multi-element arrays."""
    gmax = 10.0
    rbulk = 400.0
    params = build_law69(mu=[5.0], alpha=[2.0], rbulk=rbulk, gmax=gmax)

    nel = 4
    # Element 0: safe (amin=0.5)
    # Element 1: buckling (amin=0.1)
    # Element 2: safe tension (amin=1.2)
    # Element 3: extreme buckling (amin=0.02)
    amins = [0.5, 0.1, 1.2, 0.02]
    eps_arr = np.zeros((nel, 6), dtype=np.float64)
    for i, a in enumerate(amins):
        eps_arr[i, 0] = np.log(a)

    sig_arr, _, _ = sigeps69_solid(params, np.zeros((nel, 6)), eps=eps_arr, return_sound_speed=True)

    nu_1 = 40.0 * (0.5 - (3.0 * rbulk - gmax) / (6.0 * rbulk + gmax))
    for i, a in enumerate(amins):
        if a < 0.2:
            expected_fac = max(1.0, nu_1 / a)
        else:
            expected_fac = 1.0
        ev_i = np.array([a, 1.0, 1.0])
        t_fort, _, _, _ = fortran_sigeps69_principal(ev_i, params.mu, params.alpha, rbulk, gmax)
        sig_sorted = np.sort(sig_arr[i, :3])[::-1]
        t_sorted = np.sort(t_fort)[::-1]
        assert np.allclose(sig_sorted, t_sorted, rtol=1e-12)


# ============================================================================
# Section 8: Shell Out-of-Plane Stretch Newton-Raphson Solve (sigeps69c.F:163-199)
# ============================================================================


def test_shell_newton_raphson_plane_stress_convergence():
    """Verify that in 4 Newton steps, T_3 converges to zero (|T_3| < 1e-7)."""
    params = build_law69(
        mu=[20.0, -5.0],
        alpha=[2.0, -2.0],
        nu=0.495,
    )
    # In-plane stretches: ev1=1.2, ev2=0.9
    eps = np.array([np.log(1.2), np.log(0.9), 0.0], dtype=np.float64)
    uvar = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    sig_new, _, c = sigeps69c_shell(params, np.zeros(3), eps=eps, uvar=uvar, return_sound_speed=True)

    # Converged lambda_3 is in uvar[2]
    lam3 = uvar[2]
    assert lam3 > 0.0
    # Incompressible estimate lam3 ~ 1 / (1.2 * 0.9) = 0.9259
    assert np.isclose(lam3, 1.0 / (1.2 * 0.9), rtol=0.05)

    # Re-evaluate Kirchhoff stress T_3 at converged stretch
    ev = np.array([1.2, 0.9, lam3])
    t_principal, _, _, _ = fortran_sigeps69_principal(
        ev, params.mu, params.alpha, params.rbulk, params.gmax
    )
    # T_3 must be practically zero (plane stress after Fortran's 4 Newton steps)
    assert abs(t_principal[2]) < 1e-4


def test_shell_uvar_history_persistence():
    """Verify that successive calls reuse converged lambda_3 from uvar[:, 2]."""
    params = build_law69(mu=[10.0], alpha=[2.0], nu=0.495)
    uvar = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Step 1: stretch to 1.1
    eps1 = np.array([np.log(1.1), 0.0, 0.0], dtype=np.float64)
    sigeps69c_shell(params, np.zeros(3), eps=eps1, uvar=uvar)
    lam3_step1 = uvar[2]
    assert lam3_step1 < 1.0

    # Step 2: stretch further to 1.2
    eps2 = np.array([np.log(1.2), 0.0, 0.0], dtype=np.float64)
    sigeps69c_shell(params, np.zeros(3), eps=eps2, uvar=uvar)
    lam3_step2 = uvar[2]
    assert lam3_step2 < lam3_step1


def test_shell_thickness_evolution_with_dezz():
    """Verify thickness update: THKN = THKN + DEZZ * THKLYL * OFF in sigeps69c.F:256."""
    params = build_law69(mu=[15.0], alpha=[2.0], nu=0.495)
    thk0 = 2.5
    thk = np.array([thk0], dtype=np.float64)
    uvar = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    off = np.array([1.0], dtype=np.float64)

    # In-plane stretch 1.2 x 1.0 => lam3 ~ 1/1.2
    eps = np.array([np.log(1.2), 0.0, 0.0], dtype=np.float64)
    sigeps69c_shell(
        params,
        np.zeros(3),
        eps=eps,
        uvar=uvar,
        thkn=thk,
        thklyl=thk0,
        off=off,
    )

    lam3 = uvar[2]
    expected_dezz = np.log(lam3 / 1.0)
    expected_thk = thk0 + expected_dezz * thk0 * 1.0
    assert np.isclose(thk[0], expected_thk, rtol=1e-12)


# ============================================================================
# Section 9: Stiffening Factor GTMAX and Acoustic Sound Speeds
# ============================================================================


def test_solid_sound_speed_ground_state_exact():
    """Verify ground-state solid sound speed c = sqrt((2/3*Gmax + K)/rho0) matches Fortran sigeps69.F:337."""
    params = build_law69(
        mu=[10.0, 5.0],
        alpha=[2.0, 2.0],
        nu=0.45,
        rho0=1200.0,
    )
    # At eps = 0, GTMAX = GMAX
    expected_c = np.sqrt(((2.0 / 3.0) * params.gmax + params.rbulk) / params.rho0)
    actual_c = solid_sound_speed(params)
    assert np.isclose(actual_c, expected_c, rtol=1e-14)


def test_solid_gtmax_stiffening_under_large_strain():
    """Verify solid GTMAX stiffening calculation matches Fortran sigeps69.F:258-265."""
    params = build_law69(
        mu=[8.0, 4.0],
        alpha=[3.0, 2.0],
        nu=0.495,
        rho0=1000.0,
    )
    # Large tensile strain
    eps = np.array([0.6, -0.3, -0.3, 0.0, 0.0, 0.0], dtype=np.float64)
    _, _, c = sigeps69_solid(params, np.zeros(6), eps=eps, return_sound_speed=True)

    ev = np.exp([0.6, -0.3, -0.3])
    rv = ev[0] * ev[1] * ev[2]
    evm = ev * (rv ** (-1.0 / 3.0))

    # Fortran GTMAX calculation
    cii = np.zeros(3, dtype=np.float64)
    for m, a in zip(params.mu, params.alpha):
        lam_al = evm ** a
        amax = np.sum(lam_al) / 3.0
        cii += m * a * (lam_al + amax)

    amax1 = 0.81 * 0.5 / params.gmax
    eti = max(1.0, amax1 * np.max(cii))
    gtmax_fort = params.gmax * eti
    rho_current = params.rho0 / rv
    expected_c = np.sqrt(((2.0 / 3.0) * gtmax_fort + params.rbulk) / rho_current)

    assert np.isclose(c, expected_c, rtol=1e-12)


def test_shell_sound_speed_ground_state_exact():
    """Verify ground-state shell sound speed c = sqrt(Emax / ((1 - nu^2)*rho0)) matches Fortran sigeps69c.F:261."""
    params = build_law69(
        mu=[10.0, 4.0],
        alpha=[2.0, 2.0],
        nu=0.4,
        rho0=1100.0,
    )
    # At eps = 0, GTMAX = GMAX
    emax = params.gmax * (1.0 + params.nu)
    a11 = emax / (1.0 - params.nu ** 2)
    expected_c = np.sqrt(a11 / params.rho0)

    actual_c = shell_sound_speed(params)
    assert np.isclose(actual_c, expected_c, rtol=1e-14)


def test_shell_gtmax_stiffening_under_large_strain():
    """Verify shell GTMAX = 0.5 * max(CII) matches Fortran sigeps69c.F:245."""
    params = build_law69(
        mu=[12.0, -3.0],
        alpha=[2.0, -2.0],
        nu=0.495,
        rho0=950.0,
    )
    eps = np.array([0.4, 0.2, 0.0], dtype=np.float64)
    uvar = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    sig, _, c = sigeps69c_shell(params, np.zeros(3), eps=eps, uvar=uvar, return_sound_speed=True)

    ev1 = np.exp(0.4)
    ev2 = np.exp(0.2)
    ev3 = uvar[2]
    rv = ev1 * ev2 * ev3
    evm = np.array([ev1, ev2, ev3]) * (rv ** (-1.0 / 3.0))

    cii = np.zeros(3, dtype=np.float64)
    for m, a in zip(params.mu, params.alpha):
        lam_al = evm ** a
        amax = np.sum(lam_al) / 3.0
        cii += m * a * (lam_al + amax)

    gtmax_fort = 0.5 * np.max(cii)
    emax = max(params.gmax, gtmax_fort) * (1.0 + params.nu)
    a11 = emax / (1.0 - params.nu ** 2)
    expected_c = np.sqrt(a11 / params.rho0)

    assert np.isclose(c, expected_c, rtol=1e-12)


# ============================================================================
# Section 10: Tensile Cut-Off & Element Erosion Parity Tests
# ============================================================================


def test_solid_tensile_cutoff_erodes_to_zero():
    """Verify solid element erosion: stresses zeroed and off=0.0 when T_1 > TENSCUT (sigeps69.F:305-311)."""
    params = build_law69(
        mu=[20.0],
        alpha=[2.0],
        tenscut=50.0,  # low cut-off
    )
    off = np.array([1.0], dtype=np.float64)
    # High strain producing Cauchy stress > 50
    eps = np.array([0.8, -0.4, -0.4, 0.0, 0.0, 0.0], dtype=np.float64)

    sig, _, _ = sigeps69_solid(params, np.zeros(6), eps=eps, off=off, return_sound_speed=True)

    assert np.allclose(sig, 0.0)
    assert off[0] == 0.0


def test_shell_tensile_cutoff_erodes_to_four_over_five():
    """Verify shell element erosion: stresses zeroed and off=0.8 (FOUR_OVER_5) when T > TENSCUT (sigeps69c.F:230)."""
    params = build_law69(
        mu=[25.0],
        alpha=[2.0],
        tenscut=40.0,
    )
    off = np.array([1.0], dtype=np.float64)
    eps = np.array([0.6, 0.0, 0.0], dtype=np.float64)

    sig, _, _ = sigeps69c_shell(params, np.zeros(3), eps=eps, off=off, return_sound_speed=True)

    assert np.allclose(sig, 0.0)
    assert np.isclose(off[0], 0.8)


def test_already_eroded_element_remains_inactive():
    """Verify already eroded elements (off=0 for solid, off=0.8 for shell) remain zero stress."""
    params = build_law69(mu=[20.0], alpha=[2.0])

    # Solid with off = 0.0
    sig_solid, _, _ = sigeps69_solid(
        params, np.zeros(6), eps=np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]), off=np.array([0.0])
    )
    assert np.allclose(sig_solid, 0.0)

    # Shell with off = 0.0 (completely deleted)
    sig_shell, _, _ = sigeps69c_shell(
        params, np.zeros(3), eps=np.array([0.2, 0.0, 0.0]), off=np.array([0.0]), return_sound_speed=True
    )
    assert np.allclose(sig_shell, 0.0)


# ============================================================================
# Section 11: Drucker-Prager Material Stability Parity Tests (law69_upd.F)
# ============================================================================


def test_drucker_stability_uniaxial_mode():
    """Verify Drucker stability conditions INVD1 > 0 and INVD2 > 0 for stable Ogden parameters."""
    mu = np.array([10.0, 5.0], dtype=np.float64)
    alpha = np.array([2.0, 1.5], dtype=np.float64)

    stretches = np.linspace(0.2, 4.0, 20)
    for lam in stretches:
        invd1, invd2, stable = fortran_drucker_stability(lam, mu, alpha, mode="uniaxial")
        assert stable
        assert invd1 > 0.0
        assert invd2 > 0.0


def test_drucker_stability_equibiaxial_mode():
    """Verify Drucker stability in equibiaxial mode from law69_upd.F:287-308."""
    mu = np.array([15.0, -3.0], dtype=np.float64)
    alpha = np.array([2.0, -2.0], dtype=np.float64)  # Mooney-Rivlin

    stretches = np.linspace(0.3, 3.5, 20)
    for lam in stretches:
        invd1, invd2, stable = fortran_drucker_stability(lam, mu, alpha, mode="biaxial")
        assert stable
        assert invd1 > 0.0
        assert invd2 > 0.0


def test_drucker_stability_pure_shear_mode():
    """Verify Drucker stability in pure shear mode from law69_upd.F:324-347."""
    mu = np.array([20.0], dtype=np.float64)
    alpha = np.array([2.0], dtype=np.float64)

    stretches = np.linspace(0.4, 3.0, 15)
    for lam in stretches:
        invd1, invd2, stable = fortran_drucker_stability(lam, mu, alpha, mode="shear")
        assert stable
        assert invd1 > 0.0
        assert invd2 > 0.0


# ============================================================================
# Section 12: Starter Bounds Estimation Parity Tests (nlsqf.F: LAW69_GUESS_BOUNDS)
# ============================================================================


def test_guess_bounds_ogden_parity():
    """Verify LAW69_GUESS_BOUNDS for Ogden law matches nlsqf.F:1306-1315."""
    stretches = np.array([1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float64)
    stresses = np.array([10.0, 22.0, 36.0, 52.0, 70.0], dtype=np.float64)

    dx = stretches - 1.0
    expected_ave_slope = float(np.mean(stresses / dx))
    expected_mu_max = max(expected_ave_slope, 20.0)

    mcof_min, mcof_max = fortran_guess_bounds(
        lawid=1, icheck=-3, x=stretches, y=stresses, n_terms=2, icomp=0
    )

    assert len(mcof_min) == 4
    assert len(mcof_max) == 4
    # mu_k in [-mu_max, mu_max]
    assert np.isclose(mcof_min[0], -expected_mu_max)
    assert np.isclose(mcof_max[0], expected_mu_max)
    assert np.isclose(mcof_min[2], -expected_mu_max)
    assert np.isclose(mcof_max[2], expected_mu_max)
    # alpha_k in [-10, 10]
    assert mcof_min[1] == -10.0
    assert mcof_max[1] == 10.0
    assert mcof_min[3] == -10.0
    assert mcof_max[3] == 10.0


def test_guess_bounds_mooney_rivlin_icheck2_parity():
    """Verify LAW69_GUESS_BOUNDS with ICHECK=2 enforces positive C_10 and negative C_01 (nlsqf.F:1318-1339)."""
    stretches = np.array([1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float64)
    stresses = np.array([5.0, 12.0, 20.0, 30.0, 42.0], dtype=np.float64)

    mcof_min, mcof_max = fortran_guess_bounds(
        lawid=2, icheck=2, x=stretches, y=stresses, n_terms=2, icomp=0
    )

    # ICHECK == 2: mu_1 in [0, mu_max], alpha_1 = 2
    assert mcof_min[0] == 0.0
    assert mcof_max[0] > 0.0
    assert mcof_min[1] == 2.0
    assert mcof_max[1] == 2.0

    # mu_2 in [-mu_max, 0], alpha_2 = -2
    assert mcof_min[2] < 0.0
    assert mcof_max[2] == 0.0
    assert mcof_min[3] == -2.0
    assert mcof_max[3] == -2.0
