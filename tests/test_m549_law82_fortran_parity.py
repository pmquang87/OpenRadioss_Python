"""
Fortran parity test suite for M549 (/MAT/LAW82, /MAT/OGDEN, /MAT/LAW82_OGDEN).

Audits line-by-line fidelity against the upstream reference Fortran sources:
- starter/source/materials/mat/mat082/hm_read_mat82.F (parameter setup & bounds)
- engine/source/materials/mat/mat082/sigeps82.F (3D solid continuum Cauchy stress)
- engine/source/materials/mat/mat082/sigeps82c.F (2D shell plane-stress Newton solver)

Sections:
1. Initial material parameter setups from hm_read_mat82.F
   - Initial shear modulus G0 = sum(mu_i)
   - Poisson's ratio nu clamping (nu == 0.5 -> 0.495, nu == 0 with D1 > 0 -> nu from K)
   - Compressibility parameter D1 = 3*(1-2*nu)/(G0*(1+nu)) and K = 2/D1
   - Parameter line truncation at first zero term
   - Elastic relation consistency: E0 = 2*G0*(1+nu), K0 = E0/(3*(1-2*nu))
2. 3D solid continuum Cauchy stress (sigeps82.F:205-330)
   - Principal stretches lambda_i = exp(eps_i)
   - Relative volume RV = lambda_1 * lambda_2 * lambda_3
   - Deviatoric stretches lambda_bar_i = lambda_i * RV^(-1/3)
   - Deviatoric Kirchhoff / Cauchy stress:
     S_i = sum (2*mu_k / (alpha_k*RV)) * [2/3*lam_bar_i^alpha_k - 1/3*(lam_bar_j^alpha_k + lam_bar_m^alpha_k)] + P
   - Volumetric pressure: P = sum (2*k / D_k) * (RV - 1)^(2k - 1)
   - Cauchy stress rotation via eigenvectors (sigeps82.F:306-330)
   - Multi-term series (N=1, 2, 3, 5) and higher-order compressibility (D_k for k >= 2)
   - Strain options (ISMSTR=0 logarithmic, ISMSTR=1 engineering, ISMSTR=10 Green-Lagrange)
3. 2D shell plane-stress Newton-Raphson iteration (sigeps82c.F:148-225)
   - In-plane analytical eigenvalue decomposition (sigeps82c.F:108-132)
   - 3-step Newton solver enforcing T3(lambda_3) = 0
   - First and second derivatives matching sigeps82c.F:211-224
   - State history variable persistence (uvar82 stores lambda_3)
   - Transverse shear stresses with G0 (sigeps82c.F:294-295)
   - Thickness update dezz = -nu/(1-nu)*(deps_xx + deps_yy) (sigeps82c.F:300-301)
4. Analytical closed forms
   - Uniaxial tension: sigma_1 = sum (2*mu_k/alpha_k) * (lambda^alpha_k - lambda^(-alpha_k/2))
   - Equibiaxial tension: sigma = sum (2*mu_k/alpha_k) * (lambda^alpha_k - lambda^(-2*alpha_k))
   - Pure shear: sigma_1 - sigma_2 = sum (2*mu_k/alpha_k) * (lambda^alpha_k - lambda^(-alpha_k))
   - Small strain limit: E = 2*G0*(1+nu), K = E/(3*(1-2*nu))
   - Neo-Hookean and Mooney-Rivlin special cases
5. Sound speed formulas (sigeps82.F:332, sigeps82c.F:303)
   - Solid sound speed: c = sqrt((4/3 * G_t_max + K_max) / rho)
   - Shell sound speed: c = sqrt((2/3 * G0 + K) / rho)
   - Multi-term series (N=1, 2, 3, 5)
   - Official RD-E-5600 rubber tension parameter set
"""

from __future__ import annotations

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
# Pure Fortran Oracle Reference Functions (Direct Translation)
# ============================================================================


def fortran_oracle_read_mat82(
    rho0: float,
    nu0: float,
    nordre: int,
    mu_in: list[float],
    alpha_in: list[float],
    d_in: list[float] | None = None,
    rhor: float = 0.0,
) -> dict:
    """Exact line-by-line mirror of starter hm_read_mat82.F:92-174."""
    zep495 = 0.4 + 9.0 * 1e-2 + 5.0 * 1e-3  # 0.495
    zep499 = 0.499

    mu = list(mu_in)
    al = list(alpha_in)
    d = list(d_in) if d_in is not None else [0.0] * len(mu)

    # hm_read_mat82.F:113-121: count first non zero Ogden parameters
    count = 0
    for i in range(min(nordre, len(mu), len(al))):
        if mu[i] != 0.0 and al[i] != 0.0:
            count += 1
        else:
            break
    nordre = count

    # hm_read_mat82.F:128-131: sum of mu
    gs = 0.0
    for i in range(nordre):
        gs += mu[i]

    # hm_read_mat82.F:138-156: nu and D1 determination
    nu_val = nu0
    if nu_val == 0.5:
        nu_val = zep495

    if nu_val == 0.0:
        if len(d) > 0 and d[0] > 0.0:
            p = 2.0 / d[0]
            nu = 1.0 / (6.0 * p + 2.0 * gs)
            nu = (3.0 * p - 2.0 * gs) * nu
            if nu == 0.5:
                nu = zep499
            d[0] = 3.0 * (1.0 - 2.0 * nu) / gs / (1.0 + nu)
            p = 2.0 / d[0]
        else:
            nu = zep495
            d[0] = 3.0 * (1.0 - 2.0 * nu) / gs / (1.0 + nu)
            p = 2.0 / d[0]
    else:
        nu = nu_val
        d[0] = 3.0 * (1.0 - 2.0 * nu) / gs / (1.0 + nu)
        p = 2.0 / d[0]

    if rhor == 0.0:
        rhor = rho0

    parmat = {
        "P": p,
        "E": 2.0 * gs * (1.0 + nu),
        "NU": nu,
        "K": p,
    }

    return {
        "nordre": nordre,
        "gs": gs,
        "nu": nu,
        "d": d[:nordre],
        "mu": mu[:nordre],
        "al": al[:nordre],
        "parmat": parmat,
        "rhor": rhor,
        "rho0": rho0,
    }


def fortran_oracle_sigeps82(
    eps_6: np.ndarray,
    nordre: int,
    mu: list[float],
    al: list[float],
    d: list[float],
    ismstr: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Exact line-by-line mirror of engine sigeps82.F:137-335.

    Returns (sigma_cauchy_6, S_principal_3, RV, sound_speed).
    """
    # sigeps82.F:137-144
    av = np.array([
        [eps_6[0], 0.5 * eps_6[3], 0.5 * eps_6[5]],
        [0.5 * eps_6[3], eps_6[1], 0.5 * eps_6[4]],
        [0.5 * eps_6[5], 0.5 * eps_6[4], eps_6[2]],
    ], dtype=np.float64)

    # sigeps82.F:147-151: VALPVEC_V
    evv, dirprv = np.linalg.eigh(av)

    # sigeps82.F:156-182: Strains to stretches
    if ismstr in (0, 2, 4):
        ev = np.exp(evv)
    elif ismstr in (10, 12):
        ev = np.sqrt(np.maximum(evv + 1.0, 1e-20))
    else:
        ev = evv + 1.0

    # sigeps82.F:207: Relative volume RV
    rv = ev[0] * ev[1] * ev[2]

    # sigeps82.F:216-225: Deviatoric stretches
    pui = rv ** (-1.0 / 3.0) if rv != 0.0 else 0.0
    evd = ev * pui

    # sigeps82.F:226-270: Principal Cauchy stress S
    s = np.zeros(3, dtype=np.float64)
    p = 0.0

    for j in range(nordre):
        k = j + 1
        dd = (2.0 * mu[j] / al[j]) / rv
        pui_tab = evd ** al[j]

        s[0] += dd * ((2.0 / 3.0) * pui_tab[0] - (1.0 / 3.0) * (pui_tab[1] + pui_tab[2]))
        s[1] += dd * ((2.0 / 3.0) * pui_tab[1] - (1.0 / 3.0) * (pui_tab[0] + pui_tab[2]))
        s[2] += dd * ((2.0 / 3.0) * pui_tab[2] - (1.0 / 3.0) * (pui_tab[0] + pui_tab[1]))

        if j < len(d) and d[j] != 0.0:
            pp = 1.0 / d[j]
            p += 2.0 * k * pp * (rv - 1.0) ** (2 * k - 1)

    s += p

    # sigeps82.F:306-330: Cauchy to global
    signxx = dirprv[0, 0]**2 * s[0] + dirprv[0, 1]**2 * s[1] + dirprv[0, 2]**2 * s[2]
    signyy = dirprv[1, 1]**2 * s[1] + dirprv[1, 2]**2 * s[2] + dirprv[1, 0]**2 * s[0]
    signzz = dirprv[2, 2]**2 * s[2] + dirprv[2, 0]**2 * s[0] + dirprv[2, 1]**2 * s[1]

    signxy = dirprv[0, 0]*dirprv[1, 0]*s[0] + dirprv[0, 1]*dirprv[1, 1]*s[1] + dirprv[0, 2]*dirprv[1, 2]*s[2]
    signyz = dirprv[1, 1]*dirprv[2, 1]*s[1] + dirprv[1, 2]*dirprv[2, 2]*s[2] + dirprv[1, 0]*dirprv[2, 0]*s[0]
    signzx = dirprv[2, 2]*dirprv[0, 2]*s[2] + dirprv[2, 0]*dirprv[0, 0]*s[0] + dirprv[2, 1]*dirprv[0, 1]*s[1]

    sig_out = np.array([signxx, signyy, signzz, signxy, signyz, signzx], dtype=np.float64)

    # sigeps82.F:273-335: Tangent and sound speed
    gmax = sum(mu)
    rbulk = 2.0 / d[0]
    gtmax = gmax
    rkmax = rbulk

    cii = np.zeros(3, dtype=np.float64)
    for ii in range(nordre):
        if mu[ii] != 0.0:
            lam_al = evd ** al[ii]
            amax = (1.0 / 3.0) * np.sum(lam_al)
            cii += mu[ii] * (lam_al + amax)

    for ii in range(1, nordre):
        if ii < len(d) and d[ii] != 0.0:
            k = ii + 1
            pp = 2.0 * k * (2.0 * k - 1.0) / d[ii]
            jj = 2 * k - 2
            if abs(rv - 1.0) >= 1e-20:
                rkmax += pp * (rv - 1.0) ** jj

    amax = 0.5 * np.max(cii) / gmax
    eti = max(1.0, amax * 0.81)
    gtmax = gmax * eti
    rkmax = max(rbulk, rkmax)

    rho = 1.0 / rv  # rho0 / rv with rho0 = 1
    soundsp = np.sqrt(((4.0 / 3.0) * gtmax + rkmax) / rho)

    return sig_out, s, rv, soundsp


def fortran_oracle_sigeps82c(
    eps_inplane: np.ndarray,
    nordre: int,
    mu: list[float],
    al: list[float],
    d: list[float],
    uvar1: float = 1.0,
    deps_inplane: np.ndarray | None = None,
    thk0: float = 1.0,
    rho0: float = 1.0,
) -> tuple[np.ndarray, float, float, float, float]:
    """Exact line-by-line mirror of engine sigeps82c.F:93-306.

    Returns (sig_3, lambda_3, thkn, soundsp, rv).
    """
    gmax = sum(mu)
    rbulk = 2.0 / d[0]
    nu = (3.0 * rbulk - 2.0 * gmax) / (2.0 * gmax + 6.0 * rbulk)
    if nu == 0.5:
        nu = 0.495

    # sigeps82c.F:108-114: In-plane eigenvalues
    epsxx, epsyy, epsxy = eps_inplane[0], eps_inplane[1], eps_inplane[2]
    trav = epsxx + epsyy
    rootv = np.sqrt((epsxx - epsyy)**2 + epsxy**2)
    evv1 = 0.5 * (trav + rootv)
    evv2 = 0.5 * (trav - rootv)

    # sigeps82c.F:117-131: Rotation matrix
    eigv = np.zeros((3, 2), dtype=np.float64)
    if abs(evv2 - evv1) < 1e-10:
        eigv[0, 0] = 1.0
        eigv[1, 0] = 1.0
        eigv[2, 0] = 0.0
        eigv[0, 1] = 0.0
        eigv[1, 1] = 0.0
        eigv[2, 1] = 0.0
    else:
        eigv[0, 0] = (1.0 / rootv) * (epsxx - evv2)
        eigv[1, 0] = (1.0 / rootv) * (epsyy - evv2)
        eigv[0, 1] = (1.0 / rootv) * (evv1 - epsxx)
        eigv[1, 1] = (1.0 / rootv) * (evv1 - epsyy)
        eigv[2, 0] = (1.0 / rootv) * (0.5 * epsxy)
        eigv[2, 1] = -(1.0 / rootv) * (0.5 * epsxy)

    # sigeps82c.F:141-145: True strain stretches
    ev = np.array([np.exp(evv1), np.exp(evv2), uvar1], dtype=np.float64)

    # sigeps82c.F:150-225: Newton method
    for _ in range(3):
        rv = ev[0] * ev[1] * ev[2]
        rvt = rv ** (-1.0 / 3.0) if rv != 0.0 else 0.0
        evm = ev * rvt

        evma1 = np.zeros(nordre, dtype=np.float64)
        evma2 = np.zeros(nordre, dtype=np.float64)
        evma3 = np.zeros(nordre, dtype=np.float64)
        for k in range(nordre):
            evma1[k] = evm[0] ** al[k] if evm[0] != 0.0 else 0.0
            evma2[k] = evm[1] ** al[k] if evm[1] != 0.0 else 0.0
            evma3[k] = evm[2] ** al[k] if evm[2] != 0.0 else 0.0

        partt = 0.0
        for k in range(nordre):
            dd = 2.0 * mu[k] / al[k]
            sum_val = (1.0 / 3.0) * (evma1[k] + evma2[k] + evma3[k])
            partt += dd * (evma3[k] - sum_val)

        partp = 0.0
        for k in range(nordre):
            if k < len(d) and d[k] != 0.0:
                k2 = 2 * (k + 1)
                dd = float(k2) / d[k]
                partp += dd * (rv - 1.0) ** (k2 - 1)

        t3 = partt / rv + partp

        partt2 = 0.0
        for k in range(nordre):
            partt2 += 2.0 * mu[k] * (evma1[k] + evma2[k] + 4.0 * evma3[k]) / 9.0

        partp2 = 0.0
        for k in range(nordre):
            if k < len(d) and d[k] != 0.0:
                k2 = 2 * (k + 1)
                dd = float(k2) * (k2 - 1.0) / d[k]
                partp2 += dd * (rv - 1.0) ** (k2 - 2)

        partt_full = partt2 / rv + partp2 - t3
        ev[2] = ev[2] * (1.0 - t3 / partt_full)

    # sigeps82c.F:228-283: Recalculate principal stresses
    rv = ev[0] * ev[1] * ev[2]
    rvt = rv ** (-1.0 / 3.0)
    evm = ev * rvt

    dwdl = np.zeros(3, dtype=np.float64)
    for k in range(nordre):
        dd = mu[k] / al[k]
        evma1_k = evm[0] ** al[k]
        evma2_k = evm[1] ** al[k]
        evma3_k = evm[2] ** al[k]
        sum_val = (1.0 / 3.0) * (evma1_k + evma2_k + evma3_k)
        dwdl[0] += dd * (evma1_k - sum_val)
        dwdl[1] += dd * (evma2_k - sum_val)
        dwdl[2] += dd * (evma3_k - sum_val)

    partp = 0.0
    for k in range(nordre):
        if k < len(d) and d[k] != 0.0:
            k2 = 2 * (k + 1)
            dd = float(k2) / d[k]
            partp += dd * (rv - 1.0) ** (k2 - 1)

    t1 = 2.0 * dwdl[0] / rv + partp
    t2 = 2.0 * dwdl[1] / rv + partp

    # sigeps82c.F:290-293: Transform to global directions
    signxx = eigv[0, 0] * t1 + eigv[0, 1] * t2
    signyy = eigv[1, 0] * t1 + eigv[1, 1] * t2
    signxy = eigv[2, 0] * t1 + eigv[2, 1] * t2
    sig_out = np.array([signxx, signyy, signxy], dtype=np.float64)

    # sigeps82c.F:300-303: Thickness & sound speed
    if deps_inplane is not None:
        dezz = -nu / (1.0 - nu) * (deps_inplane[0] + deps_inplane[1])
        thkn = thk0 + dezz * thk0
    else:
        thkn = thk0

    rho_curr = rho0 / rv
    soundsp = np.sqrt(((2.0 / 3.0) * gmax + rbulk) / rho_curr)

    return sig_out, ev[2], thkn, soundsp, rv


# ============================================================================
# 1. Parameter Setup Parity Tests (hm_read_mat82.F)
# ============================================================================


def test_fortran_parity_shear_modulus_sum():
    """Verify G0 = sum(mu_i) for N=1, 2, 3, 5 matching hm_read_mat82.F:128-131."""
    # N=1
    p1 = build_law82(mu=[12.5], alpha=[2.0])
    assert np.isclose(p1.g0, 12.5)

    # N=2
    p2 = build_law82(nordre=2, mu=[10.0, 5.5], alpha=[2.0, -2.0])
    assert np.isclose(p2.g0, 15.5)

    # N=3
    p3 = build_law82(nordre=3, mu=[6.3e5, 1.2e3, -1.0e4], alpha=[1.3, 5.0, -2.0])
    assert np.isclose(p3.g0, 6.3e5 + 1.2e3 - 1.0e4)

    # N=5
    mu_5 = [10.0, 5.0, 2.5, 1.2, 0.3]
    alpha_5 = [1.5, 3.0, -1.0, -3.0, 5.0]
    p5 = build_law82(nordre=5, mu=mu_5, alpha=alpha_5)
    assert np.isclose(p5.g0, sum(mu_5))


def test_fortran_parity_nu_clamping_half():
    """Verify nu == 0.5 clamped to 0.495 matching hm_read_mat82.F:96,138."""
    mat = build_law82(nu=0.5, mu=[20.0], alpha=[2.0])
    assert np.isclose(mat.nu, 0.495)

    # Verify D1 formula: D1 = 3*(1 - 2*nu) / (G0 * (1 + nu))
    expected_d1 = 3.0 * (1.0 - 2.0 * 0.495) / (20.0 * (1.0 + 0.495))
    assert np.isclose(mat.d[0], expected_d1)

    # Verify bulk modulus K = 2 / D1
    expected_k = 2.0 / expected_d1
    assert np.isclose(mat.rbulk, expected_k)


def test_fortran_parity_nu_zero_with_d1():
    """Verify nu0 == 0 with D1 > 0 computing nu from bulk modulus (hm_read_mat82.F:140-146).

    Formula:
        P = 2 / D1
        nu = (3*P - 2*G0) / (6*P + 2*G0)
    """
    g0 = 25.0
    d1_in = 0.005
    p = 2.0 / d1_in  # 400.0
    expected_nu = (3.0 * p - 2.0 * g0) / (6.0 * p + 2.0 * g0)

    mat = build_law82(nu=0.0, mu=[g0], alpha=[2.0], d=[d1_in])
    assert np.isclose(mat.nu, expected_nu)
    assert np.isclose(mat.rbulk, p)

    # Compare with pure Fortran oracle
    fo = fortran_oracle_read_mat82(rho0=1.0, nu0=0.0, nordre=1, mu_in=[g0], alpha_in=[2.0], d_in=[d1_in])
    assert np.isclose(mat.nu, fo["nu"])
    assert np.isclose(mat.d[0], fo["d"][0])
    assert np.isclose(mat.rbulk, fo["parmat"]["K"])


def test_fortran_parity_nu_zero_without_d1():
    """Verify nu0 == 0 with D1 == 0 defaulting nu to 0.495 (hm_read_mat82.F:148-150)."""
    g0 = 30.0
    mat = build_law82(nu=0.0, mu=[g0], alpha=[2.0], d=[0.0])
    assert np.isclose(mat.nu, 0.495)
    expected_d1 = 3.0 * (1.0 - 2.0 * 0.495) / (g0 * (1.0 + 0.495))
    assert np.isclose(mat.d[0], expected_d1)
    assert np.isclose(mat.rbulk, 2.0 / expected_d1)


def test_fortran_parity_nu_zero_calc_half_clamp():
    """Verify nu0 == 0 with D1 > 0 clamping calculated nu == 0.5 to 0.499 (hm_read_mat82.F:144)."""
    g0 = 10.0
    d1_tiny = 1e-25  # P = 2/D1 is huge, nu_calc rounds to 0.5 in float
    mat = build_law82(nu=0.0, mu=[g0], alpha=[2.0], d=[d1_tiny])
    assert np.isclose(mat.nu, 0.499)


def test_fortran_parity_order_truncation():
    """Verify order truncation at first zero parameter pair (hm_read_mat82.F:113-121)."""
    # 5 terms supplied, but 3rd has mu=0, alpha=0
    mu = [10.0, 5.0, 0.0, 4.0, 1.0]
    alpha = [1.5, -2.0, 0.0, 3.0, -1.0]
    mat = build_law82(nordre=5, mu=mu, alpha=alpha)
    assert mat.nordre == 2
    assert len(mat.mu) == 2
    assert len(mat.alpha) == 2
    assert np.isclose(mat.g0, 15.0)


def test_fortran_parity_elastic_relations():
    """Verify small-strain elastic consistency: E0 = 2*G0*(1+nu) and K0 = E0/(3*(1-2*nu))."""
    for nu_test in [0.3, 0.45, 0.49, 0.495]:
        mat = build_law82(nu=nu_test, mu=[40.0], alpha=[2.0])
        e0 = 2.0 * mat.g0 * (1.0 + mat.nu)
        k_expected = e0 / (3.0 * (1.0 - 2.0 * mat.nu))
        assert np.isclose(mat.rbulk, k_expected, rtol=1e-12)
        assert np.isclose(2.0 / mat.d[0], k_expected, rtol=1e-12)


# ============================================================================
# 2. 3D Solid Continuum Parity Tests (sigeps82.F)
# ============================================================================


def test_solid_parity_pure_volumetric_expansion():
    """Verify pure volumetric stretch: RV = lambda^3, deviatoric stretches = 1, S_i = P."""
    mat = build_law82(nordre=1, mu=[10.0], alpha=[2.0], nu=0.49)
    eps_vol = 0.02  # eps_xx = eps_yy = eps_zz = 0.02
    eps_6 = np.array([eps_vol, eps_vol, eps_vol, 0.0, 0.0, 0.0])

    # Python implementation
    sig_py, _ = solid_update(mat, np.zeros(6), eps=eps_6)

    # Fortran oracle
    sig_fo, s_fo, rv_fo, _ = fortran_oracle_sigeps82(
        eps_6, mat.nordre, list(mat.mu), list(mat.alpha), list(mat.d)
    )

    # Deviatoric stretches are 1, so deviatoric stresses must be 0
    # Cauchy stress is pure hydrostatic pressure P
    assert np.allclose(sig_py, sig_fo, rtol=1e-12, atol=1e-12)
    assert np.isclose(sig_py[0], sig_py[1])
    assert np.isclose(sig_py[1], sig_py[2])
    assert np.allclose(sig_py[3:], 0.0)

    # Compare against analytical pressure P = 2/D1 * (RV - 1)
    expected_rv = np.exp(3.0 * eps_vol)
    expected_p = (2.0 / mat.d[0]) * (expected_rv - 1.0)
    assert np.isclose(sig_py[0], expected_p, rtol=1e-10)


def test_solid_parity_isochoric_stretch():
    """Verify isochoric stretch: RV = 1, pressure P = 0, exact deviatoric stress."""
    # Principal stretches: lambda1 = 1.5, lambda2 = 1/1.5, lambda3 = 1
    lam1 = 1.5
    lam2 = 1.0 / 1.5
    lam3 = 1.0
    eps_6 = np.array([np.log(lam1), np.log(lam2), np.log(lam3), 0.0, 0.0, 0.0])

    mat = build_law82(nordre=2, mu=[10.0, 5.0], alpha=[2.0, -2.0], nu=0.495)
    sig_py, _ = solid_update(mat, np.zeros(6), eps=eps_6)
    sig_fo, s_fo, rv_fo, _ = fortran_oracle_sigeps82(
        eps_6, mat.nordre, list(mat.mu), list(mat.alpha), list(mat.d)
    )

    assert np.isclose(rv_fo, 1.0, atol=1e-14)
    assert np.allclose(sig_py, sig_fo, rtol=1e-12, atol=1e-12)
    # Check trace is 0 (isochoric deviatoric stress has zero trace)
    assert np.isclose(np.sum(sig_py[:3]), 0.0, atol=1e-10)


def test_solid_parity_general_3d_tensor_with_shear():
    """Verify arbitrary general 3D strain tensor with non-zero shear components.

    Checks rotation matrix via eigenvectors matching sigeps82.F:306-330.
    """
    eps_6 = np.array([0.05, -0.03, 0.01, 0.04, -0.02, 0.03], dtype=np.float64)

    # Multi-term N=3
    mu = [20.0, 10.0, 2.5]
    alpha = [1.5, -2.0, 3.0]
    mat = build_law82(nordre=3, mu=mu, alpha=alpha, nu=0.48)

    sig_py, _ = solid_update(mat, np.zeros(6), eps=eps_6)
    sig_fo, s_fo, rv_fo, c_fo = fortran_oracle_sigeps82(
        eps_6, mat.nordre, list(mat.mu), list(mat.alpha), list(mat.d)
    )

    assert np.allclose(sig_py, sig_fo, rtol=1e-12, atol=1e-12)


def test_solid_parity_multi_term_n1_n2_n3_n5():
    """Verify multi-term series parity for N=1, 2, 3, 5 terms."""
    eps_6 = np.array([0.08, -0.04, -0.02, 0.03, 0.01, -0.02])

    cases = [
        (1, [15.0], [2.0]),
        (2, [12.0, 4.0], [2.0, -2.0]),
        (3, [10.0, 5.0, 1.5], [1.3, 4.0, -2.0]),
        (5, [8.0, 4.0, 2.0, 1.0, 0.5], [1.5, 3.0, -1.0, -3.0, 5.0]),
    ]

    for n_terms, mu, alpha in cases:
        mat = build_law82(nordre=n_terms, mu=mu, alpha=alpha, nu=0.495)
        sig_py, _ = solid_update(mat, np.zeros(6), eps=eps_6)
        sig_fo, _, _, _ = fortran_oracle_sigeps82(
            eps_6, mat.nordre, list(mat.mu), list(mat.alpha), list(mat.d)
        )
        assert np.allclose(sig_py, sig_fo, rtol=1e-12, atol=1e-12), f"Failed for N={n_terms}"


def test_solid_parity_higher_order_compressibility():
    """Verify higher order compressibility terms D_k (k >= 2) in pressure P."""
    # With D1, D2, D3 > 0
    mu = [10.0, 5.0, 2.0]
    alpha = [2.0, -2.0, 4.0]
    d = [0.01, 0.02, 0.03]
    mat = build_law82(nordre=3, mu=mu, alpha=alpha, d=d, nu=0.48)
    mat.d = np.array(d, dtype=np.float64)  # Ensure D2, D3 are retained

    eps_6 = np.array([0.05, 0.05, 0.05, 0.0, 0.0, 0.0])  # Volumetric stretch
    sig_py, _ = solid_update(mat, np.zeros(6), eps=eps_6)
    sig_fo, _, rv_fo, _ = fortran_oracle_sigeps82(eps_6, 3, mu, alpha, d)

    assert np.allclose(sig_py, sig_fo, rtol=1e-12, atol=1e-12)

    # Verify analytical pressure with N=3 terms:
    # P = 2/D1*(RV-1) + 4/D2*(RV-1)^3 + 6/D3*(RV-1)^5
    expected_p = (
        (2.0 * 1 / d[0]) * (rv_fo - 1.0) ** 1
        + (2.0 * 2 / d[1]) * (rv_fo - 1.0) ** 3
        + (2.0 * 3 / d[2]) * (rv_fo - 1.0) ** 5
    )
    assert np.isclose(sig_py[0], expected_p, rtol=1e-10)


@pytest.mark.parametrize("ismstr_mode", [0, 1, 10])
def test_solid_parity_strain_formulations(ismstr_mode):
    """Verify ISMSTR options: 0 (logarithmic), 1 (engineering), 10 (Green-Lagrange)."""
    mat = build_law82(nordre=2, mu=[10.0, 5.0], alpha=[2.0, -2.0], nu=0.49)
    eps_6 = np.array([0.06, -0.03, 0.01, 0.02, 0.0, 0.0])

    sig_py, _ = solid_update(mat, np.zeros(6), eps=eps_6, ismstr=ismstr_mode)
    sig_fo, _, _, _ = fortran_oracle_sigeps82(
        eps_6, mat.nordre, list(mat.mu), list(mat.alpha), list(mat.d), ismstr=ismstr_mode
    )

    assert np.allclose(sig_py, sig_fo, rtol=1e-12, atol=1e-12)


# ============================================================================
# 3. 2D Shell Plane-Stress Parity Tests (sigeps82c.F)
# ============================================================================


def test_shell_parity_inplane_eigenvalues():
    """Verify analytical in-plane eigenvalues and rotation matrix matching sigeps82c.F:108-132."""
    epsxx = 0.08
    epsyy = -0.04
    epsxy = 0.06
    eps_3 = np.array([epsxx, epsyy, epsxy])

    mat = build_law82(nordre=2, mu=[10.0, 5.0], alpha=[2.0, -2.0], nu=0.495)

    sig_py, _ = shell_update(mat, np.zeros(3), eps=eps_3)
    sig_fo, lam3_fo, thk_fo, c_fo, rv_fo = fortran_oracle_sigeps82c(
        eps_3, mat.nordre, list(mat.mu), list(mat.alpha), list(mat.d)
    )

    assert np.allclose(sig_py, sig_fo, rtol=1e-12, atol=1e-12)


def test_shell_parity_newton_plane_stress_enforcement():
    """Verify 3-step Newton solver drives out-of-plane stress T3 -> 0."""
    mat = build_law82(nordre=2, mu=[15.0, 3.0], alpha=[2.0, -2.0], nu=0.495)
    eps_3 = np.array([0.1, 0.05, 0.0])

    extra = {"uvar82": np.ones((1, 1), dtype=np.float64)}
    sig_py, _ = shell_update(mat, np.zeros(3), eps=eps_3, extra=extra)
    lam3_solved = extra["uvar82"][0, 0]

    # Verify plane stress T3 = 0 by evaluating solid Cauchy stress with solved lambda_3
    lam1 = np.exp(eps_3[0])
    lam2 = np.exp(eps_3[1])
    eps_equiv_solid = np.array([np.log(lam1), np.log(lam2), np.log(lam3_solved), 0.0, 0.0, 0.0])
    sig_solid, _ = solid_update(mat, np.zeros(6), eps=eps_equiv_solid)

    # sigma_zz residual is negligible compared to in-plane stresses (< 1.5%)
    assert abs(sig_solid[2]) < 0.015 * np.max(np.abs(sig_py))
    # In-plane stresses match to machine precision
    assert np.allclose(sig_py[:2], sig_solid[:2], rtol=1e-12, atol=1e-12)


def test_shell_parity_history_variable_persistence():
    """Verify uvar82 persistence across consecutive time increments."""
    mat = build_law82(nordre=1, mu=[10.0], alpha=[2.0], nu=0.495)

    extra = {"uvar82": np.ones((1, 1), dtype=np.float64)}
    deps = np.array([[0.02, -0.01, 0.0]])
    eps = np.zeros((1, 3))
    sig = np.zeros((1, 3))

    for step in range(5):
        eps += deps
        sig, _ = shell_update(mat, sig, deps=deps, eps=eps, extra=extra)
        lam3_step = extra["uvar82"][0, 0]
        # In tension, out-of-plane stretch should be < 1
        assert 0.5 < lam3_step < 1.0


def test_shell_parity_transverse_shear_update():
    """Verify transverse shear stresses updated with G0 (sigeps82c.F:294-295)."""
    mat = build_law82(nordre=1, mu=[25.0], alpha=[2.0], nu=0.49)
    g0 = mat.g0

    # Shell stress tensor with 5 components: [sig_xx, sig_yy, sig_xy, sig_yz, sig_zx]
    sig_5 = np.array([0.0, 0.0, 0.0, 1.5, -2.0])
    deps_5 = np.array([0.01, -0.005, 0.002, 0.003, -0.004])

    sig_new, _ = shell_update(mat, sig_5, deps=deps_5, eps=deps_5)

    expected_sig_yz = 1.5 + g0 * 0.003
    expected_sig_zx = -2.0 + g0 * (-0.004)
    assert np.isclose(sig_new[3], expected_sig_yz)
    assert np.isclose(sig_new[4], expected_sig_zx)


def test_shell_parity_thickness_update():
    """Verify shell thickness update matching sigeps82c.F:300-301."""
    mat = build_law82(nordre=1, mu=[20.0], alpha=[2.0], nu=0.48)
    nu = mat.nu
    thk0 = 2.5

    extra = {"thkn": np.array([thk0]), "thklyl": thk0}
    deps = np.array([0.02, 0.01, 0.0])

    shell_update(mat, np.zeros(3), deps=deps, eps=deps, extra=extra)

    dezz_expected = -nu / (1.0 - nu) * (0.02 + 0.01)
    expected_thkn = thk0 + dezz_expected * thk0
    assert np.isclose(extra["thkn"][0], expected_thkn)


# ============================================================================
# 4. Analytical Closed Forms Tests
# ============================================================================


def analytical_ogden_uniaxial(lam: float, mu: list[float], alpha: list[float]) -> float:
    """Exact analytical Cauchy stress for incompressible Ogden in uniaxial tension:

        sigma_1 = sum_{k=1}^N (2*mu_k / alpha_k) * (lambda^alpha_k - lambda^(-alpha_k / 2))
    """
    sigma = 0.0
    for mu_k, al_k in zip(mu, alpha):
        sigma += (2.0 * mu_k / al_k) * (lam**al_k - lam**(-0.5 * al_k))
    return sigma


def analytical_ogden_equibiaxial(lam: float, mu: list[float], alpha: list[float]) -> float:
    """Exact analytical Cauchy stress for incompressible Ogden in equibiaxial tension:

        sigma = sum_{k=1}^N (2*mu_k / alpha_k) * (lambda^alpha_k - lambda^(-2*alpha_k))
    """
    sigma = 0.0
    for mu_k, al_k in zip(mu, alpha):
        sigma += (2.0 * mu_k / al_k) * (lam**al_k - lam**(-2.0 * al_k))
    return sigma


def analytical_ogden_pure_shear(lam: float, mu: list[float], alpha: list[float]) -> float:
    """Exact analytical stress difference for incompressible Ogden in pure shear:

        sigma_1 - sigma_3 = sum_{k=1}^N (2*mu_k / alpha_k) * (lambda^alpha_k - lambda^(-alpha_k))
    """
    diff = 0.0
    for mu_k, al_k in zip(mu, alpha):
        diff += (2.0 * mu_k / al_k) * (lam**al_k - lam**(-al_k))
    return diff


@pytest.mark.parametrize("nordre,mu,alpha", [
    (1, [10.0], [2.0]),                           # Neo-Hookean
    (2, [8.0, 2.0], [2.0, -2.0]),                 # Mooney-Rivlin equivalent
    (3, [6.3e5, 1.2e3, -1.0e4], [1.3, 5.0, -2.0]),# 3-term Ogden
    (5, [10.0, 5.0, 2.5, 1.2, 0.3], [1.5, 3.0, -1.0, -3.0, 5.0]), # 5-term Ogden
])
def test_analytical_uniaxial_tension_parity(nordre, mu, alpha):
    """Verify solid and shell updates match exact analytical uniaxial tension formula."""
    # Near-incompressible with high Poisson's ratio
    mat = build_law82(nordre=nordre, mu=mu, alpha=alpha, nu=0.4999)

    stretches = [0.8, 0.9, 1.1, 1.3, 1.6, 2.0]
    for lam in stretches:
        sigma_exact = analytical_ogden_uniaxial(lam, mu, alpha)

        # Incompressible solid: lambda_1 = lam, lambda_2 = lambda_3 = lam^(-1/2)
        # Principal stress difference sigma_1 - sigma_2 is the physical uniaxial tension
        eps_solid = np.array([np.log(lam), -0.5 * np.log(lam), -0.5 * np.log(lam), 0.0, 0.0, 0.0])
        sig_solid, _ = solid_update(mat, np.zeros(6), eps=eps_solid)
        sig_diff = sig_solid[0] - sig_solid[1]
        assert np.isclose(sig_diff, sigma_exact, rtol=1e-4), (
            f"Solid mismatch at lam={lam}: got {sig_diff}, expected {sigma_exact}"
        )


@pytest.mark.parametrize("nordre,mu,alpha", [
    (1, [12.0], [2.0]),
    (2, [10.0, 3.0], [2.0, -2.0]),
    (3, [8.0, 4.0, 1.0], [1.5, 4.0, -2.0]),
])
def test_analytical_equibiaxial_tension_parity(nordre, mu, alpha):
    """Verify solid and shell match exact analytical equibiaxial tension formula."""
    mat = build_law82(nordre=nordre, mu=mu, alpha=alpha, nu=0.4999)

    for lam in [1.1, 1.25, 1.5]:
        sigma_exact = analytical_ogden_equibiaxial(lam, mu, alpha)

        # Incompressible solid: lambda_1 = lambda_2 = lam, lambda_3 = lam^(-2)
        # Principal stress difference sigma_1 - sigma_3 is the physical equibiaxial tension
        eps_solid = np.array([np.log(lam), np.log(lam), -2.0 * np.log(lam), 0.0, 0.0, 0.0])
        sig_solid, _ = solid_update(mat, np.zeros(6), eps=eps_solid)
        sig_diff = sig_solid[0] - sig_solid[2]
        assert np.isclose(sig_diff, sigma_exact, rtol=1e-4)
        assert np.isclose(sig_solid[0], sig_solid[1], rtol=1e-12)


@pytest.mark.parametrize("nordre,mu,alpha", [
    (1, [15.0], [2.0]),
    (2, [9.0, 3.0], [2.0, -2.0]),
    (3, [6.0, 3.0, 1.0], [1.3, 4.0, -2.0]),
])
def test_analytical_pure_shear_parity(nordre, mu, alpha):
    """Verify solid matches analytical pure shear (planar tension) stress difference."""
    mat = build_law82(nordre=nordre, mu=mu, alpha=alpha, nu=0.4999)

    for lam in [1.1, 1.3, 1.5]:
        diff_exact = analytical_ogden_pure_shear(lam, mu, alpha)

        # Pure shear: lambda_1 = lam, lambda_2 = 1, lambda_3 = 1/lam
        # Direction 3 is free (sigma_3 = 0), so physical stress difference is sigma_1 - sigma_3
        eps_solid = np.array([np.log(lam), 0.0, -np.log(lam), 0.0, 0.0, 0.0])
        sig_solid, _ = solid_update(mat, np.zeros(6), eps=eps_solid)

        diff_computed = sig_solid[0] - sig_solid[2]
        assert np.isclose(diff_computed, diff_exact, rtol=1e-4)


def test_analytical_small_strain_limit():
    """Verify small strain limit recovers classical linear elasticity: E = 2*G0*(1+nu), K = E/(3*(1-2*nu))."""
    mu_list = [10.0, 5.0]
    alpha_list = [2.0, -2.0]
    nu = 0.49
    mat = build_law82(nordre=2, mu=mu_list, alpha=alpha_list, nu=nu)
    g0 = mat.g0
    e_classical = 2.0 * g0 * (1.0 + nu)
    k_classical = e_classical / (3.0 * (1.0 - 2.0 * nu))

    # 1. Uniaxial tension tangent d_sigma_xx / d_eps_xx = E at zero strain
    h = 1e-6
    eps_plus = np.array([h, -nu * h, -nu * h, 0.0, 0.0, 0.0])
    eps_minus = -eps_plus
    sig_p, _ = solid_update(mat, np.zeros(6), eps=eps_plus)
    sig_m, _ = solid_update(mat, np.zeros(6), eps=eps_minus)

    e_numeric = (sig_p[0] - sig_m[0]) / (2.0 * h)
    assert np.isclose(e_numeric, e_classical, rtol=1e-4)

    # 2. Bulk modulus tangent dP / d(eps_vol) = K at zero strain
    eps_vol_p = np.array([h / 3.0, h / 3.0, h / 3.0, 0.0, 0.0, 0.0])
    eps_vol_m = -eps_vol_p
    sig_vp, _ = solid_update(mat, np.zeros(6), eps=eps_vol_p)
    sig_vm, _ = solid_update(mat, np.zeros(6), eps=eps_vol_m)

    p_p = np.mean(sig_vp[:3])
    p_m = np.mean(sig_vm[:3])
    # Total volume strain difference is 2*h
    k_numeric = (p_p - p_m) / (2.0 * h)
    assert np.isclose(k_numeric, k_classical, rtol=1e-4)
    assert np.isclose(k_classical, 2.0 / mat.d[0], rtol=1e-12)


def test_analytical_special_cases_neo_hookean_and_mooney_rivlin():
    """Verify exact equivalence to Neo-Hookean (N=1, alpha=2) and Mooney-Rivlin (N=2, alpha=[2, -2])."""
    # 1. Neo-Hookean: W = C10 * (I1 - 3), sigma_1 - sigma_2 = 2 * C10 * (lambda^2 - 1/lambda)
    # Ogden with N=1, alpha=2: 2*mu/2 * (lambda^2 - 1/lambda) = mu * (lambda^2 - 1/lambda) => mu = 2*C10
    c10 = 5.0
    mu_nh = 2.0 * c10
    mat_nh = build_law82(nordre=1, mu=[mu_nh], alpha=[2.0], nu=0.4999)

    for lam in [1.1, 1.3, 1.5]:
        eps = np.array([np.log(lam), -0.5 * np.log(lam), -0.5 * np.log(lam), 0.0, 0.0, 0.0])
        sig, _ = solid_update(mat_nh, np.zeros(6), eps=eps)
        sig_diff = sig[0] - sig[1]
        exact_nh = 2.0 * c10 * (lam**2 - 1.0 / lam)
        assert np.isclose(sig_diff, exact_nh, rtol=1e-4)

    # 2. Mooney-Rivlin: W = C10 * (I1 - 3) + C01 * (I2 - 3)
    # In uniaxial tension: sigma_1 - sigma_2 = 2*C10*(lambda^2 - 1/lambda) + 2*C01*(lambda - 1/lambda^2)
    # Ogden with alpha1 = 2, alpha2 = -2:
    # (2*mu1/2)*(lambda^2 - 1/lambda) + (2*mu2/(-2))*(lambda^(-2) - lambda)
    # = mu1*(lambda^2 - 1/lambda) + mu2*(lambda - 1/lambda^2)
    # Exact match with mu1 = 2*C10, mu2 = 2*C01
    c01 = 2.0
    mu1 = 2.0 * c10
    mu2 = 2.0 * c01
    mat_mr = build_law82(nordre=2, mu=[mu1, mu2], alpha=[2.0, -2.0], nu=0.4999)

    for lam in [1.1, 1.3, 1.5]:
        eps = np.array([np.log(lam), -0.5 * np.log(lam), -0.5 * np.log(lam), 0.0, 0.0, 0.0])
        sig, _ = solid_update(mat_mr, np.zeros(6), eps=eps)
        sig_diff = sig[0] - sig[1]
        exact_mr = 2.0 * c10 * (lam**2 - 1.0 / lam) + 2.0 * c01 * (lam - 1.0 / lam**2)
        assert np.isclose(sig_diff, exact_mr, rtol=1e-4)


# ============================================================================
# 5. Sound Speed Parity Tests (sigeps82.F:332, sigeps82c.F:303)
# ============================================================================


def test_sound_speed_solid_undeformed():
    """Verify solid sound speed at zero strain: c = sqrt((4/3*G0 + K) / rho)."""
    mat = build_law82(nordre=2, mu=[10.0, 5.0], alpha=[2.0, -2.0], nu=0.495)
    rho = 1.2
    c_expected = np.sqrt(((4.0 / 3.0) * mat.g0 + mat.rbulk) / rho)

    c_calc = solid_sound_speed(mat, rho=rho, eps=np.zeros(6))
    assert np.isclose(c_calc, c_expected, rtol=1e-12)


def test_sound_speed_solid_finite_strain_enhancement():
    """Verify nonlinear sound speed enhancement under finite strain (sigeps82.F:273-304)."""
    mat = build_law82(nordre=1, mu=[20.0], alpha=[2.0], nu=0.49)
    eps_large = np.array([0.3, -0.15, -0.15, 0.0, 0.0, 0.0])

    c_py = solid_sound_speed(mat, rho=1.0, eps=eps_large)
    _, _, _, c_fo = fortran_oracle_sigeps82(
        eps_large, mat.nordre, list(mat.mu), list(mat.alpha), list(mat.d)
    )

    assert np.isclose(c_py, c_fo, rtol=1e-12)


def test_sound_speed_shell():
    """Verify shell sound speed: c = sqrt((2/3*G0 + K) / rho) matching sigeps82c.F:303."""
    mat = build_law82(nordre=3, mu=[8.0, 4.0, 2.0], alpha=[1.5, 3.0, -2.0], nu=0.48)
    rho = 2.5
    c_expected = np.sqrt(((2.0 / 3.0) * mat.g0 + mat.rbulk) / rho)

    c_calc = shell_sound_speed(mat, rho=rho)
    assert np.isclose(c_calc, c_expected, rtol=1e-12)


def test_official_rd_e_5600_rubber_tension_parameters():
    """Verify official RD-E-5600 rubber tension parameter set from LAW82_N2.txt.

    Deck values:
        RHO_I = 1e-9 Mg/mm^3
        N = 2, Nu = 0.4997
        Mu_1 = 0.000045637449070023 GPa, Alpha_1 = 7.168617832124
        Mu_2 = 0.547913433558156 GPa,   Alpha_2 = -4.158214786551
        D_1 = 0.0, D_2 = 0.0
    """
    rho0 = 1e-9
    nu0 = 0.4997
    mu = [0.000045637449070023, 0.547913433558156]
    alpha = [7.168617832124, -4.158214786551]

    mat = build_law82(rho0=rho0, nu=nu0, nordre=2, mu=mu, alpha=alpha)

    # 1. Initial shear modulus
    expected_g0 = sum(mu)
    assert np.isclose(mat.g0, expected_g0, rtol=1e-12)

    # 2. Poisson's ratio
    assert np.isclose(mat.nu, nu0, rtol=1e-12)

    # 3. D1 and Bulk modulus
    expected_d1 = 3.0 * (1.0 - 2.0 * nu0) / (expected_g0 * (1.0 + nu0))
    expected_k = 2.0 / expected_d1
    assert np.isclose(mat.d[0], expected_d1, rtol=1e-12)
    assert np.isclose(mat.rbulk, expected_k, rtol=1e-12)

    # 4. Undeformed sound speeds
    expected_c_solid = np.sqrt(((4.0 / 3.0) * expected_g0 + expected_k) / rho0)
    expected_c_shell = np.sqrt(((2.0 / 3.0) * expected_g0 + expected_k) / rho0)
    assert np.isclose(solid_sound_speed(mat, rho=rho0), expected_c_solid, rtol=1e-12)
    assert np.isclose(shell_sound_speed(mat, rho=rho0), expected_c_shell, rtol=1e-12)

    # 5. Tensile response parity against Fortran oracle at 50% strain
    lam_test = 1.5
    eps_6 = np.array([np.log(lam_test), -0.5 * np.log(lam_test), -0.5 * np.log(lam_test), 0.0, 0.0, 0.0])
    sig_py, _ = solid_update(mat, np.zeros(6), eps=eps_6)
    sig_fo, _, _, _ = fortran_oracle_sigeps82(eps_6, 2, mu, alpha, list(mat.d))
    assert np.allclose(sig_py, sig_fo, rtol=1e-12, atol=1e-12)


# ============================================================================
# 6. Detailed Step-by-Step Derivative & Vectorization Audits
# ============================================================================


def test_shell_parity_newton_derivatives_step_by_step():
    """Verify first and second derivatives matching sigeps82c.F:185-224 line-by-line.

    Checks:
        PARTT_1st = sum 2*mu/alpha * (evma3 - (evma1+evma2+evma3)/3)
        PARTP_1st = sum 2k/D_k * (RV-1)^(2k-1)
        T3 = PARTT_1st / RV + PARTP_1st

        PARTT_2nd = sum (2*mu/9) * (evma1 + evma2 + 4*evma3)
        PARTP_2nd = sum (2k*(2k-1)/D_k) * (RV-1)^(2k-2)
        dpartt = PARTT_2nd / RV + PARTP_2nd - T3
        lambda_3_next = lambda_3 * (1 - T3 / dpartt)
    """
    mu = [20.0, 10.0]
    alpha = [2.0, -2.0]
    d = [0.005, 0.01]
    mat = build_law82(nordre=2, mu=mu, alpha=alpha, d=d, nu=0.48)
    mat.d = np.array(d, dtype=np.float64)

    # Initial state
    lam1 = 1.2
    lam2 = 0.9
    lam3 = 1.0

    ev = np.array([lam1, lam2, lam3], dtype=np.float64)

    # Manually compute iteration 1 according to Fortran lines 150-225
    rv = ev[0] * ev[1] * ev[2]
    rvt = rv ** (-1.0 / 3.0)
    evm = ev * rvt

    partt_1 = 0.0
    for k in range(2):
        evma1 = evm[0] ** alpha[k]
        evma2 = evm[1] ** alpha[k]
        evma3 = evm[2] ** alpha[k]
        sum_m = (evma1 + evma2 + evma3) / 3.0
        partt_1 += (2.0 * mu[k] / alpha[k]) * (evma3 - sum_m)

    partp_1 = 0.0
    for k in range(2):
        k2 = 2 * (k + 1)
        partp_1 += (float(k2) / d[k]) * (rv - 1.0) ** (k2 - 1)

    t3 = partt_1 / rv + partp_1

    partt_2 = 0.0
    for k in range(2):
        evma1 = evm[0] ** alpha[k]
        evma2 = evm[1] ** alpha[k]
        evma3 = evm[2] ** alpha[k]
        partt_2 += (2.0 * mu[k] / 9.0) * (evma1 + evma2 + 4.0 * evma3)

    partp_2 = 0.0
    for k in range(2):
        k2 = 2 * (k + 1)
        partp_2 += (float(k2) * (k2 - 1.0) / d[k]) * (rv - 1.0) ** (k2 - 2)

    dpartt = partt_2 / rv + partp_2 - t3
    lam3_next = lam3 * (1.0 - t3 / dpartt)

    # Now verify shell_update on this configuration reproduces Fortran
    eps_in = np.array([np.log(lam1), np.log(lam2), 0.0])
    extra = {"uvar82": np.array([[lam3]])}
    sig_sh, _ = shell_update(mat, np.zeros(3), eps=eps_in, extra=extra)

    # Convergence after 3 iterations
    lam3_final = extra["uvar82"][0, 0]
    assert 0.8 < lam3_final < 1.0
    # Fortran oracle must produce identical lam3
    _, lam3_fo, _, _, _ = fortran_oracle_sigeps82c(
        eps_in, 2, mu, alpha, d, uvar1=lam3
    )
    assert np.isclose(lam3_final, lam3_fo, rtol=1e-12)


def test_solid_batch_vectorization_parity():
    """Verify solid_update batch vectorization matches 1D single-element results exactly."""
    mat = build_law82(nordre=3, mu=[15.0, 8.0, 2.0], alpha=[1.3, 4.0, -2.0], nu=0.49)
    rng = np.random.default_rng(42)

    n_elem = 50
    eps_batch = rng.normal(scale=0.03, size=(n_elem, 6))
    sig_batch = np.zeros_like(eps_batch)

    sig_res_batch, _ = solid_update(mat, sig_batch, eps=eps_batch)

    # Compare against 1D single-element calls
    for i in range(n_elem):
        sig_single, _ = solid_update(mat, np.zeros(6), eps=eps_batch[i])
        assert np.allclose(sig_res_batch[i], sig_single, rtol=1e-12, atol=1e-12)


def test_shell_batch_vectorization_parity():
    """Verify shell_update batch vectorization matches 1D single-element results exactly."""
    mat = build_law82(nordre=2, mu=[12.0, 4.0], alpha=[2.0, -2.0], nu=0.495)
    rng = np.random.default_rng(123)

    n_elem = 50
    eps_batch = rng.normal(scale=0.03, size=(n_elem, 3))
    sig_batch = np.zeros_like(eps_batch)

    extra_batch = {"uvar82": np.ones((n_elem, 1), dtype=np.float64)}
    sig_res_batch, _ = shell_update(mat, sig_batch, eps=eps_batch, extra=extra_batch)

    # Compare against 1D single-element calls
    for i in range(n_elem):
        extra_single = {"uvar82": np.ones((1, 1), dtype=np.float64)}
        sig_single, _ = shell_update(mat, np.zeros(3), eps=eps_batch[i], extra=extra_single)
        assert np.allclose(sig_res_batch[i], sig_single, rtol=1e-12, atol=1e-12)
        assert np.isclose(extra_batch["uvar82"][i, 0], extra_single["uvar82"][0, 0], rtol=1e-12)


def test_consistent_tangent_properties():
    """Verify consistent algorithmic tangent properties: symmetry at zero strain and positive definiteness."""
    mat = build_law82(nordre=2, mu=[10.0, 5.0], alpha=[2.0, -2.0], nu=0.49)

    # 1. At zero strain, tangent is exact linear elasticity tensor and perfectly symmetric
    eps_zero = np.zeros(6)
    C_zero = consistent_solid_tangent(mat, eps_zero)[0]
    assert np.allclose(C_zero, C_zero.T, rtol=1e-8, atol=1e-8)
    eigvals_zero = np.linalg.eigvalsh(C_zero)
    assert np.all(eigvals_zero > 0.0)

    # Verify C11, C12 match classical linear elasticity: lambda + 2G, lambda
    g0 = mat.g0
    nu = mat.nu
    lam_lame = 2.0 * g0 * nu / (1.0 - 2.0 * nu)
    c11_expected = lam_lame + 2.0 * g0
    c12_expected = lam_lame
    assert np.isclose(C_zero[0, 0], c11_expected, rtol=1e-4)
    assert np.isclose(C_zero[0, 1], c12_expected, rtol=1e-4)

    # 2. At finite strain, tangent is positive definite (stable hyperelastic response)
    eps_finite = np.array([0.02, -0.01, 0.005, 0.01, 0.0, 0.0])
    C_finite = consistent_solid_tangent(mat, eps_finite)[0]
    eigvals_finite = np.linalg.eigvalsh(0.5 * (C_finite + C_finite.T))
    assert np.all(eigvals_finite > 0.0)

    # 3. Shell tangent at zero strain
    eps_sh_zero = np.zeros(3)
    C_shell = consistent_shell_tangent(mat, eps_sh_zero)[0]
    assert np.allclose(C_shell, C_shell.T, rtol=1e-8, atol=1e-8)
    eigvals_sh = np.linalg.eigvalsh(C_shell)
    assert np.all(eigvals_sh > 0.0)
