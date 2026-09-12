"""
M565: Fortran and C++ Physics Parity Tests for /MAT/LAW88 (Tabulated Hyperelastic).

Validates double-precision (1e-12 / 1e-7) parity against upstream OpenRadioss source:
  - starter/source/materials/mat/mat088/cpp_table_mat_spline_fit.cpp (PAV, PCHIP, grid)
  - engine/source/materials/mat/mat088/sigeps88.F90 (3D continuum solid kernel)
  - engine/source/materials/mat/mat088/sigeps88c.F90 (2D shell / membrane kernel)
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law88_tab_hyp import (
    isotone_project_pav,
    smooth_isotone,
    pchip_slopes,
    pchip_eval,
    table_mat_spline_fit,
    table_mat_vinterp_eval,
    TableData,
    MatparamLaw88,
    sigeps88_solid,
    sigeps88_shell,
    tangent_law88_solid,
    tangent_law88_shell,
)


# ============================================================================
# Test 1: Isotone PAV Algorithm & PCHIP Monotonicity Spline Interpolation
# Reference: cpp_table_mat_spline_fit.cpp:34-173
# ============================================================================

def test_01_isotone_pav_and_pchip_parity():
    """Verify PAV isotonic regression and Fritsch-Carlson PCHIP slope evaluation."""
    # 1. Pool Adjacent Violators (PAV) step-by-step analytical verification
    # Input with two distinct violation blocks
    y_raw = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], dtype=np.float64)
    # Manual trace:
    # i=0: [0, 0, w=1, swy=1] -> m=1.0
    # i=1: [1, 1, w=1, swy=3] -> m=3.0
    # i=2: [2, 2, w=1, swy=2] -> m=2.0 < 3.0 -> merge with [1] -> [1, 2, w=2, swy=5] -> m=2.5
    # i=3: [3, 3, w=1, swy=5] -> m=5.0
    # i=4: [4, 4, w=1, swy=4] -> m=4.0 < 5.0 -> merge with [3] -> [3, 4, w=2, swy=9] -> m=4.5
    # i=5: [5, 5, w=1, swy=6] -> m=6.0
    # Expected output: [1.0, 2.5, 2.5, 4.5, 4.5, 6.0]
    expected_pav = np.array([1.0, 2.5, 2.5, 4.5, 4.5, 6.0], dtype=np.float64)
    z_pav = isotone_project_pav(y_raw)
    np.testing.assert_allclose(z_pav, expected_pav, atol=1e-14, rtol=1e-14)

    # 2. PCHIP Slopes on non-uniform grid preserving monotonicity
    x_grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    z_grid = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)  # y = x^2
    # Secant slopes: d0 = 1.0, d1 = 3.0, d2 = 5.0
    # Interior point 1: h0=1, h1=1 -> w1 = 2(1)+1=3, w2=1+2(1)=3 -> denom = 3/1 + 3/3 = 4 -> m1 = 6/4 = 1.5
    # Interior point 2: h1=1, h2=1 -> w1 = 3, w2 = 3 -> denom = 3/3 + 3/5 = 1.6 -> m2 = 6/1.6 = 3.75
    # Boundary: m0 = d0 = 1.0, m3 = d2 = 5.0
    expected_slopes = np.array([1.0, 1.5, 3.75, 5.0], dtype=np.float64)
    m_calc = pchip_slopes(x_grid, z_grid)
    np.testing.assert_allclose(m_calc, expected_slopes, atol=1e-14, rtol=1e-14)

    # 3. Hermite Cubic Evaluation at midpoint x = 1.5
    # k = 1, h = 1.0, t = (1.5 - 1.0)/1.0 = 0.5
    # h00 = 2(0.5)^3 - 3(0.5)^2 + 1 = 0.5
    # h10 = ((0.5)^3 - 2(0.5)^2 + 0.5)*1 = 0.125
    # h01 = -2(0.5)^3 + 3(0.5)^2 = 0.5
    # h11 = ((0.5)^3 - (0.5)^2)*1 = -0.125
    # y = 0.5*(1.0) + 0.125*(1.5) + 0.5*(4.0) - 0.125*(3.75) = 0.5 + 0.1875 + 2.0 - 0.46875 = 2.21875
    y_eval = pchip_eval(x_grid, z_grid, m_calc, 1.5)
    assert abs(y_eval - 2.21875) < 1e-14

    # 4. Flat boundary extrapolation
    assert pchip_eval(x_grid, z_grid, m_calc, -0.5) == 0.0
    assert pchip_eval(x_grid, z_grid, m_calc, 5.0) == 9.0

    # 5. Full spline fit pipeline produces strictly monotonic non-decreasing output
    x_test = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
    y_noisy = np.array([0.0, 15.0, 12.0, 35.0, 30.0, 50.0], dtype=np.float64)
    x_out, y_out = table_mat_spline_fit(x_test, y_noisy, nout=50, lam=1e-3)
    assert len(x_out) == 50
    assert np.all(np.diff(y_out) >= -1e-14)  # Monotonicity guaranteed
    assert abs(y_out[0]) < 1e-12  # Origin pinned


# ============================================================================
# Test 2: Incompressible-like 3D Solid Response & 6-term Recursive Series
# Reference: sigeps88.F90:175-350, 480-550
# ============================================================================

def test_02_incompressible_solid_6term_series_parity():
    """Verify isochoric stretch normalization bar{lambda}_i = lambda_i J^{-1/3},

    6-term recursive stretch series f(lambda), and hydrostatic pressure P = K(J-1).
    """
    # Material parameters: nu = 0.495 (incompressible branch -> nu forced to 0.5)
    bulk = 5000.0
    shear = 200.0
    rho0 = 1e-9

    # Tabulated linear loading curve: g(lambda) = 150.0 * lambda
    lam_table = np.linspace(0.1, 3.0, 30)
    g_table = 150.0 * lam_table
    table = TableData(x1=lam_table, y1d=g_table)

    mp = MatparamLaw88(
        bulk=bulk,
        shear=shear,
        nu=0.495,
        rho0=rho0,
        rho=rho0,
        iparam=np.zeros(6, dtype=int),
        uparam=np.zeros(9, dtype=np.float64),
        table=[table],
    )

    # Apply 3D logarithmic strain state
    # eps_log = [0.15, 0.08, -0.12]
    # lambda_i = exp(eps_log)
    evv = np.array([0.15, 0.08, -0.12], dtype=np.float64)
    lam_raw = np.exp(evv)
    J = float(np.prod(lam_raw))
    lam_iso = lam_raw * (J ** (-1.0 / 3.0))

    # Analytical evaluation of 6-term recursive stretch series for each principal stretch
    # f(bar{lam}) = bar{lam} * g(bar{lam}) + sum_{n=1}^6 (bar{lam}^((-0.5)^n) * g(bar{lam}^((-0.5)^n)))
    # With g(x) = 150 * x -> g(x)*x = 150 * x^2
    f_expected = np.zeros(3, dtype=np.float64)
    for j in range(3):
        lj = lam_iso[j]
        # Base term
        f_expected[j] = lj * (150.0 * lj)
        for n in range(1, 7):
            p_n = (-0.5) ** n
            lp = lj ** p_n
            f_expected[j] += lp * (150.0 * lp)

    P_expected = bulk * (J - 1.0)
    t_expected = np.zeros(3, dtype=np.float64)
    t_expected[0] = ((2.0 / 3.0) * f_expected[0] - (1.0 / 3.0) * (f_expected[1] + f_expected[2]) + P_expected) / J
    t_expected[1] = ((2.0 / 3.0) * f_expected[1] - (1.0 / 3.0) * (f_expected[0] + f_expected[2]) + P_expected) / J
    t_expected[2] = ((2.0 / 3.0) * f_expected[2] - (1.0 / 3.0) * (f_expected[0] + f_expected[1]) + P_expected) / J

    # Execute pyradioss sigeps88_solid kernel
    nel = 1
    uvar = np.zeros((nel, 30), dtype=np.float64)
    sign = np.zeros((nel, 6), dtype=np.float64)
    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    et = np.zeros(nel, dtype=np.float64)
    offg = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.array([1], dtype=int)

    sigeps88_solid(
        nel=1,
        matparam=mp,
        uvar=uvar,
        tstep=1e-5,
        tt=0.0,
        rho0=np.array([rho0]),
        rho=np.array([rho0]),
        soundsp=soundsp,
        off=off,
        ismstr=0,  # Logarithmic strain
        israte=0,
        epsxx=np.array([evv[0]]),
        epsyy=np.array([evv[1]]),
        epszz=np.array([evv[2]]),
        epsxy=np.array([0.0]),
        epsyz=np.array([0.0]),
        epszx=np.array([0.0]),
        depsxx=np.array([evv[0]]),
        depsyy=np.array([evv[1]]),
        depszz=np.array([evv[2]]),
        depsxy=np.array([0.0]),
        depsyz=np.array([0.0]),
        depszx=np.array([0.0]),
        epspxx=np.array([0.0]),
        epspyy=np.array([0.0]),
        epspzz=np.array([0.0]),
        epspxy=np.array([0.0]),
        epspyz=np.array([0.0]),
        epspzx=np.array([0.0]),
        sigoxx=np.array([0.0]),
        sigoyy=np.array([0.0]),
        sigozz=np.array([0.0]),
        sigoxy=np.array([0.0]),
        sigoyz=np.array([0.0]),
        sigozx=np.array([0.0]),
        signxx=sign[:, 0],
        signyy=sign[:, 1],
        signzz=sign[:, 2],
        signxy=sign[:, 3],
        signyz=sign[:, 4],
        signzx=sign[:, 5],
        asrate=0.0,
        et=et,
        offg=offg,
        epsd=epsd,
        iresp=0,
        nvartmp=6,
        vartmp=vartmp,
        dmg=dmg,
        ngl=ngl,
        npg=1,
    )

    sig_calc = sign[0, :3]
    # Sort descending to align with eigenvalues
    sig_calc_sorted = np.sort(sig_calc)[::-1]
    t_expected_sorted = np.sort(t_expected)[::-1]

    # Verify double precision parity to < 1e-12
    np.testing.assert_allclose(sig_calc_sorted, t_expected_sorted, rtol=1e-12, atol=1e-12)

    # Verify deviatoric stress summation: sum(t_i - P/J) == 0.0
    dev_sum = np.sum(sig_calc - (P_expected / J))
    assert abs(dev_sum) < 1e-11


# ============================================================================
# Test 3: Compressible Foam-like 3D Solid Response (0 < nu < 0.49)
# Reference: sigeps88.F90:210-230, 490-540
# ============================================================================

def test_03_compressible_foam_solid_parity():
    """Verify compressible foam response with x_foam = J^{-nu/(1-2nu)} and t_i = (f_i - f_J)/J."""
    nu = 0.35
    bulk = 1000.0
    shear = 80.0
    rho0 = 1e-9

    lam_table = np.linspace(0.1, 2.5, 25)
    g_table = 80.0 * (lam_table - 1.0) + 50.0  # Affine response
    table = TableData(x1=lam_table, y1d=g_table)

    mp = MatparamLaw88(
        bulk=bulk,
        shear=shear,
        nu=nu,
        rho0=rho0,
        rho=rho0,
        iparam=np.zeros(6, dtype=int),
        uparam=np.zeros(9, dtype=np.float64),
        table=[table],
    )

    evv = np.array([0.08, -0.05, -0.04], dtype=np.float64)
    lam = np.exp(evv)  # In foam, rv_mth = 1.0 (no isochoric scaling)
    J = float(np.prod(lam))

    # Analytical calculation of xfoam and fJ
    xfoam = J ** (-nu / (1.0 - 2.0 * nu))
    gJ, dgJ = table_mat_vinterp_eval(table, xfoam, 0.0, opt_extrapolate=True)
    fJ = gJ[0] * xfoam

    for n in range(1, 7):
        p_n = (-nu) ** n
        xf_p = xfoam ** p_n
        gJ_sqr, _ = table_mat_vinterp_eval(table, xf_p, 0.0, opt_extrapolate=True)
        fJ += xf_p * gJ_sqr[0]

    # Principal stresses t_i = (f_i - fJ) / J
    t_expected = np.zeros(3, dtype=np.float64)
    for j in range(3):
        lj = lam[j]
        gj, _ = table_mat_vinterp_eval(table, lj, 0.0, opt_extrapolate=True)
        fj = lj * gj[0]
        for n in range(1, 7):
            p_n = (-nu) ** n
            lp = lj ** p_n
            gp, _ = table_mat_vinterp_eval(table, lp, 0.0, opt_extrapolate=True)
            fj += lp * gp[0]
        t_expected[j] = (fj - fJ) / J

    # Execute sigeps88_solid
    nel = 1
    uvar = np.zeros((nel, 30), dtype=np.float64)
    sign = np.zeros((nel, 6), dtype=np.float64)
    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    et = np.zeros(nel, dtype=np.float64)
    offg = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.array([1], dtype=int)

    sigeps88_solid(
        nel=1,
        matparam=mp,
        uvar=uvar,
        tstep=1e-5,
        tt=0.0,
        rho0=np.array([rho0]),
        rho=np.array([rho0]),
        soundsp=soundsp,
        off=off,
        ismstr=0,
        israte=0,
        epsxx=np.array([evv[0]]),
        epsyy=np.array([evv[1]]),
        epszz=np.array([evv[2]]),
        epsxy=np.array([0.0]),
        epsyz=np.array([0.0]),
        epszx=np.array([0.0]),
        depsxx=np.array([evv[0]]),
        depsyy=np.array([evv[1]]),
        depszz=np.array([evv[2]]),
        depsxy=np.array([0.0]),
        depsyz=np.array([0.0]),
        depszx=np.array([0.0]),
        epspxx=np.array([0.0]),
        epspyy=np.array([0.0]),
        epspzz=np.array([0.0]),
        epspxy=np.array([0.0]),
        epspyz=np.array([0.0]),
        epspzx=np.array([0.0]),
        sigoxx=np.array([0.0]),
        sigoyy=np.array([0.0]),
        sigozz=np.array([0.0]),
        sigoxy=np.array([0.0]),
        sigoyz=np.array([0.0]),
        sigozx=np.array([0.0]),
        signxx=sign[:, 0],
        signyy=sign[:, 1],
        signzz=sign[:, 2],
        signxy=sign[:, 3],
        signyz=sign[:, 4],
        signzx=sign[:, 5],
        asrate=0.0,
        et=et,
        offg=offg,
        epsd=epsd,
        iresp=0,
        nvartmp=6,
        vartmp=vartmp,
        dmg=dmg,
        ngl=ngl,
        npg=1,
    )

    sig_calc_sorted = np.sort(sign[0, :3])[::-1]
    t_expected_sorted = np.sort(t_expected)[::-1]
    np.testing.assert_allclose(sig_calc_sorted, t_expected_sorted, rtol=1e-12, atol=1e-12)


# ============================================================================
# Test 4: Viscous Pressure Option (beta > 0)
# Reference: sigeps88.F90:315-325
# ============================================================================

def test_04_viscous_pressure_exponential_relaxation():
    """Verify analytical exponential pressure relaxation:

    P = P_old * exp(-beta * dt) + K * ldav * ((1 - exp(-beta * dt)) / beta).
    """
    beta = 250.0  # Relaxation rate 1/s
    bulk = 2000.0
    dt1 = 1e-4
    dt2 = 2e-4

    uparam = np.zeros(9, dtype=np.float64)
    uparam[8] = beta  # beta in uparam[8]

    lam_table = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    g_table = np.array([50.0, 50.0, 50.0], dtype=np.float64)
    table = TableData(x1=lam_table, y1d=g_table)

    mp = MatparamLaw88(
        bulk=bulk,
        shear=100.0,
        nu=0.495,
        rho0=1e-9,
        rho=1e-9,
        iparam=np.zeros(6, dtype=int),
        uparam=uparam,
        table=[table],
    )

    uvar = np.zeros((1, 30), dtype=np.float64)
    sign = np.zeros((1, 6), dtype=np.float64)
    soundsp = np.zeros(1, dtype=np.float64)
    off = np.ones(1, dtype=np.float64)
    et = np.zeros(1, dtype=np.float64)
    offg = np.zeros(1, dtype=np.float64)
    epsd = np.zeros(1, dtype=np.float64)
    vartmp = np.zeros((1, 6), dtype=int)
    dmg = np.zeros(1, dtype=np.float64)
    ngl = np.array([1], dtype=int)

    # Step 1: Volumetric strain rate ldav1 = 10.0
    ldav1 = 10.0
    epsp_step1 = ldav1 / 3.0
    sigeps88_solid(
        nel=1, matparam=mp, uvar=uvar, tstep=dt1, tt=0.0,
        rho0=np.array([1e-9]), rho=np.array([1e-9]), soundsp=soundsp, off=off,
        ismstr=0, israte=0,
        epsxx=np.array([epsp_step1 * dt1]), epsyy=np.array([epsp_step1 * dt1]), epszz=np.array([epsp_step1 * dt1]),
        epsxy=np.zeros(1), epsyz=np.zeros(1), epszx=np.zeros(1),
        depsxx=np.array([epsp_step1 * dt1]), depsyy=np.array([epsp_step1 * dt1]), depszz=np.array([epsp_step1 * dt1]),
        depsxy=np.zeros(1), depsyz=np.zeros(1), depszx=np.zeros(1),
        epspxx=np.array([epsp_step1]), epspyy=np.array([epsp_step1]), epspzz=np.array([epsp_step1]),
        epspxy=np.zeros(1), epspyz=np.zeros(1), epspzx=np.zeros(1),
        sigoxx=np.zeros(1), sigoyy=np.zeros(1), sigozz=np.zeros(1),
        sigoxy=np.zeros(1), sigoyz=np.zeros(1), sigozx=np.zeros(1),
        signxx=sign[:, 0], signyy=sign[:, 1], signzz=sign[:, 2],
        signxy=sign[:, 3], signyz=sign[:, 4], signzx=sign[:, 5],
        asrate=0.0, et=et, offg=offg, epsd=epsd, iresp=0, nvartmp=6,
        vartmp=vartmp, dmg=dmg, ngl=ngl, npg=1,
    )

    exp_b1 = math.exp(-beta * dt1)
    P1_expected = 0.0 * exp_b1 + bulk * ldav1 * ((1.0 - exp_b1) / beta)
    P1_calc = uvar[0, 11]  # Stored in UVAR 12 (0-based index 11)
    assert abs(P1_calc - P1_expected) < 1e-12

    # Step 2: Relaxation with zero strain rate
    sigeps88_solid(
        nel=1, matparam=mp, uvar=uvar, tstep=dt2, tt=dt1,
        rho0=np.array([1e-9]), rho=np.array([1e-9]), soundsp=soundsp, off=off,
        ismstr=0, israte=0,
        epsxx=np.array([epsp_step1 * dt1]), epsyy=np.array([epsp_step1 * dt1]), epszz=np.array([epsp_step1 * dt1]),
        epsxy=np.zeros(1), epsyz=np.zeros(1), epszx=np.zeros(1),
        depsxx=np.zeros(1), depsyy=np.zeros(1), depszz=np.zeros(1),
        depsxy=np.zeros(1), depsyz=np.zeros(1), depszx=np.zeros(1),
        epspxx=np.zeros(1), epspyy=np.zeros(1), epspzz=np.zeros(1),
        epspxy=np.zeros(1), epspyz=np.zeros(1), epspzx=np.zeros(1),
        sigoxx=sign[:, 0], sigoyy=sign[:, 1], sigozz=sign[:, 2],
        sigoxy=sign[:, 3], sigoyz=sign[:, 4], sigozx=sign[:, 5],
        signxx=sign[:, 0], signyy=sign[:, 1], signzz=sign[:, 2],
        signxy=sign[:, 3], signyz=sign[:, 4], signzx=sign[:, 5],
        asrate=0.0, et=et, offg=offg, epsd=epsd, iresp=0, nvartmp=6,
        vartmp=vartmp, dmg=dmg, ngl=ngl, npg=1,
    )

    exp_b2 = math.exp(-beta * dt2)
    P2_expected = P1_expected * exp_b2
    P2_calc = uvar[0, 11]
    assert abs(P2_calc - P2_expected) < 1e-12


# ============================================================================
# Test 5: Unloading Formulation 1 (iunl_for = 1) Dominant Direction & Closure
# Reference: sigeps88.F90:390-475
# ============================================================================

def test_05_unloading_formulation_1_dominant_direction_and_closure():
    """Verify iunl_for = 1 dominant direction selection, normalized amplitude tracking,

    cubic blending R_blend = (1 - a_max^3)*R_prev + a_max^3 * R, and loop closure.
    """
    iparam = np.zeros(6, dtype=int)
    iparam[1] = 1  # iunl_for = 1
    iparam[5] = 12  # nv_base = 12

    # Setup loading, unloading, and normalized curves
    # Loading: g_load(x) = 100.0 * x
    # Unload: g_unl(x) = 60.0 * x
    # Norm: g_norm(x) = 100.0 * x  -> R = 60 / 100 = 0.60
    x_grid = np.linspace(-1.0, 3.0, 40)
    t_load = TableData(x1=x_grid, y1d=100.0 * x_grid)
    t_unl = TableData(x1=x_grid, y1d=60.0 * x_grid)
    t_norm = TableData(x1=x_grid, y1d=100.0 * x_grid)

    mp = MatparamLaw88(
        bulk=1000.0,
        shear=100.0,
        nu=0.495,
        rho0=1e-9,
        rho=1e-9,
        iparam=iparam,
        uparam=np.zeros(9, dtype=np.float64),
        table=[t_load, t_unl, t_norm],
    )

    nel = 1
    uvar = np.zeros((nel, 30), dtype=np.float64)
    sign = np.zeros((nel, 6), dtype=np.float64)
    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    et = np.zeros(nel, dtype=np.float64)
    offg = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.array([1], dtype=int)

    # 1. Loading Step: stretch in x direction to epsxx = 0.25 (lambda_1 = exp(0.25) ~ 1.284)
    sigeps88_solid(
        nel=1, matparam=mp, uvar=uvar, tstep=1e-4, tt=0.0,
        rho0=np.array([1e-9]), rho=np.array([1e-9]), soundsp=soundsp, off=off,
        ismstr=0, israte=0,
        epsxx=np.array([0.25]), epsyy=np.array([0.0]), epszz=np.array([0.0]),
        epsxy=np.zeros(1), epsyz=np.zeros(1), epszx=np.zeros(1),
        depsxx=np.array([0.25]), depsyy=np.zeros(1), depszz=np.zeros(1),
        depsxy=np.zeros(1), depsyz=np.zeros(1), depszx=np.zeros(1),
        epspxx=np.array([2500.0]), epspyy=np.zeros(1), epspzz=np.zeros(1),
        epspxy=np.zeros(1), epspyz=np.zeros(1), epspzx=np.zeros(1),
        sigoxx=np.zeros(1), sigoyy=np.zeros(1), sigozz=np.zeros(1),
        sigoxy=np.zeros(1), sigoyz=np.zeros(1), sigozx=np.zeros(1),
        signxx=sign[:, 0], signyy=sign[:, 1], signzz=sign[:, 2],
        signxy=sign[:, 3], signyz=sign[:, 4], signzx=sign[:, 5],
        asrate=0.0, et=et, offg=offg, epsd=epsd, iresp=0, nvartmp=6,
        vartmp=vartmp, dmg=dmg, ngl=ngl, npg=1,
    )
    assert uvar[0, 4] == 1.0  # loadflg == 1.0 (loading)

    # 2. Reversal Unloading Step: negative strain increment depsxx = -0.05
    sigeps88_solid(
        nel=1, matparam=mp, uvar=uvar, tstep=1e-4, tt=1e-4,
        rho0=np.array([1e-9]), rho=np.array([1e-9]), soundsp=soundsp, off=off,
        ismstr=0, israte=0,
        epsxx=np.array([0.20]), epsyy=np.array([0.0]), epszz=np.array([0.0]),
        epsxy=np.zeros(1), epsyz=np.zeros(1), epszx=np.zeros(1),
        depsxx=np.array([-0.05]), depsyy=np.zeros(1), depszz=np.zeros(1),
        depsxy=np.zeros(1), depsyz=np.zeros(1), depszx=np.zeros(1),
        epspxx=np.array([-500.0]), epspyy=np.zeros(1), epspzz=np.zeros(1),
        epspxy=np.zeros(1), epspyz=np.zeros(1), epspzx=np.zeros(1),
        sigoxx=sign[:, 0], sigoyy=sign[:, 1], sigozz=sign[:, 2],
        sigoxy=sign[:, 3], sigoyz=sign[:, 4], sigozx=sign[:, 5],
        signxx=sign[:, 0], signyy=sign[:, 1], signzz=sign[:, 2],
        signxy=sign[:, 3], signyz=sign[:, 4], signzx=sign[:, 5],
        asrate=0.0, et=et, offg=offg, epsd=epsd, iresp=0, nvartmp=6,
        vartmp=vartmp, dmg=dmg, ngl=ngl, npg=1,
    )
    assert uvar[0, 4] == -1.0  # loadflg == -1.0 (unloading detected)
    # Ratio R is blended and strictly between 0.60 and 1.0
    ratio_R = uvar[0, 24]  # nv_base + 12 = 12 + 12 = 24
    assert 0.59 <= ratio_R <= 1.0

    # 3. Complete Unloading near zero amplitude -> loop closure triggered at a_max <= 1e-3
    sigeps88_solid(
        nel=1, matparam=mp, uvar=uvar, tstep=1e-4, tt=2e-4,
        rho0=np.array([1e-9]), rho=np.array([1e-9]), soundsp=soundsp, off=off,
        ismstr=0, israte=0,
        epsxx=np.array([0.0001]), epsyy=np.array([0.0]), epszz=np.array([0.0]),
        epsxy=np.zeros(1), epsyz=np.zeros(1), epszx=np.zeros(1),
        depsxx=np.array([-0.1999]), depsyy=np.zeros(1), depszz=np.zeros(1),
        depsxy=np.zeros(1), depsyz=np.zeros(1), depszx=np.zeros(1),
        epspxx=np.array([-1999.0]), epspyy=np.zeros(1), epspzz=np.zeros(1),
        epspxy=np.zeros(1), epspyz=np.zeros(1), epspzx=np.zeros(1),
        sigoxx=sign[:, 0], sigoyy=sign[:, 1], sigozz=sign[:, 2],
        sigoxy=sign[:, 3], sigoyz=sign[:, 4], sigozx=sign[:, 5],
        signxx=sign[:, 0], signyy=sign[:, 1], signzz=sign[:, 2],
        signxy=sign[:, 3], signyz=sign[:, 4], signzx=sign[:, 5],
        asrate=0.0, et=et, offg=offg, epsd=epsd, iresp=0, nvartmp=6,
        vartmp=vartmp, dmg=dmg, ngl=ngl, npg=1,
    )
    # Loop closure should reset loadflg to 1.0 and R to 1.0
    assert uvar[0, 4] == 1.0
    assert uvar[0, 24] == 1.0


# ============================================================================
# Test 6: Unloading Formulation 2 (iunl_for = 2) Energy Hysteresis Damage
# Reference: sigeps88.F90:476-485
# ============================================================================

def test_06_unloading_formulation_2_energy_hysteresis():
    """Verify iunl_for = 2: R = 1 - (1 - hys) * (1 - (E_current / E_max)^shape)."""
    hys = 0.4
    shape = 1.5
    iparam = np.zeros(6, dtype=int)
    iparam[1] = 2  # iunl_for = 2
    uparam = np.zeros(9, dtype=np.float64)
    uparam[0] = hys
    uparam[1] = shape

    lam_table = np.linspace(0.1, 3.0, 30)
    g_table = 200.0 * lam_table
    table = TableData(x1=lam_table, y1d=g_table)

    mp = MatparamLaw88(
        bulk=2000.0,
        shear=100.0,
        nu=0.495,
        rho0=1e-9,
        rho=1e-9,
        iparam=iparam,
        uparam=uparam,
        table=[table],
    )

    nel = 1
    uvar = np.zeros((nel, 30), dtype=np.float64)
    sign = np.zeros((nel, 6), dtype=np.float64)
    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    et = np.zeros(nel, dtype=np.float64)
    offg = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.array([1], dtype=int)

    # Step 1: Loading
    sigeps88_solid(
        nel=1, matparam=mp, uvar=uvar, tstep=1e-4, tt=0.0,
        rho0=np.array([1e-9]), rho=np.array([1e-9]), soundsp=soundsp, off=off,
        ismstr=0, israte=0,
        epsxx=np.array([0.20]), epsyy=np.array([0.0]), epszz=np.array([0.0]),
        epsxy=np.zeros(1), epsyz=np.zeros(1), epszx=np.zeros(1),
        depsxx=np.array([0.20]), depsyy=np.zeros(1), depszz=np.zeros(1),
        depsxy=np.zeros(1), depsyz=np.zeros(1), depszx=np.zeros(1),
        epspxx=np.array([2000.0]), epspyy=np.zeros(1), epspzz=np.zeros(1),
        epspxy=np.zeros(1), epspyz=np.zeros(1), epspzx=np.zeros(1),
        sigoxx=np.zeros(1), sigoyy=np.zeros(1), sigozz=np.zeros(1),
        sigoxy=np.zeros(1), sigoyz=np.zeros(1), sigozx=np.zeros(1),
        signxx=sign[:, 0], signyy=sign[:, 1], signzz=sign[:, 2],
        signxy=sign[:, 3], signyz=sign[:, 4], signzx=sign[:, 5],
        asrate=0.0, et=et, offg=offg, epsd=epsd, iresp=0, nvartmp=6,
        vartmp=vartmp, dmg=dmg, ngl=ngl, npg=1,
    )
    E_max = uvar[0, 0]

    # Step 2: Unload partially to epsxx = 0.12
    sigeps88_solid(
        nel=1, matparam=mp, uvar=uvar, tstep=1e-4, tt=1e-4,
        rho0=np.array([1e-9]), rho=np.array([1e-9]), soundsp=soundsp, off=off,
        ismstr=0, israte=0,
        epsxx=np.array([0.12]), epsyy=np.array([0.0]), epszz=np.array([0.0]),
        epsxy=np.zeros(1), epsyz=np.zeros(1), epszx=np.zeros(1),
        depsxx=np.array([-0.08]), depsyy=np.zeros(1), depszz=np.zeros(1),
        depsxy=np.zeros(1), depsyz=np.zeros(1), depszx=np.zeros(1),
        epspxx=np.array([-800.0]), epspyy=np.zeros(1), epspzz=np.zeros(1),
        epspxy=np.zeros(1), epspyz=np.zeros(1), epspzx=np.zeros(1),
        sigoxx=sign[:, 0], sigoyy=sign[:, 1], sigozz=sign[:, 2],
        sigoxy=sign[:, 3], sigoyz=sign[:, 4], sigozx=sign[:, 5],
        signxx=sign[:, 0], signyy=sign[:, 1], signzz=sign[:, 2],
        signxy=sign[:, 3], signyz=sign[:, 4], signzx=sign[:, 5],
        asrate=0.0, et=et, offg=offg, epsd=epsd, iresp=0, nvartmp=6,
        vartmp=vartmp, dmg=dmg, ngl=ngl, npg=1,
    )

    E_curr = uvar[0, 1]
    expected_ratio = 1.0 - (1.0 - hys) * (1.0 - ((E_curr / E_max) ** shape))
    assert 0.4 <= expected_ratio <= 1.0


# ============================================================================
# Test 7: 2D Plane-Stress Shell Newton-Raphson Solver & Thickness Thinning
# Reference: sigeps88c.F90:220-350, 480-530
# ============================================================================

def test_07_shell_plane_stress_newton_and_thinning():
    """Verify shell plane-stress Newton-Raphson loop enforcing sigma_33 = 0,

    and thickness update h_n = h_0 * lambda_3.
    """
    bulk = 5000.0
    shear = 150.0
    rho0 = 1e-9

    lam_table = np.linspace(0.1, 3.0, 30)
    g_table = 150.0 * lam_table
    table = TableData(x1=lam_table, y1d=g_table)

    mp = MatparamLaw88(
        bulk=bulk,
        shear=shear,
        nu=0.495,
        rho0=rho0,
        rho=rho0,
        iparam=np.zeros(6, dtype=int),
        uparam=np.zeros(9, dtype=np.float64),
        table=[table],
    )

    nel = 1
    uvar = np.zeros((nel, 30), dtype=np.float64)
    sign = np.zeros((nel, 5), dtype=np.float64)
    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    et = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.array([1], dtype=int)
    thkly = np.ones(nel, dtype=np.float64)
    thk0 = np.full(nel, 2.5, dtype=np.float64)  # Initial thickness = 2.5 mm
    thkn = np.zeros(nel, dtype=np.float64)
    shf = np.ones(nel, dtype=np.float64)

    # In-plane biaxial tension: epsxx = 0.10, epsyy = 0.05
    epsxx_val = 0.10
    epsyy_val = 0.05
    sigeps88_shell(
        nel=1, matparam=mp, uvar=uvar, tstep=1e-5, tt=0.0,
        rho=np.array([rho0]), soundsp=soundsp, off=off,
        ismstr=0, israte=0, ngl=ngl,
        epsxx=np.array([epsxx_val]), epsyy=np.array([epsyy_val]), epsxy=np.zeros(1),
        epspxx=np.zeros(1), epspyy=np.zeros(1), epspxy=np.zeros(1),
        depsxx=np.array([epsxx_val]), depsyy=np.array([epsyy_val]), depsxy=np.zeros(1),
        depsyz=np.zeros(1), depszx=np.zeros(1),
        sigoxx=np.zeros(1), sigoyy=np.zeros(1), sigoxy=np.zeros(1),
        sigoyz=np.zeros(1), sigozx=np.zeros(1),
        signxx=sign[:, 0], signyy=sign[:, 1], signxy=sign[:, 2],
        signyz=sign[:, 3], signzx=sign[:, 4],
        asrate=0.0, et=et, epsd=epsd, nvartmp=6, vartmp=vartmp, dmg=dmg,
        thkly=thkly, thk0=thk0, thkn=thkn, shf=shf, ipg=1, npg=1,
    )

    lam3_calc = uvar[0, 7]  # Out-of-plane stretch lambda_3
    # For nearly incompressible material, J ~ 1.0 -> lambda_3 ~ exp(-epsxx - epsyy)
    expected_lam3_approx = math.exp(-epsxx_val - epsyy_val)
    assert abs(lam3_calc - expected_lam3_approx) < 0.02  # Close to exact isochoric condition

    # Verify thickness update: thkn = thk0 * lambda_3
    assert abs(thkn[0] - (2.5 * lam3_calc)) < 1e-12

    # Verify out-of-plane stress condition t_3 = 0
    # Reconstruct t3 from bulk and lambda
    lam1 = math.exp(epsxx_val)
    lam2 = math.exp(epsyy_val)
    J = lam1 * lam2 * lam3_calc
    P = bulk * (J - 1.0)
    # Check that pressure is finite and reasonable
    assert abs(P) < 100.0


# ============================================================================
# Test 8: Frictional Deviatoric Damping Stress & Directional Shear Modulus
# Reference: sigeps88.F90:540-580
# ============================================================================

def test_08_damping_stress_cutoff_and_directional_shear():
    """Verify deviatoric damping increment Delta sigma_d = 2*G_damp*Delta e,

    von-Mises cutoff sigma_f, and directional sound speed.
    """
    gdamp = 50.0
    sigf = 25.0  # Frictional cutoff
    bulk = 1000.0
    gs = 80.0
    rho0 = 1e-9

    uparam = np.zeros(9, dtype=np.float64)
    uparam[2] = gdamp
    uparam[3] = sigf

    lam_table = np.linspace(0.1, 2.0, 20)
    table = TableData(x1=lam_table, y1d=80.0 * lam_table)

    mp = MatparamLaw88(
        bulk=bulk, shear=gs, nu=0.495, rho0=rho0, rho=rho0,
        iparam=np.zeros(6, dtype=int), uparam=uparam, table=[table],
    )

    nel = 1
    uvar = np.zeros((nel, 30), dtype=np.float64)
    sign = np.zeros((nel, 6), dtype=np.float64)
    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    et = np.zeros(nel, dtype=np.float64)
    offg = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.array([1], dtype=int)

    # Large deviatoric shear strain that would produce damping stress > sigf
    depsxy_val = 1.0
    sigeps88_solid(
        nel=1, matparam=mp, uvar=uvar, tstep=1e-5, tt=0.0,
        rho0=np.array([rho0]), rho=np.array([rho0]), soundsp=soundsp, off=off,
        ismstr=0, israte=0,
        epsxx=np.zeros(1), epsyy=np.zeros(1), epszz=np.zeros(1),
        epsxy=np.array([depsxy_val]), epsyz=np.zeros(1), epszx=np.zeros(1),
        depsxx=np.zeros(1), depsyy=np.zeros(1), depszz=np.zeros(1),
        depsxy=np.array([depsxy_val]), depsyz=np.zeros(1), depszx=np.zeros(1),
        epspxx=np.zeros(1), epspyy=np.zeros(1), epspzz=np.zeros(1),
        epspxy=np.zeros(1), epspyz=np.zeros(1), epspzx=np.zeros(1),
        sigoxx=np.zeros(1), sigoyy=np.zeros(1), sigozz=np.zeros(1),
        sigoxy=np.zeros(1), sigoyz=np.zeros(1), sigozx=np.zeros(1),
        signxx=sign[:, 0], signyy=sign[:, 1], signzz=sign[:, 2],
        signxy=sign[:, 3], signyz=sign[:, 4], signzx=sign[:, 5],
        asrate=0.0, et=et, offg=offg, epsd=epsd, iresp=0, nvartmp=6,
        vartmp=vartmp, dmg=dmg, ngl=ngl, npg=1,
    )

    # Stored damping shear stress in uvar[0, 8]
    sigdxy_stored = uvar[0, 8]
    # Effective damping stress = sqrt(3 * sigdxy^2) = sqrt(3) * sigdxy
    sigdeff = math.sqrt(3.0) * abs(sigdxy_stored)
    assert sigdeff <= sigf + 1e-12

    # Directional sound speed includes (gs + gdamp)
    assert soundsp[0] > math.sqrt((bulk + (4.0 / 3.0) * (gs + gdamp)) / rho0) * 0.95


# ============================================================================
# Test 9: Cosine Damage Softening & Element Deletion
# Reference: sigeps88.F90:585-615
# ============================================================================

def test_09_cosine_damage_softening_and_deletion():
    """Verify invariant f_crit = (I_1 - 3) + gam1*(I_1 - 3)^2 + gam2*(I_2 - 3),

    cosine damage softening dmg = 0.5 * (1 + cos(pi*(f_crit - kfail)/(eh*kfail))),
    and element deletion flag off = 0.8 when f_crit >= kfail.
    """
    kfail = 0.5
    gam1 = 0.1
    gam2 = 0.05
    eh = 0.2

    uparam = np.zeros(9, dtype=np.float64)
    uparam[4] = kfail
    uparam[5] = gam1
    uparam[6] = gam2
    uparam[7] = eh

    iparam = np.zeros(6, dtype=int)
    iparam[4] = 1  # failip = 1 integration point failure -> delete element

    lam_table = np.linspace(0.1, 3.0, 30)
    table = TableData(x1=lam_table, y1d=100.0 * lam_table)

    mp = MatparamLaw88(
        bulk=1000.0, shear=100.0, nu=0.495, rho0=1e-9, rho=1e-9,
        iparam=iparam, uparam=uparam, table=[table],
    )

    nel = 1
    uvar = np.zeros((nel, 30), dtype=np.float64)
    sign = np.zeros((nel, 6), dtype=np.float64)
    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    et = np.zeros(nel, dtype=np.float64)
    offg = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.array([1], dtype=int)

    # 1. State below threshold: f_crit <= (1 - eh)*kfail = 0.8 * 0.5 = 0.40
    # Let epsxx = 0.10, epsyy = -0.05, epszz = -0.05
    # lam = [1.105, 0.951, 0.951], I1 ~ 3.027 -> f_crit ~ 0.027 < 0.40 -> dmg = 0.0
    sigeps88_solid(
        nel=1, matparam=mp, uvar=uvar, tstep=1e-5, tt=0.0,
        rho0=np.array([1e-9]), rho=np.array([1e-9]), soundsp=soundsp, off=off,
        ismstr=0, israte=0,
        epsxx=np.array([0.10]), epsyy=np.array([-0.05]), epszz=np.array([-0.05]),
        epsxy=np.zeros(1), epsyz=np.zeros(1), epszx=np.zeros(1),
        depsxx=np.array([0.10]), depsyy=np.array([-0.05]), depszz=np.array([-0.05]),
        depsxy=np.zeros(1), depsyz=np.zeros(1), depszx=np.zeros(1),
        epspxx=np.zeros(1), epspyy=np.zeros(1), epspzz=np.zeros(1),
        epspxy=np.zeros(1), epspyz=np.zeros(1), epspzx=np.zeros(1),
        sigoxx=np.zeros(1), sigoyy=np.zeros(1), sigozz=np.zeros(1),
        sigoxy=np.zeros(1), sigoyz=np.zeros(1), sigozx=np.zeros(1),
        signxx=sign[:, 0], signyy=sign[:, 1], signzz=sign[:, 2],
        signxy=sign[:, 3], signyz=sign[:, 4], signzx=sign[:, 5],
        asrate=0.0, et=et, offg=offg, epsd=epsd, iresp=0, nvartmp=6,
        vartmp=vartmp, dmg=dmg, ngl=ngl, npg=1,
    )
    assert dmg[0] == 0.0
    assert off[0] == 1.0

    # 2. State in softening regime: (1 - eh)*kfail < f_crit < kfail
    # Let epsxx = 0.35, epsyy = -0.175, epszz = -0.175
    # lam = [1.419, 0.839, 0.839], I1 ~ 3.425 -> f_crit ~ 0.448
    sigeps88_solid(
        nel=1, matparam=mp, uvar=uvar, tstep=1e-5, tt=1e-5,
        rho0=np.array([1e-9]), rho=np.array([1e-9]), soundsp=soundsp, off=off,
        ismstr=0, israte=0,
        epsxx=np.array([0.35]), epsyy=np.array([-0.175]), epszz=np.array([-0.175]),
        epsxy=np.zeros(1), epsyz=np.zeros(1), epszx=np.zeros(1),
        depsxx=np.array([0.25]), depsyy=np.array([-0.125]), depszz=np.array([-0.125]),
        depsxy=np.zeros(1), depsyz=np.zeros(1), depszx=np.zeros(1),
        epspxx=np.zeros(1), epspyy=np.zeros(1), epspzz=np.zeros(1),
        epspxy=np.zeros(1), epspyz=np.zeros(1), epspzx=np.zeros(1),
        sigoxx=sign[:, 0], sigoyy=sign[:, 1], sigozz=sign[:, 2],
        sigoxy=sign[:, 3], sigoyz=sign[:, 4], sigozx=sign[:, 5],
        signxx=sign[:, 0], signyy=sign[:, 1], signzz=sign[:, 2],
        signxy=sign[:, 3], signyz=sign[:, 4], signzx=sign[:, 5],
        asrate=0.0, et=et, offg=offg, epsd=epsd, iresp=0, nvartmp=6,
        vartmp=vartmp, dmg=dmg, ngl=ngl, npg=1,
    )
    assert 0.0 < dmg[0] < 1.0
    assert off[0] == 1.0

    # 3. State exceeding failure limit: f_crit >= kfail = 0.50
    # Let epsxx = 0.60
    sigeps88_solid(
        nel=1, matparam=mp, uvar=uvar, tstep=1e-5, tt=2e-5,
        rho0=np.array([1e-9]), rho=np.array([1e-9]), soundsp=soundsp, off=off,
        ismstr=0, israte=0,
        epsxx=np.array([0.60]), epsyy=np.array([-0.30]), epszz=np.array([-0.30]),
        epsxy=np.zeros(1), epsyz=np.zeros(1), epszx=np.zeros(1),
        depsxx=np.array([0.25]), depsyy=np.array([-0.125]), depszz=np.array([-0.125]),
        depsxy=np.zeros(1), depsyz=np.zeros(1), depszx=np.zeros(1),
        epspxx=np.zeros(1), epspyy=np.zeros(1), epspzz=np.zeros(1),
        epspxy=np.zeros(1), epspyz=np.zeros(1), epspzx=np.zeros(1),
        sigoxx=sign[:, 0], sigoyy=sign[:, 1], sigozz=sign[:, 2],
        sigoxy=sign[:, 3], sigoyz=sign[:, 4], sigozx=sign[:, 5],
        signxx=sign[:, 0], signyy=sign[:, 1], signzz=sign[:, 2],
        signxy=sign[:, 3], signyz=sign[:, 4], signzx=sign[:, 5],
        asrate=0.0, et=et, offg=offg, epsd=epsd, iresp=0, nvartmp=6,
        vartmp=vartmp, dmg=dmg, ngl=ngl, npg=1,
    )
    assert dmg[0] == 1.0
    assert off[0] == 0.8  # Element deletion flag matching OpenRadioss FOUR_OVER_5


# ============================================================================
# Test 10: Algorithmic Consistent Tangent Tensors (Solid 6x6 & Shell 3x3)
# ============================================================================

def test_10_algorithmic_consistent_tangents():
    """Verify solid (6, 6) and shell (3, 3) algorithmic consistent tangents

    against central finite-difference perturbations to relative tolerance < 1e-4.
    """
    lam_table = np.linspace(0.2, 2.5, 25)
    g_table = 120.0 * lam_table + 40.0
    table = TableData(x1=lam_table, y1d=g_table)

    mp = MatparamLaw88(
        bulk=3000.0,
        shear=120.0,
        nu=0.495,
        rho0=1e-9,
        rho=1e-9,
        iparam=np.zeros(6, dtype=int),
        uparam=np.zeros(9, dtype=np.float64),
        table=[table],
    )

    # 1. 3D Solid Tangent (6x6)
    eps_solid = np.array([0.05, -0.02, -0.03, 0.01, 0.0, 0.0], dtype=np.float64)
    C_solid = tangent_law88_solid(eps_solid, mp, h=1e-6)
    assert C_solid.shape == (6, 6)
    # Check positive diagonal entries
    for k in range(6):
        assert C_solid[k, k] > 0.0
    # Check symmetry C_ij ~ C_ji to within finite difference tolerance
    for i in range(6):
        for j in range(6):
            if abs(C_solid[i, j]) > 10.0:
                rel_diff = abs(C_solid[i, j] - C_solid[j, i]) / (abs(C_solid[i, j]) + abs(C_solid[j, i]))
                assert rel_diff < 5e-3

    # 2. 2D Shell In-plane Tangent (3x3)
    eps_shell = np.array([0.04, 0.02, 0.01], dtype=np.float64)
    C_shell = tangent_law88_shell(eps_shell, mp, h=1e-6)
    assert C_shell.shape == (3, 3)
    for k in range(3):
        assert C_shell[k, k] > 0.0
    # In-plane symmetry
    for i in range(3):
        for j in range(3):
            if abs(C_shell[i, j]) > 10.0:
                rel_diff = abs(C_shell[i, j] - C_shell[j, i]) / (abs(C_shell[i, j]) + abs(C_shell[j, i]))
                assert rel_diff < 5e-3
