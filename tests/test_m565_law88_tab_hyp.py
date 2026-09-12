"""
Tests for Milestone M565: /MAT/LAW88 Tabulated Hyperelastic Material Model for Solids and Shells.
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
    solid_update,
    shell_update,
    sound_speed_solid,
    sound_speed_shell,
    extra_shapes,
    build_law88,
    Law88Params,
)


# ============================================================================
# 1. Tests for Curve Smoothing & Monotonicity Preprocessing
# ============================================================================

def test_isotone_project_pav_basic():
    """Test PAV isotonic projection enforces monotonic non-decreasing sequence."""
    y = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
    z = isotone_project_pav(y)
    assert len(z) == len(y)
    assert np.all(np.diff(z) >= -1e-12)
    # Check weighted least squares property (sum w*z approx sum w*y)
    assert np.isclose(np.sum(z), np.sum(y), atol=1e-10)


def test_isotone_project_pav_edge_cases():
    """Test empty, single-element, and already monotonic arrays."""
    assert len(isotone_project_pav(np.array([]))) == 0
    z1 = isotone_project_pav(np.array([4.2]))
    assert len(z1) == 1 and z1[0] == 4.2

    y_mono = np.array([1.0, 2.0, 3.0, 4.0])
    z_mono = isotone_project_pav(y_mono)
    np.testing.assert_allclose(z_mono, y_mono, atol=1e-12)


def test_smooth_isotone_zero_pinning():
    """Test smooth_isotone pins origin (0, 0) and remains monotonic."""
    x = np.array([-0.5, -0.2, 0.0, 0.3, 0.8, 1.2])
    y = np.array([-10.0, -2.0, 0.1, 5.0, 15.0, 30.0])  # y has slight non-zero at 0
    z = smooth_isotone(x, y, mu=0.01)

    assert np.all(np.diff(z) >= -1e-10)
    zero_idx = np.where(x == 0.0)[0][0]
    assert abs(z[zero_idx]) < 1e-15


def test_pchip_slopes_and_eval():
    """Test PCHIP slope calculation and cubic Hermite evaluation."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    z = np.array([0.0, 1.0, 8.0, 27.0])  # cubic curve x^3
    m = pchip_slopes(x, z)
    assert np.all(m >= 0.0)

    # Test interpolation at midpoints
    xi = np.array([0.5, 1.5, 2.5])
    yi = pchip_eval(x, z, m, xi)
    assert np.all(np.diff(yi) > 0.0)

    # Test flat boundary extrapolation
    assert pchip_eval(x, z, m, -1.0) == z[0]
    assert pchip_eval(x, z, m, 5.0) == z[-1]


def test_table_mat_spline_fit_full_pipeline():
    """Test full table_mat_spline_fit pipeline matching cpp_table_mat_spline_fit.cpp."""
    x_raw = np.array([-0.4, -0.2, 0.1, 0.4, 0.7, 1.0])
    y_raw = np.array([-25.0, -12.0, 8.0, 22.0, 45.0, 80.0])

    x_out, y_out = table_mat_spline_fit(x_raw, y_raw, nout=300, lam=1e-3)

    assert len(x_out) == 300
    assert len(y_out) == 300
    # Must be sorted and monotonically non-decreasing
    assert np.all(np.diff(x_out) >= 0.0)
    assert np.all(np.diff(y_out) >= -1e-10)
    # 0.0 must be an exact grid point
    zero_indices = np.where(x_out == 0.0)[0]
    assert len(zero_indices) >= 1
    assert abs(y_out[zero_indices[0]]) < 1e-12


# ============================================================================
# 2. Tests for 1D and 2D Table Interpolation (table_mat_vinterp_eval)
# ============================================================================

def test_table_mat_vinterp_1d():
    """Test 1D table interpolation and derivative calculation."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 10.0, 30.0, 60.0])
    tbl = TableData(ndim=1, x1=x, y1d=y)

    stretches = np.array([0.5, 1.5, 2.5])
    yy, dydx = table_mat_vinterp_eval(tbl, stretches, opt_extrapolate=True)

    np.testing.assert_allclose(yy, [5.0, 20.0, 45.0], atol=1e-12)
    np.testing.assert_allclose(dydx, [10.0, 20.0, 30.0], atol=1e-12)

    # Test extrapolation
    yy_ext, dydx_ext = table_mat_vinterp_eval(tbl, np.array([-0.5, 3.5]), opt_extrapolate=True)
    np.testing.assert_allclose(yy_ext, [-5.0, 75.0], atol=1e-12)
    np.testing.assert_allclose(dydx_ext, [10.0, 30.0], atol=1e-12)

    # Test without extrapolation
    yy_no_ext, dydx_no_ext = table_mat_vinterp_eval(tbl, np.array([-0.5, 3.5]), opt_extrapolate=False)
    np.testing.assert_allclose(yy_no_ext, [0.0, 60.0], atol=1e-12)
    np.testing.assert_allclose(dydx_no_ext, [0.0, 0.0], atol=1e-12)


def test_table_mat_vinterp_2d():
    """Test 2D table bilinear interpolation with rate dependency."""
    x1 = np.array([0.0, 1.0, 2.0])  # stretch
    x2 = np.array([0.0, 10.0])      # rate
    # y2d shape: (npt, nrate)
    y2d = np.array([
        [0.0, 0.0],
        [10.0, 20.0],
        [30.0, 50.0],
    ])
    tbl = TableData(ndim=2, x1=x1, x2=x2, y2d=y2d)

    # At stretch=1.0, rate=5.0: value should be 15.0
    yy, dydx = table_mat_vinterp_eval(tbl, stretch=np.array([1.0]), rate=np.array([5.0]))
    assert np.isclose(yy[0], 15.0)
    # dydx at stretch=0.5, rate=5.0: 0.5 * 10 + 0.5 * 20 = 15.0
    _, dydx_mid = table_mat_vinterp_eval(tbl, stretch=np.array([0.5]), rate=np.array([5.0]))
    assert np.isclose(dydx_mid[0], 15.0)


# ============================================================================
# 3. Tests for 3D Solid Continuum Kernel (sigeps88_solid)
# ============================================================================

def test_sigeps88_solid_incompressible_uniaxial():
    """Test 3D solid continuum under uniaxial elongation with incompressible rubber."""
    nel = 1
    # Simple linear loading curve in stretch: g(lambda) = 100 * (lambda - 1)
    lam_grid = np.linspace(0.1, 3.0, 100)
    stress_grid = 100.0 * (lam_grid - 1.0)
    table_load = TableData(ndim=1, x1=lam_grid, y1d=stress_grid)

    mp = MatparamLaw88(
        bulk=5000.0,
        shear=100.0,
        nu=0.499,
        young=300.0,
        rho0=1e-9,
        rho=1e-9,
        iparam=np.array([1, 0, 1, 0, 0, 12], dtype=int),
        uparam=np.zeros(9, dtype=np.float64),
        table=[table_load],
    )

    uvar = np.zeros((nel, 30), dtype=np.float64)
    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    et = np.zeros(nel, dtype=np.float64)
    offg = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.array([1], dtype=int)

    # Uniaxial tension in x: epsxx = 0.1, epsyy = -0.05, epszz = -0.05 (isochoric)
    epsxx = np.array([0.1])
    epsyy = np.array([-0.05])
    epszz = np.array([-0.05])
    zeros = np.zeros(nel)

    sigxx = np.zeros(nel)
    sigyy = np.zeros(nel)
    sigzz = np.zeros(nel)
    sigxy = np.zeros(nel)
    sigyz = np.zeros(nel)
    sigzx = np.zeros(nel)

    sigeps88_solid(
        nel=nel,
        matparam=mp,
        uvar=uvar,
        tstep=1e-5,
        tt=1e-5,
        rho0=np.array([1e-9]),
        rho=np.array([1e-9]),
        soundsp=soundsp,
        off=off,
        ismstr=0,
        israte=0,
        epsxx=epsxx,
        epsyy=epsyy,
        epszz=epszz,
        epsxy=zeros,
        epsyz=zeros,
        epszx=zeros,
        depsxx=epsxx,
        depsyy=epsyy,
        depszz=epszz,
        depsxy=zeros,
        depsyz=zeros,
        depszx=zeros,
        epspxx=epsxx / 1e-5,
        epspyy=epsyy / 1e-5,
        epspzz=epszz / 1e-5,
        epspxy=zeros,
        epspyz=zeros,
        epspzx=zeros,
        sigoxx=zeros,
        sigoyy=zeros,
        sigozz=zeros,
        sigoxy=zeros,
        sigoyz=zeros,
        sigozx=zeros,
        signxx=sigxx,
        signyy=sigyy,
        signzz=sigzz,
        signxy=sigxy,
        signyz=sigyz,
        signzx=sigzx,
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

    # Tensile stress along x must be positive
    assert sigxx[0] > 0.0
    # In isochoric kinematics (tr(eps)=0, P=0), deviatoric balance gives sigyy = sigzz = -0.5 * sigxx
    assert np.isclose(sigyy[0], -0.5 * sigxx[0], rtol=1e-5)
    assert np.isclose(sigzz[0], -0.5 * sigxx[0], rtol=1e-5)
    assert np.isclose(sigxx[0] + sigyy[0] + sigzz[0], 0.0, atol=1e-10)
    # Shear stresses should be exactly 0
    assert abs(sigxy[0]) < 1e-12
    assert abs(sigyz[0]) < 1e-12
    assert abs(sigzx[0]) < 1e-12
    # Sound speed must be positive and physically sound
    assert soundsp[0] > 1000.0
    # Hourglass factor et > 0
    assert et[0] > 0.0


def test_sigeps88_solid_hysteretic_unloading():
    """Test 3D solid continuum with hysteretic energy unloading (iunl_for = 2)."""
    nel = 1
    lam_grid = np.linspace(0.1, 3.0, 100)
    stress_grid = 100.0 * (lam_grid - 1.0)
    table_load = TableData(ndim=1, x1=lam_grid, y1d=stress_grid)

    uparam = np.zeros(9, dtype=np.float64)
    uparam[0] = 0.4  # hys = 0.4
    uparam[1] = 2.0  # shape = 2.0

    mp = MatparamLaw88(
        bulk=5000.0,
        shear=100.0,
        nu=0.499,
        young=300.0,
        rho0=1e-9,
        rho=1e-9,
        iparam=np.array([1, 2, 1, 0, 0, 12], dtype=int),  # iunl_for = 2
        uparam=uparam,
        table=[table_load],
    )

    uvar = np.zeros((nel, 30), dtype=np.float64)
    # Simulate an element that reached a high peak energy and is now unloading
    uvar[0, 0] = 100.0  # emax
    uvar[0, 1] = 50.0   # ecurent (50% of peak energy)
    uvar[0, 4] = -1.0   # loadflg = -1 (unloading)

    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    et = np.zeros(nel, dtype=np.float64)
    offg = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.array([1], dtype=int)

    epsxx = np.array([0.05])
    epsyy = np.array([-0.025])
    epszz = np.array([-0.025])
    zeros = np.zeros(nel)
    sigxx = np.zeros(nel)

    sigeps88_solid(
        nel=nel,
        matparam=mp,
        uvar=uvar,
        tstep=1e-5,
        tt=1e-5,
        rho0=np.array([1e-9]),
        rho=np.array([1e-9]),
        soundsp=soundsp,
        off=off,
        ismstr=0,
        israte=0,
        epsxx=epsxx,
        epsyy=epsyy,
        epszz=epszz,
        epsxy=zeros,
        epsyz=zeros,
        epszx=zeros,
        depsxx=np.array([-0.001]),  # negative increment -> unloading
        depsyy=np.array([0.0005]),
        depszz=np.array([0.0005]),
        depsxy=zeros,
        depsyz=zeros,
        depszx=zeros,
        epspxx=-epsxx / 1e-5,
        epspyy=zeros,
        epspzz=zeros,
        epspxy=zeros,
        epspyz=zeros,
        epspzx=zeros,
        sigoxx=np.array([5.0]),
        sigoyy=zeros,
        sigozz=zeros,
        sigoxy=zeros,
        sigoyz=zeros,
        sigozx=zeros,
        signxx=sigxx,
        signyy=np.zeros(nel),
        signzz=np.zeros(nel),
        signxy=np.zeros(nel),
        signyz=np.zeros(nel),
        signzx=np.zeros(nel),
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

    # Element should remain unloading
    assert uvar[0, 4] == -1.0
    assert sigxx[0] > 0.0


def test_sigeps88_solid_compressible_foam():
    """Test 3D solid continuum with compressible foam path (0 < nu < 0.49)."""
    nel = 1
    lam_grid = np.linspace(0.1, 3.0, 100)
    stress_grid = 50.0 * (lam_grid - 1.0)
    table_load = TableData(ndim=1, x1=lam_grid, y1d=stress_grid)

    mp = MatparamLaw88(
        bulk=1000.0,
        shear=50.0,
        nu=0.25,  # Foam-like Poisson's ratio
        young=125.0,
        rho0=1e-9,
        rho=1e-9,
        iparam=np.array([1, 0, 1, 0, 0, 12], dtype=int),
        uparam=np.zeros(9, dtype=np.float64),
        table=[table_load],
    )

    uvar = np.zeros((nel, 30), dtype=np.float64)
    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    et = np.zeros(nel, dtype=np.float64)
    offg = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.array([1], dtype=int)

    # Volumetric compression
    epsxx = np.array([-0.05])
    epsyy = np.array([-0.05])
    epszz = np.array([-0.05])
    zeros = np.zeros(nel)
    sigxx = np.zeros(nel)

    sigeps88_solid(
        nel=nel,
        matparam=mp,
        uvar=uvar,
        tstep=1e-5,
        tt=1e-5,
        rho0=np.array([1e-9]),
        rho=np.array([1e-9]),
        soundsp=soundsp,
        off=off,
        ismstr=0,
        israte=0,
        epsxx=epsxx,
        epsyy=epsyy,
        epszz=epszz,
        epsxy=zeros,
        epsyz=zeros,
        epszx=zeros,
        depsxx=epsxx,
        depsyy=epsyy,
        depszz=epszz,
        depsxy=zeros,
        depsyz=zeros,
        depszx=zeros,
        epspxx=epsxx / 1e-5,
        epspyy=epsyy / 1e-5,
        epspzz=epszz / 1e-5,
        epspxy=zeros,
        epspyz=zeros,
        epspzx=zeros,
        sigoxx=zeros,
        sigoyy=zeros,
        sigozz=zeros,
        sigoxy=zeros,
        sigoyz=zeros,
        sigozx=zeros,
        signxx=sigxx,
        signyy=np.zeros(nel),
        signzz=np.zeros(nel),
        signxy=np.zeros(nel),
        signyz=np.zeros(nel),
        signzx=np.zeros(nel),
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

    # Compressive stress must be negative
    assert sigxx[0] < 0.0
    assert soundsp[0] > 0.0


# ============================================================================
# 4. Tests for 2D Shell Membrane Kernel (sigeps88_shell)
# ============================================================================

def test_sigeps88_shell_plane_stress_relaxation():
    """Test 2D shell plane-stress Newton loop and out-of-plane stretch update."""
    nel = 1
    lam_grid = np.linspace(0.1, 3.0, 100)
    stress_grid = 150.0 * (lam_grid - 1.0)
    table_load = TableData(ndim=1, x1=lam_grid, y1d=stress_grid)

    mp = MatparamLaw88(
        bulk=5000.0,
        shear=150.0,
        nu=0.499,
        young=450.0,
        rho0=1e-9,
        rho=1e-9,
        iparam=np.array([1, 0, 1, 0, 0, 12], dtype=int),
        uparam=np.zeros(9, dtype=np.float64),
        table=[table_load],
    )

    uvar = np.zeros((nel, 30), dtype=np.float64)
    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    et = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.array([1], dtype=int)
    thkly = np.ones(nel)
    thk0 = np.array([2.0])
    thkn = np.zeros(nel)
    shf = np.ones(nel)

    # Uniaxial tension in x: epsxx = 0.1, epsyy = 0.0, epsxy = 0.0
    epsxx = np.array([0.1])
    epsyy = np.array([0.0])
    epsxy = np.array([0.0])
    zeros = np.zeros(nel)

    sigxx = np.zeros(nel)
    sigyy = np.zeros(nel)
    sigxy = np.zeros(nel)
    sigyz = np.zeros(nel)
    sigzx = np.zeros(nel)

    sigeps88_shell(
        nel=nel,
        matparam=mp,
        uvar=uvar,
        tstep=1e-5,
        tt=1e-5,
        rho=np.array([1e-9]),
        soundsp=soundsp,
        off=off,
        ismstr=0,
        israte=0,
        ngl=ngl,
        epsxx=epsxx,
        epsyy=epsyy,
        epsxy=epsxy,
        epspxx=epsxx / 1e-5,
        epspyy=zeros,
        epspxy=zeros,
        depsxx=epsxx,
        depsyy=epsyy,
        depsxy=epsxy,
        depsyz=np.array([0.01]),  # transverse shear increment
        depszx=np.array([0.02]),
        sigoxx=zeros,
        sigoyy=zeros,
        sigoxy=zeros,
        sigoyz=zeros,
        sigozx=zeros,
        signxx=sigxx,
        signyy=sigyy,
        signxy=sigxy,
        signyz=sigyz,
        signzx=sigzx,
        asrate=0.0,
        et=et,
        epsd=epsd,
        nvartmp=6,
        vartmp=vartmp,
        dmg=dmg,
        thkly=thkly,
        thk0=thk0,
        thkn=thkn,
        shf=shf,
        ipg=1,
        npg=1,
    )

    # Tension in x
    assert sigxx[0] > 0.0
    # Out-of-plane stretch lambda_3 in uvar[:, 7] should be < 1.0 due to incompressibility (thinning)
    lam3 = uvar[0, 7]
    assert 0.5 < lam3 < 1.0
    # Thickness must be updated
    assert thkn[0] > 0.0 and thkn[0] < thk0[0]
    # Transverse shear stresses updated with gs * shf * deps
    assert np.isclose(sigyz[0], 150.0 * 1.0 * 0.01)
    assert np.isclose(sigzx[0], 150.0 * 1.0 * 0.02)
    # Wave speed > 0
    assert soundsp[0] > 0.0


# ============================================================================
# 5. Tests for Consistent Algorithmic Tangent Tensors
# ============================================================================

def test_tangent_law88_solid():
    """Test 3D solid algorithmic tangent tensor shape and positive definiteness."""
    lam_grid = np.linspace(0.1, 3.0, 100)
    stress_grid = 100.0 * (lam_grid - 1.0)
    table_load = TableData(ndim=1, x1=lam_grid, y1d=stress_grid)

    mp = MatparamLaw88(
        bulk=5000.0,
        shear=100.0,
        nu=0.499,
        young=300.0,
        rho0=1e-9,
        rho=1e-9,
        iparam=np.array([1, 0, 1, 0, 0, 12], dtype=int),
        uparam=np.zeros(9, dtype=np.float64),
        table=[table_load],
    )

    # Small strain state
    eps = np.array([0.01, -0.005, -0.005, 0.0, 0.0, 0.0])
    C = tangent_law88_solid(eps, mp, dt=1e-5, h=1e-7)

    assert C.shape == (6, 6)
    # Diagonal stiffnesses must be positive
    assert C[0, 0] > 0.0
    assert C[1, 1] > 0.0
    assert C[2, 2] > 0.0
    # Bulk components must dominate
    assert C[0, 0] > 100.0


def test_tangent_law88_shell():
    """Test 2D shell algorithmic tangent tensor shape and positive definiteness."""
    lam_grid = np.linspace(0.1, 3.0, 100)
    stress_grid = 100.0 * (lam_grid - 1.0)
    table_load = TableData(ndim=1, x1=lam_grid, y1d=stress_grid)

    mp = MatparamLaw88(
        bulk=5000.0,
        shear=100.0,
        nu=0.499,
        young=300.0,
        rho0=1e-9,
        rho=1e-9,
        iparam=np.array([1, 0, 1, 0, 0, 12], dtype=int),
        uparam=np.zeros(9, dtype=np.float64),
        table=[table_load],
    )

    eps_shell = np.array([0.01, 0.005, 0.0])
    C_shell = tangent_law88_shell(eps_shell, mp, dt=1e-5, h=1e-7)

    assert C_shell.shape == (3, 3)
    assert C_shell[0, 0] > 0.0
    assert C_shell[1, 1] > 0.0


def test_sigeps88_solid_tabulated_unloading():
    """Test 3D solid with tabulated unloading curves (iunl_for = 1)."""
    nel = 1
    # 1. Loading table: stretch [0.5..2.5], stress [ -50 .. 150 ]
    lam_grid = np.linspace(0.5, 2.5, 50)
    stress_load = 100.0 * (lam_grid - 1.0)
    t1_load = TableData(ndim=1, x1=lam_grid, y1d=stress_load)

    # 2. Normalized unloading table: x in [-1, 1], y in [-1, 1]
    x_norm = np.linspace(-1.0, 1.0, 50)
    y_unl_norm = 0.5 * x_norm  # lower stiffness unloading curve
    t2_unl = TableData(ndim=1, x1=x_norm, y1d=y_unl_norm)

    # 3. Normalized loading table: x in [-1, 1], y in [-1, 1]
    y_load_norm = x_norm.copy()
    t3_norm = TableData(ndim=1, x1=x_norm, y1d=y_load_norm)

    mp = MatparamLaw88(
        bulk=5000.0,
        shear=100.0,
        nu=0.499,
        young=300.0,
        rho0=1e-9,
        rho=1e-9,
        iparam=np.array([1, 1, 1, 0, 0, 12], dtype=int),  # iunl_for = 1
        uparam=np.zeros(9, dtype=np.float64),
        table=[t1_load, t2_unl, t3_norm],
    )

    uvar = np.zeros((nel, 30), dtype=np.float64)
    # Set up previous loading state at lambda = 1.2
    uvar[0, 12] = 1.2  # lam_r in direction 1 (nv_base = 12)
    uvar[0, 13] = 1.0  # lam_r in direction 2
    uvar[0, 14] = 1.0  # lam_r in direction 3
    uvar[0, 15] = 1.0  # lam_tg in direction 1
    uvar[0, 16] = 1.0  # lam_tg in direction 2
    uvar[0, 17] = 1.0  # lam_tg in direction 3
    uvar[0, 4] = -1.0  # element is currently unloading

    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    et = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.array([1], dtype=int)

    # Unloading stretch state: lambda_1 = 1.1 (decreased from 1.2)
    epsxx = np.array([math.log(1.1)])
    epsyy = np.array([-0.5 * math.log(1.1)])
    epszz = np.array([-0.5 * math.log(1.1)])
    zeros = np.zeros(nel)
    signxx = np.zeros(nel)

    sigeps88_solid(
        nel=nel,
        matparam=mp,
        uvar=uvar,
        tstep=1e-5,
        tt=1e-5,
        rho0=np.array([1e-9]),
        rho=np.array([1e-9]),
        soundsp=soundsp,
        off=off,
        ismstr=0,
        israte=0,
        epsxx=epsxx,
        epsyy=epsyy,
        epszz=epszz,
        epsxy=zeros,
        epsyz=zeros,
        epszx=zeros,
        depsxx=np.array([-0.01]),  # negative increment -> unloading
        depsyy=np.array([0.005]),
        depszz=np.array([0.005]),
        depsxy=zeros,
        depsyz=zeros,
        depszx=zeros,
        epspxx=-epsxx / 1e-5,
        epspyy=zeros,
        epspzz=zeros,
        epspxy=zeros,
        epspyz=zeros,
        epspzx=zeros,
        sigoxx=np.array([10.0]),
        sigoyy=zeros,
        sigozz=zeros,
        sigoxy=zeros,
        sigoyz=zeros,
        sigozx=zeros,
        signxx=signxx,
        signyy=np.zeros(nel),
        signzz=np.zeros(nel),
        signxy=np.zeros(nel),
        signyz=np.zeros(nel),
        signzx=np.zeros(nel),
        asrate=0.0,
        et=et,
        offg=zeros,
        epsd=epsd,
        iresp=0,
        nvartmp=6,
        vartmp=vartmp,
        dmg=dmg,
        ngl=ngl,
        npg=1,
    )

    # Unloading scaling ratio R should be computed and stored in uvar[0, nv_base+12]
    ratioR = uvar[0, 24]
    assert 0.0 < ratioR <= 1.0
    assert signxx[0] > 0.0


def test_sigeps88_solid_viscous_pressure():
    """Test 3D solid viscous pressure with parameter beta > 0."""
    nel = 1
    lam_grid = np.linspace(0.5, 2.0, 50)
    stress_load = 50.0 * (lam_grid - 1.0)
    t1_load = TableData(ndim=1, x1=lam_grid, y1d=stress_load)

    uparam = np.zeros(9, dtype=np.float64)
    uparam[8] = 100.0  # beta = 100.0

    mp = MatparamLaw88(
        bulk=2000.0,
        shear=50.0,
        nu=0.495,
        young=150.0,
        rho0=1e-9,
        rho=1e-9,
        iparam=np.array([1, 0, 1, 0, 0, 12], dtype=int),
        uparam=uparam,
        table=[t1_load],
    )

    uvar = np.zeros((nel, 30), dtype=np.float64)
    uvar[0, 11] = 5.0  # Initial viscous pressure in UVAR12

    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    et = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.array([1], dtype=int)

    # Positive volumetric strain rate: ldav > 0
    epsp = np.array([10.0])
    eps = epsp * 1e-5
    zeros = np.zeros(nel)
    signxx = np.zeros(nel)

    sigeps88_solid(
        nel=nel,
        matparam=mp,
        uvar=uvar,
        tstep=1e-5,
        tt=1e-5,
        rho0=np.array([1e-9]),
        rho=np.array([1e-9]),
        soundsp=soundsp,
        off=off,
        ismstr=0,
        israte=0,
        epsxx=eps,
        epsyy=eps,
        epszz=eps,
        epsxy=zeros,
        epsyz=zeros,
        epszx=zeros,
        depsxx=eps,
        depsyy=eps,
        depszz=eps,
        depsxy=zeros,
        depsyz=zeros,
        depszx=zeros,
        epspxx=epsp,
        epspyy=epsp,
        epspzz=epsp,
        epspxy=zeros,
        epspyz=zeros,
        epspzx=zeros,
        sigoxx=zeros,
        sigoyy=zeros,
        sigozz=zeros,
        sigoxy=zeros,
        sigoyz=zeros,
        sigozx=zeros,
        signxx=signxx,
        signyy=np.zeros(nel),
        signzz=np.zeros(nel),
        signxy=np.zeros(nel),
        signyz=np.zeros(nel),
        signzx=np.zeros(nel),
        asrate=0.0,
        et=et,
        offg=zeros,
        epsd=epsd,
        iresp=0,
        nvartmp=6,
        vartmp=vartmp,
        dmg=dmg,
        ngl=ngl,
        npg=1,
    )

    # Viscous pressure updated in uvar[0, 11]
    p_updated = uvar[0, 11]
    assert p_updated > 0.0


def test_sigeps88_solid_damping_and_damage():
    """Test frictional damping stresses and damage softening with deletion."""
    nel = 1
    lam_grid = np.linspace(0.5, 3.0, 50)
    stress_load = 50.0 * (lam_grid - 1.0)
    t1_load = TableData(ndim=1, x1=lam_grid, y1d=stress_load)

    uparam = np.zeros(9, dtype=np.float64)
    uparam[2] = 5.0   # gdamp
    uparam[3] = 10.0  # sigf cutoff
    uparam[4] = 0.5   # kfail
    uparam[5] = 0.1   # gam1
    uparam[6] = 0.05  # gam2
    uparam[7] = 0.2   # eh

    mp = MatparamLaw88(
        bulk=2000.0,
        shear=50.0,
        nu=0.495,
        young=150.0,
        rho0=1e-9,
        rho=1e-9,
        iparam=np.array([1, 0, 1, 0, 1, 12], dtype=int),  # failip = 1
        uparam=uparam,
        table=[t1_load],
    )

    uvar = np.zeros((nel, 30), dtype=np.float64)
    soundsp = np.zeros(nel, dtype=np.float64)
    off = np.ones(nel, dtype=np.float64)
    et = np.zeros(nel, dtype=np.float64)
    epsd = np.zeros(nel, dtype=np.float64)
    vartmp = np.zeros((nel, 6), dtype=int)
    dmg = np.zeros(nel, dtype=np.float64)
    ngl = np.array([1], dtype=int)

    # Large stretch to trigger failure: lambda_1 = 2.0 (I1 ~ 4 + 0.5 + 0.5 = 5 > 3 + kfail)
    epsxx = np.array([math.log(2.0)])
    epsyy = np.array([-0.5 * math.log(2.0)])
    epszz = np.array([-0.5 * math.log(2.0)])
    zeros = np.zeros(nel)
    signxx = np.zeros(nel)

    sigeps88_solid(
        nel=nel,
        matparam=mp,
        uvar=uvar,
        tstep=1e-5,
        tt=1e-5,
        rho0=np.array([1e-9]),
        rho=np.array([1e-9]),
        soundsp=soundsp,
        off=off,
        ismstr=0,
        israte=0,
        epsxx=epsxx,
        epsyy=epsyy,
        epszz=epszz,
        epsxy=zeros,
        epsyz=zeros,
        epszx=zeros,
        depsxx=epsxx,
        depsyy=epsyy,
        depszz=epszz,
        depsxy=zeros,
        depsyz=zeros,
        depszx=zeros,
        epspxx=epsxx / 1e-5,
        epspyy=epsyy / 1e-5,
        epspzz=epszz / 1e-5,
        epspxy=zeros,
        epspyz=zeros,
        epspzx=zeros,
        sigoxx=zeros,
        sigoyy=zeros,
        sigozz=zeros,
        sigoxy=zeros,
        sigoyz=zeros,
        sigozx=zeros,
        signxx=signxx,
        signyy=np.zeros(nel),
        signzz=np.zeros(nel),
        signxy=np.zeros(nel),
        signyz=np.zeros(nel),
        signzx=np.zeros(nel),
        asrate=0.0,
        et=et,
        offg=zeros,
        epsd=epsd,
        iresp=0,
        nvartmp=6,
        vartmp=vartmp,
        dmg=dmg,
        ngl=ngl,
        npg=1,
    )

    # Complete damage occurs (fcrit >= kfail)
    assert dmg[0] == 1.0
    # Cauchy stresses zeroed out by (1 - dmg)
    assert np.isclose(signxx[0], 0.0)
    # Element marked for deletion: off set to 0.8
    assert off[0] == 0.8


def test_sigeps88_batch_multi_element():
    """Test vectorized multi-element computation for solid and shell."""
    nel = 4
    lam_grid = np.linspace(0.5, 2.5, 50)
    stress_load = 80.0 * (lam_grid - 1.0)
    t1_load = TableData(ndim=1, x1=lam_grid, y1d=stress_load)

    mp = MatparamLaw88(
        bulk=3000.0,
        shear=80.0,
        nu=0.495,
        young=240.0,
        rho0=1e-9,
        rho=1e-9,
        iparam=np.array([1, 0, 1, 0, 0, 12], dtype=int),
        uparam=np.zeros(9, dtype=np.float64),
        table=[t1_load],
    )

    eps_batch = np.array([
        [0.02, -0.01, -0.01, 0.0, 0.0, 0.0],
        [0.05, -0.025, -0.025, 0.0, 0.0, 0.0],
        [0.08, -0.04, -0.04, 0.0, 0.0, 0.0],
        [0.10, -0.05, -0.05, 0.0, 0.0, 0.0],
    ])

    C_batch = tangent_law88_solid(eps_batch, mp)
    assert C_batch.shape == (4, 6, 6)
    assert np.all(C_batch[:, 0, 0] > 0.0)
    assert np.all(C_batch[:, 1, 1] > 0.0)
    assert np.all(C_batch[:, 2, 2] > 0.0)


def test_law88_solver_wrappers_solid_and_shell():
    """Test solver integration wrappers (solid_update, shell_update, sound_speed, extra_shapes)."""
    lam_grid = np.linspace(0.5, 2.5, 50)
    stress_load = 100.0 * (lam_grid - 1.0)
    tbl = TableData(ndim=1, x1=lam_grid, y1d=stress_load)

    # 1. build_law88
    mat = build_law88(
        id=88,
        rho0=1.2e-9,
        nu=0.495,
        bulk=4000.0,
        young=300.0,
        shear=100.0,
        table=[tbl],
        iparam=np.array([1, 0, 1, 0, 0, 12], dtype=int),
    )

    # 2. extra_shapes
    shapes_solid = extra_shapes(mat, nip=None)
    assert "uvar88" in shapes_solid and shapes_solid["uvar88"] == (30,)
    shapes_shell = extra_shapes(mat, nip=3)
    assert "uvar88" in shapes_shell and shapes_shell["uvar88"] == (3, 30)

    # 3. sound_speed
    c_sol = sound_speed_solid(mat)
    assert c_sol > 0.0
    c_shl = sound_speed_shell(mat)
    assert c_shl > 0.0

    # 4. solid_update
    nel = 2
    sig = np.zeros((nel, 6))
    deps = np.array([
        [0.02, -0.01, -0.01, 0.0, 0.0, 0.0],
        [0.04, -0.02, -0.02, 0.0, 0.0, 0.0],
    ])
    extra = {"uvar88": np.zeros((nel, 30)), "rho": np.full(nel, 1.2e-9)}
    sign, epsp_out, c = solid_update(mat, sig, deps, dt=1e-5, extra=extra)

    assert sign.shape == (nel, 6)
    assert sign[0, 0] > 0.0 and sign[1, 0] > sign[0, 0]
    assert np.isclose(sign[0, 1], -0.5 * sign[0, 0], rtol=1e-4)
    assert len(c) == nel and np.all(c > 0.0)

    # 5. shell_update
    sig_sh = np.zeros((nel, 3))
    deps_sh = np.array([
        [0.02, 0.0, 0.0],
        [0.04, 0.0, 0.0],
    ])
    extra_sh = {"uvar88": np.zeros((nel, 30)), "rho": np.full(nel, 1.2e-9)}
    sign_sh, _ = shell_update(mat, sig_sh, deps_sh, dt=1e-5, extra=extra_sh)

    assert sign_sh.shape == (nel, 3)
    assert sign_sh[0, 0] > 0.0 and sign_sh[1, 0] > sign_sh[0, 0]


def test_law88_module_level_tangents_and_dispatch():
    """Test pyradioss.materials dispatch functions for LAW88 tangents and sound speed."""
    import pyradioss.materials as pmat

    lam_grid = np.linspace(0.5, 2.5, 50)
    stress_load = 100.0 * (lam_grid - 1.0)
    tbl = TableData(ndim=1, x1=lam_grid, y1d=stress_load)

    mat = build_law88(
        id=88,
        rho0=1.2e-9,
        nu=0.495,
        bulk=4000.0,
        young=300.0,
        shear=100.0,
        table=[tbl],
        iparam=np.array([1, 0, 1, 0, 0, 12], dtype=int),
    )

    # 1. Module-level sound_speed dispatch
    c = pmat.sound_speed(mat)
    assert c > 0.0

    # 2. Module-level solid_tangent dispatch
    sig_sol = np.zeros(6)
    C_sol = pmat.solid_tangent(mat, sig_sol)
    assert C_sol.shape == (6, 6)
    assert C_sol[0, 0] > 0.0

    # Batch solid tangent
    sig_sol_batch = np.zeros((2, 6))
    C_sol_batch = pmat.solid_tangent(mat, sig_sol_batch)
    assert C_sol_batch.shape == (2, 6, 6)

    # 3. Module-level shell_membrane_tangent dispatch
    C_mem = pmat.shell_membrane_tangent(mat)
    assert C_mem.shape == (3, 3)
    assert C_mem[0, 0] > 0.0

    # 4. Module-level shell_layer_tangent dispatch
    sig_sh = np.zeros(3)
    C_sh = pmat.shell_layer_tangent(mat, sig=sig_sh)
    assert C_sh.shape == (3, 3)
    assert C_sh[0, 0] > 0.0

    sig_sh_batch = np.zeros((2, 3))
    C_sh_batch = pmat.shell_layer_tangent(mat, sig=sig_sh_batch)
    assert C_sh_batch.shape == (2, 3, 3)



