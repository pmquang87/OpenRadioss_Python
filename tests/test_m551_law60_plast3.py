"""
Unit tests for LAW60 — Tabulated elasto-plastic material model (/MAT/LAW60, /MAT/PLAS_T3, /MAT/FABRIC).

Tests verify:
1. Law60Params dataclass initialization, default values, and build_law60 constructor.
2. Pure elastic response for 3D solids and 2D shells (below yield).
3. Uniaxial yield and radial return for 3D solids and 2D shells.
4. Multi-rate hardening (strain-rate dependent yield curves and rate interpolation).
5. Pressure sensitivity function (ipfun and pscale hydrostatic scaling).
6. Modulus degradation (both exponential degradation and curve-based degradation).
7. Tensile damage failure scaling (eps_t1..eps_t2 degradation of yield stress).
8. Element deletion (epsp >= eps_max zeroes stress and sets off = 0).
9. Instantaneous longitudinal and shell sound speeds.
10. Algorithmic consistent tangents for solids and shells verified against finite differences.
11. Multi-element batch vectorization consistency.
"""

import numpy as np
import pytest

from pyradioss.common.tables import FunctTable
from pyradioss.materials.law60_plast3 import (
    Law60Params,
    build_law60,
    solid_update,
    shell_update,
    sound_speed,
    consistent_solid_tangent,
    consistent_shell_tangent,
)
from pyradioss.model.entities import Material


# ============================================================================
# 1. Parameter Dataclass and Builder Tests
# ============================================================================

def test_law60_params_defaults():
    """Verify default values and helper elastic constants on Law60Params."""
    p = Law60Params(e0=210000.0, nu=0.3)
    assert p.e0 == 210000.0
    assert p.nu == 0.3
    assert p.eps_max == 1e30
    assert p.eps_t1 == 1e30
    assert p.eps_t2 == 2e30
    assert p.nfunc == 1
    assert p.fsmooth == 0
    assert p.fisokin == 0.0
    assert p.fcut == 1e30
    assert p.ipfun == 0
    assert p.ifunce == 0
    assert p.pscale == 0.0
    assert p.einf == 0.0
    assert p.ce == 0.0

    expected_g0 = 210000.0 / (2.0 * (1.0 + 0.3))
    expected_c1_0 = 210000.0 / (3.0 * (1.0 - 2.0 * 0.3))
    assert np.isclose(p.G0, expected_g0)
    assert np.isclose(p.C1_0, expected_c1_0)
    assert np.isclose(p.G, expected_g0)
    assert np.isclose(p.K, expected_c1_0)
    assert np.isclose(p.E, 210000.0)


def test_build_law60_from_dict_and_curves():
    """Verify build_law60 extracts cards and maps /FUNCT curves correctly."""
    f1 = FunctTable(101, [0.0, 0.05, 0.2], [300.0, 400.0, 500.0])
    f2 = FunctTable(102, [0.0, 0.05, 0.2], [350.0, 450.0, 550.0])
    fp = FunctTable(201, [-1000.0, 0.0, 1000.0], [0.8, 1.0, 1.3])
    fe = FunctTable(301, [0.0, 0.1, 0.5], [1.0, 0.9, 0.6])
    functs = {101: f1, 102: f2, 201: fp, 301: fe}

    card_dict = {
        "id": 5,
        "MAT_RHO": 7.85e-9,
        "MAT_E": 200000.0,
        "MAT_NU": 0.28,
        "MAT_EPS": 0.35,
        "MAT_EPST1": 0.15,
        "MAT_EPST2": 0.25,
        "NFUNC": 2,
        "Fsmooth": 1,
        "MAT_HARD": 0.2,
        "Fcut": 5000.0,
        "Xr_fun": 201,
        "fct_ID_k": 301,
        "MAT_FScale": 2.0,  # Should become 1/2.0 = 0.5
        "E_R": 100000.0,
        "MAT_C1": 12.0,
        "FUN_A1": 101,
        "FUN_B1": 102,
        "MAT_ALPHA1": 1.0,
        "MAT_ALPHA2": 1.0,
        "MAT_EPSR1": 0.0,
        "MAT_EPSR2": 100.0,
    }

    params = build_law60(card_dict, functs=functs)
    assert params.id == 5
    assert np.isclose(params.rho0, 7.85e-9)
    assert np.isclose(params.e0, 200000.0)
    assert np.isclose(params.nu, 0.28)
    assert np.isclose(params.eps_max, 0.35)
    assert np.isclose(params.eps_t1, 0.15)
    assert np.isclose(params.eps_t2, 0.25)
    assert params.nfunc == 2
    assert params.fsmooth == 1
    assert np.isclose(params.fisokin, 0.2)
    assert np.isclose(params.fcut, 5000.0)
    assert params.ipfun == 201
    assert params.ifunce == 301
    assert np.isclose(params.pscale, 0.5)  # 1 / 2.0
    assert np.isclose(params.einf, 100000.0)
    assert np.isclose(params.ce, 12.0)

    # Check curves mapped
    assert len(params.curve_x) == 2
    assert np.allclose(params.curve_x[0], [0.0, 0.05, 0.2])
    assert np.allclose(params.curve_y[0], [300.0, 400.0, 500.0])
    assert np.allclose(params.rates, [0.0, 100.0])
    assert params.p_curve_x is not None
    assert params.e_curve_x is not None


def test_build_law60_from_material_entity():
    """Verify build_law60 with Material model entity."""
    f1 = FunctTable(1, [0.0, 0.1], [250.0, 350.0])
    mat = Material(id=12, law=60, rho0=2.7e-9, params={
        "E": 70000.0, "nu": 0.33, "funcs": [f1],
    })
    params = build_law60(mat)
    assert params.id == 12
    assert np.isclose(params.e0, 70000.0)
    assert np.isclose(params.nu, 0.33)
    assert len(params.curve_x) >= 1


# ============================================================================
# 2. Pure Elastic Response Tests
# ============================================================================

def test_pure_elastic_solid():
    """Test 3D solid continuum update below yield stress."""
    f1 = FunctTable(1, [0.0, 0.1], [400.0, 500.0])
    p = Law60Params(e0=200000.0, nu=0.3, rho0=7.8e-9, funcs=[f1])
    p = build_law60(p)

    sig_old = np.zeros(6)
    # Small shear and volumetric strain below yield
    deps = np.array([1e-4, -0.3e-4, -0.3e-4, 2e-4, 0.0, 0.0])
    sig_new, epsp_new, c = solid_update(p, eps=None, deps=deps, sig_old=sig_old, epsp_old=0.0, dt=1e-6)

    assert np.isclose(epsp_new, 0.0)

    # Analytical elastic prediction
    g = p.G0
    k = p.C1_0
    dav = (deps[0] + deps[1] + deps[2]) / 3.0
    sig_expected = np.zeros(6)
    sig_expected[0] = 2.0 * g * (deps[0] - dav) + k * 3.0 * dav
    sig_expected[1] = 2.0 * g * (deps[1] - dav) + k * 3.0 * dav
    sig_expected[2] = 2.0 * g * (deps[2] - dav) + k * 3.0 * dav
    sig_expected[3] = g * deps[3]
    assert np.allclose(sig_new, sig_expected, rtol=1e-5, atol=1e-5)

    # Check sound speed matches initial elastic value
    c_expected = np.sqrt((k + (4.0 / 3.0) * g) / p.rho0)
    assert np.isclose(c, c_expected)


def test_pure_elastic_shell():
    """Test 2D shell plane-stress update below yield stress."""
    f1 = FunctTable(1, [0.0, 0.1], [400.0, 500.0])
    p = Law60Params(e0=200000.0, nu=0.3, rho0=7.8e-9, funcs=[f1])
    p = build_law60(p)

    sig_old = np.zeros(3)
    deps = np.array([1e-4, 0.5e-4, 1e-4])
    extra = {"thk": 1.5}
    sig_new, epsp_new, c = shell_update(p, eps=None, deps=deps, sig_old=sig_old, epsp_old=0.0, dt=1e-6, extra=extra)

    assert np.isclose(epsp_new, 0.0)

    # Analytical plane-stress prediction
    a1 = p.e0 / (1.0 - p.nu ** 2)
    a2 = p.nu * a1
    g = p.G0
    sig_expected = np.array([
        a1 * deps[0] + a2 * deps[1],
        a2 * deps[0] + a1 * deps[1],
        g * deps[2],
    ])
    assert np.allclose(sig_new, sig_expected, rtol=1e-5, atol=1e-5)

    # Analytical layer thinning: Delta_eps_zz_el = -nu / (1 - nu) * (deps_xx + deps_yy)
    dezz_el = -p.nu / (1.0 - p.nu) * (deps[0] + deps[1])
    expected_thk = 1.5 * (1.0 + dezz_el)
    assert np.isclose(extra["thk"], expected_thk, rtol=1e-5)

    # Shell sound speed
    c_expected = np.sqrt(a1 / p.rho0)
    assert np.isclose(c, c_expected)


# ============================================================================
# 3. Uniaxial Yield and Radial Return Tests
# ============================================================================

def test_solid_uniaxial_yield_radial_return():
    """Test 3D solid radial return when straining beyond yield."""
    # Yield stress 300 MPa, hardening slope H = 1000 MPa
    f1 = FunctTable(1, [0.0, 0.1], [300.0, 400.0])
    p = Law60Params(e0=200000.0, nu=0.3, rho0=7.8e-9, funcs=[f1])
    p = build_law60(p)

    sig_old = np.zeros(6)
    # Pure shear strain creating trial stress tau_xy_tr = G * gamma_xy
    g = p.G0
    gamma_xy = 0.01  # tau_tr = 76923 * 0.01 = 769.23 MPa, sig_vm_tr = sqrt(3)*tau_tr ~ 1332.3 MPa >> 300 MPa
    deps = np.array([0.0, 0.0, 0.0, gamma_xy, 0.0, 0.0])

    sig_new, epsp_new, _ = solid_update(p, eps=None, deps=deps, sig_old=sig_old, epsp_old=0.0, dt=1e-6)

    # Calculate von Mises of converged stress
    j2 = (
        0.5 * (sig_new[0] ** 2 + sig_new[1] ** 2 + sig_new[2] ** 2)
        + sig_new[3] ** 2 + sig_new[4] ** 2 + sig_new[5] ** 2
    )
    sig_vm_converged = np.sqrt(3.0 * j2)

    # Theoretical plastic strain increment per sigeps60.F:534-545
    sig_vm_trial = np.sqrt(3.0) * (g * gamma_xy)
    sy0 = 300.0
    h = 1000.0
    expected_depsp = (sig_vm_trial - sy0) / (3.0 * g + h)

    assert np.isclose(epsp_new, expected_depsp, rtol=1e-4)
    # OpenRadioss explicit radial return scales by R = YLD(epsp_old)/sig_vm_trial to sy0
    assert np.isclose(sig_vm_converged, sy0, rtol=1e-4)


def test_shell_plane_stress_radial_return():
    """Test 2D shell plane-stress radial projection when yielding."""
    f1 = FunctTable(1, [0.0, 0.1], [250.0, 350.0])
    p = Law60Params(e0=100000.0, nu=0.3, rho0=2.7e-9, funcs=[f1])
    p = build_law60(p)

    sig_old = np.zeros(3)
    # Stretch in x-direction beyond yield: eps_xx = 0.01
    deps = np.array([0.01, 0.0, 0.0])
    extra = {"thk": 2.0}
    sig_new, epsp_new, _ = shell_update(p, eps=None, deps=deps, sig_old=sig_old, epsp_old=0.0, dt=1e-6, extra=extra)

    # Converged plane-stress von Mises
    svm = np.sqrt(sig_new[0] ** 2 + sig_new[1] ** 2 - sig_new[0] * sig_new[1] + 3.0 * sig_new[2] ** 2)

    # Theoretical return per sigeps60c.F:597-609:
    a1 = p.e0 / (1.0 - p.nu ** 2)
    a2 = p.nu * a1
    # Plane-strain constraint in y gives trial stress (a1*deps[0], a2*deps[0], 0)
    sig_vm_tr = np.sqrt(a1 ** 2 + a2 ** 2 - a1 * a2) * deps[0]
    sy0 = 250.0
    h = 1000.0
    g = p.G0
    expected_depsp = (sig_vm_tr - sy0) / (3.0 * g + h)

    assert np.isclose(epsp_new, expected_depsp, rtol=1e-4)
    assert np.isclose(svm, sy0, rtol=1e-4)

    # Check that plastic thinning occurred and thickness reduced
    assert extra["thk"] < 2.0


# ============================================================================
# 4. Multi-Rate Hardening Tests
# ============================================================================

def test_multi_rate_hardening_interpolation():
    """Verify strain rate sensitivity across multiple rate curves."""
    # 4 rate curves at rates 0, 10, 100, 1000 s^-1
    c0 = FunctTable(1, [0.0, 0.1], [300.0, 400.0])
    c1 = FunctTable(2, [0.0, 0.1], [350.0, 450.0])
    c2 = FunctTable(3, [0.0, 0.1], [400.0, 500.0])
    c3 = FunctTable(4, [0.0, 0.1], [450.0, 550.0])

    p = Law60Params(
        e0=210000.0, nu=0.3, rho0=7.8e-9,
        nfunc=4,
        funcs=[c0, c1, c2, c3],
        rate_arr=[0.0, 10.0, 100.0, 1000.0],
        fscale_arr=[1.0, 1.0, 1.0, 1.0],
    )
    p = build_law60(p)

    sig_old = np.zeros(6)
    gamma_xy = 0.005

    # Test rate 1: dt = 5e-3 -> rate ~ 0.58 s^-1 (near curve 0)
    deps = np.array([0.0, 0.0, 0.0, gamma_xy, 0.0, 0.0])
    sig_low, epsp_low, _ = solid_update(p, deps=deps, sig_old=sig_old, dt=5e-3)

    # Test rate 2: dt = 5e-5 -> rate ~ 58 s^-1 (between 10 and 100 s^-1)
    sig_mid, epsp_mid, _ = solid_update(p, deps=deps, sig_old=sig_old, dt=5e-5)

    # Test rate 3: dt = 5e-6 -> rate ~ 577 s^-1 (between 100 and 1000 s^-1)
    sig_high, epsp_high, _ = solid_update(p, deps=deps, sig_old=sig_old, dt=5e-6)

    # Dynamic flow stress increases monotonically with strain rate
    vm_low = np.sqrt(3.0 * sig_low[3] ** 2)
    vm_mid = np.sqrt(3.0 * sig_mid[3] ** 2)
    vm_high = np.sqrt(3.0 * sig_high[3] ** 2)

    assert vm_low < vm_mid < vm_high
    assert 300.0 <= vm_low <= 350.0
    assert 350.0 <= vm_mid <= 420.0
    assert 400.0 <= vm_high <= 500.0


# ============================================================================
# 5. Pressure Sensitivity Function Tests
# ============================================================================

def test_pressure_dependent_yield():
    """Verify pressure scaling function (ipfun) adjusts yield stress."""
    f_yield = FunctTable(1, [0.0, 0.1], [300.0, 300.0])
    # PFAC function: at P*pscale = 0 -> 1.0; at P*pscale = 100 -> 1.5; at P*pscale = 200 -> 2.0
    f_press = FunctTable(2, [0.0, 100.0, 200.0], [1.0, 1.5, 2.0])

    p = Law60Params(
        e0=200000.0, nu=0.3, rho0=7.8e-9,
        funcs=[f_yield],
        ipfun=2,
        pscale=1.0,  # 1 / 1.0 = 1.0
    )
    p = build_law60(p, functs={1: f_yield, 2: f_press})

    # Case A: Zero pressure initial state
    sig_zero_p = np.zeros(6)
    deps = np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0])
    sig_a, _, _ = solid_update(p, deps=deps, sig_old=sig_zero_p, dt=1e-6)
    vm_a = np.sqrt(3.0 * sig_a[3] ** 2)

    # Case B: Initial hydrostatic compression P0 = 100 MPa (so sig_ii = -100)
    sig_comp_p = np.array([-100.0, -100.0, -100.0, 0.0, 0.0, 0.0])
    sig_b, _, _ = solid_update(p, deps=deps, sig_old=sig_comp_p, dt=1e-6)
    vm_b = np.sqrt(
        0.5 * ((sig_b[0] - sig_b[1]) ** 2 + (sig_b[1] - sig_b[2]) ** 2 + (sig_b[2] - sig_b[0]) ** 2)
        + 3.0 * (sig_b[3] ** 2 + sig_b[4] ** 2 + sig_b[5] ** 2)
    )

    # At P0 = 100, PFAC = 1.5 -> Yield stress is 1.5 * 300 = 450 MPa
    assert np.isclose(vm_a, 300.0, rtol=1e-3)
    assert np.isclose(vm_b, 450.0, rtol=1e-3)


# ============================================================================
# 6. Modulus Degradation Tests
# ============================================================================

def test_exponential_modulus_degradation():
    """Verify exponential modulus degradation with plastic strain."""
    f1 = FunctTable(1, [0.0, 0.2], [200.0, 250.0])
    # E0 = 200000, Einf = 50000, ce = 10.0
    p = Law60Params(
        e0=200000.0, nu=0.3, rho0=7.8e-9,
        einf=50000.0, ce=10.0,
        funcs=[f1],
    )
    p = build_law60(p)

    # Initial sound speed at epsp = 0
    c0 = p.sound_speed_solid(epsp=0.0)
    expected_c0 = np.sqrt((p.C1_0 + (4.0 / 3.0) * p.G0) / p.rho0)
    assert np.isclose(c0, expected_c0)

    # After plastic strain epsp = 0.1
    # E_cur = 200000 - (200000 - 50000)*(1 - exp(-10*0.1)) = 200000 - 150000*(1 - e^-1) ~ 105156.4 MPa
    epsp_test = 0.1
    c_deg = p.sound_speed_solid(epsp=epsp_test)
    e_expected = 200000.0 - 150000.0 * (1.0 - np.exp(-1.0))
    g_expected = e_expected / (2.0 * (1.0 + 0.3))
    c1_expected = e_expected / (3.0 * (1.0 - 2.0 * 0.3))
    c_analytical = np.sqrt((c1_expected + (4.0 / 3.0) * g_expected) / p.rho0)

    assert np.isclose(c_deg, c_analytical, rtol=1e-4)
    assert c_deg < c0


def test_curve_modulus_degradation():
    """Verify curve-based dynamic modulus degradation (ifunce)."""
    f1 = FunctTable(1, [0.0, 0.2], [300.0, 300.0])
    # Degradation curve: scale factor drops from 1.0 at epsp=0 to 0.5 at epsp=0.2
    fe = FunctTable(2, [0.0, 0.1, 0.2], [1.0, 0.75, 0.5])

    p = Law60Params(
        e0=100000.0, nu=0.25, rho0=2.5e-9,
        ifunce=2,
        funcs=[f1],
    )
    p = build_law60(p, functs={1: f1, 2: fe})

    c_shell_0 = p.sound_speed_shell(epsp=0.0)
    c_shell_02 = p.sound_speed_shell(epsp=0.2)

    # At epsp = 0.2, E_cur is 0.5 * E0 -> sound speed drops by sqrt(0.5)
    assert np.isclose(c_shell_02, c_shell_0 * np.sqrt(0.5), rtol=1e-4)


# ============================================================================
# 7. Tensile Failure Scaling Tests
# ============================================================================

def test_tensile_failure_scaling():
    """Verify tensile failure start (eps_t1) and rupture (eps_t2) scale down yield stress."""
    f1 = FunctTable(1, [0.0, 0.2], [400.0, 400.0])
    p = Law60Params(
        e0=200000.0, nu=0.3, rho0=7.8e-9,
        eps_t1=0.05, eps_t2=0.15,
        funcs=[f1],
    )
    p = build_law60(p)

    sig_old = np.zeros(6)

    # Case 1: Total strain epst = 0.02 < eps_t1 (no damage, FAIL = 1.0)
    # Using eps parameter to control total strain
    eps_low = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    deps = np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0])
    sig_1, _, _ = solid_update(p, eps=eps_low, deps=deps, sig_old=sig_old, dt=1e-6)
    vm_1 = np.sqrt(3.0 * sig_1[3] ** 2)
    assert np.isclose(vm_1, 400.0, rtol=1e-3)

    # Case 2: Total strain epst = 0.10 midway between 0.05 and 0.15 -> FAIL = (0.15 - 0.10) / (0.15 - 0.05) = 0.5
    # For 3D principal strain with deps_xy = 0.01, eps_xx = 0.08310565 gives epst = 0.10
    eps_mid = np.array([0.08310565, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_2, _, _ = solid_update(p, eps=eps_mid, deps=deps, sig_old=sig_old, dt=1e-6)
    vm_2 = np.sqrt(3.0 * sig_2[3] ** 2)
    assert np.isclose(vm_2, 200.0, rtol=1e-3)  # 0.5 * 400 = 200 MPa

    # Case 3: Total strain epst = 0.16 >= eps_t2 -> FAIL = 0.0
    eps_rupt = np.array([0.16, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_3, _, _ = solid_update(p, eps=eps_rupt, deps=deps, sig_old=sig_old, dt=1e-6)
    vm_3 = np.sqrt(3.0 * sig_3[3] ** 2)
    assert np.isclose(vm_3, 0.0, atol=1e-3)


# ============================================================================
# 8. Element Deletion Tests
# ============================================================================

def test_element_deletion_solid_and_shell():
    """Verify element deletion when plastic strain exceeds eps_max."""
    f1 = FunctTable(1, [0.0, 0.5], [200.0, 300.0])
    p = Law60Params(
        e0=200000.0, nu=0.3, rho0=7.8e-9,
        eps_max=0.08,  # Delete at eps_p >= 0.08
        funcs=[f1],
    )
    p = build_law60(p)

    # Solid deletion
    extra_solid = {"off": np.array([1.0])}
    deps_large = np.array([[0.0, 0.0, 0.0, 0.15, 0.0, 0.0]])
    sig_s, epsp_s, _ = solid_update(p, deps=deps_large, sig_old=np.zeros((1, 6)), epsp_old=0.07, dt=1e-6, extra=extra_solid)
    assert epsp_s[0] >= 0.08
    assert np.allclose(sig_s, 0.0)
    assert extra_solid["off"][0] == 0.0

    # Shell deletion
    extra_shell = {"off": np.array([1.0]), "thk": np.array([1.0])}
    deps_shell = np.array([[0.1, 0.0, 0.0]])
    sig_sh, epsp_sh, _ = shell_update(p, deps=deps_shell, sig_old=np.zeros((1, 3)), epsp_old=0.07, dt=1e-6, extra=extra_shell)
    assert epsp_sh[0] >= 0.08
    assert np.allclose(sig_sh, 0.0)
    assert extra_shell["off"][0] == 0.0


# ============================================================================
# 9. Sound Speed Tests
# ============================================================================

def test_sound_speed_formulas():
    """Verify solid and shell sound speed calculations against analytical formulas."""
    p = Law60Params(e0=210000.0, nu=0.3, rho0=7.85e-9)
    p = build_law60(p)

    c_solid = sound_speed(p, extra={"is_solid": True})
    c_shell = sound_speed(p, extra={"is_shell": True})

    g = 210000.0 / (2.0 * 1.3)
    c1 = 210000.0 / (3.0 * 0.4)
    a1 = 210000.0 / (1.0 - 0.09)

    expected_c_solid = np.sqrt((c1 + (4.0 / 3.0) * g) / 7.85e-9)
    expected_c_shell = np.sqrt(a1 / 7.85e-9)

    assert np.isclose(c_solid, expected_c_solid)
    assert np.isclose(c_shell, expected_c_shell)


# ============================================================================
# 10. Algorithmic Consistent Tangent Verification (Finite Differences)
# ============================================================================

def test_consistent_solid_tangent_elastic():
    """Verify consistent_solid_tangent matches elastic stiffness tensor."""
    p = Law60Params(e0=200000.0, nu=0.3, rho0=7.8e-9)
    p = build_law60(p)

    sig = np.array([10.0, -5.0, 2.0, 15.0, 0.0, 0.0])
    D = consistent_solid_tangent(p, sig=sig, epsp=0.0, deps=np.zeros(1))

    # Numerical perturbation of elastic update
    h = 1e-7
    D_fd = np.zeros((6, 6))
    for j in range(6):
        d_pos = np.zeros(6)
        d_neg = np.zeros(6)
        d_pos[j] = h
        d_neg[j] = -h
        s_pos, _, _ = solid_update(p, deps=d_pos, sig_old=sig, epsp_old=0.0, dt=1e-6)
        s_neg, _, _ = solid_update(p, deps=d_neg, sig_old=sig, epsp_old=0.0, dt=1e-6)
        D_fd[:, j] = (s_pos - s_neg) / (2.0 * h)

    assert np.allclose(D, D_fd, rtol=1e-4, atol=1e-4)


def test_consistent_shell_tangent_elastic():
    """Verify consistent_shell_tangent matches plane-stress elastic stiffness tensor."""
    p = Law60Params(e0=100000.0, nu=0.3, rho0=2.7e-9)
    p = build_law60(p)

    sig = np.array([50.0, -20.0, 10.0])
    D = consistent_shell_tangent(p, sig=sig, epsp=0.0, deps=np.zeros(1))

    h = 1e-7
    D_fd = np.zeros((3, 3))
    for j in range(3):
        d_pos = np.zeros(3)
        d_neg = np.zeros(3)
        d_pos[j] = h
        d_neg[j] = -h
        s_pos, _, _ = shell_update(p, deps=d_pos, sig_old=sig, epsp_old=0.0, dt=1e-6)
        s_neg, _, _ = shell_update(p, deps=d_neg, sig_old=sig, epsp_old=0.0, dt=1e-6)
        D_fd[:, j] = (s_pos - s_neg) / (2.0 * h)

    assert np.allclose(D, D_fd, rtol=1e-4, atol=1e-4)


# ============================================================================
# 11. Multi-Element Vectorization Tests
# ============================================================================

def test_multielement_vectorization_consistency():
    """Verify that batch vectorization produces identical results to single elements."""
    f1 = FunctTable(1, [0.0, 0.05, 0.2], [300.0, 400.0, 500.0])
    p = Law60Params(e0=210000.0, nu=0.3, rho0=7.85e-9, funcs=[f1])
    p = build_law60(p)

    nel = 8
    np.random.seed(42)
    sig_batch = np.random.uniform(-50.0, 50.0, (nel, 6))
    deps_batch = np.random.uniform(-0.005, 0.005, (nel, 6))
    epsp_batch = np.random.uniform(0.0, 0.02, nel)

    # Batch call
    sig_out, epsp_out, c_out = solid_update(
        p, deps=deps_batch, sig_old=sig_batch, epsp_old=epsp_batch, dt=1e-6
    )

    # Individual single-element calls
    for i in range(nel):
        s_single, ep_single, c_single = solid_update(
            p, deps=deps_batch[i], sig_old=sig_batch[i], epsp_old=epsp_batch[i], dt=1e-6
        )
        assert np.allclose(sig_out[i], s_single, rtol=1e-12, atol=1e-12)
        assert np.isclose(epsp_out[i], ep_single, rtol=1e-12, atol=1e-12)
        assert np.isclose(c_out[i], c_single, rtol=1e-12, atol=1e-12)
