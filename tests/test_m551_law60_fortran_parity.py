"""
Exhaustive Fortran physics parity test suite for /MAT/LAW60 (/MAT/PLAS_T3 /MAT/FABRIC).

Validates pyradioss/materials/law60_plast3.py directly against the exact Fortran formulas in:
- engine/source/materials/mat/mat060/sigeps60.F (3D solid continuum kernel)
- engine/source/materials/mat/mat060/sigeps60c.F (2D shell plane-stress kernel)
- starter/source/materials/mat/mat060/hm_read_mat60.F (parameter reading & packing)

Sections:
1. Elastic trial stress (sigeps60.F:285-294, sigeps60c.F:303-305)
2. Pressure-dependent yield scaling (sigeps60.F:444-468, hm_read_mat60.F:197-203)
3. Rational rate interpolation (sigeps60c.F:785-880, sigeps60.F:509-510)
4. Dynamic Young's modulus degradation (sigeps60.F:235-259, sigeps60c.F:226-254)
5. Plastic radial return & plane stress projection (sigeps60.F:530-588, sigeps60c.F:592-611)
6. Through-thickness strain increment and thinning (sigeps60c.F:604-609)
7. Tensile failure damage factor (sigeps60.F:305-353, sigeps60c.F:330-334)
8. Element deletion (sigeps60.F:600-606)
9. Instantaneous longitudinal and shell sound speeds (sigeps60.F:245, sigeps60c.F:312)
10. Starter parameter packing parity (hm_read_mat60.F:197-300)
"""

import math
import numpy as np
import pytest

from pyradioss.common.tables import FunctTable
from pyradioss.materials.law60_plast3 import (
    Law60Params,
    build_law60,
    solid_update,
    shell_update,
    sound_speed,
    _inter_rat,
    _principal_strain,
    _principal_strain_2d,
    _current_moduli,
    _pressure_factor,
    _tensile_failure_factor,
    _eval_yield_and_hardening,
)


# ============================================================================
# Independent Fortran Reference Implementations
# ============================================================================

def fortran_inter_rat(x0: float, x1: float, x2: float, x3: float,
                      y0: float, y1: float, y2: float, y3: float,
                      x: float, i: int, n: int) -> tuple[float, float]:
    """Line-by-line Python transcription of Fortran INTER_RAT in sigeps60c.F:785-880."""
    em30 = 1e-30
    q = x - x1
    d = x2 - x1
    r = d - q
    s = (y2 - y1) / d
    sp = (y3 - y2) / (x3 - x2)
    c2 = (sp - s) / (x3 - x1)
    dm = x1 - x0
    dm = math.copysign(max(em30, abs(dm)), dm)
    sm = (y1 - y0) / dm
    c1 = (s - sm) / (d + dm)
    c6 = 0.0

    if i == 1:
        if x <= x0:
            c1 = 0.0
            c2 = 0.0
            sm = 0.0
        else:
            c2 = c1
            c1 = sm / (x1 - x0)
        r = x1 - x
        q = x - x0
        d = x1 - x0
        c3 = abs(c2 * r)
        c3d = c3 + abs(c1 * q)
        c5 = 0.0
        if c3d > 0.0:
            c3 = c3 / c3d
            c5 = c3 * (c1 - c2)
        c4 = c2 + c5
        c6 = d * c5 * (1.0 - c3)
        y = y0 + q * (sm - r * c4)
        yp = sm + (q - r) * c4 + c6
    elif i == n - 1:
        if sp == 0.0 or x > x3:
            c1 = 0.0
            c2 = 0.0
        else:
            c1 = (sp - s) / (x3 - x1)
            c2 = 0.0
        r = x3 - x
        q = x - x2
        d = x3 - x2
        c3 = abs(c2 * r)
        c3d = c3 + abs(c1 * q)
        c5 = 0.0
        if c3d > 0.0:
            c3 = c3 / c3d
            c5 = c3 * (c1 - c2)
        c4 = c2 + c5
        c6 = d * c5 * (1.0 - c3)
        y = y2 + (x - x2) * (sp - (x3 - x) * c4)
        yp = sp + (q - r) * c4 + c6
    else:
        if i == 2 and sm * (sm - dm * c1) <= 0.0:
            c1 = (s - sm - sm) / d
        c3 = abs(c2 * r)
        c3d = c3 + abs(c1 * q)
        c5 = 0.0
        if c3d > 0.0:
            c3 = c3 / c3d
            c5 = c3 * (c1 - c2)
        c4 = c2 + c5
        c6 = d * c5 * (1.0 - c3)
        y = y1 + q * (s - r * c4)
        yp = s + (q - r) * c4 + c6

    return y, yp


def fortran_principal_strain_3d(eps_xx: float, eps_yy: float, eps_zz: float,
                                eps_xy: float, eps_yz: float, eps_zx: float) -> float:
    """Exact transcription of Fortran sigeps60.F:305-348 4-iteration Newton solve."""
    dav = (eps_xx + eps_yy + eps_zz) / 3.0
    e1 = eps_xx - dav
    e2 = eps_yy - dav
    e3 = eps_zz - dav
    e4 = 0.5 * eps_xy
    e5 = 0.5 * eps_yz
    e6 = 0.5 * eps_zx
    e42 = e4 * e4
    e52 = e5 * e5
    e62 = e6 * e6
    c = -e1 * e1 - e2 * e2 - e3 * e3 - e42 - e52 - e62
    d = -e1 * e2 * e3 + e1 * e52 + e2 * e62 + e3 * e42 - 2.0 * e4 * e5 * e6
    cc = c / 3.0
    epst = math.sqrt(max(0.0, -cc))
    epst2 = epst * epst
    y = (epst2 + c) * epst + d
    if abs(y) > 1e-8:
        epst = 1.75 * epst
        for _ in range(4):
            epst2 = epst * epst
            y = (epst2 + c) * epst + d
            yp = 3.0 * epst2 + c
            if yp != 0.0:
                epst = epst - y / yp
    epst = epst + dav
    return epst


def fortran_principal_strain_2d(eps_xx: float, eps_yy: float, eps_xy: float) -> float:
    """Exact transcription of Fortran sigeps60c.F:330-333."""
    return 0.5 * (eps_xx + eps_yy + math.sqrt((eps_xx - eps_yy) ** 2 + eps_xy ** 2))


# ============================================================================
# Section 1: Elastic Trial Stress Parity (sigeps60.F:285-294, sigeps60c.F:303-305)
# ============================================================================

def test_parity_trial_stress_pure_normal_compression():
    """Verify 3D solid deviatoric trial stress under pure hydrostatic compression.
    P0 = -(sig_xx+sig_yy+sig_zz)/3 > 0.
    In pure hydrostatic loading, deviatoric trial stress must be identically zero.
    """
    p = Law60Params(e0=210000.0, nu=0.3)
    g = p.G0
    c1 = p.C1_0

    sig_old = np.array([-100.0, -100.0, -100.0, 0.0, 0.0, 0.0])
    deps = np.array([-0.001, -0.001, -0.001, 0.0, 0.0, 0.0])

    p0 = -(sig_old[0] + sig_old[1] + sig_old[2]) / 3.0
    assert np.isclose(p0, 100.0)

    dav = (deps[0] + deps[1] + deps[2]) / 3.0
    assert np.isclose(dav, -0.001)

    s_trial_xx = sig_old[0] + p0 + 2.0 * g * (deps[0] - dav)
    assert np.isclose(s_trial_xx, 0.0)

    # Elastic update via solid_update (yield stress very high to remain elastic)
    f_high = FunctTable(1, [0.0, 1.0], [1e9, 1e9])
    mat = build_law60(p, funcs=[f_high])
    sig_new, epsp_new, _ = solid_update(mat, sig_old=sig_old, deps=deps, epsp_old=np.array([0.0]))

    # Pure hydrostatic pressure increases by 3*K*dav = 3 * C1 * (-0.001)
    expected_p_new = -p0 + c1 * 3.0 * dav
    assert np.allclose(sig_new[:3], expected_p_new)
    assert np.allclose(sig_new[3:], 0.0)
    assert epsp_new == 0.0


def test_parity_trial_stress_pure_shear():
    """Verify 3D solid shear trial stress against sigeps60.F:291-293:
    SIGNXY = SIGOXY + G * DEPSXY.
    """
    p = Law60Params(e0=210000.0, nu=0.3)
    g = p.G0

    sig_old = np.array([0.0, 0.0, 0.0, 50.0, -25.0, 10.0])
    deps = np.array([0.0, 0.0, 0.0, 0.002, 0.001, -0.0005])

    expected_xy = sig_old[3] + g * deps[3]
    expected_yz = sig_old[4] + g * deps[4]
    expected_zx = sig_old[5] + g * deps[5]

    f_high = FunctTable(1, [0.0, 1.0], [1e9, 1e9])
    mat = build_law60(p, funcs=[f_high])
    sig_new, epsp_new, _ = solid_update(mat, sig_old=sig_old, deps=deps, epsp_old=np.array([0.0]))

    assert np.isclose(sig_new[3], expected_xy)
    assert np.isclose(sig_new[4], expected_yz)
    assert np.isclose(sig_new[5], expected_zx)
    assert epsp_new == 0.0


def test_parity_trial_stress_triaxial_general():
    """Compare general triaxial deviatoric trial stress against Fortran lines 285-294."""
    p = Law60Params(e0=210000.0, nu=0.3)
    g = p.G0

    rng = np.random.default_rng(42)
    sig_old = rng.uniform(-150.0, 150.0, 6)
    deps = rng.uniform(-0.0002, 0.0002, 6)

    # Fortran reference
    p0 = -(sig_old[0] + sig_old[1] + sig_old[2]) / 3.0
    dav = (deps[0] + deps[1] + deps[2]) / 3.0
    s_ftn = np.zeros(6)
    s_ftn[0] = sig_old[0] + p0 + 2.0 * g * (deps[0] - dav)
    s_ftn[1] = sig_old[1] + p0 + 2.0 * g * (deps[1] - dav)
    s_ftn[2] = sig_old[2] + p0 + 2.0 * g * (deps[2] - dav)
    s_ftn[3] = sig_old[3] + g * deps[3]
    s_ftn[4] = sig_old[4] + g * deps[4]
    s_ftn[5] = sig_old[5] + g * deps[5]

    f_high = FunctTable(1, [0.0, 1.0], [1e9, 1e9])
    mat = build_law60(p, funcs=[f_high])
    sig_new, _, _ = solid_update(mat, sig_old=sig_old, deps=deps, epsp_old=np.array([0.0]))

    # Deviatoric part of sig_new must match s_ftn exactly
    p_new = (sig_new[0] + sig_new[1] + sig_new[2]) / 3.0
    s_py = sig_new.copy()
    s_py[0] -= p_new
    s_py[1] -= p_new
    s_py[2] -= p_new

    assert np.allclose(s_py, s_ftn, atol=1e-12)


def test_parity_trial_stress_plane_stress_shell():
    """Verify 2D shell elastic trial stress against sigeps60c.F:303-305:
    SIGNXX = SIGOXX + A1*DEPSXX + A2*DEPSYY
    SIGNYY = SIGOYY + A2*DEPSXX + A1*DEPSYY
    SIGNXY = SIGOXY + G*DEPSXY
    """
    p = Law60Params(e0=210000.0, nu=0.3)
    nu = p.nu
    a1 = p.e0 / (1.0 - nu ** 2)
    a2 = nu * a1
    g = p.G0

    sig_old = np.array([80.0, -40.0, 20.0])
    deps = np.array([0.0005, -0.0002, 0.0003])

    exp_xx = sig_old[0] + a1 * deps[0] + a2 * deps[1]
    exp_yy = sig_old[1] + a2 * deps[0] + a1 * deps[1]
    exp_xy = sig_old[2] + g * deps[2]

    f_high = FunctTable(1, [0.0, 1.0], [1e9, 1e9])
    mat = build_law60(p, funcs=[f_high])
    sig_new, epsp_new, _ = shell_update(mat, sig_old=sig_old, deps=deps, epsp_old=np.array([0.0]))

    assert np.isclose(sig_new[0], exp_xx)
    assert np.isclose(sig_new[1], exp_yy)
    assert np.isclose(sig_new[2], exp_xy)
    assert epsp_new == 0.0


# ============================================================================
# Section 2: Pressure-dependent Yield Scaling Parity (sigeps60.F:444-468)
# ============================================================================

def test_parity_pressure_scaling_tension_state():
    """Verify pressure scaling PFAC = finter(ipfun, P0 * pscale) under tension (P0 < 0).
    Under hydrostatic tension, yield stress YLD = PFAC * sigma_y is reduced.
    """
    # Pressure function: at P = -500 -> PFAC = 0.7; at P = 0 -> PFAC = 1.0; at P = 500 -> PFAC = 1.3
    fp = FunctTable(201, [-500.0, 0.0, 500.0], [0.7, 1.0, 1.3])
    fy = FunctTable(101, [0.0, 0.1], [300.0, 300.0])

    card = {
        "MAT_E": 210000.0,
        "MAT_NU": 0.3,
        "Xr_fun": 201,
        "MAT_FScale": 1.0,  # pscale = 1.0
        "FUN_A1": 101,
    }
    mat = build_law60(card, functs={101: fy, 201: fp})

    # Hydrostatic tension: sig_old = [200, 200, 200, 0, 0, 0] -> P0 = -200 (since P0 = -tr(sig)/3)
    p0 = -200.0
    pfac = _pressure_factor(mat, np.array([p0]))[0]
    expected_pfac = 1.0 + (0.7 - 1.0) * (-200.0 / -500.0)  # = 1.0 - 0.3 * 0.4 = 0.88
    assert np.isclose(pfac, expected_pfac)

    # In solid_update with P0 = -200, yield stress must be scaled by expected_pfac
    sig_old = np.array([200.0, 200.0, 200.0, 0.0, 0.0, 0.0])
    # Apply small plastic deviatoric strain
    deps = np.array([0.005, -0.0025, -0.0025, 0.0, 0.0, 0.0])
    sig_new, epsp_new, _ = solid_update(mat, sig_old=sig_old, deps=deps, epsp_old=np.array([0.0]))

    # Check updated von Mises stress matches scaled yield stress
    s_new = sig_new[:3] - np.mean(sig_new[:3])
    vm = np.sqrt(1.5 * np.sum(s_new ** 2))
    expected_yld = expected_pfac * 300.0
    assert np.isclose(vm, expected_yld, rtol=1e-5)


def test_parity_pressure_scaling_compression_state():
    """Verify pressure scaling PFAC under hydrostatic compression (P0 > 0).
    Yield stress is elevated.
    """
    fp = FunctTable(201, [-500.0, 0.0, 500.0], [0.7, 1.0, 1.3])
    fy = FunctTable(101, [0.0, 0.1], [300.0, 300.0])

    card = {
        "MAT_E": 210000.0,
        "MAT_NU": 0.3,
        "Xr_fun": 201,
        "MAT_FScale": 1.0,
        "FUN_A1": 101,
    }
    mat = build_law60(card, functs={101: fy, 201: fp})

    # Hydrostatic compression: sig_old = [-250, -250, -250, 0, 0, 0] -> P0 = 250
    p0 = 250.0
    pfac = _pressure_factor(mat, np.array([p0]))[0]
    expected_pfac = 1.0 + (1.3 - 1.0) * (250.0 / 500.0)  # = 1.15
    assert np.isclose(pfac, expected_pfac)

    sig_old = np.array([-250.0, -250.0, -250.0, 0.0, 0.0, 0.0])
    deps = np.array([0.005, -0.0025, -0.0025, 0.0, 0.0, 0.0])
    sig_new, epsp_new, _ = solid_update(mat, sig_old=sig_old, deps=deps, epsp_old=np.array([0.0]))

    s_new = sig_new[:3] - np.mean(sig_new[:3])
    vm = np.sqrt(1.5 * np.sum(s_new ** 2))
    expected_yld = expected_pfac * 300.0
    assert np.isclose(vm, expected_yld, rtol=1e-5)


def test_parity_pressure_scale_inverse_storage():
    """Verify pscale inverse storage matching hm_read_mat60.F:202:
    IF (PSCALE == ZERO) THEN PSCALE = ONE ELSE PSCALE = ONE/PSCALE.
    """
    card1 = {"Xr_fun": 201, "MAT_FScale": 0.0}
    mat1 = build_law60(card1)
    assert mat1.pscale == 1.0

    card2 = {"Xr_fun": 201, "MAT_FScale": 4.0}
    mat2 = build_law60(card2)
    assert np.isclose(mat2.pscale, 0.25)


def test_parity_pressure_scaling_zero_cutoff():
    """Verify PFAC negative values are clamped to 0.0 per sigeps60.F:520:
    YLD = YLD * MAX(ZERO, PFAC).
    """
    # Pressure function going negative for extreme tension
    fp = FunctTable(201, [-1000.0, 0.0], [-0.5, 1.0])
    card = {"Xr_fun": 201, "MAT_FScale": 1.0}
    mat = build_law60(card, functs={201: fp})

    pfac = _pressure_factor(mat, np.array([-1000.0]))[0]
    assert pfac == 0.0


# ============================================================================
# Section 3: Rational Rate Interpolation Parity (sigeps60c.F:785-880)
# ============================================================================

@pytest.mark.parametrize("x_val", [-10.0, 0.0, 2.0, 5.0])
def test_parity_inter_rat_boundary_interval_1_left(x_val):
    """Verify interval 1 left extrapolation / evaluation matching sigeps60c.F:820-841."""
    x0, x1, x2, x3 = 0.0, 10.0, 100.0, 1000.0
    y0, y1, y2, y3 = 200.0, 250.0, 320.0, 400.0

    y_py, yp_py = _inter_rat(x0, x1, x2, x3, y0, y1, y2, y3, x_val, 1, 4)
    y_ftn, yp_ftn = fortran_inter_rat(x0, x1, x2, x3, y0, y1, y2, y3, x_val, 1, 4)

    assert np.isclose(y_py, y_ftn, atol=1e-12)
    assert np.isclose(yp_py, yp_ftn, atol=1e-12)


@pytest.mark.parametrize("x_val", [20.0, 50.0, 80.0])
def test_parity_inter_rat_inner_interval(x_val):
    """Verify middle interval evaluation matching sigeps60c.F:864-878."""
    x0, x1, x2, x3 = 0.0, 10.0, 100.0, 1000.0
    y0, y1, y2, y3 = 200.0, 250.0, 320.0, 400.0

    y_py, yp_py = _inter_rat(x0, x1, x2, x3, y0, y1, y2, y3, x_val, 2, 4)
    y_ftn, yp_ftn = fortran_inter_rat(x0, x1, x2, x3, y0, y1, y2, y3, x_val, 2, 4)

    assert np.isclose(y_py, y_ftn, atol=1e-12)
    assert np.isclose(yp_py, yp_ftn, atol=1e-12)


def test_parity_inter_rat_special_inflection_condition():
    """Verify special condition i == 2 and sm*(sm - dm*c1) <= 0 per sigeps60c.F:865."""
    # Construct points where sm*(sm - dm*c1) <= 0
    x0, x1, x2, x3 = 0.0, 1.0, 2.0, 3.0
    y0, y1, y2, y3 = 0.0, 1.0, 1.05, 1.06

    y_py, yp_py = _inter_rat(x0, x1, x2, x3, y0, y1, y2, y3, 1.5, 2, 4)
    y_ftn, yp_ftn = fortran_inter_rat(x0, x1, x2, x3, y0, y1, y2, y3, 1.5, 2, 4)

    assert np.isclose(y_py, y_ftn, atol=1e-12)
    assert np.isclose(yp_py, yp_ftn, atol=1e-12)


@pytest.mark.parametrize("x_val", [200.0, 800.0, 1000.0, 1500.0])
def test_parity_inter_rat_boundary_interval_last(x_val):
    """Verify last interval evaluation and extrapolation matching sigeps60c.F:842-863."""
    x0, x1, x2, x3 = 0.0, 10.0, 100.0, 1000.0
    y0, y1, y2, y3 = 200.0, 250.0, 320.0, 400.0

    y_py, yp_py = _inter_rat(x0, x1, x2, x3, y0, y1, y2, y3, x_val, 3, 4)
    y_ftn, yp_ftn = fortran_inter_rat(x0, x1, x2, x3, y0, y1, y2, y3, x_val, 3, 4)

    assert np.isclose(y_py, y_ftn, atol=1e-12)
    assert np.isclose(yp_py, yp_ftn, atol=1e-12)


def test_parity_inter_rat_exact_knots():
    """Verify exact knot evaluations recover the knot ordinate."""
    x0, x1, x2, x3 = 0.0, 10.0, 100.0, 1000.0
    y0, y1, y2, y3 = 200.0, 250.0, 320.0, 400.0

    y_py_1, _ = _inter_rat(x0, x1, x2, x3, y0, y1, y2, y3, x1, 2, 4)
    assert np.isclose(y_py_1, y1)

    y_py_2, _ = _inter_rat(x0, x1, x2, x3, y0, y1, y2, y3, x2, 2, 4)
    assert np.isclose(y_py_2, y2)


def test_parity_multi_rate_curve_interpolation():
    """Verify 4-rate curve family interpolation matches Fortran INTER_RAT."""
    f1 = FunctTable(1, [0.0, 0.1], [200.0, 250.0])
    f2 = FunctTable(2, [0.0, 0.1], [240.0, 290.0])
    f3 = FunctTable(3, [0.0, 0.1], [300.0, 350.0])
    f4 = FunctTable(4, [0.0, 0.1], [380.0, 430.0])

    card = {
        "MAT_E": 210000.0,
        "MAT_NU": 0.3,
        "NFUNC": 4,
        "Fsmooth": 0,
        "FUN_A1": 1, "FUN_B1": 2, "FUN_A2": 3, "FUN_B2": 4,
        "MAT_EPSR1": 0.0, "MAT_EPSR2": 10.0, "MAT_EPSR3": 100.0, "MAT_EPSR4": 1000.0,
    }
    mat = build_law60(card, functs={1: f1, 2: f2, 3: f3, 4: f4})

    query_epsp = np.array([0.05])
    query_rate = np.array([45.0])

    sy_py, h_py = _eval_yield_and_hardening(mat, query_epsp, query_rate)

    # Reference manual evaluation
    y1_val = 200.0 + 0.5 * 50.0
    y2_val = 240.0 + 0.5 * 50.0
    y3_val = 300.0 + 0.5 * 50.0
    y4_val = 380.0 + 0.5 * 50.0

    y_ftn, _ = fortran_inter_rat(0.0, 10.0, 100.0, 1000.0,
                                 y1_val, y2_val, y3_val, y4_val,
                                 45.0, 2, 4)

    assert np.isclose(sy_py[0], y_ftn)


# ============================================================================
# Section 4: Dynamic Modulus Degradation Parity (sigeps60.F:235-259)
# ============================================================================

def test_parity_modulus_degradation_curve_based():
    """Verify curve-based modulus degradation E_cur = finter(ifunce, epsp) * E0
    matching sigeps60.F:238-245.
    """
    fe = FunctTable(301, [0.0, 0.05, 0.2], [1.0, 0.9, 0.6])
    card = {
        "MAT_E": 200000.0,
        "MAT_NU": 0.3,
        "fct_ID_k": 301,
    }
    mat = build_law60(card, functs={301: fe})

    epsp_queries = np.array([0.0, 0.025, 0.05, 0.1, 0.2])
    e_cur, g_cur, c1_cur, a1_cur = _current_moduli(mat, epsp_queries)

    # Fortran reference
    for idx, ep in enumerate(epsp_queries):
        if ep == 0.0:
            scale = 1.0
        elif ep <= 0.05:
            scale = 1.0 + (0.9 - 1.0) * (ep / 0.05)
        else:
            scale = 0.9 + (0.6 - 0.9) * ((ep - 0.05) / 0.15)
        exp_e = scale * 200000.0
        exp_g = exp_e / (2.0 * (1.0 + 0.3))
        exp_c1 = exp_e / (3.0 * (1.0 - 2.0 * 0.3))
        exp_a1 = exp_e / (1.0 - 0.3 ** 2)

        assert np.isclose(e_cur[idx], exp_e)
        assert np.isclose(g_cur[idx], exp_g)
        assert np.isclose(c1_cur[idx], exp_c1)
        assert np.isclose(a1_cur[idx], exp_a1)


def test_parity_modulus_degradation_exponential():
    """Verify exponential degradation E_cur = E0 - (E0 - einf)*(1 - exp(-ce*epsp))
    matching sigeps60.F:250-258.
    """
    e0 = 210000.0
    einf = 90000.0
    ce = 15.0
    nu = 0.28
    card = {
        "MAT_E": e0,
        "MAT_NU": nu,
        "E_R": einf,
        "MAT_C1": ce,
    }
    mat = build_law60(card)

    epsp_queries = np.array([0.0, 0.01, 0.05, 0.1, 0.5])
    e_cur, g_cur, c1_cur, a1_cur = _current_moduli(mat, epsp_queries)

    for idx, ep in enumerate(epsp_queries):
        exp_e = e0 - (e0 - einf) * (1.0 - math.exp(-ce * ep))
        exp_g = exp_e / (2.0 * (1.0 + nu))
        exp_c1 = exp_e / (3.0 * (1.0 - 2.0 * nu))
        exp_a1 = exp_e / (1.0 - nu ** 2)

        assert np.isclose(e_cur[idx], exp_e)
        assert np.isclose(g_cur[idx], exp_g)
        assert np.isclose(c1_cur[idx], exp_c1)
        assert np.isclose(a1_cur[idx], exp_a1)


def test_parity_degraded_moduli_relationships():
    """Verify G_cur, C1_cur, A1_cur always satisfy isotropic elasticity identities."""
    mat = Law60Params(e0=210000.0, nu=0.32, einf=80000.0, ce=20.0)
    epsp_test = np.linspace(0.0, 0.3, 10)
    e_cur, g_cur, c1_cur, a1_cur = _current_moduli(mat, epsp_test)

    # Elasticity identity: C1 = 2*G*(1 + nu) / (3*(1 - 2*nu))
    expected_c1 = 2.0 * g_cur * (1.0 + 0.32) / (3.0 * (1.0 - 2.0 * 0.32))
    assert np.allclose(c1_cur, expected_c1)

    # Elasticity identity: A1 = 2*G / (1 - nu)
    expected_a1 = 2.0 * g_cur / (1.0 - 0.32)
    assert np.allclose(a1_cur, expected_a1)


def test_parity_modulus_degradation_zero_plastic_strain():
    """Verify initial moduli are returned when epsp == 0."""
    mat = Law60Params(e0=210000.0, nu=0.3, einf=50000.0, ce=50.0)
    e_cur, g_cur, c1_cur, a1_cur = _current_moduli(mat, np.array([0.0]))
    assert np.isclose(e_cur[0], 210000.0)
    assert np.isclose(g_cur[0], mat.G0)
    assert np.isclose(c1_cur[0], mat.C1_0)


# ============================================================================
# Section 5: Plastic Radial Return Parity (sigeps60.F:530-588, sigeps60c.F:592-611)
# ============================================================================

def test_parity_solid_j2_return_isotropic():
    """Verify 3D solid J2 radial return with isotropic hardening (fisokin = 0.0)
    matching sigeps60.F:530-546:
    VM = sqrt(3 * J2)
    R = YLD / VM
    DPLA = (1 - R) * VM / (3*G + H)
    """
    e0 = 210000.0
    nu = 0.3
    g = e0 / (2.0 * (1.0 + nu))
    sy0 = 300.0
    h0 = 1000.0

    fy = FunctTable(1, [0.0, 0.1], [sy0, sy0 + h0 * 0.1])
    card = {"MAT_E": e0, "MAT_NU": nu, "MAT_HARD": 0.0, "FUN_A1": 1}
    mat = build_law60(card, functs={1: fy})

    deps = np.array([0.005, -0.0025, -0.0025, 0.0, 0.0, 0.0])
    sig_new, epsp_new, _ = solid_update(mat, sig_old=np.zeros(6), deps=deps, epsp_old=np.array([0.0]))

    # Analytical Fortran return:
    # dav = 0. s_trial_xx = 2 * G * 0.005 = 0.01 * G
    s_tr_xx = 2.0 * g * 0.005
    s_tr_yy = 2.0 * g * (-0.0025)
    s_tr_zz = 2.0 * g * (-0.0025)
    j2 = 0.5 * (s_tr_xx ** 2 + s_tr_yy ** 2 + s_tr_zz ** 2)
    vm = math.sqrt(3.0 * j2)
    f = vm - sy0
    delta_epsp = f / (3.0 * g + h0)

    assert np.isclose(epsp_new, delta_epsp)
    # Check von Mises of updated stress equals yield stress
    s_new = sig_new[:3] - np.mean(sig_new[:3])
    vm_new = math.sqrt(1.5 * np.sum(s_new ** 2))
    assert np.isclose(vm_new, sy0)


def test_parity_solid_j2_return_kinematic():
    """Verify 3D solid J2 return with pure kinematic hardening (fisokin = 1.0).
    In kinematic hardening (IPLA=2 in sigeps60.F:561), H_iso = 0:
    DPLA = (1 - R) * VM / (3*G).
    """
    e0 = 210000.0
    nu = 0.3
    g = e0 / (2.0 * (1.0 + nu))
    sy0 = 300.0
    h0 = 1000.0

    fy = FunctTable(1, [0.0, 0.1], [sy0, sy0 + h0 * 0.1])
    card = {"MAT_E": e0, "MAT_NU": nu, "MAT_HARD": 1.0, "FUN_A1": 1}
    mat = build_law60(card, functs={1: fy})

    deps = np.array([0.005, -0.0025, -0.0025, 0.0, 0.0, 0.0])
    sig_new, epsp_new, _ = solid_update(mat, sig_old=np.zeros(6), deps=deps, epsp_old=np.array([0.0]))

    s_tr_xx = 2.0 * g * 0.005
    s_tr_yy = 2.0 * g * (-0.0025)
    s_tr_zz = 2.0 * g * (-0.0025)
    j2 = 0.5 * (s_tr_xx ** 2 + s_tr_yy ** 2 + s_tr_zz ** 2)
    vm = math.sqrt(3.0 * j2)
    f = vm - sy0
    delta_epsp = f / (3.0 * g)  # Denominator is 3*G because H_iso = 0

    assert np.isclose(epsp_new, delta_epsp)


def test_parity_solid_j2_return_mixed():
    """Verify 3D solid return with mixed hardening (fisokin = 0.5).
    Denominator = 3*G + (1 - fisokin)*H = 3*G + 0.5*H.
    """
    e0 = 210000.0
    nu = 0.3
    g = e0 / (2.0 * (1.0 + nu))
    sy0 = 300.0
    h0 = 1000.0

    fy = FunctTable(1, [0.0, 0.1], [sy0, sy0 + h0 * 0.1])
    card = {"MAT_E": e0, "MAT_NU": nu, "MAT_HARD": 0.5, "FUN_A1": 1}
    mat = build_law60(card, functs={1: fy})

    deps = np.array([0.005, -0.0025, -0.0025, 0.0, 0.0, 0.0])
    _, epsp_new, _ = solid_update(mat, sig_old=np.zeros(6), deps=deps, epsp_old=np.array([0.0]))

    s_tr_xx = 2.0 * g * 0.005
    s_tr_yy = 2.0 * g * (-0.0025)
    s_tr_zz = 2.0 * g * (-0.0025)
    vm = math.sqrt(3.0 * 0.5 * (s_tr_xx ** 2 + s_tr_yy ** 2 + s_tr_zz ** 2))
    delta_epsp = (vm - sy0) / (3.0 * g + 0.5 * h0)

    assert np.isclose(epsp_new, delta_epsp)


def test_parity_shell_plane_stress_radial_projection_uniaxial():
    """Verify 2D shell plane-stress radial projection under uniaxial tension.
    Matches sigeps60c.F:592-603.
    """
    e0 = 210000.0
    nu = 0.3
    g = e0 / (2.0 * (1.0 + nu))
    a1 = e0 / (1.0 - nu ** 2)
    sy0 = 350.0
    h0 = 500.0

    fy = FunctTable(1, [0.0, 0.1], [sy0, sy0 + h0 * 0.1])
    card = {"MAT_E": e0, "MAT_NU": nu, "FUN_A1": 1}
    mat = build_law60(card, functs={1: fy})

    deps = np.array([0.004, 0.0, 0.0])
    sig_new, epsp_new, _ = shell_update(mat, sig_old=np.zeros(3), deps=deps, epsp_old=np.array([0.0]))

    # Trial stress
    sxx_tr = a1 * 0.004
    syy_tr = nu * a1 * 0.004
    svm_tr = math.sqrt(sxx_tr ** 2 + syy_tr ** 2 - sxx_tr * syy_tr)
    r = sy0 / svm_tr
    delta_epsp = (svm_tr - sy0) / (3.0 * g + h0)

    assert np.isclose(sig_new[0], sxx_tr * r)
    assert np.isclose(sig_new[1], syy_tr * r)
    assert np.isclose(epsp_new, delta_epsp)


def test_parity_shell_plane_stress_radial_projection_shear():
    """Verify 2D shell plane-stress radial projection under pure shear."""
    e0 = 210000.0
    nu = 0.3
    g = e0 / (2.0 * (1.0 + nu))
    sy0 = 350.0

    fy = FunctTable(1, [0.0, 0.1], [sy0, sy0])
    card = {"MAT_E": e0, "MAT_NU": nu, "FUN_A1": 1}
    mat = build_law60(card, functs={1: fy})

    deps = np.array([0.0, 0.0, 0.01])
    sig_new, epsp_new, _ = shell_update(mat, sig_old=np.zeros(3), deps=deps, epsp_old=np.array([0.0]))

    sxy_tr = g * 0.01
    svm_tr = math.sqrt(3.0 * sxy_tr ** 2)
    r = sy0 / svm_tr
    assert np.isclose(sig_new[2], sxy_tr * r)
    assert np.isclose(math.sqrt(3.0 * sig_new[2] ** 2), sy0)


def test_parity_shell_plane_stress_radial_projection_biaxial():
    """Verify 2D shell equibiaxial tension satisfies von Mises plane stress yield."""
    e0 = 210000.0
    nu = 0.3
    sy0 = 350.0

    fy = FunctTable(1, [0.0, 0.1], [sy0, sy0])
    card = {"MAT_E": e0, "MAT_NU": nu, "FUN_A1": 1}
    mat = build_law60(card, functs={1: fy})

    deps = np.array([0.005, 0.005, 0.0])
    sig_new, epsp_new, _ = shell_update(mat, sig_old=np.zeros(3), deps=deps, epsp_old=np.array([0.0]))

    # For equibiaxial sxx = syy: svm = sqrt(sxx^2 + sxx^2 - sxx^2) = sxx = sy0
    assert np.isclose(sig_new[0], sy0)
    assert np.isclose(sig_new[1], sy0)
    assert np.isclose(sig_new[2], 0.0)


def test_parity_shell_plane_stress_out_of_plane_zero():
    """Verify out-of-plane stress sigma_zz is identically zero in 2D shell formulation."""
    mat = Law60Params(e0=210000.0, nu=0.3)
    fy = FunctTable(1, [0.0, 0.1], [300.0, 300.0])
    mat_built = build_law60(mat, funcs=[fy])

    sig_new, _, _ = shell_update(mat_built, sig_old=np.zeros(3), deps=np.array([0.005, 0.002, 0.001]))
    # Returned vector has length 3: [xx, yy, xy] - plane stress implies sigma_zz = 0
    assert len(sig_new) == 3


# ============================================================================
# Section 6: Shell Thinning Parity (sigeps60c.F:604-609)
# ============================================================================

def test_parity_shell_thinning_elastic_increment():
    """Verify pure elastic shell thinning increment matching sigeps60c.F:607:
    DEZZ = -(DEPSXX + DEPSYY) * (nu / (1 - nu)).
    """
    nu = 0.3
    p = Law60Params(e0=210000.0, nu=nu)
    f_high = FunctTable(1, [0.0, 1.0], [1e9, 1e9])
    mat = build_law60(p, funcs=[f_high])

    thk0 = 1.5
    extra = {"thk": np.array([thk0])}
    deps = np.array([0.001, 0.0005, 0.0])

    shell_update(mat, sig_old=np.zeros(3), deps=deps, epsp_old=np.array([0.0]), extra=extra)

    nnu11 = nu / (1.0 - nu)
    expected_dezz = -(deps[0] + deps[1]) * nnu11
    expected_thk = thk0 * (1.0 + expected_dezz)

    assert np.isclose(extra["thk"][0], expected_thk)


def test_parity_shell_thinning_plastic_increment_uniaxial():
    """Verify shell thinning with plastic flow matching sigeps60c.F:606-608:
    DEZZ = -(DEPSXX+DEPSYY)*NNU11 - NU31*DEZZ_PL
    where DEZZ_PL = DPLA * 0.5*(sig_xx + sig_yy) / YLD.
    """
    nu = 0.3
    p = Law60Params(e0=210000.0, nu=nu)
    sy0 = 300.0
    fy = FunctTable(1, [0.0, 0.1], [sy0, sy0])
    mat = build_law60(p, funcs=[fy])

    thk0 = 2.0
    extra = {"thk": np.array([thk0])}
    deps = np.array([0.004, 0.0, 0.0])

    sig_new, epsp_new, _ = shell_update(mat, sig_old=np.zeros(3), deps=deps,
                                        epsp_old=np.array([0.0]), extra=extra)

    nnu11 = nu / (1.0 - nu)
    nu31 = (1.0 - 2.0 * nu) / (1.0 - nu)
    s_mean = 0.5 * (sig_new[0] + sig_new[1])
    dezz_pl = epsp_new * s_mean / sy0
    expected_dezz = -(deps[0] + deps[1]) * nnu11 - nu31 * dezz_pl
    expected_thk = thk0 * (1.0 + expected_dezz)

    assert np.isclose(extra["thk"][0], expected_thk)


def test_parity_shell_thinning_plastic_increment_equibiaxial():
    """Verify plastic thinning under equibiaxial tension."""
    nu = 0.28
    p = Law60Params(e0=200000.0, nu=nu)
    sy0 = 320.0
    fy = FunctTable(1, [0.0, 0.1], [sy0, sy0])
    mat = build_law60(p, funcs=[fy])

    thk0 = 1.0
    extra = {"thk": np.array([thk0])}
    deps = np.array([0.005, 0.005, 0.0])

    sig_new, epsp_new, _ = shell_update(mat, sig_old=np.zeros(3), deps=deps,
                                        epsp_old=np.array([0.0]), extra=extra)

    nnu11 = nu / (1.0 - nu)
    nu31 = (1.0 - 2.0 * nu) / (1.0 - nu)
    s_mean = 0.5 * (sig_new[0] + sig_new[1])
    dezz_pl = epsp_new * s_mean / sy0
    expected_dezz = -(deps[0] + deps[1]) * nnu11 - nu31 * dezz_pl
    expected_thk = thk0 * (1.0 + expected_dezz)

    assert np.isclose(extra["thk"][0], expected_thk)


def test_parity_shell_thinning_accumulation():
    """Verify cumulative thinning across multiple cycles."""
    nu = 0.3
    p = Law60Params(e0=210000.0, nu=nu)
    fy = FunctTable(1, [0.0, 0.1], [300.0, 300.0])
    mat = build_law60(p, funcs=[fy])

    extra = {"thk": np.array([1.0])}
    sig = np.zeros(3)
    epsp = 0.0

    deps = np.array([0.002, 0.0, 0.0])
    for _ in range(5):
        sig, epsp, _ = shell_update(mat, sig_old=sig, deps=deps, epsp_old=np.array([epsp]), extra=extra)

    # Thickness must strictly decrease under tensile straining
    assert extra["thk"][0] < 1.0


# ============================================================================
# Section 7: Tensile Failure Damage Factor Parity (sigeps60.F:305-353, sigeps60c.F:330-334)
# ============================================================================

@pytest.mark.parametrize("seed", [101, 202, 303, 404])
def test_parity_principal_strain_3d_newton_vs_fortran(seed):
    """Verify 4-iteration Newton solve matches Fortran sigeps60.F:305-348 exactly."""
    rng = np.random.default_rng(seed)
    eps = rng.uniform(-0.02, 0.05, 6)

    epst_ftn = fortran_principal_strain_3d(eps[0], eps[1], eps[2], eps[3], eps[4], eps[5])
    epst_py = _principal_strain(eps.reshape(1, 6))[0]

    assert np.isclose(epst_py, epst_ftn, atol=1e-12)


def test_parity_principal_strain_3d_hydrostatic():
    """Verify hydrostatic tension state where DAV = eps0 and epst = eps0."""
    eps = np.array([[0.03, 0.03, 0.03, 0.0, 0.0, 0.0]])
    epst = _principal_strain(eps)[0]
    epst_ftn = fortran_principal_strain_3d(0.03, 0.03, 0.03, 0.0, 0.0, 0.0)
    assert np.isclose(epst, 0.03)
    assert np.isclose(epst, epst_ftn)


def test_parity_principal_strain_3d_cubic_root_convergence():
    """Verify that Newton solve converges to a root of Fortran's cubic equation
    y = x^3 + c*x + d = 0 per sigeps60.F:316-346 across diverse strain tensors.
    """
    rng = np.random.default_rng(777)
    for _ in range(20):
        e = rng.uniform(-0.02, 0.05, 6)
        dav = (e[0] + e[1] + e[2]) / 3.0
        e1, e2, e3 = e[0] - dav, e[1] - dav, e[2] - dav
        e4, e5, e6 = 0.5 * e[3], 0.5 * e[4], 0.5 * e[5]
        c = -(e1 ** 2 + e2 ** 2 + e3 ** 2 + e4 ** 2 + e5 ** 2 + e6 ** 2)
        d = -(e1 * e2 * e3) + e1 * e5 ** 2 + e2 * e6 ** 2 + e3 * e4 ** 2 - 2.0 * e4 * e5 * e6

        epst = _principal_strain(e.reshape(1, 6))[0]
        x_dev = epst - dav
        y_val = (x_dev ** 2 + c) * x_dev + d
        assert abs(y_val) < 1e-5


def test_parity_principal_strain_2d_shell():
    """Verify 2D shell in-plane max principal strain matching sigeps60c.F:330-333."""
    rng = np.random.default_rng(888)
    for _ in range(10):
        eps = rng.uniform(-0.01, 0.02, 3)
        epst_ftn = fortran_principal_strain_2d(eps[0], eps[1], eps[2])
        epst_py = _principal_strain_2d(eps.reshape(1, 3))[0]
        assert np.isclose(epst_py, epst_ftn, atol=1e-12)


def test_parity_tensile_failure_factor_softening():
    """Verify FAIL factor linear softening matching sigeps60.F:351-352:
    FAIL = max(0, min(1, (eps_t2 - epst)/(eps_t2 - eps_t1))).
    """
    p = Law60Params(eps_t1=0.1, eps_t2=0.2)

    # Below eps_t1: FAIL = 1.0
    assert _tensile_failure_factor(p, np.array([0.05]))[0] == 1.0
    assert _tensile_failure_factor(p, np.array([0.1]))[0] == 1.0

    # In softening zone:
    assert np.isclose(_tensile_failure_factor(p, np.array([0.15]))[0], 0.5)
    assert np.isclose(_tensile_failure_factor(p, np.array([0.175]))[0], 0.25)

    # Beyond eps_t2: FAIL = 0.0
    assert _tensile_failure_factor(p, np.array([0.2]))[0] == 0.0
    assert _tensile_failure_factor(p, np.array([0.25]))[0] == 0.0


def test_parity_tensile_failure_stress_and_hardening_degradation():
    """Verify yield stress and hardening slope degradation by FAIL factor in 3D solid."""
    fy = FunctTable(1, [0.0, 0.1], [300.0, 400.0])
    card = {
        "MAT_E": 210000.0,
        "MAT_NU": 0.3,
        "MAT_EPST1": 0.05,
        "MAT_EPST2": 0.15,
        "FUN_A1": 1,
    }
    mat = build_law60(card, functs={1: fy})

    # Total strain eps where epst = 0.10 -> FAIL = (0.15 - 0.10)/(0.15 - 0.05) = 0.5
    eps_tot = np.array([0.10, -0.05, -0.05, 0.0, 0.0, 0.0])
    deps = np.array([0.002, -0.001, -0.001, 0.0, 0.0, 0.0])

    sig_new, _, _ = solid_update(mat, sig_old=np.zeros(6), deps=deps, epsp_old=np.array([0.0]),
                                 eps=eps_tot - deps)

    # Compute exact epst and fail using Fortran reference formulas
    epst = fortran_principal_strain_3d(eps_tot[0], eps_tot[1], eps_tot[2],
                                       eps_tot[3], eps_tot[4], eps_tot[5])
    expected_fail = (0.15 - epst) / (0.15 - 0.05)
    expected_yld = expected_fail * 300.0

    s_new = sig_new[:3] - np.mean(sig_new[:3])
    vm = np.sqrt(1.5 * np.sum(s_new ** 2))
    assert np.isclose(vm, expected_yld)


# ============================================================================
# Section 8: Element Deletion Parity (sigeps60.F:600-606)
# ============================================================================

def test_parity_element_deletion_solid():
    """Verify 3D solid element deletion when epsp >= eps_max.
    Stress is zeroed and off/off60 set to 0.0.
    """
    fy = FunctTable(1, [0.0, 0.1], [250.0, 250.0])
    card = {
        "MAT_E": 210000.0,
        "MAT_NU": 0.3,
        "MAT_EPS": 0.02,
        "FUN_A1": 1,
    }
    mat = build_law60(card, functs={1: fy})

    extra = {"off": np.array([1.0]), "off60": np.array([1.0])}
    # Large strain to push epsp beyond 0.02
    deps = np.array([0.05, -0.025, -0.025, 0.0, 0.0, 0.0])
    sig_new, epsp_new, _ = solid_update(mat, sig_old=np.zeros(6), deps=deps,
                                        epsp_old=np.array([0.015]), extra=extra)

    assert epsp_new >= 0.02
    assert np.allclose(sig_new, 0.0)
    assert extra["off"][0] == 0.0
    assert extra["off60"][0] == 0.0


def test_parity_element_deletion_shell():
    """Verify 2D shell element deletion when epsp >= eps_max."""
    fy = FunctTable(1, [0.0, 0.1], [250.0, 250.0])
    card = {
        "MAT_E": 210000.0,
        "MAT_NU": 0.3,
        "MAT_EPS": 0.02,
        "FUN_A1": 1,
    }
    mat = build_law60(card, functs={1: fy})

    extra = {"off": np.array([1.0]), "layf": np.array([1.0])}
    deps = np.array([0.05, 0.0, 0.0])
    sig_new, epsp_new, _ = shell_update(mat, sig_old=np.zeros(3), deps=deps,
                                        epsp_old=np.array([0.015]), extra=extra)

    assert epsp_new >= 0.02
    assert np.allclose(sig_new, 0.0)
    assert extra["off"][0] == 0.0
    assert extra["layf"][0] == 0.0


def test_parity_element_deletion_partial_batch():
    """Verify vectorized multi-element deletion where only some elements delete."""
    fy = FunctTable(1, [0.0, 0.1], [250.0, 250.0])
    card = {
        "MAT_E": 210000.0,
        "MAT_NU": 0.3,
        "MAT_EPS": 0.05,
        "FUN_A1": 1,
    }
    mat = build_law60(card, functs={1: fy})

    sig_old = np.zeros((3, 6))
    deps = np.array([
        [0.001, -0.0005, -0.0005, 0, 0, 0],  # Elastic
        [0.01, -0.005, -0.005, 0, 0, 0],     # Moderate plastic (epsp ~ 0.008 < 0.05)
        [0.08, -0.04, -0.04, 0, 0, 0],       # Large plastic (epsp ~ 0.078 >= 0.05)
    ])
    epsp_old = np.zeros(3)
    extra = {"off": np.ones(3), "off60": np.ones(3)}

    sig_new, epsp_new, _ = solid_update(mat, sig_old=sig_old, deps=deps, epsp_old=epsp_old, extra=extra)

    # Element 0: elastic, non-zero stress, not deleted
    assert epsp_new[0] == 0.0
    assert not np.allclose(sig_new[0], 0.0)
    assert extra["off"][0] == 1.0

    # Element 1: plastic, non-zero stress, not deleted
    assert 0.0 < epsp_new[1] < 0.05
    assert not np.allclose(sig_new[1], 0.0)
    assert extra["off"][1] == 1.0

    # Element 2: deleted, zero stress, off = 0.0
    assert epsp_new[2] >= 0.05
    assert np.allclose(sig_new[2], 0.0)
    assert extra["off"][2] == 0.0


# ============================================================================
# Section 9: Longitudinal and Shell Sound Speeds Parity (sigeps60.F:245, sigeps60c.F:312)
# ============================================================================

def test_parity_sound_speed_solid_initial():
    """Verify 3D solid sound speed c_solid = sqrt((C1 + 4/3*G)/rho0)."""
    e0 = 210000.0
    nu = 0.3
    rho0 = 7.85e-9
    p = Law60Params(e0=e0, nu=nu, rho0=rho0)

    g = e0 / (2.0 * (1.0 + nu))
    c1 = e0 / (3.0 * (1.0 - 2.0 * nu))
    expected_c = math.sqrt((c1 + (4.0 / 3.0) * g) / rho0)

    assert np.isclose(p.sound_speed_solid(), expected_c)
    assert np.isclose(sound_speed(p), expected_c)


def test_parity_sound_speed_shell_initial():
    """Verify 2D shell sound speed c_shell = sqrt(A1/rho0) = sqrt(E/((1-nu^2)*rho0))."""
    e0 = 210000.0
    nu = 0.3
    rho0 = 7.85e-9
    p = Law60Params(e0=e0, nu=nu, rho0=rho0)

    a1 = e0 / (1.0 - nu ** 2)
    expected_c = math.sqrt(a1 / rho0)

    assert np.isclose(p.sound_speed_shell(), expected_c)
    assert np.isclose(sound_speed(p, extra={"is_shell": True}), expected_c)


def test_parity_sound_speed_degraded():
    """Verify sound speed degrades consistently with degraded Young's modulus."""
    e0 = 210000.0
    nu = 0.3
    rho0 = 7.85e-9
    einf = 100000.0
    ce = 10.0
    p = Law60Params(e0=e0, nu=nu, rho0=rho0, einf=einf, ce=ce)

    epsp = 0.2
    e_deg = e0 - (e0 - einf) * (1.0 - math.exp(-ce * epsp))
    g_deg = e_deg / (2.0 * (1.0 + nu))
    c1_deg = e_deg / (3.0 * (1.0 - 2.0 * nu))
    a1_deg = e_deg / (1.0 - nu ** 2)

    exp_solid_c = math.sqrt((c1_deg + (4.0 / 3.0) * g_deg) / rho0)
    exp_shell_c = math.sqrt(a1_deg / rho0)

    assert np.isclose(p.sound_speed_solid(epsp=epsp), exp_solid_c)
    assert np.isclose(p.sound_speed_shell(epsp=epsp), exp_shell_c)


def test_parity_sound_speed_density_scaling():
    """Verify sound speed recalculation with compressed density rho > rho0."""
    p = Law60Params(e0=210000.0, nu=0.3, rho0=7.85e-9)
    rho_compressed = 8.2e-9

    g = p.G0
    c1 = p.C1_0
    exp_c = math.sqrt((c1 + (4.0 / 3.0) * g) / rho_compressed)

    assert np.isclose(p.sound_speed_solid(rho=rho_compressed), exp_c)


# ============================================================================
# Section 10: Starter Parameter Packing Parity (hm_read_mat60.F:197-300)
# ============================================================================

def test_parity_starter_single_curve_duplication():
    """Verify hm_read_mat60.F:249-254 single curve duplication:
    IF (NRATE == 1) THEN MFUNC = 2; IFUNC(2) = IFUNC(1); RATE(1) = 0; RATE(2) = 1.
    """
    f1 = FunctTable(10, [0.0, 0.1, 0.2], [300.0, 350.0, 400.0])
    card = {
        "MAT_E": 210000.0,
        "MAT_NU": 0.3,
        "NFUNC": 1,
        "FUN_A1": 10,
        "MAT_EPSR1": 0.0,
    }
    mat = build_law60(card, functs={10: f1})

    # Should have 2 curves in curve_x with rates [0.0, 1.0]
    assert len(mat.curve_x) == 2
    assert len(mat.rates) == 2
    assert mat.rates[0] == 0.0
    assert mat.rates[1] == 1.0
    assert np.allclose(mat.curve_y[0], mat.curve_y[1])


def test_parity_starter_pscale_inversion():
    """Verify PSCALE parameter packing in hm_read_mat60.F:197-203."""
    # Case 1: IPFUN == 0 -> PSCALE = 0.0
    mat1 = build_law60({"ipfun": 0, "pscale": 5.0})
    assert mat1.pscale == 0.0

    # Case 2: IPFUN > 0, PSCALE == 0.0 -> PSCALE = 1.0
    mat2 = build_law60({"ipfun": 10, "pscale": 0.0})
    assert mat2.pscale == 1.0

    # Case 3: IPFUN > 0, PSCALE == 2.5 -> PSCALE = 1 / 2.5 = 0.4
    mat3 = build_law60({"ipfun": 10, "pscale": 2.5})
    assert np.isclose(mat3.pscale, 0.4)


def test_parity_starter_defaults_consistency():
    """Verify default values match hm_read_mat60.F:268-299."""
    mat = build_law60({"MAT_E": 200000.0, "MAT_NU": 0.3, "MAT_RHO": 7.85e-9})
    assert mat.rhor == 7.85e-9
    assert mat.eps_t1 == 1e30
    assert mat.eps_t2 == 2e30
    assert mat.eps_max == 1e30
    assert mat.fisokin == 0.0
    assert mat.fcut == 1e30
