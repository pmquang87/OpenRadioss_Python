"""
tests/test_m553_law58_fortran_parity.py

Milestone M553: Fortran Parity Auditor for /MAT/LAW58 (/MAT/FABR_A, /MAT/FABRIC_A).
Comprehensive direct numerical comparisons and oracle validations against upstream OpenRadioss Fortran:
- engine/source/materials/mat/mat058/sigeps58c.F
- starter/source/materials/mat/mat058/hm_read_mat58.F
- starter/source/materials/mat/mat058/cm58in3.F
- starter/source/materials/mat/mat058/law58_upd.F
"""

import math
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

from pyradioss.materials.law58_fabr_a import (
    Law58Params,
    FabricAMaterial,
    crimp_interchange,
    shell_update_law58,
    sound_speed_shell_law58,
    shell_membrane_tangent,
    tangent_law58_shell,
    init_material_axes_cm58,
    find_curve_intersection,
    build_law58,
)


class MockCurve:
    """Mock piecewise-linear curve mimicking Radioss /FUNCT."""

    def __init__(self, points: List[Tuple[float, float]]):
        self.data = np.array(points, dtype=float)
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]

    def __call__(self, val: float) -> Tuple[float, float]:
        x_clamped = max(float(self.x[0]), min(float(self.x[-1]), float(val)))
        y_val = float(np.interp(x_clamped, self.x, self.y))
        idx = np.searchsorted(self.x, x_clamped)
        if idx == 0:
            dydx = (self.y[1] - self.y[0]) / (self.x[1] - self.x[0])
        elif idx >= len(self.x):
            dydx = (self.y[-1] - self.y[-2]) / (self.x[-1] - self.x[-2])
        else:
            dydx = (self.y[idx] - self.y[idx - 1]) / (self.x[idx] - self.x[idx - 1])
        return y_val, float(dydx)


# ============================================================================
# 1. Initial Crimp Geometry & cm58in3.F Material Axes / Pre-shear Initializations
# ============================================================================

def test_initial_crimp_geometry_formulas():
    """Verify initial crimp geometry matches hm_read_mat58.F lines 208-242:
        L_c0 = 1 / N_t,  L_t0 = 1 / N_c
        D_c0 = L_c0 * (1 + S_1),  D_t0 = L_t0 * (1 + S_2)
        H_c0 = sqrt(D_c0^2 - L_c0^2),  H_t0 = sqrt(D_t0^2 - L_t0^2)
        K_c = E_1 / N_c,  K_t = E_2 / N_t
        K_fc = flex1 * K_c * H_c0 / D_c0,  K_ft = flex2 * K_t * H_t0 / D_t0
    """
    n1, n2 = 3, 5
    s1, s2 = 0.08, 0.12
    e1, e2 = 15000.0, 12000.0
    b1, b2 = 500.0, 400.0
    flex1, flex2 = 0.005, 0.008

    p = Law58Params(
        e1=e1, b1=b1, e2=e2, b2=b2,
        n1=n1, n2=n2, s1=s1, s2=s2,
        c4=flex1, c5=flex2,
    )

    # Unit cell lengths: L_c0 = 1 / N_t (warp length between weft yarns)
    #                   L_t0 = 1 / N_c (weft length between warp yarns)
    assert p.nc == n1
    assert p.nt == n2
    assert p.lc0 == pytest.approx(1.0 / n2)
    assert p.lt0 == pytest.approx(1.0 / n1)

    # Elongated fiber lengths with crimp stretch S_1, S_2
    dc0_expected = (1.0 / n2) * (1.0 + s1)
    dt0_expected = (1.0 / n1) * (1.0 + s2)
    assert p.dc0 == pytest.approx(dc0_expected)
    assert p.dt0 == pytest.approx(dt0_expected)

    # Out-of-plane crimp wave heights (amplitudes)
    hc0_expected = math.sqrt(dc0_expected**2 - (1.0 / n2)**2)
    ht0_expected = math.sqrt(dt0_expected**2 - (1.0 / n1)**2)
    assert p.hc0 == pytest.approx(hc0_expected)
    assert p.ht0 == pytest.approx(ht0_expected)

    # Axial fiber stiffnesses per yarn
    kc_expected = e1 / n1
    kt_expected = e2 / n2
    assert p.kc == pytest.approx(kc_expected)
    assert p.kt == pytest.approx(kt_expected)

    # Crimp flexural stiffnesses K_fc, K_ft
    kfc_expected = flex1 * kc_expected * hc0_expected / dc0_expected
    kft_expected = flex2 * kt_expected * ht0_expected / dt0_expected
    assert p.kfc == pytest.approx(kfc_expected)
    assert p.kft == pytest.approx(kft_expected)


def test_initial_crimp_defaults_matching_openradioss():
    """Verify default values match OpenRadioss param_c.inc (EM01=0.1, EM03=0.001):
        if S1 == 0 -> S1 = 0.1
        if S2 == 0 -> S2 = 0.1
        if flex == 0 -> flex = 0.001
        if flex1 == 0 and flex2 == 0 -> flex1 = flex2 = flex
    """
    p = Law58Params(e1=1000.0, e2=1000.0, s1=0.0, s2=0.0, flex=0.0, c4=0.0, c5=0.0)
    assert p.embc == 0.1
    assert p.embt == 0.1
    assert p.flex1 == 0.001
    assert p.flex2 == 0.001
    assert p.nc == 1
    assert p.nt == 1


def test_cm58in3_material_axes_orthogonal_and_presheared():
    """Verify material axes initialization and Trellis angle calculation matching cm58in3.F:
        TANA = (R1*R2 + S1*S2) / (R1*S2 - R2*S1)
        UVAR(6) = TANA
        UVAR(10) = TANA * G0  (SIG0 prestress offset)
        UVAR(14) = ALDT
        UVAR(40) = 1.0
    """
    p = Law58Params(e1=2000.0, e2=2000.0, g0=150.0)

    # Case A: Orthogonal yarns (DIR1 = [1, 0], DIR2 = [0, 1]) -> TANA = 0, SIG0 = 0
    res_ortho = init_material_axes_cm58(p, dir1=(1.0, 0.0), dir2=(0.0, 1.0), aldt=2.5)
    assert res_ortho["tana"] == pytest.approx(0.0)
    assert res_ortho["sig0"] == pytest.approx(0.0)
    assert res_ortho["aldt"] == 2.5
    assert res_ortho["active"] == 1.0
    assert res_ortho["crimp_thickness"] == pytest.approx(p.hc0 + p.ht0)

    # Case B: Pre-sheared yarns at 60 degrees (DIR1 along X, DIR2 at 60 deg)
    # v1 = [1, 0], v2 = [cos(60 deg), sin(60 deg)] = [0.5, sqrt(3)/2]
    # R1*R2 + S1*S2 = 0.5,  R1*S2 - R2*S1 = sqrt(3)/2
    # TANA = 0.5 / (sqrt(3)/2) = 1 / sqrt(3) = cot(60 deg) = tan(30 deg)
    theta = math.radians(60.0)
    res_shear = init_material_axes_cm58(p, dir1=(1.0, 0.0), dir2=(math.cos(theta), math.sin(theta)))
    expected_tana = 1.0 / math.tan(theta)
    assert res_shear["tana"] == pytest.approx(expected_tana)
    expected_sig0 = expected_tana * p.g0
    assert res_shear["sig0"] == pytest.approx(expected_sig0)

    # Verify that shell_update_law58 offsets initial pre-shear so initial stress is zero
    sig_init = np.zeros((1, 3))
    deps_zero = np.zeros((1, 3))
    extra = {
        "sig0": np.array([expected_sig0]),
        "tan_phi": np.array([expected_tana]),
    }
    sig_upd, _ = shell_update_law58(p, sig_init, deps_zero, dt=1e-5, extra=extra)
    # sigma_xy = G0 * tan_phi - sig0 = G0 * expected_tana - expected_sig0 = 0
    assert sig_upd[0, 2] == pytest.approx(0.0, abs=1e-12)


# ============================================================================
# 2. Crimp Interchange Iteration (sigeps58c.F:340-465)
# ============================================================================

def test_crimp_interchange_uncoupled_mode():
    """Verify uncoupled crimp deflection (sigeps58c.F:343-395):
    When warp is in tension (e_c > 0) and weft is in compression (e_t < 0):
        y_c < 0 (warp straightens), y_t > 0 (weft crinkles)
        y_c + y_t >= 0 -> no contact! F_n = 0.
    """
    p = Law58Params(e1=5000.0, e2=5000.0, flex=0.01, n1=1, n2=1, s1=0.1, s2=0.1)

    # Warp stretched by 0.5%, weft compressed by 4%
    ec = 0.005
    et = -0.04
    lc = p.lc0 * (1.0 + ec)
    lt = p.lt0 * (1.0 + et)

    yc, yt, fc, ft, fn, dc, dt = crimp_interchange(p, lc, lt, niter=5)

    # Warp yarn pulled -> wants to flatten -> yc < 0
    assert yc < 0.0
    # Weft yarn compressed -> wants to bulge -> yt > 0
    assert yt > 0.0
    # Uncoupled condition: yc + yt >= 0 -> no contact force
    assert yc + yt >= 0.0
    assert fn == pytest.approx(0.0)

    # Verify equilibrium of uncoupled Newton iteration:
    # f(y_c) = K_fc * y_c + F_c * (H_c / D_c) == 0
    hc = p.hc0 + yc
    dc_calc = math.sqrt(lc**2 + hc**2)
    dcc = dc_calc - p.dc0
    fc_calc = (p.kc - 0.5 * p.kbc * dcc) * dcc
    res_c = p.kfc * yc + fc_calc * (hc / dc_calc)
    assert abs(res_c) < 1e-4


def test_crimp_interchange_coupled_mode_and_contact_force():
    """Verify coupled crimp interchange (sigeps58c.F:399-460):
    When both yarns are in tension (e_c > 0, e_t > 0):
        y_c < 0 and y_t < 0 -> y_c + y_t < 0 -> contact occurs!
        Coupled model solves for interchange deflection y:
            y_c = y,  y_t = -y
            (K_fc + K_ft)*y + F_c*(H_c/D_c) - F_t*(H_t/D_t) = 0
            F_n = F_c*(H_c/D_c) + F_t*(H_t/D_t)
        Compare with linearized contact stiffness formula:
            F_n_linear = -(y_c_unc + y_t_unc) / (1/K_fc + 1/K_ft)
    """
    p = Law58Params(e1=10000.0, e2=8000.0, flex=0.005, n1=1, n2=1, s1=0.1, s2=0.1)

    # Biaxial tension: warp stretched 4%, weft stretched 2%
    ec = 0.04
    et = 0.02
    lc = p.lc0 * (1.0 + ec)
    lt = p.lt0 * (1.0 + et)

    yc, yt, fc, ft, fn, dc, dt = crimp_interchange(p, lc, lt, niter=5)

    # In coupled mode: y_c = y, y_t = -y
    assert yc == pytest.approx(-yt, abs=1e-10)
    # Equilibrium contact force is strictly positive
    assert fn > 0.0

    # Verify exact Fortran equilibrium equation:
    # (K_fc + K_ft)*y + F_c*(H_c/D_c) - F_t*(H_t/D_t) == 0
    hc = p.hc0 + yc
    ht = p.ht0 + yt
    dc_calc = math.sqrt(lc**2 + hc**2)
    dt_calc = math.sqrt(lt**2 + ht**2)
    dcc = dc_calc - p.dc0
    dtt = dt_calc - p.dt0
    fc_calc = (p.kc - 0.5 * p.kbc * dcc) * dcc
    ft_calc = (p.kt - 0.5 * p.kbt * dtt) * dtt

    res_coupled = (p.kfc + p.kft) * yc + fc_calc * (hc / dc_calc) - ft_calc * (ht / dt_calc)
    assert abs(res_coupled) < 1e-4

    # Fortran contact force F_n (sigeps58c.F:459)
    fn_expected = fc_calc * (hc / dc_calc) + ft_calc * (ht / dt_calc)
    assert fn == pytest.approx(fn_expected, rel=1e-5)


# ============================================================================
# 3. Membrane Tension & Fiber Modulus (Analytical & Tabulated & Unloading)
# ============================================================================

def test_analytical_membrane_tension_and_softening():
    """Verify analytical yarn tension with quadratic softening (sigeps58c.F:364-370):
        D_cc < CCL:  F_c = (K_c - 0.5 * K_bc * D_cc) * D_cc
        D_cc >= CCL: F_c = 0.5 * K_c * CCL
        sigma_xx = (F_c * L_c / D_c) * (N_c / exp(eps_yy))
    """
    e1, b1 = 4000.0, 100.0
    p = Law58Params(e1=e1, b1=b1, e2=3000.0, b2=0.0, n1=1, n2=1, s1=0.1, s2=0.1, df=0.0)

    # CCL = K_c / K_bc
    ccl_expected = p.kc / p.kbc
    assert p.ccl == pytest.approx(ccl_expected)

    # Pure 1D warp strain
    eps_xx = 0.03
    deps = np.array([[eps_xx, 0.0, 0.0]])
    sig = np.zeros((1, 3))
    sig_out, _ = shell_update_law58(p, sig, deps, dt=0.0)

    # Hand-calculate expected response
    ec = math.exp(eps_xx) - 1.0
    lc = p.lc0 * (1.0 + ec)
    # Small deformation -> hc ~ hc0, dc = sqrt(lc^2 + hc0^2)
    yc, yt, fc, ft, fn, dc, dt = crimp_interchange(p, lc, p.lt0, niter=5)
    rfac = p.nc / math.exp(0.0)  # eps_yy = 0
    expected_sxx = (fc * lc / dc) * rfac

    assert sig_out[0, 0] == pytest.approx(expected_sxx, rel=1e-5)


def test_tabulated_fiber_loading_fun_a1_fun_a2():
    """Verify tabulated loading curves FUN_A1, FUN_A2 with scale factors C1, C2
    matching sigeps58c.F:476-487:
        F_c = C_1 * f_1(D_cc)
        F_t = C_2 * f_2(D_tt)
    """
    # Define piecewise linear curve: F(elongation)
    pts1 = [(0.0, 0.0), (0.02, 100.0), (0.05, 300.0), (0.10, 800.0)]
    pts2 = [(0.0, 0.0), (0.02, 80.0), (0.05, 240.0), (0.10, 600.0)]
    curve1 = MockCurve(pts1)
    curve2 = MockCurve(pts2)

    c1, c2 = 1.5, 2.0
    p = Law58Params(
        e1=1000.0, e2=1000.0,
        fun_a1=curve1, c1=c1,
        fun_a2=curve2, c2=c2,
    )

    # Elongations
    eps_xx = 0.04
    eps_yy = 0.03
    deps = np.array([[eps_xx, eps_yy, 0.0]])
    sig = np.zeros((1, 3))
    sig_out, _ = shell_update_law58(p, sig, deps, dt=1e-5)

    ec = math.exp(eps_xx) - 1.0
    et = math.exp(eps_yy) - 1.0
    lc = p.lc0 * (1.0 + ec)
    lt = p.lt0 * (1.0 + et)
    yc, yt, fc, ft, fn, dc, dt = crimp_interchange(p, lc, lt, niter=5)

    dcc = dc - p.dc0
    dtt = dt - p.dt0
    fc_expected, _ = curve1(dcc)
    fc_expected *= c1
    ft_expected, _ = curve2(dtt)
    ft_expected *= c2

    assert fc == pytest.approx(fc_expected, rel=1e-4)
    assert ft == pytest.approx(ft_expected, rel=1e-4)


def test_fiber_unloading_hysteresis_fun_a4_fun_a5():
    """Verify unloading behavior with curves FUN_A4, FUN_A5 with scale factors C4, C5
    matching sigeps58c.F:740-750 & 883-900:
        During loading: peak reached (E_max_rl, S_max_rl)
        During unloading (d(D_c) < 0):
            eps_nc = D_cc * epsi1 / E_max_rl
            F_c = (scale4 * f4(eps_nc) / sigi1) * S_max_rl
    """
    # Curve 1 (loading) and Curve 4 (unloading)
    curve1 = MockCurve([(0.0, 0.0), (0.02, 200.0), (0.05, 500.0), (0.10, 1000.0)])
    # Unloading curve from (0,0) to peak (0.10, 1000.0) with residual shape
    curve4 = MockCurve([(0.0, 0.0), (0.02, 50.0), (0.05, 200.0), (0.10, 1000.0)])

    p = Law58Params(
        e1=5000.0, e2=5000.0,
        fun_a1=curve1, c1=1.0,
        fun_a4=curve4, scale4=1.0,
        df=0.0,
    )

    assert p.unload == 1
    # Intersection of curve1 and curve4 is at (0.10, 1000.0)
    assert p.epsi1 == pytest.approx(0.10)
    assert p.sigi1 == pytest.approx(1000.0)

    # Step 1: Load up to eps_xx = 0.06
    extra = {}
    sig = np.zeros((1, 3))
    deps_load = np.array([[0.06, 0.0, 0.0]])
    sig_peak, _ = shell_update_law58(p, sig, deps_load, dt=1e-5, extra=extra)

    sxx_peak = sig_peak[0, 0]
    assert sxx_peak > 0.0
    emax_rl_c = extra["emax_rl_c"][0]
    smax_rl_c = extra["smax_rl_c"][0]
    assert emax_rl_c > 0.0
    assert smax_rl_c > 0.0

    # Step 2: Unload by deps_xx = -0.02
    deps_unload = np.array([[-0.02, 0.0, 0.0]])
    sig_unloaded, _ = shell_update_law58(p, sig_peak, deps_unload, dt=1e-5, extra=extra)

    sxx_unloaded = sig_unloaded[0, 0]
    # Stress during unloading is lower than peak
    assert sxx_unloaded < sxx_peak
    assert sxx_unloaded > 0.0

    # Check against exact Fortran formula
    dc_cur = extra["dc_old"][0]
    dcc_cur = dc_cur - p.dc0
    eps_nc_expected = dcc_cur * p.epsi1 / emax_rl_c
    f_un_expected, _ = curve4(eps_nc_expected)
    fc_expected = (f_un_expected / p.sigi1) * smax_rl_c
    lc_cur = p.lc0 * math.exp(0.04)
    expected_sxx = (fc_expected * lc_cur / dc_cur) * p.nc
    assert sxx_unloaded == pytest.approx(expected_sxx, rel=1e-4)


# ============================================================================
# 4. Trellis Shear Behavior & Lock Angle
# ============================================================================

def test_trellis_shear_analytical_lock_angle_continuity():
    """Verify analytical Trellis shear behavior matching hm_read_mat58.F:254-256 and sigeps58c.F:525-540:
        For |tan_phi| <= tan_phi_lock:
            sigma_xy = G_0 * tan_phi
        For |tan_phi| > tan_phi_lock:
            G_post = G_t / (1 + tan_phi_lock^2)
            G_b = tan_phi_lock * (G_0 - G_post)
            sigma_xy = G_post * tan_phi + sign(tan_phi) * G_b
        Verify C^0 continuity at tan_phi = +/- tan_phi_lock!
    """
    g0 = 100.0
    gt = 500.0
    alphat = 45.0  # phi_lock = 45 deg -> tan_phi_lock = 1.0

    p = Law58Params(g0=g0, gt=gt, alphat=alphat)

    assert p.tan_lock == pytest.approx(1.0)
    # G_post = 500 / (1 + 1) = 250.0
    assert p.g_post == pytest.approx(250.0)
    # G_b = 1.0 * (100.0 - 250.0) = -150.0
    assert p.gb == pytest.approx(-150.0)

    # 1. Below lock angle: tan_phi = 0.5 -> sigma_xy = 100.0 * 0.5 = 50.0
    sig = np.zeros((1, 3))
    deps1 = np.array([[0.0, 0.0, 0.5]])
    s1, _ = shell_update_law58(p, sig, deps1, dt=1e-5)
    assert s1[0, 2] == pytest.approx(50.0)

    # 2. Right at lock angle: tan_phi = 1.0 -> sigma_xy = 100.0 * 1.0 = 100.0
    extra2 = {}
    deps2 = np.array([[0.0, 0.0, 1.0]])
    s2, _ = shell_update_law58(p, sig, deps2, dt=1e-5, extra=extra2)
    assert s2[0, 2] == pytest.approx(100.0)

    # 3. Just above lock angle: tan_phi = 1.0 + 1e-6
    deps3 = np.array([[0.0, 0.0, 1.0 + 1e-6]])
    s3, _ = shell_update_law58(p, sig, deps3, dt=1e-5)
    assert s3[0, 2] == pytest.approx(100.0, abs=1e-3)

    # 4. Beyond lock angle: tan_phi = 2.0
    # sigma_xy = G_post * 2.0 + G_b = 250.0 * 2.0 - 150.0 = 350.0
    deps4 = np.array([[0.0, 0.0, 2.0]])
    s4, _ = shell_update_law58(p, sig, deps4, dt=1e-5)
    assert s4[0, 2] == pytest.approx(350.0)

    # 5. Negative shear: tan_phi = -2.0 -> sigma_xy = -350.0 (anti-symmetry)
    deps5 = np.array([[0.0, 0.0, -2.0]])
    s5, _ = shell_update_law58(p, sig, deps5, dt=1e-5)
    assert s5[0, 2] == pytest.approx(-350.0)


def test_trellis_shear_tabulated_loading_and_unloading():
    """Verify tabulated shear loading (FUN_A3) and unloading (FUN_A6)
    matching sigeps58c.F:1550-1615:
        phi = atan(tan_phi) * 180 / pi
        Loading: sigma_xy = C_3 * f_3(phi)
        Unloading: sigma_xy = (scale6 * f6(phi * phii / phi_max) / sxyi) * sxy_max
    """
    curve3 = MockCurve([(0.0, 0.0), (15.0, 50.0), (30.0, 120.0), (60.0, 400.0)])
    curve6 = MockCurve([(0.0, 0.0), (15.0, 20.0), (30.0, 60.0), (60.0, 400.0)])

    p = Law58Params(
        e1=1000.0, e2=1000.0,
        fun_a3=curve3, c3=1.0,
        fun_a6=curve6, scale6=1.0,
    )

    assert p.unload == 1
    assert p.phii == pytest.approx(60.0)
    assert p.sxyi == pytest.approx(400.0)

    # Load up to tan_phi = tan(30 deg) ~ 0.57735
    tan_30 = math.tan(math.radians(30.0))
    extra = {}
    sig = np.zeros((1, 3))
    s_peak, _ = shell_update_law58(p, sig, np.array([[0.0, 0.0, tan_30]]), dt=1e-5, extra=extra)

    assert s_peak[0, 2] == pytest.approx(120.0, rel=1e-4)

    # Unload to tan_phi = tan(15 deg) ~ 0.26795
    tan_15 = math.tan(math.radians(15.0))
    dtan = tan_15 - tan_30
    s_unloaded, _ = shell_update_law58(p, s_peak, np.array([[0.0, 0.0, dtan]]), dt=1e-5, extra=extra)

    # Expected: phi = 15 deg, phi_max = 30 deg
    # phi_norm = 15.0 * 60.0 / 30.0 = 30.0 deg
    # f6(30 deg) = 60.0
    # sigma_xy = (60.0 / 400.0) * 120.0 = 18.0
    expected_sxy = (60.0 / 400.0) * 120.0
    assert s_unloaded[0, 2] == pytest.approx(expected_sxy, rel=1e-4)


# ============================================================================
# 5. Yarn Sliding Friction & Viscous Damping
# ============================================================================

def test_yarn_sliding_friction_coulomb_limit():
    """Verify yarn sliding friction limit tau_frot matching sigeps58c.F:550-557:
        tau_frot = (2/3) * d_S * F_n * (H_c0 + H_t0) / (L_c + L_t)
        sigma_g = tau_fold + G_frot * Delta(tan_phi)
        sigma_v_xy = sign(tau_frot, sigma_g) if |sigma_g| > tau_frot else sigma_g
    """
    ds = 0.25
    gfrot = 50.0
    p = Law58Params(
        e1=5000.0, e2=5000.0, flex=0.01,
        ds=ds, gfrot=gfrot, g0=0.0,
    )

    # Apply biaxial strain to establish normal contact force F_n > 0
    extra = {}
    sig = np.zeros((1, 3))
    # Step 1: Biaxial stretch
    shell_update_law58(p, sig, np.array([[0.04, 0.04, 0.0]]), dt=1e-5, extra=extra)
    fn = extra["fn"][0]
    assert fn > 0.0

    # Theoretical friction limit tau_frot
    lc = p.lc0 * math.exp(0.04)
    lt = p.lt0 * math.exp(0.04)
    tau_frot_expected = (2.0 / 3.0) * ds * fn * (p.hc0 + p.ht0) / (lc + lt)
    assert tau_frot_expected > 0.0

    # Step 2: Small shear increment -> elastic stick regime (|sigma_g| < tau_frot)
    d_tan_small = (0.5 * tau_frot_expected) / gfrot
    s_stick, _ = shell_update_law58(p, sig, np.array([[0.0, 0.0, d_tan_small]]), dt=1e-5, extra=extra)
    assert extra["sigv_xy"][0] == pytest.approx(gfrot * d_tan_small, rel=1e-4)

    # Step 3: Large shear increment -> Coulomb slip regime clamped at tau_frot
    d_tan_large = (10.0 * tau_frot_expected) / gfrot
    s_slip, _ = shell_update_law58(p, sig, np.array([[0.0, 0.0, d_tan_large]]), dt=1e-5, extra=extra)
    assert extra["sigv_xy"][0] == pytest.approx(tau_frot_expected, rel=1e-4)


def test_viscous_fiber_damping_scaling():
    """Verify viscous damping in normal directions matching sigeps58c.F:542-547:
        D_amp = sqrt(0.5 * rho0 * Area * Thk)
        V1 = D_f * D_amp * sqrt(N_c * K_c)
        sigma_v_xx = (deps_xx / dt) * V1
    """
    df = 0.08
    rho0 = 1.2e-6
    area = 100.0
    thk = 0.5
    dt = 1e-4
    p = Law58Params(e1=8000.0, e2=6000.0, df=df, rho0=rho0)

    extra = {"area": area, "thk": thk}
    deps_xx = 0.002
    sig = np.zeros((1, 3))
    s_out, _ = shell_update_law58(p, sig, np.array([[deps_xx, 0.0, 0.0]]), dt=dt, extra=extra)

    # Expected viscous stress
    damp = math.sqrt(0.5 * rho0 * area * thk)
    v1_expected = df * damp * math.sqrt(p.nc * p.kc)
    sigv_xx_expected = (deps_xx / dt) * v1_expected

    # Subtract elastic stress to isolate viscous damping contribution
    s_out_no_visc, _ = shell_update_law58(
        Law58Params(e1=8000.0, e2=6000.0, df=0.0, rho0=rho0),
        sig, np.array([[deps_xx, 0.0, 0.0]]), dt=dt, extra={"area": area, "thk": thk}
    )
    sigv_xx_actual = s_out[0, 0] - s_out_no_visc[0, 0]
    assert sigv_xx_actual == pytest.approx(sigv_xx_expected, rel=1e-4)


# ============================================================================
# 6. Zero-Stress Relative Area Threshold (A/A0 <= A_rel, M58_Zerostress)
# ============================================================================

def test_zero_stress_relative_area_deactivation():
    """Verify zero-stress folding deactivation matching sigeps58c.F:1739-1751:
        rel_area = 1 + e_c + e_t
        if rel_area <= A_rel:
            all stresses collapse to 0.0!
    """
    arel = 0.85
    p = Law58Params(e1=5000.0, e2=5000.0, g0=200.0, g5=100.0, arel=arel)

    # 1. Normal state (rel_area > arel): stresses non-zero
    sig = np.zeros((1, 5))
    deps_normal = np.array([[0.05, 0.02, 0.1, 0.01, 0.01]])
    s_norm, _ = shell_update_law58(p, sig, deps_normal, dt=1e-5)
    assert s_norm[0, 0] > 0.0
    assert s_norm[0, 2] > 0.0

    # 2. Large compressive folding deformation -> rel_area = 1 + ec + et < 0.85
    # e.g. ec = -0.10, et = -0.10 -> 1 + ec + et = 0.80 <= 0.85
    deps_fold = np.array([[-0.10536, -0.10536, 0.1, 0.01, 0.01]])  # exp(etc) ~ 0.90
    s_fold, _ = shell_update_law58(p, sig, deps_fold, dt=1e-5)

    # All stress components (normal, membrane shear, and transverse shear) collapse to zero!
    assert s_fold[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert s_fold[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert s_fold[0, 2] == pytest.approx(0.0, abs=1e-12)
    assert s_fold[0, 3] == pytest.approx(0.0, abs=1e-12)
    assert s_fold[0, 4] == pytest.approx(0.0, abs=1e-12)


# ============================================================================
# 7. Transverse Shear Formulation with G5
# ============================================================================

def test_transverse_shear_formulation():
    """Verify transverse shear update matching sigeps58c.F:579-580:
        sigma_yz_new = sigma_yz_old + G5 * deps_yz
        sigma_zx_new = sigma_zx_old + G5 * deps_zx
        with default G5 = G0 if G5 == 0.
    """
    g5_custom = 350.0
    p = Law58Params(e1=1000.0, g0=100.0, g5=g5_custom)

    sig_old = np.array([[0.0, 0.0, 0.0, 20.0, -15.0]])
    deps = np.array([[0.0, 0.0, 0.0, 0.02, 0.03]])

    sig_new, _ = shell_update_law58(p, sig_old, deps, dt=1e-5)

    expected_syz = 20.0 + g5_custom * 0.02
    expected_szx = -15.0 + g5_custom * 0.03

    assert sig_new[0, 3] == pytest.approx(expected_syz)
    assert sig_new[0, 4] == pytest.approx(expected_szx)

    # Check default fallback: if G5 == 0 in params, G5 defaults to G0
    p_def = Law58Params(e1=1000.0, g0=120.0, g5=0.0)
    assert p_def.g5 == 120.0


# ============================================================================
# 8. Shell Acoustic Sound Speed
# ============================================================================

def test_shell_acoustic_sound_speed():
    """Verify shell sound speed formula matching hm_read_mat58.F:319-329:
        c_shell = sqrt(max(K_c, K_t, G_0) / rho_0)
        where K_c = E_1 / N_c, K_t = E_2 / N_t.
    """
    rho0 = 1.2e-6
    e1, n1 = 6000.0, 2  # Kc = 3000.0
    e2, n2 = 8000.0, 2  # Kt = 4000.0
    g0 = 2500.0

    p = Law58Params(rho0=rho0, e1=e1, n1=n1, e2=e2, n2=n2, g0=g0)
    mat = FabricAMaterial(id=58, law=58, rho0=rho0, params=p.__dict__)

    # kmax = max(3000, 4000, 2500) = 4000.0
    c_expected = math.sqrt(4000.0 / rho0)
    c_actual = sound_speed_shell_law58(p)

    assert c_actual == pytest.approx(c_expected)
    assert mat.sound_speed_shell() == pytest.approx(c_expected)


# ============================================================================
# 9. Curve Intersection Algorithm (law58_upd.F:240-290 & func_inters.F)
# ============================================================================

def test_curve_intersection_algorithm():
    """Verify find_curve_intersection matches OpenRadioss func_inters.F:
    - Common point detection
    - Segment-segment intersection
    - Same-curve fallback to peak
    """
    # 1. Identical curves -> returns last point (func_inters.F:245-247)
    c1 = MockCurve([(0.0, 0.0), (0.05, 100.0), (0.10, 250.0)])
    x_int, y_int = find_curve_intersection(c1, 1.0, c1, 1.0)
    assert x_int == pytest.approx(0.10)
    assert y_int == pytest.approx(250.0)

    # 2. Intersecting line segments
    # Line 1: (0, 0) -> (0.10, 200.0)  slope = 2000
    # Line 2: (0, 100.0) -> (0.10, 100.0) horizontal line
    # Intersection at x = 0.05, y = 100.0
    l1 = MockCurve([(0.0, 0.0), (0.10, 200.0)])
    l2 = MockCurve([(0.0, 100.0), (0.10, 100.0)])
    x_int2, y_int2 = find_curve_intersection(l1, 1.0, l2, 1.0)
    assert x_int2 == pytest.approx(0.05)
    assert y_int2 == pytest.approx(100.0)
