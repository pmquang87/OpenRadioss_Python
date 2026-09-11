"""Unit tests for Milestone M562: /MAT/LAW66 (/MAT/PLAS_TAB_COSSER, /MAT/PLAS_COSSER)
Asymmetric Tension/Compression Tabulated Plasticity Model for Solids and Shells.

Upstream Fortran references:
  - Starter reader: starter/source/materials/mat/mat066/hm_read_mat66.F
  - Solid engine physics: engine/source/materials/mat/mat066/sigeps66.F
  - Shell engine physics: engine/source/materials/mat/mat066/sigeps66c.F
"""

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material, MatLaw66
from pyradioss.materials.law66_plas_tab import (
    Law66Params,
    build_law66,
    solid_update,
    shell_update,
    consistent_solid_tangent,
    consistent_shell_tangent,
    shell_membrane_tangent,
    sound_speed,
    sound_speed_solid,
    sound_speed_shell,
    extra_shapes,
    resolve,
)
import pyradioss.materials as materials


# ============================================================================
# 1. Parameter Dataclass: Law66Params & Properties
# ============================================================================

def test_law66_params_defaults_and_properties():
    """Verify Law66Params default construction, bounds clamping, and derived constants."""
    p = Law66Params()

    # Defaults
    assert p.rho0 == 1.0
    assert p.refer_rho == 1.0
    assert p.e == 210000.0
    assert p.nu == 0.3
    assert p.ec == 210000.0  # Defaults to e when <= 0
    assert p.pc == 0.0
    assert p.pt == 0.0
    assert p.rpct == 1.0
    assert p.chard == 0.0
    assert p.asrate == 0.0
    assert p.fsmooth == 0
    assert p.israte == 1
    assert p.epsp0 == 0.0
    assert p.cp == 1.0
    assert p.sigy == 0.0
    assert p.vp == 0

    # Derived constants for tension
    nu_denom = 1.0 - 2.0 * 0.3
    nu_sq = 1.0 - 0.3 ** 2
    expected_gt = 0.5 * 210000.0 / 1.3
    expected_c1t = 210000.0 / (3.0 * nu_denom)
    expected_a11t = 210000.0 / nu_sq
    expected_a21t = 0.3 * expected_a11t

    assert math.isclose(p.gt, expected_gt, rel_tol=1e-12)
    assert math.isclose(p.c1t, expected_c1t, rel_tol=1e-12)
    assert math.isclose(p.a11t, expected_a11t, rel_tol=1e-12)
    assert math.isclose(p.a21t, expected_a21t, rel_tol=1e-12)

    # Derived constants for compression (since ec defaults to e, they match)
    assert math.isclose(p.gc, expected_gt, rel_tol=1e-12)
    assert math.isclose(p.c1c, expected_c1t, rel_tol=1e-12)
    assert math.isclose(p.a11c, expected_a11t, rel_tol=1e-12)
    assert math.isclose(p.a21c, expected_a21t, rel_tol=1e-12)

    # Property accessors
    assert p.E == 210000.0
    assert p.nu0 == 0.3
    assert math.isclose(p.G, expected_gt, rel_tol=1e-12)
    assert math.isclose(p.K, expected_c1t, rel_tol=1e-12)
    assert p.fisokin == 0.0

    # Sound speeds
    expected_c_solid = math.sqrt((expected_c1t + (4.0 / 3.0) * expected_gt) / 1.0)
    expected_c_shell = math.sqrt(expected_a11t / 1.0)
    assert math.isclose(p.soundsp, expected_c_solid, rel_tol=1e-12)
    assert math.isclose(p.soundsp_shell, expected_c_shell, rel_tol=1e-12)


def test_law66_params_asymmetric_moduli_and_clamping():
    """Verify distinct tension and compression Young's moduli and nu clamping."""
    p = Law66Params(
        rho0=7.8e-6,
        refer_rho=7.8e-6,
        e=200000.0,    # Et
        ec=150000.0,   # Ec
        nu=0.55,       # Should be clamped to 0.499
        pc=500.0,
        pt=200.0,
        rpct=0.8,
        chard=0.4,
        israte=2,
    )

    assert p.nu == 0.499
    assert p.ec == 150000.0
    assert p.chard == 0.4
    assert p.rpct == 0.8
    assert p.israte == 2

    nu_denom = 1.0 - 2.0 * 0.499
    assert math.isclose(p.c1t, 200000.0 / (3.0 * nu_denom), rel_tol=1e-12)
    assert math.isclose(p.c1c, 150000.0 / (3.0 * nu_denom), rel_tol=1e-12)
    assert math.isclose(p.gt, 0.5 * 200000.0 / 1.499, rel_tol=1e-12)
    assert math.isclose(p.gc, 0.5 * 150000.0 / 1.499, rel_tol=1e-12)


# ============================================================================
# 2. Factory: build_law66 & Mapping Protocol
# ============================================================================

def test_build_law66_from_dataclass_and_dict():
    """Verify build_law66 constructs valid Material objects with Law66Params."""
    # Build from Law66Params
    params = Law66Params(e=180000.0, ec=120000.0, nu=0.25, rho0=2.7e-6, sigy=250.0)
    mat1 = build_law66(params)

    assert mat1.law == 66
    assert mat1.law_name == "LAW66"
    assert mat1.params["_obj"] is params
    assert mat1.params["e"] == 180000.0
    assert mat1.params["ec"] == 120000.0
    assert mat1.params["sigy"] == 250.0

    # Build from dict using Radioss keyword names
    raw_dict = {
        "id": 10,
        "title": "FoamTabMat",
        "MAT_RHO": 1.2e-6,
        "Refer_Rho": 1.2e-6,
        "MAT_E": 1000.0,
        "MAT_EC": 800.0,
        "MAT_NU": 0.2,
        "MAT_PC": 10.0,
        "MAT_PT": 5.0,
        "MAT_RPCT": 0.5,
        "MAT_HARD": 0.3,
        "MAT_asrate": 100.0,
        "Fsmooth": 1,
        "ISRATE": 1,
        "Epsilon_0": 0.01,
        "MAT_C0": 2.5,
        "SIG_Y0": 15.0,
        "VP": 1,
        "FUN_A1": 101,
        "FUN_A2": 102,
        "FScale11": 1.2,
        "FScale22": 0.9,
    }
    mat2 = build_law66(raw_dict)
    assert mat2.id == 10
    assert mat2.title == "FoamTabMat"
    p2 = mat2.params["_obj"]
    assert isinstance(p2, Law66Params)
    assert p2.e == 1000.0
    assert p2.ec == 800.0
    assert p2.pc == 10.0
    assert p2.pt == 5.0
    assert p2.rpct == 0.5
    assert p2.chard == 0.3
    assert p2.asrate == 100.0
    assert p2.fsmooth == 1
    assert p2.israte == 1
    assert p2.epsp0 == 0.01
    assert p2.cp == 2.5
    assert p2.sigy == 15.0
    assert p2.vp == 1
    assert p2.fun_a1 == 101
    assert p2.fun_a2 == 102
    assert p2.fscale11 == 1.2
    assert p2.fscale22 == 0.9


def test_matlaw66_entity_mapping_protocol():
    """Verify MatLaw66 dataclass supports mapping protocol and aliases."""
    mat_ent = MatLaw66(
        id=5,
        title="Alloy66",
        rho=7.85e-6,
        e=205000.0,
        nu=0.29,
        ec=190000.0,
        pc=200.0,
        pt=100.0,
        rpct=0.9,
        chard=0.6,
        sigy=350.0,
        fun_a1=21,
        fun_a2=22,
    )

    # Item access
    assert mat_ent["id"] == 5
    assert mat_ent["E"] == 205000.0
    assert mat_ent["rho0"] == 7.85e-6
    assert mat_ent["c_hard"] == 0.6
    assert mat_ent["fisokin"] == 0.6
    assert mat_ent["funct_idc"] == 21
    assert mat_ent["funct_idt"] == 22

    # Keys, values, items, get, in
    assert "E" in mat_ent
    assert "fisokin" in mat_ent
    assert mat_ent.get("p_c") == 200.0
    assert mat_ent.get("p_t") == 100.0
    assert mat_ent.get("nonexistent", 999) == 999

    # Build from MatLaw66 entity
    mat = build_law66(mat_ent)
    assert mat.id == 5
    p = mat.params["_obj"]
    assert p.e == 205000.0
    assert p.ec == 190000.0
    assert p.sigy == 350.0


# ============================================================================
# 3. Modulus Transition under Hydrostatic Pressure
# ============================================================================

def test_law66_pressure_modulus_transition():
    """Verify elastic modulus transition between ET (tension) and EC (compression)."""
    # Parameters: ET = 200, EC = 100, PC = 50, PT = 20, RPCT = 1.0
    p = Law66Params(
        e=200.0,
        ec=100.0,
        pc=50.0,
        pt=20.0,
        rpct=1.0,
        nu=0.25,
        rho0=1.0,
    )

    from pyradioss.materials.law66_plas_tab import _compute_elastic_moduli

    # 1. Pure tension: P <= -RPCT * PT = -20.0 -> E = ET = 200.0
    P_tens = np.array([-50.0, -30.0, -20.0])
    E, G, C1 = _compute_elastic_moduli(p, P_tens)
    np.testing.assert_allclose(E, [200.0, 200.0, 200.0])
    np.testing.assert_allclose(G, 0.5 * 200.0 / 1.25)

    # 2. Pure compression: P >= RPCT * PC = 50.0 -> E = EC = 100.0
    P_comp = np.array([50.0, 70.0, 100.0])
    E, G, C1 = _compute_elastic_moduli(p, P_comp)
    np.testing.assert_allclose(E, [100.0, 100.0, 100.0])
    np.testing.assert_allclose(G, 0.5 * 100.0 / 1.25)

    # 3. Transition: P in (-20.0, 50.0)
    # Range is RPCT * (PC + PT) = 70.0
    # At P = 15.0: fac = (50 - 15) / 70 = 35 / 70 = 0.5 -> E = 0.5 * 200 + 0.5 * 100 = 150.0
    P_mid = np.array([15.0])
    E, _, _ = _compute_elastic_moduli(p, P_mid)
    assert math.isclose(E[0], 150.0, rel_tol=1e-12)

    # At P = -6.0: fac = (50 - (-6)) / 70 = 56 / 70 = 0.8 -> E = 0.8 * 200 + 0.2 * 100 = 180.0
    P_tens_mid = np.array([-6.0])
    E, _, _ = _compute_elastic_moduli(p, P_tens_mid)
    assert math.isclose(E[0], 180.0, rel_tol=1e-12)


def test_law66_zero_pressure_limits_modulus_transition():
    """When PC = PT = 0, modulus switches at P = 0."""
    p = Law66Params(
        e=210.0,
        ec=150.0,
        pc=0.0,
        pt=0.0,
        nu=0.3,
        rho0=1.0,
    )
    from pyradioss.materials.law66_plas_tab import _compute_elastic_moduli

    P = np.array([-10.0, -1e-5, 0.0, 1e-5, 10.0])
    E, _, _ = _compute_elastic_moduli(p, P)
    assert E[0] == 210.0
    assert E[1] == 210.0
    assert E[2] == 150.0  # P >= 0 -> EC
    assert E[3] == 150.0
    assert E[4] == 150.0


# ============================================================================
# 4. Asymmetric Yield Response & Pressure Interpolation
# ============================================================================

def test_law66_asymmetric_yield_pressure_interpolation():
    """Verify asymmetric yield: tension curve YT vs compression curve YC interpolated by P."""
    # Tension curve: yield starts at 200, slope 500
    curve_t = ([0.0, 0.1], [200.0, 250.0])
    # Compression curve: yield starts at 400, slope 1000
    curve_c = ([0.0, 0.1], [400.0, 500.0])

    p = Law66Params(
        pc=100.0,
        pt=50.0,
        curve_t=curve_t,
        curve_c=curve_c,
        fscale11=1.0,
        fscale22=1.0,
    )

    from pyradioss.materials.law66_plas_tab import _compute_yield_and_hardening

    epsp = np.array([0.02, 0.02, 0.02])
    rate = np.zeros(3)

    # At epsp = 0.02:
    # YT = 200 + 0.02 * 500 = 210.0, HT = 500.0
    # YC = 400 + 0.02 * 1000 = 420.0, HC = 1000.0

    # Case 1: P <= -PT = -50.0 -> YLD = YT = 210.0, H = HT = 500.0
    P1 = np.array([-60.0, -50.0, -80.0])
    yld1, h1 = _compute_yield_and_hardening(p, P1, epsp, rate)
    np.testing.assert_allclose(yld1, [210.0, 210.0, 210.0])
    np.testing.assert_allclose(h1, [500.0, 500.0, 500.0])

    # Case 2: P >= PC = 100.0 -> YLD = YC = 420.0, H = HC = 1000.0
    P2 = np.array([100.0, 150.0, 200.0])
    yld2, h2 = _compute_yield_and_hardening(p, P2, epsp, rate)
    np.testing.assert_allclose(yld2, [420.0, 420.0, 420.0])
    np.testing.assert_allclose(h2, [1000.0, 1000.0, 1000.0])

    # Case 3: P in (-50, 100)
    # Range is PC + PT = 150.0
    # At P = 25.0: fac = (100 - 25) / 150 = 75 / 150 = 0.5
    # YLD = 0.5 * 210 + 0.5 * 420 = 315.0
    # H = 0.5 * 500 + 0.5 * 1000 = 750.0
    P3 = np.array([25.0])
    yld3, h3 = _compute_yield_and_hardening(p, P3, epsp[:1], rate[:1])
    assert math.isclose(yld3[0], 315.0, rel_tol=1e-12)
    assert math.isclose(h3[0], 750.0, rel_tol=1e-12)


# ============================================================================
# 5. Strain Rate Sensitivity Modes (ISRATE 1, 2, 3, 4) & VP
# ============================================================================

def test_law66_strain_rate_modes():
    """Verify strain rate formulations for ISRATE = 1 (power law) and ISRATE = 2 (logarithmic)."""
    # ISRATE = 1: YRATE = 1 + (rate / epsp0)**cp
    p1 = Law66Params(
        israte=1,
        epsp0=10.0,
        cp=0.5,
        sigy=100.0,
    )
    from pyradioss.materials.law66_plas_tab import _compute_yield_and_hardening

    # At rate = 40.0: YRATE = 1 + (40/10)**0.5 = 1 + 2 = 3.0
    yld1, _ = _compute_yield_and_hardening(p1, np.array([0.0]), np.array([0.0]), np.array([40.0]))
    assert math.isclose(yld1[0], 300.0, rel_tol=1e-12)

    # ISRATE = 2: YRATE = 1 + cp * ln(rate / epsp0)
    p2 = Law66Params(
        israte=2,
        epsp0=1.0,
        cp=0.1,
        sigy=200.0,
    )
    # At rate = math.exp(3.0) ~ 20.0855: YRATE = 1 + 0.1 * 3.0 = 1.3
    rate_val = math.exp(3.0)
    yld2, _ = _compute_yield_and_hardening(p2, np.array([0.0]), np.array([0.0]), np.array([rate_val]))
    assert math.isclose(yld2[0], 260.0, rel_tol=1e-12)


def test_law66_strain_rate_israte3_independent_curves():
    """Verify ISRATE = 3 independent strain rate scaling curves for tension and compression."""
    # Rate curve compression: scale doubles at rate 100
    curve_rate_c = ([0.0, 100.0], [1.0, 2.0])
    # Rate curve tension: scale triples at rate 100
    curve_rate_t = ([0.0, 100.0], [1.0, 3.0])

    p = Law66Params(
        israte=3,
        pc=50.0,
        pt=50.0,
        sigy=100.0,
        curve_rate_c=curve_rate_c,
        curve_rate_t=curve_rate_t,
        fscale33=1.0,
        fscale12=1.0,
    )
    from pyradioss.materials.law66_plas_tab import _compute_yield_and_hardening

    rate = np.array([100.0])
    # Under pure compression (P = 100 >= PC): rate multiplier is 2.0 -> yld = 200.0
    yld_c, _ = _compute_yield_and_hardening(p, np.array([100.0]), np.array([0.0]), rate)
    assert math.isclose(yld_c[0], 200.0, rel_tol=1e-12)

    # Under pure tension (P = -100 <= -PT): rate multiplier is 3.0 -> yld = 300.0
    yld_t, _ = _compute_yield_and_hardening(p, np.array([-100.0]), np.array([0.0]), rate)
    assert math.isclose(yld_t[0], 300.0, rel_tol=1e-12)


def test_law66_strain_rate_israte4_curve_families():
    """Verify ISRATE = 4 curve families with rate interpolation."""
    # Two compression curves at rates 0.0 and 100.0
    curve_c_0 = ([0.0, 0.1], [100.0, 150.0])
    curve_c_100 = ([0.0, 0.1], [200.0, 300.0])

    p = Law66Params(
        israte=4,
        curves_c=[curve_c_0, curve_c_100],
        rates_c=[0.0, 100.0],
        fp1=[1.0, 1.0],
        pc=50.0,
        pt=50.0,
    )
    from pyradioss.materials.law66_plas_tab import _compute_yield_and_hardening

    # At epsp = 0.0, rate = 50.0: midpoint between 100.0 and 200.0 = 150.0
    yld, _ = _compute_yield_and_hardening(p, np.array([100.0]), np.array([0.0]), np.array([50.0]))
    assert math.isclose(yld[0], 150.0, rel_tol=1e-12)


# ============================================================================
# 6. Mixed Isotropic-Kinematic Hardening & Bauschinger Effect
# ============================================================================

def test_law66_isotropic_vs_kinematic_hardening_bauschinger():
    """Verify mixed hardening: chard=0 is pure isotropic, chard=1 is pure kinematic with Bauschinger effect."""
    # Hardening curve: linear from 200 to 400 at epsp = 0.1 (slope H = 2000)
    curve = ([0.0, 0.1], [200.0, 400.0])

    # Case A: Pure Isotropic Hardening (chard = 0.0)
    mat_iso = build_law66(
        e=200000.0,
        nu=0.3,
        rho0=7.8e-6,
        chard=0.0,
        curve_c=curve,
        curve_t=curve,
    )

    # Case B: Pure Kinematic Hardening (chard = 1.0)
    mat_kin = build_law66(
        e=200000.0,
        nu=0.3,
        rho0=7.8e-6,
        chard=1.0,
        curve_c=curve,
        curve_t=curve,
    )

    # 1. Forward plastic step in tension: eps_xx = +0.003
    deps_fwd = np.array([[0.003, -0.0009, -0.0009, 0.0, 0.0, 0.0]])
    sig0 = np.zeros((1, 6))
    epsp0 = np.zeros(1)

    extra_iso = {"uvar66": np.zeros((1, 8))}
    extra_kin = {"uvar66": np.zeros((1, 8))}

    sig_iso, epsp_iso, _ = solid_update(mat_iso, sig0.copy(), deps_fwd.copy(), epsp=epsp0.copy(), dt=1e-6, extra=extra_iso)
    sig_kin, epsp_kin, _ = solid_update(mat_kin, sig0.copy(), deps_fwd.copy(), epsp=epsp0.copy(), dt=1e-6, extra=extra_kin)

    # Both plasticized
    assert epsp_iso[0] > 0.0
    assert epsp_kin[0] > 0.0
    # For isotropic, backstress is zero
    np.testing.assert_allclose(extra_iso["uvar66"][:, 1:7], 0.0)
    # For kinematic, backstress shifted along tension direction
    alpha_kin = extra_kin["uvar66"][0, 1:7]
    assert alpha_kin[0] > 0.0  # Positive xx backstress

    # 2. Reverse elastic-plastic step: apply compression deps_xx = -0.004
    deps_rev = np.array([[-0.004, 0.0012, 0.0012, 0.0, 0.0, 0.0]])
    sig_rev_iso, epsp_rev_iso, _ = solid_update(mat_iso, sig_iso.copy(), deps_rev.copy(), epsp=epsp_iso.copy(), dt=1e-6, extra=extra_iso)
    sig_rev_kin, epsp_rev_kin, _ = solid_update(mat_kin, sig_kin.copy(), deps_rev.copy(), epsp=epsp_kin.copy(), dt=1e-6, extra=extra_kin)

    # Under kinematic hardening, the shifted backstress causes earlier reverse yielding!
    # Therefore epsp_rev_kin accumulates more reverse plastic strain than isotropic
    d_epsp_iso = epsp_rev_iso[0] - epsp_iso[0]
    d_epsp_kin = epsp_rev_kin[0] - epsp_kin[0]
    assert d_epsp_kin > d_epsp_iso


# ============================================================================
# 7. Solid Element Stress Update & Dilatational Sound Speed
# ============================================================================

def test_law66_solid_elastic_and_plastic_update():
    """Verify solid element stress update: elastic loading, radial return, pressure response."""
    mat = build_law66(
        e=210000.0,
        nu=0.3,
        rho0=7.85e-6,
        sigy=300.0,
        curve_c=([0.0, 0.1], [300.0, 400.0]),
        curve_t=([0.0, 0.1], [300.0, 400.0]),
    )

    # Pure elastic step below yield
    # deps_xx = 0.0005 -> sigma_xx ~ E * 0.0005 ~ 105.0 < 300.0
    sig = np.zeros((1, 6))
    deps = np.array([[0.0005, -0.00015, -0.00015, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)
    extra = {"uvar66": np.zeros((1, 8))}

    sig_out, epsp_out, c_sound = solid_update(mat, sig, deps, epsp=epsp, dt=1e-6, extra=extra)

    assert epsp_out[0] == 0.0
    assert math.isclose(sig_out[0, 0], 210000.0 * 0.0005, rel_tol=1e-4)
    assert c_sound is not None
    assert c_sound[0] > 5000.0  # Steel longitudinal sound speed ~ 5800 m/s

    # Plastic step: deps_xx = 0.005 -> von Mises stress clamped to yield surface
    deps_plas = np.array([[0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0]])
    sig_plas, epsp_plas, _ = solid_update(mat, sig_out, deps_plas, epsp=epsp_out, dt=1e-6, extra=extra)

    assert epsp_plas[0] > 0.0
    # Deviatoric von Mises stress matches initial yield stress for this step
    s = sig_plas[0, :3] - np.mean(sig_plas[0, :3])
    vm = math.sqrt(1.5 * np.sum(s ** 2))
    assert math.isclose(vm, 300.0, rel_tol=1e-3)

    # Next step: yield surface has expanded with accumulated plastic strain
    deps_step2 = np.array([[0.002, -0.0006, -0.0006, 0.0, 0.0, 0.0]])
    sig_step2, epsp_step2, _ = solid_update(mat, sig_plas, deps_step2, epsp=epsp_plas, dt=1e-6, extra=extra)
    s2 = sig_step2[0, :3] - np.mean(sig_step2[0, :3])
    vm2 = math.sqrt(1.5 * np.sum(s2 ** 2))
    assert vm2 > 300.0


def test_law66_solid_multielement_vectorized():
    """Verify solid stress update handles batch/vectorized arrays of shape (n, 6)."""
    n = 5
    mat = build_law66(
        e=200000.0,
        nu=0.25,
        rho0=2.7e-6,
        sigy=250.0,
    )
    sig = np.zeros((n, 6))
    deps = np.zeros((n, 6))
    deps[:, 0] = np.linspace(0.0005, 0.005, n)
    epsp = np.zeros(n)
    extra = {"uvar66": np.zeros((n, 8))}

    sig_out, epsp_out, c_out = solid_update(mat, sig, deps, epsp=epsp, dt=1e-6, extra=extra)

    assert sig_out.shape == (n, 6)
    assert epsp_out.shape == (n,)
    assert c_out.shape == (n,)
    # Later elements in linspace plasticize while first element is elastic
    assert epsp_out[0] == 0.0
    assert epsp_out[-1] > 0.0


# ============================================================================
# 8. Shell Element Stress Update & Thickness Thinning
# ============================================================================

def test_law66_shell_elastic_and_plastic_update():
    """Verify shell element plane stress update: plane-stress Hooke's law, von Mises return, thinning."""
    mat = build_law66(
        e=210000.0,
        nu=0.3,
        rho0=7.8e-6,
        sigy=350.0,
        curve_c=([0.0, 0.1], [350.0, 450.0]),
        curve_t=([0.0, 0.1], [350.0, 450.0]),
    )

    # Shell Voigt: [xx, yy, xy]
    sig = np.zeros((1, 3))
    # Elastic strain below yield
    deps = np.array([[0.0005, 0.0, 0.0]])
    epsp = np.zeros(1)
    extra = {
        "uvar66": np.zeros((1, 8)),
        "thk": np.array([1.5]),
    }

    sig_out, epsp_out = shell_update(mat, sig, deps, epsp=epsp, dt=1e-6, extra=extra)

    assert epsp_out[0] == 0.0
    # In plane stress: sigma_xx = (E / (1 - nu^2)) * deps_xx
    c_plane = 210000.0 / (1.0 - 0.3 ** 2)
    assert math.isclose(sig_out[0, 0], c_plane * 0.0005, rel_tol=1e-4)
    assert math.isclose(sig_out[0, 1], 0.3 * c_plane * 0.0005, rel_tol=1e-4)
    # Shell thickness was thinned by elastic de_zz
    # de_zz = -(nu / (1 - nu)) * (deps_xx + deps_yy) = -(0.3 / 0.7) * 0.0005 ~ -0.000214
    expected_thk = 1.5 * (1.0 - (0.3 / 0.7) * 0.0005)
    assert math.isclose(extra["thk"][0], expected_thk, rel_tol=1e-4)

    # Plastic step: deps_xx = 0.005
    deps_plas = np.array([[0.005, 0.0, 0.0]])
    sig_plas, epsp_plas = shell_update(mat, sig_out, deps_plas, epsp=epsp_out, dt=1e-6, extra=extra)

    assert epsp_plas[0] > 0.0
    # Plane stress von Mises norm: sqrt(s_xx^2 + s_yy^2 - s_xx*s_yy + 3*s_xy^2)
    sxx, syy, sxy = sig_plas[0, 0], sig_plas[0, 1], sig_plas[0, 2]
    svm = math.sqrt(sxx ** 2 + syy ** 2 - sxx * syy + 3.0 * sxy ** 2)
    assert math.isclose(svm, 350.0 + epsp_plas[0] * 1000.0, rel_tol=1e-2)


# ============================================================================
# 9. Sound Speeds: Solid & Shell Formulations
# ============================================================================

def test_law66_sound_speed_solid_and_shell():
    """Verify sound speed calculations for solid and shell elements."""
    mat = build_law66(
        e=200000.0,
        ec=160000.0,
        nu=0.25,
        rho0=2.5e-6,
    )

    c_sol = sound_speed_solid(mat)
    c_shl = sound_speed_shell(mat)
    c_gen = sound_speed(mat)

    # Solid sound speed uses max(c1t, c1c) and gt:
    # nu_denom = 1 - 2*0.25 = 0.5
    # c1t = 200000 / (3 * 0.5) = 133333.33
    # gt = 0.5 * 200000 / 1.25 = 80000.0
    # c_solid = sqrt((133333.33 + 4/3 * 80000) / 2.5e-6) = sqrt(240000 / 2.5e-6) = sqrt(9.6e10) ~ 309838.67
    expected_c_sol = math.sqrt((133333.33333333334 + (4.0 / 3.0) * 80000.0) / 2.5e-6)
    assert math.isclose(c_sol, expected_c_sol, rel_tol=1e-5)
    assert math.isclose(c_gen, expected_c_sol, rel_tol=1e-5)

    # Shell sound speed uses a11t:
    # a11t = 200000 / (1 - 0.25^2) = 213333.33
    # c_shell = sqrt(213333.33 / 2.5e-6) = sqrt(8.5333e10) ~ 292118.69
    expected_c_shl = math.sqrt(213333.33333333334 / 2.5e-6)
    assert math.isclose(c_shl, expected_c_shl, rel_tol=1e-5)


# ============================================================================
# 10. Algorithmic Consistent Tangents (Solid & Shell)
# ============================================================================

def test_law66_consistent_solid_tangent_elastic_and_plastic():
    """Verify consistent solid tangent tensor for elastic and plastic return states."""
    mat = build_law66(
        e=200000.0,
        nu=0.3,
        rho0=7.8e-6,
        sigy=300.0,
        chard=0.0,
    )

    # 1. Pure elastic state: epsp_incr is None or zero -> standard isotropic C
    sig = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    C_el = consistent_solid_tangent(mat, sig)

    assert C_el.shape == (6, 6)
    # Symmetric
    np.testing.assert_allclose(C_el, C_el.T, rtol=1e-12)
    # Hooke matrix values
    nu_denom = (1.0 + 0.3) * (1.0 - 2.0 * 0.3)
    c11 = 200000.0 * 0.7 / nu_denom
    c12 = 200000.0 * 0.3 / nu_denom
    g = 200000.0 / (2.0 * 1.3)
    assert math.isclose(C_el[0, 0], c11, rel_tol=1e-5)
    assert math.isclose(C_el[0, 1], c12, rel_tol=1e-5)
    assert math.isclose(C_el[3, 3], g, rel_tol=1e-5)

    # 2. Plastic state: epsp_incr > 0 -> algorithmic tangent D
    epsp = np.array([0.01])
    epsp_incr = np.array([0.001])
    sig_plas = np.array([300.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    D_plas = consistent_solid_tangent(mat, sig_plas, epsp=epsp, epsp_incr=epsp_incr)
    assert D_plas.shape == (6, 6)
    # Symmetric
    np.testing.assert_allclose(D_plas, D_plas.T, rtol=1e-10)
    # Plastic softening: D[0, 0] must be strictly smaller than elastic C[0, 0]
    assert D_plas[0, 0] < C_el[0, 0]


def test_law66_consistent_shell_tangent_elastic_and_plastic():
    """Verify consistent shell tangent tensor for membrane plane stress states."""
    mat = build_law66(
        e=210000.0,
        nu=0.3,
        rho0=7.8e-6,
        sigy=350.0,
    )

    # 1. Elastic membrane tangent
    C_mem = shell_membrane_tangent(mat)
    assert C_mem.shape == (3, 3)
    np.testing.assert_allclose(C_mem, C_mem.T, rtol=1e-12)
    c_xx = 210000.0 / (1.0 - 0.3 ** 2)
    assert math.isclose(C_mem[0, 0], c_xx, rel_tol=1e-5)
    assert math.isclose(C_mem[0, 1], 0.3 * c_xx, rel_tol=1e-5)
    assert math.isclose(C_mem[2, 2], 210000.0 / (2.0 * 1.3), rel_tol=1e-5)

    # 2. Plastic shell tangent
    sig = np.array([350.0, 0.0, 0.0])
    epsp = np.array([0.02])
    epsp_incr = np.array([0.001])

    D_shell = consistent_shell_tangent(mat, sig, epsp=epsp, epsp_incr=epsp_incr)
    assert D_shell.shape == (3, 3)
    # Plastic softening along tensile loading direction
    assert D_shell[0, 0] < C_mem[0, 0]


# ============================================================================
# 11. Starter Curve Resolution & pyradioss Integration
# ============================================================================

def test_law66_starter_curve_resolution():
    """Verify resolve() hooks /FUNCT curve references from Model.functions into Law66Params."""
    from pyradioss.model import Model
    from pyradioss.common.tables import FunctTable

    c1 = FunctTable(101, [0.0, 0.05, 0.1], [250.0, 300.0, 350.0])
    c2 = FunctTable(102, [0.0, 0.05, 0.1], [180.0, 220.0, 260.0])

    model = Model()
    model.functions[101] = c1
    model.functions[102] = c2

    mat = build_law66(
        fun_a1=101,  # compression curve
        fun_a2=102,  # tension curve
    )

    resolve(mat, model)
    p = mat.params["_obj"]
    assert p.curve_c is c1
    assert p.curve_t is c2


def test_law66_materials_module_dispatch():
    """Verify law 66 dispatch hooks in pyradioss.materials."""
    mat = build_law66(
        e=200000.0,
        nu=0.3,
        rho0=7.8e-6,
        sigy=300.0,
    )

    # 1. extra_shapes
    shapes_solid = materials.extra_shapes(mat, nip=None)
    assert "uvar66" in shapes_solid
    assert shapes_solid["uvar66"] == (8,)

    shapes_shell = materials.extra_shapes(mat, nip=5)
    assert "uvar66" in shapes_shell
    assert shapes_shell["uvar66"] == (5, 8)
    assert "thk" in shapes_shell
    assert shapes_shell["thk"] == (5,)

    # 2. needs_env
    assert materials.needs_env(mat) is True

    # 3. solid_update through pyradioss.materials
    sig = np.zeros((1, 6))
    deps = np.array([[0.0001, -0.00003, -0.00003, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)
    extra = {"uvar66": np.zeros((1, 8))}
    s_out, ep_out, c_out = materials.solid_update(mat, sig, deps, epsp=epsp, dt=1e-6, extra=extra)
    assert math.isclose(s_out[0, 0], 20.0, rel_tol=1e-3)

    # 4. shell_update through pyradioss.materials
    sig_sh = np.zeros((1, 3))
    deps_sh = np.array([[0.0001, 0.0, 0.0]])
    extra_sh = {"uvar66": np.zeros((1, 8)), "thk": np.array([1.0])}
    s_sh_out, ep_sh_out = materials.shell_update(mat, sig_sh, deps_sh, epsp=epsp, dt=1e-6, extra=extra_sh)
    assert s_sh_out.shape == (1, 3)

    # 5. sound_speed through pyradioss.materials
    c_val = materials.sound_speed(mat)
    assert c_val > 0.0

    # 6. consistent tangents through pyradioss.materials
    D_sol = materials.solid_tangent(mat, sig, epsp=epsp)
    assert D_sol.shape == (1, 6, 6)
    D_sh = materials.shell_layer_tangent(mat, sig_sh, epsp=epsp)
    assert D_sh.shape == (1, 3, 3)
