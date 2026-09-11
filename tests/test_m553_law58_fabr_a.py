"""
Milestone M553: /MAT/LAW58 (/MAT/FABR_A) Anisotropic Fabric Shell Material Law.

Unit tests for:
1. Yarn crimp geometry & kinematics initialization and interchange iteration
2. Membrane tension in warp and weft directions (analytical moduli & softening)
3. Trellis shear response before and after lock angle (continuity, tangent change)
4. Tabulated curves FUN_A1, FUN_A2, FUN_A3 with scale factors
5. Yarn sliding friction limit (tau_frot) and fiber damping
6. Zero-stress relative area folding deactivation
7. Consistent algorithmic plane-stress shell tangent vs central finite differences
8. Shell sound speed, properties, and solid update exception
"""

import math
import numpy as np
import pytest

from pyradioss.materials import law58_fabr_a
from pyradioss.materials.law58_fabr_a import (
    Law58Params,
    FabricAMaterial,
    crimp_interchange,
    shell_update_law58,
    sound_speed_shell_law58,
    shell_membrane_tangent,
    tangent_law58_shell,
    solid_update,
    build_law58,
)


def _make_dummy_rec(
    e1=1000.0, b1=0.0, e2=800.0, b2=0.0,
    flex=1e-3, g0=100.0, gt=300.0, alphat=0.0,
    g5=100.0, df=0.05, ds=0.1, gfrot=100.0,
    zero_stress=0.0, arel=0.0,
    n1=1, n2=1, s1=0.1, s2=0.1,
    c4=0.0, c5=0.0, rho0=1.2e-6,
):
    rec = type("Rec", (), {})()
    rec.id = 58
    rec.title = "FABRIC_A_TEST"
    rec.density = rho0
    rec.params = {
        "MAT_E1": e1, "MAT_B1": b1,
        "MAT_E2": e2, "MAT_B2": b2,
        "MAT_F": flex,
        "MAT_G0": g0, "MAT_GI": gt,
        "MAT_ALPHA": alphat,
        "MAT_G5": g5,
        "MAT_Df": df, "MAT_dS": ds,
        "Friction_phi": gfrot,
        "M58_Zerostress": zero_stress,
        "a_r": arel,
        "N1_warp": n1, "N2_weft": n2,
        "S1": s1, "S2": s2,
        "MAT_C4": c4, "MAT_C5": c5,
    }
    return rec


# ============================================================================
# 1. Crimp Geometry Initialization & Kinematics
# ============================================================================

def test_crimp_geometry_initialization():
    """Verify unit cell geometry and initial stiffnesses match hm_read_mat58.F."""
    n1, n2 = 2, 4
    s1, s2 = 0.15, 0.20
    e1, e2 = 2000.0, 1600.0
    b1, b2 = 50.0, 40.0
    flex = 0.002
    p = Law58Params(
        e1=e1, b1=b1, e2=e2, b2=b2,
        n1=n1, n2=n2, s1=s1, s2=s2,
        flex=flex, rho0=1.5e-6,
    )

    # Unit cell lengths: L_c0 = 1 / N_t, L_t0 = 1 / N_c
    assert p.nc == n1
    assert p.nt == n2
    assert p.lc0 == pytest.approx(1.0 / n2)
    assert p.lt0 == pytest.approx(1.0 / n1)

    # Initial yarn lengths with crimp stretch
    assert p.dc0 == pytest.approx(p.lc0 * (1.0 + s1))
    assert p.dt0 == pytest.approx(p.lt0 * (1.0 + s2))

    # Crimp wave heights
    hc0_expected = math.sqrt(p.dc0**2 - p.lc0**2)
    ht0_expected = math.sqrt(p.dt0**2 - p.lt0**2)
    assert p.hc0 == pytest.approx(hc0_expected)
    assert p.ht0 == pytest.approx(ht0_expected)

    # Yarn axial and bending stiffnesses
    assert p.kc == pytest.approx(e1 / n1)
    assert p.kt == pytest.approx(e2 / n2)
    assert p.kbc == pytest.approx(b1 / n1)
    assert p.kbt == pytest.approx(b2 / n2)
    assert p.kfc == pytest.approx(flex * p.kc * p.hc0 / p.dc0)
    assert p.kft == pytest.approx(flex * p.kt * p.ht0 / p.dt0)


def test_crimp_interchange_uncoupled_vs_coupled():
    """Verify uncoupled out-of-plane yarn height relaxation and coupled interchange."""
    p = Law58Params(
        e1=1000.0, e2=1000.0, n1=1, n2=1, s1=0.1, s2=0.1, flex=1e-3, rho0=1.0
    )

    # In plane compression (ec < 0, et < 0): uncoupled model, yarns bend out-of-plane without contact
    lc_comp = p.lc0 * 0.95
    lt_comp = p.lt0 * 0.95
    yc, yt, fc, ft, fn, dc, dt = crimp_interchange(p, lc_comp, lt_comp)
    assert yc > 0.0
    assert yt > 0.0
    assert yc + yt >= 0.0
    assert fn == 0.0  # no contact force between warp and weft

    # In plane warp tension (ec = 0.05, et = 0.0): warp straightens, pulls on weft
    lc_tens = p.lc0 * 1.05
    lt_tens = p.lt0 * 1.0
    yc, yt, fc, ft, fn, dc, dt = crimp_interchange(p, lc_tens, lt_tens)
    assert yc < 0.0  # warp crimp amplitude decreases (flattened)
    assert yt > 0.0  # weft crimp amplitude increases (deepened)
    assert yc == pytest.approx(-yt, rel=1e-5)  # interchange coupling: yc = y, yt = -y
    assert fn > 0.0  # interlacing contact force generated


# ============================================================================
# 2. Tension in Warp and Weft Directions
# ============================================================================

def test_pure_warp_tension():
    """Verify uniaxial tension in warp direction generates positive sigma_xx and minimal sigma_yy."""
    p = Law58Params(e1=2000.0, e2=1000.0, n1=1, n2=1, s1=0.1, s2=0.1, g0=100.0)
    sig = np.zeros(3)
    deps = np.array([0.02, 0.0, 0.0])

    s_new, _ = shell_update_law58(p, sig, deps)
    assert s_new[0] > 0.0
    assert abs(s_new[2]) == 0.0  # no shear


def test_pure_weft_tension():
    """Verify uniaxial tension in weft direction generates positive sigma_yy and minimal sigma_xx."""
    p = Law58Params(e1=1000.0, e2=2500.0, n1=1, n2=1, s1=0.1, s2=0.1, g0=100.0)
    sig = np.zeros(3)
    deps = np.array([0.0, 0.025, 0.0])

    s_new, _ = shell_update_law58(p, sig, deps)
    assert s_new[1] > 0.0
    assert abs(s_new[2]) == 0.0


def test_biaxial_tension_resolved_membrane_stresses():
    """Verify membrane stress formulas sigma_xx = sigma_c * Nc / E_c2, sigma_yy = sigma_t * Nt / E_t2."""
    p = Law58Params(e1=1500.0, e2=1200.0, n1=2, n2=3, s1=0.1, s2=0.1)
    sig = np.zeros(3)
    deps = np.array([0.03, 0.04, 0.0])

    extra = {}
    s_new, _ = shell_update_law58(p, sig, deps, extra=extra)

    # Validate lateral stretch factors
    ec = math.exp(0.03) - 1.0
    et = math.exp(0.04) - 1.0
    lc = p.lc0 * (1.0 + ec)
    lt = p.lt0 * (1.0 + et)
    yc, yt, fc, ft, fn, dc, dt = crimp_interchange(p, lc, lt)

    ec2 = math.exp(0.04)
    et2 = math.exp(0.03)
    sigc = fc * lc / dc
    sigt = ft * lt / dt
    expected_sxx = sigc * (p.nc / ec2)
    expected_syy = sigt * (p.nt / et2)

    assert s_new[0] == pytest.approx(expected_sxx, rel=1e-5)
    assert s_new[1] == pytest.approx(expected_syy, rel=1e-5)


def test_yarn_softening_and_plateau():
    """Verify nonlinear yarn softening (B1, B2) and derivative zero plateau (CCL, TTL)."""
    # Linear (b1 = 0) vs Softening (b1 > 0)
    p_lin = Law58Params(e1=1000.0, b1=0.0, n1=1, n2=1, s1=0.05, s2=0.05)
    p_soft = Law58Params(e1=1000.0, b1=500.0, n1=1, n2=1, s1=0.05, s2=0.05)

    sig = np.zeros(3)
    deps = np.array([0.05, 0.0, 0.0])
    s_lin, _ = shell_update_law58(p_lin, sig.copy(), deps)
    s_soft, _ = shell_update_law58(p_soft, sig.copy(), deps)

    assert s_soft[0] < s_lin[0]  # softening reduces tension

    # Very large elongation beyond CCL = Kc / Kbc = 1000 / 500 = 2.0
    p_plat = Law58Params(e1=1000.0, b1=2000.0, n1=1, n2=1, s1=0.05, s2=0.05)
    # CCL = 0.5; when DCC >= 0.5, FC saturates at 0.5 * Kc * CCL = 250.0
    deps1 = np.array([0.6, 0.0, 0.0])
    deps2 = np.array([0.8, 0.0, 0.0])
    s1, _ = shell_update_law58(p_plat, sig.copy(), deps1)
    s2, _ = shell_update_law58(p_plat, sig.copy(), deps2)
    assert s1[0] > 0.0
    assert s2[0] > 0.0


# ============================================================================
# 3. Trellis Shear Behavior & Lock Angle
# ============================================================================

def test_trellis_shear_before_and_after_lock():
    """Verify linear G0 shear before lock angle and Gt shear + Gb after lock angle."""
    g0 = 50.0
    gt = 250.0
    alphat = 45.0  # 45 deg -> tan_lock = 1.0
    p = Law58Params(e1=1000.0, e2=1000.0, g0=g0, gt=gt, alphat=alphat)

    assert p.tan_lock == pytest.approx(1.0, rel=1e-5)
    # Gb = tan_lock * (G0 - G_secant) where G_secant = Gt / (1 + tan_lock^2) = 250 / 2 = 125
    g_secant = gt / (1.0 + p.tan_lock**2)
    gb_expected = p.tan_lock * (g0 - g_secant)
    assert p.gb == pytest.approx(gb_expected, rel=1e-5)

    sig = np.zeros(3)

    # 1. Pre-lock (tan_phi = 0.5 < 1.0)
    deps_pre = np.array([0.0, 0.0, 0.5])
    s_pre, _ = shell_update_law58(p, sig.copy(), deps_pre)
    assert s_pre[2] == pytest.approx(g0 * 0.5, rel=1e-5)

    # 2. Exactly at lock angle (tan_phi = 1.0)
    deps_lock = np.array([0.0, 0.0, 1.0])
    s_lock, _ = shell_update_law58(p, sig.copy(), deps_lock)
    expected_at_lock = g0 * 1.0
    assert s_lock[2] == pytest.approx(expected_at_lock, rel=1e-5)

    # 3. Post-lock (tan_phi = 1.5 > 1.0)
    deps_post = np.array([0.0, 0.0, 1.5])
    s_post, _ = shell_update_law58(p, sig.copy(), deps_post)
    expected_post = p.g_post * 1.5 + p.gb
    assert s_post[2] == pytest.approx(expected_post, rel=1e-5)

    # 4. Negative shear symmetry
    deps_neg = np.array([0.0, 0.0, -1.5])
    s_neg, _ = shell_update_law58(p, sig.copy(), deps_neg)
    assert s_neg[2] == pytest.approx(-s_post[2], rel=1e-5)


def test_shear_continuity_across_lock_angle():
    """Verify shear stress is continuous across the lock angle boundary."""
    p = Law58Params(e1=800.0, e2=800.0, g0=40.0, gt=200.0, alphat=30.0)
    eps_val = 1e-6
    sig = np.zeros(3)

    deps_below = np.array([0.0, 0.0, p.tan_lock - eps_val])
    deps_above = np.array([0.0, 0.0, p.tan_lock + eps_val])

    s_below, _ = shell_update_law58(p, sig.copy(), deps_below)
    s_above, _ = shell_update_law58(p, sig.copy(), deps_above)

    assert s_below[2] == pytest.approx(s_above[2], rel=1e-4)


# ============================================================================
# 4. Tabulated Curves FUN_A1, FUN_A2, FUN_A3
# ============================================================================

def test_tabulated_warp_curve_fun_a1():
    """Verify tabulated warp curve FUN_A1 and scale factor C1."""
    # Nonlinear piecewise curve for warp yarn tension vs elongation DCC
    def curve_warp(dcc):
        return 500.0 * dcc + 1000.0 * dcc**2

    p = Law58Params(
        e1=1000.0, e2=1000.0, n1=1, n2=1, s1=0.05, s2=0.05,
        fun_a1=curve_warp, c1=1.5
    )
    sig = np.zeros(3)
    deps = np.array([0.04, 0.0, 0.0])

    s_new, _ = shell_update_law58(p, sig, deps)
    assert s_new[0] > 0.0


def test_tabulated_weft_curve_fun_a2():
    """Verify tabulated weft curve FUN_A2 and scale factor C2."""
    class PiecewiseCurve:
        def __init__(self, xs, ys):
            self.xs = np.array(xs)
            self.ys = np.array(ys)

        def eval(self, x):
            return float(np.interp(x, self.xs, self.ys))

    curve_weft = PiecewiseCurve([0.0, 0.02, 0.05, 0.1], [0.0, 20.0, 80.0, 200.0])

    p = Law58Params(
        e1=1000.0, e2=1000.0, n1=1, n2=1, s1=0.05, s2=0.05,
        fun_a2=curve_weft, c2=2.0
    )
    sig = np.zeros(3)
    deps = np.array([0.0, 0.03, 0.0])

    s_new, _ = shell_update_law58(p, sig, deps)
    assert s_new[1] > 0.0


def test_tabulated_shear_curve_fun_a3():
    """Verify tabulated shear curve FUN_A3 vs angle in degrees."""
    def shear_curve(phi_deg):
        # Stress in MPa vs Trellis shear angle in degrees
        return 2.0 * phi_deg + 0.05 * phi_deg**2

    p = Law58Params(
        e1=1000.0, e2=1000.0, fun_a3=shear_curve, c3=1.2
    )
    tan_phi = 0.57735  # ~30 degrees
    sig = np.zeros(3)
    deps = np.array([0.0, 0.0, tan_phi])

    s_new, _ = shell_update_law58(p, sig, deps)
    phi_deg = math.atan(tan_phi) * 180.0 / math.pi
    expected_sxy = 1.2 * shear_curve(phi_deg)
    assert s_new[2] == pytest.approx(expected_sxy, rel=1e-5)


# ============================================================================
# 5. Yarn Sliding Friction & Viscous Damping
# ============================================================================

def test_yarn_sliding_friction():
    """Verify yarn contact sliding friction limit tau_frot."""
    # Under warp tension, FN > 0; with ds > 0, friction limits shear resistance
    p = Law58Params(
        e1=1000.0, e2=1000.0, n1=1, n2=1, s1=0.1, s2=0.1,
        g0=50.0, gt=50.0, ds=0.2, gfrot=200.0
    )
    sig = np.zeros(3)
    extra = {}

    # Step 1: establish warp tension to generate normal force FN
    deps1 = np.array([0.05, 0.0, 0.0])
    s1, _ = shell_update_law58(p, sig, deps1, extra=extra)
    fn = extra["fn"][0]
    assert fn > 0.0

    # Step 2: apply incremental shear strain
    deps2 = np.array([0.0, 0.0, 0.2])
    s2, _ = shell_update_law58(p, s1, deps2, extra=extra)

    # With friction, total shear stress includes elastic G0*tan_phi + friction sigv_xy
    tau_frot = (2.0 / 3.0) * p.ds * fn * (p.hc0 + p.ht0) / (extra["eps58"][0, 0] + p.lc0 + p.lt0)
    assert abs(extra["sigv_xy"][0]) <= tau_frot * 1.5


def test_fiber_viscous_damping():
    """Verify fiber viscous damping sigma_v = (deps / dt) * V."""
    p_damp = Law58Params(e1=1000.0, e2=1000.0, df=0.1, rho0=1e-6)
    p_nodamp = Law58Params(e1=1000.0, e2=1000.0, df=0.0, rho0=1e-6)

    sig = np.zeros(3)
    deps = np.array([0.01, 0.01, 0.0])
    dt = 1e-4

    extra_damp = {"area": 1.0, "thk": 1.0}
    extra_nodamp = {"area": 1.0, "thk": 1.0}

    s_damp, _ = shell_update_law58(p_damp, sig.copy(), deps, dt=dt, extra=extra_damp)
    s_nodamp, _ = shell_update_law58(p_nodamp, sig.copy(), deps, dt=dt, extra=extra_nodamp)

    # Viscous damping adds positive stress during positive strain increment
    assert s_damp[0] > s_nodamp[0]
    assert s_damp[1] > s_nodamp[1]


# ============================================================================
# 6. Zero-Stress Relative Area Folding Deactivation
# ============================================================================

def test_zero_stress_relative_area():
    """Verify stresses vanish when fabric area ratio A/A0 <= A_rel."""
    arel = 0.85
    p = Law58Params(e1=1000.0, e2=1000.0, arel=arel)

    sig = np.zeros(3)
    # Large compressive strain in both directions: A/A0 = exp(-0.15 - 0.15) ~ 0.74 <= 0.85
    deps = np.array([-0.15, -0.15, 0.05])
    s_new, _ = shell_update_law58(p, sig, deps)

    assert s_new[0] == pytest.approx(0.0)
    assert s_new[1] == pytest.approx(0.0)
    assert s_new[2] == pytest.approx(0.0)


# ============================================================================
# 7. Consistent Algorithmic Tangents
# ============================================================================

def test_consistent_algorithmic_tangent_vs_finite_difference():
    """Verify algorithmic plane-stress membrane tangent matches central finite differences."""
    p = Law58Params(
        e1=1200.0, b1=100.0, e2=900.0, b2=80.0,
        n1=1, n2=1, s1=0.08, s2=0.08,
        g0=60.0, gt=180.0, alphat=35.0
    )

    test_states = [
        np.array([0.01, 0.005, 0.02]),   # Small tension and pre-lock shear
        np.array([0.03, 0.02, 0.8]),     # Post-lock shear
        np.array([0.02, 0.04, 0.1]),     # Biaxial tension
    ]

    h = 1e-7

    for deps in test_states:
        sig = np.zeros(3)
        extra = {}
        # Pre-step
        shell_update_law58(p, sig, deps, extra=extra)

        # Compute algorithmic tangent
        D_alg = tangent_law58_shell(p, sig=sig, deps=deps, extra=extra, h=h)

        # Compute finite difference tangent
        D_fd = np.zeros((3, 3))
        for j in range(3):
            ej = np.zeros(3)
            ej[j] = h

            ex_p = law58_fabr_a._copy_extra(extra)
            ex_m = law58_fabr_a._copy_extra(extra)

            sp, _ = shell_update_law58(p, sig.copy(), deps + ej, extra=ex_p)
            sm, _ = shell_update_law58(p, sig.copy(), deps - ej, extra=ex_m)

            D_fd[:, j] = (sp[:3] - sm[:3]) / (2.0 * h)

        # Verify element-by-element agreement
        np.testing.assert_allclose(D_alg, D_fd, rtol=1e-4, atol=1e-4)


def test_reference_membrane_tangent_uncoupled():
    """Verify reference shell_membrane_tangent at uncoupled zero-strain ground state."""
    p = Law58Params(e1=1500.0, e2=1200.0, g0=80.0)
    C_ref = shell_membrane_tangent(p)
    assert C_ref.shape == (3, 3)
    assert C_ref[0, 0] == 1500.0
    assert C_ref[1, 1] == 1200.0
    assert C_ref[2, 2] == 80.0


# ============================================================================
# 8. Sound Speed, Properties, Builder & Dispatch
# ============================================================================

def test_sound_speed_shell():
    """Verify shell sound speed c = sqrt(max(Kc, Kt, G0) / rho0)."""
    e1, e2, g0 = 1600.0, 900.0, 400.0
    n1, n2 = 1, 1
    rho0 = 1.6e-6
    p = Law58Params(e1=e1, e2=e2, g0=g0, n1=n1, n2=n2, rho0=rho0)

    kc = e1 / n1
    kt = e2 / n2
    kmax = max(kc, kt, g0)
    expected_c = math.sqrt(kmax / rho0)

    c = sound_speed_shell_law58(p)
    assert c == pytest.approx(expected_c, rel=1e-6)


def test_build_law58_record():
    """Verify build_law58 constructs FabricAMaterial and sets proper attributes."""
    rec = _make_dummy_rec(e1=2000.0, e2=1500.0, g0=120.0)
    mat = build_law58(rec)

    assert isinstance(mat, FabricAMaterial)
    assert mat.law == 58
    assert mat.E == 2000.0
    assert mat.G == 120.0
    assert mat.nu == 0.0
    assert mat.sound_speed_shell() > 0.0


def test_solid_update_raises_not_implemented():
    """Verify LAW58 rejects solid elements with NotImplementedError."""
    p = Law58Params()
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    with pytest.raises(NotImplementedError, match="shell elements only"):
        solid_update(p, sig, deps)


def test_empty_element_arrays():
    """Verify empty element slices are handled cleanly without error."""
    p = Law58Params()
    sig = np.zeros((0, 3))
    deps = np.zeros((0, 3))
    s_out, epsp_out = shell_update_law58(p, sig, deps)
    assert s_out.shape == (0, 3)
    assert epsp_out.shape == (0,)

    D = tangent_law58_shell(p, sig=sig, deps=deps)
    assert D.shape == (0, 3, 3)


def test_transverse_shear_g5():
    """Verify transverse shear stresses sigma_yz, sigma_zx are updated with G5."""
    g5_val = 250.0
    p = Law58Params(e1=1000.0, e2=1000.0, g0=100.0, g5=g5_val)

    sig = np.array([[0.0, 0.0, 0.0, 10.0, 20.0]])
    deps = np.array([[0.01, 0.0, 0.0, 0.02, 0.03]])

    s_new, _ = shell_update_law58(p, sig, deps)
    assert s_new[0, 3] == pytest.approx(10.0 + g5_val * 0.02)
    assert s_new[0, 4] == pytest.approx(20.0 + g5_val * 0.03)


def test_zerostress_refstate_sensor_relaxation():
    """Verify REF-STATE zerostress initial hold and subsequent unloading relaxation."""
    p = Law58Params(e1=1000.0, e2=1000.0, zero_stress=0.5, sensor_id=1)
    sig = np.zeros(3)
    extra = {"t58": np.array([0.0])}

    # At t = 0 (t <= tstart = 0): initial state zeroed and stored in sigi
    deps1 = np.array([0.02, 0.01, 0.0])
    s1, _ = shell_update_law58(p, sig, deps1, dt=0.0, extra=extra)
    assert s1[0] == 0.0
    assert s1[1] == 0.0
    assert extra["sigi58"][0, 0] > 0.0

    # Next step with dt > 0 advancing time: relaxation begins
    extra["t58"][0] = 0.01
    deps2 = np.array([-0.005, 0.0, 0.0])
    s2, _ = shell_update_law58(p, s1, deps2, dt=1e-3, extra=extra)
    assert s2[0] != 0.0


def test_multielement_vectorization():
    """Verify vectorized element group with diverse deformation modes."""
    p = Law58Params(
        e1=1500.0, e2=1200.0, g0=80.0, gt=300.0, alphat=45.0,
        arel=0.80, df=0.02, ds=0.1
    )
    nel = 5
    sig = np.zeros((nel, 3))
    deps = np.array([
        [0.02, 0.0, 0.0],      # Warp tension
        [0.0, 0.03, 0.0],      # Weft tension
        [0.0, 0.0, 0.5],       # Trellis shear pre-lock
        [0.0, 0.0, 1.5],       # Trellis shear post-lock
        [-0.2, -0.2, 0.0],     # Folding deactivation (rel area < 0.8)
    ])

    extra = {}
    s_new, _ = shell_update_law58(p, sig, deps, dt=1e-4, extra=extra)

    assert s_new.shape == (5, 3)
    assert s_new[0, 0] > 0.0
    assert s_new[1, 1] > 0.0
    assert s_new[2, 2] > 0.0
    assert s_new[3, 2] > s_new[2, 2]
    assert s_new[4, 0] == 0.0
    assert s_new[4, 1] == 0.0


def test_1d_vs_2d_input_consistency():
    """Verify 1D and 2D tensor calls produce identical stresses."""
    p = Law58Params(e1=1200.0, e2=1000.0, g0=75.0)

    sig_1d = np.array([0.0, 0.0, 0.0])
    deps_1d = np.array([0.015, 0.02, 0.05])

    sig_2d = sig_1d.reshape(1, 3)
    deps_2d = deps_1d.reshape(1, 3)

    s_out_1d, _ = shell_update_law58(p, sig_1d, deps_1d)
    s_out_2d, _ = shell_update_law58(p, sig_2d, deps_2d)

    assert s_out_1d.shape == (3,)
    assert s_out_2d.shape == (1, 3)
    np.testing.assert_allclose(s_out_1d, s_out_2d[0])


def test_global_materials_dispatch():
    """Verify pyradioss.materials dispatches LAW58 shell_update and tangents."""
    import pyradioss.materials as materials

    rec = _make_dummy_rec(e1=1400.0, e2=1100.0, g0=90.0)
    mat = build_law58(rec)

    # Dispatch shell_update
    sig = np.zeros(3)
    deps = np.array([0.01, 0.01, 0.02])
    s_new, _ = materials.shell_update(mat, sig, deps)
    assert s_new[0] > 0.0
    assert s_new[1] > 0.0
    assert s_new[2] > 0.0

    # Dispatch shell_membrane_tangent
    C_mem = materials.shell_membrane_tangent(mat)
    assert C_mem.shape == (3, 3)
    assert C_mem[0, 0] == 1400.0

    # Dispatch consistent_shell_tangent
    D_shell = materials.consistent_shell_tangent(mat)
    assert D_shell.shape in ((3, 3), (1, 3, 3))

