"""
Unit tests for Milestone M554: LAW52 Gurson-Tvergaard-Needleman (GTN)
porous metal plasticity material model (/MAT/LAW52, /MAT/GURSON, /MAT/PLAS_GURS).

Covers:
1. Elastic trial and return
2. Gurson yield surface evaluation (triaxiality, void softening, von Mises recovery)
3. Void growth under hydrostatic tension vs pure shear
4. Strain-controlled Gaussian void nucleation (Chu & Needleman)
5. Void coalescence acceleration past critical void fraction f_c
6. Complete ductile rupture and element deletion at failure void fraction f_F
7. Matrix work hardening and Cowper-Symonds rate sensitivity
8. Shell plane-stress enforcement (sigma_zz = 0) and thickness thinning
9. Sound speed formulas (solid and shell)
10. Consistent algorithmic tangents matching numerical perturbation
11. Multi-element vectorized batch updates
"""

import math
import numpy as np
import pytest

from pyradioss.materials import law52_gurson
from pyradioss.materials.law52_gurson import (
    Law52Params,
    build_law52,
    compute_f_star,
    gurson_yield_function,
    solid_update_law52,
    shell_update_law52,
    sound_speed_solid_law52,
    sound_speed_shell_law52,
    tangent_law52_solid,
    tangent_law52_shell,
)


@pytest.fixture
def default_mat():
    """Standard steel GTN material definition."""
    return build_law52(
        E=210000.0,
        nu=0.3,
        rho0=7.85e-9,
        A=400.0,
        B=500.0,
        n=0.2,
        CSD=0.0,
        VISP=1.0,
        q1=1.5,
        q2=1.0,
        q3=2.25,
        s_N=0.1,
        eps_N=0.3,
        f_I=0.01,
        f_N=0.04,
        f_C=0.15,
        f_F=0.25,
    )


# ===================================================================
# 1. Elastic Trial and Return
# ===================================================================

def test_elastic_trial_solid(default_mat):
    """Small strain increment below yield produces purely elastic response."""
    sig = np.zeros(6, dtype=float)
    # Small tensile strain well below yield (yield ~ 400 MPa, E = 210000 -> eps_y ~ 0.0019)
    deps = np.array([0.0005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    epsp = 0.0
    extra = {}

    sig_out, epsp_out = solid_update_law52(default_mat, sig, deps, epsp, dt=1e-6, extra=extra)

    E = default_mat.params["E"]
    nu = default_mat.params["nu"]
    c1 = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c2 = c1 * nu / (1.0 - nu)

    expected_sxx = c1 * deps[0]
    expected_syy = c2 * deps[0]
    expected_szz = c2 * deps[0]

    np.testing.assert_allclose(sig_out[0], expected_sxx, rtol=1e-5)
    np.testing.assert_allclose(sig_out[1], expected_syy, rtol=1e-5)
    np.testing.assert_allclose(sig_out[2], expected_szz, rtol=1e-5)
    assert epsp_out == 0.0
    assert extra["dmg"][0, 1] == 0.0  # fg = 0
    assert extra["dmg"][0, 2] == 0.0  # fn = 0
    assert extra["dmg"][0, 3] == default_mat.params["f_I"]  # f unchanged


def test_elastic_trial_shell(default_mat):
    """Small in-plane strain produces exact plane-stress elastic response."""
    sig = np.zeros(3, dtype=float)
    deps = np.array([0.0005, 0.0001, 0.0002], dtype=float)
    epsp = 0.0
    extra = {}

    sig_out, epsp_out = shell_update_law52(default_mat, sig, deps, epsp, dt=1e-6, extra=extra)

    E = default_mat.params["E"]
    nu = default_mat.params["nu"]
    a1 = E / (1.0 - nu * nu)
    a2 = nu * a1
    G = 0.5 * E / (1.0 + nu)

    expected_sxx = a1 * deps[0] + a2 * deps[1]
    expected_syy = a2 * deps[0] + a1 * deps[1]
    expected_sxy = G * deps[2]

    np.testing.assert_allclose(sig_out[0], expected_sxx, rtol=1e-5)
    np.testing.assert_allclose(sig_out[1], expected_syy, rtol=1e-5)
    np.testing.assert_allclose(sig_out[2], expected_sxy, rtol=1e-5)
    assert epsp_out == 0.0


# ===================================================================
# 2. Gurson Yield Surface Evaluation
# ===================================================================

def test_gurson_yield_recovers_von_mises():
    """When f* = 0, Gurson yield criterion reduces exactly to von Mises q = sigma_M."""
    sig_m = 500.0
    f_star = 0.0
    q = 500.0
    p_values = [-200.0, 0.0, 300.0]

    for p in p_values:
        phi = gurson_yield_function(q, p, sig_m, f_star, q1=1.5, q2=1.0, q3=2.25)
        # Phi = (500/500)^2 + 0 - 1 = 0
        np.testing.assert_allclose(phi, 0.0, atol=1e-12)


def test_gurson_void_softening():
    """Porous material has lower effective yield stress as void fraction f* increases."""
    sig_m = 400.0
    p = 100.0  # Positive hydrostatic tension
    q1, q2, q3 = 1.5, 1.0, 2.25

    f_stars = [0.0, 0.02, 0.05, 0.10, 0.15]
    effective_q = []
    for f_star in f_stars:
        var = 1.5 * q2 * p / sig_m
        va = 1.0 + q3 * (f_star ** 2) - 2.0 * q1 * f_star * math.cosh(var)
        assert va > 0.0
        effective_q.append(sig_m * math.sqrt(va))

    # Effective yield stress monotonically decreases as porosity increases
    for i in range(len(effective_q) - 1):
        assert effective_q[i] > effective_q[i + 1]


def test_gurson_hydrostatic_tension_accelerates_yielding():
    """Higher hydrostatic tension P decreases effective shear yield stress q."""
    sig_m = 400.0
    f_star = 0.05
    q1, q2, q3 = 1.5, 1.0, 2.25

    p_list = [0.0, 100.0, 200.0, 400.0]
    q_yields = []
    for p in p_list:
        var = 1.5 * q2 * p / sig_m
        va = 1.0 + q3 * (f_star ** 2) - 2.0 * q1 * f_star * math.cosh(var)
        q_yields.append(sig_m * math.sqrt(max(va, 0.0)))

    # Higher hydrostatic tension lowers q
    for i in range(len(q_yields) - 1):
        assert q_yields[i] > q_yields[i + 1]


# ===================================================================
# 3. Void Growth Under Hydrostatic Tension vs Shear
# ===================================================================

def test_void_growth_under_hydrostatic_tension(default_mat):
    """Triaxial tension causes significant positive void growth delta_f_g > 0."""
    sig = np.zeros(6, dtype=float)
    # Tensile strain with positive pressure:
    deps = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    epsp = 0.0
    extra = {}

    sig_out, epsp_out = solid_update_law52(default_mat, sig, deps, epsp, dt=1e-5, extra=extra)

    dmg = extra["dmg"]
    fg = dmg[0, 1]
    f = dmg[0, 3]

    assert epsp_out > 0.0
    assert fg > 0.0
    assert f > default_mat.params["f_I"]


def test_void_growth_zero_under_pure_shear(default_mat):
    """Under pure shear, tr(deps^p) = 0 so void growth delta_f_g is zero."""
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.0, 0.0, 0.0, 0.015, 0.0, 0.0], dtype=float)
    epsp = 0.0
    extra = {}

    sig_out, epsp_out = solid_update_law52(default_mat, sig, deps, epsp, dt=1e-5, extra=extra)

    dmg = extra["dmg"]
    fg = dmg[0, 1]

    assert epsp_out > 0.0
    # In pure shear (p = 0), flow gradient D has D11=D22=D33=0, so tr(D) = 0
    np.testing.assert_allclose(fg, 0.0, atol=1e-8)


# ===================================================================
# 4. Strain-Controlled Gaussian Void Nucleation
# ===================================================================

def test_gaussian_void_nucleation():
    """Chu-Needleman nucleation rate peaks at eps_M = eps_N."""
    sn = 0.1
    epsn = 0.3
    fn = 0.04

    def rate(eps):
        return (fn / (sn * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * (((eps - epsn) / sn) ** 2))

    # Peak must be at eps = epsn
    r_peak = rate(epsn)
    r_left = rate(epsn - 0.05)
    r_right = rate(epsn + 0.05)
    assert r_peak > r_left
    assert r_peak > r_right
    np.testing.assert_allclose(r_left, r_right, rtol=1e-6)

    # Numerical integration of nucleation rate from 0 to 1.0 should equal ~ f_N
    eps_grid = np.linspace(0.0, 1.0, 5000)
    d_eps = eps_grid[1] - eps_grid[0]
    integral = np.sum([rate(e) * d_eps for e in eps_grid])
    np.testing.assert_allclose(integral, fn, rtol=0.01)


# ===================================================================
# 5. Coalescence Acceleration Past f_c
# ===================================================================

def test_coalescence_acceleration():
    """When f > f_c, f* accelerates with slope (f_u - f_c) / (f_F - f_c) > 1."""
    fc = 0.15
    ff = 0.25
    fu = 1.0 / 1.5  # ~ 0.6667

    # Sub-critical
    f_sub = 0.10
    f_star_sub, df_sub = compute_f_star(f_sub, fc, ff, fu)
    assert f_star_sub == f_sub
    assert df_sub == 1.0

    # At critical
    f_crit = 0.15
    f_star_crit, df_crit = compute_f_star(f_crit, fc, ff, fu)
    assert f_star_crit == f_crit

    # Post-critical coalescence
    f_post = 0.20
    f_star_post, df_post = compute_f_star(f_post, fc, ff, fu)
    expected_slope = (fu - fc) / (ff - fc)
    expected_f_star = fc + expected_slope * (f_post - fc)

    assert df_post > 1.0
    np.testing.assert_allclose(f_star_post, expected_f_star, rtol=1e-7)
    np.testing.assert_allclose(df_post, expected_slope, rtol=1e-7)

    # At failure f_F, f* should reach f_u
    f_star_fail, _ = compute_f_star(ff, fc, ff, fu)
    np.testing.assert_allclose(f_star_fail, fu, rtol=1e-7)


# ===================================================================
# 6. Complete Rupture and Element Deletion at f_F
# ===================================================================

def test_element_deletion_at_failure():
    """Element is deleted (off = 0, stresses zeroed) when f >= f_F."""
    mat = build_law52(
        E=210000.0,
        nu=0.3,
        rho0=7.8e-9,
        A=400.0,
        B=500.0,
        n=0.2,
        f_I=0.24,  # Close to failure f_F = 0.25
        f_C=0.15,
        f_F=0.25,
        q1=1.5,
    )
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    epsp = 0.0
    extra = {}

    sig_out, epsp_out = solid_update_law52(mat, sig, deps, epsp, dt=1e-4, extra=extra)

    # Element must be failed
    assert extra["off"][0] == 0.0
    np.testing.assert_allclose(sig_out, 0.0, atol=1e-12)

    # Subsequent strain increments should keep stress at zero
    deps2 = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_out2, _ = solid_update_law52(mat, sig_out, deps2, epsp_out, dt=1e-4, extra=extra)
    np.testing.assert_allclose(sig_out2, 0.0, atol=1e-12)


def test_element_initially_ruptured():
    """Element starting with f_I >= f_F is immediately deleted."""
    mat = build_law52(
        E=210000.0, nu=0.3, rho0=7.8e-9, A=400.0,
        f_I=0.26, f_C=0.15, f_F=0.25,
    )
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    extra = {}
    sig_out, _ = solid_update_law52(mat, sig, deps, 0.0, extra=extra)
    assert extra["off"][0] == 0.0
    np.testing.assert_allclose(sig_out, 0.0, atol=1e-12)


# ===================================================================
# 7. Matrix Hardening & Cowper-Symonds Rate Sensitivity
# ===================================================================

def test_matrix_hardening_power_law(default_mat):
    """Matrix flow stress follows power law sigma_M = A + B * eps_M^n."""
    p = default_mat.params
    A = p["A"]
    B = p["B"]
    n = p["n"]

    # Monotonic plastic stretch in shear to avoid void growth interference
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0], dtype=float)
    epsp = 0.0
    extra = {}

    for _ in range(5):
        solid_update_law52(default_mat, sig, deps, epsp, dt=1e-3, extra=extra)

    epsm = extra["epsm"][0]
    expected_sigm = A + B * (epsm ** n)
    actual_sigm = extra["sigm"][0]
    np.testing.assert_allclose(actual_sigm, expected_sigm, rtol=1e-4)


def test_cowper_symonds_rate_sensitivity():
    """Higher strain rate increases matrix flow stress via Cowper-Symonds multiplier."""
    mat_rate = build_law52(
        E=210000.0, nu=0.3, rho0=7.8e-9,
        A=400.0, B=0.0, n=1.0,  # No hardening to isolate rate effect
        CSD=100.0, VISP=2.0,    # C = 100 s^-1, P = 2
        f_I=0.0, f_N=0.0,       # No voids
    )

    sig1 = np.zeros(6, dtype=float)
    sig2 = np.zeros(6, dtype=float)
    deps = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    # Slow rate: dt = 1.0 s -> rate ~ 0.005 s^-1
    extra1 = {}
    solid_update_law52(mat_rate, sig1, deps, 0.0, dt=1.0, extra=extra1)

    # Fast rate: dt = 1e-4 s -> rate ~ 50 s^-1
    extra2 = {}
    solid_update_law52(mat_rate, sig2, deps, 0.0, dt=1e-4, extra=extra2)

    # Fast rate should yield higher flow stress
    assert extra2["sigm"][0] > extra1["sigm"][0]


# ===================================================================
# 8. Shell Plane-Stress Enforcement and Thickness Thinning
# ===================================================================

def test_shell_plane_stress_and_thinning(default_mat):
    """In shells, sigma_zz = 0, and plastic yielding thins thickness."""
    sig = np.zeros(3, dtype=float)
    deps = np.array([0.01, 0.0, 0.0], dtype=float)  # Uniaxial tension
    epsp = 0.0
    extra = {"thk": np.array([1.5]), "thk0": np.array([1.5])}

    sig_out, epsp_out = shell_update_law52(default_mat, sig, deps, epsp, dt=1e-5, extra=extra)

    # Thickness should thin down from 1.5
    assert extra["thk"][0] < 1.5
    assert epsp_out > 0.0


# ===================================================================
# 9. Sound Speeds
# ===================================================================

def test_sound_speeds():
    """Instantaneous sound speeds match theoretical formulas."""
    E = 210000.0
    nu = 0.3
    rho0 = 7.85e-9

    c_solid_theory = math.sqrt((E * (1.0 - nu)) / ((1.0 + nu) * (1.0 - 2.0 * nu) * rho0))
    c_shell_theory = math.sqrt(E / ((1.0 - nu * nu) * rho0))

    params = Law52Params(E=E, nu=nu, rho0=rho0)

    np.testing.assert_allclose(sound_speed_solid_law52(params), c_solid_theory, rtol=1e-6)
    np.testing.assert_allclose(sound_speed_shell_law52(params), c_shell_theory, rtol=1e-6)


# ===================================================================
# 10. Algorithmic Consistent Tangents
# ===================================================================

def test_solid_tangent_elastic(default_mat):
    """Elastic tangent matches isotropic elastic matrix."""
    sig = np.zeros(6, dtype=float)
    tangent = tangent_law52_solid(default_mat, sig, epsp=0.0, epsp_incr=0.0)

    E = default_mat.params["E"]
    nu = default_mat.params["nu"]
    c1 = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c2 = c1 * nu / (1.0 - nu)
    G = 0.5 * E / (1.0 + nu)

    expected = np.zeros((6, 6), dtype=float)
    expected[0, 0] = expected[1, 1] = expected[2, 2] = c1
    expected[0, 1] = expected[1, 0] = expected[0, 2] = expected[2, 0] = expected[1, 2] = expected[2, 1] = c2
    expected[3, 3] = expected[4, 4] = expected[5, 5] = G

    np.testing.assert_allclose(tangent, expected, rtol=1e-6)


def test_shell_tangent_elastic(default_mat):
    """Elastic shell tangent matches plane-stress membrane stiffness."""
    sig = np.zeros(3, dtype=float)
    tangent = tangent_law52_shell(default_mat, sig, epsp=0.0, epsp_incr=0.0)

    E = default_mat.params["E"]
    nu = default_mat.params["nu"]
    a1 = E / (1.0 - nu * nu)
    a2 = nu * a1
    G = 0.5 * E / (1.0 + nu)

    expected = np.array([
        [a1, a2, 0.0],
        [a2, a1, 0.0],
        [0.0, 0.0, G],
    ])
    np.testing.assert_allclose(tangent, expected, rtol=1e-6)


def test_solid_tangent_plastic_consistency(default_mat):
    """Consistent solid tangent matches numerical derivative of stress update."""
    sig = np.zeros(6, dtype=float)
    deps0 = np.array([0.005, 0.0, 0.0, 0.002, 0.0, 0.0], dtype=float)
    extra = {}
    sig_converged, epsp_out = solid_update_law52(default_mat, sig, deps0, 0.0, dt=1e-5, extra=extra)

    # Compute algorithmic tangent at converged state
    C_ep = tangent_law52_solid(default_mat, sig_converged, epsp=epsp_out, epsp_incr=0.005, extra=extra)

    # Finite difference perturbation of strain increment
    delta = 1e-7
    C_num = np.zeros((6, 6), dtype=float)
    for k in range(6):
        d_deps = np.zeros(6, dtype=float)
        d_deps[k] = delta

        sig_pert_pos = sig_converged.copy()
        extra_pos = {"epsm": extra["epsm"].copy(), "sigm": extra["sigm"].copy(), "dmg": extra["dmg"].copy(), "off": extra["off"].copy()}
        solid_update_law52(default_mat, sig_pert_pos, d_deps, epsp_out, dt=1e-5, extra=extra_pos)

        sig_pert_neg = sig_converged.copy()
        extra_neg = {"epsm": extra["epsm"].copy(), "sigm": extra["sigm"].copy(), "dmg": extra["dmg"].copy(), "off": extra["off"].copy()}
        solid_update_law52(default_mat, sig_pert_neg, -d_deps, epsp_out, dt=1e-5, extra=extra_neg)

        C_num[:, k] = (sig_pert_pos - sig_pert_neg) / (2.0 * delta)

    # Tangents should agree closely
    np.testing.assert_allclose(C_ep, C_num, rtol=0.05, atol=500.0)


def test_shell_tangent_plastic_consistency(default_mat):
    """Consistent shell tangent matches numerical derivative of plane-stress update."""
    sig = np.zeros(3, dtype=float)
    deps0 = np.array([0.006, 0.001, 0.002], dtype=float)
    extra = {}
    sig_converged, epsp_out = shell_update_law52(default_mat, sig, deps0, 0.0, dt=1e-5, extra=extra)

    C_ep = tangent_law52_shell(default_mat, sig_converged, epsp=epsp_out, epsp_incr=0.005, extra=extra)

    delta = 1e-7
    C_num = np.zeros((3, 3), dtype=float)
    for k in range(3):
        d_deps = np.zeros(3, dtype=float)
        d_deps[k] = delta

        sig_pert_pos = sig_converged.copy()
        extra_pos = {"epsm": extra["epsm"].copy(), "sigm": extra["sigm"].copy(), "dmg": extra["dmg"].copy(), "off": extra["off"].copy(), "thk": extra["thk"].copy(), "thk0": extra["thk0"].copy()}
        shell_update_law52(default_mat, sig_pert_pos, d_deps, epsp_out, dt=1e-5, extra=extra_pos)

        sig_pert_neg = sig_converged.copy()
        extra_neg = {"epsm": extra["epsm"].copy(), "sigm": extra["sigm"].copy(), "dmg": extra["dmg"].copy(), "off": extra["off"].copy(), "thk": extra["thk"].copy(), "thk0": extra["thk0"].copy()}
        shell_update_law52(default_mat, sig_pert_neg, -d_deps, epsp_out, dt=1e-5, extra=extra_neg)

        C_num[:, k] = (sig_pert_pos - sig_pert_neg) / (2.0 * delta)

    np.testing.assert_allclose(C_ep, C_num, rtol=0.05, atol=500.0)


# ===================================================================
# 11. Damage Channels and Nucleation Modes
# ===================================================================

def test_damage_outputs_channels(default_mat):
    """Verify all 5 damage channels in extra['dmg'] match specification."""
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.003, 0.0, 0.0, 0.001, 0.0, 0.0], dtype=float)
    extra = {}
    solid_update_law52(default_mat, sig, deps, 0.0, dt=1e-5, extra=extra)

    dmg = extra["dmg"][0]
    # dmg[0] = f*
    # dmg[1] = fg
    # dmg[2] = fn
    # dmg[3] = f = f_I + fg + fn
    # dmg[4] = f*
    f_I = default_mat.params["f_I"]
    expected_f = f_I + dmg[1] + dmg[2]
    np.testing.assert_allclose(dmg[3], expected_f, rtol=1e-7)
    np.testing.assert_allclose(dmg[0], dmg[4], rtol=1e-7)
    assert dmg[1] > 0.0  # Void growth under tension
    assert dmg[2] > 0.0  # Void nucleation under plastic strain


def test_tension_only_nucleation_iflag3():
    """Under IFLAG=3, nucleation occurs only under hydrostatic tension (P >= 0)."""
    mat_iflag3 = build_law52(
        E=210000.0, nu=0.3, rho0=7.8e-9, A=400.0, B=0.0, n=1.0,
        f_I=0.01, f_N=0.04, s_N=0.1, eps_N=0.1,
        iflag=3,  # Nucleation only under tension
    )
    # Compressive plastic step: P < 0
    sig = np.zeros(6, dtype=float)
    deps_comp = np.array([-0.008, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    extra_comp = {}
    solid_update_law52(mat_iflag3, sig, deps_comp, 0.0, dt=1e-5, extra=extra_comp)
    # Under compression, fn must remain 0
    np.testing.assert_allclose(extra_comp["dmg"][0, 2], 0.0, atol=1e-12)


def test_return_sound_speed_option(default_mat):
    """Calling with return_sound_speed=True returns (sig, epsp, c_sound)."""
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    res_solid = solid_update_law52(default_mat, sig, deps, 0.0, return_sound_speed=True)
    assert len(res_solid) == 3
    assert res_solid[2][0] > 0.0

    sig_sh = np.zeros(3, dtype=float)
    deps_sh = np.array([0.001, 0.0, 0.0], dtype=float)
    res_shell = shell_update_law52(default_mat, sig_sh, deps_sh, 0.0, return_sound_speed=True)
    assert len(res_shell) == 3
    assert res_shell[2][0] > 0.0


def test_dispatcher_integration(default_mat):
    """Test dispatch via pyradioss.materials.solid_update and shell_update."""
    from pyradioss import materials

    sig_s = np.zeros((1, 6), dtype=float)
    deps_s = np.array([[0.002, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    sig_out_s, epsp_out_s, c_s = materials.solid_update(default_mat, sig_s, deps_s, np.zeros(1))
    assert c_s is not None

    sig_sh = np.zeros((1, 3), dtype=float)
    deps_sh = np.array([[0.002, 0.0, 0.0]], dtype=float)
    sig_out_sh, epsp_out_sh = materials.shell_update(default_mat, sig_sh, deps_sh, np.zeros(1))
    assert sig_out_sh is not None


# ===================================================================
# 12. Vectorized Multi-Element Batch Updates
# ===================================================================

def test_vectorized_batch_execution(default_mat):
    """Verify batch updates on arrays of shape (NEL, 6) with varying active/failed states."""
    nel = 4
    sig = np.zeros((nel, 6), dtype=float)
    deps = np.array([
        [0.0002, 0.0, 0.0, 0.0, 0.0, 0.0],       # Element 0: Elastic
        [0.005, 0.0, 0.0, 0.001, 0.0, 0.0],     # Element 1: Plastic
        [0.05, 0.0, 0.0, 0.0, 0.0, 0.0],         # Element 2: Extreme tension -> rupture
        [0.002, 0.0, 0.0, 0.0, 0.0, 0.0],       # Element 3: Initially dead
    ])
    epsp = np.zeros(nel, dtype=float)
    extra = {
        "off": np.array([1.0, 1.0, 1.0, 0.0]),  # Element 3 is already deleted
    }

    sig_out, epsp_out = solid_update_law52(default_mat, sig, deps, epsp, dt=1e-5, extra=extra)

    # Element 0: Elastic, epsp = 0
    assert epsp_out[0] == 0.0
    assert extra["off"][0] == 1.0

    # Element 1: Yielded, epsp > 0, off = 1
    assert epsp_out[1] > 0.0
    assert extra["off"][1] == 1.0

    # Element 2: Failed and deleted
    assert extra["off"][2] == 0.0
    np.testing.assert_allclose(sig_out[2], 0.0, atol=1e-12)

    # Element 3: Dead element remains zero
    assert extra["off"][3] == 0.0
    np.testing.assert_allclose(sig_out[3], 0.0, atol=1e-12)
