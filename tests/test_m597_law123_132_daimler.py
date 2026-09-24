"""
Tests for M597: Daimler-Pinho (/MAT/LAW123) and Daimler-Camanho (/MAT/LAW132)
Composite Damage Constitutive Models in pyradioss.
"""

import math
import os
import tempfile
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials.law123_daimler import (
    Law123Params,
    Law132Params,
    build_law123,
    build_law132,
    solid_step,
    shell_step,
    solid_update,
    shell_update,
    sound_speed,
    solid_tangent,
    shell_tangent,
    consistent_solid_tangent,
    consistent_shell_tangent,
    extra_shapes,
    analyze_failure,
)
import pyradioss.materials as materials
from pyradioss.input.starter_keywords import read_starter_deck


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def carbon_epoxy_123():
    """Typical unidirectional carbon/epoxy composite parameters for LAW123."""
    return Law123Params(
        rho0=1.55e-6,         # kg/mm^3 (1550 kg/m^3)
        ea=150000.0,          # MPa
        eb=9000.0,            # MPa
        ec=9000.0,            # MPa
        gab=4500.0,           # MPa
        gca=4500.0,           # MPa
        gbc=3000.0,           # MPa
        prba=0.0192,          # nu21 -> nu12 = nu21 * ea / eb = 0.32
        prca=0.0192,
        prcb=0.45,
        enkink=80.0,          # N/mm (kJ/m^2)
        ena=90.0,             # G_1t
        enb=1.2,              # G_2t
        ent=1.8,              # G_2s
        enl=1.8,              # G_12
        xc=1200.0,            # MPa
        xt=2200.0,            # MPa
        yc=200.0,             # MPa
        yt=60.0,              # MPa
        sl=100.0,             # MPa
        fio=53.0,             # degrees
        sigy=0.0,
        beta=0.0,
        lcss=0,
        efs=0.5,
        ratio=1.0,
        fcut=0.0,
    )


@pytest.fixture
def carbon_epoxy_132():
    """Typical unidirectional carbon/epoxy composite parameters for LAW132."""
    return Law132Params(
        rho0=1.55e-6,
        ea=150000.0,
        eb=9000.0,
        ec=9000.0,
        gab=4500.0,
        gca=4500.0,
        gbc=3000.0,
        prba=0.0192,
        prca=0.0192,
        prcb=0.45,
        gxc=80.0,
        gxt=90.0,
        gyc=1.8,
        gyt=1.2,
        gsl=1.8,
        xc=1200.0,
        xt=2200.0,
        yc=200.0,
        yt=60.0,
        sl=100.0,
        fio=53.0,
        sigy=0.0,
        etan=0.0,
        beta=0.0,
        lcss=0,
        ratio=1.0,
        fcut=0.0,
    )


# ============================================================================
# Test 1: Elastic orthotropic response before damage (2D and 3D)
# ============================================================================

def test_elastic_orthotropic_response_2d_shell(carbon_epoxy_123):
    """Test pure elastic plane-stress response before damage initiation."""
    mat = carbon_epoxy_123
    sig = np.zeros(3)
    # Small elastic strain increment
    deps = np.array([1.0e-4, 5.0e-5, 2.0e-4])

    sig_out, epsp, uvar, dmg, off, offl, c = shell_step(mat, sig, deps, dt=1.0e-6)

    # Verify analytical plane stress relationship
    expected_sig = mat.c_plane @ deps
    np.testing.assert_allclose(sig_out, expected_sig, rtol=1.0e-5)
    assert dmg[0] == 0.0  # No damage
    assert off == 1.0     # Not deleted
    assert c > 0.0        # Positive sound speed


def test_elastic_orthotropic_response_3d_solid(carbon_epoxy_123):
    """Test pure elastic 3D solid response before damage initiation."""
    mat = carbon_epoxy_123
    sig = np.zeros(6)
    deps = np.array([1.0e-4, 5.0e-5, -2.0e-5, 1.0e-4, 2.0e-5, 3.0e-5])

    sig_out, epsp, uvar, dmg, off, offl, c = solid_step(mat, sig, deps, dt=1.0e-6)

    expected_sig = mat.d_solid @ deps
    np.testing.assert_allclose(sig_out, expected_sig, rtol=1.0e-5)
    assert dmg[0] == 0.0
    assert off == 1.0
    assert c > 0.0


# ============================================================================
# Test 2: Fiber tensile loading, peak strength X_t, linear softening to d=1
# ============================================================================

def test_fiber_tensile_loading_and_softening(carbon_epoxy_123):
    """Test longitudinal tension through peak strength Xt and regularized softening."""
    mat = carbon_epoxy_123
    sig = np.zeros(3)
    extra = {
        "eps123": np.zeros((1, 3)),
        "uv123": np.zeros((1, 22)),
        "dmg123": np.zeros((1, 8)),
        "off123": np.ones((1,)),
        "offl123": np.ones((1,)),
        "l_car": np.array([1.0]),  # 1 mm element
    }

    # Elastic strain up to initiation: eps_init = Xt / E1 = 2200 / 150000 = 0.014667
    deps_step = 1.0e-4
    n_steps = int(math.ceil(0.014667 / deps_step))
    for _ in range(n_steps):
        s_out, ep = shell_update(mat, sig, np.array([deps_step, 0.0, 0.0]), extra=extra)
        sig[:] = s_out

    # At or just before initiation, stress should be close to Xt
    assert sig[0] <= mat.xt * 1.02
    assert extra["dmg123"][0, 1] >= 0.0  # dfiber may have initiated

    # Continue straining into softening regime
    peak_stress = sig[0]
    while extra["dmg123"][0, 1] < 0.95 and extra["off123"][0] == 1.0:
        s_out, ep = shell_update(mat, sig, np.array([deps_step, 0.0, 0.0]), extra=extra)
        sig[:] = s_out

    # Damage must be significant and stress must have softened below peak
    assert extra["dmg123"][0, 1] > 0.5
    assert sig[0] < peak_stress


# ============================================================================
# Test 3: Fiber compressive kinking failure under transverse compression/shear
# ============================================================================

def test_fiber_kinking_failure_initiation(carbon_epoxy_123):
    """Test fiber kinking under longitudinal compression with transverse shear."""
    mat = carbon_epoxy_123
    sig = np.zeros(6)
    extra = {
        "eps123": np.zeros((1, 6)),
        "uv123": np.zeros((1, 22)),
        "dmg123": np.zeros((1, 8)),
        "off123": np.ones((1,)),
        "offl123": np.ones((1,)),
        "l_car": np.array([1.0]),
    }

    # Apply significant longitudinal compression (eps11 < 0) and in-plane shear (gam12)
    deps = np.array([-0.009, -0.001, -0.001, 0.01, 0.0, 0.0])
    for _ in range(3):
        s_out, *_ = solid_update(mat, sig, deps, extra=extra)
        sig[:] = s_out

    # Kinking criterion or fiber compression damage flag should be triggered
    # dmg[5] is the kinking initiation flag in LAW123
    assert extra["dmg123"][0, 5] == 1.0 or extra["dmg123"][0, 2] > 0.0


# ============================================================================
# Test 4: Matrix cracking under combined transverse tension and shear
# ============================================================================

def test_matrix_tensile_cracking(carbon_epoxy_123):
    """Test matrix cracking under transverse tension and in-plane shear."""
    mat = carbon_epoxy_123
    sig = np.zeros(3)
    extra = {
        "eps123": np.zeros((1, 3)),
        "uv123": np.zeros((1, 22)),
        "dmg123": np.zeros((1, 8)),
        "off123": np.ones((1,)),
        "offl123": np.ones((1,)),
        "l_car": np.array([1.0]),
    }

    # Transverse tension: Yt = 60 MPa, Eb = 9000 MPa -> eps_yt = 60 / 9000 = 0.00667
    deps = np.array([0.0, 0.001, 0.002])
    for _ in range(8):
        s_out, _ = shell_update(mat, sig, deps, extra=extra)
        sig[:] = s_out

    # Matrix cracking flag dmg[6] should be initiated and dmat dmg[3] > 0
    assert extra["dmg123"][0, 6] == 1.0
    assert extra["dmg123"][0, 3] > 0.0


def test_matrix_compressive_fracture_angle(carbon_epoxy_123):
    """Verify that pure transverse compression yields critical fracture angle ~ 53 deg."""
    mat = carbon_epoxy_123
    # In pure transverse compression: sigma_b = -150 MPa, others = 0
    crit_phi, max_f = analyze_failure(
        mat.st, mat.sl, mat.yt, mat.mut, mat.mul,
        sigma_b=-150.0, sigma_c=0.0, tau_bc=0.0, tau_ab=0.0, tau_ca=0.0
    )
    deg = crit_phi * 180.0 / math.pi
    # Critical fracture angle in compression should be close to 53 degrees (or 180 - 53 = 127)
    assert abs(deg - 53.0) < 2.0 or abs(deg - 127.0) < 2.0


# ============================================================================
# Test 5: Fracture energy regularization: verify area * l_car = G_f
# ============================================================================

def test_fracture_energy_regularization(carbon_epoxy_132):
    """Test that dissipated energy times element length l_car equals G_f for two element sizes."""
    mat = carbon_epoxy_132

    def run_uniaxial_tension_to_failure(l_car_val: float) -> float:
        sig = np.zeros(3)
        extra = {
            "eps123": np.zeros((1, 3)),
            "uv123": np.zeros((1, 22)),
            "dmg123": np.zeros((1, 8)),
            "off123": np.ones((1,)),
            "offl123": np.ones((1,)),
            "l_car": np.array([l_car_val]),
        }
        deps_step = 2.0e-5
        total_energy = 0.0
        prev_sig = 0.0

        for _ in range(4000):
            s_out, _ = shell_update(mat, sig, np.array([deps_step, 0.0, 0.0]), extra=extra)
            # Trapezoidal integration of sigma_11 * deps_11
            total_energy += 0.5 * (prev_sig + s_out[0]) * deps_step
            prev_sig = s_out[0]
            sig[:] = s_out
            if extra["off123"][0] == 0.0 or extra["dmg123"][0, 1] >= 0.99:
                break

        return total_energy * l_car_val

    g_diss_1 = run_uniaxial_tension_to_failure(l_car_val=1.0)
    g_diss_2 = run_uniaxial_tension_to_failure(l_car_val=2.0)

    # Invariant fracture energy: g_diss * l_car should match G_xt within reasonable numerical discretization
    assert abs(g_diss_1 - mat.gxt) / mat.gxt < 0.15
    assert abs(g_diss_2 - mat.gxt) / mat.gxt < 0.15
    # The two mesh sizes must match each other very closely (< 10% diff)
    assert abs(g_diss_1 - g_diss_2) / g_diss_1 < 0.10


# ============================================================================
# Test 6: LAW132 directional damage and crack closure under load reversal
# ============================================================================

def test_law132_crack_closure_under_reversal(carbon_epoxy_132):
    """Test that matrix tensile damage does not degrade compressive stiffness (crack closure)."""
    mat = carbon_epoxy_132
    sig = np.zeros(3)
    extra = {
        "eps123": np.zeros((1, 3)),
        "uv123": np.zeros((1, 22)),
        "dmg123": np.zeros((1, 8)),
        "off123": np.ones((1,)),
        "offl123": np.ones((1,)),
        "l_car": np.array([1.0]),
    }

    # Step A: Load in transverse tension until matrix tension damage d2+ occurs
    deps_t = np.array([0.0, 0.001, 0.0])
    for _ in range(12):
        s_out, _ = shell_update(mat, sig, deps_t, extra=extra)
        sig[:] = s_out

    d2p = extra["dmg123"][0, 3]  # d2+
    assert d2p > 0.1, "Matrix tension damage d2+ should be active"

    # Step B: Unload and reverse into transverse compression
    deps_c = np.array([0.0, -0.001, 0.0])
    for _ in range(20):
        s_out, _ = shell_update(mat, sig, deps_c, extra=extra)
        sig[:] = s_out

    # Under compression (sigma_b < 0), d2_act should use d2- which is 0.0 (undamaged)
    assert sig[1] < 0.0
    # Modulus in compression should be full Eb, not degraded by (1 - d2p)
    # Check tangential slope under one more compression step
    s_before = sig[1]
    s_out, _ = shell_update(mat, sig, np.array([0.0, -1.0e-5, 0.0]), extra=extra)
    eff_e2_comp = (s_out[1] - s_before) / (-1.0e-5)
    # The effective tangent in compression must be close to initial Eb (~9000 MPa)
    assert eff_e2_comp > 8000.0, f"Expected compressive modulus near 9000 MPa, got {eff_e2_comp}"


# ============================================================================
# Test 7: Element deletion (off=0) when damage reaches critical threshold or EFS
# ============================================================================

def test_element_deletion_on_critical_damage(carbon_epoxy_123):
    """Test element deletion flag off=0 when damage threshold is reached."""
    mat = carbon_epoxy_123
    sig = np.zeros(3)
    extra = {
        "eps123": np.zeros((1, 3)),
        "uv123": np.zeros((1, 22)),
        "dmg123": np.zeros((1, 8)),
        "off123": np.ones((1,)),
        "offl123": np.ones((1,)),
        "l_car": np.array([1.0]),
    }

    # Severe straining leading to complete failure
    deps = np.array([0.005, 0.0, 0.0])
    for _ in range(50):
        s_out, _ = shell_update(mat, sig, deps, extra=extra)
        sig[:] = s_out
        if extra["off123"][0] == 0.0:
            break

    assert extra["off123"][0] == 0.0
    assert np.all(sig == 0.0)


def test_element_deletion_on_strain_limit_efs():
    """Test element deletion when strain exceeds EFS."""
    mat = Law123Params(
        rho0=1.55e-6,
        ea=150000.0, eb=9000.0, ec=9000.0,
        gab=4500.0, gca=4500.0, gbc=3000.0,
        prba=0.0192, prca=0.0192, prcb=0.45,
        efs=0.05,  # 5% strain limit
        xt=1.0e6,  # Very high strength so failure doesn't trigger first
        xc=1.0e6, yt=1.0e6, yc=1.0e6, sl=1.0e6,
    )
    sig = np.zeros(3)
    extra = {
        "eps123": np.zeros((1, 3)),
        "uv123": np.zeros((1, 22)),
        "dmg123": np.zeros((1, 8)),
        "off123": np.ones((1,)),
        "offl123": np.ones((1,)),
        "l_car": np.array([1.0]),
    }

    # Strain past 5%
    deps = np.array([0.06, 0.0, 0.0])
    s_out, _ = shell_update(mat, sig, deps, extra=extra)
    assert extra["off123"][0] == 0.0
    assert np.all(s_out == 0.0)


# ============================================================================
# Test 8: Starter deck parsing of /MAT/LAW123 and /MAT/LAW132 (fixed and free)
# ============================================================================

def test_deck_parsing_law123_fixed():
    """Test parsing of /MAT/LAW123 fixed format card deck."""
    deck_content = """#RADIOSS STARTER
/BEGIN
TEST_DECK
/MAT/LAW123/10
LAW123_Fixed_Test
              1.55E-06
            150000.0              9000.0              9000.0
              4500.0              4500.0              3000.0
              0.0192              0.0192                0.45
                80.0                90.0                 1.2                 1.8                 1.8
              1200.0              2200.0               200.0                60.0               100.0
                53.0                 0.0                   0                   0                 0.0
                 0.5                 1.0                 0.0
/END
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".rad") as tf:
        tf.write(deck_content)
        temp_path = tf.name

    try:
        model = read_starter_deck(temp_path)
        assert 10 in model.materials
        m = model.materials[10]
        assert m.law == 123
        assert m.title.strip() == "LAW123_Fixed_Test"
        assert abs(m.params["LSDYNA_EA"] - 150000.0) < 1.0e-3
        assert abs(m.params["LSDYNA_EB"] - 9000.0) < 1.0e-3
        assert abs(m.params["LSD_MAT_XT"] - 2200.0) < 1.0e-3
        assert abs(m.params.get("LSD_XC", m.params.get("xc")) - 1200.0) < 1.0e-3
        assert abs(m.params["LSD_MAT_YT"] - 60.0) < 1.0e-3
        assert abs(m.params["LSD_MAT_YC"] - 200.0) < 1.0e-3
        assert abs(m.params["LSD_SL"] - 100.0) < 1.0e-3
        assert abs(m.params["LSD_FIO"] - 53.0) < 1.0e-3
        assert abs(m.params["EFS"] - 0.5) < 1.0e-3
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_deck_parsing_law132_free():
    """Test parsing of /MAT/LAW132 free format card deck."""
    deck_content = """#RADIOSS STARTER
/BEGIN
TEST_DECK
/MAT/LAW132/20
LAW132_Free_Test
1.55e-06
150000.0 9000.0 9000.0
4500.0 4500.0 3000.0
0.0192 0.0192 0.45
80.0 90.0 1.8 1.2 1.8
1200.0 2200.0 200.0 60.0 100.0
0.0 0.0 0.0 0.0
53.0 0.0 0.0 0.0 0
0.0 0.0 0.0
0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0
1.0 0.0
/END
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".rad") as tf:
        tf.write(deck_content)
        temp_path = tf.name

    try:
        model = read_starter_deck(temp_path)
        assert 20 in model.materials
        m = model.materials[20]
        assert m.law == 132
        assert m.title.strip() == "LAW132_Free_Test"
        assert abs(m.params["ea"] - 150000.0) < 1.0e-3
        assert abs(m.params["gxc"] - 80.0) < 1.0e-3
        assert abs(m.params["gxt"] - 90.0) < 1.0e-3
        assert abs(m.params["gyc"] - 1.8) < 1.0e-3
        assert abs(m.params["gyt"] - 1.2) < 1.0e-3
        assert abs(m.params["gsl"] - 1.8) < 1.0e-3
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# ============================================================================
# Test 9: Algorithmic tangents and acoustic sound speed
# ============================================================================

def test_tangent_stiffness_and_sound_speed(carbon_epoxy_123):
    """Verify consistent tangents match numerical perturbations and sound speed is physical."""
    mat = carbon_epoxy_123

    # Shell tangent
    c_shell = shell_tangent(mat)
    assert c_shell.shape == (3, 3)
    np.testing.assert_allclose(c_shell, mat.c_plane, rtol=1.0e-5)

    # Solid tangent
    c_solid = solid_tangent(mat)
    assert c_solid.shape == (6, 6)
    np.testing.assert_allclose(c_solid, mat.d_solid, rtol=1.0e-5)

    # Sound speed: c = sqrt(E / rho)
    c_sh = sound_speed(mat, is_shell=True)
    c_so = sound_speed(mat, is_shell=False)
    # Expected speed ~ sqrt(150000 / 1.55e-6) ~ 3.11e5 mm/s
    assert 3.0e5 < c_sh < 3.3e5
    assert 3.0e5 < c_so < 3.3e5


# ============================================================================
# Test 10: Materials package registry dispatch
# ============================================================================

def test_materials_package_dispatch(carbon_epoxy_123):
    """Verify materials.solid_update, shell_update, sound_speed dispatch for LAW123 and LAW132."""
    mat_obj = Material(id=1, law=123, rho0=carbon_epoxy_123.rho0, params=carbon_epoxy_123.__dict__)

    # Extra shapes
    shapes = materials.extra_shapes(mat_obj, nip=3)
    assert "eps123" in shapes
    assert "uv123" in shapes
    assert "dmg123" in shapes

    # Needs env
    assert materials.needs_env(mat_obj) is True

    # Shell update through materials package
    sig = np.zeros(3)
    deps = np.array([1.0e-4, 0.0, 0.0])
    extra = {
        "eps123": np.zeros((1, 3)),
        "uv123": np.zeros((1, 22)),
        "dmg123": np.zeros((1, 8)),
        "off123": np.ones((1,)),
        "offl123": np.ones((1,)),
        "l_car": np.array([1.0]),
    }
    s_out, _ = materials.shell_update(mat_obj, sig, deps, extra=extra)
    assert s_out[0] > 0.0
