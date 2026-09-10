"""
Tests for /MAT/LAW25 (/MAT/COMP_PLAS, /MAT/COMPSH, /MAT/TSAI_WU, /MAT/CRASURV)
composite anisotropic plasticity and failure model for shell and solid elements.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.materials import law25_composite
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY, GenericMaterialRecord


# ============================================================================
# 1. Constructor & Parameter Parsing Tests
# ============================================================================

def test_law25_defaults():
    """Verify default constructor attributes and derived constants."""
    mat = law25_composite.build_law25(
        id=25,
        rho0=1.5e-9,
        E1=140000.0,
        E2=10000.0,
        nu12=0.3,
        G12=5000.0,
        G23=3000.0,
        G31=5000.0,
        sigyt1=1500.0,
        sigyc1=1200.0,
        sigyt2=50.0,
        sigyc2=200.0,
        sigt12=70.0,
        sigc12=70.0,
    )
    assert mat.id == 25
    assert mat.law == 25
    assert mat.rho0 == 1.5e-9
    assert mat.title == "LAW25_COMPOSITE"

    p = mat.params
    assert p["e11"] == 140000.0
    assert p["e22"] == 10000.0
    assert p["e33"] == 140000.0  # default max(e11, e22)
    assert p["nu12"] == 0.3
    assert p["g12"] == 5000.0
    assert p["g23"] == 3000.0
    assert p["g31"] == 5000.0

    # Derived constants: nu21 = nu12 * E2 / E1
    expected_nu21 = 0.3 * 10000.0 / 140000.0
    assert p["nu21"] == pytest.approx(expected_nu21)

    expected_detc = 1.0 - 0.3 * expected_nu21
    assert p["detc"] == pytest.approx(expected_detc)

    expected_c1 = 140000.0 / expected_detc
    assert p["C1"] == pytest.approx(expected_c1)

    # Sound speed: sqrt(max(C1, G12, G23, G31, E33) / rho0)
    expected_ssp = math.sqrt(expected_c1 / 1.5e-9)
    assert p["ssp"] == pytest.approx(expected_ssp)
    assert law25_composite.sound_speed(mat) == pytest.approx(expected_ssp)


def test_law25_tsaiwu_coefficients():
    """Verify exact Tsai-Wu coefficient derivation matching read_mat25_tsaiwu.F90:310-316."""
    yt1 = 1500.0
    yc1 = 1200.0
    yt2 = 50.0
    yc2 = 200.0
    yt12 = 70.0
    yc12 = 80.0
    alpha = 1.0

    mat = law25_composite.build_law25(
        sigyt1=yt1,
        sigyc1=yc1,
        sigyt2=yt2,
        sigyc2=yc2,
        sigt12=yt12,
        sigc12=yc12,
        alpha=alpha,
    )
    p = mat.params

    expected_f1 = 1.0 / yt1 - 1.0 / yc1
    expected_f2 = 1.0 / yt2 - 1.0 / yc2
    expected_f11 = 1.0 / (yt1 * yc1)
    expected_f22 = 1.0 / (yt2 * yc2)
    expected_f33 = 1.0 / (yt12 * yc12)
    expected_f12 = -alpha / (2.0 * math.sqrt(yt1 * yc1 * yt2 * yc2))

    assert p["F1"] == pytest.approx(expected_f1)
    assert p["F2"] == pytest.approx(expected_f2)
    assert p["F11"] == pytest.approx(expected_f11)
    assert p["F22"] == pytest.approx(expected_f22)
    assert p["F33"] == pytest.approx(expected_f33)
    assert p["F12"] == pytest.approx(expected_f12)


def test_law25_crasurv_and_aliases():
    """Verify parameter aliases and CRASURV parameters."""
    mat = law25_composite.build_crasurv(
        id=99,
        MAT_RHO=2.0e-9,
        MAT_EA=100000.0,
        MAT_EB=50000.0,
        MAT_EC=80000.0,
        MAT_PRAB=0.25,
        MAT_GAB=4000.0,
        MAT_GBC=2000.0,
        MAT_GCA=3000.0,
        MAT_Iflag=1,  # CRASURV
        MAT_b1_t=0.5,
        MAT_n1_t=0.8,
        MAT_SIG1max_t=2000.0,
        MAT_c1_t=0.05,
        MAT_EPS1_t1=0.01,
        MAT_EPS2_t1=0.05,
        MAT_SIGres_t1=100.0,
        MAT_Wmax_pt1=50.0,
        MAT_DAMAGE=0.95,
        MAT_EPST1=0.02,
        MAT_EPSM1=0.06,
        MAT_SRC=0.1,
        MAT_SRP=10.0,
    )
    p = mat.params
    assert mat.id == 99
    assert p["iform"] == 1
    assert p["iflag"] == 1
    assert p["e11"] == 100000.0
    assert p["e22"] == 50000.0
    assert p["e33"] == 80000.0
    assert p["b_1t"] == 0.5
    assert p["n_1t"] == 0.8
    assert p["sig_1maxt"] == 2000.0
    assert p["eps_1t1"] == 0.01
    assert p["eps_2t1"] == 0.05
    assert p["sig_rst1"] == 100.0
    assert p["dmax"] == 0.95
    assert p["c"] == 0.1
    assert p["epdr"] == 10.0


# ============================================================================
# 2. Sound Speed & Elastic Tangent Tests
# ============================================================================

def test_law25_sound_speed_custom_density():
    """Verify sound speed scaling with density."""
    mat = law25_composite.build_law25(
        rho0=1.0e-9,
        E1=100000.0,
        E2=100000.0,
        nu12=0.0,
        G12=25000.0,
    )
    c0 = law25_composite.sound_speed(mat)
    assert c0 == pytest.approx(math.sqrt(100000.0 / 1.0e-9))

    # Compressed density
    c_comp = law25_composite.sound_speed(mat, rho=2.0e-9)
    assert c_comp == pytest.approx(math.sqrt(100000.0 / 2.0e-9))


def test_law25_membrane_tangent_matrix():
    """Verify 3x3 plane-stress membrane tangent matrix."""
    mat = law25_composite.build_law25(
        E1=100000.0,
        E2=50000.0,
        nu12=0.2,
        G12=15000.0,
    )
    c_shell = law25_composite.shell_membrane_tangent(mat)
    assert c_shell.shape == (3, 3)

    nu21 = 0.2 * 50000.0 / 100000.0  # 0.1
    detc = 1.0 - 0.2 * 0.1  # 0.98
    expected_c11 = 100000.0 / detc
    expected_c22 = 50000.0 / detc
    expected_c12 = 0.1 * expected_c11

    assert c_shell[0, 0] == pytest.approx(expected_c11)
    assert c_shell[1, 1] == pytest.approx(expected_c22)
    assert c_shell[0, 1] == pytest.approx(expected_c12)
    assert c_shell[1, 0] == pytest.approx(expected_c12)
    assert c_shell[2, 2] == pytest.approx(15000.0)
    assert c_shell[0, 2] == 0.0
    assert c_shell[2, 0] == 0.0


def test_law25_solid_stiffness_matrix():
    """Verify 6x6 3D orthotropic stiffness matrix."""
    mat = law25_composite.build_law25(
        E1=100000.0,
        E2=50000.0,
        E3=30000.0,
        nu12=0.2,
        G12=15000.0,
        G23=8000.0,
        G31=10000.0,
    )
    c_solid = law25_composite.solid_stiffness_matrix(mat)
    assert c_solid.shape == (6, 6)
    assert c_solid[2, 2] == pytest.approx(30000.0)
    assert c_solid[3, 3] == pytest.approx(15000.0)
    assert c_solid[4, 4] == pytest.approx(8000.0)
    assert c_solid[5, 5] == pytest.approx(10000.0)


# ============================================================================
# 3. Shell Stress Update Tests
# ============================================================================

def test_law25_shell_elastic_increment():
    """Verify shell elastic response for small strain increment."""
    mat = law25_composite.build_law25(
        E1=100000.0,
        E2=20000.0,
        nu12=0.25,
        G12=10000.0,
        sigyt1=1000.0,
        sigyc1=800.0,
        sigyt2=100.0,
        sigyc2=150.0,
        sigt12=50.0,
    )
    sig = np.zeros(3)
    deps = np.array([1.0e-4, 0.0, 0.0])

    sig_new, epsp_new, c_val = law25_composite.shell_update(mat, sig, deps)

    c_mat = law25_composite.shell_membrane_tangent(mat)
    expected_s11 = c_mat[0, 0] * 1.0e-4
    expected_s22 = c_mat[1, 0] * 1.0e-4

    assert sig_new[0] == pytest.approx(expected_s11, rel=1e-5)
    assert sig_new[1] == pytest.approx(expected_s22, rel=1e-5)
    assert sig_new[2] == pytest.approx(0.0, abs=1e-10)
    assert epsp_new == pytest.approx(0.0)
    assert c_val == pytest.approx(law25_composite.sound_speed(mat))


def test_law25_shell_tsaiwu_plastic_return():
    """Verify Tsai-Wu yielding and plastic return in shells."""
    mat = law25_composite.build_law25(
        E1=100000.0,
        E2=100000.0,
        nu12=0.0,
        G12=50000.0,
        sigyt1=200.0,
        sigyc1=200.0,
        sigyt2=200.0,
        sigyc2=200.0,
        sigt12=100.0,
        sigc12=100.0,
        b=0.0,
        fmax=10.0,
    )
    sig = np.zeros(3)
    # Apply strain that causes elastic trial stress s11 = 400 > 200
    deps = np.array([0.004, 0.0, 0.0])
    extra: dict[str, Any] = {}

    sig_new, epsp_new, c_val = law25_composite.shell_update(mat, sig, deps, extra=extra)

    # Returned stress should be returned onto yield surface
    p = mat.params
    f1 = p["F1"]
    f2 = p["F2"]
    f11 = p["F11"]
    f22 = p["F22"]
    f33 = p["F33"]
    f12 = p["F12"]

    s1, s2, s12 = sig_new[0], sig_new[1], sig_new[2]
    wvec = f1 * s1 + f2 * s2 + f11 * s1**2 + f22 * s2**2 + f33 * s12**2 + 2.0 * f12 * s1 * s2

    assert wvec <= 1.05  # Within tolerance of yield surface
    assert epsp_new > 0.0
    assert extra["wpla25"][0] > 0.0


def test_law25_shell_unilateral_damage():
    """Verify unilateral damage: degrades tension, preserves compression."""
    mat = law25_composite.build_law25(
        E1=100000.0,
        E2=50000.0,
        nu12=0.0,
        G12=20000.0,
        epst1=0.001,
        epsm1=0.005,
        dmax=0.8,
        sigyt1=1000.0,  # High yield stress so purely elastic/damage
        sigyc1=1000.0,
    )
    extra: dict[str, Any] = {
        "stra25": np.zeros((1, 3)),
        "dmg25": np.zeros((1, 8)),
        "wpla25": np.zeros(1),
        "off25": np.ones(1),
    }

    # Step 1: Tension exceeding epst1 to accumulate damage
    sig = np.zeros(3)
    deps1 = np.array([0.003, 0.0, 0.0])
    sig1, _, _ = law25_composite.shell_update(mat, sig, deps1, extra=extra)

    dmg_1 = extra["dmg25"][0, 1]
    assert dmg_1 > 0.0  # Damage accumulated

    # Step 2: Tensile reload with damaged modulus
    deps2 = np.array([0.0001, 0.0, 0.0])
    sig2, _, _ = law25_composite.shell_update(mat, sig1, deps2, extra=extra)
    dsig_tens = sig2[0] - sig1[0]
    expected_tens_mod = 100000.0 * (1.0 - dmg_1)
    assert dsig_tens / 0.0001 == pytest.approx(expected_tens_mod, rel=1e-3)

    # Step 3: Compressive load (sig < 0) should use intact modulus
    sig_comp_init = np.array([-100.0, 0.0, 0.0])
    deps_comp = np.array([-0.0001, 0.0, 0.0])
    sig_comp, _, _ = law25_composite.shell_update(mat, sig_comp_init, deps_comp, extra=extra)
    dsig_comp = sig_comp[0] - sig_comp_init[0]
    # In compression, de1 = 1.0 (unilateral recovery)
    assert dsig_comp / (-0.0001) == pytest.approx(100000.0, rel=1e-3)


def test_law25_element_failure_deletion():
    """Verify element deletion per IOFF failure criteria."""
    # ioff=0: deleted when wpla >= wpmax
    mat = law25_composite.build_law25(
        E1=100000.0,
        E2=100000.0,
        nu12=0.0,
        G12=50000.0,
        sigyt1=50.0,
        sigyc1=50.0,
        sigyt2=50.0,
        sigyc2=50.0,
        sigt12=50.0,
        wpmax=0.05,
        ioff=0,
    )
    extra: dict[str, Any] = {
        "stra25": np.zeros((1, 3)),
        "dmg25": np.zeros((1, 8)),
        "wpla25": np.zeros(1),
        "off25": np.ones(1),
    }

    sig = np.zeros(3)
    # Apply severe strain causing large plastic work > 0.05
    deps = np.array([0.02, 0.0, 0.0])
    sig_new, _, _ = law25_composite.shell_update(mat, sig, deps, extra=extra)

    assert extra["off25"][0] == 0.0  # Deleted
    assert np.all(sig_new == 0.0)  # Stress zeroed


# ============================================================================
# 4. Solid Stress Update Tests
# ============================================================================

def test_law25_solid_update_3d():
    """Verify 3D solid update with orthotropic elasticity and through-thickness component."""
    mat = law25_composite.build_law25(
        E1=100000.0,
        E2=50000.0,
        E3=20000.0,
        nu12=0.2,
        G12=15000.0,
        G23=8000.0,
        G31=12000.0,
        sigyt1=2000.0,
        sigyc1=2000.0,
        sigyt2=1000.0,
        sigyc2=1000.0,
        sigt12=500.0,
    )
    sig = np.zeros(6)
    deps = np.array([1e-4, 1e-4, 2e-4, 1e-4, 1e-4, 1e-4])

    sig_new, epsp_new, c_val = law25_composite.solid_update(mat, sig, deps)

    assert sig_new.shape == (6,)
    # Component 2 (zz) should be E3 * deps[2] = 20000 * 2e-4 = 4.0
    assert sig_new[2] == pytest.approx(4.0, rel=1e-4)
    # Component 4 (yz) should be G23 * deps[4] = 8000 * 1e-4 = 0.8
    assert sig_new[4] == pytest.approx(0.8, rel=1e-4)
    # Component 5 (zx) should be G31 * deps[5] = 12000 * 1e-4 = 1.2
    assert sig_new[5] == pytest.approx(1.2, rel=1e-4)
    assert epsp_new == 0.0
    assert c_val > 0.0


# ============================================================================
# 5. Tangent Consistency & Vectorization Tests
# ============================================================================

def test_law25_consistent_tangents_shapes():
    """Verify consistent tangents return (n, 3, 3) for shells and (n, 6, 6) for solids."""
    mat = law25_composite.build_law25(
        E1=100000.0,
        E2=80000.0,
        E3=50000.0,
        nu12=0.2,
        G12=30000.0,
        G23=20000.0,
        G31=25000.0,
        sigyt1=200.0,
        sigyc1=200.0,
    )

    # Shell tangents: (n, 3, 3)
    sig_shell = np.zeros((4, 3))
    t_shell = law25_composite.consistent_shell_tangent(mat, sig_shell)
    assert t_shell.shape == (4, 3, 3)

    # Single point shell
    t_shell_1 = law25_composite.consistent_shell_tangent(mat, np.zeros(3))
    assert t_shell_1.shape == (1, 3, 3)

    # Solid tangents: (n, 6, 6)
    sig_solid = np.zeros((5, 6))
    t_solid = law25_composite.consistent_solid_tangent(mat, sig_solid)
    assert t_solid.shape == (5, 6, 6)

    # Single point solid
    t_solid_1 = law25_composite.consistent_solid_tangent(mat, np.zeros(6))
    assert t_solid_1.shape == (1, 6, 6)


def test_law25_vectorized_shell_update():
    """Verify vectorized multi-element shell update matches single-element calls."""
    mat = law25_composite.build_law25(
        E1=120000.0,
        E2=30000.0,
        nu12=0.2,
        G12=10000.0,
        sigyt1=300.0,
        sigyc1=250.0,
        sigyt2=40.0,
        sigyc2=100.0,
        sigt12=60.0,
    )
    n = 3
    sigs = np.zeros((n, 3))
    deps = np.array([
        [1e-4, 0.0, 0.0],
        [0.0, 2e-4, 0.0],
        [1e-4, 1e-4, 1e-4],
    ])

    sigs_out, epsp_out, c_out = law25_composite.shell_update(mat, sigs, deps)

    assert sigs_out.shape == (n, 3)
    assert epsp_out.shape == (n,)
    assert c_out.shape == (n,)

    for i in range(n):
        s_single, e_single, _ = law25_composite.shell_update(mat, np.zeros(3), deps[i])
        np.testing.assert_allclose(sigs_out[i], s_single, rtol=1e-6)
        assert epsp_out[i] == pytest.approx(e_single)


# ============================================================================
# 6. Materials Package Dispatch & Registry Tests
# ============================================================================

def test_materials_dispatch():
    """Verify pyradioss.materials module dispatches to LAW25."""
    mat = law25_composite.build_law25(
        id=1,
        rho0=1.5e-9,
        E1=100000.0,
        E2=50000.0,
        nu12=0.2,
        G12=15000.0,
    )
    assert mat.law == 25

    # extra_shapes
    shapes_shell = materials.extra_shapes(mat, nip=3)
    assert "dmg25" in shapes_shell
    assert shapes_shell["dmg25"] == (3, 4)
    assert "stra25" in shapes_shell

    shapes_solid = materials.extra_shapes(mat)
    assert shapes_solid["dmg25"] == (4,)

    # sound_speed
    ssp = materials.sound_speed(mat)
    assert ssp > 0.0

    # shell_update returns (sig, epsp) for element kernels
    sig = np.zeros(3)
    deps = np.array([1e-4, 0.0, 0.0])
    s_new, ep_new = materials.shell_update(mat, sig, deps, dt=1e-6)
    assert s_new.shape == (3,)
    assert ep_new == pytest.approx(0.0)

    # solid_update returns (sig, epsp, c)
    sig6 = np.zeros(6)
    deps6 = np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
    s6_new, ep6_new, c6 = materials.solid_update(mat, sig6, deps6, epsp=0.0, dt=1e-6)
    assert s6_new.shape == (6,)
    assert c6 > 0.0

    # tangents
    c_mem = materials.shell_membrane_tangent(mat)
    assert c_mem.shape == (3, 3)

    t_layer = materials.shell_layer_tangent(mat, np.zeros((2, 3)), None, None)
    assert t_layer.shape == (2, 3, 3)

    t_sol = materials.solid_tangent(mat, np.zeros((2, 6)), None, None)
    assert t_sol.shape == (2, 6, 6)


def test_mat_physics_registry_law25():
    """Verify GenericMaterialRecord instantiates LAW25 via MAT_PHYSICS_REGISTRY."""
    rec = GenericMaterialRecord(
        law_name="LAW25",
        law_number=25,
        id=42,
        density=1.6e-9,
        title="CARBON_EPOXY",
        params={
            "MAT_EA": 150000.0,
            "MAT_EB": 9000.0,
            "MAT_PRAB": 0.34,
            "MAT_GAB": 4500.0,
            "MAT_SIGYT1": 1800.0,
            "MAT_SIGYC1": 1200.0,
            "MAT_SIGYT2": 40.0,
            "MAT_SIGYC2": 150.0,
            "MAT_SIGT12": 60.0,
        },
    )
    builder = MAT_PHYSICS_REGISTRY.get("LAW25")
    assert callable(builder)

    mat = builder(rec)
    assert mat.id == 42
    assert mat.law == 25
    assert mat.rho0 == 1.6e-9
    assert mat.params["e11"] == 150000.0
    assert mat.params["e22"] == 9000.0
    assert mat.params["ssp"] > 0.0


def test_law25_crasurv_directional_hardening():
    """Verify CRASURV directional hardening increases directional yield limits."""
    mat = law25_composite.build_crasurv(
        E1=100000.0,
        E2=50000.0,
        nu12=0.0,
        G12=20000.0,
        iflag=1,
        sigyt1=100.0,
        sigyc1=100.0,
        sigyt2=50.0,
        sigyc2=50.0,
        sigt12=30.0,
        b1_t=1.0,
        n1_t=1.0,
        sig1max_t=500.0,
        wplaref=1.0,
    )
    sig = np.zeros(3)
    # Apply strain in direction 1 to cause yielding
    deps1 = np.array([0.003, 0.0, 0.0])
    extra: dict[str, Any] = {}

    sig1, ep1, _ = law25_composite.shell_update(mat, sig, deps1, extra=extra)
    assert ep1 > 0.0
    assert extra["wpla25"][0] > 0.0
    assert np.isclose(sig1[0], 100.0)

    # In second step, accumulated plastic work hardens directional yield limit
    deps2 = np.array([0.001, 0.0, 0.0])
    sig2, ep2, _ = law25_composite.shell_update(mat, sig1, deps2, epsp=ep1, extra=extra)
    assert sig2[0] > 100.0
    assert ep2 > ep1


def test_law25_strain_rate_sensitivity():
    """Verify strain rate factor increases yield limit for high strain rate."""
    mat_static = law25_composite.build_law25(
        E1=100000.0,
        E2=100000.0,
        nu12=0.0,
        G12=50000.0,
        sigyt1=100.0,
        sigyc1=100.0,
        c=0.2,
        epdr=1.0,
    )
    mat_dynamic = law25_composite.build_law25(
        E1=100000.0,
        E2=100000.0,
        nu12=0.0,
        G12=50000.0,
        sigyt1=100.0,
        sigyc1=100.0,
        c=0.2,
        epdr=1.0,
    )
    deps = np.array([0.005, 0.0, 0.0])

    # Static: dt = 1.0 -> eps_dot = 0.005 < 1.0 -> epspfac = 1.0
    s_stat, _, _ = law25_composite.shell_update(mat_static, np.zeros(3), deps, dt=1.0)

    # Dynamic: dt = 1e-5 -> eps_dot = 500.0 > 1.0 -> epspfac = 1 + 0.2*log(500) > 2.2
    s_dyn, _, _ = law25_composite.shell_update(mat_dynamic, np.zeros(3), deps, dt=1e-5)

    assert s_dyn[0] > s_stat[0]


def test_law25_solid_plastic_yielding():
    """Verify 3D solid plastic yielding and work accumulation."""
    mat = law25_composite.build_law25(
        E1=100000.0,
        E2=100000.0,
        E3=50000.0,
        nu12=0.0,
        G12=50000.0,
        sigyt1=200.0,
        sigyc1=200.0,
        sigyt2=200.0,
        sigyc2=200.0,
        sigt12=100.0,
    )
    sig = np.zeros(6)
    # Apply large normal strain in dir 1
    deps = np.array([0.004, 0.0, 0.0, 0.0, 0.0, 0.0])
    extra: dict[str, Any] = {}

    sig_new, epsp_new, c_arr = law25_composite.solid_update(mat, sig, deps, extra=extra)

    assert epsp_new > 0.0
    assert sig_new[0] <= 205.0  # Returned onto yield surface
    assert c_arr > 0.0

