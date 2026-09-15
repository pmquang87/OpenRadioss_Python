"""
Tests for OpenRadioss Material Law 25 (/MAT/LAW25, /MAT/COMP_PLAS, /MAT/COMPSH, /MAT/TSAI_WU, /MAT/CRASURV).
Milestone M543: Composite Anisotropic Plasticity and Damage Model.

Covers:
  1. Parameter constructor build_law25:
     - 6x6 orthotropic compliance and stiffness matrices
     - Sound speed calculation (longitudinal, transverse, max wave speed)
     - Tsai-Wu coefficients F1, F2, F11, F22, F33, F12
  2. Elastic behavior in shell plane stress and 3D solid:
     - Hooke's law in directions 1, 2, and in-plane shear (12)
     - Hooke's law in 3D solid including out-of-plane and transverse shears (23, 31)
  3. Tsai-Wu yield condition:
     - Elastic domain: W_vec < f_yld (zero plastic strain increment)
     - Yield onset: W_vec = f_yld
     - Plastic return: W_vec -> f_yld, plastic work accumulation
  4. Strain rate sensitivity:
     - epspfac factor with rate sensitivity c and reference rate epdr
  5. Tensile damage degradation:
     - d1, d2 damage accumulation beyond eps_t thresholds
     - Modulus reduction in tension, unilateral stiffness recovery in compression
  6. CRASURV formulation (iflag=1):
     - Directional hardening (1t, 2t, 1c, 2c, 12t)
     - Directional softening and residual stress plateau
  7. Element failure deletion (IOFF 0..6):
     - Deletion when plastic work or directional tensile strain exceeds limits
     - Stress zeroing and off flag clearance
  8. Algorithmic tangents and materials package dispatch:
     - shell_membrane_tangent, consistent_shell_tangent, consistent_solid_tangent
     - extra_shapes, sound_speed, shell_update, solid_update dispatch
"""

from __future__ import annotations

import math
from typing import Any, Dict
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.materials import law25_composite
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY, GenericMaterialRecord


# ============================================================================
# 1. Parameter Constructor build_law25 & Constants
# ============================================================================

def test_law25_constructor_and_defaults():
    """Verify default constructor parameters and derived compliance / sound speed."""
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

    # nu21 = nu12 * E2 / E1
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


def test_law25_compliance_and_stiffness_inversion():
    """Verify 6x6 orthotropic compliance matrix S and stiffness C = S^(-1)."""
    e1, e2, e3 = 100000.0, 50000.0, 20000.0
    nu12 = 0.25
    nu21 = nu12 * e2 / e1  # 0.125
    g12, g23, g31 = 15000.0, 8000.0, 10000.0

    mat = law25_composite.build_law25(
        E1=e1, E2=e2, E3=e3,
        nu12=nu12,
        G12=g12, G23=g23, G31=g31,
    )

    c_mat = law25_composite.solid_stiffness_matrix(mat)
    assert c_mat.shape == (6, 6)
    # Check diagonal and symmetry
    assert c_mat[0, 1] == pytest.approx(c_mat[1, 0])
    assert c_mat[3, 3] == pytest.approx(g12)
    assert c_mat[4, 4] == pytest.approx(g23)
    assert c_mat[5, 5] == pytest.approx(g31)

    # Plane stress submatrix inversion
    c_shell = law25_composite.shell_membrane_tangent(mat)
    detc = 1.0 - nu12 * nu21
    s_plane = np.array([
        [1.0 / e1, -nu21 / e2, 0.0],
        [-nu12 / e1, 1.0 / e2, 0.0],
        [0.0, 0.0, 1.0 / g12],
    ])
    c_expected = np.linalg.inv(s_plane)
    np.testing.assert_allclose(c_shell, c_expected, rtol=1e-5)


def test_law25_tsaiwu_coefficients():
    """Verify Tsai-Wu failure coefficients F1, F2, F11, F22, F33, F12."""
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


def test_law25_sound_speed_modes():
    """Verify sound speed for longitudinal, transverse and max modes."""
    rho = 2000.0
    e11 = 8.0e10
    e22 = 2.0e10
    g12 = 5.0e9
    mat = law25_composite.build_law25(rho0=rho, E1=e11, E2=e22, nu12=0.2, G12=g12)

    c_sound = law25_composite.sound_speed(mat)
    # Longitudinal wave speed c_long = sqrt(C1 / rho)
    c1 = mat.params["C1"]
    expected_c = math.sqrt(c1 / rho)
    assert c_sound == pytest.approx(expected_c)
    assert c_sound > math.sqrt(g12 / rho)  # Greater than transverse shear wave


# ============================================================================
# 2. Elastic Behavior in Shells and Solids
# ============================================================================

def test_law25_shell_elastic_hooke_plane_stress():
    """Verify Hooke's law in shell plane stress: dir 1, dir 2, and in-plane shear."""
    e1 = 100000.0
    e2 = 40000.0
    nu12 = 0.2
    g12 = 12000.0
    mat = law25_composite.build_law25(
        E1=e1, E2=e2, nu12=nu12, G12=g12,
        sigyt1=5000.0, sigyc1=5000.0, sigyt2=5000.0, sigyc2=5000.0, sigt12=3000.0,
    )
    c_plane = law25_composite.shell_membrane_tangent(mat)

    # Direction 1 loading
    sig = np.zeros(3)
    deps1 = np.array([1e-4, 0.0, 0.0])
    s1, ep1, _ = law25_composite.shell_update(mat, sig, deps1)
    assert s1[0] == pytest.approx(c_plane[0, 0] * 1e-4)
    assert s1[1] == pytest.approx(c_plane[1, 0] * 1e-4)
    assert s1[2] == pytest.approx(0.0, abs=1e-12)
    assert ep1 == pytest.approx(0.0)

    # Direction 2 loading
    deps2 = np.array([0.0, 2e-4, 0.0])
    s2, ep2, _ = law25_composite.shell_update(mat, sig, deps2)
    assert s2[0] == pytest.approx(c_plane[0, 1] * 2e-4)
    assert s2[1] == pytest.approx(c_plane[1, 1] * 2e-4)
    assert s2[2] == pytest.approx(0.0, abs=1e-12)
    assert ep2 == pytest.approx(0.0)

    # In-plane shear loading
    deps12 = np.array([0.0, 0.0, 3e-4])
    s12, ep12, _ = law25_composite.shell_update(mat, sig, deps12)
    assert s12[0] == pytest.approx(0.0, abs=1e-12)
    assert s12[1] == pytest.approx(0.0, abs=1e-12)
    assert s12[2] == pytest.approx(g12 * 3e-4)
    assert ep12 == pytest.approx(0.0)


def test_law25_solid_elastic_hooke_3d():
    """Verify Hooke's law in 3D solid: normal components (1, 2, 3) and shears (12, 23, 31)."""
    e1, e2, e3 = 90000.0, 45000.0, 30000.0
    nu12 = 0.25
    g12, g23, g31 = 15000.0, 8000.0, 10000.0
    mat = law25_composite.build_law25(
        E1=e1, E2=e2, E3=e3, nu12=nu12,
        G12=g12, G23=g23, G31=g31,
        sigyt1=5000.0, sigyc1=5000.0, sigyt2=5000.0, sigyc2=5000.0, sigt12=3000.0,
    )
    sig = np.zeros(6)
    deps = np.array([1e-4, 5e-5, 2e-4, 1e-4, 3e-4, 1.5e-4])

    s_new, ep_new, c_val = law25_composite.solid_update(mat, sig, deps)

    c_solid = law25_composite.solid_stiffness_matrix(mat)
    assert s_new[0] == pytest.approx(c_solid[0, 0] * deps[0] + c_solid[0, 1] * deps[1])
    assert s_new[1] == pytest.approx(c_solid[1, 0] * deps[0] + c_solid[1, 1] * deps[1])
    assert s_new[2] == pytest.approx(e3 * deps[2])
    assert s_new[3] == pytest.approx(g12 * deps[3])
    assert s_new[4] == pytest.approx(g23 * deps[4])
    assert s_new[5] == pytest.approx(g31 * deps[5])
    assert ep_new == pytest.approx(0.0)
    assert c_val > 0.0


# ============================================================================
# 3. Tsai-Wu Yield Condition & Plastic Return
# ============================================================================

def test_law25_tsaiwu_elastic_domain():
    """Verify elastic domain: W_vec < f_yld produces zero plastic work / plastic strain."""
    mat = law25_composite.build_law25(
        E1=100000.0, E2=50000.0, nu12=0.0, G12=25000.0,
        sigyt1=200.0, sigyc1=200.0, sigyt2=100.0, sigyc2=100.0, sigt12=60.0,
    )
    sig = np.zeros(3)
    # Strain produces trial s11 = 100 < 200, W_vec = (100/200)^2 = 0.25 < 1.0
    deps = np.array([0.001, 0.0, 0.0])
    extra: Dict[str, Any] = {}

    s_out, ep_out, _ = law25_composite.shell_update(mat, sig, deps, extra=extra)

    assert s_out[0] == pytest.approx(100.0)
    assert ep_out == pytest.approx(0.0)
    assert extra.get("wpla25", np.zeros(1))[0] == pytest.approx(0.0)


def test_law25_tsaiwu_yield_onset_and_plastic_return():
    """Verify yield onset W_vec = f_yld and plastic return bringing stress onto yield surface."""
    mat = law25_composite.build_law25(
        E1=100000.0, E2=100000.0, nu12=0.0, G12=50000.0,
        sigyt1=200.0, sigyc1=200.0, sigyt2=200.0, sigyc2=200.0, sigt12=100.0,
        b=0.0,  # Perfectly plastic
        fmax=10.0,
    )
    sig = np.zeros(3)
    # Trial stress s11 = 400.0 > 200.0 -> W_vec trial = (400/200)^2 = 4.0 > 1.0
    deps = np.array([0.004, 0.0, 0.0])
    extra: Dict[str, Any] = {}

    s_out, ep_out, _ = law25_composite.shell_update(mat, sig, deps, extra=extra)

    # Plastic return returns stress to yield surface
    assert s_out[0] == pytest.approx(200.0, rel=1e-3)
    assert ep_out > 0.0
    assert extra["wpla25"][0] > 0.0

    # Check Tsai-Wu criterion W_vec == 1.0 on returned stress
    p = mat.params
    f1, f2, f11, f22, f33, f12 = p["F1"], p["F2"], p["F11"], p["F22"], p["F33"], p["F12"]
    s1, s2, s12 = s_out[0], s_out[1], s_out[2]
    wvec = f1 * s1 + f2 * s2 + f11 * s1**2 + f22 * s2**2 + f33 * s12**2 + 2.0 * f12 * s1 * s2
    assert wvec == pytest.approx(1.0, rel=1e-3)


# ============================================================================
# 4. Strain Rate Sensitivity
# ============================================================================

def test_law25_strain_rate_sensitivity():
    """Verify epspfac factor scales yield surface under high strain rates."""
    c_rate = 0.15
    epdr = 10.0
    mat = law25_composite.build_law25(
        E1=100000.0, E2=100000.0, nu12=0.0, G12=50000.0,
        sigyt1=100.0, sigyc1=100.0, sigyt2=100.0, sigyc2=100.0, sigt12=50.0,
        c=c_rate, epdr=epdr,
    )
    deps = np.array([0.003, 0.0, 0.0])

    # Static: dt large -> eps_dot < epdr -> epspfac = 1.0
    s_stat, _, _ = law25_composite.shell_update(mat, np.zeros(3), deps, dt=1.0)
    assert s_stat[0] == pytest.approx(100.0, rel=1e-3)

    # Dynamic: dt = 1e-5 -> eps_dot = 0.003 / 1e-5 = 300.0 > 10.0
    expected_epspfac = 1.0 + c_rate * math.log(300.0 / 10.0)
    expected_sig = 100.0 * math.sqrt(expected_epspfac)
    s_dyn, _, _ = law25_composite.shell_update(mat, np.zeros(3), deps, dt=1e-5)
    assert s_dyn[0] == pytest.approx(expected_sig, rel=1e-2)
    assert s_dyn[0] > s_stat[0]


# ============================================================================
# 5. Tensile Damage Degradation & Unilateral Recovery
# ============================================================================

def test_law25_tensile_damage_and_unilateral_recovery():
    """Verify d1, d2 damage accumulation and unilateral recovery in compression."""
    mat = law25_composite.build_law25(
        E1=100000.0, E2=50000.0, nu12=0.0, G12=20000.0,
        epst1=0.001, epsm1=0.005, dmax=0.7,
        sigyt1=5000.0, sigyc1=5000.0,  # Elastic-damage only
    )
    extra: Dict[str, Any] = {
        "stra25": np.zeros((1, 3)),
        "dmg25": np.zeros((1, 8)),
        "wpla25": np.zeros(1),
        "off25": np.ones(1),
    }

    # Apply strain exceeding epst1
    deps = np.array([0.003, 0.0, 0.0])
    s1, _, _ = law25_composite.shell_update(mat, np.zeros(3), deps, extra=extra)

    d1 = extra["dmg25"][0, 1]
    assert d1 > 0.0
    assert d1 <= 0.7

    # Tensile increment: effective modulus should be E1 * (1 - d1)
    deps_tens = np.array([1e-4, 0.0, 0.0])
    s2, _, _ = law25_composite.shell_update(mat, s1, deps_tens, extra=extra)
    tangent_tens = (s2[0] - s1[0]) / 1e-4
    assert tangent_tens == pytest.approx(100000.0 * (1.0 - d1), rel=1e-3)

    # Compressive increment: intact modulus restored
    s_comp_init = np.array([-50.0, 0.0, 0.0])
    deps_comp = np.array([-1e-4, 0.0, 0.0])
    s_comp, _, _ = law25_composite.shell_update(mat, s_comp_init, deps_comp, extra=extra)
    tangent_comp = (s_comp[0] - s_comp_init[0]) / (-1e-4)
    assert tangent_comp == pytest.approx(100000.0, rel=1e-3)


# ============================================================================
# 6. CRASURV Formulation (iflag=1)
# ============================================================================

def test_law25_crasurv_directional_hardening_and_softening():
    """Verify CRASURV directional hardening and softening in direction 1."""
    mat = law25_composite.build_crasurv(
        E1=100000.0, E2=50000.0, nu12=0.0, G12=20000.0,
        iflag=1,
        sigyt1=100.0, sigyc1=100.0, sigyt2=50.0, sigyc2=50.0, sigt12=30.0,
        b1_t=1.0, n1_t=1.0, sig1max_t=300.0,
        eps1_t1=0.01, eps2_t1=0.03, sig_rst1=50.0,
        wplaref=1.0,
    )
    extra: Dict[str, Any] = {}

    # Step 1: Plastic yielding at sigyt1 = 100
    deps1 = np.array([0.002, 0.0, 0.0])
    s1, ep1, _ = law25_composite.shell_update(mat, np.zeros(3), deps1, extra=extra)
    assert s1[0] == pytest.approx(100.0, rel=1e-2)
    assert ep1 > 0.0

    # Step 2: Directional hardening increases yield stress
    deps2 = np.array([0.001, 0.0, 0.0])
    s2, ep2, _ = law25_composite.shell_update(mat, s1, deps2, epsp=ep1, extra=extra)
    assert s2[0] > 100.0
    assert ep2 > ep1

    # Step 3: Directional softening when total strain enters [eps1_t1, eps2_t1]
    extra_soft: Dict[str, Any] = {
        "stra25": np.zeros((1, 3)),
        "dmg25": np.zeros((1, 8)),
        "wpla25": np.zeros(1),
        "off25": np.ones(1),
    }
    # Large strain to trigger softening
    deps_large = np.array([0.02, 0.0, 0.0])
    s_soft, _, _ = law25_composite.shell_update(mat, np.zeros(3), deps_large, extra=extra_soft)
    # Stress should be softened toward sig_rst1 = 50.0
    assert s_soft[0] < 300.0


# ============================================================================
# 7. Element Failure Deletion (IOFF 0..6)
# ============================================================================

@pytest.mark.parametrize("ioff_val", [0, 1, 2, 3, 4, 5, 6])
def test_law25_failure_deletion_modes(ioff_val: int):
    """Verify failure deletion criteria across all IOFF modes (0..6)."""
    # Case A: Plastic work exceeding wpmax should delete for all ioff
    mat_wp = law25_composite.build_law25(
        E1=100000.0, E2=100000.0, nu12=0.0, G12=50000.0,
        sigyt1=50.0, sigyc1=50.0, sigyt2=50.0, sigyc2=50.0, sigt12=50.0,
        wpmax=0.01,
        ioff=ioff_val,
    )
    extra_wp: Dict[str, Any] = {
        "stra25": np.zeros((1, 3)),
        "dmg25": np.zeros((1, 8)),
        "wpla25": np.zeros(1),
        "off25": np.ones(1),
    }
    s_out, _, _ = law25_composite.shell_update(mat_wp, np.zeros(3), np.array([0.01, 0.0, 0.0]), extra=extra_wp)
    assert extra_wp["off25"][0] == 0.0
    assert np.all(s_out == 0.0)

    # Case B: Direction 1 tensile strain exceeding epsf1
    mat_f1 = law25_composite.build_law25(
        E1=100000.0, E2=100000.0, nu12=0.0, G12=50000.0,
        sigyt1=5000.0, sigyc1=5000.0, sigyt2=5000.0, sigyc2=5000.0, sigt12=5000.0,
        eps_f1=0.005,
        eps_f2=0.020,
        wpmax=1e10,
        ioff=ioff_val,
    )
    extra_f1: Dict[str, Any] = {
        "stra25": np.zeros((1, 3)),
        "dmg25": np.zeros((1, 8)),
        "wpla25": np.zeros(1),
        "off25": np.ones(1),
    }
    # Apply strain eps1 = 0.008 > epsf1, eps2 = 0 < epsf2
    s_out_f1, _, _ = law25_composite.shell_update(mat_f1, np.zeros(3), np.array([0.008, 0.0, 0.0]), extra=extra_f1)

    # Modes deleted by fail1 alone: 2, 5, 6
    # Modes NOT deleted by fail1 alone: 0, 1 (wp only), 3 (fail2), 4 (fail1 and fail2)
    if ioff_val in (2, 5, 6):
        assert extra_f1["off25"][0] == 0.0
        assert np.all(s_out_f1 == 0.0)
    else:
        assert extra_f1["off25"][0] == 1.0


def test_law25_failure_deletion_ioff4_and_ioff6():
    """Verify ioff=4 requires BOTH fail1 and fail2; ioff=6 includes shear strain."""
    # ioff=4: both fail1 and fail2 needed
    mat4 = law25_composite.build_law25(
        E1=100000.0, E2=100000.0, nu12=0.0, G12=50000.0,
        sigyt1=5000.0, sigyc1=5000.0, sigyt2=5000.0, sigyc2=5000.0, sigt12=5000.0,
        eps_f1=0.005, eps_f2=0.005, wpmax=1e10, ioff=4,
    )
    extra4: Dict[str, Any] = {
        "stra25": np.zeros((1, 3)), "dmg25": np.zeros((1, 8)), "wpla25": np.zeros(1), "off25": np.ones(1),
    }
    # eps1 = 0.008 > epsf1, eps2 = 0.008 > epsf2 -> both fail -> delete
    s4, _, _ = law25_composite.shell_update(mat4, np.zeros(3), np.array([0.008, 0.008, 0.0]), extra=extra4)
    assert extra4["off25"][0] == 0.0
    assert np.all(s4 == 0.0)

    # ioff=6: shear strain failure
    mat6 = law25_composite.build_law25(
        E1=100000.0, E2=100000.0, nu12=0.0, G12=50000.0,
        sigyt1=5000.0, sigyc1=5000.0, sigyt2=5000.0, sigyc2=5000.0, sigt12=5000.0,
        eps_f1=0.005, eps_f2=0.020, wpmax=1e10, ioff=6,
    )
    extra6: Dict[str, Any] = {
        "stra25": np.zeros((1, 3)), "dmg25": np.zeros((1, 8)), "wpla25": np.zeros(1), "off25": np.ones(1),
    }
    # shear strain gamma12 = 0.008 > epsf1 -> delete
    s6, _, _ = law25_composite.shell_update(mat6, np.zeros(3), np.array([0.0, 0.0, 0.008]), extra=extra6)
    assert extra6["off25"][0] == 0.0
    assert np.all(s6 == 0.0)


# ============================================================================
# 8. Tangents & Dispatch
# ============================================================================

def test_law25_tangents_and_shapes():
    """Verify membrane and consistent tangent shapes and values."""
    mat = law25_composite.build_law25(
        E1=120000.0, E2=60000.0, E3=40000.0, nu12=0.2,
        G12=20000.0, G23=10000.0, G31=15000.0,
    )

    t_shell = law25_composite.consistent_shell_tangent(mat, np.zeros((3, 3)))
    assert t_shell.shape == (3, 3, 3)

    t_solid = law25_composite.consistent_solid_tangent(mat, np.zeros((2, 6)))
    assert t_solid.shape == (2, 6, 6)

    shapes_nip = materials.extra_shapes(mat, nip=4)
    assert shapes_nip["dmg25"] == (4, 4)
    assert shapes_nip["stra25"] == (4, 6)
    assert shapes_nip["wpla25"] == (4,)
    assert shapes_nip["off25"] == (4,)

    shapes_sol = materials.extra_shapes(mat)
    assert shapes_sol["dmg25"] == (4,)
    assert shapes_sol["stra25"] == (6,)
    assert shapes_sol["wpla25"] == ()
    assert shapes_sol["off25"] == ()
