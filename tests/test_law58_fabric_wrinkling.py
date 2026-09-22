"""
Unit tests for LAW58 A-formulation fabric wrinkling model (/MAT/LAW58, /MAT/FABR_A).
Ported from OpenRadioss Fortran sigeps58c.F / hm_read_mat58.F.

Verifies:
1. Three-state fabric deformation model:
   - TAUT (1): Both warp and weft fibers in tension
   - WRINKLED (2): Uniaxial tension/compression (one fiber tensioned, one compressed)
   - SLACK (0): Bi-compression (both warp and weft fibers compressed)
2. State classification utilities:
   - classify_fabric_state (scalar and vectorized)
   - is_taut, is_wrinkled, is_slack, get_fabric_state
3. Wrinkling mechanics:
   - Full shear stiffness in TAUT state
   - No shear stiffness when compressed (WRINKLED state)
   - Zero membrane stresses in SLACK state
4. Tangent matrix consistency across TAUT, WRINKLED, and SLACK regimes
5. History variables and extra_shapes registration
"""

import math
import numpy as np
import pytest

from pyradioss.materials import law58_fabr_a
from pyradioss.materials.law58_fabr_a import (
    FabricState,
    Law58Params,
    FabricAMaterial,
    classify_fabric_state,
    is_taut,
    is_wrinkled,
    is_slack,
    get_fabric_state,
    shell_update_law58,
    shell_update_wrinkled,
    shell_membrane_tangent,
    tangent_law58_shell,
    tangent_wrinkled_shell,
    solid_update,
    build_law58,
    extra_shapes,
)


def _make_mat(iwrinkle=1, e1=2000.0, e2=1500.0, g0=100.0, gt=300.0, **kwargs):
    """Helper to construct a LAW58 fabric material with wrinkling model enabled."""
    rec = type("Rec", (), {})()
    rec.id = 58
    rec.title = "FABRIC_WRINKLE_TEST"
    rec.density = 1.2e-6
    rec.params = {
        "MAT_E1": e1,
        "MAT_E2": e2,
        "MAT_G0": g0,
        "MAT_GI": gt,
        "MAT_F": 1e-3,
        "IWRINKLE": iwrinkle,
        **kwargs,
    }
    return build_law58(rec)


# ============================================================================
# 1. State Classification Unit Tests
# ============================================================================

def test_fabric_state_enum_values():
    """Verify FabricState enum values matching A-formulation standard."""
    assert FabricState.SLACK == 0
    assert FabricState.TAUT == 1
    assert FabricState.TENSIONED == 1
    assert FabricState.WRINKLED == 2


def test_classify_fabric_state_scalar():
    """Verify scalar classification of TAUT, WRINKLED, and SLACK states."""
    # Bi-tension -> TAUT
    st_taut = classify_fabric_state(100.0, 50.0)
    assert st_taut == FabricState.TAUT
    assert is_taut(st_taut)
    assert not is_wrinkled(st_taut)
    assert not is_slack(st_taut)

    # Warp tension, weft compression -> WRINKLED
    st_wr_warp = classify_fabric_state(100.0, -20.0)
    assert st_wr_warp == FabricState.WRINKLED
    assert is_wrinkled(st_wr_warp)
    assert not is_taut(st_wr_warp)

    # Weft tension, warp compression -> WRINKLED
    st_wr_weft = classify_fabric_state(-15.0, 80.0)
    assert st_wr_weft == FabricState.WRINKLED
    assert is_wrinkled(st_wr_weft)

    # Bi-compression -> SLACK
    st_slack = classify_fabric_state(-30.0, -40.0)
    assert st_slack == FabricState.SLACK
    assert is_slack(st_slack)
    assert not is_taut(st_slack)
    assert not is_wrinkled(st_slack)


def test_classify_fabric_state_vectorized():
    """Verify batch vectorized classification of fabric states."""
    sxx = np.array([100.0, 100.0, -10.0, -20.0])
    syy = np.array([50.0, -20.0, 80.0, -40.0])
    expected = np.array([
        FabricState.TAUT,
        FabricState.WRINKLED,
        FabricState.WRINKLED,
        FabricState.SLACK,
    ])

    states = classify_fabric_state(sxx, syy)
    assert np.array_equal(states, expected)
    assert np.array_equal(is_taut(states), [True, False, False, False])
    assert np.array_equal(is_wrinkled(states), [False, True, True, False])
    assert np.array_equal(is_slack(states), [False, False, False, True])


# ============================================================================
# 2. Constitutive Response in TAUT State
# ============================================================================

def test_taut_state_full_shear_stiffness():
    """Verify that in the TAUT state, full normal tension and Trellis shear are carried."""
    mat = _make_mat(iwrinkle=1, e1=2000.0, e2=1500.0, g0=100.0)
    sig = np.zeros(3)
    # Biaxial tension with shear
    deps = np.array([0.03, 0.02, 0.1])
    extra = {}

    s_new, _ = shell_update_law58(mat, sig, deps, extra=extra)

    assert extra["fabric_state"][0] == FabricState.TAUT
    # Warp and weft in tension
    assert s_new[0] > 0.0
    assert s_new[1] > 0.0
    # Full shear stress active: sxy approx g0 * tan_phi
    assert s_new[2] > 0.0
    assert s_new[2] == pytest.approx(100.0 * 0.1, rel=1e-3)


# ============================================================================
# 3. Constitutive Response in WRINKLED State (No Shear Stiffness)
# ============================================================================

def test_wrinkled_state_warp_tension_weft_compression():
    """Verify WRINKLED state: weft compressed -> zero shear stiffness and zero weft stress."""
    mat = _make_mat(iwrinkle=1, e1=2000.0, e2=1500.0, g0=100.0)
    sig = np.zeros(3)
    # Warp tension (+0.04), weft compression (-0.03), with shear strain (0.15)
    deps = np.array([0.04, -0.03, 0.15])
    extra = {}

    s_new, _ = shell_update_law58(mat, sig, deps, extra=extra)

    assert extra["fabric_state"][0] == FabricState.WRINKLED
    # Warp carries tension
    assert s_new[0] > 0.0
    # Weft is compressed -> zero normal stress in wrinkled state
    assert s_new[1] == 0.0
    # CRITICAL: No shear stiffness when compressed (wrinkled state)
    assert s_new[2] == 0.0


def test_wrinkled_state_weft_tension_warp_compression():
    """Verify WRINKLED state: warp compressed -> zero shear stiffness and zero warp stress."""
    mat = _make_mat(iwrinkle=1, e1=2000.0, e2=1500.0, g0=100.0)
    sig = np.zeros(3)
    # Warp compression (-0.03), weft tension (+0.04), with shear strain (0.2)
    deps = np.array([-0.03, 0.04, 0.2])
    extra = {}

    s_new, _ = shell_update_law58(mat, sig, deps, extra=extra)

    assert extra["fabric_state"][0] == FabricState.WRINKLED
    # Warp is compressed -> zero normal stress
    assert s_new[0] == 0.0
    # Weft carries tension
    assert s_new[1] > 0.0
    # CRITICAL: No shear stiffness when compressed (wrinkled state)
    assert s_new[2] == 0.0


# ============================================================================
# 4. Constitutive Response in SLACK State
# ============================================================================

def test_slack_state_zero_membrane_stresses():
    """Verify SLACK state: bi-compression zeroes all in-plane membrane stresses."""
    mat = _make_mat(iwrinkle=1, e1=2000.0, e2=1500.0, g0=100.0)
    sig = np.zeros(3)
    # Bi-compression with shear strain
    deps = np.array([-0.02, -0.03, 0.1])
    extra = {}

    s_new, _ = shell_update_law58(mat, sig, deps, extra=extra)

    assert extra["fabric_state"][0] == FabricState.SLACK
    # All in-plane stresses are zero
    assert s_new[0] == 0.0
    assert s_new[1] == 0.0
    assert s_new[2] == 0.0


# ============================================================================
# 5. Tangent Matrices in TAUT, WRINKLED, and SLACK States
# ============================================================================

def test_shell_membrane_tangent_states():
    """Verify shell_membrane_tangent reference matrix for each state."""
    mat = _make_mat(e1=2000.0, e2=1500.0, g0=100.0)

    # TAUT: full matrix
    d_taut = shell_membrane_tangent(mat, state=FabricState.TAUT)
    assert d_taut[0, 0] == 2000.0
    assert d_taut[1, 1] == 1500.0
    assert d_taut[2, 2] == 1000.0 or d_taut[2, 2] == 100.0

    # WRINKLED: shear stiffness is zero
    d_wr = shell_membrane_tangent(mat, state=FabricState.WRINKLED)
    assert d_wr[0, 0] == 2000.0
    assert d_wr[1, 1] == 1500.0
    assert d_wr[2, 2] == 0.0

    # SLACK: completely zero
    d_sl = shell_membrane_tangent(mat, state=FabricState.SLACK)
    assert np.all(d_sl == 0.0)


def test_algorithmic_tangent_wrinkling_shear_cutoff():
    """Verify algorithmic tangent D_33 vanishes in wrinkled state."""
    mat = _make_mat(iwrinkle=1, e1=2000.0, e2=1500.0, g0=100.0)
    # In wrinkled regime: warp tension, weft compression
    deps_wr = np.array([0.04, -0.03, 0.05])
    sig = np.zeros(3)

    D_wr = tangent_law58_shell(mat, sig, deps_wr)
    assert D_wr.shape == (3, 3)
    # No shear stiffness in wrinkled state: D_33 must be 0
    assert abs(D_wr[2, 2]) < 1e-6
    assert abs(D_wr[0, 2]) < 1e-6
    assert abs(D_wr[1, 2]) < 1e-6

    # In taut regime: warp tension, weft tension
    deps_taut = np.array([0.04, 0.03, 0.05])
    D_taut = tangent_law58_shell(mat, sig, deps_taut)
    # Full shear stiffness active in taut state
    assert D_taut[2, 2] > 0.0


# ============================================================================
# 6. Integration & Dispatcher Wiring
# ============================================================================

def test_extra_shapes_includes_fabric_state():
    """Verify extra_shapes includes fabric_state history allocation."""
    shapes = extra_shapes(nip=4)
    assert "fabric_state" in shapes
    assert shapes["fabric_state"] == (4,)


def test_solid_update_raises_not_implemented():
    """Verify LAW58 rejects continuum solid elements per OpenRadioss specs."""
    mat = _make_mat()
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    with pytest.raises(NotImplementedError, match="shell elements only"):
        solid_update(mat, sig, deps)


def test_shell_update_wrinkled_convenience_function():
    """Verify shell_update_wrinkled functional alias forces wrinkling model."""
    # Material created without iwrinkle
    p = Law58Params(e1=1000.0, e2=1000.0, g0=100.0, iwrinkle=0)
    sig = np.zeros(3)
    deps = np.array([0.03, -0.02, 0.1])
    extra = {}

    s_out, _ = shell_update_wrinkled(p, sig, deps, extra=extra)
    assert extra["fabric_state"][0] == FabricState.WRINKLED
    assert s_out[2] == 0.0  # shear eliminated by wrinkling model
