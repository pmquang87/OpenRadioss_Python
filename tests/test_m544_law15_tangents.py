"""
Tests for LAW15 Consistent Algorithmic Shell Tangent
(/MAT/LAW15, /MAT/CHANG, /MAT/PLAS_ANISO, /MAT/COMP_CHANG).
Milestone M544: Tangent Stiffness Audit.

Verifies:
  1. Central finite difference perturbation:
     C_num[:, j] = (sig(deps + h*e_j) - sig(deps - h*e_j)) / (2*h) with h = 1e-6 or 1e-7.
  2. Elastic regime: exact match between C_el and C_num to within 1e-6.
  3. Matrix-damaged regime: degraded transverse & shear tangent matches C_num to within 1e-6.
  4. Fiber-damaged regime: degraded orthotropic tangent matches C_num to within 1e-6.
  5. Plastic yielding regime: elastoplastic algorithmic tangent matches C_num along
     loading perturbation directions to within 1e-6 for unhardened and hardened states.
  6. Batched multi-element consistency: heterogeneous elements (elastic, fiber-damaged,
     matrix-damaged, yielding, deleted) evaluated concurrently in (n, 3, 3) arrays.
  7. Major symmetry enforcement under symmetric=True.
  8. Defensive empty array handling and package-level dispatch.
"""

from __future__ import annotations

import math
from typing import Any, Dict
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.materials import law15_chang


def _copy_extra(extra: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy arrays inside state extra dictionary."""
    return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in extra.items()}


def _build_test_material(**kwargs: Any):
    """Helper to instantiate LAW15 material with standard composite test properties."""
    defaults = dict(
        id=15,
        rho0=1.5e-9,
        E1=100000.0,
        E2=10000.0,
        nu12=0.3,
        G12=4000.0,
        G23=3000.0,
        G31=4000.0,
        sigyt1=500.0,
        sigyc1=500.0,
        sigyt2=50.0,
        sigyc2=50.0,
        sigt12=40.0,
        sigc12=40.0,
        alpha=1.0,
        b=10.0,
        n=0.8,
        s1=1e10,
        s2=1e10,
        c1=1e10,
        c2=1e10,
        s12=1e10,
        tmax=1e30,
        itype=0,
    )
    defaults.update(kwargs)
    return law15_chang.build_law15(**defaults)


# ============================================================================
# 1. Elastic Regime Tangent Verification
# ============================================================================

def test_law15_tangent_elastic_single_element():
    """Verify exact match between elastic algorithmic tangent and central FD derivative."""
    mat = _build_test_material()
    sig0 = np.array([50.0, 10.0, 5.0])
    deps0 = np.array([1e-4, 5e-5, 2e-5])
    h = 1e-7

    C_num = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        ej = np.zeros(3)
        ej[j] = h
        sp, _, _ = law15_chang.shell_update(mat, sig0, deps0 + ej)
        sm, _, _ = law15_chang.shell_update(mat, sig0, deps0 - ej)
        C_num[:, j] = (sp - sm) / (2.0 * h)

    C_alg = law15_chang.consistent_shell_tangent(mat, sig0)

    assert C_alg.shape == (3, 3)
    diff_max = np.max(np.abs(C_num - C_alg))
    assert diff_max < 1e-6, f"Elastic tangent diff {diff_max} exceeds 1e-6"

    # Verify symmetry
    assert np.allclose(C_alg, C_alg.T, atol=1e-12)


def test_law15_tangent_elastic_varying_states():
    """Verify elastic tangent across multiple non-zero initial stress states."""
    mat = _build_test_material(E1=150000.0, E2=8000.0, nu12=0.28, G12=3500.0)
    h = 1e-7

    for sig0 in [
        np.array([0.0, 0.0, 0.0]),
        np.array([120.0, -15.0, 8.0]),
        np.array([-80.0, 25.0, -12.0]),
    ]:
        deps0 = np.array([2e-5, -1e-5, 3e-5])
        C_num = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            ej = np.zeros(3)
            ej[j] = h
            sp, _, _ = law15_chang.shell_update(mat, sig0, deps0 + ej)
            sm, _, _ = law15_chang.shell_update(mat, sig0, deps0 - ej)
            C_num[:, j] = (sp - sm) / (2.0 * h)

        C_alg = law15_chang.consistent_shell_tangent(mat, sig0)
        diff_max = np.max(np.abs(C_num - C_alg))
        assert diff_max < 1e-6, f"Elastic varying state diff {diff_max} exceeds 1e-6"


# ============================================================================
# 2. Fiber Damaged Regime Tangent Verification
# ============================================================================

@pytest.mark.parametrize("dam_f", [0.8, 0.5, 0.25])
def test_law15_tangent_fiber_damaged_single_element(dam_f: float):
    """Verify degraded tangent matches central FD when fiber damage damt[0] < 1.0."""
    mat = _build_test_material()
    sig0 = np.array([40.0, 8.0, 4.0])
    deps0 = np.array([1e-4, 5e-5, 2e-5])
    h = 1e-7

    extra = {
        "damt15": np.array([[dam_f, 1.0]]),
        "sigr15": np.zeros((1, 6)),
        "wpla15": np.zeros(1),
        "off15": np.ones(1),
        "time": 0.0,
    }

    C_num = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        ej = np.zeros(3)
        ej[j] = h
        sp, _, _ = law15_chang.shell_update(mat, sig0, deps0 + ej, extra=_copy_extra(extra))
        sm, _, _ = law15_chang.shell_update(mat, sig0, deps0 - ej, extra=_copy_extra(extra))
        C_num[:, j] = (sp - sm) / (2.0 * h)

    C_alg = law15_chang.consistent_shell_tangent(mat, sig0, extra=extra)

    diff_max = np.max(np.abs(C_num - C_alg))
    assert diff_max < 1e-6, f"Fiber damaged (dam_f={dam_f}) diff {diff_max} exceeds 1e-6"

    # Verify uniform degradation relative to intact elastic membrane tangent
    C_el = law15_chang.shell_membrane_tangent(mat)
    assert np.allclose(C_alg, dam_f * C_el, atol=1e-10)


# ============================================================================
# 3. Matrix Damaged Regime Tangent Verification
# ============================================================================

@pytest.mark.parametrize("dam_m", [0.75, 0.4, 0.1])
def test_law15_tangent_matrix_damaged_single_element(dam_m: float):
    """Verify transverse & shear degraded tangent matches central FD when damt[1] < 1.0."""
    mat = _build_test_material()
    sig0 = np.array([30.0, 6.0, 3.0])
    deps0 = np.array([1e-4, 5e-5, 2e-5])
    h = 1e-7

    extra = {
        "damt15": np.array([[1.0, dam_m]]),
        "sigr15": np.zeros((1, 6)),
        "wpla15": np.zeros(1),
        "off15": np.ones(1),
        "time": 0.0,
    }

    C_num = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        ej = np.zeros(3)
        ej[j] = h
        sp, _, _ = law15_chang.shell_update(mat, sig0, deps0 + ej, extra=_copy_extra(extra))
        sm, _, _ = law15_chang.shell_update(mat, sig0, deps0 - ej, extra=_copy_extra(extra))
        C_num[:, j] = (sp - sm) / (2.0 * h)

    C_alg = law15_chang.consistent_shell_tangent(mat, sig0, extra=extra)

    diff_max = np.max(np.abs(C_num - C_alg))
    assert diff_max < 1e-6, f"Matrix damaged (dam_m={dam_m}) diff {diff_max} exceeds 1e-6"

    # Longitudinal modulus intact E11
    assert C_alg[0, 0] == pytest.approx(mat.params["E1"])
    # Transverse modulus degraded by dam_m
    assert C_alg[1, 1] == pytest.approx(mat.params["E2"] * dam_m)
    # Shear modulus degraded by dam_m
    assert C_alg[2, 2] == pytest.approx(mat.params["G12"] * dam_m)
    # Poisson coupling zeroed
    assert C_alg[0, 1] == 0.0
    assert C_alg[1, 0] == 0.0


# ============================================================================
# 4. Plastic Yielding Regime Tangent Verification
# ============================================================================

def test_law15_tangent_plastic_pure_transverse_yield():
    """Verify elastoplastic tangent along plastic loading direction under transverse tension."""
    mat = _build_test_material()
    # Stress sits on yield surface: sig_2yt = 50.0
    sig0 = np.array([0.0, 50.0, 0.0])
    deps0 = np.array([0.0, 1e-5, 0.0])
    h = 1e-7

    extra = {
        "damt15": np.array([[1.0, 1.0]]),
        "sigr15": np.zeros((1, 6)),
        "wpla15": np.zeros(1),
        "off15": np.ones(1),
        "time": 0.0,
    }

    C_num = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        ej = np.zeros(3)
        ej[j] = h
        sp, _, _ = law15_chang.shell_update(mat, sig0, deps0 + ej, extra=_copy_extra(extra))
        sm, _, _ = law15_chang.shell_update(mat, sig0, deps0 - ej, extra=_copy_extra(extra))
        C_num[:, j] = (sp - sm) / (2.0 * h)

    C_alg = law15_chang.consistent_shell_tangent(mat, sig0, extra=extra, deps=deps0)

    # Plastic loading direction is index 1 (deps_22)
    diff_dir = np.max(np.abs(C_num[:, 1] - C_alg[:, 1]))
    assert diff_dir < 1e-6, f"Plastic transverse yield col 1 diff {diff_dir} exceeds 1e-6"

    # Verify plastic softening relative to elastic stiffness
    C_el = law15_chang.shell_membrane_tangent(mat)
    assert C_alg[1, 1] < C_el[1, 1]


def test_law15_tangent_plastic_pure_shear_yield():
    """Verify elastoplastic tangent under pure shear yielding (s12 = sigt12 = 40.0)."""
    mat = _build_test_material()
    sig0 = np.array([0.0, 0.0, 40.0])
    deps0 = np.array([0.0, 0.0, 1e-5])
    h = 1e-7

    extra = {
        "damt15": np.array([[1.0, 1.0]]),
        "sigr15": np.zeros((1, 6)),
        "wpla15": np.zeros(1),
        "off15": np.ones(1),
        "time": 0.0,
    }

    C_num = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        ej = np.zeros(3)
        ej[j] = h
        sp, _, _ = law15_chang.shell_update(mat, sig0, deps0 + ej, extra=_copy_extra(extra))
        sm, _, _ = law15_chang.shell_update(mat, sig0, deps0 - ej, extra=_copy_extra(extra))
        C_num[:, j] = (sp - sm) / (2.0 * h)

    C_alg = law15_chang.consistent_shell_tangent(mat, sig0, extra=extra, deps=deps0)

    diff_dir = np.max(np.abs(C_num[:, 2] - C_alg[:, 2]))
    assert diff_dir < 1e-6, f"Pure shear yield col 2 diff {diff_dir} exceeds 1e-6"
    assert C_alg[2, 2] < 1e-6  # Ideal plasticity with zero normal work hardening has ~zero shear tangent


def test_law15_tangent_plastic_combined_normal_shear_hardened():
    """Verify elastoplastic tangent under combined normal+shear stress with work hardening."""
    mat = _build_test_material(b=15.0, n=0.75)
    f1, f2 = mat.params["F1"], mat.params["F2"]
    f11, f22, f33, f12 = mat.params["F11"], mat.params["F22"], mat.params["F33"], mat.params["F12"]

    # Pre-existing plastic work wp = 0.04
    wp0 = 0.04
    fyld0 = 1.0 + 15.0 * (wp0 ** 0.75)

    # Base stress components
    s2_base = 25.0
    rem = 1.0 - (f2 * s2_base + f22 * (s2_base ** 2))
    s12_base = math.sqrt(max(0.0, rem / f33))

    # Scale to sit exactly on the hardened yield surface
    a_w = f22 * (s2_base ** 2) + f33 * (s12_base ** 2)
    b_w = f2 * s2_base
    k_scale = (-b_w + math.sqrt(b_w ** 2 + 4.0 * a_w * fyld0)) / (2.0 * a_w)
    sig0 = np.array([0.0, k_scale * s2_base, k_scale * s12_base])

    # Strain increment into plastic domain
    deps0 = np.array([0.0, 1e-5, 1e-5])
    h = 1e-7

    extra = {
        "damt15": np.array([[1.0, 1.0]]),
        "sigr15": np.zeros((1, 6)),
        "wpla15": np.array([wp0]),
        "off15": np.ones(1),
        "time": 0.0,
    }

    C_num = np.zeros((3, 3), dtype=np.float64)
    for j in range(3):
        ej = np.zeros(3)
        ej[j] = h
        sp, _, _ = law15_chang.shell_update(mat, sig0, deps0 + ej, extra=_copy_extra(extra))
        sm, _, _ = law15_chang.shell_update(mat, sig0, deps0 - ej, extra=_copy_extra(extra))
        C_num[:, j] = (sp - sm) / (2.0 * h)

    C_alg = law15_chang.consistent_shell_tangent(mat, sig0, extra=extra, deps=deps0)

    diff_max = np.max(np.abs(C_num - C_alg))
    assert diff_max < 1e-6, f"Combined hardened plastic yield diff {diff_max} exceeds 1e-6"


# ============================================================================
# 5. Batched Multi-Element Tangent Verification
# ============================================================================

def test_law15_tangent_batched_multi_element():
    """Verify batched (n=5) evaluation across elastic, fiber-damaged, matrix-damaged, yielding, and deleted states."""
    mat = _build_test_material()
    f1, f2 = mat.params["F1"], mat.params["F2"]
    f11, f22, f33, f12 = mat.params["F11"], mat.params["F22"], mat.params["F33"], mat.params["F12"]

    s2_val = 30.0
    rem = 1.0 - (f2 * s2_val + f22 * (s2_val ** 2))
    s12_val = math.sqrt(rem / f33)

    wp0 = 0.05
    fyld0 = 1.0 + 10.0 * (wp0 ** 0.8)
    a_w = f22 * (s2_val ** 2) + f33 * (s12_val ** 2)
    b_w = f2 * s2_val
    k_scale = (-b_w + math.sqrt(b_w ** 2 + 4.0 * a_w * fyld0)) / (2.0 * a_w)
    sig_yielding = k_scale * np.array([0.0, s2_val, s12_val])

    n = 5
    sig_bat = np.zeros((n, 3), dtype=np.float64)
    sig_bat[0] = [50.0, 10.0, 5.0]        # Intact elastic
    sig_bat[1] = [50.0, 10.0, 5.0]        # Fiber damaged
    sig_bat[2] = [50.0, 10.0, 5.0]        # Matrix damaged
    sig_bat[3] = sig_yielding             # Plastic yielding
    sig_bat[4] = [50.0, 10.0, 5.0]        # Deleted element

    deps_bat = np.zeros((n, 3), dtype=np.float64)
    deps_bat[0] = [1e-4, 5e-5, 2e-5]
    deps_bat[1] = [1e-4, 5e-5, 2e-5]
    deps_bat[2] = [1e-4, 5e-5, 2e-5]
    deps_bat[3] = [0.0, 1e-5, 1e-5]
    deps_bat[4] = [1e-4, 5e-5, 2e-5]

    damt_bat = np.ones((n, 2), dtype=np.float64)
    damt_bat[1] = [0.6, 1.0]              # Fiber damaged: dam_f = 0.6
    damt_bat[2] = [1.0, 0.4]              # Matrix damaged: dam_m = 0.4

    wpla_bat = np.zeros(n, dtype=np.float64)
    wpla_bat[3] = wp0

    off_bat = np.ones(n, dtype=np.float64)
    off_bat[4] = 0.0                      # Deleted element

    extra_bat = {
        "damt15": damt_bat,
        "sigr15": np.zeros((n, 6), dtype=np.float64),
        "wpla15": wpla_bat,
        "off15": off_bat,
        "time": 0.0,
    }

    h = 1e-7
    C_num_bat = np.zeros((n, 3, 3), dtype=np.float64)

    for i in range(n):
        for j in range(3):
            ej = np.zeros((n, 3), dtype=np.float64)
            ej[i, j] = h
            sp, _, _ = law15_chang.shell_update(mat, sig_bat, deps_bat + ej, extra=_copy_extra(extra_bat))
            sm, _, _ = law15_chang.shell_update(mat, sig_bat, deps_bat - ej, extra=_copy_extra(extra_bat))
            C_num_bat[i, :, j] = (sp[i] - sm[i]) / (2.0 * h)

    C_alg_bat = law15_chang.consistent_shell_tangent(mat, sig_bat, extra=extra_bat, deps=deps_bat)

    assert C_alg_bat.shape == (n, 3, 3)

    for i in range(n):
        diff_el = np.max(np.abs(C_num_bat[i] - C_alg_bat[i]))
        assert diff_el < 1e-6, f"Batched element {i} diff {diff_el} exceeds 1e-6"

    # Deleted element must have exactly zero tangent
    assert np.all(C_alg_bat[4] == 0.0)


# ============================================================================
# 6. Major Symmetry Flag and Edge Cases
# ============================================================================

def test_law15_tangent_symmetric_flag():
    """Verify symmetric=True enforces major symmetry 0.5*(C + C.T)."""
    mat = _build_test_material()
    sig0 = np.array([0.0, 30.0, 20.0])
    deps0 = np.array([0.0, 1e-5, 1e-5])

    c_nonsym = law15_chang.consistent_shell_tangent(mat, sig0, deps=deps0, symmetric=False)
    c_sym = law15_chang.consistent_shell_tangent(mat, sig0, deps=deps0, symmetric=True)

    assert np.allclose(c_sym, 0.5 * (c_nonsym + c_nonsym.T), atol=1e-12)
    assert np.allclose(c_sym, c_sym.T, atol=1e-12)


def test_law15_tangent_empty_and_zero_shape():
    """Verify empty input returns (0, 3, 3) without errors."""
    mat = _build_test_material()
    c_empty = law15_chang.consistent_shell_tangent(mat, np.empty((0, 3)))
    assert c_empty.shape == (0, 3, 3)


def test_law15_tangent_package_dispatch():
    """Verify dispatch via materials.shell_layer_tangent and materials.consistent_shell_tangent."""
    mat = _build_test_material()
    sig = np.array([[50.0, 10.0, 5.0]])

    c1 = materials.shell_layer_tangent(mat, sig)
    c2 = materials.consistent_shell_tangent(mat, sig)

    assert c1.shape == (1, 3, 3)
    assert np.allclose(c1, c2)
