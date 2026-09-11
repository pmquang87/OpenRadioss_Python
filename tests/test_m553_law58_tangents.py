"""
LAW58 Consistent Algorithmic Tangent Stiffness Tensor Verification Suite.
Milestone M553: Consistent Tangents Auditor (/MAT/LAW58, /MAT/FABR_A).

Exhaustive verification of consistent algorithmic tangent operators in
pyradioss/materials/law58_fabr_a.py:
1. Shell membrane algorithmic tangent C^alg = d(sigma)/d(deps) of shape (NEL, 3, 3) or (3, 3),
   verified against central finite difference perturbations:
     C_num[:, j] = (sigma(deps + h * e_j) - sigma(deps - h * e_j)) / (2 * h)
   for step sizes h in [1e-7, 1e-6], achieving relative error < 1e-4 in non-linear regimes
   and < 1e-8 in linear elastic regimes.
2. Verification across all physical regimes:
   - Linear elastic tension without yarn contact (y_c + y_t >= 0)
   - Active crimp interchange coupling with yarn contact (y_c + y_t < 0)
   - Pre-locking Trellis shear (|tan phi| <= tan phi_lock)
   - Post-locking Trellis shear (|tan phi| > tan phi_lock)
   - Nonlinear curve interpolation (FUN_A1, FUN_A2, FUN_A3)
   - Unloading regimes (FUN_A4, FUN_A5, FUN_A6)
   - Zero-stress folding regime (A/A_0 <= A_rel)
   - Mixed multi-axial loading states: uniaxial warp, uniaxial weft, pure shear,
     biaxial tension, combined tension and Trellis shear.
3. Batch vectorization (n = 1, 8, 32 elements simultaneously) and element-by-element equivalence.
4. Plane-stress consistency: tangent matches directional perturbation of shell_update_law58.
5. Calling conventions, optional major symmetry, and package-level dispatch.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pytest

from pyradioss.materials.law58_fabr_a import (
    Law58Params,
    FabricAMaterial,
    build_law58,
    crimp_interchange,
    find_curve_intersection,
    init_material_axes_cm58,
    shell_membrane_tangent,
    shell_update,
    shell_update_law58,
    sound_speed_shell_law58,
    tangent_law58_shell,
    tangent_shell,
    consistent_shell_tangent,
    _copy_extra,
    _eval_curve,
)
import pyradioss.materials as materials


# =============================================================================
# Helper Utilities: Independent Central Finite Difference Numerical Tangents
# =============================================================================

def _compute_shell_numerical_tangent(
    mat: Any,
    deps: np.ndarray,
    sig: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    h: float = 1e-7,
) -> np.ndarray:
    """Compute (NEL, 3, 3) or (3, 3) numerical plane-stress membrane tangent
    via independent central finite differences:
        C_num[:, :, j] = (sigma(deps + h * e_j) - sigma(deps - h * e_j)) / (2 * h)
    """
    deps_arr = np.asarray(deps, dtype=float)
    single_deps = (deps_arr.ndim == 1)
    if single_deps:
        deps_arr = deps_arr.reshape(1, -1)
    nel = deps_arr.shape[0]

    if sig is not None:
        s_old = np.asarray(sig, dtype=float)
        single_sig = (s_old.ndim == 1)
        if single_sig:
            s_old = s_old.reshape(1, -1)
        if s_old.shape[0] == 1 and nel > 1:
            s_old = np.repeat(s_old, nel, axis=0)
    else:
        single_sig = True
        s_old = np.zeros((nel, deps_arr.shape[1]), dtype=float)

    single = single_deps and single_sig
    C_num = np.zeros((nel, 3, 3), dtype=float)

    for j in range(3):
        ej = np.zeros_like(deps_arr)
        ej[:, j] = h

        ex_p = _copy_extra(extra)
        ex_m = _copy_extra(extra)

        sp, _ = shell_update_law58(mat, s_old.copy(), deps_arr + ej, dt=dt, extra=ex_p)
        sm, _ = shell_update_law58(mat, s_old.copy(), deps_arr - ej, dt=dt, extra=ex_m)

        if sp.ndim == 1:
            sp = sp.reshape(1, -1)
            sm = sm.reshape(1, -1)

        C_num[:, :, j] = (sp[:, :3] - sm[:, :3]) / (2.0 * h)

    return C_num[0] if single else C_num


def _rel_err(C_alg: np.ndarray, C_num: np.ndarray) -> float:
    """Compute relative Frobenius norm error between algorithmic and numerical tangents."""
    norm_num = float(np.linalg.norm(C_num))
    if norm_num < 1e-12:
        return float(np.max(np.abs(C_alg - C_num)))
    return float(np.linalg.norm(C_alg - C_num) / norm_num)


# =============================================================================
# 1. Linear Elastic Regime without Yarn Contact (yc + yt >= 0)
# =============================================================================

def test_linear_elastic_uncoupled_no_yarn_contact():
    """Verify tangent matches central finite difference perturbations to < 1e-8
    in the uncoupled elastic regime where yarn heights do not make contact (yc + yt >= 0).
    """
    p = Law58Params(
        e1=2000.0, e2=1500.0, n1=1, n2=1, s1=0.1, s2=0.1,
        g0=80.0, gt=240.0, alphat=45.0,
    )

    # Reference ground state tangent recovery
    C_ref = shell_membrane_tangent(p)
    assert C_ref.shape == (3, 3)
    assert C_ref[0, 0] == 2000.0
    assert C_ref[1, 1] == 1500.0
    assert C_ref[2, 2] == 80.0

    # Compressive strains (yarns deflect outwards, yc > 0, yt > 0, fn = 0)
    deps_comp = np.array([-0.005, -0.005, 0.01], dtype=float)
    sig_init = np.zeros(3, dtype=float)

    extra: Dict[str, Any] = {}
    yc, yt, fc, ft, fn, dc, dt = crimp_interchange(p, p.lc0 * (1.0 + math.exp(-0.005) - 1.0), p.lt0 * (1.0 + math.exp(-0.005) - 1.0))
    assert yc + yt >= 0.0, f"Expected uncoupled regime yc+yt>=0, got {yc+yt}"
    assert fn == 0.0, f"Contact force must be zero without contact, got {fn}"

    for h in (1e-7, 1e-6):
        C_alg = tangent_law58_shell(p, sig=sig_init, deps=deps_comp, extra=extra, h=h)
        C_num = _compute_shell_numerical_tangent(p, deps_comp, sig=sig_init, extra=extra, h=h)

        err = _rel_err(C_alg, C_num)
        assert err < 1e-8, f"Uncoupled elastic FD error {err:.3e} exceeds 1e-8 for h={h}"

    # Diagonal terms must be strictly positive
    assert C_alg[0, 0] > 0.0
    assert C_alg[1, 1] > 0.0
    assert C_alg[2, 2] > 0.0


# =============================================================================
# 2. Active Crimp Interchange Coupling with Yarn Contact (yc + yt < 0)
# =============================================================================

def test_active_crimp_interchange_yarn_contact():
    """Verify consistent tangent in the coupled crimp interchange regime
    where interlacing yarn contact is active (yc + yt < 0, fn > 0).
    """
    p = Law58Params(
        e1=1800.0, e2=1200.0, n1=1, n2=1, s1=0.1, s2=0.1,
        flex=1e-3, g0=60.0, gt=180.0, alphat=35.0,
    )

    # 1. Warp tension with weft relaxed: warp straightens, pulls weft inward
    deps_warp = np.array([0.04, 0.0, 0.02], dtype=float)
    sig = np.zeros(3, dtype=float)
    extra_warp: Dict[str, Any] = {}

    s_new, _ = shell_update_law58(p, sig, deps_warp, extra=extra_warp)
    fn = extra_warp["fn"][0]
    yc = extra_warp["yc"][0]
    yt = extra_warp["yt"][0]
    assert fn > 0.0, f"Expected positive yarn contact force, got {fn}"
    assert math.isclose(yc, -yt, rel_tol=1e-5), f"Coupled crimp interchange requires yc = -yt, got {yc}, {yt}"

    for h in (1e-7, 1e-6):
        C_alg = tangent_law58_shell(p, sig=sig, deps=deps_warp, extra=extra_warp, h=h)
        C_num = _compute_shell_numerical_tangent(p, deps_warp, sig=sig, extra=extra_warp, h=h)

        err = _rel_err(C_alg, C_num)
        assert err < 1e-4, f"Coupled crimp warp tension FD error {err:.3e} exceeds 1e-4 for h={h}"
        assert C_alg[0, 0] > 0.0
        assert C_alg[1, 1] > 0.0

    # 2. Weft tension with warp relaxed
    deps_weft = np.array([0.0, 0.05, -0.01], dtype=float)
    extra_weft: Dict[str, Any] = {}
    shell_update_law58(p, sig, deps_weft, extra=extra_weft)
    assert extra_weft["fn"][0] > 0.0

    for h in (1e-7, 1e-6):
        C_alg = tangent_law58_shell(p, sig=sig, deps=deps_weft, extra=extra_weft, h=h)
        C_num = _compute_shell_numerical_tangent(p, deps_weft, sig=sig, extra=extra_weft, h=h)

        err = _rel_err(C_alg, C_num)
        assert err < 1e-4, f"Coupled crimp weft tension FD error {err:.3e} exceeds 1e-4 for h={h}"

    # Cross coupling between warp and weft due to Poisson crimp interchange
    assert C_alg[0, 1] != 0.0 or C_alg[1, 0] != 0.0


# =============================================================================
# 3. Pre-Locking and Post-Locking Trellis Shear Regimes
# =============================================================================

def test_trellis_shear_pre_and_post_locking():
    """Verify Trellis shear tangent before lock angle (G0) and after lock angle (G_post)."""
    g0 = 50.0
    gt = 250.0
    alphat = 45.0  # tan_lock = 1.0
    p = Law58Params(e1=1000.0, e2=1000.0, g0=g0, gt=gt, alphat=alphat)
    assert math.isclose(p.tan_lock, 1.0, rel_tol=1e-5)

    g_secant = gt / (1.0 + p.tan_lock**2)
    assert math.isclose(p.g_post, g_secant, rel_tol=1e-5)

    # 1. Pre-locking shear (|tan phi| = 0.4 <= 1.0)
    deps_pre = np.array([0.01, 0.01, 0.4], dtype=float)
    sig = np.zeros(3, dtype=float)

    for h in (1e-7, 1e-6):
        C_alg = tangent_law58_shell(p, sig=sig, deps=deps_pre, h=h)
        C_num = _compute_shell_numerical_tangent(p, deps_pre, sig=sig, h=h)

        err = _rel_err(C_alg, C_num)
        assert err < 1e-8, f"Pre-locking shear FD error {err:.3e} exceeds 1e-8 for h={h}"
        assert math.isclose(C_alg[2, 2], g0, rel_tol=1e-4)

    # Negative pre-locking shear symmetry
    deps_neg = np.array([0.01, 0.01, -0.4], dtype=float)
    C_neg = tangent_law58_shell(p, sig=sig, deps=deps_neg, h=1e-7)
    assert math.isclose(C_neg[2, 2], g0, rel_tol=1e-4)

    # 2. Post-locking shear (|tan phi| = 1.5 > 1.0)
    deps_post = np.array([0.01, 0.01, 1.5], dtype=float)
    for h in (1e-7, 1e-6):
        C_alg_post = tangent_law58_shell(p, sig=sig, deps=deps_post, h=h)
        C_num_post = _compute_shell_numerical_tangent(p, deps_post, sig=sig, h=h)

        err = _rel_err(C_alg_post, C_num_post)
        assert err < 1e-8, f"Post-locking shear FD error {err:.3e} exceeds 1e-8 for h={h}"
        assert math.isclose(C_alg_post[2, 2], p.g_post, rel_tol=1e-4)

    # Negative post-locking shear
    deps_neg_post = np.array([0.01, 0.01, -1.5], dtype=float)
    C_neg_post = tangent_law58_shell(p, sig=sig, deps=deps_neg_post, h=1e-7)
    assert math.isclose(C_neg_post[2, 2], p.g_post, rel_tol=1e-4)


def test_shear_tangent_continuity_across_lock_angle():
    """Verify shear tangent behavior immediately below and above lock angle."""
    p = Law58Params(e1=1000.0, e2=1000.0, g0=60.0, gt=200.0, alphat=30.0)
    tan_lock = p.tan_lock
    delta = 1e-5

    deps_below = np.array([0.0, 0.0, tan_lock - delta], dtype=float)
    deps_above = np.array([0.0, 0.0, tan_lock + delta], dtype=float)
    sig = np.zeros(3, dtype=float)

    C_below = tangent_law58_shell(p, sig=sig, deps=deps_below, h=1e-7)
    C_above = tangent_law58_shell(p, sig=sig, deps=deps_above, h=1e-7)

    assert math.isclose(C_below[2, 2], p.g0, rel_tol=1e-3)
    assert math.isclose(C_above[2, 2], p.g_post, rel_tol=1e-3)


# =============================================================================
# 4. Nonlinear Curve Interpolation (FUN_A1, FUN_A2, FUN_A3)
# =============================================================================

def test_nonlinear_curve_interpolation_tangents():
    """Verify tangent against numerical central finite differences when
    FUN_A1 (warp), FUN_A2 (weft), and FUN_A3 (shear) are specified.
    """
    # 1. Callable curves
    def warp_curve(dcc: float) -> float:
        return 800.0 * dcc + 2500.0 * dcc**2

    def weft_curve(dtt: float) -> float:
        return 600.0 * dtt + 1800.0 * dtt**2

    def shear_curve(phi_deg: float) -> float:
        # Trellis shear stress vs angle in degrees
        return 1.5 * phi_deg + 0.03 * phi_deg**2

    p = Law58Params(
        e1=1000.0, e2=1000.0, n1=1, n2=1, s1=0.08, s2=0.08,
        fun_a1=warp_curve, c1=1.2,
        fun_a2=weft_curve, c2=1.4,
        fun_a3=shear_curve, c3=1.1,
    )

    deps = np.array([0.03, 0.025, 0.35], dtype=float)
    sig = np.zeros(3, dtype=float)

    for h in (1e-7, 1e-6):
        C_alg = tangent_law58_shell(p, sig=sig, deps=deps, h=h)
        C_num = _compute_shell_numerical_tangent(p, deps, sig=sig, h=h)

        err = _rel_err(C_alg, C_num)
        assert err < 1e-4, f"Nonlinear curve tangent error {err:.3e} exceeds 1e-4 for h={h}"

    # 2. Piecewise tabular objects
    class TabularCurve:
        def __init__(self, xs: list[float], ys: list[float]):
            self.x = np.array(xs, dtype=float)
            self.y = np.array(ys, dtype=float)

    tab_warp = TabularCurve([0.0, 0.02, 0.05, 0.1, 0.2], [0.0, 15.0, 45.0, 110.0, 250.0])
    tab_weft = TabularCurve([0.0, 0.02, 0.05, 0.1, 0.2], [0.0, 12.0, 38.0, 95.0, 210.0])
    tab_shear = TabularCurve([0.0, 15.0, 30.0, 45.0, 60.0], [0.0, 20.0, 50.0, 90.0, 160.0])

    p_tab = Law58Params(
        e1=1000.0, e2=1000.0, n1=1, n2=1, s1=0.08, s2=0.08,
        fun_a1=tab_warp, c1=1.0,
        fun_a2=tab_weft, c2=1.0,
        fun_a3=tab_shear, c3=1.0,
    )

    deps_tab = np.array([0.04, 0.03, 0.5], dtype=float)
    for h in (1e-7, 1e-6):
        C_alg = tangent_law58_shell(p_tab, sig=sig, deps=deps_tab, h=h)
        C_num = _compute_shell_numerical_tangent(p_tab, deps_tab, sig=sig, h=h)

        err = _rel_err(C_alg, C_num)
        assert err < 1e-4, f"Tabular curve tangent error {err:.3e} exceeds 1e-4 for h={h}"


# =============================================================================
# 5. Unloading Regimes (FUN_A4, FUN_A5, FUN_A6)
# =============================================================================

def test_unloading_regimes_fun_a4_a5_a6():
    """Verify consistent tangent in unloading regimes with hysteresis curves
    FUN_A4 (warp unload), FUN_A5 (weft unload), and FUN_A6 (shear unload).
    """
    c_load_w = [(0.0, 0.0), (0.05, 50.0), (0.1, 120.0)]
    c_unload_w = [(0.0, 0.0), (0.05, 30.0), (0.1, 120.0)]

    c_load_t = [(0.0, 0.0), (0.05, 60.0), (0.1, 150.0)]
    c_unload_t = [(0.0, 0.0), (0.05, 35.0), (0.1, 150.0)]

    c_load_s = [(0.0, 0.0), (30.0, 40.0), (60.0, 100.0)]
    c_unload_s = [(0.0, 0.0), (30.0, 20.0), (60.0, 100.0)]

    p = Law58Params(
        e1=1000.0, e2=1000.0, n1=1, n2=1, s1=0.08, s2=0.08,
        fun_a1=c_load_w, fun_a4=c_unload_w, scale4=1.0,
        fun_a2=c_load_t, fun_a5=c_unload_t, scale5=1.0,
        fun_a3=c_load_s, fun_a6=c_unload_s, scale6=1.0,
    )
    assert p.unload == 1, "Unloading flag must be active when unloading curves are defined"

    # Step 1: Pre-load to high tension and shear to establish historical maxima
    extra: Dict[str, Any] = {}
    sig0 = np.zeros(3, dtype=float)
    deps_load = np.array([0.07, 0.06, 0.7], dtype=float)
    sig_loaded, _ = shell_update_law58(p, sig0, deps_load, extra=extra)

    assert sig_loaded[0] > 0.0
    assert sig_loaded[1] > 0.0
    assert sig_loaded[2] > 0.0

    # Step 2: Unload across all three directions
    deps_unload = np.array([-0.02, -0.02, -0.25], dtype=float)

    for h in (1e-7, 1e-6):
        C_alg = tangent_law58_shell(p, sig=sig_loaded, deps=deps_unload, extra=extra, h=h)
        C_num = _compute_shell_numerical_tangent(p, deps_unload, sig=sig_loaded, extra=extra, h=h)

        err = _rel_err(C_alg, C_num)
        assert err < 1e-4, f"Hysteresis unloading tangent error {err:.3e} exceeds 1e-4 for h={h}"

    # Verify unloading tangent is strictly positive on diagonal
    assert C_alg[0, 0] > 0.0
    assert C_alg[1, 1] > 0.0
    assert C_alg[2, 2] > 0.0


# =============================================================================
# 6. Zero-Stress Folding Regime (A / A_0 <= A_rel)
# =============================================================================

def test_zero_stress_folding_regime():
    """Verify tangent identically vanishes when fabric area ratio A/A0 <= A_rel
    and smooth transition tangent in the boundary layer.
    """
    arel = 0.85
    p = Law58Params(e1=1000.0, e2=1000.0, arel=arel, g0=50.0)

    # 1. Deep folding compaction: A/A0 = (1 + ec)(1 + et) ~ exp(-0.2 - 0.2) = 0.67 <= 0.85
    deps_fold = np.array([-0.2, -0.2, 0.05], dtype=float)
    sig = np.zeros(3, dtype=float)

    C_alg = tangent_law58_shell(p, sig=sig, deps=deps_fold, h=1e-7)
    C_num = _compute_shell_numerical_tangent(p, deps_fold, sig=sig, h=1e-7)

    # Stresses and tangent must be identically zero
    np.testing.assert_allclose(C_alg, 0.0, atol=1e-12)
    np.testing.assert_allclose(C_num, 0.0, atol=1e-12)

    # 2. Smooth transition zone: arel < rel_area < arel2
    # arel2 = 1.0 + 0.5 * (0.85 - 1.0) = 0.925
    # Choose strains so rel_area ~ 0.89
    deps_trans = np.array([-0.06, -0.05, 0.02], dtype=float)
    for h in (1e-7, 1e-6):
        C_trans_alg = tangent_law58_shell(p, sig=sig, deps=deps_trans, h=h)
        C_trans_num = _compute_shell_numerical_tangent(p, deps_trans, sig=sig, h=h)

        err = _rel_err(C_trans_alg, C_trans_num)
        assert err < 1e-4, f"Folding transition tangent error {err:.3e} exceeds 1e-4 for h={h}"


# =============================================================================
# 7. Mixed Multi-Axial Loading States
# =============================================================================

@pytest.mark.parametrize(
    "case_name, deps_vec",
    [
        ("pure_uniaxial_warp", np.array([0.035, 0.0, 0.0])),
        ("pure_uniaxial_weft", np.array([0.0, 0.045, 0.0])),
        ("pure_trellis_shear", np.array([0.0, 0.0, 0.3])),
        ("equibiaxial_tension", np.array([0.03, 0.03, 0.0])),
        ("nonequibiaxial_tension", np.array([0.04, 0.015, 0.0])),
        ("tension_and_prelock_shear", np.array([0.025, 0.03, 0.4])),
        ("tension_and_postlock_shear", np.array([0.03, 0.02, 1.25])),
        ("negative_shear_and_biaxial", np.array([0.02, 0.035, -0.6])),
    ],
)
def test_multiaxial_loading_states_tangents(case_name: str, deps_vec: np.ndarray):
    """Verify tangent against central finite differences across all multiaxial loading modes."""
    p = Law58Params(
        e1=1400.0, b1=120.0, e2=1100.0, b2=90.0,
        n1=2, n2=3, s1=0.1, s2=0.1,
        g0=70.0, gt=220.0, alphat=40.0,
    )
    sig = np.zeros(3, dtype=float)
    extra: Dict[str, Any] = {}

    for h in (1e-7, 1e-6):
        C_alg = tangent_law58_shell(p, sig=sig, deps=deps_vec, extra=extra, h=h)
        C_num = _compute_shell_numerical_tangent(p, deps_vec, sig=sig, extra=extra, h=h)

        err = _rel_err(C_alg, C_num)
        assert err < 1e-4, f"Multiaxial state '{case_name}' error {err:.3e} exceeds 1e-4 for h={h}"


def test_dynamic_fiber_damping_and_sliding_friction_tangent():
    """Verify tangent under dynamic rate effects: viscous fiber damping (df > 0)
    and interlacing sliding friction (ds > 0).
    """
    p = Law58Params(
        e1=1500.0, e2=1200.0, n1=1, n2=1, s1=0.1, s2=0.1,
        g0=50.0, gt=150.0, alphat=35.0,
        df=0.08, ds=0.15, gfrot=120.0, rho0=1.2e-6,
    )
    dt = 1e-5
    extra = {
        "area": 2.5,
        "thk": 1.2,
        "sigv_xy": np.zeros(1, dtype=float),
    }

    deps = np.array([0.025, 0.02, 0.15], dtype=float)
    sig = np.zeros(3, dtype=float)

    for h in (1e-7, 1e-6):
        C_alg = tangent_law58_shell(p, sig=sig, deps=deps, dt=dt, extra=extra, h=h)
        C_num = _compute_shell_numerical_tangent(p, deps, sig=sig, dt=dt, extra=extra, h=h)

        err = _rel_err(C_alg, C_num)
        assert err < 1e-4, f"Dynamic damping & friction tangent error {err:.3e} exceeds 1e-4 for h={h}"


# =============================================================================
# 8. Batch Vectorization (n = 1, 8, 32 elements simultaneously)
# =============================================================================

@pytest.mark.parametrize("nel", [1, 8, 32])
def test_batch_vectorization_and_slice_equivalence(nel: int):
    """Verify tangent evaluation across multi-element batches of size n = 1, 8, 32.
    Each slice C[i] of the vectorized batch must match the single-element tangent to within 1e-12.
    """
    p = Law58Params(
        e1=1600.0, b1=80.0, e2=1200.0, b2=60.0,
        n1=1, n2=2, s1=0.12, s2=0.10,
        g0=65.0, gt=190.0, alphat=38.0,
    )

    rng = np.random.default_rng(42 + nel)
    deps_batch = np.zeros((nel, 3), dtype=float)
    deps_batch[:, 0] = rng.uniform(0.005, 0.04, size=nel)  # warp strain
    deps_batch[:, 1] = rng.uniform(0.005, 0.035, size=nel) # weft strain
    deps_batch[:, 2] = rng.uniform(-0.8, 0.8, size=nel)    # shear strain

    sig_batch = np.zeros((nel, 3), dtype=float)

    # 1. Evaluate full batch
    C_batch = tangent_law58_shell(p, sig=sig_batch, deps=deps_batch, h=1e-7)
    assert C_batch.shape == (nel, 3, 3)

    # 2. Compare against single-element evaluations
    for i in range(nel):
        deps_single = deps_batch[i]
        sig_single = sig_batch[i]
        C_single = tangent_law58_shell(p, sig=sig_single, deps=deps_single, h=1e-7)
        assert C_single.shape == (3, 3)

        target = C_batch[i]
        np.testing.assert_allclose(
            target,
            C_single,
            rtol=1e-11,
            atol=1e-11,
            err_msg=f"Batch element {i}/{nel} mismatch vs single-element evaluation",
        )

        # Numerical finite difference check on each slice
        C_num = _compute_shell_numerical_tangent(p, deps_single, sig=sig_single, h=1e-7)
        err = _rel_err(target, C_num)
        assert err < 1e-4, f"Batch element {i} FD error {err:.3e} exceeds 1e-4"


# =============================================================================
# 9. Plane-Stress Directional Perturbation Consistency
# =============================================================================

def test_plane_stress_directional_perturbation_consistency():
    """Verify that algorithmic consistent tangent C satisfies directional perturbation identity:
        sigma(deps + delta_eps) - sigma(deps) = C * delta_eps + O(||delta_eps||^2)
    to within O(epsilon) < 1e-5 for arbitrary perturbation directions.
    """
    p = Law58Params(
        e1=1500.0, e2=1100.0, n1=1, n2=1, s1=0.1, s2=0.1,
        g0=55.0, gt=170.0, alphat=35.0,
    )
    deps_base = np.array([0.03, 0.02, 0.4], dtype=float)
    sig_base = np.zeros(3, dtype=float)
    extra_init: Dict[str, Any] = {}

    s0, _ = shell_update_law58(p, sig_base, deps_base, extra=_copy_extra(extra_init))
    C_alg = tangent_law58_shell(p, sig=sig_base, deps=deps_base, extra=_copy_extra(extra_init), h=1e-7)

    rng = np.random.default_rng(123)
    eps_scale = 1e-6

    for k in range(5):
        dir_vec = rng.standard_normal(3)
        dir_vec /= np.linalg.norm(dir_vec)
        delta_eps = eps_scale * dir_vec

        s_pert, _ = shell_update_law58(p, sig_base, deps_base + delta_eps, extra=_copy_extra(extra_init))

        actual_d_sig = s_pert[:3] - s0[:3]
        tangent_d_sig = C_alg @ delta_eps

        # Residual must be of order O(eps_scale) in forward perturbation
        residual = np.linalg.norm(actual_d_sig - tangent_d_sig)
        rel_residual = residual / np.linalg.norm(actual_d_sig)
        assert rel_residual < 1e-4, f"Perturbation {k} directional Taylor residual {rel_residual:.3e} exceeds 1e-4"


# =============================================================================
# 10. Calling Conventions, Symmetry Option, and Package-Level Dispatch
# =============================================================================

def test_calling_conventions_and_options():
    """Verify flexible calling conventions, symmetric tangent option, and package dispatch."""
    p = Law58Params(e1=1200.0, e2=1000.0, g0=50.0)

    # 1. No arguments: reference elastic membrane tangent
    D_ground = tangent_law58_shell(p)
    assert D_ground.shape == (3, 3)
    assert D_ground[0, 0] > 0.0
    assert D_ground[1, 1] > 0.0
    assert D_ground[2, 2] > 0.0

    # 2. Positional vs keyword arguments
    deps = np.array([0.02, 0.015, 0.05], dtype=float)
    sig = np.zeros(3, dtype=float)

    D_pos = tangent_law58_shell(p, sig, deps)
    D_kw = tangent_law58_shell(p, sig=sig, deps=deps)
    D_deps_only = tangent_law58_shell(p, deps=deps)

    np.testing.assert_allclose(D_pos, D_kw, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(D_pos, D_deps_only, rtol=1e-12, atol=1e-12)

    # 3. Optional major symmetry
    D_asym = tangent_law58_shell(p, sig=sig, deps=deps, symmetric=False)
    D_sym = tangent_law58_shell(p, sig=sig, deps=deps, symmetric=True)
    np.testing.assert_allclose(D_sym, D_sym.T, atol=1e-12)
    np.testing.assert_allclose(D_sym, 0.5 * (D_asym + D_asym.T), atol=1e-12)

    # 4. Package-level dispatch via materials.shell_layer_tangent
    rec = type("Rec", (), {})()
    rec.id = 58
    rec.title = "FABR_A"
    rec.density = 1.0e-6
    rec.params = {
        "MAT_E1": 2000.0,
        "MAT_E2": 1500.0,
        "MAT_G0": 60.0,
        "MAT_GI": 180.0,
        "MAT_ALPHA": 40.0,
    }
    mat = build_law58(rec)

    extra = {"eps58": np.zeros((1, 3)), "yc": np.zeros(1), "yt": np.zeros(1)}
    D_pkg = materials.shell_layer_tangent(mat, sig=np.zeros((1, 3)), extra=extra)
    assert D_pkg.shape in ((1, 3, 3), (3, 3))
    D_pkg_mat = D_pkg[0] if D_pkg.ndim == 3 else D_pkg
    assert D_pkg_mat[0, 0] > 0.0
    assert D_pkg_mat[1, 1] > 0.0
    assert D_pkg_mat[2, 2] > 0.0

    # 5. Package-level membrane tangent
    D_mem = materials.shell_membrane_tangent(mat)
    assert D_mem.shape == (3, 3)
    assert D_mem[0, 0] == 2000.0
    assert D_mem[1, 1] == 1500.0
    assert D_mem[2, 2] == 60.0
