"""
LAW48 Consistent Algorithmic Tangent Stiffness Tensor Verification Suite.
Milestone M552: Consistent Tangents Auditor.

Exhaustive verification of consistent algorithmic tangent operators in
pyradioss/materials/law48_zhao.py:
1. Solid algorithmic tangent D^alg = d(sigma)/d(deps) of shape (NEL, 6, 6) or (6, 6),
   verified against central finite difference perturbations:
     D_num[:, j] = (sigma(deps + h*e_j) - sigma(deps - h*e_j)) / (2*h)
   for step sizes h in [1e-7, 1e-6], achieving relative error < 1e-4 (plastic)
   and < 1e-8 (elastic).
2. Shell membrane plane-stress tangent C^alg = d(sigma)/d(deps) of shape (NEL, 3, 3)
   or (3, 3), verified against central finite difference perturbations for
   step sizes h in [1e-7, 1e-6].
3. Comprehensive verification across:
   - Elastic regime (pure Hookean tangent, exact major symmetry and positive definiteness)
   - Yielding with static hardening (P_A)
   - Dynamic yielding with coupled rate sensitivity (P_B) and power-law rate sensitivity (P_C)
   - Stress saturation regime (YLD = S_max + P_C with zero hardening slope)
   - Tensile damage regime (FAIL in (0, 1))
   - Fully eroded regime (element deletion via eps_p > eps_max, eps_t >= eps_r2, off = 0)
   - Multiaxial loading states (uniaxial tension, equibiaxial tension, pure shear,
     triaxial compression, mixed shear-normal)
   - Vectorization with N >= 16 elements simultaneously (N = 20 heterogeneous batch)
   - Flexible calling conventions, package-level dispatch, and major symmetry option.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pytest

from pyradioss.materials.law48_zhao import (
    Law48Params,
    build_law48,
    consistent_shell_tangent,
    consistent_solid_tangent,
    eval_yield_and_hardening,
    shell_membrane_tangent,
    shell_sound_speed,
    shell_tangent,
    shell_update,
    shell_update_law48,
    solid_sound_speed,
    solid_tangent,
    solid_update,
    solid_update_law48,
    tangent_law48_shell,
    tangent_law48_solid,
    tensile_failure_factor,
)
import pyradioss.materials as materials


# =============================================================================
# Helper Utilities: Independent Central Finite Difference Numerical Tangents
# =============================================================================

def _copy_extra(extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Deep copy state extra dictionary arrays and sub-dictionaries."""
    if extra is None:
        return None
    res: Dict[str, Any] = {}
    for k, v in extra.items():
        if isinstance(v, np.ndarray):
            res[k] = v.copy()
        elif isinstance(v, dict):
            res[k] = _copy_extra(v)
        elif hasattr(v, "copy"):
            try:
                res[k] = v.copy()
            except Exception:
                res[k] = v
        else:
            res[k] = v
    return res


def _compute_solid_numerical_tangent(
    mat: Any,
    deps: np.ndarray,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    h: float = 1e-7,
) -> np.ndarray:
    """Compute (NEL, 6, 6) or (6, 6) numerical solid tangent via central finite difference:

    D_num[:, :, j] = (sigma(deps + h * e_j) - sigma(deps - h * e_j)) / (2 * h)
    """
    deps_arr = np.asarray(deps, dtype=float)
    is_1d = (deps_arr.ndim == 1)
    if is_1d:
        deps_arr = deps_arr.reshape(1, 6)
    nel = deps_arr.shape[0]

    s_old = np.asarray(sig, dtype=float) if sig is not None else np.zeros((nel, 6), dtype=float)
    if s_old.ndim == 1:
        s_old = s_old.reshape(1, 6)

    ep_old = np.asarray(epsp, dtype=float).flatten() if epsp is not None else np.zeros(nel, dtype=float)
    if len(ep_old) == 1 and nel > 1:
        ep_old = np.full(nel, ep_old[0], dtype=float)

    D_num = np.zeros((nel, 6, 6), dtype=float)

    for j in range(6):
        ej = np.zeros_like(deps_arr)
        ej[:, j] = h

        ex_p = _copy_extra(extra)
        ex_m = _copy_extra(extra)

        sp, _, _ = solid_update_law48(mat, s_old.copy(), deps_arr + ej, epsp=ep_old.copy(), dt=dt, extra=ex_p)
        sm, _, _ = solid_update_law48(mat, s_old.copy(), deps_arr - ej, epsp=ep_old.copy(), dt=dt, extra=ex_m)

        if sp.ndim == 1:
            sp = sp.reshape(1, 6)
            sm = sm.reshape(1, 6)

        D_num[:, :, j] = (sp - sm) / (2.0 * h)

    return D_num[0] if is_1d else D_num


def _compute_shell_numerical_tangent(
    mat: Any,
    deps: np.ndarray,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    h: float = 1e-7,
) -> np.ndarray:
    """Compute (NEL, 3, 3) or (3, 3) numerical shell plane-stress tangent via central finite difference:

    C_num[:, :, j] = (sigma(deps + h * e_j) - sigma(deps - h * e_j)) / (2 * h)
    """
    deps_arr = np.asarray(deps, dtype=float)
    is_1d = (deps_arr.ndim == 1)
    if is_1d:
        deps_arr = deps_arr.reshape(1, -1)
    nel = deps_arr.shape[0]

    s_old = np.asarray(sig, dtype=float) if sig is not None else np.zeros((nel, deps_arr.shape[1]), dtype=float)
    if s_old.ndim == 1:
        s_old = s_old.reshape(1, -1)

    ep_old = np.asarray(epsp, dtype=float).flatten() if epsp is not None else np.zeros(nel, dtype=float)
    if len(ep_old) == 1 and nel > 1:
        ep_old = np.full(nel, ep_old[0], dtype=float)

    C_num = np.zeros((nel, 3, 3), dtype=float)

    for j in range(3):
        ej = np.zeros_like(deps_arr)
        ej[:, j] = h

        ex_p = _copy_extra(extra)
        ex_m = _copy_extra(extra)

        sp, _ = shell_update_law48(mat, s_old.copy(), deps_arr + ej, epsp=ep_old.copy(), dt=dt, extra=ex_p)
        sm, _ = shell_update_law48(mat, s_old.copy(), deps_arr - ej, epsp=ep_old.copy(), dt=dt, extra=ex_m)

        if sp.ndim == 1:
            sp = sp.reshape(1, -1)
            sm = sm.reshape(1, -1)

        C_num[:, :, j] = (sp[:, :3] - sm[:, :3]) / (2.0 * h)

    return C_num[0] if is_1d else C_num


def _rel_err(d_alg: np.ndarray, d_num: np.ndarray) -> float:
    """Compute relative Frobenius norm error between algorithmic and numerical tangents."""
    norm_num = float(np.linalg.norm(d_num))
    if norm_num < 1e-12:
        return float(np.max(np.abs(d_alg - d_num)))
    return float(np.linalg.norm(d_alg - d_num) / norm_num)


# =============================================================================
# 1. Pure Elastic Regime Verification
# =============================================================================

def test_solid_elastic_pure_hookean_exact():
    """Verify solid consistent tangent matches exact 3D isotropic elasticity tensor

    and central finite difference perturbations to < 1e-8.
    """
    E = 210000.0
    nu = 0.3
    mat = Law48Params(E=E, nu=nu, ca=1.0, sigy0=350.0, cb=0.0)

    # Theoretical 3D isotropic elasticity matrix
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    c11 = K + 4.0 * G / 3.0
    c12 = K - 2.0 * G / 3.0

    expected = np.zeros((6, 6), dtype=float)
    expected[0:3, 0:3] = c12
    np.fill_diagonal(expected[0:3, 0:3], c11)
    expected[3, 3] = G
    expected[4, 4] = G
    expected[5, 5] = G

    # 1. Ground state with no arguments
    D_ground = tangent_law48_solid(mat)
    np.testing.assert_allclose(D_ground, expected, rtol=1e-12, atol=1e-12)

    # 2. Major symmetry and positive definiteness
    np.testing.assert_allclose(D_ground, D_ground.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(D_ground)
    assert np.all(eigvals > 0.0), f"Elastic eigenvalues must be strictly positive: {eigvals}"

    # 3. Small strain increment below yield (pure elastic)
    deps_el = np.array([0.0001, -0.00005, 0.00008, 0.00003, -0.00002, 0.00004], dtype=float)
    for h in (1e-7, 1e-6):
        D_alg = tangent_law48_solid(mat, deps=deps_el, h=h)
        D_num = _compute_solid_numerical_tangent(mat, deps_el, h=h)

        np.testing.assert_allclose(D_alg, expected, rtol=1e-10, atol=1e-10)
        err = _rel_err(D_alg, D_num)
        assert err < 1e-8, f"Elastic solid FD error {err:.3e} exceeds 1e-8 for h={h}"


def test_shell_elastic_pure_hookean_exact():
    """Verify shell membrane tangent matches exact plane-stress elasticity matrix

    and central finite difference perturbations to < 1e-8.
    """
    E = 200000.0
    nu = 0.28
    mat = Law48Params(E=E, nu=nu, ca=1.0, sigy0=300.0, cb=0.0)

    # Theoretical plane-stress membrane matrix
    c = E / (1.0 - nu * nu)
    g = E / (2.0 * (1.0 + nu))
    expected = np.array([
        [c, nu * c, 0.0],
        [nu * c, c, 0.0],
        [0.0, 0.0, g],
    ], dtype=float)

    # 1. shell_membrane_tangent helper
    C_mem = shell_membrane_tangent(mat)
    np.testing.assert_allclose(C_mem, expected, rtol=1e-12, atol=1e-12)

    # 2. Ground state tangent_law48_shell
    C_ground = tangent_law48_shell(mat)
    np.testing.assert_allclose(C_ground, expected, rtol=1e-12, atol=1e-12)

    # 3. Major symmetry and positive definiteness
    np.testing.assert_allclose(C_ground, C_ground.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(C_ground)
    assert np.all(eigvals > 0.0), f"Elastic shell eigenvalues must be strictly positive: {eigvals}"

    # 4. Small strain increment below yield
    deps_el = np.array([0.00015, -0.00008, 0.00005], dtype=float)
    for h in (1e-7, 1e-6):
        C_alg = tangent_law48_shell(mat, deps=deps_el, h=h)
        C_num = _compute_shell_numerical_tangent(mat, deps_el, h=h)

        np.testing.assert_allclose(C_alg, expected, rtol=1e-10, atol=1e-10)
        err = _rel_err(C_alg, C_num)
        assert err < 1e-8, f"Elastic shell FD error {err:.3e} exceeds 1e-8 for h={h}"


# =============================================================================
# 2. Static Hardening Plasticity (P_A)
# =============================================================================

@pytest.mark.parametrize("cn", [0.45, 1.0001, 1.25])
def test_solid_yielding_static_hardening_fd(cn: float):
    """Verify solid consistent tangent in static work hardening regime P_A = ca*sigy0 + cb*epsp^cn

    against numerical central finite differences for h in [1e-7, 1e-6] with relative error < 1e-4.
    """
    mat = Law48Params(
        E=200000.0,
        nu=0.3,
        ca=1.0,
        sigy0=300.0,
        cb=450.0,
        cn=cn,
        eps_max=0.5,
    )
    deps_pl = np.array([0.005, -0.0015, -0.0015, 0.002, -0.001, 0.0008], dtype=float)
    sig0 = np.zeros(6, dtype=float)
    epsp0 = 0.01

    # Run solid_update to obtain converged plastic state
    s_conv, ep_conv, _ = solid_update_law48(mat, sig0.copy(), deps_pl.copy(), epsp=epsp0)
    dep = float(ep_conv - epsp0)
    assert dep > 0.0, f"Increment must cause plastic flow, got dep={dep}"

    for h in (1e-7, 1e-6):
        # 1. Test analytical return-mapping tangent
        D_ana = tangent_law48_solid(mat, sig=s_conv, epsp=epsp0, epsp_incr=dep)
        D_num = _compute_solid_numerical_tangent(mat, deps_pl, sig=sig0, epsp=epsp0, h=h)

        err_ana = _rel_err(D_ana, D_num)
        assert err_ana < 1e-4, f"Analytical solid tangent mismatch for cn={cn}, h={h}: rel_err={err_ana:.3e}"

        # 2. Test directional tangent called with deps
        D_dir = tangent_law48_solid(mat, sig=sig0, epsp=epsp0, deps=deps_pl, h=h)
        err_dir = _rel_err(D_dir, D_num)
        assert err_dir < 1e-4, f"Directional solid tangent mismatch for cn={cn}, h={h}: rel_err={err_dir:.3e}"

    # Verify plastic tangent has reduced stiffness in tensile loading direction
    C_el = tangent_law48_solid(mat)
    assert D_ana[0, 0] < C_el[0, 0], f"Plastic tangent D_00 ({D_ana[0, 0]}) must be lower than elastic ({C_el[0, 0]})"


@pytest.mark.parametrize("cn", [0.4, 1.0001, 1.2])
def test_shell_yielding_static_hardening_fd(cn: float):
    """Verify shell membrane plane-stress consistent tangent in static work hardening regime

    against numerical central finite differences for h in [1e-7, 1e-6] with relative error < 1e-4.
    """
    mat = Law48Params(
        E=210000.0,
        nu=0.3,
        ca=1.0,
        sigy0=280.0,
        cb=400.0,
        cn=cn,
        eps_max=0.5,
    )
    deps_pl = np.array([0.004, -0.001, 0.0025], dtype=float)
    sig0 = np.zeros(3, dtype=float)
    epsp0 = 0.015

    s_conv, ep_conv = shell_update_law48(mat, sig0.copy(), deps_pl.copy(), epsp=epsp0)
    dep = float(ep_conv - epsp0)
    assert dep > 0.0, f"Increment must cause plastic flow, got dep={dep}"

    for h in (1e-7, 1e-6):
        # 1. Test analytical plane-stress return tangent
        C_ana = tangent_law48_shell(mat, sig=s_conv, epsp=epsp0, epsp_incr=dep)
        C_num = _compute_shell_numerical_tangent(mat, deps_pl, sig=sig0, epsp=epsp0, h=h)

        err_ana = _rel_err(C_ana, C_num)
        assert err_ana < 1e-4, f"Analytical shell tangent mismatch for cn={cn}, h={h}: rel_err={err_ana:.3e}"

        # 2. Test directional tangent called with deps
        C_dir = tangent_law48_shell(mat, sig=sig0, epsp=epsp0, deps=deps_pl, h=h)
        err_dir = _rel_err(C_dir, C_num)
        assert err_dir < 1e-4, f"Directional shell tangent mismatch for cn={cn}, h={h}: rel_err={err_dir:.3e}"

    # Reduced stiffness verification
    C_el = shell_membrane_tangent(mat)
    assert C_ana[0, 0] < C_el[0, 0], f"Plastic shell tangent C_00 ({C_ana[0, 0]}) must be lower than elastic ({C_el[0, 0]})"


# =============================================================================
# 3. Dynamic Yielding with Rate Sensitivity (P_B, P_C)
# =============================================================================

def test_solid_dynamic_rate_sensitivity_pb_pc_fd():
    """Verify solid consistent tangent under coupled strain rate sensitivity P_B

    and high-rate power law P_C with dt > 0 against central finite differences.
    """
    mat = Law48Params(
        E=200000.0,
        nu=0.3,
        ca=1.0,
        sigy0=300.0,
        cb=350.0,
        cn=0.7,
        cc=60.0,
        cd=20.0,
        cm=0.8,
        ce=12.0,
        ck=0.6,
        eps0=1.0,
        fcut=10000.0,  # filtered strain rate
    )
    deps_pl = np.array([0.005, -0.001, -0.001, 0.0015, 0.0, 0.0], dtype=float)
    sig0 = np.zeros(6, dtype=float)
    epsp0 = 0.02
    dt = 1e-4

    extra = {"epsd48": np.zeros(1, dtype=float)}

    for h in (1e-7, 1e-6):
        D_alg = tangent_law48_solid(mat, sig=sig0, deps=deps_pl, epsp=epsp0, dt=dt, extra=extra, h=h)
        D_num = _compute_solid_numerical_tangent(mat, deps_pl, sig=sig0, epsp=epsp0, dt=dt, extra=extra, h=h)

        err = _rel_err(D_alg, D_num)
        assert err < 1e-4, f"Solid dynamic rate sensitivity tangent mismatch for h={h}: rel_err={err:.3e}"


def test_shell_dynamic_rate_sensitivity_pb_pc_fd():
    """Verify shell consistent tangent under coupled strain rate sensitivity P_B

    and high-rate power law P_C with dt > 0 against central finite differences.
    """
    mat = Law48Params(
        E=210000.0,
        nu=0.3,
        ca=1.0,
        sigy0=320.0,
        cb=380.0,
        cn=0.75,
        cc=50.0,
        cd=15.0,
        cm=0.85,
        ce=10.0,
        ck=0.55,
        eps0=1.0,
        fcut=5000.0,
    )
    deps_pl = np.array([0.004, -0.0012, 0.002], dtype=float)
    sig0 = np.zeros(3, dtype=float)
    epsp0 = 0.015
    dt = 2e-4

    extra = {"epsd48": np.zeros(1, dtype=float)}

    for h in (1e-7, 1e-6):
        C_alg = tangent_law48_shell(mat, sig=sig0, deps=deps_pl, epsp=epsp0, dt=dt, extra=extra, h=h)
        C_num = _compute_shell_numerical_tangent(mat, deps_pl, sig=sig0, epsp=epsp0, dt=dt, extra=extra, h=h)

        err = _rel_err(C_alg, C_num)
        assert err < 1e-4, f"Shell dynamic rate sensitivity tangent mismatch for h={h}: rel_err={err:.3e}"


# =============================================================================
# 4. Stress Saturation Regime (YLD = S_max + P_C)
# =============================================================================

def test_solid_stress_saturation_fd():
    """Verify solid consistent tangent in stress saturation regime YLD = S_max + P_C

    where the plastic hardening slope is zeroed (H = 0).
    """
    mat = Law48Params(
        E=200000.0,
        nu=0.3,
        ca=1.0,
        sigy0=300.0,
        cb=600.0,
        cn=1.0001,
        sig_max=360.0,  # Saturates well before uncapped yield (300 + 600*0.05 = 330, at 0.15 = 390 -> capped at 360)
    )
    deps_pl = np.array([0.008, -0.002, -0.002, 0.001, 0.0, 0.0], dtype=float)
    sig0 = np.zeros(6, dtype=float)
    epsp0 = 0.12

    # Check yield stress is capped
    yld, h_slope, _, _ = eval_yield_and_hardening(mat, epsp=epsp0, eps_dot=0.0)
    assert math.isclose(float(yld), 360.0, rel_tol=1e-6), f"Expected capped yield 360.0, got {yld}"
    assert float(h_slope) == 0.0, f"Hardening slope must be zero when capped, got {h_slope}"

    s_conv, ep_conv, _ = solid_update_law48(mat, sig0.copy(), deps_pl.copy(), epsp=epsp0)
    dep = float(ep_conv - epsp0)

    for h in (1e-7, 1e-6):
        # 1. Analytical Simo & Hughes path with H = 0
        D_ana = tangent_law48_solid(mat, sig=s_conv, epsp=epsp0, epsp_incr=dep)
        D_num = _compute_solid_numerical_tangent(mat, deps_pl, sig=sig0, epsp=epsp0, h=h)

        err = _rel_err(D_ana, D_num)
        assert err < 1e-4, f"Solid saturation analytical tangent error {err:.3e} exceeds 1e-4 for h={h}"

        # 2. Directional path with deps
        D_dir = tangent_law48_solid(mat, sig=sig0, epsp=epsp0, deps=deps_pl, h=h)
        err_dir = _rel_err(D_dir, D_num)
        assert err_dir < 1e-4, f"Solid saturation directional tangent error {err_dir:.3e} exceeds 1e-4 for h={h}"


def test_shell_stress_saturation_fd():
    """Verify shell consistent tangent in stress saturation regime YLD = S_max + P_C."""
    mat = Law48Params(
        E=210000.0,
        nu=0.3,
        ca=1.0,
        sigy0=300.0,
        cb=500.0,
        cn=1.0001,
        sig_max=350.0,
    )
    deps_pl = np.array([0.007, -0.002, 0.001], dtype=float)
    sig0 = np.zeros(3, dtype=float)
    epsp0 = 0.15

    s_conv, ep_conv = shell_update_law48(mat, sig0.copy(), deps_pl.copy(), epsp=epsp0)
    dep = float(ep_conv - epsp0)

    for h in (1e-7, 1e-6):
        C_ana = tangent_law48_shell(mat, sig=s_conv, epsp=epsp0, epsp_incr=dep)
        C_num = _compute_shell_numerical_tangent(mat, deps_pl, sig=sig0, epsp=epsp0, h=h)

        err = _rel_err(C_ana, C_num)
        assert err < 1e-4, f"Shell saturation analytical tangent error {err:.3e} exceeds 1e-4 for h={h}"

        C_dir = tangent_law48_shell(mat, sig=sig0, epsp=epsp0, deps=deps_pl, h=h)
        err_dir = _rel_err(C_dir, C_num)
        assert err_dir < 1e-4, f"Shell saturation directional tangent error {err_dir:.3e} exceeds 1e-4 for h={h}"


# =============================================================================
# 5. Tensile Damage Regime (FAIL in (0, 1))
# =============================================================================

def test_solid_tensile_damage_regime_fd():
    """Verify solid consistent tangent in tensile damage regime where FAIL in (0, 1)."""
    mat = Law48Params(
        E=200000.0,
        nu=0.3,
        ca=1.0,
        sigy0=300.0,
        cb=400.0,
        cn=1.0001,
        eps_t1=0.003,
        eps_t2=0.012,
    )
    # Total strain will reach ~0.006, giving FAIL ~ (0.012 - 0.006)/(0.012 - 0.003) ~ 0.67
    deps_pl = np.array([0.006, -0.0015, -0.0015, 0.001, 0.0, 0.0], dtype=float)
    sig0 = np.zeros(6, dtype=float)
    epsp0 = 0.01

    extra = {"eps48": np.zeros(6, dtype=float)}

    for h in (1e-7, 1e-6):
        D_alg = tangent_law48_solid(mat, sig=sig0, deps=deps_pl, epsp=epsp0, extra=extra, h=h)
        D_num = _compute_solid_numerical_tangent(mat, deps_pl, sig=sig0, epsp=epsp0, extra=extra, h=h)

        err = _rel_err(D_alg, D_num)
        assert err < 1e-4, f"Solid tensile damage tangent error {err:.3e} exceeds 1e-4 for h={h}"


def test_shell_tensile_damage_regime_fd():
    """Verify shell consistent tangent in tensile damage regime where FAIL in (0, 1)."""
    mat = Law48Params(
        E=210000.0,
        nu=0.3,
        ca=1.0,
        sigy0=300.0,
        cb=450.0,
        cn=1.0001,
        eps_t1=0.0025,
        eps_t2=0.010,
    )
    deps_pl = np.array([0.0055, -0.001, 0.0015], dtype=float)
    sig0 = np.zeros(3, dtype=float)
    epsp0 = 0.01

    extra = {"eps48": np.zeros(3, dtype=float)}

    for h in (1e-7, 1e-6):
        C_alg = tangent_law48_shell(mat, sig=sig0, deps=deps_pl, epsp=epsp0, extra=extra, h=h)
        C_num = _compute_shell_numerical_tangent(mat, deps_pl, sig=sig0, epsp=epsp0, extra=extra, h=h)

        err = _rel_err(C_alg, C_num)
        assert err < 1e-4, f"Shell tensile damage tangent error {err:.3e} exceeds 1e-4 for h={h}"


# =============================================================================
# 6. Fully Eroded Regime (Element Deletion)
# =============================================================================

def test_solid_fully_eroded_regime():
    """Verify that deleted solid elements return strictly zero tangent tensor D = 0

    across all deletion triggers: eps_p > eps_max, eps_t >= eps_r2, and off = 0.
    """
    mat = Law48Params(
        E=200000.0,
        nu=0.3,
        ca=1.0,
        sigy0=300.0,
        eps_max=0.2,
        eps_t1=0.01,
        eps_t2=0.02,
    )
    deps_norm = np.array([0.004, -0.001, -0.001, 0.001, 0.0, 0.0], dtype=float)

    # 1. Deletion via eps_p > eps_max
    D_epmax = tangent_law48_solid(mat, deps=deps_norm, epsp=0.25)
    np.testing.assert_allclose(D_epmax, 0.0, atol=1e-15)
    D_num1 = _compute_solid_numerical_tangent(mat, deps_norm, epsp=0.25)
    np.testing.assert_allclose(D_num1, 0.0, atol=1e-15)

    # 2. Deletion via eps_t >= eps_t2
    extra_rup = {"eps48": np.array([0.025, 0.0, 0.0, 0.0, 0.0, 0.0])}
    D_rup = tangent_law48_solid(mat, deps=deps_norm, epsp=0.01, extra=extra_rup)
    np.testing.assert_allclose(D_rup, 0.0, atol=1e-15)
    D_num2 = _compute_solid_numerical_tangent(mat, deps_norm, epsp=0.01, extra=extra_rup)
    np.testing.assert_allclose(D_num2, 0.0, atol=1e-15)

    # 3. Deletion via off = 0.0
    extra_off = {"off": 0.0}
    D_off = tangent_law48_solid(mat, deps=deps_norm, epsp=0.01, extra=extra_off)
    np.testing.assert_allclose(D_off, 0.0, atol=1e-15)

    # 4. Analytical path with deletion
    D_ana_del = tangent_law48_solid(mat, sig=np.ones(6), epsp=0.25, epsp_incr=0.001)
    np.testing.assert_allclose(D_ana_del, 0.0, atol=1e-15)


def test_shell_fully_eroded_regime():
    """Verify that deleted shell elements return strictly zero tangent tensor C = 0."""
    mat = Law48Params(
        E=210000.0,
        nu=0.3,
        ca=1.0,
        sigy0=300.0,
        eps_max=0.25,
        eps_t1=0.01,
        eps_t2=0.02,
    )
    deps_norm = np.array([0.004, -0.001, 0.002], dtype=float)

    # 1. eps_p > eps_max
    C_epmax = tangent_law48_shell(mat, deps=deps_norm, epsp=0.30)
    np.testing.assert_allclose(C_epmax, 0.0, atol=1e-15)
    C_num1 = _compute_shell_numerical_tangent(mat, deps_norm, epsp=0.30)
    np.testing.assert_allclose(C_num1, 0.0, atol=1e-15)

    # 2. eps_t >= eps_t2
    extra_rup = {"eps48": np.array([0.025, 0.0, 0.0])}
    C_rup = tangent_law48_shell(mat, deps=deps_norm, epsp=0.01, extra=extra_rup)
    np.testing.assert_allclose(C_rup, 0.0, atol=1e-15)

    # 3. off = 0.0 or layfail = 0.0
    extra_off = {"layfail": 0.0}
    C_off = tangent_law48_shell(mat, deps=deps_norm, epsp=0.01, extra=extra_off)
    np.testing.assert_allclose(C_off, 0.0, atol=1e-15)

    # 4. Analytical path
    C_ana_del = tangent_law48_shell(mat, sig=np.ones(3), epsp=0.35, epsp_incr=0.001)
    np.testing.assert_allclose(C_ana_del, 0.0, atol=1e-15)


# =============================================================================
# 7. Multiaxial Loading States Verification
# =============================================================================

@pytest.mark.parametrize("load_name, deps_vec", [
    ("uniaxial_tension", np.array([0.006, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ("equibiaxial_tension", np.array([0.004, 0.004, 0.0, 0.0, 0.0, 0.0])),
    ("pure_shear", np.array([0.0, 0.0, 0.0, 0.008, 0.0, 0.0])),
    ("triaxial_compression", np.array([-0.005, -0.005, -0.005, 0.0, 0.0, 0.0])),
    ("mixed_shear_normal", np.array([0.004, -0.002, 0.001, 0.003, -0.002, 0.0015])),
])
def test_solid_multiaxial_loading_fd(load_name: str, deps_vec: np.ndarray):
    """Verify solid consistent tangent across multiaxial stress/strain loading paths."""
    mat = Law48Params(
        E=200000.0,
        nu=0.3,
        ca=1.0,
        sigy0=300.0,
        cb=450.0,
        cn=0.8,
        eps_max=0.5,
    )
    sig0 = np.zeros(6, dtype=float)
    epsp0 = 0.01

    for h in (1e-7, 1e-6):
        D_alg = tangent_law48_solid(mat, sig=sig0, deps=deps_vec, epsp=epsp0, h=h)
        D_num = _compute_solid_numerical_tangent(mat, deps_vec, sig=sig0, epsp=epsp0, h=h)

        err = _rel_err(D_alg, D_num)
        assert err < 1e-4, f"Solid {load_name} tangent mismatch for h={h}: rel_err={err:.3e}"


@pytest.mark.parametrize("load_name, deps_vec", [
    ("uniaxial_tension_x", np.array([0.005, 0.0, 0.0])),
    ("uniaxial_tension_y", np.array([0.0, 0.005, 0.0])),
    ("equibiaxial_tension", np.array([0.0035, 0.0035, 0.0])),
    ("pure_shear", np.array([0.0, 0.0, 0.007])),
    ("mixed_shear_normal", np.array([0.0035, -0.002, 0.004])),
])
def test_shell_multiaxial_loading_fd(load_name: str, deps_vec: np.ndarray):
    """Verify shell plane-stress consistent tangent across multiaxial in-plane paths."""
    mat = Law48Params(
        E=210000.0,
        nu=0.3,
        ca=1.0,
        sigy0=290.0,
        cb=400.0,
        cn=0.75,
        eps_max=0.5,
    )
    sig0 = np.zeros(3, dtype=float)
    epsp0 = 0.01

    for h in (1e-7, 1e-6):
        C_alg = tangent_law48_shell(mat, sig=sig0, deps=deps_vec, epsp=epsp0, h=h)
        C_num = _compute_shell_numerical_tangent(mat, deps_vec, sig=sig0, epsp=epsp0, h=h)

        err = _rel_err(C_alg, C_num)
        assert err < 1e-4, f"Shell {load_name} tangent mismatch for h={h}: rel_err={err:.3e}"


# =============================================================================
# 8. Vectorization Verification (N >= 16 elements simultaneously)
# =============================================================================

def test_solid_vectorization_multi_element_batch():
    """Verify solid consistent tangent vectorization across N = 20 elements (>= 16)

    under heterogeneous elastic, yielding, rate-sensitive, damaged, and eroded states.
    """
    N = 20
    mat = Law48Params(
        E=200000.0,
        nu=0.3,
        ca=1.0,
        sigy0=300.0,
        cb=500.0,
        cn=0.8,
        cc=50.0,
        cd=15.0,
        cm=0.8,
        ce=10.0,
        ck=0.6,
        eps0=1.0,
        sig_max=600.0,
        eps_max=0.3,
        eps_t1=0.01,
        eps_t2=0.025,
    )

    np.random.seed(48)
    deps_batch = np.zeros((N, 6), dtype=float)
    epsp_batch = np.zeros(N, dtype=float)
    off_batch = np.ones(N, dtype=float)
    dt = 1e-4

    for i in range(N):
        if i < 4:
            # Pure elastic (small strains)
            deps_batch[i] = np.array([0.0001, -0.00004, 0.00002, 0.00003, -0.00001, 0.00002]) * (i + 1)
            epsp_batch[i] = 0.0
        elif i < 8:
            # Static plastic flow
            deps_batch[i] = np.array([0.004, -0.001, -0.001, 0.002, 0.0, 0.0]) * (1.0 + 0.1 * i)
            epsp_batch[i] = 0.01 * (i - 3)
        elif i < 12:
            # Dynamic rate-dependent flow
            deps_batch[i] = np.array([0.005, -0.0015, -0.001, 0.0025, 0.001, -0.001])
            epsp_batch[i] = 0.02
        elif i < 15:
            # Saturation / high strain
            deps_batch[i] = np.array([0.012, -0.003, -0.003, 0.004, 0.0, 0.0])
            epsp_batch[i] = 0.20
        elif i < 18:
            # Tensile damage regime
            deps_batch[i] = np.array([0.015, -0.002, -0.002, 0.001, 0.0, 0.0])
            epsp_batch[i] = 0.05
        else:
            # Fully eroded (deleted element)
            deps_batch[i] = np.array([0.005, -0.001, -0.001, 0.0, 0.0, 0.0])
            epsp_batch[i] = 0.01
            off_batch[i] = 0.0

    extra_batch = {
        "eps48": deps_batch.copy(),
        "epsd48": np.zeros(N, dtype=float),
        "off": off_batch,
    }

    # 1. Compute batched tangent
    D_batch = tangent_law48_solid(mat, deps=deps_batch, epsp=epsp_batch, dt=dt, extra=extra_batch)
    assert D_batch.shape == (N, 6, 6), f"Expected shape ({N}, 6, 6), got {D_batch.shape}"

    # 2. Verify element-by-element equivalence and numerical FD match
    for i in range(N):
        extra_i = {
            "eps48": extra_batch["eps48"][i].copy(),
            "epsd48": np.zeros(1, dtype=float),
            "off": off_batch[i],
        }
        D_single = tangent_law48_solid(mat, deps=deps_batch[i], epsp=epsp_batch[i], dt=dt, extra=extra_i)
        np.testing.assert_allclose(D_batch[i], D_single, rtol=1e-10, atol=1e-10)

        D_num = _compute_solid_numerical_tangent(mat, deps_batch[i], epsp=epsp_batch[i], dt=dt, extra=extra_i, h=1e-7)
        err = _rel_err(D_batch[i], D_num)
        if off_batch[i] <= 0.0:
            np.testing.assert_allclose(D_batch[i], 0.0, atol=1e-15)
        else:
            assert err < 1e-4, f"Element {i} in N={N} solid batch failed: rel_err={err:.3e}"


def test_shell_vectorization_multi_element_batch():
    """Verify shell consistent tangent vectorization across N = 20 elements (>= 16)."""
    N = 20
    mat = Law48Params(
        E=210000.0,
        nu=0.3,
        ca=1.0,
        sigy0=310.0,
        cb=450.0,
        cn=0.8,
        cc=40.0,
        cd=10.0,
        cm=0.8,
        ce=8.0,
        ck=0.6,
        eps0=1.0,
        eps_max=0.3,
        eps_t1=0.01,
        eps_t2=0.025,
    )

    deps_batch = np.zeros((N, 3), dtype=float)
    epsp_batch = np.zeros(N, dtype=float)
    off_batch = np.ones(N, dtype=float)
    dt = 1e-4

    for i in range(N):
        if i < 4:
            deps_batch[i] = np.array([0.0001, -0.00004, 0.00002]) * (i + 1)
            epsp_batch[i] = 0.0
        elif i < 8:
            deps_batch[i] = np.array([0.0045, -0.0015, 0.002]) * (1.0 + 0.1 * i)
            epsp_batch[i] = 0.015
        elif i < 12:
            deps_batch[i] = np.array([0.005, -0.001, 0.003])
            epsp_batch[i] = 0.02
        elif i < 15:
            deps_batch[i] = np.array([0.01, -0.002, 0.005])
            epsp_batch[i] = 0.15
        elif i < 18:
            deps_batch[i] = np.array([0.016, -0.003, 0.002])
            epsp_batch[i] = 0.05
        else:
            deps_batch[i] = np.array([0.004, -0.001, 0.001])
            epsp_batch[i] = 0.01
            off_batch[i] = 0.0

    extra_batch = {
        "eps48": deps_batch.copy(),
        "epsd48": np.zeros(N, dtype=float),
        "off": off_batch,
    }

    C_batch = tangent_law48_shell(mat, deps=deps_batch, epsp=epsp_batch, dt=dt, extra=extra_batch)
    assert C_batch.shape == (N, 3, 3)

    for i in range(N):
        extra_i = {
            "eps48": extra_batch["eps48"][i].copy(),
            "epsd48": np.zeros(1, dtype=float),
            "off": off_batch[i],
        }
        C_single = tangent_law48_shell(mat, deps=deps_batch[i], epsp=epsp_batch[i], dt=dt, extra=extra_i)
        np.testing.assert_allclose(C_batch[i], C_single, rtol=1e-10, atol=1e-10)

        C_num = _compute_shell_numerical_tangent(mat, deps_batch[i], epsp=epsp_batch[i], dt=dt, extra=extra_i, h=1e-7)
        err = _rel_err(C_batch[i], C_num)
        if off_batch[i] <= 0.0:
            np.testing.assert_allclose(C_batch[i], 0.0, atol=1e-15)
        else:
            assert err < 1e-4, f"Element {i} in N={N} shell batch failed: rel_err={err:.3e}"


# =============================================================================
# 9. Calling Conventions, Dispatchers, and Major Symmetry
# =============================================================================

def test_tangent_calling_conventions_and_shapes():
    """Verify flexible calling conventions, output shapes, package dispatchers,

    and major symmetry option symmetric=True.
    """
    mat_params = Law48Params(E=200000.0, nu=0.3, ca=1.0, sigy0=300.0)

    # 1. 1D inputs produce 2D matrices (6, 6) and (3, 3)
    sig_1d_sol = np.zeros(6, dtype=float)
    sig_1d_sh = np.zeros(3, dtype=float)
    deps_1d_sol = np.array([0.002, -0.0005, -0.0005, 0.001, 0.0, 0.0])
    deps_1d_sh = np.array([0.002, -0.0005, 0.001])

    D_1d = consistent_solid_tangent(mat_params, deps=deps_1d_sol)
    assert D_1d.shape == (6, 6)

    C_1d = consistent_shell_tangent(mat_params, deps=deps_1d_sh)
    assert C_1d.shape == (3, 3)

    # 2. 2D inputs produce 3D tensors (N, 6, 6) and (N, 3, 3)
    D_2d = consistent_solid_tangent(mat_params, deps=deps_1d_sol[None, :])
    assert D_2d.shape == (1, 6, 6)

    C_2d = consistent_shell_tangent(mat_params, deps=deps_1d_sh[None, :])
    assert C_2d.shape == (1, 3, 3)

    # 3. Empty input produces empty output
    D_empty = consistent_solid_tangent(mat_params, deps=np.empty((0, 6)))
    assert D_empty.shape == (0, 6, 6)

    C_empty = consistent_shell_tangent(mat_params, deps=np.empty((0, 3)))
    assert C_empty.shape == (0, 3, 3)

    # 4. Material entity compatibility
    class MockRecord:
        id = 48
        density = 7.8e-9
        title = "Zhao Tangent Test"
        params = {"MAT_E": 200000.0, "MAT_NU": 0.3, "MAT_SIGY": 300.0}

    mat_entity = build_law48(MockRecord())

    D_ent_1d = materials.solid_tangent(mat_entity, sig=sig_1d_sol, epsp=0.0)
    assert D_ent_1d.shape == (6, 6)

    D_ent_2d = materials.solid_tangent(mat_entity, sig=sig_1d_sol[None, :], epsp=np.zeros(1))
    assert D_ent_2d.shape == (1, 6, 6)

    C_ent_1d = materials.shell_layer_tangent(mat_entity, sig=sig_1d_sh, epsp=0.0)
    assert C_ent_1d.shape == (3, 3)

    C_ent_2d = materials.shell_layer_tangent(mat_entity, sig=sig_1d_sh[None, :], epsp=np.zeros(1))
    assert C_ent_2d.shape == (1, 3, 3)

    # 5. Major symmetry option symmetric=True
    D_sym = consistent_solid_tangent(mat_params, deps=deps_1d_sol, symmetric=True)
    np.testing.assert_allclose(D_sym, D_sym.T, atol=1e-12)

    C_sym = consistent_shell_tangent(mat_params, deps=deps_1d_sh, symmetric=True)
    np.testing.assert_allclose(C_sym, C_sym.T, atol=1e-12)
