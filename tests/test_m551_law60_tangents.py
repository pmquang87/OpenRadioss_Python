"""
Exhaustive Algorithmic Consistent Tangent Stiffness Verification Suite for LAW60.
(/MAT/LAW60, /MAT/PLAS_T3, /MAT/FABRIC).

Audits consistent_solid_tangent (NEL, 6, 6) and consistent_shell_tangent (NEL, 3, 3)
from pyradioss/materials/law60_plast3.py across:
1. Numerical central finite difference perturbation test:
   - Step sizes h in [1e-6, 1e-7]
   - D_num[:, :, j] = (sigma(eps + h * e_j) - sigma(eps - h * e_j)) / (2 * h)
   - Relative error ||D_alg - D_num|| / ||D_num|| < 1e-4
2. Solid tangent regimes:
   - Pure elastic regime (small strain eps < eps_yield)
   - Plastic loading regime (uniaxial tension, pure shear, multiaxial triaxial states)
   - Pressure-dependent states with ipfun > 0 and pre-existing hydrostatic pressure
   - Modulus degraded states with exponential ce > 0 and tabulated ifunce > 0
   - Vectorized multi-element batches (NEL = 1, 5, 20)
3. Shell plane-stress tangent regimes:
   - In-plane components [xx, yy, xy] with plane-stress condition sigma_zz = 0
   - In-plane elastic and plastic states: tension, compression, shear, multiaxial
   - Vectorized multi-element batches (NEL = 1, 5, 20)
4. Symmetry & positive-definiteness:
   - Exact major symmetry C = C.T in ground state and elastic regime
   - Positive-definiteness: strictly positive eigenvalues
   - Symmetric projection option symmetric=True
5. Element deletion & robust calling conventions:
   - Zero tangent returned for deleted elements (off <= 0.0)
   - Disambiguation across all positional and keyword calling patterns
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pytest

from pyradioss.materials.law60_plast3 import (
    Law60Params,
    build_law60,
    consistent_shell_tangent,
    consistent_solid_tangent,
    law60_consistent_shell_tangent,
    law60_consistent_solid_tangent,
    law60_shell_tangent,
    law60_solid_tangent,
    shell_membrane_tangent,
    shell_tangent,
    shell_update,
    solid_tangent,
    solid_update,
)


# =============================================================================
# Helper Utilities: Independent Central Finite Difference Numerical Tangents
# =============================================================================

def _copy_extra(extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Deep copy arrays and dicts inside state extra dictionary."""
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


def _make_law60_solid(
    e0: float = 200000.0,
    nu: float = 0.3,
    rho0: float = 7.8e-9,
    sig_y: float = 300.0,
    h_slope: float = 1000.0,
    ce: float = 0.0,
    einf: float = 0.0,
    ifunce: int = 0,
    ipfun: int = 0,
    pscale: float = 1.0,
    fisokin: float = 0.0,
    **kwargs: Any,
) -> Law60Params:
    """Build a validated Law60Params instance for 3D solid tangent audits."""
    curve_x = [np.array([0.0, 0.05, 0.2], dtype=float)]
    curve_y = [np.array([sig_y, sig_y + 0.05 * h_slope, sig_y + 0.2 * h_slope], dtype=float)]
    curve_s = [np.array([h_slope, h_slope], dtype=float)]

    p_cx, p_cy, p_cs = None, None, None
    if ipfun > 0:
        p_cx = np.array([0.0, 500.0, 1000.0], dtype=float)
        p_cy = np.array([1.0, 1.25, 1.5], dtype=float)
        p_cs = np.array([0.0005, 0.0005], dtype=float)

    e_cx, e_cy, e_cs = None, None, None
    if ifunce > 0:
        e_cx = np.array([0.0, 0.05, 0.1], dtype=float)
        e_cy = np.array([1.0, 0.8, 0.6], dtype=float)
        e_cs = np.array([-4.0, -4.0], dtype=float)

    return build_law60(
        e0=e0,
        nu=nu,
        rho0=rho0,
        ce=ce,
        einf=einf if einf > 0.0 else e0,
        ifunce=ifunce,
        ipfun=ipfun,
        pscale=pscale,
        fisokin=fisokin,
        curve_x=curve_x,
        curve_y=curve_y,
        curve_s=curve_s,
        rates=np.array([0.0], dtype=float),
        p_curve_x=p_cx,
        p_curve_y=p_cy,
        p_curve_s=p_cs,
        e_curve_x=e_cx,
        e_curve_y=e_cy,
        e_curve_s=e_cs,
        **kwargs,
    )


def _make_law60_shell(
    e0: float = 210000.0,
    nu: float = 0.3,
    rho0: float = 7.8e-9,
    sig_y: float = 300.0,
    h_slope: float = 1500.0,
    ce: float = 0.0,
    einf: float = 0.0,
    ifunce: int = 0,
    fisokin: float = 0.0,
    **kwargs: Any,
) -> Law60Params:
    """Build a validated Law60Params instance for 2D shell plane-stress tangent audits."""
    curve_x = [np.array([0.0, 0.05, 0.2], dtype=float)]
    curve_y = [np.array([sig_y, sig_y + 0.05 * h_slope, sig_y + 0.2 * h_slope], dtype=float)]
    curve_s = [np.array([h_slope, h_slope], dtype=float)]

    e_cx, e_cy, e_cs = None, None, None
    if ifunce > 0:
        e_cx = np.array([0.0, 0.05, 0.1], dtype=float)
        e_cy = np.array([1.0, 0.8, 0.6], dtype=float)
        e_cs = np.array([-4.0, -4.0], dtype=float)

    return build_law60(
        e0=e0,
        nu=nu,
        rho0=rho0,
        ce=ce,
        einf=einf if einf > 0.0 else e0,
        ifunce=ifunce,
        fisokin=fisokin,
        curve_x=curve_x,
        curve_y=curve_y,
        curve_s=curve_s,
        rates=np.array([0.0], dtype=float),
        e_curve_x=e_cx,
        e_curve_y=e_cy,
        e_curve_s=e_cs,
        **kwargs,
    )


def _compute_solid_numerical_tangent(
    mat: Law60Params,
    deps: np.ndarray,
    sig_old: Optional[np.ndarray] = None,
    epsp_old: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    h: float = 1e-7,
) -> np.ndarray:
    """Compute (NEL, 6, 6) numerical solid tangent via central finite difference:

    D_num[:, :, j] = [sigma(deps + h * e_j) - sigma(deps - h * e_j)] / (2 * h)
    """
    deps_arr = np.asarray(deps, dtype=float)
    is_1d = (deps_arr.ndim == 1)
    if is_1d:
        deps_arr = deps_arr.reshape(1, 6)
    nel = deps_arr.shape[0]

    s_old = np.asarray(sig_old, dtype=float) if sig_old is not None else np.zeros((nel, 6), dtype=float)
    if s_old.ndim == 1:
        s_old = s_old.reshape(1, 6)

    ep_old = np.asarray(epsp_old, dtype=float).flatten() if epsp_old is not None else np.zeros(nel, dtype=float)
    if len(ep_old) == 1 and nel > 1:
        ep_old = np.full(nel, ep_old[0], dtype=float)

    D_num = np.zeros((nel, 6, 6), dtype=float)

    for j in range(6):
        ej = np.zeros_like(deps_arr)
        ej[:, j] = h

        ex_p = _copy_extra(extra)
        ex_m = _copy_extra(extra)

        sp, _, _ = solid_update(mat, sig_old=s_old.copy(), deps=deps_arr + ej, epsp_old=ep_old.copy(), dt=dt, extra=ex_p)
        sm, _, _ = solid_update(mat, sig_old=s_old.copy(), deps=deps_arr - ej, epsp_old=ep_old.copy(), dt=dt, extra=ex_m)

        if sp.ndim == 1:
            sp = sp.reshape(1, 6)
            sm = sm.reshape(1, 6)

        D_num[:, :, j] = (sp - sm) / (2.0 * h)

    return D_num[0] if is_1d else D_num


def _compute_shell_numerical_tangent(
    mat: Law60Params,
    deps: np.ndarray,
    sig_old: Optional[np.ndarray] = None,
    epsp_old: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    h: float = 1e-7,
) -> np.ndarray:
    """Compute (NEL, 3, 3) numerical plane-stress tangent via central finite difference:

    D_num[:, :, j] = [sigma(deps + h * e_j) - sigma(deps - h * e_j)] / (2 * h)
    """
    deps_arr = np.asarray(deps, dtype=float)
    is_1d = (deps_arr.ndim == 1)
    if is_1d:
        deps_arr = deps_arr.reshape(1, 3)
    nel = deps_arr.shape[0]

    s_old = np.asarray(sig_old, dtype=float) if sig_old is not None else np.zeros((nel, 3), dtype=float)
    if s_old.ndim == 1:
        s_old = s_old.reshape(1, 3)

    ep_old = np.asarray(epsp_old, dtype=float).flatten() if epsp_old is not None else np.zeros(nel, dtype=float)
    if len(ep_old) == 1 and nel > 1:
        ep_old = np.full(nel, ep_old[0], dtype=float)

    D_num = np.zeros((nel, 3, 3), dtype=float)

    for j in range(3):
        ej = np.zeros_like(deps_arr)
        ej[:, j] = h

        ex_p = _copy_extra(extra)
        ex_m = _copy_extra(extra)

        sp, _, _ = shell_update(mat, sig_old=s_old.copy(), deps=deps_arr + ej, epsp_old=ep_old.copy(), dt=dt, extra=ex_p)
        sm, _, _ = shell_update(mat, sig_old=s_old.copy(), deps=deps_arr - ej, epsp_old=ep_old.copy(), dt=dt, extra=ex_m)

        if sp.ndim == 1:
            sp = sp.reshape(1, 3)
            sm = sm.reshape(1, 3)

        D_num[:, :, j] = (sp - sm) / (2.0 * h)

    return D_num[0] if is_1d else D_num


# =============================================================================
# 1. Pure Elastic Regime Verification
# =============================================================================

def test_solid_tangent_pure_elastic_exact():
    """Verify solid consistent tangent matches exact 3D isotropic elasticity matrix C_el

    and matches numerical central finite differences to < 1e-6.
    """
    e0 = 200000.0
    nu = 0.3
    mat = _make_law60_solid(e0=e0, nu=nu)

    k_bulk = e0 / (3.0 * (1.0 - 2.0 * nu))
    g_shear = e0 / (2.0 * (1.0 + nu))
    c11 = k_bulk + (4.0 / 3.0) * g_shear
    c12 = k_bulk - (2.0 / 3.0) * g_shear

    expected = np.zeros((6, 6), dtype=float)
    expected[0:3, 0:3] = c12
    np.fill_diagonal(expected[0:3, 0:3], c11)
    expected[3, 3] = g_shear
    expected[4, 4] = g_shear
    expected[5, 5] = g_shear

    # Ground state tangent with no arguments
    c_ground = consistent_solid_tangent(mat)
    np.testing.assert_allclose(c_ground, expected, rtol=1e-12, atol=1e-12)

    # Pure elastic small strain increment (sig_vm << sig_y)
    deps_el = np.array([0.0002, -0.0001, 0.00005, 0.0001, -0.00008, 0.00004], dtype=float)
    d_alg = consistent_solid_tangent(mat, deps=deps_el)
    np.testing.assert_allclose(d_alg, expected, rtol=1e-12, atol=1e-12)

    # Central FD check with h = 1e-6 and 1e-7
    for h in (1e-6, 1e-7):
        d_num = _compute_solid_numerical_tangent(mat, deps_el, h=h)
        rel_err = np.linalg.norm(d_alg - d_num) / np.linalg.norm(d_num)
        assert rel_err < 1e-6, f"Elastic solid FD relative error {rel_err:.3e} exceeds 1e-6 for h={h}"

    # Verify symmetry and positive definiteness
    np.testing.assert_allclose(d_alg, d_alg.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(d_alg)
    assert np.all(eigvals > 0.0), f"Elastic solid tangent eigenvalues must be positive, got {eigvals}"


def test_shell_tangent_pure_elastic_exact():
    """Verify shell consistent tangent matches exact plane-stress membrane elasticity C_el

    and matches numerical central finite differences to < 1e-6.
    """
    e0 = 210000.0
    nu = 0.3
    mat = _make_law60_shell(e0=e0, nu=nu)

    a1 = e0 / (1.0 - nu ** 2)
    a2 = nu * a1
    g = e0 / (2.0 * (1.0 + nu))

    expected = np.array([
        [a1, a2, 0.0],
        [a2, a1, 0.0],
        [0.0, 0.0, g],
    ], dtype=float)

    # Direct call to shell_membrane_tangent
    c_mem = shell_membrane_tangent(mat)
    np.testing.assert_allclose(c_mem, expected, rtol=1e-12, atol=1e-12)

    # Ground state shell tangent
    c_ground = consistent_shell_tangent(mat)
    np.testing.assert_allclose(c_ground, expected, rtol=1e-12, atol=1e-12)

    # Pure elastic small strain increment
    deps_el = np.array([0.0003, -0.0001, 0.0002], dtype=float)
    d_alg = consistent_shell_tangent(mat, deps=deps_el)
    np.testing.assert_allclose(d_alg, expected, rtol=1e-12, atol=1e-12)

    # Central FD check with h = 1e-6 and 1e-7
    for h in (1e-6, 1e-7):
        d_num = _compute_shell_numerical_tangent(mat, deps_el, h=h)
        rel_err = np.linalg.norm(d_alg - d_num) / np.linalg.norm(d_num)
        assert rel_err < 1e-6, f"Elastic shell FD relative error {rel_err:.3e} exceeds 1e-6 for h={h}"

    # Verify symmetry and positive definiteness
    np.testing.assert_allclose(d_alg, d_alg.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(d_alg)
    assert np.all(eigvals > 0.0), f"Elastic shell tangent eigenvalues must be positive, got {eigvals}"


# =============================================================================
# 2. Solid Plastic Loading Regime Verification
# =============================================================================

def test_solid_tangent_fd_plastic_uniaxial_tension():
    """Verify solid tangent against central finite differences in plastic uniaxial tension."""
    mat = _make_law60_solid(e0=200000.0, nu=0.3, sig_y=300.0, h_slope=1000.0)

    # Strains causing plastic yield along x, y, and z axes
    cases = [
        ("uniaxial_x", np.array([0.005, -0.001, -0.001, 0.0, 0.0, 0.0])),
        ("uniaxial_y", np.array([-0.001, 0.006, -0.001, 0.0, 0.0, 0.0])),
        ("uniaxial_z", np.array([-0.001, -0.001, 0.0055, 0.0, 0.0, 0.0])),
    ]

    for name, deps in cases:
        sig_conv, ep_conv, _ = solid_update(mat, deps=deps)
        assert ep_conv > 0.0, f"Case {name} did not yield plastically (ep={ep_conv})"

        d_alg = consistent_solid_tangent(mat, deps=deps)

        for h in (1e-6, 1e-7):
            d_num = _compute_solid_numerical_tangent(mat, deps, h=h)
            rel_err = np.linalg.norm(d_alg - d_num) / np.linalg.norm(d_num)
            assert rel_err < 1e-4, (
                f"Solid tangent mismatch in {name} with h={h}: rel_err={rel_err:.3e} >= 1e-4"
            )


def test_solid_tangent_fd_plastic_pure_shear():
    """Verify solid tangent against central finite differences in plastic pure shear."""
    mat = _make_law60_solid(e0=200000.0, nu=0.3, sig_y=300.0, h_slope=1000.0)

    # Strains causing plastic yield in xy, yz, and zx shears
    cases = [
        ("shear_xy", np.array([0.0, 0.0, 0.0, 0.012, 0.0, 0.0])),
        ("shear_yz", np.array([0.0, 0.0, 0.0, 0.0, 0.014, 0.0])),
        ("shear_zx", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.011])),
    ]

    for name, deps in cases:
        sig_conv, ep_conv, _ = solid_update(mat, deps=deps)
        assert ep_conv > 0.0, f"Case {name} did not yield plastically (ep={ep_conv})"

        d_alg = consistent_solid_tangent(mat, deps=deps)

        for h in (1e-6, 1e-7):
            d_num = _compute_solid_numerical_tangent(mat, deps, h=h)
            rel_err = np.linalg.norm(d_alg - d_num) / np.linalg.norm(d_num)
            assert rel_err < 1e-4, (
                f"Solid shear tangent mismatch in {name} with h={h}: rel_err={rel_err:.3e} >= 1e-4"
            )


def test_solid_tangent_fd_plastic_multiaxial_triaxial():
    """Verify solid tangent against central finite differences in general multiaxial / triaxial states."""
    mat = _make_law60_solid(e0=200000.0, nu=0.3, sig_y=300.0, h_slope=1000.0)

    # General 3D multiaxial states with tension, compression, and shear
    cases = [
        ("multiaxial_1", np.array([0.004, -0.002, 0.001, 0.003, 0.002, -0.001])),
        ("triaxial_tension", np.array([0.006, 0.002, -0.001, 0.002, 0.001, 0.003])),
        ("triaxial_compression", np.array([-0.005, -0.001, 0.002, -0.002, 0.003, 0.001])),
        ("general_shear_normal", np.array([0.0035, -0.0015, -0.001, 0.005, 0.004, 0.003])),
    ]

    for name, deps in cases:
        sig_conv, ep_conv, _ = solid_update(mat, deps=deps)
        assert ep_conv > 0.0, f"Case {name} did not yield plastically (ep={ep_conv})"

        d_alg = consistent_solid_tangent(mat, deps=deps)

        for h in (1e-6, 1e-7):
            d_num = _compute_solid_numerical_tangent(mat, deps, h=h)
            rel_err = np.linalg.norm(d_alg - d_num) / np.linalg.norm(d_num)
            assert rel_err < 1e-4, (
                f"Solid triaxial tangent mismatch in {name} with h={h}: rel_err={rel_err:.3e} >= 1e-4"
            )


def test_solid_tangent_stiffness_reduction_along_flow():
    """Verify that tangent stiffness along the plastic flow direction is strictly less

    than elastic stiffness (elastoplastic softening/flow projection).
    """
    mat = _make_law60_solid(e0=200000.0, nu=0.3, sig_y=300.0, h_slope=1000.0)
    deps = np.array([0.005, -0.001, -0.001, 0.0, 0.0, 0.0])

    d_alg = consistent_solid_tangent(mat, deps=deps)
    c_el = consistent_solid_tangent(mat)

    # The flow direction unit tensor is along dev(deps)
    flow_dir = np.array([2.0 / np.sqrt(6.0), -1.0 / np.sqrt(6.0), -1.0 / np.sqrt(6.0), 0.0, 0.0, 0.0])
    tangent_stiffness = float(flow_dir @ d_alg @ flow_dir)
    elastic_stiffness = float(flow_dir @ c_el @ flow_dir)

    assert tangent_stiffness < elastic_stiffness, (
        f"Tangential stiffness along flow ({tangent_stiffness:.1f}) should be strictly less "
        f"than elastic stiffness ({elastic_stiffness:.1f})"
    )


# =============================================================================
# 3. Pressure-Dependent States (ipfun > 0)
# =============================================================================

def test_solid_tangent_fd_pressure_dependent():
    """Verify solid tangent when pressure-dependent yield factor ipfun > 0 is active

    under non-zero initial hydrostatic pressure.
    """
    mat = _make_law60_solid(
        e0=200000.0, nu=0.3, sig_y=300.0, h_slope=1000.0,
        ipfun=1, pscale=1.0,
    )

    # Initial hydrostatic stress state: p0 = -tr(sig_old)/3 = 300.0 (compression)
    sig_old = np.array([-300.0, -300.0, -300.0, 0.0, 0.0, 0.0])
    deps = np.array([0.005, -0.001, -0.001, 0.0, 0.0, 0.0])

    sig_conv, ep_conv, _ = solid_update(mat, sig_old=sig_old, deps=deps)
    assert ep_conv > 0.0, f"Element did not yield with ipfun=1 (ep={ep_conv})"

    d_alg = consistent_solid_tangent(mat, sig_old, deps=deps)

    for h in (1e-6, 1e-7):
        d_num = _compute_solid_numerical_tangent(mat, deps, sig_old=sig_old, h=h)
        rel_err = np.linalg.norm(d_alg - d_num) / np.linalg.norm(d_num)
        assert rel_err < 1e-4, (
            f"Pressure-dependent solid tangent error {rel_err:.3e} exceeds 1e-4 for h={h}"
        )


# =============================================================================
# 4. Modulus Degraded States (ce > 0 and ifunce > 0)
# =============================================================================

def test_solid_tangent_fd_modulus_degraded_ce():
    """Verify solid tangent with exponential modulus degradation ce > 0 and einf < e0

    starting with pre-existing plastic strain.
    """
    e0 = 200000.0
    einf = 100000.0
    ce = 10.0
    mat = _make_law60_solid(
        e0=e0, nu=0.3, sig_y=300.0, h_slope=1000.0,
        ce=ce, einf=einf,
    )

    epsp_old = np.array([0.02], dtype=float)
    deps = np.array([0.005, -0.001, -0.001, 0.0, 0.0, 0.0])

    sig_conv, ep_conv, _ = solid_update(mat, deps=deps, epsp_old=epsp_old)
    assert ep_conv > epsp_old[0], f"Plastic strain did not advance: {ep_conv} <= {epsp_old[0]}"

    d_alg = consistent_solid_tangent(mat, deps=deps, epsp=epsp_old)

    for h in (1e-6, 1e-7):
        d_num = _compute_solid_numerical_tangent(mat, deps, epsp_old=epsp_old, h=h)
        rel_err = np.linalg.norm(d_alg - d_num) / np.linalg.norm(d_num)
        assert rel_err < 1e-4, (
            f"ce degraded solid tangent error {rel_err:.3e} exceeds 1e-4 for h={h}"
        )


def test_solid_tangent_fd_modulus_degraded_ifunce():
    """Verify solid tangent with tabulated modulus degradation ifunce > 0

    starting with pre-existing plastic strain.
    """
    mat = _make_law60_solid(
        e0=200000.0, nu=0.3, sig_y=300.0, h_slope=1000.0,
        ifunce=1,
    )

    epsp_old = np.array([0.025], dtype=float)
    deps = np.array([0.005, -0.001, -0.001, 0.002, 0.001, 0.0])

    sig_conv, ep_conv, _ = solid_update(mat, deps=deps, epsp_old=epsp_old)
    assert ep_conv > epsp_old[0], f"Plastic strain did not advance: {ep_conv} <= {epsp_old[0]}"

    d_alg = consistent_solid_tangent(mat, deps=deps, epsp=epsp_old)

    for h in (1e-6, 1e-7):
        d_num = _compute_solid_numerical_tangent(mat, deps, epsp_old=epsp_old, h=h)
        rel_err = np.linalg.norm(d_alg - d_num) / np.linalg.norm(d_num)
        assert rel_err < 1e-4, (
            f"ifunce degraded solid tangent error {rel_err:.3e} exceeds 1e-4 for h={h}"
        )


# =============================================================================
# 5. Shell Plane-Stress Tangent Verification
# =============================================================================

def test_shell_tangent_fd_plastic_tension():
    """Verify shell plane-stress tangent against central finite differences in plastic tension."""
    mat = _make_law60_shell(e0=210000.0, nu=0.3, sig_y=300.0, h_slope=1500.0)

    # In-plane tension increments along x and y
    cases = [
        ("tension_xx", np.array([0.003, 0.0, 0.0])),
        ("tension_yy", np.array([0.0, 0.0035, 0.0])),
        ("tension_equibiaxial", np.array([0.0025, 0.0025, 0.0])),
    ]

    for name, deps in cases:
        sig_conv, ep_conv, _ = shell_update(mat, deps=deps)
        assert ep_conv > 0.0, f"Shell case {name} did not yield: ep={ep_conv}"

        d_alg = consistent_shell_tangent(mat, deps=deps)

        for h in (1e-6, 1e-7):
            d_num = _compute_shell_numerical_tangent(mat, deps, h=h)
            rel_err = np.linalg.norm(d_alg - d_num) / np.linalg.norm(d_num)
            assert rel_err < 1e-4, (
                f"Shell tension tangent error in {name} with h={h}: rel_err={rel_err:.3e} >= 1e-4"
            )


def test_shell_tangent_fd_plastic_compression():
    """Verify shell plane-stress tangent against central finite differences in plastic compression."""
    mat = _make_law60_shell(e0=210000.0, nu=0.3, sig_y=300.0, h_slope=1500.0)

    # In-plane compression increments
    cases = [
        ("compression_xx", np.array([-0.003, 0.0, 0.0])),
        ("compression_yy", np.array([0.0, -0.0035, 0.0])),
        ("compression_biaxial", np.array([-0.002, -0.002, 0.0])),
    ]

    for name, deps in cases:
        sig_conv, ep_conv, _ = shell_update(mat, deps=deps)
        assert ep_conv > 0.0, f"Shell compression case {name} did not yield: ep={ep_conv}"

        d_alg = consistent_shell_tangent(mat, deps=deps)

        for h in (1e-6, 1e-7):
            d_num = _compute_shell_numerical_tangent(mat, deps, h=h)
            rel_err = np.linalg.norm(d_alg - d_num) / np.linalg.norm(d_num)
            assert rel_err < 1e-4, (
                f"Shell compression tangent error in {name} with h={h}: rel_err={rel_err:.3e} >= 1e-4"
            )


def test_shell_tangent_fd_plastic_shear():
    """Verify shell plane-stress tangent against central finite differences in plastic in-plane shear."""
    mat = _make_law60_shell(e0=210000.0, nu=0.3, sig_y=300.0, h_slope=1500.0)

    cases = [
        ("shear_xy_pos", np.array([0.0, 0.0, 0.005])),
        ("shear_xy_neg", np.array([0.0, 0.0, -0.005])),
    ]

    for name, deps in cases:
        sig_conv, ep_conv, _ = shell_update(mat, deps=deps)
        assert ep_conv > 0.0, f"Shell shear case {name} did not yield: ep={ep_conv}"

        d_alg = consistent_shell_tangent(mat, deps=deps)

        for h in (1e-6, 1e-7):
            d_num = _compute_shell_numerical_tangent(mat, deps, h=h)
            rel_err = np.linalg.norm(d_alg - d_num) / np.linalg.norm(d_num)
            assert rel_err < 1e-4, (
                f"Shell shear tangent error in {name} with h={h}: rel_err={rel_err:.3e} >= 1e-4"
            )


def test_shell_tangent_fd_plastic_multiaxial():
    """Verify shell plane-stress tangent against central finite differences in general in-plane multiaxial states."""
    mat = _make_law60_shell(e0=210000.0, nu=0.3, sig_y=300.0, h_slope=1500.0)

    cases = [
        ("multiaxial_1", np.array([0.0025, -0.0015, 0.002])),
        ("multiaxial_2", np.array([-0.002, 0.003, -0.0025])),
        ("multiaxial_3", np.array([0.003, 0.001, 0.0035])),
    ]

    for name, deps in cases:
        sig_conv, ep_conv, _ = shell_update(mat, deps=deps)
        assert ep_conv > 0.0, f"Shell multiaxial case {name} did not yield: ep={ep_conv}"

        d_alg = consistent_shell_tangent(mat, deps=deps)

        for h in (1e-6, 1e-7):
            d_num = _compute_shell_numerical_tangent(mat, deps, h=h)
            rel_err = np.linalg.norm(d_alg - d_num) / np.linalg.norm(d_num)
            assert rel_err < 1e-4, (
                f"Shell multiaxial tangent error in {name} with h={h}: rel_err={rel_err:.3e} >= 1e-4"
            )


# =============================================================================
# 6. Vectorized Multi-Element Batches (NEL = 1, 5, 20)
# =============================================================================

@pytest.mark.parametrize("nel", [1, 5, 20])
def test_solid_tangent_vectorized_batches(nel: int):
    """Verify vectorized multi-element batches (NEL = 1, 5, 20) for 3D solids

    match element-by-element evaluation and numerical finite differences to < 1e-4.
    """
    mat = _make_law60_solid(e0=200000.0, nu=0.3, sig_y=300.0, h_slope=1000.0)

    rng = np.random.default_rng(42 + nel)
    # Generate varied strain increments: some elastic, some plastic, some shear, some multiaxial
    deps_batch = rng.uniform(-0.004, 0.006, size=(nel, 6))

    d_batch = consistent_solid_tangent(mat, deps=deps_batch)
    assert d_batch.shape == (nel, 6, 6)

    # 1D slice input returns (6, 6)
    d_1d = consistent_solid_tangent(mat, deps=deps_batch[0])
    assert d_1d.shape == (6, 6)

    d_batch_2d = d_batch.reshape(nel, 6, 6)

    for i in range(nel):
        d_single = consistent_solid_tangent(mat, deps=deps_batch[i])
        np.testing.assert_allclose(
            d_batch_2d[i], d_single, rtol=1e-12, atol=1e-12,
            err_msg=f"Element {i} in batch NEL={nel} differs from single-element evaluation"
        )

        d_num = _compute_solid_numerical_tangent(mat, deps_batch[i], h=1e-7)
        rel_err = np.linalg.norm(d_batch_2d[i] - d_num) / np.linalg.norm(d_num)
        assert rel_err < 1e-4, f"Element {i} in batch NEL={nel} has FD error {rel_err:.3e} >= 1e-4"


@pytest.mark.parametrize("nel", [1, 5, 20])
def test_shell_tangent_vectorized_batches(nel: int):
    """Verify vectorized multi-element batches (NEL = 1, 5, 20) for 2D shells

    match element-by-element evaluation and numerical finite differences to < 1e-4.
    """
    mat = _make_law60_shell(e0=210000.0, nu=0.3, sig_y=300.0, h_slope=1500.0)

    rng = np.random.default_rng(100 + nel)
    deps_batch = rng.uniform(-0.003, 0.005, size=(nel, 3))

    d_batch = consistent_shell_tangent(mat, deps=deps_batch)
    assert d_batch.shape == (nel, 3, 3)

    # 1D slice input returns (3, 3)
    d_1d = consistent_shell_tangent(mat, deps=deps_batch[0])
    assert d_1d.shape == (3, 3)

    d_batch_2d = d_batch.reshape(nel, 3, 3)

    for i in range(nel):
        d_single = consistent_shell_tangent(mat, deps=deps_batch[i])
        np.testing.assert_allclose(
            d_batch_2d[i], d_single, rtol=1e-12, atol=1e-12,
            err_msg=f"Shell element {i} in batch NEL={nel} differs from single-element evaluation"
        )

        d_num = _compute_shell_numerical_tangent(mat, deps_batch[i], h=1e-7)
        rel_err = np.linalg.norm(d_batch_2d[i] - d_num) / np.linalg.norm(d_num)
        assert rel_err < 1e-4, f"Shell element {i} in batch NEL={nel} has FD error {rel_err:.3e} >= 1e-4"


# =============================================================================
# 7. Symmetry & Positive-Definiteness Verification
# =============================================================================

def test_solid_tangent_symmetry_ground_state():
    """Verify exact major symmetry and positive definiteness of solid tangent tensor."""
    mat = _make_law60_solid(e0=200000.0, nu=0.28)

    # 1. Ground state
    d_ground = consistent_solid_tangent(mat)
    np.testing.assert_allclose(d_ground, d_ground.T, atol=1e-12)
    eigs_ground = np.linalg.eigvalsh(d_ground)
    assert np.all(eigs_ground > 0.0), f"Ground state eigenvalues must be strictly positive, got {eigs_ground}"

    # 2. Pure elastic states
    d_elastic = consistent_solid_tangent(mat, deps=np.array([1e-5, -3e-6, -2e-6, 1e-5, 0.0, 0.0]))
    np.testing.assert_allclose(d_elastic, d_elastic.T, atol=1e-12)
    eigs_elastic = np.linalg.eigvalsh(d_elastic)
    assert np.all(eigs_elastic > 0.0), f"Elastic state eigenvalues must be strictly positive, got {eigs_elastic}"

    # 3. Plastic states: solid radial return is symmetric
    d_plastic = consistent_solid_tangent(mat, deps=np.array([0.005, -0.001, -0.001, 0.002, 0.0, 0.0]))
    np.testing.assert_allclose(d_plastic, d_plastic.T, atol=1e-12)
    eigs_plastic = np.linalg.eigvalsh(d_plastic)
    assert np.all(eigs_plastic >= -1e-10), f"Plastic state eigenvalues must be non-negative, got {eigs_plastic}"


def test_shell_tangent_symmetry_ground_state():
    """Verify exact symmetry and positive definiteness of shell tangent in ground and elastic states."""
    mat = _make_law60_shell(e0=210000.0, nu=0.3)

    # 1. Ground state
    d_ground = consistent_shell_tangent(mat)
    np.testing.assert_allclose(d_ground, d_ground.T, atol=1e-12)
    eigs_ground = np.linalg.eigvalsh(d_ground)
    assert np.all(eigs_ground > 0.0), f"Ground state eigenvalues must be strictly positive, got {eigs_ground}"

    # 2. Pure elastic states
    d_elastic = consistent_shell_tangent(mat, deps=np.array([1e-5, -2e-6, 3e-6]))
    np.testing.assert_allclose(d_elastic, d_elastic.T, atol=1e-12)
    eigs_elastic = np.linalg.eigvalsh(d_elastic)
    assert np.all(eigs_elastic > 0.0), f"Elastic state eigenvalues must be strictly positive, got {eigs_elastic}"


def test_tangents_with_symmetric_flag():
    """Verify symmetric=True projects tangent tensors to exact major symmetry (0.5*(D + D^T))."""
    mat_sol = _make_law60_solid(e0=200000.0, nu=0.3)
    mat_sh = _make_law60_shell(e0=210000.0, nu=0.3)

    deps_sol = np.array([0.005, -0.001, -0.001, 0.003, 0.002, 0.001])
    deps_sh = np.array([0.003, 0.001, 0.002])

    d_sol_sym = consistent_solid_tangent(mat_sol, deps=deps_sol, symmetric=True)
    d_sh_sym = consistent_shell_tangent(mat_sh, deps=deps_sh, symmetric=True)

    np.testing.assert_allclose(d_sol_sym, d_sol_sym.T, atol=1e-12)
    np.testing.assert_allclose(d_sh_sym, d_sh_sym.T, atol=1e-12)


# =============================================================================
# 8. Element Deletion & State Handling
# =============================================================================

def test_tangent_element_deletion():
    """Verify deleted elements (off <= 0.0) return zero tangent stiffness."""
    mat_sol = _make_law60_solid(e0=200000.0, nu=0.3)
    mat_sh = _make_law60_shell(e0=210000.0, nu=0.3)

    # 1. Single deleted element
    extra_del = {"off": np.array([0.0])}
    d_sol_del = consistent_solid_tangent(mat_sol, deps=np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0]), extra=extra_del)
    np.testing.assert_allclose(d_sol_del, np.zeros((6, 6)), atol=1e-12)

    d_sh_del = consistent_shell_tangent(mat_sh, deps=np.array([0.001, 0.0, 0.0]), extra=extra_del)
    np.testing.assert_allclose(d_sh_del, np.zeros((3, 3)), atol=1e-12)

    # 2. Mixed batch: element 0 active, element 1 deleted
    extra_mixed = {"off": np.array([1.0, 0.0])}
    deps_sol_batch = np.array([
        [0.002, -0.0005, -0.0005, 0.0, 0.0, 0.0],
        [0.002, -0.0005, -0.0005, 0.0, 0.0, 0.0],
    ])
    d_sol_batch = consistent_solid_tangent(mat_sol, deps=deps_sol_batch, extra=extra_mixed)
    assert np.linalg.norm(d_sol_batch[0]) > 0.0, "Active element must have non-zero tangent"
    np.testing.assert_allclose(d_sol_batch[1], np.zeros((6, 6)), atol=1e-12)


# =============================================================================
# 9. API Calling Conventions & Aliases
# =============================================================================

def test_tangent_calling_conventions_and_aliases():
    """Verify consistent tangent functions support all expected calling patterns and aliases."""
    mat_sol = _make_law60_solid(e0=200000.0, nu=0.3)
    mat_sh = _make_law60_shell(e0=210000.0, nu=0.3)

    deps_sol = np.array([0.004, -0.001, -0.001, 0.0, 0.0, 0.0])
    sig_sol, ep_sol, _ = solid_update(mat_sol, deps=deps_sol)

    deps_sh = np.array([0.003, 0.0, 0.0])
    sig_sh, ep_sh, _ = shell_update(mat_sh, deps=deps_sh)

    # Mode 1: by deps
    d1_sol = consistent_solid_tangent(mat_sol, deps=deps_sol)
    d1_sh = consistent_shell_tangent(mat_sh, deps=deps_sh)

    # Mode 2: by sig and epsp_incr
    d2_sol = consistent_solid_tangent(mat_sol, sig=sig_sol, epsp=0.0, epsp_incr=ep_sol)
    d2_sh = consistent_shell_tangent(mat_sh, sig=sig_sh, epsp=0.0, epsp_incr=ep_sh)

    np.testing.assert_allclose(d1_sol, d2_sol, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(d1_sh, d2_sh, rtol=1e-6, atol=1e-6)

    # Mode 3: aliases
    np.testing.assert_allclose(solid_tangent(mat_sol, deps=deps_sol), d1_sol, atol=1e-12)
    np.testing.assert_allclose(shell_tangent(mat_sh, deps=deps_sh), d1_sh, atol=1e-12)
    np.testing.assert_allclose(law60_solid_tangent(mat_sol, deps=deps_sol), d1_sol, atol=1e-12)
    np.testing.assert_allclose(law60_shell_tangent(mat_sh, deps=deps_sh), d1_sh, atol=1e-12)
    np.testing.assert_allclose(law60_consistent_solid_tangent(mat_sol, deps=deps_sol), d1_sol, atol=1e-12)
    np.testing.assert_allclose(law60_consistent_shell_tangent(mat_sh, deps=deps_sh), d1_sh, atol=1e-12)
