"""
LAW43 Consistent Algorithmic Tangent Stiffness Tensor Audit Suite.
Milestone M548: Tangent Stiffness Auditor.

Comprehensive verification of consistent algorithmic tangent operators in
pyradioss/materials/law43_hill_tab.py:
  1. Directional numerical derivative check:
     Central-difference numerical perturbation:
       C_num[:, j] = (sigma(deps + h*e_j) - sigma(deps - h*e_j)) / (2*h)
     Verified against consistent_shell_tangent and consistent_solid_tangent for h = 1e-6 and 1e-7.
  2. Pure elastic regime:
     - Shell membrane tangent matches exact plane-stress elastic matrix:
       [[A1, A2, 0], [A2, A1, 0], [0, 0, G]].
     - Solid tangent matches 3D isotropic elasticity matrix with bulk modulus C1 and shear modulus G.
  3. Plastic flow regime:
     - Consistent algorithmic tangent matches central finite difference perturbations to < 1e-4.
     - Tangent exhibits reduced stiffness along flow direction compared to elastic stiffness.
  4. Modulus degradation regime:
     - Tangent reflects degraded elastic modulus E(epsp) for both CE exponential and curve degradation.
     - Unloading stiffness matches degraded elastic matrix.
  5. Iso-kinematic hardening regime:
     - Tangent remains consistent when FISOKIN > 0 and back-stresses are active.
     - State extra dictionaries are non-destructively copied during perturbation.
  6. Element deletion regime:
     - Zero tangent tensor returned when element is deleted (off <= 0.0 or layfail == 0.0).
     - Mixed active/deleted batches correctly zero deleted elements while preserving active ones.
  7. Vectorization, batching, and shapes:
     - 1D inputs (3,) and (6,) produce (3, 3) and (6, 6).
     - 2D batched inputs (N, 3) and (N, 6) produce (N, 3, 3) and (N, 6, 6) matching element-by-element.
  8. Major symmetry:
     - Enforced when symmetric=True for both shell and solid tangents.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pytest

from pyradioss.materials import (
    consistent_shell_tangent,
    consistent_solid_tangent,
    law43_hill_tab,
    shell_layer_tangent,
    shell_membrane_tangent,
    shell_update,
    solid_tangent,
    solid_update,
)
from pyradioss.model.entities import Material


# =============================================================================
# Helper Utilities
# =============================================================================

def _copy_extra(extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Deep copy arrays inside state extra dictionary."""
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


def _make_law43_shell(
    e0: float = 210000.0,
    nu: float = 0.3,
    r00: float = 1.0,
    r45: float = 1.0,
    r90: float = 1.0,
    iyield: int = 0,
    fisokin: float = 0.0,
    ce: float = 0.0,
    einf: Optional[float] = None,
    opte: int = 0,
    curves: Optional[list] = None,
    rho0: float = 7.85e-9,
    **kwargs: Any,
) -> Material:
    """Build a validated LAW43 Material instance for shell tests."""
    if einf is None:
        einf = e0
    if curves is None:
        curves = [([0.0, 0.05, 0.2], [300.0, 450.0, 600.0], 0.0)]

    return law43_hill_tab.build_law43(
        id=43,
        rho0=rho0,
        E=e0,
        E0=e0,
        nu=nu,
        R00=r00,
        R45=r45,
        R90=r90,
        Iyield=iyield,
        FISOKIN=fisokin,
        CE=ce,
        Einf=einf,
        OPTE=opte,
        curves=curves,
        **kwargs,
    )


def _make_law43_solid(
    e0: float = 200000.0,
    nu: float = 0.28,
    r00: float = 1.0,
    r45: float = 1.0,
    r90: float = 1.0,
    iyield: int = 0,
    curves: Optional[list] = None,
    rho0: float = 7.85e-9,
    **kwargs: Any,
) -> Material:
    """Build a validated LAW43 Material instance for solid tests."""
    if curves is None:
        curves = [([0.0, 0.05, 0.2], [350.0, 500.0, 650.0], 0.0)]

    return law43_hill_tab.build_law43(
        id=43,
        rho0=rho0,
        E=e0,
        E0=e0,
        nu=nu,
        R00=r00,
        R45=r45,
        R90=r90,
        Iyield=iyield,
        curves=curves,
        **kwargs,
    )


def _compute_shell_numerical_tangent(
    mat: Material,
    sig_init: np.ndarray,
    deps_base: np.ndarray,
    epsp_init: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    h: float = 1e-7,
) -> np.ndarray:
    """Compute (3, 3) numerical plane-stress tangent via central finite difference:

    C_num[:, j] = [sigma(deps_base + h * e_j) - sigma(deps_base - h * e_j)] / (2 * h)
    """
    c_num = np.zeros((3, 3), dtype=float)
    base = np.asarray(deps_base, dtype=float).reshape(1, -1)

    for j in range(3):
        ej = np.zeros_like(base)
        ej[0, j] = h

        ex_p = _copy_extra(extra)
        sp_init = np.asarray(sig_init, dtype=float).copy().reshape(1, -1)
        ep_p = np.asarray(epsp_init, dtype=float).copy().flatten() if epsp_init is not None else None
        sp, _, _ = law43_hill_tab.shell_update(mat, sp_init, base + ej, epsp=ep_p, dt=dt, extra=ex_p)

        ex_m = _copy_extra(extra)
        sm_init = np.asarray(sig_init, dtype=float).copy().reshape(1, -1)
        ep_m = np.asarray(epsp_init, dtype=float).copy().flatten() if epsp_init is not None else None
        sm, _, _ = law43_hill_tab.shell_update(mat, sm_init, base - ej, epsp=ep_m, dt=dt, extra=ex_m)

        c_num[:, j] = (sp[0, :3] - sm[0, :3]) / (2.0 * h)

    return c_num


def _compute_solid_numerical_tangent(
    mat: Material,
    sig_init: np.ndarray,
    deps_base: np.ndarray,
    epsp_init: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    h: float = 1e-7,
) -> np.ndarray:
    """Compute (6, 6) numerical 3D solid tangent via central finite difference:

    C_num[:, j] = [sigma(deps_base + h * e_j) - sigma(deps_base - h * e_j)] / (2 * h)
    """
    c_num = np.zeros((6, 6), dtype=float)
    base = np.asarray(deps_base, dtype=float).reshape(1, -1)

    for j in range(6):
        ej = np.zeros_like(base)
        ej[0, j] = h

        ex_p = _copy_extra(extra)
        sp_init = np.asarray(sig_init, dtype=float).copy().reshape(1, -1)
        ep_p = np.asarray(epsp_init, dtype=float).copy().flatten() if epsp_init is not None else None
        sp, _, _ = law43_hill_tab.solid_update(mat, sp_init, base + ej, epsp=ep_p, dt=dt, extra=ex_p)

        ex_m = _copy_extra(extra)
        sm_init = np.asarray(sig_init, dtype=float).copy().reshape(1, -1)
        ep_m = np.asarray(epsp_init, dtype=float).copy().flatten() if epsp_init is not None else None
        sm, _, _ = law43_hill_tab.solid_update(mat, sm_init, base - ej, epsp=ep_m, dt=dt, extra=ex_m)

        c_num[:, j] = (sp[0, :6] - sm[0, :6]) / (2.0 * h)

    return c_num


# =============================================================================
# 1. Pure Elastic Regime Verification
# =============================================================================

def test_shell_tangent_elastic_exact_match():
    """Verify shell membrane tangent matches exact plane-stress elastic matrix:
    [[A1, A2, 0], [A2, A1, 0], [0, 0, G]] and FD perturbation to < 1e-6.
    """
    e0 = 210000.0
    nu = 0.3
    mat = _make_law43_shell(e0=e0, nu=nu)

    # Theoretical plane-stress coefficients
    a1_exact = e0 / (1.0 - nu ** 2)
    a2_exact = nu * a1_exact
    g_exact = 0.5 * e0 / (1.0 + nu)

    expected = np.array([
        [a1_exact, a2_exact, 0.0],
        [a2_exact, a1_exact, 0.0],
        [0.0, 0.0, g_exact],
    ], dtype=float)

    # 1. Direct call to shell_membrane_tangent
    c_mem = law43_hill_tab.shell_membrane_tangent(mat)
    np.testing.assert_allclose(c_mem, expected, rtol=1e-12, atol=1e-12)

    # 2. Package-level dispatch
    c_mem_pkg = shell_membrane_tangent(mat)
    np.testing.assert_allclose(c_mem_pkg, expected, rtol=1e-12, atol=1e-12)

    # 3. Direct call to consistent_shell_tangent without deps (pure elastic)
    sig0 = np.array([[50.0, -20.0, 15.0]])
    c_alg = law43_hill_tab.consistent_shell_tangent(mat, sig0, epsp_incr=0.0)
    np.testing.assert_allclose(c_alg[0], expected, rtol=1e-12, atol=1e-12)

    # 4. Central FD perturbation check with small elastic strain increment
    deps_el = np.array([[0.0002, -0.0001, 0.00015]])
    for h in (1e-6, 1e-7):
        c_fd = law43_hill_tab.consistent_shell_tangent(mat, sig0, deps=deps_el, h=h)
        c_num = _compute_shell_numerical_tangent(mat, sig0, deps_el, h=h)

        np.testing.assert_allclose(c_fd[0], expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(c_num, expected, rtol=1e-6, atol=1e-6)
        rel_diff = np.linalg.norm(c_fd[0] - expected) / np.linalg.norm(expected)
        assert rel_diff < 1e-6, f"Elastic shell FD error {rel_diff:.3e} exceeds 1e-6 for h={h}"

    # Verify symmetry and positive definiteness
    assert np.allclose(c_mem, c_mem.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(c_mem)
    assert np.all(eigvals > 0.0), f"Elastic tangent eigenvalues must be positive, got {eigvals}"


def test_solid_tangent_elastic_exact_match():
    """Verify solid tangent matches 3D isotropic elasticity matrix with bulk C1 and shear G."""
    e0 = 200000.0
    nu = 0.28
    mat = _make_law43_solid(e0=e0, nu=nu)

    k_bulk = e0 / (3.0 * (1.0 - 2.0 * nu))
    g_shear = 0.5 * e0 / (1.0 + nu)
    c11 = k_bulk + 4.0 / 3.0 * g_shear
    c12 = k_bulk - 2.0 / 3.0 * g_shear

    expected = np.zeros((6, 6), dtype=float)
    expected[0, 0] = expected[1, 1] = expected[2, 2] = c11
    expected[0, 1] = expected[0, 2] = expected[1, 0] = expected[1, 2] = expected[2, 0] = expected[2, 1] = c12
    expected[3, 3] = expected[4, 4] = expected[5, 5] = g_shear

    # 1. Direct call to consistent_solid_tangent without deps
    sig0 = np.array([[20.0, -10.0, 5.0, 2.0, 1.0, -3.0]])
    c_sol = law43_hill_tab.consistent_solid_tangent(mat, sig0)
    np.testing.assert_allclose(c_sol[0], expected, rtol=1e-12, atol=1e-12)

    # 2. Central FD perturbation with small elastic strain increment
    deps_el = np.array([[0.0001, -0.00005, 0.00008, 0.00003, -0.00002, 0.00004]])
    for h in (1e-6, 1e-7):
        c_fd = law43_hill_tab.consistent_solid_tangent(mat, sig0, deps=deps_el, h=h)
        c_num = _compute_solid_numerical_tangent(mat, sig0, deps_el, h=h)

        np.testing.assert_allclose(c_fd[0], expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(c_num, expected, rtol=1e-6, atol=1e-6)
        rel_diff = np.linalg.norm(c_fd[0] - expected) / np.linalg.norm(expected)
        assert rel_diff < 1e-6, f"Elastic solid FD error {rel_diff:.3e} exceeds 1e-6 for h={h}"

    # Verify symmetry and positive definiteness
    assert np.allclose(expected, expected.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(expected)
    assert np.all(eigvals > 0.0), f"Elastic solid tangent eigenvalues must be positive, got {eigvals}"


# =============================================================================
# 2. Plastic Flow Regime Verification
# =============================================================================

def test_shell_tangent_fd_plastic_flow():
    """Verify consistent algorithmic tangent against central finite difference perturbations
    dsigma / ddeps with h = 1e-6 and 1e-7 in plastic flow regime.
    """
    mat = _make_law43_shell(
        e0=210000.0,
        nu=0.3,
        r00=1.6,
        r45=1.2,
        r90=1.9,
        curves=[([0.0, 0.02, 0.1], [320.0, 450.0, 600.0], 0.0)],
    )

    # Multi-axial strain increments that cause distinct plastic flow states
    test_cases = [
        # (name, sig_init, deps)
        ("uniaxial_x", np.array([[150.0, 0.0, 0.0]]), np.array([[0.004, -0.001, 0.0005]])),
        ("transverse_y", np.array([[0.0, 180.0, 0.0]]), np.array([[-0.001, 0.0045, 0.0005]])),
        ("pure_shear", np.array([[0.0, 0.0, 80.0]]), np.array([[0.0002, -0.0002, 0.006]])),
        ("biaxial", np.array([[120.0, 120.0, 20.0]]), np.array([[0.0035, 0.003, 0.001]])),
    ]

    for name, sig0, deps in test_cases:
        # Run base update to confirm plastic flow
        s_base, ep_base, _ = law43_hill_tab.shell_update(mat, sig0.copy(), deps.copy(), epsp=0.0)
        assert ep_base > 0.0, f"Case {name} did not yield plastically: ep={ep_base}"

        for h in (1e-6, 1e-7):
            c_fd = law43_hill_tab.consistent_shell_tangent(mat, sig0, deps=deps, h=h)
            c_num = _compute_shell_numerical_tangent(mat, sig0, deps, h=h)

            diff_max = np.max(np.abs(c_fd - c_num))
            rel_err = diff_max / np.max(np.abs(c_num))
            assert rel_err < 1e-4, (
                f"Plastic flow shell tangent mismatch in {name} with h={h}: "
                f"max diff={diff_max:.3e}, rel err={rel_err:.3e}"
            )

        # In plastic flow, the tangent along loading direction must be strictly softer than elastic
        c_fd_mat = c_fd[0] if c_fd.ndim == 3 else c_fd
        c_el = law43_hill_tab.shell_membrane_tangent(mat)
        if name == "uniaxial_x":
            assert c_fd_mat[0, 0] < c_el[0, 0], "Plastic tangent C_xx should be lower than elastic A1"
        elif name == "pure_shear":
            assert c_fd_mat[2, 2] < c_el[2, 2], "Plastic tangent C_xy should be lower than elastic G"

        # Tangent diagonal elements must remain positive
        assert np.all(np.diag(c_fd_mat) > 0.0), f"Tangent diagonal must be positive in {name}: {np.diag(c_fd_mat)}"


def test_solid_tangent_fd_plastic_flow():
    """Verify consistent solid tangent against central finite difference perturbations
    dsigma / ddeps with h = 1e-6 and 1e-7 in 3D plastic flow regime.
    """
    mat = _make_law43_solid(
        e0=200000.0,
        nu=0.28,
        curves=[([0.0, 0.02, 0.1], [300.0, 420.0, 580.0], 0.0)],
    )

    test_cases = [
        # (name, sig_init, deps)
        ("uniaxial_tension", np.zeros((1, 6)), np.array([[0.0035, -0.001, -0.001, 0.0, 0.0, 0.0]])),
        ("pure_shear_xy", np.zeros((1, 6)), np.array([[0.0, 0.0, 0.0, 0.005, 0.0, 0.0]])),
        ("triaxial_flow", np.array([[100.0, -50.0, 20.0, 15.0, 0.0, 0.0]]),
         np.array([[0.003, -0.0015, 0.001, 0.002, 0.001, 0.0005]])),
    ]

    for name, sig0, deps in test_cases:
        s_base, ep_base, _ = law43_hill_tab.solid_update(mat, sig0.copy(), deps.copy(), epsp=0.0)
        assert ep_base > 0.0, f"Case {name} did not yield plastically: ep={ep_base}"

        for h in (1e-6, 1e-7):
            c_fd = law43_hill_tab.consistent_solid_tangent(mat, sig0, deps=deps, h=h)
            c_num = _compute_solid_numerical_tangent(mat, sig0, deps, h=h)

            diff_max = np.max(np.abs(c_fd - c_num))
            rel_err = diff_max / np.max(np.abs(c_num))
            assert rel_err < 1e-4, (
                f"Plastic flow solid tangent mismatch in {name} with h={h}: "
                f"max diff={diff_max:.3e}, rel err={rel_err:.3e}"
            )

        # Diagonals remain positive
        c_fd_mat = c_fd[0] if c_fd.ndim == 3 else c_fd
        assert np.all(np.diag(c_fd_mat) > 0.0), f"Solid tangent diagonal must be positive: {np.diag(c_fd_mat)}"


# =============================================================================
# 3. Modulus Degradation Regime Verification
# =============================================================================

def test_shell_tangent_fd_with_modulus_degradation():
    """Verify tangent reflects degraded elastic modulus E(epsp) for both exponential and curve degradation."""
    e0 = 200000.0
    nu = 0.3
    ce = 10.0
    einf = 120000.0

    mat_exp = _make_law43_shell(
        e0=e0, nu=nu, ce=ce, einf=einf,
        curves=[([0.0, 0.1], [300.0, 500.0], 0.0)],
    )

    # 1. Check degraded elastic matrix for various epsp values
    test_epsp = [0.02, 0.05, 0.10]
    for ep_val in test_epsp:
        # Theoretical degraded E
        e_deg_expected = e0 - (e0 - einf) * (1.0 - math.exp(-ce * ep_val))
        a1_deg = e_deg_expected / (1.0 - nu ** 2)
        a2_deg = nu * a1_deg
        g_deg = 0.5 * e_deg_expected / (1.0 + nu)

        expected_deg = np.array([
            [a1_deg, a2_deg, 0.0],
            [a2_deg, a1_deg, 0.0],
            [0.0, 0.0, g_deg],
        ], dtype=float)

        # Verify shell_membrane_tangent with epsp
        c_mem_deg = law43_hill_tab.shell_membrane_tangent(mat_exp, epsp=ep_val)
        np.testing.assert_allclose(c_mem_deg, expected_deg, rtol=1e-10)

        # Verify elastic unloading from degraded state matches expected degraded elasticity
        sig_unld = np.array([[40.0, -10.0, 5.0]])
        deps_unld = np.array([[0.0001, -0.00005, 0.00008]])
        extra_deg = {"pla43": np.array([ep_val])}

        c_fd_unld = law43_hill_tab.consistent_shell_tangent(
            mat_exp, sig_unld, epsp=ep_val, deps=deps_unld, extra=extra_deg, h=1e-7
        )
        c_num_unld = _compute_shell_numerical_tangent(
            mat_exp, sig_unld, deps_unld, epsp_init=np.array([ep_val]), extra=extra_deg, h=1e-7
        )

        c_fd_mat = c_fd_unld[0] if c_fd_unld.ndim == 3 else c_fd_unld
        np.testing.assert_allclose(c_fd_mat, expected_deg, rtol=1e-5)
        np.testing.assert_allclose(c_num_unld, expected_deg, rtol=1e-5)

    # 2. Active plastic flow with ongoing modulus degradation
    sig_act = np.array([[120.0, 20.0, 10.0]])
    deps_act = np.array([[0.003, -0.001, 0.001]])
    extra_act = {"pla43": np.array([0.03])}

    c_fd_act = law43_hill_tab.consistent_shell_tangent(
        mat_exp, sig_act, epsp=0.03, deps=deps_act, extra=extra_act, h=1e-7
    )
    c_num_act = _compute_shell_numerical_tangent(
        mat_exp, sig_act, deps_act, epsp_init=np.array([0.03]), extra=extra_act, h=1e-7
    )

    diff_act = np.max(np.abs(c_fd_act - c_num_act))
    rel_act = diff_act / np.max(np.abs(c_num_act))
    assert rel_act < 1e-4, f"Degradation plastic tangent rel error {rel_act:.3e} exceeds 1e-4"


# =============================================================================
# 4. Iso-Kinematic Hardening Regime Verification
# =============================================================================

def test_shell_tangent_fd_with_isokinematic_hardening():
    """Verify tangent remains consistent when FISOKIN > 0 and back-stresses are active."""
    mat_kin = _make_law43_shell(
        e0=210000.0,
        nu=0.3,
        fisokin=0.5,  # 50% kinematic, 50% isotropic
        curves=[([0.0, 0.05, 0.15], [300.0, 450.0, 600.0], 0.0)],
    )

    # State with pre-existing back-stresses alpha = [alpha_xx, alpha_yy, alpha_xy, edot]
    alpha_init = np.array([50.0, -20.0, 15.0, 0.0])
    extra_init = {
        "uvar43": alpha_init.copy(),
        "pla43": np.array([0.01]),
        "off43": np.array([1.0]),
    }

    sig0 = np.array([[180.0, -30.0, 25.0]])
    deps = np.array([[0.003, -0.001, 0.0015]])

    # Make sure extra_init is not mutated across the test
    extra_run = _copy_extra(extra_init)
    c_fd = law43_hill_tab.consistent_shell_tangent(
        mat_kin, sig0, epsp=0.01, deps=deps, extra=extra_run, h=1e-7
    )

    # Verify extra_init was not modified
    np.testing.assert_allclose(extra_init["uvar43"], alpha_init, err_msg="State dict mutated!")

    c_num = _compute_shell_numerical_tangent(
        mat_kin, sig0, deps, epsp_init=np.array([0.01]), extra=extra_init, h=1e-7
    )

    diff_max = np.max(np.abs(c_fd - c_num))
    rel_err = diff_max / np.max(np.abs(c_num))
    assert rel_err < 1e-4, f"Kinematic hardening tangent rel err {rel_err:.3e} exceeds 1e-4"

    # Reverse loading check (Bauschinger effect)
    deps_rev = np.array([[-0.0035, 0.001, -0.001]])
    c_fd_rev = law43_hill_tab.consistent_shell_tangent(
        mat_kin, sig0, epsp=0.01, deps=deps_rev, extra=extra_init, h=1e-7
    )
    c_num_rev = _compute_shell_numerical_tangent(
        mat_kin, sig0, deps_rev, epsp_init=np.array([0.01]), extra=extra_init, h=1e-7
    )
    rel_err_rev = np.max(np.abs(c_fd_rev - c_num_rev)) / np.max(np.abs(c_num_rev))
    assert rel_err_rev < 1e-4, f"Reverse loading kinematic rel err {rel_err_rev:.3e} exceeds 1e-4"


# =============================================================================
# 5. Element Deletion Regime Verification
# =============================================================================

def test_tangent_deleted_element_zero():
    """Verify zero tangent tensor when element is deleted (off == 0.0 or layfail == 0.0)."""
    mat_sh = _make_law43_shell()
    mat_so = _make_law43_solid()

    sig3 = np.array([[100.0, 50.0, 20.0]])
    deps3 = np.array([[0.002, -0.0005, 0.001]])

    sig6 = np.array([[100.0, 50.0, 20.0, 10.0, 5.0, 2.0]])
    deps6 = np.array([[0.002, -0.0005, 0.0005, 0.001, 0.0, 0.0]])

    # 1. Single element shell deletion via extra dict
    for off_key in ("off43", "off", "layfail"):
        extra_del = {off_key: np.array([0.0])}
        # Elastic membrane tangent
        c_mem = law43_hill_tab.shell_membrane_tangent(mat_sh, extra=extra_del)
        np.testing.assert_allclose(c_mem, np.zeros((3, 3)), atol=1e-14)

        # Consistent shell tangent without deps
        c_sh_nodeps = law43_hill_tab.consistent_shell_tangent(mat_sh, sig3, extra=extra_del)
        np.testing.assert_allclose(c_sh_nodeps[0], np.zeros((3, 3)), atol=1e-14)

        # Consistent shell tangent with deps
        c_sh_deps = law43_hill_tab.consistent_shell_tangent(mat_sh, sig3, deps=deps3, extra=extra_del)
        np.testing.assert_allclose(c_sh_deps[0], np.zeros((3, 3)), atol=1e-14)

    # 2. Single element solid deletion
    for off_key in ("off43", "off"):
        extra_del = {off_key: np.array([0.0])}
        c_so_nodeps = law43_hill_tab.consistent_solid_tangent(mat_so, sig6, extra=extra_del)
        np.testing.assert_allclose(c_so_nodeps[0], np.zeros((6, 6)), atol=1e-14)

        c_so_deps = law43_hill_tab.consistent_solid_tangent(mat_so, sig6, deps=deps6, extra=extra_del)
        np.testing.assert_allclose(c_so_deps[0], np.zeros((6, 6)), atol=1e-14)

    # 3. Mixed active/deleted batch: elem 0 active (off=1.0), elem 1 deleted (off=0.0), elem 2 active (off=1.0)
    sig_batch = np.tile(sig3, (3, 1))
    deps_batch = np.tile(deps3, (3, 1))
    extra_mixed = {"off43": np.array([1.0, 0.0, 1.0])}

    c_sh_mixed = law43_hill_tab.consistent_shell_tangent(mat_sh, sig_batch, deps=deps_batch, extra=extra_mixed)
    assert c_sh_mixed.shape == (3, 3, 3)

    # Elem 1 must be strictly zero
    np.testing.assert_allclose(c_sh_mixed[1], np.zeros((3, 3)), atol=1e-14)
    # Elem 0 and 2 must be non-zero and identical
    assert np.linalg.norm(c_sh_mixed[0]) > 0.0
    assert np.linalg.norm(c_sh_mixed[2]) > 0.0
    np.testing.assert_allclose(c_sh_mixed[0], c_sh_mixed[2], atol=1e-12)

    # Solid mixed batch
    sig6_batch = np.tile(sig6, (3, 1))
    deps6_batch = np.tile(deps6, (3, 1))
    c_so_mixed = law43_hill_tab.consistent_solid_tangent(mat_so, sig6_batch, deps=deps6_batch, extra=extra_mixed)
    assert c_so_mixed.shape == (3, 6, 6)
    np.testing.assert_allclose(c_so_mixed[1], np.zeros((6, 6)), atol=1e-14)
    assert np.linalg.norm(c_so_mixed[0]) > 0.0
    np.testing.assert_allclose(c_so_mixed[0], c_so_mixed[2], atol=1e-12)


# =============================================================================
# 6. Tangent Batching and Shape Verification
# =============================================================================

def test_tangent_batching_and_shapes():
    """Verify 1D shapes (3,) -> (3, 3) and (6,) -> (6, 6), and 2D batched shapes (N, 3, 3) and (N, 6, 6)."""
    mat_sh = _make_law43_shell()
    mat_so = _make_law43_solid()

    # 1. 1D input shapes
    sig1d_3 = np.array([40.0, 20.0, 10.0])
    deps1d_3 = np.array([0.001, -0.0003, 0.0005])

    c_1d_sh = law43_hill_tab.consistent_shell_tangent(mat_sh, sig1d_3, deps=deps1d_3)
    assert c_1d_sh.shape == (3, 3)

    sig1d_6 = np.array([40.0, 20.0, 10.0, 5.0, -2.0, 3.0])
    deps1d_6 = np.array([0.001, -0.0003, 0.0002, 0.0005, 0.0, 0.0])

    c_1d_so = law43_hill_tab.consistent_solid_tangent(mat_so, sig1d_6, deps=deps1d_6)
    assert c_1d_so.shape == (6, 6)

    # 2. Batched input shapes for N = 1, 4, 16
    for n in (1, 4, 16):
        sig_sh_n = np.tile(sig1d_3, (n, 1))
        deps_sh_n = np.tile(deps1d_3, (n, 1))

        # Add slight variation per row
        factors = np.linspace(0.8, 1.2, n)[:, None]
        sig_sh_n = sig_sh_n * factors
        deps_sh_n = deps_sh_n * factors

        c_batch_sh = law43_hill_tab.consistent_shell_tangent(mat_sh, sig_sh_n, deps=deps_sh_n)
        assert c_batch_sh.shape == (n, 3, 3)

        # Check equivalence with element-by-element evaluation
        for i in range(n):
            c_single_i = law43_hill_tab.consistent_shell_tangent(mat_sh, sig_sh_n[i], deps=deps_sh_n[i])
            np.testing.assert_allclose(c_batch_sh[i], c_single_i, rtol=1e-12, atol=1e-12)

        sig_so_n = np.tile(sig1d_6, (n, 1)) * factors
        deps_so_n = np.tile(deps1d_6, (n, 1)) * factors

        c_batch_so = law43_hill_tab.consistent_solid_tangent(mat_so, sig_so_n, deps=deps_so_n)
        assert c_batch_so.shape == (n, 6, 6)

        for i in range(n):
            c_single_so_i = law43_hill_tab.consistent_solid_tangent(mat_so, sig_so_n[i], deps=deps_so_n[i])
            np.testing.assert_allclose(c_batch_so[i], c_single_so_i, rtol=1e-12, atol=1e-12)

    # 3. Empty input defense
    empty_sh = law43_hill_tab.consistent_shell_tangent(mat_sh, np.empty((0, 3)))
    assert empty_sh.shape == (0, 3, 3)

    empty_so = law43_hill_tab.consistent_solid_tangent(mat_so, np.empty((0, 6)))
    assert empty_so.shape == (0, 6, 6)


# =============================================================================
# 7. Major Symmetry Enforcement
# =============================================================================

def test_tangent_symmetric_option():
    """Verify symmetric option enforces C_ij = C_ji for both shell and solid tangents."""
    mat_sh = _make_law43_shell(
        curves=[([0.0, 0.05, 0.15], [300.0, 450.0, 600.0], 0.0)],
        fisokin=0.4,
    )
    mat_so = _make_law43_solid(
        curves=[([0.0, 0.05, 0.15], [320.0, 480.0, 620.0], 0.0)],
    )

    # Non-symmetric plastic loading state
    sig3 = np.array([[180.0, -40.0, 25.0]])
    deps3 = np.array([[0.0035, -0.001, 0.002]])

    # 1. Shell without symmetry enforcement can have mild non-symmetry
    c_raw = law43_hill_tab.consistent_shell_tangent(mat_sh, sig3, deps=deps3, symmetric=False)[0]
    # Shell with symmetry enforcement
    c_sym = law43_hill_tab.consistent_shell_tangent(mat_sh, sig3, deps=deps3, symmetric=True)[0]

    np.testing.assert_allclose(c_sym, c_sym.T, atol=1e-12)
    np.testing.assert_allclose(c_sym, 0.5 * (c_raw + c_raw.T), atol=1e-12)

    # 2. Solid with symmetry enforcement
    sig6 = np.array([[150.0, -60.0, 30.0, 20.0, -10.0, 15.0]])
    deps6 = np.array([[0.003, -0.001, 0.001, 0.002, -0.001, 0.001]])

    c_so_raw = law43_hill_tab.consistent_solid_tangent(mat_so, sig6, deps=deps6, symmetric=False)[0]
    c_so_sym = law43_hill_tab.consistent_solid_tangent(mat_so, sig6, deps=deps6, symmetric=True)[0]

    np.testing.assert_allclose(c_so_sym, c_so_sym.T, atol=1e-12)
    np.testing.assert_allclose(c_so_sym, 0.5 * (c_so_raw + c_so_raw.T), atol=1e-12)

    # 3. Batched symmetry
    sig_b = np.tile(sig3, (4, 1))
    deps_b = np.tile(deps3, (4, 1))
    c_b_sym = law43_hill_tab.consistent_shell_tangent(mat_sh, sig_b, deps=deps_b, symmetric=True)
    for i in range(4):
        np.testing.assert_allclose(c_b_sym[i], c_b_sym[i].T, atol=1e-12)


# =============================================================================
# 8. Package-level Integration and Dispatches
# =============================================================================

def test_package_dispatches_and_signatures():
    """Verify package-level dispatcher functions shell_layer_tangent and solid_tangent for LAW43."""
    mat_sh = _make_law43_shell()
    mat_so = _make_law43_solid()

    sig3 = np.array([[50.0, 20.0, 10.0]])
    sig6 = np.array([[50.0, 20.0, 10.0, 5.0, 2.0, 1.0]])

    # shell_layer_tangent dispatch
    c_sh_layer = shell_layer_tangent(mat_sh, sig3, epsp=np.zeros(1), epsp_incr=np.zeros(1))
    assert c_sh_layer.shape == (1, 3, 3)

    # solid_tangent dispatch
    c_so_disp = solid_tangent(mat_so, sig6, epsp=np.zeros(1), epsp_incr=np.zeros(1))
    assert c_so_disp.shape == (1, 6, 6)
