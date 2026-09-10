"""Tests for LAW22 Consistent Algorithmic Tangent Stiffness Tensors
(/MAT/LAW22, /MAT/DAMA, /MAT/PLAS_DAMA).
Milestone M545: Tangent Stiffness Audit.

Verifies:
  1. Central finite difference perturbation:
     C_num[:, i] = (sig(deps + h*e_i) - sig(deps - h*e_i)) / (2*h) with h = 1e-6 or 1e-7.
  2. Elastic regime (undamaged): exact match between C_algo and C_num to within 1e-6,
     and exact agreement with elastic constitutive matrices (shell_membrane_tangent, solid_tangent).
  3. Damaged elastic regime (epseq >= eps_dam): degraded modulus E_curr = alpe * E
     matches C_num to within 1e-5.
  4. Plastic yielding regime: elastoplastic return mapping consistent tangent matches C_num
     and directional derivative consistency along plastic flow direction (< 1e-4 relative / abs).
  5. Both shell (3x3) and solid (6x6) tensors for single element (1D and 2D) and vectorized batches.
  6. Inactive/failed elements (off22 <= 0 or epseq >= eps_max) produce strictly zero tangent stiffness.
  7. Major symmetry enforcement under symmetric=True.
  8. Defensive empty array handling and package-level dispatchers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.materials.law22_dama import (
    build_law22,
    shell_update,
    solid_update,
    consistent_shell_tangent,
    consistent_solid_tangent,
    shell_membrane_tangent,
    solid_tangent,
)
from pyradioss.materials.law22_dama import _P_PLANE
from pyradioss.model.entities import Material


def _copy_extra(extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Deep copy arrays inside state extra dictionary."""
    if extra is None:
        return None
    return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in extra.items()}


def _build_test_material_shell(**kwargs: Any) -> Material:
    """Helper to instantiate LAW22 material for shell tests."""
    defaults = dict(
        id=22,
        rho0=7.8e-9,
        E=200000.0,
        nu=0.3,
        a=250.0,
        b=500.0,
        n=0.5,
        eps_dam=0.01,
        E_tan=-4000.0,
        eps_max=0.08,
        sig_max=800.0,
        c=0.0,
        eps_dot_0=1.0,
        ICC=1,
    )
    defaults.update(kwargs)
    return build_law22(**defaults)


def _build_test_material_solid(**kwargs: Any) -> Material:
    """Helper to instantiate LAW22 material for solid tests."""
    defaults = dict(
        id=22,
        rho0=7.8e-9,
        E=210000.0,
        nu=0.28,
        a=300.0,
        b=600.0,
        n=0.5,
        eps_dam=0.015,
        E_tan=-3000.0,
        eps_max=0.10,
        sig_max=900.0,
        c=0.0,
        eps_dot_0=1.0,
        ICC=2,
    )
    defaults.update(kwargs)
    return build_law22(**defaults)


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
        sp, _, _ = shell_update(mat, sp_init, base + ej, epsp=ep_p, dt=dt, extra=ex_p)

        ex_m = _copy_extra(extra)
        sm_init = np.asarray(sig_init, dtype=float).copy().reshape(1, -1)
        ep_m = np.asarray(epsp_init, dtype=float).copy().flatten() if epsp_init is not None else None
        sm, _, _ = shell_update(mat, sm_init, base - ej, epsp=ep_m, dt=dt, extra=ex_m)

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
        sp, _, _ = solid_update(mat, sp_init, base + ej, epsp=ep_p, dt=dt, extra=ex_p)

        ex_m = _copy_extra(extra)
        sm_init = np.asarray(sig_init, dtype=float).copy().reshape(1, -1)
        ep_m = np.asarray(epsp_init, dtype=float).copy().flatten() if epsp_init is not None else None
        sm, _, _ = solid_update(mat, sm_init, base - ej, epsp=ep_m, dt=dt, extra=ex_m)

        c_num[:, j] = (sp[0, :6] - sm[0, :6]) / (2.0 * h)

    return c_num


# ============================================================================
# 1. Shell Tangent Tests (Plane Stress, 3x3)
# ============================================================================

class TestShellConsistentTangent:
    """Audit suite for consistent plane-stress shell tangent (3x3)."""

    def test_shell_elastic_single_element(self):
        """Verify undamaged elastic shell tangent matches numerical tangent < 1e-6."""
        mat = _build_test_material_shell()
        sig0 = np.array([[50.0, 20.0, 10.0]])
        deps0 = np.array([[0.0003, -0.0001, 0.0002]])
        h = 1e-7

        extra = {}
        s_base, ep_base, _ = shell_update(mat, sig0.copy(), deps0.copy(), extra=extra)
        assert ep_base[0] == 0.0

        c_num = _compute_shell_numerical_tangent(mat, sig0, deps0, h=h)
        c_alg = consistent_shell_tangent(mat, s_base, extra=extra)[0]

        diff_max = np.max(np.abs(c_num - c_alg))
        assert diff_max < 1e-6, f"Elastic shell tangent diff {diff_max:.3e} exceeds 1e-6"

        c_el = shell_membrane_tangent(mat)
        np.testing.assert_allclose(c_alg, c_el, atol=1e-12)
        assert np.allclose(c_alg, c_alg.T, atol=1e-12)

    def test_shell_elastic_varying_states(self):
        """Verify elastic shell tangent across multiple multi-axial initial stress states."""
        mat = _build_test_material_shell()
        h = 1e-7

        test_states = [
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[120.0, -40.0, 15.0]]),
            np.array([[-80.0, 60.0, -25.0]]),
            np.array([[0.0, 0.0, 50.0]]),
        ]

        for sig0 in test_states:
            deps0 = np.array([[0.0002, -0.0001, 0.00015]])
            extra = {}
            s_base, _, _ = shell_update(mat, sig0.copy(), deps0.copy(), extra=extra)

            c_num = _compute_shell_numerical_tangent(mat, sig0, deps0, h=h)
            c_alg = consistent_shell_tangent(mat, s_base, extra=extra)[0]

            diff_max = np.max(np.abs(c_num - c_alg))
            assert diff_max < 1e-6, f"Varying elastic state diff {diff_max:.3e} exceeds 1e-6"

    @pytest.mark.parametrize("epsp_val", [0.015, 0.025, 0.04])
    def test_shell_damaged_elastic_regime(self, epsp_val: float):
        """Verify damaged elastic regime (epseq >= eps_dam) degrades modulus E_curr = alpe * E (< 1e-5)."""
        mat = _build_test_material_shell(eps_dam=0.01)
        sig0 = np.array([[30.0, 10.0, 5.0]])
        # Small strain increment so it stays within damaged elastic boundary
        deps0 = np.array([[0.0001, -0.00005, 0.00008]])
        epsp_init = np.array([epsp_val])
        extra = {"epsp22": epsp_init.copy(), "off22": np.array([1.0])}
        h = 1e-7

        s_base, ep_base, _ = shell_update(mat, sig0.copy(), deps0.copy(), epsp=epsp_init.copy(), extra=extra)
        assert ep_base[0] == pytest.approx(epsp_val)

        alpe = extra["alpe22"][0]
        assert alpe < 1.0, f"Expected degradation factor alpe < 1.0, got {alpe}"
        assert alpe > 0.0

        c_num = _compute_shell_numerical_tangent(mat, sig0, deps0, epsp_init=epsp_init, extra=extra, h=h)
        c_alg = consistent_shell_tangent(mat, s_base, epsp=ep_base, extra=extra)[0]

        diff_max = np.max(np.abs(c_num - c_alg))
        assert diff_max < 1e-5, f"Damaged elastic shell diff {diff_max:.3e} exceeds 1e-5 for epsp={epsp_val}"

        # Verify exact analytical degraded stiffness
        E_curr = alpe * mat.params["E"]
        nu = mat.params["nu"]
        a1 = E_curr / (1.0 - nu ** 2)
        a2 = nu * a1
        G_curr = 0.5 * E_curr / (1.0 + nu)
        c_expected = np.array([[a1, a2, 0.0], [a2, a1, 0.0], [0.0, 0.0, G_curr]])
        np.testing.assert_allclose(c_alg, c_expected, atol=1e-10)

    def test_shell_plastic_yield_undamaged(self):
        """Verify plastic yield state consistency and directional derivative along flow direction."""
        mat = _build_test_material_shell(eps_dam=0.02)
        sig0 = np.array([[100.0, 50.0, 20.0]])
        # Strain increment causing plastic deformation while epseq < eps_dam
        deps0 = np.array([[0.003, 0.001, 0.0005]])
        epsp_init = np.zeros(1)
        extra_init = {"epsp22": epsp_init.copy(), "off22": np.array([1.0])}
        extra_run = _copy_extra(extra_init)
        h = 1e-7

        s_base, ep_base, _ = shell_update(mat, sig0.copy(), deps0.copy(), epsp=epsp_init.copy(), extra=extra_run)
        assert extra_run["dpla"][0] > 0.0, "Element must yield plastically"
        assert ep_base[0] < mat.params["eps_dam"], "Plastic step must be undamaged"

        c_num = _compute_shell_numerical_tangent(mat, sig0, deps0, epsp_init=epsp_init, extra=extra_init, h=h)
        c_alg = consistent_shell_tangent(mat, s_base, epsp=ep_base, extra=extra_run)[0]

        # Full matrix difference
        diff_max = np.max(np.abs(c_num - c_alg))
        assert diff_max < 1e-4, f"Plastic shell tangent diff {diff_max:.3e} exceeds 1e-4"

        # Directional derivative consistency along plastic flow direction
        # In plane-stress von Mises, flow direction vector in strain space is proportional to P @ sig
        flow_vec = _P_PLANE @ s_base[0, :3]
        flow_vec = flow_vec / np.linalg.norm(flow_vec)

        d_sig_num = c_num @ flow_vec
        d_sig_alg = c_alg @ flow_vec
        flow_err = np.linalg.norm(d_sig_num - d_sig_alg)
        assert flow_err < 1e-4, f"Flow direction error {flow_err:.3e} exceeds 1e-4"

    def test_shell_plastic_yield_damaged(self):
        """Verify plastic yielding in pre-damaged state (epseq >= eps_dam)."""
        mat = _build_test_material_shell(eps_dam=0.01)
        sig0 = np.array([[80.0, 40.0, 10.0]])
        deps0 = np.array([[0.015, 0.005, 0.005]])
        epsp_init = np.array([0.02])
        extra = {"epsp22": epsp_init.copy(), "off22": np.array([1.0])}
        h = 1e-7

        s_base, ep_base, _ = shell_update(mat, sig0.copy(), deps0.copy(), epsp=epsp_init.copy(), extra=extra)
        assert extra["dpla"][0] > 0.0, "Element must yield plastically"
        assert ep_base[0] > mat.params["eps_dam"]

        c_num = _compute_shell_numerical_tangent(mat, sig0, deps0, epsp_init=epsp_init, extra=extra, h=h)
        c_alg = consistent_shell_tangent(mat, s_base, epsp=ep_base, extra=extra)[0]

        diff_max = np.max(np.abs(c_num - c_alg))
        assert diff_max < 1e-4, f"Damaged yielding shell tangent diff {diff_max:.3e} exceeds 1e-4"

        flow_vec = _P_PLANE @ s_base[0, :3]
        flow_vec = flow_vec / np.linalg.norm(flow_vec)
        flow_err = np.linalg.norm((c_num - c_alg) @ flow_vec)
        assert flow_err < 1e-4, f"Damaged flow direction error {flow_err:.3e} exceeds 1e-4"

    def test_shell_failed_and_deleted_element(self):
        """Verify deleted or failed shell elements produce strictly zero tangent stiffness."""
        mat = _build_test_material_shell(eps_max=0.05)
        sig0 = np.array([[100.0, 50.0, 20.0], [100.0, 50.0, 20.0]])
        epsp0 = np.array([0.01, 0.06])  # element 1 exceeded eps_max
        extra = {"off22": np.array([0.0, 1.0])}  # element 0 deleted

        c_alg = consistent_shell_tangent(mat, sig0, epsp=epsp0, extra=extra)
        assert c_alg.shape == (2, 3, 3)
        np.testing.assert_allclose(c_alg[0], 0.0, atol=1e-14)
        np.testing.assert_allclose(c_alg[1], 0.0, atol=1e-14)

    def test_shell_batched_multi_element(self):
        """Verify vectorized batched evaluation matches serial evaluation across heterogeneous states."""
        mat = _build_test_material_shell(eps_dam=0.01, eps_max=0.08)
        n = 6

        sig_batch = np.array([
            [50.0, 20.0, 10.0],     # 0: undamaged elastic
            [30.0, 10.0, 5.0],      # 1: damaged elastic
            [100.0, 50.0, 20.0],    # 2: undamaged yielding
            [80.0, 40.0, 10.0],     # 3: damaged yielding
            [100.0, 50.0, 20.0],    # 4: deleted
            [100.0, 50.0, 20.0],    # 5: failed
        ])

        deps_batch = np.array([
            [0.0003, -0.0001, 0.0002],  # elastic
            [0.0001, -0.00005, 0.00008], # damaged elastic
            [0.003, 0.001, 0.0005],     # yielding
            [0.015, 0.005, 0.005],      # damaged yielding
            [0.003, 0.001, 0.0005],     # deleted
            [0.003, 0.001, 0.0005],     # failed
        ])

        epsp_batch = np.array([0.0, 0.02, 0.0, 0.02, 0.0, 0.09])
        off_batch = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0])

        extra_batch = {
            "epsp22": epsp_batch.copy(),
            "off22": off_batch.copy(),
        }

        s_base, ep_base, _ = shell_update(mat, sig_batch.copy(), deps_batch.copy(), epsp=epsp_batch.copy(), extra=extra_batch)

        c_batch = consistent_shell_tangent(mat, s_base, epsp=ep_base, extra=extra_batch)
        assert c_batch.shape == (n, 3, 3)

        # Serial comparison
        for i in range(n):
            ex_i = {
                "epsp22": np.array([epsp_batch[i]]),
                "off22": np.array([off_batch[i]]),
            }
            s_i, ep_i, _ = shell_update(mat, sig_batch[i:i+1].copy(), deps_batch[i:i+1].copy(), epsp=np.array([epsp_batch[i]]), extra=ex_i)
            c_i = consistent_shell_tangent(mat, s_i, epsp=ep_i, extra=ex_i)
            np.testing.assert_allclose(c_batch[i], c_i[0], atol=1e-12)

        # Elements 4 and 5 must be zero
        np.testing.assert_allclose(c_batch[4], 0.0, atol=1e-14)
        np.testing.assert_allclose(c_batch[5], 0.0, atol=1e-14)

    def test_shell_symmetric_flag(self):
        """Verify symmetric=True enforces major symmetry 0.5 * (C + C.T)."""
        mat = _build_test_material_shell()
        sig0 = np.array([[100.0, 50.0, 20.0]])
        deps0 = np.array([[0.003, 0.001, 0.0005]])
        extra = {}
        s_base, ep_base, _ = shell_update(mat, sig0.copy(), deps0.copy(), extra=extra)

        c_asym = consistent_shell_tangent(mat, s_base, extra=extra, symmetric=False)[0]
        c_sym = consistent_shell_tangent(mat, s_base, extra=extra, symmetric=True)[0]

        np.testing.assert_allclose(c_sym, 0.5 * (c_asym + c_asym.T), atol=1e-14)
        np.testing.assert_allclose(c_sym, c_sym.T, atol=1e-14)


# ============================================================================
# 2. Solid Tangent Tests (3D Solid, 6x6)
# ============================================================================

class TestSolidConsistentTangent:
    """Audit suite for consistent 3D solid tangent (6x6)."""

    def test_solid_elastic_single_element(self):
        """Verify undamaged elastic solid tangent matches numerical tangent < 1e-6."""
        mat = _build_test_material_solid()
        sig0 = np.array([[50.0, 20.0, 10.0, 5.0, 0.0, 0.0]])
        deps0 = np.array([[0.0002, -0.0001, 0.0001, 0.00015, 0.0, 0.0]])
        h = 1e-7

        extra = {}
        s_base, ep_base, _ = solid_update(mat, sig0.copy(), deps0.copy(), extra=extra)
        assert ep_base[0] == 0.0

        c_num = _compute_solid_numerical_tangent(mat, sig0, deps0, h=h)
        c_alg = consistent_solid_tangent(mat, s_base, extra=extra)[0]

        diff_max = np.max(np.abs(c_num - c_alg))
        assert diff_max < 1e-6, f"Elastic solid tangent diff {diff_max:.3e} exceeds 1e-6"

        c_el = solid_tangent(mat)
        np.testing.assert_allclose(c_alg, c_el, atol=1e-12)
        assert np.allclose(c_alg, c_alg.T, atol=1e-12)

    def test_solid_elastic_varying_states(self):
        """Verify elastic solid tangent across multiple 3D initial stress states."""
        mat = _build_test_material_solid()
        h = 1e-7

        test_states = [
            np.zeros((1, 6)),
            np.array([[120.0, -60.0, 30.0, -15.0, 20.0, -10.0]]),
            np.array([[200.0, 200.0, 200.0, 0.0, 0.0, 0.0]]),  # Hydrostatic
            np.array([[0.0, 0.0, 0.0, 40.0, -30.0, 20.0]]),     # Pure shear
        ]

        for sig0 in test_states:
            deps0 = np.array([[0.0001, -0.00005, 0.00008, 0.0001, -0.00005, 0.00005]])
            extra = {}
            s_base, _, _ = solid_update(mat, sig0.copy(), deps0.copy(), extra=extra)

            c_num = _compute_solid_numerical_tangent(mat, sig0, deps0, h=h)
            c_alg = consistent_solid_tangent(mat, s_base, extra=extra)[0]

            diff_max = np.max(np.abs(c_num - c_alg))
            assert diff_max < 1e-6, f"Varying elastic solid state diff {diff_max:.3e} exceeds 1e-6"

    @pytest.mark.parametrize("epsp_val", [0.02, 0.035, 0.05])
    def test_solid_damaged_elastic_regime(self, epsp_val: float):
        """Verify damaged elastic regime degrades shear modulus G_curr = alpe * G (< 1e-5)."""
        mat = _build_test_material_solid(eps_dam=0.015)
        sig0 = np.array([[40.0, 20.0, 10.0, 5.0, 0.0, 0.0]])
        deps0 = np.array([[0.0001, -0.00005, -0.00005, 0.00005, 0.0, 0.0]])
        epsp_init = np.array([epsp_val])
        extra = {"epsp22": epsp_init.copy(), "off22": np.array([1.0])}
        h = 1e-7

        s_base, ep_base, _ = solid_update(mat, sig0.copy(), deps0.copy(), epsp=epsp_init.copy(), extra=extra)
        assert ep_base[0] == pytest.approx(epsp_val)

        alpe = extra["alpe22"][0]
        assert alpe < 1.0, f"Expected degradation factor alpe < 1.0, got {alpe}"
        assert alpe > 0.0

        c_num = _compute_solid_numerical_tangent(mat, sig0, deps0, epsp_init=epsp_init, extra=extra, h=h)
        c_alg = consistent_solid_tangent(mat, s_base, epsp=ep_base, extra=extra)[0]

        diff_max = np.max(np.abs(c_num - c_alg))
        assert diff_max < 1e-5, f"Damaged elastic solid diff {diff_max:.3e} exceeds 1e-5 for epsp={epsp_val}"

        # Verify bulk modulus K is intact while shear modulus G is degraded by alpe
        K = mat.params["K"]
        Gi = alpe * mat.params["G"]
        ee = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        I_dev = np.diag([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.5, 0.5, 0.5])
        I_dev[0, 1] = I_dev[0, 2] = I_dev[1, 0] = I_dev[1, 2] = I_dev[2, 0] = I_dev[2, 1] = -1.0 / 3.0
        c_expected = K * np.outer(ee, ee) + 2.0 * Gi * I_dev
        np.testing.assert_allclose(c_alg, c_expected, atol=1e-10)

    def test_solid_plastic_yield_undamaged(self):
        """Verify 3D J2 plastic yield state consistency and directional derivative along flow direction."""
        mat = _build_test_material_solid(eps_dam=0.03)
        sig0 = np.array([[50.0, 20.0, 10.0, 5.0, 0.0, 0.0]])
        deps0 = np.array([[0.005, -0.001, -0.001, 0.002, 0.0, 0.0]])
        epsp_init = np.zeros(1)
        extra_init = {"epsp22": epsp_init.copy(), "off22": np.array([1.0])}
        extra_run = _copy_extra(extra_init)
        h = 1e-7

        s_base, ep_base, _ = solid_update(mat, sig0.copy(), deps0.copy(), epsp=epsp_init.copy(), extra=extra_run)
        assert extra_run["dpla"][0] > 0.0, "Solid must yield plastically"
        assert ep_base[0] < mat.params["eps_dam"]

        c_num = _compute_solid_numerical_tangent(mat, sig0, deps0, epsp_init=epsp_init, extra=extra_init, h=h)
        c_alg = consistent_solid_tangent(mat, s_base, epsp=ep_base, extra=extra_run)[0]

        diff_max = np.max(np.abs(c_num - c_alg))
        assert diff_max < 1e-4, f"Plastic solid tangent diff {diff_max:.3e} exceeds 1e-4"

        # Directional derivative along flow direction
        s_dev = s_base[0].copy()
        pm = (s_dev[0] + s_dev[1] + s_dev[2]) / 3.0
        s_dev[0:3] -= pm
        snorm = math.sqrt(s_dev[0]**2 + s_dev[1]**2 + s_dev[2]**2 + 2.0 * (s_dev[3]**2 + s_dev[4]**2 + s_dev[5]**2))
        Nv = s_dev / snorm
        flow_vec = np.array([Nv[0], Nv[1], Nv[2], 2.0 * Nv[3], 2.0 * Nv[4], 2.0 * Nv[5]])
        flow_vec = flow_vec / np.linalg.norm(flow_vec)

        flow_err = np.linalg.norm((c_num - c_alg) @ flow_vec)
        assert flow_err < 1e-4, f"Solid flow direction error {flow_err:.3e} exceeds 1e-4"

    def test_solid_plastic_yield_damaged(self):
        """Verify 3D J2 plastic yielding in damaged state (epseq >= eps_dam)."""
        mat = _build_test_material_solid(eps_dam=0.015)
        sig0 = np.array([[120.0, -80.0, 40.0, -15.0, 20.0, -10.0]])
        deps0 = np.array([[0.01, 0.005, -0.005, 0.01, 0.005, -0.005]])
        epsp_init = np.array([0.02])
        extra = {"epsp22": epsp_init.copy(), "off22": np.array([1.0])}
        h = 1e-7

        s_base, ep_base, _ = solid_update(mat, sig0.copy(), deps0.copy(), epsp=epsp_init.copy(), extra=extra)
        assert extra["dpla"][0] > 0.0, "Solid must yield plastically"
        assert ep_base[0] > mat.params["eps_dam"]

        c_num = _compute_solid_numerical_tangent(mat, sig0, deps0, epsp_init=epsp_init, extra=extra, h=h)
        c_alg = consistent_solid_tangent(mat, s_base, epsp=ep_base, extra=extra)[0]

        diff_max = np.max(np.abs(c_num - c_alg))
        assert diff_max < 1e-4, f"Damaged yielding solid tangent diff {diff_max:.3e} exceeds 1e-4"

        s_dev = s_base[0].copy()
        pm = (s_dev[0] + s_dev[1] + s_dev[2]) / 3.0
        s_dev[0:3] -= pm
        snorm = math.sqrt(s_dev[0]**2 + s_dev[1]**2 + s_dev[2]**2 + 2.0 * (s_dev[3]**2 + s_dev[4]**2 + s_dev[5]**2))
        Nv = s_dev / snorm
        flow_vec = np.array([Nv[0], Nv[1], Nv[2], 2.0 * Nv[3], 2.0 * Nv[4], 2.0 * Nv[5]])
        flow_vec = flow_vec / np.linalg.norm(flow_vec)

        flow_err = np.linalg.norm((c_num - c_alg) @ flow_vec)
        assert flow_err < 1e-4, f"Damaged solid flow direction error {flow_err:.3e} exceeds 1e-4"

    def test_solid_failed_and_deleted_element(self):
        """Verify deleted or failed solid elements produce strictly zero tangent stiffness."""
        mat = _build_test_material_solid(eps_max=0.08)
        sig0 = np.array([[50.0, 20.0, 10.0, 5.0, 0.0, 0.0], [50.0, 20.0, 10.0, 5.0, 0.0, 0.0]])
        epsp0 = np.array([0.02, 0.09])  # element 1 exceeded eps_max
        extra = {"off22": np.array([0.0, 1.0])}  # element 0 deleted

        c_alg = consistent_solid_tangent(mat, sig0, epsp=epsp0, extra=extra)
        assert c_alg.shape == (2, 6, 6)
        np.testing.assert_allclose(c_alg[0], 0.0, atol=1e-14)
        np.testing.assert_allclose(c_alg[1], 0.0, atol=1e-14)

    def test_solid_batched_multi_element(self):
        """Verify vectorized batched evaluation matches serial evaluation across heterogeneous states."""
        mat = _build_test_material_solid(eps_dam=0.015, eps_max=0.10)
        n = 6

        sig_batch = np.array([
            [50.0, 20.0, 10.0, 5.0, 0.0, 0.0],                   # 0: undamaged elastic
            [40.0, 20.0, 10.0, 5.0, 0.0, 0.0],                   # 1: damaged elastic
            [50.0, 20.0, 10.0, 5.0, 0.0, 0.0],                   # 2: undamaged yielding
            [120.0, -80.0, 40.0, -15.0, 20.0, -10.0],            # 3: damaged yielding
            [50.0, 20.0, 10.0, 5.0, 0.0, 0.0],                   # 4: deleted
            [50.0, 20.0, 10.0, 5.0, 0.0, 0.0],                   # 5: failed
        ])

        deps_batch = np.array([
            [0.0002, -0.0001, 0.0001, 0.00015, 0.0, 0.0],        # elastic
            [0.0001, -0.00005, -0.00005, 0.00005, 0.0, 0.0],     # damaged elastic
            [0.005, -0.001, -0.001, 0.002, 0.0, 0.0],           # yielding
            [0.01, 0.005, -0.005, 0.01, 0.005, -0.005],          # damaged yielding
            [0.005, -0.001, -0.001, 0.002, 0.0, 0.0],           # deleted
            [0.005, -0.001, -0.001, 0.002, 0.0, 0.0],           # failed
        ])

        epsp_batch = np.array([0.0, 0.025, 0.0, 0.025, 0.0, 0.12])
        off_batch = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0])

        extra_batch = {
            "epsp22": epsp_batch.copy(),
            "off22": off_batch.copy(),
        }

        s_base, ep_base, _ = solid_update(mat, sig_batch.copy(), deps_batch.copy(), epsp=epsp_batch.copy(), extra=extra_batch)

        c_batch = consistent_solid_tangent(mat, s_base, epsp=ep_base, extra=extra_batch)
        assert c_batch.shape == (n, 6, 6)

        # Serial comparison
        for i in range(n):
            ex_i = {
                "epsp22": np.array([epsp_batch[i]]),
                "off22": np.array([off_batch[i]]),
            }
            s_i, ep_i, _ = solid_update(mat, sig_batch[i:i+1].copy(), deps_batch[i:i+1].copy(), epsp=np.array([epsp_batch[i]]), extra=ex_i)
            c_i = consistent_solid_tangent(mat, s_i, epsp=ep_i, extra=ex_i)
            np.testing.assert_allclose(c_batch[i], c_i[0], atol=1e-12)

        # Elements 4 and 5 must be zero
        np.testing.assert_allclose(c_batch[4], 0.0, atol=1e-14)
        np.testing.assert_allclose(c_batch[5], 0.0, atol=1e-14)

    def test_solid_symmetric_flag(self):
        """Verify symmetric=True enforces major symmetry 0.5 * (C + C.T)."""
        mat = _build_test_material_solid()
        sig0 = np.array([[50.0, 20.0, 10.0, 5.0, 0.0, 0.0]])
        deps0 = np.array([[0.005, -0.001, -0.001, 0.002, 0.0, 0.0]])
        extra = {}
        s_base, ep_base, _ = solid_update(mat, sig0.copy(), deps0.copy(), extra=extra)

        c_asym = consistent_solid_tangent(mat, s_base, extra=extra, symmetric=False)[0]
        c_sym = consistent_solid_tangent(mat, s_base, extra=extra, symmetric=True)[0]

        np.testing.assert_allclose(c_sym, 0.5 * (c_asym + c_asym.T), atol=1e-14)
        np.testing.assert_allclose(c_sym, c_sym.T, atol=1e-14)


# ============================================================================
# 3. Robustness, 1D/Empty Arrays, and Package Dispatch
# ============================================================================

class TestRobustnessAndDispatch:
    """Audit edge cases, 1D array support, empty array support, and package dispatch."""

    def test_shell_1d_input_array(self):
        """Verify 1D input (3,) returns (3, 3) tangent tensor."""
        mat = _build_test_material_shell()
        sig_1d = np.array([50.0, 20.0, 10.0])
        c_1d = consistent_shell_tangent(mat, sig_1d)
        assert c_1d.shape == (3, 3)

        c_2d = consistent_shell_tangent(mat, sig_1d[None, :])
        assert c_2d.shape == (1, 3, 3)
        np.testing.assert_allclose(c_1d, c_2d[0])

    def test_solid_1d_input_array(self):
        """Verify 1D input (6,) returns (6, 6) tangent tensor."""
        mat = _build_test_material_solid()
        sig_1d = np.array([50.0, 20.0, 10.0, 5.0, 0.0, 0.0])
        c_1d = consistent_solid_tangent(mat, sig_1d)
        assert c_1d.shape == (6, 6)

        c_2d = consistent_solid_tangent(mat, sig_1d[None, :])
        assert c_2d.shape == (1, 6, 6)
        np.testing.assert_allclose(c_1d, c_2d[0])

    def test_empty_input_arrays(self):
        """Verify empty input arrays return (0, 3, 3) and (0, 6, 6) respectively."""
        mat_sh = _build_test_material_shell()
        c_sh_empty = consistent_shell_tangent(mat_sh, np.empty((0, 3)))
        assert c_sh_empty.shape == (0, 3, 3)

        mat_sol = _build_test_material_solid()
        c_sol_empty = consistent_solid_tangent(mat_sol, np.empty((0, 6)))
        assert c_sol_empty.shape == (0, 6, 6)

    def test_package_dispatch_shell(self):
        """Verify pyradioss.materials dispatch to consistent_shell_tangent."""
        mat = _build_test_material_shell()
        sig = np.array([[50.0, 20.0, 10.0]])
        c1 = materials.shell_layer_tangent(mat, sig)
        c2 = materials.consistent_shell_tangent(mat, sig)
        c_direct = consistent_shell_tangent(mat, sig)

        assert c1.shape == (1, 3, 3)
        np.testing.assert_allclose(c1, c_direct)
        np.testing.assert_allclose(c2, c_direct)

    def test_package_dispatch_solid(self):
        """Verify pyradioss.materials dispatch to consistent_solid_tangent."""
        mat = _build_test_material_solid()
        sig = np.array([[50.0, 20.0, 10.0, 5.0, 0.0, 0.0]])
        c1 = materials.solid_tangent(mat, sig)
        c2 = materials.consistent_solid_tangent(mat, sig)
        c_direct = consistent_solid_tangent(mat, sig)

        assert c1.shape == (1, 6, 6)
        np.testing.assert_allclose(c1, c_direct)
        np.testing.assert_allclose(c2, c_direct)
