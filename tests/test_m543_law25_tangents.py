"""
LAW25 Tangent Stiffness Tensor Audit Suite.
Milestone M543: Composite Anisotropic Plasticity and Damage Model (/MAT/LAW25).

Audits the algorithmic consistent tangent stiffness tensors in
pyradioss/materials/law25_composite.py:
  1. Elastic compliance inversion:
     Orthotropic compliance matrix S inverted analytically and compared against
     shell_membrane_tangent and solid_stiffness_matrix (within 1e-6).
  2. Damaged state:
     Degraded moduli under damage (d1, d2) and unilateral contact recovery in compression
     verified against central finite difference perturbation of shell_update and
     solid_update (within 1e-5).
  3. Plastic yield state:
     Consistent algorithmic elastoplastic tangent verified against central finite difference
     perturbation and directional consistency condition m^T * C_algo = 0.
  4. Multi-element vectorization:
     Identical evaluations between 1D (3, 3) / (6, 6), single-element (1, 3, 3) / (1, 6, 6),
     and batched (N, 3, 3) / (N, 6, 6) across mixed elastic, damaged, yielding, and failed states.
  5. Materials dispatcher:
     Integration with pyradioss.materials.consistent_shell_tangent and consistent_solid_tangent.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.materials import law25_composite
from pyradioss.model.entities import Material


# =============================================================================
# Helper Utilities
# =============================================================================

def _make_law25(
    e11: float = 120000.0,
    e22: float = 60000.0,
    e33: float = 40000.0,
    nu12: float = 0.25,
    g12: float = 20000.0,
    g23: float = 12000.0,
    g31: float = 15000.0,
    sigyt1: float = 800.0,
    sigyc1: float = 600.0,
    sigyt2: float = 100.0,
    sigyc2: float = 200.0,
    sigt12: float = 80.0,
    sigc12: float = 80.0,
    b: float = 0.0,
    n: float = 1.0,
    dmax: float = 0.999,
    epst1: float = 1e30,
    epsm1: float = 1e30,
    epst2: float = 1e30,
    epsm2: float = 1e30,
    rho0: float = 1.6e-9,
) -> Material:
    """Build a validated LAW25 Material instance for tangent auditing."""
    return law25_composite.build_law25(
        id=25,
        rho0=rho0,
        E1=e11,
        E2=e22,
        E3=e33,
        nu12=nu12,
        G12=g12,
        G23=g23,
        G31=g31,
        sigyt1=sigyt1,
        sigyc1=sigyc1,
        sigyt2=sigyt2,
        sigyc2=sigyc2,
        sigt12=sigt12,
        sigc12=sigc12,
        b=b,
        n=n,
        dmax=dmax,
        epst1=epst1,
        epsm1=epsm1,
        epst2=epst2,
        epsm2=epsm2,
    )


def _compute_numerical_shell_tangent(
    mat: Material,
    sig_init: np.ndarray,
    deps_base: np.ndarray,
    epsp_init: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    h: float = 1e-6,
) -> np.ndarray:
    """Compute (3, 3) numerical plane-stress tangent via central finite difference:

    C_num[:, j] = [sigma(deps_base + h * e_j) - sigma(deps_base - h * e_j)] / (2 * h)
    """
    c_num = np.zeros((3, 3), dtype=np.float64)
    base = np.asarray(deps_base, dtype=np.float64).flatten()

    for j in range(3):
        # Forward perturbation
        dp = base.copy()
        dp[j] += h
        sp_init = np.asarray(sig_init, dtype=np.float64).copy()
        ep_init = np.asarray(epsp_init, dtype=np.float64).copy() if epsp_init is not None else None
        ex_p = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in extra.items()} if extra is not None else None
        sp, _, _ = law25_composite.shell_update(mat, sp_init, dp, ep_init, dt=dt, extra=ex_p)

        # Backward perturbation
        dm = base.copy()
        dm[j] -= h
        sm_init = np.asarray(sig_init, dtype=np.float64).copy()
        em_init = np.asarray(epsp_init, dtype=np.float64).copy() if epsp_init is not None else None
        ex_m = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in extra.items()} if extra is not None else None
        sm, _, _ = law25_composite.shell_update(mat, sm_init, dm, em_init, dt=dt, extra=ex_m)

        s_plus = sp[0] if sp.ndim == 2 else sp
        s_minus = sm[0] if sm.ndim == 2 else sm
        c_num[:, j] = (s_plus[:3] - s_minus[:3]) / (2.0 * h)

    return c_num


def _compute_numerical_solid_tangent(
    mat: Material,
    sig_init: np.ndarray,
    deps_base: np.ndarray,
    epsp_init: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    h: float = 1e-6,
) -> np.ndarray:
    """Compute (6, 6) numerical 3D solid tangent via central finite difference:

    C_num[:, j] = [sigma(deps_base + h * e_j) - sigma(deps_base - h * e_j)] / (2 * h)
    """
    c_num = np.zeros((6, 6), dtype=np.float64)
    base = np.asarray(deps_base, dtype=np.float64).flatten()

    for j in range(6):
        dp = base.copy()
        dp[j] += h
        sp_init = np.asarray(sig_init, dtype=np.float64).copy()
        ep_init = np.asarray(epsp_init, dtype=np.float64).copy() if epsp_init is not None else None
        ex_p = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in extra.items()} if extra is not None else None
        sp, _, _ = law25_composite.solid_update(mat, sp_init, dp, ep_init, dt=dt, extra=ex_p)

        dm = base.copy()
        dm[j] -= h
        sm_init = np.asarray(sig_init, dtype=np.float64).copy()
        em_init = np.asarray(epsp_init, dtype=np.float64).copy() if epsp_init is not None else None
        ex_m = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in extra.items()} if extra is not None else None
        sm, _, _ = law25_composite.solid_update(mat, sm_init, dm, em_init, dt=dt, extra=ex_m)

        s_plus = sp[0] if sp.ndim == 2 else sp
        s_minus = sm[0] if sm.ndim == 2 else sm
        c_num[:, j] = (s_plus[:6] - s_minus[:6]) / (2.0 * h)

    return c_num


# =============================================================================
# 1. Elastic State Compliance Inversion
# =============================================================================

class TestElasticStateComplianceInversion:
    """Verify orthotropic compliance inversion match within 1e-6."""

    def test_shell_plane_stress_compliance_inversion(self):
        """Orthotropic plane-stress compliance S_inv matches shell_membrane_tangent within 1e-6."""
        e1, e2 = 140000.0, 70000.0
        nu12 = 0.28
        g12 = 28000.0
        nu21 = nu12 * e2 / e1

        # 3x3 plane-stress compliance
        s_mat = np.array([
            [1.0 / e1, -nu21 / e2, 0.0],
            [-nu12 / e1, 1.0 / e2, 0.0],
            [0.0, 0.0, 1.0 / g12],
        ], dtype=np.float64)

        c_expected = np.linalg.inv(s_mat)

        mat = _make_law25(e11=e1, e22=e2, nu12=nu12, g12=g12)
        c_shell = law25_composite.shell_membrane_tangent(mat)
        c_tan_1d = law25_composite.consistent_shell_tangent(mat, np.zeros(3))
        c_tan_2d = law25_composite.consistent_shell_tangent(mat, np.zeros((1, 3)))[0]

        max_diff_mem = np.max(np.abs(c_expected - c_shell))
        max_diff_1d = np.max(np.abs(c_expected - c_tan_1d))
        max_diff_2d = np.max(np.abs(c_expected - c_tan_2d))

        assert max_diff_mem < 1e-6, f"shell_membrane_tangent error {max_diff_mem:.3e} exceeds 1e-6"
        assert max_diff_1d < 1e-6, f"consistent_shell_tangent 1D error {max_diff_1d:.3e} exceeds 1e-6"
        assert max_diff_2d < 1e-6, f"consistent_shell_tangent 2D error {max_diff_2d:.3e} exceeds 1e-6"

    def test_solid_3d_orthotropic_compliance_inversion(self):
        """Orthotropic 3D solid compliance S_inv matches solid_stiffness_matrix within 1e-6."""
        e1, e2, e3 = 135000.0, 65000.0, 45000.0
        nu12 = 0.26
        g12, g23, g31 = 22000.0, 14000.0, 18000.0
        nu21 = nu12 * e2 / e1

        # 6x6 orthotropic compliance for LAW25 solid structure
        s_mat = np.zeros((6, 6), dtype=np.float64)
        s_mat[0, 0] = 1.0 / e1
        s_mat[0, 1] = -nu21 / e2
        s_mat[1, 0] = -nu12 / e1
        s_mat[1, 1] = 1.0 / e2
        s_mat[2, 2] = 1.0 / e3
        s_mat[3, 3] = 1.0 / g12
        s_mat[4, 4] = 1.0 / g23
        s_mat[5, 5] = 1.0 / g31

        c_expected = np.linalg.inv(s_mat)

        mat = _make_law25(e11=e1, e22=e2, e33=e3, nu12=nu12, g12=g12, g23=g23, g31=g31)
        c_solid = law25_composite.solid_stiffness_matrix(mat)
        c_tan_1d = law25_composite.consistent_solid_tangent(mat, np.zeros(6))
        c_tan_2d = law25_composite.consistent_solid_tangent(mat, np.zeros((1, 6)))[0]

        max_diff_sol = np.max(np.abs(c_expected - c_solid))
        max_diff_1d = np.max(np.abs(c_expected - c_tan_1d))
        max_diff_2d = np.max(np.abs(c_expected - c_tan_2d))

        assert max_diff_sol < 1e-6, f"solid_stiffness_matrix error {max_diff_sol:.3e} exceeds 1e-6"
        assert max_diff_1d < 1e-6, f"consistent_solid_tangent 1D error {max_diff_1d:.3e} exceeds 1e-6"
        assert max_diff_2d < 1e-6, f"consistent_solid_tangent 2D error {max_diff_2d:.3e} exceeds 1e-6"

    def test_elastic_numerical_fd_perturbation_match(self):
        """Numerical perturbation of update matches elastic tangent within 1e-6 relative error."""
        mat = _make_law25(e11=110000.0, e22=55000.0, nu12=0.2, g12=20000.0)
        deps_shell = np.array([0.0003, 0.0002, 0.00015])
        c_num_shell = _compute_numerical_shell_tangent(mat, np.zeros(3), deps_shell, h=1e-7)
        c_ana_shell = law25_composite.consistent_shell_tangent(mat, np.zeros(3))
        rel_err_sh = np.linalg.norm(c_num_shell - c_ana_shell) / np.linalg.norm(c_ana_shell)
        assert rel_err_sh < 1e-6, f"Shell elastic FD relative error {rel_err_sh:.3e} exceeds 1e-6"

        deps_solid = np.array([0.0002, 0.0001, 0.0003, 0.00015, 0.0001, 0.0002])
        c_num_solid = _compute_numerical_solid_tangent(mat, np.zeros(6), deps_solid, h=1e-7)
        c_ana_solid = law25_composite.consistent_solid_tangent(mat, np.zeros(6))
        rel_err_so = np.linalg.norm(c_num_solid - c_ana_solid) / np.linalg.norm(c_ana_solid)
        assert rel_err_so < 1e-6, f"Solid elastic FD relative error {rel_err_so:.3e} exceeds 1e-6"


# =============================================================================
# 2. Damaged State Verification
# =============================================================================

class TestDamagedStateTangents:
    """Verify damaged state stiffness match within 1e-5."""

    def test_shell_damaged_stiffness_fd_match(self):
        """Shell tangent under damage (d1=0.35, d2=0.45) matches numerical FD within 1e-5."""
        mat = _make_law25(
            e11=120000.0, e22=60000.0, nu12=0.22, g12=24000.0,
            sigyt1=5000.0, sigyc1=5000.0, sigyt2=5000.0, sigyc2=5000.0, sigt12=3000.0,
        )
        d1_val, d2_val = 0.35, 0.45
        extra = {"dmg25": np.array([[0.0, d1_val, d2_val, 0.0]])}
        sig_init = np.array([50.0, 30.0, 10.0])
        deps_base = np.array([0.0004, 0.0002, 0.0002])

        c_num = _compute_numerical_shell_tangent(mat, sig_init, deps_base, extra=extra, h=1e-7)
        c_ana = law25_composite.consistent_shell_tangent(mat, sig_init, extra=extra)

        rel_err = np.linalg.norm(c_num - c_ana) / np.linalg.norm(c_ana)
        assert rel_err < 1e-5, f"Damaged shell tangent relative error {rel_err:.3e} exceeds 1e-5"

        # Explicit check of degraded moduli
        de1 = 1.0 - d1_val
        de2 = 1.0 - d2_val
        assert c_ana[0, 0] == pytest.approx(120000.0 * de1, rel=1e-5)
        assert c_ana[1, 1] == pytest.approx(60000.0 * de2, rel=1e-5)
        assert c_ana[0, 1] == pytest.approx(0.0, abs=1e-8)  # scale1 = 0 when damaged
        assert c_ana[2, 2] == pytest.approx(de1 * de2 * 24000.0, rel=1e-5)

    def test_shell_unilateral_recovery_in_compression(self):
        """Under compression (sig < 0), intact elastic modulus is recovered."""
        mat = _make_law25(
            e11=100000.0, e22=50000.0, nu12=0.2, g12=20000.0,
            sigyt1=5000.0, sigyc1=5000.0, sigyt2=5000.0, sigyc2=5000.0, sigt12=3000.0,
        )
        extra = {"dmg25": np.array([[0.0, 0.5, 0.5, 0.0]])}
        # Negative stress state -> compression recovers intact moduli
        sig_comp = np.array([-100.0, -50.0, 0.0])
        c_comp = law25_composite.consistent_shell_tangent(mat, sig_comp, extra=extra)
        c_intact = law25_composite.shell_membrane_tangent(mat)

        np.testing.assert_allclose(c_comp, c_intact, rtol=1e-6)

    def test_solid_damaged_stiffness_fd_match(self):
        """Solid 6x6 tangent under damage matches numerical FD within 1e-5."""
        mat = _make_law25(
            e11=110000.0, e22=55000.0, e33=40000.0, nu12=0.25,
            g12=20000.0, g23=12000.0, g31=15000.0,
            sigyt1=5000.0, sigyc1=5000.0, sigyt2=5000.0, sigyc2=5000.0, sigt12=3000.0,
        )
        d1_val, d2_val = 0.25, 0.35
        extra = {"dmg25": np.array([[0.0, d1_val, d2_val, 0.0]])}
        sig_init = np.array([40.0, 20.0, 10.0, 5.0, 5.0, 5.0])
        deps_base = np.array([0.0003, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001])

        c_num = _compute_numerical_solid_tangent(mat, sig_init, deps_base, extra=extra, h=1e-7)
        c_ana = law25_composite.consistent_solid_tangent(mat, sig_init, extra=extra)

        rel_err = np.linalg.norm(c_num - c_ana) / np.linalg.norm(c_ana)
        assert rel_err < 1e-5, f"Damaged solid tangent relative error {rel_err:.3e} exceeds 1e-5"

        de1 = 1.0 - d1_val
        de2 = 1.0 - d2_val
        assert c_ana[0, 0] == pytest.approx(110000.0 * de1, rel=1e-5)
        assert c_ana[1, 1] == pytest.approx(55000.0 * de2, rel=1e-5)
        assert c_ana[2, 2] == pytest.approx(40000.0, rel=1e-5)
        assert c_ana[3, 3] == pytest.approx(de1 * de2 * 20000.0, rel=1e-5)
        assert c_ana[4, 4] == pytest.approx(de2 * 12000.0, rel=1e-5)
        assert c_ana[5, 5] == pytest.approx(de1 * 15000.0, rel=1e-5)

    def test_element_deletion_zero_stiffness(self):
        """Failed element (off=0) returns zero tangent stiffness."""
        mat = _make_law25()
        extra_failed = {"off25": np.zeros(1)}

        c_sh = law25_composite.consistent_shell_tangent(mat, np.zeros(3), extra=extra_failed)
        assert np.all(c_sh == 0.0)

        c_so = law25_composite.consistent_solid_tangent(mat, np.zeros(6), extra=extra_failed)
        assert np.all(c_so == 0.0)


# =============================================================================
# 3. Plastic Yield State Verification
# =============================================================================

class TestPlasticYieldStateTangents:
    """Verify plastic yield state tangents against finite difference and directional consistency."""

    def test_shell_plastic_yield_fd_match(self):
        """Shell algorithmic tangent matches central difference in plastic yield regime."""
        mat = _make_law25(
            e11=100000.0, e22=100000.0, nu12=0.0, g12=50000.0,
            sigyt1=200.0, sigyc1=200.0, sigyt2=200.0, sigyc2=200.0,
            sigt12=100.0, sigc12=100.0, b=0.0,
        )
        sig_init = np.zeros(3)
        deps_base = np.array([0.003, 0.001, 0.001])

        # Step produces plastic yield: trial s11 = 300 > 200
        s_mid, ep_mid, _ = law25_composite.shell_update(mat, sig_init, deps_base)
        assert ep_mid > 0.0

        c_num = _compute_numerical_shell_tangent(mat, sig_init, deps_base, h=1e-6)
        c_algo = law25_composite.consistent_shell_tangent(mat, sig_init, deps=deps_base)

        rel_err = np.linalg.norm(c_num - c_algo) / np.linalg.norm(c_algo)
        assert rel_err < 1e-4, f"Shell plastic yield FD error {rel_err:.3e} exceeds 1e-4"

    def test_shell_directional_consistency_tangent_to_yield_surface(self):
        """Yield surface normal m satisfies m^T * C_algo = 0 (directional consistency)."""
        mat = _make_law25(
            e11=120000.0, e22=60000.0, nu12=0.2, g12=25000.0,
            sigyt1=300.0, sigyc1=300.0, sigyt2=150.0, sigyc2=150.0,
            sigt12=80.0, sigc12=80.0, b=0.0,
        )
        deps = np.array([0.004, 0.002, 0.0015])
        s_out, _, _ = law25_composite.shell_update(mat, np.zeros(3), deps)

        c_algo = law25_composite.consistent_shell_tangent(mat, s_out)

        p = mat.params
        f1, f2, f11, f22, f33, f12 = p["F1"], p["F2"], p["F11"], p["F22"], p["F33"], p["F12"]
        m_grad = np.array([
            f1 + 2.0 * f11 * s_out[0] + 2.0 * f12 * s_out[1],
            f2 + 2.0 * f22 * s_out[1] + 2.0 * f12 * s_out[0],
            2.0 * f33 * s_out[2],
        ])

        normal_proj = m_grad @ c_algo
        norm_proj = np.linalg.norm(normal_proj)
        norm_c = np.linalg.norm(c_algo)
        norm_m = np.linalg.norm(m_grad)
        rel_proj = norm_proj / (norm_c * norm_m)

        assert rel_proj < 1e-6, f"Yield surface normal projection {rel_proj:.3e} exceeds 1e-6"

    def test_solid_plastic_yield_fd_match(self):
        """Solid 6x6 tangent matches central difference under multi-axial plastic yield."""
        mat = _make_law25(
            e11=100000.0, e22=100000.0, e33=80000.0, nu12=0.0,
            g12=50000.0, g23=40000.0, g31=40000.0,
            sigyt1=200.0, sigyc1=200.0, sigyt2=200.0, sigyc2=200.0,
            sigt12=100.0, sigc12=100.0, b=0.0,
        )
        sig_init = np.zeros(6)
        deps_base = np.array([0.003, 0.001, 0.0005, 0.001, 0.0002, 0.0002])

        s_mid, ep_mid, _ = law25_composite.solid_update(mat, sig_init, deps_base)
        assert ep_mid > 0.0

        c_num = _compute_numerical_solid_tangent(mat, sig_init, deps_base, h=1e-6)
        c_algo = law25_composite.consistent_solid_tangent(mat, sig_init, deps=deps_base)

        rel_err = np.linalg.norm(c_num - c_algo) / np.linalg.norm(c_algo)
        assert rel_err < 1e-4, f"Solid plastic yield FD error {rel_err:.3e} exceeds 1e-4"

        # Out-of-plane components remain elastic
        assert c_algo[2, 2] == pytest.approx(80000.0)
        assert c_algo[4, 4] == pytest.approx(40000.0)
        assert c_algo[5, 5] == pytest.approx(40000.0)

    def test_plastic_hardening_slope(self):
        """Nonlinear isotropic hardening (b=0.5, n=0.5) tangent matches numerical derivative."""
        mat = _make_law25(
            e11=100000.0, e22=100000.0, nu12=0.0, g12=50000.0,
            sigyt1=200.0, sigyc1=200.0, sigyt2=200.0, sigyc2=200.0,
            sigt12=100.0, sigc12=100.0, b=0.5, n=0.5,
        )
        extra = {"wpla25": np.array([0.01])}
        sig_init = np.zeros(3)
        deps_base = np.array([0.003, 0.001, 0.001])

        c_num = _compute_numerical_shell_tangent(mat, sig_init, deps_base, extra=extra, h=1e-6)
        c_algo = law25_composite.consistent_shell_tangent(mat, sig_init, extra=extra, deps=deps_base)

        rel_err = np.linalg.norm(c_num - c_algo) / np.linalg.norm(c_algo)
        assert rel_err < 1e-4, f"Hardening plastic tangent relative error {rel_err:.3e} exceeds 1e-4"


# =============================================================================
# 4. Vectorized Batches and Tensor Shapes
# =============================================================================

class TestVectorizedBatchesAndShapes:
    """Verify single-element (1D and 2D) and vectorized batches for shells and solids."""

    def test_shell_single_element_1d_and_2d(self):
        """Shell tangent produces identical values for 1D (3,) and 2D (1, 3) inputs."""
        mat = _make_law25()
        sig_1d = np.array([50.0, 20.0, 10.0])
        sig_2d = sig_1d[None, :]

        c_1d = law25_composite.consistent_shell_tangent(mat, sig_1d)
        c_2d = law25_composite.consistent_shell_tangent(mat, sig_2d)

        assert c_1d.shape == (3, 3)
        assert c_2d.shape == (1, 3, 3)
        np.testing.assert_allclose(c_1d, c_2d[0], rtol=1e-12)

    def test_solid_single_element_1d_and_2d(self):
        """Solid tangent produces identical values for 1D (6,) and 2D (1, 6) inputs."""
        mat = _make_law25()
        sig_1d = np.array([50.0, 20.0, 15.0, 10.0, 5.0, 5.0])
        sig_2d = sig_1d[None, :]

        c_1d = law25_composite.consistent_solid_tangent(mat, sig_1d)
        c_2d = law25_composite.consistent_solid_tangent(mat, sig_2d)

        assert c_1d.shape == (6, 6)
        assert c_2d.shape == (1, 6, 6)
        np.testing.assert_allclose(c_1d, c_2d[0], rtol=1e-12)

    def test_shell_vectorized_batch_mixed_states(self):
        """Batch of N=16 mixed elastic, damaged, yielding, and failed shell elements."""
        mat = _make_law25(
            e11=100000.0, e22=50000.0, nu12=0.2, g12=20000.0,
            sigyt1=200.0, sigyc1=200.0, sigyt2=100.0, sigyc2=100.0, sigt12=60.0,
        )
        n = 16
        sig_batch = np.zeros((n, 3))
        epsp_batch = np.zeros(n)
        epsp_incr = np.zeros(n)
        dmg_batch = np.zeros((n, 4))
        off_batch = np.ones(n)

        # 0..3: Elastic
        sig_batch[0:4] = np.array([50.0, 20.0, 10.0])

        # 4..7: Damaged
        sig_batch[4:8] = np.array([40.0, 15.0, 5.0])
        dmg_batch[4:8, 1] = 0.3
        dmg_batch[4:8, 2] = 0.4

        # 8..11: Yielding
        sig_batch[8:12] = np.array([200.0, 0.0, 0.0])  # sits on yield surface
        epsp_incr[8:12] = 0.001

        # 12..15: Failed
        off_batch[12:16] = 0.0

        extra_batch = {
            "dmg25": dmg_batch,
            "off25": off_batch,
        }

        c_batch = law25_composite.consistent_shell_tangent(
            mat, sig_batch, epsp=epsp_batch, epsp_incr=epsp_incr, extra=extra_batch
        )
        assert c_batch.shape == (n, 3, 3)

        # Verify against individual evaluations
        for i in range(n):
            ex_i = {
                "dmg25": dmg_batch[i:i+1],
                "off25": off_batch[i:i+1],
            }
            c_single = law25_composite.consistent_shell_tangent(
                mat, sig_batch[i:i+1], epsp=epsp_batch[i:i+1],
                epsp_incr=epsp_incr[i:i+1], extra=ex_i
            )[0]
            np.testing.assert_allclose(c_batch[i], c_single, rtol=1e-12, atol=1e-12)

    def test_solid_vectorized_batch_mixed_states(self):
        """Batch of N=16 mixed elastic, damaged, yielding, and failed solid elements."""
        mat = _make_law25(
            e11=100000.0, e22=50000.0, e33=40000.0, nu12=0.2,
            g12=20000.0, g23=12000.0, g31=15000.0,
            sigyt1=200.0, sigyc1=200.0, sigyt2=100.0, sigyc2=100.0, sigt12=60.0,
        )
        n = 16
        sig_batch = np.zeros((n, 6))
        epsp_batch = np.zeros(n)
        epsp_incr = np.zeros(n)
        dmg_batch = np.zeros((n, 4))
        off_batch = np.ones(n)

        # 0..3: Elastic
        sig_batch[0:4] = np.array([40.0, 20.0, 15.0, 5.0, 5.0, 5.0])

        # 4..7: Damaged
        sig_batch[4:8] = np.array([30.0, 15.0, 10.0, 5.0, 5.0, 5.0])
        dmg_batch[4:8, 1] = 0.2
        dmg_batch[4:8, 2] = 0.35

        # 8..11: Yielding
        sig_batch[8:12] = np.array([200.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        epsp_incr[8:12] = 0.001

        # 12..15: Failed
        off_batch[12:16] = 0.0

        extra_batch = {
            "dmg25": dmg_batch,
            "off25": off_batch,
        }

        c_batch = law25_composite.consistent_solid_tangent(
            mat, sig_batch, epsp=epsp_batch, epsp_incr=epsp_incr, extra=extra_batch
        )
        assert c_batch.shape == (n, 6, 6)

        for i in range(n):
            ex_i = {
                "dmg25": dmg_batch[i:i+1],
                "off25": off_batch[i:i+1],
            }
            c_single = law25_composite.consistent_solid_tangent(
                mat, sig_batch[i:i+1], epsp=epsp_batch[i:i+1],
                epsp_incr=epsp_incr[i:i+1], extra=ex_i
            )[0]
            np.testing.assert_allclose(c_batch[i], c_single, rtol=1e-12, atol=1e-12)

    def test_symmetric_flag_enforcement(self):
        """symmetric=True strictly enforces C_ij == C_ji for non-symmetric algorithmic tangent."""
        mat = _make_law25(
            e11=100000.0, e22=100000.0, nu12=0.0, g12=50000.0,
            sigyt1=200.0, sigyc1=200.0, sigyt2=200.0, sigyc2=200.0,
            sigt12=100.0, sigc12=100.0, b=0.0,
        )
        deps = np.array([0.003, 0.001, 0.001])
        c_unsym = law25_composite.consistent_shell_tangent(mat, np.zeros(3), deps=deps, symmetric=False)
        c_sym = law25_composite.consistent_shell_tangent(mat, np.zeros(3), deps=deps, symmetric=True)

        assert not np.allclose(c_unsym, c_unsym.T)
        np.testing.assert_allclose(c_sym, c_sym.T, atol=1e-15)


# =============================================================================
# 5. Materials Package Dispatcher
# =============================================================================

class TestMaterialsPackageDispatcher:
    """Verify pyradioss.materials dispatching to law25_composite."""

    def test_shell_tangent_dispatcher(self):
        """materials.consistent_shell_tangent routes to LAW25."""
        mat = _make_law25()
        sig = np.zeros((2, 3))
        # Test without optional epsp, epsp_incr
        c_disp = materials.consistent_shell_tangent(mat, sig)
        c_dir = law25_composite.consistent_shell_tangent(mat, sig)
        np.testing.assert_allclose(c_disp, c_dir, rtol=1e-12)

        # Test with explicit epsp, epsp_incr
        epsp = np.zeros(2)
        epsp_incr = np.zeros(2)
        c_disp_full = materials.consistent_shell_tangent(mat, sig, epsp, epsp_incr)
        np.testing.assert_allclose(c_disp_full, c_dir, rtol=1e-12)

    def test_solid_tangent_dispatcher(self):
        """materials.consistent_solid_tangent routes to LAW25."""
        mat = _make_law25()
        sig = np.zeros((2, 6))
        # Test without optional epsp, epsp_incr
        c_disp = materials.consistent_solid_tangent(mat, sig)
        c_dir = law25_composite.consistent_solid_tangent(mat, sig)
        np.testing.assert_allclose(c_disp, c_dir, rtol=1e-12)

        # Test with explicit epsp, epsp_incr
        epsp = np.zeros(2)
        epsp_incr = np.zeros(2)
        c_disp_full = materials.consistent_solid_tangent(mat, sig, epsp, epsp_incr)
        np.testing.assert_allclose(c_disp_full, c_dir, rtol=1e-12)
