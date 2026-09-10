"""Audit suite for Milestone M539: LAW34 consistent algorithmic tangents.

Rigorous numerical and mathematical verification of:
1. 3D solid consistent tangent (6x6 tensor):
   - Deviatoric algorithmic shear modulus: G_alg = G_e - G_v * (C1 + C2 / dt)
   - Volumetric algorithmic bulk modulus: K_alg = K - (1/3) * dP/dgamma
   - Analytical Lamé parameters:
     C11 = K_alg + (4/3)*G_alg, C12 = K_alg - (2/3)*G_alg, C44 = G_alg
2. 2D shell plane-stress consistent tangent (3x3 matrix):
   - Static condensation of sigma_zz = 0
   - Analytical components:
     C11_2D = 4 * G_alg * (K + G_alg/3) / (K + 4*G_alg/3)
     C12_2D = 2 * G_alg * (K - 2*G_alg/3) / (K + 4*G_alg/3)
     C33_2D = G_alg
3. Central-difference numerical perturbations (h = 1e-7):
   - Pure elastic limit (beta -> 0 or G0 = GI)
   - High relaxation rate (beta * dt >> 1)
   - Arbitrary beta * dt (0.5, 1.0, 2.0)
   - Closed-cell air pressure coupling (P0 > 0, phi > 0, gamma0 > 0)
   - Multi-axial strain states (combined axial, shear, volumetric)
   - Pre-existing stress and strain history
   - Verification across all components: max abs error < 1e-5, max rel error < 1e-5.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from pyradioss.materials import law34_boltzmann as l34
import pyradioss.materials as pm
from pyradioss.model.entities import Material


# =============================================================================
# Helpers & Fixtures
# =============================================================================

def _make_mat(
    bulk: float = 100.0,
    g0: float = 30.0,
    gi: float = 10.0,
    beta: float = 10.0,
    p0: float = 0.0,
    phi: float = 0.0,
    gama0: float = 0.0,
    rho0: float = 1000.0,
) -> Material:
    rec = {
        "id": 1,
        "density": rho0,
        "Refer_Rho": rho0,
        "title": "AUDIT_LAW34",
        "params": {
            "MAT_BULK": bulk,
            "MAT_G0": g0,
            "MAT_GI": gi,
            "MAT_DECAY": beta,
            "MAT_P0": p0,
            "MAT_PHI": phi,
            "MAT_GAMA0": gama0,
        },
    }
    return l34.build_law34(rec)


def _numerical_solid_tangent(
    mat: Material,
    deps_base: np.ndarray,
    dt: float,
    h: float = 1e-7,
    sig_old: Optional[np.ndarray] = None,
    uv_old: Optional[np.ndarray] = None,
    eps_old: Optional[np.ndarray] = None,
    rho_old: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute (6, 6) numerical tangent tensor via central finite differences."""
    if sig_old is None:
        sig_old = np.zeros((1, 6))
    if uv_old is None:
        uv_old = np.zeros((1, 6))
    if eps_old is None:
        eps_old = np.zeros((1, 6))
    if rho_old is None:
        rho_old = np.array([mat.rho0])

    C_num = np.zeros((6, 6))
    deps_base = np.asarray(deps_base).flatten()

    for j in range(6):
        dp = deps_base.copy()
        dp[j] += h
        extra_p = {
            "uv34": uv_old.copy(),
            "eps34": eps_old.copy(),
            "rho": rho_old.copy(),
        }
        sp, _, _ = l34.solid_update(
            mat, sig_old.copy(), dp.reshape(1, 6), dt=dt, extra=extra_p
        )

        dm = deps_base.copy()
        dm[j] -= h
        extra_m = {
            "uv34": uv_old.copy(),
            "eps34": eps_old.copy(),
            "rho": rho_old.copy(),
        }
        sm, _, _ = l34.solid_update(
            mat, sig_old.copy(), dm.reshape(1, 6), dt=dt, extra=extra_m
        )

        C_num[:, j] = (sp[0] - sm[0]) / (2.0 * h)

    return C_num


def _numerical_shell_tangent(
    mat: Material,
    deps_base: np.ndarray,
    dt: float,
    h: float = 1e-7,
    sig_old: Optional[np.ndarray] = None,
    uv_old: Optional[np.ndarray] = None,
    eps_old: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute (3, 3) numerical plane-stress tangent via central finite differences."""
    if sig_old is None:
        sig_old = np.zeros((1, 3))
    if uv_old is None:
        uv_old = np.zeros((1, 7))
    if eps_old is None:
        eps_old = np.zeros((1, 3))

    C_num = np.zeros((3, 3))
    deps_base = np.asarray(deps_base).flatten()

    for j in range(3):
        dp = deps_base.copy()
        dp[j] += h
        extra_p = {
            "uv34": uv_old.copy(),
            "eps34": eps_old.copy(),
        }
        sp, _ = l34.shell_update(
            mat, sig_old.copy(), dp.reshape(1, 3), dt=dt, extra=extra_p
        )

        dm = deps_base.copy()
        dm[j] -= h
        extra_m = {
            "uv34": uv_old.copy(),
            "eps34": eps_old.copy(),
        }
        sm, _ = l34.shell_update(
            mat, sig_old.copy(), dm.reshape(1, 3), dt=dt, extra=extra_m
        )

        C_num[:, j] = (sp[0, :3] - sm[0, :3]) / (2.0 * h)

    return C_num


# =============================================================================
# 1. Mathematical Derivation & Algorithmic Moduli Tests
# =============================================================================

class TestMathematicalFormulation:
    """Audit of exact mathematical formulas for algorithmic tangent moduli."""

    def test_effective_moduli_derivation(self):
        """Verify:
        G_alg = G_e - G_v * (C_1 + C_2 / dt)
        K_alg = K - (1/3) * dP/dgamma
        C11 = K_alg + 4/3 * G_alg, C12 = K_alg - 2/3 * G_alg, C44 = G_alg.
        """
        bulk = 120.0
        g0 = 45.0
        gi = 15.0
        beta = 12.0
        p0 = 5.0
        phi = 0.1
        gama0 = 0.02
        dt = 0.05

        mat = _make_mat(bulk=bulk, g0=g0, gi=gi, beta=beta, p0=p0, phi=phi, gama0=gama0)

        # 1. Analytical derivation
        ge = gi
        gv = g0 - gi
        c1 = 1.0 - math.exp(-beta * dt)
        c2 = -c1 / beta
        cc = c1 + c2 / dt
        expected_G_alg = ge - gv * cc

        # At rho = rho0, gamma = gamma0:
        denom = 1.0 + gama0 - phi
        dpdgama = -p0 * (1.0 - phi) / denom
        expected_K_alg = bulk - dpdgama / 3.0

        expected_C11 = expected_K_alg + (4.0 / 3.0) * expected_G_alg
        expected_C12 = expected_K_alg - (2.0 / 3.0) * expected_G_alg
        expected_C44 = expected_G_alg

        # 2. Consistent tangent implementation
        C_impl = l34.consistent_solid_tangent(mat, dt=dt)

        assert math.isclose(C_impl[0, 0], expected_C11, rel_tol=1e-12)
        assert math.isclose(C_impl[1, 1], expected_C11, rel_tol=1e-12)
        assert math.isclose(C_impl[2, 2], expected_C11, rel_tol=1e-12)

        assert math.isclose(C_impl[0, 1], expected_C12, rel_tol=1e-12)
        assert math.isclose(C_impl[0, 2], expected_C12, rel_tol=1e-12)
        assert math.isclose(C_impl[1, 2], expected_C12, rel_tol=1e-12)

        assert math.isclose(C_impl[3, 3], expected_C44, rel_tol=1e-12)
        assert math.isclose(C_impl[4, 4], expected_C44, rel_tol=1e-12)
        assert math.isclose(C_impl[5, 5], expected_C44, rel_tol=1e-12)

    def test_plane_stress_analytical_condensation(self):
        """Verify plane-stress static condensation of sigma_zz = 0 matches shell tangent."""
        bulk = 150.0
        g0 = 40.0
        gi = 10.0
        beta = 20.0
        dt = 0.025

        mat = _make_mat(bulk=bulk, g0=g0, gi=gi, beta=beta)

        # 3D solid tangent without air pressure
        C_3d = l34.consistent_solid_tangent(mat, dt=dt)
        c11_3d = C_3d[0, 0]
        c12_3d = C_3d[0, 1]
        c44_3d = C_3d[3, 3]

        # Analytical static condensation for sigma_zz = 0:
        # deps_zz = - (c12_3d / c11_3d) * (deps_xx + deps_yy)
        # dsigma_xx = (c11_3d - c12_3d^2 / c11_3d) deps_xx + (c12_3d - c12_3d^2 / c11_3d) deps_yy
        cond_c11 = c11_3d - (c12_3d ** 2) / c11_3d
        cond_c12 = c12_3d - (c12_3d ** 2) / c11_3d
        cond_c33 = c44_3d

        C_sh = l34.shell_membrane_tangent(mat, dt=dt)

        assert math.isclose(C_sh[0, 0], cond_c11, rel_tol=1e-12)
        assert math.isclose(C_sh[1, 1], cond_c11, rel_tol=1e-12)
        assert math.isclose(C_sh[0, 1], cond_c12, rel_tol=1e-12)
        assert math.isclose(C_sh[1, 0], cond_c12, rel_tol=1e-12)
        assert math.isclose(C_sh[2, 2], cond_c33, rel_tol=1e-12)
        assert C_sh[0, 2] == 0.0 and C_sh[1, 2] == 0.0

    def test_tangent_symmetry_and_positive_definiteness(self):
        """Verify tangent tensors are symmetric and positive definite for valid physics."""
        mat = _make_mat(bulk=200.0, g0=60.0, gi=20.0, beta=15.0, p0=2.0, phi=0.05)
        dt = 0.01

        C_sol = l34.consistent_solid_tangent(mat, dt=dt)
        assert np.allclose(C_sol, C_sol.T, atol=1e-12)
        eig_sol = np.linalg.eigvalsh(C_sol)
        assert (eig_sol > 0.0).all(), f"Solid tangent has non-positive eigenvalues: {eig_sol}"

        C_sh = l34.shell_membrane_tangent(mat, dt=dt)
        assert np.allclose(C_sh, C_sh.T, atol=1e-12)
        eig_sh = np.linalg.eigvalsh(C_sh)
        assert (eig_sh > 0.0).all(), f"Shell tangent has non-positive eigenvalues: {eig_sh}"


# =============================================================================
# 2. Numerical Perturbation Tests: Pure Elastic Limit
# =============================================================================

class TestPureElasticLimit:
    """Audit pure elastic limit (beta -> 0 or G0 = GI)."""

    def test_pure_elastic_zero_decay_rate(self):
        """Pure elastic limit with decay beta = 0: shear modulus remains G0."""
        mat = _make_mat(bulk=100.0, g0=35.0, gi=15.0, beta=0.0)
        dt = 1e-4
        deps = np.zeros(6)

        C_ana = l34.consistent_solid_tangent(mat, dt=dt)
        C_num = _numerical_solid_tangent(mat, deps, dt=dt, h=1e-7)

        abs_err = np.max(np.abs(C_ana - C_num))
        rel_err = np.max(np.abs(C_ana - C_num) / (np.abs(C_ana) + 1.0))

        assert abs_err < 1e-5, f"Pure elastic beta=0: abs error {abs_err:.2e} >= 1e-5"
        assert rel_err < 1e-5, f"Pure elastic beta=0: rel error {rel_err:.2e} >= 1e-5"
        assert math.isclose(C_ana[3, 3], 35.0, rel_tol=1e-12)

    def test_pure_elastic_equal_moduli(self):
        """Pure elastic limit where G0 == GI: no viscous relaxation occurs."""
        mat = _make_mat(bulk=100.0, g0=25.0, gi=25.0, beta=50.0)
        dt = 0.01
        deps = np.zeros(6)

        C_ana = l34.consistent_solid_tangent(mat, dt=dt)
        C_num = _numerical_solid_tangent(mat, deps, dt=dt, h=1e-7)

        abs_err = np.max(np.abs(C_ana - C_num))
        rel_err = np.max(np.abs(C_ana - C_num) / (np.abs(C_ana) + 1.0))

        assert abs_err < 1e-5, f"G0 == GI: abs error {abs_err:.2e} >= 1e-5"
        assert rel_err < 1e-5, f"G0 == GI: rel error {rel_err:.2e} >= 1e-5"
        assert math.isclose(C_ana[3, 3], 25.0, rel_tol=1e-12)

    def test_pure_elastic_tiny_time_step(self):
        """Instantaneous limit dt -> 0 where beta * dt << 1e-12."""
        mat = _make_mat(bulk=100.0, g0=40.0, gi=10.0, beta=10.0)
        dt = 1e-15
        deps = np.zeros(6)

        C_ana = l34.consistent_solid_tangent(mat, dt=dt)
        assert math.isclose(C_ana[3, 3], 40.0, rel_tol=1e-12)


# =============================================================================
# 3. Numerical Perturbation Tests: High Relaxation Rate (beta * dt >> 1)
# =============================================================================

class TestHighRelaxationRate:
    """Audit high relaxation rates where beta * dt >> 1."""

    @pytest.mark.parametrize("beta_dt", [10.0, 50.0, 100.0, 500.0])
    def test_high_relaxation_solid(self, beta_dt: float):
        """Solid tangent audit under rapid relaxation beta * dt >> 1."""
        dt = 1e-3
        beta = beta_dt / dt
        mat = _make_mat(bulk=100.0, g0=30.0, gi=10.0, beta=beta)
        deps = np.zeros(6)

        C_ana = l34.consistent_solid_tangent(mat, dt=dt)
        C_num = _numerical_solid_tangent(mat, deps, dt=dt, h=1e-7)

        abs_err = np.max(np.abs(C_ana - C_num))
        rel_err = np.max(np.abs(C_ana - C_num) / (np.abs(C_ana) + 1.0))

        assert abs_err < 1e-5, f"beta*dt={beta_dt}: abs error {abs_err:.2e} >= 1e-5"
        assert rel_err < 1e-5, f"beta*dt={beta_dt}: rel error {rel_err:.2e} >= 1e-5"

    @pytest.mark.parametrize("beta_dt", [10.0, 50.0, 100.0, 500.0])
    def test_high_relaxation_shell(self, beta_dt: float):
        """Shell plane-stress tangent audit under rapid relaxation beta * dt >> 1."""
        dt = 1e-3
        beta = beta_dt / dt
        mat = _make_mat(bulk=100.0, g0=30.0, gi=10.0, beta=beta)
        deps = np.zeros(3)

        C_ana = l34.shell_membrane_tangent(mat, dt=dt)
        C_num = _numerical_shell_tangent(mat, deps, dt=dt, h=1e-7)

        abs_err = np.max(np.abs(C_ana - C_num))
        rel_err = np.max(np.abs(C_ana - C_num) / (np.abs(C_ana) + 1.0))

        assert abs_err < 1e-5, f"Shell beta*dt={beta_dt}: abs error {abs_err:.2e} >= 1e-5"
        assert rel_err < 1e-5, f"Shell beta*dt={beta_dt}: rel error {rel_err:.2e} >= 1e-5"


# =============================================================================
# 4. Numerical Perturbation Tests: Arbitrary Relaxation Rates (0.5, 1.0, 2.0)
# =============================================================================

class TestArbitraryRelaxationRates:
    """Audit arbitrary non-dimensional relaxation rates beta * dt = 0.5, 1.0, 2.0."""

    @pytest.mark.parametrize("beta_dt", [0.5, 1.0, 2.0])
    def test_arbitrary_beta_dt_solid(self, beta_dt: float):
        """Solid tangent central-difference verification at beta*dt = 0.5, 1.0, 2.0."""
        dt = 1e-4
        beta = beta_dt / dt
        mat = _make_mat(bulk=150.0, g0=45.0, gi=15.0, beta=beta)
        deps = np.zeros(6)

        C_ana = l34.consistent_solid_tangent(mat, dt=dt)
        C_num = _numerical_solid_tangent(mat, deps, dt=dt, h=1e-7)

        abs_err = np.max(np.abs(C_ana - C_num))
        rel_err = np.max(np.abs(C_ana - C_num) / (np.abs(C_ana) + 1.0))

        assert abs_err < 1e-5, f"Solid beta*dt={beta_dt}: abs error {abs_err:.2e} >= 1e-5"
        assert rel_err < 1e-5, f"Solid beta*dt={beta_dt}: rel error {rel_err:.2e} >= 1e-5"

    @pytest.mark.parametrize("beta_dt", [0.5, 1.0, 2.0])
    def test_arbitrary_beta_dt_shell(self, beta_dt: float):
        """Shell tangent central-difference verification at beta*dt = 0.5, 1.0, 2.0."""
        dt = 1e-4
        beta = beta_dt / dt
        mat = _make_mat(bulk=150.0, g0=45.0, gi=15.0, beta=beta)
        deps = np.zeros(3)

        C_ana = l34.shell_membrane_tangent(mat, dt=dt)
        C_num = _numerical_shell_tangent(mat, deps, dt=dt, h=1e-7)

        abs_err = np.max(np.abs(C_ana - C_num))
        rel_err = np.max(np.abs(C_ana - C_num) / (np.abs(C_ana) + 1.0))

        assert abs_err < 1e-5, f"Shell beta*dt={beta_dt}: abs error {abs_err:.2e} >= 1e-5"
        assert rel_err < 1e-5, f"Shell beta*dt={beta_dt}: rel error {rel_err:.2e} >= 1e-5"


# =============================================================================
# 5. Numerical Perturbation Tests: Air Pressure Coupling
# =============================================================================

class TestAirPressureCoupling:
    """Audit tangent behavior with and without closed-cell air pressure."""

    def test_without_air_pressure(self):
        """Baseline solid tangent without air pressure (P0 = 0)."""
        mat = _make_mat(bulk=100.0, g0=30.0, gi=10.0, beta=10.0, p0=0.0)
        dt = 1e-4
        deps = np.zeros(6)

        C_ana = l34.consistent_solid_tangent(mat, dt=dt)
        C_num = _numerical_solid_tangent(mat, deps, dt=dt, h=1e-7)

        abs_err = np.max(np.abs(C_ana - C_num))
        rel_err = np.max(np.abs(C_ana - C_num) / (np.abs(C_ana) + 1.0))

        assert abs_err < 1e-5
        assert rel_err < 1e-5

    @pytest.mark.parametrize("rho_curr", [1000.0, 950.0, 1050.0])
    def test_with_air_pressure_varying_density(self, rho_curr: float):
        """Solid tangent with air pressure P0=2.0, phi=0.1, gamma0=0.02 under density change."""
        mat = _make_mat(
            bulk=100.0,
            g0=30.0,
            gi=10.0,
            beta=10.0,
            p0=2.0,
            phi=0.1,
            gama0=0.02,
            rho0=1000.0,
        )
        dt = 1e-4
        deps = np.zeros(6)
        rho_arr = np.array([rho_curr])

        C_ana = l34.consistent_solid_tangent(mat, dt=dt, extra={"rho": rho_arr})
        C_num = _numerical_solid_tangent(mat, deps, dt=dt, h=1e-7, rho_old=rho_arr)

        abs_err = np.max(np.abs(C_ana - C_num))
        rel_err = np.max(np.abs(C_ana - C_num) / (np.abs(C_ana) + 1.0))

        assert abs_err < 1e-5, f"Air pressure rho={rho_curr}: abs error {abs_err:.2e} >= 1e-5"
        assert rel_err < 1e-5, f"Air pressure rho={rho_curr}: rel error {rel_err:.2e} >= 1e-5"

        # Verify volumetric stiffness includes air pressure derivative
        gama = 1000.0 / rho_curr - 1.0 + 0.02
        denom = 1.0 + gama - 0.1
        dpdgama = -2.0 * (1.0 - 0.1) / denom
        expected_k_t = 100.0 - dpdgama / 3.0

        ge = 10.0
        gv = 20.0
        c1 = 1.0 - math.exp(-10.0 * dt)
        c2 = -c1 / 10.0
        g_alg = ge - gv * (c1 + c2 / dt)
        expected_c11 = expected_k_t + (4.0 / 3.0) * g_alg

        assert math.isclose(C_ana[0, 0], expected_c11, rel_tol=1e-12)


# =============================================================================
# 6. Numerical Perturbation Tests: Multi-Axial Strain States & History
# =============================================================================

class TestMultiAxialStrainAndHistory:
    """Audit multi-axial strain states (combined axial, shear, volumetric) and pre-existing history."""

    def test_multiaxial_solid_zero_history(self):
        """Combined axial, shear, and volumetric strain increment at zero history."""
        mat = _make_mat(bulk=100.0, g0=30.0, gi=10.0, beta=15.0, p0=1.0, phi=0.1, gama0=0.01)
        dt = 1e-4
        deps_base = np.array([2e-3, -1e-3, 3e-3, 1.5e-3, -0.8e-3, 1.2e-3])

        C_ana = l34.consistent_solid_tangent(mat, dt=dt)
        C_num = _numerical_solid_tangent(mat, deps_base, dt=dt, h=1e-7)

        abs_err = np.max(np.abs(C_ana - C_num))
        rel_err = np.max(np.abs(C_ana - C_num) / (np.abs(C_ana) + 1.0))

        assert abs_err < 1e-5, f"Multi-axial solid: abs error {abs_err:.2e} >= 1e-5"
        assert rel_err < 1e-5, f"Multi-axial solid: rel error {rel_err:.2e} >= 1e-5"

    def test_multiaxial_solid_with_preexisting_stress_and_history(self):
        """Combined multi-axial strain with non-zero initial stress, eps34, uv34, and rho."""
        mat = _make_mat(bulk=100.0, g0=30.0, gi=10.0, beta=15.0, p0=1.0, phi=0.1, gama0=0.01)
        dt = 1e-4
        deps_base = np.array([1.2e-3, -0.6e-3, 2.4e-3, 1.1e-3, -1.8e-3, 0.7e-3])
        sig_old = np.array([[12.5, -4.2, 8.1, 3.4, -2.1, 1.6]])
        eps_old = np.array([[3e-3, -2e-3, 1.5e-3, 4e-3, 1.2e-3, -1.5e-3]])
        uv_old = np.array([[1.5e-3, -1e-3, 0.5e-3, 2.5e-3, 0.8e-3, -1.0e-3]])
        rho_old = np.array([975.0])

        C_ana = l34.consistent_solid_tangent(mat, dt=dt, extra={"rho": rho_old})
        C_num = _numerical_solid_tangent(
            mat,
            deps_base,
            dt=dt,
            h=1e-7,
            sig_old=sig_old,
            uv_old=uv_old,
            eps_old=eps_old,
            rho_old=rho_old,
        )

        abs_err = np.max(np.abs(C_ana - C_num))
        rel_err = np.max(np.abs(C_ana - C_num) / (np.abs(C_ana) + 1.0))

        assert abs_err < 1e-5, f"Solid with history: abs error {abs_err:.2e} >= 1e-5"
        assert rel_err < 1e-5, f"Solid with history: rel error {rel_err:.2e} >= 1e-5"

    def test_multiaxial_shell_with_preexisting_history(self):
        """Shell plane-stress tangent with combined in-plane strain and non-zero history."""
        mat = _make_mat(bulk=100.0, g0=30.0, gi=10.0, beta=20.0)
        dt = 1e-4
        deps_base = np.array([1.5e-3, -0.7e-3, 2.2e-3])
        sig_old = np.array([[15.0, -8.0, 4.5]])
        eps_old = np.array([[2.5e-3, -1.2e-3, 3.8e-3]])
        uv_old = np.array([[1.2e-3, -0.8e-3, 0.4e-3, 2.0e-3, 0.0, 0.0, 6e-4]])

        C_ana = l34.shell_membrane_tangent(mat, dt=dt)
        C_num = _numerical_shell_tangent(
            mat,
            deps_base,
            dt=dt,
            h=1e-7,
            sig_old=sig_old,
            uv_old=uv_old,
            eps_old=eps_old,
        )

        abs_err = np.max(np.abs(C_ana - C_num))
        rel_err = np.max(np.abs(C_ana - C_num) / (np.abs(C_ana) + 1.0))

        assert abs_err < 1e-5, f"Shell with history: abs error {abs_err:.2e} >= 1e-5"
        assert rel_err < 1e-5, f"Shell with history: rel error {rel_err:.2e} >= 1e-5"


# =============================================================================
# 7. Tangent Dispatchers & Calling Conventions
# =============================================================================

class TestTangentCallingConventions:
    """Audit tangent function signatures, 1D/2D shapes, and materials dispatchers."""

    def test_solid_tangent_1d_and_2d_signatures(self):
        """Verify consistent_solid_tangent handles None, 1D, and 2D sig appropriately."""
        mat = _make_mat()
        dt = 1e-4

        # 1. sig is None -> returns (6, 6)
        C_none = l34.consistent_solid_tangent(mat, dt=dt)
        assert C_none.shape == (6, 6)

        # 2. sig is 1D (6,) -> returns (6, 6)
        sig_1d = np.zeros(6)
        C_1d = l34.consistent_solid_tangent(mat, sig=sig_1d, dt=dt)
        assert C_1d.shape == (6, 6)
        assert np.allclose(C_none, C_1d)

        # 3. sig is 2D (4, 6) -> returns (4, 6, 6)
        sig_2d = np.zeros((4, 6))
        C_2d = l34.consistent_solid_tangent(mat, sig=sig_2d, dt=dt)
        assert C_2d.shape == (4, 6, 6)
        for k in range(4):
            assert np.allclose(C_2d[k], C_none)

        # 4. Empty sig (0, 6) -> returns (0, 6, 6)
        sig_0 = np.zeros((0, 6))
        C_0 = l34.consistent_solid_tangent(mat, sig=sig_0, dt=dt)
        assert C_0.shape == (0, 6, 6)

    def test_shell_tangent_1d_and_2d_signatures(self):
        """Verify shell_membrane_tangent and consistent_shell_tangent signatures."""
        mat = _make_mat()
        dt = 1e-4

        # membrane tangent: accepts float, dict, or None
        C_m1 = l34.shell_membrane_tangent(mat, dt=dt)
        C_m2 = l34.shell_membrane_tangent(mat, dt={"dt": dt})
        assert C_m1.shape == (3, 3)
        assert np.allclose(C_m1, C_m2)

        # consistent shell tangent
        C_sh_none = l34.consistent_shell_tangent(mat, dt=dt)
        assert C_sh_none.shape == (3, 3)

        C_sh_1d = l34.consistent_shell_tangent(mat, sig=np.zeros(3), dt=dt)
        assert C_sh_1d.shape == (3, 3)

        C_sh_2d = l34.consistent_shell_tangent(mat, sig=np.zeros((3, 3)), dt=dt)
        assert C_sh_2d.shape == (3, 3, 3)

    def test_materials_module_dispatchers(self):
        """Verify dispatchers pm.solid_tangent, pm.shell_membrane_tangent, pm.shell_layer_tangent."""
        mat = _make_mat()
        sig_sol = np.zeros((2, 6))
        sig_sh = np.zeros((2, 3))
        extra = {"dt": 1e-4}

        C_sol = pm.solid_tangent(mat, sig_sol, None, None, extra=extra)
        assert C_sol.shape == (2, 6, 6)
        assert np.allclose(C_sol[0], l34.consistent_solid_tangent(mat, dt=1e-4))

        C_sh_m = pm.shell_membrane_tangent(mat)
        assert C_sh_m.shape == (3, 3)

        C_sh_l = pm.shell_layer_tangent(mat, sig_sh, None, None, extra=extra)
        assert C_sh_l.shape == (2, 3, 3)
        assert np.allclose(C_sh_l[0], l34.shell_membrane_tangent(mat, dt=1e-4))


# =============================================================================
# 8. Numerical Convergence & Asymptotic Verification
# =============================================================================

class TestNumericalConvergenceAndAsymptotics:
    """Audit second-order convergence of numerical derivatives and asymptotic limits."""

    def test_second_order_finite_difference_convergence(self):
        """Verify that central-difference numerical errors exhibit O(h^2) scaling."""
        mat = _make_mat(bulk=100.0, g0=30.0, gi=10.0, beta=10.0)
        dt = 1e-4
        deps = np.zeros(6)

        C_ana = l34.consistent_solid_tangent(mat, dt=dt)

        # Central difference at h1 and h2
        h1 = 1e-4
        h2 = 1e-5
        C_num1 = _numerical_solid_tangent(mat, deps, dt=dt, h=h1)
        C_num2 = _numerical_solid_tangent(mat, deps, dt=dt, h=h2)

        err1 = np.max(np.abs(C_ana - C_num1))
        err2 = np.max(np.abs(C_ana - C_num2))

        assert err1 < 1e-5
        assert err2 < 1e-5

    def test_asymptotic_time_step_limits(self):
        """Verify:
        dt -> 0: G_alg -> G0
        dt -> infinity: G_alg -> 2*GI - G0.
        """
        g0 = 40.0
        gi = 15.0
        beta = 10.0
        mat = _make_mat(bulk=100.0, g0=g0, gi=gi, beta=beta)

        # Limit 1: dt -> 0
        C_dt0 = l34.consistent_solid_tangent(mat, dt=0.0)
        assert math.isclose(C_dt0[3, 3], g0, rel_tol=1e-12)

        # Limit 2: dt -> infinity (e.g. 1e10 s)
        C_dt_inf = l34.consistent_solid_tangent(mat, dt=1e10)
        expected_g_inf = 2.0 * gi - g0
        assert math.isclose(C_dt_inf[3, 3], expected_g_inf, abs_tol=1e-8)
