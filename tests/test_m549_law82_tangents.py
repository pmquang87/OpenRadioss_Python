"""
Exhaustive Algorithmic Tangent Stiffness Verification Suite for LAW82 (/MAT/LAW82, /MAT/OGDEN).

Audits consistent_solid_tangent (n, 6, 6) and consistent_shell_tangent (n, 3, 3)
from pyradioss/materials/law82_ogden.py across:
1. Numerical central finite difference perturbation test:
   - Step sizes h in [1e-6, 1e-7]
   - C_num_ij = (sigma_i(eps + h * e_j) - sigma_i(eps - h * e_j)) / (2 * h)
   - Relative error ||C_alg - C_num|| / ||C_num|| < 1e-4
2. Multiple deformation states:
   - Ground state (eps = 0): exact recovery of isotropic elasticity tensor
     C_ijkl = lambda * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
     and 2D plane stress isotropic elasticity tensor
   - Moderate tensile strain (eps_11 = 0.1, 0.2)
   - Large tensile strain (eps_11 = 0.5, 1.0)
   - Compressive strain (eps_11 = -0.1, -0.3)
   - Shear strain (gamma_12 = 0.1, 0.2, 0.3, 0.6)
   - Multi-axial combined strain states
   - Pure volumetric expansion (J = 1.1) and compaction (J = 0.9)
3. Symmetry checks:
   - Exact major symmetry at ground state and pure volumetric states
   - Approximate symmetry C_ij ~= C_ji under moderate deformation
   - Nonlinear continuum mechanics identity: C_12 - C_21 = sigma_yy - sigma_xx, etc.
4. Positive-definiteness:
   - Eigenvalues of C and symmetric part 0.5 * (C + C.T) strictly positive (v^T C v > 0)
5. Multi-term models:
   - N = 1, 2, 3, 5 Ogden series
6. Batch vectorization:
   - (n, 6, 6) for solids and (n, 3, 3) for shells across element batches
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.materials.law82_ogden import (
    OgdenParams,
    build_law82,
    solid_update,
    shell_update,
    consistent_solid_tangent,
    consistent_shell_tangent,
    solid_tangent,
    shell_tangent,
    law82_consistent_solid_tangent,
    law82_consistent_shell_tangent,
)


# ============================================================================
# Helper Functions: Independent Central Finite Difference Numerical Tangents
# ============================================================================


def compute_numerical_solid_tangent(
    mat: OgdenParams,
    eps: np.ndarray,
    h: float = 1e-6,
    dt: float = 1e-3,
    ismstr: int = 0,
) -> np.ndarray:
    """Compute numerical tangent (n, 6, 6) by central finite difference perturbation.

    C_num_ij = (sigma_i(eps + h * e_j) - sigma_i(eps - h * e_j)) / (2 * h)
    """
    eps_arr = np.asarray(eps, dtype=np.float64)
    if eps_arr.ndim == 1:
        eps_arr = eps_arr.reshape(1, -1)
    n = eps_arr.shape[0]

    C_num = np.zeros((n, 6, 6), dtype=np.float64)
    dummy_sig = np.zeros((n, 6), dtype=np.float64)
    dummy_deps = np.zeros((n, 6), dtype=np.float64)

    for j in range(6):
        ep = eps_arr.copy()
        ep[:, j] += h
        em = eps_arr.copy()
        em[:, j] -= h

        sp, _ = solid_update(mat, dummy_sig, dummy_deps, ep, dt=dt, ismstr=ismstr)
        sm, _ = solid_update(mat, dummy_sig, dummy_deps, em, dt=dt, ismstr=ismstr)

        C_num[:, :, j] = (sp - sm) / (2.0 * h)

    return C_num


def compute_numerical_shell_tangent(
    mat: OgdenParams,
    eps: np.ndarray,
    h: float = 1e-6,
    dt: float = 1e-3,
    ismstr: int = 0,
) -> np.ndarray:
    """Compute numerical tangent (n, 3, 3) by central finite difference perturbation.

    C_num_ij = (sigma_i(eps + h * e_j) - sigma_i(eps - h * e_j)) / (2 * h)
    """
    eps_arr = np.asarray(eps, dtype=np.float64)
    if eps_arr.ndim == 1:
        eps_arr = eps_arr.reshape(1, -1)
    n = eps_arr.shape[0]

    C_num = np.zeros((n, 3, 3), dtype=np.float64)
    dummy_sig = np.zeros((n, 3), dtype=np.float64)
    dummy_deps = np.zeros((n, 3), dtype=np.float64)

    for j in range(3):
        ep = eps_arr.copy()
        ep[:, j] += h
        em = eps_arr.copy()
        em[:, j] -= h

        sp, _ = shell_update(mat, dummy_sig, dummy_deps, ep, dt=dt, ismstr=ismstr)
        sm, _ = shell_update(mat, dummy_sig, dummy_deps, em, dt=dt, ismstr=ismstr)

        C_num[:, :, j] = (sp[:, :3] - sm[:, :3]) / (2.0 * h)

    return C_num


def relative_tangent_error(C_alg: np.ndarray, C_num: np.ndarray) -> float:
    """Compute Frobenius relative error ||C_alg - C_num|| / ||C_num||."""
    norm_num = np.linalg.norm(C_num)
    if norm_num < 1e-12:
        return float(np.linalg.norm(C_alg - C_num))
    return float(np.linalg.norm(C_alg - C_num) / norm_num)


# ============================================================================
# 1. Ground State Tangents: Exact Linear Isotropic Elasticity Recovery
# ============================================================================


class TestGroundStateTangents:
    """Verify ground state (eps = 0) recovery of classical isotropic elasticity tensors."""

    @pytest.mark.parametrize("nu", [0.3, 0.45, 0.48, 0.495])
    @pytest.mark.parametrize("g0", [1.0, 10.0, 50.0])
    def test_solid_ground_state_exact_recovery(self, nu: float, g0: float):
        """Solid tangent at eps=0 must match isotropic elasticity tensor:

        C_ijkl = lambda * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
        In 6x6 Voigt:
            C_11 = C_22 = C_33 = lambda + 2 * mu
            C_12 = C_13 = C_23 = lambda
            C_44 = C_55 = C_66 = mu
            All other components = 0.
        """
        mat = build_law82(nordre=1, mu=[g0], alpha=[2.0], nu=nu)
        eps_zero = np.zeros((1, 6))

        C_alg = consistent_solid_tangent(mat, eps_zero)[0]

        mu = g0
        lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)

        C_expected = np.zeros((6, 6))
        # Normal components
        C_expected[0, 0] = C_expected[1, 1] = C_expected[2, 2] = lam + 2.0 * mu
        C_expected[0, 1] = C_expected[1, 0] = lam
        C_expected[0, 2] = C_expected[2, 0] = lam
        C_expected[1, 2] = C_expected[2, 1] = lam
        # Shear components
        C_expected[3, 3] = C_expected[4, 4] = C_expected[5, 5] = mu

        assert np.allclose(C_alg, C_expected, rtol=1e-5, atol=1e-6)
        # Exact major symmetry
        assert np.allclose(C_alg, C_alg.T, atol=1e-8)

        # Spectral validation: 1 volumetric mode (3*K = 3*lambda + 2*mu), 5 isochoric modes (2*mu)
        eigs = np.sort(np.linalg.eigvalsh(C_alg))
        K_bulk = lam + (2.0 / 3.0) * mu
        expected_vol_eig = 3.0 * K_bulk
        assert np.all(eigs > 0.0)
        assert np.isclose(eigs[-1], expected_vol_eig, rtol=1e-5)

    @pytest.mark.parametrize("nu", [0.3, 0.45, 0.48, 0.495])
    @pytest.mark.parametrize("g0", [1.0, 10.0, 50.0])
    def test_shell_ground_state_exact_recovery(self, nu: float, g0: float):
        """Shell tangent at eps=0 must match plane stress isotropic elasticity tensor:

        C_11 = C_22 = E / (1 - nu^2)
        C_12 = C_21 = nu * E / (1 - nu^2)
        C_33 = G
        All other components = 0.
        """
        mat = build_law82(nordre=1, mu=[g0], alpha=[2.0], nu=nu)
        eps_zero = np.zeros((1, 3))

        C_alg = consistent_shell_tangent(mat, eps_zero)[0]

        E_mod = 2.0 * g0 * (1.0 + nu)
        c11_plane = E_mod / (1.0 - nu ** 2)
        c12_plane = nu * c11_plane
        c33_plane = g0

        C_expected = np.array([
            [c11_plane, c12_plane, 0.0],
            [c12_plane, c11_plane, 0.0],
            [0.0,       0.0,       c33_plane],
        ])

        assert np.allclose(C_alg, C_expected, rtol=1e-5, atol=1e-6)
        assert np.allclose(C_alg, C_alg.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(C_alg) > 0.0)


# ============================================================================
# 2. Numerical Central Finite Difference Verification (Solid Continuum)
# ============================================================================


class TestSolidFiniteDifferenceTangents:
    """Verify consistent_solid_tangent against central FD across deformation states."""

    @pytest.fixture
    def ogden_mat(self) -> OgdenParams:
        """2-term Ogden rubber material."""
        return build_law82(nordre=2, mu=[10.0, 2.0], alpha=[2.0, -2.0], nu=0.45)

    @pytest.mark.parametrize("h", [1e-6, 1e-7])
    def test_solid_ground_state_fd(self, ogden_mat: OgdenParams, h: float):
        """Ground state eps = 0 central FD test."""
        eps = np.zeros(6)
        C_alg = consistent_solid_tangent(ogden_mat, eps, h=h)[0]
        C_num = compute_numerical_solid_tangent(ogden_mat, eps, h=h)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("eps_11", [0.1, 0.2])
    @pytest.mark.parametrize("h", [1e-6, 5e-7])
    def test_solid_moderate_tensile_strain_fd(self, ogden_mat: OgdenParams, eps_11: float, h: float):
        """Moderate tensile strain states eps_11 in [0.1, 0.2] with Poisson contraction."""
        nu = ogden_mat.nu
        eps = np.array([eps_11, -nu * eps_11, -nu * eps_11, 0.0, 0.0, 0.0])

        C_alg = consistent_solid_tangent(ogden_mat, eps, h=h)[0]
        C_num = compute_numerical_solid_tangent(ogden_mat, eps, h=h)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("eps_11", [0.5, 1.0])
    @pytest.mark.parametrize("h", [1e-6, 5e-7])
    def test_solid_large_tensile_strain_fd(self, ogden_mat: OgdenParams, eps_11: float, h: float):
        """Large tensile strain states eps_11 in [0.5, 1.0]."""
        nu = ogden_mat.nu
        eps = np.array([eps_11, -nu * eps_11, -nu * eps_11, 0.0, 0.0, 0.0])

        C_alg = consistent_solid_tangent(ogden_mat, eps, h=h)[0]
        C_num = compute_numerical_solid_tangent(ogden_mat, eps, h=h)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("eps_11", [-0.1, -0.3])
    @pytest.mark.parametrize("h", [1e-6, 5e-7])
    def test_solid_compressive_strain_fd(self, ogden_mat: OgdenParams, eps_11: float, h: float):
        """Compressive strain states eps_11 in [-0.1, -0.3]."""
        nu = ogden_mat.nu
        eps = np.array([eps_11, -nu * eps_11, -nu * eps_11, 0.0, 0.0, 0.0])

        C_alg = consistent_solid_tangent(ogden_mat, eps, h=h)[0]
        C_num = compute_numerical_solid_tangent(ogden_mat, eps, h=h)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("gamma_12", [0.1, 0.2, 0.3, 0.6])
    @pytest.mark.parametrize("h", [1e-6, 5e-7])
    def test_solid_shear_strain_fd(self, ogden_mat: OgdenParams, gamma_12: float, h: float):
        """Pure shear strain states gamma_12 in [0.1, 0.2, 0.3, 0.6]."""
        eps = np.array([0.0, 0.0, 0.0, gamma_12, 0.0, 0.0])

        C_alg = consistent_solid_tangent(ogden_mat, eps, h=h)[0]
        C_num = compute_numerical_solid_tangent(ogden_mat, eps, h=h)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("h", [1e-6, 5e-7])
    def test_solid_multiaxial_combined_state_fd(self, ogden_mat: OgdenParams, h: float):
        """Combined 3D multi-axial strain state."""
        eps = np.array([0.15, -0.05, 0.02, 0.08, -0.04, 0.03])

        C_alg = consistent_solid_tangent(ogden_mat, eps, h=h)[0]
        C_num = compute_numerical_solid_tangent(ogden_mat, eps, h=h)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("J", [1.1, 0.9])
    @pytest.mark.parametrize("h", [1e-6, 5e-7])
    def test_solid_pure_volumetric_fd(self, ogden_mat: OgdenParams, J: float, h: float):
        """Pure volumetric expansion (J = 1.1) and compaction (J = 0.9)."""
        e_vol = np.log(J) / 3.0
        eps = np.array([e_vol, e_vol, e_vol, 0.0, 0.0, 0.0])

        C_alg = consistent_solid_tangent(ogden_mat, eps, h=h)[0]
        C_num = compute_numerical_solid_tangent(ogden_mat, eps, h=h)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4


# ============================================================================
# 3. Numerical Central Finite Difference Verification (Shell Plane Stress)
# ============================================================================


class TestShellFiniteDifferenceTangents:
    """Verify consistent_shell_tangent against central FD across deformation states."""

    @pytest.fixture
    def ogden_mat(self) -> OgdenParams:
        """2-term Ogden rubber material."""
        return build_law82(nordre=2, mu=[10.0, 2.0], alpha=[2.0, -2.0], nu=0.45)

    @pytest.mark.parametrize("h", [1e-6, 1e-7])
    def test_shell_ground_state_fd(self, ogden_mat: OgdenParams, h: float):
        """Ground state eps = 0 central FD test."""
        eps = np.zeros(3)
        C_alg = consistent_shell_tangent(ogden_mat, eps, h=h)[0]
        C_num = compute_numerical_shell_tangent(ogden_mat, eps, h=h)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("eps_11", [0.1, 0.2])
    @pytest.mark.parametrize("h", [1e-6, 5e-7])
    def test_shell_moderate_tensile_strain_fd(self, ogden_mat: OgdenParams, eps_11: float, h: float):
        """Moderate in-plane tensile strain states."""
        eps = np.array([eps_11, -ogden_mat.nu * eps_11, 0.0])

        C_alg = consistent_shell_tangent(ogden_mat, eps, h=h)[0]
        C_num = compute_numerical_shell_tangent(ogden_mat, eps, h=h)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("eps_11", [-0.1, -0.3])
    @pytest.mark.parametrize("h", [1e-6, 5e-7])
    def test_shell_compressive_strain_fd(self, ogden_mat: OgdenParams, eps_11: float, h: float):
        """Compressive in-plane strain states."""
        eps = np.array([eps_11, -ogden_mat.nu * eps_11, 0.0])

        C_alg = consistent_shell_tangent(ogden_mat, eps, h=h)[0]
        C_num = compute_numerical_shell_tangent(ogden_mat, eps, h=h)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("gamma_12", [0.1, 0.2, 0.3, 0.6])
    @pytest.mark.parametrize("h", [1e-6, 5e-7])
    def test_shell_shear_strain_fd(self, ogden_mat: OgdenParams, gamma_12: float, h: float):
        """In-plane shear strain states gamma_12 in [0.1, 0.2, 0.3, 0.6]."""
        eps = np.array([0.0, 0.0, gamma_12])

        C_alg = consistent_shell_tangent(ogden_mat, eps, h=h)[0]
        C_num = compute_numerical_shell_tangent(ogden_mat, eps, h=h)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("h", [1e-6, 5e-7])
    def test_shell_multiaxial_combined_state_fd(self, ogden_mat: OgdenParams, h: float):
        """Combined in-plane multi-axial strain state."""
        eps = np.array([0.15, -0.05, 0.08])

        C_alg = consistent_shell_tangent(ogden_mat, eps, h=h)[0]
        C_num = compute_numerical_shell_tangent(ogden_mat, eps, h=h)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("h", [1e-6, 5e-7])
    def test_shell_large_tensile_strain_fd(self, ogden_mat: OgdenParams, h: float):
        """Large in-plane tensile strain eps_11 = 0.5 with Poisson contraction."""
        eps = np.array([0.5, -0.225, 0.0])

        C_alg = consistent_shell_tangent(ogden_mat, eps, h=h)[0]
        C_num = compute_numerical_shell_tangent(ogden_mat, eps, h=h)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4


# ============================================================================
# 4. Symmetry Verification & Continuum Mechanics Stress Identity
# ============================================================================


class TestTangentSymmetry:
    """Verify major symmetry properties and exact continuum stress-tangent identities."""

    @pytest.fixture
    def ogden_mat(self) -> OgdenParams:
        return build_law82(nordre=2, mu=[10.0, 2.0], alpha=[2.0, -2.0], nu=0.45)

    def test_solid_ground_state_symmetry(self, ogden_mat: OgdenParams):
        """Exact major symmetry C = C.T at ground state."""
        C = consistent_solid_tangent(ogden_mat, np.zeros(6))[0]
        asym = np.max(np.abs(C - C.T))
        assert asym < 1e-8

    def test_shell_ground_state_symmetry(self, ogden_mat: OgdenParams):
        """Exact major symmetry C = C.T at ground state for shells."""
        C = consistent_shell_tangent(ogden_mat, np.zeros(3))[0]
        asym = np.max(np.abs(C - C.T))
        assert asym < 1e-8

    @pytest.mark.parametrize("J", [1.1, 0.9])
    def test_solid_volumetric_exact_symmetry(self, ogden_mat: OgdenParams, J: float):
        """Exact symmetry C = C.T under pure volumetric deformation (spherical stress)."""
        e_vol = np.log(J) / 3.0
        eps = np.array([e_vol, e_vol, e_vol, 0.0, 0.0, 0.0])
        C = consistent_solid_tangent(ogden_mat, eps)[0]
        asym = np.max(np.abs(C - C.T))
        assert asym < 1e-8

    @pytest.mark.parametrize("eps", [
        np.array([0.05, -0.02, 0.01, 0.03, -0.01, 0.02]),
        np.array([0.15, -0.05, 0.02, 0.08, -0.04, 0.03]),
        np.array([0.3, -0.1, -0.1, 0.0, 0.0, 0.0]),
        np.array([-0.2, 0.1, 0.1, 0.05, 0.0, 0.0]),
    ])
    def test_solid_continuum_mechanics_tangent_asymmetry_identity(
        self, ogden_mat: OgdenParams, eps: np.ndarray
    ):
        """In finite elasticity, Cauchy stress derivative w.r.t logarithmic strain satisfies:

            C_12 - C_21 = sigma_22 - sigma_11
            C_13 - C_31 = sigma_33 - sigma_11
            C_23 - C_32 = sigma_33 - sigma_22
        """
        C = consistent_solid_tangent(ogden_mat, eps)[0]
        sig, _ = solid_update(ogden_mat, np.zeros((1, 6)), None, eps.reshape(1, 6))
        sig = sig[0]

        # C_12 - C_21 == sigma_22 - sigma_11
        assert np.isclose(C[0, 1] - C[1, 0], sig[1] - sig[0], rtol=1e-4, atol=1e-5)
        # C_13 - C_31 == sigma_33 - sigma_11
        assert np.isclose(C[0, 2] - C[2, 0], sig[2] - sig[0], rtol=1e-4, atol=1e-5)
        # C_23 - C_32 == sigma_33 - sigma_22
        assert np.isclose(C[1, 2] - C[2, 1], sig[2] - sig[1], rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("eps", [
        np.array([0.05, -0.02, 0.03]),
        np.array([0.15, -0.05, 0.08]),
        np.array([-0.1, 0.045, 0.02]),
    ])
    def test_shell_tangent_approximate_symmetry(self, ogden_mat: OgdenParams, eps: np.ndarray):
        """Shell tangent exhibits approximate symmetry C_ij ~= C_ji under moderate deformation."""
        C = consistent_shell_tangent(ogden_mat, eps)[0]
        norm_C = np.linalg.norm(C)
        asym = np.linalg.norm(C - C.T) / norm_C
        # For moderate strains in shells, relative asymmetry is well within 15%
        assert asym < 0.15


# ============================================================================
# 5. Positive-Definiteness Verification (Stability v^T C v > 0)
# ============================================================================


class TestPositiveDefiniteness:
    """Verify positive definiteness of algorithmic tangents across deformation states."""

    @pytest.fixture
    def ogden_mat(self) -> OgdenParams:
        return build_law82(nordre=2, mu=[10.0, 2.0], alpha=[2.0, -2.0], nu=0.45)

    @pytest.mark.parametrize("eps", [
        np.zeros(6),
        np.array([0.1, -0.045, -0.045, 0.0, 0.0, 0.0]),
        np.array([0.2, -0.09, -0.09, 0.0, 0.0, 0.0]),
        np.array([0.5, -0.225, -0.225, 0.0, 0.0, 0.0]),
        np.array([1.0, -0.45, -0.45, 0.0, 0.0, 0.0]),
        np.array([-0.1, 0.045, 0.045, 0.0, 0.0, 0.0]),
        np.array([-0.3, 0.135, 0.135, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.6, 0.0, 0.0]),
        np.array([0.15, -0.05, 0.02, 0.08, -0.04, 0.03]),
        np.full(6, np.log(1.1) / 3.0) * np.array([1, 1, 1, 0, 0, 0]),
        np.full(6, np.log(0.9) / 3.0) * np.array([1, 1, 1, 0, 0, 0]),
    ])
    def test_solid_tangent_positive_definiteness(
        self, ogden_mat: OgdenParams, eps: np.ndarray
    ):
        """For all stable elastic states, v^T C v > 0 requires min_eig(0.5*(C+C^T)) > 0."""
        C = consistent_solid_tangent(ogden_mat, eps)[0]

        # Check eigenvalues of real matrix C
        real_eigs = np.linalg.eigvals(C).real
        assert np.all(real_eigs > 0.0), f"Negative eigenvalue found in C: {real_eigs}"

        # Check quadratic form v^T C v > 0 via symmetric part eigenvalues
        sym_eigs = np.linalg.eigvalsh(0.5 * (C + C.T))
        assert np.all(sym_eigs > 0.0), f"Negative quadratic form eigenvalue: {sym_eigs}"

    @pytest.mark.parametrize("eps", [
        np.zeros(3),
        np.array([0.1, -0.045, 0.0]),
        np.array([0.2, -0.09, 0.0]),
        np.array([0.5, -0.225, 0.0]),
        np.array([-0.1, 0.045, 0.0]),
        np.array([-0.3, 0.135, 0.0]),
        np.array([0.0, 0.0, 0.2]),
        np.array([0.0, 0.0, 0.6]),
        np.array([0.15, -0.05, 0.08]),
    ])
    def test_shell_tangent_positive_definiteness(
        self, ogden_mat: OgdenParams, eps: np.ndarray
    ):
        """For all stable shell states, min_eig(0.5*(C+C^T)) > 0 and real_eigs(C) > 0."""
        C = consistent_shell_tangent(ogden_mat, eps)[0]

        real_eigs = np.linalg.eigvals(C).real
        assert np.all(real_eigs > 0.0), f"Negative eigenvalue found in shell C: {real_eigs}"

        sym_eigs = np.linalg.eigvalsh(0.5 * (C + C.T))
        assert np.all(sym_eigs > 0.0), f"Negative quadratic form eigenvalue: {sym_eigs}"


# ============================================================================
# 6. Multi-Term Models: N = 1, 2, 3, 5
# ============================================================================


class TestMultiTermModels:
    """Verify tangent stiffness across N=1, 2, 3, 5 Ogden expansions."""

    MODELS = [
        (1, [15.0], [2.0], 0.475),
        (2, [10.0, 2.0], [2.0, -2.0], 0.45),
        (3, [6.3, 0.12, -0.1], [1.3, 5.0, -2.0], 0.48),
        (5, [10.0, 5.0, -2.0, 1.0, 0.5], [2.0, -2.0, 3.0, -3.0, 1.5], 0.45),
    ]

    @pytest.mark.parametrize("n_order,mu,alpha,nu", MODELS)
    def test_multiterm_solid_tangent_fd(
        self, n_order: int, mu: list[float], alpha: list[float], nu: float
    ):
        """Verify N-term Ogden solid tangent against central FD under multi-axial strain."""
        mat = build_law82(nordre=n_order, mu=mu, alpha=alpha, nu=nu)
        eps = np.array([0.08, -0.03, 0.02, 0.05, -0.02, 0.01])

        C_alg = consistent_solid_tangent(mat, eps)[0]
        C_num = compute_numerical_solid_tangent(mat, eps)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4

        sym_eigs = np.linalg.eigvalsh(0.5 * (C_alg + C_alg.T))
        assert np.all(sym_eigs > 0.0)

    @pytest.mark.parametrize("n_order,mu,alpha,nu", MODELS)
    def test_multiterm_shell_tangent_fd(
        self, n_order: int, mu: list[float], alpha: list[float], nu: float
    ):
        """Verify N-term Ogden shell tangent against central FD under multi-axial strain."""
        mat = build_law82(nordre=n_order, mu=mu, alpha=alpha, nu=nu)
        eps = np.array([0.08, -0.03, 0.05])

        C_alg = consistent_shell_tangent(mat, eps)[0]
        C_num = compute_numerical_shell_tangent(mat, eps)[0]

        rel_err = relative_tangent_error(C_alg, C_num)
        assert rel_err < 1e-4

        sym_eigs = np.linalg.eigvalsh(0.5 * (C_alg + C_alg.T))
        assert np.all(sym_eigs > 0.0)

    @pytest.mark.parametrize("n_order,mu,alpha,nu", MODELS)
    def test_multiterm_ground_state_shear_modulus(
        self, n_order: int, mu: list[float], alpha: list[float], nu: float
    ):
        """Ground-state shear components C_44, C_55, C_66 must equal sum(mu_k)."""
        mat = build_law82(nordre=n_order, mu=mu, alpha=alpha, nu=nu)
        g0_expected = sum(mu)

        C_solid = consistent_solid_tangent(mat, np.zeros(6))[0]
        assert np.isclose(C_solid[3, 3], g0_expected, rtol=1e-5)
        assert np.isclose(C_solid[4, 4], g0_expected, rtol=1e-5)
        assert np.isclose(C_solid[5, 5], g0_expected, rtol=1e-5)

        C_shell = consistent_shell_tangent(mat, np.zeros(3))[0]
        assert np.isclose(C_shell[2, 2], g0_expected, rtol=1e-5)


# ============================================================================
# 7. Batch Vectorization Tests: (n, 6, 6) and (n, 3, 3)
# ============================================================================


class TestBatchVectorization:
    """Verify batch vectorization of solid and shell tangents across multiple elements."""

    @pytest.fixture
    def ogden_mat(self) -> OgdenParams:
        return build_law82(nordre=2, mu=[10.0, 2.0], alpha=[2.0, -2.0], nu=0.45)

    @pytest.mark.parametrize("n_elem", [1, 5, 16, 64])
    def test_solid_batch_vectorization(self, ogden_mat: OgdenParams, n_elem: int):
        """Batch (n, 6, 6) evaluation must match element-by-element evaluation."""
        rng = np.random.RandomState(12345 + n_elem)
        eps_batch = rng.uniform(-0.05, 0.05, size=(n_elem, 6))

        C_batch = consistent_solid_tangent(ogden_mat, eps_batch)
        assert C_batch.shape == (n_elem, 6, 6)

        # Compare against single-element calls
        for i in range(n_elem):
            C_single = consistent_solid_tangent(ogden_mat, eps_batch[i:i + 1])
            assert np.allclose(C_batch[i], C_single[0], atol=1e-12)

        # Verify batch numerical FD matches batch algorithmic tangent
        C_num_batch = compute_numerical_solid_tangent(ogden_mat, eps_batch)
        rel_err = relative_tangent_error(C_batch, C_num_batch)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("n_elem", [1, 5, 16, 64])
    def test_shell_batch_vectorization(self, ogden_mat: OgdenParams, n_elem: int):
        """Batch (n, 3, 3) evaluation must match element-by-element evaluation."""
        rng = np.random.RandomState(54321 + n_elem)
        eps_batch = rng.uniform(-0.05, 0.05, size=(n_elem, 3))

        C_batch = consistent_shell_tangent(ogden_mat, eps_batch)
        assert C_batch.shape == (n_elem, 3, 3)

        # Compare against single-element calls
        for i in range(n_elem):
            C_single = consistent_shell_tangent(ogden_mat, eps_batch[i:i + 1])
            assert np.allclose(C_batch[i], C_single[0], atol=1e-12)

        # Verify batch numerical FD matches batch algorithmic tangent
        C_num_batch = compute_numerical_shell_tangent(ogden_mat, eps_batch)
        rel_err = relative_tangent_error(C_batch, C_num_batch)
        assert rel_err < 1e-4


# ============================================================================
# 8. Deformation Gradient Input & Tangent Function Aliases
# ============================================================================


class TestDeformationGradientAndAliases:
    """Verify input formats (deformation gradient F) and function alias dispatches."""

    def test_solid_tangent_from_deformation_gradient(self):
        """Verify consistent_solid_tangent accepts F with shape (n, 3, 3)."""
        mat = build_law82(nordre=1, mu=[10.0], alpha=[2.0], nu=0.45)

        # Pure stretch along axes: F = diag(lambda_1, lambda_2, lambda_3)
        lam1, lam2, lam3 = 1.1, 0.95, 0.95
        F = np.diag([lam1, lam2, lam3]).reshape(1, 3, 3)

        C_F = consistent_solid_tangent(mat, F)
        assert C_F.shape == (1, 6, 6)

        # Equivalent logarithmic strain: eps = [ln(lam1), ln(lam2), ln(lam3), 0, 0, 0]
        eps_equiv = np.array([[np.log(lam1), np.log(lam2), np.log(lam3), 0.0, 0.0, 0.0]])
        C_eps = consistent_solid_tangent(mat, eps_equiv)

        assert np.allclose(C_F, C_eps, atol=1e-10)

    def test_tangent_aliases(self):
        """Verify all package dispatch aliases point to the correct implementations."""
        mat = build_law82()
        eps_s = np.zeros((1, 6))
        eps_sh = np.zeros((1, 3))

        c1 = consistent_solid_tangent(mat, eps_s)
        c2 = solid_tangent(mat, eps_s)
        c3 = law82_consistent_solid_tangent(mat, eps_s)
        assert np.array_equal(c1, c2)
        assert np.array_equal(c1, c3)

        c4 = consistent_shell_tangent(mat, eps_sh)
        c5 = shell_tangent(mat, eps_sh)
        c6 = law82_consistent_shell_tangent(mat, eps_sh)
        assert np.array_equal(c4, c5)
        assert np.array_equal(c4, c6)
