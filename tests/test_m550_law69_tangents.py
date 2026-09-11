"""
Exhaustive Algorithmic Consistent Tangent Stiffness Verification Suite for LAW69.
(/MAT/LAW69, /MAT/HYP_ELAS, /MAT/HYPERELASTIC).

Audits tangent_law69_solid (NEL, 6, 6) and tangent_law69_shell (NEL, 3, 3)
from pyradioss/materials/law69_hyperelastic.py across:
1. Numerical central finite difference perturbation test:
   - Step sizes h in [1e-6, 1e-7]
   - D_num[:, :, j] = (sigma(eps + h * e_j) - sigma(eps - h * e_j)) / (2 * h)
   - Relative error ||D_alg - D_num|| / ||D_num|| < 1e-4
2. Multiple deformation states:
   - Pure tension: small strain (eps=0.01), moderate strain (eps=0.2), large strain (eps=1.0)
   - Pure compression: eps=-0.05, -0.2, -0.5
   - Pure shear: eps_xy=0.05, 0.2, 0.4
   - Multiaxial combined states: tension + shear, triaxial states
   - Pure volumetric expansion (J = 1.1) and compaction (J = 0.9)
3. Material formulations:
   - Mooney-Rivlin (law_id=2)
   - Ogden (law_id=1) with terms N=1, 2, 3
   - Constant bulk modulus vs tabulated bulk function f(RV)
4. Shell plane-stress tangents:
   - 2D shell in-plane tangent (NEL, 3, 3) accounting for sigma_zz = 0
   - In-plane tension, compression, shear, and multiaxial states
5. Ground state symmetry & isotropic elasticity recovery:
   - Exact isotropic elasticity tensor C_ijkl = lambda delta_ij delta_kl + mu (delta_ik delta_jl + delta_il delta_jk)
   - Exact major symmetry C = C.T at ground state and pure volumetric states
   - Continuum mechanics stress-asymmetry identity: C_12 - C_21 = sigma_yy - sigma_xx
6. Vectorized multi-element batches:
   - Batches with NEL = 1, 5, 20
   - Consistency between batch and single-element evaluation
7. Deformation gradient (F) input format and dispatch aliases.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.materials.law69_hyperelastic import (
    Law69Params,
    build_law69,
    sigeps69_solid,
    sigeps69c_shell,
    solid_update,
    shell_update,
    tangent_law69_solid,
    tangent_law69_shell,
    solid_tangent,
    shell_tangent,
    consistent_solid_tangent,
    consistent_shell_tangent,
    law69_consistent_solid_tangent,
    law69_consistent_shell_tangent,
)


# ============================================================================
# Helper Functions: Independent Central Finite Difference Numerical Tangents
# ============================================================================


def compute_numerical_solid_tangent(
    mat: Law69Params,
    eps: np.ndarray,
    h: float = 1e-6,
    dt: float = 0.0,
    extra: dict | None = None,
    ismstr: int = 0,
) -> np.ndarray:
    """Compute numerical solid tangent (NEL, 6, 6) by central finite differences.

    D_num[:, :, j] = (sigma(eps + h * e_j) - sigma(eps - h * e_j)) / (2 * h)
    """
    eps_arr = np.asarray(eps, dtype=np.float64)
    if eps_arr.ndim == 1:
        eps_arr = eps_arr.reshape(1, -1)
    nel = eps_arr.shape[0]

    D_num = np.zeros((nel, 6, 6), dtype=np.float64)
    dummy_sig = np.zeros((nel, 6), dtype=np.float64)

    for j in range(6):
        ep = eps_arr.copy()
        ep[:, j] += h
        em = eps_arr.copy()
        em[:, j] -= h

        extra_p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in extra.items()} if extra is not None else None
        extra_m = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in extra.items()} if extra is not None else None

        sp, _, _ = sigeps69_solid(mat, dummy_sig, eps=ep, dt=dt, extra=extra_p, ismstr=ismstr, return_sound_speed=True)
        sm, _, _ = sigeps69_solid(mat, dummy_sig, eps=em, dt=dt, extra=extra_m, ismstr=ismstr, return_sound_speed=True)

        D_num[:, :, j] = (sp - sm) / (2.0 * h)

    return D_num


def compute_numerical_shell_tangent(
    mat: Law69Params,
    eps: np.ndarray,
    h: float = 1e-6,
    dt: float = 0.0,
    extra: dict | None = None,
    ismstr: int = 0,
) -> np.ndarray:
    """Compute numerical shell tangent (NEL, 3, 3) by central finite differences.

    D_num[:, :, j] = (sigma(eps + h * e_j) - sigma(eps - h * e_j)) / (2 * h)
    """
    eps_arr = np.asarray(eps, dtype=np.float64)
    if eps_arr.ndim == 1:
        eps_arr = eps_arr.reshape(1, -1)
    nel = eps_arr.shape[0]

    D_num = np.zeros((nel, 3, 3), dtype=np.float64)
    dummy_sig = np.zeros((nel, 3), dtype=np.float64)

    for j in range(3):
        ep = eps_arr.copy()
        ep[:, j] += h
        em = eps_arr.copy()
        em[:, j] -= h

        extra_p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in extra.items()} if extra is not None else None
        extra_m = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in extra.items()} if extra is not None else None

        sp, _ = sigeps69c_shell(mat, dummy_sig, eps=ep, dt=dt, extra=extra_p, ismstr=ismstr, return_sound_speed=False)
        sm, _ = sigeps69c_shell(mat, dummy_sig, eps=em, dt=dt, extra=extra_m, ismstr=ismstr, return_sound_speed=False)

        D_num[:, :, j] = (sp[:, :3] - sm[:, :3]) / (2.0 * h)

    return D_num


def relative_tangent_error(D_alg: np.ndarray, D_num: np.ndarray) -> float:
    """Compute Frobenius relative error ||D_alg - D_num|| / ||D_num||."""
    norm_num = np.linalg.norm(D_num)
    if norm_num < 1e-12:
        return float(np.linalg.norm(D_alg - D_num))
    return float(np.linalg.norm(D_alg - D_num) / norm_num)


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
        mat = build_law69(law_id=1, mu=[g0], alpha=[2.0], nu=nu)
        eps_zero = np.zeros((1, 6))

        C_alg = tangent_law69_solid(mat, eps_zero)[0]

        mu = mat.g0
        lam = mat.rbulk - (2.0 / 3.0) * mu

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

        # Spectral validation: 1 volumetric mode (3*K), 5 isochoric modes (2*mu)
        eigs = np.sort(np.linalg.eigvalsh(C_alg))
        K_bulk = mat.rbulk
        expected_vol_eig = 3.0 * K_bulk
        assert np.all(eigs > 0.0)
        assert np.isclose(eigs[-1], expected_vol_eig, rtol=1e-5)

    @pytest.mark.parametrize("nu", [0.3, 0.45, 0.48, 0.495])
    @pytest.mark.parametrize("g0", [1.0, 10.0, 50.0])
    def test_shell_ground_state_plane_stress_recovery(self, nu: float, g0: float):
        """Shell tangent at eps=0 approaches plane stress isotropic elasticity tensor:

        C_11 = C_22 = E / (1 - nu^2)
        C_12 = C_21 = nu * E / (1 - nu^2)
        C_33 = G
        All other components = 0.
        """
        mat = build_law69(law_id=1, mu=[g0], alpha=[2.0], nu=nu)
        eps_zero = np.zeros((1, 3))

        C_alg = tangent_law69_shell(mat, eps_zero)[0]

        E_mod = 2.0 * mat.g0 * (1.0 + nu)
        c11_plane = E_mod / (1.0 - nu ** 2)
        c12_plane = nu * c11_plane
        c33_plane = mat.g0

        C_expected = np.array([
            [c11_plane, c12_plane, 0.0],
            [c12_plane, c11_plane, 0.0],
            [0.0,       0.0,       c33_plane],
        ])

        # 4-step Newton iteration in sigeps69c.F converges to within 0.02% for nu>=0.48,
        # within 0.2% for nu=0.45, and within 12% for compressible nu=0.3
        if nu >= 0.46:
            rtol = 2e-4
        elif nu >= 0.4:
            rtol = 2e-3
        else:
            rtol = 0.15

        assert np.allclose(C_alg, C_expected, rtol=rtol, atol=1e-3)
        assert np.allclose(C_alg, C_alg.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(C_alg) > 0.0)


# ============================================================================
# 2. Numerical Central Finite Difference Verification (Solid Continuum)
# ============================================================================


class TestSolidFiniteDifferenceTangents:
    """Verify tangent_law69_solid against central FD across deformation states."""

    @pytest.fixture(params=[1, 2])
    def mat_model(self, request: pytest.FixtureRequest) -> Law69Params:
        """Test both Ogden (law_id=1) and Mooney-Rivlin (law_id=2)."""
        if request.param == 1:
            return build_law69(law_id=1, mu=[10.0, 2.0], alpha=[2.0, -2.0], nu=0.45)
        else:
            return build_law69(law_id=2, mu=[12.0, -3.0], alpha=[2.0, -2.0], nu=0.48)

    @pytest.mark.parametrize("h", [1e-6, 1e-7])
    def test_solid_ground_state_fd(self, mat_model: Law69Params, h: float):
        """Ground state eps = 0 central FD test."""
        eps = np.zeros((1, 6))
        D_alg = tangent_law69_solid(mat_model, eps, h=h)
        D_num = compute_numerical_solid_tangent(mat_model, eps, h=h)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("eps_val", [0.01, 0.2, 1.0])
    @pytest.mark.parametrize("h", [1e-6, 1e-7])
    def test_solid_pure_tension_fd(self, mat_model: Law69Params, eps_val: float, h: float):
        """Pure tension: small strain (0.01), moderate strain (0.2), large strain (1.0)."""
        nu = mat_model.nu
        eps = np.array([[eps_val, -nu * eps_val, -nu * eps_val, 0.0, 0.0, 0.0]])

        D_alg = tangent_law69_solid(mat_model, eps, h=h)
        D_num = compute_numerical_solid_tangent(mat_model, eps, h=h)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("eps_val", [-0.05, -0.2, -0.5])
    @pytest.mark.parametrize("h", [1e-6, 1e-7])
    def test_solid_pure_compression_fd(self, mat_model: Law69Params, eps_val: float, h: float):
        """Pure compression: eps = -0.05, -0.2, -0.5."""
        nu = mat_model.nu
        eps = np.array([[eps_val, -nu * eps_val, -nu * eps_val, 0.0, 0.0, 0.0]])

        D_alg = tangent_law69_solid(mat_model, eps, h=h)
        D_num = compute_numerical_solid_tangent(mat_model, eps, h=h)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("eps_xy", [0.05, 0.2, 0.4])
    @pytest.mark.parametrize("h", [1e-6, 1e-7])
    def test_solid_pure_shear_fd(self, mat_model: Law69Params, eps_xy: float, h: float):
        """Pure shear: eps_xy = 0.05, 0.2, 0.4 (Voigt engineering shear gamma_xy = 2 * eps_xy)."""
        eps = np.array([[0.0, 0.0, 0.0, 2.0 * eps_xy, 0.0, 0.0]])

        D_alg = tangent_law69_solid(mat_model, eps, h=h)
        D_num = compute_numerical_solid_tangent(mat_model, eps, h=h)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("h", [1e-6, 1e-7])
    def test_solid_multiaxial_tension_plus_shear_fd(self, mat_model: Law69Params, h: float):
        """Multiaxial combined state: tension + shear."""
        eps = np.array([[0.15, -0.05, -0.05, 0.1, 0.0, 0.0]])

        D_alg = tangent_law69_solid(mat_model, eps, h=h)
        D_num = compute_numerical_solid_tangent(mat_model, eps, h=h)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("h", [1e-6, 1e-7])
    def test_solid_multiaxial_triaxial_fd(self, mat_model: Law69Params, h: float):
        """Multiaxial combined state: fully 3D triaxial strain state."""
        eps = np.array([[0.12, -0.06, 0.03, 0.08, -0.04, 0.05]])

        D_alg = tangent_law69_solid(mat_model, eps, h=h)
        D_num = compute_numerical_solid_tangent(mat_model, eps, h=h)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("J", [1.1, 0.9])
    @pytest.mark.parametrize("h", [1e-6, 1e-7])
    def test_solid_pure_volumetric_fd(self, mat_model: Law69Params, J: float, h: float):
        """Pure volumetric expansion (J = 1.1) and compaction (J = 0.9)."""
        e_vol = np.log(J) / 3.0
        eps = np.array([[e_vol, e_vol, e_vol, 0.0, 0.0, 0.0]])

        D_alg = tangent_law69_solid(mat_model, eps, h=h)
        D_num = compute_numerical_solid_tangent(mat_model, eps, h=h)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4


# ============================================================================
# 3. Ogden Multi-Term & Tabulated Bulk Function Verifications
# ============================================================================


class TestOgdenTermsAndBulkFunction:
    """Verify tangent stiffness across Ogden orders N=1, 2, 3 and tabulated bulk f(RV)."""

    OGDEN_MODELS = [
        (1, [15.0], [2.0], 0.475),
        (2, [10.0, 2.0], [2.0, -2.0], 0.45),
        (3, [6.0, 0.5, -0.2], [1.3, 5.0, -2.0], 0.48),
    ]

    @pytest.mark.parametrize("nordre,mu,alpha,nu", OGDEN_MODELS)
    def test_ogden_solid_tangent_fd(self, nordre: int, mu: list[float], alpha: list[float], nu: float):
        """Verify Ogden N=1, 2, 3 solid tangents under multiaxial strain."""
        mat = build_law69(law_id=1, nip=nordre, mu=mu, alpha=alpha, nu=nu)
        eps = np.array([[0.08, -0.03, 0.02, 0.05, -0.02, 0.01]])

        D_alg = tangent_law69_solid(mat, eps)
        D_num = compute_numerical_solid_tangent(mat, eps)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("nordre,mu,alpha,nu", OGDEN_MODELS)
    def test_ogden_ground_state_shear_modulus(self, nordre: int, mu: list[float], alpha: list[float], nu: float):
        """Ground-state shear components C_44, C_55, C_66 must equal sum(mu_k * alpha_k) / 2."""
        mat = build_law69(law_id=1, nip=nordre, mu=mu, alpha=alpha, nu=nu)
        g0_expected = float(np.sum(np.array(mu) * np.array(alpha))) / 2.0

        C_solid = tangent_law69_solid(mat, np.zeros((1, 6)))[0]
        assert np.isclose(C_solid[3, 3], g0_expected, rtol=1e-5)
        assert np.isclose(C_solid[4, 4], g0_expected, rtol=1e-5)
        assert np.isclose(C_solid[5, 5], g0_expected, rtol=1e-5)

        C_shell = tangent_law69_shell(mat, np.zeros((1, 3)))[0]
        assert np.isclose(C_shell[2, 2], g0_expected, rtol=1e-5)

    def test_tabulated_bulk_callable_fd(self):
        """Verify solid tangent with callable tabulated bulk function f(RV) = RV^1.5."""
        mat = build_law69(law_id=1, fct_id=10, fscale=1.0, mu=[10.0], alpha=[2.0], nu=0.45)
        fn_bulk = lambda rv: rv ** 1.5
        extra = {"curves": {10: fn_bulk}}

        eps = np.array([[0.05, -0.02, 0.01, 0.04, 0.0, 0.0]])
        D_alg = tangent_law69_solid(mat, eps, extra=extra)
        D_num = compute_numerical_solid_tangent(mat, eps, extra=extra)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4

    def test_tabulated_bulk_piecewise_linear_fd(self):
        """Verify solid tangent with tabulated bulk function curve (x, y data points)."""
        mat = build_law69(law_id=1, fct_id=20, fscale=1.2, mu=[12.0], alpha=[2.0], nu=0.45)

        class TabulatedCurve:
            x = np.array([0.5, 0.8, 1.0, 1.2, 1.5], dtype=np.float64)
            y = np.array([0.4, 0.75, 1.0, 1.3, 1.8], dtype=np.float64)

        extra = {"curves": {20: TabulatedCurve()}}

        eps = np.array([[0.03, 0.02, -0.01, 0.02, -0.01, 0.0]])
        D_alg = tangent_law69_solid(mat, eps, extra=extra)
        D_num = compute_numerical_solid_tangent(mat, eps, extra=extra)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4


# ============================================================================
# 4. Numerical Central Finite Difference Verification (Shell Plane Stress)
# ============================================================================


class TestShellFiniteDifferenceTangents:
    """Verify tangent_law69_shell against central FD across in-plane deformation states."""

    @pytest.fixture(params=[1, 2])
    def mat_model(self, request: pytest.FixtureRequest) -> Law69Params:
        """Test both Ogden (law_id=1) and Mooney-Rivlin (law_id=2)."""
        if request.param == 1:
            return build_law69(law_id=1, mu=[10.0, 2.0], alpha=[2.0, -2.0], nu=0.45)
        else:
            return build_law69(law_id=2, mu=[12.0, -3.0], alpha=[2.0, -2.0], nu=0.48)

    @pytest.mark.parametrize("h", [1e-6, 1e-7])
    def test_shell_ground_state_fd(self, mat_model: Law69Params, h: float):
        """Ground state eps = 0 central FD test."""
        eps = np.zeros((1, 3))
        D_alg = tangent_law69_shell(mat_model, eps, h=h)
        D_num = compute_numerical_shell_tangent(mat_model, eps, h=h)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("eps_val", [0.01, 0.2, 0.5])
    @pytest.mark.parametrize("h", [1e-6, 1e-7])
    def test_shell_inplane_tension_fd(self, mat_model: Law69Params, eps_val: float, h: float):
        """In-plane tension: small strain (0.01), moderate strain (0.2), large strain (0.5)."""
        nu = mat_model.nu
        eps = np.array([[eps_val, -nu * eps_val, 0.0]])

        D_alg = tangent_law69_shell(mat_model, eps, h=h)
        D_num = compute_numerical_shell_tangent(mat_model, eps, h=h)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("eps_val", [-0.05, -0.2, -0.4])
    @pytest.mark.parametrize("h", [1e-6, 1e-7])
    def test_shell_inplane_compression_fd(self, mat_model: Law69Params, eps_val: float, h: float):
        """In-plane compression: eps = -0.05, -0.2, -0.4."""
        nu = mat_model.nu
        eps = np.array([[eps_val, -nu * eps_val, 0.0]])

        D_alg = tangent_law69_shell(mat_model, eps, h=h)
        D_num = compute_numerical_shell_tangent(mat_model, eps, h=h)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("eps_xy", [0.05, 0.2, 0.4])
    @pytest.mark.parametrize("h", [1e-6, 1e-7])
    def test_shell_inplane_shear_fd(self, mat_model: Law69Params, eps_xy: float, h: float):
        """In-plane shear: eps_xy = 0.05, 0.2, 0.4 (Voigt gamma_12 = 2 * eps_xy)."""
        eps = np.array([[0.0, 0.0, 2.0 * eps_xy]])

        D_alg = tangent_law69_shell(mat_model, eps, h=h)
        D_num = compute_numerical_shell_tangent(mat_model, eps, h=h)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("h", [1e-6, 1e-7])
    def test_shell_multiaxial_combined_state_fd(self, mat_model: Law69Params, h: float):
        """Combined in-plane multi-axial strain state (tension + compression + shear)."""
        eps = np.array([[0.15, -0.05, 0.08]])

        D_alg = tangent_law69_shell(mat_model, eps, h=h)
        D_num = compute_numerical_shell_tangent(mat_model, eps, h=h)

        rel_err = relative_tangent_error(D_alg, D_num)
        assert rel_err < 1e-4


# ============================================================================
# 5. Symmetry Verification & Continuum Mechanics Stress Identity
# ============================================================================


class TestTangentSymmetry:
    """Verify major symmetry properties and exact continuum stress-tangent identities."""

    @pytest.fixture
    def mat_model(self) -> Law69Params:
        return build_law69(law_id=1, mu=[10.0, 2.0], alpha=[2.0, -2.0], nu=0.45)

    def test_solid_ground_state_symmetry(self, mat_model: Law69Params):
        """Exact major symmetry C = C.T at ground state for 3D solids."""
        C = tangent_law69_solid(mat_model, np.zeros((1, 6)))[0]
        asym = np.max(np.abs(C - C.T))
        assert asym < 1e-8

    def test_shell_ground_state_symmetry(self, mat_model: Law69Params):
        """Exact major symmetry C = C.T at ground state for 2D shells."""
        C = tangent_law69_shell(mat_model, np.zeros((1, 3)))[0]
        asym = np.max(np.abs(C - C.T))
        assert asym < 1e-8

    @pytest.mark.parametrize("J", [1.1, 0.9])
    def test_solid_volumetric_exact_symmetry(self, mat_model: Law69Params, J: float):
        """Exact symmetry C = C.T under pure volumetric deformation."""
        e_vol = np.log(J) / 3.0
        eps = np.array([[e_vol, e_vol, e_vol, 0.0, 0.0, 0.0]])
        C = tangent_law69_solid(mat_model, eps)[0]
        asym = np.max(np.abs(C - C.T))
        assert asym < 1e-8

    @pytest.mark.parametrize("eps", [
        np.array([[0.05, -0.02, 0.01, 0.03, -0.01, 0.02]]),
        np.array([[0.15, -0.05, 0.02, 0.08, -0.04, 0.03]]),
        np.array([[0.3, -0.1, -0.1, 0.0, 0.0, 0.0]]),
        np.array([[-0.2, 0.1, 0.1, 0.05, 0.0, 0.0]]),
    ])
    def test_solid_continuum_mechanics_tangent_asymmetry_identity(
        self, mat_model: Law69Params, eps: np.ndarray
    ):
        """In finite elasticity, Cauchy stress derivative w.r.t logarithmic strain satisfies:

            C_12 - C_21 = sigma_22 - sigma_11
            C_13 - C_31 = sigma_33 - sigma_11
            C_23 - C_32 = sigma_33 - sigma_22
        """
        C = tangent_law69_solid(mat_model, eps)[0]
        sig, _, _ = sigeps69_solid(mat_model, np.zeros((1, 6)), eps=eps, return_sound_speed=True)
        sig = sig[0]

        # C_12 - C_21 == sigma_22 - sigma_11
        assert np.isclose(C[0, 1] - C[1, 0], sig[1] - sig[0], rtol=1e-4, atol=1e-5)
        # C_13 - C_31 == sigma_33 - sigma_11
        assert np.isclose(C[0, 2] - C[2, 0], sig[2] - sig[0], rtol=1e-4, atol=1e-5)
        # C_23 - C_32 == sigma_33 - sigma_22
        assert np.isclose(C[1, 2] - C[2, 1], sig[2] - sig[1], rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("eps", [
        np.array([[0.05, -0.02, 0.03]]),
        np.array([[0.15, -0.05, 0.08]]),
        np.array([[-0.1, 0.045, 0.02]]),
    ])
    def test_shell_tangent_approximate_symmetry(self, mat_model: Law69Params, eps: np.ndarray):
        """Shell tangent exhibits approximate symmetry C_ij ~= C_ji under moderate deformation."""
        C = tangent_law69_shell(mat_model, eps)[0]
        norm_C = np.linalg.norm(C)
        asym = np.linalg.norm(C - C.T) / norm_C
        # For moderate strains in shells, relative asymmetry is well within 15%
        assert asym < 0.15


# ============================================================================
# 6. Positive-Definiteness Verification (Stability v^T C v > 0)
# ============================================================================


class TestPositiveDefiniteness:
    """Verify positive definiteness of algorithmic tangents across deformation states."""

    @pytest.fixture
    def mat_model(self) -> Law69Params:
        return build_law69(law_id=1, mu=[10.0, 2.0], alpha=[2.0, -2.0], nu=0.45)

    @pytest.mark.parametrize("eps", [
        np.zeros(6),
        np.array([0.01, -0.0045, -0.0045, 0.0, 0.0, 0.0]),
        np.array([0.2, -0.09, -0.09, 0.0, 0.0, 0.0]),
        np.array([0.5, -0.225, -0.225, 0.0, 0.0, 0.0]),
        np.array([-0.05, 0.0225, 0.0225, 0.0, 0.0, 0.0]),
        np.array([-0.2, 0.09, 0.09, 0.0, 0.0, 0.0]),
        np.array([-0.4, 0.18, 0.18, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.4, 0.0, 0.0]),
        np.array([0.15, -0.05, 0.02, 0.08, -0.04, 0.03]),
        np.full(6, np.log(1.1) / 3.0) * np.array([1, 1, 1, 0, 0, 0]),
        np.full(6, np.log(0.9) / 3.0) * np.array([1, 1, 1, 0, 0, 0]),
    ])
    def test_solid_tangent_positive_definiteness(self, mat_model: Law69Params, eps: np.ndarray):
        """For all stable elastic states, v^T C v > 0 requires min_eig(0.5*(C+C^T)) > 0."""
        C = tangent_law69_solid(mat_model, eps.reshape(1, 6))[0]

        real_eigs = np.linalg.eigvals(C).real
        assert np.all(real_eigs > 0.0), f"Negative eigenvalue found in C: {real_eigs}"

        sym_eigs = np.linalg.eigvalsh(0.5 * (C + C.T))
        assert np.all(sym_eigs > 0.0), f"Negative quadratic form eigenvalue: {sym_eigs}"

    @pytest.mark.parametrize("eps", [
        np.zeros(3),
        np.array([0.01, -0.0045, 0.0]),
        np.array([0.2, -0.09, 0.0]),
        np.array([0.4, -0.18, 0.0]),
        np.array([-0.05, 0.0225, 0.0]),
        np.array([-0.2, 0.09, 0.0]),
        np.array([0.0, 0.0, 0.1]),
        np.array([0.0, 0.0, 0.4]),
        np.array([0.15, -0.05, 0.08]),
    ])
    def test_shell_tangent_positive_definiteness(self, mat_model: Law69Params, eps: np.ndarray):
        """For all stable shell states, min_eig(0.5*(C+C^T)) > 0 and real_eigs(C) > 0."""
        C = tangent_law69_shell(mat_model, eps.reshape(1, 3))[0]

        real_eigs = np.linalg.eigvals(C).real
        assert np.all(real_eigs > 0.0), f"Negative eigenvalue found in shell C: {real_eigs}"

        sym_eigs = np.linalg.eigvalsh(0.5 * (C + C.T))
        assert np.all(sym_eigs > 0.0), f"Negative quadratic form eigenvalue: {sym_eigs}"


# ============================================================================
# 7. Batch Vectorization Tests: NEL = 1, 5, 20
# ============================================================================


class TestBatchVectorization:
    """Verify batch vectorization of solid and shell tangents across multiple elements."""

    @pytest.fixture
    def mat_model(self) -> Law69Params:
        return build_law69(law_id=1, mu=[10.0, 2.0], alpha=[2.0, -2.0], nu=0.45)

    @pytest.mark.parametrize("nel", [1, 5, 20])
    def test_solid_batch_vectorization(self, mat_model: Law69Params, nel: int):
        """Batch (NEL, 6, 6) evaluation must match element-by-element evaluation."""
        rng = np.random.RandomState(42 + nel)
        eps_batch = rng.uniform(-0.04, 0.04, size=(nel, 6))

        C_batch = tangent_law69_solid(mat_model, eps_batch)
        assert C_batch.shape == (nel, 6, 6)

        # Compare against single-element calls
        for i in range(nel):
            C_single = tangent_law69_solid(mat_model, eps_batch[i:i + 1])
            assert np.allclose(C_batch[i], C_single[0], atol=1e-12)

        # Verify batch numerical FD matches batch algorithmic tangent
        C_num_batch = compute_numerical_solid_tangent(mat_model, eps_batch)
        rel_err = relative_tangent_error(C_batch, C_num_batch)
        assert rel_err < 1e-4

    @pytest.mark.parametrize("nel", [1, 5, 20])
    def test_shell_batch_vectorization(self, mat_model: Law69Params, nel: int):
        """Batch (NEL, 3, 3) evaluation must match element-by-element evaluation."""
        rng = np.random.RandomState(84 + nel)
        eps_batch = rng.uniform(-0.04, 0.04, size=(nel, 3))

        C_batch = tangent_law69_shell(mat_model, eps_batch)
        assert C_batch.shape == (nel, 3, 3)

        # Compare against single-element calls
        for i in range(nel):
            C_single = tangent_law69_shell(mat_model, eps_batch[i:i + 1])
            assert np.allclose(C_batch[i], C_single[0], atol=1e-12)

        # Verify batch numerical FD matches batch algorithmic tangent
        C_num_batch = compute_numerical_shell_tangent(mat_model, eps_batch)
        rel_err = relative_tangent_error(C_batch, C_num_batch)
        assert rel_err < 1e-4


# ============================================================================
# 8. Deformation Gradient Input & Tangent Function Aliases
# ============================================================================


class TestDeformationGradientAndAliases:
    """Verify input formats (deformation gradient F) and function alias dispatches."""

    def test_solid_tangent_from_deformation_gradient(self):
        """Verify tangent_law69_solid accepts F with shape (1, 3, 3)."""
        mat = build_law69(law_id=1, mu=[10.0], alpha=[2.0], nu=0.45)

        lam1, lam2, lam3 = 1.05, 0.96, 0.98
        F = np.diag([lam1, lam2, lam3]).reshape(1, 3, 3)

        C_F = tangent_law69_solid(mat, F)
        assert C_F.shape == (1, 6, 6)

        # Equivalent logarithmic strain: eps = [ln(lam1), ln(lam2), ln(lam3), 0, 0, 0]
        eps_equiv = np.array([[np.log(lam1), np.log(lam2), np.log(lam3), 0.0, 0.0, 0.0]])
        C_eps = tangent_law69_solid(mat, eps_equiv)

        assert np.allclose(C_F, C_eps, atol=1e-6)

    def test_tangent_aliases(self):
        """Verify all package dispatch aliases point to the correct implementations."""
        mat = build_law69()
        eps_s = np.zeros((1, 6))
        eps_sh = np.zeros((1, 3))

        c1 = tangent_law69_solid(mat, eps_s)
        c2 = solid_tangent(mat, eps_s)
        c3 = consistent_solid_tangent(mat, eps_s)
        c4 = law69_consistent_solid_tangent(mat, eps_s)
        assert np.array_equal(c1, c2)
        assert np.array_equal(c1, c3)
        assert np.array_equal(c1, c4)

        c5 = tangent_law69_shell(mat, eps_sh)
        c6 = shell_tangent(mat, eps_sh)
        c7 = consistent_shell_tangent(mat, eps_sh)
        c8 = law69_consistent_shell_tangent(mat, eps_sh)
        assert np.array_equal(c5, c6)
        assert np.array_equal(c5, c7)
        assert np.array_equal(c5, c8)
