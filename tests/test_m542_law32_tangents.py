"""
LAW32 Algorithmic Consistent Tangent Stiffness Tensor Audit Suite.
Milestone M542 Wave 2 Auditor 2.

Rigorously audits the algorithmic consistent tangent stiffness tensor in
pyradioss/materials/law32_hill.py:
  1. Directional numerical derivative check:
     Central-difference numerical perturbation:
       C_num[:, j] = (sigma(deps + h*e_j) - sigma(deps - h*e_j)) / (2*h)
     Verified against consistent_shell_tangent for h = 1e-6 and 1e-7.
     Relative error strictly bounded (< 1e-4 in plastic flow, < 1e-6 in elastic regime).
  2. Major symmetry:
     Verify C_ij = C_ji for all i, j in [0, 1, 2] in elastic regime and with symmetric=True.
  3. Positive definiteness:
     Verify eigenvalues are positive in stable elastic and hardening plastic regimes.
  4. Multi-axial strain states:
     - Uniaxial tension along fiber direction (theta = 0 deg)
     - Uniaxial tension transverse to fiber (theta = 90 deg)
     - Uniaxial tension at theta = 45 deg
     - Biaxial tension (eps_xx = eps_yy)
     - Pure shear (gamma_xy)
     - General combined in-plane strain state
  5. Lankford anisotropy sensitivity:
     - Isotropic steel (R00 = R45 = R90 = 1.0)
     - High normal anisotropy (R00 = R90 = 2.0, R45 = 1.5)
     - Planar anisotropy (R00 = 1.8, R45 = 1.2, R90 = 2.2)
  6. Multi-element vectorization consistency:
     Identical evaluations between single-element (1, 3, 3) / 1D (3, 3) and batched (N, 3, 3)
     for N = 16 and N = 64.
  7. Robustness, failure deletion, and dispatcher integration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from pyradioss.materials import (
    consistent_shell_tangent,
    law32_hill,
    shell_layer_tangent,
    shell_membrane_tangent,
    shell_update,
)
from pyradioss.model.entities import Material


# =============================================================================
# Helper Utilities
# =============================================================================

def _make_law32(
    e: float = 210000.0,
    nu: float = 0.3,
    a_yield: float = 300.0,
    b: float = 1.0,
    n: float = 0.0,
    r00: float = 1.0,
    r45: float = 1.0,
    r90: float = 1.0,
    i_yield: int = 0,
    ipla: int = 0,
    sig_max: float = 1e30,
    eps_max: float = 1e30,
    rho0: float = 7.85e-9,
) -> Material:
    """Build a validated LAW32 Material instance with explicit parameters."""
    return law32_hill.build_law32(
        id=1,
        rho0=rho0,
        E=e,
        nu=nu,
        A=a_yield,
        B=b,
        n=n,
        r00=r00,
        r45=r45,
        r90=r90,
        i_yield=i_yield,
        ipla=ipla,
        sig_max=sig_max,
        eps_max=eps_max,
    )


def _compute_numerical_tangent(
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
        # Forward perturbation
        dp = base.copy()
        dp[0, j] += h
        sp_init = np.asarray(sig_init, dtype=float).copy().reshape(1, -1)
        ep_init = np.asarray(epsp_init, dtype=float).copy().flatten() if epsp_init is not None else np.zeros(1)
        ex_p = extra.copy() if extra is not None else None
        sp, _ = law32_hill.shell_update(mat, sp_init, dp, ep_init, dt=dt, extra=ex_p)

        # Backward perturbation
        dm = base.copy()
        dm[0, j] -= h
        sm_init = np.asarray(sig_init, dtype=float).copy().reshape(1, -1)
        em_init = np.asarray(epsp_init, dtype=float).copy().flatten() if epsp_init is not None else np.zeros(1)
        ex_m = extra.copy() if extra is not None else None
        sm, _ = law32_hill.shell_update(mat, sm_init, dm, em_init, dt=dt, extra=ex_m)

        c_num[:, j] = (sp[0, :3] - sm[0, :3]) / (2.0 * h)

    return c_num


# =============================================================================
# 1. Directional Numerical Derivative Verification
# =============================================================================

class TestDirectionalNumericalDerivative:
    """Audit central-difference numerical perturbation against consistent algorithmic tangent."""

    def test_directional_derivative_elastic_regime(self):
        """Relative error < 1e-6 in pure elastic regime for h=1e-6 and h=1e-7."""
        mat = _make_law32(e=205000.0, nu=0.29, a_yield=450.0, b=1.0, n=0.0)
        deps_base = np.array([[0.0002, -0.0001, 0.00015]])
        sig_init = np.zeros((1, 3))
        epsp_init = np.zeros(1)

        sig_c, epsp_c = law32_hill.shell_update(mat, sig_init.copy(), deps_base, epsp_init.copy(), dt=0.0)
        dep_incr = epsp_c - epsp_init
        c_ana = law32_hill.consistent_shell_tangent(mat, sig_c, epsp_c, dep_incr, symmetric=False)[0]

        for h in (1e-6, 1e-7):
            c_num = _compute_numerical_tangent(mat, sig_init, deps_base, epsp_init, h=h)
            rel_err = np.linalg.norm(c_num - c_ana) / np.linalg.norm(c_ana)
            assert rel_err < 1e-6, f"Elastic numerical derivative error {rel_err:.3e} exceeds 1e-6 for h={h}"

    def test_directional_derivative_plastic_flow_regime(self):
        """Relative error < 1e-4 in plastic flow regime for h=1e-6 and h=1e-7."""
        mat = _make_law32(
            e=210000.0, nu=0.3, a_yield=300.0, b=1.0, n=0.0,
            r00=1.6, r45=1.2, r90=1.9,
        )
        deps_base = np.array([[0.004, -0.0012, 0.0025]])
        sig_init = np.array([[120.0, 30.0, 20.0]])
        epsp_init = np.array([0.001])

        sig_c, epsp_c = law32_hill.shell_update(mat, sig_init.copy(), deps_base, epsp_init.copy(), dt=0.0)
        dep_incr = epsp_c - epsp_init
        assert dep_incr[0] > 0.0, "Test setup must undergo plastic deformation"

        c_ana = law32_hill.consistent_shell_tangent(mat, sig_c, epsp_c, dep_incr, symmetric=False)[0]

        for h in (1e-6, 1e-7):
            c_num = _compute_numerical_tangent(mat, sig_init, deps_base, epsp_init, h=h)
            rel_err = np.linalg.norm(c_num - c_ana) / np.linalg.norm(c_ana)
            assert rel_err < 1e-4, f"Plastic numerical derivative error {rel_err:.3e} exceeds 1e-4 for h={h}"

    def test_directional_derivative_arbitrary_perturbation_directions(self):
        """Directional derivatives along non-coordinate direction vectors match to < 1e-4."""
        mat = _make_law32(
            e=210000.0, nu=0.3, a_yield=320.0, b=1.0, n=0.0,
            r00=1.4, r45=1.1, r90=1.7,
        )
        deps_base = np.array([[0.0035, -0.001, 0.002]])
        sig_init = np.array([[150.0, -40.0, 25.0]])
        epsp_init = np.array([0.0015])

        sig_c, epsp_c = law32_hill.shell_update(mat, sig_init.copy(), deps_base, epsp_init.copy(), dt=0.0)
        dep_incr = epsp_c - epsp_init
        c_ana = law32_hill.consistent_shell_tangent(mat, sig_c, epsp_c, dep_incr, symmetric=False)[0]

        directions = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.6, -0.4, 0.3]),
            np.array([-0.3, 0.7, -0.2]),
        ]
        h = 1e-7

        for v in directions:
            v_norm = v / np.linalg.norm(v)
            dp = deps_base + h * v_norm
            dm = deps_base - h * v_norm

            sp, _ = law32_hill.shell_update(mat, sig_init.copy(), dp, epsp_init.copy(), dt=0.0)
            sm, _ = law32_hill.shell_update(mat, sig_init.copy(), dm, epsp_init.copy(), dt=0.0)

            d_sig_num = (sp[0, :3] - sm[0, :3]) / (2.0 * h)
            d_sig_ana = c_ana @ v_norm

            rel_err = np.linalg.norm(d_sig_num - d_sig_ana) / np.linalg.norm(d_sig_ana)
            assert rel_err < 1e-4, f"Directional derivative error {rel_err:.3e} exceeds 1e-4 for dir={v}"

    def test_directional_derivative_step_size_convergence(self):
        """Numerical error decreases predictably as perturbation h is refined."""
        mat = _make_law32(e=210000.0, nu=0.3, a_yield=300.0, b=1.0, n=0.0)
        deps_base = np.array([[0.003, -0.001, 0.002]])
        sig_init = np.array([[100.0, 20.0, 10.0]])
        epsp_init = np.array([0.0005])

        sig_c, epsp_c = law32_hill.shell_update(mat, sig_init.copy(), deps_base, epsp_init.copy(), dt=0.0)
        dep_incr = epsp_c - epsp_init
        c_ana = law32_hill.consistent_shell_tangent(mat, sig_c, epsp_c, dep_incr, symmetric=False)[0]

        err_1e5 = np.linalg.norm(_compute_numerical_tangent(mat, sig_init, deps_base, epsp_init, h=1e-5) - c_ana)
        err_1e6 = np.linalg.norm(_compute_numerical_tangent(mat, sig_init, deps_base, epsp_init, h=1e-6) - c_ana)
        err_1e7 = np.linalg.norm(_compute_numerical_tangent(mat, sig_init, deps_base, epsp_init, h=1e-7) - c_ana)

        assert err_1e6 < err_1e5
        assert err_1e7 < err_1e6

    def test_elastic_unloading_kuhn_tucker(self):
        """Unloading with reversed strain yields exactly the elastic membrane tangent."""
        mat = _make_law32(e=210000.0, nu=0.3, a_yield=300.0, b=1.0, n=0.0)
        sig = np.array([[250.0, 0.0, 0.0]])
        epsp = np.array([0.005])
        deps_unload = np.array([[-0.001, 0.0, 0.0]])

        sig_new, epsp_new = law32_hill.shell_update(mat, sig.copy(), deps_unload, epsp.copy(), dt=0.0)
        dep_incr = epsp_new - epsp
        assert dep_incr[0] == 0.0, "Elastic unloading must produce zero plastic increment"

        c_tan = law32_hill.consistent_shell_tangent(mat, sig_new, epsp_new, dep_incr)[0]
        c_el = law32_hill.shell_membrane_tangent(mat)
        np.testing.assert_allclose(c_tan, c_el, atol=1e-12)


# =============================================================================
# 2. Major Symmetry Verification
# =============================================================================

class TestMajorSymmetry:
    """Audit C_ij = C_ji for all i, j in [0, 1, 2]."""

    def test_elastic_membrane_tangent_symmetry(self):
        """Elastic plane-stress tangent is symmetric to machine precision."""
        mat = _make_law32(e=210000.0, nu=0.3)
        c_el = law32_hill.shell_membrane_tangent(mat)
        assert c_el.shape == (3, 3)
        np.testing.assert_allclose(c_el, c_el.T, atol=1e-14)
        for i in range(3):
            for j in range(3):
                assert math.isclose(c_el[i, j], c_el[j, i], abs_tol=1e-12)

    def test_algorithmic_tangent_symmetrized(self):
        """consistent_shell_tangent(..., symmetric=True) satisfies C_ij == C_ji strictly."""
        mat = _make_law32(
            e=210000.0, nu=0.3, a_yield=300.0, b=0.02, n=0.25,
            r00=1.8, r45=1.2, r90=2.1,
        )
        test_states = [
            np.array([[220.0, -80.0, 30.0]]),
            np.array([[150.0, 150.0, 0.0]]),
            np.array([[0.0, 0.0, 90.0]]),
            np.array([[280.0, 40.0, -35.0]]),
        ]
        epsp = np.array([0.005])
        epsp_incr = np.array([0.001])

        for sig in test_states:
            c_sym = law32_hill.consistent_shell_tangent(
                mat, sig, epsp, epsp_incr, symmetric=True
            )[0]
            asym_max = np.max(np.abs(c_sym - c_sym.T))
            assert asym_max < 1e-12, f"Asymmetry {asym_max:.2e} exceeds 1e-12 for stress {sig}"
            for i in range(3):
                for j in range(3):
                    assert c_sym[i, j] == pytest.approx(c_sym[j, i], abs=1e-12)

    def test_symmetry_across_orientation_angles(self):
        """Symmetry holds under material orientation angles theta in [0, 30, 45, 60, 90, 120]."""
        mat = _make_law32(
            e=210000.0, nu=0.3, a_yield=290.0, b=1.0, n=0.0,
            r00=1.7, r45=1.3, r90=2.0,
        )
        angles = [0.0, math.pi / 6.0, math.pi / 4.0, math.pi / 3.0, math.pi / 2.0, 2.0 * math.pi / 3.0]
        sig = np.array([[210.0, 60.0, 20.0]])
        epsp = np.array([0.003])
        epsp_incr = np.array([0.0008])

        for th in angles:
            extra = {"theta": th}
            c_sym = law32_hill.consistent_shell_tangent(
                mat, sig, epsp, epsp_incr, extra=extra, symmetric=True
            )[0]
            np.testing.assert_allclose(c_sym, c_sym.T, atol=1e-12)


# =============================================================================
# 3. Positive Definiteness Verification
# =============================================================================

class TestPositiveDefiniteness:
    """Verify eigenvalues are positive in stable elastic and hardening plastic regimes."""

    def test_elastic_regime_eigenvalues_positive(self):
        """All 3 eigenvalues of the elastic tangent are strictly positive."""
        for nu in (0.1, 0.25, 0.3, 0.45, 0.49):
            mat = _make_law32(e=200000.0, nu=nu)
            c_el = law32_hill.shell_membrane_tangent(mat)
            eigvals = np.linalg.eigvalsh(c_el)
            assert np.all(eigvals > 0.0), f"Non-positive eigenvalues {eigvals} at nu={nu}"
            # Analytical eigenvalues: E/(1-nu), E/(1+nu) = 2G, G
            e1 = mat.params["E"] / (1.0 - nu)
            e2 = mat.params["E"] / (1.0 + nu)
            e3 = mat.params["G"]
            expected = np.sort([e1, e2, e3])
            np.testing.assert_allclose(np.sort(eigvals), expected, rtol=1e-10)

    def test_hardening_plastic_regime_eigenvalues_positive(self):
        """In hardening plastic flow (H > 0), all eigenvalues are strictly positive."""
        mat = _make_law32(
            e=210000.0, nu=0.3, a_yield=300.0, b=0.01, n=0.3,
            r00=1.5, r45=1.2, r90=1.8,
        )
        deps_list = [
            np.array([[0.004, -0.0012, 0.0]]),
            np.array([[0.003, 0.003, 0.0]]),
            np.array([[0.0, 0.0, 0.006]]),
            np.array([[0.003, -0.001, 0.002]]),
        ]

        for deps in deps_list:
            sig = np.zeros((1, 3))
            extra = {"hardening": True}
            s_out, ep_out = law32_hill.shell_update(mat, sig, deps, dt=0.0, extra=extra)
            c_tan = law32_hill.consistent_shell_tangent(mat, s_out, ep_out, ep_out, extra=extra, symmetric=False)[0]

            eigvals = np.linalg.eigvals(c_tan)
            real_eigs = np.real(eigvals)
            assert np.all(real_eigs > 0.0), f"Negative/zero eigenvalue found: {real_eigs} for deps={deps}"

    def test_perfect_plasticity_eigenvalues_non_negative(self):
        """In perfect plasticity (H = 0), eigenvalues are non-negative (lambda_min >= -1e-8)."""
        mat = _make_law32(
            e=210000.0, nu=0.3, a_yield=300.0, b=1.0, n=0.0,
            r00=1.6, r45=1.2, r90=2.0,
        )
        deps = np.array([[0.005, -0.0015, 0.002]])
        s_out, ep_out = law32_hill.shell_update(mat, np.zeros((1, 3)), deps, dt=0.0)
        c_tan = law32_hill.consistent_shell_tangent(mat, s_out, ep_out, ep_out, symmetric=False)[0]

        eigvals = np.sort(np.real(np.linalg.eigvals(c_tan)))
        # Minimum eigenvalue corresponds to the flow direction (zero stiffness at constant flow stress)
        assert eigvals[0] >= -1e-8, f"Min eigenvalue {eigvals[0]} below zero threshold"
        # The remaining two eigenvalues remain strictly positive elastic/tangential modes
        assert eigvals[1] > 1000.0
        assert eigvals[2] > 1000.0


# =============================================================================
# 4. Multi-Axial Strain States
# =============================================================================

class TestMultiAxialStrainStates:
    """Audit consistent tangent under 6 canonical multi-axial plane-stress deformation states."""

    @pytest.fixture
    def aniso_mat(self) -> Material:
        return _make_law32(
            e=210000.0, nu=0.3, a_yield=300.0, b=1.0, n=0.0,
            r00=1.8, r45=1.2, r90=2.2,
        )

    def test_multiaxial_uniaxial_tension_fiber_0deg(self, aniso_mat):
        """Uniaxial tension along fiber direction (theta = 0 deg)."""
        nu = 0.3
        deps = np.array([[0.005, -nu * 0.005, 0.0]])
        extra = {"theta": 0.0}

        sig_c, epsp_c = law32_hill.shell_update(aniso_mat, np.zeros((1, 3)), deps, dt=0.0, extra=extra)
        dep_incr = epsp_c
        c_ana = law32_hill.consistent_shell_tangent(aniso_mat, sig_c, epsp_c, dep_incr, extra=extra, symmetric=False)[0]
        c_num = _compute_numerical_tangent(aniso_mat, np.zeros((1, 3)), deps, extra=extra)

        rel_err = np.linalg.norm(c_num - c_ana) / np.linalg.norm(c_ana)
        assert rel_err < 1e-4, f"Error {rel_err:.3e} exceeds 1e-4 in 0 deg uniaxial tension"

    def test_multiaxial_uniaxial_tension_transverse_90deg(self, aniso_mat):
        """Uniaxial tension transverse to fiber (theta = 90 deg)."""
        nu = 0.3
        deps = np.array([[-nu * 0.005, 0.005, 0.0]])
        extra = {"theta": 0.0}

        sig_c, epsp_c = law32_hill.shell_update(aniso_mat, np.zeros((1, 3)), deps, dt=0.0, extra=extra)
        dep_incr = epsp_c
        c_ana = law32_hill.consistent_shell_tangent(aniso_mat, sig_c, epsp_c, dep_incr, extra=extra, symmetric=False)[0]
        c_num = _compute_numerical_tangent(aniso_mat, np.zeros((1, 3)), deps, extra=extra)

        rel_err = np.linalg.norm(c_num - c_ana) / np.linalg.norm(c_ana)
        assert rel_err < 1e-4, f"Error {rel_err:.3e} exceeds 1e-4 in 90 deg uniaxial tension"

    def test_multiaxial_uniaxial_tension_diagonal_45deg(self, aniso_mat):
        """Uniaxial tension at theta = 45 deg to rolling direction."""
        nu = 0.3
        deps = np.array([[0.005, -nu * 0.005, 0.0]])
        extra = {"theta": math.pi / 4.0}

        sig_c, epsp_c = law32_hill.shell_update(aniso_mat, np.zeros((1, 3)), deps, dt=0.0, extra=extra)
        dep_incr = epsp_c
        c_ana = law32_hill.consistent_shell_tangent(aniso_mat, sig_c, epsp_c, dep_incr, extra=extra, symmetric=False)[0]
        c_num = _compute_numerical_tangent(aniso_mat, np.zeros((1, 3)), deps, extra=extra)

        rel_err = np.linalg.norm(c_num - c_ana) / np.linalg.norm(c_ana)
        assert rel_err < 1e-4, f"Error {rel_err:.3e} exceeds 1e-4 in 45 deg uniaxial tension"

    def test_multiaxial_biaxial_tension(self, aniso_mat):
        """Equibiaxial tension (eps_xx == eps_yy)."""
        deps = np.array([[0.004, 0.004, 0.0]])

        sig_c, epsp_c = law32_hill.shell_update(aniso_mat, np.zeros((1, 3)), deps, dt=0.0)
        dep_incr = epsp_c
        c_ana = law32_hill.consistent_shell_tangent(aniso_mat, sig_c, epsp_c, dep_incr, symmetric=False)[0]
        c_num = _compute_numerical_tangent(aniso_mat, np.zeros((1, 3)), deps)

        rel_err = np.linalg.norm(c_num - c_ana) / np.linalg.norm(c_ana)
        assert rel_err < 1e-4, f"Error {rel_err:.3e} exceeds 1e-4 in biaxial tension"

    def test_multiaxial_pure_shear(self, aniso_mat):
        """Pure shear deformation (gamma_xy != 0, eps_xx = eps_yy = 0)."""
        deps = np.array([[0.0, 0.0, 0.008]])

        sig_c, epsp_c = law32_hill.shell_update(aniso_mat, np.zeros((1, 3)), deps, dt=0.0)
        dep_incr = epsp_c
        c_ana = law32_hill.consistent_shell_tangent(aniso_mat, sig_c, epsp_c, dep_incr, symmetric=False)[0]
        c_num = _compute_numerical_tangent(aniso_mat, np.zeros((1, 3)), deps)

        rel_err = np.linalg.norm(c_num - c_ana) / np.linalg.norm(c_ana)
        assert rel_err < 1e-4, f"Error {rel_err:.3e} exceeds 1e-4 in pure shear"

    def test_multiaxial_general_combined_in_plane(self, aniso_mat):
        """General combined in-plane strain state (eps_xx, eps_yy, gamma_xy all active)."""
        deps = np.array([[0.0035, -0.0015, 0.0025]])
        extra = {"theta": math.pi / 6.0}

        sig_c, epsp_c = law32_hill.shell_update(aniso_mat, np.zeros((1, 3)), deps, dt=0.0, extra=extra)
        dep_incr = epsp_c
        c_ana = law32_hill.consistent_shell_tangent(aniso_mat, sig_c, epsp_c, dep_incr, extra=extra, symmetric=False)[0]
        c_num = _compute_numerical_tangent(aniso_mat, np.zeros((1, 3)), deps, extra=extra)

        rel_err = np.linalg.norm(c_num - c_ana) / np.linalg.norm(c_ana)
        assert rel_err < 1e-4, f"Error {rel_err:.3e} exceeds 1e-4 in combined state"


# =============================================================================
# 5. Lankford Anisotropy Sensitivity
# =============================================================================

class TestLankfordAnisotropySensitivity:
    """Audit sensitivity of consistent tangent to Lankford anisotropy parameters R00, R45, R90."""

    def test_lankford_isotropic_steel(self):
        """Isotropic steel: R00 = R45 = R90 = 1.0 reduces Hill coefficients to von Mises."""
        mat = _make_law32(r00=1.0, r45=1.0, r90=1.0)
        p = mat.params
        assert p["A11"] == pytest.approx(1.0)
        assert p["A22"] == pytest.approx(1.0)
        assert p["A1122"] == pytest.approx(1.0)
        assert p["A12"] == pytest.approx(3.0)

        deps = np.array([[0.003, -0.001, 0.002]])
        s_out, ep_out = law32_hill.shell_update(mat, np.zeros((1, 3)), deps, dt=0.0)
        c_ana = law32_hill.consistent_shell_tangent(mat, s_out, ep_out, ep_out, symmetric=False)[0]
        c_num = _compute_numerical_tangent(mat, np.zeros((1, 3)), deps)

        rel_err = np.linalg.norm(c_num - c_ana) / np.linalg.norm(c_ana)
        assert rel_err < 1e-4

    def test_lankford_high_normal_anisotropy(self):
        """High normal anisotropy (deep drawing): R00 = R90 = 2.0, R45 = 1.5."""
        mat = _make_law32(r00=2.0, r45=1.5, r90=2.0)
        p = mat.params
        # R_bar = (2 + 2*1.5 + 2)/4 = 1.75; H = 1.75/2.75 = 7/11
        # A11 = A22 = (7/11) * (1 + 0.5) = 21/22 ~ 0.9545
        assert p["A11"] == pytest.approx(p["A22"])
        assert p["A11"] < 1.0

        deps = np.array([[0.004, -0.001, 0.0015]])
        s_out, ep_out = law32_hill.shell_update(mat, np.zeros((1, 3)), deps, dt=0.0)
        c_ana = law32_hill.consistent_shell_tangent(mat, s_out, ep_out, ep_out, symmetric=False)[0]
        c_num = _compute_numerical_tangent(mat, np.zeros((1, 3)), deps)

        rel_err = np.linalg.norm(c_num - c_ana) / np.linalg.norm(c_ana)
        assert rel_err < 1e-4

    def test_lankford_planar_anisotropy(self):
        """Planar anisotropy: R00 = 1.8, R45 = 1.2, R90 = 2.2."""
        mat = _make_law32(r00=1.8, r45=1.2, r90=2.2)
        p = mat.params
        assert p["A11"] != pytest.approx(p["A22"])

        deps = np.array([[0.003, -0.0008, 0.002]])
        s_out, ep_out = law32_hill.shell_update(mat, np.zeros((1, 3)), deps, dt=0.0)
        c_ana = law32_hill.consistent_shell_tangent(mat, s_out, ep_out, ep_out, symmetric=False)[0]
        c_num = _compute_numerical_tangent(mat, np.zeros((1, 3)), deps)

        rel_err = np.linalg.norm(c_num - c_ana) / np.linalg.norm(c_ana)
        assert rel_err < 1e-4

    def test_lankford_yield_normalization_flag(self):
        """i_yield = 1 normalizes A11 to exactly 1.0 and scales all other Hill coefficients."""
        mat_norm = _make_law32(r00=1.6, r45=1.1, r90=2.4, i_yield=1)
        assert mat_norm.params["A11"] == pytest.approx(1.0)

        deps = np.array([[0.004, -0.001, 0.002]])
        s_out, ep_out = law32_hill.shell_update(mat_norm, np.zeros((1, 3)), deps, dt=0.0)
        c_ana = law32_hill.consistent_shell_tangent(mat_norm, s_out, ep_out, ep_out, symmetric=False)[0]
        c_num = _compute_numerical_tangent(mat_norm, np.zeros((1, 3)), deps)

        rel_err = np.linalg.norm(c_num - c_ana) / np.linalg.norm(c_ana)
        assert rel_err < 1e-4


# =============================================================================
# 6. Multi-Element Vectorization Consistency
# =============================================================================

class TestMultiElementVectorizationConsistency:
    """Verify tangent evaluations match identically whether evaluated single-element or batched."""

    @pytest.mark.parametrize("n_elements", [16, 64])
    def test_batched_vs_serial_consistency(self, n_elements: int):
        """Batched (N, 3, 3) tangent matches serial (1, 3, 3) to machine precision."""
        mat = _make_law32(
            e=210000.0, nu=0.3, a_yield=300.0, b=0.02, n=0.25,
            r00=1.7, r45=1.3, r90=2.0,
        )
        rng = np.random.default_rng(42 + n_elements)

        sig_batch = rng.uniform(50.0, 220.0, size=(n_elements, 3))
        epsp_batch = rng.uniform(0.001, 0.02, size=n_elements)
        epsp_incr_batch = rng.uniform(0.0002, 0.003, size=n_elements)
        theta_batch = rng.uniform(-np.pi, np.pi, size=n_elements)
        extra_batch = {"theta": theta_batch, "hardening": True}

        # Batched evaluation
        d_batch = law32_hill.consistent_shell_tangent(
            mat, sig_batch, epsp_batch, epsp_incr_batch, extra=extra_batch
        )
        assert d_batch.shape == (n_elements, 3, 3)

        # Serial element-by-element evaluation
        for i in range(n_elements):
            ex_i = {"theta": theta_batch[i], "hardening": True}
            d_single = law32_hill.consistent_shell_tangent(
                mat, sig_batch[i:i+1], epsp_batch[i:i+1], epsp_incr_batch[i:i+1], extra=ex_i
            )
            np.testing.assert_allclose(d_batch[i], d_single[0], atol=1e-12)

    def test_single_element_1d_input_shape(self):
        """1D input arrays of shape (3,) produce shape (3, 3) tangent identical to (1, 3)."""
        mat = _make_law32(e=210000.0, nu=0.3, a_yield=300.0, b=1.0, n=0.0)
        sig_1d = np.array([180.0, -40.0, 25.0])
        epsp_val = 0.002
        dep_val = 0.0005
        extra = {"theta": 0.4}

        d_1d = law32_hill.consistent_shell_tangent(mat, sig_1d, epsp_val, dep_val, extra=extra)
        d_2d = law32_hill.consistent_shell_tangent(mat, sig_1d.reshape(1, 3), [epsp_val], [dep_val], extra=extra)

        assert d_1d.shape == (3, 3)
        assert d_2d.shape == (1, 3, 3)
        np.testing.assert_allclose(d_1d, d_2d[0], atol=1e-14)

    def test_mixed_elastic_and_plastic_batch(self):
        """Batches containing mixed elastic (deps_p = 0) and plastic elements compute correctly."""
        mat = _make_law32(e=210000.0, nu=0.3, a_yield=300.0, b=1.0, n=0.0)
        sig = np.array([
            [50.0, 10.0, 5.0],     # elastic
            [250.0, -40.0, 20.0],  # plastic
            [80.0, 20.0, 0.0],     # elastic
            [270.0, 60.0, -30.0],  # plastic
        ])
        epsp = np.array([0.0, 0.002, 0.0, 0.004])
        epsp_incr = np.array([0.0, 0.001, 0.0, 0.0015])

        d_batch = law32_hill.consistent_shell_tangent(mat, sig, epsp, epsp_incr)
        c_el = law32_hill.shell_membrane_tangent(mat)

        np.testing.assert_allclose(d_batch[0], c_el, atol=1e-12)
        assert not np.allclose(d_batch[1], c_el)
        np.testing.assert_allclose(d_batch[2], c_el, atol=1e-12)
        assert not np.allclose(d_batch[3], c_el)


# =============================================================================
# 7. Robustness, Failure Deletion, and Dispatcher Integration
# =============================================================================

class TestRobustnessAndDispatcher:
    """Audit boundary conditions, failed element deletion, and global dispatcher."""

    def test_empty_stress_input(self):
        """Empty input arrays return empty (0, 3, 3) tangent tensor."""
        mat = _make_law32()
        d_empty = law32_hill.consistent_shell_tangent(mat, np.empty((0, 3)))
        assert d_empty.shape == (0, 3, 3)

    def test_failed_element_zero_stiffness(self):
        """Failed elements (off = 0 or eps_p >= eps_max) have strictly zero tangent stiffness."""
        eps_max = 0.01
        mat = _make_law32(eps_max=eps_max)
        sig = np.array([[100.0, 50.0, 10.0], [200.0, 80.0, 20.0]])
        epsp = np.array([0.002, 0.015])  # element 1 exceeded eps_max
        epsp_incr = np.array([0.0005, 0.001])
        extra = {"off32": np.array([1.0, 0.0])}  # element 1 deleted

        d_tangent = law32_hill.consistent_shell_tangent(mat, sig, epsp, epsp_incr, extra=extra)
        assert not np.allclose(d_tangent[0], 0.0)
        np.testing.assert_allclose(d_tangent[1], 0.0, atol=1e-14)

    def test_global_shell_layer_tangent_dispatcher(self):
        """pyradioss.materials.shell_layer_tangent dispatches to LAW32 consistent tangent."""
        mat = _make_law32(a_yield=300.0, r00=1.5, r45=1.2, r90=1.8)
        sig = np.array([[180.0, -50.0, 25.0]])
        epsp = np.array([0.002])
        epsp_incr = np.array([0.0006])
        extra = {"theta": 0.3}

        d_direct = law32_hill.consistent_shell_tangent(mat, sig, epsp, epsp_incr, extra=extra)
        d_dispatched = shell_layer_tangent(mat, sig, epsp, epsp_incr, extra=extra)

        np.testing.assert_allclose(d_direct, d_dispatched, atol=1e-14)

    def test_maximum_stress_cap_zero_hardening_modulus(self):
        """When flow stress reaches sig_max, hardening slope is zeroed (H_eff = 0)."""
        sig_max = 350.0
        mat = _make_law32(a_yield=300.0, b=0.01, n=0.5, sig_max=sig_max)
        # Large plastic strain where A * (b + epsp)^n = 300 * sqrt(2.01) ~ 425 > sig_max = 350
        epsp = np.array([2.0])
        epsp_incr = np.array([0.001])
        sig = np.array([[350.0, 0.0, 0.0]])
        extra = {"hardening": True}

        c_capped = law32_hill.consistent_shell_tangent(mat, sig, epsp, epsp_incr, extra=extra, symmetric=False)[0]
        # At H_eff = 0, minimum eigenvalue must be zero
        eigvals = np.sort(np.real(np.linalg.eigvals(c_capped)))
        assert eigvals[0] == pytest.approx(0.0, abs=1e-8)
