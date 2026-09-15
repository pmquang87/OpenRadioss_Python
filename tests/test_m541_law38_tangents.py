"""Audit suite for Milestone M541: LAW38 / VISC_TAB consistent algorithmic tangents.

Auditor 2 in Wave 2 for M541: LAW38 Tangent & Numerical Derivative Auditor.
Rigorously audits the algorithmic consistent tangent stiffness tensor in
`pyradioss/materials/law38_visc_tab.py`:

1. Directional numerical derivative check:
   Central-difference numerical perturbation:
     C_{num, ijkl} = [sigma_{ij}(eps + h e_{kl}) - sigma_{ij}(eps - h e_{kl})] / (2h)
   Compare with analytical algorithmic tangent tensor for h = 1e-6 or 1e-7.
   Verify relative error is strictly bounded:
     - < 1e-5 in linear and quasi-linear small-strain regimes
     - < 1e-3 in nonlinear large-deformation regimes
2. Tangent symmetry:
   Verify major symmetry C_{ijkl} = C_{klij} (C == C.T in Voigt form)
   and minor symmetry C_{ijkl} = C_{jikl}.
3. Positive definiteness:
   Verify all eigenvalues are strictly positive in stable elastic and
   compressive deformation regimes.
4. Multi-axial strain states:
   - Pure hydrostatic compression (J < 1) with and without air pressure
   - Uniaxial tension and compression
   - Pure shear (gamma_xy, gamma_yz, gamma_zx)
   - General 3D mixed multi-axial strain
   - Rate-dependent states with dynamic viscosity
5. Batched vectorization consistency:
   Verify tangent evaluations match identically whether evaluated single-element
   (1, 6, 6) or batched (N, 6, 6) for N = 16 and N = 64.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

import pyradioss.materials as pm
from pyradioss.materials import law38_visc_tab as l38
from pyradioss.model.entities import Material


# =============================================================================
# Helper Fixtures & Builders
# =============================================================================

def _make_law38(
    e0: float = 100.0,
    nu_t: float = 0.2,
    nu_c: float = 0.2,
    rho0: float = 1.0,
    load_curve: Optional[Tuple[Any, Any]] = None,
    load_curves: Optional[list] = None,
    kcompair: int = 0,
    p0: float = 0.0,
    phi: float = 0.0,
    relaxp: float = 0.0,
    hyster: float = 1.0,
    beta: float = 0.0,
    iflag: int = 0,
    itotal: int = 0,
    efinal: Optional[float] = None,
    epsfin: float = 1.0,
    lamda: float = 1.0,
    viscosity: float = 1.0e30,
    eps_tab: Optional[list] = None,
    fscale_tab: Optional[list] = None,
    mat_id: int = 1,
    title: str = "AUDIT_LAW38",
    **kwargs: Any,
) -> Material:
    """Build a LAW38 material instance with specified parameters."""
    if load_curves is None:
        if load_curve is not None:
            load_curves = [load_curve]
        else:
            load_curves = [([0.0, 0.2, 0.5], [0.0, 20.0, 60.0])]

    params: Dict[str, Any] = {
        "MAT_E": e0,
        "MAT_NU": nu_t,
        "MAT_NUt": nu_c,
        "rho0": rho0,
        "load_curves": load_curves,
        "MAT_Kair": kcompair,
        "MAT_P0": p0,
        "MAT_POROS": phi,
        "MAT_PR": relaxp,
        "MAT_HYST": hyster,
        "MAT_RELX": beta,
        "MAT_IFLAG": iflag,
        "ITOTAL": itotal,
        "MAT_Efinal": efinal if efinal is not None else e0,
        "MAT_Epsfinal": epsfin,
        "MAT_Lamda": lamda,
        "MAT_MaxVisc": viscosity,
        **kwargs,
    }
    if eps_tab is not None:
        params["eps_tab"] = eps_tab
    if fscale_tab is not None:
        params["fscale_tab"] = fscale_tab

    rec = {
        "id": mat_id,
        "title": title,
        "rho0": rho0,
        "params": params,
    }
    return l38.build_law38(rec)


def _compute_numerical_tangent(
    mat: Material,
    deps0: np.ndarray,
    dt: float = 0.01,
    extra: Optional[Dict[str, Any]] = None,
    sig0: Optional[np.ndarray] = None,
    epsp0: Optional[np.ndarray] = None,
    h: float = 1e-6,
) -> np.ndarray:
    """Compute (n, 6, 6) numerical tangent tensor via central finite differences:

    C_{num, ijkl} = [sigma_{ij}(eps + h e_{kl}) - sigma_{ij}(eps - h e_{kl})] / (2h)
    """
    deps_arr = np.atleast_2d(np.asarray(deps0, dtype=float))
    n = deps_arr.shape[0]
    assert deps_arr.shape[1] == 6, f"Expected (n, 6), got {deps_arr.shape}"

    if sig0 is None:
        sig0 = np.zeros((n, 6), dtype=float)
    else:
        sig0 = np.atleast_2d(np.asarray(sig0, dtype=float))

    if epsp0 is None:
        epsp0 = np.zeros(n, dtype=float)

    C_num = np.zeros((n, 6, 6), dtype=float)

    for j in range(6):
        dp = deps_arr.copy()
        dm = deps_arr.copy()
        dp[:, j] += h
        dm[:, j] -= h

        extra_p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in extra.items()} if extra else None
        extra_m = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in extra.items()} if extra else None

        sp, _, _ = l38.solid_update(mat, sig0.copy(), dp, epsp=epsp0, dt=dt, extra=extra_p)
        sm, _, _ = l38.solid_update(mat, sig0.copy(), dm, epsp=epsp0, dt=dt, extra=extra_m)

        C_num[:, :, j] = (sp - sm) / (2.0 * h)

    return C_num


# =============================================================================
# 1. Mathematical Structure, Symmetry & Positive Definiteness
# =============================================================================

class TestMathematicalStructure:
    """Audit mathematical structure, symmetry, and positive definiteness."""

    def test_analytical_linear_elastic_structure(self):
        """Verify analytical Hookean structure in linear elastic limit:
        C11 = C22 = C33 = E(1 - nu) / ((1 + nu)(1 - 2*nu))
        C12 = C13 = C23 = E*nu / ((1 + nu)(1 - 2*nu))
        C44 = C55 = C66 = E / (2*(1 + nu))
        All other components = 0.
        """
        e0 = 200.0
        nu = 0.25
        mat = _make_law38(e0=e0, nu_t=nu, nu_c=nu, load_curve=([0.0, 1.0], [0.0, 200.0]))

        expected_c11 = e0 * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        expected_c12 = e0 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        expected_c44 = e0 / (2.0 * (1.0 + nu))

        C = l38.consistent_solid_tangent(mat, dt=0.01)
        assert C.shape == (1, 6, 6)

        # Diagonal normal terms
        for i in range(3):
            assert math.isclose(C[0, i, i], expected_c11, rel_tol=1e-5)
        # Off-diagonal normal terms
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert math.isclose(C[0, i, j], expected_c12, rel_tol=1e-5)
        # Shear diagonal terms
        for s in (3, 4, 5):
            assert math.isclose(C[0, s, s], expected_c44, rel_tol=1e-5)

        # Shear-normal cross couplings are zero in linear elastic isotropic limit
        for i in range(3):
            for s in (3, 4, 5):
                assert abs(C[0, i, s]) < 1e-6
                assert abs(C[0, s, i]) < 1e-6

    def test_tangent_symmetry_major_and_minor(self):
        """Verify major symmetry C_{ijkl} = C_{klij} and minor symmetry C_{ijkl} = C_{jikl}."""
        mat = _make_law38(
            e0=150.0, nu_t=0.22, nu_c=0.22,
            load_curve=([0.0, 0.2, 0.5], [0.0, 30.0, 90.0]),
            kcompair=1, p0=3.0, phi=0.1,
        )

        test_strains = [
            np.zeros((1, 6)),
            np.array([[-0.03, -0.03, -0.03, 0, 0, 0]]),
            np.array([[-0.04, 0.01, -0.02, 0.015, -0.01, 0.005]]),
            np.array([[0, 0, 0, 0.05, 0, 0]]),
        ]

        for deps in test_strains:
            C = l38.consistent_solid_tangent(mat, deps, dt=0.01, symmetric=True)
            # C[0] must equal C[0].T to machine precision
            sym_diff = np.max(np.abs(C[0] - C[0].T))
            assert sym_diff < 1e-10, f"Tangent asymmetry {sym_diff:.2e} on strain {deps}"

    def test_positive_definiteness_in_compressive_and_elastic_regimes(self):
        """Verify all 6 eigenvalues of the tangent are strictly positive in stable deformation."""
        mat = _make_law38(
            e0=120.0, nu_t=0.2, nu_c=0.2,
            load_curve=([0.0, 0.1, 0.4], [0.0, 12.0, 60.0]),
            kcompair=1, p0=2.5, phi=0.05,
        )

        regimes = [
            ("Zero strain", np.zeros((1, 6))),
            ("Moderate compression", np.array([[-0.05, -0.05, -0.05, 0, 0, 0]])),
            ("High compression J=0.75", np.array([[-0.1, -0.1, -0.1, 0, 0, 0]])),
            ("Uniaxial compression", np.array([[-0.06, 0, 0, 0, 0, 0]])),
            ("Pure shear", np.array([[0, 0, 0, 0.04, 0.02, -0.03]])),
            ("Mixed multi-axial", np.array([[-0.04, -0.02, 0.01, 0.02, -0.01, 0.015]])),
        ]

        for desc, deps in regimes:
            C = l38.consistent_solid_tangent(mat, deps, dt=0.01)
            eigvals = np.linalg.eigvalsh(C[0])
            min_eig = float(np.min(eigvals))
            assert min_eig > 0.0, (
                f"Negative/zero eigenvalue {min_eig:.3e} in regime {desc}: {eigvals}"
            )

    def test_shell_update_raises_not_implemented(self):
        """Verify shell_update raises NotImplementedError (solids only)."""
        mat = _make_law38()
        with pytest.raises(NotImplementedError, match="3D solid elements only"):
            l38.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)), None, 0.0)


# =============================================================================
# 2. Directional Numerical Derivative Verification
# =============================================================================

class TestDirectionalNumericalDerivative:
    """Audit central-difference perturbation check against algorithmic tangent."""

    def test_directional_derivative_linear_elastic(self):
        """Verify relative error is strictly < 1e-5 in uncompressed linear elastic regime."""
        mat = _make_law38(
            e0=180.0, nu_t=0.25, nu_c=0.25,
            load_curve=([0.0, 1.0], [0.0, 180.0]),
        )
        deps = np.zeros((1, 6))

        C_ana = l38.consistent_solid_tangent(mat, deps, dt=0.01, symmetric=True)
        C_num = _compute_numerical_tangent(mat, deps, dt=0.01, h=1e-6)

        diff = np.abs(C_ana[0] - C_num[0])
        scale = np.maximum(np.abs(C_ana[0]), 1.0)
        max_rel_err = float(np.max(diff / scale))

        assert max_rel_err < 1e-5, f"Linear elastic relative error {max_rel_err:.2e} >= 1e-5"

    def test_directional_derivative_hydrostatic_compression(self):
        """Verify relative error is strictly < 1e-5 in pure hydrostatic compression."""
        mat = _make_law38(
            e0=100.0, nu_t=0.2, nu_c=0.2,
            load_curve=([0.0, 0.2, 0.5], [0.0, 20.0, 60.0]),
        )
        deps = np.array([[-0.04, -0.04, -0.04, 0, 0, 0]])

        C_ana = l38.consistent_solid_tangent(mat, deps, dt=0.01, symmetric=True)
        C_num = _compute_numerical_tangent(mat, deps, dt=0.01, h=1e-6)

        diff = np.abs(C_ana[0] - C_num[0])
        scale = np.maximum(np.abs(C_ana[0]), 1.0)
        max_rel_err = float(np.max(diff / scale))

        assert max_rel_err < 1e-5, f"Hydrostatic relative error {max_rel_err:.2e} >= 1e-5"

    def test_directional_derivative_quasilinear_small_strain(self):
        """Verify relative error is strictly < 1e-5 in small-strain quasi-linear state."""
        mat = _make_law38(
            e0=120.0, nu_t=0.22, nu_c=0.22,
            load_curve=([0.0, 0.2, 0.5], [0.0, 24.0, 70.0]),
        )
        deps = np.array([[-1e-5, -1e-5, -1e-5, 0, 0, 0]])

        C_ana = l38.consistent_solid_tangent(mat, deps, dt=0.01, symmetric=True)
        C_num = _compute_numerical_tangent(mat, deps, dt=0.01, h=1e-6)

        diff = np.abs(C_ana[0] - C_num[0])
        scale = np.maximum(np.abs(C_ana[0]), 1.0)
        max_rel_err = float(np.max(diff / scale))

        assert max_rel_err < 1e-5, f"Quasi-linear relative error {max_rel_err:.2e} >= 1e-5"

    def test_directional_derivative_exact_directional_mode(self):
        """Verify exact directional derivative mode (symmetric=False) matches C_num to < 1e-6."""
        mat = _make_law38(
            e0=100.0, nu_t=0.2, nu_c=0.2,
            load_curve=([0.0, 0.2, 0.5], [0.0, 20.0, 60.0]),
        )
        deps = np.array([[-0.03, 0.01, -0.02, 0.02, -0.01, 0.015]])

        # In exact directional derivative mode, h=1e-7 matches C_num(h=1e-6) to < 1e-6
        C_dir = l38.consistent_solid_tangent(mat, deps, dt=0.01, symmetric=False, h=1e-7)
        C_num = _compute_numerical_tangent(mat, deps, dt=0.01, h=1e-6)

        diff = np.abs(C_dir[0] - C_num[0])
        scale = np.maximum(np.abs(C_dir[0]), 1.0)
        max_rel_err = float(np.max(diff / scale))

        assert max_rel_err < 1e-6, f"Directional derivative relative error {max_rel_err:.2e} >= 1e-6"

    def test_directional_derivative_nonlinear_compression_bounded(self):
        """Verify relative error is bounded < 1e-3 in nonlinear compression."""
        mat = _make_law38(
            e0=100.0, nu_t=0.2, nu_c=0.2,
            load_curve=([0.0, 0.2, 0.5], [0.0, 20.0, 60.0]),
        )
        deps = np.array([[-0.05, -0.05, -0.05, 0, 0, 0]])

        C_ana = l38.consistent_solid_tangent(mat, deps, dt=0.01, symmetric=True)
        C_num = _compute_numerical_tangent(mat, deps, dt=0.01, h=1e-6)

        diff = np.abs(C_ana[0] - C_num[0])
        scale = np.maximum(np.abs(C_ana[0]), 1.0)
        max_rel_err = float(np.max(diff / scale))

        assert max_rel_err < 1e-3, f"Nonlinear compression relative error {max_rel_err:.2e} >= 1e-3"

    def test_directional_derivative_nonlinear_tension_bounded(self):
        """Verify relative error is bounded < 1e-3 in nonlinear tension stiffening regime."""
        mat = _make_law38(
            e0=100.0, efinal=300.0, epsfin=0.5, lamda=2.0,
            load_curve=([0.0, 1.0], [0.0, 100.0]),
        )
        deps = np.array([[+0.05, +0.05, +0.05, 0, 0, 0]])

        C_ana = l38.consistent_solid_tangent(mat, deps, dt=0.01, symmetric=True)
        C_num = _compute_numerical_tangent(mat, deps, dt=0.01, h=1e-6)

        diff = np.abs(C_ana[0] - C_num[0])
        scale = np.maximum(np.abs(C_ana[0]), 1.0)
        max_rel_err = float(np.max(diff / scale))

        assert max_rel_err < 1e-3, f"Nonlinear tension relative error {max_rel_err:.2e} >= 1e-3"


# =============================================================================
# 3. Multi-Axial Strain States
# =============================================================================

class TestMultiAxialStrainStates:
    """Audit multi-axial strain states: hydrostatic, uniaxial, shear, 3D mixed, viscous."""

    def test_pure_hydrostatic_compression_without_air(self):
        """Pure hydrostatic compression J < 1 without air pressure."""
        mat = _make_law38(
            e0=100.0, nu_t=0.2, nu_c=0.2,
            load_curve=([0.0, 0.2, 0.5], [0.0, 20.0, 60.0]),
            kcompair=0,
        )
        deps = np.array([[-0.04, -0.04, -0.04, 0, 0, 0]])

        C = l38.consistent_solid_tangent(mat, deps, dt=0.01)
        assert math.isclose(C[0, 0, 0] - C[0, 0, 1], 2.0 * C[0, 3, 3], rel_tol=1e-4)
        eigvals = np.linalg.eigvalsh(C[0])
        assert np.all(eigvals > 0.0)

    def test_pure_hydrostatic_compression_with_air_pressure(self):
        """Pure hydrostatic compression J < 1 with closed-cell air pressure (P0 > 0)."""
        mat_no_air = _make_law38(
            e0=100.0, nu_t=0.2, nu_c=0.2,
            load_curve=([0.0, 0.2, 0.5], [0.0, 20.0, 60.0]),
            kcompair=0,
        )
        mat_air = _make_law38(
            e0=100.0, nu_t=0.2, nu_c=0.2,
            load_curve=([0.0, 0.2, 0.5], [0.0, 20.0, 60.0]),
            kcompair=1, p0=10.0, phi=0.1,
        )
        deps = np.array([[-0.05, -0.05, -0.05, 0, 0, 0]])

        C_no_air = l38.consistent_solid_tangent(mat_no_air, deps, dt=0.01)
        C_air = l38.consistent_solid_tangent(mat_air, deps, dt=0.01)

        assert C_air[0, 0, 0] > C_no_air[0, 0, 0]
        assert C_air[0, 0, 1] > C_no_air[0, 0, 1]
        assert math.isclose(C_air[0, 3, 3], C_no_air[0, 3, 3], rel_tol=1e-3)
        assert np.all(np.linalg.eigvalsh(C_air[0]) > 0.0)

    def test_air_pressure_relaxation_decay(self):
        """Air pressure stiffness decays over time according to exp(-relaxp * t)."""
        mat = _make_law38(
            e0=100.0, nu_t=0.2, nu_c=0.2,
            kcompair=1, p0=5.0, phi=0.1, relaxp=10.0,
        )
        deps = np.array([[-0.05, -0.05, -0.05, 0, 0, 0]])

        C_t0 = l38.consistent_solid_tangent(mat, deps, dt=0.01, extra={"time": 0.0})
        C_t1 = l38.consistent_solid_tangent(mat, deps, dt=0.01, extra={"time": 0.5})

        assert C_t0[0, 0, 0] > C_t1[0, 0, 0]

    def test_uniaxial_compression_and_tension(self):
        """Uniaxial compression (eps_xx < 0) and uniaxial tension (eps_xx > 0)."""
        mat = _make_law38(
            e0=100.0, efinal=250.0, nu_t=0.2, nu_c=0.2,
            load_curve=([0.0, 0.2, 0.5], [0.0, 20.0, 60.0]),
        )

        deps_comp = np.array([[-0.04, 0, 0, 0, 0, 0]])
        C_comp = l38.consistent_solid_tangent(mat, deps_comp, dt=0.01)
        assert np.all(np.linalg.eigvalsh(C_comp[0]) > 0.0)
        assert C_comp[0, 0, 0] > 0.0

        deps_tens = np.array([[+0.04, 0, 0, 0, 0, 0]])
        C_tens = l38.consistent_solid_tangent(mat, deps_tens, dt=0.01)
        assert np.all(np.linalg.eigvalsh(C_tens[0]) > 0.0)
        assert C_tens[0, 0, 0] > 0.0

    @pytest.mark.parametrize("shear_idx,label", [(3, "xy"), (4, "yz"), (5, "zx")])
    def test_pure_shear_modes(self, shear_idx: int, label: str):
        """Pure shear modes gamma_xy, gamma_yz, gamma_zx."""
        mat = _make_law38(
            e0=100.0, nu_t=0.2, nu_c=0.2,
            load_curve=([0.0, 0.2, 0.5], [0.0, 20.0, 60.0]),
        )
        deps = np.zeros((1, 6))
        deps[0, shear_idx] = 0.04

        C = l38.consistent_solid_tangent(mat, deps, dt=0.01)
        assert C[0, shear_idx, shear_idx] > 0.0, f"Non-positive shear stiffness for {label}"
        eigvals = np.linalg.eigvalsh(C[0])
        assert np.all(eigvals > 0.0), f"Non-positive eigenvalues under pure shear {label}: {eigvals}"

    def test_general_3d_mixed_multiaxial_strain(self):
        """General 3D mixed multi-axial strain with simultaneous normal and shear components."""
        mat = _make_law38(
            e0=100.0, nu_t=0.2, nu_c=0.2,
            load_curve=([0.0, 0.2, 0.5], [0.0, 20.0, 60.0]),
            kcompair=1, p0=3.0, phi=0.1,
        )
        deps = np.array([[-0.03, 0.015, -0.025, 0.02, -0.012, 0.018]])

        C = l38.consistent_solid_tangent(mat, deps, dt=0.01)
        assert C.shape == (1, 6, 6)
        eigvals = np.linalg.eigvalsh(C[0])
        assert np.all(eigvals > 0.0)
        assert np.allclose(C[0], C[0].T, atol=1e-10)

    def test_rate_dependent_dynamic_viscosity(self):
        """Rate-dependent states with dynamic viscosity and strain-rate tables."""
        curves = [
            ([0.0, 0.2, 0.5], [0.0, 20.0, 60.0]),
            ([0.0, 0.2, 0.5], [0.0, 30.0, 90.0]),
            ([0.0, 0.2, 0.5], [0.0, 45.0, 135.0]),
        ]
        mat = _make_law38(
            e0=100.0, nu_t=0.2, nu_c=0.2,
            load_curves=curves,
            eps_tab=[0.0, 10.0, 100.0],
            fscale_tab=[1.0, 1.0, 1.0],
            viscosity=100.0,
        )

        deps_fast = np.array([[-0.03, -0.01, 0.005, 0.02, 0, 0]])
        C_slow = l38.consistent_solid_tangent(mat, deps_fast, dt=0.1)
        C_fast = l38.consistent_solid_tangent(mat, deps_fast, dt=0.001)

        assert C_fast[0, 0, 0] > C_slow[0, 0, 0]
        assert np.all(np.linalg.eigvalsh(C_fast[0]) > 0.0)


# =============================================================================
# 4. Batched Vectorization Consistency
# =============================================================================

class TestBatchedVectorizationConsistency:
    """Audit single-element vs batched N=16 and N=64 element consistency."""

    def test_batched_consistency_n16(self):
        """Verify (16, 6, 6) batched tangent matches 16 single-element evaluations."""
        mat = _make_law38(
            e0=100.0, nu_t=0.2, nu_c=0.2,
            load_curve=([0.0, 0.2, 0.5], [0.0, 20.0, 60.0]),
            kcompair=1, p0=2.0, phi=0.1,
        )
        np.random.seed(42)
        deps_batch = np.random.randn(16, 6) * 0.02

        C_batch = l38.consistent_solid_tangent(mat, deps_batch, dt=0.01)
        assert C_batch.shape == (16, 6, 6)

        for i in range(16):
            C_single = l38.consistent_solid_tangent(mat, deps_batch[i:i+1], dt=0.01)
            np.testing.assert_allclose(
                C_batch[i], C_single[0], atol=1e-12,
                err_msg=f"Element {i} in N=16 batch differs from single evaluation"
            )

    def test_batched_consistency_n64(self):
        """Verify (64, 6, 6) batched tangent matches single-element evaluations."""
        mat = _make_law38(
            e0=120.0, nu_t=0.25, nu_c=0.25,
            load_curve=([0.0, 0.2, 0.5], [0.0, 25.0, 75.0]),
        )
        np.random.seed(123)
        deps_batch = np.random.randn(64, 6) * 0.015

        C_batch = l38.consistent_solid_tangent(mat, deps_batch, dt=0.01)
        assert C_batch.shape == (64, 6, 6)

        for i in (0, 15, 32, 47, 63):
            C_single = l38.consistent_solid_tangent(mat, deps_batch[i:i+1], dt=0.01)
            np.testing.assert_allclose(
                C_batch[i], C_single[0], atol=1e-12,
                err_msg=f"Element {i} in N=64 batch differs from single evaluation"
            )

    def test_batched_symmetry_and_definiteness(self):
        """Verify all elements in a large batch are symmetric and positive definite."""
        mat = _make_law38(e0=100.0, nu_t=0.2, nu_c=0.2)
        deps_batch = np.random.randn(32, 6) * 0.01
        C_batch = l38.consistent_solid_tangent(mat, deps_batch, dt=0.01)

        for i in range(32):
            np.testing.assert_allclose(C_batch[i], C_batch[i].T, atol=1e-10)
            eigvals = np.linalg.eigvalsh(C_batch[i])
            assert np.all(eigvals > 0.0)


# =============================================================================
# 5. Defensive Edge Cases & Dispatch Wiring
# =============================================================================

class TestDefensiveEdgeCases:
    """Audit empty inputs, zero/negative dt, octahedral formulation, and dispatch."""

    def test_empty_array_handling(self):
        """Empty input arrays return empty (0, 6, 6) tangent tensor."""
        mat = _make_law38()
        deps_empty = np.empty((0, 6), dtype=float)
        C = l38.consistent_solid_tangent(mat, deps_empty)
        assert C.shape == (0, 6, 6)

    def test_zero_and_negative_dt(self):
        """Zero or negative dt handled defensively without division-by-zero."""
        mat = _make_law38()
        deps = np.array([[-0.02, -0.02, -0.02, 0, 0, 0]])

        C_zero = l38.consistent_solid_tangent(mat, deps, dt=0.0)
        assert C_zero.shape == (1, 6, 6)
        assert np.all(np.linalg.eigvalsh(C_zero[0]) > 0.0)

        C_neg = l38.consistent_solid_tangent(mat, deps, dt=-1e-4)
        assert C_neg.shape == (1, 6, 6)
        assert np.all(np.linalg.eigvalsh(C_neg[0]) > 0.0)

    def test_octahedral_formulation_iflag1(self):
        """Octahedral strain formulation branch (iflag=1) tangent."""
        e0 = 160.0
        nu = 0.25
        mat = _make_law38(e0=e0, nu_t=nu, nu_c=nu, iflag=1)
        deps = np.array([[-0.03, 0.01, -0.02, 0.01, 0, 0]])

        C = l38.consistent_solid_tangent(mat, deps, dt=0.01)
        assert C.shape == (1, 6, 6)
        assert np.allclose(C[0], C[0].T, atol=1e-10)
        assert np.all(np.linalg.eigvalsh(C[0]) > 0.0)

    def test_materials_dispatch_integration(self):
        """Verify dispatch through pyradioss.materials.solid_tangent."""
        mat = _make_law38(e0=100.0, nu_t=0.2, nu_c=0.2)
        sig = np.zeros((3, 6))
        epsp = np.zeros(3)
        deps = np.zeros((3, 6))

        C = pm.solid_tangent(mat, sig, epsp, deps)
        assert C.shape == (3, 6, 6)
        for i in range(3):
            assert np.allclose(C[i], C[i].T, atol=1e-10)
            assert np.all(np.linalg.eigvalsh(C[i]) > 0.0)
