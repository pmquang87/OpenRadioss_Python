"""
Milestone M556: Algorithmic Consistent Tangent Verification Suite for /MAT/LAW21 (/MAT/DPRAG).

Rigorously verifies the algorithmic consistent solid tangent stiffness tensors in
`pyradioss/materials/law21_dprag.py` across:
1. Solid algorithmic tangent D^alg = d(sigma)/d(deps) of shape (nel, 6, 6) or (6, 6),
   verified against independent central finite difference perturbations:
     D_num[:, j] = (sigma(deps + h*e_j) - sigma(deps - h*e_j)) / (2*h)
   with step sizes h in [1e-7, 1e-6], achieving relative error < 1e-4 in plastic regimes
   and < 1e-8 in elastic regimes.
2. Verification across all physical regimes:
   - Linear elastic regime (J2 < G0, exact Hookean matrix with Kt and G)
   - Linear Drucker-Prager yield regime (A2 = 0, A1 > 0)
   - Parabolic Drucker-Prager yield regime (A2 > 0)
   - von Mises cap regime (G0 = Amax, dG0/dPtot = 0)
   - Root cutoff regime (Ptot <= P*, deviatoric tangent vanishes identically)
   - Tensile fracture cutoff regime (P <= Pmin)
   - Compaction EOS loading regime (Kt = dP/dmu)
   - Compaction EOS hysteretic unloading regime (Kt = Kunload)
   - Multi-axial loading states: uniaxial tension, uniaxial compression, equibiaxial tension,
     triaxial hydrostatic compression, pure shear, mixed normal-shear states.
   - Deactivated elements (off = 0.0, tangent vanishes identically).
3. Batch vectorization (n = 1, 8, 32 elements simultaneously).
4. Calling conventions: positional and keyword invocations, symmetric=True producing D = D^T,
   and package-level dispatch via materials.solid_tangent and materials.consistent_solid_tangent.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pytest

import pyradioss.materials as materials
from pyradioss.materials.law21_dprag import (
    build_law21,
    solid_update_law21,
    sound_speed_solid_law21,
    tangent_law21_solid,
    consistent_solid_tangent,
    solid_tangent,
    shell_update_law21,
    _copy_extra,
    _ensure_params,
)
from pyradioss.model.entities import Material


# =============================================================================
# Helper Utilities: Independent Central Finite Difference Numerical Tangents
# =============================================================================

def _compute_num_solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray],
    deps: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
    dt: float = 1.0,
    epsp: Optional[Union[float, np.ndarray]] = None,
    h: float = 1e-7,
) -> np.ndarray:
    """Compute (NEL, 6, 6) or (6, 6) numerical solid tangent via central finite difference:
        D_num[:, :, j] = (sigma(deps + h * e_j) - sigma(deps - h * e_j)) / (2 * h)
    """
    deps_arr = np.asarray(deps, dtype=float)
    single_deps = (deps_arr.ndim == 1)
    if single_deps:
        deps_arr = deps_arr.reshape(1, -1)
    nel = deps_arr.shape[0]

    if sig is not None:
        s0 = np.asarray(sig, dtype=float).copy()
        single_sig = (s0.ndim == 1)
        if single_sig:
            s0 = s0.reshape(1, -1)
        if s0.shape[0] == 1 and nel > 1:
            s0 = np.repeat(s0, nel, axis=0)
    else:
        single_sig = True
        s0 = np.zeros((nel, deps_arr.shape[1]), dtype=float)

    single = single_deps and single_sig

    if epsp is not None:
        ep_arr = np.asarray(epsp, dtype=float).flatten()
        if len(ep_arr) == 1 and nel > 1:
            ep_arr = np.full(nel, ep_arr[0])
    else:
        ep_arr = np.zeros(nel, dtype=float)

    D_num = np.zeros((nel, 6, 6), dtype=float)
    for j in range(6):
        ej = np.zeros_like(deps_arr)
        ej[:, j] = h
        ex_p = _copy_extra(extra)
        ex_m = _copy_extra(extra)
        sp = solid_update_law21(mat, s0.copy(), deps=deps_arr + ej, extra=ex_p, dt=dt, epsp=ep_arr.copy(), return_tuple=False)
        sm = solid_update_law21(mat, s0.copy(), deps=deps_arr - ej, extra=ex_m, dt=dt, epsp=ep_arr.copy(), return_tuple=False)
        sp_stress = sp[0] if isinstance(sp, tuple) else sp
        sm_stress = sm[0] if isinstance(sm, tuple) else sm
        if sp_stress.ndim == 1:
            sp_stress = sp_stress.reshape(1, -1)
            sm_stress = sm_stress.reshape(1, -1)
        D_num[:, :, j] = (sp_stress[:, :6] - sm_stress[:, :6]) / (2.0 * h)

    return D_num[0] if single else D_num


def _rel_error(D_alg: np.ndarray, D_num: np.ndarray) -> float:
    """Compute relative Frobenius norm error between algorithmic and numerical tangents."""
    norm_num = float(np.linalg.norm(D_num))
    if norm_num < 1e-12:
        return float(np.max(np.abs(D_alg - D_num)))
    return float(np.linalg.norm(D_alg - D_num) / norm_num)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def soil_linear_dp_mat() -> Material:
    """Linear Drucker-Prager geological material (sand/soil, A2=0, A1=0.6, C1=bulk)."""
    return build_law21({
        "id": 1,
        "rho0": 2000.0,
        "E": 30000.0,
        "nu": 0.25,
        "a0": 100.0,
        "a1": 0.6,
        "a2": 0.0,
        "amax": 5000.0,
        "c1": 20000.0,
        "bunl": 20000.0,
    })


@pytest.fixture
def concrete_parabolic_dp_mat() -> Material:
    """Parabolic Drucker-Prager concrete material (A2>0, A1>0, non-linear envelope)."""
    return build_law21({
        "id": 2,
        "rho0": 2400.0,
        "E": 35000.0,
        "nu": 0.20,
        "a0": 80.0,
        "a1": 0.8,
        "a2": 0.002,
        "amax": 1500.0,
        "c1": 25000.0,
        "bunl": 30000.0,
    })


@pytest.fixture
def compaction_eos_mat() -> Material:
    """Compaction material with non-linear EOS curve and hysteretic unloading."""
    xs = np.array([0.0, 0.02, 0.05, 0.10])
    ys = np.array([0.0, 500.0, 1500.0, 4000.0])
    return build_law21({
        "id": 3,
        "rho0": 2200.0,
        "E": 40000.0,
        "nu": 0.22,
        "a0": 120.0,
        "a1": 0.5,
        "a2": 0.001,
        "amax": 3000.0,
        "c1": 45000.0,
        "bunl": 90000.0,
        "mumax": 0.10,
        "curve": (xs, ys),
    })


# =============================================================================
# 1. Linear Elastic Regime & Analytical Hookean Recovery
# =============================================================================

class TestElasticTangents:
    def test_elastic_analytic_hookean(self, soil_linear_dp_mat: Material):
        """Verify tangent without deps matches analytical 3D Hookean stiffness matrix."""
        D = tangent_law21_solid(soil_linear_dp_mat)
        assert D.shape == (6, 6)

        p = soil_linear_dp_mat.params
        g = p["G"]
        k = p["c1"]

        expected = np.zeros((6, 6), dtype=float)
        expected[0, 0] = expected[1, 1] = expected[2, 2] = k + (4.0 / 3.0) * g
        expected[0, 1] = expected[0, 2] = expected[1, 0] = expected[1, 2] = expected[2, 0] = expected[2, 1] = k - (2.0 / 3.0) * g
        expected[3, 3] = expected[4, 4] = expected[5, 5] = g

        np.testing.assert_allclose(D, expected, rtol=1e-12, atol=1e-12)

    def test_elastic_central_fd_h_sensitivity(self, soil_linear_dp_mat: Material):
        """Small sub-yield strain increment verified against central differences with h in [1e-7, 1e-6]."""
        deps_elastic = np.array([1.0e-5, -3.0e-6, -2.0e-6, 1.5e-6, 0.0, 0.0])
        sig0 = np.zeros(6)

        for h_step in (1e-7, 5e-7, 1e-6):
            D_alg = tangent_law21_solid(soil_linear_dp_mat, sig0, deps=deps_elastic, h=h_step)
            D_num = _compute_num_solid_tangent(soil_linear_dp_mat, sig0, deps_elastic, h=h_step)
            err = _rel_error(D_alg, D_num)
            assert err < 1e-8, f"Elastic tangent error {err:.3e} exceeds 1e-8 at h={h_step}"

    def test_elastic_multiaxial_subyield_states(self, concrete_parabolic_dp_mat: Material):
        """Multi-axial small strain states within elastic limit."""
        test_states = [
            np.array([2e-5, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 2e-5, 0.0, 0.0]),
            np.array([1.5e-5, 1.5e-5, 1.5e-5, 1.0e-5, 0.5e-5, 0.5e-5]),
            np.array([-1e-5, 2e-5, -1e-5, 0.0, 1e-5, 0.0]),
        ]
        sig0 = np.zeros(6)
        for deps in test_states:
            D_alg = tangent_law21_solid(concrete_parabolic_dp_mat, sig0, deps=deps, h=1e-7)
            D_num = _compute_num_solid_tangent(concrete_parabolic_dp_mat, sig0, deps, h=1e-6)
            err = _rel_error(D_alg, D_num)
            assert err < 1e-8, f"Multi-axial elastic tangent error {err:.3e} exceeds 1e-8 for {deps}"


# =============================================================================
# 2. Drucker-Prager Yield Regimes (Linear, Parabolic, von Mises Cap, Cutoffs)
# =============================================================================

class TestDruckerPragerYieldRegimes:
    def test_linear_drucker_prager_yield(self, soil_linear_dp_mat: Material):
        """Linear Drucker-Prager yield regime (A2 = 0, A1 > 0)."""
        deps_yield = np.array([0.005, -0.001, -0.001, 0.002, 0.0, 0.0])
        sig0 = np.zeros(6)

        for h_test in (1e-7, 1e-6):
            D_alg = tangent_law21_solid(soil_linear_dp_mat, sig0, deps=deps_yield, h=h_test)
            D_num = _compute_num_solid_tangent(soil_linear_dp_mat, sig0, deps_yield, h=h_test)
            err = _rel_error(D_alg, D_num)
            assert err < 1e-4, f"Linear DP yield error {err:.3e} exceeds 1e-4 at h={h_test}"

    def test_parabolic_drucker_prager_yield(self, concrete_parabolic_dp_mat: Material):
        """Parabolic Drucker-Prager yield regime (A2 > 0)."""
        deps_yield = np.array([0.006, -0.0015, -0.0015, 0.003, 0.001, 0.0])
        sig0 = np.zeros(6)

        for h_test in (1e-7, 1e-6):
            D_alg = tangent_law21_solid(concrete_parabolic_dp_mat, sig0, deps=deps_yield, h=h_test)
            D_num = _compute_num_solid_tangent(concrete_parabolic_dp_mat, sig0, deps_yield, h=h_test)
            err = _rel_error(D_alg, D_num)
            assert err < 1e-4, f"Parabolic DP yield error {err:.3e} exceeds 1e-4 at h={h_test}"

    def test_von_mises_cap_regime(self):
        """von Mises cap regime where G0 = Amax and dG0/dPtot = 0."""
        mat = build_law21({
            "id": 10,
            "rho0": 2000.0,
            "E": 30000.0,
            "nu": 0.2,
            "a0": 50.0,
            "a1": 2.0,
            "a2": 0.01,
            "amax": 120.0,
            "c1": 20000.0,
            "bunl": 20000.0,
        })
        deps_cap = np.array([0.005, -0.001, -0.001, 0.002, 0.0, 0.0])
        sig0 = np.zeros(6)

        for h_test in (1e-7, 1e-6):
            D_alg = tangent_law21_solid(mat, sig0, deps=deps_cap, h=h_test)
            D_num = _compute_num_solid_tangent(mat, sig0, deps_cap, h=h_test)
            err = _rel_error(D_alg, D_num)
            assert err < 1e-4, f"von Mises cap error {err:.3e} exceeds 1e-4 at h={h_test}"

    def test_root_cutoff_regime(self):
        """Root cutoff regime (Ptot <= P*) where deviatoric yield envelope collapses to 0."""
        # For A0=100, A1=1.0, A2=0, P* = -100.0. Under tension P = C1 * mu < 0.
        mat = build_law21({
            "id": 11,
            "rho0": 2000.0,
            "E": 30000.0,
            "nu": 0.2,
            "a0": 100.0,
            "a1": 1.0,
            "a2": 0.0,
            "amax": 5000.0,
            "c1": 20000.0,
            "bunl": 20000.0,
            "pext": 0.0,
        })
        # tr(deps) = 0.018 > 0 -> mu = -0.018 -> P = -360.0 < P* (-100.0)
        deps_root = np.array([0.006, 0.006, 0.006, 0.001, 0.0, 0.0])
        sig0 = np.zeros(6)

        for h_test in (1e-7, 1e-6):
            D_alg = tangent_law21_solid(mat, sig0, deps=deps_root, h=h_test)
            D_num = _compute_num_solid_tangent(mat, sig0, deps_root, h=h_test)
            err = _rel_error(D_alg, D_num)
            assert err < 1e-4, f"Root cutoff error {err:.3e} exceeds 1e-4 at h={h_test}"

    def test_tensile_fracture_cutoff_regime(self):
        """Tensile fracture cutoff regime (P <= Pmin) where tangent vanishes identically."""
        mat = build_law21({
            "id": 12,
            "rho0": 2000.0,
            "E": 30000.0,
            "nu": 0.2,
            "a0": 100.0,
            "a1": 1.0,
            "a2": 0.0,
            "amax": 5000.0,
            "c1": 20000.0,
            "bunl": 20000.0,
            "pmin": -10.0,
        })
        # Tension causes P < Pmin -> stresses pinned to -Pmin
        deps_pmin = np.array([0.002, 0.002, 0.002, 0.001, 0.0, 0.0])
        sig0 = np.zeros(6)

        D_alg = tangent_law21_solid(mat, sig0, deps=deps_pmin, h=1e-7)
        np.testing.assert_allclose(D_alg, 0.0, atol=1e-12)


# =============================================================================
# 3. Compaction EOS Regimes (Loading Curve & Hysteretic Unloading)
# =============================================================================

class TestCompactionEOSTangents:
    def test_compaction_eos_loading_curve(self, compaction_eos_mat: Material):
        """Compaction EOS virgin loading regime following tabulated curve."""
        deps_load = np.array([-0.01, -0.01, -0.01, 0.002, 0.0, 0.0])
        sig0 = np.zeros(6)
        extra0 = {"mu": 0.0, "mu_bak": 0.0}

        for h_test in (1e-7, 1e-6):
            D_alg = tangent_law21_solid(compaction_eos_mat, sig0, deps=deps_load, extra=extra0, h=h_test)
            D_num = _compute_num_solid_tangent(compaction_eos_mat, sig0, deps_load, extra=extra0, h=h_test)
            err = _rel_error(D_alg, D_num)
            assert err < 1e-4, f"Compaction EOS loading error {err:.3e} exceeds 1e-4 at h={h_test}"

    def test_compaction_eos_hysteretic_unloading(self, compaction_eos_mat: Material):
        """Compaction EOS hysteretic unloading with evolving modulus Kunload."""
        # Previous historical compaction mu_bak = 0.05
        # Current partial volumetric unloading to mu = 0.035
        deps_unl = np.array([0.005, 0.005, 0.005, 0.002, 0.0, 0.0])
        sig0 = np.zeros(6)
        extra_unl = {"mu": 0.05, "mu_bak": 0.05}

        for h_test in (1e-7, 1e-6):
            D_alg = tangent_law21_solid(compaction_eos_mat, sig0, deps=deps_unl, extra=extra_unl, h=h_test)
            D_num = _compute_num_solid_tangent(compaction_eos_mat, sig0, deps_unl, extra=extra_unl, h=h_test)
            err = _rel_error(D_alg, D_num)
            assert err < 1e-4, f"Compaction EOS unloading error {err:.3e} exceeds 1e-4 at h={h_test}"


# =============================================================================
# 4. Multi-axial Loading States
# =============================================================================

class TestMultiaxialLoadingStates:
    def test_uniaxial_tension(self, soil_linear_dp_mat: Material):
        """Uniaxial tension loading state."""
        deps = np.array([0.005, -0.001, -0.001, 0.0, 0.0, 0.0])
        sig0 = np.zeros(6)
        D_alg = tangent_law21_solid(soil_linear_dp_mat, sig0, deps=deps, h=1e-7)
        D_num = _compute_num_solid_tangent(soil_linear_dp_mat, sig0, deps, h=1e-6)
        assert _rel_error(D_alg, D_num) < 1e-4

    def test_uniaxial_compression(self, concrete_parabolic_dp_mat: Material):
        """Uniaxial compression loading state."""
        deps = np.array([-0.006, 0.0012, 0.0012, 0.0, 0.0, 0.0])
        sig0 = np.zeros(6)
        D_alg = tangent_law21_solid(concrete_parabolic_dp_mat, sig0, deps=deps, h=1e-7)
        D_num = _compute_num_solid_tangent(concrete_parabolic_dp_mat, sig0, deps, h=1e-6)
        assert _rel_error(D_alg, D_num) < 1e-4

    def test_equibiaxial_tension(self, soil_linear_dp_mat: Material):
        """Equibiaxial tension state (deps_xx = deps_yy)."""
        deps = np.array([0.004, 0.004, -0.002, 0.0, 0.0, 0.0])
        sig0 = np.zeros(6)
        D_alg = tangent_law21_solid(soil_linear_dp_mat, sig0, deps=deps, h=1e-7)
        D_num = _compute_num_solid_tangent(soil_linear_dp_mat, sig0, deps, h=1e-6)
        assert _rel_error(D_alg, D_num) < 1e-4

    def test_triaxial_hydrostatic_compression(self, soil_linear_dp_mat: Material):
        """Triaxial hydrostatic compression (pure volumetric, J2=0)."""
        deps = np.array([-0.003, -0.003, -0.003, 0.0, 0.0, 0.0])
        sig0 = np.zeros(6)
        D_alg = tangent_law21_solid(soil_linear_dp_mat, sig0, deps=deps, h=1e-7)
        D_num = _compute_num_solid_tangent(soil_linear_dp_mat, sig0, deps, h=1e-6)
        assert _rel_error(D_alg, D_num) < 1e-8

    def test_pure_shear_planes(self, concrete_parabolic_dp_mat: Material):
        """Pure shear across xy, yz, and zx shear planes."""
        sig0 = np.zeros(6)
        shear_states = [
            np.array([0.0, 0.0, 0.0, 0.008, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.008, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.008]),
        ]
        for deps in shear_states:
            D_alg = tangent_law21_solid(concrete_parabolic_dp_mat, sig0, deps=deps, h=1e-7)
            D_num = _compute_num_solid_tangent(concrete_parabolic_dp_mat, sig0, deps, h=1e-6)
            assert _rel_error(D_alg, D_num) < 1e-4

    def test_mixed_normal_shear(self, concrete_parabolic_dp_mat: Material):
        """General mixed multi-axial normal and shear strain state."""
        deps = np.array([0.004, -0.002, 0.001, 0.003, -0.002, 0.0015])
        sig0 = np.zeros(6)
        D_alg = tangent_law21_solid(concrete_parabolic_dp_mat, sig0, deps=deps, h=1e-7)
        D_num = _compute_num_solid_tangent(concrete_parabolic_dp_mat, sig0, deps, h=1e-6)
        assert _rel_error(D_alg, D_num) < 1e-4


# =============================================================================
# 5. Deactivated Elements
# =============================================================================

class TestDeactivatedElements:
    def test_deactivated_single_element(self, soil_linear_dp_mat: Material):
        """Single deactivated element with off=0.0 produces identically zero tangent."""
        deps = np.array([0.005, -0.001, -0.001, 0.002, 0.0, 0.0])
        sig0 = np.zeros(6)
        extra = {"off": np.array([0.0])}

        D = tangent_law21_solid(soil_linear_dp_mat, sig0, deps=deps, extra=extra)
        np.testing.assert_allclose(D, 0.0, atol=1e-12)

    def test_mixed_active_and_deactivated_batch(self, soil_linear_dp_mat: Material):
        """Batch with mixed active and deactivated elements."""
        n = 8
        deps = np.tile(np.array([0.005, -0.001, -0.001, 0.002, 0.0, 0.0]), (n, 1))
        sig = np.zeros((n, 6))
        off = np.ones(n)
        off[2] = 0.0
        off[5] = 0.0
        extra = {"off": off}

        D = tangent_law21_solid(soil_linear_dp_mat, sig, deps=deps, extra=extra)
        assert D.shape == (n, 6, 6)
        assert np.all(D[2] == 0.0)
        assert np.all(D[5] == 0.0)
        assert np.linalg.norm(D[0]) > 0.0
        assert np.linalg.norm(D[1]) > 0.0


# =============================================================================
# 6. Batch Vectorization (n = 1, 8, 32)
# =============================================================================

class TestBatchVectorization:
    @pytest.mark.parametrize("n", [1, 8, 32])
    def test_vectorization_batches(self, concrete_parabolic_dp_mat: Material, n: int):
        """Verify batch vectorization for n = 1, 8, 32 elements."""
        rng = np.random.default_rng(42)
        base_deps = np.array([0.004, -0.0015, -0.001, 0.002, 0.001, 0.0])
        perturbations = rng.uniform(-1e-4, 1e-4, size=(n, 6))
        deps = np.tile(base_deps, (n, 1)) + perturbations
        sig = np.zeros((n, 6))

        D_batch = tangent_law21_solid(concrete_parabolic_dp_mat, sig, deps=deps)
        assert D_batch.shape == (n, 6, 6)

        # Verify element-by-element equivalence
        for i in range(n):
            D_single = tangent_law21_solid(concrete_parabolic_dp_mat, sig[i], deps=deps[i])
            np.testing.assert_allclose(D_batch[i], D_single, rtol=1e-10, atol=1e-10)

    def test_shapes_1d_2d_empty(self, soil_linear_dp_mat: Material):
        """Verify shape handling for 1D, 2D, and empty arrays."""
        # 1D
        d_1d = tangent_law21_solid(soil_linear_dp_mat, np.zeros(6))
        assert d_1d.shape == (6, 6)

        # 2D (1, 6)
        d_2d = tangent_law21_solid(soil_linear_dp_mat, np.zeros((1, 6)))
        assert d_2d.shape == (1, 6, 6)

        # Empty (0, 6)
        d_empty = tangent_law21_solid(soil_linear_dp_mat, np.empty((0, 6)))
        assert d_empty.shape == (0, 6, 6)


# =============================================================================
# 7. Calling Conventions, Symmetry, Aliases, and Dispatch
# =============================================================================

class TestCallingConventionsAndSymmetry:
    def test_positional_and_keyword_invocations(self, soil_linear_dp_mat: Material):
        """Verify calling conventions: positional, keyword, and omitted sig."""
        deps = np.array([0.005, -0.001, -0.001, 0.002, 0.0, 0.0])
        sig = np.zeros(6)
        extra = {"mu": 0.0, "mu_bak": 0.0}

        D1 = tangent_law21_solid(soil_linear_dp_mat, sig, deps, extra=extra)
        D2 = tangent_law21_solid(soil_linear_dp_mat, sig=sig, deps=deps, extra=extra)
        D3 = tangent_law21_solid(soil_linear_dp_mat, None, deps, extra=extra)
        D4 = tangent_law21_solid(soil_linear_dp_mat, deps=deps, extra=extra)
        D5 = tangent_law21_solid(soil_linear_dp_mat, sig=sig, d_eps=deps, extra=extra)

        np.testing.assert_allclose(D1, D2, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(D3, D4, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(D1, D5, rtol=1e-12, atol=1e-12)

    def test_symmetric_option(self, concrete_parabolic_dp_mat: Material):
        """Verify symmetric=True produces D = D^T and matches symmetrized unsymmetric tangent."""
        deps = np.array([0.005, -0.001, -0.001, 0.002, 0.0, 0.0])

        D_unsym = tangent_law21_solid(concrete_parabolic_dp_mat, deps=deps, symmetric=False)
        D_sym = tangent_law21_solid(concrete_parabolic_dp_mat, deps=deps, symmetric=True)

        # Pressure-deviatoric coupling produces minor unsymmetry in non-associative Drucker-Prager
        asym_diff = float(np.max(np.abs(D_unsym - D_unsym.T)))
        assert asym_diff > 0.1, f"Expected non-symmetric plastic tangent, asym={asym_diff}"

        # Symmetrized tangent must satisfy D = D^T
        sym_diff = float(np.max(np.abs(D_sym - D_sym.T)))
        assert sym_diff < 1e-12, f"Symmetrized tangent violates symmetry: {sym_diff}"
        np.testing.assert_allclose(D_sym, 0.5 * (D_unsym + D_unsym.T), atol=1e-12)

    def test_aliases(self, soil_linear_dp_mat: Material):
        """Verify module aliases consistent_solid_tangent and solid_tangent."""
        deps = np.array([0.003, -0.001, -0.001, 0.001, 0.0, 0.0])
        D_orig = tangent_law21_solid(soil_linear_dp_mat, deps=deps)
        D_alias1 = consistent_solid_tangent(soil_linear_dp_mat, deps=deps)
        D_alias2 = solid_tangent(soil_linear_dp_mat, deps=deps)

        np.testing.assert_allclose(D_orig, D_alias1, atol=1e-12)
        np.testing.assert_allclose(D_orig, D_alias2, atol=1e-12)

    def test_package_level_dispatch(self, soil_linear_dp_mat: Material):
        """Verify package-level dispatch via materials.solid_tangent and materials.consistent_solid_tangent."""
        D_pkg1 = materials.solid_tangent(soil_linear_dp_mat, sig=np.zeros(6))
        D_pkg2 = materials.consistent_solid_tangent(soil_linear_dp_mat, sig=np.zeros(6))
        assert D_pkg1.shape == (6, 6)
        assert D_pkg2.shape == (6, 6)
        np.testing.assert_allclose(D_pkg1, D_pkg2, atol=1e-12)

    def test_shell_tangent_raises_not_implemented(self, soil_linear_dp_mat: Material):
        """LAW21 is solid-only; shell update and tangent must raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="solid elements only"):
            shell_update_law21()

        with pytest.raises(NotImplementedError, match="solid elements only"):
            materials.shell_layer_tangent(soil_linear_dp_mat)
