"""
Milestone M555: Algorithmic Consistent Tangent Verification Suite for /MAT/LAW57 (/MAT/BARLAT3).

Rigorously verifies the algorithmic consistent plane-stress membrane tangent stiffness tensors
in `pyradioss/materials/law57_barlat.py` across:
1. Shell membrane algorithmic tangent C^alg = d(sigma)/d(deps) of shape (nel, 3, 3) or (3, 3),
   verified against independent central finite difference perturbations:
     C_num[:, j] = (sigma(deps + h*e_j) - sigma(deps - h*e_j)) / (2*h)
   with step sizes h in [1e-7, 1e-6], achieving relative error < 1e-4 in plastic regimes
   and < 1e-8 in elastic regimes.
2. Verification across all physical regimes:
   - Elastic regime (analytical Hookean plane-stress recovery)
   - Initial yielding along rolling direction (0 deg), diagonal (45 deg), transverse (90 deg)
   - Equibiaxial tension (sigma_xx = sigma_yy) and non-equibiaxial tension
   - Pure shear (sigma_xy)
   - Mixed normal-shear states
   - Tabulated hardening curves across different strain rates and rate filtering
   - Kinematic hardening (F_isokin in [0, 1]) and backstress tensor evolution
   - Dynamic Young's modulus degradation (exponential CE and tabulated ifunce)
   - Tensile damage scaling regime (FAIL in (0, 1))
   - Ruptured/eroded regime (epsp >= EPSMAX, FAIL <= 0, or off <= 0: tangent vanishes identically)
   - Different Barlat exponents (m = 2, 4, 6, 8) and diverse Lankford parameters (steel vs aluminum)
   - Vectorization with n = 1, 8, 32 elements simultaneously.
3. Calling conventions: positional and keyword invocations, symmetric=True producing C = C^T,
   and package-level dispatch via materials.shell_layer_tangent and materials.shell_membrane_tangent.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pytest

import pyradioss.materials as materials
from pyradioss.materials.law57_barlat import (
    Law57Params,
    BarlatParams,
    build_law57,
    barlat_params,
    calculp2,
    barlat_equivalent_stress,
    shell_update_law57,
    sound_speed_shell_law57,
    tangent_law57_shell,
    consistent_shell_tangent,
    shell_tangent,
    shell_membrane_tangent,
    _copy_extra,
)


# =============================================================================
# Helper Utilities: Independent Central Finite Difference Numerical Tangents
# =============================================================================

def _compute_num_shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray],
    deps: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
    dt: float = 0.0,
    epsp: Optional[Union[float, np.ndarray]] = None,
    h: float = 1e-7,
) -> np.ndarray:
    """Compute (NEL, 3, 3) or (3, 3) numerical plane-stress membrane tangent
    via independent directional central finite differences:
        C_num[:, :, j] = (sigma(deps + h * e_j) - sigma(deps - h * e_j)) / (2 * h)
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

    C_num = np.zeros((nel, 3, 3), dtype=float)
    for j in range(3):
        ej = np.zeros_like(deps_arr)
        ej[:, j] = h
        ex_p = _copy_extra(extra)
        ex_m = _copy_extra(extra)
        sp = shell_update_law57(mat, s0.copy(), deps_arr + ej, extra=ex_p, dt=dt, epsp=ep_arr.copy(), return_sound_speed=False)
        sm = shell_update_law57(mat, s0.copy(), deps_arr - ej, extra=ex_m, dt=dt, epsp=ep_arr.copy(), return_sound_speed=False)
        sp_stress = sp[0] if isinstance(sp, tuple) else sp
        sm_stress = sm[0] if isinstance(sm, tuple) else sm
        if sp_stress.ndim == 1:
            sp_stress = sp_stress.reshape(1, -1)
            sm_stress = sm_stress.reshape(1, -1)
        C_num[:, :, j] = (sp_stress[:, :3] - sm_stress[:, :3]) / (2.0 * h)

    return C_num[0] if single else C_num


def _rel_error(C_alg: np.ndarray, C_num: np.ndarray) -> float:
    """Compute relative Frobenius norm error between algorithmic and numerical tangents."""
    norm_num = float(np.linalg.norm(C_num))
    if norm_num < 1e-12:
        return float(np.max(np.abs(C_alg - C_num)))
    return float(np.linalg.norm(C_alg - C_num) / norm_num)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def isotropic_mat() -> Law57Params:
    """Isotropic baseline (R00=R45=R90=1, m=6)."""
    return build_law57(
        id=1,
        E=200000.0,
        nu=0.3,
        r00=1.0,
        r45=1.0,
        r90=1.0,
        m=6.0,
        sigy0=200.0,
        curves=[([0.0, 0.05, 0.10], [200.0, 280.0, 340.0], 0.0)],
    )


@pytest.fixture
def steel_mat() -> Law57Params:
    """Anisotropic deep-drawing sheet steel (BCC, m=6, R00=1.8, R45=1.3, R90=2.1)."""
    return build_law57(
        id=2,
        E=210000.0,
        nu=0.3,
        r00=1.8,
        r45=1.3,
        r90=2.1,
        m=6.0,
        sigy0=220.0,
        curves=[([0.0, 0.02, 0.08], [220.0, 310.0, 420.0], 0.0)],
    )


@pytest.fixture
def aluminum_mat() -> Law57Params:
    """Anisotropic automotive aluminum sheet AA6016 (FCC, m=8, R00=0.65, R45=0.48, R90=0.75)."""
    return build_law57(
        id=3,
        E=70000.0,
        nu=0.33,
        r00=0.65,
        r45=0.48,
        r90=0.75,
        m=8.0,
        sigy0=120.0,
        curves=[([0.0, 0.03, 0.12], [120.0, 190.0, 260.0], 0.0)],
    )


# =============================================================================
# 1. Elastic Regime & Hookean Plane-Stress Recovery
# =============================================================================

class TestElasticTangents:
    def test_elastic_analytic_hookean(self, steel_mat: Law57Params):
        """Verify tangent without deps matches analytical Hookean plane-stress stiffness."""
        C = tangent_law57_shell(steel_mat)
        assert C.shape == (3, 3)

        E = steel_mat.E
        nu = steel_mat.nu
        a11 = E / (1.0 - nu ** 2)
        a12 = nu * a11
        g = E / (2.0 * (1.0 + nu))

        expected = np.array([
            [a11, a12, 0.0],
            [a12, a11, 0.0],
            [0.0, 0.0, g],
        ])
        np.testing.assert_allclose(C, expected, rtol=1e-12, atol=1e-12)

    def test_elastic_small_perturbation_h_sensitivity(self, steel_mat: Law57Params):
        """Small sub-yield strain increment verified against central differences with h in [1e-7, 1e-6]."""
        deps_elastic = np.array([1.0e-5, -3.0e-6, 2.0e-6])
        C_alg = tangent_law57_shell(steel_mat, deps=deps_elastic, h=1e-7)

        for h_step in (1e-7, 5e-7, 1e-6):
            C_num = _compute_num_shell_tangent(steel_mat, None, deps_elastic, h=h_step)
            err = _rel_error(C_alg, C_num)
            assert err < 1e-8, f"Elastic tangent error {err:.3e} exceeds 1e-8 at h={h_step}"

    def test_elastic_multiaxial_states(self, aluminum_mat: Law57Params):
        """Multi-axial small strains within elastic limit."""
        test_states = [
            np.array([2e-5, 0.0, 0.0]),
            np.array([0.0, 2e-5, 0.0]),
            np.array([0.0, 0.0, 3e-5]),
            np.array([1.5e-5, 1.5e-5, 1.0e-5]),
            np.array([-1e-5, 2e-5, -1e-5]),
        ]
        for deps in test_states:
            C_alg = tangent_law57_shell(aluminum_mat, deps=deps)
            C_num = _compute_num_shell_tangent(aluminum_mat, None, deps, h=1e-6)
            err = _rel_error(C_alg, C_num)
            assert err < 1e-8, f"Multi-axial elastic tangent error {err:.3e} exceeds 1e-8 for {deps}"


# =============================================================================
# 2. Initial Yielding Along Principal Anisotropy Directions (0, 45, 90 deg)
# =============================================================================

class TestAnisotropicYieldingTangents:
    def test_yielding_rolling_0deg(self, steel_mat: Law57Params):
        """Uniaxial yielding along rolling direction (0 deg)."""
        deps_0 = np.array([0.003, -0.0009, 0.0])
        C_alg = tangent_law57_shell(steel_mat, deps=deps_0, h=1e-7)

        for h_test in (1e-7, 1e-6):
            C_num = _compute_num_shell_tangent(steel_mat, None, deps_0, h=h_test)
            err = _rel_error(C_alg, C_num)
            assert err < 1e-4, f"0 deg yielding error {err:.3e} exceeds 1e-4 at h={h_test}"

    def test_yielding_transverse_90deg(self, steel_mat: Law57Params):
        """Uniaxial yielding along transverse direction (90 deg)."""
        deps_90 = np.array([-0.0009, 0.003, 0.0])
        C_alg = tangent_law57_shell(steel_mat, deps=deps_90, h=1e-7)

        for h_test in (1e-7, 1e-6):
            C_num = _compute_num_shell_tangent(steel_mat, None, deps_90, h=h_test)
            err = _rel_error(C_alg, C_num)
            assert err < 1e-4, f"90 deg yielding error {err:.3e} exceeds 1e-4 at h={h_test}"

    def test_yielding_diagonal_45deg(self, steel_mat: Law57Params):
        """Yielding along diagonal direction (45 deg) with combined tension and shear."""
        deps_45 = np.array([0.0015, 0.0015, 0.003])
        C_alg = tangent_law57_shell(steel_mat, deps=deps_45, h=1e-7)

        for h_test in (1e-7, 1e-6):
            C_num = _compute_num_shell_tangent(steel_mat, None, deps_45, h=h_test)
            err = _rel_error(C_alg, C_num)
            assert err < 1e-4, f"45 deg yielding error {err:.3e} exceeds 1e-4 at h={h_test}"

    def test_anisotropic_directional_difference(self, steel_mat: Law57Params):
        """Verify that anisotropic parameters (R00 != R90) produce distinct tangent components."""
        deps_0 = np.array([0.003, -0.0009, 0.0])
        deps_90 = np.array([-0.0009, 0.003, 0.0])

        C_0 = tangent_law57_shell(steel_mat, deps=deps_0)
        C_90 = tangent_law57_shell(steel_mat, deps=deps_90)

        # Anisotropic difference between 0 and 90 deg response
        diff_diag = abs(C_0[0, 0] - C_90[1, 1])
        assert diff_diag > 100.0, f"Expected distinct anisotropic tangent responses, got diff {diff_diag}"


# =============================================================================
# 3. Equibiaxial, Non-Equibiaxial, Pure Shear, and Mixed Stress States
# =============================================================================

class TestStressStateTangents:
    def test_equibiaxial_tension(self, steel_mat: Law57Params):
        """Equibiaxial tension state (deps_xx = deps_yy > yield)."""
        deps_biax = np.array([0.003, 0.003, 0.0])
        C_alg = tangent_law57_shell(steel_mat, deps=deps_biax, h=1e-7)

        for h_test in (1e-7, 1e-6):
            C_num = _compute_num_shell_tangent(steel_mat, None, deps_biax, h=h_test)
            err = _rel_error(C_alg, C_num)
            assert err < 1e-4, f"Equibiaxial tangent error {err:.3e} exceeds 1e-4 at h={h_test}"

    def test_nonequibiaxial_tension(self, aluminum_mat: Law57Params):
        """Non-equibiaxial tension state (deps_xx != deps_yy)."""
        deps_non_eq = np.array([0.004, 0.001, 0.0])
        C_alg = tangent_law57_shell(aluminum_mat, deps=deps_non_eq, h=1e-7)

        for h_test in (1e-7, 1e-6):
            C_num = _compute_num_shell_tangent(aluminum_mat, None, deps_non_eq, h=h_test)
            err = _rel_error(C_alg, C_num)
            assert err < 1e-4, f"Non-equibiaxial tangent error {err:.3e} exceeds 1e-4 at h={h_test}"

    def test_pure_shear(self, steel_mat: Law57Params):
        """Pure in-plane shear state (deps_xy >> 0, deps_xx = deps_yy = 0)."""
        deps_shear = np.array([0.0, 0.0, 0.005])
        C_alg = tangent_law57_shell(steel_mat, deps=deps_shear, h=1e-7)

        for h_test in (1e-7, 1e-6):
            C_num = _compute_num_shell_tangent(steel_mat, None, deps_shear, h=h_test)
            err = _rel_error(C_alg, C_num)
            assert err < 1e-4, f"Pure shear tangent error {err:.3e} exceeds 1e-4 at h={h_test}"

    def test_mixed_normal_shear(self, steel_mat: Law57Params):
        """Combined normal and shear plastic deformation state."""
        deps_mixed = np.array([0.003, -0.0015, 0.0025])
        C_alg = tangent_law57_shell(steel_mat, deps=deps_mixed, h=1e-7)

        for h_test in (1e-7, 1e-6):
            C_num = _compute_num_shell_tangent(steel_mat, None, deps_mixed, h=h_test)
            err = _rel_error(C_alg, C_num)
            assert err < 1e-4, f"Mixed state tangent error {err:.3e} exceeds 1e-4 at h={h_test}"

    def test_directional_taylor_series_residual(self, steel_mat: Law57Params):
        """Directional Taylor test: sigma(deps + delta) - sigma(deps) = C_alg @ delta + O(||delta||^2)."""
        deps_base = np.array([0.0025, -0.001, 0.0015])
        s0 = shell_update_law57(steel_mat, np.zeros(3), deps_base, return_sound_speed=False)[0]
        C_alg = tangent_law57_shell(steel_mat, deps=deps_base, h=1e-7)

        rng = np.random.default_rng(42)
        for _ in range(4):
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            delta = 1e-6 * direction

            s_p = shell_update_law57(steel_mat, np.zeros(3), deps_base + delta, return_sound_speed=False)[0]
            s_m = shell_update_law57(steel_mat, np.zeros(3), deps_base - delta, return_sound_speed=False)[0]
            actual_dsig_central = (s_p[:3] - s_m[:3]) / 2.0
            pred_dsig = C_alg @ delta

            residual_central = np.linalg.norm(actual_dsig_central - pred_dsig)
            rel_res_central = residual_central / np.linalg.norm(actual_dsig_central)
            assert rel_res_central < 1e-4, f"Central Taylor residual {rel_res_central:.3e} exceeds 1e-4"

            # Forward Taylor series: residual is O(||delta||), bound by 1e-3
            actual_dsig_fwd = s_p[:3] - s0[:3]
            residual_fwd = np.linalg.norm(actual_dsig_fwd - pred_dsig)
            rel_res_fwd = residual_fwd / np.linalg.norm(actual_dsig_fwd)
            assert rel_res_fwd < 1e-3, f"Forward Taylor residual {rel_res_fwd:.3e} exceeds 1e-3"


# =============================================================================
# 4. Tabulated Hardening Curves Across Different Strain Rates
# =============================================================================

class TestTabulatedHardeningAndRateTangents:
    @pytest.fixture
    def rate_mat(self) -> Law57Params:
        """LAW57 with tabulated hardening curves at multiple strain rates."""
        c_static = ([0.0, 0.05, 0.15], [200.0, 290.0, 360.0], 0.0)
        c_med = ([0.0, 0.05, 0.15], [240.0, 340.0, 420.0], 100.0)
        c_high = ([0.0, 0.05, 0.15], [290.0, 400.0, 490.0], 1000.0)
        return build_law57(
            id=4,
            E=200000.0,
            nu=0.3,
            sigy0=200.0,
            curves=[c_static, c_med, c_high],
            asrate=50.0,
            israte=1,
            vp=0,
        )

    def test_static_rate_regime(self, rate_mat: Law57Params):
        """Very small rate (static curve regime)."""
        deps = np.array([0.003, -0.001, 0.001])
        dt = 1.0  # epsd ~ 0.003 < 100.0
        C_alg = tangent_law57_shell(rate_mat, deps=deps, dt=dt, h=1e-7)

        C_num = _compute_num_shell_tangent(rate_mat, None, deps, dt=dt, h=1e-6)
        err = _rel_error(C_alg, C_num)
        assert err < 1e-4, f"Static rate tangent error {err:.3e} exceeds 1e-4"

    def test_intermediate_rate_interpolation(self, rate_mat: Law57Params):
        """Intermediate rate regime interpolating between c_med and c_high."""
        deps = np.array([0.003, -0.001, 0.001])
        dt = 1e-5  # epsd ~ 300 /s, between 100 and 1000
        C_alg = tangent_law57_shell(rate_mat, deps=deps, dt=dt, h=1e-7)

        C_num = _compute_num_shell_tangent(rate_mat, None, deps, dt=dt, h=1e-6)
        err = _rel_error(C_alg, C_num)
        assert err < 1e-4, f"Intermediate rate tangent error {err:.3e} exceeds 1e-4"

    def test_high_rate_saturation(self, rate_mat: Law57Params):
        """High dynamic strain rate regime."""
        deps = np.array([0.003, -0.001, 0.001])
        dt = 1e-6  # epsd ~ 3000 /s > 1000
        C_alg = tangent_law57_shell(rate_mat, deps=deps, dt=dt, h=1e-7)

        C_num = _compute_num_shell_tangent(rate_mat, None, deps, dt=dt, h=1e-6)
        err = _rel_error(C_alg, C_num)
        assert err < 1e-4, f"High rate tangent error {err:.3e} exceeds 1e-4"


# =============================================================================
# 5. Mixed Isotropic / Kinematic Hardening (F_isokin in [0, 1])
# =============================================================================

class TestKinematicHardeningTangents:
    def test_pure_isotropic_hardening(self, steel_mat: Law57Params):
        """F_isokin = 0.0 (pure isotropic hardening)."""
        deps = np.array([0.003, -0.001, 0.0015])
        extra = {"pla57": np.array([0.01])}
        C_alg = tangent_law57_shell(steel_mat, deps=deps, extra=extra, h=1e-7)

        C_num = _compute_num_shell_tangent(steel_mat, None, deps, extra=extra, h=1e-6)
        err = _rel_error(C_alg, C_num)
        assert err < 1e-4, f"Pure isotropic tangent error {err:.3e} exceeds 1e-4"

    def test_mixed_hardening(self):
        """F_isokin = 0.5 (mixed isotropic / kinematic hardening)."""
        mat = build_law57(
            id=5,
            E=200000.0,
            nu=0.3,
            sigy0=200.0,
            fisokin=0.5,
            curves=[([0.0, 0.05], [200.0, 320.0], 0.0)],
        )
        deps = np.array([0.003, -0.001, 0.0015])
        extra = {"pla57": np.array([0.02]), "sigb57": np.array([[20.0, -10.0, 5.0]])}
        C_alg = tangent_law57_shell(mat, deps=deps, extra=extra, h=1e-7)

        C_num = _compute_num_shell_tangent(mat, None, deps, extra=extra, h=1e-6)
        err = _rel_error(C_alg, C_num)
        assert err < 1e-4, f"Mixed hardening tangent error {err:.3e} exceeds 1e-4"

    def test_pure_kinematic_hardening(self):
        """F_isokin = 1.0 (pure kinematic hardening)."""
        mat = build_law57(
            id=6,
            E=200000.0,
            nu=0.3,
            sigy0=200.0,
            fisokin=1.0,
            curves=[([0.0, 0.05], [200.0, 320.0], 0.0)],
        )
        deps = np.array([0.003, -0.001, 0.0015])
        extra = {"pla57": np.array([0.02]), "sigb57": np.array([[30.0, -15.0, 8.0]])}
        C_alg = tangent_law57_shell(mat, deps=deps, extra=extra, h=1e-7)

        C_num = _compute_num_shell_tangent(mat, None, deps, extra=extra, h=1e-6)
        err = _rel_error(C_alg, C_num)
        assert err < 1e-4, f"Pure kinematic tangent error {err:.3e} exceeds 1e-4"


# =============================================================================
# 6. Dynamic Young's Modulus Degradation (E(epsp) Active)
# =============================================================================

class TestDynamicYoungModulusTangents:
    def test_exponential_modulus_degradation_elastic(self):
        """Exponential degradation: E(epsp) = E0 - (E0 - Einf) * (1 - exp(-CE * epsp))."""
        mat = build_law57(
            id=7,
            E=200000.0,
            nu=0.3,
            sigy0=200.0,
            ce=25.0,
            einf=140000.0,
        )
        # Check degraded elastic matrix when epsp is provided
        C_init = tangent_law57_shell(mat, epsp=0.0)
        C_deg = tangent_law57_shell(mat, epsp=0.04)

        # Expected E at epsp = 0.04: 200000 - 60000 * (1 - exp(-1.0)) ~ 162072.77
        expected_E = 200000.0 - (200000.0 - 140000.0) * (1.0 - math.exp(-25.0 * 0.04))
        expected_a11 = expected_E / (1.0 - 0.3 ** 2)

        assert C_deg[0, 0] < C_init[0, 0]
        assert C_deg[0, 0] == pytest.approx(expected_a11, rel=1e-5)

    def test_exponential_modulus_degradation_plastic(self):
        """Plastic tangent with active exponential modulus degradation."""
        mat = build_law57(
            id=8,
            E=200000.0,
            nu=0.3,
            sigy0=200.0,
            ce=20.0,
            einf=150000.0,
            curves=[([0.0, 0.05], [200.0, 300.0], 0.0)],
        )
        deps = np.array([0.003, -0.001, 0.001])
        epsp = 0.02
        C_alg = tangent_law57_shell(mat, deps=deps, epsp=epsp, h=1e-7)

        C_num = _compute_num_shell_tangent(mat, None, deps, epsp=epsp, h=1e-6)
        err = _rel_error(C_alg, C_num)
        assert err < 1e-4, f"Degraded modulus plastic tangent error {err:.3e} exceeds 1e-4"

    def test_tabulated_modulus_degradation(self):
        """Tabulated modulus degradation curve E(epsp) via ifunce."""
        mat = build_law57(
            id=9,
            E=200000.0,
            nu=0.3,
            sigy0=200.0,
            ifunce=10,
            curves=[([0.0, 0.05], [200.0, 300.0], 0.0)],
            E_curve_x=[0.0, 0.05, 0.10],
            E_curve_y=[200000.0, 160000.0, 130000.0],
        )
        deps = np.array([0.003, -0.001, 0.001])
        extra = {"pla57": np.array([0.025])}
        C_alg = tangent_law57_shell(mat, deps=deps, extra=extra, h=1e-7)

        C_num = _compute_num_shell_tangent(mat, None, deps, extra=extra, h=1e-6)
        err = _rel_error(C_alg, C_num)
        assert err < 1e-4, f"Tabulated modulus plastic tangent error {err:.3e} exceeds 1e-4"


# =============================================================================
# 7. Tensile Damage Scaling & Rupture/Erosion Regimes
# =============================================================================

class TestDamageAndErosionTangents:
    def test_tensile_damage_scaling_regime(self):
        """Tensile damage scaling regime (FAIL in (0, 1)) where EPSR1 < epst < EPSR2."""
        mat = build_law57(
            id=10,
            E=200000.0,
            nu=0.3,
            sigy0=100000.0,  # Ensure purely elastic-damage regime
            epsr1=0.01,
            epsr2=0.03,
        )
        # Total strain epst ~ 0.02 (halfway between epsr1 and epsr2, FAIL = 0.5)
        deps = np.array([0.02, 0.0, 0.0])
        extra = {"eps57": np.zeros((1, 3)), "dmg57": np.zeros((1, 3))}

        C_alg = tangent_law57_shell(mat, deps=deps, extra=extra, h=1e-7)
        C_num = _compute_num_shell_tangent(mat, None, deps, extra=extra, h=1e-6)

        err = _rel_error(C_alg, C_num)
        assert err < 1e-4, f"Tensile damage scaling tangent error {err:.3e} exceeds 1e-4"

    def test_ruptured_regime_tensile_epsr2(self):
        """Element ruptured/eroded when epst >= EPSR2 (FAIL <= 0): tangent vanishes identically."""
        mat = build_law57(
            id=11,
            E=200000.0,
            nu=0.3,
            sigy0=100000.0,
            epsr1=0.01,
            epsr2=0.03,
        )
        # epst = 0.035 >= epsr2
        deps = np.array([0.035, 0.0, 0.0])
        extra = {"eps57": np.zeros((1, 3)), "dmg57": np.zeros((1, 3))}

        C = tangent_law57_shell(mat, deps=deps, extra=extra)
        assert np.all(C == 0.0), f"Expected zero tangent for tensile ruptured element, got:\n{C}"

    def test_ruptured_regime_epsmax(self):
        """Element ruptured/eroded when plastic strain epsp >= EPSMAX: tangent vanishes identically."""
        mat = build_law57(
            id=12,
            E=200000.0,
            nu=0.3,
            sigy0=200.0,
            epsmax=0.05,
            curves=[([0.0, 0.10], [200.0, 300.0], 0.0)],
        )
        extra = {"pla57": np.array([0.055])}  # epsp > epsmax
        deps = np.array([0.002, 0.0, 0.0])

        C = tangent_law57_shell(mat, deps=deps, extra=extra)
        assert np.all(C == 0.0), f"Expected zero tangent for epsmax eroded element, got:\n{C}"

    def test_eroded_flag_off_zero(self, steel_mat: Law57Params):
        """Element with off = 0.0 (already deleted): tangent vanishes identically."""
        extra = {"off57": np.array([0.0])}
        deps = np.array([0.003, -0.001, 0.0])

        C = tangent_law57_shell(steel_mat, deps=deps, extra=extra)
        assert np.all(C == 0.0), f"Expected zero tangent for deleted element (off=0), got:\n{C}"


# =============================================================================
# 8. Diverse Barlat Exponents (m = 2, 4, 6, 8) and Lankford Anisotropies
# =============================================================================

class TestBarlatExponentsAndAnisotropies:
    @pytest.mark.parametrize("m_exp", [2.0, 4.0, 6.0, 8.0])
    @pytest.mark.parametrize("aniso", [
        ("steel", 1.8, 1.3, 2.1),
        ("aluminum", 0.65, 0.48, 0.75),
    ])
    def test_barlat_exponent_and_anisotropy_combinations(self, m_exp: float, aniso: Tuple[str, float, float, float]):
        """Verify tangent against central finite differences for m in (2, 4, 6, 8) and diverse Lankford params."""
        mat_name, r00, r45, r90 = aniso
        mat = build_law57(
            id=20,
            E=200000.0,
            nu=0.3,
            sigy0=200.0,
            r00=r00,
            r45=r45,
            r90=r90,
            m=m_exp,
            curves=[([0.0, 0.05], [200.0, 300.0], 0.0)],
        )
        deps = np.array([0.003, -0.001, 0.001])
        C_alg = tangent_law57_shell(mat, deps=deps, h=1e-7)
        C_num = _compute_num_shell_tangent(mat, None, deps, h=1e-6)

        err = _rel_error(C_alg, C_num)
        assert err < 1e-4, f"m={m_exp} {mat_name} error {err:.3e} exceeds 1e-4"


# =============================================================================
# 9. Batched Element Vectorization (n = 1, 8, 32 elements)
# =============================================================================

class TestBatchedVectorization:
    def test_single_element_equivalence(self, steel_mat: Law57Params):
        """Check return shapes: 1D input yields (3, 3), 2D input yields (1, 3, 3)."""
        deps_1d = np.array([0.003, -0.001, 0.001])
        deps_2d = deps_1d[None, :]

        C_1d = tangent_law57_shell(steel_mat, deps=deps_1d)
        C_2d = tangent_law57_shell(steel_mat, deps=deps_2d)

        assert C_1d.shape == (3, 3)
        assert C_2d.shape == (1, 3, 3)
        np.testing.assert_allclose(C_1d, C_2d[0], rtol=1e-12, atol=1e-12)

    def test_batched_8_elements(self, steel_mat: Law57Params):
        """Batch of n = 8 elements verified against element-by-element execution."""
        n = 8
        rng = np.random.default_rng(101)
        deps = rng.uniform(-0.002, 0.004, size=(n, 3))
        sig = rng.uniform(-50.0, 50.0, size=(n, 3))
        pla = np.linspace(0.0, 0.03, n)
        extra = {"pla57": pla}

        C_batch = tangent_law57_shell(steel_mat, sig=sig, deps=deps, extra=extra)
        assert C_batch.shape == (n, 3, 3)

        for i in range(n):
            ex_i = {"pla57": np.array([pla[i]])}
            C_single = tangent_law57_shell(steel_mat, sig=sig[i], deps=deps[i], extra=ex_i)
            np.testing.assert_allclose(C_batch[i], C_single, rtol=1e-10, atol=1e-10)

    def test_batched_32_elements_heterogeneous(self):
        """Batch of n = 32 elements spanning diverse physical regimes simultaneously."""
        mat = build_law57(
            id=30,
            E=200000.0,
            nu=0.3,
            sigy0=200.0,
            epsmax=0.06,
            epsr1=0.015,
            epsr2=0.035,
            curves=[([0.0, 0.05], [200.0, 320.0], 0.0)],
        )
        n = 32
        deps = np.zeros((n, 3), dtype=float)
        pla = np.zeros(n, dtype=float)
        off = np.ones(n, dtype=float)
        dmg = np.zeros((n, 3), dtype=float)
        eps_tot = np.zeros((n, 3), dtype=float)

        for i in range(n):
            mode = i % 8
            if mode == 0:
                # Elastic sub-yield
                deps[i] = np.array([1e-5, -3e-6, 1e-6])
            elif mode == 1:
                # Active plastic rolling 0 deg
                deps[i] = np.array([0.003, -0.0009, 0.0])
                pla[i] = 0.01
            elif mode == 2:
                # Active plastic transverse 90 deg
                deps[i] = np.array([-0.0009, 0.003, 0.0])
                pla[i] = 0.02
            elif mode == 3:
                # Equibiaxial tension
                deps[i] = np.array([0.003, 0.003, 0.0])
                pla[i] = 0.015
            elif mode == 4:
                # Pure shear
                deps[i] = np.array([0.0, 0.0, 0.005])
                pla[i] = 0.025
            elif mode == 5:
                # Tensile damage scaling regime
                deps[i] = np.array([0.02, 0.0, 0.0])
            elif mode == 6:
                # Eroded by epsmax
                deps[i] = np.array([0.003, 0.0, 0.0])
                pla[i] = 0.07  # > epsmax = 0.06
            else:
                # Eroded by off = 0.0
                deps[i] = np.array([0.003, 0.0, 0.0])
                off[i] = 0.0

        extra = {
            "pla57": pla,
            "off57": off,
            "dmg57": dmg,
            "eps57": eps_tot,
        }

        C_batch = tangent_law57_shell(mat, deps=deps, extra=extra)
        assert C_batch.shape == (n, 3, 3)

        # Check eroded elements have exactly zero tangent
        for i in range(n):
            if i % 8 in (6, 7):
                assert np.all(C_batch[i] == 0.0)

        # Verify element-by-element equivalence
        for i in range(n):
            ex_i = {
                "pla57": np.array([pla[i]]),
                "off57": np.array([off[i]]),
                "dmg57": dmg[i:i+1].copy(),
                "eps57": eps_tot[i:i+1].copy(),
            }
            C_single = tangent_law57_shell(mat, deps=deps[i], extra=ex_i)
            np.testing.assert_allclose(C_batch[i], C_single, rtol=1e-10, atol=1e-10)


# =============================================================================
# 10. Calling Conventions, Symmetry Option, and Package Dispatch
# =============================================================================

class TestCallingConventionsAndSymmetry:
    def test_positional_and_keyword_invocations(self, steel_mat: Law57Params):
        """Verify calling conventions: positional, keyword, and omitted sig."""
        deps = np.array([0.003, -0.001, 0.002])
        sig = np.array([40.0, 20.0, 10.0])
        extra = {"pla57": np.array([0.01])}

        C1 = tangent_law57_shell(steel_mat, sig, deps, extra=extra)
        C2 = tangent_law57_shell(steel_mat, sig=sig, deps=deps, extra=extra)
        C3 = tangent_law57_shell(steel_mat, None, deps, extra=extra)
        C4 = tangent_law57_shell(steel_mat, deps=deps, extra=extra)

        np.testing.assert_allclose(C1, C2, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(C3, C4, rtol=1e-12, atol=1e-12)

    def test_symmetric_option(self, steel_mat: Law57Params):
        """Verify symmetric=True produces C = C^T and matches symmetrized unsymmetric tangent."""
        deps = np.array([0.003, -0.001, 0.002])
        extra = {"pla57": np.array([0.01])}

        C_unsym = tangent_law57_shell(steel_mat, deps=deps, extra=extra, symmetric=False)
        C_sym = tangent_law57_shell(steel_mat, deps=deps, extra=extra, symmetric=True)

        # Anisotropic non-associative flow yields natural minor asymmetry in plane stress
        asym_diff = float(np.max(np.abs(C_unsym - C_unsym.T)))
        assert asym_diff > 1.0, f"Expected non-symmetric plastic tangent, asym={asym_diff}"

        # Symmetrized tangent must satisfy C = C^T
        sym_diff = float(np.max(np.abs(C_sym - C_sym.T)))
        assert sym_diff < 1e-12, f"Symmetrized tangent violates symmetry: {sym_diff}"
        np.testing.assert_allclose(C_sym, 0.5 * (C_unsym + C_unsym.T), atol=1e-12)

    def test_package_level_dispatch(self, steel_mat: Law57Params):
        """Verify package-level dispatch via materials.shell_layer_tangent and shell_membrane_tangent."""
        # 1. shell_membrane_tangent
        C_mem = materials.shell_membrane_tangent(steel_mat)
        assert C_mem.shape == (3, 3)
        assert C_mem[0, 0] > 0.0
        assert C_mem[1, 1] > 0.0
        assert C_mem[2, 2] > 0.0

        # 2. shell_layer_tangent
        deps = np.array([0.003, -0.001, 0.001])
        C_layer = materials.shell_layer_tangent(steel_mat, sig=np.zeros((1, 3)))
        assert C_layer.shape in ((1, 3, 3), (3, 3))

    def test_aliases(self, steel_mat: Law57Params):
        """Verify module aliases consistent_shell_tangent, shell_tangent, shell_membrane_tangent."""
        deps = np.array([0.002, -0.0005, 0.001])
        C_orig = tangent_law57_shell(steel_mat, deps=deps)
        C_alias1 = consistent_shell_tangent(steel_mat, deps=deps)
        C_alias2 = shell_tangent(steel_mat, deps=deps)
        C_alias3 = shell_membrane_tangent(steel_mat, deps=deps)

        np.testing.assert_allclose(C_orig, C_alias1, atol=1e-12)
        np.testing.assert_allclose(C_orig, C_alias2, atol=1e-12)
        np.testing.assert_allclose(C_orig, C_alias3, atol=1e-12)
