"""Audit suite for Milestone M540: LAW37 / BIPHAS consistent algorithmic tangents.

Rigorous numerical and mathematical verification of:
1. Mathematical derivation of the (6, 6) algorithmic tangent stiffness matrix:
   C^{alg} = d(sigma) / d(Delta_eps)
   - Biphasic hydrodynamic deviatoric rate-viscous response:
     sigma_v = 2 * mu * (Delta_eps / Delta_t) + mu_vol * (tr(Delta_eps) / Delta_t) * I
     d(sigma_v) / d(Delta_eps) = (2 * mu / Delta_t) * I_sym + (mu_vol / Delta_t) * (1 (x) 1)
   - Hydrostatic pressure response:
     -P_{total} * I
     Delta_P ~ -K_eff * tr(Delta_eps) = -K_eff * (Delta_eps_xx + Delta_eps_yy + Delta_eps_zz)
     d(-P_{total} * I) / d(Delta_eps) = K_eff * (1 (x) 1)
     where K_eff = rho * c^2 from the mixture sound speed (Wood's formula).
   - Combined algorithmic tangent:
     C11 = C22 = C33 = K_eff + mu_vol / Delta_t + 2 * mu / Delta_t
     C12 = C13 = C23 = K_eff + mu_vol / Delta_t
     C44 = C55 = C66 = mu / Delta_t
     All 24 shear-normal and shear-shear off-diagonal components are zero.
2. Numerical verification via central finite-difference perturbation (h = 1e-7):
   - Pure liquid limit (alpha1 = 1.0)
   - Pure gas limit (alpha1 = 0.0)
   - Two-phase mixture (0 < alpha1 < 1)
   - Non-zero shear viscosity (nu_l > 0, nu_g > 0) and volumetric viscosity (nu_vol_l > 0, nu_vol_g > 0)
   - Multi-axial strain states (combined volumetric compression, shear, and transverse tension)
   - Dual solvers: ISOLVER = 1 (legacy 2-iteration chord solver) and ISOLVER = 2 (Newton-Raphson)
   - Maximum error < 1e-5 across all 36 components (6x6).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from pyradioss import materials as pm
from pyradioss.materials import law37_biphas as l37
from pyradioss.model.entities import Material


# =============================================================================
# Helper Fixtures & Builders
# =============================================================================

def _make_law37(
    rho_l0: float = 1000.0,
    c_l: float = 2.2e9,
    alpha1: float = 0.8,
    nu_l: float = 1e-3,
    nu_vol_l: float = 2e-3,
    rho_g0: float = 1.2,
    gamma: float = 1.4,
    p0: float = 1.01325e5,
    nu_g: float = 1.8e-5,
    nu_vol_g: float = 3.0e-5,
    rho0: float = 0.0,
    psh: float = 0.0,
    isolver: int = 2,
    mat_id: int = 1,
    title: str = "AUDIT_LAW37",
    **kwargs: Any,
) -> Material:
    """Build a LAW37 material instance with explicit parameters."""
    params: Dict[str, Any] = {
        "rho_l0": rho_l0,
        "c_l": c_l,
        "alpha1": alpha1,
        "nu_l": nu_l,
        "nu_vol_l": nu_vol_l,
        "rho_g0": rho_g0,
        "gamma": gamma,
        "p0": p0,
        "nu_g": nu_g,
        "nu_vol_g": nu_vol_g,
        "rho0": rho0,
        "psh": psh,
        "isolver": isolver,
        **kwargs,
    }
    rec = {
        "id": mat_id,
        "title": title,
        "params": params,
    }
    return l37.build_law37(rec)


def _compute_numerical_tangent(
    mat: Material,
    deps0: np.ndarray,
    dt: float,
    rho_base: float,
    uv37_base: Optional[np.ndarray] = None,
    h: float = 1e-7,
    mode: str = "full",
) -> np.ndarray:
    """Compute (6, 6) numerical tangent matrix via central finite differences.

    Parameters
    ----------
    mat : Material
        LAW37 material instance.
    deps0 : np.ndarray
        Base strain increment (1, 6).
    dt : float
        Time step size.
    rho_base : float
        Baseline density at the start of increment.
    uv37_base : np.ndarray, optional
        Baseline state vector (1, 5).
    h : float, default=1e-7
        Perturbation step size.
    mode : {"full", "viscous", "pressure"}
        "full": volumetric mass conservation rho = rho_base / (1 + tr(deps)) and strain-rate viscous stress.
        "viscous": fixed density rho_base, isolates d(sigma_v) / d(deps).
        "pressure": zero strain rates, isolates d(-P*I) / d(deps).

    Returns
    -------
    C_num : np.ndarray
        (6, 6) finite difference tangent matrix.
    """
    deps0_flat = np.asarray(deps0, dtype=float).flatten()
    assert deps0_flat.shape == (6,)
    C_num = np.zeros((6, 6), dtype=float)

    for j in range(6):
        dp = deps0_flat.copy()
        dm = deps0_flat.copy()
        dp[j] += h
        dm[j] -= h

        dp_2d = dp.reshape(1, 6)
        dm_2d = dm.reshape(1, 6)

        if mode == "viscous":
            rhop = rho_base
            rhom = rho_base
            uv_p = uv37_base.copy() if uv37_base is not None else None
            uv_m = uv37_base.copy() if uv37_base is not None else None
            deps_rate_p = dp_2d
            deps_rate_m = dm_2d
        elif mode == "pressure":
            trp = float(dp[0] + dp[1] + dp[2])
            trm = float(dm[0] + dm[1] + dm[2])
            rhop = rho_base / (1.0 + trp)
            rhom = rho_base / (1.0 + trm)
            uv_p = uv37_base.copy() if uv37_base is not None else None
            uv_m = uv37_base.copy() if uv37_base is not None else None
            if uv_p is not None and uv_base_b1_valid(uv37_base):
                uv_p[0, 0] = uv37_base[0, 0] * (rhop / rho_base)
                uv_m[0, 0] = uv37_base[0, 0] * (rhom / rho_base)
            deps_rate_p = np.zeros((1, 6))
            deps_rate_m = np.zeros((1, 6))
        else:  # "full"
            trp = float(dp[0] + dp[1] + dp[2])
            trm = float(dm[0] + dm[1] + dm[2])
            rhop = rho_base / (1.0 + trp)
            rhom = rho_base / (1.0 + trm)
            uv_p = uv37_base.copy() if uv37_base is not None else None
            uv_m = uv37_base.copy() if uv37_base is not None else None
            if uv_p is not None and uv_base_b1_valid(uv37_base):
                uv_p[0, 0] = uv37_base[0, 0] * (rhop / rho_base)
                uv_m[0, 0] = uv37_base[0, 0] * (rhom / rho_base)
            deps_rate_p = dp_2d
            deps_rate_m = dm_2d

        extrap: Dict[str, Any] = {"rho": np.array([rhop]), "dt": dt}
        extram: Dict[str, Any] = {"rho": np.array([rhom]), "dt": dt}
        if uv_p is not None:
            extrap["uv37"] = uv_p
        if uv_m is not None:
            extram["uv37"] = uv_m

        sp, _, _ = l37.solid_update(mat, np.zeros((1, 6)), deps_rate_p, dt=dt, extra=extrap)
        sm, _, _ = l37.solid_update(mat, np.zeros((1, 6)), deps_rate_m, dt=dt, extra=extram)

        C_num[:, j] = (sp[0] - sm[0]) / (2.0 * h)

    return C_num


def uv_base_b1_valid(uv: Optional[np.ndarray]) -> bool:
    """Check if uv history array contains initialized state."""
    if uv is None:
        return False
    return bool(uv.shape[0] > 0 and (uv[0, 1] > 0.0 or uv[0, 2] > 0.0))


def _assert_tangent_close(C_ana: np.ndarray, C_num: np.ndarray, tol: float = 1e-5, label: str = "") -> None:
    """Verify max relative and absolute error across all 36 components is < tol."""
    diff = np.abs(C_ana - C_num)
    scale = np.maximum(np.abs(C_ana), 1.0)
    rel_err = diff / scale
    max_rel = float(np.max(rel_err))
    max_abs = float(np.max(diff))

    assert max_rel < tol, (
        f"{label} verification failed: max relative error {max_rel:.3e} >= {tol:.1e} "
        f"(max abs error {max_abs:.3e})\n"
        f"C_ana:\n{np.round(C_ana, 4)}\nC_num:\n{np.round(C_num, 4)}"
    )


# =============================================================================
# 1. Mathematical Derivation & Exact Algorithmic Moduli Tests
# =============================================================================

class TestMathematicalDerivation:
    """Audit exact mathematical structure of the consistent algorithmic tangent."""

    def test_analytical_tangent_structure(self):
        """Verify Lamé-type structure of C_alg:
        C11 = C22 = C33 = K_eff + mu_vol / dt + 2 * mu / dt
        C12 = C13 = C23 = K_eff + mu_vol / dt
        C44 = C55 = C66 = mu / dt
        All other 24 components are identically 0.
        """
        mat = _make_law37(alpha1=0.8, nu_l=1.2e-3, nu_vol_l=2.4e-3, nu_g=1.5e-5, nu_vol_g=2.5e-5, isolver=2)
        rho_base = mat.rho0
        dt = 1e-5
        extra = {"rho": np.array([rho_base]), "dt": dt}

        # Initialize state
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        C = l37.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra, dt=dt)[0]

        # Analytical components
        uv37 = extra["uv37"]
        b1 = uv37[0, 0]
        b2 = rho_base - b1
        rho1 = uv37[0, 2]
        rho2 = uv37[0, 1]
        nu_l = mat.params["nu_l"]
        nu_vol_l = mat.params["nu_vol_l"]
        nu_g = mat.params["nu_g"]
        nu_vol_g = mat.params["nu_vol_g"]

        mu = (b1 * rho1 * nu_l + b2 * rho2 * nu_g) / rho_base
        mu_vol = (b1 * rho1 * nu_vol_l + b2 * rho2 * nu_vol_g) / rho_base
        gt = mu / dt
        k_vol = mu_vol / dt

        c_sound = float(l37.sound_speed(mat, rho=rho_base, extra=extra))
        k_eff = rho_base * (c_sound ** 2)

        c11_expected = k_eff + k_vol + 2.0 * gt
        c12_expected = k_eff + k_vol
        c44_expected = gt

        # Diagonal normal blocks
        for i in range(3):
            assert math.isclose(C[i, i], c11_expected, rel_tol=1e-12)

        # Off-diagonal normal blocks
        assert math.isclose(C[0, 1], c12_expected, rel_tol=1e-12)
        assert math.isclose(C[0, 2], c12_expected, rel_tol=1e-12)
        assert math.isclose(C[1, 2], c12_expected, rel_tol=1e-12)
        assert math.isclose(C[1, 0], c12_expected, rel_tol=1e-12)
        assert math.isclose(C[2, 0], c12_expected, rel_tol=1e-12)
        assert math.isclose(C[2, 1], c12_expected, rel_tol=1e-12)

        # Diagonal shear blocks
        for s in (3, 4, 5):
            assert math.isclose(C[s, s], c44_expected, rel_tol=1e-12)

        # Off-diagonal shear-normal and shear-shear blocks
        zero_indices = [(i, j) for i in range(6) for j in range(6) if (i >= 3 or j >= 3) and i != j]
        assert len(zero_indices) == 24
        for i, j in zero_indices:
            assert C[i, j] == 0.0

    def test_major_symmetry_and_positive_definiteness(self):
        """Verify C_alg is symmetric and positive-definite for physically valid parameters."""
        mat = _make_law37(alpha1=0.7, nu_l=2.0e-3, nu_vol_l=3.0e-3, nu_g=1.8e-5, nu_vol_g=2.5e-5, isolver=2)
        dt = 1e-4
        extra = {"rho": np.array([mat.rho0]), "dt": dt}
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        C = l37.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra, dt=dt)[0]

        # Major symmetry: C = C^T
        asym = np.max(np.abs(C - C.T))
        assert asym == 0.0

        # Eigenvalues must all be strictly positive
        eigenvalues = np.linalg.eigvalsh(C)
        assert np.all(eigenvalues > 0.0), f"Negative eigenvalues found: {eigenvalues}"

    def test_inviscid_fluid_degeneracy(self):
        """When viscosities are zero (nu=0, nu_vol=0), C_alg reduces to pure rank-1 bulk response."""
        mat = _make_law37(alpha1=0.8, nu_l=0.0, nu_vol_l=0.0, nu_g=0.0, nu_vol_g=0.0, isolver=2)
        dt = 1e-5
        extra = {"rho": np.array([mat.rho0]), "dt": dt}
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        C = l37.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra, dt=dt)[0]
        c_sound = float(l37.sound_speed(mat, rho=mat.rho0, extra=extra))
        k_eff = mat.rho0 * (c_sound ** 2)

        # All shear moduli must be strictly zero
        assert C[3, 3] == 0.0
        assert C[4, 4] == 0.0
        assert C[5, 5] == 0.0

        # Normal block is pure K_eff * 1 (x) 1
        expected_normal = np.full((3, 3), k_eff)
        np.testing.assert_allclose(C[:3, :3], expected_normal, rtol=1e-12)

    def test_woods_formula_mixture_bulk_modulus(self):
        """Verify Wood's mixture formula: 1 / K_eff = alpha_v1 / K1 + alpha_v2 / K2.
        K1 = C_l = 2.2e9 Pa, K2 = gamma * P0 = 1.4 * 101325 = 141855 Pa.
        """
        c_l = 2.2e9
        gamma = 1.4
        p0 = 101325.0
        rho_l0 = 1000.0
        rho_g0 = 1.2
        k1 = c_l
        k2 = gamma * p0

        for a1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            mat = _make_law37(alpha1=a1, c_l=c_l, gamma=gamma, p0=p0, rho_l0=rho_l0, rho_g0=rho_g0, isolver=2)
            rho0 = mat.rho0
            extra = {"rho": np.array([rho0]), "dt": 1e-5}
            l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=1e-5, extra=extra)

            uv37 = extra["uv37"]
            av1 = uv37[0, 3]
            av2 = uv37[0, 4]
            k_wood_expected = 1.0 / (av1 / k1 + av2 / k2)

            c_mix = float(l37.sound_speed(mat, rho=rho0, extra=extra))
            k_eff_computed = rho0 * (c_mix ** 2)

            assert math.isclose(k_eff_computed, k_wood_expected, rel_tol=1e-10)


# =============================================================================
# 2. Pure Liquid Limit (alpha1 = 1.0) Tests
# =============================================================================

class TestPureLiquidLimit:
    """Audit consistent tangent in the pure liquid limit (alpha1 = 1.0)."""

    @pytest.mark.parametrize("isolver", [1, 2])
    def test_pure_liquid_viscous_tangent(self, isolver: int):
        """Pure liquid viscous stress perturbation matches analytical viscous tangent."""
        nu_l = 1.5e-3
        nu_vol_l = 2.5e-3
        mat = _make_law37(alpha1=1.0, nu_l=nu_l, nu_vol_l=nu_vol_l, isolver=isolver)
        rho0 = 1000.0
        dt = 1e-5
        extra = {"rho": np.array([rho0]), "dt": dt}
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        C_ana = l37.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra, dt=dt)[0]
        c = float(l37.sound_speed(mat, rho=rho0, extra=extra))
        k_eff = rho0 * (c ** 2)

        # Subtract hydrostatic bulk stiffness to obtain purely viscous tangent
        C_ana_visc = C_ana.copy()
        C_ana_visc[:3, :3] -= k_eff

        C_num_visc = _compute_numerical_tangent(
            mat, np.zeros((1, 6)), dt, rho0, uv37_base=extra["uv37"], h=1e-7, mode="viscous"
        )
        _assert_tangent_close(C_ana_visc, C_num_visc, tol=1e-6, label=f"Pure liquid ISOLVER={isolver} viscous")

    @pytest.mark.parametrize("isolver", [1, 2])
    def test_pure_liquid_pressure_tangent(self, isolver: int):
        """Pure liquid hydrostatic pressure perturbation matches K_eff = C_l = 2.2e9."""
        mat = _make_law37(alpha1=1.0, nu_l=0.0, nu_vol_l=0.0, isolver=isolver, c_l=2.2e9)
        rho0 = 1000.0
        dt = 1e-5
        extra = {"rho": np.array([rho0]), "dt": dt}
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        # For ISOLVER 1, catastrophic cancellation of float64 in sqrt(b^2 - a*c) occurs at h < 1e-6.
        # h = 1e-6 gives exact precision for ISOLVER 1, h = 1e-7 for ISOLVER 2.
        h = 1e-6 if isolver == 1 else 1e-7
        C_num_press = _compute_numerical_tangent(
            mat, np.zeros((1, 6)), dt, rho0, uv37_base=extra["uv37"], h=h, mode="pressure"
        )

        expected_press = np.zeros((6, 6))
        expected_press[:3, :3] = 2.2e9

        _assert_tangent_close(expected_press, C_num_press, tol=1e-5, label=f"Pure liquid ISOLVER={isolver} pressure")

    @pytest.mark.parametrize("isolver", [1, 2])
    def test_pure_liquid_full_tangent_all_36_components(self, isolver: int):
        """Full coupled consistent tangent matches numerical derivatives for all 36 components."""
        nu_l = 2.0e-3
        nu_vol_l = 3.5e-3
        mat = _make_law37(alpha1=1.0, nu_l=nu_l, nu_vol_l=nu_vol_l, isolver=isolver, c_l=2.2e9)
        rho0 = 1000.0
        dt = 1e-5
        extra = {"rho": np.array([rho0]), "dt": dt}
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        C_ana = l37.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra, dt=dt)[0]
        h = 1e-6 if isolver == 1 else 1e-7
        C_num = _compute_numerical_tangent(
            mat, np.zeros((1, 6)), dt, rho0, uv37_base=extra["uv37"], h=h, mode="full"
        )

        _assert_tangent_close(C_ana, C_num, tol=1e-5, label=f"Pure liquid ISOLVER={isolver} full tangent")


# =============================================================================
# 3. Pure Gas Limit (alpha1 = 0.0) Tests
# =============================================================================

class TestPureGasLimit:
    """Audit consistent tangent in the pure gas limit (alpha1 = 0.0)."""

    @pytest.mark.parametrize("isolver", [1, 2])
    def test_pure_gas_viscous_tangent(self, isolver: int):
        """Pure gas viscous stress perturbation matches gas viscous moduli."""
        nu_g = 1.8e-5
        nu_vol_g = 3.2e-5
        mat = _make_law37(alpha1=0.0, nu_g=nu_g, nu_vol_g=nu_vol_g, isolver=isolver)
        rho0 = 1.2
        dt = 1e-5
        extra = {"rho": np.array([rho0]), "dt": dt}
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        C_ana = l37.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra, dt=dt)[0]
        c = float(l37.sound_speed(mat, rho=rho0, extra=extra))
        k_eff = rho0 * (c ** 2)

        C_ana_visc = C_ana.copy()
        C_ana_visc[:3, :3] -= k_eff

        C_num_visc = _compute_numerical_tangent(
            mat, np.zeros((1, 6)), dt, rho0, uv37_base=extra["uv37"], h=1e-7, mode="viscous"
        )
        _assert_tangent_close(C_ana_visc, C_num_visc, tol=1e-6, label=f"Pure gas ISOLVER={isolver} viscous")

    def test_pure_gas_pressure_tangent_isolver2(self):
        """Pure gas hydrostatic pressure perturbation matches isentropic gas bulk modulus K = gamma * P0."""
        gamma = 1.4
        p0 = 1.0e5
        k_gas_expected = gamma * p0
        mat = _make_law37(alpha1=0.0, gamma=gamma, p0=p0, nu_g=0.0, nu_vol_g=0.0, isolver=2)
        rho0 = 1.2
        dt = 1e-5
        extra = {"rho": np.array([rho0]), "dt": dt}
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        C_num_press = _compute_numerical_tangent(
            mat, np.zeros((1, 6)), dt, rho0, uv37_base=extra["uv37"], h=1e-7, mode="pressure"
        )

        expected_press = np.zeros((6, 6))
        expected_press[:3, :3] = k_gas_expected

        _assert_tangent_close(expected_press, C_num_press, tol=1e-5, label="Pure gas ISOLVER=2 pressure")

    def test_pure_gas_full_tangent_all_36_components(self):
        """Full coupled consistent tangent in pure gas matches all 36 components under ISOLVER=2."""
        mat = _make_law37(alpha1=0.0, nu_g=1.5e-5, nu_vol_g=2.8e-5, isolver=2)
        rho0 = 1.2
        dt = 1e-5
        extra = {"rho": np.array([rho0]), "dt": dt}
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        C_ana = l37.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra, dt=dt)[0]
        C_num = _compute_numerical_tangent(
            mat, np.zeros((1, 6)), dt, rho0, uv37_base=extra["uv37"], h=1e-7, mode="full"
        )

        _assert_tangent_close(C_ana, C_num, tol=1e-5, label="Pure gas full tangent")


# =============================================================================
# 4. Two-Phase Mixture (0 < alpha1 < 1) Tests
# =============================================================================

class TestTwoPhaseMixture:
    """Audit consistent tangent for arbitrary two-phase liquid-gas mixtures."""

    @pytest.mark.parametrize("alpha1", [0.1, 0.3, 0.5, 0.8, 0.95])
    @pytest.mark.parametrize("isolver", [1, 2])
    def test_two_phase_mixture_viscous_tangent(self, alpha1: float, isolver: int):
        """Viscous tangent verification across volume fractions and dual solvers."""
        mat = _make_law37(
            alpha1=alpha1,
            nu_l=1.0e-3,
            nu_vol_l=2.0e-3,
            nu_g=1.8e-5,
            nu_vol_g=3.0e-5,
            isolver=isolver,
        )
        rho0 = mat.rho0
        dt = 1e-5
        extra = {"rho": np.array([rho0]), "dt": dt}
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        C_ana = l37.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra, dt=dt)[0]
        c = float(l37.sound_speed(mat, rho=rho0, extra=extra))
        k_eff = rho0 * (c ** 2)

        C_ana_visc = C_ana.copy()
        C_ana_visc[:3, :3] -= k_eff

        C_num_visc = _compute_numerical_tangent(
            mat, np.zeros((1, 6)), dt, rho0, uv37_base=extra["uv37"], h=1e-7, mode="viscous"
        )
        _assert_tangent_close(C_ana_visc, C_num_visc, tol=1e-6, label=f"Two-phase alpha={alpha1} isolver={isolver} viscous")

    @pytest.mark.parametrize("alpha1", [0.1, 0.3, 0.5, 0.8, 0.95])
    def test_two_phase_mixture_pressure_tangent_isolver2(self, alpha1: float):
        """Pressure tangent matches Wood's mixture bulk modulus across volume fractions under ISOLVER=2."""
        mat = _make_law37(
            alpha1=alpha1,
            nu_l=0.0,
            nu_vol_l=0.0,
            nu_g=0.0,
            nu_vol_g=0.0,
            isolver=2,
        )
        rho0 = mat.rho0
        dt = 1e-5
        extra = {"rho": np.array([rho0]), "dt": dt}
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        c = float(l37.sound_speed(mat, rho=rho0, extra=extra))
        k_eff = rho0 * (c ** 2)

        # h = 1e-6 clears the Newton-Raphson stopping tolerance (tol=1e-10) noise
        # where tol / (2*h) ~ 1e-4 relative noise occurs at h = 1e-7.
        C_num_press = _compute_numerical_tangent(
            mat, np.zeros((1, 6)), dt, rho0, uv37_base=extra["uv37"], h=1e-6, mode="pressure"
        )

        expected_press = np.zeros((6, 6))
        expected_press[:3, :3] = k_eff

        _assert_tangent_close(expected_press, C_num_press, tol=1e-5, label=f"Two-phase alpha={alpha1} pressure")

    @pytest.mark.parametrize("alpha1", [0.1, 0.3, 0.5, 0.8, 0.95])
    def test_two_phase_mixture_full_tangent_isolver2(self, alpha1: float):
        """Full coupled consistent tangent verification across volume fractions under ISOLVER=2."""
        mat = _make_law37(
            alpha1=alpha1,
            nu_l=1.2e-3,
            nu_vol_l=2.2e-3,
            nu_g=1.6e-5,
            nu_vol_g=2.7e-5,
            isolver=2,
        )
        rho0 = mat.rho0
        dt = 1e-5
        extra = {"rho": np.array([rho0]), "dt": dt}
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        C_ana = l37.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra, dt=dt)[0]
        C_num = _compute_numerical_tangent(
            mat, np.zeros((1, 6)), dt, rho0, uv37_base=extra["uv37"], h=1e-6, mode="full"
        )

        _assert_tangent_close(C_ana, C_num, tol=1e-5, label=f"Two-phase alpha={alpha1} full tangent")


# =============================================================================
# 5. Multi-Axial Strain States Tests
# =============================================================================

class TestMultiAxialStrainStates:
    """Audit consistent tangent under combined volumetric compression, shear, and tension."""

    @pytest.fixture
    def multiaxial_states(self) -> list[Tuple[str, np.ndarray]]:
        """Representative 3D multi-axial strain increments."""
        return [
            (
                "volumetric_compression_plus_shear",
                np.array([[-0.002, -0.0015, -0.001, 0.003, -0.002, 0.0015]]),
            ),
            (
                "volumetric_tension_plus_shear",
                np.array([[0.0015, 0.001, 0.0008, -0.0025, 0.0018, -0.0012]]),
            ),
            (
                "pure_isochoric_shear",
                np.array([[0.001, -0.0005, -0.0005, 0.004, -0.003, 0.002]]),
            ),
            (
                "triaxial_unequal_shear",
                np.array([[-0.003, 0.001, -0.0008, 0.0015, 0.0022, -0.0018]]),
            ),
        ]

    def test_multiaxial_viscous_tangents(self, multiaxial_states):
        """Viscous tangent matches numerical perturbation on arbitrary multi-axial strain states."""
        mat = _make_law37(alpha1=0.75, nu_l=1.4e-3, nu_vol_l=2.6e-3, nu_g=1.8e-5, nu_vol_g=3.0e-5, isolver=2)
        rho0 = mat.rho0
        dt = 1e-5
        extra = {"rho": np.array([rho0]), "dt": dt}
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        C_ana = l37.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra, dt=dt)[0]
        c = float(l37.sound_speed(mat, rho=rho0, extra=extra))
        k_eff = rho0 * (c ** 2)

        C_ana_visc = C_ana.copy()
        C_ana_visc[:3, :3] -= k_eff

        for name, deps in multiaxial_states:
            C_num_visc = _compute_numerical_tangent(
                mat, deps, dt, rho0, uv37_base=extra["uv37"], h=1e-7, mode="viscous"
            )
            _assert_tangent_close(C_ana_visc, C_num_visc, tol=1e-6, label=f"Multiaxial {name} viscous")

    def test_multiaxial_directional_derivatives(self, multiaxial_states):
        """Directional derivatives along unit vectors match C_alg @ d for multi-axial states."""
        mat = _make_law37(alpha1=0.8, nu_l=1.0e-3, nu_vol_l=2.0e-3, nu_g=1.5e-5, nu_vol_g=2.5e-5, isolver=2)
        rho0 = mat.rho0
        dt = 1e-5
        extra = {"rho": np.array([rho0]), "dt": dt}
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        C_ana = l37.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra, dt=dt)[0]

        test_directions: list[Tuple[str, np.ndarray]] = [
            ("hydrostatic_dilation", np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) / math.sqrt(3.0)),
            ("pure_shear_xy", np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])),
            (
                "combined_multiaxial",
                np.array([1.0, -0.5, -0.5, 1.0, -1.0, 0.5]) / np.linalg.norm([1.0, -0.5, -0.5, 1.0, -1.0, 0.5]),
            ),
        ]
        for name, deps in multiaxial_states:
            test_directions.append((name, deps[0] / np.linalg.norm(deps[0])))

        h = 1e-6
        for name, d_vec in test_directions:
            dp = (h * d_vec).reshape(1, 6)
            dm = (-h * d_vec).reshape(1, 6)

            trp = float(dp[0, 0] + dp[0, 1] + dp[0, 2])
            trm = float(dm[0, 0] + dm[0, 1] + dm[0, 2])
            rhop = rho0 / (1.0 + trp)
            rhom = rho0 / (1.0 + trm)

            uv_p = extra["uv37"].copy()
            uv_p[0, 0] *= (rhop / rho0)
            uv_m = extra["uv37"].copy()
            uv_m[0, 0] *= (rhom / rho0)

            sp, _, _ = l37.solid_update(
                mat, np.zeros((1, 6)), dp, dt=dt, extra={"rho": np.array([rhop]), "dt": dt, "uv37": uv_p}
            )
            sm, _, _ = l37.solid_update(
                mat, np.zeros((1, 6)), dm, dt=dt, extra={"rho": np.array([rhom]), "dt": dt, "uv37": uv_m}
            )

            directional_fd = (sp[0] - sm[0]) / (2.0 * h)
            directional_ana = C_ana @ d_vec

            rel_err = np.linalg.norm(directional_fd - directional_ana) / np.linalg.norm(directional_ana)
            assert rel_err < 1e-5, f"Directional derivative for {name} failed: rel_err={rel_err:.3e}"


# =============================================================================
# 6. Solvers Audit (ISOLVER = 1 vs ISOLVER = 2)
# =============================================================================

class TestSolversAudit:
    """Audit algorithmic tangent behavior across ISOLVER = 1 and ISOLVER = 2."""

    def test_identical_viscous_tangent_across_solvers(self):
        """Deviatoric rate-viscous tangent is identical between ISOLVER 1 and ISOLVER 2."""
        mat1 = _make_law37(alpha1=0.7, nu_l=1.5e-3, nu_vol_l=2.5e-3, nu_g=2.0e-5, nu_vol_g=3.0e-5, isolver=1)
        mat2 = _make_law37(alpha1=0.7, nu_l=1.5e-3, nu_vol_l=2.5e-3, nu_g=2.0e-5, nu_vol_g=3.0e-5, isolver=2)
        rho0 = mat1.rho0
        dt = 1e-5

        extra1 = {"rho": np.array([rho0]), "dt": dt}
        extra2 = {"rho": np.array([rho0]), "dt": dt}
        l37.solid_update(mat1, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra1)
        l37.solid_update(mat2, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra2)

        C_visc1 = _compute_numerical_tangent(mat1, np.zeros((1, 6)), dt, rho0, uv37_base=extra1["uv37"], mode="viscous")
        C_visc2 = _compute_numerical_tangent(mat2, np.zeros((1, 6)), dt, rho0, uv37_base=extra2["uv37"], mode="viscous")

        np.testing.assert_allclose(C_visc1, C_visc2, rtol=1e-12, atol=1e-12)

    def test_isolver1_mixture_sound_speed_override(self):
        """When extra['mixture_sound_speed']=True, ISOLVER 1 tangent uses Wood's mixture sound speed."""
        mat = _make_law37(alpha1=0.8, nu_l=1.0e-3, nu_vol_l=2.0e-3, nu_g=1.5e-5, nu_vol_g=2.5e-5, isolver=1)
        rho0 = mat.rho0
        dt = 1e-5
        extra = {"rho": np.array([rho0]), "dt": dt, "mixture_sound_speed": True}
        l37.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra)

        C_mix = l37.consistent_solid_tangent(mat, np.zeros((1, 6)), extra=extra, dt=dt)[0]

        # Compare against ISOLVER 2 which uses Wood's formula by default
        mat2 = _make_law37(alpha1=0.8, nu_l=1.0e-3, nu_vol_l=2.0e-3, nu_g=1.5e-5, nu_vol_g=2.5e-5, isolver=2)
        extra2 = {"rho": np.array([rho0]), "dt": dt}
        l37.solid_update(mat2, np.zeros((1, 6)), np.zeros((1, 6)), dt=dt, extra=extra2)
        C_isolver2 = l37.consistent_solid_tangent(mat2, np.zeros((1, 6)), extra=extra2, dt=dt)[0]

        np.testing.assert_allclose(C_mix, C_isolver2, rtol=1e-10)

    def test_call_signatures_and_vectorization(self):
        """Verify tangent call signatures (1D sig, 2D batched sig, None, empty)."""
        mat = _make_law37(alpha1=0.8, isolver=2)
        rho0 = mat.rho0
        dt = 1e-5

        # 1. 1D stress input (6,)
        sig_1d = np.zeros(6)
        extra_1d = {"rho": rho0, "dt": dt}
        D_1d = l37.consistent_solid_tangent(mat, sig_1d, extra=extra_1d)
        assert D_1d.shape == (1, 6, 6)

        # 2. 2D batched stress input (nel, 6)
        nel = 8
        sig_batch = np.zeros((nel, 6))
        extra_batch = {"rho": np.full(nel, rho0), "dt": dt}
        D_batch = l37.consistent_solid_tangent(mat, sig_batch, extra=extra_batch)
        assert D_batch.shape == (nel, 6, 6)
        for i in range(nel):
            np.testing.assert_allclose(D_batch[i], D_1d[0])

        # 3. Empty stress input (0, 6)
        D_empty = l37.consistent_solid_tangent(mat, np.zeros((0, 6)))
        assert D_empty.shape == (0, 6, 6)

        # 4. Dispatch through pyradioss.materials.solid_tangent
        D_disp = pm.solid_tangent(mat, sig_batch, None, None, extra=extra_batch)
        assert D_disp.shape == (nel, 6, 6)
        np.testing.assert_allclose(D_disp, D_batch)
