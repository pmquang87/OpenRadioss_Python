"""Fortran parity test suite for Milestone M540: LAW37 (/MAT/BIPHAS, /MAT/BIPHASIC).

Validates numerical and physical parity against upstream OpenRadioss reference files:
1. engine/source/materials/mat/mat037/sigeps37.F
2. starter/source/materials/mat/mat037/hm_read_mat37.F
3. starter/source/materials/mat/mat037/m37init.F

Audited areas:
1. Cycle 0 initialization (TIME == 0, sigeps37.F lines 180-197 and m37init.F lines 83-105):
   - Hydrostatic pressure: P = max(1e-30, -1/3 tr(sigma_0)) - P_sh
   - mu1 + 1 = (P - P0)/C1 + 1
   - mu2 + 1 = (P / P0)**(1/gamma)
   - rho1 = rho10 * (mu1 + 1), rho2 = rho20 * (mu2 + 1)
   - alpha_v = (rho - rho2) / (rho1 - rho2)
   - uv37[:, 0] = alpha_v * rho1 = B_1
   - uv37[:, 1] = rho2, uv37[:, 2] = rho1
   - uv37[:, 3] = alpha_v, uv37[:, 4] = 1 - alpha_v
2. Legacy Solver (ISOLVER = 1, sigeps37.F lines 342-392):
   - 2-iteration chord linearization with quadratic solve for rho1
   - Total relative pressure P_total = max(P_min, P) + P0 + P_sh
   - Sound speed c = sqrt(C1 / rho1)
3. Newton Solver (ISOLVER = 2, sigeps37.F lines 234-336):
   - Mass conservation: M1 = B1 * V, M2 = M - M1
   - Pure phase limits (M1/M < 1e-10 pure gas, M2/M < 1e-10 pure liquid)
   - 2D Newton iteration on volume and pressure equilibrium
   - Step clamping: drho1 in [-0.5 rho1, 3 rho1], drho2 in [-0.5 rho2, 3 rho2]
   - Wood's mixture formula: 1 / (rho * c^2) = alpha_v1 / SSP1 + alpha_v2 / SSP2
   - Total pressure: P_total = max(P_min, P) + P_sh
4. Viscous stresses (sigeps37.F lines 319-328 / 374-383):
   - Dynamic viscosity mu = (B1*rho1*nu_l + B2*rho2*nu_g) / rho
   - Volumetric viscosity mu_vol = (B1*rho1*nu_vol_l + B2*rho2*nu_vol_g) / rho
   - sigma_v = 2 * mu * edot + mu_vol * tr(edot) * I
   - sigma = -P_total * I + sigma_v
   - Maximum damping modulus: viscmax = 2*mu + mu_vol
5. Boundary element check (GAM * C1 < 1e-30, sigeps37.F lines 208-232).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law37_biphas import (
    build_law37,
    init_uv37,
    solid_update,
    sound_speed,
    consistent_solid_tangent,
)


# =============================================================================
# Helper: Fortran Oracle for Legacy Solver (sigeps37.F lines 342-392)
# =============================================================================
def fortran_legacy_solver_oracle(
    rho: float,
    uv37_in: np.ndarray,
    c1: float,
    rho10: float,
    rho20: float,
    gamma: float,
    p0: float,
    pmin: float,
    pshift: float,
) -> tuple[float, float, float, np.ndarray]:
    """Exact line-by-line re-implementation of Fortran sigeps37.F lines 342-392."""
    r1 = c1 / rho10
    rho2 = uv37_in[1]
    b1 = uv37_in[0]
    b2 = rho - b1

    # --- Iteration 1 (lines 345-356)
    pold = p0 * (rho2 / rho20) ** gamma
    r2 = gamma * pold / rho2
    c2 = -(1.0 - gamma) * pold + p0
    c12 = c1 - c2
    a = r1
    b = 0.5 * (b1 * r1 + b2 * r2 + c12)
    c_quad = b1 * c12
    rho1 = (b + math.sqrt(max(0.0, b * b - a * c_quad))) / a
    p = r1 * rho1 - c1
    rhn2 = max(1e-30, (p + c2) / r2)

    # --- Iteration 2 (lines 358-363)
    pn2 = pold + p0 * (rhn2 / rho20) ** gamma
    r2 = gamma * pn2 / (rho2 + rhn2)
    b = 0.5 * (b1 * r1 + b2 * r2 + c12)
    rho1 = (b + math.sqrt(max(0.0, b * b - a * c_quad))) / a
    p = r1 * rho1 - c1
    rho2 = max(1e-30, (p + c2) / r2)

    # Output (lines 365-370)
    uv_out = np.zeros(5, dtype=float)
    uv_out[0] = max(0.0, b1)
    uv_out[1] = max(0.0, rho2)
    uv_out[2] = max(0.0, rho1)
    alpha_v1 = b1 / rho1 if rho1 > 1e-30 else 0.0
    if alpha_v1 < 1e-20:
        alpha_v1 = 0.0
    uv_out[3] = max(0.0, alpha_v1)
    uv_out[4] = max(0.0, 1.0 - alpha_v1)

    p_total = max(pmin, p) + p0 + pshift
    ssp = math.sqrt(c1 / rho1)

    return p_total, ssp, p, uv_out


# =============================================================================
# Helper: Fortran Oracle for Newton Solver (sigeps37.F lines 234-336)
# =============================================================================
def fortran_newton_solver_oracle(
    rho: float,
    uv37_in: np.ndarray,
    c1: float,
    rho10: float,
    rho20: float,
    gamma: float,
    p0: float,
    pmin: float,
    pshift: float,
    vol: float = 1.0,
) -> tuple[float, float, float, np.ndarray]:
    """Exact line-by-line re-implementation of Fortran sigeps37.F lines 234-336."""
    r1 = c1 / rho10
    mas = rho * vol
    mas1 = uv37_in[0] * vol
    mas2 = mas - mas1
    rho2 = uv37_in[1]
    rho1 = uv37_in[2]

    uv_out = np.zeros(5, dtype=float)

    if mas1 / mas < 1e-10:
        # Phase 2 (pure gas)
        uv_out[0] = 0.0
        uv_out[3] = 0.0
        uv_out[4] = 1.0
        rho2 = mas / vol
        uv_out[1] = rho2
        uv_out[2] = rho1
        p = p0 * (rho2 / rho20) ** gamma
    elif mas2 / mas < 1e-10:
        # Phase 1 (pure liquid)
        rho1 = mas / vol
        uv_out[0] = rho1
        uv_out[2] = rho1
        uv_out[1] = rho2
        uv_out[3] = 1.0
        uv_out[4] = 0.0
        p = r1 * rho1 - c1 + p0
    else:
        # 2D Newton iteration
        tol = 1e-10
        niter = 20
        it = 1
        err = 1e30
        while it < niter and err > tol:
            p1 = r1 * rho1 - c1 + p0
            p2 = p0 * (rho2 / rho20) ** gamma
            f1 = mas1 / rho1 + mas2 / rho2 - vol
            f2 = p1 - p2
            df11 = -mas1 / (rho1 * rho1)
            df12 = -mas2 / (rho2 * rho2)
            df21 = r1
            df22 = -gamma * p0 / (rho20 ** gamma) * (rho2 ** (gamma - 1.0))
            det = df11 * df22 - df12 * df21
            drho1 = (-df22 * f1 + df12 * f2) / det
            drho2 = (df21 * f1 - df11 * f2) / det
            drho1 = min(3.0 * rho1, max(drho1, -0.5 * rho1))
            drho2 = min(3.0 * rho2, max(drho2, -0.5 * rho2))
            rho1 += drho1
            rho2 += drho2
            err = abs(drho1 / rho1) + abs(drho2 / rho2)
            it += 1

        p = r1 * rho1 - c1 + p0
        uv_out[0] = uv37_in[0]
        uv_out[1] = rho2
        uv_out[2] = rho1
        alpha_v1 = uv_out[0] / rho1 if rho1 > 1e-30 else 0.0
        if alpha_v1 < 1e-20:
            alpha_v1 = 0.0
        uv_out[3] = alpha_v1
        uv_out[4] = 1.0 - alpha_v1

    # Wood's formula for sound speed (lines 293-314)
    ssp1 = r1 * rho1
    ssp2 = gamma * p0 * (rho2 / rho20) ** gamma
    term1 = (uv_out[3] / ssp1) if ssp1 > 0.0 else 0.0
    term2 = (uv_out[4] / ssp2) if ssp2 > 0.0 else 0.0
    ssp_tot = term1 + term2
    ssp = math.sqrt(1.0 / (ssp_tot * rho)) if (ssp_tot > 0.0 and rho > 0.0) else 1e-30

    p_total = max(pmin, p) + pshift
    np.maximum(0.0, uv_out, out=uv_out)

    return p_total, ssp, p, uv_out


# =============================================================================
# Test Suite 1: Cycle 0 Initialization (sigeps37.F lines 180-197)
# =============================================================================
class TestLaw37FortranInitCycle0:
    """Verifies cycle 0 state initialization against sigeps37.F lines 180-197."""

    def test_init_zero_stress_relative_pressure(self):
        """TIME == 0, sigma == 0, default relative pressure formulation P_sh = -P0."""
        c1 = 2.2e9
        rho10 = 1000.0
        rho20 = 1.2
        gamma = 1.4
        p0 = 1.01325e5
        alpha1 = 0.75

        mat = build_law37({
            "rho_l0": rho10,
            "c_l": c1,
            "alpha1": alpha1,
            "rho_g0": rho20,
            "gamma": gamma,
            "p0": p0,
            "psh": 0.0,  # P_sh = -P0 in hm_read_mat37.F lines 180-184
        })

        rho0 = rho10 * alpha1 + (1.0 - alpha1) * rho20
        uv37 = init_uv37(mat, nel=1, rho=rho0)

        # Expected from sigeps37.F lines 180-193:
        # P = max(1e-30, 0) - (-P0) = P0
        # mu1 + 1 = (P0 - P0)/C1 + 1 = 1.0 -> rho1 = rho10
        # mu2 + 1 = (P0/P0)**(1/gamma) = 1.0 -> rho2 = rho20
        # a = (rho0 - rho20) / (rho10 - rho20) = alpha1 = 0.75
        # B1 = a * rho1 = 0.75 * 1000.0 = 750.0
        assert math.isclose(uv37[0, 0], 750.0, rel_tol=1e-12)
        assert math.isclose(uv37[0, 1], rho20, rel_tol=1e-12)
        assert math.isclose(uv37[0, 2], rho10, rel_tol=1e-12)
        assert math.isclose(uv37[0, 3], 0.75, rel_tol=1e-12)
        assert math.isclose(uv37[0, 4], 0.25, rel_tol=1e-12)

    def test_init_hydrostatic_pre_stress(self):
        """TIME == 0 with non-zero initial stress sigma_0."""
        c1 = 2.2e9
        rho10 = 1000.0
        rho20 = 1.2
        gamma = 1.4
        p0 = 1.0e5
        alpha1 = 0.8
        p_sh = -p0  # PSHIFT = -P0

        mat = build_law37({
            "rho_l0": rho10,
            "c_l": c1,
            "alpha1": alpha1,
            "rho_g0": rho20,
            "gamma": gamma,
            "p0": p0,
            "psh": 0.0,
        })

        # Pre-stressed state: sig0 = -1e6 Pa (1 MPa hydrostatic pressure)
        sig0 = np.full((1, 6), 0.0)
        sig0[0, :3] = -1.0e6
        rho = 805.0

        uv37 = init_uv37(mat, nel=1, rho=rho, sig=sig0)

        # sigeps37.F formula:
        sig_hydro = -(-1.0e6 - 1.0e6 - 1.0e6) / 3.0  # = 1.0e6
        p_fortran = max(1e-30, sig_hydro) - p_sh  # = 1.0e6 - (-1.0e5) = 1.1e6 Pa
        mu1p1 = (p_fortran - p0) / c1 + 1.0
        mu2p1 = (p_fortran / p0) ** (1.0 / gamma)
        expected_rho1 = rho10 * mu1p1
        expected_rho2 = rho20 * mu2p1
        expected_a = (rho - expected_rho2) / (expected_rho1 - expected_rho2)
        expected_b1 = expected_a * expected_rho1

        assert math.isclose(uv37[0, 0], expected_b1, rel_tol=1e-10)
        assert math.isclose(uv37[0, 1], expected_rho2, rel_tol=1e-10)
        assert math.isclose(uv37[0, 2], expected_rho1, rel_tol=1e-10)
        assert math.isclose(uv37[0, 3], expected_a, rel_tol=1e-10)
        assert math.isclose(uv37[0, 4], 1.0 - expected_a, rel_tol=1e-10)

    def test_solid_update_time_zero_triggers_init(self):
        """Verify that extra={'time': 0.0} triggers cycle 0 initialization."""
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.6,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
        })
        rho0 = 1000.0 * 0.6 + 1.2 * 0.4
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))
        # Provide pre-existing dummy uv37
        dummy_uv37 = np.full((1, 5), 999.0)
        extra = {"rho": rho0, "uv37": dummy_uv37, "time": 0.0}

        solid_update(mat, sig, deps, dt=1e-5, extra=extra)

        # Must have been overwritten by cycle 0 initialization
        assert math.isclose(extra["uv37"][0, 1], 1.2, rel_tol=1e-6)
        assert math.isclose(extra["uv37"][0, 2], 1000.0, rel_tol=1e-6)
        assert math.isclose(extra["uv37"][0, 3], 0.6, rel_tol=1e-6)


# =============================================================================
# Test Suite 2: Legacy Solver Parity (ISOLVER = 1, sigeps37.F lines 342-392)
# =============================================================================
class TestLaw37LegacySolverParity:
    """Rigorous differential verification of ISOLVER = 1 against Fortran oracle."""

    @pytest.mark.parametrize("compression", [0.98, 0.999, 1.0, 1.001, 1.02])
    def test_isolver1_step_by_step_oracle_parity(self, compression):
        c1 = 2.2e9
        rho10 = 1000.0
        rho20 = 1.2
        gamma = 1.4
        p0 = 1.0e5
        pmin = -p0
        pshift = -p0

        mat = build_law37({
            "rho_l0": rho10,
            "c_l": c1,
            "alpha1": 0.8,
            "rho_g0": rho20,
            "gamma": gamma,
            "p0": p0,
            "isolver": 1,
            "psh": 0.0,
        })

        rho_ref = rho10 * 0.8 + rho20 * 0.2
        rho_curr = rho_ref * compression

        # Create initialized uv37
        uv37_test = np.zeros((1, 5), dtype=float)
        b1_val = 0.8 * rho10 * compression
        uv37_test[0, 0] = b1_val
        uv37_test[0, 1] = rho20 * (compression ** 0.5)
        uv37_test[0, 2] = rho10 * (compression ** 0.1)
        uv37_test[0, 3] = b1_val / uv37_test[0, 2]
        uv37_test[0, 4] = 1.0 - uv37_test[0, 3]

        # 1. Evaluate with Fortran Oracle
        p_oracle, ssp_oracle, _, uv_oracle = fortran_legacy_solver_oracle(
            rho=rho_curr,
            uv37_in=uv37_test[0].copy(),
            c1=c1,
            rho10=rho10,
            rho20=rho20,
            gamma=gamma,
            p0=p0,
            pmin=pmin,
            pshift=pshift,
        )

        # 2. Evaluate with pyradioss solid_update
        sig = np.zeros((1, 6), dtype=float)
        deps = np.zeros((1, 6), dtype=float)
        extra = {"rho": rho_curr, "uv37": uv37_test.copy(), "time": 1e-4}

        sig_out, _, soundsp = solid_update(mat, sig, deps, dt=1e-5, extra=extra)

        # Verify exact numerical parity
        assert math.isclose(-sig_out[0, 0], p_oracle, rel_tol=1e-12, abs_tol=1e-6)
        assert math.isclose(-sig_out[0, 1], p_oracle, rel_tol=1e-12, abs_tol=1e-6)
        assert math.isclose(-sig_out[0, 2], p_oracle, rel_tol=1e-12, abs_tol=1e-6)
        assert math.isclose(soundsp[0], ssp_oracle, rel_tol=1e-12)

        # Verify state variable array parity
        for k in range(5):
            assert math.isclose(extra["uv37"][0, k], uv_oracle[k], rel_tol=1e-12, abs_tol=1e-12)

    def test_b1_preservation_without_overwrite(self):
        """Verify B_1 = uv37[..., 0] is preserved and not overwritten by alpha1."""
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.5,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
            "isolver": 1,
        })
        # Set arbitrary B1 = 600 kg/m3 with rho = 700 kg/m3
        uv37 = np.zeros((1, 5), dtype=float)
        uv37[0, 0] = 600.0
        uv37[0, 1] = 1.3
        uv37[0, 2] = 1005.0
        uv37[0, 3] = 600.0 / 1005.0
        uv37[0, 4] = 1.0 - uv37[0, 3]

        extra = {"rho": 700.0, "uv37": uv37, "time": 1e-4}
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))

        solid_update(mat, sig, deps, dt=1e-5, extra=extra)

        # B1 must remain 600.0
        assert math.isclose(extra["uv37"][0, 0], 600.0, rel_tol=1e-12)


# =============================================================================
# Test Suite 3: Newton Solver Parity (ISOLVER = 2, sigeps37.F lines 234-336)
# =============================================================================
class TestLaw37NewtonSolverParity:
    """Rigorous differential verification of ISOLVER = 2 against Fortran oracle."""

    @pytest.mark.parametrize("compression", [0.99, 1.0, 1.002, 1.01])
    def test_isolver2_step_by_step_oracle_parity(self, compression):
        c1 = 2.2e9
        rho10 = 1000.0
        rho20 = 1.2
        gamma = 1.4
        p0 = 1.0e5
        pmin = -p0
        pshift = -p0

        mat = build_law37({
            "rho_l0": rho10,
            "c_l": c1,
            "alpha1": 0.85,
            "rho_g0": rho20,
            "gamma": gamma,
            "p0": p0,
            "isolver": 2,
            "psh": 0.0,
        })

        rho_ref = rho10 * 0.85 + rho20 * 0.15
        rho_curr = rho_ref * compression

        uv37_test = np.zeros((1, 5), dtype=float)
        b1_val = 0.85 * rho10 * compression
        uv37_test[0, 0] = b1_val
        uv37_test[0, 1] = rho20 * (compression ** 0.7)
        uv37_test[0, 2] = rho10 * (compression ** 0.1)
        uv37_test[0, 3] = b1_val / uv37_test[0, 2]
        uv37_test[0, 4] = 1.0 - uv37_test[0, 3]

        p_oracle, ssp_oracle, _, uv_oracle = fortran_newton_solver_oracle(
            rho=rho_curr,
            uv37_in=uv37_test[0].copy(),
            c1=c1,
            rho10=rho10,
            rho20=rho20,
            gamma=gamma,
            p0=p0,
            pmin=pmin,
            pshift=pshift,
        )

        sig = np.zeros((1, 6), dtype=float)
        deps = np.zeros((1, 6), dtype=float)
        extra = {"rho": rho_curr, "uv37": uv37_test.copy(), "time": 1e-4}

        sig_out, _, soundsp = solid_update(mat, sig, deps, dt=1e-5, extra=extra)

        assert math.isclose(-sig_out[0, 0], p_oracle, rel_tol=1e-9, abs_tol=1e-4)
        assert math.isclose(soundsp[0], ssp_oracle, rel_tol=1e-9)
        for k in range(5):
            assert math.isclose(extra["uv37"][0, k], uv_oracle[k], rel_tol=1e-9, abs_tol=1e-9)

    def test_pure_gas_limit_branch(self):
        """Verify pure gas branch M1/M < 1e-10 (sigeps37.F lines 250-257)."""
        c1 = 2.2e9
        rho10 = 1000.0
        rho20 = 1.2
        gamma = 1.4
        p0 = 1.0e5
        mat = build_law37({
            "rho_l0": rho10,
            "c_l": c1,
            "alpha1": 0.0,
            "rho_g0": rho20,
            "gamma": gamma,
            "p0": p0,
            "isolver": 2,
        })
        rho = 1.3
        extra = {"rho": rho, "time": 1e-4}
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))

        sig_out, _, c = solid_update(mat, sig, deps, dt=1e-5, extra=extra)

        # Expected: P = P0 * (rho / rho20)^gamma
        p_gas = p0 * ((rho / rho20) ** gamma)
        p_rel = p_gas - p0  # pshift = -p0
        assert math.isclose(-sig_out[0, 0], p_rel, rel_tol=1e-5)
        # Sound speed c = sqrt(gamma * P_gas / rho)
        expected_c = math.sqrt(gamma * p_gas / rho)
        assert math.isclose(c[0], expected_c, rel_tol=1e-5)
        # Volume fractions: alpha_v1 = 0, alpha_v2 = 1
        assert math.isclose(extra["uv37"][0, 3], 0.0, abs_tol=1e-12)
        assert math.isclose(extra["uv37"][0, 4], 1.0, rel_tol=1e-12)

    def test_pure_liquid_limit_branch(self):
        """Verify pure liquid branch M2/M < 1e-10 (sigeps37.F lines 258-265)."""
        c1 = 2.2e9
        rho10 = 1000.0
        rho20 = 1.2
        gamma = 1.4
        p0 = 1.0e5
        mat = build_law37({
            "rho_l0": rho10,
            "c_l": c1,
            "alpha1": 1.0,
            "rho_g0": rho20,
            "gamma": gamma,
            "p0": p0,
            "isolver": 2,
        })
        rho = 1008.0
        extra = {"rho": rho, "time": 1e-4}
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))

        sig_out, _, c = solid_update(mat, sig, deps, dt=1e-5, extra=extra)

        # Expected: P = R1 * rho - C1 + P0
        # P_rel = P + pshift = (R1 * rho - C1 + P0) - P0 = C1 * (rho / rho10 - 1)
        expected_p_rel = c1 * (rho / rho10 - 1.0)
        assert math.isclose(-sig_out[0, 0], expected_p_rel, rel_tol=1e-5)
        # Sound speed: c = sqrt(C1 / rho10)
        expected_c = math.sqrt(c1 / rho10)
        assert math.isclose(c[0], expected_c, rel_tol=1e-5)
        # Volume fractions: alpha_v1 = 1, alpha_v2 = 0
        assert math.isclose(extra["uv37"][0, 3], 1.0, rel_tol=1e-12)
        assert math.isclose(extra["uv37"][0, 4], 0.0, abs_tol=1e-12)


# =============================================================================
# Test Suite 4: Viscous Damping Stresses (sigeps37.F lines 319-328 / 374-383)
# =============================================================================
class TestLaw37ViscousStressParity:
    """Verifies viscous stresses and maximum damping modulus."""

    def test_dynamic_and_bulk_viscosity_formulas(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 2.2e9,
            "alpha1": 0.8,
            "nu_l": 1.2e-3,
            "nu_vol_l": 2.4e-3,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
            "nu_g": 1.8e-5,
            "nu_vol_g": 3.6e-5,
            "isolver": 2,
        })
        rho = 800.24
        extra = {"rho": rho, "time": 1e-4}
        sig = np.zeros((1, 6))
        deps = np.zeros((1, 6))
        dt = 1e-5

        # Normal + shear strain rate
        deps[0, 0] = 1e-4  # edot_xx = 10 s^-1
        deps[0, 1] = 5e-5  # edot_yy = 5 s^-1
        deps[0, 3] = 2e-4  # gamma_dot_xy = 20 s^-1

        solid_update(mat, sig, deps, dt=dt, extra=extra)

        uv37 = extra["uv37"]
        b1 = uv37[0, 0]
        b2 = rho - b1
        rho1 = uv37[0, 2]
        rho2 = uv37[0, 1]

        # Fortran lines 319-328:
        mu_fortran = (b1 * rho1 * 1.2e-3 + b2 * rho2 * 1.8e-5) / rho
        mu_vol_fortran = (b1 * rho1 * 2.4e-3 + b2 * rho2 * 3.6e-5) / rho
        vis2 = 2.0 * mu_fortran
        vis3 = mu_vol_fortran

        edot_vol = 10.0 + 5.0
        vv = vis3 * edot_vol
        expected_sigv_xx = vis2 * 10.0 + vv
        expected_sigv_yy = vis2 * 5.0 + vv
        expected_sigv_zz = vv
        expected_sigv_xy = mu_fortran * 20.0

        # Mean pressure
        p_hydro = -(sig[0, 0] + sig[0, 1] + sig[0, 2] - (expected_sigv_xx + expected_sigv_yy + expected_sigv_zz)) / 3.0

        assert math.isclose(sig[0, 0], -p_hydro + expected_sigv_xx, rel_tol=1e-6)
        assert math.isclose(sig[0, 1], -p_hydro + expected_sigv_yy, rel_tol=1e-6)
        assert math.isclose(sig[0, 2], -p_hydro + expected_sigv_zz, rel_tol=1e-6)
        assert math.isclose(sig[0, 3], expected_sigv_xy, rel_tol=1e-6)

        # viscmax check (line 330/385: VISCMAX = VIS2 + VIS3)
        assert "viscmax" in extra
        assert math.isclose(float(extra["viscmax"][0]), float(vis2 + vis3), rel_tol=1e-6)


# =============================================================================
# Test Suite 5: Boundary Element Check (sigeps37.F lines 208-232)
# =============================================================================
class TestLaw37BoundaryElementParity:
    """Verifies boundary element check when GAM * C1 < 1e-30."""

    def test_boundary_element_gas_limit(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 0.0,  # GAM * C1 == 0
            "alpha1": 0.0,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
        })
        sig = np.array([[10.0, 20.0, 30.0, 4.0, 5.0, 6.0]])
        deps = np.zeros((1, 6))
        uv37 = np.zeros((1, 5))
        uv37[0, 2] = 400.0  # uv37[2] / rho_l0 = 0.4 < 0.5
        extra = {"rho": 1.25, "uv37": uv37, "time": 1e-4}

        sig_out, _, c = solid_update(mat, sig, deps, dt=1e-5, extra=extra)

        # Sound speed must be 1e-30
        assert math.isclose(c[0], 1e-30, abs_tol=1e-35)
        # Stress must be unmodified
        assert np.allclose(sig_out, [[10.0, 20.0, 30.0, 4.0, 5.0, 6.0]])
        # UVAR states
        assert math.isclose(extra["uv37"][0, 0], 0.0, abs_tol=1e-12)
        assert math.isclose(extra["uv37"][0, 1], 1.25, rel_tol=1e-6)
        assert math.isclose(extra["uv37"][0, 3], 0.0, abs_tol=1e-12)
        assert math.isclose(extra["uv37"][0, 4], 1.0, rel_tol=1e-12)

    def test_boundary_element_liquid_limit(self):
        mat = build_law37({
            "rho_l0": 1000.0,
            "c_l": 0.0,  # GAM * C1 == 0
            "alpha1": 1.0,
            "rho_g0": 1.2,
            "gamma": 1.4,
            "p0": 1.0e5,
        })
        sig = np.array([[10.0, 20.0, 30.0, 4.0, 5.0, 6.0]])
        deps = np.zeros((1, 6))
        uv37 = np.zeros((1, 5))
        uv37[0, 2] = 800.0  # uv37[2] / rho_l0 = 0.8 >= 0.5
        extra = {"rho": 1005.0, "uv37": uv37, "time": 1e-4}

        sig_out, _, c = solid_update(mat, sig, deps, dt=1e-5, extra=extra)

        assert math.isclose(c[0], 1e-30, abs_tol=1e-35)
        assert np.allclose(sig_out, [[10.0, 20.0, 30.0, 4.0, 5.0, 6.0]])
        assert math.isclose(extra["uv37"][0, 0], 800.0, rel_tol=1e-6)
        assert math.isclose(extra["uv37"][0, 1], 1.2, rel_tol=1e-6)
        assert math.isclose(extra["uv37"][0, 3], 1.0, rel_tol=1e-12)
        assert math.isclose(extra["uv37"][0, 4], 0.0, abs_tol=1e-12)
