"""
Fortran Parity Verification Tests for LAW25 Composite Material Model (/MAT/LAW25).
Milestone M543: Auditor 2A Verification Suite.

Validates exact parity against reference OpenRadioss Fortran subroutines:
  - starter/source/materials/mat/mat025/read_mat25_tsaiwu.F90
  - starter/source/materials/mat/mat025/read_mat25_crasurv.F90
  - starter/source/materials/mat/mat025/hm_read_mat25.F
  - engine/source/materials/mat/mat025/sigeps25c.F
  - engine/source/materials/mat/mat025/mat25_tsaiwu_c.F90
  - engine/source/materials/mat/mat025/mat25_crasurv_c.F90
  - engine/source/materials/mat/mat025/m25law.F
  - engine/source/materials/mat/mat025/m25crak.F
  - engine/source/materials/mat/mat025/m25delam.F
"""

from __future__ import annotations

import math
from typing import Any, Dict
import numpy as np
import pytest

from pyradioss.materials import law25_composite
from pyradioss.materials.law25_composite import (
    build_law25,
    build_crasurv,
    tsai_wu_coefficients,
    tsai_wu_yield_criterion,
    tsai_wu_flow_normal,
    hardening_yield,
    strain_rate_factor,
    tensile_damage,
    sound_speed_law25,
    sound_speed,
    extra_shapes,
    shell_membrane_tangent,
    consistent_shell_tangent,
    solid_stiffness_matrix,
    consistent_solid_tangent,
    shell_update,
    solid_update,
)


# ============================================================================
# 1. Tsai-Wu Failure Coefficients Parity (read_mat25_tsaiwu.F90:310-316)
# ============================================================================

class TestTsaiWuCoefficients:
    """Verifies Tsai-Wu coefficients against exact Fortran lines 310-316."""

    def test_symmetric_tension_compression(self):
        """When sigyt == sigyc, linear coefficients F1 and F2 must be zero."""
        sig1 = 1000.0
        sig2 = 100.0
        sig12 = 60.0
        coeffs = tsai_wu_coefficients(sig1, sig1, sig2, sig2, sig12, alpha=1.0)

        assert coeffs["F1"] == pytest.approx(0.0, abs=1e-15)
        assert coeffs["F2"] == pytest.approx(0.0, abs=1e-15)
        assert coeffs["F11"] == pytest.approx(1.0 / (sig1 * sig1))
        assert coeffs["F22"] == pytest.approx(1.0 / (sig2 * sig2))
        assert coeffs["F33"] == pytest.approx(1.0 / (sig12 * sig12))
        expected_f12 = -0.5 * 1.0 * math.sqrt(coeffs["F11"] * coeffs["F22"])
        assert coeffs["F12"] == pytest.approx(expected_f12)

    def test_asymmetric_tension_compression(self):
        """Asymmetric tension/compression yields nonzero F1, F2."""
        sigyt1, sigyc1 = 1500.0, 1200.0
        sigyt2, sigyc2 = 50.0, 250.0
        sig12 = 80.0
        alpha = 1.0

        coeffs = tsai_wu_coefficients(sigyt1, sigyc1, sigyt2, sigyc2, sig12, alpha=alpha)

        # Fortran: F1 = 1/sigyt1 - 1/sigyc1
        expected_f1 = 1.0 / 1500.0 - 1.0 / 1200.0
        expected_f2 = 1.0 / 50.0 - 1.0 / 250.0
        expected_f11 = 1.0 / (1500.0 * 1200.0)
        expected_f22 = 1.0 / (50.0 * 250.0)
        expected_f33 = 1.0 / (80.0 * 80.0)
        expected_f12 = -1.0 / (2.0 * math.sqrt(1500.0 * 1200.0 * 50.0 * 250.0))

        assert coeffs["F1"] == pytest.approx(expected_f1, rel=1e-14)
        assert coeffs["F2"] == pytest.approx(expected_f2, rel=1e-14)
        assert coeffs["F11"] == pytest.approx(expected_f11, rel=1e-14)
        assert coeffs["F22"] == pytest.approx(expected_f22, rel=1e-14)
        assert coeffs["F33"] == pytest.approx(expected_f33, rel=1e-14)
        assert coeffs["F12"] == pytest.approx(expected_f12, rel=1e-14)

    def test_alpha_reduction_factor(self):
        """alpha scales F12 linearly."""
        yt1, yc1 = 1000.0, 1000.0
        yt2, yc2 = 100.0, 100.0
        c_alpha1 = tsai_wu_coefficients(yt1, yc1, yt2, yc2, 50.0, alpha=1.0)
        c_alpha0 = tsai_wu_coefficients(yt1, yc1, yt2, yc2, 50.0, alpha=0.0)
        c_alpha_half = tsai_wu_coefficients(yt1, yc1, yt2, yc2, 50.0, alpha=0.5)

        assert c_alpha0["F12"] == 0.0
        assert c_alpha_half["F12"] == pytest.approx(0.5 * c_alpha1["F12"])

    def test_separate_shear_tension_compression(self):
        """Separate sigyt12 and sigyc12 passed as a tuple."""
        sigyt12, sigyc12 = 60.0, 90.0
        coeffs = tsai_wu_coefficients(1000.0, 1000.0, 100.0, 100.0, (sigyt12, sigyc12))
        expected_f33 = 1.0 / (60.0 * 90.0)
        assert coeffs["F33"] == pytest.approx(expected_f33)


# ============================================================================
# 2. Tsai-Wu Yield Criterion Parity (mat25_tsaiwu_c.F90:480-484)
# ============================================================================

class TestTsaiWuYieldCriterion:
    """Verifies W = F1*s1 + F2*s2 + F11*s1^2 + F22*s2^2 + 2*F12*s1*s2 + F33*s12^2."""

    @pytest.fixture
    def coeffs(self):
        return tsai_wu_coefficients(
            sigyt1=1200.0, sigyc1=1000.0,
            sigyt2=60.0, sigyc2=200.0,
            sig12=75.0, alpha=1.0,
        )

    def test_pure_uniaxial_tension_1(self, coeffs):
        """Pure tensile stress along dir 1 at sigyt1 must give W = 1.0 exactly."""
        s1 = 1200.0
        s2 = 0.0
        s12 = 0.0
        w = tsai_wu_yield_criterion(s1, s2, s12, **coeffs)
        assert w == pytest.approx(1.0, rel=1e-12)

    def test_pure_uniaxial_compression_1(self, coeffs):
        """Pure compressive stress along dir 1 at -sigyc1 must give W = 1.0 exactly."""
        s1 = -1000.0
        s2 = 0.0
        s12 = 0.0
        w = tsai_wu_yield_criterion(s1, s2, s12, **coeffs)
        assert w == pytest.approx(1.0, rel=1e-12)

    def test_pure_uniaxial_tension_2(self, coeffs):
        """Pure tensile stress along dir 2 at sigyt2 must give W = 1.0 exactly."""
        s1 = 0.0
        s2 = 60.0
        s12 = 0.0
        w = tsai_wu_yield_criterion(s1, s2, s12, **coeffs)
        assert w == pytest.approx(1.0, rel=1e-12)

    def test_pure_uniaxial_compression_2(self, coeffs):
        """Pure compressive stress along dir 2 at -sigyc2 must give W = 1.0 exactly."""
        s1 = 0.0
        s2 = -200.0
        s12 = 0.0
        w = tsai_wu_yield_criterion(s1, s2, s12, **coeffs)
        assert w == pytest.approx(1.0, rel=1e-12)

    def test_pure_shear_12(self, coeffs):
        """Pure shear stress at sig12 must give W = 1.0 exactly."""
        s1 = 0.0
        s2 = 0.0
        s12 = 75.0
        w = tsai_wu_yield_criterion(s1, s2, s12, **coeffs)
        assert w == pytest.approx(1.0, rel=1e-12)

        # Shear sign symmetry: negative shear gives identical W
        w_neg = tsai_wu_yield_criterion(s1, s2, -75.0, **coeffs)
        assert w_neg == pytest.approx(1.0, rel=1e-12)

    def test_elastic_interior_and_plastic_exterior(self, coeffs):
        """Points inside yield surface have W < 1; points outside have W > 1."""
        w_in = tsai_wu_yield_criterion(600.0, 30.0, 30.0, **coeffs)
        assert w_in < 1.0

        w_out = tsai_wu_yield_criterion(1300.0, 70.0, 80.0, **coeffs)
        assert w_out > 1.0


# ============================================================================
# 3. Flow Vector Normal Parity (mat25_tsaiwu_c.F90:501-503)
# ============================================================================

class TestTsaiWuFlowNormal:
    """Verifies gradient dF/ds matches analytical derivatives and finite differences."""

    @pytest.fixture
    def coeffs(self):
        return tsai_wu_coefficients(
            sigyt1=1500.0, sigyc1=1200.0,
            sigyt2=80.0, sigyc2=180.0,
            sig12=65.0, alpha=0.8,
        )

    def test_flow_normal_analytical_formulas(self, coeffs):
        """Check exact formula:
        dF/ds1 = F1 + 2*F11*s1 + 2*F12*s2
        dF/ds2 = F2 + 2*F22*s2 + 2*F12*s1
        dF/ds12 = 2*F33*s12
        """
        s1, s2, s12 = 500.0, 40.0, 30.0
        df1, df2, df12 = tsai_wu_flow_normal(s1, s2, s12, **coeffs)

        expected_df1 = coeffs["F1"] + 2.0 * coeffs["F11"] * s1 + 2.0 * coeffs["F12"] * s2
        expected_df2 = coeffs["F2"] + 2.0 * coeffs["F22"] * s2 + 2.0 * coeffs["F12"] * s1
        expected_df12 = 2.0 * coeffs["F33"] * s12

        assert df1 == pytest.approx(expected_df1, rel=1e-14)
        assert df2 == pytest.approx(expected_df2, rel=1e-14)
        assert df12 == pytest.approx(expected_df12, rel=1e-14)

    def test_flow_normal_finite_difference_check(self, coeffs):
        """Check gradient normal against central finite differences of W."""
        s1, s2, s12 = 600.0, 50.0, 40.0
        df1, df2, df12 = tsai_wu_flow_normal(s1, s2, s12, **coeffs)

        eps = 1e-6
        fd_1 = (tsai_wu_yield_criterion(s1 + eps, s2, s12, **coeffs) -
                tsai_wu_yield_criterion(s1 - eps, s2, s12, **coeffs)) / (2.0 * eps)
        fd_2 = (tsai_wu_yield_criterion(s1, s2 + eps, s12, **coeffs) -
                tsai_wu_yield_criterion(s1, s2 - eps, s12, **coeffs)) / (2.0 * eps)
        fd_12 = (tsai_wu_yield_criterion(s1, s2, s12 + eps, **coeffs) -
                 tsai_wu_yield_criterion(s1, s2, s12 - eps, **coeffs)) / (2.0 * eps)

        assert df1 == pytest.approx(fd_1, rel=1e-5)
        assert df2 == pytest.approx(fd_2, rel=1e-5)
        assert df12 == pytest.approx(fd_12, rel=1e-5)


# ============================================================================
# 4. Plastic Work and Hardening (mat25_tsaiwu_c.F90:463-474)
# ============================================================================

class TestHardeningYield:
    """Verifies f_yld = min(fmax, (1 + b * wpla^n) * epspfac)."""

    def test_perfect_plasticity(self):
        """When b = 0, f_yld equals epspfac regardless of wpla."""
        assert hardening_yield(wpla=0.0, b=0.0, n=1.0, epspfac=1.0) == 1.0
        assert hardening_yield(wpla=10.0, b=0.0, n=1.0, epspfac=1.5) == 1.5

    def test_linear_hardening(self):
        """When n = 1, hardening increases linearly with plastic work."""
        wpla = 2.5
        b = 0.4
        epspfac = 1.2
        expected = (1.0 + 0.4 * 2.5) * 1.2  # (1 + 1) * 1.2 = 2.4
        assert hardening_yield(wpla, b, n=1.0, epspfac=epspfac) == pytest.approx(expected)

    def test_power_law_hardening(self):
        """When n = 0.5, hardening follows square root of plastic work."""
        wpla = 4.0
        b = 0.5
        expected = (1.0 + 0.5 * math.sqrt(4.0))  # 1 + 1 = 2.0
        assert hardening_yield(wpla, b, n=0.5, epspfac=1.0) == pytest.approx(expected)

    def test_fmax_saturation(self):
        """f_yld is capped strictly at fmax."""
        wpla = 100.0
        b = 10.0
        fmax = 5.0
        assert hardening_yield(wpla, b, n=1.0, epspfac=1.0, fmax=fmax) == 5.0


# ============================================================================
# 5. Strain Rate Factor Parity (mat25_tsaiwu_c.F90:453-462 & Cowper-Symonds)
# ============================================================================

class TestStrainRateFactor:
    """Verifies strain rate factors: Fortran logarithmic law and Cowper-Symonds."""

    def test_fortran_log_strain_rate(self):
        """Fortran law: epspfac = 1 + c * log(eps_dot / epdr) for eps_dot > epdr."""
        c = 0.15
        epdr = 10.0

        # Sub-threshold: eps_dot <= epdr -> epspfac = 1.0
        assert strain_rate_factor(5.0, c=c, epdr=epdr, formulation="log") == 1.0
        assert strain_rate_factor(10.0, c=c, epdr=epdr, formulation="log") == 1.0

        # Super-threshold: eps_dot = 1000.0
        expected = 1.0 + 0.15 * math.log(1000.0 / 10.0)
        assert strain_rate_factor(1000.0, c=c, epdr=epdr, formulation="log") == pytest.approx(expected)

    def test_cowper_symonds_power_law(self):
        """Cowper-Symonds: epspfac = 1 + (eps_dot / c)^(1 / epdr)."""
        c = 40.0
        epdr = 5.0  # exponent P
        eps_dot = 80.0
        expected = 1.0 + (80.0 / 40.0) ** (1.0 / 5.0)
        assert strain_rate_factor(eps_dot, c=c, epdr=epdr, formulation="cowper_symonds") == pytest.approx(expected)

    def test_zero_rate_or_coefficient(self):
        """When c = 0 or eps_dot = 0, epspfac must be exactly 1.0."""
        assert strain_rate_factor(0.0, c=0.1, epdr=1.0) == 1.0
        assert strain_rate_factor(100.0, c=0.0, epdr=1.0) == 1.0


# ============================================================================
# 6. Damage Evolution Parity (m25crak.F:74-76 & m25law.F:264-266)
# ============================================================================

class TestTensileDamageEvolution:
    """Verifies dam1 = (epst - epst1)/(epsm1 - epst1), dam2 = dam1 * epsm1 / epst."""

    def test_below_threshold_zero_damage(self):
        """Strain below epst1 produces no damage."""
        assert tensile_damage(epst=0.002, epst_limit=0.005, epsm_limit=0.02) == 0.0

    def test_at_onset_threshold(self):
        """Strain exactly at epst1 produces zero damage."""
        assert tensile_damage(epst=0.005, epst_limit=0.005, epsm_limit=0.02) == 0.0

    def test_at_maximum_strain_threshold(self):
        """Strain at epsm1 produces damage = dmax."""
        epst1 = 0.005
        epsm1 = 0.02
        dmax = 0.95
        # At epst = epsm1: dam1 = 1.0, dam2 = 1.0 * epsm1 / epsm1 = 1.0
        # dmg = min(dam2, dmax) = dmax
        assert tensile_damage(epst=epsm1, epst_limit=epst1, epsm_limit=epsm1, dmax=dmax) == pytest.approx(dmax)

    def test_intermediate_damage(self):
        """Damage between epst1 and epsm1 matches exact Fortran formula."""
        epst1 = 0.004
        epsm1 = 0.020
        epst = 0.012
        dmax = 0.999

        dam1 = (0.012 - 0.004) / (0.020 - 0.004)  # 0.008 / 0.016 = 0.5
        dam2 = dam1 * 0.020 / 0.012  # 0.5 * 20 / 12 = 10 / 12 = 0.8333333333333334
        expected = min(dam2, dmax)
        assert tensile_damage(epst, epst1, epsm1, dmax=dmax) == pytest.approx(expected)

    def test_monotonic_damage_accumulation(self):
        """Damage cannot decrease when strain drops (history variable dmg_old)."""
        epst1, epsm1 = 0.005, 0.025
        d1 = tensile_damage(0.015, epst1, epsm1, dmax=0.99)
        # Unload to below threshold
        d2 = tensile_damage(0.002, epst1, epsm1, dmax=0.99, dmg_old=d1)
        assert d2 == d1


# ============================================================================
# 7. Sound Speed Parity (read_mat25_tsaiwu.F90:288)
# ============================================================================

class TestSoundSpeedParity:
    """Verifies c = sqrt(max(C1, G12, G23, G31) / rho0) where C1 = max(E1, E2)/(1 - nu12*nu21)."""

    def test_c1_dominates(self):
        """When C1 > G_max, wave speed is governed by in-plane stiffness C1."""
        e11 = 140000.0
        e22 = 10000.0
        nu12 = 0.3
        rho0 = 1.5e-9
        nu21 = nu12 * e22 / e11
        detc = 1.0 - nu12 * nu21
        c1 = max(e11, e22) / detc
        expected_ssp = math.sqrt(c1 / rho0)

        calc_ssp = sound_speed_law25(e11, e22, nu12, 5000.0, 3000.0, 5000.0, rho0)
        assert calc_ssp == pytest.approx(expected_ssp, rel=1e-12)

    def test_shear_dominates(self):
        """When G12 > C1, wave speed is governed by shear modulus G12."""
        e11 = 1000.0
        e22 = 1000.0
        nu12 = 0.0
        g12 = 5000.0
        rho0 = 2.0e-9
        expected_ssp = math.sqrt(g12 / rho0)

        calc_ssp = sound_speed_law25(e11, e22, nu12, g12, 100.0, 100.0, rho0)
        assert calc_ssp == pytest.approx(expected_ssp, rel=1e-12)

    def test_e33_excluded_from_sound_speed(self):
        """Fortran read_mat25_tsaiwu.F90 line 288 ssp = sqrt(max(c1,gmax)/rho0).
        Verify that out-of-plane E33 does NOT inflate sound speed."""
        e11, e22 = 100000.0, 50000.0
        nu12 = 0.2
        g12, g23, g31 = 10000.0, 5000.0, 5000.0
        rho0 = 1.0e-9

        # Build material with huge E33 = 1e9
        mat = build_law25(
            E1=e11, E2=e22, E3=1.0e9, nu12=nu12,
            G12=g12, G23=g23, G31=g31, rho0=rho0,
        )
        c_sound = sound_speed(mat)

        nu21 = nu12 * e22 / e11
        c1 = e11 / (1.0 - nu12 * nu21)
        expected_c = math.sqrt(c1 / rho0)
        assert c_sound == pytest.approx(expected_c, rel=1e-10)
        # Verify it did not use 1e9
        assert c_sound < math.sqrt(1.0e9 / rho0)


# ============================================================================
# 8. Damage Array & State Variables Parity (hm_read_mat25.F & sigeps25c.F)
# ============================================================================

class TestDamageArrayParity:
    """Verifies DMG array layout and global failure index tracking."""

    def test_extra_shapes_dimensions(self):
        """Check persistent variable allocations match hm_read_mat25.F:121-136."""
        mat_tsaiwu = build_law25(iform=0)
        shapes_tw = extra_shapes(mat_tsaiwu, nip=4)
        # NMOD = 3 for Tsai-Wu -> NDMG = 1 + 3 = 4 (padded to >= 4)
        assert shapes_tw["dmg25"][1] >= 4
        assert shapes_tw["stra25"] == (4, 6)
        assert shapes_tw["crak25"] == (4, 2)

        mat_crasurv = build_law25(iform=1)
        shapes_cras = extra_shapes(mat_crasurv, nip=4)
        # NMOD = 6 for CRASURV -> NDMG = 1 + 6 = 7
        assert shapes_cras["dmg25"][1] >= 7

    def test_global_failure_index_tracking(self):
        """DMG[0] tracks max(DMG[1], DMG[2], DMG[3]) as in sigeps25c.F:288."""
        mat = build_law25(
            E1=100000.0, E2=50000.0, nu12=0.0, G12=20000.0,
            epst1=0.001, epsm1=0.005, dmax=0.8,
            wpmax=10.0,
        )
        extra: Dict[str, Any] = {
            "stra25": np.zeros((1, 3)),
            "dmg25": np.zeros((1, 8)),
            "wpla25": np.zeros(1),
            "off25": np.ones(1),
        }

        # Strain exceeding epst1 causes tensile damage in dir 1
        deps = np.array([0.003, 0.0, 0.0])
        law25_composite.shell_update(mat, np.zeros(3), deps, extra=extra)

        dmg = extra["dmg25"][0]
        # dmg[1] is dir 1 damage
        assert dmg[1] > 0.0
        # dmg[0] must be equal to max of active damage components
        assert dmg[0] == pytest.approx(max(dmg[1], dmg[2], dmg[3]))
