"""
Fortran Parity Verification Suite for Material Law 43 (/MAT/LAW43, /MAT/HILL_TAB).

Direct line-by-line parity against reference Fortran OpenRadioss files:
  - Starter: starter/source/materials/mat/mat043/hm_read_mat43.F
  - Engine shells: engine/source/materials/mat/mat043/sigeps43c.F
  - Engine solids / generalized shells: engine/source/materials/mat/mat043/sigeps43g.F
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials import law43_hill_tab
from pyradioss.materials.law43_hill_tab import (
    build_law43,
    shell_update,
    solid_update,
    sound_speed,
    extra_shapes,
    resolve,
    shell_membrane_tangent,
    consistent_shell_tangent,
    consistent_solid_tangent,
    _eval_yield_stress,
    _eval_young_modulus,
)


# ============================================================================
# 1. Starter Parity: build_law43 vs hm_read_mat43.F
# ============================================================================

class TestStarterParity:
    """Parity tests against starter/source/materials/mat/mat043/hm_read_mat43.F."""

    def test_hill_constants_defaults_von_mises(self):
        """hm_read_mat43.F lines 216-228: R00=1, R45=1, R90=1 -> von Mises (1, 1, 1, 3)."""
        mat = build_law43(id=43, E=210000.0, nu=0.3, R00=1.0, R45=1.0, R90=1.0)
        p = mat.params

        # Exact Fortran formulas:
        # R = (R0 + 2*R45 + R90) * 0.25 = (1 + 2 + 1) * 0.25 = 1.0
        # H = R / (1 + R) = 1.0 / 2.0 = 0.5
        # A01 = H * (1 + 1/R00) = 0.5 * 2 = 1.0
        # A02 = H * (1 + 1/R90) = 0.5 * 2 = 1.0
        # A03 = 2 * H = 1.0
        # A12 = (2*R45 + 1) * (A01 + A02 - A03) = 3 * (1 + 1 - 1) = 3.0
        assert p["R_bar"] == pytest.approx(1.0, rel=1e-15)
        assert p["H_hill"] == pytest.approx(0.5, rel=1e-15)
        assert p["A01"] == pytest.approx(1.0, rel=1e-15)
        assert p["A02"] == pytest.approx(1.0, rel=1e-15)
        assert p["A03"] == pytest.approx(1.0, rel=1e-15)
        assert p["A12"] == pytest.approx(3.0, rel=1e-15)

    def test_hill_constants_anisotropic_exact_analytical(self):
        """hm_read_mat43.F lines 216-228 with arbitrary Lankford parameters."""
        r00 = 1.75
        r45 = 1.25
        r90 = 2.10

        mat = build_law43(id=1, E=70000.0, nu=0.33, R00=r00, R45=r45, R90=r90, Iyield=0)
        p = mat.params

        # Analytical Fortran calculation
        r_fortran = (r00 + 2.0 * r45 + r90) * 0.25  # (1.75 + 2.5 + 2.10)*0.25 = 1.5875
        h_fortran = r_fortran / (1.0 + r_fortran)   # 1.5875 / 2.5875 = 0.6135265700483091
        a01_f = h_fortran * (1.0 + 1.0 / r00)       # 0.61352657 * (1 + 1/1.75)
        a02_f = h_fortran * (1.0 + 1.0 / r90)       # 0.61352657 * (1 + 1/2.10)
        a03_f = 2.0 * h_fortran
        a12_f = (2.0 * r45 + 1.0) * (a01_f + a02_f - a03_f)

        assert p["R_bar"] == pytest.approx(r_fortran, rel=1e-15)
        assert p["H_hill"] == pytest.approx(h_fortran, rel=1e-15)
        assert p["A01"] == pytest.approx(a01_f, rel=1e-15)
        assert p["A02"] == pytest.approx(a02_f, rel=1e-15)
        assert p["A03"] == pytest.approx(a03_f, rel=1e-15)
        assert p["A12"] == pytest.approx(a12_f, rel=1e-15)

    def test_hill_constants_iyield_normalization(self):
        """hm_read_mat43.F lines 223-228: If IR0 > 0: A02/=A01, A03/=A01, A12/=A01, A01=1."""
        r00 = 1.75
        r45 = 1.25
        r90 = 2.10

        mat = build_law43(id=1, E=70000.0, nu=0.33, R00=r00, R45=r45, R90=r90, Iyield=1)
        p = mat.params

        r_fortran = (r00 + 2.0 * r45 + r90) * 0.25
        h_fortran = r_fortran / (1.0 + r_fortran)
        a01_unnorm = h_fortran * (1.0 + 1.0 / r00)
        a02_unnorm = h_fortran * (1.0 + 1.0 / r90)
        a03_unnorm = 2.0 * h_fortran
        a12_unnorm = (2.0 * r45 + 1.0) * (a01_unnorm + a02_unnorm - a03_unnorm)

        assert p["A01"] == pytest.approx(1.0, rel=1e-15)
        assert p["A02"] == pytest.approx(a02_unnorm / a01_unnorm, rel=1e-15)
        assert p["A03"] == pytest.approx(a03_unnorm / a01_unnorm, rel=1e-15)
        assert p["A12"] == pytest.approx(a12_unnorm / a01_unnorm, rel=1e-15)

    def test_lankford_zero_defaults_to_one(self):
        """hm_read_mat43.F lines 166-168: IF(R0==ZERO) R0=ONE, etc."""
        mat = build_law43(id=1, E=210000.0, nu=0.3, R00=0.0, R45=0.0, R90=0.0)
        p = mat.params
        assert p["R00"] == 1.0
        assert p["R45"] == 1.0
        assert p["R90"] == 1.0
        assert p["A01"] == pytest.approx(1.0)
        assert p["A12"] == pytest.approx(3.0)

    def test_failure_strains_infinity_defaults(self):
        """hm_read_mat43.F lines 169-170: EPSR1=INFINITY, EPSR2=TWO*INFINITY."""
        mat = build_law43(id=1, E=210000.0, nu=0.3, EPSR1=0.0, EPSR2=0.0)
        p = mat.params
        assert p["EPSR1"] >= 1.0e30
        assert p["EPSR2"] >= 2.0e30

    def test_asrate_israte_starter_logic(self):
        """hm_read_mat43.F lines 177-188:
        IF ASRATE /= 0 -> ISRATE=1
        ELSE IF ISRATE /= 0 -> ASRATE = 10000
        ELSE ASRATE = 0
        """
        # Case 1: ASRATE given
        m1 = build_law43(id=1, E=1000.0, ASRATE=50.0, ISRATE=0)
        assert m1.params["ISRATE"] == 1
        assert m1.params["ASRATE"] == 50.0

        # Case 2: Only ISRATE given
        m2 = build_law43(id=2, E=1000.0, ASRATE=0.0, ISRATE=1)
        assert m2.params["ISRATE"] == 1
        assert m2.params["ASRATE"] == 10000.0

        # Case 3: Neither given
        m3 = build_law43(id=3, E=1000.0, ASRATE=0.0, ISRATE=0)
        assert m3.params["ISRATE"] == 0
        assert m3.params["ASRATE"] == 0.0

    def test_single_curve_duplication_and_rate_zero_prepend(self):
        """hm_read_mat43.F lines 191-207:
        If NRATE==1 -> NUM_FUNC=2, RATE(1)=0, RATE(2)=1, IFUNC(2)=IFUNC(1)
        If NRATE>1 and RATE(1) /= 0 -> prepend rate 0
        """
        # Single curve
        m1 = build_law43(id=1, E=1000.0, curves=[([0.0, 0.1], [300.0, 400.0], 10.0)])
        assert len(m1.params["rates"]) == 2
        assert m1.params["rates"][0] == 0.0
        assert m1.params["rates"][1] == 1.0

        # Multi curve with lowest rate > 0
        m2 = build_law43(id=2, E=1000.0, curves=[
            ([0.0, 0.1], [350.0, 450.0], 50.0),
            ([0.0, 0.1], [400.0, 500.0], 200.0),
        ])
        assert len(m2.params["rates"]) == 3
        assert m2.params["rates"][0] == 0.0
        assert m2.params["rates"][1] == 50.0
        assert m2.params["rates"][2] == 200.0

    def test_acoustic_sound_speeds_starter(self):
        """hm_read_mat43.F lines 211, 241-243:
        c_shell = sqrt(A1 / rho0)
        c_solid = sqrt((C1 + 4G/3) / rho0)
        """
        e0 = 206000.0
        nu = 0.3
        rho0 = 7.8e-9

        a1_f = e0 / (1.0 - nu ** 2)
        g_f = 0.5 * e0 / (1.0 + nu)
        c1_f = e0 / (3.0 * (1.0 - 2.0 * nu))
        c_shell_f = math.sqrt(a1_f / rho0)
        c_solid_f = math.sqrt((c1_f + (4.0 / 3.0) * g_f) / rho0)

        mat = build_law43(id=1, rho0=rho0, E=e0, nu=nu)
        assert mat.params["c_shell"] == pytest.approx(c_shell_f, rel=1e-15)
        assert mat.params["c_solid"] == pytest.approx(c_solid_f, rel=1e-15)
        assert sound_speed(mat) == pytest.approx(c_shell_f, rel=1e-15)
        assert sound_speed(mat, extra={"is_solid": True}) == pytest.approx(c_solid_f, rel=1e-15)


# ============================================================================
# 2. Engine Shell Parity: shell_update vs sigeps43c.F
# ============================================================================

class TestEngineShellParity:
    """Parity tests against engine/source/materials/mat/mat043/sigeps43c.F."""

    def test_trial_stress_elastic_exact(self):
        """sigeps43c.F lines 172-176:
        SIGNXX = SIGOXX - UVAR(2) + A1*DEPSXX + A2*DEPSYY
        SIGNYY = SIGOYY - UVAR(3) + A2*DEPSXX + A1*DEPSYY
        SIGNXY = SIGOXY - UVAR(4) + G*DEPSXY
        SIGNYZ = SIGOYZ + GS*DEPSYZ
        SIGNZX = SIGOZX + GS*DEPSZX
        """
        e0 = 210000.0
        nu = 0.3
        shf = 5.0 / 6.0
        a1 = e0 / (1.0 - nu ** 2)
        a2 = nu * a1
        g = 0.5 * e0 / (1.0 + nu)
        gs = g * shf

        mat = build_law43(id=1, E=e0, nu=nu, curves=[([0.0, 1.0], [1.0e6, 1.0e6], 0.0)])

        sig_old = np.array([100.0, 50.0, 20.0, 5.0, -5.0])
        alpha = np.array([30.0, 15.0, 10.0])  # uvar 0, 1, 2
        deps = np.array([1.0e-4, -3.0e-5, 2.0e-4, 1.0e-5, 2.0e-5])

        extra = {
            "uvar43": np.array([alpha[0], alpha[1], alpha[2], 0.0]),
            "shf": shf,
        }

        sig_out, pla_out, c_sound = shell_update(mat, sig_old.copy(), deps, dt=1.0e-6, extra=extra)

        expected_sxx = (sig_old[0] - alpha[0]) + a1 * deps[0] + a2 * deps[1] + alpha[0]
        expected_syy = (sig_old[1] - alpha[1]) + a2 * deps[0] + a1 * deps[1] + alpha[1]
        expected_sxy = (sig_old[2] - alpha[2]) + g * deps[2] + alpha[2]
        expected_syz = sig_old[3] + gs * deps[3]
        expected_szx = sig_old[4] + gs * deps[4]

        assert pla_out == 0.0
        assert sig_out[0] == pytest.approx(expected_sxx, rel=1e-12)
        assert sig_out[1] == pytest.approx(expected_syy, rel=1e-12)
        assert sig_out[2] == pytest.approx(expected_sxy, rel=1e-12)
        assert sig_out[3] == pytest.approx(expected_syz, rel=1e-12)
        assert sig_out[4] == pytest.approx(expected_szx, rel=1e-12)

    def test_dynamic_modulus_degradation_exponential(self):
        """sigeps43c.F lines 157-167:
        E(I) = E(I) - (E(I) - EINF) * (1 - EXP(-CE * PLA(I)))
        """
        e0 = 200000.0
        einf = 120000.0
        ce = 15.0
        nu = 0.28
        mat = build_law43(id=1, E=e0, nu=nu, CE=ce, Einf=einf)

        pla = np.array([0.035])
        e_deg, a1, a2, g, g3 = _eval_young_modulus(mat, pla)

        expected_e = e0 - (e0 - einf) * (1.0 - math.exp(-ce * pla[0]))
        expected_a1 = expected_e / (1.0 - nu ** 2)
        expected_a2 = nu * expected_a1
        expected_g = 0.5 * expected_e / (1.0 + nu)
        expected_g3 = 3.0 * expected_g

        assert e_deg[0] == pytest.approx(expected_e, rel=1e-14)
        assert a1[0] == pytest.approx(expected_a1, rel=1e-14)
        assert a2[0] == pytest.approx(expected_a2, rel=1e-14)
        assert g[0] == pytest.approx(expected_g, rel=1e-14)
        assert g3[0] == pytest.approx(expected_g3, rel=1e-14)

    def test_dynamic_modulus_degradation_curve(self):
        """sigeps43c.F lines 144-156:
        ESCALE = FINTER(..., PLA)
        E = ESCALE * E0
        """
        e0 = 200000.0
        mat = build_law43(
            id=1, E=e0, nu=0.3, IFUNCE=1,
            E_curve_x=[0.0, 0.05, 0.10],
            E_curve_y=[1.0, 0.85, 0.70]
        )
        pla = np.array([0.05])
        e_deg, _, _, _, _ = _eval_young_modulus(mat, pla)
        assert e_deg[0] == pytest.approx(0.85 * e0, rel=1e-14)

    def test_strain_rate_measure_and_linear_filter(self):
        """sigeps43c.F lines 183-189 and mulawc.F90 lines 691-695:
        EPSP = 0.5 * (|edxx + edyy| + sqrt((edxx - edyy)^2 + edxy^2))
        Filtered: asrate_f = min(1.0, Fcut * dt), edot = asrate_f * edot_inst + (1 - asrate_f) * edot_prev
        """
        mat_inst = build_law43(id=1, E=200000.0, nu=0.3, ISRATE=0, curves=[([0.0, 0.1], [300.0, 400.0], 0.0)])
        fcut = 500.0
        mat_filt = build_law43(id=2, E=200000.0, nu=0.3, ISRATE=1, ASRATE=fcut, curves=[([0.0, 0.1], [300.0, 400.0], 0.0)])

        dt = 2.0e-4
        deps = np.array([[2.0e-3, -1.0e-3, 1.5e-3]])
        edxx = deps[0, 0] / dt  # 10.0
        edyy = deps[0, 1] / dt  # -5.0
        edxy = deps[0, 2] / dt  # 7.5

        # Analytical Fortran instantaneous formula
        expected_edot_inst = 0.5 * (abs(edxx + edyy) + math.sqrt((edxx - edyy) ** 2 + edxy ** 2))

        # Filtered formula: alpha_f = min(1.0, 500 * 2e-4) = 0.1
        alpha_f = min(1.0, fcut * dt)
        assert alpha_f == pytest.approx(0.1)

        edot_prev = 2.0
        extra_filt = {"uvar43": np.array([[0.0, 0.0, 0.0, edot_prev]])}
        extra_inst = {"uvar43": np.array([[0.0, 0.0, 0.0, edot_prev]])}

        sig = np.zeros((1, 3))
        shell_update(mat_inst, sig.copy(), deps, dt=dt, extra=extra_inst)
        shell_update(mat_filt, sig.copy(), deps, dt=dt, extra=extra_filt)

        assert extra_inst["uvar43"][0, 3] == pytest.approx(expected_edot_inst, rel=1e-12)
        expected_edot_filt = alpha_f * expected_edot_inst + (1.0 - alpha_f) * edot_prev
        assert extra_filt["uvar43"][0, 3] == pytest.approx(expected_edot_filt, rel=1e-12)

    def test_tensile_failure_and_scaling(self):
        """sigeps43c.F lines 193-196 and 234-235:
        EPST = 0.5 * (exx + eyy + sqrt((exx - eyy)^2 + exy^2))
        FAIL = clamp((EPSR2 - EPST) / (EPSR2 - EPSR1), 0.0, 1.0)
        """
        epsr1 = 0.01
        epsr2 = 0.03
        mat = build_law43(
            id=1, E=200000.0, nu=0.3, EPSR1=epsr1, EPSR2=epsr2,
            curves=[([0.0, 0.1], [300.0, 300.0], 0.0)]
        )
        # Strain giving epst = 0.02 (exactly halfway between 0.01 and 0.03 -> FAIL = 0.5)
        extra = {"eps": np.array([[0.02, 0.0, 0.0]])}
        sig = np.zeros((1, 3))
        deps = np.array([[1.0e-3, 0.0, 0.0]])

        sig_out, pla_out, _ = shell_update(mat, sig, deps, dt=1.0e-5, extra=extra)

        # Hill equivalent stress should be scaled by FAIL = 0.5: 300 * 0.5 = 150
        svm = math.sqrt(sig_out[0, 0] ** 2 + sig_out[0, 1] ** 2 - sig_out[0, 0] * sig_out[0, 1] + 3.0 * sig_out[0, 2] ** 2)
        assert svm == pytest.approx(150.0, rel=1e-3)

    def test_multi_curve_bracket_linear_extrapolation(self):
        """sigeps43c.F line 233: FAC = (EPSP - RATE1) / (RATE2 - RATE1), no upper cap."""
        mat = build_law43(
            id=1, E=200000.0, nu=0.3,
            curves=[
                ([0.0, 0.1], [300.0, 400.0], 0.0),
                ([0.0, 0.1], [350.0, 450.0], 100.0),
            ]
        )
        # Test strain rate = 200.0 (beyond max rate 100.0)
        # Bracket is [0, 100], FAC = (200 - 0) / (100 - 0) = 2.0
        # Yield stress at epsp=0 is 300 + 2.0 * (350 - 300) = 400.0
        y, h, y0 = _eval_yield_stress(mat, np.array([0.0]), np.array([200.0]))
        assert y[0] == pytest.approx(400.0, rel=1e-12)
        assert y0[0] == pytest.approx(400.0, rel=1e-12)

    def test_hill_equivalent_stress_formula(self):
        """sigeps43c.F lines 251-255:
        SVM = sqrt(A01*sigxx^2 + A02*sigyy^2 - A03*sigxx*sigyy + A12*sigxy^2)
        """
        r00, r45, r90 = 1.5, 1.2, 1.8
        mat = build_law43(id=1, E=200000.0, nu=0.3, R00=r00, R45=r45, R90=r90)
        p = mat.params
        a01, a02, a03, a12 = p["A01"], p["A02"], p["A03"], p["A12"]

        sxx, syy, sxy = 250.0, -120.0, 75.0
        expected_svm = math.sqrt(a01 * sxx ** 2 + a02 * syy ** 2 - a03 * sxx * syy + a12 * sxy ** 2)

        # Run elastic step with very high yield
        mat_high = build_law43(
            id=1, E=200000.0, nu=0.3, R00=r00, R45=r45, R90=r90,
            curves=[([0.0, 0.1], [1.0e6, 1.0e6], 0.0)]
        )
        extra = {"seq": 0.0}
        shell_update(mat_high, np.array([[sxx, syy, sxy]]), np.zeros((1, 3)), dt=1.0e-5, extra=extra)
        assert extra["seq"] == pytest.approx(expected_svm, rel=1e-12)

    def test_plastic_thinning_dezz_parity(self):
        """sigeps43c.F lines 257-259 and 371-374:
        DEZZ_el = -(DEPSXX + DEPSYY) * NNU1
        DEZZ_pl = -NU5 * DPLA * S1 / YLD
        where NNU1 = nu / (1 - nu), NU5 = 1 - NNU1, S1 = A01*SXX + A02*SYY - 0.5*A03*(SXX + SYY)
        """
        nu = 0.3
        sigy0 = 300.0
        mat = build_law43(
            id=1, E=200000.0, nu=nu, R00=1.0, R45=1.0, R90=1.0,
            curves=[([0.0, 0.1], [sigy0, sigy0], 0.0)]
        )
        thk0 = 1.5
        extra = {"thk": np.array([thk0]), "thk0": thk0, "off": np.array([1.0])}

        # Apply plastic strain in xx: deps_xx = 0.005, deps_yy = 0.0
        deps = np.array([[0.005, 0.0, 0.0]])
        sig_out, pla_out, _ = shell_update(mat, np.zeros((1, 3)), deps, dt=1.0e-5, extra=extra)

        assert pla_out > 0.0
        # Shell should have thinned (thk < thk0)
        assert extra["thk"][0] < thk0
        # In incompressible plasticity (nu=0.5 -> nu5=0 -> Poisson thinning),
        # with nu=0.3, DEZZ = DEZZ_el + DEZZ_pl < 0
        nnu1 = nu / (1.0 - nu)
        nu5 = 1.0 - nnu1
        dezz_el = -deps[0, 0] * nnu1
        s1 = sig_out[0, 0] - 0.5 * sig_out[0, 0]  # for A01=1, A03=1, syy=0 -> S1 = 0.5*sxx
        dezz_pl = -nu5 * pla_out * s1 / sigy0
        expected_thk = thk0 + (dezz_el + dezz_pl) * thk0
        assert extra["thk"][0] == pytest.approx(expected_thk, rel=1e-3)

    def test_mixed_isotropic_kinematic_hardening_parity(self):
        """sigeps43c.F lines 241-242 and 375-382:
        YLD = (1 - FISOKIN)*Y(pla) + FISOKIN*Y0
        Backstress update: alpha += (2*P1 + P2) * DR0, etc.
        """
        sigy0 = 300.0
        h_mod = 10000.0
        fisokin = 0.4  # 40% kinematic, 60% isotropic
        mat = build_law43(
            id=1, E=200000.0, nu=0.3, FISOKIN=fisokin,
            curves=[([0.0, 0.1], [sigy0, sigy0 + h_mod * 0.1], 0.0)]
        )
        extra = {"uvar43": np.zeros((1, 4))}
        deps = np.array([[0.005, -0.3 * 0.005, 0.0]])
        sig_out, pla_out, _ = shell_update(mat, np.zeros((1, 3)), deps, dt=1.0e-5, extra=extra)

        assert pla_out > 0.0
        alpha_xx = extra["uvar43"][0, 0]
        alpha_yy = extra["uvar43"][0, 1]
        alpha_xy = extra["uvar43"][0, 2]

        assert alpha_xx > 0.0
        # In sigeps43c.F lines 375-382:
        # P1 = A01*sxx - 0.5*A03*syy, P2 = A02*syy - 0.5*A03*sxx
        # dalpha_xx = (2*P1 + P2) * dr0
        # dalpha_yy = (2*P2 + P1) * dr0
        sxx_eff = sig_out[0, 0] - alpha_xx
        syy_eff = sig_out[0, 1] - alpha_yy
        p1 = sxx_eff - 0.5 * syy_eff
        p2 = syy_eff - 0.5 * sxx_eff
        ratio_expected = (2.0 * p2 + p1) / (2.0 * p1 + p2)
        assert (alpha_yy / alpha_xx) == pytest.approx(ratio_expected, rel=1e-4)
        assert alpha_xy == pytest.approx(0.0, abs=1e-12)

    def test_element_deletion_epsmax_parity(self):
        """sigeps43c.F lines 386-388:
        IF(PLA > EPSMAX .AND. OFF == ONE) OFF = FOUR_OVER_5 (0.8)
        """
        epsmax = 0.02
        mat = build_law43(
            id=1, E=200000.0, nu=0.3, EPSMAX=epsmax,
            curves=[([0.0, 0.1], [300.0, 300.0], 0.0)]
        )
        extra = {
            "pla43": np.array([0.025]),  # > epsmax
            "off43": np.array([1.0]),
        }
        sig = np.array([[200.0, 0.0, 0.0]])
        deps = np.array([[1.0e-4, 0.0, 0.0]])

        shell_update(mat, sig, deps, dt=1.0e-5, extra=extra)
        assert extra["off43"][0] == pytest.approx(0.8, rel=1e-15)


# ============================================================================
# 3. Engine Solid / Generalized Shell Parity: solid_update vs sigeps43g.F
# ============================================================================

class TestEngineSolidParity:
    """Parity tests against engine/source/materials/mat/mat043/sigeps43g.F."""

    def test_generalized_shell_8comp_elastic_bending(self):
        """sigeps43g.F lines 169-174 and 221-226:
        C1 = thk0 / 12, AM1 = A1 * C1, AM2 = A2 * C1, GM = G * C1
        MOMNXX = MOMOXX + AM1*DEPBXX + AM2*DEPBYY
        """
        e0 = 210000.0
        nu = 0.3
        thk0 = 2.4
        a1 = e0 / (1.0 - nu ** 2)
        a2 = nu * a1
        g = 0.5 * e0 / (1.0 + nu)

        c1_factor = thk0 * (1.0 / 12.0)
        am1 = a1 * c1_factor
        am2 = a2 * c1_factor
        gm = g * c1_factor

        mat = build_law43(
            id=1, E=e0, nu=nu,
            curves=[([0.0, 1.0], [1.0e6, 1.0e6], 0.0)]
        )
        sig8 = np.zeros((1, 8))
        # Curvature increments: kxx = 1.0e-4, kyy = -5.0e-5, kxy = 2.0e-4
        deps8 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-4, -5.0e-5, 2.0e-4]])
        extra = {"thk0": np.array([thk0])}

        sig_out, pla_out, c_sound = solid_update(mat, sig8, deps8, dt=1.0e-5, extra=extra)

        assert pla_out == 0.0
        assert sig_out[0, 5] == pytest.approx(am1 * deps8[0, 5] + am2 * deps8[0, 6], rel=1e-12)
        assert sig_out[0, 6] == pytest.approx(am2 * deps8[0, 5] + am1 * deps8[0, 6], rel=1e-12)
        assert sig_out[0, 7] == pytest.approx(gm * deps8[0, 7], rel=1e-12)

    def test_generalized_shell_coupling_coefficients(self):
        """sigeps43g.F lines 297-302:
        GAMA = 1.5*(C1 + Y) / (1.5*C1 + Y) where C1 = pla * E
        CM = 16 * GAMA^2
        CNM = (4/sqrt(3)) * GAMA
        QTIER = (4/3) * GAMA^2
        """
        e0 = 200000.0
        yld = 300.0
        pla = 0.02
        c1_pla = pla * e0
        expected_gama = 1.5 * (c1_pla + yld) / (1.5 * c1_pla + yld)
        expected_cm = 16.0 * expected_gama ** 2
        expected_cnm = (4.0 / math.sqrt(3.0)) * expected_gama
        expected_qtier = (4.0 / 3.0) * expected_gama ** 2

        mat = build_law43(id=1, E=e0, nu=0.3, curves=[([0.0, 0.1], [yld, yld], 0.0)])

        # Run plastic step with moments
        sig8 = np.array([[200.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0]])
        deps8 = np.array([[0.003, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0]])
        extra = {"pla43": np.array([pla]), "thk0": np.array([2.0])}

        sig_out, pla_out, _ = solid_update(mat, sig8, deps8, dt=1.0e-5, extra=extra)
        assert pla_out > pla

    def test_3d_continuum_hydrostatic_invariance(self):
        """Pure hydrostatic pressure must produce zero plastic strain (J2 / Hill plane invariance)."""
        mat = build_law43(
            id=1, E=210000.0, nu=0.3,
            curves=[([0.0, 0.1], [300.0, 400.0], 0.0)]
        )
        sig6 = np.zeros((1, 6))
        # High hydrostatic pressure
        deps6 = np.array([[-0.01, -0.01, -0.01, 0.0, 0.0, 0.0]])
        sig_out, pla_out, _ = solid_update(mat, sig6, deps6, dt=1.0e-5)

        k_bulk = 210000.0 / (3.0 * (1.0 - 2.0 * 0.3))
        expected_p = -k_bulk * (-0.03)  # positive tension / negative comp
        assert pla_out == 0.0
        assert sig_out[0, 0] == pytest.approx(-k_bulk * 0.03, rel=1e-12)
        assert sig_out[0, 1] == pytest.approx(-k_bulk * 0.03, rel=1e-12)
        assert sig_out[0, 2] == pytest.approx(-k_bulk * 0.03, rel=1e-12)

    def test_3d_continuum_pure_shear_yield(self):
        """In pure shear, yield stress is sigy0 / sqrt(3) when isotropic (A12=3)."""
        sigy0 = 350.0
        mat = build_law43(
            id=1, E=210000.0, nu=0.3, R00=1.0, R45=1.0, R90=1.0,
            curves=[([0.0, 0.1], [sigy0, sigy0], 0.0)]
        )
        sig6 = np.zeros((1, 6))
        deps6 = np.array([[0.0, 0.0, 0.0, 0.02, 0.0, 0.0]])

        sig_out, pla_out, _ = solid_update(mat, sig6, deps6, dt=1.0e-5)
        assert pla_out > 0.0
        expected_tau = sigy0 / math.sqrt(3.0)
        assert sig_out[0, 3] == pytest.approx(expected_tau, rel=1e-2)


# ============================================================================
# 4. Tangent Operators & Numerical Differentiation
# ============================================================================

class TestTangentOperators:
    """Verification of consistent tangent operators."""

    def test_membrane_tangent_elastic(self):
        """Consistent membrane tangent matches analytical plane-stress tensor."""
        e0 = 200000.0
        nu = 0.25
        mat = build_law43(id=1, E=e0, nu=nu)
        c_mat = shell_membrane_tangent(mat)

        a1 = e0 / (1.0 - nu ** 2)
        a2 = nu * a1
        g = 0.5 * e0 / (1.0 + nu)

        expected = np.array([
            [a1, a2, 0.0],
            [a2, a1, 0.0],
            [0.0, 0.0, g],
        ])
        np.testing.assert_allclose(c_mat, expected, rtol=1e-14)

    def test_consistent_shell_tangent_fd_symmetry(self):
        """Finite-difference verified tangent has positive eigenvalues and matches FD."""
        mat = build_law43(
            id=1, E=200000.0, nu=0.3,
            curves=[([0.0, 0.1], [300.0, 500.0], 0.0)]
        )
        sig = np.array([150.0, -50.0, 30.0])
        deps = np.array([2.0e-3, -6.0e-4, 1.0e-3])

        tan = consistent_shell_tangent(mat, sig, deps=deps, dt=1.0e-5, symmetric=True)
        assert tan.shape == (3, 3)
        np.testing.assert_allclose(tan, tan.T, rtol=1e-12)
        eigenvalues = np.linalg.eigvalsh(tan)
        assert np.all(eigenvalues > 0.0)

    def test_consistent_solid_tangent_fd_symmetry(self):
        """Finite-difference solid tangent has positive eigenvalues and is symmetric."""
        mat = build_law43(
            id=1, E=200000.0, nu=0.3,
            curves=[([0.0, 0.1], [300.0, 500.0], 0.0)]
        )
        sig = np.zeros(6)
        deps = np.array([1.5e-3, -4.5e-4, -4.5e-4, 8.0e-4, 0.0, 0.0])

        tan = consistent_solid_tangent(mat, sig, deps=deps, dt=1.0e-5, symmetric=True)
        assert tan.shape == (6, 6)
        np.testing.assert_allclose(tan, tan.T, rtol=1e-12)
        eigenvalues = np.linalg.eigvalsh(tan)
        assert np.all(eigenvalues > 0.0)
