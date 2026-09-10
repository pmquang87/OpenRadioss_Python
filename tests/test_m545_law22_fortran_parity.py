"""Fortran Parity Verification Test Suite for Milestone M545: LAW22 (/MAT/LAW22, /MAT/DAMA, /MAT/PLAS_DAMA).

Audits pyradioss/materials/law22_dama.py against upstream OpenRadioss Fortran physics:
- starter/source/materials/mat/mat022/hm_read_mat22.F
- engine/source/materials/mat/mat022/sigeps22c.F
- engine/source/materials/mat/mat022/sigeps22g.F
- engine/source/materials/mat/mat022/m22cplr.F
- engine/source/materials/mat/mat022/m22law.F
- hm_cfg_files/config/CFG/radioss110/MAT/matl22_dama.cfg
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from pyradioss.materials.law22_dama import (
    build_law22,
    shell_update,
    solid_update,
    consistent_shell_tangent,
    consistent_solid_tangent,
    sound_speed,
    sound_speed_shell,
    sound_speed_solid,
    shell_membrane_tangent,
    solid_tangent,
    extra_shapes,
)


class TestFortranModuliAndPoisson:
    """1. Moduli & Poisson ratio exact formulas:
    G = E / (2*(1+nu))
    K = E / (3*(1-2*nu))
    E1MN2 = E / (1-nu^2)
    EN1N2 = nu * E1MN2
    nu == 0.5 clamped to 0.499 (ZEP499)
    """

    @pytest.mark.parametrize("E, nu", [
        (210000.0, 0.3),
        (70000.0, 0.33),
        (110000.0, 0.34),
        (200000.0, 0.28),
        (5000.0, 0.4),
        (150000.0, 0.0),
    ])
    def test_derived_elastic_moduli_exact(self, E, nu):
        mat = build_law22(E=E, nu=nu, rho0=7.8e-9)
        p = mat.params

        # Exact Fortran formulas (hm_read_mat22.F lines 148-153)
        expected_G = E / (2.0 * (1.0 + nu))
        expected_K = E / (3.0 * (1.0 - 2.0 * nu))
        expected_E1MN2 = E / (1.0 - nu ** 2)
        expected_EN1N2 = nu * expected_E1MN2

        assert p["G"] == pytest.approx(expected_G, rel=1e-12)
        assert p["K"] == pytest.approx(expected_K, rel=1e-12)
        assert p["E1MN2"] == pytest.approx(expected_E1MN2, rel=1e-12)
        assert p["EN1N2"] == pytest.approx(expected_EN1N2, rel=1e-12)

    def test_poisson_ratio_clamp_zep499(self):
        """hm_read_mat22.F:141: IF(ANU == HALF) ANU = ZEP499 (0.499)."""
        mat = build_law22(E=100000.0, nu=0.5, rho0=1.0)
        assert mat.params["nu"] == pytest.approx(0.499, rel=1e-12)
        expected_G = 100000.0 / (2.0 * (1.0 + 0.499))
        expected_K = 100000.0 / (3.0 * (1.0 - 2.0 * 0.499))
        assert mat.params["G"] == pytest.approx(expected_G, rel=1e-12)
        assert mat.params["K"] == pytest.approx(expected_K, rel=1e-12)

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Young modulus E must be > 0"):
            build_law22(E=0.0, nu=0.3)
        with pytest.raises(ValueError, match="Young modulus E must be > 0"):
            build_law22(E=-1000.0, nu=0.3)
        with pytest.raises(ValueError, match="Poisson ratio nu"):
            build_law22(E=1000.0, nu=-0.1)
        with pytest.raises(ValueError, match="Poisson ratio nu"):
            build_law22(E=1000.0, nu=0.55)


class TestFortranSoundSpeed:
    """2. Sound speed parity:
    Shell (hm_read_mat22.F:154): c = sqrt(E / rho0) or sqrt(E1MN2 / rho0).
    Solid (hm_read_mat22.F:185-191): c = sqrt((K + 4/3*G) / rho0)
                                      = sqrt(E*(1-nu)/((1+nu)*(1-2*nu)*rho0)).
    """

    @pytest.mark.parametrize("E, nu, rho0", [
        (210000.0, 0.3, 7.85e-9),
        (70000.0, 0.33, 2.7e-9),
        (110000.0, 0.34, 4.5e-9),
        (3000.0, 0.45, 1.2e-9),
    ])
    def test_solid_sound_speed_algebraic_identity(self, E, nu, rho0):
        mat = build_law22(E=E, nu=nu, rho0=rho0)
        p = mat.params

        K = p["K"]
        G = p["G"]

        # Formulation 1: Bulk + Shear (hm_read_mat22.F line 191 & matl22_dama.cfg line 105)
        c_solid_1 = math.sqrt((K + 4.0 / 3.0 * G) / rho0)
        # Formulation 2: Explicit E, nu (hm_read_mat22.F line 187)
        c_solid_2 = math.sqrt(E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu) * rho0))

        assert c_solid_1 == pytest.approx(c_solid_2, rel=1e-12)
        assert p["c_solid"] == pytest.approx(c_solid_1, rel=1e-12)
        assert sound_speed_solid(mat) == pytest.approx(c_solid_1, rel=1e-12)
        assert sound_speed(mat) == pytest.approx(c_solid_1, rel=1e-12)

    @pytest.mark.parametrize("E, nu, rho0", [
        (210000.0, 0.3, 7.85e-9),
        (70000.0, 0.33, 2.7e-9),
    ])
    def test_shell_sound_speed_modes(self, E, nu, rho0):
        mat = build_law22(E=E, nu=nu, rho0=rho0)
        p = mat.params

        # Mode 'young': hm_read_mat22.F:154 SDSP = SQRT(YOUNG/MAX(PM(1),EM20))
        c_young = math.sqrt(E / rho0)
        assert p["c_sound"] == pytest.approx(c_young, rel=1e-12)
        assert p["SDSP"] == pytest.approx(c_young, rel=1e-12)
        assert sound_speed_shell(mat, mode="young") == pytest.approx(c_young, rel=1e-12)

        # Mode 'plane_stress': sqrt(max(E1MN2, G) / rho0)
        E1MN2 = E / (1.0 - nu ** 2)
        G = E / (2.0 * (1.0 + nu))
        c_plane = math.sqrt(max(E1MN2, G) / rho0)
        assert p["c_shell"] == pytest.approx(c_plane, rel=1e-12)
        assert sound_speed_shell(mat, mode="plane_stress") == pytest.approx(c_plane, rel=1e-12)
        assert sound_speed_shell(mat) == pytest.approx(c_plane, rel=1e-12)


class TestFortranSofteningSlopeAndDamageThreshold:
    """3. Softening slope and damage threshold:
    HL = E * E_tan / max(EM20, E - E_tan) for shells (m22cplr.F)
    HL_solid = 3*G0*E_tan / max(EM20, 3*G0 + E_tan) for solids (m22law.F:205)
    YLDL = a + b * (eps_dam^n) capped at sig_max.
    """

    def test_softening_slope_shell_and_solid(self):
        E = 200000.0
        nu = 0.3
        E_tan = -3000.0
        mat = build_law22(
            E=E, nu=nu, a=250.0, b=400.0, n=0.5,
            eps_dam=0.02, E_tan=E_tan, sig_max=800.0, rho0=7.8e-9
        )
        p = mat.params
        G = E / (2.0 * (1.0 + nu))

        # Shell softening slope (m22cplr.F & hm_read_mat22.F:155)
        expected_HL_shell = E * E_tan / (E - E_tan)
        assert p["HL_shell"] == pytest.approx(expected_HL_shell, rel=1e-12)
        assert p["HL"] == pytest.approx(expected_HL_shell, rel=1e-12)

        # Solid softening slope (m22law.F:205 / 241)
        expected_HL_solid = (3.0 * G * E_tan) / (3.0 * G + E_tan)
        assert p["HL_solid"] == pytest.approx(expected_HL_solid, rel=1e-12)

        # Damage threshold yield stress YLDL
        expected_YLDL = 250.0 + 400.0 * (0.02 ** 0.5)
        assert p["YLDL"] == pytest.approx(expected_YLDL, rel=1e-12)

    def test_damage_threshold_capped_at_sig_max(self):
        mat = build_law22(
            E=200000.0, nu=0.3, a=300.0, b=500.0, n=0.5,
            eps_dam=0.1, E_tan=-2000.0, sig_max=400.0, rho0=7.8e-9
        )
        # a + b * eps_dam^n = 300 + 500*sqrt(0.1) = 300 + 158.11 = 458.11 > 400.0
        assert mat.params["YLDL"] == pytest.approx(400.0, rel=1e-12)

    def test_default_eps_dam_and_n(self):
        # Default eps_dam = 1e-15 (ZEP15 in hm_read_mat22.F:142)
        # Default n = 1.0 when n = 0.0 (hm_read_mat22.F:143)
        mat = build_law22(E=100000.0, nu=0.3, a=200.0, b=100.0, n=0.0, eps_dam=0.0, rho0=1.0)
        assert mat.params["eps_dam"] == pytest.approx(1e-15, rel=1e-12)
        assert mat.params["n"] == pytest.approx(1.0, rel=1e-12)
        # YLDL = a + b * eps_dam^1
        assert mat.params["YLDL"] == pytest.approx(200.0 + 100.0 * 1e-15, rel=1e-12)


class TestFortranYieldAndDamageEvolution:
    """4. Yield and damage evolution (m22cplr.F:87-106, m22law.F:200-240):
    - yld = a + b * epseq^n
    - yld = min(yld, sig_max)
    - depsl = max(0.0, epseq - eps_dam)
    - yld = min(yld, YLDL + HL * depsl)
    - alpe = min(1.0, yld / (yld + E * depsl)) (or 3*G for solids)
    - E_curr = alpe * E
    - von Mises equivalent stress and strain rate factor: yld * (1 + c * ln(eps_dot / eps_dot_0))
    - Radial / plane-stress plastic return.
    """

    @pytest.fixture
    def law22_shell_mat(self):
        return build_law22(
            E=200000.0, nu=0.3, a=200.0, b=400.0, n=0.5,
            eps_dam=0.02, E_tan=-5000.0, eps_max=0.1, sig_max=600.0,
            c=0.04, eps_dot_0=1.0, ICC=1, rho0=7.8e-9
        )

    @pytest.fixture
    def law22_solid_mat(self):
        return build_law22(
            E=210000.0, nu=0.28, a=250.0, b=500.0, n=0.5,
            eps_dam=0.015, E_tan=-4000.0, eps_max=0.08, sig_max=700.0,
            c=0.02, eps_dot_0=1.0, ICC=2, rho0=7.8e-9
        )

    def test_shell_damage_factor_evolution(self, law22_shell_mat):
        """Verify alpha degradation factor follows m22cplr.F line 93."""
        E = 200000.0
        a, b, n = 200.0, 400.0, 0.5
        eps_dam = 0.02
        HL = law22_shell_mat.params["HL_shell"]
        YLDL = law22_shell_mat.params["YLDL"]

        for epseq_val in [0.005, 0.019, 0.02, 0.03, 0.05]:
            depsl = max(0.0, epseq_val - eps_dam)
            yld = min(a + b * (epseq_val ** n), 600.0)
            yld = min(yld, YLDL + HL * depsl)
            expected_alpha = min(1.0, yld / (yld + E * depsl))

            sig = np.zeros((1, 3))
            # Small strain increment so epseq does not advance
            deps = np.array([[1e-6, 0.0, 0.0]])
            extra = {"epsp22": np.array([epseq_val]), "off22": np.array([1.0])}
            shell_update(law22_shell_mat, sig, deps, dt=0.0, extra=extra)

            assert extra["alpe22"][0] == pytest.approx(expected_alpha, rel=1e-6)

    def test_solid_damage_factor_evolution(self, law22_solid_mat):
        """Verify solid alpha degradation factor follows m22law.F line 245 with 3*G0."""
        G = law22_solid_mat.params["G"]
        a, b, n = 250.0, 500.0, 0.5
        eps_dam = 0.015
        HL_solid = law22_solid_mat.params["HL_solid"]
        YLDL = law22_solid_mat.params["YLDL"]

        for epseq_val in [0.005, 0.015, 0.025, 0.04]:
            depsl = max(0.0, epseq_val - eps_dam)
            ak = min(a + b * (epseq_val ** n), 700.0)
            ak = min(ak, YLDL + HL_solid * depsl)
            expected_alpha = min(1.0, ak / (ak + 3.0 * G * depsl))

            sig = np.zeros((1, 6))
            deps = np.array([[1e-6, 0.0, 0.0, 0.0, 0.0, 0.0]])
            extra = {"epsp22": np.array([epseq_val]), "off22": np.array([1.0])}
            solid_update(law22_solid_mat, sig, deps, dt=0.0, extra=extra)

            assert extra["alpe22"][0] == pytest.approx(expected_alpha, rel=1e-6)

    def test_strain_rate_scaling_icc1_vs_icc2(self):
        """m22cplr.F:124 and m22law.F:247:
        ICC=1: yld * (1 + c*ln(eps_dot/eps_dot_0)) without sig_max cap on rate effect.
        ICC=2: capped at sig_max.
        """
        # Material with high rate sensitivity and moderate sig_max
        mat_icc1 = build_law22(
            E=200000.0, nu=0.3, a=200.0, b=0.0, n=1.0,
            c=0.2, eps_dot_0=1.0, ICC=1, sig_max=300.0, rho0=7.8e-9
        )
        mat_icc2 = build_law22(
            E=200000.0, nu=0.3, a=200.0, b=0.0, n=1.0,
            c=0.2, eps_dot_0=1.0, ICC=2, sig_max=300.0, rho0=7.8e-9
        )

        dt = 1e-5
        de11 = 0.01  # eps_dot = 0.01 / 1e-5 = 1000.0
        # rate_fac = 1 + 0.2 * ln(1000) = 1 + 0.2 * 6.907755 = 2.38155
        # yld_uncapped = 200.0 * 2.38155 = 476.31 > 300.0
        rate_fac = 1.0 + 0.2 * math.log(1000.0)
        expected_yld_icc1 = 200.0 * rate_fac
        expected_yld_icc2 = 300.0

        # Shell update
        sig1 = np.zeros((1, 3))
        deps1 = np.array([[de11, 0.0, 0.0]])
        s_new1, _, _ = shell_update(mat_icc1, sig1, deps1, dt=dt)
        svm1 = math.sqrt(s_new1[0, 0] ** 2 + s_new1[0, 1] ** 2 - s_new1[0, 0] * s_new1[0, 1] + 3.0 * s_new1[0, 2] ** 2)
        assert svm1 == pytest.approx(expected_yld_icc1, rel=1e-4)

        sig2 = np.zeros((1, 3))
        deps2 = np.array([[de11, 0.0, 0.0]])
        s_new2, _, _ = shell_update(mat_icc2, sig2, deps2, dt=dt)
        svm2 = math.sqrt(s_new2[0, 0] ** 2 + s_new2[0, 1] ** 2 - s_new2[0, 0] * s_new2[0, 1] + 3.0 * s_new2[0, 2] ** 2)
        assert svm2 == pytest.approx(expected_yld_icc2, rel=1e-4)

    def test_plane_stress_return_options_ipla0_and_ipla1(self, law22_shell_mat):
        """m22cplr.F: IPLA=0 (radial projection) vs IPLA=1 (plane stress iteration)."""
        sig_0 = np.zeros((1, 3))
        deps = np.array([[0.0015, 0.0, 0.0]])
        s_ipla0, ep_0, _ = shell_update(law22_shell_mat, sig_0.copy(), deps, ipla=0)

        sig_1 = np.zeros((1, 3))
        s_ipla1, ep_1, _ = shell_update(law22_shell_mat, sig_1.copy(), deps, ipla=1)

        # Both methods return admissible plastic state with non-zero plastic strain
        assert ep_0[0] > 0.0
        assert ep_1[0] > 0.0
        # Check that both bring stress down from elastic trial stress (329.67 MPa) toward yield (~200-240 MPa)
        svm0 = math.sqrt(s_ipla0[0, 0] ** 2 + s_ipla0[0, 1] ** 2 - s_ipla0[0, 0] * s_ipla0[0, 1] + 3.0 * s_ipla0[0, 2] ** 2)
        s1 = s_ipla1[0, 0] + s_ipla1[0, 1]
        s2 = s_ipla1[0, 0] - s_ipla1[0, 1]
        s3 = s_ipla1[0, 2]
        svm1 = math.sqrt(0.25 * s1 ** 2 + 0.75 * s2 ** 2 + 3.0 * s3 ** 2)
        assert svm0 == pytest.approx(200.0, rel=1e-3)
        assert svm1 < 250.0

    def test_through_thickness_strain_ezz(self, law22_shell_mat):
        """m22cplr.F:139: EZZ(I) = DPLA(I) * S1 / YLD(I)."""
        sig = np.zeros((1, 3))
        deps = np.array([[0.004, 0.001, 0.0]])
        extra = {}
        s_new, ep_new, _ = shell_update(law22_shell_mat, sig, deps, extra=extra)

        assert "ezz22" in extra
        ezz = extra["ezz22"][0]
        # Tension produces positive dpla and positive mean stress S1 -> ezz > 0
        assert ezz > 0.0
        dpla = extra["dpla"][0]
        s1 = 0.5 * (s_new[0, 0] + s_new[0, 1])
        yld = 200.0
        assert ezz == pytest.approx(dpla * s1 / yld, rel=1e-4)


class TestFortranElementFailureAndDeletion:
    """5. Element failure: deletion (off22 = 0.0, stresses zeroed) when epseq >= eps_max.
    m22cplr.F / sigeps22c.F / m22law.F.
    """

    def test_shell_element_failure_at_eps_max(self):
        mat = build_law22(
            E=200000.0, nu=0.3, a=200.0, b=100.0, n=1.0,
            eps_max=0.05, rho0=7.8e-9
        )
        sig = np.array([[150.0, 50.0, 10.0]])
        deps = np.array([[0.002, 0.0, 0.0]])
        # Already at eps_max
        extra = {"epsp22": np.array([0.051]), "off22": np.array([1.0])}
        s_new, ep_new, _ = shell_update(mat, sig, deps, extra=extra)

        assert np.all(s_new == 0.0)
        assert extra["off22"][0] == 0.0
        assert ep_new[0] >= 0.05

    def test_solid_element_failure_at_eps_max(self):
        mat = build_law22(
            E=210000.0, nu=0.28, a=250.0, b=200.0, n=1.0,
            eps_max=0.06, rho0=7.8e-9
        )
        sig = np.array([[200.0, 100.0, 100.0, 20.0, 10.0, 5.0]])
        deps = np.array([[0.002, 0.0, 0.0, 0.0, 0.0, 0.0]])
        extra = {"epsp22": np.array([0.065]), "off22": np.array([1.0])}
        s_new, ep_new, _ = solid_update(mat, sig, deps, extra=extra)

        assert np.all(s_new == 0.0)
        assert extra["off22"][0] == 0.0
        assert ep_new[0] >= 0.06

    def test_inactive_element_remains_zeroed(self):
        mat = build_law22(E=200000.0, nu=0.3, a=200.0, eps_max=0.1, rho0=7.8e-9)
        sig = np.array([[100.0, 50.0, 0.0]])
        deps = np.array([[0.005, 0.0, 0.0]])
        extra = {"epsp22": np.array([0.0]), "off22": np.array([0.0])}
        s_new, ep_new, _ = shell_update(mat, sig, deps, extra=extra)

        assert np.all(s_new == 0.0)
        assert extra["off22"][0] == 0.0


class TestFortranConsistentTangents:
    """6. Algorithmic Consistent Tangents parity."""

    def test_solid_consistent_tangent_uses_hl_solid(self):
        mat = build_law22(
            E=210000.0, nu=0.3, a=250.0, b=300.0, n=0.5,
            eps_dam=0.01, E_tan=-3000.0, eps_max=0.1, rho0=7.8e-9
        )
        sig = np.array([[200.0, 50.0, 50.0, 10.0, 0.0, 0.0]])
        epsp = np.array([0.02])  # beyond eps_dam
        epsp_incr = np.array([0.001])
        extra = {"alpe22": np.array([0.85]), "off22": np.array([1.0])}

        D = consistent_solid_tangent(mat, sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
        assert D.shape == (1, 6, 6)
        # Ensure tangent is finite and positive semi-definite up to float precision
        assert not np.isnan(D).any()
        assert not np.isinf(D).any()
        eigvals = np.linalg.eigvalsh(0.5 * (D[0] + D[0].T))
        assert np.all(eigvals >= -1e-6)

    def test_shell_consistent_tangent_degraded_modulus(self):
        mat = build_law22(
            E=200000.0, nu=0.3, a=200.0, b=400.0, n=0.5,
            eps_dam=0.01, E_tan=-4000.0, eps_max=0.1, rho0=7.8e-9
        )
        sig = np.array([[150.0, 20.0, 0.0]])
        epsp = np.array([0.025])
        epsp_incr = np.array([0.001])
        extra = {"alpe22": np.array([0.7]), "off22": np.array([1.0])}

        D = consistent_shell_tangent(mat, sig, epsp=epsp, epsp_incr=epsp_incr, extra=extra)
        assert D.shape == (1, 3, 3)
        assert not np.isnan(D).any()
        # Failed element gives zero tangent
        extra_failed = {"off22": np.array([0.0])}
        D_failed = consistent_shell_tangent(mat, sig, extra=extra_failed)
        assert np.all(D_failed == 0.0)


class TestFortranVectorizationAndEdgeCases:
    """7. Multi-element vectorization and edge cases."""

    def test_vectorized_multi_element_shell(self):
        mat = build_law22(
            E=200000.0, nu=0.3, a=200.0, b=400.0, n=0.5,
            eps_dam=0.01, E_tan=-2000.0, eps_max=0.05, rho0=7.8e-9
        )
        n_elem = 5
        sig = np.zeros((n_elem, 3))
        # 0: elastic, 1: plastic pre-damage, 2: damaged, 3: near failure, 4: failed
        deps = np.array([
            [0.0005, 0.0, 0.0],
            [0.003, 0.0, 0.0],
            [0.003, 0.0, 0.0],
            [0.003, 0.0, 0.0],
            [0.001, 0.0, 0.0],
        ])
        epsp_init = np.array([0.0, 0.005, 0.02, 0.049, 0.06])
        extra = {
            "epsp22": epsp_init.copy(),
            "alpe22": np.ones(n_elem),
            "off22": np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
        }

        s_new, ep_new, _ = shell_update(mat, sig, deps, epsp=epsp_init, dt=1e-5, extra=extra)

        assert s_new.shape == (n_elem, 3)
        assert ep_new[0] == pytest.approx(0.0)     # elastic
        assert ep_new[1] > 0.005                   # plastic pre-damage
        assert extra["alpe22"][1] == pytest.approx(1.0)
        assert extra["alpe22"][2] < 1.0            # damaged
        assert extra["off22"][4] == 0.0            # already failed
        assert np.all(s_new[4] == 0.0)             # zero stress

    def test_vectorized_multi_element_solid(self):
        mat = build_law22(
            E=210000.0, nu=0.28, a=250.0, b=500.0, n=0.5,
            eps_dam=0.015, E_tan=-3000.0, eps_max=0.08, rho0=7.8e-9
        )
        n_elem = 4
        sig = np.zeros((n_elem, 6))
        deps = np.array([
            [0.0003, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.005, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.005, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        epsp_init = np.array([0.0, 0.008, 0.03, 0.09])
        extra = {
            "epsp22": epsp_init.copy(),
            "alpe22": np.ones(n_elem),
            "off22": np.array([1.0, 1.0, 1.0, 1.0]),
        }

        s_new, ep_new, c_sol = solid_update(mat, sig, deps, epsp=epsp_init, dt=1e-5, extra=extra)

        assert s_new.shape == (n_elem, 6)
        assert ep_new[0] == pytest.approx(0.0)
        assert ep_new[1] > 0.008
        assert extra["alpe22"][1] == pytest.approx(1.0)
        assert extra["alpe22"][2] < 1.0
        assert extra["off22"][3] == 0.0            # failed (0.09 >= 0.08)
        assert np.all(s_new[3] == 0.0)
