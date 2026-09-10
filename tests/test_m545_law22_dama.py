"""Tests for Milestone M545: LAW22 Kernel Builder (/MAT/LAW22, /MAT/DAMA, /MAT/PLAS_DAMA).

Verifies pyradioss/materials/law22_dama.py against exact OpenRadioss Fortran physics:
- starter/source/materials/mat/mat022/hm_read_mat22.F
- engine/source/materials/mat/mat022/sigeps22c.F
- engine/source/materials/mat/mat022/sigeps22g.F
- engine/source/materials/mat/mat022/m22cplr.F
- engine/source/materials/mat/mat022/m22law.F
- hm_cfg_files/config/CFG/radioss110/MAT/matl22_dama.cfg
"""

from __future__ import annotations

import math
from pathlib import Path
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
import pyradioss.materials as materials
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog


class TestBuildLaw22:
    """Test parameter parsing and derived constant computation."""

    def test_standard_parameters_and_derived_constants(self):
        mat = build_law22(
            E=210000.0,
            nu=0.3,
            a=300.0,
            b=450.0,
            n=0.4,
            eps_max=0.2,
            sig_max=700.0,
            c=0.02,
            eps_dot_0=0.001,
            ICC=2,
            eps_dam=0.02,
            E_tan=-2000.0,
            rho0=7.85e-9,
            refer_rho=7.85e-9,
        )
        assert mat.law == 22
        assert mat.rho0 == pytest.approx(7.85e-9)
        p = mat.params

        # Derived elastic constants
        expected_G = 210000.0 / (2.0 * 1.3)
        expected_K = 210000.0 / (3.0 * (1.0 - 2.0 * 0.3))
        expected_E1MN2 = 210000.0 / (1.0 - 0.3 ** 2)
        expected_EN1N2 = 0.3 * expected_E1MN2

        assert p["G"] == pytest.approx(expected_G)
        assert p["K"] == pytest.approx(expected_K)
        assert p["E1MN2"] == pytest.approx(expected_E1MN2)
        assert p["EN1N2"] == pytest.approx(expected_EN1N2)

        # Softening slope HL and YLDL
        expected_HL = 210000.0 * (-2000.0) / (210000.0 - (-2000.0))
        expected_YLDL = min(700.0, 300.0 + 450.0 * (0.02 ** 0.4))
        assert p["HL"] == pytest.approx(expected_HL)
        assert p["YLDL"] == pytest.approx(expected_YLDL)

        # Sound speeds
        expected_c_shell = math.sqrt(max(expected_E1MN2, expected_G) / 7.85e-9)
        expected_c_solid = math.sqrt((expected_K + 4.0 / 3.0 * expected_G) / 7.85e-9)
        assert p["c_shell"] == pytest.approx(expected_c_shell)
        assert p["c_solid"] == pytest.approx(expected_c_solid)

        # Sound speed functions
        assert sound_speed(mat) == pytest.approx(expected_c_solid)
        assert sound_speed_solid(mat) == pytest.approx(expected_c_solid)
        assert sound_speed_shell(mat) == pytest.approx(expected_c_shell)

    def test_defaults_and_fallbacks(self):
        mat = build_law22(E=100000.0, nu=0.25, rho0=2.5e-9)
        p = mat.params

        assert p["refer_rho"] == pytest.approx(2.5e-9)
        assert p["n"] == pytest.approx(1.0)
        assert math.isinf(p["eps_max"])
        assert math.isinf(p["sig_max"])
        assert p["c"] == pytest.approx(0.0)
        assert p["eps_dot_0"] == pytest.approx(1.0)
        assert p["ICC"] == 1
        assert p["eps_dam"] == pytest.approx(1e-15)
        assert p["E_tan"] == pytest.approx(0.0)
        assert p["HL"] == pytest.approx(0.0)

    def test_nu_clamp_and_validation(self):
        # nu = 0.5 clamped to 0.499
        mat = build_law22(E=100000.0, nu=0.5, rho0=1.0)
        assert mat.params["nu"] == pytest.approx(0.499)

        # Negative E raises ValueError
        with pytest.raises(ValueError, match="Young modulus E must be > 0"):
            build_law22(E=-100.0, nu=0.3)

        # nu outside [0, 0.5) raises ValueError
        with pytest.raises(ValueError, match="Poisson ratio nu"):
            build_law22(E=1000.0, nu=0.6)

    def test_extra_shapes(self):
        sh_shapes = extra_shapes(nip=5)
        assert sh_shapes["epsp22"] == (5,)
        assert sh_shapes["alpe22"] == (5,)
        assert sh_shapes["off22"] == (5,)

        sol_shapes = extra_shapes(nip=None)
        assert sol_shapes["epsp22"] == ()
        assert sol_shapes["alpe22"] == ()
        assert sol_shapes["off22"] == ()


class TestShellUpdate:
    """Test plane-stress shell kernel update (m22cplr.F)."""

    @pytest.fixture
    def mat22(self):
        return build_law22(
            E=200000.0,
            nu=0.3,
            a=250.0,
            b=500.0,
            n=0.5,
            eps_dam=0.01,
            E_tan=-4000.0,
            eps_max=0.08,
            sig_max=800.0,
            c=0.05,
            eps_dot_0=1.0,
            ICC=1,
            rho0=7.8e-9,
        )

    def test_shell_elastic_step(self, mat22):
        sig = np.zeros((1, 3))
        deps = np.array([[0.0005, 0.0, 0.0]])
        extra = {}
        s_new, ep_new, cs = shell_update(mat22, sig, deps, dt=1e-5, extra=extra)

        # Theoretical elastic stress: a1 * de11 = 200000 / (1 - 0.09) * 0.0005 ~ 109.89 MPa
        a1 = 200000.0 / (1.0 - 0.3 ** 2)
        assert s_new[0, 0] == pytest.approx(a1 * 0.0005)
        assert s_new[0, 1] == pytest.approx(0.3 * a1 * 0.0005)
        assert s_new[0, 2] == pytest.approx(0.0)
        assert ep_new[0] == pytest.approx(0.0)
        assert extra["alpe22"][0] == pytest.approx(1.0)
        assert extra["off22"][0] == pytest.approx(1.0)

    def test_shell_plastic_hardening(self, mat22):
        # Step large enough to yield without reaching eps_dam (eps_dam = 0.01)
        sig = np.zeros((1, 3))
        deps = np.array([[0.003, 0.0, 0.0]])
        extra = {}
        s_new, ep_new, _ = shell_update(mat22, sig, deps, dt=1e-5, extra=extra)

        assert ep_new[0] > 0.0
        assert ep_new[0] < 0.01
        # alpha remains 1.0 before eps_dam
        assert extra["alpe22"][0] == pytest.approx(1.0)
        # In-plane von Mises matches start-of-step yield scaled by rate factor (m22cplr.F explicit radial projection)
        svm = math.sqrt(s_new[0, 0] ** 2 + s_new[0, 1] ** 2 - s_new[0, 0] * s_new[0, 1] + 3.0 * s_new[0, 2] ** 2)
        epsp_rate = 0.003 / 1e-5
        rate_fac = 1.0 + 0.05 * math.log(epsp_rate / 1.0)
        expected_yld = 250.0 * rate_fac
        assert svm == pytest.approx(expected_yld, rel=1e-4)

    def test_shell_damage_softening(self, mat22):
        # Start at epsp beyond eps_dam
        sig = np.zeros((1, 3))
        deps = np.array([[0.002, 0.0, 0.0]])
        epsp_init = np.array([0.02])
        extra = {"epsp22": epsp_init, "off22": np.array([1.0])}

        s_new, ep_new, _ = shell_update(mat22, sig, deps, epsp=epsp_init, dt=1e-5, extra=extra)

        # Modulus degradation alpha must be < 1.0
        alpe = extra["alpe22"][0]
        assert alpe < 1.0
        assert alpe > 0.0

    def test_shell_element_failure(self, mat22):
        # Strain causing plastic strain to exceed eps_max = 0.08
        sig = np.zeros((1, 3))
        deps = np.array([[0.001, 0.0, 0.0]])
        epsp_init = np.array([0.085])
        extra = {"epsp22": epsp_init, "off22": np.array([1.0])}

        s_new, ep_new, _ = shell_update(mat22, sig, deps, epsp=epsp_init, dt=1e-5, extra=extra)

        # Failed element should have zero stress and off22 = 0.0
        assert s_new[0, 0] == pytest.approx(0.0)
        assert s_new[0, 1] == pytest.approx(0.0)
        assert s_new[0, 2] == pytest.approx(0.0)
        assert extra["off22"][0] == pytest.approx(0.0)

    def test_shell_transverse_shear_handling(self, mat22):
        # Stress and deps with 5 components [xx, yy, xy, yz, zx]
        sig = np.zeros((1, 5))
        deps = np.array([[0.0005, 0.0, 0.0, 0.001, 0.002]])
        s_new, _, _ = shell_update(mat22, sig, deps, dt=1e-5)
        G = mat22.params["G"]
        assert s_new[0, 3] == pytest.approx(G * 0.001)
        assert s_new[0, 4] == pytest.approx(G * 0.002)


class TestSolidUpdate:
    """Test 3D solid kernel update (m22law.F)."""

    @pytest.fixture
    def mat22(self):
        return build_law22(
            E=210000.0,
            nu=0.28,
            a=300.0,
            b=600.0,
            n=0.5,
            eps_dam=0.015,
            E_tan=-3000.0,
            eps_max=0.1,
            sig_max=900.0,
            c=0.03,
            eps_dot_0=1.0,
            ICC=2,
            rho0=7.8e-9,
        )

    def test_solid_elastic_step(self, mat22):
        sig = np.zeros((1, 6))
        deps = np.array([[0.0004, 0.0, 0.0, 0.0, 0.0, 0.0]])
        extra = {}
        s_new, ep_new, c_sol = solid_update(mat22, sig, deps, dt=1e-5, extra=extra)

        # 1D uniaxial strain: C_11 = K + 4/3*G
        C11 = mat22.params["K"] + 4.0 / 3.0 * mat22.params["G"]
        assert s_new[0, 0] == pytest.approx(C11 * 0.0004)
        assert ep_new[0] == pytest.approx(0.0)
        assert extra["alpe22"][0] == pytest.approx(1.0)
        assert extra["off22"][0] == pytest.approx(1.0)
        assert c_sol == pytest.approx(mat22.params["c_solid"])

    def test_solid_plastic_radial_return(self, mat22):
        sig = np.zeros((1, 6))
        deps = np.array([[0.008, 0.0, 0.0, 0.0, 0.0, 0.0]])
        extra = {}
        s_new, ep_new, _ = solid_update(mat22, sig, deps, dt=1e-5, extra=extra)

        assert ep_new[0] > 0.0
        # Deviatoric stress von Mises should match hardened yield
        p = (s_new[0, 0] + s_new[0, 1] + s_new[0, 2]) / 3.0
        s_dev = s_new[0].copy()
        s_dev[0:3] -= p
        j2 = 0.5 * np.sum(s_dev[0:3] ** 2) + np.sum(s_dev[3:] ** 2)
        svm = math.sqrt(3.0 * j2)

        # Deviatoric von Mises matches start-of-step yield scaled by rate factor (m22law.F explicit return)
        epd = 0.008 / 1e-5
        expected_yld = 300.0 * (1.0 + 0.03 * math.log(epd / 1.0))
        expected_yld = min(expected_yld, 900.0)

        assert svm == pytest.approx(expected_yld, rel=1e-4)

    def test_solid_element_failure(self, mat22):
        sig = np.zeros((1, 6))
        deps = np.array([[0.001, 0.0, 0.0, 0.0, 0.0, 0.0]])
        epsp_init = np.array([0.12])
        extra = {"epsp22": epsp_init, "off22": np.array([1.0])}

        s_new, ep_new, _ = solid_update(mat22, sig, deps, epsp=epsp_init, dt=1e-5, extra=extra)

        assert np.all(s_new == 0.0)
        assert extra["off22"][0] == pytest.approx(0.0)


class TestConsistentTangents:
    """Test algorithmic plane-stress and 3D solid tangents."""

    @pytest.fixture
    def mat22(self):
        return build_law22(
            E=200000.0,
            nu=0.3,
            a=200.0,
            b=300.0,
            n=0.5,
            eps_dam=0.01,
            E_tan=-2000.0,
            eps_max=0.05,
            rho0=7.8e-9,
        )

    def test_shell_tangent_elastic_and_plastic(self, mat22):
        sig = np.array([[100.0, 20.0, 0.0]])
        # Elastic point
        D_el = consistent_shell_tangent(mat22, sig, epsp=np.array([0.0]), epsp_incr=np.array([0.0]))
        Ce = shell_membrane_tangent(mat22)
        assert np.allclose(D_el[0], Ce)

        # Plastic point
        D_pl = consistent_shell_tangent(mat22, sig, epsp=np.array([0.005]), epsp_incr=np.array([0.001]))
        assert D_pl.shape == (1, 3, 3)
        assert not np.allclose(D_pl[0], Ce)

        # Failed element
        extra = {"off22": np.array([0.0])}
        D_failed = consistent_shell_tangent(mat22, sig, extra=extra)
        assert np.all(D_failed == 0.0)

    def test_solid_tangent_elastic_and_plastic(self, mat22):
        sig = np.array([[100.0, 50.0, 50.0, 10.0, 0.0, 0.0]])
        # Elastic point
        D_el = consistent_solid_tangent(mat22, sig, epsp=np.array([0.0]), epsp_incr=np.array([0.0]))
        Ce = solid_tangent(mat22)
        assert np.allclose(D_el[0], Ce)

        # Plastic point
        D_pl = consistent_solid_tangent(mat22, sig, epsp=np.array([0.005]), epsp_incr=np.array([0.001]))
        assert D_pl.shape == (1, 6, 6)
        assert not np.allclose(D_pl[0], Ce)

        # Failed element
        extra = {"off22": np.array([0.0])}
        D_failed = consistent_solid_tangent(mat22, sig, extra=extra)
        assert np.all(D_failed == 0.0)


class TestStarterDeckParsing:
    """Test /MAT/LAW22, /MAT/DAMA, and /MAT/PLAS_DAMA deck parsing."""

    def test_parse_mat_law22_deck(self, tmp_path: Path):
        deck = """/BEGIN
/MAT/LAW22/1001
DAMA test steel
7.85e-9 0.0
210000.0 0.3
300.0 450.0 0.5 0.1 700.0
0.02 0.001 2
0.015 -2500.0
/END
"""
        p = tmp_path / "TEST_0000.rad"
        p.write_text(deck, encoding="ascii")
        blocks = read_deck(str(p))
        log = MessageLog()
        model = Model()
        parse_starter_deck(blocks, model, log)

        assert 1001 in model.materials
        mat = model.materials[1001]
        assert mat.law == 22
        assert mat.rho0 == pytest.approx(7.85e-9)
        assert mat.E == pytest.approx(210000.0)
        assert mat.params["a"] == pytest.approx(300.0)
        assert mat.params["b"] == pytest.approx(450.0)
        assert mat.params["n"] == pytest.approx(0.5)
        assert mat.params["eps_dam"] == pytest.approx(0.015)
        assert mat.params["E_tan"] == pytest.approx(-2500.0)

    def test_parse_mat_plas_dama_alias(self, tmp_path: Path):
        deck = """/BEGIN
/MAT/PLAS_DAMA/2002
Plas Dama alias
7.8e-9
200000.0 0.29
250.0 400.0 0.45 0.15 650.0
0.0 1.0 1
0.01 -1500.0
/END
"""
        p = tmp_path / "TEST_0000.rad"
        p.write_text(deck, encoding="ascii")
        blocks = read_deck(str(p))
        log = MessageLog()
        model = Model()
        parse_starter_deck(blocks, model, log)

        assert 2002 in model.materials
        mat = model.materials[2002]
        assert mat.law == 22
        assert mat.params["a"] == pytest.approx(250.0)
        assert mat.params["E_tan"] == pytest.approx(-1500.0)


class TestMaterialsPackageDispatch:
    """Test materials.__init__ dispatch for LAW22."""

    def test_package_dispatch(self):
        mat = build_law22(
            E=210000.0, nu=0.3, a=300.0, b=500.0, n=0.5,
            eps_dam=0.01, E_tan=-2000.0, eps_max=0.05, rho0=7.8e-9
        )

        # sound_speed dispatch
        cs = materials.sound_speed(mat)
        assert cs == pytest.approx(mat.params["c_solid"])

        # extra_shapes dispatch
        sh_shapes = materials.extra_shapes(mat, nip=3)
        assert "epsp22" in sh_shapes
        assert "alpe22" in sh_shapes
        assert "off22" in sh_shapes

        # shell_update dispatch
        sig_sh = np.zeros((1, 3))
        deps_sh = np.array([[0.0005, 0.0, 0.0]])
        s_out, ep_out = materials.shell_update(mat, sig_sh, deps_sh)
        assert s_out.shape == (1, 3)

        # solid_update dispatch
        sig_sol = np.zeros((1, 6))
        deps_sol = np.array([[0.0005, 0.0, 0.0, 0.0, 0.0, 0.0]])
        s6_out, ep6_out, c_out = materials.solid_update(mat, sig_sol, deps_sol)
        assert s6_out.shape == (1, 6)
        assert c_out == pytest.approx(mat.params["c_solid"])

        # tangents dispatch
        D_sol = materials.solid_tangent(mat, sig_sol)
        assert D_sol.shape == (1, 6, 6)
        D_sh = materials.shell_layer_tangent(mat, sig_sh)
        assert D_sh.shape == (1, 3, 3)
