"""
Milestone M543: /MAT/LAW25 Input Layer, Starter Checks & DeckWriter Roundtrip Audit.

Comprehensive audit suite covering:
  1. Full roundtrip (Tsai-Wu iform=0 and CRASURV iform=1) preserving all parameters with precision.
  2. All keyword aliases: /MAT/LAW25, /MAT/COMP_PLAS, /MAT/COMPSH, /MAT/TSAI_WU, /MAT/CRASURV, /MAT/COMPOSITE_PLAS.
  3. All Starter check diagnostics:
     - rho <= 0
     - E1 <= 0 or E2 <= 0
     - nu12 * nu21 >= 1.0 (detc <= 0)
     - G12 <= 0 or G23 <= 0 or G31 <= 0
     - yield stresses <= 0 (sigyt1, sigyc1, sigyt2, sigyc2, sigt12, sigc12)
     - hardening exponent n > 1.0 (Tsai-Wu and CRASURV)
     - damage dmax < 0 or dmax > 1.0
     - epsm < epst (damage strain ordering in dir 1, dir 2, and CRASURV directional softening)
     - incompatible element types (trusses, beams, springs)
  4. DeckWriter method chaining, unit_id emission, and positional argument styles.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck, read_mat_law25, KEYWORD_PARSERS
from pyradioss.model.model import Model
from pyradioss.model.entities import Material, MatLaw25
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    check_mat_law25,
    check_materials,
    check_model,
)


def _parse_string(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "DECK_0000.rad"
    p.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


class TestRoundtripTsaiWuAllParameters:
    def test_tsai_wu_roundtrip_all_fields(self, tmp_path: Path):
        d = StarterDeck("AUDIT_TSAI_WU")
        d.title("TSAI_WU_FULL_ROUNDTRIP")

        ret = d.mat_law25(
            mat_id=101,
            title="Carbon_Epoxy_TsaiWu",
            rho=1.55e-9,
            rho_ref=1.55e-9,
            e11=145000.0,
            e22=9800.0,
            nu12=0.32,
            iform=0,
            e33=9800.0,
            g12=4800.0,
            g23=3200.0,
            g31=4800.0,
            eps_f1=0.025,
            eps_f2=0.015,
            eps_t1=0.006,
            eps_m1=0.022,
            eps_t2=0.004,
            eps_m2=0.012,
            dmax=0.92,
            wpmax=6.5,
            wpref=1.2,
            ioff=2,
            b=0.45,
            n=0.75,
            fmax=8.5,
            sig_1yt=1650.0,
            sig_2yt=48.0,
            sig_1yc=1250.0,
            sig_2yc=180.0,
            alpha=1.1,
            sig_12yc=85.0,
            sig_12yt=72.0,
            c=0.12,
            eps_rate_0=15.0,
            icc=1,
            gamma_ini=0.008,
            gamma_max=0.035,
            d3max=0.88,
            fsmooth=1,
            fcut=1200.0,
        )
        assert ret is d, "mat_law25 must return self for method chaining"

        rendered = d.render()
        assert "/MAT/LAW25/101" in rendered
        assert "Carbon_Epoxy_TsaiWu" in rendered

        model, log = _parse_string(tmp_path, rendered)
        assert len(log.errors) == 0, f"Parse errors: {log.errors}"
        assert 101 in model.mat_law25s

        mat = model.mat_law25s[101]
        assert mat.id == 101
        assert mat.title == "Carbon_Epoxy_TsaiWu"
        assert mat.rho0 == pytest.approx(1.55e-9)
        assert mat.rhor == pytest.approx(1.55e-9)
        assert mat.e11 == pytest.approx(145000.0)
        assert mat.e22 == pytest.approx(9800.0)
        assert mat.nu12 == pytest.approx(0.32)
        assert mat.iform == 0
        assert mat.e33 == pytest.approx(9800.0)
        assert mat.g12 == pytest.approx(4800.0)
        assert mat.g23 == pytest.approx(3200.0)
        assert mat.g31 == pytest.approx(4800.0)
        assert mat.eps_f1 == pytest.approx(0.025)
        assert mat.eps_f2 == pytest.approx(0.015)
        assert mat.eps_t1 == pytest.approx(0.006)
        assert mat.eps_m1 == pytest.approx(0.022)
        assert mat.eps_t2 == pytest.approx(0.004)
        assert mat.eps_m2 == pytest.approx(0.012)
        assert mat.dmax == pytest.approx(0.92)
        assert mat.wpmax == pytest.approx(6.5)
        assert mat.wpref == pytest.approx(1.2)
        assert mat.ioff == 2
        assert mat.b == pytest.approx(0.45)
        assert mat.n == pytest.approx(0.75)
        assert mat.fmax == pytest.approx(8.5)
        assert mat.sig_1yt == pytest.approx(1650.0)
        assert mat.sig_2yt == pytest.approx(48.0)
        assert mat.sig_1yc == pytest.approx(1250.0)
        assert mat.sig_2yc == pytest.approx(180.0)
        assert mat.alpha == pytest.approx(1.1)
        assert mat.sig_12yc == pytest.approx(85.0)
        assert mat.sig_12yt == pytest.approx(72.0)
        assert mat.c == pytest.approx(0.12)
        assert mat.eps_rate_0 == pytest.approx(15.0)
        assert mat.icc == 1
        assert mat.gamma_ini == pytest.approx(0.008)
        assert mat.gamma_max == pytest.approx(0.035)
        assert mat.d3max == pytest.approx(0.88)
        assert mat.fsmooth == 1
        assert mat.fcut == pytest.approx(1200.0)

        assert 101 in model.materials
        assert model.materials[101].law == 25


class TestRoundtripCrasurvAllParameters:
    def test_crasurv_roundtrip_all_fields(self, tmp_path: Path):
        d = StarterDeck("AUDIT_CRASURV")
        d.title("CRASURV_FULL_ROUNDTRIP")

        ret = d.mat_crasurv(
            mat_id=202,
            title="Crash_Composite_CRASURV",
            rho=1.48e-9,
            rho_ref=1.48e-9,
            e11=130000.0,
            e22=35000.0,
            nu12=0.28,
            e33=18000.0,
            g12=9200.0,
            g23=4600.0,
            g31=7500.0,
            eps_f1=0.032,
            eps_f2=0.018,
            eps_t1=0.007,
            eps_m1=0.026,
            eps_t2=0.005,
            eps_m2=0.016,
            dmax=0.85,
            wpmax=12.0,
            wpref=1.5,
            ioff=1,
            iflawp=1,
            c=0.08,
            eps_rate_0=2.5,
            alpha=1.05,
            icc=1,
            # Tension dir 1
            sig_1yt=1100.0,
            b_1t=0.55,
            n_1t=0.82,
            sig_1maxt=1550.0,
            c_1t=0.025,
            eps_1t1=0.012,
            eps_2t1=0.032,
            sig_rst1=220.0,
            wpmax_t1=55.0,
            # Tension dir 2
            sig_2yt=110.0,
            b_2t=0.25,
            n_2t=0.88,
            sig_2maxt=160.0,
            c_2t=0.015,
            eps_1t2=0.009,
            eps_2t2=0.022,
            sig_rst2=35.0,
            wpmax_t2=22.0,
            # Compression dir 1
            sig_1yc=850.0,
            b_1c=0.42,
            n_1c=0.86,
            sig_1maxc=1280.0,
            c_1c=0.022,
            eps_1c1=0.014,
            eps_2c1=0.038,
            sig_rsc1=260.0,
            wpmax_c1=65.0,
            # Compression dir 2
            sig_2yc=220.0,
            b_2c=0.32,
            n_2c=0.92,
            sig_2maxc=270.0,
            c_2c=0.012,
            eps_1c2=0.011,
            eps_2c2=0.028,
            sig_rsc2=55.0,
            wpmax_c2=32.0,
            # Shear dir 12
            sig_12yt=85.0,
            b_12t=0.35,
            n_12t=0.84,
            sig_12maxt=130.0,
            c_12t=0.012,
            eps_1t12=0.016,
            eps_2t12=0.042,
            sig_rst12=28.0,
            wpmax_t12=42.0,
            # Delamination and filtering
            gamma_ini=0.012,
            gamma_max=0.045,
            d3max=0.90,
            fsmooth=1,
            fcut=1500.0,
        )
        assert ret is d, "mat_crasurv must return self for chaining"

        rendered = d.render()
        assert "/MAT/CRASURV/202" in rendered
        assert "Crash_Composite_CRASURV" in rendered

        model, log = _parse_string(tmp_path, rendered)
        assert len(log.errors) == 0, f"Parse errors: {log.errors}"
        assert 202 in model.mat_law25s

        mat = model.mat_law25s[202]
        assert mat.id == 202
        assert mat.title == "Crash_Composite_CRASURV"
        assert mat.iform == 1
        assert mat.rho0 == pytest.approx(1.48e-9)
        assert mat.e11 == pytest.approx(130000.0)
        assert mat.e22 == pytest.approx(35000.0)
        assert mat.nu12 == pytest.approx(0.28)
        assert mat.e33 == pytest.approx(18000.0)
        assert mat.g12 == pytest.approx(9200.0)
        assert mat.g23 == pytest.approx(4600.0)
        assert mat.g31 == pytest.approx(7500.0)
        assert mat.eps_f1 == pytest.approx(0.032)
        assert mat.eps_f2 == pytest.approx(0.018)
        assert mat.eps_t1 == pytest.approx(0.007)
        assert mat.eps_m1 == pytest.approx(0.026)
        assert mat.eps_t2 == pytest.approx(0.005)
        assert mat.eps_m2 == pytest.approx(0.016)
        assert mat.dmax == pytest.approx(0.85)
        assert mat.wpmax == pytest.approx(12.0)
        assert mat.wpref == pytest.approx(1.5)
        assert mat.ioff == 1
        assert mat.iflawp == 1
        assert mat.c == pytest.approx(0.08)
        assert mat.eps_rate_0 == pytest.approx(2.5)
        assert mat.alpha == pytest.approx(1.05)
        assert mat.icc == 1

        # Dir 1 tension
        assert mat.sig_1yt == pytest.approx(1100.0)
        assert mat.b_1t == pytest.approx(0.55)
        assert mat.n_1t == pytest.approx(0.82)
        assert mat.sig_1maxt == pytest.approx(1550.0)
        assert mat.c_1t == pytest.approx(0.025)
        assert mat.eps_1t1 == pytest.approx(0.012)
        assert mat.eps_2t1 == pytest.approx(0.032)
        assert mat.sig_rst1 == pytest.approx(220.0)
        assert mat.wpmax_t1 == pytest.approx(55.0)

        # Dir 2 tension
        assert mat.sig_2yt == pytest.approx(110.0)
        assert mat.b_2t == pytest.approx(0.25)
        assert mat.n_2t == pytest.approx(0.88)
        assert mat.sig_2maxt == pytest.approx(160.0)
        assert mat.c_2t == pytest.approx(0.015)
        assert mat.eps_1t2 == pytest.approx(0.009)
        assert mat.eps_2t2 == pytest.approx(0.022)
        assert mat.sig_rst2 == pytest.approx(35.0)
        assert mat.wpmax_t2 == pytest.approx(22.0)

        # Dir 1 compression
        assert mat.sig_1yc == pytest.approx(850.0)
        assert mat.b_1c == pytest.approx(0.42)
        assert mat.n_1c == pytest.approx(0.86)
        assert mat.sig_1maxc == pytest.approx(1280.0)
        assert mat.c_1c == pytest.approx(0.022)
        assert mat.eps_1c1 == pytest.approx(0.014)
        assert mat.eps_2c1 == pytest.approx(0.038)
        assert mat.sig_rsc1 == pytest.approx(260.0)
        assert mat.wpmax_c1 == pytest.approx(65.0)

        # Dir 2 compression
        assert mat.sig_2yc == pytest.approx(220.0)
        assert mat.b_2c == pytest.approx(0.32)
        assert mat.n_2c == pytest.approx(0.92)
        assert mat.sig_2maxc == pytest.approx(270.0)
        assert mat.c_2c == pytest.approx(0.012)
        assert mat.eps_1c2 == pytest.approx(0.011)
        assert mat.eps_2c2 == pytest.approx(0.028)
        assert mat.sig_rsc2 == pytest.approx(55.0)
        assert mat.wpmax_c2 == pytest.approx(32.0)

        # Dir 12 shear
        assert mat.sig_12yt == pytest.approx(85.0)
        assert mat.b_12t == pytest.approx(0.35)
        assert mat.n_12t == pytest.approx(0.84)
        assert mat.sig_12maxt == pytest.approx(130.0)
        assert mat.c_12t == pytest.approx(0.012)
        assert mat.eps_1t12 == pytest.approx(0.016)
        assert mat.eps_2t12 == pytest.approx(0.042)
        assert mat.sig_rst12 == pytest.approx(28.0)
        assert mat.wpmax_t12 == pytest.approx(42.0)

        # Delamination & filter
        assert mat.gamma_ini == pytest.approx(0.012)
        assert mat.gamma_max == pytest.approx(0.045)
        assert mat.d3max == pytest.approx(0.90)
        assert mat.fsmooth == 1
        assert mat.fcut == pytest.approx(1500.0)


class TestAllKeywordAliases:
    @pytest.mark.parametrize("alias, method_name", [
        ("LAW25", "mat_law25"),
        ("COMP_PLAS", "mat_comp_plas"),
        ("COMPSH", "mat_compsh"),
        ("TSAI_WU", "mat_tsai_wu"),
        ("CRASURV", "mat_crasurv"),
        ("COMPOSITE_PLAS", "mat_composite_plas"),
    ])
    def test_deck_writer_alias_methods(self, alias: str, method_name: str, tmp_path: Path):
        d = StarterDeck("ALIAS_TEST")
        method = getattr(d, method_name)
        ret = method(
            mat_id=50,
            title=f"Test_{alias}",
            rho=1.5e-9,
            rho_ref=1.5e-9,
            e11=100000.0,
            e22=50000.0,
            nu12=0.25,
            g12=10000.0,
            g23=5000.0,
            g31=8000.0,
            sig_1yt=1000.0,
            sig_2yt=100.0,
            sig_1yc=800.0,
            sig_2yc=200.0,
            sig_12yt=60.0,
            sig_12yc=60.0,
        )
        assert ret is d, f"{method_name} must return self"

        rendered = d.render()
        assert f"/MAT/{alias}/50" in rendered

        model, log = _parse_string(tmp_path, rendered)
        assert len(log.errors) == 0, f"Errors parsing {alias}: {log.errors}"
        assert 50 in model.mat_law25s
        mat = model.mat_law25s[50]
        assert mat.e11 == pytest.approx(100000.0)
        assert mat.sig_1yt == pytest.approx(1000.0)

    def test_model_alias_dictionaries(self):
        m = Model()
        mat = MatLaw25(id=99, e11=120000.0)
        m.mat_law25s[99] = mat

        assert m.mat_comp_plas[99] is mat
        assert m.mat_compshs[99] is mat
        assert m.mat_tsai_wus[99] is mat
        assert m.mat_crasurvs[99] is mat
        assert m.mat_composite_plas[99] is mat
        assert m.mat_composite_plass[99] is mat


class TestStarterCheckDiagnostics:
    def _baseline_params(self) -> dict:
        return {
            "rho0": 1.5e-9,
            "e11": 100000.0,
            "e22": 50000.0,
            "nu12": 0.25,
            "g12": 10000.0,
            "g23": 5000.0,
            "g31": 8000.0,
            "sigyt1": 1000.0,
            "sigyc1": 800.0,
            "sigyt2": 100.0,
            "sigyc2": 200.0,
            "sigt12": 60.0,
            "sigc12": 60.0,
            "n": 0.8,
            "dmax": 0.9,
            "epst1": 0.005,
            "epsm1": 0.02,
            "epst2": 0.003,
            "epsm2": 0.015,
        }

    def test_valid_passes_checks(self):
        mat = Material(id=1, law=25, rho0=1.5e-9, params=self._baseline_params())
        log = MessageLog()
        check_mat_law25(mat, log)
        assert len(log.errors) == 0

    @pytest.mark.parametrize("rho_val", [0.0, -1.5e-9])
    def test_diagnostic_rho_le_zero(self, rho_val: float):
        p = self._baseline_params()
        p["rho0"] = rho_val
        mat = Material(id=1, law=25, rho0=rho_val, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any("initial density RHO must be > 0" in e for e in log.errors)

    @pytest.mark.parametrize("param_key, err_text", [
        ("e11", "Young's modulus E1 must be > 0"),
        ("e22", "Young's modulus E2 must be > 0"),
    ])
    @pytest.mark.parametrize("bad_val", [0.0, -100.0])
    def test_diagnostic_youngs_moduli_le_zero(self, param_key: str, err_text: str, bad_val: float):
        p = self._baseline_params()
        p[param_key] = bad_val
        mat = Material(id=1, law=25, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any(err_text in e for e in log.errors)

    def test_diagnostic_detc_le_zero(self):
        p = self._baseline_params()
        p["nu12"] = 2.0
        mat = Material(id=1, law=25, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any("detc = 1 - nu12*nu21 must be > 0" in e for e in log.errors)

    @pytest.mark.parametrize("g_key, err_name", [
        ("g12", "G12"),
        ("g23", "G23"),
        ("g31", "G31"),
    ])
    @pytest.mark.parametrize("bad_val", [0.0, -500.0])
    def test_diagnostic_shear_moduli_le_zero(self, g_key: str, err_name: str, bad_val: float):
        p = self._baseline_params()
        p[g_key] = bad_val
        mat = Material(id=1, law=25, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any(f"shear modulus {err_name} must be > 0" in e for e in log.errors)

    @pytest.mark.parametrize("stress_key, err_str", [
        ("sigyt1", "tensile yield stress in dir 1 (sigyt1) must be > 0"),
        ("sigyc1", "compressive yield stress in dir 1 (sigyc1) must be > 0"),
        ("sigyt2", "tensile yield stress in dir 2 (sigyt2) must be > 0"),
        ("sigyc2", "compressive yield stress in dir 2 (sigyc2) must be > 0"),
        ("sigt12", "tensile shear yield stress in dir 12 (sigt12) must be > 0"),
        ("sigc12", "compressive shear yield stress in dir 12 (sigc12) must be > 0"),
    ])
    @pytest.mark.parametrize("bad_val", [0.0, -250.0])
    def test_diagnostic_yield_stresses_le_zero(self, stress_key: str, err_str: str, bad_val: float):
        p = self._baseline_params()
        p[stress_key] = bad_val
        mat = Material(id=1, law=25, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any(err_str in e for e in log.errors)

    def test_diagnostic_hardening_n_gt_one_tsaiwu(self):
        p = self._baseline_params()
        p["n"] = 1.25
        mat = Material(id=1, law=25, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any("hardening exponent n must be <= 1.0" in e for e in log.errors)

    @pytest.mark.parametrize("n_key", ["n1_t", "n1_c", "n2_t", "n2_c", "n12_t"])
    def test_diagnostic_hardening_n_gt_one_crasurv(self, n_key: str):
        p = self._baseline_params()
        p["iform"] = 1
        p[n_key] = 1.3
        mat = Material(id=1, law=25, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any("hardening exponent n must be <= 1.0" in e for e in log.errors)

    @pytest.mark.parametrize("bad_dmax", [-0.1, 1.05, 2.0])
    def test_diagnostic_damage_dmax_bounds(self, bad_dmax: float):
        p = self._baseline_params()
        p["dmax"] = bad_dmax
        mat = Material(id=1, law=25, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any("maximum damage dmax must be in [0, 1]" in e for e in log.errors)

    def test_diagnostic_epsm_lt_epst(self):
        p1 = self._baseline_params()
        p1["epst1"] = 0.02
        p1["epsm1"] = 0.005
        mat1 = Material(id=1, law=25, rho0=1.5e-9, params=p1)
        log1 = MessageLog()
        check_mat_law25(mat1, log1)
        assert any("maximum damage strain eps_m1 must be >= damage initiation strain eps_t1" in e for e in log1.errors)

        p2 = self._baseline_params()
        p2["epst2"] = 0.015
        p2["epsm2"] = 0.003
        mat2 = Material(id=2, law=25, rho0=1.5e-9, params=p2)
        log2 = MessageLog()
        check_mat_law25(mat2, log2)
        assert any("maximum damage strain eps_m2 must be >= damage initiation strain eps_t2" in e for e in log2.errors)

    def test_diagnostic_crasurv_eps2_lt_eps1(self):
        p = self._baseline_params()
        p["iform"] = 1
        p["eps_1t1"] = 0.03
        p["eps_2t1"] = 0.01
        mat = Material(id=1, law=25, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law25(mat, log)
        assert any("CRASURV maximum damage strain eps_2t1 must be >= eps_1t1" in e for e in log.errors)

    @pytest.mark.parametrize("incompatible_elem", ["trusses", "beams", "springs"])
    def test_diagnostic_incompatible_element_types(self, incompatible_elem: str):
        model = Model()
        model.add_nodes(np.array([1, 2]), np.zeros((2, 3)))
        mat = Material(id=1, law=25, rho0=1.5e-9, params=self._baseline_params())
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [(incompatible_elem, FakeGroup())]
        log = MessageLog()
        check_model(model, log)
        assert any(f"is not supported for {incompatible_elem} elements" in e for e in log.errors)

    @pytest.mark.parametrize("compatible_elem", ["shells", "sh3n", "quads", "bricks", "tetras", "penta6", "pyra5"])
    def test_compatible_element_types_accepted(self, compatible_elem: str):
        model = Model()
        model.add_nodes(np.array([1, 2, 3, 4]), np.zeros((4, 3)))
        mat = Material(id=1, law=25, rho0=1.5e-9, params=self._baseline_params())
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [(compatible_elem, FakeGroup())]
        log = MessageLog()
        check_model(model, log)
        assert not any("is not supported for" in e for e in log.errors)


class TestEdgeCasesAndPrecision:
    def test_unit_id_in_header(self, tmp_path: Path):
        d = StarterDeck("UNIT_ID_TEST")
        d.mat_law25(mat_id=1, unit_id=3, rho=1.5e-9, e11=1e5, e22=1e5, g12=1e4, sigyt1=100)
        rendered = d.render()
        assert "/MAT/LAW25/1/3" in rendered

        model, log = _parse_string(tmp_path, rendered)
        assert len(log.errors) == 0
        assert 1 in model.mat_law25s
        assert any(ref[0] == "MAT" and ref[1] == 1 and ref[2] == 3 for ref in model.raw_unit_refs)

    def test_positional_rho_without_title(self, tmp_path: Path):
        d = StarterDeck("POS_TEST")
        d.mat_law25(10, 1.55e-9, e11=140000.0, e22=10000.0, nu12=0.3, g12=5000.0, sigyt1=1500.0)
        rendered = d.render()
        assert "/MAT/LAW25/10" in rendered

        model, log = _parse_string(tmp_path, rendered)
        assert len(log.errors) == 0
        assert 10 in model.mat_law25s
        assert model.mat_law25s[10].rho0 == pytest.approx(1.55e-9)

    def test_keyword_parsers_registered(self):
        for alias in ("MAT_LAW25", "LAW25", "MAT_COMP_PLAS", "COMP_PLAS", "MAT_COMPSH", "COMPSH",
                      "MAT_TSAI_WU", "TSAI_WU", "MAT_CRASURV", "CRASURV", "MAT_COMPOSITE_PLAS", "COMPOSITE_PLAS"):
            assert alias in KEYWORD_PARSERS, f"{alias} missing from KEYWORD_PARSERS"
            assert KEYWORD_PARSERS[alias].__name__ == "read_mat"
