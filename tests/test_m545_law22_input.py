"""
Tests for Milestone M545: /MAT/LAW22 (/MAT/DAMA, /MAT/PLAS_DAMA) input layer.

Covers:
  1. Card layout constants and aliases (MAT_LAW22_1..5, MAT_DAMA_*, MAT_PLAS_DAMA_*).
  2. Starter reading of /MAT/LAW22, /MAT/DAMA, /MAT/PLAS_DAMA in fixed format.
  3. Starter reading in free format (comma-separated and whitespace-separated).
  4. Starter validation checks (check_mat_law22):
     - Negative and zero density (rho0 <= 0)
     - Negative and zero Young's modulus (E <= 0)
     - Invalid Poisson's ratio (nu < 0 or nu >= 0.5)
     - Non-positive yield stress (a <= 0)
     - Hardening exponent bound (n > 1.0 per hm_read_mat22.F line 213)
     - Reference strain rate bound (eps_dot_0 <= 0 per hm_read_mat22.F line 221)
     - Softening damage slope bound (E_tan > 0 per hm_read_mat22.F line 229 / matl22_dama.cfg)
  5. Element family compatibility in check_model:
     - Accepted element families: shells, shells_qbat, shells_qeph, sh3n, quads, bricks, tetras, penta6, pyra5
     - Rejected element families: trusses, beams, springs
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import Material, MatLaw22, MatDama, MatPlasDama
from pyradioss.starter.checks import _ALLOWED_LAWS, check_mat_law22, check_model


def _parse_string(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    """Helper to write and parse a deck from a string."""
    p = tmp_path / "DECK_0000.rad"
    p.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


# ============================================================================
# 1. Card Layout Constants & Aliases
# ============================================================================

class TestLaw22CardLayouts:
    """Card layout constants and LAYOUTS dictionary verification (M545)."""

    def test_constants_and_aliases(self):
        assert cl.MAT_LAW22_1 == (20, 20)
        assert cl.MAT_LAW22_2 == (20, 20)
        assert cl.MAT_LAW22_3 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW22_4 == (20, 20, 10)
        assert cl.MAT_LAW22_5 == (20, 20)

        for i in range(1, 6):
            law_val = getattr(cl, f"MAT_LAW22_{i}")
            assert getattr(cl, f"MAT_DAMA_{i}") == law_val
            assert getattr(cl, f"MAT_PLAS_DAMA_{i}") == law_val
            assert getattr(cl, f"MAT_LAW22_CFG_{i}") == law_val
            assert getattr(cl, f"MAT_DAMA_CFG_{i}") == law_val
            assert getattr(cl, f"MAT_PLAS_DAMA_CFG_{i}") == law_val

    def test_layouts_dict_entries(self):
        for i in range(1, 6):
            expected = list(getattr(cl, f"MAT_LAW22_{i}"))
            assert cl.LAYOUTS[f"MAT_LAW22_{i}"] == expected
            assert cl.LAYOUTS[f"MAT_DAMA_{i}"] == expected
            assert cl.LAYOUTS[f"MAT_PLAS_DAMA_{i}"] == expected


# ============================================================================
# 2. Starter Reading (Fixed and Free Formats & Aliases)
# ============================================================================

class TestLaw22Reader:
    """Fixed- and free-format starter reader tests for LAW22 and aliases."""

    def test_read_law22_fixed_format(self, tmp_path: Path):
        """Read 5-card /MAT/LAW22 in fixed format."""
        deck = """
/BEGIN
LAW22_FIXED_TEST
                  -1
/MAT/LAW22/1
Steel_DP600_DAMA
#              RHO_I               RHO_O
             7.85e-9             7.85e-9
#                  E                  Nu
            210000.0                 0.3
#                  a                   b                   n             Eps_max           SIGMA_max
               350.0               450.0                 0.5                0.25               900.0
#                  c           Eps_dot_0       ICC
                0.03                 1.0         1
#            Eps_dam                 E_t
                0.05            -10000.0
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 1 in model.mat_law22s
        mat = model.mat_law22s[1]
        assert mat.id == 1
        assert mat.rho0 == pytest.approx(7.85e-9)
        assert mat.rhor == pytest.approx(7.85e-9)
        assert mat.e == pytest.approx(210000.0)
        assert mat.nu == pytest.approx(0.3)
        assert mat.a == pytest.approx(350.0)
        assert mat.b == pytest.approx(450.0)
        assert mat.n == pytest.approx(0.5)
        assert mat.eps_max == pytest.approx(0.25)
        assert mat.sig_max == pytest.approx(900.0)
        assert mat.c == pytest.approx(0.03)
        assert mat.eps_dot_0 == pytest.approx(1.0)
        assert mat.icc == 1
        assert mat.eps_dam == pytest.approx(0.05)
        assert mat.e_tan == pytest.approx(-10000.0)

        # Instantiated Material
        assert 1 in model.materials
        m = model.materials[1]
        assert m.law == 22
        assert m.params["E"] == pytest.approx(210000.0)
        assert m.params["E_tan"] == pytest.approx(-10000.0)

    def test_read_dama_alias_fixed_format(self, tmp_path: Path):
        """Read /MAT/DAMA alias in fixed format."""
        deck = """
/BEGIN
DAMA_TEST
                  -1
/MAT/DAMA/2
Alu_DAMA_Test
#              RHO_I
              2.7e-9
#                  E                  Nu
             70000.0                0.33
#                  a                   b                   n             Eps_max           SIGMA_max
               180.0               220.0                 0.4                0.15               400.0
#                  c           Eps_dot_0       ICC
                 0.0                 1.0         1
#            Eps_dam                 E_t
                0.08             -5000.0
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 2 in model.mat_damas
        mat = model.mat_damas[2]
        assert mat.id == 2
        assert mat.rho0 == pytest.approx(2.7e-9)
        assert mat.e == pytest.approx(70000.0)
        assert mat.nu == pytest.approx(0.33)
        assert mat.a == pytest.approx(180.0)
        assert mat.b == pytest.approx(220.0)
        assert mat.n == pytest.approx(0.4)
        assert mat.eps_dam == pytest.approx(0.08)
        assert mat.e_tan == pytest.approx(-5000.0)

    def test_read_plas_dama_alias_fixed_format(self, tmp_path: Path):
        """Read /MAT/PLAS_DAMA alias in fixed format."""
        deck = """
/BEGIN
PLAS_DAMA_TEST
                  -1
/MAT/PLAS_DAMA/3
Polymer_DAMA
#              RHO_I
              1.2e-9
#                  E                  Nu
              3000.0                0.38
#                  a                   b                   n             Eps_max           SIGMA_max
                50.0                30.0                 0.6                 0.5               100.0
#                  c           Eps_dot_0       ICC
                0.05                 0.5         2
#            Eps_dam                 E_t
                0.10             -1000.0
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 3 in model.mat_plas_damas
        mat = model.mat_plas_damas[3]
        assert mat.id == 3
        assert mat.rho0 == pytest.approx(1.2e-9)
        assert mat.e == pytest.approx(3000.0)
        assert mat.nu == pytest.approx(0.38)
        assert mat.a == pytest.approx(50.0)
        assert mat.b == pytest.approx(30.0)
        assert mat.icc == 2
        assert mat.eps_dam == pytest.approx(0.10)
        assert mat.e_tan == pytest.approx(-1000.0)

    def test_read_law22_free_format_comma(self, tmp_path: Path):
        """Read /MAT/LAW22 in comma-separated free format."""
        deck = """
/BEGIN
LAW22_FREE_COMMA
                  -1
/MAT/LAW22/10
Free_Comma_Test
7.85e-9, 7.85e-9
210000.0, 0.3
350.0, 450.0, 0.5, 0.25, 900.0
0.03, 1.0, 1
0.05, -10000.0
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 10 in model.mat_law22s
        mat = model.mat_law22s[10]
        assert mat.id == 10
        assert mat.rho0 == pytest.approx(7.85e-9)
        assert mat.e == pytest.approx(210000.0)
        assert mat.nu == pytest.approx(0.3)
        assert mat.a == pytest.approx(350.0)
        assert mat.b == pytest.approx(450.0)
        assert mat.n == pytest.approx(0.5)
        assert mat.eps_max == pytest.approx(0.25)
        assert mat.sig_max == pytest.approx(900.0)
        assert mat.c == pytest.approx(0.03)
        assert mat.eps_dot_0 == pytest.approx(1.0)
        assert mat.icc == 1
        assert mat.eps_dam == pytest.approx(0.05)
        assert mat.e_tan == pytest.approx(-10000.0)

    def test_read_law22_free_format_whitespace(self, tmp_path: Path):
        """Read /MAT/LAW22 in whitespace-separated free format."""
        deck = """
/BEGIN
LAW22_FREE_WS
                  -1
/MAT/LAW22/11
Free_Whitespace_Test
7.85e-9
205000.0 0.29
300.0 400.0 0.45 0.30 850.0
0.02 0.8 1
0.06 -8000.0
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 11 in model.mat_law22s
        mat = model.mat_law22s[11]
        assert mat.id == 11
        assert mat.rho0 == pytest.approx(7.85e-9)
        assert mat.e == pytest.approx(205000.0)
        assert mat.nu == pytest.approx(0.29)
        assert mat.a == pytest.approx(300.0)
        assert mat.b == pytest.approx(400.0)
        assert mat.n == pytest.approx(0.45)
        assert mat.eps_dam == pytest.approx(0.06)
        assert mat.e_tan == pytest.approx(-8000.0)


# ============================================================================
# 3. Parameter Validation Checks (check_mat_law22)
# ============================================================================

class TestLaw22StarterChecks:
    """Starter checks parameter validation for /MAT/LAW22 (M545)."""

    def _valid_params(self) -> dict:
        return {
            "rho0": 7.85e-9,
            "e": 210000.0,
            "nu": 0.3,
            "a": 350.0,
            "b": 450.0,
            "n": 0.5,
            "eps_max": 0.25,
            "sig_max": 900.0,
            "c": 0.03,
            "eps_dot_0": 1.0,
            "icc": 1,
            "eps_dam": 0.05,
            "e_tan": -10000.0,
        }

    def test_valid_law22_passes(self):
        mat = Material(id=1, law=22, rho0=7.85e-9, params=self._valid_params())
        log = MessageLog()
        check_mat_law22(mat, log)
        assert len(log.errors) == 0

    def test_negative_density_rejected(self):
        p = self._valid_params()
        p["rho0"] = -7.85e-9
        mat = Material(id=1, law=22, rho0=-7.85e-9, params=p)
        log = MessageLog()
        check_mat_law22(mat, log)
        assert any("initial density RHO must be > 0" in e for e in log.errors)

    def test_zero_young_modulus_rejected(self):
        p = self._valid_params()
        p["e"] = 0.0
        mat = Material(id=1, law=22, rho0=7.85e-9, params=p)
        log = MessageLog()
        check_mat_law22(mat, log)
        assert any("Young's modulus E must be > 0" in e for e in log.errors)

        p2 = self._valid_params()
        p2["e"] = -200000.0
        mat2 = Material(id=2, law=22, rho0=7.85e-9, params=p2)
        log2 = MessageLog()
        check_mat_law22(mat2, log2)
        assert any("Young's modulus E must be > 0" in e for e in log2.errors)

    def test_invalid_poisson_ratio_rejected(self):
        p = self._valid_params()
        p["nu"] = 0.5
        mat = Material(id=1, law=22, rho0=7.85e-9, params=p)
        log = MessageLog()
        check_mat_law22(mat, log)
        assert any("Poisson's ratio NU must be in [0, 0.5)" in e for e in log.errors)

        p2 = self._valid_params()
        p2["nu"] = -0.1
        mat2 = Material(id=2, law=22, rho0=7.85e-9, params=p2)
        log2 = MessageLog()
        check_mat_law22(mat2, log2)
        assert any("Poisson's ratio NU must be in [0, 0.5)" in e for e in log2.errors)

    def test_negative_yield_stress_rejected(self):
        p = self._valid_params()
        p["a"] = -100.0
        mat = Material(id=1, law=22, rho0=7.85e-9, params=p)
        log = MessageLog()
        check_mat_law22(mat, log)
        assert any("yield stress a (SIGY) must be > 0" in e for e in log.errors)

        p2 = self._valid_params()
        p2["a"] = 0.0
        mat2 = Material(id=2, law=22, rho0=7.85e-9, params=p2)
        log2 = MessageLog()
        check_mat_law22(mat2, log2)
        assert any("yield stress a (SIGY) must be > 0" in e for e in log2.errors)

    def test_hardening_exponent_exceeding_one_rejected(self):
        """Hardening exponent n > 1.0 rejected per hm_read_mat22.F line 213."""
        p = self._valid_params()
        p["n"] = 1.2
        mat = Material(id=1, law=22, rho0=7.85e-9, params=p)
        log = MessageLog()
        check_mat_law22(mat, log)
        assert any("hardening exponent n must be <= 1.0" in e for e in log.errors)

    def test_nonpositive_strain_rate_rejected(self):
        """Reference strain rate eps_dot_0 <= 0 rejected per hm_read_mat22.F line 221."""
        p = self._valid_params()
        p["eps_dot_0"] = 0.0
        mat = Material(id=1, law=22, rho0=7.85e-9, params=p)
        log = MessageLog()
        check_mat_law22(mat, log)
        assert any("reference strain rate EPS_DOT_0 (srp) must be > 0" in e for e in log.errors)

        p2 = self._valid_params()
        p2["eps_dot_0"] = -1.0
        mat2 = Material(id=2, law=22, rho0=7.85e-9, params=p2)
        log2 = MessageLog()
        check_mat_law22(mat2, log2)
        assert any("reference strain rate EPS_DOT_0 (srp) must be > 0" in e for e in log2.errors)

    def test_positive_softening_slope_rejected(self):
        """Softening damage slope E_tan > 0 rejected per hm_read_mat22.F line 229 / matl22_dama.cfg."""
        p = self._valid_params()
        p["e_tan"] = 1000.0
        mat = Material(id=1, law=22, rho0=7.85e-9, params=p)
        log = MessageLog()
        check_mat_law22(mat, log)
        assert any("softening damage slope E_tan must be <= 0.0" in e for e in log.errors)


# ============================================================================
# 4. Element Family Compatibility (check_model)
# ============================================================================

class TestLaw22ElementCompatibility:
    """Verify LAW22 element family compatibility in check_model."""

    def test_allowed_on_shells_and_solids(self):
        """LAW22 is allowed on shells and 3D solids."""
        allowed_families = (
            "shells", "shells_qbat", "shells_qeph", "sh3n", "quads",
            "bricks", "tetras", "penta6", "pyra5"
        )
        for fam in allowed_families:
            assert 22 in _ALLOWED_LAWS[fam]
            assert "22" in _ALLOWED_LAWS[fam]
            assert "LAW22" in _ALLOWED_LAWS[fam]
            assert "DAMA" in _ALLOWED_LAWS[fam]
            assert "PLAS_DAMA" in _ALLOWED_LAWS[fam]

    @pytest.mark.parametrize("family", ["trusses", "beams", "springs"])
    def test_rejected_on_1d_elements(self, family: str):
        """LAW22 applied to 1D beams/trusses/springs causes check_model error."""
        model = Model()
        model.add_nodes(np.array([1, 2]), np.zeros((2, 3)))
        mat = Material(id=1, law=22, rho0=7.85e-9, params={"e": 210000.0, "nu": 0.3, "a": 350.0})
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [(family, FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert any(
            f"is not supported for {family} elements" in e or f"is not ported for {family} elements" in e
            for e in log.errors
        )
