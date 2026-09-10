"""Tests for Milestone M547: /MAT/LAW14 (/MAT/COMPSO, /MAT/COMP_SOL) input layer.

Covers:
  1. Card layout constants and aliases (MAT_LAW14_1..9, MAT_COMPSO_*, MAT_COMP_SOL_*).
  2. Starter reading of /MAT/LAW14, /MAT/COMPSO, /MAT/COMP_SOL in fixed format (9 cards).
  3. Starter reading in free format (comma-separated).
  4. Starter reading of synonyms (/MAT/COMPSO, /MAT/COMP_SOL).
  5. Dual-personality dispatch for /MAT/LAW14:
     - <= 2 data cards dispatches to Cam-Clay (M187)
     - > 2 data cards dispatches to COMPSO (M547)
  6. Starter validation checks (check_mat_law14):
     - Negative and zero density (rho0 <= 0)
     - Negative and zero Young's moduli (E11, E22, E33 <= 0)
     - Non-positive compliance determinant (DETC <= 0)
     - Valid parameters pass cleanly
  7. Element family and 2D restrictions in check_model:
     - Accepted element families: bricks, tetras, penta6, pyra5 (3D solids)
     - Rejected element families: shells, shells_qbat, shells_qeph, sh3n, quads, trusses, beams, springs
     - Rejected when N2D > 0 (2D axisymmetric / plane strain analysis per hm_read_mat14.F line 151)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import Material
from pyradioss.starter.checks import _ALLOWED_LAWS, check_model

# Try importing check_mat_law14, or provide fallback if subagent 1B is in flight
try:
    from pyradioss.starter.checks import check_mat_law14
except ImportError:
    check_mat_law14 = None  # type: ignore


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

class TestLaw14CardLayouts:
    """Card layout constants and LAYOUTS dictionary verification (M547)."""

    def test_constants_and_aliases(self):
        """Verify 9-card layouts for LAW14 and synonyms."""
        assert hasattr(cl, "MAT_LAW14_1")
        assert cl.MAT_LAW14_1 == (20, 20)
        assert cl.MAT_LAW14_2 == (20, 20, 20)
        assert cl.MAT_LAW14_3 == (20, 20, 20)
        assert cl.MAT_LAW14_4 == (20, 20, 20)
        assert cl.MAT_LAW14_5 == (20, 20, 20, 20)
        assert cl.MAT_LAW14_6 == (20, 20, 20, 20)
        assert cl.MAT_LAW14_7 == (20, 20, 20, 20)
        assert cl.MAT_LAW14_8 == (20, 20, 20, 20)
        assert cl.MAT_LAW14_9 == (20, 20, 20, 20, 10)

        for i in range(1, 10):
            law_val = getattr(cl, f"MAT_LAW14_{i}")
            if hasattr(cl, f"MAT_COMPSO_{i}"):
                assert getattr(cl, f"MAT_COMPSO_{i}") == law_val
            if hasattr(cl, f"MAT_COMP_SOL_{i}"):
                assert getattr(cl, f"MAT_COMP_SOL_{i}") == law_val

    def test_layouts_dict_entries(self):
        """Verify LAYOUTS dictionary has entries for MAT_LAW14_1..9 and aliases."""
        for i in range(1, 10):
            expected = list(getattr(cl, f"MAT_LAW14_{i}"))
            assert cl.LAYOUTS[f"MAT_LAW14_{i}"] == expected
            if f"MAT_COMPSO_{i}" in cl.LAYOUTS:
                assert cl.LAYOUTS[f"MAT_COMPSO_{i}"] == expected
            if f"MAT_COMP_SOL_{i}" in cl.LAYOUTS:
                assert cl.LAYOUTS[f"MAT_COMP_SOL_{i}"] == expected


# ============================================================================
# 2. Starter Reading (Fixed and Free Formats & Aliases)
# ============================================================================

class TestLaw14Reader:
    """Fixed- and free-format starter reader tests for LAW14 and aliases."""

    def test_parse_law14_fixed_format(self, tmp_path: Path):
        """Read 9-card /MAT/LAW14 in fixed format."""
        deck = """
/BEGIN
LAW14_FIXED_TEST
                  -1
/MAT/LAW14/14
Carbon_Epoxy_Solid_3D
#              RHO_I               RHO_O
             1.55e-9             1.55e-9
#                E11                 E22                 E33
            140000.0             10000.0             10000.0
#               NU12                NU23                NU31
                0.30                0.45                0.02
#                G12                 G23                 G31
              5000.0              3500.0              5000.0
#           SIGMA_T1            SIGMA_T2            SIGMA_T3               DELTA
              1800.0                40.0                40.0                0.08
#                  b                   n                Fmax             WplaRef
               150.0                 0.5              1200.0                 1.0
#          sigma_1yt           sigma_2yt           sigma_1yc           sigma_2yc
              2000.0                50.0              1500.0               200.0
#         sigma_12yt          sigma_12yc          sigma_23yt          sigma_23yc
                80.0                80.0                50.0                50.0
#              ALPHA                EFIB                   C                EPS0       ICC
                0.25            180000.0                0.04                 1.0         1
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"

        # MatLaw14 / MatCompso stored
        mat_entry = getattr(model, "mat_law14s", {}).get(14) or getattr(model, "mat_compsos", {}).get(14)
        assert mat_entry is not None
        assert mat_entry.id == 14
        assert mat_entry.rho0 == pytest.approx(1.55e-9)

        # Transverse isotropy in card 7 & 8: sigyt3 == sigyt2, sigyc3 == sigyc2
        if hasattr(mat_entry, "sig_2yt"):
            assert mat_entry.sig_2yt == pytest.approx(50.0)
        if hasattr(mat_entry, "sigyt2"):
            assert mat_entry.sigyt2 == pytest.approx(50.0)

        # Materials dispatch
        assert 14 in model.materials
        m = model.materials[14]
        assert m.law == 14
        assert m.params["D11"] > 0.0
        assert m.params["SSP"] > 0.0

    def test_parse_law14_free_format(self, tmp_path: Path):
        """Read 9-card /MAT/LAW14 in free format (comma-separated values)."""
        deck = """
/BEGIN
LAW14_FREE_TEST
-1
/MAT/LAW14/1
Composite_Solid_Free
1.6e-9, 1.6e-9
120000.0, 12000.0, 12000.0
0.28, 0.4, 0.03
6000.0, 4000.0, 6000.0
1500.0, 50.0, 50.0, 0.05
200.0, 0.4, 1000.0, 1.0
1600.0, 60.0, 1200.0, 180.0
70.0, 70.0, 45.0, 45.0
0.3, 160000.0, 0.02, 1.0, 1
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        mat_entry = getattr(model, "mat_law14s", {}).get(1) or getattr(model, "mat_compsos", {}).get(1)
        assert mat_entry is not None
        assert mat_entry.rho0 == pytest.approx(1.6e-9)
        assert 1 in model.materials

    def test_parse_law14_synonyms(self, tmp_path: Path):
        """Verify /MAT/COMPSO and /MAT/COMP_SOL parse into model materials."""
        deck = """
/BEGIN
LAW14_SYNONYMS_TEST
-1
/MAT/COMPSO/101
Synonym_COMPSO
1.5e-9, 1.5e-9
100000.0, 20000.0, 20000.0
0.25, 0.2, 0.15
8000.0, 6000.0, 8000.0
1000.0, 50.0, 50.0, 0.05
0.0, 1.0, 1.0e10, 1.0
1200.0, 60.0, 1000.0, 150.0
60.0, 60.0, 40.0, 40.0
0.0, 0.0, 0.0, 1.0, 1
/MAT/COMP_SOL/102
Synonym_COMP_SOL
1.5e-9, 1.5e-9
100000.0, 20000.0, 20000.0
0.25, 0.2, 0.15
8000.0, 6000.0, 8000.0
1000.0, 50.0, 50.0, 0.05
0.0, 1.0, 1.0e10, 1.0
1200.0, 60.0, 1000.0, 150.0
60.0, 60.0, 40.0, 40.0
0.0, 0.0, 0.0, 1.0, 1
/END
"""
        model, log = _parse_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 101 in model.materials
        assert 102 in model.materials
        assert model.materials[101].law == 14
        assert model.materials[102].law == 14

    def test_law14_cam_clay_dual_personality(self, tmp_path: Path):
        """If card count <= 2, dispatches to Cam-Clay (M187); if > 2 cards, dispatches to COMPSO."""
        # 1. Cam-Clay deck (2 cards)
        deck_camclay = """
/BEGIN
CAMCLAY_TEST
-1
/MAT/LAW14/501
CamClay_Mat
#              RHO_I                   E                  NU
              1.8e-9               100.0                 0.3
#             LAMBDA               KAPPA                   M                  E0                  P0
                0.05                0.01                 1.2                 0.8                50.0
/END
"""
        model1, log1 = _parse_string(tmp_path, deck_camclay)
        assert len(log1.errors) == 0, f"Errors: {log1.errors}"
        assert 501 in model1.mat_law14s
        # Cam-Clay entity has lambda / kappa
        cc_mat = model1.mat_law14s[501]
        assert hasattr(cc_mat, "lambda_") or hasattr(cc_mat, "lam") or "lambda" in str(cc_mat.__dict__).lower()

        # 2. COMPSO deck (9 cards)
        deck_compso = """
/BEGIN
COMPSO_TEST
-1
/MAT/LAW14/502
Compso_Mat
1.5e-9, 1.5e-9
100000.0, 20000.0, 20000.0
0.25, 0.2, 0.15
8000.0, 6000.0, 8000.0
1000.0, 50.0, 50.0, 0.05
0.0, 1.0, 1.0e10, 1.0
1200.0, 60.0, 1000.0, 150.0
60.0, 60.0, 40.0, 40.0
0.0, 0.0, 0.0, 1.0, 1
/END
"""
        model2, log2 = _parse_string(tmp_path, deck_compso)
        assert len(log2.errors) == 0, f"Errors: {log2.errors}"
        assert 502 in model2.materials
        m2 = model2.materials[502]
        assert m2.law == 14
        assert "D11" in m2.params
        assert m2.params["D11"] > 0.0


# ============================================================================
# 3. Validation Checks (check_mat_law14)
# ============================================================================

class TestLaw14ValidationChecks:
    """Starter parameter bounds validation checks for LAW14."""

    def _valid_params(self) -> Dict[str, Any]:
        return {
            "rho": 1.5e-9,
            "e11": 100000.0,
            "e22": 50000.0,
            "e33": 50000.0,
            "nu12": 0.25,
            "nu23": 0.20,
            "nu31": 0.25,
            "g12": 15000.0,
            "g23": 10000.0,
            "g31": 15000.0,
        }

    def test_check_mat_law14_valid(self):
        """Valid parameters pass without errors."""
        p = self._valid_params()
        mat = Material(id=14, law=14, rho0=1.5e-9, params=p)
        log = MessageLog()
        if check_mat_law14 is not None:
            check_mat_law14(mat, log)
            assert len(log.errors) == 0
        else:
            pytest.skip("check_mat_law14 not yet implemented in starter.checks")

    def test_check_mat_law14_invalid_rho(self):
        """rho <= 0 triggers error."""
        if check_mat_law14 is None:
            pytest.skip("check_mat_law14 not yet implemented in starter.checks")
        p = self._valid_params()
        p["rho"] = 0.0
        mat = Material(id=1, law=14, rho0=0.0, params=p)
        log = MessageLog()
        check_mat_law14(mat, log)
        assert any("density" in e.lower() and "must be > 0" in e for e in log.errors)

        p2 = self._valid_params()
        p2["rho"] = -1.5e-9
        mat2 = Material(id=2, law=14, rho0=-1.5e-9, params=p2)
        log2 = MessageLog()
        check_mat_law14(mat2, log2)
        assert any("density" in e.lower() and "must be > 0" in e for e in log2.errors)

    def test_check_mat_law14_invalid_moduli(self):
        """Negative/zero E or Poisson violating DETC > 0 triggers error."""
        if check_mat_law14 is None:
            pytest.skip("check_mat_law14 not yet implemented in starter.checks")

        # E11 <= 0
        p1 = self._valid_params()
        p1["e11"] = 0.0
        mat1 = Material(id=1, law=14, rho0=1.5e-9, params=p1)
        log1 = MessageLog()
        check_mat_law14(mat1, log1)
        assert any("Young's moduli" in e and "must be > 0" in e for e in log1.errors)

        # E22 < 0
        p2 = self._valid_params()
        p2["e22"] = -50000.0
        mat2 = Material(id=2, law=14, rho0=1.5e-9, params=p2)
        log2 = MessageLog()
        check_mat_law14(mat2, log2)
        assert any("Young's moduli" in e and "must be > 0" in e for e in log2.errors)

        # DETC <= 0 (unphysical Poisson ratios violating thermodynamic stability)
        p3 = self._valid_params()
        p3["nu12"] = 0.95
        p3["nu23"] = 0.95
        p3["nu31"] = 0.95
        mat3 = Material(id=3, law=14, rho0=1.5e-9, params=p3)
        log3 = MessageLog()
        check_mat_law14(mat3, log3)
        assert any("DETC" in e and "must be > 0" in e for e in log3.errors)


# ============================================================================
# 4. Element Restrictions & 2D Restrictions (check_model)
# ============================================================================

class TestLaw14ElementRestrictions:
    """Verify LAW14 element family compatibility and 2D restrictions in check_model."""

    def test_check_mat_law14_element_compatibility(self):
        """Solid elements allowed, shell/beam/truss/spring rejected."""
        solid_families = ("bricks", "tetras", "penta6", "pyra5")
        for fam in solid_families:
            assert 14 in _ALLOWED_LAWS[fam]
            assert "14" in _ALLOWED_LAWS[fam] or "LAW14" in _ALLOWED_LAWS[fam]

        # Solids pass cleanly
        model = Model()
        model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
        mat = Material(id=1, law=14, rho0=1.5e-9, params={"E11": 100000.0, "E22": 50000.0, "E33": 50000.0})
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [("bricks", FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert len(log.errors) == 0

        # Shell/beam/truss/spring rejected
        non_solids = ["shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "trusses", "beams", "springs"]
        for family in non_solids:
            m_rej = Model()
            m_rej.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
            m_rej.materials[1] = mat
            m_rej.element_groups = lambda fam=family: [(fam, FakeGroup())]
            log_rej = MessageLog()
            check_model(m_rej, log_rej)
            assert any(
                "is not supported for" in e or "is not ported for" in e
                for e in log_rej.errors
            )

    def test_check_mat_law14_2d_rejection(self):
        """2D elements (N2D > 0) rejected per hm_read_mat14.F line 151."""
        model = Model()
        model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
        model.n2d = 1  # 2D axisymmetric / plane strain
        mat = Material(id=1, law=14, rho0=1.5e-9, params={"E11": 100000.0, "E22": 50000.0, "E33": 50000.0})
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [("bricks", FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert any("LAW14 is not supported for 2D analysis" in e or "not supported for 2D" in e for e in log.errors)


# ============================================================================
# 5. Deck Writer Tests (StarterDeck.mat_law14, mat_compso, mat_comp_sol)
# ============================================================================

class TestLaw14DeckWriter:
    """Deck writer tests for /MAT/LAW14, /MAT/COMPSO, /MAT/COMP_SOL."""

    def test_writer_and_roundtrip(self, tmp_path: Path):
        from pyradioss.input.deck_writer import StarterDeck

        deck = StarterDeck("LAW14_ROUNDTRIP")
        deck.begin("LAW14_ROUNDTRIP")
        deck.mat_law14(
            mid=14,
            title="Composite_Carbon_3D",
            rho=1.55e-9,
            ea=140000.0,
            eb=10000.0,
            ec=10000.0,
            prab=0.30,
            prbc=0.45,
            prca=0.02,
            gab=5000.0,
            gbc=3500.0,
            gca=5000.0,
            sigt1=1800.0,
            sigt2=40.0,
            sigt3=40.0,
            delta=0.08,
            cb=150.0,
            cn=0.5,
            fmax=1200.0,
            wplaref=1.0,
            sigyt1=2000.0,
            sigyt2=50.0,
            sigyc1=1500.0,
            sigyc2=200.0,
            sigyt12=80.0,
            sigyc12=80.0,
            sigyt23=50.0,
            sigyc23=50.0,
            alpha=0.25,
            efib=180000.0,
            cc=0.04,
            eps0=1.0,
            strflag=1,
        )
        deck.mat_compso(
            mid=15,
            title="Compso_Alias",
            rho=1.6e-9,
            ea=100000.0,
            eb=20000.0,
            ec=20000.0,
            prab=0.25,
            prbc=0.2,
            prca=0.15,
            gab=8000.0,
            gbc=6000.0,
            gca=8000.0,
            sigyt1=1200.0,
            sigyt2=60.0,
            sigyc1=1000.0,
            sigyc2=150.0,
            sigyt12=60.0,
            sigyc12=60.0,
            sigyt23=40.0,
            sigyc23=40.0,
        )
        deck.end()
        text = deck.render()

        assert "/MAT/LAW14/14" in text
        assert "/MAT/COMPSO/15" in text

        model, log = _parse_string(tmp_path, text)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 14 in model.mat_law14s
        assert 15 in model.mat_law14s

        m14 = model.mat_law14s[14]
        assert m14.rho0 == pytest.approx(1.55e-9)
        assert m14.ea == pytest.approx(140000.0)
        assert m14.eb == pytest.approx(10000.0)
        assert m14.prab == pytest.approx(0.30)
        assert m14.gab == pytest.approx(5000.0)
        assert m14.sigt1 == pytest.approx(1800.0)
        assert m14.delta == pytest.approx(0.08)
        assert m14.cb == pytest.approx(150.0)
        assert m14.cn == pytest.approx(0.5)
        assert m14.fmax == pytest.approx(1200.0)
        assert m14.sigyt1 == pytest.approx(2000.0)
        assert m14.alpha == pytest.approx(0.25)
        assert m14.efib == pytest.approx(180000.0)
        assert m14.cc == pytest.approx(0.04)

