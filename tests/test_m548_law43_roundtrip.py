"""Milestone M548: /MAT/LAW43 (/MAT/HILL_TAB, /MAT/HILL_PLAS_TAB) Roundtrip & Restart Serialization Auditor.

Audits:
  1. Card Format Roundtrip:
     - 5 fixed-format cards + repeated curve cards per radioss140/MAT/matl43_HILL_TAB.cfg
       and card_layouts.py (MAT_LAW43_1..5, MAT_LAW43_CURVE).
     - Card-by-card audit:
         Card 1: RHO, Refer_Rho (%20lg%20lg)
         Card 2: E, nu (%20lg%20lg)
         Card 3: Yr_fun, Einf, C (%10d[10s]%20lg%20lg or %10d%20lg%20lg)
         Card 4: R00, R45, R90, CHard, Iyield (%20lg%20lg%20lg%20lg%10d)
         Card 5: EPS_max, EPST1, EPST2, Fcut, Fsmooth (%20lg%20lg%20lg%20lg%10d)
         Curve cards: fct_ID, Fscale, EPS_DOT (%10d[10s]%20lg%20lg or %10d%20lg%20lg)
     - Free format reading (space-delimited, comma-delimited, and legacy 4-card format).
     - StarterDeck.mat_law43, mat_hill_tab, and mat_hill_plas_tab writer methods.
     - Full roundtrip: generate deck string -> parse via read_mat_law43 -> assert exact parameter and curve matches.

  2. Keyword Synonyms:
     - Verify /MAT/LAW43, /MAT/HILL_TAB, /MAT/HILL_PLAS_TAB, and /MAT/LAW43_HILL_TAB parse identically.

  3. Starter Checks & Element Compatibility:
     - Verify shells (SHELL, SH3N, QBAT, QEPH), quads, and solids (bricks, tetras, penta6, pyra5) pass validation.
     - Verify beams (BEAM), trusses (TRUSS), springs (SPRING) are strictly rejected with error messages.
     - Verify parameter bounds checks:
         - rho0 <= 0 rejected.
         - E <= 0 rejected.
         - nu < 0 or nu >= 0.5 rejected.
         - R00 <= 0, R45 <= 0, R90 <= 0 rejected.
         - FISOKIN < 0 or FISOKIN > 1 rejected (ANCMSG 913).
         - zero curves / no plasticity curves rejected (ANCMSG 366).

  4. Restart (.rst) Pickling & State Serialization:
     - extra_shapes dictionary for nip=None and nip=5 pickled and deserialized without corruption.
     - MatLaw43 dataclass entity pickled and deserialized.
     - Material object built by build_law43 pickled and deserialized with sound speeds and parameters preserved.
     - Element state arrays (pla43, uvar43, off43, edot43, thk43) pickled and deserialized with assert_allclose.
     - Model object containing LAW43 materials and aliases pickled and restored.
"""

from __future__ import annotations

import math
from pathlib import Path
import pickle
from typing import Any, Dict, List
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials import law43_hill_tab
from pyradioss.materials.law43_hill_tab import (
    build_law43,
    extra_shapes,
    shell_update,
    solid_update,
    sound_speed,
)
from pyradioss.model.entities import Material, MatLaw43, MatHillTab, MatHillPlasTab, MatLaw43HillTab
from pyradioss.model.model import Model
from pyradioss.starter.checks import _ALLOWED_LAWS, check_mat_law43, check_model


def _parse_deck_string(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    """Helper to write and parse a deck string into Model and MessageLog."""
    deck_path = tmp_path / "ROUNDTRIP_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def _get_mat_cards(deck_str: str) -> List[str]:
    """Extract material data cards from rendered deck (skipping headers/comments)."""
    lines = deck_str.splitlines()
    mat_idx = next(i for i, line in enumerate(lines) if line.startswith("/MAT/"))
    cards = []
    for line in lines[mat_idx + 2:]:
        if line.startswith("/"):
            break
        if not line.startswith("#"):
            cards.append(line)
    return cards


# ============================================================================
# 1. Five Cards Strict Fixed Format & Card Layout Audit (matl43_HILL_TAB.cfg)
# ============================================================================

class TestLaw43FixedFormatAudit:
    """Audit of all 5 cards + curves in strict fixed format per radioss140/MAT/matl43_HILL_TAB.cfg."""

    def test_card_layout_constants_and_aliases(self):
        """Verify layout tuples and aliases in card_layouts.py."""
        assert cl.MAT_LAW43_1 == (20, 20)
        assert cl.MAT_LAW43_2 == (20, 20)
        assert cl.MAT_LAW43_3 == (10, 20, 20)
        assert cl.MAT_LAW43_4 == (20, 20, 20, 20, 10)
        assert cl.MAT_LAW43_5 == (20, 20, 20, 20, 10)
        assert cl.MAT_LAW43_CURVE == (10, 20, 20)

        for name in ("MAT_LAW43", "MAT_HILL_TAB", "MAT_HILL_PLAS_TAB", "MAT_LAW43_HILL_TAB"):
            for i in range(1, 6):
                assert getattr(cl, f"{name}_{i}") == getattr(cl, f"MAT_LAW43_{i}")
                assert getattr(cl, f"{name}_CFG_{i}") == getattr(cl, f"MAT_LAW43_{i}")
                assert cl.LAYOUTS[f"{name}_{i}"] == list(getattr(cl, f"MAT_LAW43_{i}"))
                assert cl.LAYOUTS[f"{name}_CFG_{i}"] == list(getattr(cl, f"MAT_LAW43_{i}"))
            assert getattr(cl, f"{name}_CURVE") == cl.MAT_LAW43_CURVE
            assert getattr(cl, f"{name}_CFG_CURVE") == cl.MAT_LAW43_CURVE
            assert cl.LAYOUTS[f"{name}_CURVE"] == list(cl.MAT_LAW43_CURVE)
            assert cl.LAYOUTS[f"{name}_CFG_CURVE"] == list(cl.MAT_LAW43_CURVE)

    def test_card1_rho_refer_rho(self, tmp_path: Path):
        """Card 1: RHO, Refer_Rho (%20lg%20lg). Tests dual density and single density."""
        # Dual density
        d1 = StarterDeck("CARD1_DUAL").mat_law43(
            mid=1, rho=7.8e-6, refer_rho=8.0e-6,
            e=210000.0, nu=0.3,
            curves=[{"fct_id": 101, "fscale": 1.0, "eps_dot": 0.0}],
        )
        cards1 = _get_mat_cards(d1.render())
        f1 = cl.split_fixed(cards1[0], cl.MAT_LAW43_1)
        assert float(f1[0]) == pytest.approx(7.8e-6)
        assert float(f1[1]) == pytest.approx(8.0e-6)

        model1, log1 = _parse_deck_string(tmp_path, d1.render())
        assert len(log1.errors) == 0
        mat1 = model1.mat_law43s[1]
        assert mat1.rho0 == pytest.approx(7.8e-6)
        assert mat1.rhor == pytest.approx(8.0e-6)

        # Single density (refer_rho defaults to rho0)
        d2 = StarterDeck("CARD1_SINGLE").mat_law43(
            mid=2, rho=2.7e-6,
            e=70000.0, nu=0.33,
            curves=[{"fct_id": 101, "fscale": 1.0, "eps_dot": 0.0}],
        )
        cards2 = _get_mat_cards(d2.render())
        f2 = cl.split_fixed(cards2[0], cl.MAT_LAW43_1)
        assert float(f2[0]) == pytest.approx(2.7e-6)

        model2, log2 = _parse_deck_string(tmp_path, d2.render())
        assert len(log2.errors) == 0
        mat2 = model2.mat_law43s[2]
        assert mat2.rho0 == pytest.approx(2.7e-6)
        assert mat2.rhor == pytest.approx(2.7e-6)

    def test_card2_e_nu(self, tmp_path: Path):
        """Card 2: E, nu (%20lg%20lg)."""
        d = StarterDeck("CARD2_TEST").mat_law43(
            mid=1, rho=7.85e-6, e=205000.0, nu=0.29,
            curves=[{"fct_id": 101, "fscale": 1.0, "eps_dot": 0.0}],
        )
        cards = _get_mat_cards(d.render())
        f = cl.split_fixed(cards[1], cl.MAT_LAW43_2)
        assert float(f[0]) == pytest.approx(205000.0)
        assert float(f[1]) == pytest.approx(0.29)

        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        mat = model.mat_law43s[1]
        assert mat.e == pytest.approx(205000.0)
        assert mat.nu == pytest.approx(0.29)

    def test_card3_ifunce_einf_ce(self, tmp_path: Path):
        """Card 3: Yr_fun, Einf, C (%10d%20lg%20lg and CFG %10d[10s]%20lg%20lg)."""
        d = StarterDeck("CARD3_TEST").mat_law43(
            mid=1, rho=7.8e-6, e=210000.0, nu=0.3,
            ifunce=501, einf=180000.0, ce=25.5,
            curves=[{"fct_id": 101, "fscale": 1.0, "eps_dot": 0.0}],
        )
        cards = _get_mat_cards(d.render())
        f = cl.split_fixed(cards[2], cl.MAT_LAW43_3)
        assert int(f[0]) == 501
        assert float(f[1]) == pytest.approx(180000.0)
        assert float(f[2]) == pytest.approx(25.5)

        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        mat = model.mat_law43s[1]
        assert mat.ifunce == 501
        assert mat.einf == pytest.approx(180000.0)
        assert mat.ce == pytest.approx(25.5)

        # CFG format with 10 blank spaces between Yr_fun and Einf
        cfg_card3 = f"{501:10d}          {180000.0:20.6f}{25.5:20.6f}"
        deck_cfg = f"""#RADIOSS STARTER
/BEGIN
Test CFG Card 3
                  10                   0
/MAT/LAW43/1
Test CFG
{7.8e-6:20.6e}
{210000.0:20.6f}{0.3:20.6f}
{cfg_card3}
{1.1:20.6f}{1.2:20.6f}{1.3:20.6f}{0.2:20.6f}{1:10d}
{0.2:20.6f}{0.05:20.6f}{0.1:20.6f}{500.0:20.6f}{1:10d}
{101:10d}          {1.0:20.6f}{0.0:20.6f}
"""
        model_cfg, log_cfg = _parse_deck_string(tmp_path, deck_cfg)
        assert len(log_cfg.errors) == 0
        mat_cfg = model_cfg.mat_law43s[1]
        assert mat_cfg.ifunce == 501
        assert mat_cfg.einf == pytest.approx(180000.0)
        assert mat_cfg.ce == pytest.approx(25.5)

    def test_card4_r00_r45_r90_chard_iyield(self, tmp_path: Path):
        """Card 4: R00, R45, R90, CHard, Iyield (%20lg%20lg%20lg%20lg%10d)."""
        d = StarterDeck("CARD4_TEST").mat_law43(
            mid=1, rho=7.8e-6, e=210000.0, nu=0.3,
            r00=1.65, r45=1.25, r90=1.85, chard=0.35, iyield=1,
            curves=[{"fct_id": 101, "fscale": 1.0, "eps_dot": 0.0}],
        )
        cards = _get_mat_cards(d.render())
        f = cl.split_fixed(cards[3], cl.MAT_LAW43_4)
        assert float(f[0]) == pytest.approx(1.65)
        assert float(f[1]) == pytest.approx(1.25)
        assert float(f[2]) == pytest.approx(1.85)
        assert float(f[3]) == pytest.approx(0.35)
        assert int(f[4]) == 1

        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        mat = model.mat_law43s[1]
        assert mat.r00 == pytest.approx(1.65)
        assert mat.r45 == pytest.approx(1.25)
        assert mat.r90 == pytest.approx(1.85)
        assert mat.chard == pytest.approx(0.35)
        assert mat.fisokin == pytest.approx(0.35)
        assert mat.iyield == 1

    def test_card5_failure_and_filtering(self, tmp_path: Path):
        """Card 5: EPS_max, EPST1, EPST2, Fcut, Fsmooth (%20lg%20lg%20lg%20lg%10d)."""
        d = StarterDeck("CARD5_TEST").mat_law43(
            mid=1, rho=7.8e-6, e=210000.0, nu=0.3,
            eps_max=0.35, epst1=0.12, epst2=0.25, fcut=1000.0, fsmooth=1,
            curves=[{"fct_id": 101, "fscale": 1.0, "eps_dot": 0.0}],
        )
        cards = _get_mat_cards(d.render())
        f = cl.split_fixed(cards[4], cl.MAT_LAW43_5)
        assert float(f[0]) == pytest.approx(0.35)
        assert float(f[1]) == pytest.approx(0.12)
        assert float(f[2]) == pytest.approx(0.25)
        assert float(f[3]) == pytest.approx(1000.0)
        assert int(f[4]) == 1

        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        mat = model.mat_law43s[1]
        assert mat.eps_max == pytest.approx(0.35)
        assert mat.epst1 == pytest.approx(0.12)
        assert mat.epst2 == pytest.approx(0.25)
        assert mat.fcut == pytest.approx(1000.0)
        assert mat.asrate == pytest.approx(1000.0)
        assert mat.fsmooth == 1
        assert mat.israte == 1

    def test_repeated_curve_cards_fixed(self, tmp_path: Path):
        """Repeated curve cards (MAT_LAW43_CURVE: [10, 20, 20])."""
        curve_defs = [
            {"fct_id": 201, "fscale": 1.0, "eps_dot": 0.0},
            {"fct_id": 202, "fscale": 1.15, "eps_dot": 10.0},
            {"fct_id": 203, "fscale": 1.30, "eps_dot": 100.0},
        ]
        d = StarterDeck("CURVES_TEST").mat_law43(
            mid=1, rho=7.8e-6, e=210000.0, nu=0.3,
            curves=curve_defs,
        )
        cards = _get_mat_cards(d.render())
        assert len(cards) == 8  # 5 parameter cards + 3 curve cards

        for i, cdef in enumerate(curve_defs):
            cc = cl.split_fixed(cards[5 + i], cl.MAT_LAW43_CURVE)
            assert int(cc[0]) == cdef["fct_id"]
            assert float(cc[1]) == pytest.approx(cdef["fscale"])
            assert float(cc[2]) == pytest.approx(cdef["eps_dot"])

        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        mat = model.mat_law43s[1]
        assert len(mat.curves) == 3
        for i, cdef in enumerate(curve_defs):
            assert mat.curves[i]["fct_id"] == cdef["fct_id"]
            assert mat.curves[i]["funct_id"] == cdef["fct_id"]
            assert mat.curves[i]["fscale"] == pytest.approx(cdef["fscale"])
            assert mat.curves[i]["eps_dot"] == pytest.approx(cdef["eps_dot"])

    def test_full_fixed_roundtrip_all_parameters(self, tmp_path: Path):
        """Generate deck string via StarterDeck.mat_law43 -> parse -> assert exact match."""
        curves = [
            {"fct_id": 301, "fscale": 1.0, "eps_dot": 0.0},
            {"fct_id": 302, "fscale": 1.05, "eps_dot": 0.1},
            {"fct_id": 303, "fscale": 1.25, "eps_dot": 500.0},
        ]
        deck = StarterDeck("FULL_ROUNDTRIP").mat_law43(
            mid=43,
            title="Aluminum_6061_T6_Tabulated_Hill",
            rho=2.7e-6,
            rhor=2.75e-6,
            e=70000.0,
            nu=0.33,
            ifunce=501,
            einf=65000.0,
            ce=12.0,
            r00=1.45,
            r45=1.15,
            r90=1.75,
            chard=0.40,
            iyield=1,
            eps_max=0.28,
            epst1=0.10,
            epst2=0.22,
            fcut=2000.0,
            fsmooth=1,
            curves=curves,
        )

        model, log = _parse_deck_string(tmp_path, deck.render())
        assert len(log.errors) == 0
        assert 43 in model.mat_law43s
        m = model.mat_law43s[43]

        assert m.id == 43
        assert m.title == "Aluminum_6061_T6_Tabulated_Hill"
        assert m.rho0 == pytest.approx(2.7e-6)
        assert m.rhor == pytest.approx(2.75e-6)
        assert m.e == pytest.approx(70000.0)
        assert m.nu == pytest.approx(0.33)
        assert m.ifunce == 501
        assert m.yr_fun == 501
        assert m.einf == pytest.approx(65000.0)
        assert m.efib == pytest.approx(65000.0)
        assert m.ce == pytest.approx(12.0)
        assert m.c == pytest.approx(12.0)
        assert m.r00 == pytest.approx(1.45)
        assert m.r0 == pytest.approx(1.45)
        assert m.r45 == pytest.approx(1.15)
        assert m.r90 == pytest.approx(1.75)
        assert m.chard == pytest.approx(0.40)
        assert m.fisokin == pytest.approx(0.40)
        assert m.c_hard == pytest.approx(0.40)
        assert m.iyield == 1
        assert m.eps_max == pytest.approx(0.28)
        assert m.eps == pytest.approx(0.28)
        assert m.epst1 == pytest.approx(0.10)
        assert m.eps_t == pytest.approx(0.10)
        assert m.epst2 == pytest.approx(0.22)
        assert m.eps_m == pytest.approx(0.22)
        assert m.fcut == pytest.approx(2000.0)
        assert m.asrate == pytest.approx(2000.0)
        assert m.fsmooth == 1
        assert m.israte == 1
        assert m.num_curves == 3
        assert len(m.curves) == 3
        for idx in range(3):
            assert m.curves[idx]["fct_id"] == curves[idx]["fct_id"]
            assert m.curves[idx]["fscale"] == pytest.approx(curves[idx]["fscale"])
            assert m.curves[idx]["eps_dot"] == pytest.approx(curves[idx]["eps_dot"])

    def test_writer_from_mat_law43_instance(self, tmp_path: Path):
        """Pass a MatLaw43 instance directly to StarterDeck.mat_law43."""
        orig = MatLaw43(
            id=10,
            title="Entity_Export",
            rho0=7.8e-6,
            rhor=7.8e-6,
            e=210000.0,
            nu=0.3,
            ifunce=401,
            einf=190000.0,
            ce=15.0,
            r00=1.2,
            r45=1.1,
            r90=1.3,
            chard=0.25,
            iyield=0,
            eps_max=0.3,
            epst1=0.08,
            epst2=0.18,
            fcut=1500.0,
            fsmooth=1,
            curves=[{"fct_id": 1, "fscale": 1.0, "eps_dot": 0.0}],
        )
        deck = StarterDeck("OBJ_TEST").mat_law43(orig)
        model, log = _parse_deck_string(tmp_path, deck.render())
        assert len(log.errors) == 0
        restored = model.mat_law43s[10]
        assert restored.e == pytest.approx(210000.0)
        assert restored.r00 == pytest.approx(1.2)
        assert restored.r45 == pytest.approx(1.1)
        assert restored.chard == pytest.approx(0.25)
        assert restored.fcut == pytest.approx(1500.0)


# ============================================================================
# 2. Free Format Audit (Space and Comma-Delimited)
# ============================================================================

class TestLaw43FreeFormatAudit:
    """Audit free format parsing for /MAT/LAW43."""

    def test_free_format_5card_space_delimited(self, tmp_path: Path):
        """Standard 5 cards space-delimited free format."""
        deck = """#RADIOSS STARTER
/BEGIN
Test Free 5 Card Space
                  10                   0
/MAT/LAW43/1
Free Space Steel
7.85e-6 7.85e-6
210000.0 0.28
501 195000.0 30.0
1.35 1.15 1.55 0.5 1
0.25 0.08 0.16 3000.0 1
101 1.0 0.0
102 1.2 100.0
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        m = model.mat_law43s[1]
        assert m.rho0 == pytest.approx(7.85e-6)
        assert m.e == pytest.approx(210000.0)
        assert m.nu == pytest.approx(0.28)
        assert m.ifunce == 501
        assert m.einf == pytest.approx(195000.0)
        assert m.ce == pytest.approx(30.0)
        assert m.r00 == pytest.approx(1.35)
        assert m.r45 == pytest.approx(1.15)
        assert m.r90 == pytest.approx(1.55)
        assert m.chard == pytest.approx(0.5)
        assert m.iyield == 1
        assert m.eps_max == pytest.approx(0.25)
        assert m.epst1 == pytest.approx(0.08)
        assert m.epst2 == pytest.approx(0.16)
        assert m.fcut == pytest.approx(3000.0)
        assert m.fsmooth == 1
        assert len(m.curves) == 2
        assert m.curves[0]["fct_id"] == 101
        assert m.curves[1]["fct_id"] == 102
        assert m.curves[1]["eps_dot"] == pytest.approx(100.0)

    def test_free_format_5card_comma_delimited(self, tmp_path: Path):
        """Standard 5 cards comma-delimited free format."""
        deck = """#RADIOSS STARTER
/BEGIN
Test Free 5 Card Comma
                  10                   0
/MAT/LAW43/2
Free Comma Steel
7.85e-6, 7.85e-6
210000.0, 0.30
501, 200000.0, 45.0
1.2, 1.1, 1.4, 0.2, 0
0.2, 0.05, 0.15, 2500.0, 1
101, 1.0, 0.0
102, 1.1, 50.0
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        m = model.mat_law43s[2]
        assert m.rho0 == pytest.approx(7.85e-6)
        assert m.e == pytest.approx(210000.0)
        assert m.nu == pytest.approx(0.30)
        assert m.ifunce == 501
        assert m.einf == pytest.approx(200000.0)
        assert m.ce == pytest.approx(45.0)
        assert m.r00 == pytest.approx(1.2)
        assert m.r45 == pytest.approx(1.1)
        assert m.r90 == pytest.approx(1.4)
        assert m.chard == pytest.approx(0.2)
        assert m.iyield == 0
        assert m.eps_max == pytest.approx(0.2)
        assert m.fcut == pytest.approx(2500.0)
        assert len(m.curves) == 2

    def test_free_format_legacy_4card(self, tmp_path: Path):
        """Legacy 4-card format (card 2 has [E, nu, ifunce, einf, ce])."""
        deck = """#RADIOSS STARTER
/BEGIN
Test Free Legacy 4 Card
                  10                   0
/MAT/LAW43/3
Legacy 4-Card Steel
7.8e-6, 0
210000.0, 0.3, 501, 20000.0, 0.4
1.1, 1.2, 1.3, 0.1, 1
0.2, 0.01, 0.02, 2, 1, 0.0
502, 1.0, 0.0
503, 1.0, 50.0
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        m = model.mat_law43s[3]
        assert m.e == pytest.approx(210000.0)
        assert m.nu == pytest.approx(0.3)
        assert m.ifunce == 501
        assert m.einf == pytest.approx(20000.0)
        assert m.ce == pytest.approx(0.4)
        assert m.r00 == pytest.approx(1.1)
        assert m.r45 == pytest.approx(1.2)
        assert m.r90 == pytest.approx(1.3)
        assert m.chard == pytest.approx(0.1)
        assert m.iyield == 1
        assert len(m.curves) == 2
        assert m.curves[0]["funct_id"] == 502
        assert m.curves[1]["eps_dot"] == pytest.approx(50.0)


# ============================================================================
# 3. Keyword Synonyms Verification
# ============================================================================

class TestLaw43KeywordSynonyms:
    """Verify /MAT/LAW43, /MAT/HILL_TAB, /MAT/HILL_PLAS_TAB, /MAT/LAW43_HILL_TAB parse identically."""

    @pytest.mark.parametrize("kw", [
        "/MAT/LAW43",
        "/MAT/HILL_TAB",
        "/MAT/HILL_PLAS_TAB",
        "/MAT/LAW43_HILL_TAB",
    ])
    def test_synonyms_parse_identically(self, tmp_path: Path, kw: str):
        """All keyword synonyms produce identical MatLaw43 and Material definitions."""
        deck_text = f"""#RADIOSS STARTER
/BEGIN
Test Synonym {kw}
                  10                   0
{kw}/43
Synonym Test Part
7.85e-9             7.85e-9
210000.0            0.30
       501          180000.0            15.0
1.45                1.25                1.65                0.35         1
0.30                0.10                0.20                1000.0       1
       101          1.0                 0.0
       102          1.15                50.0
"""
        model, log = _parse_deck_string(tmp_path, deck_text)
        assert len(log.errors) == 0
        assert 43 in model.mat_law43s

        mat = model.mat_law43s[43]
        assert isinstance(mat, MatLaw43)
        assert mat.rho0 == pytest.approx(7.85e-9)
        assert mat.e == pytest.approx(210000.0)
        assert mat.nu == pytest.approx(0.30)
        assert mat.ifunce == 501
        assert mat.einf == pytest.approx(180000.0)
        assert mat.ce == pytest.approx(15.0)
        assert mat.r00 == pytest.approx(1.45)
        assert mat.r45 == pytest.approx(1.25)
        assert mat.r90 == pytest.approx(1.65)
        assert mat.chard == pytest.approx(0.35)
        assert mat.fisokin == pytest.approx(0.35)
        assert mat.iyield == 1
        assert mat.eps_max == pytest.approx(0.30)
        assert mat.epst1 == pytest.approx(0.10)
        assert mat.epst2 == pytest.approx(0.20)
        assert mat.fcut == pytest.approx(1000.0)
        assert mat.fsmooth == 1
        assert len(mat.curves) == 2

        # Verify built material in model.materials
        assert 43 in model.materials
        m_phys = model.materials[43]
        assert m_phys.id == 43
        assert m_phys.law == 43
        assert m_phys.params["E0"] == pytest.approx(210000.0)
        assert m_phys.params["R00"] == pytest.approx(1.45)
        assert m_phys.params["Iyield"] == 1

        # Verify model dictionary aliases
        assert model.mat_hill_tabs is model.mat_law43s
        assert model.mat_hill_plas_tabs is model.mat_law43s
        assert model.mat_law43_hill_tabs is model.mat_law43s

    def test_deck_writer_synonym_methods(self):
        """Verify StarterDeck methods mat_law43, mat_hill_tab, mat_hill_plas_tab, mat_law43_hill_tab."""
        curves = [{"fct_id": 1, "fscale": 1.0, "eps_dot": 0.0}]
        d1 = StarterDeck("D1").mat_law43(mid=1, rho=7.8e-6, e=2.1e5, nu=0.3, curves=curves).render()
        d2 = StarterDeck("D2").mat_hill_tab(mid=1, rho=7.8e-6, e=2.1e5, nu=0.3, curves=curves).render()
        d3 = StarterDeck("D3").mat_hill_plas_tab(mid=1, rho=7.8e-6, e=2.1e5, nu=0.3, curves=curves).render()
        d4 = StarterDeck("D4").mat_law43_hill_tab(mid=1, rho=7.8e-6, e=2.1e5, nu=0.3, curves=curves).render()

        assert "/MAT/LAW43/1" in d1
        assert "/MAT/HILL_TAB/1" in d2
        assert "/MAT/HILL_PLAS_TAB/1" in d3
        assert "/MAT/LAW43_HILL_TAB/1" in d4

        cards1 = _get_mat_cards(d1)
        cards2 = _get_mat_cards(d2)
        cards3 = _get_mat_cards(d3)
        cards4 = _get_mat_cards(d4)
        assert cards1 == cards2 == cards3 == cards4


# ============================================================================
# 4. Starter Checks & Element Compatibility
# ============================================================================

class TestLaw43StarterChecksAndCompatibility:
    """Audit element family compatibility and parameter bounds checks for /MAT/LAW43."""

    def test_allowed_element_families_pass(self):
        """Shells (shells, shells_qbat, shells_qeph, sh3n), quads, and solids (bricks, tetras, penta6, pyra5) pass validation."""
        allowed_families = (
            "shells",
            "shells_qbat",
            "shells_qeph",
            "sh3n",
            "quads",
            "bricks",
            "tetras",
            "penta6",
            "pyra5",
        )

        for fam in allowed_families:
            assert 43 in _ALLOWED_LAWS[fam]
            assert "43" in _ALLOWED_LAWS[fam]
            assert "LAW43" in _ALLOWED_LAWS[fam]
            assert "HILL_TAB" in _ALLOWED_LAWS[fam]

            model = Model()
            model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
            mat = Material(
                id=1,
                law=43,
                rho0=7.85e-9,
                params={
                    "E0": 210000.0,
                    "nu": 0.3,
                    "R00": 1.0,
                    "R45": 1.0,
                    "R90": 1.0,
                    "FISOKIN": 0.0,
                    "curves": [{"fct_id": 1, "fscale": 1.0, "eps_dot": 0.0}],
                },
            )
            model.materials[1] = mat

            class FakeGroup:
                def __init__(self):
                    self.state = {"slices": [(slice(0, 1), mat, None)]}

            model.element_groups = lambda f=fam: [(f, FakeGroup())]
            log = MessageLog()
            check_model(model, log)
            assert len(log.errors) == 0, f"Unexpected error for allowed family {fam}: {log.errors}"

    def test_disallowed_element_families_rejected(self):
        """Beams (BEAM), trusses (TRUSS), and springs (SPRING) are strictly rejected with error messages."""
        disallowed = ("beams", "trusses", "springs")
        mat = Material(
            id=1,
            law=43,
            rho0=7.85e-9,
            params={
                "E0": 210000.0,
                "nu": 0.3,
                "R00": 1.0,
                "R45": 1.0,
                "R90": 1.0,
                "FISOKIN": 0.0,
                "curves": [{"fct_id": 1, "fscale": 1.0, "eps_dot": 0.0}],
            },
        )

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        for fam in disallowed:
            model = Model()
            model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
            model.materials[1] = mat
            model.element_groups = lambda f=fam: [(f, FakeGroup())]
            log = MessageLog()
            check_model(model, log)
            assert any(
                "/MAT/LAW43/1 (/MAT/HILL_TAB) is not supported for" in e and fam in e
                for e in log.errors
            ), f"Disallowed family {fam} was not rejected properly: {log.errors}"

    def test_bounds_rho_nonpositive(self):
        """rho0 <= 0 rejected with error."""
        for bad_rho in (0.0, -7.8e-6):
            mat = Material(
                id=1,
                law=43,
                rho0=bad_rho,
                params={
                    "E0": 210000.0,
                    "nu": 0.3,
                    "R00": 1.0,
                    "R45": 1.0,
                    "R90": 1.0,
                    "curves": [{"fct_id": 1, "fscale": 1.0, "eps_dot": 0.0}],
                },
            )
            log = MessageLog()
            check_mat_law43(mat, log)
            assert any("initial density RHO must be > 0" in e for e in log.errors)

    def test_bounds_e_nonpositive(self):
        """E <= 0 rejected with error."""
        for bad_e in (0.0, -1000.0):
            mat = Material(
                id=1,
                law=43,
                rho0=7.8e-6,
                params={
                    "E0": bad_e,
                    "nu": 0.3,
                    "R00": 1.0,
                    "R45": 1.0,
                    "R90": 1.0,
                    "curves": [{"fct_id": 1, "fscale": 1.0, "eps_dot": 0.0}],
                },
            )
            log = MessageLog()
            check_mat_law43(mat, log)
            assert any("Young's modulus E must be > 0" in e for e in log.errors)

    def test_bounds_nu_invalid(self):
        """nu < 0 or nu >= 0.5 rejected with error."""
        for bad_nu in (-0.1, 0.5, 0.55):
            mat = Material(
                id=1,
                law=43,
                rho0=7.8e-6,
                params={
                    "E0": 210000.0,
                    "nu": bad_nu,
                    "R00": 1.0,
                    "R45": 1.0,
                    "R90": 1.0,
                    "curves": [{"fct_id": 1, "fscale": 1.0, "eps_dot": 0.0}],
                },
            )
            log = MessageLog()
            check_mat_law43(mat, log)
            assert any("Poisson's ratio NU must satisfy 0 <= NU < 0.5" in e for e in log.errors)

    def test_bounds_lankford_nonpositive(self):
        """R00 <= 0, R45 <= 0, R90 <= 0 rejected with error."""
        for r_key, msg_key in [("r00", "R00"), ("r45", "R45"), ("r90", "R90")]:
            for bad_r in (0.0, -0.5):
                p = {
                    "E0": 210000.0,
                    "nu": 0.3,
                    "r00": 1.0,
                    "r45": 1.0,
                    "r90": 1.0,
                    "curves": [{"fct_id": 1, "fscale": 1.0, "eps_dot": 0.0}],
                }
                p[r_key] = bad_r
                mat = Material(id=1, law=43, rho0=7.8e-6, params=p)
                log = MessageLog()
                check_mat_law43(mat, log)
                assert any(f"Lankford coefficient {msg_key} must be > 0" in e for e in log.errors)

    def test_bounds_fisokin_out_of_range(self):
        """FISOKIN < 0 or FISOKIN > 1 rejected with ANCMSG 913."""
        for bad_fisokin in (-0.1, 1.2):
            mat = Material(
                id=1,
                law=43,
                rho0=7.8e-6,
                params={
                    "E0": 210000.0,
                    "nu": 0.3,
                    "R00": 1.0,
                    "R45": 1.0,
                    "R90": 1.0,
                    "fisokin": bad_fisokin,
                    "curves": [{"fct_id": 1, "fscale": 1.0, "eps_dot": 0.0}],
                },
            )
            log = MessageLog()
            check_mat_law43(mat, log)
            assert any(
                "iso-kinematic hardening factor FISOKIN must be in [0, 1]" in e and "ANCMSG 913" in e
                for e in log.errors
            )

    def test_bounds_zero_curves_ancmsg_366(self):
        """Zero plasticity curves defined rejected with ANCMSG 366."""
        mat_empty = Material(
            id=1,
            law=43,
            rho0=7.8e-6,
            params={
                "E0": 210000.0,
                "nu": 0.3,
                "R00": 1.0,
                "R45": 1.0,
                "R90": 1.0,
                "fisokin": 0.0,
                "curves": [],
            },
        )
        log = MessageLog()
        check_mat_law43(mat_empty, log)
        assert any("no plasticity hardening curve defined" in e and "ANCMSG 366" in e for e in log.errors)


# ============================================================================
# 5. Restart (.rst) Pickling & State Serialization Audit
# ============================================================================

class TestLaw43RestartAndStateSerialization:
    """Verify state pickling and unpickling fidelity for Material, MatLaw43, and extra_shapes arrays."""

    def test_extra_shapes_pickling(self):
        """extra_shapes dictionary for nip=None and nip=5 survives pickling and unpickling."""
        # Unlayered / solid formulation (nip=None)
        s_solid = extra_shapes(None)
        raw_solid = pickle.dumps(s_solid)
        res_solid = pickle.loads(raw_solid)
        assert res_solid == s_solid
        assert res_solid["pla43"] == ()
        assert res_solid["uvar43"] == (4,)
        assert res_solid["off43"] == ()
        assert res_solid["edot43"] == ()
        assert res_solid["thk43"] == ()

        # Layered shell formulation (nip=5 integration points)
        s_shell = extra_shapes(None, nip=5)
        raw_shell = pickle.dumps(s_shell)
        res_shell = pickle.loads(raw_shell)
        assert res_shell == s_shell
        assert res_shell["pla43"] == (5,)
        assert res_shell["uvar43"] == (5, 4)
        assert res_shell["off43"] == (5,)
        assert res_shell["edot43"] == (5,)
        assert res_shell["thk43"] == (5,)

    def test_mat_law43_entity_pickling(self):
        """MatLaw43 entity dataclass survives pickling and unpickling."""
        orig = MatLaw43(
            id=43,
            title="Entity_Pickle_Test",
            rho0=7.85e-9,
            rhor=7.85e-9,
            e=210000.0,
            nu=0.30,
            ifunce=501,
            einf=180000.0,
            ce=25.0,
            r00=1.45,
            r45=1.25,
            r90=1.65,
            chard=0.35,
            fisokin=0.35,
            iyield=1,
            eps_max=0.30,
            epst1=0.10,
            epst2=0.20,
            fcut=1000.0,
            asrate=1000.0,
            fsmooth=1,
            israte=1,
            curves=[
                {"fct_id": 101, "fscale": 1.0, "eps_dot": 0.0},
                {"fct_id": 102, "fscale": 1.15, "eps_dot": 50.0},
            ],
        )

        raw = pickle.dumps(orig)
        restored = pickle.loads(raw)

        assert restored.id == orig.id
        assert restored.title == orig.title
        assert restored.rho0 == pytest.approx(orig.rho0)
        assert restored.e == pytest.approx(orig.e)
        assert restored.nu == pytest.approx(orig.nu)
        assert restored.r00 == pytest.approx(orig.r00)
        assert restored.chard == pytest.approx(orig.chard)
        assert restored.fisokin == pytest.approx(orig.fisokin)
        assert restored.num_curves == 2
        assert len(restored.curves) == 2
        assert restored.curves[0]["fct_id"] == 101
        assert restored.curves[1]["eps_dot"] == pytest.approx(50.0)

    def test_material_and_built_material_pickling(self):
        """Material instance built via build_law43 can be pickled and unpickled without corruption."""
        orig = MatLaw43(
            id=43,
            rho0=7.85e-9,
            e=210000.0,
            nu=0.30,
            r00=1.5,
            r45=1.2,
            r90=1.8,
            chard=0.4,
            curves=[{"fct_id": 1, "points": ([0.0, 0.05, 0.1], [300.0, 450.0, 550.0])}],
        )
        mat = build_law43(orig)

        raw = pickle.dumps(mat)
        restored = pickle.loads(raw)

        assert restored.id == mat.id
        assert restored.law == 43
        assert restored.rho0 == pytest.approx(mat.rho0)
        assert restored.params["A01"] == pytest.approx(mat.params["A01"])
        assert restored.params["A02"] == pytest.approx(mat.params["A02"])
        assert restored.params["A03"] == pytest.approx(mat.params["A03"])
        assert restored.params["A12"] == pytest.approx(mat.params["A12"])
        assert sound_speed(restored) == pytest.approx(sound_speed(mat), rel=1e-12)
        assert restored.sound_speed_solid() == pytest.approx(mat.sound_speed_solid(), rel=1e-12)
        assert restored.sound_speed_shell() == pytest.approx(mat.sound_speed_shell(), rel=1e-12)

    def test_element_state_arrays_pickling(self):
        """Element persistent state arrays (pla43, uvar43, off43, edot43, thk43) survive restart pickling."""
        n_elem = 24
        rng = np.random.default_rng(12345)
        state = {
            "pla43": rng.uniform(0.0, 0.25, size=n_elem),
            "uvar43": rng.uniform(-100.0, 100.0, size=(n_elem, 4)),
            "off43": np.ones(n_elem, dtype=float),
            "edot43": rng.uniform(0.0, 500.0, size=n_elem),
            "thk43": rng.uniform(0.95, 1.05, size=n_elem),
        }
        # Simulate failure deletion on some elements
        state["off43"][3] = 0.0
        state["off43"][11] = 0.0

        raw = pickle.dumps(state)
        restored = pickle.loads(raw)

        for k in state:
            assert k in restored
            np.testing.assert_allclose(restored[k], state[k], err_msg=f"State array {k} mismatch")

    def test_model_pickling_with_law43(self):
        """A complete Model with LAW43 materials and aliases pickles and unpickles cleanly."""
        model = Model()
        model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
        mat_entity = MatLaw43(
            id=43,
            rho0=7.85e-9,
            e=210000.0,
            nu=0.3,
            r00=1.4,
            r45=1.2,
            r90=1.6,
            curves=[{"fct_id": 1, "points": ([0.0, 0.1], [300.0, 500.0])}],
        )
        model.mat_law43s[43] = mat_entity
        model.materials[43] = build_law43(mat_entity)

        # Verify aliases before pickling
        assert model.mat_hill_tabs is model.mat_law43s
        assert model.mat_hill_plas_tabs is model.mat_law43s
        assert model.mat_law43_hill_tabs is model.mat_law43s

        raw = pickle.dumps(model)
        restored = pickle.loads(raw)

        assert 43 in restored.mat_law43s
        assert 43 in restored.materials
        assert restored.mat_hill_tabs is restored.mat_law43s
        assert restored.mat_hill_plas_tabs is restored.mat_law43s
        assert restored.mat_law43_hill_tabs is restored.mat_law43s
        assert sound_speed(restored.materials[43]) == pytest.approx(sound_speed(model.materials[43]))
        assert restored.materials[43].sound_speed_solid() == pytest.approx(model.materials[43].sound_speed_solid())
