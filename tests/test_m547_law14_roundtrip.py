"""Milestone M547: /MAT/LAW14 (/MAT/COMPSO, /MAT/COMP_SOL) Roundtrip & Integration Auditor.

Audits:
  1. Card Format Roundtrip:
     - Strict fixed format (20-character fields) per matl14_compso.cfg (radioss2020)
       and card_layouts.py (MAT_LAW14_1..9, MAT_COMPSO_1..9, MAT_COMP_SOL_1..9).
     - Card-by-card audit:
         Card 1: RHO_I, Refer_Rho (%20lg%20lg)
         Card 2: E11, E22, E33 (%20lg%20lg%20lg)
         Card 3: NU12, NU23, NU31 (%20lg%20lg%20lg)
         Card 4: G12, G23, G31 (%20lg%20lg%20lg)
         Card 5: SIGT1, SIGT2, SIGT3, DELTA (%20lg%20lg%20lg%20lg)
         Card 6: CB, CN, FMAX, WPLAREF (%20lg%20lg%20lg%20lg)
         Card 7: SIGYT1, SIGYT2, SIGYC1, SIGYC2 (%20lg%20lg%20lg%20lg)
         Card 8: SIGYT12, SIGYC12, SIGYT23, SIGYC23 (%20lg%20lg%20lg%20lg)
         Card 9: ALPHA, EFIB, CC, EPS0, STRFLAG (%20lg%20lg%20lg%20lg%10d)
     - Free format reading (comma-separated and space-separated).
     - StarterDeck.mat_law14, mat_compso, and mat_comp_sol writer methods with keywords and MatLaw14 instances.
     - Full roundtrip: generate deck string -> parse with read_mat_compso / read_mat_law14 -> verify exact parameter match.

  2. Dual-Personality Disambiguation:
     - /MAT/LAW14 with <= 2 cards routed to read_mat_cam_clay (Cam-Clay geotechnical model).
     - /MAT/LAW14 with > 2 cards (3..9 cards) routed to read_mat_compso (composite solid model).
     - /MAT/COMPSO and /MAT/COMP_SOL always routed to read_mat_compso even if <= 2 cards.

  3. Starter Checks & Element Compatibility:
     - 3D solids (bricks, tetras, penta6, pyra5) pass validation cleanly.
     - Non-solids (shells, shells_qbat, shells_qeph, sh3n, beams, trusses, springs, quads) are rejected with error messages.
     - 2D solids / axisymmetric models (N2D > 0) rejected per hm_read_mat14.F:151 (ANCMSG 305).
     - Parameter checks:
         - rho0 <= 0 rejected.
         - E11 <= 0, E22 <= 0, E33 <= 0 rejected with ANCMSG 306.
         - Compliance matrix determinant DETC <= 0 rejected with ANCMSG 307.
         - Shear moduli G12 < 0, G23 < 0, G31 < 0 rejected.

  4. State Serialization (Restart / Pickling):
     - extra_shapes dictionary for nip=1 and nip=4 pickled and unpickled without data corruption.
     - Material object built by build_law14 pickled and unpickled with exact attributes and sound_speed_solid().
     - Element persistent state arrays (dam14, epe14, epc14, wpla14, off14, epsf14, sigf14, tsaiwu14)
       pickled and unpickled with assert_allclose.
     - Model object containing MatLaw14 materials and element groups pickled and restored.
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
from pyradioss.materials import law14_compso
from pyradioss.materials.law14_compso import (
    build_law14,
    extra_shapes,
    solid_update,
    sound_speed,
)
from pyradioss.model.entities import Material, MatLaw14, MatCompso, MatCompSol, MatCamClay
from pyradioss.model.model import Model
from pyradioss.starter.checks import _ALLOWED_LAWS, check_mat_law14, check_model


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
# 1. Nine Cards Strict Fixed Format & Card Layout Audit (matl14_compso.cfg)
# ============================================================================

class TestLaw14CardsFixedFormatAudit:
    """Audit of all 9 cards in strict fixed format according to matl14_compso.cfg."""

    def test_card_layout_constants_and_aliases(self):
        """Verify layout tuples and aliases in card_layouts.py."""
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
            val14 = getattr(cl, f"MAT_LAW14_{i}")
            assert getattr(cl, f"MAT_COMPSO_{i}") == val14
            assert getattr(cl, f"MAT_COMP_SOL_{i}") == val14
            assert cl.LAYOUTS[f"MAT_LAW14_{i}"] == list(val14)
            assert cl.LAYOUTS[f"MAT_COMPSO_{i}"] == list(val14)
            assert cl.LAYOUTS[f"MAT_COMP_SOL_{i}"] == list(val14)

    def test_card1_rho_refer_rho_roundtrip(self, tmp_path: Path):
        """Card 1: RHO_I, Refer_Rho (%20lg%20lg). Tests dual density and single density."""
        # Dual density
        d1 = StarterDeck("CARD1_DUAL").mat_law14(
            mid=1, rho=1.55e-9, refer_rho=1.60e-9,
            ea=140000.0, eb=10000.0, ec=10000.0,
            prab=0.3, prbc=0.4, prca=0.02,
            gab=5000.0, gbc=3500.0, gca=5000.0,
        )
        cards1 = _get_mat_cards(d1.render())
        card1 = cards1[0]
        assert len(card1) == 40
        f1 = cl.split_fixed(card1, cl.MAT_LAW14_1)
        assert float(f1[0]) == pytest.approx(1.55e-9)
        assert float(f1[1]) == pytest.approx(1.60e-9)

        model1, log1 = _parse_deck_string(tmp_path, d1.render())
        assert len(log1.errors) == 0
        mat1 = model1.mat_law14s[1]
        assert mat1.rho0 == pytest.approx(1.55e-9)
        assert mat1.rhor == pytest.approx(1.60e-9)

        # Single density (refer_rho defaults to rho0)
        d2 = StarterDeck("CARD1_SINGLE").mat_law14(
            mid=2, rho=1.75e-9,
            ea=120000.0, eb=12000.0, ec=12000.0,
            prab=0.28, prbc=0.35, prca=0.02,
            gab=6000.0, gbc=4000.0, gca=6000.0,
        )
        cards2 = _get_mat_cards(d2.render())
        card1_single = cards2[0]
        assert len(card1_single) == 20
        model2, log2 = _parse_deck_string(tmp_path, d2.render())
        assert len(log2.errors) == 0
        mat2 = model2.mat_law14s[2]
        assert mat2.rho0 == pytest.approx(1.75e-9)
        assert mat2.rhor == pytest.approx(1.75e-9)

    def test_card2_elastic_moduli_roundtrip(self, tmp_path: Path):
        """Card 2: E11, E22, E33 (%20lg%20lg%20lg). Width = 60."""
        d = StarterDeck("CARD2_TEST").mat_law14(
            mid=1, rho=1.5e-9, ea=145000.0, eb=11200.0, ec=9800.0,
            prab=0.3, prbc=0.4, prca=0.02, gab=5000.0, gbc=3500.0, gca=5000.0,
        )
        cards = _get_mat_cards(d.render())
        card2 = cards[1]
        assert len(card2) == 60
        f2 = cl.split_fixed(card2, cl.MAT_LAW14_2)
        assert [float(x) for x in f2] == pytest.approx([145000.0, 11200.0, 9800.0])

        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        mat = model.mat_law14s[1]
        assert mat.ea == pytest.approx(145000.0)
        assert mat.eb == pytest.approx(11200.0)
        assert mat.ec == pytest.approx(9800.0)

    def test_card3_poisson_ratios_roundtrip(self, tmp_path: Path):
        """Card 3: NU12, NU23, NU31 (%20lg%20lg%20lg). Width = 60."""
        d = StarterDeck("CARD3_TEST").mat_law14(
            mid=1, rho=1.5e-9, ea=100000.0, eb=10000.0, ec=10000.0,
            prab=0.29, prbc=0.42, prca=0.025, gab=5000.0, gbc=3500.0, gca=5000.0,
        )
        cards = _get_mat_cards(d.render())
        card3 = cards[2]
        assert len(card3) == 60
        f3 = cl.split_fixed(card3, cl.MAT_LAW14_3)
        assert [float(x) for x in f3] == pytest.approx([0.29, 0.42, 0.025])

        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        mat = model.mat_law14s[1]
        assert mat.prab == pytest.approx(0.29)
        assert mat.prbc == pytest.approx(0.42)
        assert mat.prca == pytest.approx(0.025)

    def test_card4_shear_moduli_roundtrip(self, tmp_path: Path):
        """Card 4: G12, G23, G31 (%20lg%20lg%20lg). Width = 60."""
        d = StarterDeck("CARD4_TEST").mat_law14(
            mid=1, rho=1.5e-9, ea=100000.0, eb=10000.0, ec=10000.0,
            prab=0.3, prbc=0.4, prca=0.02, gab=5400.0, gbc=3600.0, gca=5200.0,
        )
        cards = _get_mat_cards(d.render())
        card4 = cards[3]
        assert len(card4) == 60
        f4 = cl.split_fixed(card4, cl.MAT_LAW14_4)
        assert [float(x) for x in f4] == pytest.approx([5400.0, 3600.0, 5200.0])

        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        mat = model.mat_law14s[1]
        assert mat.gab == pytest.approx(5400.0)
        assert mat.gbc == pytest.approx(3600.0)
        assert mat.gca == pytest.approx(5200.0)

    def test_card5_tensile_strengths_damage_roundtrip(self, tmp_path: Path):
        """Card 5: SIGT1, SIGT2, SIGT3, DELTA (%20lg%20lg%20lg%20lg). Width = 80."""
        d = StarterDeck("CARD5_TEST").mat_law14(
            mid=1, rho=1.5e-9, ea=100000.0, eb=10000.0, ec=10000.0,
            prab=0.3, prbc=0.4, prca=0.02, gab=5000.0, gbc=3500.0, gca=5000.0,
            sigt1=1850.0, sigt2=45.0, sigt3=42.0, delta=0.075,
        )
        cards = _get_mat_cards(d.render())
        card5 = cards[4]
        assert len(card5) == 80
        f5 = cl.split_fixed(card5, cl.MAT_LAW14_5)
        assert [float(x) for x in f5] == pytest.approx([1850.0, 45.0, 42.0, 0.075])

        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        mat = model.mat_law14s[1]
        assert mat.sigt1 == pytest.approx(1850.0)
        assert mat.sigt2 == pytest.approx(45.0)
        assert mat.sigt3 == pytest.approx(42.0)
        assert mat.delta == pytest.approx(0.075)

    def test_card6_hardening_roundtrip(self, tmp_path: Path):
        """Card 6: CB, CN, FMAX, WPLAREF (%20lg%20lg%20lg%20lg). Width = 80."""
        d = StarterDeck("CARD6_TEST").mat_law14(
            mid=1, rho=1.5e-9, ea=100000.0, eb=10000.0, ec=10000.0,
            prab=0.3, prbc=0.4, prca=0.02, gab=5000.0, gbc=3500.0, gca=5000.0,
            cb=140.0, cn=0.55, fmax=1250.0, wplaref=1.2,
        )
        cards = _get_mat_cards(d.render())
        card6 = cards[5]
        assert len(card6) == 80
        f6 = cl.split_fixed(card6, cl.MAT_LAW14_6)
        assert [float(x) for x in f6] == pytest.approx([140.0, 0.55, 1250.0, 1.2])

        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        mat = model.mat_law14s[1]
        assert mat.cb == pytest.approx(140.0)
        assert mat.cn == pytest.approx(0.55)
        assert mat.fmax == pytest.approx(1250.0)
        assert mat.wplaref == pytest.approx(1.2)

    def test_card7_yield_stresses_normal_roundtrip(self, tmp_path: Path):
        """Card 7: SIGYT1, SIGYT2, SIGYC1, SIGYC2 (%20lg%20lg%20lg%20lg). Width = 80."""
        d = StarterDeck("CARD7_TEST").mat_law14(
            mid=1, rho=1.5e-9, ea=100000.0, eb=10000.0, ec=10000.0,
            prab=0.3, prbc=0.4, prca=0.02, gab=5000.0, gbc=3500.0, gca=5000.0,
            sigyt1=2050.0, sigyt2=55.0, sigyc1=1550.0, sigyc2=210.0,
        )
        cards = _get_mat_cards(d.render())
        card7 = cards[6]
        assert len(card7) == 80
        f7 = cl.split_fixed(card7, cl.MAT_LAW14_7)
        assert [float(x) for x in f7] == pytest.approx([2050.0, 55.0, 1550.0, 210.0])

        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        mat = model.mat_law14s[1]
        assert mat.sigyt1 == pytest.approx(2050.0)
        assert mat.sigyt2 == pytest.approx(55.0)
        assert mat.sigyc1 == pytest.approx(1550.0)
        assert mat.sigyc2 == pytest.approx(210.0)

    def test_card8_yield_stresses_shear_roundtrip(self, tmp_path: Path):
        """Card 8: SIGYT12, SIGYC12, SIGYT23, SIGYC23 (%20lg%20lg%20lg%20lg). Width = 80."""
        d = StarterDeck("CARD8_TEST").mat_law14(
            mid=1, rho=1.5e-9, ea=100000.0, eb=10000.0, ec=10000.0,
            prab=0.3, prbc=0.4, prca=0.02, gab=5000.0, gbc=3500.0, gca=5000.0,
            sigyt12=85.0, sigyc12=82.0, sigyt23=52.0, sigyc23=51.0,
        )
        cards = _get_mat_cards(d.render())
        card8 = cards[7]
        assert len(card8) == 80
        f8 = cl.split_fixed(card8, cl.MAT_LAW14_8)
        assert [float(x) for x in f8] == pytest.approx([85.0, 82.0, 52.0, 51.0])

        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        mat = model.mat_law14s[1]
        assert mat.sigt12 == pytest.approx(85.0)
        assert mat.sigc12 == pytest.approx(82.0)
        assert mat.sigt23 == pytest.approx(52.0)
        assert mat.sigc23 == pytest.approx(51.0)

    def test_card9_fiber_strainrate_roundtrip(self, tmp_path: Path):
        """Card 9: ALPHA, EFIB, CC, EPS0, STRFLAG (%20lg%20lg%20lg%20lg%10d). Width = 90."""
        d = StarterDeck("CARD9_TEST").mat_law14(
            mid=1, rho=1.5e-9, ea=100000.0, eb=10000.0, ec=10000.0,
            prab=0.3, prbc=0.4, prca=0.02, gab=5000.0, gbc=3500.0, gca=5000.0,
            alpha=0.35, efib=195000.0, cc=0.035, eps0=1.5, strflag=2,
        )
        cards = _get_mat_cards(d.render())
        card9 = cards[8]
        assert len(card9) == 90
        f9 = cl.split_fixed(card9, cl.MAT_LAW14_9)
        assert float(f9[0]) == pytest.approx(0.35)
        assert float(f9[1]) == pytest.approx(195000.0)
        assert float(f9[2]) == pytest.approx(0.035)
        assert float(f9[3]) == pytest.approx(1.5)
        assert int(f9[4]) == 2

        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        mat = model.mat_law14s[1]
        assert mat.alpha == pytest.approx(0.35)
        assert mat.efib == pytest.approx(195000.0)
        assert mat.cc == pytest.approx(0.035)
        assert mat.eps0 == pytest.approx(1.5)
        assert mat.strflag == 2


# ============================================================================
# 2. Free Format Reading (Comma- & Space-Delimited)
# ============================================================================

class TestLaw14FreeFormatAudit:
    """Audit of free-format deck reading (comma- and whitespace-delimited)."""

    def test_free_format_comma_delimited(self, tmp_path: Path):
        """Read 9 cards separated by commas."""
        deck = """
/BEGIN
FREE_COMMA_TEST
-1
/MAT/LAW14/10
T300_914_Composite
1.55e-9, 1.55e-9
138000.0, 9500.0, 9500.0
0.32, 0.45, 0.02
4800.0, 3200.0, 4800.0
1500.0, 40.0, 40.0, 0.08
120.0, 0.6, 1100.0, 1.0
1800.0, 45.0, 1400.0, 190.0
75.0, 75.0, 48.0, 48.0
0.28, 175000.0, 0.03, 1.0, 1
/END
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 10 in model.mat_law14s
        mat = model.mat_law14s[10]
        assert mat.rho0 == pytest.approx(1.55e-9)
        assert mat.ea == pytest.approx(138000.0)
        assert mat.eb == pytest.approx(9500.0)
        assert mat.prab == pytest.approx(0.32)
        assert mat.gab == pytest.approx(4800.0)
        assert mat.sigt1 == pytest.approx(1500.0)
        assert mat.delta == pytest.approx(0.08)
        assert mat.cb == pytest.approx(120.0)
        assert mat.cn == pytest.approx(0.6)
        assert mat.fmax == pytest.approx(1100.0)
        assert mat.sigyt1 == pytest.approx(1800.0)
        assert mat.sigt12 == pytest.approx(75.0)
        assert mat.alpha == pytest.approx(0.28)
        assert mat.efib == pytest.approx(175000.0)
        assert mat.cc == pytest.approx(0.03)
        assert mat.strflag == 1

        # Built material verification
        assert 10 in model.materials
        m = model.materials[10]
        assert m.law == 14
        assert m.params["D11"] > 0.0
        assert m.params["SSP"] > 0.0

    def test_free_format_space_delimited(self, tmp_path: Path):
        """Read 9 cards separated by variable whitespace."""
        deck = """
/BEGIN
FREE_SPACE_TEST
-1
/MAT/COMPSO/20
Space_Delimited_Material
1.6e-9   1.6e-9
125000.0   11000.0   11000.0
0.28   0.38   0.025
5200.0   3400.0   5200.0
1600.0   45.0   45.0   0.06
110.0   0.5   1050.0   1.0
1700.0   50.0   1300.0   175.0
70.0   70.0   45.0   45.0
0.3   165000.0   0.025   1.0   2
/END
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        assert 20 in model.mat_law14s
        mat = model.mat_law14s[20]
        assert mat.rho0 == pytest.approx(1.6e-9)
        assert mat.ea == pytest.approx(125000.0)
        assert mat.strflag == 2
        assert model.materials[20].law == 14


# ============================================================================
# 3. StarterDeck Emitters & Direct MatLaw14 Instance Roundtrip
# ============================================================================

class TestLaw14DeckWriterSynonyms:
    """StarterDeck emitter audit for mat_law14, mat_compso, and mat_comp_sol."""

    def test_writer_synonyms_headers(self):
        """Verify headers generated by StarterDeck.mat_law14, mat_compso, mat_comp_sol."""
        d1 = StarterDeck("DECK1").mat_law14(mid=1, rho=1.5e-9, ea=1e5, eb=1e4, ec=1e4, prab=0.3, prbc=0.3, prca=0.03, gab=5e3, gbc=3e3, gca=5e3)
        assert "/MAT/LAW14/1" in d1.render()

        d2 = StarterDeck("DECK2").mat_compso(mid=2, rho=1.5e-9, ea=1e5, eb=1e4, ec=1e4, prab=0.3, prbc=0.3, prca=0.03, gab=5e3, gbc=3e3, gca=5e3)
        assert "/MAT/COMPSO/2" in d2.render()

        d3 = StarterDeck("DECK3").mat_comp_sol(mid=3, rho=1.5e-9, ea=1e5, eb=1e4, ec=1e4, prab=0.3, prbc=0.3, prca=0.03, gab=5e3, gbc=3e3, gca=5e3)
        assert "/MAT/COMP_SOL/3" in d3.render()

    def test_writer_from_matlaw14_instance_roundtrip(self, tmp_path: Path):
        """Pass a MatLaw14 instance directly to StarterDeck methods."""
        mat_orig = MatLaw14(
            id=42, title="Direct_Instance_Mat",
            rho0=1.52e-9, rhor=1.55e-9,
            ea=142000.0, eb=9800.0, ec=9800.0,
            prab=0.31, prbc=0.44, prca=0.021,
            gab=4900.0, gbc=3300.0, gca=4900.0,
            sigt1=1750.0, sigt2=42.0, sigt3=42.0, delta=0.065,
            cb=135.0, cn=0.52, fmax=1150.0, wplaref=1.1,
            sigyt1=1950.0, sigyt2=48.0, sigyc1=1450.0, sigyc2=195.0,
            sigt12=78.0, sigc12=78.0, sigt23=49.0, sigc23=49.0,
            alpha=0.26, efib=182000.0, cc=0.038, eps0=1.2, strflag=1,
        )

        d = StarterDeck("INSTANCE_ROUNDTRIP").mat_law14(mat_orig)
        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        assert 42 in model.mat_law14s
        mat_parsed = model.mat_law14s[42]

        assert mat_parsed.id == 42
        assert mat_parsed.rho0 == pytest.approx(1.52e-9)
        assert mat_parsed.rhor == pytest.approx(1.55e-9)
        assert mat_parsed.ea == pytest.approx(142000.0)
        assert mat_parsed.eb == pytest.approx(9800.0)
        assert mat_parsed.ec == pytest.approx(9800.0)
        assert mat_parsed.prab == pytest.approx(0.31)
        assert mat_parsed.prbc == pytest.approx(0.44)
        assert mat_parsed.prca == pytest.approx(0.021)
        assert mat_parsed.gab == pytest.approx(4900.0)
        assert mat_parsed.gbc == pytest.approx(3300.0)
        assert mat_parsed.gca == pytest.approx(4900.0)
        assert mat_parsed.sigt1 == pytest.approx(1750.0)
        assert mat_parsed.sigt2 == pytest.approx(42.0)
        assert mat_parsed.sigt3 == pytest.approx(42.0)
        assert mat_parsed.delta == pytest.approx(0.065)
        assert mat_parsed.cb == pytest.approx(135.0)
        assert mat_parsed.cn == pytest.approx(0.52)
        assert mat_parsed.fmax == pytest.approx(1150.0)
        assert mat_parsed.wplaref == pytest.approx(1.1)
        assert mat_parsed.sigyt1 == pytest.approx(1950.0)
        assert mat_parsed.sigyt2 == pytest.approx(48.0)
        assert mat_parsed.sigyc1 == pytest.approx(1450.0)
        assert mat_parsed.sigyc2 == pytest.approx(195.0)
        assert mat_parsed.sigt12 == pytest.approx(78.0)
        assert mat_parsed.sigc12 == pytest.approx(78.0)
        assert mat_parsed.sigt23 == pytest.approx(49.0)
        assert mat_parsed.sigc23 == pytest.approx(49.0)
        assert mat_parsed.alpha == pytest.approx(0.26)
        assert mat_parsed.efib == pytest.approx(182000.0)
        assert mat_parsed.cc == pytest.approx(0.038)
        assert mat_parsed.eps0 == pytest.approx(1.2)
        assert mat_parsed.strflag == 1

    def test_writer_parameter_aliases_kwargs(self, tmp_path: Path):
        """Verify kwargs aliases (e11/ea, nu12/prab, b/cb, n/cn, etc.) in StarterDeck."""
        d = StarterDeck("KWARGS_ALIAS").mat_law14(
            mid=7,
            title="Kwargs_Aliases",
            rho0=1.5e-9,
            rhor=1.5e-9,
            e11=110000.0,
            e22=10000.0,
            e33=10000.0,
            nu12=0.3,
            nu23=0.4,
            nu31=0.02,
            g12=5000.0,
            g23=3000.0,
            g31=5000.0,
            b=125.0,
            n=0.55,
            sig=1100.0,
            wpref=1.05,
            sig_1yt=1800.0,
            sig_2yt=50.0,
            sig_1yc=1300.0,
            sig_2yc=180.0,
            sig_12yt=75.0,
            sig_12yc=75.0,
            sig_23yt=45.0,
            sig_23yc=45.0,
            alpha_fib=0.25,
            e_fib=170000.0,
            src=0.03,
            srp=1.0,
            icc=1,
        )
        model, log = _parse_deck_string(tmp_path, d.render())
        assert len(log.errors) == 0
        mat = model.mat_law14s[7]
        assert mat.ea == pytest.approx(110000.0)
        assert mat.cb == pytest.approx(125.0)
        assert mat.cn == pytest.approx(0.55)
        assert mat.fmax == pytest.approx(1100.0)
        assert mat.sigyt1 == pytest.approx(1800.0)
        assert mat.alpha == pytest.approx(0.25)


# ============================================================================
# 4. Dual-Personality Disambiguation Audit
# ============================================================================

class TestLaw14DualPersonalityDisambiguation:
    """Audit dual-personality routing for /MAT/LAW14 vs /MAT/COMPSO / /MAT/COMP_SOL."""

    def test_law14_one_card_routes_to_cam_clay(self, tmp_path: Path):
        """/MAT/LAW14 with 1 data card routes to read_mat_cam_clay."""
        deck = """
/BEGIN
CAMCLAY_1CARD
-1
/MAT/LAW14/101
CamClay_1Card
1.8e-9, 100.0, 0.3
/END
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        assert 101 in model.mat_law14s
        mat = model.mat_law14s[101]
        assert hasattr(mat, "lambda_") or hasattr(mat, "lamda")
        assert mat.ea == 0.0

    def test_law14_two_cards_routes_to_cam_clay(self, tmp_path: Path):
        """/MAT/LAW14 with 2 data cards routes to read_mat_cam_clay."""
        deck = """
/BEGIN
CAMCLAY_2CARDS
-1
/MAT/LAW14/102
CamClay_2Cards
1.8e-9, 100.0, 0.3
1.2, 0.05, 0.01, 0.8, 50.0
/END
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        assert 102 in model.mat_law14s
        mat = model.mat_law14s[102]
        assert hasattr(mat, "kappa")
        assert getattr(mat, "kappa", 0.0) == pytest.approx(0.01)

    def test_law14_three_cards_routes_to_compso(self, tmp_path: Path):
        """/MAT/LAW14 with 3 data cards routes to read_mat_compso."""
        deck = """
/BEGIN
COMPSO_3CARDS
-1
/MAT/LAW14/201
Compso_3Cards
1.5e-9, 1.5e-9
100000.0, 10000.0, 10000.0
0.3, 0.4, 0.02
/END
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        assert 201 in model.materials
        m = model.materials[201]
        assert m.law == 14
        assert "D11" in m.params
        assert m.params["D11"] > 0.0

    def test_law14_five_cards_routes_to_compso(self, tmp_path: Path):
        """/MAT/LAW14 with 5 data cards routes to read_mat_compso."""
        deck = """
/BEGIN
COMPSO_5CARDS
-1
/MAT/LAW14/202
Compso_5Cards
1.5e-9, 1.5e-9
100000.0, 10000.0, 10000.0
0.3, 0.4, 0.02
5000.0, 3000.0, 5000.0
1500.0, 40.0, 40.0, 0.05
/END
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        assert 202 in model.materials
        assert model.mat_law14s[202].sigt1 == pytest.approx(1500.0)

    def test_compso_synonym_with_one_or_two_cards_never_routes_to_cam_clay(self, tmp_path: Path):
        """/MAT/COMPSO and /MAT/COMP_SOL with <= 2 cards always route to read_mat_compso."""
        deck = """
/BEGIN
COMPSO_SHORT_TEST
-1
/MAT/COMPSO/301
Compso_Short
1.5e-9, 1.5e-9
100000.0, 10000.0, 10000.0
/MAT/COMP_SOL/302
CompSol_Short
1.5e-9, 1.5e-9
100000.0, 10000.0, 10000.0
/END
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        assert 301 in model.mat_law14s
        assert 302 in model.mat_law14s
        assert model.mat_law14s[301].ea == pytest.approx(100000.0)
        assert model.mat_law14s[302].ea == pytest.approx(100000.0)
        assert not hasattr(model.mat_law14s[301], "kappa") or model.mat_law14s[301].kappa == 0.0


# ============================================================================
# 5. Starter Checks & Element Compatibility Audit
# ============================================================================

class TestLaw14StarterChecksAndElementCompatibility:
    """Starter checks verifying element families, 2D rejections, and parameter bounds."""

    def test_allowed_solid_element_families(self):
        """Solid element families (bricks, tetras, penta6, pyra5) accept LAW14."""
        solid_families = ("bricks", "tetras", "penta6", "pyra5")
        for fam in solid_families:
            assert fam in _ALLOWED_LAWS
            assert 14 in _ALLOWED_LAWS[fam]
            assert "14" in _ALLOWED_LAWS[fam] or "LAW14" in _ALLOWED_LAWS[fam]

        # Validation check_model passes cleanly for 3D solids
        for fam in solid_families:
            model = Model()
            model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
            mat = Material(id=1, law=14, rho0=1.5e-9, params={"E11": 100000.0, "E22": 10000.0, "E33": 10000.0})
            model.materials[1] = mat

            class FakeGroup:
                def __init__(self):
                    self.state = {"slices": [(slice(0, 1), mat, None)]}

            model.element_groups = lambda f=fam: [(f, FakeGroup())]
            log = MessageLog()
            check_model(model, log)
            assert len(log.errors) == 0, f"Unexpected error for solid family {fam}: {log.errors}"

    def test_disallowed_element_families_rejected(self):
        """Non-solids (shells, beams, trusses, springs, quads) are strictly rejected with error messages."""
        disallowed = ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "beams", "trusses", "springs")
        mat = Material(id=1, law=14, rho0=1.5e-9, params={"E11": 100000.0, "E22": 10000.0, "E33": 10000.0})

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
                "/MAT/LAW14/1 (/MAT/COMPSO) is not supported for" in e or "is not supported for" in e
                for e in log.errors
            ), f"Failed to reject disallowed family {fam}: {log.errors}"

    def test_2d_analysis_rejected_ancmsg_305(self):
        """2D analysis (N2D > 0) rejected per hm_read_mat14.F:151 (ANCMSG 305)."""
        model = Model()
        model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
        model.n2d = 1  # 2D model
        mat = Material(id=1, law=14, rho0=1.5e-9, params={"E11": 100000.0, "E22": 10000.0, "E33": 10000.0})
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [("bricks", FakeGroup())]
        log = MessageLog()
        check_model(model, log)
        assert any(
            "/MAT/LAW14/1: LAW14 is not supported for 2D analysis (N2D > 0) (ANCMSG 305)" in e
            for e in log.errors
        )

    def test_parameter_bounds_rho0(self):
        """Initial density rho0 <= 0 rejected with error."""
        mat_zero = Material(id=1, law=14, rho0=0.0, params={"rho": 0.0, "e11": 1e5, "e22": 1e4, "e33": 1e4})
        log1 = MessageLog()
        check_mat_law14(mat_zero, log1)
        assert any("initial density RHO must be > 0" in e for e in log1.errors)

        mat_neg = Material(id=2, law=14, rho0=-1.5e-9, params={"rho": -1.5e-9, "e11": 1e5, "e22": 1e4, "e33": 1e4})
        log2 = MessageLog()
        check_mat_law14(mat_neg, log2)
        assert any("initial density RHO must be > 0" in e for e in log2.errors)

    def test_parameter_bounds_moduli_ancmsg_306(self):
        """Moduli E11, E22, E33 <= 0 rejected with ANCMSG 306."""
        # E11 = 0
        mat1 = Material(id=1, law=14, rho0=1.5e-9, params={"e11": 0.0, "e22": 1e4, "e33": 1e4})
        log1 = MessageLog()
        check_mat_law14(mat1, log1)
        assert any("Young's moduli E11, E22, E33 must be > 0" in e and "ANCMSG 306" in e for e in log1.errors)

        # E22 < 0
        mat2 = Material(id=2, law=14, rho0=1.5e-9, params={"e11": 1e5, "e22": -1e4, "e33": 1e4})
        log2 = MessageLog()
        check_mat_law14(mat2, log2)
        assert any("Young's moduli E11, E22, E33 must be > 0" in e and "ANCMSG 306" in e for e in log2.errors)

        # E33 <= 0
        mat3 = Material(id=3, law=14, rho0=1.5e-9, params={"e11": 1e5, "e22": 1e4, "e33": 0.0})
        log3 = MessageLog()
        check_mat_law14(mat3, log3)
        assert any("Young's moduli E11, E22, E33 must be > 0" in e and "ANCMSG 306" in e for e in log3.errors)

    def test_parameter_bounds_compliance_detc_ancmsg_307(self):
        """Compliance determinant DETC <= 0 rejected with ANCMSG 307."""
        # Unphysical Poisson ratios violating thermodynamic stability (positive definite compliance)
        p = {
            "e11": 100000.0, "e22": 10000.0, "e33": 10000.0,
            "nu12": 0.95, "nu23": 0.95, "nu31": 0.95,
            "g12": 5000.0, "g23": 3000.0, "g31": 5000.0,
            "rho": 1.5e-9
        }
        mat = Material(id=1, law=14, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law14(mat, log)
        assert any("compliance matrix determinant DETC must be > 0" in e and "ANCMSG 307" in e for e in log.errors)

    def test_parameter_bounds_shear_moduli(self):
        """Negative shear moduli G12, G23, G31 rejected."""
        p = {
            "e11": 100000.0, "e22": 10000.0, "e33": 10000.0,
            "nu12": 0.3, "nu23": 0.4, "nu31": 0.02,
            "g12": -5000.0, "g23": 3000.0, "g31": 5000.0,
            "rho": 1.5e-9
        }
        mat = Material(id=1, law=14, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law14(mat, log)
        assert any("shear moduli G12, G23, G31 must be >= 0" in e for e in log.errors)


# ============================================================================
# 6. State Serialization (Restart / Pickling) Audit
# ============================================================================

class TestLaw14StateSerializationAndPickling:
    """Verify restart serialization fidelity of extra_shapes, Material, and element state."""

    def test_extra_shapes_pickling(self):
        """extra_shapes dictionary for nip=1 and nip=4 survives pickling and unpickling."""
        # nip = 1 (single integration point solid)
        s1 = extra_shapes(nip=1)
        raw1 = pickle.dumps(s1)
        restored1 = pickle.loads(raw1)
        assert restored1 == s1
        assert restored1["dam14"] == (5,)
        assert restored1["epe14"] == (3,)
        assert restored1["epc14"] == (3,)
        assert restored1["wpla14"] == ()
        assert restored1["off14"] == ()
        assert restored1["epsf14"] == ()
        assert restored1["sigf14"] == ()
        assert restored1["tsaiwu14"] == ()

        # nip = 4 (4-point reduced integration solid)
        s4 = extra_shapes(nip=4)
        raw4 = pickle.dumps(s4)
        restored4 = pickle.loads(raw4)
        assert restored4 == s4
        assert restored4["dam14"] == (4, 5)
        assert restored4["epe14"] == (4, 3)
        assert restored4["epc14"] == (4, 3)
        assert restored4["wpla14"] == (4,)
        assert restored4["off14"] == (4,)
        assert restored4["epsf14"] == (4,)
        assert restored4["sigf14"] == (4,)
        assert restored4["tsaiwu14"] == (4,)

    def test_material_and_built_material_pickling(self):
        """Material instance built via build_law14 can be pickled and unpickled without corruption."""
        mat_law = MatLaw14(
            id=14,
            title="Pickle_LAW14_Test",
            rho0=1.55e-9,
            rhor=1.55e-9,
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
            sigt12=80.0,
            sigc12=80.0,
            sigt23=50.0,
            sigc23=50.0,
            alpha=0.25,
            efib=180000.0,
            cc=0.04,
            eps0=1.0,
            strflag=1,
        )
        mat = build_law14(mat_law)

        raw = pickle.dumps(mat)
        restored = pickle.loads(raw)

        assert restored.id == mat.id
        assert restored.law == 14
        assert restored.rho0 == pytest.approx(mat.rho0)
        assert restored.title == mat.title
        assert restored.law_name == mat.law_name
        np.testing.assert_allclose(restored.d_mat, mat.d_mat, rtol=1e-12)
        assert restored.ssp == pytest.approx(mat.ssp, rel=1e-12)
        assert restored.sound_speed_solid() == pytest.approx(mat.sound_speed_solid(), rel=1e-12)
        assert restored.params["D11"] == pytest.approx(mat.params["D11"], rel=1e-12)
        assert restored.params["SSP"] == pytest.approx(mat.params["SSP"], rel=1e-12)

    def test_element_state_arrays_pickling(self):
        """Element persistent state arrays survive restart pickling with zero corruption."""
        n_elem = 16
        state = {
            "dam14": np.random.default_rng(42).uniform(0.0, 0.5, size=(n_elem, 5)),
            "epe14": np.random.default_rng(43).uniform(1e-4, 1e-2, size=(n_elem, 3)),
            "epc14": np.random.default_rng(44).uniform(0.0, 1e-3, size=(n_elem, 3)),
            "wpla14": np.random.default_rng(45).uniform(0.0, 10.0, size=(n_elem,)),
            "off14": np.zeros(n_elem, dtype=float),
            "epsf14": np.random.default_rng(46).uniform(1e-4, 5e-3, size=(n_elem,)),
            "sigf14": np.random.default_rng(47).uniform(100.0, 800.0, size=(n_elem,)),
            "tsaiwu14": np.random.default_rng(48).uniform(0.1, 0.9, size=(n_elem,)),
        }

        raw = pickle.dumps(state)
        restored = pickle.loads(raw)

        for k in state:
            assert k in restored
            np.testing.assert_allclose(restored[k], state[k], err_msg=f"Mismatch in state array {k}")

    def test_model_pickling_with_law14(self):
        """A complete Model with LAW14 materials and aliases pickles and unpickles cleanly."""
        model = Model()
        model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
        mat_entity = MatLaw14(id=1, rho0=1.5e-9, ea=140000.0, eb=10000.0, ec=10000.0)
        model.mat_law14s[1] = mat_entity
        model.materials[1] = build_law14(mat_entity)

        # Aliases test
        assert model.mat_compsos is model.mat_law14s
        assert model.mat_comp_sols is model.mat_law14s

        raw = pickle.dumps(model)
        restored = pickle.loads(raw)

        assert 1 in restored.mat_law14s
        assert 1 in restored.materials
        assert restored.mat_compsos is restored.mat_law14s
        assert restored.mat_comp_sols is restored.mat_law14s
        assert restored.mat_law14s[1].ea == pytest.approx(140000.0)
        assert restored.materials[1].law == 14
        assert restored.materials[1].sound_speed_solid() > 0.0


# ============================================================================
# 7. Model Storage Aliases & Entity Properties / Setters
# ============================================================================

class TestLaw14ModelStorageAndProperties:
    """Verify Model storage aliases and MatLaw14 property setters."""

    def test_model_storage_aliases(self):
        """Verify model.mat_law14s, mat_compsos, mat_comp_sols, mat_camclays aliasing."""
        m = Model()
        mat = MatLaw14(id=1, rho0=1.5e-9, ea=100000.0)
        m.mat_law14s[1] = mat

        assert m.mat_compsos is m.mat_law14s
        assert m.mat_comp_sols is m.mat_law14s
        assert m.mat_camclays is m.mat_law14s
        assert 1 in m.mat_compsos
        assert 1 in m.mat_comp_sols
        assert 1 in m.mat_camclays

    def test_mat_law14_property_setters(self):
        """Test all OpenRadioss parameter aliases and setters on MatLaw14."""
        mat = MatLaw14()

        # Density
        mat.rho = 1.55e-9
        assert mat.rho0 == pytest.approx(1.55e-9) and mat.rho == pytest.approx(1.55e-9)
        mat.refer_rho = 1.60e-9
        assert mat.rhor == pytest.approx(1.60e-9) and mat.refer_rho == pytest.approx(1.60e-9)

        # Moduli
        mat.e11 = 145000.0
        assert mat.ea == pytest.approx(145000.0) and mat.e11 == pytest.approx(145000.0)
        mat.e22 = 10500.0
        assert mat.eb == pytest.approx(10500.0) and mat.e22 == pytest.approx(10500.0)
        mat.e33 = 10200.0
        assert mat.ec == pytest.approx(10200.0) and mat.e33 == pytest.approx(10200.0)

        # Poisson's ratios
        mat.nu12 = 0.31
        assert mat.prab == pytest.approx(0.31) and mat.nu12 == pytest.approx(0.31)
        mat.nu23 = 0.44
        assert mat.prbc == pytest.approx(0.44) and mat.nu23 == pytest.approx(0.44)
        mat.nu31 = 0.022
        assert mat.prca == pytest.approx(0.022) and mat.nu31 == pytest.approx(0.022)

        # Shear moduli
        mat.g12 = 5100.0
        assert mat.gab == pytest.approx(5100.0) and mat.g12 == pytest.approx(5100.0)
        mat.g23 = 3300.0
        assert mat.gbc == pytest.approx(3300.0) and mat.g23 == pytest.approx(3300.0)
        mat.g31 = 5100.0
        assert mat.gca == pytest.approx(5100.0) and mat.g31 == pytest.approx(5100.0)

        # Tensile strengths
        mat.sig_t1 = 1780.0
        assert mat.sigt1 == pytest.approx(1780.0) and mat.sig_t1 == pytest.approx(1780.0)
        mat.sig_t2 = 41.0
        assert mat.sigt2 == pytest.approx(41.0) and mat.sig_t2 == pytest.approx(41.0)
        mat.sig_t3 = 39.0
        assert mat.sigt3 == pytest.approx(39.0) and mat.sig_t3 == pytest.approx(39.0)

        # Hardening
        mat.b = 145.0
        assert mat.cb == pytest.approx(145.0) and mat.b == pytest.approx(145.0)
        mat.n = 0.54
        assert mat.cn == pytest.approx(0.54) and mat.n == pytest.approx(0.54)

        # Yield stresses
        mat.sig_1yt = 2020.0
        assert mat.sigyt1 == pytest.approx(2020.0) and mat.sig_1yt == pytest.approx(2020.0)
        mat.sig_2yt = 52.0
        assert mat.sigyt2 == pytest.approx(52.0) and mat.sig_2yt == pytest.approx(52.0)
        mat.sig_1yc = 1520.0
        assert mat.sigyc1 == pytest.approx(1520.0) and mat.sig_1yc == pytest.approx(1520.0)
        mat.sig_2yc = 205.0
        assert mat.sigyc2 == pytest.approx(205.0) and mat.sig_2yc == pytest.approx(205.0)

        mat.sig_12yt = 82.0
        assert mat.sigt12 == pytest.approx(82.0) and mat.sig_12yt == pytest.approx(82.0)
        mat.sig_12yc = 79.0
        assert mat.sigc12 == pytest.approx(79.0) and mat.sig_12yc == pytest.approx(79.0)
        mat.sig_23yt = 51.0
        assert mat.sigt23 == pytest.approx(51.0) and mat.sig_23yt == pytest.approx(51.0)
        mat.sig_23yc = 49.0
        assert mat.sigc23 == pytest.approx(49.0) and mat.sig_23yc == pytest.approx(49.0)

        # Rate
        mat.c = 0.042
        assert mat.cc == pytest.approx(0.042) and mat.c == pytest.approx(0.042)
