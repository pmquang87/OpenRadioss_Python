"""Milestone M546: DeckWriter, CFG Synonyms, and Roundtrip Auditor for /MAT/LAW12 (/MAT/3D_COMP, /MAT/COMP_3D).

Audits all 10 cards per 3d_comp_12.cfg and hm_read_mat12.F:
  Card 1: RHO_I, Refer_Rho (%20lg%20lg)
  Card 2: E11, E22, E33 (%20lg%20lg%20lg)
  Card 3: NU12, NU23, NU31 (%20lg%20lg%20lg)
  Card 4: G12, G23, G31 (%20lg%20lg%20lg)
  Card 5: SIGT1, SIGT2, SIGT3, DELTA (%20lg%20lg%20lg%20lg)
  Card 6: CB, CN, FMAX, WPLAREF (%20lg%20lg%20lg%20lg)
  Card 7: SIGYT1, SIGYT2, SIGYC1, SIGYC2 (%20lg%20lg%20lg%20lg)
  Card 8: SIGYT12, SIGYC12, SIGYT23, SIGYC23 (%20lg%20lg%20lg%20lg)
  Card 9: SIGYT3, SIGYC3, SIGYT13, SIGYC13 (%20lg%20lg%20lg%20lg)
  Card 10: ALPHA, EFIB, CC, EPS0, STRFLAG (%20lg%20lg%20lg%20lg%10d)
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
from pyradioss.model.entities import Material, MatLaw12, Mat3dComp, MatComp3d, Mat3parbi
from pyradioss.model.model import Model
from pyradioss.starter.checks import _ALLOWED_LAWS, check_mat_law12, check_model


def _parse_deck_string(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    """Helper to parse a deck string into Model and MessageLog."""
    deck_path = tmp_path / "ROUNDTRIP_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def _get_mat_cards(deck_str: str) -> List[str]:
    """Extract material data cards from rendered deck (skipping /BEGIN, /TITLE, and /MAT header/title)."""
    lines = deck_str.splitlines()
    mat_idx = next(i for i, line in enumerate(lines) if line.startswith("/MAT/"))
    cards = []
    # lines[mat_idx] is /MAT/..., lines[mat_idx + 1] is title card
    for line in lines[mat_idx + 2:]:
        if line.startswith("/"):
            break
        if not line.startswith("#"):
            cards.append(line)
    return cards


# ============================================================================
# 1. Ten Cards Layout & Parsing Audit (3d_comp_12.cfg)
# ============================================================================

class TestLaw12CardsAudit:
    """Audit of all 10 cards according to 3d_comp_12.cfg and hm_read_mat12.F."""

    def test_card1_rho_refer_rho_roundtrip(self, tmp_path: Path):
        """Card 1: RHO_I, Refer_Rho (%20lg%20lg). Single and dual values."""
        # Dual densities
        deck1 = StarterDeck("CARD1_DUAL").mat_law12(
            mid=1, rho=1.6e-9, refer_rho=1.65e-9,
            e11=100000.0, e22=10000.0, e33=10000.0,
            nu12=0.3, nu23=0.3, nu31=0.03,
            g12=5000.0, g23=3000.0, g31=5000.0,
        ).write()
        cards1 = _get_mat_cards(deck1)
        card1 = cards1[0]
        assert len(card1) == 40
        c1_fields = cl.split_fixed(card1, cl.MAT_LAW12_1)
        assert float(c1_fields[0]) == pytest.approx(1.6e-9)
        assert float(c1_fields[1]) == pytest.approx(1.65e-9)

        model1, log1 = _parse_deck_string(tmp_path, deck1)
        assert len(log1.errors) == 0
        assert model1.mat_law12s[1].rho0 == pytest.approx(1.6e-9)
        assert model1.mat_law12s[1].rhor == pytest.approx(1.65e-9)

        # Single density (refer_rho defaults to rho0 in Starter)
        deck2 = StarterDeck("CARD1_SINGLE").mat_law12(
            mid=2, rho=1.75e-9,
            e11=120000.0, e22=12000.0, e33=12000.0,
            nu12=0.28, nu23=0.35, nu31=0.02,
            g12=6000.0, g23=4000.0, g31=6000.0,
        ).write()
        cards2 = _get_mat_cards(deck2)
        card1_single = cards2[0]
        assert len(card1_single) == 20
        model2, log2 = _parse_deck_string(tmp_path, deck2)
        assert len(log2.errors) == 0
        assert model2.mat_law12s[2].rho0 == pytest.approx(1.75e-9)
        assert model2.mat_law12s[2].rhor == pytest.approx(1.75e-9)

    def test_card2_elastic_moduli_roundtrip(self, tmp_path: Path):
        """Card 2: E11, E22, E33 (%20lg%20lg%20lg)."""
        deck = StarterDeck("CARD2_TEST").mat_law12(
            mid=1, rho=1.5e-9, e11=145000.0, e22=11200.0, e33=9800.0,
            nu12=0.3, nu23=0.4, nu31=0.02, g12=5000.0, g23=3500.0, g31=5000.0
        ).write()
        cards = _get_mat_cards(deck)
        card2 = cards[1]
        assert len(card2) == 60
        f = cl.split_fixed(card2, cl.MAT_LAW12_2)
        assert [float(x) for x in f] == pytest.approx([145000.0, 11200.0, 9800.0])

        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.mat_law12s[1]
        assert mat.e11 == pytest.approx(145000.0)
        assert mat.e22 == pytest.approx(11200.0)
        assert mat.e33 == pytest.approx(9800.0)

    def test_card3_poissons_ratios_roundtrip(self, tmp_path: Path):
        """Card 3: NU12, NU23, NU31 (%20lg%20lg%20lg)."""
        deck = StarterDeck("CARD3_TEST").mat_law12(
            mid=1, rho=1.5e-9, e11=100000.0, e22=10000.0, e33=10000.0,
            nu12=0.29, nu23=0.42, nu31=0.015, g12=4500.0, g23=3200.0, g31=4500.0
        ).write()
        cards = _get_mat_cards(deck)
        card3 = cards[2]
        assert len(card3) == 60
        f = cl.split_fixed(card3, cl.MAT_LAW12_3)
        assert [float(x) for x in f] == pytest.approx([0.29, 0.42, 0.015])

        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.mat_law12s[1]
        assert mat.nu12 == pytest.approx(0.29)
        assert mat.nu23 == pytest.approx(0.42)
        assert mat.nu31 == pytest.approx(0.015)

    def test_card4_shear_moduli_roundtrip(self, tmp_path: Path):
        """Card 4: G12, G23, G31 (%20lg%20lg%20lg)."""
        deck = StarterDeck("CARD4_TEST").mat_law12(
            mid=1, rho=1.5e-9, e11=100000.0, e22=10000.0, e33=10000.0,
            nu12=0.3, nu23=0.4, nu31=0.02, g12=5600.0, g23=3800.0, g31=5400.0
        ).write()
        cards = _get_mat_cards(deck)
        card4 = cards[3]
        assert len(card4) == 60
        f = cl.split_fixed(card4, cl.MAT_LAW12_4)
        assert [float(x) for x in f] == pytest.approx([5600.0, 3800.0, 5400.0])

        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.mat_law12s[1]
        assert mat.g12 == pytest.approx(5600.0)
        assert mat.g23 == pytest.approx(3800.0)
        assert mat.g31 == pytest.approx(5400.0)

    def test_card5_tensile_strength_and_damage_roundtrip(self, tmp_path: Path):
        """Card 5: SIGT1, SIGT2, SIGT3, DELTA (%20lg%20lg%20lg%20lg)."""
        deck = StarterDeck("CARD5_TEST").mat_law12(
            mid=1, rho=1.5e-9, e11=100000.0, e22=10000.0, e33=10000.0,
            nu12=0.3, nu23=0.4, nu31=0.02, g12=5000.0, g23=3500.0, g31=5000.0,
            sigt1=1950.0, sigt2=45.0, sigt3=42.0, delta=0.12
        ).write()
        cards = _get_mat_cards(deck)
        card5 = cards[4]
        assert len(card5) == 80
        f = cl.split_fixed(card5, cl.MAT_LAW12_5)
        assert [float(x) for x in f] == pytest.approx([1950.0, 45.0, 42.0, 0.12])

        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.mat_law12s[1]
        assert mat.sig_t1 == pytest.approx(1950.0)
        assert mat.sig_t2 == pytest.approx(45.0)
        assert mat.sig_t3 == pytest.approx(42.0)
        assert mat.delta == pytest.approx(0.12)

    def test_card6_plastic_hardening_roundtrip(self, tmp_path: Path):
        """Card 6: CB, CN, FMAX, WPLAREF (%20lg%20lg%20lg%20lg)."""
        deck = StarterDeck("CARD6_TEST").mat_law12(
            mid=1, rho=1.5e-9, e11=100000.0, e22=10000.0, e33=10000.0,
            nu12=0.3, nu23=0.4, nu31=0.02, g12=5000.0, g23=3500.0, g31=5000.0,
            cb=175.0, cn=0.65, fmax=1350.0, wplaref=1.25
        ).write()
        cards = _get_mat_cards(deck)
        card6 = cards[5]
        assert len(card6) == 80
        f = cl.split_fixed(card6, cl.MAT_LAW12_6)
        assert [float(x) for x in f] == pytest.approx([175.0, 0.65, 1350.0, 1.25])

        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.mat_law12s[1]
        assert mat.b == pytest.approx(175.0)
        assert mat.n == pytest.approx(0.65)
        assert mat.fmax == pytest.approx(1350.0)
        assert mat.wplaref == pytest.approx(1.25)

    def test_card7_yield_stresses_normal_roundtrip(self, tmp_path: Path):
        """Card 7: SIGYT1, SIGYT2, SIGYC1, SIGYC2 (%20lg%20lg%20lg%20lg)."""
        deck = StarterDeck("CARD7_TEST").mat_law12(
            mid=1, rho=1.5e-9, e11=100000.0, e22=10000.0, e33=10000.0,
            nu12=0.3, nu23=0.4, nu31=0.02, g12=5000.0, g23=3500.0, g31=5000.0,
            sigyt1=2100.0, sigyt2=55.0, sigyc1=1600.0, sigyc2=220.0
        ).write()
        cards = _get_mat_cards(deck)
        card7 = cards[6]
        assert len(card7) == 80
        f = cl.split_fixed(card7, cl.MAT_LAW12_7)
        assert [float(x) for x in f] == pytest.approx([2100.0, 55.0, 1600.0, 220.0])

        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.mat_law12s[1]
        assert mat.sig_1yt == pytest.approx(2100.0)
        assert mat.sig_2yt == pytest.approx(55.0)
        assert mat.sig_1yc == pytest.approx(1600.0)
        assert mat.sig_2yc == pytest.approx(220.0)

    def test_card8_shear_yield_stresses_12_23_roundtrip(self, tmp_path: Path):
        """Card 8: SIGYT12, SIGYC12, SIGYT23, SIGYC23 (%20lg%20lg%20lg%20lg)."""
        deck = StarterDeck("CARD8_TEST").mat_law12(
            mid=1, rho=1.5e-9, e11=100000.0, e22=10000.0, e33=10000.0,
            nu12=0.3, nu23=0.4, nu31=0.02, g12=5000.0, g23=3500.0, g31=5000.0,
            sigyt12=85.0, sigyc12=82.0, sigyt23=52.0, sigyc23=51.0
        ).write()
        cards = _get_mat_cards(deck)
        card8 = cards[7]
        assert len(card8) == 80
        f = cl.split_fixed(card8, cl.MAT_LAW12_8)
        assert [float(x) for x in f] == pytest.approx([85.0, 82.0, 52.0, 51.0])

        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.mat_law12s[1]
        assert mat.sig_12yt == pytest.approx(85.0)
        assert mat.sig_12yc == pytest.approx(82.0)
        assert mat.sig_23yt == pytest.approx(52.0)
        assert mat.sig_23yc == pytest.approx(51.0)

    def test_card9_yield_stresses_3_13_roundtrip(self, tmp_path: Path):
        """Card 9: SIGYT3, SIGYC3, SIGYT13, SIGYC13 (%20lg%20lg%20lg%20lg)."""
        deck = StarterDeck("CARD9_TEST").mat_law12(
            mid=1, rho=1.5e-9, e11=100000.0, e22=10000.0, e33=10000.0,
            nu12=0.3, nu23=0.4, nu31=0.02, g12=5000.0, g23=3500.0, g31=5000.0,
            sigyt3=53.0, sigyc3=215.0, sigyt13=84.0, sigyc13=83.0
        ).write()
        cards = _get_mat_cards(deck)
        card9 = cards[8]
        assert len(card9) == 80
        f = cl.split_fixed(card9, cl.MAT_LAW12_9)
        assert [float(x) for x in f] == pytest.approx([53.0, 215.0, 84.0, 83.0])

        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.mat_law12s[1]
        assert mat.sig_3yt == pytest.approx(53.0)
        assert mat.sig_3yc == pytest.approx(215.0)
        assert mat.sig_13yt == pytest.approx(84.0)
        assert mat.sig_13yc == pytest.approx(83.0)

    def test_card10_fiber_and_strain_rate_roundtrip(self, tmp_path: Path):
        """Card 10: ALPHA, EFIB, CC, EPS0, STRFLAG (%20lg%20lg%20lg%20lg%10d)."""
        deck = StarterDeck("CARD10_TEST").mat_law12(
            mid=1, rho=1.5e-9, e11=100000.0, e22=10000.0, e33=10000.0,
            nu12=0.3, nu23=0.4, nu31=0.02, g12=5000.0, g23=3500.0, g31=5000.0,
            alpha=0.35, efib=195000.0, cc=0.05, eps0=1.5, strflag=2
        ).write()
        cards = _get_mat_cards(deck)
        card10 = cards[9]
        assert len(card10) == 90
        f = cl.split_fixed(card10, cl.MAT_LAW12_10)
        assert [float(x) for x in f[:4]] == pytest.approx([0.35, 195000.0, 0.05, 1.5])
        assert int(f[4]) == 2

        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.mat_law12s[1]
        assert mat.alpha == pytest.approx(0.35)
        assert mat.efib == pytest.approx(195000.0)
        assert mat.c == pytest.approx(0.05)
        assert mat.eps0 == pytest.approx(1.5)
        assert mat.icc == 2


# ============================================================================
# 2. DeckWriter Fixed Widths, Aliases & Extreme Numbers
# ============================================================================

class TestLaw12DeckWriter:
    """DeckWriter formatting, aliases, and numerical extremes."""

    def test_deckwriter_all_ten_cards_fixed_widths(self):
        """Verify exact character widths for all 10 cards matching card_layouts.py."""
        deck = StarterDeck("ALL_CARDS_WIDTHS").mat_law12(
            mid=42, title="Complete_Law12",
            rho=1.55e-9, refer_rho=1.55e-9,
            e11=140000.0, e22=10000.0, e33=10000.0,
            nu12=0.3, nu23=0.45, nu31=0.02,
            g12=5000.0, g23=3500.0, g31=5000.0,
            sigt1=1800.0, sigt2=40.0, sigt3=40.0, delta=0.08,
            cb=150.0, cn=0.5, fmax=1200.0, wplaref=1.0,
            sigyt1=2000.0, sigyt2=50.0, sigyc1=1500.0, sigyc2=200.0,
            sigyt12=80.0, sigyc12=80.0, sigyt23=50.0, sigyc23=50.0,
            sigyt3=50.0, sigyc3=200.0, sigyt13=80.0, sigyc13=80.0,
            alpha=0.25, efib=180000.0, cc=0.04, eps0=1.0, strflag=1,
        ).write()

        cards = _get_mat_cards(deck)
        assert len(cards) == 10
        expected_widths = [
            sum(cl.MAT_LAW12_1),
            sum(cl.MAT_LAW12_2),
            sum(cl.MAT_LAW12_3),
            sum(cl.MAT_LAW12_4),
            sum(cl.MAT_LAW12_5),
            sum(cl.MAT_LAW12_6),
            sum(cl.MAT_LAW12_7),
            sum(cl.MAT_LAW12_8),
            sum(cl.MAT_LAW12_9),
            sum(cl.MAT_LAW12_10),
        ]
        for i, expected_w in enumerate(expected_widths):
            assert len(cards[i]) == expected_w, f"Card {i+1} width mismatch: got {len(cards[i])}, expected {expected_w}"

    def test_deckwriter_aliases(self):
        """Verify mat_3d_comp, mat_comp_3d, and mat_3parbi write proper headers."""
        deck_3d_comp = StarterDeck("ALIAS_1").mat_3d_comp(mid=10, rho=1.5e-9, e11=1e5, e22=1e4, e33=1e4).write()
        assert "/MAT/3D_COMP/10" in deck_3d_comp

        deck_comp_3d = StarterDeck("ALIAS_2").mat_comp_3d(mid=20, rho=1.5e-9, e11=1e5, e22=1e4, e33=1e4).write()
        assert "/MAT/COMP_3D/20" in deck_comp_3d

        deck_3parbi = StarterDeck("ALIAS_3").mat_3parbi(mid=30, rho=1.5e-9, e11=1e5, e22=1e4, e33=1e4).write()
        assert "/MAT/3PARBI/30" in deck_3parbi

    def test_deckwriter_extreme_numbers(self, tmp_path: Path):
        """Format extreme numbers (1e-15, 1e20, high precision) without line overflow."""
        deck = StarterDeck("EXTREMES").mat_law12(
            mid=999,
            rho=1.23456789e-15, refer_rho=1.23456789e-15,
            e11=1.987654321e20, e22=1.987654321e19, e33=1.987654321e19,
            nu12=0.123456789012345, nu23=0.234567890123456, nu31=0.0123456789012,
            g12=5.6789012345e19, g23=3.4567890123e19, g31=5.6789012345e19,
            sigt1=1e20, sigt2=1e18, sigt3=1e18, delta=1e-12,
            cb=1e-8, cn=0.999999999, fmax=1e20, wplaref=1e-6,
            sigyt1=1e20, sigyt2=1e18, sigyc1=1e20, sigyc2=1e18,
            sigyt12=1e18, sigyc12=1e18, sigyt23=1e18, sigyc23=1e18,
            sigyt3=1e18, sigyc3=1e18, sigyt13=1e18, sigyc13=1e18,
            alpha=0.333333333333, efib=2e20, cc=1e-10, eps0=1e-5, strflag=1
        ).write()

        cards = _get_mat_cards(deck)
        assert len(cards) == 10
        expected_widths = [40, 60, 60, 60, 80, 80, 80, 80, 80, 90]
        for i, expected_w in enumerate(expected_widths):
            assert len(cards[i]) == expected_w, f"Card {i+1} overflow/underflow: {len(cards[i])} != {expected_w}"

        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        mat = model.mat_law12s[999]
        assert mat.rho0 == pytest.approx(1.23456789e-15, rel=1e-5)
        assert mat.e11 == pytest.approx(1.987654321e20, rel=1e-5)
        assert mat.delta == pytest.approx(1e-12, rel=1e-5)
        assert mat.alpha == pytest.approx(0.333333333333, rel=1e-5)

    def test_deckwriter_from_mat_law12_instance(self, tmp_path: Path):
        """Pass a MatLaw12 instance directly to StarterDeck.mat_law12(mat)."""
        mat = MatLaw12(
            id=55, title="Instance_Export",
            rho0=1.6e-9, rhor=1.6e-9,
            e11=130000.0, e22=9000.0, e33=9000.0,
            nu12=0.32, nu23=0.41, nu31=0.025,
            g12=4800.0, g23=3100.0, g31=4800.0,
            sig_t1=1700.0, sig_t2=38.0, sig_t3=38.0, delta=0.07,
            b=140.0, n=0.55, fmax=1100.0, wplaref=1.0,
            sig_1yt=1900.0, sig_2yt=48.0, sig_1yc=1400.0, sig_2yc=190.0,
            sig_12yt=75.0, sig_12yc=75.0, sig_23yt=48.0, sig_23yc=48.0,
            sig_3yt=48.0, sig_3yc=190.0, sig_13yt=75.0, sig_13yc=75.0,
            alpha=0.28, efib=170000.0, c=0.03, eps0=1.0, icc=1,
            law_name="3D_COMP"
        )
        deck = StarterDeck("FROM_INSTANCE").mat_law12(mat).write()
        assert "/MAT/3D_COMP/55" in deck

        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0
        mat_read = model.mat_law12s[55]
        assert mat_read.id == 55
        assert mat_read.e11 == pytest.approx(130000.0)
        assert mat_read.sig_1yt == pytest.approx(1900.0)
        assert mat_read.alpha == pytest.approx(0.28)


# ============================================================================
# 3. Reader: Fixed Blank Defaults, Free Format & Comments
# ============================================================================

class TestLaw12ReaderRobustness:
    """Reader handling of Fortran defaults, free format, comments, and synonyms."""

    def test_reader_blank_fields_fortran_defaults(self, tmp_path: Path):
        """Fixed-format deck with blank fields assumes exact Fortran defaults (hm_read_mat12.F:154, 178-200)."""
        # Leave DELTA, CN, FMAX, WPLAREF, CC, EPS0, ICC blank
        deck = """
/BEGIN
LAW12_DEFAULTS
                  -1
/MAT/LAW12/7
Defaults_Test
#              RHO_I               RHO_O
             1.5e-9
#                E11                 E22                 E33
            100000.0             10000.0             10000.0
#               NU12                NU23                NU31
                0.25                0.35                0.02
#                G12                 G23                 G31
              5000.0              3000.0              5000.0
#              SIGT1               SIGT2               SIGT3               DELTA
              1000.0                40.0                40.0
#                  b                   n                Fmax             WplaRef
                 0.0
#             SIG1YT              SIG2YT              SIG1YC              SIG2YC
              1200.0                50.0              1000.0               150.0
#            SIG12YT             SIG12YC             SIG23YT             SIG23YC
                60.0                60.0                40.0                40.0
#             SIG3YT              SIG3YC             SIG13YT             SIG13YC
                50.0               150.0                60.0                60.0
#              ALPHA                EFIB                   C                EPS0       ICC
                 0.0                 0.0
/END
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        mat = model.mat_law12s[7]
        assert mat.rho0 == pytest.approx(1.5e-9)
        assert mat.rhor == pytest.approx(1.5e-9)  # RHOR default = RHO0
        assert mat.delta == pytest.approx(0.05)   # DELTA default = 0.05
        assert mat.n == pytest.approx(1.0)        # CN default = 1.0
        assert mat.fmax == pytest.approx(1.0e10)  # FMAX default = 1.0e10
        assert mat.wplaref == pytest.approx(1.0)  # WPLAREF default = 1.0
        assert mat.eps0 == pytest.approx(1.0)     # EPS0 default = 1.0 when CC == 0
        assert mat.icc == 1                       # ICC default = 1

    def test_reader_free_format_comma_and_space(self, tmp_path: Path):
        """Free-format deck with comma and space delimiters."""
        deck = """
/BEGIN
FREE_FORMAT_LAW12
-1
/MAT/LAW12/8
Free_Test
1.6e-9, 1.6e-9
110000.0, 11000.0, 11000.0
0.28, 0.38, 0.022
5500.0, 3200.0, 5500.0
1600.0, 42.0, 42.0, 0.06
120.0, 0.6, 1150.0, 1.0
1800.0, 48.0, 1300.0, 180.0
72.0, 72.0, 46.0, 46.0
48.0, 180.0, 72.0, 72.0
0.3, 165000.0, 0.035, 1.0, 1
/END
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        mat = model.mat_law12s[8]
        assert mat.e11 == pytest.approx(110000.0)
        assert mat.nu12 == pytest.approx(0.28)
        assert mat.b == pytest.approx(120.0)
        assert mat.sig_1yt == pytest.approx(1800.0)

    def test_reader_comments_blank_lines_and_cards(self, tmp_path: Path):
        """Comments (#), inline comments, blank lines, and whitespace cards do not disrupt card order."""
        deck = """
/BEGIN
LAW12_COMMENTS_TEST
                  -1
# This is a pre-material comment line

/MAT/LAW12/9
Comment_Deck_Test
# Card 1 with inline comment
             1.55e-9             1.55e-9
# Card 2
            140000.0             10000.0             10000.0

# Blank line above was skipped
                0.30                0.45                0.02
# Card 4
              5000.0              3500.0              5000.0
# Card 5
              1800.0                40.0                40.0                0.08
# Card 6
               150.0                 0.5              1200.0                 1.0
# Card 7
              2000.0                50.0              1500.0               200.0
# Card 8
                80.0                80.0                50.0                50.0
# Card 9
                50.0               200.0                80.0                80.0
# Card 10
                0.25            180000.0                0.04                 1.0         1
/END
"""
        model, log = _parse_deck_string(tmp_path, deck)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        mat = model.mat_law12s[9]
        assert mat.id == 9
        assert mat.e11 == pytest.approx(140000.0)
        assert mat.delta == pytest.approx(0.08)
        assert mat.icc == 1

    def test_reader_keyword_synonyms(self, tmp_path: Path):
        """All keyword synonyms /MAT/LAW12, /MAT/3D_COMP, /MAT/COMP_3D, /MAT/3PARBI parse."""
        synonyms = [
            ("LAW12", 1),
            ("3D_COMP", 2),
            ("COMP_3D", 3),
            ("3PARBI", 4),
            ("RAGAB", 5),
            ("LAW12_3D_COMP", 6),
            ("LAW12_COMP_3D", 7),
            ("LAW12_3PARBI", 8),
        ]
        deck_parts = ["/BEGIN\nSYNONYMS_TEST\n-1"]
        for syn, mid in synonyms:
            deck_parts.append(f"""
/MAT/{syn}/{mid}
Mat_{syn}
1.5e-9, 1.5e-9
100000.0, 10000.0, 10000.0
0.3, 0.4, 0.02
5000.0, 3500.0, 5000.0
1500.0, 40.0, 40.0, 0.05
0.0, 1.0, 1e10, 1.0
1500.0, 50.0, 1200.0, 180.0
70.0, 70.0, 45.0, 45.0
50.0, 180.0, 70.0, 70.0
0.25, 180000.0, 0.0, 1.0, 1""")
        deck_parts.append("/END")
        deck_str = "\n".join(deck_parts)

        model, log = _parse_deck_string(tmp_path, deck_str)
        assert len(log.errors) == 0, f"Errors: {log.errors}"
        for _, mid in synonyms:
            assert mid in model.mat_law12s
            assert model.mat_law12s[mid].id == mid
            assert model.mat_law12s[mid].e11 == pytest.approx(100000.0)


# ============================================================================
# 4. Model & Entities: Aliases, Properties & Pickle Serialization
# ============================================================================

class TestLaw12ModelAndEntities:
    """Model storage aliases, MatLaw12 properties, and pickle roundtrip."""

    def test_model_mat_law12_alias_dictionaries(self):
        """Verify model.mat_law12s, mat_3d_comps, mat_comp_3ds, mat_3parbis aliasing."""
        model = Model()
        mat = MatLaw12(id=1, rho0=1.5e-9, e11=100000.0)
        model.mat_law12s[1] = mat

        assert model.mat_3d_comps is model.mat_law12s
        assert model.mat_comp_3ds is model.mat_law12s
        assert model.mat_3parbis is model.mat_law12s
        assert 1 in model.mat_3d_comps
        assert 1 in model.mat_comp_3ds
        assert 1 in model.mat_3parbis

    def test_mat_law12_properties_and_setters(self):
        """Test all OpenRadioss CFG parameter aliases and setters on MatLaw12."""
        mat = MatLaw12()

        # Density
        mat.rho = 1.6e-9
        assert mat.rho0 == 1.6e-9 and mat.rho == 1.6e-9
        mat.refer_rho = 1.65e-9
        assert mat.rhor == 1.65e-9 and mat.refer_rho == 1.65e-9

        # Moduli
        mat.ea = 150000.0
        assert mat.e11 == 150000.0 and mat.ea == 150000.0
        mat.eb = 12000.0
        assert mat.e22 == 12000.0 and mat.eb == 12000.0
        mat.ec = 11000.0
        assert mat.e33 == 11000.0 and mat.ec == 11000.0

        # Poisson's ratios
        mat.prab = 0.28
        assert mat.nu12 == 0.28 and mat.prab == 0.28
        mat.prbc = 0.38
        assert mat.nu23 == 0.38 and mat.prbc == 0.38
        mat.prca = 0.02
        assert mat.nu31 == 0.02 and mat.prca == 0.02

        # Shear moduli
        mat.gab = 5200.0
        assert mat.g12 == 5200.0 and mat.gab == 5200.0
        mat.gbc = 3400.0
        assert mat.g23 == 3400.0 and mat.gbc == 3400.0
        mat.gca = 5100.0
        assert mat.g31 == 5100.0 and mat.gca == 5100.0

        # Tensile strength & damage
        mat.sigt1 = 1850.0
        assert mat.sig_t1 == 1850.0 and mat.sigt1 == 1850.0
        mat.sigt2 = 42.0
        assert mat.sig_t2 == 42.0 and mat.sigt2 == 42.0
        mat.sigt3 = 39.0
        assert mat.sig_t3 == 39.0 and mat.sigt3 == 39.0

        # Hardening
        mat.cb = 135.0
        assert mat.b == 135.0 and mat.cb == 135.0
        mat.cn = 0.58
        assert mat.n == 0.58 and mat.cn == 0.58
        mat.wpref = 1.15
        assert mat.wplaref == 1.15 and mat.wpref == 1.15

        # Yield stresses
        mat.sigyt1 = 1950.0
        assert mat.sig_1yt == 1950.0 and mat.sigyt1 == 1950.0
        mat.sigyt2 = 52.0
        assert mat.sig_2yt == 52.0 and mat.sigyt2 == 52.0
        mat.sigyc1 = 1450.0
        assert mat.sig_1yc == 1450.0 and mat.sigyc1 == 1450.0
        mat.sigyc2 = 195.0
        assert mat.sig_2yc == 195.0 and mat.sigyc2 == 195.0

        mat.sigyt12 = 78.0
        assert mat.sig_12yt == 78.0 and mat.sigyt12 == 78.0
        mat.sigyc12 = 76.0
        assert mat.sig_12yc == 76.0 and mat.sigyc12 == 76.0
        mat.sigyt23 = 49.0
        assert mat.sig_23yt == 49.0 and mat.sigyt23 == 49.0
        mat.sigyc23 = 48.0
        assert mat.sig_23yc == 48.0 and mat.sigyc23 == 48.0

        mat.sigyt3 = 51.0
        assert mat.sig_3yt == 51.0 and mat.sigyt3 == 51.0
        mat.sigyc3 = 192.0
        assert mat.sig_3yc == 192.0 and mat.sigyc3 == 192.0
        mat.sigyt13 = 77.0
        assert mat.sig_13yt == 77.0 and mat.sigyt13 == 77.0
        mat.sigyc13 = 75.0
        assert mat.sig_13yc == 75.0 and mat.sigyc13 == 75.0

        # Rate & flag
        mat.cc = 0.045
        assert mat.c == 0.045 and mat.cc == 0.045
        mat.strflag = 2
        assert mat.icc == 2 and mat.strflag == 2

    def test_pickle_restart_simulation(self):
        """Simulate restart file (.rst) pickle serialization of MatLaw12 and Model."""
        mat = MatLaw12(
            id=10, title="Pickle_Test",
            rho0=1.55e-9, rhor=1.55e-9,
            e11=140000.0, e22=10000.0, e33=10000.0,
            nu12=0.3, nu23=0.45, nu31=0.02,
            g12=5000.0, g23=3500.0, g31=5000.0,
            sig_t1=1800.0, sig_t2=40.0, sig_t3=40.0, delta=0.08,
            b=150.0, n=0.5, fmax=1200.0, wplaref=1.0,
            sig_1yt=2000.0, sig_2yt=50.0, sig_1yc=1500.0, sig_2yc=200.0,
            sig_12yt=80.0, sig_12yc=80.0, sig_23yt=50.0, sig_23yc=50.0,
            sig_3yt=50.0, sig_3yc=200.0, sig_13yt=80.0, sig_13yc=80.0,
            alpha=0.25, efib=180000.0, c=0.04, eps0=1.0, icc=1
        )
        data = pickle.dumps(mat)
        restored_mat = pickle.loads(data)
        assert restored_mat.id == mat.id
        assert restored_mat.rho0 == pytest.approx(mat.rho0)
        assert restored_mat.e11 == pytest.approx(mat.e11)
        assert restored_mat.b == pytest.approx(mat.b)
        assert restored_mat.sig_1yt == pytest.approx(mat.sig_1yt)
        assert restored_mat.cb == pytest.approx(mat.cb)

        # Whole model pickle
        model = Model()
        model.mat_law12s[10] = mat
        model_data = pickle.dumps(model)
        restored_model = pickle.loads(model_data)
        assert 10 in restored_model.mat_law12s
        assert restored_model.mat_3d_comps[10].e11 == pytest.approx(140000.0)


# ============================================================================
# 5. Starter Checks & ANCMSG Citations
# ============================================================================

class TestLaw12StarterChecks:
    """Starter checks verifying bounds and citing exact Fortran error numbers ANCMSG 305, 306, 307."""

    def test_starter_check_missing_density(self):
        """Density RHO <= 0 triggers error."""
        mat = Material(id=1, law=12, rho0=0.0, params={"rho": 0.0, "e11": 1e5, "e22": 1e4, "e33": 1e4})
        log = MessageLog()
        check_mat_law12(mat, log)
        assert any("initial density RHO must be > 0" in e for e in log.errors)

    def test_starter_check_moduli_ancmsg_306(self):
        """Zero or negative Young's modulus E11, E22, or E33 triggers ANCMSG 306."""
        # E11 = 0
        mat1 = Material(id=1, law=12, rho0=1.5e-9, params={"e11": 0.0, "e22": 1e4, "e33": 1e4})
        log1 = MessageLog()
        check_mat_law12(mat1, log1)
        assert any("Young's moduli E11, E22, E33 must be > 0" in e and "ANCMSG 306" in e for e in log1.errors)

        # E22 < 0
        mat2 = Material(id=2, law=12, rho0=1.5e-9, params={"e11": 1e5, "e22": -1e4, "e33": 1e4})
        log2 = MessageLog()
        check_mat_law12(mat2, log2)
        assert any("Young's moduli E11, E22, E33 must be > 0" in e and "ANCMSG 306" in e for e in log2.errors)

        # E33 <= 0
        mat3 = Material(id=3, law=12, rho0=1.5e-9, params={"e11": 1e5, "e22": 1e4, "e33": 0.0})
        log3 = MessageLog()
        check_mat_law12(mat3, log3)
        assert any("Young's moduli E11, E22, E33 must be > 0" in e and "ANCMSG 306" in e for e in log3.errors)

    def test_starter_check_detc_ancmsg_307(self):
        """Negative or zero compliance determinant DETC triggers ANCMSG 307."""
        # Extremely high Poisson ratios violate positive-definiteness of compliance matrix
        p = {
            "e11": 100000.0, "e22": 10000.0, "e33": 10000.0,
            "nu12": 0.95, "nu23": 0.95, "nu31": 0.95,
            "g12": 5000.0, "g23": 3000.0, "g31": 5000.0,
            "rho": 1.5e-9
        }
        mat = Material(id=1, law=12, rho0=1.5e-9, params=p)
        log = MessageLog()
        check_mat_law12(mat, log)
        assert any("compliance matrix determinant DETC must be > 0" in e and "ANCMSG 307" in e for e in log.errors)

    def test_starter_check_2d_analysis_ancmsg_305(self):
        """2D analysis (N2D > 0) with LAW12 triggers ANCMSG 305."""
        model = Model()
        model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
        model.n2d = 1  # 2D model
        mat = Material(id=1, law=12, rho0=1.5e-9, params={"e11": 1e5, "e22": 1e4, "e33": 1e4})
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [("bricks", FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert any("LAW12 is not supported for 2D analysis" in e and "ANCMSG 305" in e for e in log.errors)

    @pytest.mark.parametrize("non_solid", [
        "shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "trusses", "beams", "springs"
    ])
    def test_starter_check_non_solid_elements(self, non_solid: str):
        """Non-solid element types paired with LAW12 fail element slice compatibility checks."""
        model = Model()
        model.add_nodes(np.arange(1, 9), np.zeros((8, 3)))
        mat = Material(id=1, law=12, rho0=1.5e-9, params={"e11": 1e5, "e22": 1e4, "e33": 1e4})
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [(non_solid, FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert any(
            f"is not supported for {non_solid} elements" in e or f"is not ported for {non_solid} elements" in e
            for e in log.errors
        )

    def test_complete_roundtrip_write_parse(self, tmp_path: Path):
        """Complete end-to-end write -> parse -> verify pipeline for /MAT/LAW12."""
        orig_mat = MatLaw12(
            id=12, title="Full_Roundtrip",
            rho0=1.55e-9, rhor=1.55e-9,
            e11=142000.0, e22=10500.0, e33=10500.0,
            nu12=0.31, nu23=0.44, nu31=0.021,
            g12=5100.0, g23=3600.0, g31=5100.0,
            sig_t1=1820.0, sig_t2=41.0, sig_t3=41.0, delta=0.085,
            b=155.0, n=0.52, fmax=1250.0, wplaref=1.05,
            sig_1yt=2050.0, sig_2yt=52.0, sig_1yc=1520.0, sig_2yc=205.0,
            sig_12yt=82.0, sig_12yc=82.0, sig_23yt=51.0, sig_23yc=51.0,
            sig_3yt=52.0, sig_3yc=205.0, sig_13yt=82.0, sig_13yc=82.0,
            alpha=0.26, efib=185000.0, c=0.042, eps0=1.0, icc=1
        )

        deck_str = StarterDeck("FULL_ROUNDTRIP").mat_law12(orig_mat).write()
        model, log = _parse_deck_string(tmp_path, deck_str)
        assert len(log.errors) == 0

        parsed = model.mat_law12s[12]
        assert parsed.id == orig_mat.id
        assert parsed.rho0 == pytest.approx(orig_mat.rho0)
        assert parsed.rhor == pytest.approx(orig_mat.rhor)
        assert parsed.e11 == pytest.approx(orig_mat.e11)
        assert parsed.e22 == pytest.approx(orig_mat.e22)
        assert parsed.e33 == pytest.approx(orig_mat.e33)
        assert parsed.nu12 == pytest.approx(orig_mat.nu12)
        assert parsed.nu23 == pytest.approx(orig_mat.nu23)
        assert parsed.nu31 == pytest.approx(orig_mat.nu31)
        assert parsed.g12 == pytest.approx(orig_mat.g12)
        assert parsed.g23 == pytest.approx(orig_mat.g23)
        assert parsed.g31 == pytest.approx(orig_mat.g31)
        assert parsed.sig_t1 == pytest.approx(orig_mat.sig_t1)
        assert parsed.sig_t2 == pytest.approx(orig_mat.sig_t2)
        assert parsed.sig_t3 == pytest.approx(orig_mat.sig_t3)
        assert parsed.delta == pytest.approx(orig_mat.delta)
        assert parsed.b == pytest.approx(orig_mat.b)
        assert parsed.n == pytest.approx(orig_mat.n)
        assert parsed.fmax == pytest.approx(orig_mat.fmax)
        assert parsed.wplaref == pytest.approx(orig_mat.wplaref)
        assert parsed.sig_1yt == pytest.approx(orig_mat.sig_1yt)
        assert parsed.sig_2yt == pytest.approx(orig_mat.sig_2yt)
        assert parsed.sig_1yc == pytest.approx(orig_mat.sig_1yc)
        assert parsed.sig_2yc == pytest.approx(orig_mat.sig_2yc)
        assert parsed.sig_12yt == pytest.approx(orig_mat.sig_12yt)
        assert parsed.sig_12yc == pytest.approx(orig_mat.sig_12yc)
        assert parsed.sig_23yt == pytest.approx(orig_mat.sig_23yt)
        assert parsed.sig_23yc == pytest.approx(orig_mat.sig_23yc)
        assert parsed.sig_3yt == pytest.approx(orig_mat.sig_3yt)
        assert parsed.sig_3yc == pytest.approx(orig_mat.sig_3yc)
        assert parsed.sig_13yt == pytest.approx(orig_mat.sig_13yt)
        assert parsed.sig_13yc == pytest.approx(orig_mat.sig_13yc)
        assert parsed.alpha == pytest.approx(orig_mat.alpha)
        assert parsed.efib == pytest.approx(orig_mat.efib)
        assert parsed.c == pytest.approx(orig_mat.c)
        assert parsed.eps0 == pytest.approx(orig_mat.eps0)
        assert parsed.icc == orig_mat.icc
