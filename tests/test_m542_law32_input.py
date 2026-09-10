"""Tests for Milestone M542: /MAT/LAW32 and /MAT/HILL input layer.

Covers:
1. Card layout constants, aliases, and LAYOUTS dictionary registration.
2. Deck writer emitter StarterDeck.mat_law32 and alias mat_hill.
3. Fixed-format 5-card width alignment, comments, and field formatting.
4. CFG catalogue and mat_reader synonym resolution (/MAT/HILL vs /MAT/LAW32).
5. Fixed-format and free-format (whitespace and comma-separated) deck parsing into Material and MatLaw32 entities.
6. Default value injection matching matl32_hill.cfg and hm_read_mat32.F.
7. Deck roundtrip parsing (StarterDeck -> parse_starter_deck).
8. Starter validation checks: shell element acceptance, solid element rejection, parameter checks.
9. Backward compatibility with legacy M187 3-card tabulated format.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.mat_reader import catalogue
from pyradioss.input.cfg_catalogue import (
    LAW_MAP,
    LAW_SYNONYMS,
    KEYWORD_NAME_MAP,
    SYNONYMS,
    canonical_law_name,
    law_number,
)
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck, read_mat_law32, read_mat_hill
from pyradioss.model.model import Model
from pyradioss.model.entities import Material, MatLaw32, MatHill
from pyradioss.starter.checks import _ALLOWED_LAWS, check_model, check_mat_law32


def block_lines(text: str, header_prefix: str) -> list[str]:
    """The lines of the block starting with header_prefix (incl. header)."""
    lines = text.splitlines()
    out, active = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("/"):
            active = s.startswith(header_prefix)
        if active:
            out.append(ln)
    return out


def data_cards(lines: list[str]) -> list[str]:
    """Block lines minus header and comments (blank cards kept)."""
    return [ln for ln in lines[1:] if not ln.lstrip().startswith("#")]


def _parse_string(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    """Helper to write and parse a deck."""
    p = tmp_path / "DECK_0000.rad"
    p.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


# ============================================================================
# 1. Card Layouts & Aliases
# ============================================================================

class TestLaw32CardLayouts:
    """Card layout constants and LAYOUTS dictionary verification (M542)."""

    def test_constants_and_aliases(self):
        assert cl.MAT_LAW32_1 == (20, 20)
        assert cl.MAT_LAW32_2 == (20, 20)
        assert cl.MAT_LAW32_3 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW32_4 == (20, 20)
        assert cl.MAT_LAW32_5 == (20, 20, 20)

        # Aliases
        for i in range(1, 6):
            law_val = getattr(cl, f"MAT_LAW32_{i}")
            assert getattr(cl, f"MAT_LAW32_CFG_{i}") == law_val
            assert getattr(cl, f"MAT_HILL_{i}") == law_val
            assert getattr(cl, f"MAT_HILL_CFG_{i}") == law_val

    def test_layouts_dictionary(self):
        for i in range(1, 6):
            for prefix in ("MAT_LAW32_", "MAT_LAW32_CFG_", "MAT_HILL_", "MAT_HILL_CFG_"):
                key = f"{prefix}{i}"
                assert key in cl.LAYOUTS, f"Missing key {key} in LAYOUTS"
                expected = getattr(cl, f"MAT_LAW32_{i}")
                assert list(cl.LAYOUTS[key]) == list(expected)


# ============================================================================
# 2. CFG Catalogue & Synonyms
# ============================================================================

class TestLaw32CfgCatalogue:
    """CFG catalogue, synonym mapping, and schema tests."""

    def test_synonyms_and_law_map(self):
        assert LAW_MAP["LAW32"] == 32
        assert LAW_MAP["HILL"] == 32
        assert KEYWORD_NAME_MAP["LAW32"] == 32
        assert KEYWORD_NAME_MAP["HILL"] == 32

        assert LAW_SYNONYMS["LAW32"] == "LAW32"
        assert LAW_SYNONYMS["HILL"] == "LAW32"
        assert SYNONYMS["HILL"] == "LAW32"

        assert law_number("LAW32") == 32
        assert law_number("HILL") == 32
        assert law_number("MAT_HILL") == 32
        assert law_number("/MAT/HILL") == 32
        assert law_number("/MAT/LAW32") == 32
        assert law_number("32") == 32

        assert canonical_law_name("HILL") == "LAW32"
        assert canonical_law_name("LAW32") == "LAW32"
        assert canonical_law_name("/MAT/HILL") == "LAW32"
        assert canonical_law_name("/MAT/LAW32") == "LAW32"
        assert canonical_law_name("32") == "LAW32"

    def test_catalogue_schema(self):
        cat = catalogue()
        schema_law = cat.schema("LAW32")
        schema_hill = cat.schema("HILL")
        assert schema_law is not None
        assert schema_hill is not None
        assert schema_law.law_number == 32
        assert schema_hill.law_number == 32

        expected_attrs = [
            "MAT_RHO", "Refer_Rho", "MAT_E", "MAT_NU",
            "MAT_SIGY", "MAT_BETA", "MAT_HARD", "MAT_EPS", "MAT_SIG",
            "MAT_SRP", "MAT_SRC", "MAT_R00", "MAT_R45", "MAT_R90",
        ]
        for attr in expected_attrs:
            assert attr in schema_law.attributes, f"Missing attribute {attr} in schema"


# ============================================================================
# 3. Deck Writer
# ============================================================================

class TestLaw32DeckWriter:
    """StarterDeck emitter tests for /MAT/LAW32 and /MAT/HILL."""

    def test_emit_mat_law32_all_cards(self):
        deck = StarterDeck("M542")
        deck.mat_law32(
            mat_id=1,
            rho=7.85e-9,
            e=210000.0,
            nu=0.3,
            sigy=350.0,
            beta=150.0,
            hard=0.45,
            eps=0.25,
            sig=550.0,
            srp=1.0e-3,
            src=0.15,
            r00=1.8,
            r45=1.2,
            r90=2.1,
            title="STEEL_HILL_48",
            rhor=7.80e-9,
        )
        rendered = deck.render()
        lines = block_lines(rendered, "/MAT/LAW32/1")
        assert len(lines) >= 12
        assert lines[0] == "/MAT/LAW32/1"
        assert lines[1] == "STEEL_HILL_48"

        # Check header comments matching matl32_hill.cfg
        assert any("#              RHO_I               RHO_O" in ln for ln in lines)
        assert any("#                  E                  NU" in ln for ln in lines)
        assert any("#                  A           EPSILON_0                   n             EPS_max           SIGMA_max" in ln for ln in lines)
        assert any("#          EPS_DOT_0                   m" in ln for ln in lines)
        assert any("#                r00                 r45                 r90" in ln for ln in lines)

        cards = data_cards(lines)
        # 5 data cards + title card = 6 cards total (cards[0] is title)
        assert len(cards) == 6

        # Card 1: rho, rhor (%20lg%20lg = 40 chars)
        assert len(cards[1]) == 40
        assert math.isclose(float(cards[1][:20]), 7.85e-9, rel_tol=1e-5)
        assert math.isclose(float(cards[1][20:40]), 7.80e-9, rel_tol=1e-5)

        # Card 2: e, nu (%20lg%20lg = 40 chars)
        assert len(cards[2]) == 40
        assert math.isclose(float(cards[2][:20]), 210000.0, rel_tol=1e-5)
        assert math.isclose(float(cards[2][20:40]), 0.3, rel_tol=1e-5)

        # Card 3: sigy, beta, hard, eps, sig (%20lg*5 = 100 chars)
        assert len(cards[3]) == 100
        assert math.isclose(float(cards[3][:20]), 350.0, rel_tol=1e-5)
        assert math.isclose(float(cards[3][20:40]), 150.0, rel_tol=1e-5)
        assert math.isclose(float(cards[3][40:60]), 0.45, rel_tol=1e-5)
        assert math.isclose(float(cards[3][60:80]), 0.25, rel_tol=1e-5)
        assert math.isclose(float(cards[3][80:100]), 550.0, rel_tol=1e-5)

        # Card 4: srp, src (%20lg%20lg = 40 chars)
        assert len(cards[4]) == 40
        assert math.isclose(float(cards[4][:20]), 1.0e-3, rel_tol=1e-5)
        assert math.isclose(float(cards[4][20:40]), 0.15, rel_tol=1e-5)

        # Card 5: r00, r45, r90 (%20lg*3 = 60 chars)
        assert len(cards[5]) == 60
        assert math.isclose(float(cards[5][:20]), 1.8, rel_tol=1e-5)
        assert math.isclose(float(cards[5][20:40]), 1.2, rel_tol=1e-5)
        assert math.isclose(float(cards[5][40:60]), 2.1, rel_tol=1e-5)

    def test_emit_mat_hill_alias(self):
        deck = StarterDeck("M542")
        deck.mat_hill(
            mat_id=2,
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            sigy=250.0,
            title="ALU_HILL",
        )
        rendered = deck.render()
        lines = block_lines(rendered, "/MAT/HILL/2")
        assert len(lines) >= 7
        assert lines[0] == "/MAT/HILL/2"
        assert lines[1] == "ALU_HILL"
        assert any("#              RHO_I" in ln for ln in lines)

    def test_emit_mat_law32_with_positional_title(self):
        deck = StarterDeck("M542")
        deck.mat_law32(
            3,
            "POSITIONAL_TITLE",
            7.8e-9,
            210000.0,
            0.3,
            300.0,
            100.0,
            0.5,
            0.2,
            500.0,
            1.0,
            0.0,
            1.5,
            1.2,
            1.8,
        )
        rendered = deck.render()
        lines = block_lines(rendered, "/MAT/LAW32/3")
        assert lines[0] == "/MAT/LAW32/3"
        assert lines[1] == "POSITIONAL_TITLE"
        cards = data_cards(lines)
        assert math.isclose(float(cards[1][:20]), 7.8e-9, rel_tol=1e-5)
        assert math.isclose(float(cards[2][:20]), 210000.0, rel_tol=1e-5)
        assert math.isclose(float(cards[3][:20]), 300.0, rel_tol=1e-5)


# ============================================================================
# 4. Keyword Parsing: Fixed and Free Format
# ============================================================================

class TestLaw32KeywordParsing:
    """Fixed-format, free-format, alias resolution, and default injection tests."""

    def test_parse_fixed_format_5card(self, tmp_path: Path):
        deck = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/LAW32/10
Sheet Metal Hill Anisotropic
#              RHO_I               RHO_O
             7.85E-9              7.80E-9
#                  E                  NU
             210000.                 0.3
#                  A           EPSILON_0                   n             EPS_max           SIGMA_max
                350.                150.                0.45                0.25                550.
#          EPS_DOT_0                   m
              0.0010                0.15
#                r00                 r45                 r90
                 1.8                 1.2                 2.1
"""
        model, log = _parse_string(tmp_path, deck)
        assert not log.errors
        assert 10 in model.mat_law32s
        assert 10 in model.materials

        mat32 = model.mat_law32s[10]
        assert isinstance(mat32, MatLaw32)
        assert mat32.id == 10
        assert mat32.title == "Sheet Metal Hill Anisotropic"
        assert mat32.rho0 == pytest.approx(7.85e-9)
        assert mat32.rhor == pytest.approx(7.80e-9)
        assert mat32.e == pytest.approx(210000.0)
        assert mat32.nu == pytest.approx(0.3)
        assert mat32.sigy == pytest.approx(350.0)
        assert mat32.beta == pytest.approx(150.0)
        assert mat32.hard == pytest.approx(0.45)
        assert mat32.eps == pytest.approx(0.25)
        assert mat32.sig == pytest.approx(550.0)
        assert mat32.srp == pytest.approx(1.0e-3)
        assert mat32.src == pytest.approx(0.15)
        assert mat32.r00 == pytest.approx(1.8)
        assert mat32.r45 == pytest.approx(1.2)
        assert mat32.r90 == pytest.approx(2.1)

        # Mathematical alias properties on MatLaw32
        assert mat32.a == pytest.approx(350.0)
        assert mat32.b == pytest.approx(150.0)
        assert mat32.n == pytest.approx(0.45)
        assert mat32.eps_max == pytest.approx(0.25)
        assert mat32.sig_max == pytest.approx(550.0)
        assert mat32.eps0 == pytest.approx(1.0e-3)
        assert mat32.eps_dot_0 == pytest.approx(1.0e-3)
        assert mat32.m == pytest.approx(0.15)

        # Generic Material entity
        mat = model.materials[10]
        assert mat.law == 32
        assert mat.law_name == "LAW32"
        assert mat.rho0 == pytest.approx(7.85e-9)
        assert mat.params["E"] == pytest.approx(210000.0)
        assert mat.params["nu"] == pytest.approx(0.3)
        assert mat.params["A"] == pytest.approx(350.0)
        assert mat.params["B"] == pytest.approx(150.0)
        assert mat.params["n"] == pytest.approx(0.45)
        assert mat.params["eps_max"] == pytest.approx(0.25)
        assert mat.params["sig_max"] == pytest.approx(550.0)
        assert mat.params["eps0"] == pytest.approx(1.0e-3)
        assert mat.params["m"] == pytest.approx(0.15)
        assert mat.params["r00"] == pytest.approx(1.8)
        assert mat.params["r45"] == pytest.approx(1.2)
        assert mat.params["r90"] == pytest.approx(2.1)

        # Check Hill coefficients computed
        r = 0.25 * (1.8 + 2.0 * 1.2 + 2.1)
        h = r / (1.0 + r)
        a11 = h * (1.0 + 1.0 / 1.8)
        assert mat.params["A11"] == pytest.approx(a11)

    def test_parse_free_format_whitespace(self, tmp_path: Path):
        deck = """\
/MAT/LAW32/20
Free Format Hill
7.85e-9 7.80e-9
210000.0 0.3
350.0 150.0 0.45 0.25 550.0
0.001 0.15
1.8 1.2 2.1
"""
        model, log = _parse_string(tmp_path, deck)
        assert not log.errors
        assert 20 in model.mat_law32s
        m = model.mat_law32s[20]
        assert m.rho0 == pytest.approx(7.85e-9)
        assert m.e == pytest.approx(210000.0)
        assert m.nu == pytest.approx(0.3)
        assert m.sigy == pytest.approx(350.0)
        assert m.r00 == pytest.approx(1.8)
        assert m.r45 == pytest.approx(1.2)
        assert m.r90 == pytest.approx(2.1)

    def test_parse_free_format_comma_separated(self, tmp_path: Path):
        deck = """\
/MAT/LAW32/30
Comma Format Hill
7.85e-9, 7.80e-9
210000.0, 0.3
350.0, 150.0, 0.45, 0.25, 550.0
0.001, 0.15
1.8, 1.2, 2.1
"""
        model, log = _parse_string(tmp_path, deck)
        assert not log.errors
        assert 30 in model.mat_law32s
        m = model.mat_law32s[30]
        assert m.rho0 == pytest.approx(7.85e-9)
        assert m.sigy == pytest.approx(350.0)
        assert m.r00 == pytest.approx(1.8)

    def test_alias_resolution(self, tmp_path: Path):
        deck = """\
/MAT/HILL/40
Hill Keyword Alias
7.8e-9
205000.0 0.29
320.0 100.0 0.4 0.2 500.0
1.0 0.0
1.5 1.1 1.9
"""
        model, log = _parse_string(tmp_path, deck)
        assert not log.errors
        assert 40 in model.mat_law32s
        assert 40 in model.mat_hills
        assert 40 in model.materials

        mat = model.materials[40]
        assert mat.law == 32
        assert mat.params["E"] == pytest.approx(205000.0)
        assert mat.params["r00"] == pytest.approx(1.5)

    def test_default_value_injection(self, tmp_path: Path):
        deck = """\
/MAT/LAW32/50
Defaults Test
7.85e-9
210000.0 0.5
0 0 0 0 0
0 0
0 0 0
"""
        model, log = _parse_string(tmp_path, deck)
        assert not log.errors
        m = model.mat_law32s[50]
        # rhor defaults to rho0
        assert m.rhor == pytest.approx(7.85e-9)
        # nu=0.5 clamps to 0.499999
        assert m.nu == pytest.approx(0.499999)
        # sigy (A) defaults to 1e30
        assert m.sigy == pytest.approx(1.0e30)
        # beta (B) defaults to 0.0
        assert m.beta == pytest.approx(0.0)
        # hard (n) defaults to 1.0
        assert m.hard == pytest.approx(1.0)
        # eps_max and sig_max default to 1e30
        assert m.eps == pytest.approx(1.0e30)
        assert m.sig == pytest.approx(1.0e30)
        # srp defaults to 1.0
        assert m.srp == pytest.approx(1.0)
        assert m.src == pytest.approx(0.0)
        # Lankford parameters default to 1.0 (isotropic Hill)
        assert m.r00 == pytest.approx(1.0)
        assert m.r45 == pytest.approx(1.0)
        assert m.r90 == pytest.approx(1.0)


# ============================================================================
# 5. Roundtrip Parsing
# ============================================================================

class TestLaw32Roundtrip:
    """StarterDeck emission -> parse_starter_deck roundtrip testing."""

    def test_roundtrip_mat_law32(self, tmp_path: Path):
        deck = StarterDeck("M542_ROUNDTRIP")
        deck.mat_law32(
            mat_id=101,
            rho=7.8e-9,
            e=205000.0,
            nu=0.28,
            sigy=310.0,
            beta=120.0,
            hard=0.38,
            eps=0.22,
            sig=480.0,
            srp=0.005,
            src=0.12,
            r00=1.65,
            r45=1.15,
            r90=1.95,
            title="ROUNDTRIP_STEEL",
            rhor=7.75e-9,
        )
        rendered = deck.render()
        model, log = _parse_string(tmp_path, rendered)
        assert not log.errors
        assert 101 in model.mat_law32s
        m = model.mat_law32s[101]

        assert m.id == 101
        assert m.title == "ROUNDTRIP_STEEL"
        assert m.rho0 == pytest.approx(7.8e-9)
        assert m.rhor == pytest.approx(7.75e-9)
        assert m.e == pytest.approx(205000.0)
        assert m.nu == pytest.approx(0.28)
        assert m.sigy == pytest.approx(310.0)
        assert m.beta == pytest.approx(120.0)
        assert m.hard == pytest.approx(0.38)
        assert m.eps == pytest.approx(0.22)
        assert m.sig == pytest.approx(480.0)
        assert m.srp == pytest.approx(0.005)
        assert m.src == pytest.approx(0.12)
        assert m.r00 == pytest.approx(1.65)
        assert m.r45 == pytest.approx(1.15)
        assert m.r90 == pytest.approx(1.95)

    def test_roundtrip_mat_hill(self, tmp_path: Path):
        deck = StarterDeck("M542_HILL_ROUNDTRIP")
        deck.mat_hill(
            mat_id=102,
            rho=2.7e-9,
            e=69000.0,
            nu=0.33,
            sigy=220.0,
            beta=80.0,
            hard=0.25,
            eps=0.18,
            sig=340.0,
            r00=0.85,
            r45=0.65,
            r90=0.95,
            title="ROUNDTRIP_ALU",
        )
        rendered = deck.render()
        model, log = _parse_string(tmp_path, rendered)
        assert not log.errors
        assert 102 in model.mat_law32s
        m = model.mat_law32s[102]
        assert m.id == 102
        assert m.rho0 == pytest.approx(2.7e-9)
        assert m.e == pytest.approx(69000.0)
        assert m.sigy == pytest.approx(220.0)
        assert m.r00 == pytest.approx(0.85)


# ============================================================================
# 6. Starter Validation Checks
# ============================================================================

class TestLaw32StarterChecks:
    """Starter checks for LAW32/HILL element compatibility and parameter bounds."""

    def test_parameter_bounds_valid(self):
        log = MessageLog()
        mat = MatLaw32(
            id=1, rho0=7.8e-9, e=210000.0, nu=0.3,
            sigy=350.0, beta=100.0, hard=0.45,
            eps=0.25, sig=500.0, srp=1.0, src=0.1,
            r00=1.5, r45=1.2, r90=1.8
        )
        check_mat_law32(mat, log)
        assert not log.errors

    def test_parameter_bounds_invalid(self):
        log = MessageLog()
        # Invalid: rho0 <= 0, e <= 0, nu >= 0.5, sigy <= 0, hard > 1.0, srp <= 0, r00 <= 0
        mat = MatLaw32(
            id=2, rho0=-1.0, e=-100.0, nu=0.6,
            sigy=0.0, beta=0.0, hard=1.5,
            eps=0.0, sig=0.0, srp=-1.0, src=0.0,
            r00=0.0, r45=-0.5, r90=0.0
        )
        check_mat_law32(mat, log)
        assert log.errors
        error_msgs = " ".join(log.errors)
        assert "initial density RHO must be > 0" in error_msgs
        assert "Young's modulus E must be > 0" in error_msgs
        assert "Poisson's ratio NU must be in [0, 0.5)" in error_msgs
        assert "yield stress A (SIGY) must be > 0" in error_msgs
        assert "hardening exponent n must be <= 1.0" in error_msgs
        assert "reference strain rate EPS_DOT_0" in error_msgs
        assert "Lankford parameter r00 must be > 0" in error_msgs
        assert "Lankford parameter r45 must be > 0" in error_msgs
        assert "Lankford parameter r90 must be > 0" in error_msgs

        )
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [("bricks", FakeGroup())]

        log = MessageLog()
        check_model(model, log)
        assert log.errors
        error_msgs = " ".join(log.errors)
        assert "/MAT/LAW32/1 (/MAT/HILL) is not supported for bricks elements" in error_msgs


# ============================================================================
# 7. Backward Compatibility with M187
# ============================================================================

class TestLaw32LegacyM187:
    """Ensure legacy 3-card tabulated format from M187 still parses without error."""

    def test_legacy_3card_fixed(self, tmp_path: Path):
        deck = """\
/MAT/LAW32/701
Hill Tabulated Anisotropic
              2.7E-9              70000.              68000.              65000.                 0.3                0.32                0.29
              26000.              25000.              24000.
       111       122       133       112       123       131
"""
        model, log = _parse_string(tmp_path, deck)
        assert not log.errors
        assert 701 in model.mat_law32s
        m = model.mat_law32s[701]
        assert m.rho0 == pytest.approx(2.7e-9)
        assert m.e1 == pytest.approx(70000.0)
        assert m.e2 == pytest.approx(68000.0)
        assert m.e3 == pytest.approx(65000.0)
        assert m.nu12 == pytest.approx(0.3)
        assert m.nu23 == pytest.approx(0.32)
        assert m.nu31 == pytest.approx(0.29)
        assert m.g12 == pytest.approx(26000.0)
        assert m.g23 == pytest.approx(25000.0)
        assert m.g31 == pytest.approx(24000.0)
        assert m.fct_id11 == 111
        assert m.fct_id22 == 122
        assert m.fct_id33 == 133
        assert m.fct_id12 == 112
        assert m.fct_id23 == 123
        assert m.fct_id31 == 131
