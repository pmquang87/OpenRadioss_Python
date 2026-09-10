"""Comprehensive input & roundtrip audit test suite for /MAT/LAW32 and /MAT/HILL (M542).

Auditor 4: LAW32 Input & Roundtrip Auditor (Wave 2 for M542).

Rigorously audits:
1. Card formatting precision audit:
   - Fixed-format width checks for all 5 cards (20-character columns) matching matl32_hill.cfg.
   - Card 1: RHO_I, optional RHO_O (20, 20)
   - Card 2: E, NU (20, 20)
   - Card 3: A, B, n, EPS_max, SIGMA_max (20, 20, 20, 20, 20)
   - Card 4: EPS_DOT_0, m (20, 20)
   - Card 5: r00, r45, r90 (20, 20, 20)
2. Roundtrip fidelity:
   - Complete StarterDeck with /MAT/LAW32 and /MAT/HILL across all fields.
   - 100% parameter preservation across MatLaw32 dataclass and Material.params.
   - Distinct Refer_Rho (rhor != rho) and omitted/equal Refer_Rho (rhor == rho).
   - Regular, blank, and numeric title card roundtrip.
3. Free-format robustness:
   - Comma-separated without spaces (e.g. 7.85e-9,7.80e-9).
   - Comma-separated with spaces (e.g. 7.85e-9, 7.80e-9).
   - Whitespace-separated free format.
   - Consecutive commas (empty fields) preserving defaults across all cards.
   - Cards with trailing comments (# and $) and trailing blanks.
   - Free-format deck without title card (numeric card detection).
4. Synonym & alias resolution:
   - Exact equivalence of /MAT/HILL and /MAT/LAW32.
   - Header variants: /MAT/LAW32/id, /MAT/HILL/id, /LAW32/id, /HILL/id.
   - Aliases in model: model.mat_hills is model.mat_law32s.
   - Dataclass aliases: MatHill is MatLaw32.
   - Catalogue schema and synonym lookups.
5. Defensive edge cases:
   - Missing optional cards or trailing cards (early EOF).
   - Default value injection when cards omit fields (default b=0, hard=1.0, eps=1e30, sig=1e30, srp=1.0, r=1.0).
   - Clamping nu=0.5 -> 0.499999.
   - Rate-independent srp default when src=0.0 (hm_read_mat32.F line 147).
   - Malformed inputs handled gracefully with clear errors or warnings.
   - Parameter bounds validation via check_mat_law32.
   - Element compatibility validation (shells accepted, solids rejected).
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input.card_layouts import split_fixed, fmt_float, blank, BLANK_CARD
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import Card, read_deck
from pyradioss.input.mat_reader import catalogue
from pyradioss.input.cfg_catalogue import (
    LAW_MAP,
    LAW_SYNONYMS,
    KEYWORD_NAME_MAP,
    SYNONYMS,
    canonical_law_name,
    law_number,
)
from pyradioss.input.starter_keywords import parse_starter_deck, _is_numeric_card, read_mat_law32, read_mat_hill
from pyradioss.model.model import Model
from pyradioss.model.entities import Material, MatLaw32, MatHill
from pyradioss.starter.checks import check_mat_law32, check_model, _ALLOWED_LAWS


def _extract_block_cards(deck_text: str, header: str) -> list[str]:
    """Extract raw lines of a block without header and comments."""
    lines = deck_text.splitlines()
    out = []
    active = False
    for line in lines:
        s = line.strip()
        if s.startswith("/"):
            active = s.startswith(header)
            continue
        if active:
            if not s.startswith("#") and not s.startswith("$"):
                out.append(line)
    return out


def _parse_deck_string(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    """Helper to write and parse a deck."""
    p = tmp_path / "AUDIT_DECK_0000.rad"
    p.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


class TestCardFormattingPrecisionAudit:
    """Card formatting precision audit for all 5 cards of /MAT/LAW32 and /MAT/HILL."""

    def test_all_5_card_layouts_constants(self):
        """Audit field layout tuple definitions against radioss110 CFG matl32_hill.cfg."""
        assert cl.MAT_LAW32_1 == (20, 20)
        assert cl.MAT_LAW32_2 == (20, 20)
        assert cl.MAT_LAW32_3 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW32_4 == (20, 20)
        assert cl.MAT_LAW32_5 == (20, 20, 20)

        for i in range(1, 6):
            for pfx in ("MAT_LAW32_", "MAT_LAW32_CFG_", "MAT_HILL_", "MAT_HILL_CFG_"):
                key = f"{pfx}{i}"
                assert key in cl.LAYOUTS, f"Missing {key} in LAYOUTS"
                expected = getattr(cl, f"MAT_LAW32_{i}")
                assert list(cl.LAYOUTS[key]) == list(expected)

    def test_fixed_format_field_widths_with_rhor(self):
        """Audit fixed-format width, column alignments, and field slicing when RHO_O is present."""
        deck = StarterDeck("AUDIT_FIXED_WIDTH_RHOR")
        deck.mat_law32(
            mat_id=1,
            rho=7.85e-9,
            rhor=7.80e-9,
            e=210000.0,
            nu=0.3,
            sigy=350.0,
            beta=150.0,
            hard=0.45,
            eps=0.25,
            sig=550.0,
            srp=0.001,
            src=0.15,
            r00=1.8,
            r45=1.2,
            r90=2.1,
            title="STEEL_HILL_FIXED",
        )
        rendered = deck.render()
        cards = _extract_block_cards(rendered, "/MAT/LAW32/1")

        assert cards[0] == "STEEL_HILL_FIXED"
        card1, card2, card3, card4, card5 = cards[1:6]

        assert len(card1) == 40
        assert len(card2) == 40
        assert len(card3) == 100
        assert len(card4) == 40
        assert len(card5) == 60

        c1_fields = split_fixed(card1, cl.LAYOUTS["MAT_LAW32_1"])
        assert len(c1_fields) == 2
        assert math.isclose(float(c1_fields[0]), 7.85e-9, rel_tol=1e-5)
        assert math.isclose(float(c1_fields[1]), 7.80e-9, rel_tol=1e-5)

        c2_fields = split_fixed(card2, cl.LAYOUTS["MAT_LAW32_2"])
        assert len(c2_fields) == 2
        assert math.isclose(float(c2_fields[0]), 210000.0, rel_tol=1e-5)
        assert math.isclose(float(c2_fields[1]), 0.3, rel_tol=1e-5)

        c3_fields = split_fixed(card3, cl.LAYOUTS["MAT_LAW32_3"])
        assert len(c3_fields) == 5
        assert math.isclose(float(c3_fields[0]), 350.0, rel_tol=1e-5)
        assert math.isclose(float(c3_fields[1]), 150.0, rel_tol=1e-5)
        assert math.isclose(float(c3_fields[2]), 0.45, rel_tol=1e-5)
        assert math.isclose(float(c3_fields[3]), 0.25, rel_tol=1e-5)
        assert math.isclose(float(c3_fields[4]), 550.0, rel_tol=1e-5)

        c4_fields = split_fixed(card4, cl.LAYOUTS["MAT_LAW32_4"])
        assert len(c4_fields) == 2
        assert math.isclose(float(c4_fields[0]), 0.001, rel_tol=1e-5)
        assert math.isclose(float(c4_fields[1]), 0.15, rel_tol=1e-5)

        c5_fields = split_fixed(card5, cl.LAYOUTS["MAT_LAW32_5"])
        assert len(c5_fields) == 3
        assert math.isclose(float(c5_fields[0]), 1.8, rel_tol=1e-5)
        assert math.isclose(float(c5_fields[1]), 1.2, rel_tol=1e-5)
        assert math.isclose(float(c5_fields[2]), 2.1, rel_tol=1e-5)

    def test_fixed_format_field_widths_without_rhor(self):
        """Audit that Card 1 emits exactly 20 characters when RHO_O is omitted."""
        deck = StarterDeck("AUDIT_NO_RHOR")
        deck.mat_law32(
            mat_id=2,
            rho=2.7e-9,
            e=70000.0,
            nu=0.33,
            sigy=220.0,
            title="ALU_NO_RHOR",
        )
        rendered = deck.render()
        cards = _extract_block_cards(rendered, "/MAT/LAW32/2")
        card1 = cards[1]
        assert len(card1) == 20
        assert math.isclose(float(card1[:20]), 2.7e-9, rel_tol=1e-5)

    def test_comment_lines_match_cfg_cards(self):
        """Audit that generated comment cards match radioss110/radioss90 CFG exactly."""
        deck = StarterDeck("AUDIT_COMMENTS")
        deck.mat_law32(mat_id=3, rho=7.8e-9, rhor=7.7e-9, e=200000.0, nu=0.3)
        rendered = deck.render()
        lines = rendered.splitlines()

        assert any("#              RHO_I               RHO_O" in ln for ln in lines)
        assert any("#                  E                  NU" in ln for ln in lines)
        assert any("#                  A           EPSILON_0                   n             EPS_max           SIGMA_max" in ln for ln in lines)
        assert any("#          EPS_DOT_0                   m" in ln for ln in lines)
        assert any("#                r00                 r45                 r90" in ln for ln in lines)


class TestRoundtripFidelityAudit:
    """Rigorous roundtrip fidelity audit across all fields, aliases, and titles."""

    @pytest.mark.parametrize("use_hill_alias", [False, True])
    @pytest.mark.parametrize("distinct_rhor", [False, True])
    @pytest.mark.parametrize("title_variant", ["EXPLICIT_HILL_STEEL", "", "12345"])
    def test_complete_roundtrip_fidelity(
        self, tmp_path: Path, use_hill_alias: bool, distinct_rhor: bool, title_variant: str
    ):
        """Assert 100% parameter preservation across all fields with both aliases and title styles."""
        mat_id = 42
        rho0 = 7.85e-9
        rhor = 7.80e-9 if distinct_rhor else 7.85e-9
        e = 210000.0
        nu = 0.31
        sigy = 360.0
        beta = 140.0
        hard = 0.42
        eps = 0.28
        sig = 560.0
        srp = 0.002
        src = 0.18
        r00 = 1.75
        r45 = 1.25
        r90 = 2.05

        full_params = {
            "mat_id": mat_id,
            "rho": rho0,
            "rhor": rhor if distinct_rhor else None,
            "e": e,
            "nu": nu,
            "sigy": sigy,
            "beta": beta,
            "hard": hard,
            "eps": eps,
            "sig": sig,
            "srp": srp,
            "src": src,
            "r00": r00,
            "r45": r45,
            "r90": r90,
            "title": title_variant,
        }

        deck = StarterDeck("AUDIT_ROUNDTRIP")
        if use_hill_alias:
            deck.mat_hill(**full_params)
        else:
            deck.mat_law32(**full_params)

        deck_path = tmp_path / f"roundtrip_{use_hill_alias}_{distinct_rhor}_{title_variant}.rad"
        deck.write(deck_path)

        model = parse_starter_deck(str(deck_path))

        assert mat_id in model.materials
        assert mat_id in model.mat_law32s
        assert mat_id in model.mat_hills

        m32 = model.mat_law32s[mat_id]
        p = model.materials[mat_id].params

        # MatLaw32 dataclass field verification
        assert m32.id == mat_id
        assert m32.title == title_variant
        assert math.isclose(m32.rho0, rho0, rel_tol=1e-5)
        assert math.isclose(m32.rhor, rhor, rel_tol=1e-5)
        assert math.isclose(m32.e, e, rel_tol=1e-5)
        assert math.isclose(m32.nu, nu, rel_tol=1e-5)
        assert math.isclose(m32.sigy, sigy, rel_tol=1e-5)
        assert math.isclose(m32.beta, beta, rel_tol=1e-5)
        assert math.isclose(m32.hard, hard, rel_tol=1e-5)
        assert math.isclose(m32.eps, eps, rel_tol=1e-5)
        assert math.isclose(m32.sig, sig, rel_tol=1e-5)
        assert math.isclose(m32.srp, srp, rel_tol=1e-5)
        assert math.isclose(m32.src, src, rel_tol=1e-5)
        assert math.isclose(m32.r00, r00, rel_tol=1e-5)
        assert math.isclose(m32.r45, r45, rel_tol=1e-5)
        assert math.isclose(m32.r90, r90, rel_tol=1e-5)

        # Dataclass property aliases
        assert math.isclose(m32.a, sigy, rel_tol=1e-5)
        assert math.isclose(m32.b, beta, rel_tol=1e-5)
        assert math.isclose(m32.n, hard, rel_tol=1e-5)
        assert math.isclose(m32.eps_max, eps, rel_tol=1e-5)
        assert math.isclose(m32.sig_max, sig, rel_tol=1e-5)
        assert math.isclose(m32.eps0, srp, rel_tol=1e-5)
        assert math.isclose(m32.eps_dot_0, srp, rel_tol=1e-5)
        assert math.isclose(m32.m, src, rel_tol=1e-5)

        # Material.params dictionary verification
        assert math.isclose(p["rho"], rho0, rel_tol=1e-5)
        assert math.isclose(p["rhor"], rhor, rel_tol=1e-5)
        assert math.isclose(p["E"], e, rel_tol=1e-5)
        assert math.isclose(p["nu"], nu, rel_tol=1e-5)
        assert math.isclose(p["A"], sigy, rel_tol=1e-5)
        assert math.isclose(p["B"], beta, rel_tol=1e-5)
        assert math.isclose(p["n"], hard, rel_tol=1e-5)
        assert math.isclose(p["eps_max"], eps, rel_tol=1e-5)
        assert math.isclose(p["sig_max"], sig, rel_tol=1e-5)
        assert math.isclose(p["srp"], srp, rel_tol=1e-5)
        assert math.isclose(p["src"], src, rel_tol=1e-5)
        assert math.isclose(p["r00"], r00, rel_tol=1e-5)
        assert math.isclose(p["r45"], r45, rel_tol=1e-5)
        assert math.isclose(p["r90"], r90, rel_tol=1e-5)

        # Hill anisotropic yield surface coefficients
        r_mean = 0.25 * (r00 + 2.0 * r45 + r90)
        h_val = r_mean / (1.0 + r_mean)
        a11 = h_val * (1.0 + 1.0 / r00)
        a22 = h_val * (1.0 + 1.0 / r90)
        a1122 = 2.0 * h_val
        a12 = 2.0 * h_val * (r45 + 0.5) * (1.0 / r00 + 1.0 / r90)

        assert math.isclose(p["A11"], a11, rel_tol=1e-5)
        assert math.isclose(p["A22"], a22, rel_tol=1e-5)
        assert math.isclose(p["A1122"], a1122, rel_tol=1e-5)
        assert math.isclose(p["A12"], a12, rel_tol=1e-5)


class TestFreeFormatRobustnessAudit:
    """Rigorous free-format parsing audit: comma, whitespace, consecutive commas, comments."""

    def test_comma_separated_without_spaces(self, tmp_path: Path):
        """Audit free-format parsing with compact comma separation without spaces."""
        deck_text = """/BEGIN
COMMA_COMPACT
2022  0
/MAT/LAW32/11
COMMA_COMPACT_TITLE
7.85e-9,7.80e-9
210000.0,0.3
350.0,150.0,0.45,0.25,550.0
0.001,0.15
1.8,1.2,2.1
/END
"""
        model, log = _parse_deck_string(tmp_path, deck_text)
        assert not log.errors
        assert 11 in model.mat_law32s
        m = model.mat_law32s[11]

        assert math.isclose(m.rho0, 7.85e-9, rel_tol=1e-5)
        assert math.isclose(m.rhor, 7.80e-9, rel_tol=1e-5)
        assert math.isclose(m.e, 210000.0, rel_tol=1e-5)
        assert math.isclose(m.nu, 0.3, rel_tol=1e-5)
        assert math.isclose(m.sigy, 350.0, rel_tol=1e-5)
        assert math.isclose(m.beta, 150.0, rel_tol=1e-5)
        assert math.isclose(m.hard, 0.45, rel_tol=1e-5)
        assert math.isclose(m.eps, 0.25, rel_tol=1e-5)
        assert math.isclose(m.sig, 550.0, rel_tol=1e-5)
        assert math.isclose(m.srp, 0.001, rel_tol=1e-5)
        assert math.isclose(m.src, 0.15, rel_tol=1e-5)
        assert math.isclose(m.r00, 1.8, rel_tol=1e-5)
        assert math.isclose(m.r45, 1.2, rel_tol=1e-5)
        assert math.isclose(m.r90, 2.1, rel_tol=1e-5)

    def test_comma_separated_with_spaces(self, tmp_path: Path):
        """Audit free-format parsing with comma + space separation."""
        deck_text = """/BEGIN
COMMA_SPACE
2022  0
/MAT/HILL/12
COMMA_SPACE_TITLE
7.85e-9, 7.80e-9
210000.0, 0.3
350.0, 150.0, 0.45, 0.25, 550.0
0.001, 0.15
1.8, 1.2, 2.1
/END
"""
        model, log = _parse_deck_string(tmp_path, deck_text)
        assert not log.errors
        assert 12 in model.mat_law32s
        m = model.mat_law32s[12]
        assert math.isclose(m.rho0, 7.85e-9, rel_tol=1e-5)
        assert math.isclose(m.e, 210000.0, rel_tol=1e-5)
        assert math.isclose(m.r00, 1.8, rel_tol=1e-5)

    def test_whitespace_separated_free_format(self, tmp_path: Path):
        """Audit free-format parsing with spaces and tabs."""
        deck_text = """/BEGIN
WHITESPACE_FREE
2022  0
/MAT/LAW32/13
WHITESPACE_TITLE
\t7.85e-9\t\t7.80e-9
210000.0   0.3
350.0   150.0   0.45   0.25   550.0
0.001   0.15
1.8   1.2   2.1
/END
"""
        model, log = _parse_deck_string(tmp_path, deck_text)
        assert not log.errors
        assert 13 in model.mat_law32s
        m = model.mat_law32s[13]
        assert math.isclose(m.rho0, 7.85e-9, rel_tol=1e-5)
        assert math.isclose(m.r45, 1.2, rel_tol=1e-5)

    def test_consecutive_commas_default_injection(self, tmp_path: Path):
        """Audit consecutive commas inserting empty fields taking exact defaults."""
        deck_text = """/BEGIN
CONSEC_COMMAS
2022  0
/MAT/LAW32/14
CONSEC_COMMAS_TITLE
7.85e-9
210000.0,0.3
350.0,,0.45,,550.0
,0.15
,,2.1
/END
"""
        model, log = _parse_deck_string(tmp_path, deck_text)
        assert not log.errors
        assert 14 in model.mat_law32s
        m = model.mat_law32s[14]

        assert math.isclose(m.rho0, 7.85e-9, rel_tol=1e-5)
        assert math.isclose(m.rhor, 7.85e-9, rel_tol=1e-5)
        assert math.isclose(m.e, 210000.0, rel_tol=1e-5)
        assert math.isclose(m.nu, 0.3, rel_tol=1e-5)
        assert math.isclose(m.sigy, 350.0, rel_tol=1e-5)
        assert math.isclose(m.beta, 0.0, rel_tol=1e-5)
        assert math.isclose(m.hard, 0.45, rel_tol=1e-5)
        assert math.isclose(m.eps, 1.0e30, rel_tol=1e-5)
        assert math.isclose(m.sig, 550.0, rel_tol=1e-5)
        assert math.isclose(m.srp, 1.0, rel_tol=1e-5)
        assert math.isclose(m.src, 0.15, rel_tol=1e-5)
        assert math.isclose(m.r00, 1.0, rel_tol=1e-5)
        assert math.isclose(m.r45, 1.0, rel_tol=1e-5)
        assert math.isclose(m.r90, 2.1, rel_tol=1e-5)

    def test_trailing_comments_and_blanks(self, tmp_path: Path):
        """Audit free-format and fixed cards with inline comments (# and $) and trailing whitespace."""
        deck_text = """/BEGIN
COMMENTS_TEST
2022  0
/MAT/LAW32/15
TRAILING COMMENTS AND BLANKS
7.85e-9, 7.80e-9 # initial and ref density
210000.0 0.3     $ young modulus and poisson ratio
350.0, 150.0, 0.45, 0.25, 550.0   # A, B, n, eps_max, sig_max
0.001 0.15       $ strain rate parameters
1.8, 1.2, 2.1    # Lankford coefficients
/END
"""
        model, log = _parse_deck_string(tmp_path, deck_text)
        assert not log.errors
        assert 15 in model.mat_law32s
        m = model.mat_law32s[15]
        assert math.isclose(m.rho0, 7.85e-9, rel_tol=1e-5)
        assert math.isclose(m.e, 210000.0, rel_tol=1e-5)
        assert math.isclose(m.sigy, 350.0, rel_tol=1e-5)
        assert math.isclose(m.srp, 0.001, rel_tol=1e-5)
        assert math.isclose(m.r90, 2.1, rel_tol=1e-5)

    def test_numeric_card_detection_with_commas(self):
        """Audit _is_numeric_card correctly recognizes comma-separated numeric cards without title."""
        assert _is_numeric_card(Card("7.85e-9,7.80e-9"))
        assert _is_numeric_card(Card("7.85e-9, 7.80e-9"))
        assert _is_numeric_card(Card("210000.0,0.3"))
        assert _is_numeric_card(Card("350.0,150.0,0.45,0.25,550.0"))
        assert not _is_numeric_card(Card("STEEL SHEET HILL"))
        assert not _is_numeric_card(Card("STEEL, HILL ANISOTROPIC"))

    def test_free_format_deck_without_title_card(self, tmp_path: Path):
        """Audit free-format deck starting directly with numeric cards (no title)."""
        deck_text = """/BEGIN
NO_TITLE_DECK
2022  0
/MAT/LAW32/16
7.85e-9, 7.80e-9
210000.0, 0.3
350.0, 150.0, 0.45, 0.25, 550.0
0.001, 0.15
1.8, 1.2, 2.1
/END
"""
        model, log = _parse_deck_string(tmp_path, deck_text)
        assert not log.errors
        assert 16 in model.mat_law32s
        m = model.mat_law32s[16]
        assert m.title == ""
        assert math.isclose(m.rho0, 7.85e-9, rel_tol=1e-5)
        assert math.isclose(m.e, 210000.0, rel_tol=1e-5)
        assert math.isclose(m.sigy, 350.0, rel_tol=1e-5)


class TestSynonymAndAliasResolutionAudit:
    """Verify /MAT/HILL and /MAT/LAW32 exact synonym and syntax handling."""

    def test_cfg_catalogue_synonyms(self):
        """Audit LAW_MAP, LAW_SYNONYMS, and catalogue schema mapping."""
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

        cat = catalogue()
        assert cat.schema("LAW32") is not None
        assert cat.schema("HILL") is not None
        assert cat.schema("HILL").law_number == 32
        assert cat.schema("LAW32").law_number == 32

    def test_model_dict_and_dataclass_aliases(self):
        """Audit that model.mat_hills is identical to model.mat_law32s."""
        m = Model()
        assert m.mat_hills is m.mat_law32s
        assert MatHill is MatLaw32

    @pytest.mark.parametrize("header", [
        "/MAT/LAW32/55",
        "/MAT/HILL/55",
        "/LAW32/55",
        "/HILL/55",
    ])
    def test_header_syntax_variants(self, tmp_path: Path, header: str):
        """Audit all 4 keyword header syntax variants for LAW32 / HILL."""
        deck_text = f"""/BEGIN
SYNONYM_TEST
2022  0
{header}
SYNONYM_TEST_MATERIAL
             7.85E-9             7.80E-9
             210000.                 0.3
                350.                150.                0.45                0.25                550.
               0.001                0.15
                 1.8                 1.2                 2.1
/END
"""
        deck_file = tmp_path / f"synonym_{header.replace('/', '_')}.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        model = parse_starter_deck(str(deck_file))

        assert 55 in model.materials
        assert 55 in model.mat_law32s
        assert 55 in model.mat_hills

        mat = model.materials[55]
        m32 = model.mat_law32s[55]
        mh = model.mat_hills[55]

        assert mat.law == 32
        assert m32 is mh
        assert m32.id == 55
        assert math.isclose(m32.e, 210000.0, rel_tol=1e-5)
        assert math.isclose(m32.r00, 1.8, rel_tol=1e-5)


class TestDefensiveEdgeCasesAudit:
    """Audit missing optional cards, default value injection, malformed inputs, bounds."""

    def test_missing_optional_cards_early_eof(self, tmp_path: Path):
        """Audit graceful handling and defaults when cards 3, 4, 5 are omitted (early EOF)."""
        deck_text = """/BEGIN
EARLY_EOF
2022  0
/MAT/LAW32/61
EARLY_EOF_MAT
7.85e-9
210000.0 0.3
/END
"""
        model, log = _parse_deck_string(tmp_path, deck_text)
        assert not log.errors
        assert 61 in model.mat_law32s
        m = model.mat_law32s[61]

        assert math.isclose(m.rho0, 7.85e-9, rel_tol=1e-5)
        assert math.isclose(m.rhor, 7.85e-9, rel_tol=1e-5)
        assert math.isclose(m.e, 210000.0, rel_tol=1e-5)
        assert math.isclose(m.nu, 0.3, rel_tol=1e-5)

        assert math.isclose(m.sigy, 1.0e30, rel_tol=1e-5)
        assert math.isclose(m.beta, 0.0, rel_tol=1e-5)
        assert math.isclose(m.hard, 1.0, rel_tol=1e-5)
        assert math.isclose(m.eps, 1.0e30, rel_tol=1e-5)
        assert math.isclose(m.sig, 1.0e30, rel_tol=1e-5)
        assert math.isclose(m.srp, 1.0, rel_tol=1e-5)
        assert math.isclose(m.src, 0.0, rel_tol=1e-5)
        assert math.isclose(m.r00, 1.0, rel_tol=1e-5)
        assert math.isclose(m.r45, 1.0, rel_tol=1e-5)
        assert math.isclose(m.r90, 1.0, rel_tol=1e-5)

    def test_default_value_injections_zero_fields(self, tmp_path: Path):
        """Audit that zero/blank fields inject defaults per hm_read_mat32.F."""
        deck_text = """/BEGIN
ZERO_FIELDS
2022  0
/MAT/LAW32/62
ZERO_FIELDS_MAT
7.85e-9 0.0
210000.0 0.5
0.0 0.0 0.0 0.0 0.0
0.0 0.0
0.0 0.0 0.0
/END
"""
        model, log = _parse_deck_string(tmp_path, deck_text)
        assert not log.errors
        assert 62 in model.mat_law32s
        m = model.mat_law32s[62]

        assert math.isclose(m.rhor, 7.85e-9, rel_tol=1e-5)
        assert math.isclose(m.nu, 0.499999, rel_tol=1e-5)
        assert math.isclose(m.sigy, 1.0e30, rel_tol=1e-5)
        assert math.isclose(m.beta, 0.0, rel_tol=1e-5)
        assert math.isclose(m.hard, 1.0, rel_tol=1e-5)
        assert math.isclose(m.eps, 1.0e30, rel_tol=1e-5)
        assert math.isclose(m.sig, 1.0e30, rel_tol=1e-5)
        assert math.isclose(m.srp, 1.0, rel_tol=1e-5)
        assert math.isclose(m.src, 0.0, rel_tol=1e-5)
        assert math.isclose(m.r00, 1.0, rel_tol=1e-5)
        assert math.isclose(m.r45, 1.0, rel_tol=1e-5)
        assert math.isclose(m.r90, 1.0, rel_tol=1e-5)

    def test_rate_independent_srp_default(self, tmp_path: Path):
        """Audit hm_read_mat32.F line 147: IF(CM==ZERO) EPS0 = ONE."""
        deck_text = """/BEGIN
RATE_INDEP
2022  0
/MAT/LAW32/63
RATE_INDEP_MAT
7.85e-9
210000.0 0.3
350.0 150.0 0.45 0.25 550.0
0.005 0.0
1.8 1.2 2.1
/END
"""
        model, log = _parse_deck_string(tmp_path, deck_text)
        assert not log.errors
        assert 63 in model.mat_law32s
        m = model.mat_law32s[63]
        assert math.isclose(m.srp, 1.0, rel_tol=1e-5)
        assert math.isclose(m.src, 0.0, rel_tol=1e-5)

    def test_missing_data_cards_graceful_error(self, tmp_path: Path):
        """Audit that an empty /MAT/LAW32 block logs an error without crashing."""
        deck_text = """/BEGIN
EMPTY_BLOCK
2022  0
/MAT/LAW32/99
/END
"""
        model, log = _parse_deck_string(tmp_path, deck_text)
        assert log.errors
        assert any("/MAT/LAW32/99: missing data card" in err for err in log.errors)
        assert 99 not in model.mat_law32s

    def test_malformed_numeric_input_graceful_error(self, tmp_path: Path):
        """Audit that non-numeric characters on cards are caught gracefully with an error."""
        deck_text = """/BEGIN
BAD_DATA
2022  0
/MAT/LAW32/88
TITLE
7.85e-9
INVALID_FLOAT_HERE 0.3
/END
"""
        model, log = _parse_deck_string(tmp_path, deck_text)
        assert log.errors
        assert any("could not convert string to float" in err or "malformed" in err for err in log.errors)
        assert 88 not in model.mat_law32s

    def test_parameter_bounds_validation(self):
        """Audit check_mat_law32 parameter bounds checking."""
        log = MessageLog()
        m_good = MatLaw32(
            id=1, rho0=7.85e-9, e=210000.0, nu=0.3, sigy=350.0, hard=0.5, srp=1.0,
            r00=1.0, r45=1.0, r90=1.0
        )
        check_mat_law32(m_good, log)
        assert not log.errors

        log = MessageLog()
        m_bad_rho = MatLaw32(id=2, rho0=-1.0e-9, e=210000.0, nu=0.3, sigy=350.0)
        check_mat_law32(m_bad_rho, log)
        assert any("initial density RHO must be > 0" in e for e in log.errors)

        log = MessageLog()
        m_bad_e = MatLaw32(id=3, rho0=7.85e-9, e=0.0, nu=0.3, sigy=350.0)
        check_mat_law32(m_bad_e, log)
        assert any("Young's modulus E must be > 0" in e for e in log.errors)

        log = MessageLog()
        m_bad_nu = MatLaw32(id=4, rho0=7.85e-9, e=210000.0, nu=0.52, sigy=350.0)
        check_mat_law32(m_bad_nu, log)
        assert any("Poisson's ratio NU must be in [0, 0.5)" in e for e in log.errors)

        log = MessageLog()
        m_bad_sigy = MatLaw32(id=5, rho0=7.85e-9, e=210000.0, nu=0.3, sigy=-10.0)
        check_mat_law32(m_bad_sigy, log)
        assert any("yield stress A (SIGY) must be > 0" in e for e in log.errors)

        log = MessageLog()
        m_bad_hard = MatLaw32(id=6, rho0=7.85e-9, e=210000.0, nu=0.3, sigy=350.0, hard=1.2)
        check_mat_law32(m_bad_hard, log)
        assert any("hardening exponent n must be <= 1.0" in e for e in log.errors)

        log = MessageLog()
        m_bad_srp = MatLaw32(id=7, rho0=7.85e-9, e=210000.0, nu=0.3, sigy=350.0, srp=-0.01)
        check_mat_law32(m_bad_srp, log)
        assert any("reference strain rate EPS_DOT_0 (srp) must be > 0" in e for e in log.errors)

        log = MessageLog()
        m_bad_r = MatLaw32(id=8, rho0=7.85e-9, e=210000.0, nu=0.3, sigy=350.0, r00=-0.5, r45=0.0, r90=-1.0)
        check_mat_law32(m_bad_r, log)
        assert any("Lankford parameter r00 must be > 0" in e for e in log.errors)
        assert any("Lankford parameter r45 must be > 0" in e for e in log.errors)
        assert any("Lankford parameter r90 must be > 0" in e for e in log.errors)

    def test_element_compatibility_shells_allowed_solids_rejected(self):
        """Audit element family check: shells/sh3n allowed, bricks/solids rejected."""
        assert 32 in _ALLOWED_LAWS["shells"]
        assert "LAW32" in _ALLOWED_LAWS["shells"]
        assert "HILL" in _ALLOWED_LAWS["shells"]
        assert 32 in _ALLOWED_LAWS["shells_qbat"]
        assert 32 in _ALLOWED_LAWS["shells_qeph"]
        assert 32 in _ALLOWED_LAWS["sh3n"]

        for solid_family in ("bricks", "tetras", "penta6", "pyra5", "trusses", "beams"):
            assert 32 not in _ALLOWED_LAWS[solid_family]
            assert "LAW32" not in _ALLOWED_LAWS[solid_family]
            assert "HILL" not in _ALLOWED_LAWS[solid_family]

        model = Model()
        model.add_nodes(np.array([1, 2, 3, 4]), np.zeros((4, 3)))
        mat = MatLaw32(
            id=1, rho0=7.85e-9, e=210000.0, nu=0.3,
            sigy=350.0, beta=150.0, hard=0.5,
            eps=0.25, sig=550.0, srp=1.0, src=0.0,
            r00=1.0, r45=1.0, r90=1.0,
        )
        model.materials[1] = mat

        class FakeSolidGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model.element_groups = lambda: [("bricks", FakeSolidGroup())]

        log = MessageLog()
        check_model(model, log)
        assert log.errors
        assert any("/MAT/LAW32/1 (/MAT/HILL) is not supported for bricks elements" in e for e in log.errors)
