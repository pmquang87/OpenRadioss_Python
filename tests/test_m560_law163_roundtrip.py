"""
Milestone M560: /MAT/LAW163 (/MAT/CRUSHABLE_FOAM, /MAT/CRUSH_FOAM)
Exhaustive Roundtrip, Negative Validation, Boundary Cases & Restart (.rst) Serialization Auditor Suite.

Fortran origins:
  - starter/source/materials/mat/mat163/hm_read_mat163.F90 (card reader, parameters, default fallbacks)
  - engine/source/materials/mat/mat163/sigeps163.F90 (constitutive stress update, filtering, viscous damping)
  - engine/source/output/restart/wrrestp.F & rdresb.F
  - starter/source/restart/ddsplit/wrrest.F
  - hm_cfg_files/config/CFG/radioss2025/MAT/matl163_crushable_foam.cfg
"""

from __future__ import annotations

import math
from pathlib import Path
import pickle
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import solid_hexa8, solid_tetra4
from pyradioss.input.card_layouts import CARD_LAYOUTS, split_fixed
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.starter_keywords import (
    parse_starter_deck,
    read_mat_law163,
    read_starter_deck,
)
from pyradioss.materials.law163_crush_foam import (
    Law163Params,
    build_law163,
    solid_update,
    extra_shapes,
    sound_speed_solid_law163,
)
from pyradioss.model.entities import (
    MatCrushableFoam,
    MatCrushFoam,
    MatLaw163,
    Material,
    Part,
)
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_mat_law163,
    check_materials,
    check_model,
)
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers
# ============================================================================

def _parse_deck_str(tmp_path: Path, text: str, name: str = "TEST_LAW163") -> Tuple[Model, MessageLog]:
    """Write deck text to file and parse into Model and MessageLog."""
    deck_path = tmp_path / f"{name}_0000.rad"
    deck_path.write_text(text.strip() + "\n", encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def _assert_law163_all_fields_exact(
    m: MatLaw163,
    mat: Material | None = None,
    *,
    rho: float,
    e: float,
    nu: float,
    tsc: float,
    damp: float = 0.10,
    ncycle: int = 12,
    tab_id: int = 0,
    epsd_ref: float = 0.0,
    fscale: float = 1.0,
    srclmt: float = 1.0e20,
    nrs: int = 0,
    title: str = "",
) -> None:
    """Assert exact (floating-point tolerance) equality for all LAW163 parameters."""
    assert m.rho == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
    assert m.e == pytest.approx(e, rel=1e-6, abs=1e-12)
    assert m.E == pytest.approx(e, rel=1e-6, abs=1e-12)
    assert m.nu == pytest.approx(nu, rel=1e-6, abs=1e-12)
    assert m.tsc == pytest.approx(tsc, rel=1e-6, abs=1e-12)
    assert m.damp == pytest.approx(damp, rel=1e-6, abs=1e-12)
    assert m.ncycle == ncycle
    assert m.tab_id == tab_id
    assert m.epsd_ref == pytest.approx(epsd_ref, rel=1e-6, abs=1e-12)
    assert m.fscale == pytest.approx(fscale, rel=1e-6, abs=1e-12)
    assert m.srclmt == pytest.approx(srclmt, rel=1e-6, abs=1e-12)
    assert m.nrs == nrs
    if title:
        assert m.title == title

    # Derived Moduli
    expected_g = e / (2.0 * (1.0 + nu))
    expected_bulk = e / (3.0 * (1.0 - 2.0 * nu))
    expected_cii = expected_bulk + 4.0 / 3.0 * expected_g
    expected_cij = expected_bulk - 2.0 / 3.0 * expected_g

    assert m.G == pytest.approx(expected_g, rel=1e-6, abs=1e-12)
    assert m.bulk == pytest.approx(expected_bulk, rel=1e-6, abs=1e-12)
    assert m.K == pytest.approx(expected_bulk, rel=1e-6, abs=1e-12)
    assert m.cii == pytest.approx(expected_cii, rel=1e-6, abs=1e-12)
    assert m.cij == pytest.approx(expected_cij, rel=1e-6, abs=1e-12)

    # Check Material entity in model.materials
    if mat is not None:
        assert mat.law == 163
        assert mat.rho0 == pytest.approx(rho, rel=1e-6, abs=1e-12)
        p = mat.params
        assert p["rho"] == pytest.approx(rho, rel=1e-6, abs=1e-12)
        assert p["rho0"] == pytest.approx(rho, rel=1e-6, abs=1e-12)
        assert p["e"] == pytest.approx(e, rel=1e-6, abs=1e-12)
        assert p["nu"] == pytest.approx(nu, rel=1e-6, abs=1e-12)
        assert p["tsc"] == pytest.approx(tsc, rel=1e-6, abs=1e-12)
        assert p["damp"] == pytest.approx(damp, rel=1e-6, abs=1e-12)
        assert p["ncycle"] == ncycle
        assert p["tab_id"] == tab_id
        assert p["epsd_ref"] == pytest.approx(epsd_ref, rel=1e-6, abs=1e-12)
        assert p["fscale"] == pytest.approx(fscale, rel=1e-6, abs=1e-12)
        assert p["srclmt"] == pytest.approx(srclmt, rel=1e-6, abs=1e-12)
        assert p["nrs"] == nrs
        if title:
            assert mat.title == title


# ============================================================================
# 1. Fixed-Format Deck Roundtrip
# ============================================================================

class TestLaw163FixedFormatRoundtrip:
    """Audit StarterDeck.mat_law163 fixed-format emission and re-reading."""

    def test_fixed_format_card_columns_and_roundtrip(self, tmp_path: Path):
        """Verify exact 20/10-column layout of all 3 data cards and complete roundtrip."""
        vals = {
            "rho": 1.2e-3,
            "e": 250.0,
            "nu": 0.25,
            "tsc": 5.0,
            "damp": 0.15,
            "ncycle": 10,
            "tab_id": 101,
            "epsd_ref": 0.01,
            "fscale": 1.5,
            "srclmt": 1.0e15,
            "nrs": 1,
        }

        deck = StarterDeck("AUDIT_FIXED_LAW163")
        deck.mat_law163(
            mat_id=163,
            title="Foam-Fixed-Standard",
            fixed_format=True,
            **vals,
        )

        rendered = deck.render()
        lines = [line for line in rendered.splitlines() if line.strip() and not line.startswith("#")]

        mat_idx = None
        for idx, line in enumerate(lines):
            if line.startswith("/MAT/LAW163/163"):
                mat_idx = idx
                break
        assert mat_idx is not None, "Failed to locate /MAT/LAW163/163 header in rendered deck"

        # Card 0: Title
        assert lines[mat_idx + 1].strip() == "Foam-Fixed-Standard"

        # Cards 1 to 3
        card_lines = lines[mat_idx + 2 : mat_idx + 5]
        assert len(card_lines) == 3, f"Expected 3 data cards, got {len(card_lines)}"

        # Card 1: RHO (%20lg) -> width 20
        c1 = split_fixed(card_lines[0], CARD_LAYOUTS["MAT_LAW163_1"])
        assert len(c1) == 1
        assert float(c1[0]) == pytest.approx(vals["rho"])

        # Card 2: E, NU, TSC, DAMP, blank(10), NCYCLE (%20lg%20lg%20lg%20lg%10s%10d) -> width 100
        assert len(card_lines[1]) == 100
        c2 = split_fixed(card_lines[1], CARD_LAYOUTS["MAT_LAW163_2"])
        assert len(c2) == 6
        assert float(c2[0]) == pytest.approx(vals["e"])
        assert float(c2[1]) == pytest.approx(vals["nu"])
        assert float(c2[2]) == pytest.approx(vals["tsc"])
        assert float(c2[3]) == pytest.approx(vals["damp"])
        assert int(c2[5]) == vals["ncycle"]

        # Card 3: blank(10), TAB_ID, EPSD_REF, FSCALE, SRCLMT, blank(10), NRS (%10s%10d%20lg%20lg%20lg%10s%10d) -> width 100
        assert len(card_lines[2]) == 100
        c3 = split_fixed(card_lines[2], CARD_LAYOUTS["MAT_LAW163_3"])
        assert len(c3) == 7
        assert int(c3[1]) == vals["tab_id"]
        assert float(c3[2]) == pytest.approx(vals["epsd_ref"])
        assert float(c3[3]) == pytest.approx(vals["fscale"])
        assert float(c3[4]) == pytest.approx(vals["srclmt"])
        assert int(c3[6]) == vals["nrs"]

        # Re-parse via read_starter_deck
        rad_path = tmp_path / "audit_fixed_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors, f"Errors encountered parsing fixed deck: {log.errors}"
        assert 163 in model.mat_law163s
        assert 163 in model.materials

        _assert_law163_all_fields_exact(
            model.mat_law163s[163],
            model.materials[163],
            title="Foam-Fixed-Standard",
            **vals,
        )

    def test_fixed_format_defaults_and_minimal_cards(self, tmp_path: Path):
        """Verify default fallbacks when cards are minimal or omitted."""
        deck = StarterDeck("AUDIT_MINIMAL_LAW163")
        deck.mat_law163(
            mat_id=1,
            title="Minimal-Foam",
            rho=1.5e-3,
            e=100.0,
            nu=0.3,
            tsc=2.0,
            fixed_format=True,
        )
        rad_path = tmp_path / "audit_min_0000.rad"
        deck.write(str(rad_path))

        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        m = model.mat_law163s[1]
        assert m.damp == pytest.approx(0.10)
        assert m.ncycle == 12
        assert m.tab_id == 0
        assert m.epsd_ref == pytest.approx(0.0)
        assert m.fscale == pytest.approx(1.0)
        assert m.srclmt == pytest.approx(1.0e20)
        assert m.nrs == 0

    def test_fixed_format_1_card_and_2_cards(self, tmp_path: Path):
        """Verify reading decks with only 1 card or 2 cards."""
        text_1card = """
/MAT/LAW163/1
Single Card Foam
               1.200E-03
"""
        model1, log1 = _parse_deck_str(tmp_path, text_1card, name="ONE_CARD")
        assert not log1.has_errors
        m1 = model1.mat_law163s[1]
        assert m1.rho == pytest.approx(1.2e-3)
        assert m1.e == pytest.approx(0.0)
        assert m1.nu == pytest.approx(0.0)
        assert m1.damp == pytest.approx(0.10)
        assert m1.ncycle == 12
        assert m1.tab_id == 0

        text_2cards = """
/MAT/LAW163/2
Two Card Foam
               1.200E-03
               2.500E+02               2.500E-01               5.000E+00               1.500E-01                  10
"""
        model2, log2 = _parse_deck_str(tmp_path, text_2cards, name="TWO_CARDS")
        assert not log2.has_errors
        m2 = model2.mat_law163s[2]
        assert m2.rho == pytest.approx(1.2e-3)
        assert m2.e == pytest.approx(250.0)
        assert m2.nu == pytest.approx(0.25)
        assert m2.tsc == pytest.approx(5.0)
        assert m2.damp == pytest.approx(0.15)
        assert m2.ncycle == 10
        assert m2.tab_id == 0
        assert m2.fscale == pytest.approx(1.0)
        assert m2.srclmt == pytest.approx(1.0e20)
        assert m2.nrs == 0

    def test_fixed_format_entity_invocation(self, tmp_path: Path):
        """Verify passing MatLaw163, MatCrushableFoam, MatCrushFoam, and Material objects."""
        # 1. MatLaw163 entity
        m_entity = MatLaw163(
            id=10,
            title="Entity-Foam",
            rho=1.8e-3,
            e=300.0,
            nu=0.2,
            tsc=4.0,
            damp=0.12,
            ncycle=8,
            tab_id=202,
            epsd_ref=0.05,
            fscale=1.2,
            srclmt=1e16,
            nrs=1,
        )
        deck = StarterDeck("AUDIT_ENTITY")
        deck.mat_law163(m_entity)
        rad_path = tmp_path / "entity_0000.rad"
        deck.write(str(rad_path))
        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        _assert_law163_all_fields_exact(
            model.mat_law163s[10],
            model.materials[10],
            rho=1.8e-3, e=300.0, nu=0.2, tsc=4.0, damp=0.12, ncycle=8,
            tab_id=202, epsd_ref=0.05, fscale=1.2, srclmt=1e16, nrs=1,
            title="Entity-Foam",
        )

        # 2. Material entity
        mat_entity = Material(
            id=20,
            law=163,
            rho0=2.0e-3,
            title="Material-Foam",
            params={
                "rho0": 2.0e-3, "e": 400.0, "nu": 0.22, "tsc": 6.0,
                "damp": 0.18, "ncycle": 14, "tab_id": 303,
                "epsd_ref": 0.02, "fscale": 2.0, "srclmt": 1e18, "nrs": 0,
            }
        )
        deck2 = StarterDeck("AUDIT_MATERIAL_ENTITY")
        deck2.mat_law163(mat_entity)
        rad_path2 = tmp_path / "mat_entity_0000.rad"
        deck2.write(str(rad_path2))
        model2, log2 = read_starter_deck(str(rad_path2))
        assert not log2.has_errors
        _assert_law163_all_fields_exact(
            model2.mat_law163s[20],
            model2.materials[20],
            rho=2.0e-3, e=400.0, nu=0.22, tsc=6.0, damp=0.18, ncycle=14,
            tab_id=303, epsd_ref=0.02, fscale=2.0, srclmt=1e18, nrs=0,
            title="Material-Foam",
        )

    def test_fixed_format_positional_arguments(self, tmp_path: Path):
        """Verify positional argument invocation of mat_law163."""
        deck = StarterDeck("AUDIT_POS_LAW163")
        deck.mat_law163(
            163,
            "Pos-Title",
            1.2e-3,  # rho
            250.0,   # e
            0.25,    # nu
            5.0,     # tsc
            0.15,    # damp
            10,      # ncycle
            101,     # tab_id
            0.01,    # epsd_ref
            1.5,     # fscale
            1.0e15,  # srclmt
            1,       # nrs
        )
        rad_path = tmp_path / "audit_pos_0000.rad"
        deck.write(str(rad_path))
        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        _assert_law163_all_fields_exact(
            model.mat_law163s[163],
            model.materials[163],
            rho=1.2e-3, e=250.0, nu=0.25, tsc=5.0, damp=0.15, ncycle=10,
            tab_id=101, epsd_ref=0.01, fscale=1.5, srclmt=1.0e15, nrs=1,
            title="Pos-Title",
        )

    def test_fixed_format_keyword_aliases(self, tmp_path: Path):
        """Verify keyword argument aliases (rho0, young, mat_nu, lsdyna_tsc, lsd_tid, etc.)."""
        deck = StarterDeck("AUDIT_ALIASES_LAW163")
        deck.mat_law163(
            mat_id=5,
            title="Alias-Foam",
            rho0=1.1e-3,
            young=220.0,
            mat_nu=0.24,
            lsdyna_tsc=4.5,
            lsd_mat_damp=0.14,
            lsd_ncycle=11,
            lsd_tid=505,
            epsd_ref=0.03,
            fscale=1.3,
            lsd_srclmt=1.0e14,
            nrsflag=1,
        )
        rad_path = tmp_path / "audit_alias_0000.rad"
        deck.write(str(rad_path))
        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        _assert_law163_all_fields_exact(
            model.mat_law163s[5],
            model.materials[5],
            rho=1.1e-3, e=220.0, nu=0.24, tsc=4.5, damp=0.14, ncycle=11,
            tab_id=505, epsd_ref=0.03, fscale=1.3, srclmt=1.0e14, nrs=1,
            title="Alias-Foam",
        )


# ============================================================================
# 2. Free-Format Roundtrip
# ============================================================================

class TestLaw163FreeFormatRoundtrip:
    """Audit free-format (comma and whitespace delimited) emission and parsing."""

    def test_free_format_comma_delimited(self, tmp_path: Path):
        """Verify free format with comma and space delimiters."""
        deck = StarterDeck("AUDIT_FREE_COMMA")
        deck.mat_law163(
            mat_id=7,
            title="Free-Comma-Foam",
            fixed_format=False,
            comma=True,
            rho=1.3e-3,
            e=260.0,
            nu=0.28,
            tsc=4.8,
            damp=0.16,
            ncycle=9,
            tab_id=707,
            epsd_ref=0.04,
            fscale=1.4,
            srclmt=1e17,
            nrs=1,
        )
        rendered = deck.render()
        assert "," in rendered
        rad_path = tmp_path / "free_comma_0000.rad"
        deck.write(str(rad_path))
        model, log = read_starter_deck(str(rad_path))
        assert not log.has_errors
        _assert_law163_all_fields_exact(
            model.mat_law163s[7],
            model.materials[7],
            rho=1.3e-3, e=260.0, nu=0.28, tsc=4.8, damp=0.16, ncycle=9,
            tab_id=707, epsd_ref=0.04, fscale=1.4, srclmt=1e17, nrs=1,
            title="Free-Comma-Foam",
        )

    def test_free_format_compact_comma(self, tmp_path: Path):
        """Verify parsing free-format with compact commas (no whitespace)."""
        text = """
/MAT/LAW163/8
Compact Comma Foam
1.3e-3
260.0,0.28,4.8,0.16,9
707,0.04,1.4,1e17,1
"""
        model, log = _parse_deck_str(tmp_path, text, name="COMPACT_COMMA")
        assert not log.has_errors
        _assert_law163_all_fields_exact(
            model.mat_law163s[8],
            model.materials[8],
            rho=1.3e-3, e=260.0, nu=0.28, tsc=4.8, damp=0.16, ncycle=9,
            tab_id=707, epsd_ref=0.04, fscale=1.4, srclmt=1e17, nrs=1,
            title="Compact Comma Foam",
        )

    def test_free_format_space_delimited(self, tmp_path: Path):
        """Verify parsing free-format with whitespace separation."""
        text = """
/MAT/LAW163/9
Whitespace Foam
1.4e-3
270.0 0.29 4.9 0.17 8
808 0.05 1.5 2e17 0
"""
        model, log = _parse_deck_str(tmp_path, text, name="WHITESPACE")
        assert not log.has_errors
        _assert_law163_all_fields_exact(
            model.mat_law163s[9],
            model.materials[9],
            rho=1.4e-3, e=270.0, nu=0.29, tsc=4.9, damp=0.17, ncycle=8,
            tab_id=808, epsd_ref=0.05, fscale=1.5, srclmt=2e17, nrs=0,
            title="Whitespace Foam",
        )

    def test_free_format_7_token_card3(self, tmp_path: Path):
        """Verify Card 3 written with 7 tokens including leading/trailing blanks/zeros."""
        text = """
/MAT/LAW163/12
Seven Token Card3 Foam
1.2e-3
250.0, 0.25, 5.0, 0.15, 10
0, 101, 0.01, 1.5, 1.0e15, 0, 1
"""
        model, log = _parse_deck_str(tmp_path, text, name="SEVEN_TOKEN")
        assert not log.has_errors
        _assert_law163_all_fields_exact(
            model.mat_law163s[12],
            model.materials[12],
            rho=1.2e-3, e=250.0, nu=0.25, tsc=5.0, damp=0.15, ncycle=10,
            tab_id=101, epsd_ref=0.01, fscale=1.5, srclmt=1.0e15, nrs=1,
            title="Seven Token Card3 Foam",
        )

    def test_free_format_comments_and_blank_lines(self, tmp_path: Path):
        """Verify parser robustness against comments (# and $) and blank lines."""
        text = """
# Material Header Comment
/MAT/LAW163/15
# Title next line
Commented Foam
# Card 1
1.2e-3 # density

# Card 2
250.0, 0.25, 5.0, 0.15, 10 $ elastic and damping
# Card 3
$ blank field, tab, epsd, fscale, srclmt, blank, nrs
0, 101, 0.01, 1.5, 1.0e15, 0, 1
"""
        model, log = _parse_deck_str(tmp_path, text, name="COMMENTS_BLANKS")
        assert not log.has_errors
        _assert_law163_all_fields_exact(
            model.mat_law163s[15],
            model.materials[15],
            rho=1.2e-3, e=250.0, nu=0.25, tsc=5.0, damp=0.15, ncycle=10,
            tab_id=101, epsd_ref=0.01, fscale=1.5, srclmt=1.0e15, nrs=1,
            title="Commented Foam",
        )

    def test_cross_dialect_roundtrip(self, tmp_path: Path):
        """Verify cross-dialect roundtrip: free-format -> parse -> fixed-format -> parse -> exact equality."""
        free_text = """
/MAT/LAW163/99
Cross Dialect Foam
1.25e-3
240.0, 0.22, 4.2, 0.18, 11
909, 0.025, 1.35, 5.0e16, 1
"""
        model_free, log_free = _parse_deck_str(tmp_path, free_text, name="CROSS_FREE")
        assert not log_free.has_errors
        m_free = model_free.mat_law163s[99]

        deck_fixed = StarterDeck("CROSS_FIXED")
        deck_fixed.mat_law163(m_free, fixed_format=True)
        rad_fixed = tmp_path / "cross_fixed_0000.rad"
        deck_fixed.write(str(rad_fixed))

        model_round, log_round = read_starter_deck(str(rad_fixed))
        assert not log_round.has_errors
        m_round = model_round.mat_law163s[99]

        assert m_round.rho == pytest.approx(m_free.rho, rel=1e-6, abs=1e-12)
        assert m_round.e == pytest.approx(m_free.e, rel=1e-6, abs=1e-12)
        assert m_round.nu == pytest.approx(m_free.nu, rel=1e-6, abs=1e-12)
        assert m_round.tsc == pytest.approx(m_free.tsc, rel=1e-6, abs=1e-12)
        assert m_round.damp == pytest.approx(m_free.damp, rel=1e-6, abs=1e-12)
        assert m_round.ncycle == m_free.ncycle
        assert m_round.tab_id == m_free.tab_id
        assert m_round.epsd_ref == pytest.approx(m_free.epsd_ref, rel=1e-6, abs=1e-12)
        assert m_round.fscale == pytest.approx(m_free.fscale, rel=1e-6, abs=1e-12)
        assert m_round.srclmt == pytest.approx(m_free.srclmt, rel=1e-6, abs=1e-12)
        assert m_round.nrs == m_free.nrs
        assert m_round.title == m_free.title


# ============================================================================
# 3. Keyword Synonyms
# ============================================================================

class TestLaw163KeywordSynonyms:
    """Audit /MAT/LAW163, /MAT/CRUSHABLE_FOAM, /MAT/CRUSH_FOAM keyword synonyms and unit ID."""

    def test_keyword_synonyms_and_unit_id(self, tmp_path: Path):
        """Verify all keyword synonyms parse into model.mat_law163s and dictionary aliases."""
        text = """
/MAT/LAW163/1
Law 163 Standard
1.2e-3
250.0, 0.25, 5.0, 0.15, 10
101, 0.01, 1.5, 1.0e15, 1
/MAT/CRUSHABLE_FOAM/2
Crushable Foam Keyword
1.3e-3
260.0, 0.26, 5.1, 0.16, 11
102, 0.02, 1.6, 1.0e16, 0
/MAT/CRUSH_FOAM/3
Crush Foam Keyword
1.4e-3
270.0, 0.27, 5.2, 0.17, 12
103, 0.03, 1.7, 1.0e17, 1
/MAT/LAW163/4/2
Foam With Unit System
1.5e-3
280.0, 0.28, 5.3, 0.18, 13
104, 0.04, 1.8, 1.0e18, 0
"""
        model, log = _parse_deck_str(tmp_path, text, name="SYNONYMS")
        assert not log.has_errors
        assert set(model.mat_law163s.keys()) == {1, 2, 3, 4}
        assert set(model.mat_crushable_foams.keys()) == {1, 2, 3, 4}
        assert set(model.mat_crush_foams.keys()) == {1, 2, 3, 4}

        # Check entity types
        for mid in (1, 2, 3, 4):
            m = model.mat_law163s[mid]
            assert isinstance(m, MatLaw163)
            assert isinstance(m, MatCrushableFoam)
            assert isinstance(m, MatCrushFoam)

        # Check StarterDeck helper methods
        deck = StarterDeck("SYNONYM_WRITER")
        deck.mat_crushable_foam(10, "Crushable", rho=1.0e-3, e=200.0, nu=0.2)
        deck.mat_crush_foam(11, "Crush", rho=1.1e-3, e=210.0, nu=0.21)
        rad_path = tmp_path / "syn_writer_0000.rad"
        deck.write(str(rad_path))
        m_written, log_written = read_starter_deck(str(rad_path))
        assert not log_written.has_errors
        assert 10 in m_written.mat_law163s
        assert 11 in m_written.mat_law163s


# ============================================================================
# 4. Negative Starter Diagnostics
# ============================================================================

class TestLaw163NegativeDiagnostics:
    """Audit starter negative diagnostics in check_mat_law163 and check_materials."""

    def test_negative_and_zero_density(self):
        """Error when initial density RHO <= 0."""
        log1 = MessageLog()
        m1 = MatLaw163(id=1, rho=0.0, e=250.0, nu=0.25)
        check_mat_law163(m1, log1)
        assert log1.has_errors
        assert any("initial density RHO must be > 0" in e for e in log1.errors)

        log2 = MessageLog()
        m2 = MatLaw163(id=2, rho=-1.2e-3, e=250.0, nu=0.25)
        check_mat_law163(m2, log2)
        assert log2.has_errors
        assert any("initial density RHO must be > 0" in e for e in log2.errors)

    def test_negative_and_zero_young(self):
        """Error when Young's modulus E <= 0."""
        log1 = MessageLog()
        m1 = MatLaw163(id=1, rho=1.2e-3, e=0.0, nu=0.25)
        check_mat_law163(m1, log1)
        assert log1.has_errors
        assert any("Young's modulus E must be > 0" in e for e in log1.errors)

        log2 = MessageLog()
        m2 = MatLaw163(id=2, rho=1.2e-3, e=-250.0, nu=0.25)
        check_mat_law163(m2, log2)
        assert log2.has_errors
        assert any("Young's modulus E must be > 0" in e for e in log2.errors)

    def test_poisson_ratio_bounds(self):
        """Error when nu < 0 or nu >= 0.5 (ANCMSG 1514)."""
        # Negative nu
        log1 = MessageLog()
        m1 = MatLaw163(id=1, rho=1.2e-3, e=250.0, nu=-0.1)
        check_mat_law163(m1, log1)
        assert log1.has_errors
        assert any("ANCMSG 1514" in e for e in log1.errors)

        # nu == 0.5 (incompressible limit)
        log2 = MessageLog()
        m2 = MatLaw163(id=2, rho=1.2e-3, e=250.0, nu=0.5)
        check_mat_law163(m2, log2)
        assert log2.has_errors
        assert any("ANCMSG 1514" in e for e in log2.errors)

        # nu > 0.5
        log3 = MessageLog()
        m3 = MatLaw163(id=3, rho=1.2e-3, e=250.0, nu=0.55)
        check_mat_law163(m3, log3)
        assert log3.has_errors
        assert any("ANCMSG 1514" in e for e in log3.errors)

        # Valid bounds: nu = 0.0 and nu = 0.499
        log4 = MessageLog()
        m4 = MatLaw163(id=4, rho=1.2e-3, e=250.0, nu=0.0)
        check_mat_law163(m4, log4)
        assert not log4.has_errors

        log5 = MessageLog()
        m5 = MatLaw163(id=5, rho=1.2e-3, e=250.0, nu=0.499)
        check_mat_law163(m5, log5)
        assert not log5.has_errors

    def test_negative_damping(self):
        """Error when damping coefficient DAMP < 0."""
        log1 = MessageLog()
        m1 = MatLaw163(id=1, rho=1.2e-3, e=250.0, nu=0.25, damp=-0.05)
        check_mat_law163(m1, log1)
        assert log1.has_errors
        assert any("damping coefficient DAMP must be >= 0" in e for e in log1.errors)

        log2 = MessageLog()
        m2 = MatLaw163(id=2, rho=1.2e-3, e=250.0, nu=0.25, damp=0.0)
        check_mat_law163(m2, log2)
        assert not log2.has_errors

    def test_2d_analysis_rejected(self):
        """Reject 2D analysis (N2D > 0) with ANCMSG 305."""
        model = Model()
        model.n2d = 1
        m = MatLaw163(id=1, rho=1.2e-3, e=250.0, nu=0.25)
        model.mat_law163s[1] = m
        model.materials[1] = Material(id=1, law=163, rho0=1.2e-3, params={"rho": 1.2e-3, "e": 250.0, "nu": 0.25})

        log = MessageLog()
        check_mat_law163(model, 1, m, log)
        assert log.has_errors
        assert any("2D analysis" in e and "ANCMSG 305" in e for e in log.errors)

    def test_incompatible_shell_elements(self):
        """Reject shell elements with ANCMSG 305."""
        class DummyEl:
            def __init__(self, mid: int):
                self.mat_id = mid

        class DummyGrp:
            def __init__(self, els: list):
                self._els = {i: e for i, e in enumerate(els)}
            def values(self):
                return self._els.values()

        class DummyModel:
            def __init__(self, grps: list, n2d: int = 0):
                self._grps = grps
                self.n2d = n2d
                self.materials: dict = {}
            def element_groups(self):
                return self._grps

        m = MatLaw163(id=1, rho=1.2e-3, e=250.0, nu=0.25)
        for sh_type in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
            model_sh = DummyModel([(sh_type, DummyGrp([DummyEl(1)]))])
            log_sh = MessageLog()
            check_mat_law163(model=model_sh, mat=m, log=log_sh)
            assert log_sh.has_errors, f"Shell element {sh_type} should be rejected"
            assert any("shell elements" in e and "ANCMSG 305" in e for e in log_sh.errors)

        # 2. Via parts dictionary
        model2 = Model()
        model2.mat_law163s[1] = m
        model2.materials[1] = Material(id=1, law=163, rho0=1.2e-3, params={"rho": 1.2e-3, "e": 250.0, "nu": 0.25})
        p_sh = Part(id=1, prop_id=1, mat_id=1, title="Shell Part")
        p_sh.elem_type = "SHELL"
        model2.parts[1] = p_sh
        log2 = MessageLog()
        check_mat_law163(model2, 1, m, log2)
        assert log2.has_errors
        assert any("shell elements" in e and "ANCMSG 305" in e for e in log2.errors)

    def test_incompatible_1d_elements(self):
        """Reject 1D elements with ANCMSG 306."""
        class DummyEl:
            def __init__(self, mid: int):
                self.mat_id = mid

        class DummyGrp:
            def __init__(self, els: list):
                self._els = {i: e for i, e in enumerate(els)}
            def values(self):
                return self._els.values()

        class DummyModel:
            def __init__(self, grps: list, n2d: int = 0):
                self._grps = grps
                self.n2d = n2d
                self.materials: dict = {}
            def element_groups(self):
                return self._grps

        m = MatLaw163(id=1, rho=1.2e-3, e=250.0, nu=0.25)
        for d1_type in ("trusses", "beams", "springs"):
            model_1d = DummyModel([(d1_type, DummyGrp([DummyEl(1)]))])
            log_1d = MessageLog()
            check_mat_law163(model=model_1d, mat=m, log=log_1d)
            assert log_1d.has_errors, f"1D element {d1_type} should be rejected"
            assert any("1D elements" in e and "ANCMSG 306" in e for e in log_1d.errors)

        # 2. Via parts dictionary
        model2 = Model()
        model2.mat_law163s[1] = m
        model2.materials[1] = Material(id=1, law=163, rho0=1.2e-3, params={"rho": 1.2e-3, "e": 250.0, "nu": 0.25})
        p_bm = Part(id=1, prop_id=1, mat_id=1, title="Beam Part")
        p_bm.elem_type = "BEAM"
        model2.parts[1] = p_bm
        log2 = MessageLog()
        check_mat_law163(model2, 1, m, log2)
        assert log2.has_errors
        assert any("1D elements" in e and "ANCMSG 306" in e for e in log2.errors)

    def test_compatible_solid_elements(self):
        """Accept solid elements without diagnostic errors."""
        class DummyEl:
            def __init__(self, mid: int):
                self.mat_id = mid

        class DummyGrp:
            def __init__(self, els: list):
                self._els = {i: e for i, e in enumerate(els)}
            def values(self):
                return self._els.values()

        class DummyModel:
            def __init__(self, grps: list, n2d: int = 0):
                self._grps = grps
                self.n2d = n2d
                self.materials: dict = {}
            def element_groups(self):
                return self._grps

        m = MatLaw163(id=1, rho=1.2e-3, e=250.0, nu=0.25)
        for s_type in ("solids", "bricks", "tetras", "penta6", "pyra5"):
            model_s = DummyModel([(s_type, DummyGrp([DummyEl(1)]))])
            log_s = MessageLog()
            check_mat_law163(model=model_s, mat=m, log=log_s)
            assert not log_s.has_errors, f"Solid element {s_type} should be accepted"

    def test_checks_registry_and_dispatch(self):
        """Verify registry membership and check_materials dispatch."""
        for syn in (163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM"):
            assert syn in _MAT_CHECKS, f"{syn} missing from _MAT_CHECKS"
            assert _MAT_CHECKS[syn] is check_mat_law163

        for fam in ("bricks", "tetras", "penta6", "pyra5", "solids"):
            if fam in _ALLOWED_LAWS and _ALLOWED_LAWS[fam] is not None:
                assert 163 in _ALLOWED_LAWS[fam], f"163 missing from {fam} allowed laws"

        for fam in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads", "trusses", "beams"):
            if fam in _ALLOWED_LAWS and _ALLOWED_LAWS[fam] is not None:
                assert 163 not in _ALLOWED_LAWS[fam], f"163 unexpectedly allowed for {fam}"

        # check_materials dispatcher
        model = Model()
        model.materials[1] = Material(id=1, law=163, rho0=-1.0, params={"rho0": -1.0, "e": 250.0, "nu": 0.25})
        log = MessageLog()
        check_materials(model, log)
        assert log.has_errors
        assert any("initial density RHO must be > 0" in e for e in log.errors)


# ============================================================================
# 5. Boundary Values & Defaults
# ============================================================================

class TestLaw163BoundaryValuesAndDefaults:
    """Audit boundary conditions, extreme Poisson ratio, defaults, wave speeds, and mapping protocol."""

    def test_boundary_poisson_ratio(self):
        """Verify moduli at boundary nu=0.0 and nu near 0.5."""
        # nu = 0.0 (foam with zero lateral expansion)
        m0 = MatLaw163(id=1, rho=1.0, e=300.0, nu=0.0)
        assert m0.G == pytest.approx(150.0)
        assert m0.bulk == pytest.approx(100.0)
        assert m0.K == pytest.approx(100.0)
        assert m0.cii == pytest.approx(100.0 + 4.0 / 3.0 * 150.0)  # = 300.0
        assert m0.cij == pytest.approx(100.0 - 2.0 / 3.0 * 150.0)  # = 0.0

        # nu = 0.499 (nearly incompressible)
        m_incomp = MatLaw163(id=2, rho=1.0, e=300.0, nu=0.499)
        expected_k = 300.0 / (3.0 * (1.0 - 2.0 * 0.499))
        assert m_incomp.bulk == pytest.approx(expected_k)
        assert m_incomp.K == pytest.approx(expected_k)
        assert m_incomp.bulk > 40000.0

    def test_boundary_nrs_flag(self, tmp_path: Path):
        """Verify NRS flag boundary and clamping in reader."""
        deck_text = """
/MAT/LAW163/1
NRS Zero
1.2e-3
250.0, 0.25, 5.0, 0.15, 10
0, 0.0, 1.0, 1.0e20, 0
/MAT/LAW163/2
NRS One
1.2e-3
250.0, 0.25, 5.0, 0.15, 10
0, 0.0, 1.0, 1.0e20, 1
/MAT/LAW163/3
NRS Clamped High
1.2e-3
250.0, 0.25, 5.0, 0.15, 10
0, 0.0, 1.0, 1.0e20, 5
"""
        model, log = _parse_deck_str(tmp_path, deck_text, name="NRS_TEST")
        assert not log.has_errors
        assert model.mat_law163s[1].nrs == 0
        assert model.mat_law163s[2].nrs == 1
        assert model.mat_law163s[3].nrs == 1

    def test_derived_wave_speeds(self):
        """Verify sound speed calculations for solid and shell elements."""
        m = MatLaw163(id=1, rho=1.2e-3, e=250.0, nu=0.25)
        # c_solid = sqrt(cii / rho)
        c_solid = math.sqrt(m.cii / 1.2e-3)
        # c_shell = sqrt(e / (rho * (1 - nu^2)))
        c_shell = math.sqrt(250.0 / (1.2e-3 * (1.0 - 0.25**2)))

        assert m.sound_speed == pytest.approx(c_shell)
        assert sound_speed_solid_law163(m) == pytest.approx(c_solid)

    def test_mapping_protocol(self):
        """Verify mapping protocol (__getitem__, __setitem__, __contains__, get, keys, items, values)."""
        m = MatLaw163(id=1, rho=1.2e-3, e=250.0, nu=0.25, tsc=5.0)
        assert m["rho"] == pytest.approx(1.2e-3)
        assert m["e"] == pytest.approx(250.0)
        assert m["nu"] == pytest.approx(0.25)
        assert m.get("tsc") == pytest.approx(5.0)
        assert m.get("nonexistent", 99.0) == 99.0
        assert "rho" in m
        assert "e" in m
        assert "nonexistent" not in m

        # Mutate via __setitem__
        m["tsc"] = 6.5
        assert m.tsc == pytest.approx(6.5)

        # keys, items, values
        keys = list(m.keys())
        assert "rho" in keys
        assert "e" in keys
        assert "nu" in keys
        items = dict(m.items())
        assert items["tsc"] == pytest.approx(6.5)


# ============================================================================
# 6. Restart (.rst) Serialization
# ============================================================================

class TestLaw163RestartAndSerialization:
    """Audit MatLaw163 and Model serialization, .rst file writing/reading, and cycle continuation."""

    def test_pickle_mat_law163(self):
        """Verify MatLaw163 dataclass pickling fidelity."""
        m = MatLaw163(
            id=163,
            title="Pickle-Foam",
            rho=1.2e-3,
            e=250.0,
            nu=0.25,
            tsc=5.0,
            damp=0.15,
            ncycle=10,
            tab_id=101,
            epsd_ref=0.01,
            fscale=1.5,
            srclmt=1.0e15,
            nrs=1,
        )
        serialized = pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL)
        restored = pickle.loads(serialized)

        assert isinstance(restored, MatLaw163)
        _assert_law163_all_fields_exact(
            restored,
            rho=1.2e-3, e=250.0, nu=0.25, tsc=5.0, damp=0.15, ncycle=10,
            tab_id=101, epsd_ref=0.01, fscale=1.5, srclmt=1.0e15, nrs=1,
            title="Pickle-Foam",
        )

    def test_write_read_restart_with_law163_model(self, tmp_path: Path):
        """Verify write_restart and read_restart preserve LAW163 persistent material arrays."""
        deck = StarterDeck("RST_LAW163_ARRAYS")
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0), (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
            (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0), (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0),
            (9, 2.0, 0.0, 0.0), (10, 2.0, 1.0, 0.0), (11, 2.0, 0.0, 1.0), (12, 2.0, 1.0, 1.0),
        ])
        deck.brick(1, [
            [1, 1, 2, 3, 4, 5, 6, 7, 8],
            [2, 2, 9, 10, 3, 6, 11, 12, 7],
        ])
        deck.prop_solid(1, "SolidProp")
        deck.part(1, "FoamPart", prop_id=1, mat_id=1)
        deck.mat_law163(
            mat_id=1,
            title="Restart-Foam",
            rho=1.2e-3,
            e=250.0,
            nu=0.25,
            tsc=5.0,
            damp=0.15,
            ncycle=10,
            tab_id=101,
            epsd_ref=0.01,
            fscale=1.5,
            srclmt=1.0e15,
            nrs=1,
        )

        rad_path = tmp_path / "rst_law163_0000.rad"
        deck.write(str(rad_path))

        model = run_starter(str(rad_path))
        assert model is not None

        groups = dict(model.element_groups())
        bg = groups.get("bricks") or groups.get("solids")
        st = bg.state

        # Synthesize realistic persistent material state arrays for LAW163 (2 elements)
        uvar163_synth = np.array([
            [0.01, 10.0],
            [0.02, 10.0],
        ], dtype=float)
        epsd163_synth = np.array([0.05, 0.06], dtype=float)
        sigv_synth = np.array([
            [0.12, 0.05, 0.05, 0.02, 0.0, 0.0],
            [0.24, 0.10, 0.10, 0.04, 0.0, 0.0],
        ], dtype=float)
        uvar1_synth = uvar163_synth[:, 0].copy()
        epsd_synth = epsd163_synth.copy()
        sig_synth = np.array([
            [5.0, 1.0, 1.0, 0.5, 0.0, 0.0],
            [6.0, 1.2, 1.2, 0.6, 0.0, 0.0],
        ], dtype=float)

        st["mat_extra"]["uvar163"] = uvar163_synth.copy()
        st["mat_extra"]["epsd163"] = epsd163_synth.copy()
        st["mat_extra"]["sigv"] = sigv_synth.copy()
        st["mat_extra"]["uvar1"] = uvar1_synth.copy()
        st["mat_extra"]["epsd"] = epsd_synth.copy()
        st["sig"] = sig_synth.copy()

        # Write restart .rst
        rst_path = tmp_path / "rst_law163_0001.rst"
        engine_dict = {
            "cycle": 50,
            "t": 0.005,
            "dt": 1.0e-7,
        }
        write_restart(model, str(rst_path), engine=engine_dict)

        # Read back from restart .rst
        rest_model, rest_engine = read_restart(str(rst_path))
        assert rest_engine["cycle"] == 50
        assert rest_engine["t"] == pytest.approx(0.005)
        assert rest_engine["dt"] == pytest.approx(1.0e-7)

        # Assert exact preservation of LAW163 state arrays
        rest_bg = dict(rest_model.element_groups()).get("bricks") or dict(rest_model.element_groups()).get("solids")
        rest_extra = rest_bg.state["mat_extra"]
        rest_st = rest_bg.state

        np.testing.assert_array_equal(rest_extra["uvar163"], uvar163_synth)
        np.testing.assert_array_equal(rest_extra["epsd163"], epsd163_synth)
        np.testing.assert_array_equal(rest_extra["sigv"], sigv_synth)
        np.testing.assert_array_equal(rest_extra["uvar1"], uvar1_synth)
        np.testing.assert_array_equal(rest_extra["epsd"], epsd_synth)
        np.testing.assert_array_equal(rest_st["sig"], sig_synth)

    def test_constitutive_dynamic_cycle_continuation_from_restart(self):
        """Constitutive stress update continuation from restart matches uninterrupted simulation within 10^-12."""
        mat = build_law163({
            "id": 163,
            "rho0": 1.2e-3,
            "e": 250.0,
            "nu": 0.25,
            "tsc": 5.0,
            "damp": 0.15,
            "ncycle": 10,
            "tab_id": 0,
            "epsd_ref": 0.0,
            "fscale": 1.0,
            "srclmt": 1.0e20,
            "nrs": 0,
        })

        dt = 1.0e-5
        deps_steps = [
            np.array([[-0.005, 0.001, 0.001, 0.0005, 0.0, 0.0]]),
            np.array([[-0.010, 0.002, 0.002, 0.0010, 0.0, 0.0]]),
            np.array([[-0.015, 0.003, 0.003, 0.0015, 0.0, 0.0]]),
            np.array([[-0.020, 0.004, 0.004, 0.0020, 0.0, 0.0]]),
        ]

        # 1. Uninterrupted run: 4 steps
        sig_uninterrupted = np.zeros((1, 6))
        epsp_uninterrupted = np.zeros(1)
        extra_uninterrupted = {
            "le": np.array([10.0]),
        }

        for deps in deps_steps:
            sig_uninterrupted, epsp_uninterrupted, _ = solid_update(
                mat,
                sig_uninterrupted,
                deps=deps,
                epsp=epsp_uninterrupted,
                dt=dt,
                extra=extra_uninterrupted,
                return_tuple=True,
            )

        # 2. Resumed run: 2 steps, snapshot/restart, continue 2 steps
        sig_restarted = np.zeros((1, 6))
        epsp_restarted = np.zeros(1)
        extra_restarted = {
            "le": np.array([10.0]),
        }

        for deps in deps_steps[:2]:
            sig_restarted, epsp_restarted, _ = solid_update(
                mat,
                sig_restarted,
                deps=deps,
                epsp=epsp_restarted,
                dt=dt,
                extra=extra_restarted,
                return_tuple=True,
            )

        # Snapshot state via pickle (simulating restart file write/read)
        snap = pickle.dumps({
            "sig": sig_restarted.copy(),
            "epsp": epsp_restarted.copy(),
            "extra": {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in extra_restarted.items()},
        })

        # Restore from snapshot
        restored = pickle.loads(snap)
        sig_resumed = restored["sig"]
        epsp_resumed = restored["epsp"]
        extra_resumed = restored["extra"]

        # Continue next 2 steps
        for deps in deps_steps[2:]:
            sig_resumed, epsp_resumed, _ = solid_update(
                mat,
                sig_resumed,
                deps=deps,
                epsp=epsp_resumed,
                dt=dt,
                extra=extra_resumed,
                return_tuple=True,
            )

        # Assert exact continuation within 10^-12
        np.testing.assert_allclose(sig_resumed, sig_uninterrupted, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(epsp_resumed, epsp_uninterrupted, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["uvar163"], extra_uninterrupted["uvar163"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["epsd163"], extra_uninterrupted["epsd163"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["sigv"], extra_uninterrupted["sigv"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["uvar1"], extra_uninterrupted["uvar1"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(extra_resumed["epsd"], extra_uninterrupted["epsd"], rtol=1e-12, atol=1e-12)
