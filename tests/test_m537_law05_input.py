"""
Tests for Milestone M537: /MAT/LAW5 and /MAT/JWL card layouts, deck writer emitter,
CFG catalogue synonyms, starter checks, and roundtrip parsing fidelity.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input import deck_writer as dw
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.mat_reader import catalogue, parse_generic_mat, CfgCatalogue
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.checks import _ALLOWED_LAWS, check_model


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


class TestLaw5CardLayouts:
    """Card layout constants and LAYOUTS dictionary verification (M537)."""

    def test_constants_and_aliases(self):
        assert cl.MAT_LAW5_1 == (20, 20)
        assert cl.MAT_LAW5_2 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW5_3 == (20, 20, 20, 20, 10, 10)
        assert cl.MAT_LAW5_4 == (20, 20, 20)
        assert cl.MAT_LAW5_5_OPT1 == (20, 20)
        assert cl.MAT_LAW5_5_OPT2 == (20, 20, 20)

        assert cl.MAT_LAW5_CFG_1 == (20, 20)
        assert cl.MAT_LAW5_CFG_2 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW5_CFG_3 == (20, 20, 20, 20, 10, 10)
        assert cl.MAT_LAW5_CFG_4 == (20, 20, 20)
        assert cl.MAT_LAW5_CFG_5_OPT1 == (20, 20)
        assert cl.MAT_LAW5_CFG_5_OPT2 == (20, 20, 20)

        assert cl.MAT_JWL_1 == (20, 20)
        assert cl.MAT_JWL_2 == (20, 20, 20, 20, 20)
        assert cl.MAT_JWL_3 == (20, 20, 20, 20, 10, 10)
        assert cl.MAT_JWL_4 == (20, 20, 20)
        assert cl.MAT_JWL_5_OPT1 == (20, 20)
        assert cl.MAT_JWL_5_OPT2 == (20, 20, 20)

        assert cl.MAT_JWL_CFG_1 == (20, 20)
        assert cl.MAT_JWL_CFG_2 == (20, 20, 20, 20, 20)
        assert cl.MAT_JWL_CFG_3 == (20, 20, 20, 20, 10, 10)
        assert cl.MAT_JWL_CFG_4 == (20, 20, 20)
        assert cl.MAT_JWL_CFG_5_OPT1 == (20, 20)
        assert cl.MAT_JWL_CFG_5_OPT2 == (20, 20, 20)

    def test_layouts_dictionary(self):
        assert cl.CARD_LAYOUTS["MAT_LAW5_1"] == [20, 20]
        assert cl.CARD_LAYOUTS["MAT_LAW5_2"] == [20, 20, 20, 20, 20]
        assert cl.CARD_LAYOUTS["MAT_LAW5_3"] == [20, 20, 20, 20, 10, 10]
        assert cl.CARD_LAYOUTS["MAT_LAW5_4"] == [20, 20, 20]
        assert cl.CARD_LAYOUTS["MAT_LAW5_5_OPT1"] == [20, 20]
        assert cl.CARD_LAYOUTS["MAT_LAW5_5_OPT2"] == [20, 20, 20]

        assert cl.CARD_LAYOUTS["MAT_JWL_1"] == [20, 20]
        assert cl.CARD_LAYOUTS["MAT_JWL_2"] == [20, 20, 20, 20, 20]
        assert cl.CARD_LAYOUTS["MAT_JWL_3"] == [20, 20, 20, 20, 10, 10]
        assert cl.CARD_LAYOUTS["MAT_JWL_4"] == [20, 20, 20]
        assert cl.CARD_LAYOUTS["MAT_JWL_5_OPT1"] == [20, 20]
        assert cl.CARD_LAYOUTS["MAT_JWL_5_OPT2"] == [20, 20, 20]


class TestLaw5DeckWriter:
    """Deck writer emitter tests for /MAT/LAW5 and /MAT/JWL (M537)."""

    def test_mat_law5_basic_emission(self):
        """Card layout test for standard JWL without afterburning (4 data cards)."""
        d = dw.StarterDeck("TNT")
        d.mat_law5(
            mat_id=1,
            title="TNT_Explosive",
            rho=1.63e-6,
            a=3.712e5,
            b=3.23e3,
            r1=4.15,
            r2=0.95,
            omega=0.30,
            d=6.93,
            pcj=2.1e4,
            e0=7.0e3,
            eadd=0.0,
            ibfrac=0,
            qopt=0,
            p0=1.0,
            psh=0.0,
            bunreacted=100.0,
        )
        rendered = d.render()
        cards = data_cards(block_lines(rendered, "/MAT/LAW5/1"))
        assert len(cards) == 5  # Title + 4 data cards
        assert cards[0] == "TNT_Explosive"

        # Card 1: RHO_I (20 cols)
        assert float(cards[1][0:20]) == 1.63e-6
        assert len(cards[1]) == 20

        # Card 2: A, B, R1, R2, Omega (5x 20 cols)
        assert float(cards[2][0:20]) == 3.712e5
        assert float(cards[2][20:40]) == 3.23e3
        assert float(cards[2][40:60]) == 4.15
        assert float(cards[2][60:80]) == 0.95
        assert float(cards[2][80:100]) == 0.30

        # Card 3: D, P_CJ, E0, Eadd, I_BFRAC, Qopt (4x 20 cols, 2x 10 cols)
        assert float(cards[3][0:20]) == 6.93
        assert float(cards[3][20:40]) == 2.1e4
        assert float(cards[3][40:60]) == 7.0e3
        assert float(cards[3][60:80]) == 0.0
        assert int(cards[3][80:90]) == 0
        assert int(cards[3][90:100]) == 0

        # Card 4: P0, PSH, Bunreacted (3x 20 cols)
        assert float(cards[4][0:20]) == 1.0
        assert float(cards[4][20:40]) == 0.0
        assert float(cards[4][40:60]) == 100.0

    def test_mat_law5_with_rho_ref(self):
        """Card 1 emits both RHO_I and RHO_O when rho_ref is provided."""
        d = dw.StarterDeck("TNT_REF")
        d.mat_law5(
            mat_id=2,
            title="TNT_Dual_Rho",
            rho=1.63e-6,
            rho_ref=1.60e-6,
            a=3.712e5,
            b=3.23e3,
            r1=4.15,
            r2=0.95,
            omega=0.30,
            d=6.93,
            pcj=2.1e4,
            e0=7.0e3,
        )
        rendered = d.render()
        cards = data_cards(block_lines(rendered, "/MAT/LAW5/2"))
        assert len(cards) == 5
        assert float(cards[1][0:20]) == 1.63e-6
        assert float(cards[1][20:40]) == 1.60e-6

    def test_mat_law5_afterburning_qopt1(self):
        """Card 5 emits TSTART and TSTOP for QOPT in (0, 1, 2) when Eadd > 0."""
        d = dw.StarterDeck("TNT_AB")
        d.mat_law5(
            mat_id=3,
            title="Afterburn_Const",
            rho=1.63e-6,
            a=3.712e5,
            b=3.23e3,
            r1=4.15,
            r2=0.95,
            omega=0.30,
            d=6.93,
            pcj=2.1e4,
            e0=7.0e3,
            eadd=1200.0,
            ibfrac=1,
            qopt=1,
            p0=0.0,
            psh=0.0,
            bunreacted=0.0,
            tstart=0.25,
            tstop=1.75,
        )
        rendered = d.render()
        cards = data_cards(block_lines(rendered, "/MAT/LAW5/3"))
        assert len(cards) == 6  # Title + 5 data cards
        # Card 3: check Eadd and QOPT
        assert float(cards[3][60:80]) == 1200.0
        assert int(cards[3][80:90]) == 1
        assert int(cards[3][90:100]) == 1
        # Card 5: TSTART, TSTOP
        assert float(cards[5][0:20]) == 0.25
        assert float(cards[5][20:40]) == 1.75

    def test_mat_law5_afterburning_qopt3_miller(self):
        """Card 5 emits a, m, n for Miller's extension (QOPT=3)."""
        d = dw.StarterDeck("TNT_MILLER")
        d.mat_law5(
            mat_id=4,
            title="Miller_Extension",
            rho=1.63e-6,
            a=3.712e5,
            b=3.23e3,
            r1=4.15,
            r2=0.95,
            omega=0.30,
            d=6.93,
            pcj=2.1e4,
            e0=7.0e3,
            eadd=1500.0,
            ibfrac=2,
            qopt=3,
            a_mil=2.5,
            m_mil=0.75,
            n_mil=1.8,
        )
        rendered = d.render()
        cards = data_cards(block_lines(rendered, "/MAT/LAW5/4"))
        assert len(cards) == 6
        assert float(cards[5][0:20]) == 2.5
        assert float(cards[5][20:40]) == 0.75
        assert float(cards[5][40:60]) == 1.8

    def test_mat_jwl_alias_and_headers(self):
        d = dw.StarterDeck("TNT_JWL")
        assert d.mat_jwl == d.mat_law5

        # With unit_id
        d.mat_law5(mat_id=5, rho=1.63e-6, a=1.0, b=2.0, r1=3.0, r2=4.0, omega=0.5,
                   d=6.0, pcj=7.0, e0=8.0, unit_id=2)
        assert "/MAT/LAW5/5/2" in d.render()

        # With law_name="JWL"
        d.mat_jwl(mat_id=6, rho=1.63e-6, a=1.0, b=2.0, r1=3.0, r2=4.0, omega=0.5,
                  d=6.0, pcj=7.0, e0=8.0, law_name="JWL")
        assert "/MAT/JWL/6" in d.render()


class TestMatReaderAndCfgCatalogue:
    """Verification of CFG catalogue schemas and JWL synonym mapping (M537)."""

    def test_jwl_synonym_in_catalogue(self):
        assert "JWL" in CfgCatalogue._SYNONYMS
        assert CfgCatalogue._SYNONYMS["JWL"] == "LAW5"

    def test_schema_lookup(self):
        cat = catalogue()
        schema_law5 = cat.schema("LAW5")
        schema_jwl = cat.schema("JWL")
        assert schema_law5 is not None
        assert schema_jwl is not None
        assert schema_law5.path == schema_jwl.path
        assert "MAT_A" in schema_law5.attributes
        assert "MAT_PC" in schema_law5.attributes
        assert "MAT_E0" in schema_law5.attributes
        assert "Omega" in schema_law5.attributes


class TestRoundtripParsingFidelity:
    """Full roundtrip fidelity: deck_writer -> deck_reader -> parse_generic_mat."""

    def test_roundtrip_law5_and_jwl(self, tmp_path: Path):
        d = dw.StarterDeck("ROUNDTRIP_DECK")
        # Material 1: Standard LAW5 without afterburning
        d.mat_law5(
            mat_id=1,
            title="Mat_Standard_JWL",
            rho=1.63e-6,
            rho_ref=1.63e-6,
            a=3.712e5,
            b=3.23e3,
            r1=4.15,
            r2=0.95,
            omega=0.30,
            d=6.93,
            pcj=2.1e4,
            e0=7.0e3,
            eadd=0.0,
            ibfrac=0,
            qopt=0,
            p0=1.0,
            psh=0.0,
            bunreacted=50.0,
        )
        # Material 2: LAW5 with afterburning QOPT=1
        d.mat_law5(
            mat_id=2,
            title="Mat_Afterburning_Const",
            rho=1.80e-6,
            a=4.0e5,
            b=4.0e3,
            r1=4.5,
            r2=1.0,
            omega=0.35,
            d=7.5,
            pcj=2.5e4,
            e0=8.0e3,
            eadd=1000.0,
            ibfrac=1,
            qopt=1,
            tstart=0.5,
            tstop=2.0,
        )
        # Material 3: JWL with Miller's extension QOPT=3
        d.mat_jwl(
            mat_id=3,
            title="Mat_Miller_Ext",
            rho=1.75e-6,
            a=5.0e5,
            b=5.0e3,
            r1=5.0,
            r2=1.1,
            omega=0.40,
            d=8.0,
            pcj=3.0e4,
            e0=9.0e3,
            eadd=1500.0,
            ibfrac=2,
            qopt=3,
            a_mil=2.0,
            m_mil=0.8,
            n_mil=1.5,
            law_name="JWL",
        )

        deck_text = d.render()
        deck_file = tmp_path / "test_law5_deck.rad"
        deck_file.write_text(deck_text, encoding="utf-8")

        blocks = read_deck(str(deck_file))
        mat_blocks = [b for b in blocks if len(b.parts) >= 2 and b.parts[0].upper() == "MAT"]
        assert len(mat_blocks) == 3

        log = MessageLog()

        # Parse Material 1
        rec1 = parse_generic_mat(mat_blocks[0], log)
        assert rec1 is not None
        assert rec1.id == 1
        assert rec1.density == pytest.approx(1.63e-6)
        assert rec1.params["Refer_Rho"] == pytest.approx(1.63e-6)
        assert rec1.params["MAT_A"] == pytest.approx(3.712e5)
        assert rec1.params["MAT_B"] == pytest.approx(3.23e3)
        assert rec1.params["MAT_PDIR1"] == pytest.approx(4.15)
        assert rec1.params["MAT_PDIR2"] == pytest.approx(0.95)
        assert rec1.params["Omega"] == pytest.approx(0.30)
        assert rec1.params["MAT_D"] == pytest.approx(6.93)
        assert rec1.params["MAT_PC"] == pytest.approx(2.1e4)
        assert rec1.params["MAT_E0"] == pytest.approx(7.0e3)
        assert rec1.params["MAT_E"] == pytest.approx(0.0)
        assert rec1.params["MAT_IBFRAC"] == 0
        assert rec1.params["QOPT"] == 0
        assert rec1.params["LAW5_P0"] == pytest.approx(1.0)
        assert rec1.params["BUNREACTED"] == pytest.approx(50.0)

        # Parse Material 2
        rec2 = parse_generic_mat(mat_blocks[1], log)
        assert rec2 is not None
        assert rec2.id == 2
        assert rec2.density == pytest.approx(1.80e-6)
        assert rec2.params["MAT_E"] == pytest.approx(1000.0)
        assert rec2.params["MAT_IBFRAC"] == 1
        assert rec2.params["QOPT"] == 1
        assert rec2.params["TSTART"] == pytest.approx(0.5)
        assert rec2.params["TSTOP"] == pytest.approx(2.0)

        # Parse Material 3
        rec3 = parse_generic_mat(mat_blocks[2], log)
        assert rec3 is not None
        assert rec3.id == 3
        assert rec3.density == pytest.approx(1.75e-6)
        assert rec3.params["MAT_E"] == pytest.approx(1500.0)
        assert rec3.params["MAT_IBFRAC"] == 2
        assert rec3.params["QOPT"] == 3
        assert rec3.params["LAW5_A"] == pytest.approx(2.0)
        assert rec3.params["LAW5_M"] == pytest.approx(0.8)
        assert rec3.params["LAW5_N"] == pytest.approx(1.5)

        assert len(log.errors) == 0

    def test_normalize_roundtrip(self, tmp_path: Path):
        """Test StarterDeck.normalize_deck with LAW5 and JWL."""
        deck_text = """\
# RADIOSS STARTER
/BEGIN
NORM_TEST
                2022
/MAT/LAW5/1
Standard LAW5
               1.6e-6
              3.7e+05              3.2e+03                 4.15                 0.95                  0.3
                 6.93              2.1e+04              7.0e+03                  0.0         0         0
                  0.0                  0.0                  0.0
/MAT/JWL/2
Afterburn JWL
               1.7e-6
              4.0e+05              4.0e+03                  4.5                  1.0                 0.35
                  7.5              2.5e+04              8.0e+03               1000.0         1         1
                  0.0                  0.0                  0.0
                  0.5                  2.0
/END
"""
        d = dw.starter_deck_from_lines(deck_text.splitlines(), "NORM_TEST")
        norm = d.render()

        assert "/MAT/LAW5/1" in norm
        assert "/MAT/JWL/2" in norm

        cards1 = data_cards(block_lines(norm, "/MAT/LAW5/1"))
        assert len(cards1) == 5
        assert float(cards1[1][0:20]) == pytest.approx(1.6e-6)
        assert float(cards1[2][0:20]) == pytest.approx(3.7e5)

        cards2 = data_cards(block_lines(norm, "/MAT/JWL/2"))
        assert len(cards2) == 6
        assert float(cards2[1][0:20]) == pytest.approx(1.7e-6)
        assert float(cards2[5][0:20]) == pytest.approx(0.5)
        assert float(cards2[5][20:40]) == pytest.approx(2.0)


class TestStarterChecksAllowedLaws:
    """Verify checks.py element family compatibility rules for LAW5/JWL (M537)."""

    def test_allowed_laws_contains_law5_and_jwl(self):
        for key in (5, "5", "LAW5", "JWL"):
            assert key in _ALLOWED_LAWS["bricks"], f"{key} missing in bricks"
            assert key in _ALLOWED_LAWS["tetras"], f"{key} missing in tetras"

    def test_check_model_accepts_law5_on_solid(self):
        import numpy as np
        from pyradioss.model.entities import Material
        model = Model()
        model.add_nodes(np.arange(1, 9, dtype=np.int64), np.zeros((8, 3)))
        mat = Material(id=1, law=5, rho0=1.63e-6, title="TNT", params={"E": 1e9, "nu": 0.3})
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model._brick_groups = {"brics": FakeGroup()}
        model.element_groups = lambda: [("bricks", model._brick_groups["brics"])]

        log = MessageLog()
        check_model(model, log)
        assert len(log.errors) == 0, f"Unexpected errors: {log.errors}"
