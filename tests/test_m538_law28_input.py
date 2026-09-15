"""
Tests for Milestone M538: /MAT/LAW28 and /MAT/HONEYCOMB card layouts, deck writer emitter,
CFG catalogue synonyms, starter checks, and roundtrip parsing fidelity.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input import deck_writer as dw
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.mat_reader import catalogue, parse_generic_mat, CfgCatalogue
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import Material
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


class TestLaw28CardLayouts:
    """Card layout constants and LAYOUTS dictionary verification (M538)."""

    def test_constants_and_aliases(self):
        assert cl.MAT_LAW28_1 == (20, 20)
        assert cl.MAT_LAW28_2 == (20, 20, 20)
        assert cl.MAT_LAW28_3 == (20, 20, 20)
        assert cl.MAT_LAW28_4 == (10, 10, 10, 10, 20, 20, 20)
        assert cl.MAT_LAW28_5 == (20, 20, 20)
        assert cl.MAT_LAW28_6 == (10, 10, 10, 10, 20, 20, 20)
        assert cl.MAT_LAW28_7 == (20, 20, 20)

        assert cl.MAT_LAW28_CFG_1 == (20, 20)
        assert cl.MAT_LAW28_CFG_2 == (20, 20, 20)
        assert cl.MAT_LAW28_CFG_3 == (20, 20, 20)
        assert cl.MAT_LAW28_CFG_4 == (10, 10, 10, 10, 20, 20, 20)
        assert cl.MAT_LAW28_CFG_5 == (20, 20, 20)
        assert cl.MAT_LAW28_CFG_6 == (10, 10, 10, 10, 20, 20, 20)
        assert cl.MAT_LAW28_CFG_7 == (20, 20, 20)

        assert cl.MAT_HONEYCOMB_1 == (20, 20)
        assert cl.MAT_HONEYCOMB_2 == (20, 20, 20)
        assert cl.MAT_HONEYCOMB_3 == (20, 20, 20)
        assert cl.MAT_HONEYCOMB_4 == (10, 10, 10, 10, 20, 20, 20)
        assert cl.MAT_HONEYCOMB_5 == (20, 20, 20)
        assert cl.MAT_HONEYCOMB_6 == (10, 10, 10, 10, 20, 20, 20)
        assert cl.MAT_HONEYCOMB_7 == (20, 20, 20)

        assert cl.MAT_HONEYCOMB_CFG_1 == (20, 20)
        assert cl.MAT_HONEYCOMB_CFG_2 == (20, 20, 20)
        assert cl.MAT_HONEYCOMB_CFG_3 == (20, 20, 20)
        assert cl.MAT_HONEYCOMB_CFG_4 == (10, 10, 10, 10, 20, 20, 20)
        assert cl.MAT_HONEYCOMB_CFG_5 == (20, 20, 20)
        assert cl.MAT_HONEYCOMB_CFG_6 == (10, 10, 10, 10, 20, 20, 20)
        assert cl.MAT_HONEYCOMB_CFG_7 == (20, 20, 20)

    def test_layouts_dictionary(self):
        expected_widths = {
            1: [20, 20],
            2: [20, 20, 20],
            3: [20, 20, 20],
            4: [10, 10, 10, 10, 20, 20, 20],
            5: [20, 20, 20],
            6: [10, 10, 10, 10, 20, 20, 20],
            7: [20, 20, 20],
        }
        for i in range(1, 8):
            w = expected_widths[i]
            assert cl.LAYOUTS[f"MAT_LAW28_{i}"] == w
            assert cl.LAYOUTS[f"MAT_LAW28_CFG_{i}"] == w
            assert cl.LAYOUTS[f"MAT_HONEYCOMB_{i}"] == w
            assert cl.LAYOUTS[f"MAT_HONEYCOMB_CFG_{i}"] == w
            assert cl.CARD_LAYOUTS[f"MAT_LAW28_{i}"] == w
            assert cl.CARD_LAYOUTS[f"MAT_LAW28_CFG_{i}"] == w
            assert cl.CARD_LAYOUTS[f"MAT_HONEYCOMB_{i}"] == w
            assert cl.CARD_LAYOUTS[f"MAT_HONEYCOMB_CFG_{i}"] == w

    def test_split_fixed_cutting(self):
        # Card 4: 4x 10, 3x 20
        raw_c4 = f"{11:>10d}{22:>10d}{33:>10d}{1:>10d}{1.1:>20.4f}{1.2:>20.4f}{1.3:>20.4f}"
        cut4 = cl.split_fixed(raw_c4, cl.LAYOUTS["MAT_LAW28_4"])
        assert cut4 == ["11", "22", "33", "1", "1.1000", "1.2000", "1.3000"]

        # Card 6: 4x 10, 3x 20
        raw_c6 = f"{12:>10d}{23:>10d}{31:>10d}{0:>10d}{0.9:>20.4f}{1.0:>20.4f}{1.05:>20.4f}"
        cut6 = cl.split_fixed(raw_c6, cl.LAYOUTS["MAT_HONEYCOMB_6"])
        assert cut6 == ["12", "23", "31", "0", "0.9000", "1.0000", "1.0500"]


class TestLaw28DeckWriter:
    """Deck writer emitter tests for /MAT/LAW28 and /MAT/HONEYCOMB (M538)."""

    def test_mat_law28_basic_emission(self):
        d = dw.StarterDeck("HONEYCOMB_DECK")
        d.mat_law28(
            mat_id=1,
            title="Honeycomb_Test",
            rho=3.5e-4,
            e11=100.0,
            e22=120.0,
            e33=500.0,
            g12=40.0,
            g23=60.0,
            g31=50.0,
            fun_a1=11,
            fun_b1=22,
            fun_a2=33,
            gflag=1,
            fscale11=1.0,
            fscale22=1.1,
            fscale33=1.2,
            eps_max11=0.5,
            eps_max22=0.4,
            eps_max33=0.8,
            fun_a3=12,
            fun_b3=23,
            fun_a4=31,
            vflag=0,
            fscale12=0.9,
            fscale23=1.0,
            fscale31=1.05,
            eps_max12=0.3,
            eps_max23=0.35,
            eps_max31=0.45,
        )
        rendered = d.render()
        cards = data_cards(block_lines(rendered, "/MAT/LAW28/1"))
        assert len(cards) == 8  # Title + 7 data cards
        assert cards[0] == "Honeycomb_Test"

        # Card 1: RHO_I (20 cols)
        assert float(cards[1][0:20]) == pytest.approx(3.5e-4)
        assert len(cards[1]) == 20

        # Card 2: E11, E22, E33 (3x 20 cols)
        assert float(cards[2][0:20]) == pytest.approx(100.0)
        assert float(cards[2][20:40]) == pytest.approx(120.0)
        assert float(cards[2][40:60]) == pytest.approx(500.0)

        # Card 3: G12, G23, G31 (3x 20 cols)
        assert float(cards[3][0:20]) == pytest.approx(40.0)
        assert float(cards[3][20:40]) == pytest.approx(60.0)
        assert float(cards[3][40:60]) == pytest.approx(50.0)

        # Card 4: fun_a1, fun_b1, fun_a2, gflag (4x 10 cols), fscale11, fscale22, fscale33 (3x 20 cols)
        assert int(cards[4][0:10]) == 11
        assert int(cards[4][10:20]) == 22
        assert int(cards[4][20:30]) == 33
        assert int(cards[4][30:40]) == 1
        assert float(cards[4][40:60]) == pytest.approx(1.0)
        assert float(cards[4][60:80]) == pytest.approx(1.1)
        assert float(cards[4][80:100]) == pytest.approx(1.2)

        # Card 5: eps_max11, eps_max22, eps_max33 (3x 20 cols)
        assert float(cards[5][0:20]) == pytest.approx(0.5)
        assert float(cards[5][20:40]) == pytest.approx(0.4)
        assert float(cards[5][40:60]) == pytest.approx(0.8)

        # Card 6: fun_a3, fun_b3, fun_a4, vflag (4x 10 cols), fscale12, fscale23, fscale31 (3x 20 cols)
        assert int(cards[6][0:10]) == 12
        assert int(cards[6][10:20]) == 23
        assert int(cards[6][20:30]) == 31
        assert int(cards[6][30:40]) == 0
        assert float(cards[6][40:60]) == pytest.approx(0.9)
        assert float(cards[6][60:80]) == pytest.approx(1.0)
        assert float(cards[6][80:100]) == pytest.approx(1.05)

        # Card 7: eps_max12, eps_max23, eps_max31 (3x 20 cols)
        assert float(cards[7][0:20]) == pytest.approx(0.3)
        assert float(cards[7][20:40]) == pytest.approx(0.35)
        assert float(cards[7][40:60]) == pytest.approx(0.45)

    def test_mat_law28_with_rho_ref(self):
        d = dw.StarterDeck("HONEYCOMB_REF")
        d.mat_law28(
            mat_id=2,
            title="Dual_Rho_Honeycomb",
            rho=3.5e-4,
            rho_ref=3.0e-4,
            e11=100.0,
            e22=120.0,
            e33=500.0,
        )
        rendered = d.render()
        cards = data_cards(block_lines(rendered, "/MAT/LAW28/2"))
        assert len(cards) == 8
        assert float(cards[1][0:20]) == pytest.approx(3.5e-4)
        assert float(cards[1][20:40]) == pytest.approx(3.0e-4)

    def test_mat_honeycomb_alias_and_headers(self):
        d = dw.StarterDeck("HONEYCOMB_ALIAS")
        assert d.mat_honeycomb == d.mat_law28

        # With unit_id
        d.mat_law28(mat_id=3, rho=3.5e-4, e11=100.0, unit_id=5)
        assert "/MAT/LAW28/3/5" in d.render()

        # With law_name="HONEYCOMB"
        d.mat_honeycomb(mat_id=4, rho=3.5e-4, e11=100.0, law_name="HONEYCOMB")
        assert "/MAT/HONEYCOMB/4" in d.render()


class TestMatReaderAndCfgCatalogue:
    """Verification of CFG catalogue schemas and HONEYCOMB synonym mapping (M538)."""

    def test_honeycomb_synonym_in_catalogue(self):
        assert "HONEYCOMB" in CfgCatalogue._SYNONYMS
        assert CfgCatalogue._SYNONYMS["HONEYCOMB"] == "LAW28"

    def test_schema_lookup(self):
        cat = catalogue()
        schema_law28 = cat.schema("LAW28")
        schema_honeycomb = cat.schema("HONEYCOMB")
        assert schema_law28 is not None
        assert schema_honeycomb is not None
        assert schema_law28.path == schema_honeycomb.path

        expected_attrs = [
            "MAT_RHO", "Refer_Rho", "MAT_EA", "MAT_EB", "MAT_EC",
            "MAT_GAB", "MAT_GBC", "MAT_GCA", "FUN_A1", "FUN_B1", "FUN_A2",
            "Gflag", "FScale11", "FScale22", "FScale33", "MAT_EPSR1", "MAT_EPSR2", "MAT_EPSR3",
            "FUN_A3", "FUN_B3", "FUN_A4", "Vflag", "FScale12", "FScale23", "FScale13",
            "MAT_EPSR4", "MAT_EPSR5", "MAT_EPSR6",
        ]
        for attr in expected_attrs:
            assert attr in schema_law28.attributes, f"Missing attribute {attr} in LAW28 schema"

    def test_parse_generic_mat_honeycomb(self, tmp_path: Path):
        deck_text = """\
# RADIOSS STARTER
/BEGIN
GENERIC_TEST
                2022
/MAT/HONEYCOMB/101
Honeycomb Generic
               3.5e-4              3.2e-4
                100.0               120.0               500.0
                 40.0                60.0                50.0
        11        22        33         1                 1.0                 1.1                 1.2
                  0.5                 0.4                 0.8
        12        23        31         0                 0.9                 1.0                1.05
                  0.3                0.35                0.45
/END
"""
        deck_file = tmp_path / "generic_honeycomb.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        blocks = read_deck(str(deck_file))
        mat_blocks = [b for b in blocks if len(b.parts) >= 2 and b.parts[0].upper() == "MAT"]
        assert len(mat_blocks) == 1

        log = MessageLog()
        rec = parse_generic_mat(mat_blocks[0], log)
        assert rec is not None
        assert rec.id == 101
        assert rec.density == pytest.approx(3.5e-4)
        assert rec.params["Refer_Rho"] == pytest.approx(3.2e-4)
        assert rec.params["MAT_EA"] == pytest.approx(100.0)
        assert rec.params["MAT_EB"] == pytest.approx(120.0)
        assert rec.params["MAT_EC"] == pytest.approx(500.0)
        assert rec.params["MAT_GAB"] == pytest.approx(40.0)
        assert rec.params["MAT_GBC"] == pytest.approx(60.0)
        assert rec.params["MAT_GCA"] == pytest.approx(50.0)
        assert rec.params["FUN_A1"] == 11
        assert rec.params["FUN_B1"] == 22
        assert rec.params["FUN_A2"] == 33
        assert rec.params["Gflag"] == 1
        assert rec.params["FScale11"] == pytest.approx(1.0)
        assert rec.params["FScale22"] == pytest.approx(1.1)
        assert rec.params["FScale33"] == pytest.approx(1.2)
        assert rec.params["MAT_EPSR1"] == pytest.approx(0.5)
        assert rec.params["MAT_EPSR2"] == pytest.approx(0.4)
        assert rec.params["MAT_EPSR3"] == pytest.approx(0.8)
        assert rec.params["FUN_A3"] == 12
        assert rec.params["FUN_B3"] == 23
        assert rec.params["FUN_A4"] == 31
        assert rec.params["Vflag"] == 0
        assert rec.params["FScale12"] == pytest.approx(0.9)
        assert rec.params["FScale23"] == pytest.approx(1.0)
        assert rec.params["FScale13"] == pytest.approx(1.05)
        assert rec.params["MAT_EPSR4"] == pytest.approx(0.3)
        assert rec.params["MAT_EPSR5"] == pytest.approx(0.35)
        assert rec.params["MAT_EPSR6"] == pytest.approx(0.45)
        assert len(log.errors) == 0


class TestStarterKeywordsParsing:
    """Starter keyword parser tests for /MAT/LAW28 and /MAT/HONEYCOMB in free and fixed format."""

    def test_parse_starter_law28_fixed_and_free(self, tmp_path: Path):
        deck_fixed = """\
# RADIOSS STARTER
/BEGIN
STARTER_TEST_FIXED
                2022
/MAT/LAW28/201
LAW28 Fixed Format
            3.500E-4            3.000E-4
            100.0000            120.0000            500.0000
             40.0000             60.0000             50.0000
        11        22        33         1              1.0000              1.1000              1.2000
              0.5000              0.4000              0.8000
        12        23        31         0              0.9000              1.0000              1.0500
              0.3000              0.3500              0.4500
/END
"""
        file_fixed = tmp_path / "starter_law28_fixed.rad"
        file_fixed.write_text(deck_fixed, encoding="utf-8")
        blocks_fixed = read_deck(str(file_fixed))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks_fixed, model, log)
        assert len(log.errors) == 0

        # Mat 201 (LAW28 Fixed)
        assert 201 in model.mat_law28s
        m1 = model.mat_law28s[201]
        assert m1.id == 201
        assert m1.title == "LAW28 Fixed Format"
        assert m1.rho0 == pytest.approx(3.5e-4)
        assert m1.ref_rho == pytest.approx(3.0e-4)
        assert m1.e11 == pytest.approx(100.0)
        assert m1.e22 == pytest.approx(120.0)
        assert m1.e33 == pytest.approx(500.0)
        assert m1.g12 == pytest.approx(40.0)
        assert m1.g23 == pytest.approx(60.0)
        assert m1.g31 == pytest.approx(50.0)
        assert m1.fun_a1 == 11
        assert m1.fun_b1 == 22
        assert m1.fun_a2 == 33
        assert m1.fun_id11 == 11
        assert m1.fun_id22 == 22
        assert m1.fun_id33 == 33
        assert m1.gflag == 1
        assert m1.fscale11 == pytest.approx(1.0)
        assert m1.fscale22 == pytest.approx(1.1)
        assert m1.fscale33 == pytest.approx(1.2)
        assert m1.epsr1 == pytest.approx(0.5)
        assert m1.epsr2 == pytest.approx(0.4)
        assert m1.epsr3 == pytest.approx(0.8)
        assert m1.eps_max11 == pytest.approx(0.5)
        assert m1.eps_max22 == pytest.approx(0.4)
        assert m1.eps_max33 == pytest.approx(0.8)
        assert m1.fun_a3 == 12
        assert m1.fun_b3 == 23
        assert m1.fun_a4 == 31
        assert m1.fun_id12 == 12
        assert m1.fun_id23 == 23
        assert m1.fun_id31 == 31
        assert m1.vflag == 0
        assert m1.fscale12 == pytest.approx(0.9)
        assert m1.fscale23 == pytest.approx(1.0)
        assert m1.fscale13 == pytest.approx(1.05)
        assert m1.epsr4 == pytest.approx(0.3)
        assert m1.epsr5 == pytest.approx(0.35)
        assert m1.epsr6 == pytest.approx(0.45)
        assert m1.eps_max12 == pytest.approx(0.3)
        assert m1.eps_max23 == pytest.approx(0.35)
        assert m1.eps_max31 == pytest.approx(0.45)

        # Free format deck
        deck_free = """\
# RADIOSS STARTER
/MAT/HONEYCOMB/202
HONEYCOMB Free Format
3.2e-4
80.0 90.0 400.0
30.0 50.0 45.0
10 20 30 0 1.0 1.0 1.0
0.4 0.35 0.7
11 21 31 1 1.0 1.0 1.0
0.25 0.3 0.4
/END
"""
        file_free = tmp_path / "starter_honeycomb_free.rad"
        file_free.write_text(deck_free, encoding="utf-8")
        blocks_free = read_deck(str(file_free))
        model2 = Model()
        log2 = MessageLog()
        parse_starter_deck(blocks_free, model2, log2)
        assert len(log2.errors) == 0

        # Mat 202 (HONEYCOMB Free)
        assert 202 in model2.mat_law28s
        m2 = model2.mat_law28s[202]
        assert m2.id == 202
        assert m2.title == "HONEYCOMB Free Format"
        assert m2.rho0 == pytest.approx(3.2e-4)
        assert m2.e11 == pytest.approx(80.0)
        assert m2.e22 == pytest.approx(90.0)
        assert m2.e33 == pytest.approx(400.0)
        assert m2.g12 == pytest.approx(30.0)
        assert m2.g23 == pytest.approx(50.0)
        assert m2.g31 == pytest.approx(45.0)
        assert m2.fun_a1 == 10
        assert m2.fun_b1 == 20
        assert m2.fun_a2 == 30
        assert m2.gflag == 0
        assert m2.fscale11 == pytest.approx(1.0)
        assert m2.fscale22 == pytest.approx(1.0)
        assert m2.fscale33 == pytest.approx(1.0)
        assert m2.epsr1 == pytest.approx(0.4)
        assert m2.epsr2 == pytest.approx(0.35)
        assert m2.epsr3 == pytest.approx(0.7)
        assert m2.fun_a3 == 11
        assert m2.fun_b3 == 21
        assert m2.fun_a4 == 31
        assert m2.vflag == 1
        assert m2.fscale12 == pytest.approx(1.0)
        assert m2.fscale23 == pytest.approx(1.0)
        assert m2.fscale13 == pytest.approx(1.0)
        assert m2.epsr4 == pytest.approx(0.25)
        assert m2.epsr5 == pytest.approx(0.3)
        assert m2.epsr6 == pytest.approx(0.4)


class TestRoundtripParsingFidelity:
    """Full roundtrip fidelity: deck_writer -> deck_reader -> parse_starter_deck & normalize."""

    def test_roundtrip_law28_and_honeycomb(self, tmp_path: Path):
        d = dw.StarterDeck("ROUNDTRIP_DECK")
        # Material 1: Standard LAW28 with rho_ref
        d.mat_law28(
            mat_id=1,
            title="Mat_Standard_LAW28",
            rho=3.5e-4,
            rho_ref=3.2e-4,
            e11=100.0,
            e22=120.0,
            e33=500.0,
            g12=40.0,
            g23=60.0,
            g31=50.0,
            fun_a1=11,
            fun_b1=22,
            fun_a2=33,
            gflag=1,
            fscale11=1.0,
            fscale22=1.1,
            fscale33=1.2,
            eps_max11=0.5,
            eps_max22=0.4,
            eps_max33=0.8,
            fun_a3=12,
            fun_b3=23,
            fun_a4=31,
            vflag=0,
            fscale12=0.9,
            fscale23=1.0,
            fscale31=1.05,
            eps_max12=0.3,
            eps_max23=0.35,
            eps_max31=0.45,
            law_name="LAW28",
        )
        # Material 2: Emitted via mat_honeycomb alias with law_name="HONEYCOMB"
        d.mat_honeycomb(
            mat_id=2,
            title="Mat_Honeycomb_Alias",
            rho=2.8e-4,
            e11=80.0,
            e22=95.0,
            e33=350.0,
            g12=25.0,
            g23=45.0,
            g31=35.0,
            fun_a1=101,
            fun_b1=102,
            fun_a2=103,
            gflag=0,
            fscale11=1.0,
            fscale22=1.0,
            fscale33=1.0,
            eps_max11=0.6,
            eps_max22=0.6,
            eps_max33=0.7,
            fun_a3=104,
            fun_b3=105,
            fun_a4=106,
            vflag=1,
            fscale12=1.0,
            fscale23=1.0,
            fscale31=1.0,
            eps_max12=0.5,
            eps_max23=0.5,
            eps_max31=0.5,
            law_name="HONEYCOMB",
        )

        deck_text = d.render()
        deck_file = tmp_path / "test_roundtrip.rad"
        deck_file.write_text(deck_text, encoding="utf-8")

        blocks = read_deck(str(deck_file))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0

        # Verify Mat 1
        assert 1 in model.mat_law28s
        m1 = model.mat_law28s[1]
        assert m1.rho0 == pytest.approx(3.5e-4)
        assert m1.ref_rho == pytest.approx(3.2e-4)
        assert m1.e11 == pytest.approx(100.0)
        assert m1.e22 == pytest.approx(120.0)
        assert m1.e33 == pytest.approx(500.0)
        assert m1.g12 == pytest.approx(40.0)
        assert m1.g23 == pytest.approx(60.0)
        assert m1.g31 == pytest.approx(50.0)
        assert m1.fun_a1 == 11
        assert m1.fun_b1 == 22
        assert m1.fun_a2 == 33
        assert m1.gflag == 1
        assert m1.fscale11 == pytest.approx(1.0)
        assert m1.fscale22 == pytest.approx(1.1)
        assert m1.fscale33 == pytest.approx(1.2)
        assert m1.epsr1 == pytest.approx(0.5)
        assert m1.epsr2 == pytest.approx(0.4)
        assert m1.epsr3 == pytest.approx(0.8)
        assert m1.fun_a3 == 12
        assert m1.fun_b3 == 23
        assert m1.fun_a4 == 31
        assert m1.vflag == 0
        assert m1.fscale12 == pytest.approx(0.9)
        assert m1.fscale23 == pytest.approx(1.0)
        assert m1.fscale13 == pytest.approx(1.05)
        assert m1.epsr4 == pytest.approx(0.3)
        assert m1.epsr5 == pytest.approx(0.35)
        assert m1.epsr6 == pytest.approx(0.45)

        # Verify Mat 2
        assert 2 in model.mat_law28s
        m2 = model.mat_law28s[2]
        assert m2.rho0 == pytest.approx(2.8e-4)
        assert m2.e11 == pytest.approx(80.0)
        assert m2.e22 == pytest.approx(95.0)
        assert m2.e33 == pytest.approx(350.0)
        assert m2.g12 == pytest.approx(25.0)
        assert m2.g23 == pytest.approx(45.0)
        assert m2.g31 == pytest.approx(35.0)
        assert m2.fun_a1 == 101
        assert m2.fun_b1 == 102
        assert m2.fun_a2 == 103
        assert m2.gflag == 0
        assert m2.fscale11 == pytest.approx(1.0)
        assert m2.fscale22 == pytest.approx(1.0)
        assert m2.fscale33 == pytest.approx(1.0)
        assert m2.epsr1 == pytest.approx(0.6)
        assert m2.epsr2 == pytest.approx(0.6)
        assert m2.epsr3 == pytest.approx(0.7)
        assert m2.fun_a3 == 104
        assert m2.fun_b3 == 105
        assert m2.fun_a4 == 106
        assert m2.vflag == 1
        assert m2.fscale12 == pytest.approx(1.0)
        assert m2.fscale23 == pytest.approx(1.0)
        assert m2.fscale13 == pytest.approx(1.0)
        assert m2.epsr4 == pytest.approx(0.5)
        assert m2.epsr5 == pytest.approx(0.5)
        assert m2.epsr6 == pytest.approx(0.5)

    def test_normalize_roundtrip(self, tmp_path: Path):
        """Test StarterDeck.starter_deck_from_lines with LAW28 and HONEYCOMB."""
        deck_text = """\
# RADIOSS STARTER
/BEGIN
NORM_TEST
                2022
/MAT/LAW28/1
Standard LAW28
            3.500E-4            3.200E-4
            100.0000            120.0000            500.0000
             40.0000             60.0000             50.0000
        11        22        33         1              1.0000              1.1000              1.2000
              0.5000              0.4000              0.8000
        12        23        31         0              0.9000              1.0000              1.0500
              0.3000              0.3500              0.4500
/MAT/HONEYCOMB/2
Honeycomb Material
            2.800E-4
             80.0000             95.0000            350.0000
             25.0000             45.0000             35.0000
       101       102       103         0              1.0000              1.0000              1.0000
              0.6000              0.6000              0.7000
       104       105       106         1              1.0000              1.0000              1.0000
              0.5000              0.5000              0.5000
/END
"""
        d = dw.starter_deck_from_lines(deck_text.splitlines(), "NORM_TEST")
        norm = d.render()

        assert "/MAT/LAW28/1" in norm
        assert "/MAT/HONEYCOMB/2" in norm

        cards1 = data_cards(block_lines(norm, "/MAT/LAW28/1"))
        assert len(cards1) == 8
        assert float(cards1[1][0:20]) == pytest.approx(3.5e-4)
        assert float(cards1[1][20:40]) == pytest.approx(3.2e-4)
        assert float(cards1[2][0:20]) == pytest.approx(100.0)

        cards2 = data_cards(block_lines(norm, "/MAT/HONEYCOMB/2"))
        assert len(cards2) == 8
        assert float(cards2[1][0:20]) == pytest.approx(2.8e-4)
        assert float(cards2[2][0:20]) == pytest.approx(80.0)


class TestStarterChecksAllowedLaws:
    """Verify checks.py element family compatibility rules for LAW28/HONEYCOMB (M538)."""

    def test_allowed_laws_contains_law28_and_honeycomb(self):
        for key in (28, "28", "LAW28", "HONEYCOMB"):
            assert key in _ALLOWED_LAWS["bricks"], f"{key} missing in bricks"
            assert key in _ALLOWED_LAWS["tetras"], f"{key} missing in tetras"

    def test_check_model_accepts_law28_on_solid(self):
        model = Model()
        model.add_nodes(np.arange(1, 9, dtype=np.int64), np.zeros((8, 3)))
        mat = Material(id=1, law=28, rho0=3.5e-4, title="Honeycomb", params={"E11": 100.0, "E22": 120.0, "E33": 500.0})
        model.materials[1] = mat

        class FakeGroup:
            def __init__(self):
                self.state = {"slices": [(slice(0, 1), mat, None)]}

        model._brick_groups = {"brics": FakeGroup()}
        model.element_groups = lambda: [("bricks", model._brick_groups["brics"])]

        log = MessageLog()
        check_model(model, log)
        assert len(log.errors) == 0, f"Unexpected errors: {log.errors}"


class TestLaw28EdgeCasesAndRobustness:
    """Comprehensive edge cases: missing cards, blank defaults, dual density, arbitrary titles/IDs."""

    def test_missing_optional_cards_fixed_format(self, tmp_path: Path):
        # Case A: 3 data cards only (Cards 1..3; 4..7 omitted)
        deck_3cards = """\
# RADIOSS STARTER
/BEGIN
TEST_3CARDS
                2022
/MAT/LAW28/301
3 Cards Only
            3.500E-4
            100.0000            120.0000            500.0000
             40.0000             60.0000             50.0000
/END
"""
        f3 = tmp_path / "law28_3cards.rad"
        f3.write_text(deck_3cards, encoding="utf-8")
        blocks = read_deck(str(f3))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0
        assert 301 in model.mat_law28s
        m = model.mat_law28s[301]
        assert m.rho0 == pytest.approx(3.5e-4)
        assert m.e11 == pytest.approx(100.0)
        assert m.fun_a1 == 0
        assert m.fscale11 == pytest.approx(1.0)
        assert m.fscale12 == pytest.approx(1.0)
        # Material params resolved via build_law28
        mat = model.materials[301]
        assert mat.law == 28
        assert mat.params["rho_ref"] == pytest.approx(3.5e-4)
        assert mat.params["fun_id11"] == 0
        assert mat.params["fun_id12"] == 0
        assert mat.params["fscale11"] == pytest.approx(1.0)
        assert mat.params["fscale12"] == pytest.approx(1.0)
        assert mat.params["eps_max11"] == pytest.approx(1.0e30)
        assert mat.params["eps_max12"] == pytest.approx(1.0e30)

        # Case B: 4 data cards only (Cards 1..4; 5..7 omitted)
        deck_4cards = """\
# RADIOSS STARTER
/BEGIN
TEST_4CARDS
                2022
/MAT/LAW28/302
4 Cards Only
            3.500E-4
            100.0000            120.0000            500.0000
             40.0000             60.0000             50.0000
        11        22        33         1              1.5000              1.6000              1.7000
/END
"""
        f4 = tmp_path / "law28_4cards.rad"
        f4.write_text(deck_4cards, encoding="utf-8")
        blocks4 = read_deck(str(f4))
        model4 = Model()
        log4 = MessageLog()
        parse_starter_deck(blocks4, model4, log4)
        assert len(log4.errors) == 0
        assert 302 in model4.mat_law28s
        mat4 = model4.materials[302]
        assert mat4.params["fun_id11"] == 11
        assert mat4.params["fun_id22"] == 22
        assert mat4.params["fun_id33"] == 33
        assert mat4.params["gflag"] == 1
        assert mat4.params["fscale11"] == pytest.approx(1.5)
        assert mat4.params["fscale22"] == pytest.approx(1.6)
        assert mat4.params["fscale33"] == pytest.approx(1.7)
        assert mat4.params["eps_max11"] == pytest.approx(1.0e30)
        assert mat4.params["fun_id12"] == 0
        assert mat4.params["fscale12"] == pytest.approx(1.0)
        assert mat4.params["eps_max12"] == pytest.approx(1.0e30)

        # Case C: 6 data cards (Cards 1..6; Card 7 omitted)
        deck_6cards = """\
# RADIOSS STARTER
/BEGIN
TEST_6CARDS
                2022
/MAT/LAW28/303
6 Cards Only
            3.500E-4
            100.0000            120.0000            500.0000
             40.0000             60.0000             50.0000
        11        22        33         1              1.0000              1.1000              1.2000
              0.5000              0.4000              0.8000
        12        23        31         0              0.9000              1.0000              1.0500
/END
"""
        f6 = tmp_path / "law28_6cards.rad"
        f6.write_text(deck_6cards, encoding="utf-8")
        blocks6 = read_deck(str(f6))
        model6 = Model()
        log6 = MessageLog()
        parse_starter_deck(blocks6, model6, log6)
        assert len(log6.errors) == 0
        assert 303 in model6.mat_law28s
        mat6 = model6.materials[303]
        assert mat6.params["eps_max11"] == pytest.approx(0.5)
        assert mat6.params["fun_id12"] == 12
        assert mat6.params["eps_max12"] == pytest.approx(1.0e30)

    def test_missing_optional_cards_free_format(self, tmp_path: Path):
        deck_free = """\
# RADIOSS STARTER
/MAT/HONEYCOMB/310
Free 3 Cards
3.2e-4
80.0 90.0 400.0
30.0 50.0 45.0
/MAT/HONEYCOMB/311
Free 4 Cards
3.2e-4
80.0 90.0 400.0
30.0 50.0 45.0
11 22 33 0 1.2 1.3 1.4
/END
"""
        f_free = tmp_path / "law28_free_truncated.rad"
        f_free.write_text(deck_free, encoding="utf-8")
        blocks = read_deck(str(f_free))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0
        assert 310 in model.materials
        assert 311 in model.materials
        m310 = model.materials[310]
        assert m310.params["fun_id11"] == 0
        assert m310.params["eps_max11"] == pytest.approx(1.0e30)
        m311 = model.materials[311]
        assert m311.params["fun_id11"] == 11
        assert m311.params["fscale11"] == pytest.approx(1.2)
        assert m311.params["eps_max11"] == pytest.approx(1.0e30)

    def test_blank_and_default_fields(self, tmp_path: Path):
        """When fields on Card 4 or Card 6 are blank/omitted, fscale defaults to 1.0."""
        deck_blank_fields = """\
# RADIOSS STARTER
/BEGIN
TEST_BLANKS
                2022
/MAT/LAW28/320
Blank Fields Test
            3.500E-4
            100.0000            120.0000            500.0000
             40.0000             60.0000             50.0000
        11        22        33         1
                                
        12        23        31         0
                                
/END
"""
        fb = tmp_path / "law28_blank_fields.rad"
        fb.write_text(deck_blank_fields, encoding="utf-8")
        blocks = read_deck(str(fb))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0
        assert 320 in model.mat_law28s
        mat = model.materials[320]
        assert mat.params["fscale11"] == pytest.approx(1.0)
        assert mat.params["fscale22"] == pytest.approx(1.0)
        assert mat.params["fscale33"] == pytest.approx(1.0)
        assert mat.params["fscale12"] == pytest.approx(1.0)
        assert mat.params["fscale23"] == pytest.approx(1.0)
        assert mat.params["fscale31"] == pytest.approx(1.0)
        assert mat.params["eps_max11"] == pytest.approx(1.0e30)
        assert mat.params["eps_max12"] == pytest.approx(1.0e30)

    def test_dual_density_variations(self, tmp_path: Path):
        deck = """\
# RADIOSS STARTER
/BEGIN
TEST_RHO
                2022
/MAT/LAW28/330
Single Density
            3.500E-4
            100.0000            120.0000            500.0000
             40.0000             60.0000             50.0000
/MAT/LAW28/331
Dual Density
            3.500E-4            2.900E-4
            100.0000            120.0000            500.0000
             40.0000             60.0000             50.0000
/END
"""
        frho = tmp_path / "law28_rho.rad"
        frho.write_text(deck, encoding="utf-8")
        blocks = read_deck(str(frho))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0

        # Single density: rho_ref defaults to rho0
        mat330 = model.materials[330]
        assert mat330.rho0 == pytest.approx(3.5e-4)
        assert mat330.params["rho0"] == pytest.approx(3.5e-4)
        assert mat330.params["rho_ref"] == pytest.approx(3.5e-4)

        # Dual density: rho_ref takes specified value
        mat331 = model.materials[331]
        assert mat331.rho0 == pytest.approx(3.5e-4)
        assert mat331.params["rho0"] == pytest.approx(3.5e-4)
        assert mat331.params["rho_ref"] == pytest.approx(2.9e-4)

    def test_arbitrary_ids_and_titles_roundtrip(self, tmp_path: Path):
        d = dw.StarterDeck("ARBITRARY_TEST")
        # 1. Very large integer ID
        d.mat_law28(mat_id=987654321, title="Large_ID_Mat", rho=1.0e-3, e11=100.0)
        # 2. Title with special characters
        d.mat_law28(mat_id=10, title="Honeycomb [Nomex] Type-42 (t=0.5mm) #1", rho=2.0e-3, e11=200.0)
        # 3. Numeric title in fixed format
        d.mat_law28(mat_id=20, title="999888", rho=3.0e-3, e11=300.0)
        # 4. Empty title
        d.mat_law28(mat_id=30, title="", rho=4.0e-3, e11=400.0)

        rendered = d.render()
        f_arb = tmp_path / "arbitrary.rad"
        f_arb.write_text(rendered, encoding="utf-8")

        blocks = read_deck(str(f_arb))
        model = Model()
        log = MessageLog()
        parse_starter_deck(blocks, model, log)
        assert len(log.errors) == 0

        assert 987654321 in model.mat_law28s
        assert model.mat_law28s[987654321].title == "Large_ID_Mat"
        assert 10 in model.mat_law28s
        assert model.mat_law28s[10].title == "Honeycomb [Nomex] Type-42 (t=0.5mm) #1"
        assert 20 in model.mat_law28s
        assert model.mat_law28s[20].title == "999888"
        assert 30 in model.mat_law28s
        assert model.mat_law28s[30].title == ""

        # Normalize roundtrip
        d_norm = dw.starter_deck_from_lines(rendered.splitlines(), "ARBITRARY_NORM")
        norm_rendered = d_norm.render()
        assert "/MAT/LAW28/987654321" in norm_rendered
        assert "/MAT/LAW28/10" in norm_rendered
        assert "/MAT/LAW28/20" in norm_rendered
        assert "/MAT/LAW28/30" in norm_rendered
