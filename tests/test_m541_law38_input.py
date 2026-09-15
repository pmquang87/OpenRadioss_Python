"""Tests for Milestone M541: /MAT/LAW38 and /MAT/VISC_TAB input layer.

Covers:
1. Card layout constants, aliases, and LAYOUTS dictionary registration.
2. Deck writer emitter StarterDeck.mat_law38 and alias mat_visc_tab.
3. Fixed-format 12-card width alignment, comments, and field formatting.
4. CFG catalogue and mat_reader synonym resolution.
5. Multi-curve table parsing (Fscale, Epsilon, Funct_Id_Load, Funct_Id_UnLoad).
6. Fixed-format and free-format deck parsing into Material and MatLaw38 entities.
7. Deck roundtrip parsing (StarterDeck -> parse_starter_deck).
8. Alias resolution (/MAT/VISC_TAB vs /MAT/LAW38 vs /VISC_TAB).
9. Starter validation checks: solid element acceptance, shell element rejection, parameter checks.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.mat_reader import catalogue
from pyradioss.input.cfg_catalogue import LAW_MAP, LAW_SYNONYMS, KEYWORD_NAME_MAP, SYNONYMS, canonical_law_name, law_number
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import Material, MatLaw38
from pyradioss.starter.checks import _ALLOWED_LAWS, check_mat_law38


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


# ============================================================================
# 1. Card Layouts & Aliases
# ============================================================================

class TestLaw38CardLayouts:
    """Card layout constants and LAYOUTS dictionary verification (M541)."""

    def test_constants_and_aliases(self):
        assert cl.MAT_LAW38_1 == (20, 20)
        assert cl.MAT_LAW38_2 == (20, 20, 20, 20, 10, 10)
        assert cl.MAT_LAW38_3 == (20, 20, 20, 10, 10, 20)
        assert cl.MAT_LAW38_4 == (10, 10, 20)
        assert cl.MAT_LAW38_5 == (20, 20, 20, 20)
        assert cl.MAT_LAW38_6 == (10, 10, 20, 20, 20, 20)
        assert cl.MAT_LAW38_7 == (10, 10, 20, 10)
        assert cl.MAT_LAW38_8 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW38_9 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW38_10 == (20, 20, 20, 20, 20)
        assert cl.MAT_LAW38_11 == (10, 10, 10, 10, 10)
        assert cl.MAT_LAW38_12 == (10, 10, 10, 10, 10)

        # Aliases
        for i in range(1, 13):
            law_val = getattr(cl, f"MAT_LAW38_{i}")
            assert getattr(cl, f"MAT_LAW38_CFG_{i}") == law_val
            assert getattr(cl, f"MAT_VISC_TAB_{i}") == law_val
            assert getattr(cl, f"MAT_VISC_TAB_CFG_{i}") == law_val

    def test_layouts_dictionary(self):
        for i in range(1, 13):
            for prefix in ("MAT_LAW38_", "MAT_LAW38_CFG_", "MAT_VISC_TAB_", "MAT_VISC_TAB_CFG_"):
                key = f"{prefix}{i}"
                assert key in cl.LAYOUTS, f"Missing key {key} in LAYOUTS"
                expected = getattr(cl, f"MAT_LAW38_{i}")
                assert list(cl.LAYOUTS[key]) == list(expected)


# ============================================================================
# 2. CFG Catalogue & Synonyms
# ============================================================================

class TestLaw38CfgCatalogue:
    """CFG catalogue, synonym mapping, and schema tests."""

    def test_synonyms_and_law_map(self):
        assert LAW_MAP["LAW38"] == 38
        assert LAW_MAP["VISC_TAB"] == 38
        assert KEYWORD_NAME_MAP["LAW38"] == 38
        assert KEYWORD_NAME_MAP["VISC_TAB"] == 38

        assert LAW_SYNONYMS["LAW38"] == "LAW38"
        assert LAW_SYNONYMS["VISC_TAB"] == "LAW38"
        assert SYNONYMS["VISC_TAB"] == "LAW38"

        assert law_number("LAW38") == 38
        assert law_number("VISC_TAB") == 38
        assert law_number("MAT_VISC_TAB") == 38
        assert law_number("38") == 38

        assert canonical_law_name("VISC_TAB") == "LAW38"
        assert canonical_law_name("LAW38") == "LAW38"
        assert canonical_law_name("38") == "LAW38"

    def test_catalogue_schema(self):
        cat = catalogue()
        schema_law = cat.schema("LAW38")
        schema_visc = cat.schema("VISC_TAB")
        assert schema_law is not None
        assert schema_visc is not None
        assert schema_law.law_number == 38
        assert "Fscale_i" in schema_law.attributes
        assert "Epsilon_i" in schema_law.attributes
        assert "Funct_Id_Load" in schema_law.attributes
        assert "Funct_Id_UnLoad" in schema_law.attributes


# ============================================================================
# 3. Deck Writer
# ============================================================================

class TestLaw38DeckWriter:
    """StarterDeck emitter tests for /MAT/LAW38 and /MAT/VISC_TAB."""

    def test_emit_mat_law38_all_cards(self):
        deck = StarterDeck("M541")
        deck.mat_law38(
            id=1,
            rho=1.2e-6,
            e=100.0,
            nu=0.35,
            nu_t=0.35,
            nu_c=0.38,
            rv=1.5,
            iflag=1,
            itotal=2,
            beta=0.05,
            h=0.8,
            damp1=0.4,
            gflag=1,
            vflag=1,
            theta=0.7,
            kair=1,
            np=201,
            pscale=1.2,
            p0=0.1013,
            pr=0.01,
            pmax=5.0,
            poros=0.85,
            ful=301,
            alpha_unload=0.9,
            eps_unload=0.02,
            a=1.1,
            b=0.95,
            nfunc=2,
            cutoff=10.0,
            iinsta=1,
            efinal=150.0,
            epsfinal=0.8,
            lamda=1.2,
            maxvisc=2.0,
            tol=0.05,
            fscale=[1.0, 1.2],
            epsilon=[0.01, 1.0],
            funct_id_load=[101, 102],
            funct_id_unload=[103, 104],
            title="FOAM_VISCOELASTIC",
            rhor=1.25e-6,
        )
        rendered = deck.render()
        lines = block_lines(rendered, "/MAT/LAW38/1")
        assert len(lines) >= 14
        assert lines[0] == "/MAT/LAW38/1"
        assert lines[1] == "FOAM_VISCOELASTIC"

        cards = data_cards(lines)
        # 12 data cards + title card = 13 cards total (cards[0] is title)
        assert len(cards) == 13

        # Card 1: rho, rhor (%20lg%20lg = 40 chars)
        assert len(cards[1]) == 40
        assert math.isclose(float(cards[1][:20]), 1.2e-6, rel_tol=1e-5)
        assert math.isclose(float(cards[1][20:40]), 1.25e-6, rel_tol=1e-5)

        # Card 2: e, nu_t, nu_c, rv, iflag, itotal (%20lg*4%10d%10d = 100 chars)
        assert len(cards[2]) == 100
        assert math.isclose(float(cards[2][:20]), 100.0, rel_tol=1e-5)
        assert math.isclose(float(cards[2][20:40]), 0.35, rel_tol=1e-5)
        assert math.isclose(float(cards[2][40:60]), 0.38, rel_tol=1e-5)
        assert math.isclose(float(cards[2][60:80]), 1.5, rel_tol=1e-5)
        assert int(cards[2][80:90]) == 1
        assert int(cards[2][90:100]) == 2

        # Card 3: beta, h, damp1, gflag, vflag, theta (%20lg*3%10d%10d%20lg = 100 chars)
        assert len(cards[3]) == 100
        assert math.isclose(float(cards[3][:20]), 0.05, rel_tol=1e-5)
        assert math.isclose(float(cards[3][20:40]), 0.8, rel_tol=1e-5)
        assert math.isclose(float(cards[3][40:60]), 0.4, rel_tol=1e-5)
        assert int(cards[3][60:70]) == 1
        assert int(cards[3][70:80]) == 1
        assert math.isclose(float(cards[3][80:100]), 0.7, rel_tol=1e-5)

        # Card 4: kair, np, pscale (%10d%10d%20lg = 40 chars)
        assert len(cards[4]) == 40
        assert int(cards[4][:10]) == 1
        assert int(cards[4][10:20]) == 201
        assert math.isclose(float(cards[4][20:40]), 1.2, rel_tol=1e-5)

        # Card 5: p0, pr, pmax, poros (%20lg*4 = 80 chars)
        assert len(cards[5]) == 80
        assert math.isclose(float(cards[5][:20]), 0.1013, rel_tol=1e-5)
        assert math.isclose(float(cards[5][20:40]), 0.01, rel_tol=1e-5)
        assert math.isclose(float(cards[5][40:60]), 5.0, rel_tol=1e-5)
        assert math.isclose(float(cards[5][60:80]), 0.85, rel_tol=1e-5)

        # Card 6: ful, blank, alpha_unload, eps_unload, a, b (%10d 10x %20lg*4 = 100 chars)
        assert len(cards[6]) == 100
        assert int(cards[6][:10]) == 301
        assert cards[6][10:20].strip() == ""
        assert math.isclose(float(cards[6][20:40]), 0.9, rel_tol=1e-5)
        assert math.isclose(float(cards[6][40:60]), 0.02, rel_tol=1e-5)
        assert math.isclose(float(cards[6][60:80]), 1.1, rel_tol=1e-5)
        assert math.isclose(float(cards[6][80:100]), 0.95, rel_tol=1e-5)

        # Card 7: m_func, blank, cutoff, iinsta (%10d 10x %20lg%10d = 50 chars)
        assert len(cards[7]) == 50
        assert int(cards[7][:10]) == 2
        assert cards[7][10:20].strip() == ""
        assert math.isclose(float(cards[7][20:40]), 10.0, rel_tol=1e-5)
        assert int(cards[7][40:50]) == 1

        # Card 8: efinal, epsfinal, lamda, maxvisc, tol (%20lg*5 = 100 chars)
        assert len(cards[8]) == 100
        assert math.isclose(float(cards[8][:20]), 150.0, rel_tol=1e-5)
        assert math.isclose(float(cards[8][20:40]), 0.8, rel_tol=1e-5)
        assert math.isclose(float(cards[8][40:60]), 1.2, rel_tol=1e-5)
        assert math.isclose(float(cards[8][60:80]), 2.0, rel_tol=1e-5)
        assert math.isclose(float(cards[8][80:100]), 0.05, rel_tol=1e-5)

        # Card 9: Fscale (2 entries, 40 chars)
        assert len(cards[9]) == 40
        assert math.isclose(float(cards[9][:20]), 1.0, rel_tol=1e-5)
        assert math.isclose(float(cards[9][20:40]), 1.2, rel_tol=1e-5)

        # Card 10: Epsilon (2 entries, 40 chars)
        assert len(cards[10]) == 40
        assert math.isclose(float(cards[10][:20]), 0.01, rel_tol=1e-5)
        assert math.isclose(float(cards[10][20:40]), 1.0, rel_tol=1e-5)

        # Card 11: Funct_Id_Load (2 entries, 20 chars)
        assert len(cards[11]) == 20
        assert int(cards[11][:10]) == 101
        assert int(cards[11][10:20]) == 102

        # Card 12: Funct_Id_UnLoad (2 entries, 20 chars)
        assert len(cards[12]) == 20
        assert int(cards[12][:10]) == 103
        assert int(cards[12][10:20]) == 104

    def test_emit_mat_visc_tab_alias(self):
        deck = StarterDeck("M541")
        deck.mat_visc_tab(
            id=2,
            rho=1.5e-6,
            e=50.0,
            nu=0.4,
            title="VISC_TAB_TEST",
        )
        rendered = deck.render()
        lines = block_lines(rendered, "/MAT/VISC_TAB/2")
        assert len(lines) >= 14
        assert lines[0] == "/MAT/VISC_TAB/2"
        assert lines[1] == "VISC_TAB_TEST"


# ============================================================================
# 4. Fixed-Format Parsing
# ============================================================================

class TestLaw38FixedFormatParsing:
    """Fixed-format 12-card deck parsing into Material and MatLaw38 entities."""

    def test_parse_fixed_12_card_deck(self, tmp_path: Path):
        deck_text = """#RADIOSS STARTER
/BEGIN
TEST_RUN
      2022         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/MAT/LAW38/10
POLYMER_FOAM_TEST
#        Init. dens.          Ref. dens.
             1.2e-06            1.25e-06
#                  E                nu_t                nu_c                  Rv     Iflag     Itota
               120.0                0.32                0.36                 1.8         1         2
#               Beta                   H                 R_D       K_R       K_D     Instant-mod-upd
                0.08                 0.9                0.45         1         1                0.75
#     Kair        Np              Pscale
         1       201                 1.5
#                 P0                  Rp                Pmax                 Phi
              0.1013                0.02                 6.0                0.80
#      ful                  alpha_unload        Eps_._unload                   a                   b
       301                           0.8                0.03                 1.2                 0.9
#   m_func                        CUToff    Iinsta
         3                          12.5         1
#            E-final          Epsi-final              Lambda                VISC                 Tol
               180.0                 0.9                 1.3                 2.5                0.04
# Scale factors
                 1.0                 1.1                 1.2
# Strain rates
               0.001                0.01                 0.1
# Loading functions
       101       102       103
# Unloading functions
       201       202       203
/END
"""
        deck_file = tmp_path / "test_fixed_0000.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        model = parse_starter_deck(str(deck_file))

        assert 10 in model.materials
        assert 10 in model.mat_law38s

        mat = model.materials[10]
        assert mat.law == 38
        assert math.isclose(mat.rho0, 1.2e-6, rel_tol=1e-5)
        assert math.isclose(mat.E, 120.0, rel_tol=1e-5)
        assert math.isclose(mat.nu, 0.36, rel_tol=1e-5)  # max(nu_t, nu_c)

        p = mat.params
        assert math.isclose(p["rho"], 1.2e-6, rel_tol=1e-5)
        assert math.isclose(p["rhor"], 1.25e-6, rel_tol=1e-5)
        assert math.isclose(p["e"], 120.0, rel_tol=1e-5)
        assert math.isclose(p["nu_t"], 0.32, rel_tol=1e-5)
        assert math.isclose(p["nu_c"], 0.36, rel_tol=1e-5)
        assert math.isclose(p["rv"], 1.8, rel_tol=1e-5)
        assert p["iflag"] == 1
        assert p["itotal"] == 2
        assert math.isclose(p["beta"], 0.08, rel_tol=1e-5)
        assert math.isclose(p["h"], 0.9, rel_tol=1e-5)
        assert math.isclose(p["r_d"], 0.45, rel_tol=1e-5)
        assert p["k_r"] == 1
        assert p["k_d"] == 1
        assert math.isclose(p["instant_mod_upd"], 0.75, rel_tol=1e-5)
        assert p["kair"] == 1
        assert p["np"] == 201
        assert math.isclose(p["pscale"], 1.5, rel_tol=1e-5)
        assert math.isclose(p["p0"], 0.1013, rel_tol=1e-5)
        assert math.isclose(p["rp"], 0.02, rel_tol=1e-5)
        assert math.isclose(p["pmax"], 6.0, rel_tol=1e-5)
        assert math.isclose(p["phi"], 0.80, rel_tol=1e-5)
        assert p["ful"] == 301
        assert math.isclose(p["alpha_unload"], 0.8, rel_tol=1e-5)
        assert math.isclose(p["eps_unload"], 0.03, rel_tol=1e-5)
        assert math.isclose(p["a"], 1.2, rel_tol=1e-5)
        assert math.isclose(p["b"], 0.9, rel_tol=1e-5)
        assert p["m_func"] == 3
        assert math.isclose(p["cutoff"], 12.5, rel_tol=1e-5)
        assert p["iinsta"] == 1
        assert math.isclose(p["e_final"], 180.0, rel_tol=1e-5)
        assert math.isclose(p["epsi_final"], 0.9, rel_tol=1e-5)
        assert math.isclose(p["lamb"], 1.3, rel_tol=1e-5)
        assert math.isclose(p["visc"], 2.5, rel_tol=1e-5)
        assert math.isclose(p["tol"], 0.04, rel_tol=1e-5)

        # Multi-curve table verification
        assert len(p["fscale_i"]) == 3
        assert [pytest.approx(x, rel=1e-5) for x in p["fscale_i"]] == [1.0, 1.1, 1.2]
        assert [pytest.approx(x, rel=1e-5) for x in p["epsilon_i"]] == [0.001, 0.01, 0.1]
        assert p["funct_id_load"] == [101, 102, 103]
        assert p["funct_id_unload"] == [201, 202, 203]

        # MatLaw38 dataclass verification
        m38 = model.mat_law38s[10]
        assert isinstance(m38, MatLaw38)
        assert m38.fscale == p["fscale_i"]
        assert m38.epsilon == p["epsilon_i"]
        assert m38.funct_id_load == [101, 102, 103]
        assert m38.funct_id_unload == [201, 202, 203]


# ============================================================================
# 5. Free-Format Parsing
# ============================================================================

class TestLaw38FreeFormatParsing:
    """Free-format deck parsing tests."""

    def test_parse_free_format_deck(self, tmp_path: Path):
        deck_text = """/MAT/LAW38/20
FREE_FORMAT_TEST
1.0e-06 1.0e-06
80.0 0.3 0.35 1.0 0 1
0.05 1.0 0.5 0 0 0.67
0 0 1.0
0.1 0.0 10.0 0.7
10 0.95 0.01 1.0 1.0
2 5.0 0
100.0 1.0 1.0 0.0 1.0
1.0 1.5
0.01 0.1
501 502
601 602
/END
"""
        deck_file = tmp_path / "test_free.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        model = parse_starter_deck(str(deck_file))

        assert 20 in model.materials
        mat = model.materials[20]
        assert mat.law == 38
        assert math.isclose(mat.rho0, 1.0e-6, rel_tol=1e-5)
        assert math.isclose(mat.E, 80.0, rel_tol=1e-5)
        p = mat.params
        assert math.isclose(p["nu_t"], 0.3, rel_tol=1e-5)
        assert math.isclose(p["nu_c"], 0.35, rel_tol=1e-5)
        assert p["ful"] == 10
        assert math.isclose(p["alpha_unload"], 0.95, rel_tol=1e-5)
        assert math.isclose(p["eps_unload"], 0.01, rel_tol=1e-5)
        assert p["m_func"] == 2
        assert math.isclose(p["cutoff"], 5.0, rel_tol=1e-5)
        assert p["fscale_i"] == [1.0, 1.5]
        assert p["epsilon_i"] == [0.01, 0.1]
        assert p["funct_id_load"] == [501, 502]
        assert p["funct_id_unload"] == [601, 602]


# ============================================================================
# 6. Roundtrip Parsing
# ============================================================================

class TestLaw38Roundtrip:
    """Roundtrip test: StarterDeck writer -> parse_starter_deck."""

    def test_roundtrip_law38(self, tmp_path: Path):
        deck = StarterDeck("ROUNDTRIP")
        deck.mat_law38(
            id=5,
            rho=2.5e-6,
            e=250.0,
            nu=0.28,
            nu_t=0.28,
            nu_c=0.33,
            rv=2.0,
            iflag=0,
            itotal=1,
            beta=0.1,
            h=0.85,
            damp1=0.6,
            gflag=1,
            vflag=2,
            theta=0.65,
            kair=1,
            np=401,
            pscale=2.0,
            p0=0.1013,
            pr=0.05,
            pmax=8.0,
            poros=0.75,
            ful=501,
            alpha_unload=0.85,
            eps_unload=0.04,
            a=1.05,
            b=0.98,
            nfunc=4,
            cutoff=15.0,
            iinsta=1,
            efinal=300.0,
            epsfinal=0.7,
            lamda=1.1,
            maxvisc=3.5,
            tol=0.02,
            fscale=[1.0, 1.1, 1.2, 1.3],
            epsilon=[0.001, 0.01, 0.1, 1.0],
            funct_id_load=[11, 12, 13, 14],
            funct_id_unload=[21, 22, 23, 24],
            title="ROUNDTRIP_FOAM",
            rhor=2.55e-6,
        )
        p = tmp_path / "roundtrip_0000.rad"
        deck.write(str(p))
        model = parse_starter_deck(str(p))

        assert 5 in model.materials
        assert 5 in model.mat_law38s

        m = model.materials[5]
        assert m.law == 38
        assert math.isclose(m.rho0, 2.5e-6, rel_tol=1e-5)
        assert math.isclose(m.E, 250.0, rel_tol=1e-5)
        assert math.isclose(m.nu, 0.33, rel_tol=1e-5)

        p_dict = m.params
        assert math.isclose(p_dict["rhor"], 2.55e-6, rel_tol=1e-5)
        assert math.isclose(p_dict["nu_t"], 0.28, rel_tol=1e-5)
        assert math.isclose(p_dict["nu_c"], 0.33, rel_tol=1e-5)
        assert math.isclose(p_dict["beta"], 0.1, rel_tol=1e-5)
        assert math.isclose(p_dict["h"], 0.85, rel_tol=1e-5)
        assert math.isclose(p_dict["r_d"], 0.6, rel_tol=1e-5)
        assert p_dict["k_r"] == 1
        assert p_dict["k_d"] == 2
        assert math.isclose(p_dict["theta"], 0.65, rel_tol=1e-5)
        assert p_dict["np"] == 401
        assert math.isclose(p_dict["pscale"], 2.0, rel_tol=1e-5)
        assert math.isclose(p_dict["pmax"], 8.0, rel_tol=1e-5)
        assert math.isclose(p_dict["poros"], 0.75, rel_tol=1e-5)
        assert p_dict["ful"] == 501
        assert math.isclose(p_dict["alpha_unload"], 0.85, rel_tol=1e-5)
        assert math.isclose(p_dict["eps_unload"], 0.04, rel_tol=1e-5)
        assert math.isclose(p_dict["a"], 1.05, rel_tol=1e-5)
        assert math.isclose(p_dict["b"], 0.98, rel_tol=1e-5)
        assert p_dict["m_func"] == 4
        assert math.isclose(p_dict["cutoff"], 15.0, rel_tol=1e-5)
        assert math.isclose(p_dict["efinal"], 300.0, rel_tol=1e-5)
        assert math.isclose(p_dict["visc"], 3.5, rel_tol=1e-5)

        assert len(p_dict["fscale_i"]) == 4
        assert [pytest.approx(x, rel=1e-5) for x in p_dict["fscale_i"]] == [1.0, 1.1, 1.2, 1.3]
        assert [pytest.approx(x, rel=1e-5) for x in p_dict["epsilon_i"]] == [0.001, 0.01, 0.1, 1.0]
        assert p_dict["funct_id_load"] == [11, 12, 13, 14]
        assert p_dict["funct_id_unload"] == [21, 22, 23, 24]


# ============================================================================
# 7. Alias Resolution
# ============================================================================

class TestLaw38Aliases:
    """Header keyword alias resolution (/MAT/VISC_TAB vs /MAT/LAW38 vs /VISC_TAB)."""

    def test_alias_mat_visc_tab(self, tmp_path: Path):
        deck_text = """#RADIOSS STARTER
/BEGIN
ALIAS_RUN
      2022         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/MAT/VISC_TAB/101
ALIAS_VISC_TAB
#        Init. dens.
             1.0e-06
#                  E                nu_t                nu_c                  Rv     Iflag     Itota
                50.0                 0.3                 0.3                 0.0         0         0
#               Beta                   H                 R_D       K_R       K_D     Instant-mod-upd
                 0.0                 1.0                 0.5         0         0                0.67
#     Kair        Np              Pscale
         0         0                 1.0
#                 P0                  Rp                Pmax                 Phi
                 0.0                 0.0                 0.0                 0.0
#      ful                  alpha_unload        Eps_._unload                   a                   b
         0                           1.0                 0.0                 1.0                 1.0
#   m_func                        CUToff    Iinsta
         1                           0.0         0
#            E-final          Epsi-final              Lambda                VISC                 Tol
                50.0                 1.0                 1.0                 0.0                 1.0
# Scale factors
                 1.0
# Strain rates
                 0.0
# Loading functions
                 100
# Unloading functions
                 100
/END
"""
        deck_file = tmp_path / "test_alias1_0000.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        model = parse_starter_deck(str(deck_file))
        assert 101 in model.materials
        assert 101 in model.mat_law38s
        mat = model.materials[101]
        assert mat.law == 38
        assert mat.title == "ALIAS_VISC_TAB"
        assert math.isclose(mat.E, 50.0, rel_tol=1e-5)

    def test_alias_standalone_visc_tab(self, tmp_path: Path):
        deck_text = """#RADIOSS STARTER
/BEGIN
ALIAS_RUN2
      2022         0
                  Mg                  mm                   s
                  Mg                  mm                   s
/VISC_TAB/102
STANDALONE_VISC_TAB
#        Init. dens.
             1.0e-06
#                  E                nu_t                nu_c                  Rv     Iflag     Itota
                60.0                 0.3                 0.3                 0.0         0         0
#               Beta                   H                 R_D       K_R       K_D     Instant-mod-upd
                 0.0                 1.0                 0.5         0         0                0.67
#     Kair        Np              Pscale
         0         0                 1.0
#                 P0                  Rp                Pmax                 Phi
                 0.0                 0.0                 0.0                 0.0
#      ful                  alpha_unload        Eps_._unload                   a                   b
         0                           1.0                 0.0                 1.0                 1.0
#   m_func                        CUToff    Iinsta
         1                           0.0         0
#            E-final          Epsi-final              Lambda                VISC                 Tol
                60.0                 1.0                 1.0                 0.0                 1.0
# Scale factors
                 1.0
# Strain rates
                 0.0
# Loading functions
                 100
# Unloading functions
                 100
/END
"""
        deck_file = tmp_path / "test_alias2_0000.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        model = parse_starter_deck(str(deck_file))
        assert 102 in model.materials
        assert model.materials[102].law == 38


# ============================================================================
# 8. Starter Validation Checks
# ============================================================================

class TestLaw38ValidationChecks:
    """Starter checks integration for LAW38 / VISC_TAB."""

    def test_allowed_laws_includes_law38(self):
        for group in ("bricks", "tetras", "penta6", "pyra5"):
            assert 38 in _ALLOWED_LAWS[group]
            assert "LAW38" in _ALLOWED_LAWS[group]
            assert "VISC_TAB" in _ALLOWED_LAWS[group]

        # Shells and beams reject LAW38
        assert 38 not in _ALLOWED_LAWS["shells"]
        assert 38 not in _ALLOWED_LAWS["beams"]

    def test_check_mat_law38_valid(self):
        log = MessageLog()
        mat = Material(
            id=1, law=38, rho0=1.2e-6,
            params={"rho": 1.2e-6, "e": 100.0, "nu_t": 0.3, "nu_c": 0.35, "nfunc": 1}
        )
        check_mat_law38(mat, log)
        assert len(log.errors) == 0

    def test_check_mat_law38_invalid_density(self):
        log = MessageLog()
        mat = Material(
            id=1, law=38, rho0=0.0,
            params={"rho": 0.0, "e": 100.0, "nu_t": 0.3, "nu_c": 0.35, "nfunc": 1}
        )
        check_mat_law38(mat, log)
        assert len(log.errors) > 0
        assert any("initial density RHO must be > 0" in str(e) for e in log.errors)

    def test_check_mat_law38_invalid_modulus(self):
        log = MessageLog()
        mat = Material(
            id=2, law=38, rho0=1.2e-6,
            params={"rho": 1.2e-6, "e": 0.0, "nu_t": 0.3, "nu_c": 0.35, "nfunc": 1}
        )
        check_mat_law38(mat, log)
        assert len(log.errors) > 0
        assert any("Young's modulus E must be > 0" in str(e) for e in log.errors)

    def test_check_mat_law38_invalid_poisson(self):
        log = MessageLog()
        mat = Material(
            id=3, law=38, rho0=1.2e-6,
            params={"rho": 1.2e-6, "e": 100.0, "nu_t": 0.55, "nu_c": 0.35, "nfunc": 1}
        )
        check_mat_law38(mat, log)
        assert len(log.errors) > 0
        assert any("tensile Poisson's ratio nu_t must be in [0, 0.5)" in str(e) for e in log.errors)

    def test_check_mat_law38_invalid_nfunc(self):
        log = MessageLog()
        mat = Material(
            id=4, law=38, rho0=1.2e-6,
            params={"rho": 1.2e-6, "e": 100.0, "nu_t": 0.3, "nu_c": 0.35, "nfunc": 6}
        )
        check_mat_law38(mat, log)
        assert len(log.errors) > 0
        assert any("number of functions nfunc must be between 1 and 5" in str(e) for e in log.errors)
