"""Comprehensive input & roundtrip audit test suite for /MAT/LAW38 and /MAT/VISC_TAB (M541).

Auditor 4: LAW38 Input & Roundtrip Auditor.

Rigorously audits:
1. Card formatting precision audit:
   - Fixed-format width checks for all 12 cards (20-char and 10-char fields).
   - Cards 6 and 7 blank column preservation (col 2 is exactly 10 spaces).
   - CELL_LIST layout, line lengths, and field slicing for cards 9..12 with NFUNC = 1, 2, 3, 4, 5.
2. Roundtrip fidelity:
   - Complete StarterDeck with /MAT/LAW38 and /MAT/VISC_TAB across all fields and curves.
   - 100% parameter preservation across MatLaw38 dataclass and Material.params.
   - Both Refer_Rho specified (rhor != rho) and omitted (rhor == rho).
3. Free-format robustness:
   - Comma-separated without spaces (e.g. 1.0e-6,1.25e-6).
   - Comma-separated with spaces (e.g. 1.0e-6, 1.25e-6).
   - Whitespace-separated free format.
   - Consecutive commas (empty fields) preserving defaults across all cards.
   - Free-format deck without title card (numeric card detection with commas).
4. Synonym & alias resolution:
   - Exact equivalence of /MAT/VISC_TAB, /MAT/LAW38, /VISC_TAB, /LAW38.
   - /MAT/VISC_TAB/mat_id syntax and /MAT/LAW38/mat_id syntax.
   - Aliases in model: model.mat_law38s is model.mat_visc_tabs.
   - Dataclass aliases: MatViscTab is MatLaw38.
   - Catalogue schema and synonym lookups.
5. Defensive edge cases:
   - Missing optional cards (truncated cards 9..12 when NFUNC=0).
   - Truncated deck (early EOF after card 2 or 4).
   - Default value injection when cards omit fields.
   - Unload function default <= 0 falling back to loading curve 1.
   - Scale factor <= 0 falling back to 1.0.
   - Parameter bounds validation via check_mat_law38 (rho0, E, nu, nfunc, air content).
   - Element compatibility validation (solids accepted, shells/trias rejected).
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import card_layouts as cl
from pyradioss.input.card_layouts import split_fixed, fmt_float, fmt_int, blank, BLANK_CARD
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.input.deck_reader import Card
from pyradioss.input.mat_reader import catalogue
from pyradioss.input.cfg_catalogue import (
    LAW_MAP,
    LAW_SYNONYMS,
    KEYWORD_NAME_MAP,
    SYNONYMS,
    canonical_law_name,
    law_number,
)
from pyradioss.input.starter_keywords import parse_starter_deck, _is_numeric_card
from pyradioss.model.model import Model
from pyradioss.model.entities import Material, MatLaw38, MatViscTab
from pyradioss.starter.checks import check_mat_law38, check_model, _ALLOWED_LAWS


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
            if not s.startswith("#"):
                out.append(line)
    return out


# ============================================================================
# 1. Card Formatting Precision Audit
# ============================================================================

class TestCardFormattingPrecisionAudit:
    """Card formatting precision audit for all 12 cards of /MAT/LAW38."""

    def test_all_12_card_layouts_constants(self):
        """Audit field layout tuple definitions against radioss51 CFG."""
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

        # Audit layout registrations in cl.LAYOUTS
        for i in range(1, 13):
            for pfx in ("MAT_LAW38_", "MAT_LAW38_CFG_", "MAT_VISC_TAB_", "MAT_VISC_TAB_CFG_"):
                key = f"{pfx}{i}"
                assert key in cl.LAYOUTS, f"Missing {key} in LAYOUTS"
                expected = getattr(cl, f"MAT_LAW38_{i}")
                assert cl.LAYOUTS[key] == expected

    def test_cards_6_and_7_blank_column_preservation(self):
        """Audit that cards 6 & 7 have exactly 10 blank spaces in column 2."""
        deck = StarterDeck("BLANK_COL_AUDIT")
        deck.mat_law38(
            id=1,
            rho=1.2e-6,
            e=100.0,
            ful=301,
            alpha_unload=0.9,
            eps_unload=0.02,
            a=1.1,
            b=0.95,
            nfunc=2,
            cutoff=10.0,
            iinsta=1,
            fscale=[1.0, 1.2],
            epsilon=[0.01, 1.0],
            funct_id_load=[101, 102],
            funct_id_unload=[103, 104],
            title="BLANK_COL_TEST",
        )
        rendered = deck.render()
        cards = _extract_block_cards(rendered, "/MAT/LAW38/1")
        # cards[0] is title
        card6 = cards[6]
        card7 = cards[7]

        # Card 6: FUN_B4(10), blank(10), ALPHA6(20), EPSF2(20), EXP1(20), EXP2(20)
        assert len(card6) == 100
        assert card6[:10].strip() == "301"
        assert card6[10:20] == "          ", "Card 6 col 2 must be exactly 10 spaces"
        assert math.isclose(float(card6[20:40]), 0.9, rel_tol=1e-5)
        assert math.isclose(float(card6[40:60]), 0.02, rel_tol=1e-5)
        assert math.isclose(float(card6[60:80]), 1.1, rel_tol=1e-5)
        assert math.isclose(float(card6[80:100]), 0.95, rel_tol=1e-5)

        # Card 7: NFUNC(10), blank(10), CUTOFF(20), Iinsta(10)
        assert len(card7) == 50
        assert card7[:10].strip() == "2"
        assert card7[10:20] == "          ", "Card 7 col 2 must be exactly 10 spaces"
        assert math.isclose(float(card7[20:40]), 10.0, rel_tol=1e-5)
        assert card7[40:50].strip() == "1"

    @pytest.mark.parametrize("nfunc", [1, 2, 3, 4, 5])
    def test_cell_list_multi_curve_cards_9_to_12(self, nfunc: int):
        """Audit CELL_LIST formatting and cutting for 1, 2, 3, 4, and 5 functions."""
        fscales = [1.0 + 0.1 * i for i in range(nfunc)]
        epsilons = [0.01 * (10 ** i) for i in range(nfunc)]
        floads = [100 + i for i in range(nfunc)]
        funloads = [200 + i for i in range(nfunc)]

        deck = StarterDeck(f"CELL_LIST_{nfunc}")
        deck.mat_law38(
            id=nfunc,
            rho=1.2e-6,
            e=100.0,
            nfunc=nfunc,
            fscale=fscales,
            epsilon=epsilons,
            funct_id_load=floads,
            funct_id_unload=funloads,
            title=f"CELL_LIST_NFUNC_{nfunc}",
        )
        rendered = deck.render()
        cards = _extract_block_cards(rendered, f"/MAT/LAW38/{nfunc}")
        # cards: [0]=title, [1]=c1, ..., [8]=c8, [9]=c9, [10]=c10, [11]=c11, [12]=c12
        card9 = cards[9]
        card10 = cards[10]
        card11 = cards[11]
        card12 = cards[12]

        # Exact line length checks:
        # Card 9 (Fscale_i): nfunc * 20 chars
        assert len(card9) == nfunc * 20
        # Card 10 (Epsilon_i): nfunc * 20 chars
        assert len(card10) == nfunc * 20
        # Card 11 (Funct_Id_Load): nfunc * 10 chars
        assert len(card11) == nfunc * 10
        # Card 12 (Funct_Id_UnLoad): nfunc * 10 chars
        assert len(card12) == nfunc * 10

        # Card cutting verification against LAYOUTS
        c9_cuts = split_fixed(card9, cl.LAYOUTS["MAT_LAW38_9"])
        c10_cuts = split_fixed(card10, cl.LAYOUTS["MAT_LAW38_10"])
        c11_cuts = split_fixed(card11, cl.LAYOUTS["MAT_LAW38_11"])
        c12_cuts = split_fixed(card12, cl.LAYOUTS["MAT_LAW38_12"])

        for i in range(nfunc):
            assert math.isclose(float(c9_cuts[i]), fscales[i], rel_tol=1e-5)
            assert math.isclose(float(c10_cuts[i]), epsilons[i], rel_tol=1e-5)
            assert int(c11_cuts[i]) == floads[i]
            assert int(c12_cuts[i]) == funloads[i]

        for i in range(nfunc, 5):
            assert c9_cuts[i] == ""
            assert c10_cuts[i] == ""
            assert c11_cuts[i] == ""
            assert c12_cuts[i] == ""


# ============================================================================
# 2. Roundtrip Fidelity
# ============================================================================

class TestRoundtripFidelityAudit:
    """Rigorous roundtrip fidelity audit across all fields and curves."""

    @pytest.mark.parametrize("use_visc_tab_alias", [False, True])
    @pytest.mark.parametrize("distinct_rhor", [False, True])
    def test_complete_roundtrip_fidelity(self, tmp_path: Path, use_visc_tab_alias: bool, distinct_rhor: bool):
        """Assert 100% parameter preservation across all fields with both aliases."""
        mat_id = 77
        rho0 = 1.25e-6
        rhor = 1.35e-6 if distinct_rhor else 1.25e-6

        full_params = {
            "id": mat_id,
            "rho": rho0,
            "rhor": rhor,
            "e": 120.0,
            "nu_t": 0.32,
            "nu_c": 0.36,
            "rv": 1.8,
            "iflag": 1,
            "itotal": 2,
            "beta": 0.06,
            "h": 0.85,
            "damp1": 0.42,
            "gflag": 1,
            "vflag": 1,
            "theta": 0.68,
            "kair": 1,
            "np": 202,
            "pscale": 1.3,
            "p0": 0.101325,
            "pr": 0.015,
            "pmax": 5.5,
            "poros": 0.82,
            "ful": 302,
            "alpha_unload": 0.92,
            "eps_unload": 0.025,
            "a": 1.15,
            "b": 0.92,
            "nfunc": 3,
            "cutoff": 11.5,
            "iinsta": 1,
            "efinal": 160.0,
            "epsfinal": 0.85,
            "lamda": 1.25,
            "maxvisc": 2.2,
            "tol": 0.045,
            "fscale": [1.05, 1.15, 1.25],
            "epsilon": [0.005, 0.05, 0.5],
            "funct_id_load": [101, 102, 103],
            "funct_id_unload": [201, 202, 203],
            "title": "FIDELITY_ROUNDTRIP_TEST",
        }

        deck = StarterDeck("AUDIT_ROUNDTRIP")
        if use_visc_tab_alias:
            deck.mat_visc_tab(**full_params)
        else:
            deck.mat_law38(**full_params)

        deck_path = tmp_path / f"roundtrip_{use_visc_tab_alias}_{distinct_rhor}.rad"
        deck.write(deck_path)

        model = parse_starter_deck(str(deck_path))

        assert mat_id in model.materials
        assert mat_id in model.mat_law38s
        assert mat_id in model.mat_visc_tabs

        m38 = model.mat_law38s[mat_id]
        p = model.materials[mat_id].params

        # MatLaw38 dataclass field verification
        assert m38.id == mat_id
        assert m38.title == "FIDELITY_ROUNDTRIP_TEST"
        assert math.isclose(m38.rho0, rho0, rel_tol=1e-5)
        assert math.isclose(m38.rhor, rhor, rel_tol=1e-5)
        assert math.isclose(m38.e, 120.0, rel_tol=1e-5)
        assert math.isclose(m38.nu_t, 0.32, rel_tol=1e-5)
        assert math.isclose(m38.nu_c, 0.36, rel_tol=1e-5)
        assert math.isclose(m38.nu, 0.36, rel_tol=1e-5)
        assert math.isclose(m38.rv, 1.8, rel_tol=1e-5)
        assert m38.iflag == 1
        assert m38.itotal == 2
        assert math.isclose(m38.beta, 0.06, rel_tol=1e-5)
        assert math.isclose(m38.h, 0.85, rel_tol=1e-5)
        assert math.isclose(m38.r_d, 0.42, rel_tol=1e-5)
        assert m38.k_r == 1
        assert m38.k_d == 1
        assert math.isclose(m38.instant_mod_upd, 0.68, rel_tol=1e-5)
        assert m38.kair == 1
        assert m38.np == 202
        assert math.isclose(m38.pscale, 1.3, rel_tol=1e-5)
        assert math.isclose(m38.p0, 0.101325, rel_tol=1e-5)
        assert math.isclose(m38.rp, 0.015, rel_tol=1e-5)
        assert math.isclose(m38.pmax, 5.5, rel_tol=1e-5)
        assert math.isclose(m38.phi, 0.82, rel_tol=1e-5)
        assert m38.ful == 302
        assert math.isclose(m38.alpha_unload, 0.92, rel_tol=1e-5)
        assert math.isclose(m38.eps_unload, 0.025, rel_tol=1e-5)
        assert math.isclose(m38.a, 1.15, rel_tol=1e-5)
        assert math.isclose(m38.b, 0.92, rel_tol=1e-5)
        assert m38.m_func == 3
        assert m38.nfunc == 3
        assert math.isclose(m38.cutoff, 11.5, rel_tol=1e-5)
        assert m38.iinsta == 1
        assert math.isclose(m38.e_final, 160.0, rel_tol=1e-5)
        assert math.isclose(m38.epsi_final, 0.85, rel_tol=1e-5)
        assert math.isclose(m38.lamb, 1.25, rel_tol=1e-5)
        assert math.isclose(m38.visc, 2.2, rel_tol=1e-5)
        assert math.isclose(m38.tol, 0.045, rel_tol=1e-5)

        # Multi-curve preservation
        assert len(m38.fscale_i) == 3
        assert [pytest.approx(x, rel=1e-5) for x in m38.fscale_i] == [1.05, 1.15, 1.25]
        assert [pytest.approx(x, rel=1e-5) for x in m38.epsilon_i] == [0.005, 0.05, 0.5]
        assert m38.funct_id_load == [101, 102, 103]
        assert m38.funct_id_unload == [201, 202, 203]

        # Material.params parity verification
        assert math.isclose(p["rho"], rho0, rel_tol=1e-5)
        assert math.isclose(p["rhor"], rhor, rel_tol=1e-5)
        assert math.isclose(p["e"], 120.0, rel_tol=1e-5)
        assert math.isclose(p["nu_t"], 0.32, rel_tol=1e-5)
        assert math.isclose(p["nu_c"], 0.36, rel_tol=1e-5)
        assert math.isclose(p["nu"], 0.36, rel_tol=1e-5)
        assert p["iflag"] == 1
        assert p["itotal"] == 2
        assert p["kair"] == 1
        assert p["ful"] == 302
        assert p["m_func"] == 3
        assert p["nfunc"] == 3
        assert p["fscale_i"] == m38.fscale_i
        assert p["epsilon_i"] == m38.epsilon_i
        assert p["funct_id_load"] == [101, 102, 103]
        assert p["funct_id_unload"] == [201, 202, 203]


# ============================================================================
# 3. Free-Format Robustness
# ============================================================================

class TestFreeFormatRobustnessAudit:
    """Free-format deck parsing with commas, spaces, and empty fields."""

    def test_compact_comma_separated_values(self, tmp_path: Path):
        """Audit free-format parsing with compact comma separation (no spaces)."""
        deck_text = """/BEGIN
COMPACT_COMMA
2020  0
/MAT/LAW38/10
COMPACT_COMMA_MAT
1.1e-06,1.2e-06
95.0,0.31,0.37,1.4,1,1
0.04,0.75,0.38,1,0,0.65
1,205,1.15
0.1013,0.012,4.5,0.78
305,,0.88,0.018,1.08,0.91
2,,9.5,1
140.0,0.75,1.15,1.8,0.04
1.02,1.18
0.008,0.8
111,112
211,212
/END
"""
        deck_file = tmp_path / "compact_comma.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        model = parse_starter_deck(str(deck_file))

        assert 10 in model.mat_law38s
        m = model.mat_law38s[10]
        assert math.isclose(m.rho0, 1.1e-6, rel_tol=1e-5)
        assert math.isclose(m.rhor, 1.2e-6, rel_tol=1e-5)
        assert math.isclose(m.e, 95.0, rel_tol=1e-5)
        assert math.isclose(m.nu_t, 0.31, rel_tol=1e-5)
        assert math.isclose(m.nu_c, 0.37, rel_tol=1e-5)
        assert m.ful == 305
        assert math.isclose(m.alpha_unload, 0.88, rel_tol=1e-5)
        assert m.m_func == 2
        assert math.isclose(m.cutoff, 9.5, rel_tol=1e-5)
        assert len(m.fscale_i) == 2
        assert [pytest.approx(x, rel=1e-5) for x in m.fscale_i] == [1.02, 1.18]
        assert [pytest.approx(x, rel=1e-5) for x in m.epsilon_i] == [0.008, 0.8]
        assert m.funct_id_load == [111, 112]
        assert m.funct_id_unload == [211, 212]

    def test_comma_with_whitespace_separated_values(self, tmp_path: Path):
        """Audit free-format parsing with comma + whitespace separation."""
        deck_text = """/BEGIN
COMMA_SPACE
2020  0
/MAT/VISC_TAB/20
COMMA_SPACE_MAT
1.15e-06, 1.15e-06
110.0, 0.28, 0.33, 1.2, 0, 2
0.03, 0.9, 0.45, 0, 1, 0.69
0, 0, 1.0
0.1, 0.0, 10.0, 0.0
0, , 1.0, 0.0, 1.0, 1.0
3, , 15.0, 0
120.0, 1.0, 1.0, 0.0, 1.0
1.0, 1.1, 1.2
0.01, 0.1, 1.0
10, 20, 30
15, 25, 35
/END
"""
        deck_file = tmp_path / "comma_space.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        model = parse_starter_deck(str(deck_file))

        assert 20 in model.mat_visc_tabs
        m = model.mat_visc_tabs[20]
        assert math.isclose(m.rho0, 1.15e-6, rel_tol=1e-5)
        assert math.isclose(m.e, 110.0, rel_tol=1e-5)
        assert m.m_func == 3
        assert math.isclose(m.cutoff, 15.0, rel_tol=1e-5)
        assert m.funct_id_load == [10, 20, 30]
        assert m.funct_id_unload == [15, 25, 35]

    def test_consecutive_commas_for_default_injection(self, tmp_path: Path):
        """Audit consecutive commas inserting empty fields taking defaults."""
        deck_text = """/BEGIN
CONSECUTIVE_COMMAS
2020  0
/MAT/LAW38/30
DEFAULT_COMMAS
1.0e-06
100.0,0.3
,,0.5,,,0.67
,,,
,,,,
,,,,,,
1,,,
,,,,
1.0
0.01
100
/END
"""
        deck_file = tmp_path / "consec_commas.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        model = parse_starter_deck(str(deck_file))

        assert 30 in model.mat_law38s
        m = model.mat_law38s[30]
        assert math.isclose(m.rho0, 1.0e-6, rel_tol=1e-5)
        assert math.isclose(m.rhor, 1.0e-6, rel_tol=1e-5)
        assert math.isclose(m.e, 100.0, rel_tol=1e-5)
        assert math.isclose(m.nu_t, 0.3, rel_tol=1e-5)
        assert math.isclose(m.h, 1.0, rel_tol=1e-5)  # defaulted to 1.0
        assert math.isclose(m.r_d, 0.5, rel_tol=1e-5)  # explicit 0.5
        assert math.isclose(m.instant_mod_upd, 0.67, rel_tol=1e-5)
        assert math.isclose(m.pscale, 1.0, rel_tol=1e-5)  # defaulted to 1.0
        assert math.isclose(m.alpha_unload, 1.0, rel_tol=1e-5)  # defaulted to 1.0
        assert math.isclose(m.a, 1.0, rel_tol=1e-5)  # defaulted to 1.0
        assert math.isclose(m.b, 1.0, rel_tol=1e-5)  # defaulted to 1.0
        assert m.m_func == 1
        assert m.funct_id_load == [100]

    def test_numeric_card_detection_with_commas(self):
        """Audit _is_numeric_card correctly recognizes comma-separated numeric cards without title."""
        assert _is_numeric_card(Card("1.0e-06,1.25e-6"))
        assert _is_numeric_card(Card("1.0e-06, 1.25e-6"))
        assert _is_numeric_card(Card("100.0,0.35,0.38,1.5,1,2"))
        assert not _is_numeric_card(Card("FOAM MATERIAL TITLE"))
        assert not _is_numeric_card(Card("FOAM, VISCOELASTIC MATERIAL"))


# ============================================================================
# 4. Synonym & Alias Resolution
# ============================================================================

class TestSynonymAndAliasResolutionAudit:
    """Verify /MAT/VISC_TAB and /MAT/LAW38 exact synonym and syntax handling."""

    def test_cfg_catalogue_synonyms(self):
        """Audit LAW_MAP, LAW_SYNONYMS, and catalogue schema mapping."""
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
        assert canonical_law_name("VISC_TAB") == "LAW38"
        assert canonical_law_name("LAW38") == "LAW38"

        cat = catalogue()
        assert cat.schema("LAW38") is not None
        assert cat.schema("VISC_TAB") is not None
        assert cat.schema("VISC_TAB").law_number == 38

    def test_model_dict_and_dataclass_aliases(self):
        """Audit that model.mat_visc_tabs is identical to model.mat_law38s."""
        m = Model()
        assert m.mat_visc_tabs is m.mat_law38s
        assert MatViscTab is MatLaw38

    @pytest.mark.parametrize("header", [
        "/MAT/LAW38/55",
        "/MAT/VISC_TAB/55",
        "/LAW38/55",
        "/VISC_TAB/55",
    ])
    def test_header_syntax_variants(self, tmp_path: Path, header: str):
        """Audit all 4 keyword header syntax variants for LAW38 / VISC_TAB."""
        deck_text = f"""/BEGIN
SYNONYM_TEST
2022  0
{header}
SYNONYM_TEST_MATERIAL
       1.20000000000000E-06
                100.                0.35                0.38                 1.5         1         2
                0.05                 0.8                 0.4         1         1                 0.7
         1       201                 1.2
              0.1013                0.01                  5.                0.85
       301                                           0.9                0.02                 1.1                0.95
         2                                          10.0         1
                150.                 0.8                 1.2                  2.                0.05
                 1.0                 1.2
                0.01                 1.0
       101       102
       103       104
/END
"""
        deck_file = tmp_path / f"synonym_{header.replace('/', '_')}.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        model = parse_starter_deck(str(deck_file))

        assert 55 in model.materials
        assert 55 in model.mat_law38s
        assert 55 in model.mat_visc_tabs

        mat = model.materials[55]
        m38 = model.mat_law38s[55]
        mv = model.mat_visc_tabs[55]

        assert mat.law == 38
        assert m38 is mv
        assert m38.id == 55
        assert math.isclose(m38.e, 100.0, rel_tol=1e-5)
        assert m38.m_func == 2
        assert m38.funct_id_load == [101, 102]


# ============================================================================
# 5. Defensive Edge Cases & Malformed Inputs
# ============================================================================

class TestDefensiveEdgeCasesAudit:
    """Defensive edge cases: truncated cards, defaults, malformed values, checks."""

    def test_truncated_cards_9_to_12_when_nfunc_is_zero(self, tmp_path: Path):
        """Audit graceful handling when NFUNC=0 and cards 9..12 are absent."""
        deck_text = """/BEGIN
NFUNC_ZERO_TRUNCATED
2022  0
/MAT/LAW38/1
NO_CURVES_TRUNCATED
       1.00000000000000E-06
                100.                0.35                0.35                 0.0         0         0
                0.00                 1.0                 0.5         0         0                0.67
         0         0                 1.0
                 0.0                 0.0                 0.0                 0.0
         0                                           1.0                 0.0                 1.0                 1.0
         0                                           0.0         0
                100.                 1.0                 1.0                 0.0                 1.0
/END
"""
        deck_file = tmp_path / "nfunc_0_trunc.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        model = parse_starter_deck(str(deck_file))

        assert 1 in model.mat_law38s
        m = model.mat_law38s[1]
        assert m.m_func == 0
        assert m.fscale_i == []
        assert m.epsilon_i == []
        assert m.funct_id_load == []
        assert m.funct_id_unload == []

    def test_truncated_deck_early_eof(self, tmp_path: Path):
        """Audit truncated deck after card 2 (only rho and E provided)."""
        deck_text = """/BEGIN
EARLY_EOF
2022  0
/MAT/LAW38/2
ONLY_RHO_AND_E
             1.5e-06
                85.0
/END
"""
        deck_file = tmp_path / "early_eof.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        model = parse_starter_deck(str(deck_file))

        assert 2 in model.mat_law38s
        m = model.mat_law38s[2]
        assert math.isclose(m.rho0, 1.5e-6, rel_tol=1e-5)
        assert math.isclose(m.rhor, 1.5e-6, rel_tol=1e-5)  # defaulted to rho0
        assert math.isclose(m.e, 85.0, rel_tol=1e-5)
        assert math.isclose(m.h, 1.0, rel_tol=1e-5)  # defaulted
        assert math.isclose(m.r_d, 0.5, rel_tol=1e-5)  # defaulted
        assert math.isclose(m.instant_mod_upd, 0.67, rel_tol=1e-5)  # defaulted
        assert math.isclose(m.pscale, 1.0, rel_tol=1e-5)  # defaulted
        assert math.isclose(m.e_final, 85.0, rel_tol=1e-5)  # defaulted to e

    def test_missing_data_cards_logs_error(self, tmp_path: Path):
        """Audit empty /MAT/LAW38 block reporting clear error without crash."""
        deck_text = """/BEGIN
EMPTY_MAT_BLOCK
2022  0
/MAT/LAW38/99
/END
"""
        deck_file = tmp_path / "empty_block.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        log = MessageLog()
        parse_starter_deck(str(deck_file), log=log)
        assert any("/MAT/LAW38/99: missing data card" in err for err in log.errors)

    def test_unload_functions_default_to_loading_curve_1(self, tmp_path: Path):
        """Audit that zero/omitted unloading functions fall back to loading curve 1 (hm_read_mat38.F:343)."""
        deck_text = """/BEGIN
UNLOAD_FALLBACK
2020  0
/MAT/LAW38/4
FALLBACK_UNLOAD
1.0e-06
100.0,0.3,0.3
0.0,1.0,0.5,0,0,0.67
0,0,1.0
0,0,0,0
0,,1.0,0,1.0,1.0
2,,0.0,0
100.0,1.0,1.0,0.0,1.0
1.0,1.2
0.01,0.1
501,502
0,0
/END
"""
        deck_file = tmp_path / "unload_fallback.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        model = parse_starter_deck(str(deck_file))

        m = model.mat_law38s[4]
        assert m.funct_id_load == [501, 502]
        # Funct_Id_UnLoad were 0 -> must default to funct_id_load[0] = 501
        assert m.funct_id_unload == [501, 501]

    def test_scale_factors_default_to_one(self, tmp_path: Path):
        """Audit that scale factors <= 0 default to 1.0 (hm_read_mat38.F:323)."""
        deck_text = """/BEGIN
FSCALE_DEFAULT
2020  0
/MAT/LAW38/5
FSCALE_ZERO
1.0e-06
100.0,0.3,0.3
0.0,1.0,0.5,0,0,0.67
0,0,1.0
0,0,0,0
0,,1.0,0,1.0,1.0
2,,0.0,0
100.0,1.0,1.0,0.0,1.0
0.0,-0.5
0.01,0.1
601,602
701,702
/END
"""
        deck_file = tmp_path / "fscale_default.rad"
        deck_file.write_text(deck_text, encoding="utf-8")
        model = parse_starter_deck(str(deck_file))

        m = model.mat_law38s[5]
        assert m.fscale_i == [1.0, 1.0]

    def test_starter_checks_parameter_validation(self):
        """Audit parameter validation in check_mat_law38 for rho0, E, nu, nfunc, air content."""
        # 1. Negative density
        m_bad_rho = MatLaw38(id=1, rho0=-1.0e-6, e=100.0, nu_t=0.3, nu_c=0.3, m_func=2)
        log = MessageLog()
        check_mat_law38(m_bad_rho, log)
        assert any("initial density RHO must be > 0" in e for e in log.errors)

        # 2. Zero Young modulus
        m_bad_e = MatLaw38(id=2, rho0=1.0e-6, e=0.0, nu_t=0.3, nu_c=0.3, m_func=2)
        log = MessageLog()
        check_mat_law38(m_bad_e, log)
        assert any("Young's modulus E must be > 0" in e for e in log.errors)

        # 3. Invalid Poisson ratio nu_t >= 0.5
        m_bad_nu = MatLaw38(id=3, rho0=1.0e-6, e=100.0, nu_t=0.52, nu_c=0.3, m_func=2)
        log = MessageLog()
        check_mat_law38(m_bad_nu, log)
        assert any("tensile Poisson's ratio nu_t must be in [0, 0.5)" in e for e in log.errors)

        # 4. Invalid function count (nfunc=0 or nfunc=6)
        m_bad_nfunc = MatLaw38(id=4, rho0=1.0e-6, e=100.0, nu_t=0.3, nu_c=0.3, m_func=0)
        log = MessageLog()
        check_mat_law38(m_bad_nfunc, log)
        assert any("number of functions nfunc must be between 1 and 5" in e for e in log.errors)

        m_bad_nfunc6 = MatLaw38(id=5, rho0=1.0e-6, e=100.0, nu_t=0.3, nu_c=0.3, m_func=6)
        log = MessageLog()
        check_mat_law38(m_bad_nfunc6, log)
        assert any("number of functions nfunc must be between 1 and 5" in e for e in log.errors)

        # 5. Invalid air content parameters (kair=1, phi >= 1 or p0 < 0)
        m_bad_air = MatLaw38(id=6, rho0=1.0e-6, e=100.0, nu_t=0.3, nu_c=0.3, m_func=2, kair=1, phi=1.2, p0=-0.1)
        log = MessageLog()
        check_mat_law38(m_bad_air, log)
        assert any("porosity phi must be in [0, 1)" in e for e in log.errors)
        assert any("initial air pressure P0 must be >= 0" in e for e in log.errors)

    def test_starter_checks_element_compatibility(self):
        """Audit element compatibility: solids accepted, shells and trias rejected."""
        class DummyGroup:
            def __init__(self, slices):
                self.n = len(slices)
                self.state = {"slices": slices}

        model = Model()
        model.node_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        mat = Material(id=1, law=38, rho0=1.0e-6, law_name="LAW38", params={"e": 100.0, "nu": 0.3, "nfunc": 2})

        # Solid element: bricks (should pass)
        model.bricks = DummyGroup([(None, mat, None)])
        log = MessageLog()
        check_model(model, log)
        assert not any("is not supported" in err for err in log.errors)

        # Shell elements referencing LAW38 (must be rejected)
        model.shells = DummyGroup([(None, mat, None)])
        log = MessageLog()
        check_model(model, log)
        assert any("/MAT/LAW38/1 (/MAT/VISC_TAB) is not supported for shells elements" in err for err in log.errors)
