"""
Roundtrip, DeckWriter, and Starter Validation Audit for Milestone M545:
/MAT/LAW22 (/MAT/DAMA, /MAT/PLAS_DAMA).

Cites:
  - starter/source/materials/mat/mat022/hm_read_mat22.F
  - hm_cfg_files/config/CFG/radioss110/MAT/matl22_dama.cfg
  - pyradioss/input/deck_writer.py
  - pyradioss/input/starter_keywords.py
  - pyradioss/starter/checks.py
  - pyradioss/materials/law22_dama.py
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.input.deck_writer import StarterDeck, _conv_mat, read_lines_to_blocks
from pyradioss.input.card_layouts import LAYOUTS, split_fixed
import pyradioss.input.card_layouts as cl
from pyradioss.input.cfg_catalogue import CfgCatalogue
from pyradioss.input.starter_keywords import (
    KEYWORD_PARSERS,
    KeywordBlock,
    Card,
    read_mat,
    read_mat_law22,
    read_mat_dama,
    read_mat_plas_dama,
    parse_starter_deck,
)
from pyradioss.model.model import Model
from pyradioss.model.entities import MatLaw22, Material
from pyradioss.starter.checks import check_mat_law22, check_materials, check_model, _ALLOWED_LAWS
from pyradioss.common.messages import MessageLog


# ============================================================================
# 1. Complete 5-Card Fixed Format Output & Specification Verification
# ============================================================================

def test_deck_writer_5_cards_fixed_format_spec():
    """Verify StarterDeck.mat_law22 produces exactly 5 data cards matching OpenRadioss spec."""
    deck = StarterDeck("LAW22_5CARDS")
    deck.mat_law22(
        mid=1,
        title="DP600_DAMA_SPEC",
        rho=7.85e-9,
        refer_rho=7.80e-9,
        e=210000.0,
        nu=0.3,
        a=350.0,
        b=450.0,
        n=0.5,
        eps_max=0.25,
        sig_max=900.0,
        c=0.03,
        eps_dot_0=1.0,
        icc=1,
        eps_dam=0.05,
        e_tan=-10000.0,
    )
    rendered = deck.render()
    lines = [line.rstrip() for line in rendered.splitlines() if line.strip() and not line.startswith("#")]

    # Find /MAT/LAW22/1
    mat_idx = [i for i, line in enumerate(lines) if line.startswith("/MAT/LAW22/1")][0]
    assert lines[mat_idx] == "/MAT/LAW22/1"
    assert lines[mat_idx + 1] == "DP600_DAMA_SPEC"

    # Exactly 5 data cards (lines mat_idx + 2 to mat_idx + 6)
    data_cards = lines[mat_idx + 2 : mat_idx + 7]
    assert len(data_cards) == 5

    # Card 1: RHO_I Refer_Rho (%20lg%20lg)
    c1 = split_fixed(data_cards[0], LAYOUTS["MAT_LAW22_1"])
    assert float(c1[0]) == pytest.approx(7.85e-9)
    assert float(c1[1]) == pytest.approx(7.80e-9)

    # Card 2: E Nu (%20lg%20lg)
    c2 = split_fixed(data_cards[1], LAYOUTS["MAT_LAW22_2"])
    assert float(c2[0]) == pytest.approx(210000.0)
    assert float(c2[1]) == pytest.approx(0.3)

    # Card 3: a b n Eps_max SIGMA_max (%20lg%20lg%20lg%20lg%20lg)
    c3 = split_fixed(data_cards[2], LAYOUTS["MAT_LAW22_3"])
    assert float(c3[0]) == pytest.approx(350.0)
    assert float(c3[1]) == pytest.approx(450.0)
    assert float(c3[2]) == pytest.approx(0.5)
    assert float(c3[3]) == pytest.approx(0.25)
    assert float(c3[4]) == pytest.approx(900.0)

    # Card 4: c Eps_dot_0 ICC (%20lg%20lg%10d)
    c4 = split_fixed(data_cards[3], LAYOUTS["MAT_LAW22_4"])
    assert float(c4[0]) == pytest.approx(0.03)
    assert float(c4[1]) == pytest.approx(1.0)
    assert int(c4[2]) == 1

    # Card 5: Eps_dam E_t (%20lg%20lg)
    c5 = split_fixed(data_cards[4], LAYOUTS["MAT_LAW22_5"])
    assert float(c5[0]) == pytest.approx(0.05)
    assert float(c5[1]) == pytest.approx(-10000.0)


def test_deck_writer_single_density_card_when_refer_rho_none_or_zero():
    """Verify Card 1 contains single RHO field when refer_rho is 0 or None."""
    # refer_rho is None
    deck1 = StarterDeck("RHO_NONE")
    deck1.mat_law22(mid=2, title="T1", rho=7.85e-9, refer_rho=None, e=200000.0, nu=0.3, a=300.0)
    lines1 = [line.rstrip() for line in deck1.render().splitlines() if line.strip() and not line.startswith("#")]
    idx1 = [i for i, line in enumerate(lines1) if line.startswith("/MAT/LAW22/2")][0]
    card1_line1 = lines1[idx1 + 2]
    toks1 = card1_line1.split()
    assert len(toks1) == 1
    assert float(toks1[0]) == pytest.approx(7.85e-9)

    # refer_rho is 0.0
    deck2 = StarterDeck("RHO_ZERO")
    deck2.mat_law22(mid=3, title="T2", rho=7.85e-9, refer_rho=0.0, e=200000.0, nu=0.3, a=300.0)
    lines2 = [line.rstrip() for line in deck2.render().splitlines() if line.strip() and not line.startswith("#")]
    idx2 = [i for i, line in enumerate(lines2) if line.startswith("/MAT/LAW22/3")][0]
    card1_line2 = lines2[idx2 + 2]
    toks2 = card1_line2.split()
    assert len(toks2) == 1
    assert float(toks2[0]) == pytest.approx(7.85e-9)


# ============================================================================
# 2. All Keyword Aliases: /MAT/LAW22, /MAT/DAMA, /MAT/PLAS_DAMA
# ============================================================================

@pytest.mark.parametrize("method_name,expected_kw", [
    ("mat_law22", "LAW22"),
    ("mat_dama", "DAMA"),
    ("mat_plas_dama", "PLAS_DAMA"),
])
def test_all_keyword_aliases_emitter_and_parsing(method_name: str, expected_kw: str):
    """Verify all 3 alias methods write correct header and roundtrip through parsing."""
    deck = StarterDeck("ALIAS_TEST")
    fn = getattr(deck, method_name)
    fn(
        mid=101,
        title=f"TEST_{expected_kw}",
        rho=7.85e-9,
        e=210000.0,
        nu=0.3,
        a=350.0,
        b=450.0,
        n=0.5,
        eps_max=0.25,
        sig_max=900.0,
        c=0.03,
        eps_dot_0=1.0,
        icc=1,
        eps_dam=0.05,
        e_tan=-10000.0,
    )
    rendered = deck.render()
    assert f"/MAT/{expected_kw}/101" in rendered

    # Parse rendered deck with read_lines_to_blocks and read_mat
    blocks = read_lines_to_blocks(rendered.splitlines())
    mat_blocks = [b for b in blocks if b.parts and b.parts[0] == "MAT"]
    assert len(mat_blocks) == 1
    blk = mat_blocks[0]
    assert blk.parts[0] == "MAT"
    assert blk.parts[1] == expected_kw
    assert blk.user_id == 101

    model = Model()
    log = MessageLog()
    read_mat(blk, model, log)
    assert not log.has_errors
    assert 101 in model.mat_law22s
    m22 = model.mat_law22s[101]
    assert m22.law_name == expected_kw
    assert m22.rho0 == pytest.approx(7.85e-9)
    assert m22.e == pytest.approx(210000.0)
    assert m22.nu == pytest.approx(0.3)
    assert m22.a == pytest.approx(350.0)
    assert m22.b == pytest.approx(450.0)
    assert m22.n == pytest.approx(0.5)
    assert m22.eps_dam == pytest.approx(0.05)
    assert m22.e_tan == pytest.approx(-10000.0)

    # Active material created
    assert 101 in model.materials
    active = model.materials[101]
    assert active.law == 22
    assert active.params["E"] == pytest.approx(210000.0)
    assert active.params["E_tan"] == pytest.approx(-10000.0)

    # Check direct aliases on Model
    assert 101 in model.mat_damas
    if expected_kw == "PLAS_DAMA":
        assert 101 in model.mat_plas_damas

    # Check that KEYWORD_PARSERS dispatches alias directly
    direct_parser = KEYWORD_PARSERS.get(f"MAT_{expected_kw}")
    assert direct_parser is not None
    assert direct_parser is read_mat

    bare_parser = KEYWORD_PARSERS.get(expected_kw)
    assert bare_parser is not None
    assert bare_parser is read_mat


def test_direct_parser_functions_exported():
    """Verify read_mat_law22, read_mat_dama, read_mat_plas_dama can be called directly."""
    deck_str = """
/MAT/LAW22/50
Direct_Parser_Test
  7.85e-9  7.80e-9
 210000.0      0.3
    350.0    450.0      0.5     0.25    900.0
     0.03      1.0        1
     0.05 -10000.0
"""
    blocks = read_lines_to_blocks(deck_str.splitlines())
    blk = blocks[0]

    for fn in [read_mat_law22, read_mat_dama, read_mat_plas_dama]:
        m = Model()
        l = MessageLog()
        fn(blk, m, l)
        assert not l.has_errors
        assert 50 in m.mat_law22s
        mat = m.mat_law22s[50]
        assert mat.rho0 == pytest.approx(7.85e-9)
        assert mat.rhor == pytest.approx(7.80e-9)
        assert mat.e == pytest.approx(210000.0)
        assert mat.e_tan == pytest.approx(-10000.0)


# ============================================================================
# 3. Comprehensive 14-Parameter Roundtrip Audit (Fixed & Free Formats)
# ============================================================================

def test_all_14_parameters_roundtrip_fixed_and_free():
    """Verify all 14 parameters roundtrip cleanly in fixed and free format."""
    params = {
        "rho": 7.82e-9,
        "refer_rho": 7.78e-9,
        "e": 208000.0,
        "nu": 0.285,
        "a": 340.0,
        "b": 420.0,
        "n": 0.48,
        "eps_max": 0.22,
        "sig_max": 880.0,
        "c": 0.025,
        "eps_dot_0": 0.8,
        "icc": 2,
        "eps_dam": 0.06,
        "e_tan": -8500.0,
    }

    # 1. Emit using StarterDeck
    deck = StarterDeck("PARAM14_TEST")
    deck.mat_law22(
        mid=42,
        title="ALL_14_PARAMS",
        **params,
    )
    rendered_fixed = deck.render()

    # 2. Roundtrip via Fixed-format parsing
    blocks_fixed = read_lines_to_blocks(rendered_fixed.splitlines())
    blk_fixed = [b for b in blocks_fixed if b.parts and b.parts[0] == "MAT"][0]
    blk_fixed.fixed = True
    model_fixed = Model()
    log_fixed = MessageLog()
    read_mat_law22(blk_fixed, model_fixed, log_fixed)
    assert not log_fixed.has_errors
    m_fix = model_fixed.mat_law22s[42]

    assert m_fix.rho0 == pytest.approx(params["rho"])
    assert m_fix.rhor == pytest.approx(params["refer_rho"])
    assert m_fix.e == pytest.approx(params["e"])
    assert m_fix.nu == pytest.approx(params["nu"])
    assert m_fix.a == pytest.approx(params["a"])
    assert m_fix.b == pytest.approx(params["b"])
    assert m_fix.n == pytest.approx(params["n"])
    assert m_fix.eps_max == pytest.approx(params["eps_max"])
    assert m_fix.sig_max == pytest.approx(params["sig_max"])
    assert m_fix.c == pytest.approx(params["c"])
    assert m_fix.eps_dot_0 == pytest.approx(params["eps_dot_0"])
    assert m_fix.icc == params["icc"]
    assert m_fix.eps_dam == pytest.approx(params["eps_dam"])
    assert m_fix.e_tan == pytest.approx(params["e_tan"])

    # 3. Roundtrip via Free-format parsing (whitespace)
    free_cards_ws = [
        Card(raw="ALL_14_FREE_WS"),
        Card(raw=f"{params['rho']} {params['refer_rho']}"),
        Card(raw=f"{params['e']} {params['nu']}"),
        Card(raw=f"{params['a']} {params['b']} {params['n']} {params['eps_max']} {params['sig_max']}"),
        Card(raw=f"{params['c']} {params['eps_dot_0']} {params['icc']}"),
        Card(raw=f"{params['eps_dam']} {params['e_tan']}"),
    ]
    blk_free_ws = KeywordBlock(
        keyword="/MAT/LAW22/43",
        user_id=43,
        parts=["MAT", "LAW22", "43"],
        cards=free_cards_ws,
        fixed=False,
    )
    model_free_ws = Model()
    log_free_ws = MessageLog()
    read_mat_law22(blk_free_ws, model_free_ws, log_free_ws)
    assert not log_free_ws.has_errors
    m_free_ws = model_free_ws.mat_law22s[43]

    assert m_free_ws.rho0 == pytest.approx(params["rho"])
    assert m_free_ws.rhor == pytest.approx(params["refer_rho"])
    assert m_free_ws.e == pytest.approx(params["e"])
    assert m_free_ws.nu == pytest.approx(params["nu"])
    assert m_free_ws.a == pytest.approx(params["a"])
    assert m_free_ws.b == pytest.approx(params["b"])
    assert m_free_ws.n == pytest.approx(params["n"])
    assert m_free_ws.eps_max == pytest.approx(params["eps_max"])
    assert m_free_ws.sig_max == pytest.approx(params["sig_max"])
    assert m_free_ws.c == pytest.approx(params["c"])
    assert m_free_ws.eps_dot_0 == pytest.approx(params["eps_dot_0"])
    assert m_free_ws.icc == params["icc"]
    assert m_free_ws.eps_dam == pytest.approx(params["eps_dam"])
    assert m_free_ws.e_tan == pytest.approx(params["e_tan"])

    # 4. Roundtrip via Free-format parsing (comma-separated)
    free_cards_comma = [
        Card(raw="ALL_14_FREE_COMMA"),
        Card(raw=f"{params['rho']}, {params['refer_rho']}"),
        Card(raw=f"{params['e']}, {params['nu']}"),
        Card(raw=f"{params['a']}, {params['b']}, {params['n']}, {params['eps_max']}, {params['sig_max']}"),
        Card(raw=f"{params['c']}, {params['eps_dot_0']}, {params['icc']}"),
        Card(raw=f"{params['eps_dam']}, {params['e_tan']}"),
    ]
    blk_free_comma = KeywordBlock(
        keyword="/MAT/LAW22/44",
        user_id=44,
        parts=["MAT", "LAW22", "44"],
        cards=free_cards_comma,
        fixed=False,
    )
    model_free_comma = Model()
    log_free_comma = MessageLog()
    read_mat_law22(blk_free_comma, model_free_comma, log_free_comma)
    assert not log_free_comma.has_errors
    m_free_comma = model_free_comma.mat_law22s[44]

    assert m_free_comma.rho0 == pytest.approx(params["rho"])
    assert m_free_comma.rhor == pytest.approx(params["refer_rho"])
    assert m_free_comma.e == pytest.approx(params["e"])
    assert m_free_comma.nu == pytest.approx(params["nu"])
    assert m_free_comma.a == pytest.approx(params["a"])
    assert m_free_comma.b == pytest.approx(params["b"])
    assert m_free_comma.n == pytest.approx(params["n"])
    assert m_free_comma.eps_max == pytest.approx(params["eps_max"])
    assert m_free_comma.sig_max == pytest.approx(params["sig_max"])
    assert m_free_comma.c == pytest.approx(params["c"])
    assert m_free_comma.eps_dot_0 == pytest.approx(params["eps_dot_0"])
    assert m_free_comma.icc == params["icc"]
    assert m_free_comma.eps_dam == pytest.approx(params["eps_dam"])
    assert m_free_comma.e_tan == pytest.approx(params["e_tan"])


def test_all_kwarg_aliases_accepted_by_deck_writer():
    """Verify all kwarg alias spellings in StarterDeck.mat_law22 produce identical output."""
    alias_kwargs = {
        "rho0": 7.85e-9,
        "rho_ref": 7.80e-9,
        "young": 210000.0,
        "poisson": 0.3,
        "sigy": 350.0,
        "beta": 450.0,
        "hard": 0.5,
        "epsmax": 0.25,
        "sigmax": 900.0,
        "src": 0.03,
        "srp": 1.0,
        "strflag": 1,
        "damage": 0.05,
        "etan": -10000.0,
    }
    deck = StarterDeck("KWARG_ALIAS")
    deck.mat_law22(mid=60, title="ALIAS_TITLE", **alias_kwargs)
    rendered = deck.render()

    blocks = read_lines_to_blocks(rendered.splitlines())
    blk = [b for b in blocks if b.parts and b.parts[0] == "MAT"][0]
    m = Model()
    log = MessageLog()
    read_mat_law22(blk, m, log)
    assert not log.has_errors
    mat = m.mat_law22s[60]

    assert mat.rho0 == pytest.approx(7.85e-9)
    assert mat.rhor == pytest.approx(7.80e-9)
    assert mat.e == pytest.approx(210000.0)
    assert mat.nu == pytest.approx(0.3)
    assert mat.a == pytest.approx(350.0)
    assert mat.b == pytest.approx(450.0)
    assert mat.n == pytest.approx(0.5)
    assert mat.eps_max == pytest.approx(0.25)
    assert mat.sig_max == pytest.approx(900.0)
    assert mat.c == pytest.approx(0.03)
    assert mat.eps_dot_0 == pytest.approx(1.0)
    assert mat.icc == 1
    assert mat.eps_dam == pytest.approx(0.05)
    assert mat.e_tan == pytest.approx(-10000.0)


# ============================================================================
# 4. Deck Conversion Fidelity (_conv_mat)
# ============================================================================

def test_deck_conversion_fidelity_with_interleaved_comments():
    """Verify _conv_mat parses fixed and free format blocks with interleaved comments and converts faithfully."""
    deck = StarterDeck("CONV_TEST")
    block = KeywordBlock(
        keyword="/MAT/LAW22/88",
        user_id=88,
        parts=["MAT", "LAW22", "88"],
        cards=[
            Card(raw="LAW22_CONV_TEST"),
            Card(raw="# Card 1: Density"),
            Card(raw="7.85e-9 7.80e-9"),
            Card(raw="# Card 2: Elasticity"),
            Card(raw="210000.0 0.3"),
            Card(raw="# Card 3: Plasticity"),
            Card(raw="350.0 450.0 0.5 0.25 900.0"),
            Card(raw="# Card 4: Rate effects"),
            Card(raw="0.03 1.0 1"),
            Card(raw="# Card 5: Damage"),
            Card(raw="0.05 -10000.0"),
        ],
        source="test:conv",
        fixed=False,
    )
    _conv_mat(deck, block)
    content = deck.render()
    assert "/MAT/LAW22/88" in content
    assert "LAW22_CONV_TEST" in content

    # Re-parse through read_lines_to_blocks and read_mat_law22
    blocks = read_lines_to_blocks(content.splitlines())
    mat_blocks = [b for b in blocks if b.parts and b.parts[0] == "MAT"]
    assert len(mat_blocks) == 1
    m = Model()
    log = MessageLog()
    read_mat_law22(mat_blocks[0], m, log)
    assert not log.has_errors
    m88 = m.mat_law22s[88]
    assert m88.rho0 == pytest.approx(7.85e-9)
    assert m88.rhor == pytest.approx(7.80e-9)
    assert m88.e == pytest.approx(210000.0)
    assert m88.nu == pytest.approx(0.3)
    assert m88.a == pytest.approx(350.0)
    assert m88.b == pytest.approx(450.0)
    assert m88.n == pytest.approx(0.5)
    assert m88.eps_max == pytest.approx(0.25)
    assert m88.sig_max == pytest.approx(900.0)
    assert m88.c == pytest.approx(0.03)
    assert m88.eps_dot_0 == pytest.approx(1.0)
    assert m88.icc == 1
    assert m88.eps_dam == pytest.approx(0.05)
    assert m88.e_tan == pytest.approx(-10000.0)


# ============================================================================
# 5. CFG Catalogue & matl22_dama.cfg Schema Verification
# ============================================================================

def test_cfg_interpreter_matl22_dama():
    """Verify CfgCatalogue schema loads and maps all attributes of matl22_dama.cfg."""
    cat = CfgCatalogue()
    schema = cat.schema("LAW22")
    assert schema is not None
    assert "matl22_dama.cfg" in schema.path.lower()

    # Verify DAMA routes to matl22_dama.cfg
    schema_dama = cat.schema("DAMA")
    assert schema_dama is not None
    assert "matl22_dama.cfg" in schema_dama.path.lower()

    # Note: in Altair/OpenRadioss CFG, PLAS_DAMA is also mapped in matl23_plas_dama.cfg
    schema_plas_dama = cat.schema("PLAS_DAMA")
    assert schema_plas_dama is not None
    assert "matl23_plas_dama.cfg" in schema_plas_dama.path.lower() or "matl22_dama.cfg" in schema_plas_dama.path.lower()

    # Check that key attribute names exist in schema
    attrs = schema.attributes
    for attr in [
        "MAT_RHO", "Refer_Rho", "MAT_E", "MAT_NU", "MAT_SIGY",
        "MAT_BETA", "MAT_HARD", "MAT_EPS", "MAT_SIG", "MAT_SRC",
        "MAT_SRP", "STRFLAG", "MAT_DAMAGE", "MAT_ETAN",
    ]:
        assert attr in attrs, f"Missing attribute {attr} in matl22_dama.cfg schema"


# ============================================================================
# 6. Starter Check Diagnostics: Complete Bounds Audit
# ============================================================================

def test_starter_check_diagnostics_density():
    """Verify diagnostics for rho <= 0 (zero and negative)."""
    # rho = 0.0
    m_zero = MatLaw22(id=1, rho0=0.0, e=210000.0, nu=0.3, a=350.0, n=0.5, eps_dot_0=1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_zero, log)
    assert any("initial density RHO must be > 0" in e for e in log.errors)

    # rho < 0.0
    m_neg = MatLaw22(id=2, rho0=-7.85e-9, e=210000.0, nu=0.3, a=350.0, n=0.5, eps_dot_0=1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_neg, log)
    assert any("initial density RHO must be > 0" in e for e in log.errors)


def test_starter_check_diagnostics_young_modulus():
    """Verify diagnostics for E <= 0 (zero and negative)."""
    # E = 0.0
    m_zero = MatLaw22(id=3, rho0=7.85e-9, e=0.0, nu=0.3, a=350.0, n=0.5, eps_dot_0=1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_zero, log)
    assert any("Young's modulus E must be > 0" in e for e in log.errors)

    # E < 0.0
    m_neg = MatLaw22(id=4, rho0=7.85e-9, e=-210000.0, nu=0.3, a=350.0, n=0.5, eps_dot_0=1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_neg, log)
    assert any("Young's modulus E must be > 0" in e for e in log.errors)


def test_starter_check_diagnostics_poisson_ratio():
    """Verify diagnostics for nu < 0 and nu >= 0.5."""
    # nu < 0
    m_neg = MatLaw22(id=5, rho0=7.85e-9, e=210000.0, nu=-0.1, a=350.0, n=0.5, eps_dot_0=1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_neg, log)
    assert any("Poisson's ratio NU must be in [0, 0.5)" in e for e in log.errors)

    # nu = 0.5
    m_half = MatLaw22(id=6, rho0=7.85e-9, e=210000.0, nu=0.5, a=350.0, n=0.5, eps_dot_0=1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_half, log)
    assert any("Poisson's ratio NU must be in [0, 0.5)" in e for e in log.errors)

    # nu > 0.5
    m_high = MatLaw22(id=7, rho0=7.85e-9, e=210000.0, nu=0.6, a=350.0, n=0.5, eps_dot_0=1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_high, log)
    assert any("Poisson's ratio NU must be in [0, 0.5)" in e for e in log.errors)

    # nu in [0, 0.5) valid
    m_ok = MatLaw22(id=8, rho0=7.85e-9, e=210000.0, nu=0.3, a=350.0, n=0.5, eps_dot_0=1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_ok, log)
    assert not log.has_errors


def test_starter_check_diagnostics_yield_stress():
    """Verify diagnostics for yield stress a <= 0 (zero and negative)."""
    # a = 0.0
    m_zero = MatLaw22(id=9, rho0=7.85e-9, e=210000.0, nu=0.3, a=0.0, n=0.5, eps_dot_0=1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_zero, log)
    assert any("yield stress a (SIGY) must be > 0" in e for e in log.errors)

    # a < 0.0
    m_neg = MatLaw22(id=10, rho0=7.85e-9, e=210000.0, nu=0.3, a=-350.0, n=0.5, eps_dot_0=1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_neg, log)
    assert any("yield stress a (SIGY) must be > 0" in e for e in log.errors)


def test_starter_check_diagnostics_hardening_exponent():
    """Verify diagnostics for hardening exponent n > 1.0 per hm_read_mat22.F line 213."""
    # n = 1.2 > 1.0 rejected
    m_high = MatLaw22(id=11, rho0=7.85e-9, e=210000.0, nu=0.3, a=350.0, n=1.2, eps_dot_0=1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_high, log)
    assert any("hardening exponent n must be <= 1.0" in e for e in log.errors)

    # n = 1.0 accepted
    m_one = MatLaw22(id=12, rho0=7.85e-9, e=210000.0, nu=0.3, a=350.0, n=1.0, eps_dot_0=1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_one, log)
    assert not log.has_errors

    # n = 0.5 accepted
    m_half = MatLaw22(id=13, rho0=7.85e-9, e=210000.0, nu=0.3, a=350.0, n=0.5, eps_dot_0=1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_half, log)
    assert not log.has_errors


def test_starter_check_diagnostics_reference_strain_rate():
    """Verify diagnostics for reference strain rate eps_dot_0 <= 0 per hm_read_mat22.F line 221."""
    # eps_dot_0 = 0.0 rejected
    m_zero = MatLaw22(id=14, rho0=7.85e-9, e=210000.0, nu=0.3, a=350.0, n=0.5, eps_dot_0=0.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_zero, log)
    assert any("reference strain rate EPS_DOT_0 (srp) must be > 0" in e for e in log.errors)

    # eps_dot_0 < 0.0 rejected
    m_neg = MatLaw22(id=15, rho0=7.85e-9, e=210000.0, nu=0.3, a=350.0, n=0.5, eps_dot_0=-1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_neg, log)
    assert any("reference strain rate EPS_DOT_0 (srp) must be > 0" in e for e in log.errors)

    # eps_dot_0 > 0 accepted
    m_ok = MatLaw22(id=16, rho0=7.85e-9, e=210000.0, nu=0.3, a=350.0, n=0.5, eps_dot_0=1.0, e_tan=-1000.0)
    log = MessageLog()
    check_mat_law22(m_ok, log)
    assert not log.has_errors


def test_starter_check_diagnostics_softening_slope():
    """Verify diagnostics for softening damage slope E_tan > 0 per hm_read_mat22.F line 229 / matl22_dama.cfg."""
    # E_tan > 0 rejected
    m_pos = MatLaw22(id=17, rho0=7.85e-9, e=210000.0, nu=0.3, a=350.0, n=0.5, eps_dot_0=1.0, e_tan=5000.0)
    log = MessageLog()
    check_mat_law22(m_pos, log)
    assert any("softening damage slope E_tan must be <= 0.0" in e for e in log.errors)

    # E_tan = 0.0 accepted
    m_zero = MatLaw22(id=18, rho0=7.85e-9, e=210000.0, nu=0.3, a=350.0, n=0.5, eps_dot_0=1.0, e_tan=0.0)
    log = MessageLog()
    check_mat_law22(m_zero, log)
    assert not log.has_errors

    # E_tan < 0.0 accepted
    m_neg = MatLaw22(id=19, rho0=7.85e-9, e=210000.0, nu=0.3, a=350.0, n=0.5, eps_dot_0=1.0, e_tan=-10000.0)
    log = MessageLog()
    check_mat_law22(m_neg, log)
    assert not log.has_errors


def test_check_materials_invokes_check_mat_law22():
    """Verify check_materials checks all LAW22 materials in model."""
    model = Model()
    # Bad material in model.materials
    mat_bad = MatLaw22(id=21, rho0=0.0, e=210000.0, nu=0.3, a=350.0, n=0.5, eps_dot_0=1.0, e_tan=-1000.0)
    model.materials[21] = mat_bad
    log = MessageLog()
    check_materials(model, log)
    assert any("/MAT/LAW22/21: initial density RHO must be > 0" in e for e in log.errors)


# ============================================================================
# 7. Element Family Compatibility in check_model
# ============================================================================

@pytest.mark.parametrize("incompatible_elem", ["trusses", "beams", "springs"])
def test_check_model_rejects_incompatible_elements(incompatible_elem: str):
    """Verify check_model rejects LAW22 on trusses, beams, and springs with clear diagnostic message."""
    model = Model()
    model.add_nodes(np.array([1, 2]), np.zeros((2, 3)))
    mat22 = MatLaw22(
        id=22, rho0=7.85e-9, e=210000.0, nu=0.3, a=350.0, b=450.0, n=0.5,
        eps_max=0.25, sig_max=900.0, c=0.03, eps_dot_0=1.0, icc=1,
        eps_dam=0.05, e_tan=-10000.0,
    )
    model.materials[22] = mat22

    class DummyGroup:
        def __init__(self):
            self.state = {"slices": [(slice(0, 1), mat22, None)]}

    model.element_groups = lambda: [(incompatible_elem, DummyGroup())]
    log = MessageLog()
    check_model(model, log)
    assert any(
        f"/MAT/LAW22/22 (/MAT/DAMA) is not supported for {incompatible_elem} elements" in e
        for e in log.errors
    )


@pytest.mark.parametrize("compatible_elem", [
    "bricks", "tetras", "penta6", "pyra5",
    "shells", "shells_qbat", "shells_qeph", "sh3n", "quads",
])
def test_check_model_accepts_compatible_elements(compatible_elem: str):
    """Verify check_model accepts LAW22 on solids and shells."""
    model = Model()
    model.add_nodes(np.array([1, 2]), np.zeros((2, 3)))
    mat22 = MatLaw22(
        id=22, rho0=7.85e-9, e=210000.0, nu=0.3, a=350.0, b=450.0, n=0.5,
        eps_max=0.25, sig_max=900.0, c=0.03, eps_dot_0=1.0, icc=1,
        eps_dam=0.05, e_tan=-10000.0,
    )
    model.materials[22] = mat22

    class DummyGroup:
        def __init__(self):
            self.state = {"slices": [(slice(0, 1), mat22, None)]}

    model.element_groups = lambda: [(compatible_elem, DummyGroup())]
    log = MessageLog()
    check_model(model, log)
    assert not log.has_errors


# ============================================================================
# 8. Unit ID and Header Extensions
# ============================================================================

def test_deck_writer_unit_id_support():
    """Verify StarterDeck.mat_law22 writes unit_id into header and starter parser resolves it."""
    deck = StarterDeck("UNIT_TEST")
    deck.mat_law22(
        mid=7,
        unit_id=3,
        title="UNIT_LAW22",
        rho=7.85e-9,
        e=210000.0,
        nu=0.3,
        a=350.0,
        b=450.0,
        n=0.5,
        eps_max=0.25,
        sig_max=900.0,
        c=0.03,
        eps_dot_0=1.0,
        icc=1,
        eps_dam=0.05,
        e_tan=-10000.0,
    )
    rendered = deck.render()
    assert "/MAT/LAW22/7/3" in rendered

    # Parse and check unit_refs
    blocks = read_lines_to_blocks(rendered.splitlines())
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert not log.has_errors
    assert any(ref[0] == "MAT" and ref[1] == 7 and ref[2] == 3 for ref in model.raw_unit_refs)
