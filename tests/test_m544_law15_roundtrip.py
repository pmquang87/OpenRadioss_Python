"""
Roundtrip, DeckWriter, and Starter Validation Audit for Milestone M544:
/MAT/LAW15 (/MAT/CHANG, /MAT/PLAS_ANISO, /MAT/COMP_CHANG).

Cites:
  - starter/source/materials/mat/mat015/hm_read_mat15.F
  - hm_cfg_files/config/CFG/radioss110/MAT/matl15_chang.cfg
  - pyradioss/input/deck_writer.py
  - pyradioss/input/starter_keywords.py
  - pyradioss/starter/checks.py
  - pyradioss/materials/law15_chang.py
"""

from __future__ import annotations

import pytest
import numpy as np

from pyradioss.input.deck_writer import StarterDeck, _conv_mat, read_lines_to_blocks
from pyradioss.input.card_layouts import LAYOUTS, split_fixed
from pyradioss.input.cfg_catalogue import CfgCatalogue
from pyradioss.input.starter_keywords import (
    KEYWORD_PARSERS,
    KeywordBlock,
    Card,
    read_mat,
    read_mat_law15,
    read_mat_chang,
    read_mat_plas_aniso,
    read_mat_comp_chang,
)
from pyradioss.model.model import Model
from pyradioss.model.entities import MatLaw15, Material
from pyradioss.starter.checks import check_mat_law15, check_materials, check_model, _ALLOWED_LAWS
from pyradioss.common.messages import MessageLog
from pyradioss import materials


# ============================================================================
# 1. Complete 7-Card Fixed Format Output & Specification Verification
# ============================================================================

def test_deck_writer_7_cards_fixed_format_spec():
    """Verify StarterDeck.mat_law15 produces exactly 7 data cards matching OpenRadioss spec."""
    deck = StarterDeck("LAW15_7CARDS")
    deck.mat_law15(
        mid=15,
        title="CHANG_7_CARD_SPEC",
        rho=1.5e-9,
        refer_rho=0.0,
        e11=150000.0,
        e22=9000.0,
        nu12=0.3,
        g12=4500.0,
        g23=3000.0,
        g31=4500.0,
        b=100.0,
        n=0.5,
        fmax=2000.0,
        wpmax=50.0,
        wpref=1.0,
        ioff=1,
        sig_1yt=1800.0,
        sig_2yt=40.0,
        sig_1yc=1200.0,
        sig_2yc=160.0,
        alpha=1.0,
        sig_12yc=60.0,
        sig_12yt=60.0,
        c=0.05,
        eps_dot_0=1.0,
        icc=1,
    )
    rendered = deck.render()
    lines = [line.rstrip() for line in rendered.split("\n") if line.strip() and not line.startswith("#")]

    # Find /MAT/LAW15/15
    mat_idx = [i for i, line in enumerate(lines) if line.startswith("/MAT/LAW15/15")][0]
    assert lines[mat_idx] == "/MAT/LAW15/15"
    assert lines[mat_idx + 1] == "CHANG_7_CARD_SPEC"

    # Exactly 7 data cards (lines mat_idx + 2 to mat_idx + 8)
    data_cards = lines[mat_idx + 2 : mat_idx + 9]
    assert len(data_cards) == 7

    # Card 1: RHO_I (%20lg)
    c1 = split_fixed(data_cards[0], LAYOUTS["MAT_LAW15_1"])
    assert float(c1[0]) == pytest.approx(1.5e-9)

    # Card 2: E11 E22 nu12 (%20lg%20lg%20lg)
    c2 = split_fixed(data_cards[1], LAYOUTS["MAT_LAW15_2"])
    assert float(c2[0]) == pytest.approx(150000.0)
    assert float(c2[1]) == pytest.approx(9000.0)
    assert float(c2[2]) == pytest.approx(0.3)

    # Card 3: G12 G23 G31 (%20lg%20lg%20lg)
    c3 = split_fixed(data_cards[2], LAYOUTS["MAT_LAW15_3"])
    assert float(c3[0]) == pytest.approx(4500.0)
    assert float(c3[1]) == pytest.approx(3000.0)
    assert float(c3[2]) == pytest.approx(4500.0)

    # Card 4: b n fmax (%20lg%20lg%20lg)
    c4 = split_fixed(data_cards[3], LAYOUTS["MAT_LAW15_4"])
    assert float(c4[0]) == pytest.approx(100.0)
    assert float(c4[1]) == pytest.approx(0.5)
    assert float(c4[2]) == pytest.approx(2000.0)

    # Card 5: Wpmax Wpref Ioff (%20lg%20lg%10d)
    c5 = split_fixed(data_cards[4], LAYOUTS["MAT_LAW15_5"])
    assert float(c5[0]) == pytest.approx(50.0)
    assert float(c5[1]) == pytest.approx(1.0)
    assert int(c5[2]) == 1

    # Card 6: sigma_1yt sigma_2yt sigma_1yc sigma_2yc alpha (%20lg%20lg%20lg%20lg%20lg)
    c6 = split_fixed(data_cards[5], LAYOUTS["MAT_LAW15_6"])
    assert float(c6[0]) == pytest.approx(1800.0)
    assert float(c6[1]) == pytest.approx(40.0)
    assert float(c6[2]) == pytest.approx(1200.0)
    assert float(c6[3]) == pytest.approx(160.0)
    assert float(c6[4]) == pytest.approx(1.0)

    # Card 7: sigma_12yc sigma_12yt c Eps_dot_0 ICC (%20lg%20lg%20lg%20lg%10d)
    c7 = split_fixed(data_cards[6], LAYOUTS["MAT_LAW15_7"])
    assert float(c7[0]) == pytest.approx(60.0)
    assert float(c7[1]) == pytest.approx(60.0)
    assert float(c7[2]) == pytest.approx(0.05)
    assert float(c7[3]) == pytest.approx(1.0)
    assert int(c7[4]) == 1


# ============================================================================
# 2. Complete 9-Card Fixed Format Output Verification
# ============================================================================

def test_deck_writer_9_cards_fixed_format_spec():
    """Verify StarterDeck.mat_law15 produces all 9 cards when failure/rate parameters are set."""
    deck = StarterDeck("LAW15_9CARDS")
    deck.mat_law15(
        mid=20,
        title="CHANG_9_CARD_SPEC",
        rho=1.6e-9,
        refer_rho=1.55e-9,
        e11=140000.0,
        e22=8500.0,
        nu12=0.28,
        g12=4200.0,
        g23=2800.0,
        g31=4200.0,
        b=80.0,
        n=0.6,
        fmax=2200.0,
        wpmax=60.0,
        wpref=1.5,
        ioff=2,
        sig_1yt=1600.0,
        sig_2yt=35.0,
        sig_1yc=1100.0,
        sig_2yc=150.0,
        alpha=0.95,
        sig_12yc=55.0,
        sig_12yt=55.0,
        c=0.04,
        eps_dot_0=0.5,
        icc=2,
        beta=0.75,
        tmax=0.005,
        s1=1900.0,
        s2=45.0,
        s12=65.0,
        fsmooth=1,
        fcut=10000.0,
        c1=1300.0,
        c2=170.0,
    )
    rendered = deck.render()
    lines = [line.rstrip() for line in rendered.split("\n") if line.strip() and not line.startswith("#")]

    # Find /MAT/LAW15/20
    mat_idx = [i for i, line in enumerate(lines) if line.startswith("/MAT/LAW15/20")][0]
    assert lines[mat_idx] == "/MAT/LAW15/20"
    assert lines[mat_idx + 1] == "CHANG_9_CARD_SPEC"

    data_cards = lines[mat_idx + 2 : mat_idx + 11]
    assert len(data_cards) == 9

    # Card 1 with RHO_O
    c1 = split_fixed(data_cards[0], LAYOUTS["MAT_LAW15_1"])
    assert float(c1[0]) == pytest.approx(1.6e-9)
    assert float(c1[1]) == pytest.approx(1.55e-9)

    # Card 8: beta Tmax S1 S2 S12 (%20lg%20lg%20lg%20lg%20lg)
    c8 = split_fixed(data_cards[7], LAYOUTS["MAT_LAW15_8"])
    assert float(c8[0]) == pytest.approx(0.75)
    assert float(c8[1]) == pytest.approx(0.005)
    assert float(c8[2]) == pytest.approx(1900.0)
    assert float(c8[3]) == pytest.approx(45.0)
    assert float(c8[4]) == pytest.approx(65.0)

    # Card 9: Fsmooth Fcut C1 C2 (%10d%20lg%20lg%20lg)
    c9 = split_fixed(data_cards[8], LAYOUTS["MAT_LAW15_9"])
    assert int(c9[0]) == 1
    assert float(c9[1]) == pytest.approx(10000.0)
    assert float(c9[2]) == pytest.approx(1300.0)
    assert float(c9[3]) == pytest.approx(170.0)


# ============================================================================
# 3. All Keyword Aliases: /MAT/LAW15, /MAT/CHANG, /MAT/PLAS_ANISO, /MAT/COMP_CHANG
# ============================================================================

@pytest.mark.parametrize("method_name,expected_kw", [
    ("mat_law15", "LAW15"),
    ("mat_chang", "CHANG"),
    ("mat_plas_aniso", "PLAS_ANISO"),
    ("mat_comp_chang", "COMP_CHANG"),
])
def test_all_keyword_aliases_emitter_and_parsing(method_name: str, expected_kw: str):
    """Verify all 4 alias methods write correct header and roundtrip through parsing."""
    deck = StarterDeck("ALIAS_TEST")
    fn = getattr(deck, method_name)
    fn(
        mid=101,
        title=f"TEST_{expected_kw}",
        rho=1.5e-9,
        e11=150000.0,
        e22=9000.0,
        nu12=0.3,
        g12=4500.0,
        g23=3000.0,
        g31=4500.0,
        b=100.0,
        n=0.5,
        fmax=2000.0,
        wpmax=50.0,
        wpref=1.0,
        ioff=1,
        sig_1yt=1800.0,
        sig_2yt=40.0,
        sig_1yc=1200.0,
        sig_2yc=160.0,
        alpha=1.0,
        sig_12yc=60.0,
        sig_12yt=60.0,
        c=0.05,
        eps_dot_0=1.0,
        icc=1,
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
    assert 101 in model.mat_law15s
    m15 = model.mat_law15s[101]
    assert m15.law_name == expected_kw
    assert m15.rho0 == pytest.approx(1.5e-9)
    assert m15.e11 == pytest.approx(150000.0)

    # Check that KEYWORD_PARSERS dispatches alias directly as well
    direct_parser = KEYWORD_PARSERS.get(f"MAT_{expected_kw}")
    assert direct_parser is not None


# ============================================================================
# 4. Comprehensive 32-Parameter Roundtrip Audit (Fixed & Free Formats)
# ============================================================================

def test_all_32_parameters_roundtrip_fixed_and_free():
    """Verify all 32 parameters and aliases roundtrip cleanly in fixed and free format."""
    # Target parameter set
    params = {
        "rho": 1.55e-9,
        "refer_rho": 1.50e-9,
        "E1": 145000.0,
        "E2": 8800.0,
        "nu12": 0.29,
        "G12": 4300.0,
        "G23": 2900.0,
        "G31": 4300.0,
        "b": 95.0,
        "hard": 0.55,
        "sig": 2100.0,
        "wpmax": 55.0,
        "wpref": 1.2,
        "itype": 3,
        "sigyt1": 1750.0,
        "sigyt2": 38.0,
        "sigyc1": 1150.0,
        "sigyc2": 155.0,
        "alpha": 0.98,
        "sigc12": 58.0,
        "sigt12": 58.0,
        "src": 0.045,
        "srp": 0.8,
        "strflag": 2,
        "beta_s": 0.7,
        "tmax": 0.008,
        "s12": 68.0,
        "s1": 1950.0,
        "s2": 48.0,
        "c1": 1350.0,
        "c2": 175.0,
        "fsmooth": 1,
        "fcut": 8500.0,
    }

    # 1. Emit using StarterDeck with exact parameter alias names
    deck = StarterDeck("PARAM32_TEST")
    deck.mat_law15(
        mid=42,
        title="ALL_32_PARAMS",
        **params,
    )
    rendered_fixed = deck.render()

    # 2. Roundtrip via Fixed-format parsing
    blocks_fixed = read_lines_to_blocks(rendered_fixed.splitlines())
    blk_fixed = [b for b in blocks_fixed if b.parts and b.parts[0] == "MAT"][0]
    blk_fixed.fixed = True
    model_fixed = Model()
    log_fixed = MessageLog()
    read_mat_law15(blk_fixed, model_fixed, log_fixed)
    assert not log_fixed.has_errors
    m_fix = model_fixed.mat_law15s[42]

    # Verify all 32 parameters in fixed format
    assert m_fix.rho0 == pytest.approx(params["rho"])
    assert m_fix.rhor == pytest.approx(params["refer_rho"])
    assert m_fix.e11 == pytest.approx(params["E1"])
    assert m_fix.e22 == pytest.approx(params["E2"])
    assert m_fix.nu12 == pytest.approx(params["nu12"])
    assert m_fix.g12 == pytest.approx(params["G12"])
    assert m_fix.g23 == pytest.approx(params["G23"])
    assert m_fix.g31 == pytest.approx(params["G31"])
    assert m_fix.b == pytest.approx(params["b"])
    assert m_fix.n == pytest.approx(params["hard"])
    assert m_fix.fmax == pytest.approx(params["sig"])
    assert m_fix.wpmax == pytest.approx(params["wpmax"])
    assert m_fix.wpref == pytest.approx(params["wpref"])
    assert m_fix.ioff == params["itype"]
    assert m_fix.sig_1yt == pytest.approx(params["sigyt1"])
    assert m_fix.sig_2yt == pytest.approx(params["sigyt2"])
    assert m_fix.sig_1yc == pytest.approx(params["sigyc1"])
    assert m_fix.sig_2yc == pytest.approx(params["sigyc2"])
    assert m_fix.alpha == pytest.approx(params["alpha"])
    assert m_fix.sig_12yc == pytest.approx(params["sigc12"])
    assert m_fix.sig_12yt == pytest.approx(params["sigt12"])
    assert m_fix.c == pytest.approx(params["src"])
    assert m_fix.eps_dot_0 == pytest.approx(params["srp"])
    assert m_fix.icc == params["strflag"]
    assert m_fix.beta == pytest.approx(params["beta_s"])
    assert m_fix.tmax == pytest.approx(params["tmax"])
    assert m_fix.s1 == pytest.approx(params["s1"])
    assert m_fix.s2 == pytest.approx(params["s2"])
    assert m_fix.s12 == pytest.approx(params["s12"])
    assert isinstance(m_fix.c1, float)
    assert isinstance(m_fix.c2, float)
    assert m_fix.c1 == pytest.approx(params["c1"])
    assert m_fix.c2 == pytest.approx(params["c2"])
    assert m_fix.fsmooth == params["fsmooth"]
    assert m_fix.fcut == pytest.approx(params["fcut"])

    # 3. Roundtrip via Free-format parsing
    free_cards = [
        Card(raw="ALL_32_FREE"),
        Card(raw=f"{params['rho']} {params['refer_rho']}"),
        Card(raw=f"{params['E1']} {params['E2']} {params['nu12']}"),
        Card(raw=f"{params['G12']} {params['G23']} {params['G31']}"),
        Card(raw=f"{params['b']} {params['hard']} {params['sig']}"),
        Card(raw=f"{params['wpmax']} {params['wpref']} {params['itype']}"),
        Card(raw=f"{params['sigyt1']} {params['sigyt2']} {params['sigyc1']} {params['sigyc2']} {params['alpha']}"),
        Card(raw=f"{params['sigc12']} {params['sigt12']} {params['src']} {params['srp']} {params['strflag']}"),
        Card(raw=f"{params['beta_s']} {params['tmax']} {params['s1']} {params['s2']} {params['s12']}"),
        Card(raw=f"{params['fsmooth']} {params['fcut']} {params['c1']} {params['c2']}"),
    ]
    blk_free = KeywordBlock(
        keyword="/MAT/LAW15/43",
        user_id=43,
        parts=["MAT", "LAW15", "43"],
        cards=free_cards,
        fixed=False,
    )
    model_free = Model()
    log_free = MessageLog()
    read_mat_law15(blk_free, model_free, log_free)
    assert not log_free.has_errors
    m_free = model_free.mat_law15s[43]

    assert m_free.rho0 == pytest.approx(params["rho"])
    assert m_free.rhor == pytest.approx(params["refer_rho"])
    assert m_free.e11 == pytest.approx(params["E1"])
    assert m_free.e22 == pytest.approx(params["E2"])
    assert m_free.nu12 == pytest.approx(params["nu12"])
    assert m_free.g12 == pytest.approx(params["G12"])
    assert m_free.g23 == pytest.approx(params["G23"])
    assert m_free.g31 == pytest.approx(params["G31"])
    assert m_free.b == pytest.approx(params["b"])
    assert m_free.n == pytest.approx(params["hard"])
    assert m_free.fmax == pytest.approx(params["sig"])
    assert m_free.wpmax == pytest.approx(params["wpmax"])
    assert m_free.wpref == pytest.approx(params["wpref"])
    assert m_free.ioff == params["itype"]
    assert m_free.sig_1yt == pytest.approx(params["sigyt1"])
    assert m_free.sig_2yt == pytest.approx(params["sigyt2"])
    assert m_free.sig_1yc == pytest.approx(params["sigyc1"])
    assert m_free.sig_2yc == pytest.approx(params["sigyc2"])
    assert m_free.alpha == pytest.approx(params["alpha"])
    assert m_free.sig_12yc == pytest.approx(params["sigc12"])
    assert m_free.sig_12yt == pytest.approx(params["sigt12"])
    assert m_free.c == pytest.approx(params["src"])
    assert m_free.eps_dot_0 == pytest.approx(params["srp"])
    assert m_free.icc == params["strflag"]
    assert m_free.beta == pytest.approx(params["beta_s"])
    assert m_free.tmax == pytest.approx(params["tmax"])
    assert m_free.s1 == pytest.approx(params["s1"])
    assert m_free.s2 == pytest.approx(params["s2"])
    assert m_free.s12 == pytest.approx(params["s12"])
    assert m_free.c1 == pytest.approx(params["c1"])
    assert m_free.c2 == pytest.approx(params["c2"])
    assert m_free.fsmooth == params["fsmooth"]
    assert m_free.fcut == pytest.approx(params["fcut"])


# ============================================================================
# 5. 7-Card Fixed Format Default Parameters Roundtrip
# ============================================================================

def test_7_cards_defaults_and_no_shadowing():
    """Verify that a 7-card deck sets c1=0.0 and c2=0.0 as floats, not lists."""
    deck = StarterDeck("DEFAULT_7CARDS")
    deck.mat_law15(
        mid=77,
        title="CHANG_DEFAULT_7",
        rho=1.5e-9,
        e11=150000.0,
        e22=9000.0,
        nu12=0.3,
        g12=4500.0,
        g23=3000.0,
        g31=4500.0,
        b=100.0,
        n=0.5,
        fmax=2000.0,
        wpmax=50.0,
        wpref=1.0,
        ioff=1,
        sig_1yt=1800.0,
        sig_2yt=40.0,
        sig_1yc=1200.0,
        sig_2yc=160.0,
        alpha=1.0,
        sig_12yc=60.0,
        sig_12yt=60.0,
        c=0.05,
        eps_dot_0=1.0,
        icc=1,
    )
    blocks = read_lines_to_blocks(deck.render().splitlines())
    blk = [b for b in blocks if b.parts and b.parts[0] == "MAT"][0]
    blk.fixed = True
    model = Model()
    log = MessageLog()
    read_mat_law15(blk, model, log)
    assert not log.has_errors
    m = model.mat_law15s[77]

    # Crucial regression check: c1 and c2 must be 0.0 floats, not card token lists!
    assert isinstance(m.c1, float)
    assert isinstance(m.c2, float)
    assert m.c1 == 0.0
    assert m.c2 == 0.0
    assert m.beta == 0.0
    assert m.tmax == 0.0
    assert m.s1 == 0.0
    assert m.s2 == 0.0
    assert m.s12 == 0.0
    assert m.fsmooth == 0
    assert m.fcut == 0.0

    # Active material created with fallback strengths
    active_mat = model.materials[77]
    assert active_mat.params["ioff"] == 1
    assert active_mat.params["s1"] == pytest.approx(1800.0)


# ============================================================================
# 6. Free Format Parsing & Deck Conversion Fidelity
# ============================================================================

def test_deck_conversion_fidelity_with_interleaved_comments():
    """Verify _conv_mat parses fixed and free format blocks with interleaved comments."""
    deck = StarterDeck("CONV_TEST")
    block = KeywordBlock(
        keyword="/MAT/LAW15/88",
        user_id=88,
        parts=["MAT", "LAW15", "88"],
        cards=[
            Card(raw="CHANG_CONV_TEST"),
            Card(raw="# Card 1: Density"),
            Card(raw="1.5e-9"),
            Card(raw="# Card 2: Moduli"),
            Card(raw="150000.0 9000.0 0.3"),
            Card(raw="# Card 3: Shear moduli"),
            Card(raw="4500.0 3000.0 4500.0"),
            Card(raw="# Card 4: Hardening"),
            Card(raw="100.0 0.5 2000.0"),
            Card(raw="# Card 5: Plastic work"),
            Card(raw="50.0 1.0 1"),
            Card(raw="# Card 6: Yield stresses"),
            Card(raw="1800.0 40.0 1200.0 160.0 1.0"),
            Card(raw="# Card 7: Shear yield"),
            Card(raw="60.0 60.0 0.05 1.0 1"),
        ],
        source="test:conv",
        fixed=False,
    )
    _conv_mat(deck, block)
    content = deck.render()
    assert "/MAT/LAW15/88" in content
    assert "CHANG_CONV_TEST" in content

    # Re-parse through read_lines_to_blocks and read_mat_law15
    blocks = read_lines_to_blocks(content.splitlines())
    mat_blocks = [b for b in blocks if b.parts and b.parts[0] == "MAT"]
    assert len(mat_blocks) == 1
    m = Model()
    log = MessageLog()
    read_mat_law15(mat_blocks[0], m, log)
    assert not log.has_errors
    m88 = m.mat_law15s[88]
    assert m88.rho0 == pytest.approx(1.5e-9)
    assert m88.e11 == pytest.approx(150000.0)
    assert m88.nu12 == pytest.approx(0.3)


# ============================================================================
# 7. CFG Interpreter & matl15_chang.cfg Schema Verification
# ============================================================================

def test_cfg_interpreter_matl15_chang():
    """Verify CfgCatalogue schema loads and maps all attributes of matl15_chang.cfg."""
    cat = CfgCatalogue()
    schema = cat.schema("LAW15")
    assert schema is not None
    assert "matl15_chang.cfg" in schema.path.lower()

    # Check that key attribute names exist in schema
    common_attrs = schema.attributes
    for attr in [
        "MAT_RHO", "MAT_EA", "MAT_EB", "MAT_PRAB", "MAT_GAB", "MAT_GBC", "MAT_GCA",
        "MAT_BETA", "MAT_HARD", "MAT_SIG", "WPMAX", "WPREF", "Itype",
        "MAT_SIGYT1", "MAT_SIGYT2", "MAT_SIGYC1", "MAT_SIGYC2", "MAT_ALPHA",
        "MAT_SIGC12", "MAT_SIGT12", "MAT_SRC", "MAT_SRP", "STRFLAG",
        "MAT_Beta", "MAT_TMAX", "MCHANG_S1", "MCHANG_S2", "MCHANG_S12",
        "Fsmooth", "Fcut", "MCHANG_C1", "MCHANG_C2"
    ]:
        assert attr in common_attrs, f"Missing attribute {attr} in matl15_chang.cfg schema"


# ============================================================================
# 8. Starter Check Diagnostics: Complete Bounds & Compatibility Audit
# ============================================================================

def test_starter_check_diagnostics_density_moduli_detc():
    """Verify diagnostics for rho <= 0, E1/E2 <= 0, detc <= 0, G12/G23/G31 <= 0."""
    # rho <= 0
    m_rho = MatLaw15(id=1, rho0=0.0, e11=150000.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
                     sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0)
    log = MessageLog()
    check_mat_law15(m_rho, log)
    assert any("initial density RHO must be > 0" in e for e in log.errors)

    # E1 <= 0
    m_e1 = MatLaw15(id=2, rho0=1.5e-9, e11=-100.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
                    sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0)
    log = MessageLog()
    check_mat_law15(m_e1, log)
    assert any("Young's modulus E1 must be > 0" in e for e in log.errors)

    # E2 <= 0
    m_e2 = MatLaw15(id=3, rho0=1.5e-9, e11=150000.0, e22=0.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
                    sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0)
    log = MessageLog()
    check_mat_law15(m_e2, log)
    assert any("Young's modulus E2 must be > 0" in e for e in log.errors)

    # detc = 1 - nu12*nu21 <= 0 (nu12 * nu21 >= 1.0)
    # nu21 = nu12 * E2 / E1. If E1=1000, E2=1000, nu12=1.0 -> nu21=1.0 -> detc = 0.0 <= 0
    m_detc = MatLaw15(id=4, rho0=1.5e-9, e11=1000.0, e22=1000.0, nu12=1.0, g12=4500.0, g23=3000.0, g31=4500.0,
                      sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0)
    log = MessageLog()
    check_mat_law15(m_detc, log)
    assert any("detc = 1 - nu12*nu21 must be > 0" in e for e in log.errors)

    # G12 <= 0, G23 <= 0, G31 <= 0
    for g_field in ["g12", "g23", "g31"]:
        kwargs = {"rho0": 1.5e-9, "e11": 150000.0, "e22": 9000.0, "nu12": 0.3,
                  "g12": 4500.0, "g23": 3000.0, "g31": 4500.0,
                  "sig_1yt": 1800.0, "sig_2yt": 40.0, "sig_1yc": 1200.0, "sig_2yc": 160.0,
                  "sig_12yc": 60.0, "sig_12yt": 60.0}
        kwargs[g_field] = 0.0
        m_g = MatLaw15(id=5, **kwargs)
        log = MessageLog()
        check_mat_law15(m_g, log)
        assert any(f"shear modulus {g_field.upper()} must be > 0" in e for e in log.errors)


def test_starter_check_diagnostics_yield_stresses():
    """Verify diagnostics for all 6 yield stresses <= 0."""
    yield_fields = [
        ("sig_1yt", "tensile yield stress in dir 1 (sig_1yt) must be > 0"),
        ("sig_1yc", "compressive yield stress in dir 1 (sig_1yc) must be > 0"),
        ("sig_2yt", "tensile yield stress in dir 2 (sig_2yt) must be > 0"),
        ("sig_2yc", "compressive yield stress in dir 2 (sig_2yc) must be > 0"),
        ("sig_12yc", "compressive shear yield stress in dir 12 (sig_12yc) must be > 0"),
        ("sig_12yt", "tensile shear yield stress in dir 12 (sig_12yt) must be > 0"),
    ]
    for field, msg in yield_fields:
        kwargs = {
            "rho0": 1.5e-9, "e11": 150000.0, "e22": 9000.0, "nu12": 0.3,
            "g12": 4500.0, "g23": 3000.0, "g31": 4500.0,
            "sig_1yt": 1800.0, "sig_2yt": 40.0, "sig_1yc": 1200.0, "sig_2yc": 160.0,
            "sig_12yc": 60.0, "sig_12yt": 60.0,
        }
        kwargs[field] = 0.0
        m = MatLaw15(id=10, **kwargs)
        log = MessageLog()
        check_mat_law15(m, log)
        assert any(msg in e for e in log.errors), f"Failed to catch {field} <= 0"


def test_starter_check_diagnostics_hardening_bounds():
    """Verify diagnostics for hardening exponent b > 1.0, n > 1.0, and passing when <= 1.0."""
    # b > 1.0 rejected
    m_b = MatLaw15(id=11, rho0=1.5e-9, e11=150000.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
                   sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0,
                   b=1.2, n=0.5)
    log = MessageLog()
    check_mat_law15(m_b, log)
    assert any("hardening parameter b must be <= 1.0" in e for e in log.errors)

    # b <= 1.0 accepted
    m_b_ok = MatLaw15(id=12, rho0=1.5e-9, e11=150000.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
                      sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0,
                      b=1.0, n=1.0)
    log = MessageLog()
    check_mat_law15(m_b_ok, log)
    assert not log.has_errors

    # n > 1.0 rejected
    m_n = MatLaw15(id=13, rho0=1.5e-9, e11=150000.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
                   sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0,
                   b=0.5, n=1.5)
    log = MessageLog()
    check_mat_law15(m_n, log)
    assert any("hardening exponent n must be <= 1.0" in e for e in log.errors)


def test_starter_check_diagnostics_strengths_and_tmax():
    """Verify diagnostics for strengths <= 0 and tmax <= 0 when failure criteria are active."""
    strength_fields = [
        ("s1", "longitudinal tensile strength S1 must be > 0"),
        ("s2", "transverse tensile strength S2 must be > 0"),
        ("s12", "shear strength S12 must be > 0"),
        ("c1", "longitudinal compressive strength C1 must be > 0"),
        ("c2", "transverse compressive strength C2 must be > 0"),
    ]
    for field, msg in strength_fields:
        kwargs = {
            "rho0": 1.5e-9, "e11": 150000.0, "e22": 9000.0, "nu12": 0.3,
            "g12": 4500.0, "g23": 3000.0, "g31": 4500.0,
            "sig_1yt": 1800.0, "sig_2yt": 40.0, "sig_1yc": 1200.0, "sig_2yc": 160.0,
            "sig_12yc": 60.0, "sig_12yt": 60.0,
            "tmax": 0.01, "s1": 2000.0, "s2": 50.0, "s12": 70.0, "c1": 1400.0, "c2": 180.0,
        }
        kwargs[field] = 0.0
        m = MatLaw15(id=20, **kwargs)
        log = MessageLog()
        check_mat_law15(m, log)
        assert any(msg in e for e in log.errors), f"Failed to catch {field} <= 0"

    # tmax <= 0 when failure active
    kwargs = {
        "rho0": 1.5e-9, "e11": 150000.0, "e22": 9000.0, "nu12": 0.3,
        "g12": 4500.0, "g23": 3000.0, "g31": 4500.0,
        "sig_1yt": 1800.0, "sig_2yt": 40.0, "sig_1yc": 1200.0, "sig_2yc": 160.0,
        "sig_12yc": 60.0, "sig_12yt": 60.0,
        "tmax": 0.0, "s1": 2000.0, "s2": 50.0, "s12": 70.0, "c1": 1400.0, "c2": 180.0,
    }
    m_tmax = MatLaw15(id=21, **kwargs)
    log = MessageLog()
    check_mat_law15(m_tmax, log)
    assert any("stress relaxation time tmax must be > 0" in e for e in log.errors)


@pytest.mark.parametrize("incompatible_elem", ["bricks", "tetras", "penta6", "pyra5", "trusses", "beams", "springs"])
def test_check_model_rejects_incompatible_elements(incompatible_elem: str):
    """Verify check_model rejects LAW15 on solid, truss, beam, and spring element groups with clear diagnostics."""
    model = Model()
    model.add_nodes(np.array([1, 2]), np.zeros((2, 3)))
    mat15 = MatLaw15(
        id=15, rho0=1.5e-9, e11=150000.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
        sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0,
        b=0.8, n=0.5, fmax=2000.0, wpmax=50.0, wpref=1.0, ioff=1,
    )
    model.materials[15] = mat15

    class DummyGroup:
        def __init__(self):
            self.state = {"slices": [(slice(0, 1), mat15, None)]}

    model.element_groups = lambda: [(incompatible_elem, DummyGroup())]
    log = MessageLog()
    check_model(model, log)
    assert any(f"is not supported for {incompatible_elem} elements" in e for e in log.errors)


@pytest.mark.parametrize("compatible_elem", ["shells", "shells_qbat", "shells_qeph", "sh3n", "quads"])
def test_check_model_accepts_compatible_elements(compatible_elem: str):
    """Verify check_model accepts LAW15 on shell families (shells, qbat, qeph, sh3n, quads)."""
    model = Model()
    model.add_nodes(np.array([1, 2]), np.zeros((2, 3)))
    mat15 = MatLaw15(
        id=15, rho0=1.5e-9, e11=150000.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
        sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0,
        b=0.8, n=0.5, fmax=2000.0, wpmax=50.0, wpref=1.0, ioff=1,
    )
    model.materials[15] = mat15

    class DummyGroup:
        def __init__(self):
            self.state = {"slices": [(slice(0, 1), mat15, None)]}

    model.element_groups = lambda: [(compatible_elem, DummyGroup())]
    log = MessageLog()
    check_model(model, log)
    assert not log.has_errors


# ============================================================================
# 9. Fluent Chaining & Edge Cases
# ============================================================================

def test_deck_writer_fluent_chaining():
    """Verify StarterDeck.mat_law15 and aliases support fluent chaining."""
    deck = StarterDeck("CHAIN_TEST")
    res = (
        deck.mat_law15(1, "MAT1", rho=1.5e-9, e11=1e5, e22=1e4, nu12=0.3, g12=5e3, g23=5e3, g31=5e3,
                       sig_1yt=1000, sig_2yt=50, sig_1yc=800, sig_2yc=100, sig_12yc=50, sig_12yt=50)
            .mat_chang(2, "MAT2", rho=1.5e-9, e11=1e5, e22=1e4, nu12=0.3, g12=5e3, g23=5e3, g31=5e3,
                       sig_1yt=1000, sig_2yt=50, sig_1yc=800, sig_2yc=100, sig_12yc=50, sig_12yt=50)
            .mat_plas_aniso(3, "MAT3", rho=1.5e-9, e11=1e5, e22=1e4, nu12=0.3, g12=5e3, g23=5e3, g31=5e3,
                            sig_1yt=1000, sig_2yt=50, sig_1yc=800, sig_2yc=100, sig_12yc=50, sig_12yt=50)
            .mat_comp_chang(4, "MAT4", rho=1.5e-9, e11=1e5, e22=1e4, nu12=0.3, g12=5e3, g23=5e3, g31=5e3,
                            sig_1yt=1000, sig_2yt=50, sig_1yc=800, sig_2yc=100, sig_12yc=50, sig_12yt=50)
    )
    assert res is deck
    rendered = deck.render()
    assert "/MAT/LAW15/1" in rendered
    assert "/MAT/CHANG/2" in rendered
    assert "/MAT/PLAS_ANISO/3" in rendered
    assert "/MAT/COMP_CHANG/4" in rendered


def test_unit_system_roundtrip():
    """Verify unit_id is included in header and roundtrips."""
    deck = StarterDeck("UNIT_TEST")
    deck.mat_law15(
        mid=55,
        title="UNIT_MAT",
        unit_id=2,
        rho=1.5e-9, e11=150000.0, e22=9000.0, nu12=0.3, g12=4500.0, g23=3000.0, g31=4500.0,
        sig_1yt=1800.0, sig_2yt=40.0, sig_1yc=1200.0, sig_2yc=160.0, sig_12yc=60.0, sig_12yt=60.0,
    )
    rendered = deck.render()
    assert "/MAT/LAW15/55/2" in rendered
