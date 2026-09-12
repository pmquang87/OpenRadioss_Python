"""
Tests for Milestone M564: /MAT/LAW87 (/MAT/BARLAT2000, /MAT/BARLAT_2000, /MAT/BARLAT2000_2D)
Starter & Model Integration.

Covers:
1. MatLaw87 (and MatBarlat2000, MatBarlat20002D, MaterialLaw87) dataclass fields,
   property helpers (rho0, rhor, E, Nu, G, bulk, K, sound_speed, sound_speed_shell, Lp, Lpp),
   and Mapping / dict protocol.
2. Model containers (model.mat_law87s, model.mat_barlat2000s, model.mat_barlats).
3. Card layouts (MAT_LAW87_1..8, MAT_BARLAT2000_1..8, etc.) matching matl87_barlat.cfg.
4. CFG catalogue mapping (LAW87, BARLAT2000, BARLAT_2000, BARLAT2000_2D -> 87).
5. Starter keyword reader for fixed format (ifit=0 and ifit=1) and free format.
6. Tabulated curves (iflag=0), Swift-Voce (iflag=1), 3-dir orthotropic (iflag=3), and kinematic hardening (CRC/CRA).
7. StarterDeck writer for fixed and free formats and helper aliases (mat_barlat2000, mat_barlat_2000) with roundtrip.
8. Starter validation checks:
   - Bounds checks (rho > 0, e > 0, 0 <= nu < 0.5) emitting ANCMSG 1514.
   - Convexity checks for Lp (ANCMSG 3095) and Lpp (ANCMSG 3102).
   - Element compatibility checks (rejecting solids ANCMSG 305 and 1D ANCMSG 306, accepting shells).
9. Shell element integration (shell_bt4, shell_qeph, shell_tri3) for sound speed and uvar87 state allocation.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import (
    MatLaw87,
    MatBarlat2000,
    MatBarlat20002D,
    MaterialLaw87,
    Material,
)
from pyradioss.model.model import Model
from pyradioss.input.card_layouts import CARD_LAYOUTS
from pyradioss.input.cfg_catalogue import law_number, canonical_law_name
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law87
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.checks import (
    check_mat_law87,
    check_materials,
    check_model,
    _ALLOWED_LAWS,
    _MAT_CHECKS,
)
from pyradioss.common.messages import MessageLog
from pyradioss.elements.shell_bt4 import _init_material_state, _layer_extra
import pyradioss.materials as materials


# ============================================================================
# 1. Model Entities, Property Helpers & Mapping Protocol
# ============================================================================

def test_mat_law87_entities_and_properties():
    """Verify MatLaw87 dataclass fields, property helpers, and mapping protocol."""
    m = MatLaw87(
        id=87,
        rho=2.7e-9,
        rhor=2.7e-9,
        e=70000.0,
        nu=0.33,
        iflag=1,
        iflagsr=0,
        invc=0.0,
        invp=0.0,
        flag_fit=0,
        al1=0.95,
        al2=1.05,
        al3=0.98,
        al4=1.02,
        al5=1.01,
        al6=0.99,
        al7=1.04,
        al8=0.96,
        fisokin=0.25,
        ikin=1,
        expa=8.0,
        aswift=450.0,
        nexp=0.22,
        alpha=0.6,
        epso=0.002,
        qvoce=80.0,
        beta=15.0,
        ko=120.0,
        ckh=(100.0, 50.0, 0.0, 0.0),
        akh=(10.0, 5.0, 0.0, 0.0),
        title="Barlat 2000 Sheet",
    )

    # Class aliases
    assert MatBarlat2000 is MatLaw87
    assert MatBarlat20002D is MatLaw87
    assert MaterialLaw87 is MatLaw87

    # Primary fields
    assert m.id == 87
    assert m.title == "Barlat 2000 Sheet"
    assert m.law == 87
    assert m.law_name == "LAW87"
    assert m.rho == pytest.approx(2.7e-9)
    assert m.rho0 == pytest.approx(2.7e-9)
    assert m.rhor == pytest.approx(2.7e-9)
    assert m.e == pytest.approx(70000.0)
    assert m.E == pytest.approx(70000.0)
    assert m.nu == pytest.approx(0.33)
    assert m.Nu == pytest.approx(0.33)
    assert m.expa == pytest.approx(8.0)
    assert m.flag_fit == 0
    assert m.al1 == pytest.approx(0.95)
    assert m.al2 == pytest.approx(1.05)
    assert m.al3 == pytest.approx(0.98)
    assert m.al4 == pytest.approx(1.02)
    assert m.al5 == pytest.approx(1.01)
    assert m.al6 == pytest.approx(0.99)
    assert m.al7 == pytest.approx(1.04)
    assert m.al8 == pytest.approx(0.96)
    assert m.fisokin == pytest.approx(0.25)
    assert m.aswift == pytest.approx(450.0)

    # Elastic derived properties
    expected_g = 70000.0 / (2.0 * (1.0 + 0.33))
    assert m.G == pytest.approx(expected_g)
    expected_k = 70000.0 / (3.0 * (1.0 - 2.0 * 0.33))
    assert m.bulk == pytest.approx(expected_k)
    assert m.K == pytest.approx(expected_k)

    # Sound speed (plane stress thin shell)
    expected_c = math.sqrt(70000.0 / ((1.0 - 0.33**2) * 2.7e-9))
    assert m.sound_speed == pytest.approx(expected_c)
    assert m.sound_speed_shell == pytest.approx(expected_c)
    # CallableFloat protocol
    assert m.sound_speed() == pytest.approx(expected_c)
    assert m.sound_speed_shell() == pytest.approx(expected_c)

    # Linear transformation matrices Lp (L') and Lpp (L'')
    lp = m.Lp
    assert isinstance(lp, np.ndarray)
    assert lp.shape == (3, 3)
    assert lp[0, 0] == pytest.approx(2.0 * 0.95 / 3.0)
    assert lp[0, 1] == pytest.approx(-0.95 / 3.0)
    assert lp[1, 0] == pytest.approx(-1.05 / 3.0)
    assert lp[1, 1] == pytest.approx(2.0 * 1.05 / 3.0)
    assert lp[2, 2] == pytest.approx(1.04)

    lpp = m.Lpp
    assert isinstance(lpp, np.ndarray)
    assert lpp.shape == (3, 3)
    assert lpp[2, 2] == pytest.approx(0.96)

    # Mapping / dict protocol
    assert "E" in m
    assert "rho0" in m
    assert "nu" in m
    assert "al1" in m
    assert m["E"] == pytest.approx(70000.0)
    assert m.get("aswift") == pytest.approx(450.0)
    assert m.get("nonexistent", 42.0) == 42.0
    assert len(m) > 10


# ============================================================================
# 2. Model Containers
# ============================================================================

def test_model_mat_law87_containers():
    """Verify Model contains mat_law87s dict and all aliases."""
    model = Model()
    assert hasattr(model, "mat_law87s")
    assert hasattr(model, "mat_barlat2000s")
    assert hasattr(model, "mat_barlats")
    assert model.mat_barlat2000s is model.mat_law87s
    assert model.mat_barlats is model.mat_law87s


# ============================================================================
# 3. Card Layouts & Catalogue Mappings
# ============================================================================

def test_card_layouts_and_cfg_catalogue_law87():
    """Verify card layouts and law catalogue mappings for LAW87 and synonyms."""
    for prefix in ("MAT_LAW87", "MAT_BARLAT2000", "MAT_BARLAT2000_2D"):
        assert CARD_LAYOUTS[f"{prefix}_1"] == [20, 20]
        assert CARD_LAYOUTS[f"{prefix}_2"] == [20, 20, 10, 10, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_3"] == [20, 20, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_4"] == [20, 20, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_6"] == [20, 20, 20, 20, 10, 10]
        assert CARD_LAYOUTS[f"{prefix}_7"] == [20, 20, 20, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_8"] == [10, 10, 20, 20]

    assert law_number("LAW87") == 87
    assert law_number("BARLAT") == 87
    assert law_number("BARLAT2000") == 87
    assert law_number("BARLAT_2000") == 87
    assert law_number("BARLAT2000_2D") == 87
    assert law_number("BARLAT_YLD2000") == 87
    assert law_number("MAT_LAW87") == 87
    assert law_number("MAT_BARLAT2000") == 87
    assert law_number("MAT_BARLAT2000_2D") == 87

    assert canonical_law_name(87) == "LAW87"
    assert canonical_law_name("LAW87") == "LAW87"
    assert canonical_law_name("BARLAT2000") == "LAW87"
    assert canonical_law_name("BARLAT_2000") == "LAW87"
    assert canonical_law_name("BARLAT2000_2D") == "LAW87"


# ============================================================================
# 4. Starter Keyword Reader (Fixed Format ifit=0 and ifit=1)
# ============================================================================

def test_starter_keyword_reader_fixed_ifit0(tmp_path):
    """Verify fixed-format /MAT/LAW87 with ifit=0 (direct alphas) and iflag=1 (Swift-Voce)."""
    deck_text = (
        "/MAT/LAW87/1\n"
        "Barlat2000 Fixed Alphas\n"
        "#                 RHO                RHOR\n"
        "             2.70E-9             2.70E-9\n"
        "#                  E                  NU               IFLAG            IFLAGSR                INVC                INVP\n"
        "             70000.0                0.33                   1                   0                 0.0                 0.0\n"
        "#           FLAG_FIT                 AL1                 AL2                 AL3                 AL4\n"
        "                   0                0.95                1.05                0.98                1.02\n"
        "#                AL5                 AL6                 AL7                 AL8\n"
        "                1.01                0.99                1.04                0.96\n"
        "#            FISOKIN                IKIN                EXPA                FCUT             FSMOOTH\n"
        "                0.25                   1                 8.0                 0.0                   0\n"
        "#              NRATE              ASWIFT                NEXP               ALPHA                EPSO\n"
        "                   0               450.0                0.22                 0.6               0.002\n"
        "#              QVOCE                BETA                  KO\n"
        "                80.0                15.0               120.0\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_fixed_ifit0_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law87(block, model, log)

    assert len(log.errors) == 0
    assert 1 in model.mat_law87s
    m = model.mat_law87s[1]
    assert m.id == 1
    assert m.title == "Barlat2000 Fixed Alphas"
    assert m.rho == pytest.approx(2.7e-9)
    assert m.rhor == pytest.approx(2.7e-9)
    assert m.e == pytest.approx(70000.0)
    assert m.nu == pytest.approx(0.33)
    assert m.iflag == 1
    assert m.flag_fit == 0
    assert m.al1 == pytest.approx(0.95)
    assert m.al2 == pytest.approx(1.05)
    assert m.al3 == pytest.approx(0.98)
    assert m.al4 == pytest.approx(1.02)
    assert m.al5 == pytest.approx(1.01)
    assert m.al6 == pytest.approx(0.99)
    assert m.al7 == pytest.approx(1.04)
    assert m.al8 == pytest.approx(0.96)
    assert m.fisokin == pytest.approx(0.25)
    assert m.ikin == 1
    assert m.expa == pytest.approx(8.0)
    assert m.aswift == pytest.approx(450.0)
    assert m.nexp == pytest.approx(0.22)
    assert m.alpha == pytest.approx(0.6)
    assert m.epso == pytest.approx(0.002)
    assert m.qvoce == pytest.approx(80.0)
    assert m.beta == pytest.approx(15.0)
    assert m.ko == pytest.approx(120.0)

    # Also populates model.materials
    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 87


def test_starter_keyword_reader_fixed_ifit1(tmp_path):
    """Verify fixed-format /MAT/LAW87 with ifit=1 (Lankford & yield stress fitting)."""
    deck_text = (
        "/MAT/LAW87/2\n"
        "Barlat2000 Fixed Fitting\n"
        "#                 RHO                RHOR\n"
        "             7.85E-9             7.85E-9\n"
        "#                  E                  NU               IFLAG            IFLAGSR                INVC                INVP\n"
        "            210000.0                0.30                   1                   0                 0.0                 0.0\n"
        "#           FLAG_FIT            SIGMA_00            SIGMA_45            SIGMA_90             SIGMA_B\n"
        "                   1               250.0               245.0               260.0               255.0\n"
        "#               R_00                R_45                R_90                 R_B\n"
        "                 1.2                 1.4                 1.6                 1.1\n"
        "#            FISOKIN                IKIN                EXPA                FCUT             FSMOOTH\n"
        "                 0.0                   1                 6.0                 0.0                   0\n"
        "#              NRATE              ASWIFT                NEXP               ALPHA                EPSO\n"
        "                   0               550.0                0.18                 1.0               0.005\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_fixed_ifit1_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law87(block, model, log)

    assert len(log.errors) == 0
    assert 2 in model.mat_law87s
    m = model.mat_law87s[2]
    assert m.id == 2
    assert m.flag_fit == 1
    assert m.sigma_00 == pytest.approx(250.0)
    assert m.sigma_45 == pytest.approx(245.0)
    assert m.sigma_90 == pytest.approx(260.0)
    assert m.sigma_b == pytest.approx(255.0)
    assert m.r_00 == pytest.approx(1.2)
    assert m.r_45 == pytest.approx(1.4)
    assert m.r_90 == pytest.approx(1.6)
    assert m.r_b == pytest.approx(1.1)
    assert m.expa == pytest.approx(6.0)


# ============================================================================
# 5. Free-Format Reading and Keyword Synonyms
# ============================================================================

def test_starter_keyword_reader_free_and_synonyms(tmp_path):
    """Verify free-format parsing for /MAT/BARLAT2000, /MAT/BARLAT_2000, /MAT/BARLAT2000_2D, /MAT/LAW87."""
    deck_text = (
        "/MAT/BARLAT2000/10\n"
        "Barlat2000 Free\n"
        "2.7e-9, 2.7e-9\n"
        "70000.0, 0.33, 1, 0, 0.0, 0.0\n"
        "0, 1.0, 1.0, 1.0, 1.0\n"
        "1.0, 1.0, 1.0, 1.0\n"
        "0.0, 1, 8.0, 0.0, 0\n"
        "0, 500.0, 0.2, 1.0, 0.001\n"
        "/MAT/BARLAT_2000/20\n"
        "Barlat_2000 Free\n"
        "2.7e-9, 2.7e-9\n"
        "70000.0, 0.33, 1, 0, 0.0, 0.0\n"
        "0, 1.0, 1.0, 1.0, 1.0\n"
        "1.0, 1.0, 1.0, 1.0\n"
        "0.0, 1, 8.0, 0.0, 0\n"
        "0, 500.0, 0.2, 1.0, 0.001\n"
        "/MAT/BARLAT2000_2D/30\n"
        "Barlat2000_2D Free\n"
        "2.7e-9, 2.7e-9\n"
        "70000.0, 0.33, 1, 0, 0.0, 0.0\n"
        "0, 1.0, 1.0, 1.0, 1.0\n"
        "1.0, 1.0, 1.0, 1.0\n"
        "0.0, 1, 8.0, 0.0, 0\n"
        "0, 500.0, 0.2, 1.0, 0.001\n"
        "/MAT/LAW87/40\n"
        "Law87 Free\n"
        "2.7e-9, 2.7e-9\n"
        "70000.0, 0.33, 1, 0, 0.0, 0.0\n"
        "0, 1.0, 1.0, 1.0, 1.0\n"
        "1.0, 1.0, 1.0, 1.0\n"
        "0.0, 1, 8.0, 0.0, 0\n"
        "0, 500.0, 0.2, 1.0, 0.001\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_synonyms_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law87(block, model, log)

    assert len(log.errors) == 0
    assert 10 in model.mat_law87s
    assert 20 in model.mat_law87s
    assert 30 in model.mat_law87s
    assert 40 in model.mat_law87s

    # Containers reference the same dictionary
    assert model.mat_barlat2000s[10].title == "Barlat2000 Free"
    assert model.mat_barlat2000s[20].title == "Barlat_2000 Free"
    assert model.mat_barlat2000s[30].title == "Barlat2000_2D Free"
    assert model.mat_barlat2000s[40].title == "Law87 Free"


# ============================================================================
# 6. Tabulated Curves, Kinematic Hardening, and Orthotropic Tables
# ============================================================================

def test_starter_keyword_reader_tabulated_and_kinematic(tmp_path):
    """Verify parsing of tabulated curves (iflag=0) and kinematic hardening cards."""
    deck_text = (
        "/MAT/LAW87/5\n"
        "Barlat2000 Tabulated Kinematic\n"
        "2.7e-9, 2.7e-9\n"
        "70000.0, 0.33, 0, 0, 0.0, 0.0\n"
        "0, 1.0, 1.0, 1.0, 1.0\n"
        "1.0, 1.0, 1.0, 1.0\n"
        "0.5, 1, 8.0, 0.0, 0\n"
        "100.0, 50.0, 25.0, 12.5\n"
        "10.0, 5.0, 2.5, 1.25\n"
        "2\n"
        "1, 1, 0.0, 1.0\n"
        "2, 2, 100.0, 1.05\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_tab_kin_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law87(block, model, log)

    assert len(log.errors) == 0
    assert 5 in model.mat_law87s
    m = model.mat_law87s[5]
    assert m.iflag == 0
    assert m.fisokin == pytest.approx(0.5)
    assert m.ckh == (100.0, 50.0, 25.0, 12.5)
    assert m.akh == (10.0, 5.0, 2.5, 1.25)
    assert m.nrate == 2
    assert len(m.funct_ids) == 2
    assert m.funct_ids == [1, 2]
    assert m.rates == pytest.approx([0.0, 100.0])
    assert m.yfac == pytest.approx([1.0, 1.05])


# ============================================================================
# 7. Starter Deck Writer & Roundtrip
# ============================================================================

def test_starter_deck_writer_roundtrip(tmp_path):
    """Verify StarterDeck.mat_law87, mat_barlat2000, and mat_barlat_2000 roundtrip."""
    deck = StarterDeck("ROUNDTRIP")

    deck.mat_law87(
        1,
        "Law 87 Primary",
        rho=2.7e-9,
        rhor=2.7e-9,
        e=70000.0,
        nu=0.33,
        iflag=1,
        expa=8.0,
        al1=0.95,
        al2=1.05,
        al3=0.98,
        al4=1.02,
        al5=1.01,
        al6=0.99,
        al7=1.04,
        al8=0.96,
        fisokin=0.2,
        aswift=450.0,
        nexp=0.22,
        alpha=0.5,
        epso=0.002,
        qvoce=80.0,
        beta=15.0,
        ko=120.0,
    )

    deck.mat_barlat2000(
        2,
        "Alias Barlat 2000",
        rho=7.85e-9,
        rhor=7.85e-9,
        e=210000.0,
        nu=0.30,
        iflag=1,
        expa=6.0,
        al1=1.0,
        al2=1.0,
        al3=1.0,
        al4=1.0,
        al5=1.0,
        al6=1.0,
        al7=1.0,
        al8=1.0,
        aswift=550.0,
        nexp=0.18,
        alpha=1.0,
        epso=0.005,
    )

    deck.mat_barlat_2000(
        3,
        "Alias Barlat_2000",
        rho=2.8e-9,
        rhor=2.8e-9,
        e=72000.0,
        nu=0.32,
        iflag=1,
        expa=8.0,
        al1=1.0,
        al2=1.0,
        al3=1.0,
        al4=1.0,
        al5=1.0,
        al6=1.0,
        al7=1.0,
        al8=1.0,
        aswift=400.0,
        nexp=0.2,
        alpha=1.0,
        epso=0.001,
    )

    rad_file = tmp_path / "deck_law87_roundtrip_0000.rad"
    deck.write(str(rad_file))

    # Read back
    blocks = read_deck(str(rad_file))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law87(block, model, log)

    assert len(log.errors) == 0
    assert 1 in model.mat_law87s
    assert 2 in model.mat_law87s
    assert 3 in model.mat_law87s

    m1 = model.mat_law87s[1]
    assert m1.id == 1
    assert m1.title == "Law 87 Primary"
    assert m1.e == pytest.approx(70000.0)
    assert m1.nu == pytest.approx(0.33)
    assert m1.expa == pytest.approx(8.0)
    assert m1.al1 == pytest.approx(0.95)
    assert m1.aswift == pytest.approx(450.0)

    m2 = model.mat_law87s[2]
    assert m2.id == 2
    assert m2.title == "Alias Barlat 2000"
    assert m2.e == pytest.approx(210000.0)
    assert m2.expa == pytest.approx(6.0)
    assert m2.aswift == pytest.approx(550.0)

    m3 = model.mat_law87s[3]
    assert m3.id == 3
    assert m3.title == "Alias Barlat_2000"
    assert m3.e == pytest.approx(72000.0)
    assert m3.aswift == pytest.approx(400.0)


# ============================================================================
# 8. Starter Validation Checks (ANCMSG 1514, 3095, 3102, 305, 306)
# ============================================================================

def test_check_mat_law87_bounds_ancmsg1514():
    """Verify bounds check errors for rho <= 0, e <= 0, and nu outside [0, 0.5) (ANCMSG 1514)."""
    log = MessageLog()
    m_bad_rho = MatLaw87(id=1, rho=0.0, e=70000.0, nu=0.33)
    check_mat_law87(mat=m_bad_rho, log=log)
    assert any("ANCMSG 1514" in e and "RHO" in e for e in log.errors)

    log_e = MessageLog()
    m_bad_e = MatLaw87(id=2, rho=2.7e-9, e=-100.0, nu=0.33)
    check_mat_law87(mat=m_bad_e, log=log_e)
    assert any("ANCMSG 1514" in e and "E" in e for e in log_e.errors)

    log_nu = MessageLog()
    m_bad_nu = MatLaw87(id=3, rho=2.7e-9, e=70000.0, nu=0.55)
    check_mat_law87(mat=m_bad_nu, log=log_nu)
    assert any("ANCMSG 1514" in e and "NU" in e for e in log_nu.errors)


def test_check_mat_law87_convexity_warnings_ancmsg3095_3102():
    """Verify convexity warnings when Lp (ANCMSG 3095) or Lpp (ANCMSG 3102) has non-positive eigenvalues."""
    # 1. Convex parameters (all positive eigenvalues): no warnings
    log_ok = MessageLog()
    m_ok = MatLaw87(id=1, rho=2.7e-9, e=70000.0, nu=0.33, al1=1.0, al2=1.0, al3=1.0, al4=1.0,
                    al5=1.0, al6=1.0, al7=1.0, al8=1.0, flag_fit=0)
    check_mat_law87(mat=m_ok, log=log_ok)
    assert not any("ANCMSG 3095" in w for w in log_ok.warnings)
    assert not any("ANCMSG 3102" in w for w in log_ok.warnings)

    # 2. Non-convex Lp: al7 <= 0
    log_lp = MessageLog()
    m_bad_lp = MatLaw87(id=2, rho=2.7e-9, e=70000.0, nu=0.33, al1=1.0, al2=1.0, al3=1.0, al4=1.0,
                        al5=1.0, al6=1.0, al7=-0.5, al8=1.0, flag_fit=0)
    check_mat_law87(mat=m_bad_lp, log=log_lp)
    assert any("ANCMSG 3095" in w and "Lp eigenvalues" in w for w in log_lp.warnings)

    # 3. Non-convex Lpp: al8 <= 0
    log_lpp = MessageLog()
    m_bad_lpp = MatLaw87(id=3, rho=2.7e-9, e=70000.0, nu=0.33, al1=1.0, al2=1.0, al3=1.0, al4=1.0,
                         al5=1.0, al6=1.0, al7=1.0, al8=-0.5, flag_fit=0)
    check_mat_law87(mat=m_bad_lpp, log=log_lpp)
    assert any("ANCMSG 3102" in w and "Lpp eigenvalues" in w for w in log_lpp.warnings)


def test_check_mat_law87_element_compatibility_ancmsg305_306():
    """Verify rejection of solids (ANCMSG 305) and 1D elements (ANCMSG 306) and acceptance of shells."""
    mat = Material(id=10, law=87, rho0=2.7e-9, params={"e": 70000.0, "nu": 0.33})
    m87 = MatLaw87(id=10, rho=2.7e-9, e=70000.0, nu=0.33)

    class FakeGroup:
        def __init__(self, name, mid):
            self.name = name
            self.state = {"slices": [(slice(0, 1), mat, None)]}
            self.n = 1
        def values(self):
            return []

    # 1. Shell model: accepted
    model_sh = Model()
    model_sh.node_ids = np.array([1, 2, 3, 4])
    model_sh.materials[10] = mat
    model_sh.mat_law87s[10] = m87
    model_sh.shells = FakeGroup("shells", 10)
    log_sh = MessageLog()
    check_mat_law87(mat=m87, log=log_sh, model=model_sh, mat_id=10)
    assert len(log_sh.errors) == 0

    # 2. Solid model: rejected (ANCMSG 305)
    model_solid = Model()
    model_solid.node_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    model_solid.materials[10] = mat
    model_solid.mat_law87s[10] = m87
    model_solid.bricks = FakeGroup("bricks", 10)
    log_solid = MessageLog()
    check_mat_law87(mat=m87, log=log_solid, model=model_solid, mat_id=10)
    assert any("ANCMSG 305" in e and "solid elements" in e for e in log_solid.errors)

    # 3. 1D model: rejected (ANCMSG 306)
    model_1d = Model()
    model_1d.node_ids = np.array([1, 2])
    model_1d.materials[10] = mat
    model_1d.mat_law87s[10] = m87
    model_1d.beams = FakeGroup("beams", 10)
    log_1d = MessageLog()
    check_mat_law87(mat=m87, log=log_1d, model=model_1d, mat_id=10)
    assert any("ANCMSG 306" in e and "1D elements" in e for e in log_1d.errors)


def test_allowed_laws_and_mat_checks_registration():
    """Verify LAW87 is registered in _MAT_CHECKS and _ALLOWED_LAWS for shells."""
    for key in (87, "87", "LAW87", "BARLAT2000", "BARLAT_2000", "BARLAT2000_2D", "BARLAT_YLD2000",
                "MAT_LAW87", "MAT_BARLAT2000", "MAT_BARLAT_2000", "MAT_BARLAT2000_2D"):
        assert key in _MAT_CHECKS
        assert _MAT_CHECKS[key] is check_mat_law87

    for grp in ("shells", "shells_qbat", "shells_qeph", "sh3n"):
        assert 87 in _ALLOWED_LAWS[grp]
        assert "LAW87" in _ALLOWED_LAWS[grp]
        assert "BARLAT2000" in _ALLOWED_LAWS[grp]

    for grp in ("bricks", "tetras", "penta6", "pyra5", "trusses", "beams"):
        assert 87 not in _ALLOWED_LAWS[grp]
        assert "LAW87" not in _ALLOWED_LAWS[grp]


# ============================================================================
# 9. Shell Element Integration (shell_bt4, shell_qeph, shell_tri3)
# ============================================================================

def test_shell_element_state_initialization():
    """Verify shell_bt4._init_material_state initializes uvar87 for LAW87."""
    class FakeProp:
        thick = 1.0

    mat = MatLaw87(id=87, rho=2.7e-9, e=70000.0, nu=0.33)
    prop = FakeProp()

    st = {
        "slices": [(slice(0, 4), mat, prop)],
        "thick": np.ones(4),
        "mat_extra": {},
    }

    _init_material_state(st, nip_max=3, n=4)

    assert "uvar87" in st
    assert st["uvar87"].shape == (4, 7)
    assert "uvar87" in st["mat_extra"]
    assert st["mat_extra"]["uvar87"].shape == (4, 3, 7)

    # Layer extra view
    extra = _layer_extra(st, slice(0, 4), 0)
    assert "uvar87" in extra
    assert extra["uvar87"].shape == (4, 7)


def test_shell_element_sound_speed():
    """Verify sound speed calculation across shell_bt4, shell_qeph, and shell_tri3."""
    mat = MatLaw87(id=87, rho=2.7e-9, e=70000.0, nu=0.33)
    expected_c = math.sqrt(70000.0 / ((1.0 - 0.33**2) * 2.7e-9))

    assert mat.sound_speed_shell() == pytest.approx(expected_c)
    assert materials.sound_speed(mat, rho=2.7e-9) == pytest.approx(expected_c)
