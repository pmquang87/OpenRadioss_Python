"""Tests for M556: /MAT/LAW21 and /MAT/DPRAG Starter & Model Integration.

Validates:
1. MatLaw21 (and MatDprag, MatDuckhub) dataclass fields, property helpers
   (rho0, rhor, G, sound_speed_solid, sound_speed, params), and mapping protocol
   (__getitem__, __contains__, get, keys).
2. Card layouts MAT_LAW21_1..6 and MAT_DPRAG_1..6 matching matl21_dprag.cfg.
3. CFG catalogue mapping (/MAT/LAW21, /MAT/DPRAG, DPRAG, MAT_DPRAG, LAW21_DPRAG, DUCKHUB -> 21; DPRAG1 -> 10).
4. Starter keyword reader for fixed and free formats, and default value rules per hm_read_mat21.F.
5. StarterDeck writer for 20-col fixed format and comma-delimited free format, with round-trip.
6. Starter checks: parameter bounds (rho, E, nu, C1), warnings (yield surface A1, A2, discriminant),
   element compatibility (solids permitted, shells ANCMSG 305, 1D ANCMSG 306, 2D analysis ANCMSG 305).
"""

from pathlib import Path
import math
import pytest

from pyradioss.model.entities import MatLaw21, MatDprag, MatDuckhub, Material
from pyradioss.model.model import Model
from pyradioss.input.card_layouts import CARD_LAYOUTS
from pyradioss.input.cfg_catalogue import law_number, canonical_law_name
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law21, KEYWORD_PARSERS
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.checks import check_mat_law21, check_model, _ALLOWED_LAWS, _MAT_CHECKS
from pyradioss.common.messages import MessageLog


# ============================================================================
# 1. Model Entities & Properties
# ============================================================================

def test_mat_law21_entities_and_properties():
    """Verify MatLaw21 dataclass fields, property helpers, and mapping protocol."""
    m = MatLaw21(
        id=5,
        rho=2.5e-3,
        refer_rho=2.55e-3,
        e=30000.0,
        nu=0.2,
        a0=10.0,
        a1=0.5,
        a2=0.01,
        amax=150.0,
        ifunc=3,
        c1=20000.0,
        pfscale=1.2,
        pmin=-100.0,
        pext=50.0,
        bunl=25000.0,
        mumax=0.08,
        title="Drucker-Prager Concrete",
    )

    # Aliases
    assert MatDprag is MatLaw21
    assert MatDuckhub is MatLaw21

    # Basic attributes & aliases
    assert m.id == 5
    assert m.rho == pytest.approx(2.5e-3)
    assert m.refer_rho == pytest.approx(2.55e-3)
    assert m.rho0 == pytest.approx(2.5e-3)
    assert m.rhor == pytest.approx(2.55e-3)
    assert m.E == pytest.approx(30000.0)
    assert m.Nu == pytest.approx(0.2)
    assert m.a0 == pytest.approx(10.0)
    assert m.a1 == pytest.approx(0.5)
    assert m.a2 == pytest.approx(0.01)
    assert m.amax == pytest.approx(150.0)
    assert m.ifunc == 3
    assert m.c1 == pytest.approx(20000.0)
    assert m.pfscale == pytest.approx(1.2)
    assert m.pmin == pytest.approx(-100.0)
    assert m.pext == pytest.approx(50.0)
    assert m.bunl == pytest.approx(25000.0)
    assert m.mumax == pytest.approx(0.08)

    # Shear modulus G = E / (2 * (1 + nu))
    expected_g = 30000.0 / (2.0 * (1.0 + 0.2))
    assert m.G == pytest.approx(expected_g)

    # Solid sound speed c = sqrt((C1 + 4/3*G) / rho0)
    expected_c_solid = math.sqrt((20000.0 + (4.0 / 3.0) * expected_g) / 2.5e-3)
    assert m.sound_speed_solid == pytest.approx(expected_c_solid)
    assert m.sound_speed_solid() == pytest.approx(expected_c_solid)

    # 1D bar sound speed c0 = sqrt(E / rho0) per matl21_dprag.cfg DRAWABLES SOUND_SPEED
    expected_c0 = math.sqrt(30000.0 / 2.5e-3)
    assert m.sound_speed == pytest.approx(expected_c0)
    assert m.sound_speed() == pytest.approx(expected_c0)

    # Setters
    m.E = 35000.0
    assert m.e == pytest.approx(35000.0)
    m.Nu = 0.25
    assert m.nu == pytest.approx(0.25)
    expected_g_new = 35000.0 / (2.0 * (1.0 + 0.25))
    assert m.G == pytest.approx(expected_g_new)

    # Mapping protocol
    assert m["id"] == 5
    assert m["E"] == pytest.approx(35000.0)
    assert m["c1"] == pytest.approx(20000.0)
    assert m["a1"] == pytest.approx(0.5)
    assert "bunl" in m
    assert "pfscale" in m
    assert m.get("pext") == pytest.approx(50.0)
    assert m.get("unknown_key", 999.0) == 999.0
    assert "rho" in m.keys()

    # Params dict
    p = m.params
    assert p["a0"] == pytest.approx(10.0)
    assert p["a1"] == pytest.approx(0.5)
    assert p["ifunc"] == 3
    assert p["c1"] == pytest.approx(20000.0)


def test_mat_law21_default_refer_rho():
    """Verify rhor falls back to rho0 when refer_rho is 0."""
    m = MatLaw21(id=1, rho=2.0e-3, refer_rho=0.0, e=10000.0, nu=0.3, c1=8000.0)
    assert m.rhor == pytest.approx(2.0e-3)


# ============================================================================
# 2. CFG Catalogue & Card Layouts
# ============================================================================

def test_cfg_catalogue_and_card_layouts():
    """Verify CFG catalogue mappings and CARD_LAYOUTS entries."""
    for syn in ("/MAT/LAW21", "LAW21", "/MAT/DPRAG", "DPRAG", "MAT_DPRAG", "LAW21_DPRAG", "DUCKHUB"):
        assert law_number(syn) == 21
        assert canonical_law_name(syn) == "LAW21"

    # DPRAG1 must remain LAW10
    assert law_number("/MAT/DPRAG1") == 10
    assert law_number("DPRAG1") == 10

    # Layout dimensions matching matl21_dprag.cfg
    assert CARD_LAYOUTS["MAT_LAW21_1"] == [20, 20]
    assert CARD_LAYOUTS["MAT_LAW21_2"] == [20, 20]
    assert CARD_LAYOUTS["MAT_LAW21_3"] == [20, 20, 20, 20]
    assert CARD_LAYOUTS["MAT_LAW21_4"] == [10, 10, 20, 20]
    assert CARD_LAYOUTS["MAT_LAW21_5"] == [20]
    assert CARD_LAYOUTS["MAT_LAW21_6"] == [20, 20]

    assert CARD_LAYOUTS["MAT_DPRAG_1"] == [20, 20]
    assert CARD_LAYOUTS["MAT_DPRAG_2"] == [20, 20]
    assert CARD_LAYOUTS["MAT_DPRAG_3"] == [20, 20, 20, 20]
    assert CARD_LAYOUTS["MAT_DPRAG_4"] == [10, 10, 20, 20]
    assert CARD_LAYOUTS["MAT_DPRAG_5"] == [20]
    assert CARD_LAYOUTS["MAT_DPRAG_6"] == [20, 20]


# ============================================================================
# 3. Starter Keyword Reader (Fixed & Free Formats)
# ============================================================================

def test_read_mat_law21_fixed_format(tmp_path: Path):
    """Test reading 6-card fixed-format /MAT/LAW21 deck."""
    deck_text = """\
#RADIOSS STARTER
/BEGIN
test_law21_fixed
                  90                   1
/MAT/LAW21/10
Drucker-Prager Fixed Test
#              RHO_I          Ref. dens.
              2.5E-3             2.55E-3
#                  E                  NU
             30000.0                 0.2
#                 a0                  a1                  a2                amax
                12.0                 0.6                0.02               180.0
#      ifunc                          C1             PFscale
           4                     22000.0                 1.5
#               Pmin
              -250.0
#               Bunl               Mumax
             26000.0                0.05
/END
"""
    f = tmp_path / "deck21_fixed.rad"
    f.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(f))
    model = Model()
    log = MessageLog()

    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law21(block, model, log)

    assert not log.has_errors
    assert 10 in model.mat_law21s
    m = model.mat_law21s[10]

    assert m.id == 10
    assert m.rho == pytest.approx(2.5e-3)
    assert m.refer_rho == pytest.approx(2.55e-3)
    assert m.e == pytest.approx(30000.0)
    assert m.nu == pytest.approx(0.2)
    assert m.a0 == pytest.approx(12.0)
    assert m.a1 == pytest.approx(0.6)
    assert m.a2 == pytest.approx(0.02)
    assert m.amax == pytest.approx(180.0)
    assert m.ifunc == 4
    assert m.c1 == pytest.approx(22000.0)
    assert m.pfscale == pytest.approx(1.5)
    assert m.pmin == pytest.approx(-250.0)
    assert m.bunl == pytest.approx(26000.0)
    assert m.mumax == pytest.approx(0.05)


def test_read_mat_law21_free_format_comma(tmp_path: Path):
    """Test reading comma-separated free-format /MAT/DPRAG deck."""
    deck_text = """\
#RADIOSS STARTER
/BEGIN
test_law21_free_comma
                  90                   1
/MAT/DPRAG/20
Drucker-Prager Free Comma
2.4e-3, 2.4e-3
25000.0, 0.18
8.0, 0.4, 0.005, 120.0
2, 18000.0, 1.0
-150.0
20000.0, 0.04
/END
"""
    f = tmp_path / "deck21_free_comma.rad"
    f.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(f))
    model = Model()
    log = MessageLog()

    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law21(block, model, log)

    assert not log.has_errors
    assert 20 in model.mat_law21s
    m = model.mat_law21s[20]

    assert m.id == 20
    assert m.rho == pytest.approx(2.4e-3)
    assert m.e == pytest.approx(25000.0)
    assert m.nu == pytest.approx(0.18)
    assert m.a0 == pytest.approx(8.0)
    assert m.a1 == pytest.approx(0.4)
    assert m.a2 == pytest.approx(0.005)
    assert m.amax == pytest.approx(120.0)
    assert m.ifunc == 2
    assert m.c1 == pytest.approx(18000.0)
    assert m.pfscale == pytest.approx(1.0)
    assert m.pmin == pytest.approx(-150.0)
    assert m.bunl == pytest.approx(20000.0)
    assert m.mumax == pytest.approx(0.04)


def test_read_mat_law21_free_format_spaces(tmp_path: Path):
    """Test reading space-separated free-format /MAT/LAW21 deck."""
    deck_text = """\
#RADIOSS STARTER
/BEGIN
test_law21_free_spaces
                  90                   1
/MAT/LAW21/30
Drucker-Prager Free Spaces
2.6e-3 2.6e-3
32000.0 0.22
15.0 0.7 0.03 200.0
5 24000.0 1.3
-300.0
28000.0 0.06
/END
"""
    f = tmp_path / "deck21_free_spaces.rad"
    f.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(f))
    model = Model()
    log = MessageLog()

    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law21(block, model, log)

    assert not log.has_errors
    assert 30 in model.mat_law21s
    m = model.mat_law21s[30]

    assert m.id == 30
    assert m.rho == pytest.approx(2.6e-3)
    assert m.e == pytest.approx(32000.0)
    assert m.c1 == pytest.approx(24000.0)
    assert m.ifunc == 5
    assert m.pmin == pytest.approx(-300.0)
    assert m.bunl == pytest.approx(28000.0)
    assert m.mumax == pytest.approx(0.06)


def test_read_mat_law21_defaults_injection(tmp_path: Path):
    """Test upstream default value injection (refer_rho=rho, pmin=-1e30, bunl=c1, amax=1e20, mumax=1e20, pfscale=1.0)."""
    deck_text = """\
#RADIOSS STARTER
/BEGIN
test_law21_defaults
                  90                   1
/MAT/LAW21/40
Minimal LAW21 Card
#              RHO_I          Ref. dens.
              2.3E-3
#                  E                  NU
             28000.0                 0.2
#                 a0                  a1                  a2                amax
                10.0                 0.5                 0.0
#      ifunc                          C1             PFscale
                                 21000.0
#               Pmin

#               Bunl               Mumax

/END
"""
    f = tmp_path / "deck21_defaults.rad"
    f.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(f))
    model = Model()
    log = MessageLog()

    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law21(block, model, log)

    assert not log.has_errors
    assert 40 in model.mat_law21s
    m = model.mat_law21s[40]

    assert m.rho == pytest.approx(2.3e-3)
    assert m.refer_rho == pytest.approx(2.3e-3)
    assert m.amax == pytest.approx(1.0e20)
    assert m.pfscale == pytest.approx(1.0)
    assert m.pmin == pytest.approx(-1.0e30)
    assert m.bunl == pytest.approx(21000.0)
    assert m.mumax == pytest.approx(1.0e20)


def test_keyword_parsers_registration():
    """Verify KEYWORD_PARSERS contains MAT/LAW21, MAT/DPRAG, and MAT/DUCKHUB."""
    assert "MAT/LAW21" in KEYWORD_PARSERS
    assert "MAT/DPRAG" in KEYWORD_PARSERS
    assert "MAT/DUCKHUB" in KEYWORD_PARSERS
    assert "LAW21" in KEYWORD_PARSERS
    assert "DPRAG" in KEYWORD_PARSERS


# ============================================================================
# 4. StarterDeck Writer & Round-Trip
# ============================================================================

def test_deck_writer_fixed_and_free(tmp_path: Path):
    """Test writing /MAT/LAW21 and /MAT/DPRAG with fixed and free formats, plus round-trip."""
    deck_fixed = StarterDeck("test_m556_fixed")
    deck_fixed.mat_law21(
        mid=1,
        title="Drucker-Prager Fixed",
        rho=2.5e-3,
        refer_rho=2.55e-3,
        e=30000.0,
        nu=0.2,
        a0=12.0,
        a1=0.6,
        a2=0.02,
        amax=180.0,
        ifunc=4,
        c1=22000.0,
        pfscale=1.5,
        pmin=-250.0,
        bunl=26000.0,
        mumax=0.05,
        fixed_format=True,
    )
    fixed_text = deck_fixed.render()
    assert "/MAT/LAW21/1" in fixed_text
    assert "Drucker-Prager Fixed" in fixed_text

    # Write to file and re-read
    f_fixed = tmp_path / "deck_writer_fixed.rad"
    f_fixed.write_text(fixed_text + "\n/END\n", encoding="utf-8")
    blocks = read_deck(str(f_fixed))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law21(block, model, log)

    assert not log.has_errors
    m = model.mat_law21s[1]
    assert m.rho == pytest.approx(2.5e-3)
    assert m.refer_rho == pytest.approx(2.55e-3)
    assert m.e == pytest.approx(30000.0)
    assert m.nu == pytest.approx(0.2)
    assert m.a0 == pytest.approx(12.0)
    assert m.a1 == pytest.approx(0.6)
    assert m.a2 == pytest.approx(0.02)
    assert m.amax == pytest.approx(180.0)
    assert m.ifunc == 4
    assert m.c1 == pytest.approx(22000.0)
    assert m.pfscale == pytest.approx(1.5)
    assert m.pmin == pytest.approx(-250.0)
    assert m.bunl == pytest.approx(26000.0)
    assert m.mumax == pytest.approx(0.05)

    # Free format via mat_dprag
    deck_free = StarterDeck("test_m556_free")
    deck_free.mat_dprag(
        mid=2,
        title="Drucker-Prager Free",
        rho=2.4e-3,
        e=25000.0,
        nu=0.18,
        a0=8.0,
        a1=0.4,
        a2=0.005,
        amax=120.0,
        c1=18000.0,
        fixed_format=False,
        comma=True,
    )
    free_text = deck_free.render()
    assert "/MAT/DPRAG/2" in free_text
    assert "8.0, 0.4, 0.005, 120.0" in free_text


# ============================================================================
# 5. Starter Checks and Diagnostics
# ============================================================================

def test_check_mat_law21_bounds():
    """Verify error detection on parameter bounds in check_mat_law21."""
    log = MessageLog()
    mat = MatLaw21(id=1, rho=2.5e-3, e=30000.0, nu=0.2, c1=20000.0, a0=10.0, a1=0.5, a2=0.01)
    check_mat_law21(mat=mat, log=log)
    assert not log.has_errors

    # 1. Density rho <= 0
    log_rho = MessageLog()
    mat_bad_rho = MatLaw21(id=1, rho=0.0, e=30000.0, nu=0.2, c1=20000.0)
    check_mat_law21(mat=mat_bad_rho, log=log_rho)
    assert log_rho.has_errors
    assert any("initial density RHO must be > 0" in m for m in log_rho.errors)

    # 2. Young's modulus E <= 0
    log_e = MessageLog()
    mat_bad_e = MatLaw21(id=2, rho=2.5e-3, e=-100.0, nu=0.2, c1=20000.0)
    check_mat_law21(mat=mat_bad_e, log=log_e)
    assert log_e.has_errors
    assert any("Young's modulus E must be > 0" in m for m in log_e.errors)

    # 3. Poisson's ratio nu < 0 or >= 0.5
    log_nu1 = MessageLog()
    mat_bad_nu1 = MatLaw21(id=3, rho=2.5e-3, e=30000.0, nu=-0.1, c1=20000.0)
    check_mat_law21(mat=mat_bad_nu1, log=log_nu1)
    assert log_nu1.has_errors
    assert any("0 <= NU < 0.5" in m for m in log_nu1.errors)

    log_nu2 = MessageLog()
    mat_bad_nu2 = MatLaw21(id=4, rho=2.5e-3, e=30000.0, nu=0.5, c1=20000.0)
    check_mat_law21(mat=mat_bad_nu2, log=log_nu2)
    assert log_nu2.has_errors

    # 4. Bulk modulus C1 <= 0 (ANCMSG 829 error)
    log_c1 = MessageLog()
    mat_bad_c1 = MatLaw21(id=5, rho=2.5e-3, e=30000.0, nu=0.2, c1=0.0)
    check_mat_law21(mat=mat_bad_c1, log=log_c1)
    assert log_c1.has_errors
    assert any("tensile bulk modulus C1 must be > 0" in m and "ANCMSG 829" in m for m in log_c1.errors)


def test_check_mat_law21_warnings():
    """Verify yield surface warning messages per hm_read_mat21.F."""
    # 1. Inverted yield surface: a1 < 0 and a2 == 0
    log1 = MessageLog()
    mat1 = MatLaw21(id=1, rho=2.5e-3, e=30000.0, nu=0.2, c1=20000.0, a0=10.0, a1=-0.5, a2=0.0)
    check_mat_law21(mat=mat1, log=log1)
    assert not log1.has_errors
    assert any("INVERTED YIELD SURFACE. CHECK A1 SIGN." in m and "ANCMSG 829" in m for m in log1.warnings)

    # 2. Untypical yield surface: a2 < 0
    log2 = MessageLog()
    mat2 = MatLaw21(id=2, rho=2.5e-3, e=30000.0, nu=0.2, c1=20000.0, a0=10.0, a1=0.5, a2=-0.01)
    check_mat_law21(mat=mat2, log=log2)
    assert not log2.has_errors
    assert any("UNTYPICAL YIELD SURFACE. CHECK A2 SIGN." in m and "ANCMSG 829" in m for m in log2.warnings)

    # 3. No root discriminant: a2 != 0 and a1^2 - 4*a0*a2 < 0
    # a1 = 1.0, a0 = 10.0, a2 = 1.0 -> delta = 1 - 40 = -39 < 0
    log3 = MessageLog()
    mat3 = MatLaw21(id=3, rho=2.5e-3, e=30000.0, nu=0.2, c1=20000.0, a0=10.0, a1=1.0, a2=1.0)
    check_mat_law21(mat=mat3, log=log3)
    assert not log3.has_errors
    assert any("YIELD SURFACE HAS NO ROOT." in m and "ANCMSG 829" in m for m in log3.warnings)


def test_check_mat_law21_element_compatibility():
    """Verify solid elements are supported while shell (ANCMSG 305) and 1D (ANCMSG 306) elements are rejected."""
    # Dispatch dictionary
    assert 21 in _MAT_CHECKS
    assert "LAW21" in _MAT_CHECKS
    assert "DPRAG" in _MAT_CHECKS
    assert "MAT_DPRAG" in _MAT_CHECKS
    assert "LAW21_DPRAG" in _MAT_CHECKS
    assert "DUCKHUB" in _MAT_CHECKS

    # _ALLOWED_LAWS permits LAW21 on solids only
    assert 21 in _ALLOWED_LAWS["bricks"]
    assert 21 in _ALLOWED_LAWS["tetras"]
    assert 21 in _ALLOWED_LAWS["penta6"]
    assert 21 in _ALLOWED_LAWS["pyra5"]
    assert 21 in _ALLOWED_LAWS["solids"]

    assert 21 not in _ALLOWED_LAWS["shells"]
    assert 21 not in _ALLOWED_LAWS["shells_qbat"]
    assert 21 not in _ALLOWED_LAWS["shells_qeph"]
    assert 21 not in _ALLOWED_LAWS["sh3n"]
    assert 21 not in _ALLOWED_LAWS["quads"]
    assert 21 not in _ALLOWED_LAWS["trusses"]
    assert 21 not in _ALLOWED_LAWS["beams"]

    # Incompatible element rejection via check_mat_law21
    class DummyElement:
        def __init__(self, mid):
            self.mat_id = mid

    class DummyGroup:
        def __init__(self, elements):
            self.els = {i: el for i, el in enumerate(elements)}
        def values(self):
            return self.els.values()

    class DummyModel:
        def __init__(self, groups, n2d=0):
            self.grps = groups
            self.n2d = n2d
        def element_groups(self):
            return self.grps

    # Test solid element acceptance
    model_solid = DummyModel([("bricks", DummyGroup([DummyElement(1)]))])
    mat21 = MatLaw21(id=1, rho=2.5e-3, e=30000.0, nu=0.2, c1=20000.0)
    log_solid = MessageLog()
    check_mat_law21(model=model_solid, mat=mat21, log=log_solid)
    assert not log_solid.has_errors

    # Test shell element rejection (ANCMSG 305)
    model_shell = DummyModel([("shells", DummyGroup([DummyElement(1)]))])
    log_shell = MessageLog()
    check_mat_law21(model=model_shell, mat=mat21, log=log_shell)
    assert log_shell.has_errors
    assert any("ANCMSG 305" in m for m in log_shell.errors)

    # Test 1D element rejection (ANCMSG 306)
    model_1d = DummyModel([("beams", DummyGroup([DummyElement(1)]))])
    log_1d = MessageLog()
    check_mat_law21(model=model_1d, mat=mat21, log=log_1d)
    assert log_1d.has_errors
    assert any("ANCMSG 306" in m for m in log_1d.errors)

    # Test 2D analysis rejection (N2D > 0, ANCMSG 305)
    model_2d = DummyModel([("bricks", DummyGroup([DummyElement(1)]))], n2d=1)
    log_2d = MessageLog()
    check_mat_law21(model=model_2d, mat=mat21, log=log_2d)
    assert log_2d.has_errors
    assert any("N2D > 0" in m and "ANCMSG 305" in m for m in log_2d.errors)
