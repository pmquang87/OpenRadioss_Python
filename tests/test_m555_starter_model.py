"""Tests for M555: /MAT/LAW57 and /MAT/BARLAT3 Starter & Model Integration.

Validates:
1. MatLaw57 and MatLaw57Curve dataclasses, properties (rho0, rhor, sound_speed_shell,
   E, G, Nu, epsp_max, params), and mapping protocol (__getitem__, __contains__, get).
2. Card layouts MAT_LAW57_1..5, MAT_LAW57_CURVE, MAT_BARLAT3_1..5, MAT_BARLAT3_CURVE.
3. CFG catalogue mapping (/MAT/LAW57, /MAT/BARLAT3, MAT_BARLAT3, LAW57_BARLAT3 -> 57).
4. Starter keyword reader for fixed and free formats, and default value rules per hm_read_mat57.F90.
5. StarterDeck writer for 20-col fixed format and comma-delimited free format.
6. Starter checks: parameter bounds (rho, E, nu, Lankford, m, tensile strains) and
   element compatibility (shells permitted, solid elements ANCMSG 305, 1D elements ANCMSG 306).
"""

from pathlib import Path
import math
import pytest

from pyradioss.model.entities import MatLaw57, MatLaw57Curve, MatBarlat3, Material
from pyradioss.model.model import Model
from pyradioss.input.card_layouts import CARD_LAYOUTS
from pyradioss.input.cfg_catalogue import law_number, canonical_law_name
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law57, KEYWORD_PARSERS
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.checks import check_mat_law57, check_model, _ALLOWED_LAWS, _MAT_CHECKS
from pyradioss.common.messages import MessageLog


# ============================================================================
# 1. Model Entities & Properties
# ============================================================================

def test_mat_law57_entities_and_properties():
    """Verify MatLaw57 dataclass fields, property helpers, and mapping protocol."""
    c1 = MatLaw57Curve(fct_id=101, fscale=1.05, eps=0.01)
    assert c1.func_id == 101
    assert c1.scale == 1.05
    assert c1.rate == 0.01
    assert c1["fct_id"] == 101
    assert c1["scale"] == 1.05
    assert c1.get("rate") == 0.01

    m = MatLaw57(
        id=7,
        rho=7.8e-3,
        refer_rho=7.85e-3,
        e=210000.0,
        nu=0.3,
        ifunce=12,
        einf=190000.0,
        ce=15.0,
        r00=1.2,
        r45=1.1,
        r90=1.4,
        chard=250.0,
        m=6.0,
        eps_max=0.35,
        eps_t1=0.40,
        eps_t2=0.45,
        fcut=1000.0,
        fsmooth=1,
        vp=1,
        curves=[c1],
        title="Barlat Test Material",
    )

    # Basic aliases and properties
    assert MatBarlat3 is MatLaw57
    assert m.rho0 == pytest.approx(7.8e-3)
    assert m.rhor == pytest.approx(7.85e-3)
    assert m.E == pytest.approx(210000.0)
    assert m.Nu == pytest.approx(0.3)
    assert m.epsp_max == pytest.approx(0.35)

    # Shear modulus G = E / (2*(1+nu))
    expected_g = 210000.0 / (2.0 * (1.0 + 0.3))
    assert m.G == pytest.approx(expected_g)

    # Shell plane-stress sound speed c = sqrt(E / (rho0 * (1 - nu^2)))
    expected_c = math.sqrt(210000.0 / (7.8e-3 * (1.0 - 0.3**2)))
    assert m.sound_speed_shell == pytest.approx(expected_c)
    assert m.sound_speed_shell() == pytest.approx(expected_c)
    assert m.sound_speed == pytest.approx(expected_c)

    # Setters
    m.E = 200000.0
    assert m.e == pytest.approx(200000.0)
    m.Nu = 0.29
    assert m.nu == pytest.approx(0.29)
    m.epsp_max = 0.5
    assert m.eps_max == pytest.approx(0.5)

    # Mapping protocol
    assert m["id"] == 7
    assert m["E"] == pytest.approx(200000.0)
    assert m["r00"] == pytest.approx(1.2)
    assert "r45" in m
    assert "curves" in m
    assert m.get("chard") == pytest.approx(250.0)
    assert m.get("unknown_key", "default") == "default"

    # Params dict
    p = m.params
    assert p["r90"] == pytest.approx(1.4)
    assert p["ifunce"] == 12
    assert p["vp"] == 1


# ============================================================================
# 2. CFG Catalogue & Card Layouts
# ============================================================================

def test_cfg_catalogue_and_card_layouts():
    """Verify CFG catalogue mappings and CARD_LAYOUTS entries."""
    for syn in ("/MAT/LAW57", "LAW57", "/MAT/BARLAT3", "BARLAT3", "MAT_BARLAT3", "LAW57_BARLAT3"):
        assert law_number(syn) == 57
        assert canonical_law_name(syn) == "LAW57"

    assert CARD_LAYOUTS["MAT_LAW57_1"] == [20, 20]
    assert CARD_LAYOUTS["MAT_LAW57_2"] == [20, 20]
    assert CARD_LAYOUTS["MAT_LAW57_3"] == [10, 10, 20, 20]
    assert CARD_LAYOUTS["MAT_LAW57_4"] == [20, 20, 20, 20, 20]
    assert CARD_LAYOUTS["MAT_LAW57_5"] == [20, 20, 20, 20, 10, 10]
    assert CARD_LAYOUTS["MAT_LAW57_CURVE"] == [10, 10, 20, 20]

    assert CARD_LAYOUTS["MAT_BARLAT3_1"] == [20, 20]
    assert CARD_LAYOUTS["MAT_BARLAT3_2"] == [20, 20]
    assert CARD_LAYOUTS["MAT_BARLAT3_3"] == [10, 10, 20, 20]
    assert CARD_LAYOUTS["MAT_BARLAT3_4"] == [20, 20, 20, 20, 20]
    assert CARD_LAYOUTS["MAT_BARLAT3_5"] == [20, 20, 20, 20, 10, 10]
    assert CARD_LAYOUTS["MAT_BARLAT3_CURVE"] == [10, 10, 20, 20]


# ============================================================================
# 3. Starter Keyword Reader (Fixed & Free Formats)
# ============================================================================

def test_read_mat_law57_fixed_5card(tmp_path: Path):
    """Test reading 5-card fixed-format /MAT/LAW57 deck."""
    deck_text = """\
#RADIOSS STARTER
/BEGIN
test_law57_5card_fixed
                  90                   1
/MAT/LAW57/10
Barlat 3 5-Card Fixed
#              RHO_I          Ref. dens.
              7.8E-3              7.8E-3
#                  E                  NU
            210000.0                 0.3
#FUNCT_IDE                          EINF                  CE
        12                      195000.0                10.0
#                r00                 r45                 r90              C_hard                   m
                 1.3                 1.1                 1.5               350.0                 6.0
#           EPSP_max               EPS_T               EPS_M                Fcut   Fsmooth        VP
                 0.4                 0.5                 0.6              2500.0         1         1
# funct_ID                      Fscale_i               EPS_i
        21                           1.0                 0.0
        22                           1.2                50.0
/END
"""
    f = tmp_path / "deck57_fixed.rad"
    f.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(f))
    model = Model()
    log = MessageLog()

    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law57(block, model, log)

    assert not log.has_errors
    assert 10 in model.mat_law57s
    m = model.mat_law57s[10]

    assert m.id == 10
    assert m.rho == pytest.approx(7.8e-3)
    assert m.refer_rho == pytest.approx(7.8e-3)
    assert m.e == pytest.approx(210000.0)
    assert m.nu == pytest.approx(0.3)
    assert m.ifunce == 12
    assert m.einf == pytest.approx(195000.0)
    assert m.ce == pytest.approx(10.0)
    assert m.r00 == pytest.approx(1.3)
    assert m.r45 == pytest.approx(1.1)
    assert m.r90 == pytest.approx(1.5)
    assert m.chard == pytest.approx(350.0)
    assert m.m == pytest.approx(6.0)
    assert m.eps_max == pytest.approx(0.4)
    assert m.eps_t1 == pytest.approx(0.5)
    assert m.eps_t2 == pytest.approx(0.6)
    assert m.fcut == pytest.approx(2500.0)
    assert m.fsmooth == 1
    assert m.vp == 1
    assert len(m.curves) == 2
    assert m.curves[0].fct_id == 21
    assert m.curves[1].fct_id == 22
    assert m.curves[1].fscale == pytest.approx(1.2)
    assert m.curves[1].eps == pytest.approx(50.0)

    # Check model.materials
    assert 10 in model.materials
    mat = model.materials[10]
    assert mat.law == 57
    assert mat.params["m"] == pytest.approx(6.0)


def test_read_mat_law57_free_format(tmp_path: Path):
    """Test reading free-format /MAT/BARLAT3 with comma delimiters."""
    deck_text = """\
/BEGIN
test_law57_free
/MAT/BARLAT3/20
Barlat 3 Free Comma
2.7E-3, 2.7E-3
70000.0, 0.33
0, 0.0, 0.0
0.9, 1.0, 1.2, 400.0, 8.0
0.25, 0.35, 0.45, 1.0E30, 0, 0
101, 1.0, 0.0
102, 1.15, 100.0
/END
"""
    f = tmp_path / "deck57_free.rad"
    f.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(f))
    model = Model()
    log = MessageLog()

    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law57(block, model, log)

    assert not log.has_errors
    assert 20 in model.mat_law57s
    m = model.mat_law57s[20]

    assert m.rho == pytest.approx(2.7e-3)
    assert m.e == pytest.approx(70000.0)
    assert m.nu == pytest.approx(0.33)
    assert m.r00 == pytest.approx(0.9)
    assert m.r45 == pytest.approx(1.0)
    assert m.r90 == pytest.approx(1.2)
    assert m.chard == pytest.approx(400.0)
    assert m.m == pytest.approx(8.0)
    assert m.eps_max == pytest.approx(0.25)
    assert len(m.curves) == 2
    assert m.curves[1].fct_id == 102
    assert m.curves[1].fscale == pytest.approx(1.15)


def test_read_mat_law57_defaults(tmp_path: Path):
    """Verify Fortran-faithful defaults per hm_read_mat57.F90."""
    deck_text = """\
/BEGIN
test_law57_defaults
/MAT/LAW57/30
Barlat Defaults
7.8E-3, 0.0
210000.0, 0.3
0, 0.0, 0.0
0.0, 0.0, 0.0, 0.0, 0.0
0.0, 0.0, 0.0, 0.0, 0, 5
501, 1.0, 0.0
/END
"""
    f = tmp_path / "deck57_defaults.rad"
    f.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(f))
    model = Model()
    log = MessageLog()

    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law57(block, model, log)

    assert not log.has_errors
    m = model.mat_law57s[30]
    # if refer_rho == 0 -> rho
    assert m.refer_rho == pytest.approx(7.8e-3)
    # if r00, r45, r90 == 0 -> 1.0
    assert m.r00 == pytest.approx(1.0)
    assert m.r45 == pytest.approx(1.0)
    assert m.r90 == pytest.approx(1.0)
    # if m == 0 -> 6.0
    assert m.m == pytest.approx(6.0)
    # if eps_max == 0 -> 1e30
    assert m.eps_max == pytest.approx(1.0e30)
    # if eps_t1 == 0 -> 1e30, eps_t2 == 0 -> 2e30
    assert m.eps_t1 == pytest.approx(1.0e30)
    assert m.eps_t2 == pytest.approx(2.0e30)
    # if fcut <= 0 -> 1e30
    assert m.fcut == pytest.approx(1.0e30)
    # vp clamped to [0, 1]
    assert m.vp == 1


# ============================================================================
# 4. Deck Writer (StarterDeck) & Round-Trip
# ============================================================================

def test_deck_writer_fixed_and_free(tmp_path: Path):
    """Test StarterDeck.mat_law57 and mat_barlat3 emission in fixed and free formats."""
    curves = [
        MatLaw57Curve(fct_id=1, fscale=1.0, eps=0.0),
        {"fct_id": 2, "fscale": 1.1, "eps": 10.0},
    ]

    deck = StarterDeck("test_m555_fixed")
    deck.mat_law57(
        mid=1,
        title="Barlat 57 Fixed",
        rho=7.85e-3,
        refer_rho=7.85e-3,
        e=210000.0,
        nu=0.3,
        ifunce=10,
        einf=180000.0,
        ce=20.0,
        r00=1.2,
        r45=1.1,
        r90=1.3,
        chard=300.0,
        m=6.0,
        eps_max=0.4,
        eps_t1=0.5,
        eps_t2=0.6,
        fcut=5000.0,
        fsmooth=1,
        vp=1,
        curves=curves,
        fixed_format=True,
    )
    deck_text = deck.render()
    assert "/MAT/LAW57/1" in deck_text
    assert "Barlat 57 Fixed" in deck_text

    # Write and re-parse
    f_fixed = tmp_path / "writer_fixed.rad"
    full_deck = "/BEGIN\ntest_roundtrip\n                  90                   1\n" + deck_text + "\n/END\n"
    f_fixed.write_text(full_deck, encoding="utf-8")

    blocks = read_deck(str(f_fixed))
    model = Model()
    log = MessageLog()
    for b in blocks:
        if b.key0 == "MAT":
            read_mat_law57(b, model, log)

    assert not log.has_errors
    m = model.mat_law57s[1]
    assert m.rho == pytest.approx(7.85e-3)
    assert m.e == pytest.approx(210000.0)
    assert m.ifunce == 10
    assert m.einf == pytest.approx(180000.0)
    assert m.r00 == pytest.approx(1.2)
    assert m.m == pytest.approx(6.0)
    assert m.fcut == pytest.approx(5000.0)
    assert len(m.curves) == 2
    assert m.curves[1].fct_id == 2
    assert m.curves[1].fscale == pytest.approx(1.1)

    # Free format via mat_barlat3
    deck_free = StarterDeck("test_m555_free")
    deck_free.mat_barlat3(
        mid=2,
        title="Barlat 3 Free",
        rho=2.7e-3,
        e=70000.0,
        nu=0.33,
        r00=0.8,
        r45=0.9,
        r90=1.1,
        m=8.0,
        curves=[(11, 1.0, 0.0)],
        fixed_format=False,
    )
    free_text = deck_free.render()
    assert "/MAT/BARLAT3/2" in free_text
    assert "0.8, 0.9, 1.1" in free_text


# ============================================================================
# 5. Starter Checks and Diagnostics
# ============================================================================

def test_check_mat_law57_bounds():
    """Verify error detection on parameter bounds in check_mat_law57."""
    log = MessageLog()
    mat = MatLaw57(id=1, rho=7.8e-3, e=210000.0, nu=0.3, r00=1.0, r45=1.0, r90=1.0, m=6.0)
    check_mat_law57(mat=mat, log=log)
    assert not log.has_errors

    # 1. Density rho <= 0
    log_rho = MessageLog()
    mat_bad_rho = MatLaw57(id=1, rho=0.0, e=210000.0, nu=0.3)
    check_mat_law57(mat=mat_bad_rho, log=log_rho)
    assert log_rho.has_errors
    assert any("initial density RHO must be > 0" in m for m in log_rho.errors)

    # 2. Young's modulus E <= 0
    log_e = MessageLog()
    mat_bad_e = MatLaw57(id=2, rho=7.8e-3, e=-100.0, nu=0.3)
    check_mat_law57(mat=mat_bad_e, log=log_e)
    assert log_e.has_errors
    assert any("Young's modulus E must be > 0" in m for m in log_e.errors)

    # 3. Poisson's ratio nu < 0 or >= 0.5
    log_nu1 = MessageLog()
    mat_bad_nu1 = MatLaw57(id=3, rho=7.8e-3, e=210000.0, nu=-0.1)
    check_mat_law57(mat=mat_bad_nu1, log=log_nu1)
    assert log_nu1.has_errors
    assert any("0 <= NU < 0.5" in m for m in log_nu1.errors)

    log_nu2 = MessageLog()
    mat_bad_nu2 = MatLaw57(id=4, rho=7.8e-3, e=210000.0, nu=0.5)
    check_mat_law57(mat=mat_bad_nu2, log=log_nu2)
    assert log_nu2.has_errors

    # 4. Lankford coefficients <= 0
    log_r = MessageLog()
    mat_bad_r = MatLaw57(id=5, rho=7.8e-3, e=210000.0, nu=0.3, r00=0.0, r45=-0.5, r90=0.0)
    check_mat_law57(mat=mat_bad_r, log=log_r)
    assert log_r.has_errors
    assert any("Lankford parameter R00 must be > 0" in m for m in log_r.errors)
    assert any("Lankford parameter R45 must be > 0" in m for m in log_r.errors)
    assert any("Lankford parameter R90 must be > 0" in m for m in log_r.errors)

    # 5. Exponent m < 1.0
    log_m = MessageLog()
    mat_bad_m = MatLaw57(id=6, rho=7.8e-3, e=210000.0, nu=0.3, m=0.5)
    check_mat_law57(mat=mat_bad_m, log=log_m)
    assert log_m.has_errors
    assert any("Barlat exponent m must be >= 1.0" in m for m in log_m.errors)

    # 6. Tensile failure strains eps_t2 <= eps_t1 (when eps_t1 < 1e20)
    log_epst = MessageLog()
    mat_bad_epst = MatLaw57(id=7, rho=7.8e-3, e=210000.0, nu=0.3, eps_t1=0.5, eps_t2=0.4)
    check_mat_law57(mat=mat_bad_epst, log=log_epst)
    assert log_epst.has_errors
    assert any("EPS_t2 must be > EPS_t1" in m for m in log_epst.errors)


def test_check_mat_law57_element_compatibility():
    """Verify shell elements are supported while solid (ANCMSG 305) and 1D (ANCMSG 306) elements are rejected."""
    # Dispatch dictionary
    assert 57 in _MAT_CHECKS
    assert "LAW57" in _MAT_CHECKS
    assert "BARLAT3" in _MAT_CHECKS

    # _ALLOWED_LAWS permits LAW57 on shells only
    assert 57 in _ALLOWED_LAWS["shells"]
    assert 57 in _ALLOWED_LAWS["shells_qbat"]
    assert 57 in _ALLOWED_LAWS["shells_qeph"]
    assert 57 in _ALLOWED_LAWS["sh3n"]
    assert 57 in _ALLOWED_LAWS["quads"]

    assert 57 not in _ALLOWED_LAWS["bricks"]
    assert 57 not in _ALLOWED_LAWS["tetras"]
    assert 57 not in _ALLOWED_LAWS["trusses"]
    assert 57 not in _ALLOWED_LAWS["beams"]

    # Incompatible element rejection via check_mat_law57
    class DummyElement:
        def __init__(self, mid):
            self.mat_id = mid

    class DummyGroup:
        def __init__(self, elements):
            self.els = {i: el for i, el in enumerate(elements)}
        def values(self):
            return self.els.values()

    class DummyModel:
        def __init__(self, groups):
            self.grps = groups
        def element_groups(self):
            return self.grps

    # Test solid element rejection (ANCMSG 305)
    model_solid = DummyModel([("bricks", DummyGroup([DummyElement(1)]))])
    mat57 = MatLaw57(id=1, rho=7.8e-3, e=210000.0, nu=0.3)
    log_solid = MessageLog()
    check_mat_law57(model=model_solid, mat=mat57, log=log_solid)
    assert log_solid.has_errors
    assert any("ANCMSG 305" in m for m in log_solid.errors)

    # Test 1D element rejection (ANCMSG 306)
    model_1d = DummyModel([("beams", DummyGroup([DummyElement(1)]))])
    log_1d = MessageLog()
    check_mat_law57(model=model_1d, mat=mat57, log=log_1d)
    assert log_1d.has_errors
    assert any("ANCMSG 306" in m for m in log_1d.errors)

    # Test shell element acceptance
    model_shell = DummyModel([("shells", DummyGroup([DummyElement(1)]))])
    log_shell = MessageLog()
    check_mat_law57(model=model_shell, mat=mat57, log=log_shell)
    assert not log_shell.has_errors
