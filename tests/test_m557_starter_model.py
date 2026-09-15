"""Tests for M557: /MAT/LAW49, /MAT/STEINB, /MAT/STEINBERG Starter & Model Integration.

Validates:
1. MatLaw49 (and MatSteinb, MatSteinberg, MatSteinbergGuinan) dataclass fields,
   property helpers (rho0, rhor, G, G0, bulk, C1, sound_speed, sound_speed_solid, params),
   and mapping protocol (__getitem__, __contains__, get, keys).
2. Card layouts MAT_LAW49_1..5, MAT_STEINB_1..5, MAT_STEINBERG_1..5, MAT_STEINBERG_GUINAN_1..5
   matching matl49_steinb.cfg.
3. CFG catalogue mapping (/MAT/LAW49, /MAT/STEINB, STEINBERG, STEINBERG_GUINAN, MAT_STEINB, etc. -> 49).
4. Starter keyword reader for fixed and free formats, and default value rules per hm_read_mat49.F.
5. StarterDeck writer for 20-col fixed format and comma-delimited free format, with round-trip.
6. Starter checks: parameter bounds (rho, E, nu, f), element compatibility
   (solids permitted, shells ANCMSG 305, 1D ANCMSG 306, 2D analysis ANCMSG 305).
"""

from pathlib import Path
import math
import pytest

from pyradioss.model.entities import MatLaw49, MatSteinb, MatSteinberg, MatSteinbergGuinan, Material
from pyradioss.model.model import Model
from pyradioss.input.card_layouts import CARD_LAYOUTS
from pyradioss.input.cfg_catalogue import law_number, canonical_law_name
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law49, KEYWORD_PARSERS
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.checks import check_mat_law49, check_model, _ALLOWED_LAWS, _MAT_CHECKS
from pyradioss.common.messages import MessageLog


# ============================================================================
# 1. Model Entities & Properties
# ============================================================================

def test_mat_law49_entities_and_properties():
    """Verify MatLaw49 dataclass fields, property helpers, and mapping protocol."""
    m = MatLaw49(
        id=7,
        rho=8.96,
        refer_rho=8.96,
        e0=1.24e5,
        nu=0.34,
        sigy=120.0,
        beta=36.0,
        n=0.45,
        eps_max=1e20,
        sigma_max=640.0,
        t0=300.0,
        tmelt=1356.0,
        rhoc_p=3.45e3,
        pmin=-1e20,
        b1=2.8e-5,
        b2=2.8e-5,
        h=3.8e-4,
        f=0.5,
        title="Steinberg Copper",
    )

    # Aliases
    assert MatSteinb is MatLaw49
    assert MatSteinberg is MatLaw49
    assert MatSteinbergGuinan is MatLaw49

    # Attributes & aliases
    assert m.id == 7
    assert m.rho == pytest.approx(8.96)
    assert m.refer_rho == pytest.approx(8.96)
    assert m.rho0 == pytest.approx(8.96)
    assert m.rhor == pytest.approx(8.96)
    assert m.E == pytest.approx(1.24e5)
    assert m.e == pytest.approx(1.24e5)
    assert m.e0 == pytest.approx(1.24e5)
    assert m.nu == pytest.approx(0.34)
    assert m.Nu == pytest.approx(0.34)
    assert m.sigy == pytest.approx(120.0)
    assert m.sigma_0 == pytest.approx(120.0)
    assert m.beta == pytest.approx(36.0)
    assert m.n == pytest.approx(0.45)
    assert m.hard == pytest.approx(0.45)
    assert m.sigma_max == pytest.approx(640.0)
    assert m.t0 == pytest.approx(300.0)
    assert m.tmelt == pytest.approx(1356.0)
    assert m.rhoc_p == pytest.approx(3.45e3)
    assert m.pmin == pytest.approx(-1e20)
    assert m.b1 == pytest.approx(2.8e-5)
    assert m.b2 == pytest.approx(2.8e-5)
    assert m.h == pytest.approx(3.8e-4)
    assert m.f == pytest.approx(0.5)

    # Derived shear modulus G0 = E0 / (2 * (1 + nu))
    expected_g0 = 1.24e5 / (2.0 * (1.0 + 0.34))
    assert m.G == pytest.approx(expected_g0)
    assert m.G0 == pytest.approx(expected_g0)

    # Derived bulk modulus K = E0 / (3 * (1 - 2*nu))
    expected_bulk = 1.24e5 / (3.0 * (1.0 - 2.0 * 0.34))
    assert m.bulk == pytest.approx(expected_bulk)
    assert m.C1 == pytest.approx(expected_bulk)

    # Sound speeds
    expected_c_solid = math.sqrt((expected_bulk + (4.0 / 3.0) * expected_g0) / 8.96)
    assert m.sound_speed_solid == pytest.approx(expected_c_solid)
    assert m.sound_speed_solid() == pytest.approx(expected_c_solid)

    expected_c0 = math.sqrt(1.24e5 / 8.96)
    assert m.sound_speed == pytest.approx(expected_c0)
    assert m.sound_speed() == pytest.approx(expected_c0)

    # Setters
    m.E = 1.3e5
    assert m.e0 == pytest.approx(1.3e5)
    m.Nu = 0.35
    assert m.nu == pytest.approx(0.35)
    m.sigma_0 = 130.0
    assert m.sigy == pytest.approx(130.0)
    m.hard = 0.5
    assert m.n == pytest.approx(0.5)

    # Mapping protocol
    assert "rho" in m
    assert "e0" in m
    assert "nu" in m
    assert "sigy" in m
    assert "G" in m
    assert "bulk" in m
    assert m["rho"] == pytest.approx(8.96)
    assert m.get("b1") == pytest.approx(2.8e-5)
    assert m.get("nonexistent", 99.0) == 99.0


def test_model_mat_law49_containers():
    """Verify Model contains mat_law49s dict and aliases."""
    model = Model()
    assert hasattr(model, "mat_law49s")
    assert hasattr(model, "mat_steinbs")
    assert hasattr(model, "mat_steinbergs")
    assert hasattr(model, "mat_steinberg_guinans")
    assert model.mat_steinbs is model.mat_law49s
    assert model.mat_steinbergs is model.mat_law49s
    assert model.mat_steinberg_guinans is model.mat_law49s


# ============================================================================
# 2. Card Layouts & Catalogue Mappings
# ============================================================================

def test_card_layouts_law49():
    """Verify card layouts MAT_LAW49_1..5 and alias layouts."""
    for prefix in ("MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "MAT_STEINBERG_GUINAN"):
        layout1 = CARD_LAYOUTS[f"{prefix}_1"]
        layout2 = CARD_LAYOUTS[f"{prefix}_2"]
        layout3 = CARD_LAYOUTS[f"{prefix}_3"]
        layout4 = CARD_LAYOUTS[f"{prefix}_4"]
        layout5 = CARD_LAYOUTS[f"{prefix}_5"]

        assert list(layout1) == [20, 20]
        assert list(layout2) == [20, 20]
        assert list(layout3) == [20, 20, 20, 20, 20]
        assert list(layout4) == [20, 20, 20, 20]
        assert list(layout5) == [20, 20, 20, 20]


def test_cfg_catalogue_law49():
    """Verify CFG catalogue law_number and canonical_law_name mappings."""
    for synonym in (
        "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN",
        "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "MAT_STEINBERG_GUINAN", "LAW49_STEINB",
    ):
        assert law_number(synonym) == 49
        assert canonical_law_name(synonym) == "LAW49"


def test_keyword_parsers_registered():
    """Verify all LAW49 keyword synonyms are registered in KEYWORD_PARSERS."""
    for kw in (
        "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN",
        "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "MAT_STEINBERG_GUINAN",
        "MAT/LAW49", "MAT/STEINB", "MAT/STEINBERG", "MAT/STEINBERG_GUINAN",
        "LAW49_STEINB",
    ):
        assert kw in KEYWORD_PARSERS
        assert KEYWORD_PARSERS[kw] == read_mat_law49


# ============================================================================
# 3. Starter Keyword Reader
# ============================================================================

def test_read_mat_law49_fixed_format(tmp_path: Path):
    """Test parsing 20-col fixed format /MAT/LAW49 card."""
    deck_text = (
        "/BEGIN\n"
        "test_law49_fixed\n"
        "/MAT/LAW49/101\n"
        "Copper OFHC Shock\n"
        "#                 rho            rhor\n"
        "                8.96                8.96\n"
        "#                  E                  nu\n"
        "             1.24E+5                0.34\n"
        "#              sigma            beta                   n              epsm            sigmam\n"
        "               120.0                36.0                0.45               1E+20               640.0\n"
        "#                 T0               Tmelt                 sph                pmin\n"
        "               300.0              1356.0              3.45E3              -1E+20\n"
        "#                 b1                  b2                   h                   f\n"
        "              2.8E-5              2.8E-5              3.8E-4                 0.5\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_fixed_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")

    blocks = read_deck(str(deck_file))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law49(block, model, log)

    assert not log.has_errors
    assert 101 in model.mat_law49s
    assert 101 in model.materials

    m = model.mat_law49s[101]
    assert m.id == 101
    assert m.title == "Copper OFHC Shock"
    assert m.rho == pytest.approx(8.96)
    assert m.refer_rho == pytest.approx(8.96)
    assert m.e0 == pytest.approx(1.24e5)
    assert m.nu == pytest.approx(0.34)
    assert m.sigy == pytest.approx(120.0)
    assert m.beta == pytest.approx(36.0)
    assert m.n == pytest.approx(0.45)
    assert m.eps_max == pytest.approx(1e20)
    assert m.sigma_max == pytest.approx(640.0)
    assert m.t0 == pytest.approx(300.0)
    assert m.tmelt == pytest.approx(1356.0)
    assert m.rhoc_p == pytest.approx(3.45e3)
    assert m.pmin == pytest.approx(-1e20)
    assert m.b1 == pytest.approx(2.8e-5)
    assert m.b2 == pytest.approx(2.8e-5)
    assert m.h == pytest.approx(3.8e-4)
    assert m.f == pytest.approx(0.5)


def test_read_mat_law49_free_format_defaults(tmp_path: Path):
    """Test parsing free-format comma-delimited /MAT/STEINB with default parameters."""
    deck_text = (
        "/BEGIN\n"
        "test_steinb_free\n"
        "/MAT/STEINB/202\n"
        "Minimal Steinberg\n"
        "7.85\n"
        "2.1e5, 0.3\n"
        "300.0, 40.0, 0.5\n"
        "# All thermal cards omitted or partial\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_free_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")

    blocks = read_deck(str(deck_file))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law49(block, model, log)

    assert not log.has_errors
    assert 202 in model.mat_law49s
    m = model.mat_law49s[202]

    assert m.id == 202
    assert m.rho == pytest.approx(7.85)
    assert m.refer_rho == pytest.approx(7.85)  # defaulted to rho
    assert m.e0 == pytest.approx(2.1e5)
    assert m.nu == pytest.approx(0.3)
    assert m.sigy == pytest.approx(300.0)
    assert m.beta == pytest.approx(40.0)
    assert m.n == pytest.approx(0.5)
    # Default values per hm_read_mat49.F
    assert m.eps_max == pytest.approx(1.0e20)
    assert m.sigma_max == pytest.approx(1.0e20)
    assert m.t0 == pytest.approx(300.0)
    assert m.tmelt == pytest.approx(1.0e20)
    assert m.pmin == pytest.approx(-1.0e20)


# ============================================================================
# 4. StarterDeck Writer & Round-Trip
# ============================================================================

def test_starter_deck_writer_fixed_and_roundtrip(tmp_path: Path):
    """Verify StarterDeck.mat_law49 creates valid cards that round-trip through read_deck."""
    deck = StarterDeck("steinb_roundtrip")
    deck.mat_law49(
        mat_id=1,
        title="Aluminum 6061-T6",
        rho=2.7,
        e0=7.2e4,
        nu=0.33,
        sig0=290.0,
        beta=125.0,
        n=0.1,
        eps_max=1e20,
        sigma_max=600.0,
        t0=300.0,
        tmelt=925.0,
        rhoc_p=2.43e3,
        pmin=-1.2e3,
        b1=6.5e-5,
        b2=6.5e-5,
        h=6.1e-4,
        f=0.2,
        fixed_format=True,
    )

    deck_path = tmp_path / "deck_out_0000.rad"
    deck.write(deck_path)

    # Read back
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law49(block, model, log)

    assert not log.has_errors
    assert 1 in model.mat_law49s
    m = model.mat_law49s[1]
    assert m.title == "Aluminum 6061-T6"
    assert m.rho == pytest.approx(2.7)
    assert m.e0 == pytest.approx(7.2e4)
    assert m.nu == pytest.approx(0.33)
    assert m.sigy == pytest.approx(290.0)
    assert m.beta == pytest.approx(125.0)
    assert m.n == pytest.approx(0.1)
    assert m.sigma_max == pytest.approx(600.0)
    assert m.tmelt == pytest.approx(925.0)
    assert m.rhoc_p == pytest.approx(2.43e3)
    assert m.pmin == pytest.approx(-1.2e3)
    assert m.b1 == pytest.approx(6.5e-5)
    assert m.b2 == pytest.approx(6.5e-5)
    assert m.h == pytest.approx(6.1e-4)
    assert m.f == pytest.approx(0.2)


def test_starter_deck_writer_free_format(tmp_path: Path):
    """Verify free-format writer round-trip."""
    deck = StarterDeck("steinb_free")
    deck.mat_steinb(
        mat_id=5,
        title="Tantalum",
        rho=16.69,
        e0=1.86e5,
        nu=0.35,
        sig0=770.0,
        beta=22.0,
        n=0.28,
        fixed_format=False,
        comma_delimited=True,
    )
    deck_path = tmp_path / "deck_free_out_0000.rad"
    deck.write(deck_path)

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law49(block, model, log)

    assert not log.has_errors
    assert 5 in model.mat_law49s
    m = model.mat_law49s[5]
    assert m.rho == pytest.approx(16.69)
    assert m.e0 == pytest.approx(1.86e5)
    assert m.sigy == pytest.approx(770.0)


# ============================================================================
# 5. Starter Parameter & Model Checks
# ============================================================================

def test_check_mat_law49_valid():
    """Verify valid LAW49 passes checks without error."""
    m = MatLaw49(
        id=1,
        rho=8.96,
        e0=1.24e5,
        nu=0.34,
        sigy=120.0,
        f=0.5,
    )
    log = MessageLog()
    check_mat_law49(mat=m, log=log)
    assert not log.has_errors


def test_check_mat_law49_invalid_parameters():
    """Verify parameter error messages for rho, E, nu, f."""
    # 1. Non-positive density
    log_rho = MessageLog()
    m1 = MatLaw49(id=1, rho=0.0, e0=1.0e5, nu=0.3)
    check_mat_law49(mat=m1, log=log_rho)
    assert log_rho.has_errors
    assert any("initial density RHO must be > 0" in m for m in log_rho.errors)

    # 2. Non-positive Young's modulus
    log_e = MessageLog()
    m2 = MatLaw49(id=2, rho=7.8, e0=-100.0, nu=0.3)
    check_mat_law49(mat=m2, log=log_e)
    assert log_e.has_errors
    assert any("Young's modulus E must be > 0" in m for m in log_e.errors)

    # 3. Poisson's ratio out of bounds (ANCMSG 1514)
    log_nu1 = MessageLog()
    m3 = MatLaw49(id=3, rho=7.8, e0=2.0e5, nu=0.5)
    check_mat_law49(mat=m3, log=log_nu1)
    assert log_nu1.has_errors
    assert any("Poisson's ratio nu must satisfy 0 <= nu < 0.5" in m and "1514" in m for m in log_nu1.errors)

    log_nu2 = MessageLog()
    m3b = MatLaw49(id=3, rho=7.8, e0=2.0e5, nu=-0.1)
    check_mat_law49(mat=m3b, log=log_nu2)
    assert log_nu2.has_errors
    assert any("Poisson's ratio nu must satisfy 0 <= nu < 0.5" in m and "1514" in m for m in log_nu2.errors)

    # 4. Negative F coefficient (ANCMSG 1513)
    log_f = MessageLog()
    m4 = MatLaw49(id=4, rho=7.8, e0=2.0e5, nu=0.3, f=-0.5)
    check_mat_law49(mat=m4, log=log_f)
    assert log_f.has_errors
    assert any("F coefficient must be >= 0" in m and "1513" in m for m in log_f.errors)


def test_check_mat_law49_element_compatibility():
    """Verify solid elements are supported while shell (ANCMSG 305) and 1D (ANCMSG 306) elements are rejected."""
    # Dispatch dictionary
    assert 49 in _MAT_CHECKS
    assert "LAW49" in _MAT_CHECKS
    assert "STEINB" in _MAT_CHECKS
    assert "STEINBERG" in _MAT_CHECKS

    # _ALLOWED_LAWS permits LAW49 on solids only
    assert 49 in _ALLOWED_LAWS["bricks"]
    assert 49 in _ALLOWED_LAWS["tetras"]
    assert 49 in _ALLOWED_LAWS["penta6"]
    assert 49 in _ALLOWED_LAWS["pyra5"]
    assert 49 in _ALLOWED_LAWS["solids"]

    assert 49 not in _ALLOWED_LAWS["shells"]
    assert 49 not in _ALLOWED_LAWS["shells_qbat"]
    assert 49 not in _ALLOWED_LAWS["shells_qeph"]
    assert 49 not in _ALLOWED_LAWS["sh3n"]
    assert 49 not in _ALLOWED_LAWS["quads"]
    assert 49 not in _ALLOWED_LAWS["trusses"]
    assert 49 not in _ALLOWED_LAWS["beams"]

    # Incompatible element rejection via check_mat_law49
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
    mat49 = MatLaw49(id=1, rho=8.96, e0=1.24e5, nu=0.34)
    log_solid = MessageLog()
    check_mat_law49(model=model_solid, mat=mat49, log=log_solid)
    assert not log_solid.has_errors

    # Test shell element rejection (ANCMSG 305)
    model_shell = DummyModel([("shells", DummyGroup([DummyElement(1)]))])
    log_shell = MessageLog()
    check_mat_law49(model=model_shell, mat=mat49, log=log_shell)
    assert log_shell.has_errors
    assert any("ANCMSG 305" in m for m in log_shell.errors)

    # Test 1D element rejection (ANCMSG 306)
    model_1d = DummyModel([("beams", DummyGroup([DummyElement(1)]))])
    log_1d = MessageLog()
    check_mat_law49(model=model_1d, mat=mat49, log=log_1d)
    assert log_1d.has_errors
    assert any("ANCMSG 306" in m for m in log_1d.errors)

    # Test 2D analysis rejection (N2D > 0, ANCMSG 305)
    model_2d = DummyModel([("bricks", DummyGroup([DummyElement(1)]))], n2d=1)
    log_2d = MessageLog()
    check_mat_law49(model=model_2d, mat=mat49, log=log_2d)
    assert log_2d.has_errors
    assert any("N2D > 0" in m and "ANCMSG 305" in m for m in log_2d.errors)
