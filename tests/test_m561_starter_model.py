"""Tests for Milestone M561: /MAT/LAW73 (/MAT/BARLAT2000, /MAT/HILL_THERM, /MAT/THERM_HILL) Starter & Model Integration.

Validates:
1. MatLaw73 (and MatHillTherm, MatBarlat2000, MatThermalHill, MatThermHill) dataclass fields,
   property helpers (rho0, rhor, E, Nu, G, A01..A12, sound_speed, sound_speed_shell),
   CFG attribute alias properties (fun_a1, yr_fun, efib, c, t_initial, spheat, epst1, epst2, epsp_max),
   and mapping protocol (__getitem__, __setitem__, __contains__, get, keys, values, items).
2. Model containers (mat_law73s, mat_therm_hills, mat_barlat2000s).
3. Card layouts (MAT_LAW73_1..7, MAT_BARLAT2000_1..7, MAT_HILL_THERM_1..7, MAT_THERM_HILL_1..7) matching matl73_73.cfg.
4. CFG catalogue mapping (LAW73, BARLAT2000, HILL_THERM, THERM_HILL -> 73).
5. Starter keyword reader for fixed and free 7-card formats, populating model.mat_law73s and model.materials.
6. Legacy 5-card format backward compatibility (M191).
7. Starter diagnostic checks: parameter bounds (rho > 0, E > 0, 0 <= nu < 0.5 with ANCMSG 1514, epsr1 < epsr2 with ANCMSG 1044),
   solid element rejection (ANCMSG 305), 1D element rejection (ANCMSG 306), and shell element acceptance.
8. StarterDeck writer for fixed format and helper aliases (mat_hill_therm, mat_therm_hill, mat_barlat2000) with roundtrip.
9. Shell element state initialization (uvar73 of shape (n, 7) and (n, nip, 7) in shell_bt4).
10. Material dispatch integration (MATERIAL_SHELL_DISPATCH, solid rejection, tangents, and sound speed).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import (
    MatLaw73,
    MatHillTherm,
    MatThermalHill,
    MatThermHill,
    Material,
)
from pyradioss.model.model import Model
from pyradioss.input.card_layouts import CARD_LAYOUTS
from pyradioss.input.cfg_catalogue import law_number, canonical_law_name
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law73
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.checks import (
    check_mat_law73,
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

def test_mat_law73_entities_and_properties():
    """Verify MatLaw73 dataclass fields, property helpers, and mapping protocol."""
    m = MatLaw73(
        id=73,
        rho=2.7e-9,
        e=70000.0,
        nu=0.33,
        r00=1.2,
        r45=1.4,
        r90=1.6,
        chard=0.25,
        iyield=1,
        eps_max=0.5,
        epsr1=0.2,
        epsr2=0.35,
        ifunce=2,
        einf=50000.0,
        ce=10.0,
        table_id=101,
        fscale=1.1,
        pscale=0.9,
        t0=295.0,
        rhocp=2.4e-3,
        title="Sheet Metal Law73",
    )

    # Class aliases
    assert MatHillTherm is MatLaw73
    assert MatThermalHill is MatLaw73
    assert MatThermHill is MatLaw73

    # Primary fields
    assert m.id == 73
    assert m.title == "Sheet Metal Law73"
    assert m.law == 73
    assert m.law_name == "LAW73"
    assert m.rho == pytest.approx(2.7e-9)
    assert m.rho0 == pytest.approx(2.7e-9)
    assert m.rhor == pytest.approx(2.7e-9)
    assert m.e == pytest.approx(70000.0)
    assert m.E == pytest.approx(70000.0)
    assert m.nu == pytest.approx(0.33)
    assert m.Nu == pytest.approx(0.33)
    assert m.r00 == pytest.approx(1.2)
    assert m.r45 == pytest.approx(1.4)
    assert m.r90 == pytest.approx(1.6)
    assert m.chard == pytest.approx(0.25)
    assert m.iyield == 1
    assert m.eps_max == pytest.approx(0.5)
    assert m.epsr1 == pytest.approx(0.2)
    assert m.epsr2 == pytest.approx(0.35)
    assert m.ifunce == 2
    assert m.einf == pytest.approx(50000.0)
    assert m.ce == pytest.approx(10.0)
    assert m.table_id == 101
    assert m.fscale == pytest.approx(1.1)
    assert m.pscale == pytest.approx(0.9)
    assert m.t0 == pytest.approx(295.0)
    assert m.rhocp == pytest.approx(2.4e-3)

    # Derived Moduli & Lankford Hill coefficients
    expected_g = 70000.0 / (2.0 * (1.0 + 0.33))
    r = 0.25 * (1.2 + 2.0 * 1.4 + 1.6)
    h = r / (1.0 + r)
    raw_a01 = h * (1.0 + 1.0 / 1.2)
    raw_a02 = h * (1.0 + 1.0 / 1.6)
    raw_a03 = 2.0 * h
    raw_a12 = (2.0 * 1.4 + 1.0) * (raw_a01 + raw_a02 - raw_a03)
    # iyield == 1 -> normalize by raw_a01
    expected_a01 = 1.0
    expected_a02 = raw_a02 / raw_a01
    expected_a03 = raw_a03 / raw_a01
    expected_a12 = raw_a12 / raw_a01

    assert m.G == pytest.approx(expected_g)
    assert m.A01 == pytest.approx(expected_a01)
    assert m.A02 == pytest.approx(expected_a02)
    assert m.A03 == pytest.approx(expected_a03)
    assert m.A12 == pytest.approx(expected_a12)

    expected_c_acoustic = math.sqrt(70000.0 / 2.7e-9)
    expected_c_shell = math.sqrt(70000.0 / (2.7e-9 * (1.0 - 0.33**2)))
    assert m.sound_speed == pytest.approx(expected_c_acoustic)
    assert m.sound_speed_shell == pytest.approx(expected_c_shell)

    # CFG alias properties
    m.fun_a1 = 205
    m.yr_fun = 3
    m.efib = 60000.0
    m.c = 15.0
    m.t_initial = 310.0
    m.spheat = 3.0e-3
    m.epst1 = 0.22
    m.epst2 = 0.38
    m.epsp_max = 0.65
    assert m.table_id == 205
    assert m.ifunce == 3
    assert m.einf == pytest.approx(60000.0)
    assert m.ce == pytest.approx(15.0)
    assert m.t0 == pytest.approx(310.0)
    assert m.rhocp == pytest.approx(3.0e-3)
    assert m.epsr1 == pytest.approx(0.22)
    assert m.epsr2 == pytest.approx(0.38)
    assert m.eps_max == pytest.approx(0.65)

    # Mapping protocol
    assert "rho" in m
    assert "e" in m
    assert "nu" in m
    assert "r00" in m
    assert "table_id" in m
    assert m["e"] == pytest.approx(70000.0)
    assert m.get("ce") == pytest.approx(15.0)
    assert m.get("missing_key", -1.0) == -1.0

    # Key/value setting
    m["r00"] = 1.35
    assert m["r00"] == pytest.approx(1.35)
    assert m.r00 == pytest.approx(1.35)

    with pytest.raises(KeyError, match="Cannot set unknown attribute"):
        m["unknown_param"] = 123.45


def test_model_mat_law73_containers():
    """Verify Model contains mat_law73s dict and aliases."""
    model = Model()
    assert hasattr(model, "mat_law73s")
    assert hasattr(model, "mat_therm_hills")
    assert model.mat_therm_hills is model.mat_law73s


# ============================================================================
# 2. Card Layouts & Catalogue Mappings
# ============================================================================

def test_card_layouts_and_cfg_catalogue_law73():
    """Verify card layouts and law catalogue mappings for LAW73 and synonyms."""
    for prefix in ("MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL"):
        assert CARD_LAYOUTS[f"{prefix}_1"] == [20]
        assert CARD_LAYOUTS[f"{prefix}_2"] == [20, 20]
        assert CARD_LAYOUTS[f"{prefix}_3"] == [10, 10, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_4"] == [20, 20, 20, 20, 10, 10]
        assert CARD_LAYOUTS[f"{prefix}_5"] == [20, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_6"] == [10, 10, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_7"] == [20, 20]

    assert law_number("LAW73") == 73
    assert law_number("HILL_THERM") == 73
    assert law_number("THERM_HILL") == 73
    assert law_number("MAT_LAW73") == 73
    assert law_number("MAT_HILL_THERM") == 73
    assert law_number("MAT_THERM_HILL") == 73

    assert canonical_law_name(73) == "LAW73"
    assert canonical_law_name("HILL_THERM") == "LAW73"
    assert canonical_law_name("THERM_HILL") == "LAW73"


# ============================================================================
# 3. Starter Keyword Reader
# ============================================================================

def test_starter_keyword_reader_fixed_7card(tmp_path):
    """Verify fixed-format 7-card /MAT/LAW73 parsing."""
    deck_text = (
        "/MAT/LAW73/1\n"
        "Thermal Hill Orthotropic Shell\n"
        "#                 RHO\n"
        "             2.70E-9\n"
        "#                  E                  NU\n"
        "             70000.0                0.33\n"
        "#FUNCT_IDE                          EINF                  CE\n"
        "         3                       55000.0                12.5\n"
        "#                R00                 R45                 R90              C_HARD   Iyield0\n"
        "                1.15                1.35                1.55                0.20         1\n"
        "#           EPSP_MAX              EPS_T1              EPS_T2\n"
        "                0.60                0.25                0.40\n"
        "#    TABLE                  SIGMA_SCALE         EPSPT_SCALE\n"
        "       202                           1.2                 0.8\n"
        "#                 TI                  CP\n"
        "               298.0             2.50E-3\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_fixed7_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law73(block, model, log)

    assert len(log.errors) == 0
    assert 1 in model.mat_law73s
    assert 1 in model.materials

    m = model.mat_law73s[1]
    assert m.id == 1
    assert m.title == "Thermal Hill Orthotropic Shell"
    assert m.rho == pytest.approx(2.7e-9)
    assert m.e == pytest.approx(70000.0)
    assert m.nu == pytest.approx(0.33)
    assert m.ifunce == 3
    assert m.einf == pytest.approx(55000.0)
    assert m.ce == pytest.approx(12.5)
    assert m.r00 == pytest.approx(1.15)
    assert m.r45 == pytest.approx(1.35)
    assert m.r90 == pytest.approx(1.55)
    assert m.chard == pytest.approx(0.20)
    assert m.iyield == 1
    assert m.eps_max == pytest.approx(0.60)
    assert m.epsr1 == pytest.approx(0.25)
    assert m.epsr2 == pytest.approx(0.40)
    assert m.table_id == 202
    assert m.fscale == pytest.approx(1.2)
    assert m.pscale == pytest.approx(0.8)
    assert m.t0 == pytest.approx(298.0)
    assert m.rhocp == pytest.approx(2.5e-3)

    # Check Material record
    mat = model.materials[1]
    assert mat.id == 1
    assert mat.law == 73
    assert mat.rho0 == pytest.approx(2.7e-9)
    assert mat.params["e"] == pytest.approx(70000.0)
    assert mat.params["table_id"] == 202


def test_starter_keyword_reader_free_and_synonyms(tmp_path):
    """Verify free-format /MAT/LAW73, /MAT/HILL_THERM, and /MAT/THERM_HILL parsing."""
    deck_text = (
        "/MAT/LAW73/10\n"
        "Law73 Free\n"
        "2.8e-9\n"
        "72000.0 0.32\n"
        "0 0.0 0.0\n"
        "1.1 1.3 1.5 0.15 0\n"
        "0.5 0.2 0.35\n"
        "301 1.0 1.0\n"
        "293.0 2.2e-3\n"
        "/MAT/HILL_THERM/20\n"
        "Hill Therm Free\n"
        "7.8e-9\n"
        "210000.0 0.30\n"
        "1 180000.0 8.0\n"
        "1.0 1.2 1.4 0.0 1\n"
        "0.8 0.3 0.5\n"
        "401 1.05 0.95\n"
        "300.0 3.5e-3\n"
        "/MAT/THERM_HILL/30\n"
        "Therm Hill Free Short\n"
        "7.85e-9\n"
        "205000.0 0.29\n"
        "0 0.0 0.0\n"
        "1.2 1.4 1.6 0.1 0\n"
        "0.7 0.2 0.4\n"
        "501 1.0 1.0\n"
        "295.0 0.0\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_free7_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law73(block, model, log)

    assert len(log.errors) == 0
    assert 10 in model.mat_law73s
    assert 20 in model.mat_law73s
    assert 30 in model.mat_law73s

    m10 = model.mat_law73s[10]
    assert m10.id == 10
    assert m10.title == "Law73 Free"
    assert m10.rho == pytest.approx(2.8e-9)
    assert m10.e == pytest.approx(72000.0)
    assert m10.nu == pytest.approx(0.32)
    assert m10.r00 == pytest.approx(1.1)
    assert m10.r45 == pytest.approx(1.3)
    assert m10.r90 == pytest.approx(1.5)
    assert m10.chard == pytest.approx(0.15)
    assert m10.table_id == 301

    m20 = model.mat_law73s[20]
    assert m20.id == 20
    assert m20.title == "Hill Therm Free"
    assert m20.rho == pytest.approx(7.8e-9)
    assert m20.e == pytest.approx(210000.0)
    assert m20.nu == pytest.approx(0.30)
    assert m20.ifunce == 1
    assert m20.einf == pytest.approx(180000.0)
    assert m20.ce == pytest.approx(8.0)
    assert m20.iyield == 1
    assert m20.table_id == 401

    m30 = model.mat_law73s[30]
    assert m30.id == 30
    assert m30.title == "Therm Hill Free Short"
    assert m30.rho == pytest.approx(7.85e-9)
    assert m30.e == pytest.approx(205000.0)
    assert m30.nu == pytest.approx(0.29)
    assert m30.table_id == 501
    assert m30.t0 == pytest.approx(295.0)


def test_starter_keyword_reader_legacy_5card_backward_compatibility(tmp_path):
    """Verify legacy 5-card /MAT/LAW73 format (M191) parses cleanly."""
    deck_text = (
        "/MAT/LAW73/73\n"
        "Thermal Hill Orthotropic M191\n"
        "#             RHO_0               RHO_R\n"
        "              2.7e-6                   0\n"
        "#                 E                  NU                 R00                 R45                 R90\n"
        "            70000.0                0.33                 1.2                 1.1                 1.5\n"
        "#             CHARD             EPS_MAX               EPST1               EPST2              FUN_A1\n"
        "                0.1                 0.2                0.01                0.02                   201\n"
        "#            FSCALE              PSCALE           T_INITIAL              SPHEAT              IYIELD\n"
        "                1.0                 1.0               293.0               900.0                   1\n"
        "#            YR_FUN                EFIB                   C\n"
        "                202             10000.0                 0.5\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_legacy5_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law73(block, model, log)

    assert len(log.errors) == 0
    assert 73 in model.mat_law73s
    m = model.mat_law73s[73]
    assert m.id == 73
    assert m.title == "Thermal Hill Orthotropic M191"
    assert m.rho == pytest.approx(2.7e-6)
    assert m.e == pytest.approx(70000.0)
    assert m.nu == pytest.approx(0.33)
    assert m.r00 == pytest.approx(1.2)
    assert m.r45 == pytest.approx(1.1)
    assert m.r90 == pytest.approx(1.5)
    assert m.chard == pytest.approx(0.1)
    assert m.eps_max == pytest.approx(0.2)
    assert m.epsr1 == pytest.approx(0.01)
    assert m.epsr2 == pytest.approx(0.02)
    assert m.table_id == 201
    assert m.fun_a1 == 201
    assert m.fscale == pytest.approx(1.0)
    assert m.pscale == pytest.approx(1.0)
    assert m.t0 == pytest.approx(293.0)
    assert m.t_initial == pytest.approx(293.0)
    assert m.rhocp == pytest.approx(900.0)
    assert m.spheat == pytest.approx(900.0)
    assert m.iyield == 1
    assert m.ifunce == 202
    assert m.yr_fun == 202
    assert m.einf == pytest.approx(10000.0)
    assert m.efib == pytest.approx(10000.0)
    assert m.ce == pytest.approx(0.5)
    assert m.c == pytest.approx(0.5)


# ============================================================================
# 4. Starter Validation Checks
# ============================================================================

def test_starter_checks_valid():
    """Verify clean LAW73 material passes validation without errors."""
    m = MatLaw73(id=1, rho=2.7e-9, e=70000.0, nu=0.33, r00=1.2, r45=1.3, r90=1.4,
                 epsr1=0.2, epsr2=0.35)
    log = MessageLog()
    check_mat_law73(mat=m, log=log)
    assert len(log.errors) == 0


def test_starter_checks_parameter_bounds():
    """Verify parameter bounds checks for rho, E, nu (ANCMSG 1514), Lankford R, and failure strains (ANCMSG 1044)."""
    # 1. Invalid density
    m_rho = MatLaw73(id=1, rho=0.0, e=70000.0, nu=0.33)
    log1 = MessageLog()
    check_mat_law73(mat=m_rho, log=log1)
    assert any("initial density RHO must be > 0" in e for e in log1.errors)

    # 2. Invalid Young's modulus
    m_e = MatLaw73(id=2, rho=2.7e-9, e=-1000.0, nu=0.33)
    log2 = MessageLog()
    check_mat_law73(mat=m_e, log=log2)
    assert any("Young's modulus E must be > 0" in e for e in log2.errors)

    # 3. Invalid Poisson's ratio (negative) -> ANCMSG 1514
    m_nu1 = MatLaw73(id=3, rho=2.7e-9, e=70000.0, nu=-0.05)
    log3 = MessageLog()
    check_mat_law73(mat=m_nu1, log=log3)
    assert any("ANCMSG 1514" in e for e in log3.errors)

    # 4. Invalid Poisson's ratio (>= 0.5) -> ANCMSG 1514
    m_nu2 = MatLaw73(id=4, rho=2.7e-9, e=70000.0, nu=0.50)
    log4 = MessageLog()
    check_mat_law73(mat=m_nu2, log=log4)
    assert any("ANCMSG 1514" in e for e in log4.errors)

    # 5. Invalid Lankford parameters
    m_r = MatLaw73(id=5, rho=2.7e-9, e=70000.0, nu=0.33, r00=-0.5, r45=0.0, r90=-1.0)
    log5 = MessageLog()
    check_mat_law73(mat=m_r, log=log5)
    assert any("Lankford parameter R00 must be > 0" in e for e in log5.errors)
    assert any("Lankford parameter R45 must be > 0" in e for e in log5.errors)
    assert any("Lankford parameter R90 must be > 0" in e for e in log5.errors)

    # 6. Failure strains: epsr1 >= epsr2 -> ANCMSG 1044
    m_eps = MatLaw73(id=6, rho=2.7e-9, e=70000.0, nu=0.33, epsr1=0.40, epsr2=0.30)
    log6 = MessageLog()
    check_mat_law73(mat=m_eps, log=log6)
    assert any("ANCMSG 1044" in e for e in log6.errors)


def test_starter_checks_elements_compatibility():
    """Verify solid rejection (ANCMSG 305) and 1D rejection (ANCMSG 306)."""
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    mat = Material(id=10, law=73, rho0=2.7e-9, params={"e": 70000.0, "nu": 0.33})
    m73 = MatLaw73(id=10, rho=2.7e-9, e=70000.0, nu=0.33)
    model.materials[10] = mat
    model.mat_law73s[10] = m73

    class FakeGroup:
        def __init__(self, name, mid):
            self.name = name
            self.state = {"slices": [(slice(0, 1), mat, None)]}
            self.n = 1
        def values(self):
            return []

    # Solid element group (bricks) -> ANCMSG 305
    model_solid = Model()
    model_solid.node_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    model_solid.materials[10] = mat
    model_solid.mat_law73s[10] = m73
    model_solid.bricks = FakeGroup("bricks", 10)
    log_solid = MessageLog()
    check_mat_law73(mat=m73, log=log_solid, model=model_solid, mat_id=10)
    assert any("ANCMSG 305" in e and "solid elements" in e for e in log_solid.errors)

    # 1D element group (beams) -> ANCMSG 306
    model_1d = Model()
    model_1d.node_ids = np.array([1, 2])
    model_1d.materials[10] = mat
    model_1d.mat_law73s[10] = m73
    model_1d.beams = FakeGroup("beams", 10)
    log_1d = MessageLog()
    check_mat_law73(mat=m73, log=log_1d, model=model_1d, mat_id=10)
    assert any("ANCMSG 306" in e and "1D elements" in e for e in log_1d.errors)


def test_allowed_laws_and_mat_checks_registration():
    """Verify LAW73 is registered in _MAT_CHECKS and _ALLOWED_LAWS for shells."""
    for key in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL",
                "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL"):
        assert key in _MAT_CHECKS
        assert _MAT_CHECKS[key] is check_mat_law73

    for grp in ("shells", "shells_qbat", "shells_qeph", "sh3n"):
        assert 73 in _ALLOWED_LAWS[grp]
        assert "LAW73" in _ALLOWED_LAWS[grp]
        assert "HILL_THERM" in _ALLOWED_LAWS[grp]

    for grp in ("bricks", "tetras", "penta6", "pyra5", "trusses", "beams"):
        assert 73 not in _ALLOWED_LAWS[grp]
        assert "LAW73" not in _ALLOWED_LAWS[grp]


# ============================================================================
# 5. Starter Deck Writer & Roundtrip
# ============================================================================

def test_deck_writer_and_roundtrip(tmp_path):
    """Verify StarterDeck.mat_law73, aliases, and roundtrip parsing."""
    deck = StarterDeck("test_hill_therm")
    deck.mat_law73(
        1,
        "Law 73 Primary",
        rho=2.7e-9,
        e=70000.0,
        nu=0.33,
        r00=1.1,
        r45=1.3,
        r90=1.5,
        chard=0.2,
        iyield=1,
        eps_max=0.5,
        epsr1=0.2,
        epsr2=0.35,
        ifunce=2,
        einf=50000.0,
        ce=10.0,
        table_id=201,
        fscale=1.1,
        pscale=0.9,
        t0=295.0,
        rhocp=2.4e-3,
    )
    deck.mat_hill_therm(
        2,
        "Alias Hill Therm",
        rho=7.8e-9,
        e=210000.0,
        nu=0.30,
        r00=1.0,
        r45=1.2,
        r90=1.4,
        chard=0.0,
        iyield=0,
        eps_max=0.7,
        epsr1=0.3,
        epsr2=0.45,
        ifunce=0,
        einf=0.0,
        ce=0.0,
        table_id=202,
        fscale=1.0,
        pscale=1.0,
        t0=300.0,
        rhocp=3.5e-3,
    )
    deck.mat_therm_hill(
        3,
        "Alias Therm Hill",
        rho=7.85e-9,
        e=205000.0,
        nu=0.29,
        r00=1.2,
        r45=1.4,
        r90=1.6,
        chard=0.1,
        iyield=1,
        eps_max=0.6,
        epsr1=0.25,
        epsr2=0.4,
        ifunce=1,
        einf=170000.0,
        ce=5.0,
        table_id=203,
        fscale=1.05,
        pscale=0.95,
        t0=298.0,
        rhocp=3.2e-3,
    )
    deck.mat_law73(
        4,
        "Law 73 Secondary",
        rho=2.8e-9,
        e=72000.0,
        nu=0.32,
        r00=1.15,
        r45=1.25,
        r90=1.35,
        chard=0.3,
        iyield=0,
        eps_max=0.55,
        epsr1=0.22,
        epsr2=0.38,
        ifunce=0,
        einf=0.0,
        ce=0.0,
        table_id=204,
        fscale=1.0,
        pscale=1.0,
        t0=293.0,
        rhocp=2.3e-3,
    )

    deck_file = tmp_path / "deck_law73_roundtrip_0000.rad"
    deck.write(str(deck_file))

    # Parse back
    blocks = read_deck(str(deck_file))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law73(block, model, log)

    assert len(log.errors) == 0
    assert 1 in model.mat_law73s
    assert 2 in model.mat_law73s
    assert 3 in model.mat_law73s
    assert 4 in model.mat_law73s

    m1 = model.mat_law73s[1]
    assert m1.id == 1
    assert m1.title == "Law 73 Primary"
    assert m1.rho == pytest.approx(2.7e-9)
    assert m1.e == pytest.approx(70000.0)
    assert m1.nu == pytest.approx(0.33)
    assert m1.r00 == pytest.approx(1.1)
    assert m1.r45 == pytest.approx(1.3)
    assert m1.r90 == pytest.approx(1.5)
    assert m1.table_id == 201
    assert m1.t0 == pytest.approx(295.0)

    m2 = model.mat_law73s[2]
    assert m2.id == 2
    assert m2.title == "Alias Hill Therm"
    assert m2.rho == pytest.approx(7.8e-9)
    assert m2.e == pytest.approx(210000.0)
    assert m2.table_id == 202

    m3 = model.mat_law73s[3]
    assert m3.id == 3
    assert m3.title == "Alias Therm Hill"
    assert m3.rho == pytest.approx(7.85e-9)
    assert m3.e == pytest.approx(205000.0)
    assert m3.table_id == 203

    m4 = model.mat_law73s[4]
    assert m4.id == 4
    assert m4.title == "Law 73 Secondary"
    assert m4.rho == pytest.approx(2.8e-9)
    assert m4.e == pytest.approx(72000.0)
    assert m4.table_id == 204


# ============================================================================
# 6. Shell Element State Initialization & Material Extra
# ============================================================================

def test_shell_element_state_initialization():
    """Verify shell_bt4._init_material_state initializes uvar73 for LAW73."""
    class FakePart:
        def __init__(self, mat, prop):
            self.mat = mat
            self.prop = prop

    class FakeProp:
        def __init__(self):
            self.thick = 1.0
            self.nip = 3

    class FakeGroup:
        def __init__(self, mat, prop):
            self.n = 5
            self.state = {
                "slices": [(slice(0, 5), mat, prop)],
                "mat_extra": {},
            }

    mat73 = Material(
        id=1,
        law=73,
        law_name="LAW73",
        rho0=2.7e-9,
        params={"e": 70000.0, "nu": 0.33, "eps_max": 0.5},
    )
    prop = FakeProp()
    grp = FakeGroup(mat73, prop)

    _init_material_state(grp, nip_max=3)

    st = grp.state
    assert "uvar73" in st
    assert st["uvar73"].shape == (5, 7)
    assert "uvar73" in st["mat_extra"]
    assert st["mat_extra"]["uvar73"].shape == (5, 3, 7)
    assert st["chk_fail"] is True

    # Test _layer_extra
    st["layfail"] = np.zeros((5, 3), dtype=bool)
    extra = _layer_extra(st, slice(0, 5), 1)
    assert "uvar73" in extra
    assert "uvar" in extra
    assert extra["uvar73"].shape == (5, 7)
    assert extra["uvar"].shape == (5, 7)


# ============================================================================
# 7. Material Dispatch Integration
# ============================================================================

def test_materials_dispatch_integration():
    """Verify materials package dispatch, tangents, and sound speed for LAW73."""
    mat = Material(
        id=1,
        law=73,
        law_name="LAW73",
        rho0=2.7e-9,
        params={
            "e": 70000.0,
            "nu": 0.33,
            "r00": 1.0,
            "r45": 1.0,
            "r90": 1.0,
        },
    )

    # Shell dispatch registered
    for synonym in (73, "73", "LAW73", "HILL_THERM", "THERM_HILL",
                    "MAT_LAW73", "MAT_HILL_THERM", "MAT_THERM_HILL"):
        assert synonym in materials.MATERIAL_SHELL_DISPATCH
        assert synonym in materials.MATERIAL_SOLID_DISPATCH

    # Solid update rejected
    with pytest.raises(NotImplementedError, match="shell elements only"):
        materials.solid_update(mat, np.zeros((1, 6)), np.zeros((1, 6)))

    # Solid tangent rejected
    with pytest.raises(NotImplementedError, match="no solid tangent"):
        materials.solid_tangent(mat, np.zeros((1, 6)))

    # Shell membrane tangent
    c_mem = materials.shell_membrane_tangent(mat)
    assert c_mem.shape == (3, 3)
    c_iso = 70000.0 / (1.0 - 0.33**2)
    assert c_mem[0, 0] == pytest.approx(c_iso)
    assert c_mem[1, 1] == pytest.approx(c_iso)
    assert c_mem[0, 1] == pytest.approx(0.33 * c_iso)

    # Shell layer tangent
    c_lay = materials.shell_layer_tangent(mat, sig=np.zeros((2, 3)))
    assert c_lay.shape == (2, 3, 3)

    # Sound speed
    c_snd = materials.sound_speed(mat, rho=2.7e-9)
    expected_c = math.sqrt(70000.0 / (2.7e-9 * (1.0 - 0.33**2)))
    assert c_snd == pytest.approx(expected_c)
