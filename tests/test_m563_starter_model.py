"""Tests for Milestone M563: /MAT/LAW74 (/MAT/HILL_3D, /MAT/ORTH_PLAS, /MAT/THERM_HILL) Starter & Model Integration.

Validates:
1. MatLaw74 (and MatHill3D, MatOrthPlas, MaterialLaw74) dataclass fields,
   property helpers (rho0, rhor, E, nu, Nu, G, bulk, K, sound_speed, sound_speed_solid),
   Hill coefficients (FF, GG, HH, LL, MM, NN and lowercase aliases),
   CFG attribute alias properties (fun_a1, yr_fun, efib, c, t_initial, spheat, epst1, epst2,
   sigt1..3, sigyt1..3, etc.),
   and mapping protocol (__getitem__, __setitem__, __contains__, get, keys, values, items).
2. Model containers (mat_law74s, mat_hill_3ds, mat_orth_plass, mat_hill_therms).
3. Card layouts (MAT_LAW74_1..8, MAT_HILL_3D_1..8, MAT_ORTH_PLAS_1..8, MAT_LAW74_CFG_1..8) matching matl74_74.cfg.
4. CFG catalogue mapping (LAW74, HILL_3D, ORTH_PLAS, MAT_LAW74, MAT_HILL_3D, MAT_ORTH_PLAS -> 74).
5. Starter keyword reader for fixed and free 8-card formats, populating model.mat_law74s and model.materials.
6. Legacy 7-card format backward compatibility (radioss110 without card 8).
7. Starter diagnostic checks: parameter bounds (rho > 0, E > 0, 0 <= nu < 0.5 with ANCMSG 1514,
   S11Y..S31Y > 0 with ANCMSG 822, epsr1 < epsr2 with ANCMSG 1044),
   shell element rejection (ANCMSG 305), 1D element rejection (ANCMSG 306), 2D analysis rejection (ANCMSG 305),
   and solid element acceptance.
8. StarterDeck writer for fixed and free format and helper aliases (mat_hill_3d, mat_orth_plas) with roundtrip.
9. Solid element state initialization (uvar74 of shape (n, 10), temp initialized to t0).
10. Material dispatch integration (MATERIAL_SOLID_DISPATCH, shell rejection, solid update, tangents, and sound speed).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import (
    MatLaw74,
    MatHill3D,
    MatOrthPlas,
    MaterialLaw74,
    Material,
)
from pyradioss.model.model import Model
from pyradioss.input.card_layouts import CARD_LAYOUTS
from pyradioss.input.cfg_catalogue import law_number, canonical_law_name
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law74
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.checks import (
    check_mat_law74,
    check_materials,
    check_model,
    _ALLOWED_LAWS,
    _MAT_CHECKS,
)
from pyradioss.common.messages import MessageLog
from pyradioss.elements.solid_hexa8 import _init_material_state
import pyradioss.materials as materials


# ============================================================================
# 1. Model Entities, Property Helpers & Mapping Protocol
# ============================================================================

def test_mat_law74_entities_and_properties():
    """Verify MatLaw74 dataclass fields, property helpers, and mapping protocol."""
    m = MatLaw74(
        id=74,
        rho=7.8e-9,
        e=210000.0,
        nu=0.30,
        eps_max=0.5,
        epsr1=0.2,
        epsr2=0.35,
        ifunce=2,
        einf=180000.0,
        ce=10.0,
        fsmooth=1,
        chard=0.25,
        fcut=5000.0,
        s11y=400.0,
        s22y=420.0,
        s33y=450.0,
        s12y=230.0,
        s23y=240.0,
        s31y=250.0,
        table_id=101,
        fscale=1.1,
        pscale=0.9,
        t0=295.0,
        rhocp=3.5e-3,
        title="3D Hill Plasticity Law74",
    )

    # Class aliases
    assert MatHill3D is MatLaw74
    assert MatOrthPlas is MatLaw74
    assert MaterialLaw74 is MatLaw74

    # Primary fields
    assert m.id == 74
    assert m.title == "3D Hill Plasticity Law74"
    assert m.law == 74
    assert m.law_name == "LAW74"
    assert m.rho == pytest.approx(7.8e-9)
    assert m.rho0 == pytest.approx(7.8e-9)
    assert m.rhor == pytest.approx(7.8e-9)
    assert m.e == pytest.approx(210000.0)
    assert m.E == pytest.approx(210000.0)
    assert m.nu == pytest.approx(0.30)
    assert m.Nu == pytest.approx(0.30)
    assert m.eps_max == pytest.approx(0.5)
    assert m.epsr1 == pytest.approx(0.2)
    assert m.epsr2 == pytest.approx(0.35)
    assert m.ifunce == 2
    assert m.einf == pytest.approx(180000.0)
    assert m.ce == pytest.approx(10.0)
    assert m.fsmooth == 1
    assert m.chard == pytest.approx(0.25)
    assert m.fcut == pytest.approx(5000.0)
    assert m.s11y == pytest.approx(400.0)
    assert m.s22y == pytest.approx(420.0)
    assert m.s33y == pytest.approx(450.0)
    assert m.s12y == pytest.approx(230.0)
    assert m.s23y == pytest.approx(240.0)
    assert m.s31y == pytest.approx(250.0)
    assert m.table_id == 101
    assert m.fscale == pytest.approx(1.1)
    assert m.pscale == pytest.approx(0.9)
    assert m.t0 == pytest.approx(295.0)
    assert m.rhocp == pytest.approx(3.5e-3)

    # Derived Moduli
    expected_g = 210000.0 / (2.0 * (1.0 + 0.30))
    expected_bulk = 210000.0 / (3.0 * (1.0 - 2.0 * 0.30))
    assert m.G == pytest.approx(expected_g)
    assert m.bulk == pytest.approx(expected_bulk)
    assert m.K == pytest.approx(expected_bulk)

    # Derived Hill coefficients (hm_read_mat74.F lines 229-234)
    expected_ff = 0.5 * (1.0 / 420.0**2 + 1.0 / 450.0**2 - 1.0 / 400.0**2)
    expected_gg = 0.5 * (1.0 / 400.0**2 + 1.0 / 450.0**2 - 1.0 / 420.0**2)
    expected_hh = 0.5 * (1.0 / 400.0**2 + 1.0 / 420.0**2 - 1.0 / 450.0**2)
    expected_ll = 0.5 / 240.0**2
    expected_mm = 0.5 / 250.0**2
    expected_nn = 0.5 / 230.0**2

    assert m.FF == pytest.approx(expected_ff)
    assert m.GG == pytest.approx(expected_gg)
    assert m.HH == pytest.approx(expected_hh)
    assert m.LL == pytest.approx(expected_ll)
    assert m.MM == pytest.approx(expected_mm)
    assert m.NN == pytest.approx(expected_nn)

    # Lowercase aliases
    assert m.ff == pytest.approx(expected_ff)
    assert m.gg == pytest.approx(expected_gg)
    assert m.hh == pytest.approx(expected_hh)
    assert m.ll == pytest.approx(expected_ll)
    assert m.mm == pytest.approx(expected_mm)
    assert m.nn == pytest.approx(expected_nn)

    # Sound speed: sound_speed = sqrt(E/rho), sound_speed_solid = sqrt((K + 4/3*G)/rho)
    expected_c_acoustic = math.sqrt(210000.0 / 7.8e-9)
    expected_c_solid = math.sqrt((expected_bulk + 4.0 / 3.0 * expected_g) / 7.8e-9)
    assert m.sound_speed == pytest.approx(expected_c_acoustic)
    assert m.sound_speed_solid == pytest.approx(expected_c_solid)

    # CallableFloat check
    assert m.sound_speed_solid() == pytest.approx(expected_c_solid)

    # CFG alias properties
    m.fun_a1 = 205
    m.yr_fun = 3
    m.efib = 190000.0
    m.c = 15.0
    m.t_initial = 310.0
    m.spheat = 4.0e-3
    m.epst1 = 0.22
    m.epst2 = 0.38
    m.epsp_max = 0.65
    m.sigt1 = 410.0
    m.sigt2 = 430.0
    m.sigt3 = 460.0
    m.sigyt1 = 235.0
    m.sigyt2 = 245.0
    m.sigyt3 = 255.0

    assert m.table_id == 205
    assert m.ifunce == 3
    assert m.einf == pytest.approx(190000.0)
    assert m.ce == pytest.approx(15.0)
    assert m.t0 == pytest.approx(310.0)
    assert m.rhocp == pytest.approx(4.0e-3)
    assert m.epsr1 == pytest.approx(0.22)
    assert m.epsr2 == pytest.approx(0.38)
    assert m.eps_max == pytest.approx(0.65)
    assert m.s11y == pytest.approx(410.0)
    assert m.s22y == pytest.approx(430.0)
    assert m.s33y == pytest.approx(460.0)
    assert m.s12y == pytest.approx(235.0)
    assert m.s23y == pytest.approx(245.0)
    assert m.s31y == pytest.approx(255.0)

    # Mapping protocol
    assert "rho" in m
    assert "e" in m
    assert "nu" in m
    assert "s11y" in m
    assert "table_id" in m
    assert m["e"] == pytest.approx(210000.0)
    assert m.get("ce") == pytest.approx(15.0)
    assert m.get("missing_key", -1.0) == -1.0

    # Key/value setting
    m["s11y"] = 415.0
    assert m["s11y"] == pytest.approx(415.0)
    assert m.s11y == pytest.approx(415.0)

    with pytest.raises(KeyError, match="Cannot set unknown attribute"):
        m["unknown_param"] = 123.45


def test_model_mat_law74_containers():
    """Verify Model contains mat_law74s dict and aliases."""
    model = Model()
    assert hasattr(model, "mat_law74s")
    assert hasattr(model, "mat_hill_3ds")
    assert hasattr(model, "mat_orth_plass")
    assert hasattr(model, "mat_hill_therms")
    assert model.mat_hill_3ds is model.mat_law74s
    assert model.mat_orth_plass is model.mat_law74s
    assert model.mat_hill_therms is model.mat_law74s


# ============================================================================
# 2. Card Layouts & Catalogue Mappings
# ============================================================================

def test_card_layouts_and_cfg_catalogue_law74():
    """Verify card layouts and law catalogue mappings for LAW74 and synonyms."""
    for prefix in ("MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS", "MAT_LAW74_CFG"):
        assert CARD_LAYOUTS[f"{prefix}_1"] == [20, 20]
        assert CARD_LAYOUTS[f"{prefix}_2"] == [20, 20, 20, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_3"] == [10, 10, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_4"] == [10, 10, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_5"] == [20, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_6"] == [20, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_7"] == [10, 10, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_8"] == [20, 20]

    for key in ("LAW74", "HILL_3D", "ORTH_PLAS", "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS"):
        assert law_number(key) == 74
        assert canonical_law_name(key) in ("LAW74", "HILL_3D", "ORTH_PLAS")

    # Allowed laws verification in checks.py
    for fam in ("bricks", "tetras", "penta6", "pyra5", "solids"):
        assert 74 in _ALLOWED_LAWS[fam]
        assert "LAW74" in _ALLOWED_LAWS[fam]
        assert "HILL_3D" in _ALLOWED_LAWS[fam]
        assert "ORTH_PLAS" in _ALLOWED_LAWS[fam]

    # Verify not allowed in shells or 1D elements
    assert 74 not in _ALLOWED_LAWS["shells"]
    assert 74 not in _ALLOWED_LAWS["trusses"]
    assert 74 not in _ALLOWED_LAWS["beams"]


# ============================================================================
# 3. Starter Keyword Reader: Fixed & Free Formats
# ============================================================================

def test_starter_keyword_reader_fixed8_format(tmp_path):
    """Verify 8-card fixed format parser for /MAT/LAW74, /MAT/HILL_3D, /MAT/ORTH_PLAS."""
    deck_text = (
        "/MAT/LAW74/1\n"
        "Law 74 Fixed Card 8-line\n"
        "#                 RHO\n"
        "               7.8e-9\n"
        "#                   E                  NU             EPS_MAX               EPSR1               EPSR2\n"
        "             210000.0                 0.3                 0.5                 0.2                0.35\n"
        "#   IFUNCE               EINF                  CE\n"
        "         2           180000.0                10.0\n"
        "#  FSMOOTH              CHARD                FCUT\n"
        "         1               0.25              5000.0\n"
        "#                S11Y                S22Y                S33Y\n"
        "                400.0               420.0               450.0\n"
        "#                S12Y                S23Y                S31Y\n"
        "                230.0               240.0               250.0\n"
        "# TABLE_ID             FSCALE              PSCALE\n"
        "       101                1.1                 0.9\n"
        "#                  T0               RHOCP\n"
        "                295.0              3.5e-3\n"
        "/MAT/HILL_3D/2\n"
        "Hill 3D Fixed Alias\n"
        "#                 RHO\n"
        "               2.7e-9\n"
        "#                   E                  NU             EPS_MAX               EPSR1               EPSR2\n"
        "              70000.0                0.33                 0.6                0.25                 0.4\n"
        "#   IFUNCE               EINF                  CE\n"
        "         0                0.0                 0.0\n"
        "#  FSMOOTH              CHARD                FCUT\n"
        "         0                0.0                 0.0\n"
        "#                S11Y                S22Y                S33Y\n"
        "                150.0               160.0               170.0\n"
        "#                S12Y                S23Y                S31Y\n"
        "                 85.0                90.0                95.0\n"
        "# TABLE_ID             FSCALE              PSCALE\n"
        "       102                1.0                 1.0\n"
        "#                  T0               RHOCP\n"
        "                293.0              2.4e-3\n"
        "/MAT/ORTH_PLAS/3\n"
        "Orth Plas Fixed Alias\n"
        "#                 RHO\n"
        "               7.9e-9\n"
        "#                   E                  NU             EPS_MAX               EPSR1               EPSR2\n"
        "             205000.0                0.29                 0.4                0.15                 0.3\n"
        "#   IFUNCE               EINF                  CE\n"
        "         1           175000.0                 5.0\n"
        "#  FSMOOTH              CHARD                FCUT\n"
        "         1                0.2              4000.0\n"
        "#                S11Y                S22Y                S33Y\n"
        "                380.0               390.0               410.0\n"
        "#                S12Y                S23Y                S31Y\n"
        "                210.0               220.0               230.0\n"
        "# TABLE_ID             FSCALE              PSCALE\n"
        "       103               1.05                0.95\n"
        "#                  T0               RHOCP\n"
        "                300.0              3.8e-3\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_fixed8_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law74(block, model, log)

    assert len(log.errors) == 0
    assert 1 in model.mat_law74s
    assert 2 in model.mat_law74s
    assert 3 in model.mat_law74s

    # Materials container populated
    assert 1 in model.materials
    assert 2 in model.materials
    assert 3 in model.materials

    m1 = model.mat_law74s[1]
    assert m1.id == 1
    assert m1.title == "Law 74 Fixed Card 8-line"
    assert m1.rho == pytest.approx(7.8e-9)
    assert m1.e == pytest.approx(210000.0)
    assert m1.nu == pytest.approx(0.3)
    assert m1.eps_max == pytest.approx(0.5)
    assert m1.epsr1 == pytest.approx(0.2)
    assert m1.epsr2 == pytest.approx(0.35)
    assert m1.ifunce == 2
    assert m1.einf == pytest.approx(180000.0)
    assert m1.ce == pytest.approx(10.0)
    assert m1.fsmooth == 1
    assert m1.chard == pytest.approx(0.25)
    assert m1.fcut == pytest.approx(5000.0)
    assert m1.s11y == pytest.approx(400.0)
    assert m1.s22y == pytest.approx(420.0)
    assert m1.s33y == pytest.approx(450.0)
    assert m1.s12y == pytest.approx(230.0)
    assert m1.s23y == pytest.approx(240.0)
    assert m1.s31y == pytest.approx(250.0)
    assert m1.table_id == 101
    assert m1.fscale == pytest.approx(1.1)
    assert m1.pscale == pytest.approx(0.9)
    assert m1.t0 == pytest.approx(295.0)
    assert m1.rhocp == pytest.approx(3.5e-3)

    m2 = model.mat_law74s[2]
    assert m2.id == 2
    assert m2.title == "Hill 3D Fixed Alias"
    assert m2.rho == pytest.approx(2.7e-9)
    assert m2.e == pytest.approx(70000.0)
    assert m2.s11y == pytest.approx(150.0)
    assert m2.s22y == pytest.approx(160.0)
    assert m2.s33y == pytest.approx(170.0)
    assert m2.table_id == 102

    m3 = model.mat_law74s[3]
    assert m3.id == 3
    assert m3.title == "Orth Plas Fixed Alias"
    assert m3.rho == pytest.approx(7.9e-9)
    assert m3.e == pytest.approx(205000.0)
    assert m3.s11y == pytest.approx(380.0)
    assert m3.table_id == 103


def test_starter_keyword_reader_free8_format(tmp_path):
    """Verify 8-card free-format parser for /MAT/LAW74."""
    deck_text = (
        "/MAT/LAW74/10\n"
        "Hill 3D Free\n"
        "7.85e-9\n"
        "200000.0 0.31 0.45 0.18 0.32\n"
        "1 170000.0 8.0\n"
        "1 0.3 6000.0\n"
        "390.0 410.0 430.0\n"
        "220.0 230.0 240.0\n"
        "201 1.05 0.95\n"
        "298.0 3.2e-3\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_free8_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law74(block, model, log)

    assert len(log.errors) == 0
    assert 10 in model.mat_law74s
    m = model.mat_law74s[10]
    assert m.id == 10
    assert m.rho == pytest.approx(7.85e-9)
    assert m.e == pytest.approx(200000.0)
    assert m.nu == pytest.approx(0.31)
    assert m.eps_max == pytest.approx(0.45)
    assert m.epsr1 == pytest.approx(0.18)
    assert m.epsr2 == pytest.approx(0.32)
    assert m.ifunce == 1
    assert m.einf == pytest.approx(170000.0)
    assert m.ce == pytest.approx(8.0)
    assert m.fsmooth == 1
    assert m.chard == pytest.approx(0.3)
    assert m.fcut == pytest.approx(6000.0)
    assert m.s11y == pytest.approx(390.0)
    assert m.s22y == pytest.approx(410.0)
    assert m.s33y == pytest.approx(430.0)
    assert m.s12y == pytest.approx(220.0)
    assert m.s23y == pytest.approx(230.0)
    assert m.s31y == pytest.approx(240.0)
    assert m.table_id == 201
    assert m.fscale == pytest.approx(1.05)
    assert m.pscale == pytest.approx(0.95)
    assert m.t0 == pytest.approx(298.0)
    assert m.rhocp == pytest.approx(3.2e-3)


def test_starter_keyword_reader_legacy_7card_backward_compatibility(tmp_path):
    """Verify legacy 7-card format without card 8 (radioss110 format)."""
    deck_text = (
        "/MAT/LAW74/74\n"
        "Hill 3D Legacy 7 Card\n"
        "#                 RHO\n"
        "               7.8e-9\n"
        "#                   E                  NU             EPS_MAX               EPSR1               EPSR2\n"
        "             210000.0                 0.3                 0.5                 0.2                0.35\n"
        "#   IFUNCE               EINF                  CE\n"
        "         0                0.0                 0.0\n"
        "#  FSMOOTH              CHARD                FCUT\n"
        "         0                0.0                 0.0\n"
        "#                S11Y                S22Y                S33Y\n"
        "                400.0               420.0               450.0\n"
        "#                S12Y                S23Y                S31Y\n"
        "                230.0               240.0               250.0\n"
        "# TABLE_ID             FSCALE              PSCALE\n"
        "       101                1.0                 1.0\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_legacy7_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law74(block, model, log)

    assert len(log.errors) == 0
    assert 74 in model.mat_law74s
    m = model.mat_law74s[74]
    assert m.id == 74
    assert m.rho == pytest.approx(7.8e-9)
    assert m.e == pytest.approx(210000.0)
    assert m.s11y == pytest.approx(400.0)
    assert m.table_id == 101
    # Defaults from 7-card read
    assert m.t0 == pytest.approx(293.0)
    assert m.rhocp == pytest.approx(0.0)


# ============================================================================
# 4. Starter Diagnostic Checks
# ============================================================================

def test_starter_checks_valid():
    """Verify valid LAW74 material passes validation without errors."""
    m = MatLaw74(
        id=1,
        rho=7.8e-9,
        e=210000.0,
        nu=0.30,
        s11y=400.0,
        s22y=420.0,
        s33y=450.0,
        s12y=230.0,
        s23y=240.0,
        s31y=250.0,
        epsr1=0.2,
        epsr2=0.35,
    )
    log = MessageLog()
    check_mat_law74(mat=m, log=log)
    assert len(log.errors) == 0


def test_starter_checks_parameter_bounds():
    """Verify parameter bounds checks for rho, E, nu (ANCMSG 1514), yield stresses (ANCMSG 822), and failure strains (ANCMSG 1044)."""
    # 1. Invalid density
    m_rho = MatLaw74(id=1, rho=0.0, e=210000.0, nu=0.3, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
    log1 = MessageLog()
    check_mat_law74(mat=m_rho, log=log1)
    assert any("initial density RHO must be > 0" in str(err) for err in log1.errors)

    # 2. Invalid Young's modulus
    m_e = MatLaw74(id=2, rho=7.8e-9, e=0.0, nu=0.3, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
    log2 = MessageLog()
    check_mat_law74(mat=m_e, log=log2)
    assert any("Young's modulus E must be > 0" in str(err) for err in log2.errors)

    # 3. Invalid Poisson's ratio (ANCMSG 1514)
    m_nu_neg = MatLaw74(id=3, rho=7.8e-9, e=210000.0, nu=-0.1, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
    log3 = MessageLog()
    check_mat_law74(mat=m_nu_neg, log=log3)
    assert any("ANCMSG 1514" in str(err) for err in log3.errors)

    m_nu_half = MatLaw74(id=4, rho=7.8e-9, e=210000.0, nu=0.5, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
    log4 = MessageLog()
    check_mat_law74(mat=m_nu_half, log=log4)
    assert any("ANCMSG 1514" in str(err) for err in log4.errors)

    # 4. Invalid directional yield stresses (ANCMSG 822)
    for s_name in ("s11y", "s22y", "s33y", "s12y", "s23y", "s31y"):
        kwargs = {"rho": 7.8e-9, "e": 210000.0, "nu": 0.3, "s11y": 400.0, "s22y": 420.0, "s33y": 450.0, "s12y": 230.0, "s23y": 240.0, "s31y": 250.0}
        kwargs[s_name] = 0.0
        m_s = MatLaw74(id=5, **kwargs)
        log5 = MessageLog()
        check_mat_law74(mat=m_s, log=log5)
        assert any(f"yield stress parameter {s_name.upper()} must be > 0" in str(err) and "ANCMSG 822" in str(err) for err in log5.errors)

    # 5. Invalid failure strains epsr1 >= epsr2 (ANCMSG 1044)
    m_epsr = MatLaw74(id=6, rho=7.8e-9, e=210000.0, nu=0.3, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0,
                      epsr1=0.35, epsr2=0.2)
    log6 = MessageLog()
    check_mat_law74(mat=m_epsr, log=log6)
    assert any("ANCMSG 1044" in str(err) for err in log6.errors)


def test_starter_checks_element_compatibility():
    """Verify LAW74 rejects shells (ANCMSG 305), 1D elements (ANCMSG 306), 2D analysis (ANCMSG 305), and accepts solids."""
    class FakeElement:
        def __init__(self, mat_id):
            self.mat_id = mat_id

    class FakeElementGroup:
        def __init__(self, elements):
            self._elements = elements

        def values(self):
            return self._elements

    class FakePart:
        def __init__(self, mat_id, elem_type):
            self.mat_id = mat_id
            self.elem_type = elem_type

    m = MatLaw74(id=74, rho=7.8e-9, e=210000.0, nu=0.3, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)

    # 1. Shell element rejection (ANCMSG 305)
    model_shell = Model()
    model_shell.materials[74] = m
    model_shell.element_groups = lambda: [("shells", FakeElementGroup([FakeElement(74)]))]
    log_shell = MessageLog()
    check_mat_law74(model=model_shell, mat_id=74, log=log_shell)
    assert any("ANCMSG 305" in str(err) and "shell elements" in str(err) for err in log_shell.errors)

    # 2. 1D element rejection (ANCMSG 306)
    model_beam = Model()
    model_beam.materials[74] = m
    model_beam.element_groups = lambda: [("beams", FakeElementGroup([FakeElement(74)]))]
    log_beam = MessageLog()
    check_mat_law74(model=model_beam, mat_id=74, log=log_beam)
    assert any("ANCMSG 306" in str(err) and "1D elements" in str(err) for err in log_beam.errors)

    # 3. 2D analysis rejection (ANCMSG 305)
    model_2d = Model()
    model_2d.n2d = 1
    model_2d.materials[74] = m
    log_2d = MessageLog()
    check_mat_law74(model=model_2d, mat_id=74, log=log_2d)
    assert any("ANCMSG 305" in str(err) and "2D analysis" in str(err) for err in log_2d.errors)

    # 4. Solid element acceptance
    model_solid = Model()
    model_solid.materials[74] = m
    model_solid.element_groups = lambda: [("bricks", FakeElementGroup([FakeElement(74)]))]
    log_solid = MessageLog()
    check_mat_law74(model=model_solid, mat_id=74, log=log_solid)
    assert len(log_solid.errors) == 0


def test_starter_checks_model_dispatch():
    """Verify check_materials and check_model invoke check_mat_law74."""
    model = Model()
    m = MatLaw74(id=74, rho=-1.0, e=210000.0, nu=0.3, s11y=400.0, s22y=420.0, s33y=450.0, s12y=230.0, s23y=240.0, s31y=250.0)
    model.materials[74] = m
    log = MessageLog()

    check_materials(model, log)
    assert any("/MAT/LAW74/74: initial density RHO must be > 0" in str(err) for err in log.errors)

    # In _MAT_CHECKS
    assert 74 in _MAT_CHECKS
    assert "LAW74" in _MAT_CHECKS
    assert "HILL_3D" in _MAT_CHECKS
    assert "ORTH_PLAS" in _MAT_CHECKS
    assert _MAT_CHECKS[74] is check_mat_law74


# ============================================================================
# 5. StarterDeck Writer & Roundtrip Verification
# ============================================================================

def test_starter_deck_writer_and_roundtrip(tmp_path):
    """Verify StarterDeck.mat_law74 and alias methods write cards that parse back identically."""
    deck = StarterDeck("Test LAW74 Roundtrip")
    deck.mat_law74(
        1,
        "Law 74 Primary",
        rho=7.8e-9,
        e=210000.0,
        nu=0.30,
        eps_max=0.5,
        epsr1=0.2,
        epsr2=0.35,
        ifunce=2,
        einf=180000.0,
        ce=10.0,
        fsmooth=1,
        chard=0.25,
        fcut=5000.0,
        s11y=400.0,
        s22y=420.0,
        s33y=450.0,
        s12y=230.0,
        s23y=240.0,
        s31y=250.0,
        table_id=201,
        fscale=1.1,
        pscale=0.9,
        t0=295.0,
        rhocp=3.5e-3,
    )
    deck.mat_hill_3d(
        2,
        "Alias Hill 3D",
        rho=2.7e-9,
        e=70000.0,
        nu=0.33,
        eps_max=0.6,
        epsr1=0.25,
        epsr2=0.4,
        ifunce=0,
        einf=0.0,
        ce=0.0,
        fsmooth=0,
        chard=0.0,
        fcut=0.0,
        s11y=150.0,
        s22y=160.0,
        s33y=170.0,
        s12y=85.0,
        s23y=90.0,
        s31y=95.0,
        table_id=202,
        fscale=1.0,
        pscale=1.0,
        t0=293.0,
        rhocp=2.4e-3,
    )
    deck.mat_orth_plas(
        3,
        "Alias Orth Plas",
        rho=7.9e-9,
        e=205000.0,
        nu=0.29,
        eps_max=0.4,
        epsr1=0.15,
        epsr2=0.3,
        ifunce=1,
        einf=175000.0,
        ce=5.0,
        fsmooth=1,
        chard=0.2,
        fcut=4000.0,
        s11y=380.0,
        s22y=390.0,
        s33y=410.0,
        s12y=210.0,
        s23y=220.0,
        s31y=230.0,
        table_id=203,
        fscale=1.05,
        pscale=0.95,
        t0=300.0,
        rhocp=3.8e-3,
    )

    deck_file = tmp_path / "deck_law74_roundtrip_0000.rad"
    deck.write(str(deck_file))

    # Parse back
    blocks = read_deck(str(deck_file))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law74(block, model, log)

    assert len(log.errors) == 0
    assert 1 in model.mat_law74s
    assert 2 in model.mat_law74s
    assert 3 in model.mat_law74s

    m1 = model.mat_law74s[1]
    assert m1.id == 1
    assert m1.title == "Law 74 Primary"
    assert m1.rho == pytest.approx(7.8e-9)
    assert m1.e == pytest.approx(210000.0)
    assert m1.nu == pytest.approx(0.30)
    assert m1.s11y == pytest.approx(400.0)
    assert m1.s22y == pytest.approx(420.0)
    assert m1.s33y == pytest.approx(450.0)
    assert m1.s12y == pytest.approx(230.0)
    assert m1.s23y == pytest.approx(240.0)
    assert m1.s31y == pytest.approx(250.0)
    assert m1.table_id == 201
    assert m1.t0 == pytest.approx(295.0)

    m2 = model.mat_law74s[2]
    assert m2.id == 2
    assert m2.title == "Alias Hill 3D"
    assert m2.rho == pytest.approx(2.7e-9)
    assert m2.e == pytest.approx(70000.0)
    assert m2.s11y == pytest.approx(150.0)
    assert m2.table_id == 202

    m3 = model.mat_law74s[3]
    assert m3.id == 3
    assert m3.title == "Alias Orth Plas"
    assert m3.rho == pytest.approx(7.9e-9)
    assert m3.e == pytest.approx(205000.0)
    assert m3.s11y == pytest.approx(380.0)
    assert m3.table_id == 203


# ============================================================================
# 6. Solid Element State Initialization & Material Extra
# ============================================================================

def test_solid_element_state_initialization():
    """Verify solid_hexa8._init_material_state initializes uvar74 and temp for LAW74."""
    class FakeProp:
        def __init__(self):
            self.params = {}

    class FakeGroup:
        def __init__(self, mat, prop):
            self.n = 4
            self.state = {
                "slices": [(slice(0, 4), mat, prop)],
                "mat_extra": {},
            }

    mat74 = Material(
        id=1,
        law=74,
        law_name="LAW74",
        rho0=7.8e-9,
        params={
            "e": 210000.0,
            "nu": 0.30,
            "s11y": 400.0,
            "s22y": 420.0,
            "s33y": 450.0,
            "s12y": 230.0,
            "s23y": 240.0,
            "s31y": 250.0,
            "t0": 305.0,
            "eps_max": 0.5,
        },
    )
    prop = FakeProp()
    grp = FakeGroup(mat74, prop)
    dndx0 = np.zeros((4, 8, 3))

    _init_material_state(grp, dndx0)

    st = grp.state
    assert "mat_extra" in st
    assert "uvar74" in st["mat_extra"]
    assert st["mat_extra"]["uvar74"].shape == (4, 10)
    assert "temp" in st["mat_extra"]
    assert np.all(st["mat_extra"]["temp"] == pytest.approx(305.0))
    assert st["chk_fail"] is True


# ============================================================================
# 7. Material Dispatch Integration
# ============================================================================

def test_materials_dispatch_integration():
    """Verify materials package dispatch, solid update, tangents, and sound speed for LAW74."""
    mat = Material(
        id=1,
        law=74,
        law_name="LAW74",
        rho0=7.8e-9,
        params={
            "e": 210000.0,
            "nu": 0.30,
            "s11y": 400.0,
            "s22y": 420.0,
            "s33y": 450.0,
            "s12y": 230.0,
            "s23y": 240.0,
            "s31y": 250.0,
        },
    )

    # Solid dispatch registered
    for synonym in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS",
                    "MAT_LAW74", "MAT_HILL_3D", "MAT_ORTH_PLAS"):
        assert synonym in materials.MATERIAL_SOLID_DISPATCH

    # Shell update rejected
    with pytest.raises(NotImplementedError, match="solid elements only"):
        materials.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)))

    # Solid update executes
    sig = np.zeros((2, 6))
    deps = np.zeros((2, 6))
    deps[:, 0] = 0.001
    uvar74 = np.zeros((2, 10))
    extra = {"uvar74": uvar74, "rho": np.full(2, 7.8e-9)}

    sig_out, epsp_out, c_out = materials.solid_update(
        mat, sig, deps, epsp=np.zeros(2), dt=1.0e-6, extra=extra
    )
    assert sig_out.shape == (2, 6)
    assert epsp_out.shape == (2,)
    assert c_out.shape == (2,)

    # Solid tangent
    c_tan = materials.solid_tangent(mat, sig=np.zeros((2, 6)))
    assert c_tan.shape == (2, 6, 6)

    # Sound speed
    c_snd = materials.sound_speed(mat, rho=7.8e-9)
    expected_k = 210000.0 / (3.0 * (1.0 - 2.0 * 0.3))
    expected_g = 210000.0 / (2.0 * (1.0 + 0.3))
    expected_c = math.sqrt((expected_k + 4.0 / 3.0 * expected_g) / 7.8e-9)
    assert c_snd == pytest.approx(expected_c)
