"""Tests for Milestone M559: /MAT/LAW50 (/MAT/VISC_HONEY, /MAT/HYP_FOAM) Starter & Model Integration.

Validates:
1. MatLaw50 (and MatViscHoney, MatHypFoam) dataclass fields, property helpers,
   and mapping protocol (__getitem__, __setitem__, __contains__, get).
2. Model containers (mat_law50s, mat_visc_honeys, mat_hyp_foams).
3. Card layouts (MAT_LAW50_1..25, MAT_VISC_HONEY_1..25, MAT_HYP_FOAM_1..25) matching mat_law50.cfg.
4. CFG catalogue mapping (LAW50, VISC_HONEY, HYP_FOAM -> 50).
5. Starter keyword reader for fixed (24 and 25 cards) and free format, default irate=2.
6. StarterDeck writer for fixed format and helper aliases (mat_visc_honey, mat_hyp_foam).
7. Starter diagnostic checks: parameter bounds (rho, moduli > 0, compaction bounds),
   2D analysis rejection (ANCMSG 305), shell rejection (ANCMSG 305), 1D rejection (ANCMSG 306),
   and solid element compatibility.
8. Solid element integration (solid_hexa8, solid_tetra4) with sound speed and failure/deletion.
"""

from __future__ import annotations

import math
import os
import numpy as np
import pytest

from pyradioss.model.entities import MatLaw50, MatViscHoney, MatHypFoam, Material
from pyradioss.model.model import Model
from pyradioss.input.card_layouts import CARD_LAYOUTS
from pyradioss.input.cfg_catalogue import law_number, canonical_law_name
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law50, KEYWORD_PARSERS, parse_starter_deck
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.checks import check_mat_law50, check_materials, check_model, _ALLOWED_LAWS, _MAT_CHECKS
from pyradioss.common.messages import MessageLog
from pyradioss.elements import solid_hexa8, solid_tetra4


# ============================================================================
# 1. Model Entities & Properties
# ============================================================================

def test_mat_law50_entities_and_properties():
    """Verify MatLaw50 dataclass fields, property helpers, and mapping protocol."""
    m = MatLaw50(
        id=50,
        rho=1.5e-3,
        refer_rho=1.5e-3,
        ea=100.0,
        eb=200.0,
        ec=300.0,
        gab=40.0,
        gbc=50.0,
        gca=60.0,
        asrate=25.0,
        irate=2,
        gflag=1,
        vflag=-1,
        eps_max11=0.2,
        eps_max22=0.3,
        eps_max33=0.4,
        eps_max12=0.5,
        eps_max23=0.6,
        eps_max31=0.7,
        ecomp=800.0,
        pr=0.25,
        sigy=80.0,
        et=20.0,
        vcomp=0.3,
        title="Honeycomb Core",
    )

    # Aliases
    assert MatViscHoney is MatLaw50
    assert MatHypFoam is MatLaw50

    # Fields & Aliases
    assert m.id == 50
    assert m.rho == pytest.approx(1.5e-3)
    assert m.rho0 == pytest.approx(1.5e-3)
    assert m.refer_rho == pytest.approx(1.5e-3)
    assert m.rhor == pytest.approx(1.5e-3)
    assert m.e11 == pytest.approx(100.0)
    assert m.e22 == pytest.approx(200.0)
    assert m.e33 == pytest.approx(300.0)
    assert m.g12 == pytest.approx(40.0)
    assert m.g23 == pytest.approx(50.0)
    assert m.g31 == pytest.approx(60.0)
    assert m.nu == pytest.approx(0.25)
    assert m.pr == pytest.approx(0.25)
    assert m.hcomp == pytest.approx(20.0)
    assert m.et == pytest.approx(20.0)

    # Derived Moduli
    assert m.E == pytest.approx(300.0)
    assert m.G == pytest.approx(60.0)
    expected_gcomp = 800.0 / (1.0 + 0.25)
    assert m.gcomp == pytest.approx(expected_gcomp)
    expected_bulk = 800.0 / (3.0 * (1.0 - 2.0 * 0.25))
    assert m.bulk == pytest.approx(expected_bulk)
    assert m.K == pytest.approx(expected_bulk)

    # Sound Speed (uncompacted state: sqrt(max(E, G) / rho))
    expected_c_solid = math.sqrt(max(m.ea, m.eb, m.ec, m.gab, m.gbc, m.gca) / 1.5e-3)
    assert m.sound_speed_solid == pytest.approx(expected_c_solid)
    assert m.sound_speed_solid() == pytest.approx(expected_c_solid)

    # Setters
    m.nu = 0.30
    assert m.pr == pytest.approx(0.30)
    m.hcomp = 30.0
    assert m.et == pytest.approx(30.0)

    # Mapping protocol
    assert "rho" in m
    assert "ea" in m
    assert "gflag" in m
    assert "ecomp" in m
    assert m["ea"] == pytest.approx(100.0)
    assert m.get("vcomp") == pytest.approx(0.3)
    assert m.get("missing_key", 99.0) == 99.0
    m["custom_param"] = 123.4
    assert m["custom_param"] == pytest.approx(123.4)


def test_model_mat_law50_containers():
    """Verify Model contains mat_law50s dict and aliases."""
    model = Model()
    assert hasattr(model, "mat_law50s")
    assert hasattr(model, "mat_visc_honeys")
    assert hasattr(model, "mat_hyp_foams")
    assert model.mat_visc_honeys is model.mat_law50s
    assert model.mat_hyp_foams is model.mat_law50s


# ============================================================================
# 2. Card Layouts & Catalogue Mappings
# ============================================================================

def test_card_layouts_law50():
    """Verify card layouts MAT_LAW50_1..25 and aliases."""
    for prefix in ("MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM"):
        # Cards 1..3
        assert CARD_LAYOUTS[f"{prefix}_1"] == [20, 20]
        assert CARD_LAYOUTS[f"{prefix}_2"] == [20, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_3"] == [20, 20, 20]
        # Card 4: asrate, irate
        assert CARD_LAYOUTS[f"{prefix}_4"] == [20, 10]
        # Card 5: gflag, eps_max11, eps_max22, eps_max33
        assert CARD_LAYOUTS[f"{prefix}_5"] == [10, 20, 20, 20]
        # Card 25: ecomp, pr, sigy, et, vcomp
        assert CARD_LAYOUTS[f"{prefix}_25"] == [20, 20, 20, 20, 20]


def test_cfg_catalogue_law50():
    """Verify canonical law name and law number for LAW50 and its aliases."""
    for name in ("LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM"):
        assert law_number(name) == 50
    assert canonical_law_name("VISC_HONEY") in ("LAW50", "VISC_HONEY")
    assert canonical_law_name("HYP_FOAM") in ("LAW50", "VISC_HONEY", "HYP_FOAM")


# ============================================================================
# 3. Starter Keyword Reader (Fixed & Free Formats)
# ============================================================================

def test_starter_keyword_reader_fixed_uncompacted(tmp_path):
    """Verify reading fixed-format 24-card uncompacted LAW50 deck."""
    deck_text = (
        "/MAT/LAW50/1\n"
        "Uncompacted Honeycomb\n"
        "               0.001                 0.0\n"
        "               100.0               200.0               300.0\n"
        "                40.0                50.0                60.0\n"
        "                 0.0                    \n"  # asrate=0, irate=blank -> default 2
        "         0               0.2                 0.3                 0.4\n"
        "         0         0         0         0         0\n"
        "                 1.0                 1.0                 1.0                 1.0                 1.0\n"
        "                 0.0                 0.0                 0.0                 0.0                 0.0\n"
        "         0         0         0         0         0\n"
        "                 1.0                 1.0                 1.0                 1.0                 1.0\n"
        "                 0.0                 0.0                 0.0                 0.0                 0.0\n"
        "         0         0         0         0         0\n"
        "                 1.0                 1.0                 1.0                 1.0                 1.0\n"
        "                 0.0                 0.0                 0.0                 0.0                 0.0\n"
        "         0               0.5                 0.6                 0.7\n"
        "         0         0         0         0         0\n"
        "                 1.0                 1.0                 1.0                 1.0                 1.0\n"
        "                 0.0                 0.0                 0.0                 0.0                 0.0\n"
        "         0         0         0         0         0\n"
        "                 1.0                 1.0                 1.0                 1.0                 1.0\n"
        "                 0.0                 0.0                 0.0                 0.0                 0.0\n"
        "         0         0         0         0         0\n"
        "                 1.0                 1.0                 1.0                 1.0                 1.0\n"
        "                 0.0                 0.0                 0.0                 0.0                 0.0\n"
        "/END\n"
    )

    deck_file = tmp_path / "deck_uncomp_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law50(block, model, log)

    assert len(log.errors) == 0
    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 50
    assert mat.rho0 == pytest.approx(0.001)
    assert mat.params["ea"] == pytest.approx(100.0)
    assert mat.params["eb"] == pytest.approx(200.0)
    assert mat.params["ec"] == pytest.approx(300.0)
    assert mat.params["gab"] == pytest.approx(40.0)
    assert mat.params["gbc"] == pytest.approx(50.0)
    assert mat.params["gca"] == pytest.approx(60.0)
    assert mat.params["irate"] == 2  # default
    assert mat.params["icompact"] == 0


def test_starter_keyword_reader_fixed_compacted(tmp_path):
    """Verify reading fixed-format 25-card compacted LAW50 deck."""
    deck_text = (
        "/MAT/VISC_HONEY/2\n"
        "Compacted Honeycomb\n"
        "               0.001                 0.0\n"
        "               100.0               200.0               300.0\n"
        "                40.0                50.0                60.0\n"
        "                10.0         1          \n"  # asrate=10, irate=1
        "         1               0.2                 0.3                 0.4\n"
        "       101       102         0         0         0\n"
        "                 1.5                 2.0                 1.0                 1.0                 1.0\n"
        "                 0.0                10.0                 0.0                 0.0                 0.0\n"
        "         0         0         0         0         0\n"
        "                 1.0                 1.0                 1.0                 1.0                 1.0\n"
        "                 0.0                 0.0                 0.0                 0.0                 0.0\n"
        "         0         0         0         0         0\n"
        "                 1.0                 1.0                 1.0                 1.0                 1.0\n"
        "                 0.0                 0.0                 0.0                 0.0                 0.0\n"
        "        -1               0.5                 0.6                 0.7\n"
        "       201         0         0         0         0\n"
        "                 1.2                 1.0                 1.0                 1.0                 1.0\n"
        "                 0.0                 0.0                 0.0                 0.0                 0.0\n"
        "         0         0         0         0         0\n"
        "                 1.0                 1.0                 1.0                 1.0                 1.0\n"
        "                 0.0                 0.0                 0.0                 0.0                 0.0\n"
        "         0         0         0         0         0\n"
        "                 1.0                 1.0                 1.0                 1.0                 1.0\n"
        "                 0.0                 0.0                 0.0                 0.0                 0.0\n"
        "               500.0                0.25                50.0                10.0                 0.2\n"
        "/END\n"
    )

    deck_file = tmp_path / "deck_comp_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law50(block, model, log)

    assert len(log.errors) == 0
    assert 2 in model.materials
    mat = model.materials[2]
    assert mat.params["irate"] == 1
    assert mat.params["gflag"] == 1
    assert mat.params["vflag"] == -1
    assert mat.params["ecomp"] == pytest.approx(500.0)
    assert mat.params["pr"] == pytest.approx(0.25)
    assert mat.params["sigy"] == pytest.approx(50.0)
    assert mat.params["et"] == pytest.approx(10.0)
    assert mat.params["vcomp"] == pytest.approx(0.2)
    assert mat.params["icompact"] == 1
    assert mat.params["yfun11"][0] == 101
    assert mat.params["yfun11"][1] == 102
    assert mat.params["sfac11"][0] == pytest.approx(1.5)


def test_starter_keyword_reader_free_format(tmp_path):
    """Verify free-format card parsing with comma separation."""
    deck_text = (
        "/MAT/HYP_FOAM/3\n"
        "Free Format Foam\n"
        "0.002, 0.0\n"
        "150.0, 250.0, 350.0\n"
        "45.0, 55.0, 65.0\n"
        "15.0, 2\n"
        "0, 0.25, 0.35, 0.45\n"
        "0, 0, 0, 0, 0\n"
        "1.0, 1.0, 1.0, 1.0, 1.0\n"
        "0.0, 0.0, 0.0, 0.0, 0.0\n"
        "0, 0, 0, 0, 0\n"
        "1.0, 1.0, 1.0, 1.0, 1.0\n"
        "0.0, 0.0, 0.0, 0.0, 0.0\n"
        "0, 0, 0, 0, 0\n"
        "1.0, 1.0, 1.0, 1.0, 1.0\n"
        "0.0, 0.0, 0.0, 0.0, 0.0\n"
        "0, 0.55, 0.65, 0.75\n"
        "0, 0, 0, 0, 0\n"
        "1.0, 1.0, 1.0, 1.0, 1.0\n"
        "0.0, 0.0, 0.0, 0.0, 0.0\n"
        "0, 0, 0, 0, 0\n"
        "1.0, 1.0, 1.0, 1.0, 1.0\n"
        "0.0, 0.0, 0.0, 0.0, 0.0\n"
        "0, 0, 0, 0, 0\n"
        "1.0, 1.0, 1.0, 1.0, 1.0\n"
        "0.0, 0.0, 0.0, 0.0, 0.0\n"
        "600.0, 0.2, 60.0, 15.0, 0.25\n"
        "/END\n"
    )

    deck_file = tmp_path / "deck_free_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law50(block, model, log)

    assert len(log.errors) == 0
    assert 3 in model.materials
    mat = model.materials[3]
    assert mat.rho0 == pytest.approx(0.002)
    assert mat.params["ea"] == pytest.approx(150.0)
    assert mat.params["eb"] == pytest.approx(250.0)
    assert mat.params["ec"] == pytest.approx(350.0)
    assert mat.params["ecomp"] == pytest.approx(600.0)
    assert mat.params["vcomp"] == pytest.approx(0.25)


# ============================================================================
# 4. StarterDeck Writer & Roundtrip
# ============================================================================

def test_deck_writer_roundtrip(tmp_path):
    """Verify StarterDeck.mat_law50 and aliases roundtrip through file."""
    deck = StarterDeck("M559_ROUNDTRIP")
    deck.mat_law50(
        mat_id=1,
        rho=0.001,
        ea=110.0,
        eb=210.0,
        ec=310.0,
        gab=41.0,
        gbc=51.0,
        gca=61.0,
        asrate=12.0,
        irate=1,
        gflag=1,
        vflag=-1,
        eps_max11=0.21,
        eps_max22=0.31,
        eps_max33=0.41,
        eps_max12=0.51,
        eps_max23=0.61,
        eps_max31=0.71,
        yfun11=[10, 20],
        sfac11=[1.1, 2.2],
        eps11=[0.0, 5.0],
        ecomp=700.0,
        pr=0.22,
        sigy=75.0,
        et=12.0,
        vcomp=0.28,
        title="MatLaw50 Write",
    )

    deck.mat_visc_honey(
        mat_id=2,
        rho=0.002,
        ea=120.0,
        eb=220.0,
        ec=320.0,
        gab=42.0,
        gbc=52.0,
        gca=62.0,
        title="MatViscHoney Write",
    )

    deck.mat_hyp_foam(
        mat_id=3,
        rho=0.003,
        ea=130.0,
        eb=230.0,
        ec=330.0,
        gab=43.0,
        gbc=53.0,
        gca=63.0,
        title="MatHypFoam Write",
    )

    out_file = os.path.join(tmp_path, "TEST_0000.rad")
    deck.write(out_file)

    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(out_file), model, log)

    assert len(log.errors) == 0
    assert 1 in model.materials
    assert 2 in model.materials
    assert 3 in model.materials

    m1 = model.materials[1]
    assert m1.law == 50
    assert m1.rho0 == pytest.approx(0.001)
    assert m1.params["ea"] == pytest.approx(110.0)
    assert m1.params["irate"] == 1
    assert m1.params["ecomp"] == pytest.approx(700.0)
    assert m1.params["yfun11"][0] == 10
    assert m1.params["yfun11"][1] == 20

    m2 = model.materials[2]
    assert m2.law == 50
    assert m2.params["eb"] == pytest.approx(220.0)

    m3 = model.materials[3]
    assert m3.law == 50
    assert m3.params["ec"] == pytest.approx(330.0)


# ============================================================================
# 5. Diagnostic Checks (Bounds & Incompatibilities)
# ============================================================================

def test_starter_checks_valid():
    """Verify valid LAW50 material passes check_mat_law50 without errors."""
    m = MatLaw50(
        id=1,
        rho=1e-3,
        ea=100.0,
        eb=100.0,
        ec=100.0,
        gab=40.0,
        gbc=40.0,
        gca=40.0,
        ecomp=500.0,
        pr=0.25,
        sigy=50.0,
        vcomp=0.3,
    )
    log = MessageLog()
    check_mat_law50(mat=m, log=log)
    assert len(log.errors) == 0


def test_starter_checks_density():
    """Verify check_mat_law50 rejects zero or negative density."""
    m_zero = MatLaw50(id=1, rho=0.0, ea=100.0, eb=100.0, ec=100.0, gab=40.0, gbc=40.0, gca=40.0)
    log = MessageLog()
    check_mat_law50(mat=m_zero, log=log)
    assert any("initial density RHO must be > 0" in e for e in log.errors)

    m_neg = MatLaw50(id=2, rho=-0.005, ea=100.0, eb=100.0, ec=100.0, gab=40.0, gbc=40.0, gca=40.0)
    log = MessageLog()
    check_mat_law50(mat=m_neg, log=log)
    assert any("initial density RHO must be > 0" in e for e in log.errors)


def test_starter_checks_moduli():
    """Verify check_mat_law50 requires positive EA, EB, EC, GAB, GBC, GCA (ANCMSG 306)."""
    for mod_name in ("ea", "eb", "ec", "gab", "gbc", "gca"):
        kwargs = dict(id=10, rho=1e-3, ea=100.0, eb=100.0, ec=100.0, gab=40.0, gbc=40.0, gca=40.0)
        kwargs[mod_name] = 0.0
        m = MatLaw50(**kwargs)
        log = MessageLog()
        check_mat_law50(mat=m, log=log)
        assert any(f"elastic modulus {mod_name.upper()} must be > 0" in e and "ANCMSG 306" in e for e in log.errors)


def test_starter_checks_compaction_bounds():
    """Verify check_mat_law50 compaction parameter validation."""
    # ECOMP <= 0
    m_ecomp = MatLaw50(id=1, rho=1e-3, ea=100.0, eb=100.0, ec=100.0, gab=40.0, gbc=40.0, gca=40.0,
                        ecomp=0.0, pr=0.25, sigy=50.0, vcomp=0.3)
    log = MessageLog()
    check_mat_law50(mat=m_ecomp, log=log)
    assert any("compacted Young's modulus ECOMP must be > 0" in e for e in log.errors)

    # PR < 0 or >= 0.5
    m_pr_high = MatLaw50(id=2, rho=1e-3, ea=100.0, eb=100.0, ec=100.0, gab=40.0, gbc=40.0, gca=40.0,
                         ecomp=500.0, pr=0.5, sigy=50.0, vcomp=0.3)
    log = MessageLog()
    check_mat_law50(mat=m_pr_high, log=log)
    assert any("compacted Poisson's ratio NU/PR must satisfy 0 <= NU < 0.5" in e for e in log.errors)

    m_pr_neg = MatLaw50(id=3, rho=1e-3, ea=100.0, eb=100.0, ec=100.0, gab=40.0, gbc=40.0, gca=40.0,
                        ecomp=500.0, pr=-0.1, sigy=50.0, vcomp=0.3)
    log = MessageLog()
    check_mat_law50(mat=m_pr_neg, log=log)
    assert any("compacted Poisson's ratio NU/PR must satisfy 0 <= NU < 0.5" in e for e in log.errors)

    # VCOMP <= 0 or > 1.0
    m_vcomp_zero = MatLaw50(id=4, rho=1e-3, ea=100.0, eb=100.0, ec=100.0, gab=40.0, gbc=40.0, gca=40.0,
                           ecomp=500.0, pr=0.25, sigy=50.0, vcomp=0.0)
    log = MessageLog()
    check_mat_law50(mat=m_vcomp_zero, log=log)
    assert any("compaction volume fraction VCOMP must satisfy 0 < VCOMP <= 1" in e for e in log.errors)

    m_vcomp_high = MatLaw50(id=5, rho=1e-3, ea=100.0, eb=100.0, ec=100.0, gab=40.0, gbc=40.0, gca=40.0,
                           ecomp=500.0, pr=0.25, sigy=50.0, vcomp=1.2)
    log = MessageLog()
    check_mat_law50(mat=m_vcomp_high, log=log)
    assert any("compaction volume fraction VCOMP must satisfy 0 < VCOMP <= 1" in e for e in log.errors)


def test_starter_checks_2d_analysis_rejection():
    """Verify check_model rejects LAW50 when n2d > 0 (ANCMSG 305)."""
    model = Model()
    model.n2d = 1
    model.node_ids = np.array([1, 2, 3, 4])
    m = MatLaw50(id=1, rho=1e-3, ea=100.0, eb=100.0, ec=100.0, gab=40.0, gbc=40.0, gca=40.0)
    model.materials[1] = m
    log = MessageLog()
    check_model(model, log)
    assert any("LAW50 is not supported for 2D analysis" in e and "ANCMSG 305" in e for e in log.errors)


def test_starter_checks_allowed_laws():
    """Verify 50 and aliases are in _ALLOWED_LAWS for solid elements and _MAT_CHECKS."""
    for el_type in ("bricks", "tetras", "penta6", "pyra5"):
        allowed = _ALLOWED_LAWS[el_type]
        for key in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM"):
            assert key in allowed, f"{key} missing in _ALLOWED_LAWS[{el_type}]"

    for key in (50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM"):
        assert key in _MAT_CHECKS, f"{key} missing in _MAT_CHECKS"


# ============================================================================
# 6. Element Kernel Time Step & Update Integration
# ============================================================================

def test_solid_hexa8_law50_dt_and_sound_speed():
    """Verify solid_hexa8 computes valid critical time step and sound speed for LAW50."""
    class DummyGroup:
        def __init__(self):
            self.n = 1
            self.conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
            self.state = {
                "slices": [],
                "off": np.array([1.0]),
                "mass": np.array([1.0e-3]),
                "sig": np.zeros((1, 6)),
                "epsp": np.zeros(1),
                "eint": np.zeros(1),
                "ehour": np.zeros(1),
                "vol0": np.array([1.0]),
                "dtfac": np.array([0.9]),
                "lc_scale": np.array([1.0]),
                "mat_extra": {"off50": np.array([1.0])},
                "chk_fail": False,
            }

    # Cube [0, 1]^3
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])

    mat = Material(id=1, rho0=1.0e-3, law=50)
    mat.params = {
        "rho": 1.0e-3, "rho0": 1.0e-3,
        "ea": 100.0, "eb": 200.0, "ec": 300.0,
        "gab": 40.0, "gbc": 50.0, "gca": 60.0,
    }

    grp = DummyGroup()
    grp.state["slices"] = [(slice(0, 1), mat, None)]

    fint = np.zeros_like(x)
    mint = np.zeros_like(x)
    dt_crit = solid_hexa8.forces(grp, x=x, v=None, vr=None, dt=0.0, fint=fint, mint=mint)
    assert len(dt_crit) == 1
    assert dt_crit[0] > 0.0
    assert not np.isnan(dt_crit[0])
    assert not np.isinf(dt_crit[0])


def test_solid_tetra4_law50_dt_and_sound_speed():
    """Verify solid_tetra4 computes valid critical time step and sound speed for LAW50."""
    class DummyTetGroup:
        def __init__(self):
            self.n = 1
            self.conn = np.array([[0, 1, 2, 3]])
            self.state = {
                "slices": [],
                "off": np.array([1.0]),
                "mass": np.array([1.0e-3 / 6.0]),
                "sig": np.zeros((1, 6)),
                "epsp": np.zeros(1),
                "eint": np.zeros(1),
                "ehour": np.zeros(1),
                "vol0": np.array([1.0 / 6.0]),
                "dtfac": np.array([0.9]),
                "j_prev": np.array([1.0]),
                "qvw_pend": np.zeros(1),
                "mat_extra": {"off50": np.array([1.0])},
                "chk_fail": False,
            }

    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    mat = Material(id=1, rho0=1.0e-3, law=50)
    mat.params = {
        "rho": 1.0e-3, "rho0": 1.0e-3,
        "ea": 100.0, "eb": 200.0, "ec": 300.0,
        "gab": 40.0, "gbc": 50.0, "gca": 60.0,
    }

    grp = DummyTetGroup()
    grp.state["slices"] = [(slice(0, 1), mat, None)]

    v = np.zeros((4, 3))
    fint = np.zeros_like(x)
    mint = np.zeros_like(x)
    dt_crit = solid_tetra4.forces(grp, x=x, v=v, vr=None, dt=1.0e-5, fint=fint, mint=mint)
    assert len(dt_crit) == 1
    assert dt_crit[0] > 0.0
    assert not np.isnan(dt_crit[0])
    assert not np.isnan(dt_crit[0])
