"""Tests for Milestone M560: /MAT/LAW163 (/MAT/CRUSHABLE_FOAM, /MAT/CRUSH_FOAM) Starter & Model Integration.

Validates:
1. MatLaw163 (and MatCrushableFoam, MatCrushFoam) dataclass fields, property helpers,
   and mapping protocol (__getitem__, __setitem__, __contains__, get, keys, values, items).
2. Model containers (mat_law163s, mat_crushable_foams, mat_crush_foams).
3. Card layouts (MAT_LAW163_1..3, MAT_CRUSHABLE_FOAM_1..3, MAT_CRUSH_FOAM_1..3) matching matl163_crushable_foam.cfg.
4. CFG catalogue mapping (LAW163, CRUSHABLE_FOAM, CRUSH_FOAM -> 163).
5. Starter keyword reader for fixed and free formats, populating model.mat_law163s and model.materials.
6. Starter diagnostic checks: parameter bounds (rho > 0, E > 0, 0 <= nu < 0.5 with ANCMSG 1514, damp >= 0),
   2D analysis rejection (ANCMSG 305), shell rejection (ANCMSG 305), 1D rejection (ANCMSG 306),
   and solid element compatibility.
7. StarterDeck writer for fixed and free format and helper aliases (mat_crushable_foam, mat_crush_foam) with roundtrip.
8. Law 163 constitutive physics and solid element integration (solid_hexa8, solid_tetra4).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import MatLaw163, MatCrushableFoam, MatCrushFoam, Material
from pyradioss.model.model import Model
from pyradioss.input.card_layouts import CARD_LAYOUTS
from pyradioss.input.cfg_catalogue import law_number, canonical_law_name
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law163
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.checks import (
    check_mat_law163,
    check_materials,
    check_model,
    _ALLOWED_LAWS,
    _MAT_CHECKS,
)
from pyradioss.common.messages import MessageLog
from pyradioss.materials import law163_crush_foam
from pyradioss.elements import solid_hexa8, solid_tetra4


# ============================================================================
# 1. Model Entities & Properties
# ============================================================================

def test_mat_law163_entities_and_properties():
    """Verify MatLaw163 dataclass fields, property helpers, and mapping protocol."""
    m = MatLaw163(
        id=163,
        rho=1.2e-3,
        e=250.0,
        nu=0.25,
        tsc=5.0,
        damp=0.15,
        ncycle=10,
        tab_id=101,
        epsd_ref=0.01,
        fscale=1.5,
        srclmt=1.0e15,
        nrs=1,
        title="Foam Pad",
    )

    # Aliases
    assert MatCrushableFoam is MatLaw163
    assert MatCrushFoam is MatLaw163

    # Primary fields & defaults
    assert m.id == 163
    assert m.title == "Foam Pad"
    assert m.law == 163
    assert m.law_name == "LAW163"
    assert m.rho == pytest.approx(1.2e-3)
    assert m.rho0 == pytest.approx(1.2e-3)
    assert m.e == pytest.approx(250.0)
    assert m.E == pytest.approx(250.0)
    assert m.nu == pytest.approx(0.25)
    assert m.tsc == pytest.approx(5.0)
    assert m.damp == pytest.approx(0.15)
    assert m.ncycle == 10
    assert m.tab_id == 101
    assert m.epsd_ref == pytest.approx(0.01)
    assert m.fscale == pytest.approx(1.5)
    assert m.srclmt == pytest.approx(1.0e15)
    assert m.nrs == 1

    # Derived Moduli
    expected_g = 250.0 / (2.0 * (1.0 + 0.25))
    expected_bulk = 250.0 / (3.0 * (1.0 - 2.0 * 0.25))
    expected_cii = expected_bulk + 4.0 / 3.0 * expected_g
    expected_cij = expected_bulk - 2.0 / 3.0 * expected_g
    expected_c_solid = math.sqrt(expected_cii / 1.2e-3)
    expected_c_shell = math.sqrt(250.0 / (1.2e-3 * (1.0 - 0.25**2)))

    assert m.G == pytest.approx(expected_g)
    assert m.bulk == pytest.approx(expected_bulk)
    assert m.K == pytest.approx(expected_bulk)
    assert m.cii == pytest.approx(expected_cii)
    assert m.cij == pytest.approx(expected_cij)
    assert m.sound_speed == pytest.approx(expected_c_shell)
    assert m.sound_speed_solid == pytest.approx(expected_c_solid)
    assert m.sound_speed_solid() == pytest.approx(expected_c_solid)

    # Mapping protocol
    assert "rho" in m
    assert "e" in m
    assert "nu" in m
    assert "tab_id" in m
    assert "damp" in m
    assert m["e"] == pytest.approx(250.0)
    assert m.get("tsc") == pytest.approx(5.0)
    assert m.get("missing_key", 99.0) == 99.0

    # Key/value setting
    m["custom_param"] = 42.0
    assert m["custom_param"] == pytest.approx(42.0)
    assert "custom_param" in m
    assert list(m.keys())[:3] == ["id", "rho", "e"]


def test_model_mat_law163_containers():
    """Verify Model contains mat_law163s dict and aliases."""
    model = Model()
    assert hasattr(model, "mat_law163s")
    assert hasattr(model, "mat_crushable_foams")
    assert hasattr(model, "mat_crush_foams")
    assert model.mat_crushable_foams is model.mat_law163s
    assert model.mat_crush_foams is model.mat_law163s


# ============================================================================
# 2. Card Layouts & Catalogue Mappings
# ============================================================================

def test_card_layouts_and_cfg_catalogue_law163():
    """Verify card layouts and law catalogue mappings for LAW163."""
    for prefix in ("MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM"):
        assert CARD_LAYOUTS[f"{prefix}_1"] == [20]
        assert CARD_LAYOUTS[f"{prefix}_2"] == [20, 20, 20, 20, 10, 10]
        assert CARD_LAYOUTS[f"{prefix}_3"] == [10, 10, 20, 20, 20, 10, 10]

    assert law_number("LAW163") == 163
    assert law_number("CRUSHABLE_FOAM") == 163
    assert law_number("CRUSH_FOAM") == 163
    assert law_number("MAT_LAW163") == 163
    assert law_number("MAT_CRUSHABLE_FOAM") == 163
    assert law_number("MAT_CRUSH_FOAM") == 163

    assert canonical_law_name(163) == "LAW163"
    assert canonical_law_name("CRUSHABLE_FOAM") == "LAW163"
    assert canonical_law_name("CRUSH_FOAM") == "LAW163"


# ============================================================================
# 3. Starter Keyword Reader
# ============================================================================

def test_starter_keyword_reader_fixed(tmp_path):
    """Verify fixed-format /MAT/LAW163 parsing."""
    deck_text = (
        "/MAT/LAW163/1\n"
        "Foam Core Fixed\n"
        "#                 RHO\n"
        "             1.50E-3\n"
        "#                  E                  NU                 TSC                DAMP              NCYCLE\n"
        "             200.000               0.300               4.000               0.120                  15\n"
        "#             TAB_ID            EPSD_REF              FSCALE           SRC_LIMIT                 NRS\n"
        "                  42             0.05000               1.200             1.00E12                   1\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_fixed_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law163(block, model, log)

    assert len(log.errors) == 0
    assert 1 in model.mat_law163s
    assert 1 in model.materials

    m = model.mat_law163s[1]
    assert m.id == 1
    assert m.title == "Foam Core Fixed"
    assert m.rho == pytest.approx(1.5e-3)
    assert m.e == pytest.approx(200.0)
    assert m.nu == pytest.approx(0.30)
    assert m.tsc == pytest.approx(4.0)
    assert m.damp == pytest.approx(0.12)
    assert m.ncycle == 15
    assert m.tab_id == 42
    assert m.epsd_ref == pytest.approx(0.05)
    assert m.fscale == pytest.approx(1.20)
    assert m.srclmt == pytest.approx(1.0e12)
    assert m.nrs == 1

    # Check Material record
    mat = model.materials[1]
    assert mat.id == 1
    assert mat.law == 163
    assert mat.rho0 == pytest.approx(1.5e-3)
    assert mat.params["e"] == pytest.approx(200.0)
    assert mat.params["tab_id"] == 42


def test_starter_keyword_reader_free_and_aliases(tmp_path):
    """Verify free-format /MAT/CRUSHABLE_FOAM and /MAT/CRUSH_FOAM parsing with defaults."""
    deck_text = (
        "/MAT/CRUSHABLE_FOAM/10\n"
        "Crushable Foam Free 5-tokens\n"
        "1.0e-3\n"
        "150.0 0.20 2.5 0.10 12\n"
        "501 0.0 1.0 1.0e20 0\n"
        "/MAT/CRUSH_FOAM/20\n"
        "Crush Foam Free with Blanks\n"
        "2.0e-3\n"
        "300.0 0.35 10.0\n"
        "0 0 0 0\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_free_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law163(block, model, log)

    assert len(log.errors) == 0
    assert 10 in model.mat_law163s
    assert 20 in model.mat_law163s

    m10 = model.mat_law163s[10]
    assert m10.id == 10
    assert m10.title == "Crushable Foam Free 5-tokens"
    assert m10.rho == pytest.approx(1.0e-3)
    assert m10.e == pytest.approx(150.0)
    assert m10.nu == pytest.approx(0.20)
    assert m10.tsc == pytest.approx(2.5)
    assert m10.tab_id == 501
    assert m10.fscale == pytest.approx(1.0)
    assert m10.nrs == 0

    m20 = model.mat_law163s[20]
    assert m20.id == 20
    assert m20.title == "Crush Foam Free with Blanks"
    assert m20.rho == pytest.approx(2.0e-3)
    assert m20.e == pytest.approx(300.0)
    assert m20.nu == pytest.approx(0.35)
    assert m20.tsc == pytest.approx(10.0)
    # Defaults applied
    assert m20.damp == pytest.approx(0.10)
    assert m20.ncycle == 12
    assert m20.fscale == pytest.approx(1.0)
    assert m20.srclmt == pytest.approx(1.0e20)


# ============================================================================
# 4. Starter Validation Checks
# ============================================================================

def test_starter_checks_valid():
    """Verify clean LAW163 material passes validation without errors."""
    m = MatLaw163(id=1, rho=1.0e-3, e=100.0, nu=0.25, damp=0.10)
    log = MessageLog()
    check_mat_law163(mat=m, log=log)
    assert len(log.errors) == 0


def test_starter_checks_parameter_bounds():
    """Verify bounds checks for rho, E, nu, and damp."""
    # 1. Invalid density
    m_rho = MatLaw163(id=1, rho=0.0, e=100.0, nu=0.25)
    log1 = MessageLog()
    check_mat_law163(mat=m_rho, log=log1)
    assert any("initial density RHO must be > 0" in e for e in log1.errors)

    # 2. Invalid Young's modulus
    m_e = MatLaw163(id=2, rho=1.0e-3, e=0.0, nu=0.25)
    log2 = MessageLog()
    check_mat_law163(mat=m_e, log=log2)
    assert any("Young's modulus E must be > 0" in e for e in log2.errors)

    # 3. Invalid Poisson's ratio (negative) -> ANCMSG 1514
    m_nu1 = MatLaw163(id=3, rho=1.0e-3, e=100.0, nu=-0.1)
    log3 = MessageLog()
    check_mat_law163(mat=m_nu1, log=log3)
    assert any("ANCMSG 1514" in e for e in log3.errors)

    # 4. Invalid Poisson's ratio (>= 0.5) -> ANCMSG 1514
    m_nu2 = MatLaw163(id=4, rho=1.0e-3, e=100.0, nu=0.50)
    log4 = MessageLog()
    check_mat_law163(mat=m_nu2, log=log4)
    assert any("ANCMSG 1514" in e for e in log4.errors)

    # 5. Invalid damping (< 0)
    m_damp = MatLaw163(id=5, rho=1.0e-3, e=100.0, nu=0.25, damp=-0.05)
    log5 = MessageLog()
    check_mat_law163(mat=m_damp, log=log5)
    assert any("damping coefficient DAMP must be >= 0" in e for e in log5.errors)


def test_starter_checks_2d_and_elements_compatibility():
    """Verify 2D analysis rejection (ANCMSG 305), shell rejection (ANCMSG 305), and 1D rejection (ANCMSG 306)."""
    # 2D analysis check
    model = Model()
    model.n2d = 1
    model.node_ids = np.array([1, 2, 3, 4])
    m = MatLaw163(id=1, rho=1.0e-3, e=100.0, nu=0.25)
    model.mat_law163s[1] = m
    log = MessageLog()
    check_model(model, log)
    assert any("ANCMSG 305" in e and "2D analysis" in e for e in log.errors)

    # Shell element compatibility
    model3d = Model()
    model3d.n2d = 0
    model3d.node_ids = np.array([1, 2, 3, 4])
    mat = Material(id=10, law=163, rho0=1.0e-3, params={"e": 100.0, "nu": 0.25})
    model3d.materials[10] = mat
    model3d.mat_law163s[10] = m

    class FakeGroup:
        def __init__(self, name, mid):
            self.name = name
            self.state = {"slices": [(slice(0, 1), mat, None)]}
            self.n = 1
        def values(self):
            return []

    # Shell elements -> ANCMSG 305
    model3d.shells = FakeGroup("shells", 10)
    log_shell = MessageLog()
    check_model(model3d, log_shell)
    assert any("ANCMSG 305" in e and "shell elements" in e for e in log_shell.errors)

    # 1D elements -> ANCMSG 306
    model3d_1d = Model()
    model3d_1d.n2d = 0
    model3d_1d.node_ids = np.array([1, 2])
    model3d_1d.materials[10] = mat
    model3d_1d.mat_law163s[10] = m
    model3d_1d.beams = FakeGroup("beams", 10)
    log_beam = MessageLog()
    check_model(model3d_1d, log_beam)
    assert any("ANCMSG 306" in e and "1D elements" in e for e in log_beam.errors)


def test_allowed_laws_and_mat_checks_registration():
    """Verify LAW163 is registered in _MAT_CHECKS and _ALLOWED_LAWS for solids."""
    for key in (163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM"):
        assert key in _MAT_CHECKS
        assert _MAT_CHECKS[key] is check_mat_law163

    for grp in ("bricks", "tetras", "penta6", "pyra5"):
        assert 163 in _ALLOWED_LAWS[grp]
        assert "LAW163" in _ALLOWED_LAWS[grp]
        assert "CRUSHABLE_FOAM" in _ALLOWED_LAWS[grp]

    for grp in ("shells", "shells_qbat", "shells_qeph", "sh3n", "trusses", "beams"):
        assert 163 not in _ALLOWED_LAWS[grp]
        assert "LAW163" not in _ALLOWED_LAWS[grp]


# ============================================================================
# 5. Starter Deck Writer & Roundtrip
# ============================================================================

def test_deck_writer_and_roundtrip(tmp_path):
    """Verify StarterDeck.mat_law163, aliases, and roundtrip parsing."""
    deck = StarterDeck("test_foam")
    deck.mat_law163(
        1,
        "Foam Law 163 Fixed",
        rho=1.2e-3,
        e=220.0,
        nu=0.28,
        tsc=4.5,
        damp=0.15,
        ncycle=14,
        tab_id=202,
        epsd_ref=0.02,
        fscale=1.1,
        srclmt=1.0e16,
        nrs=1,
    )
    deck.mat_crushable_foam(
        2,
        "Foam Alias Crushable",
        rho=1.8e-3,
        e=310.0,
        nu=0.33,
        tsc=8.0,
        damp=0.10,
        ncycle=12,
        tab_id=303,
        epsd_ref=0.0,
        fscale=1.0,
        srclmt=1.0e20,
        nrs=0,
    )
    deck.mat_crush_foam(
        3,
        "Foam Alias Crush",
        rho=0.9e-3,
        e=90.0,
        nu=0.15,
        tsc=1.2,
        damp=0.08,
        ncycle=8,
        tab_id=404,
        epsd_ref=0.01,
        fscale=0.9,
        srclmt=5.0e14,
        nrs=1,
    )

    deck_file = tmp_path / "deck_roundtrip_0000.rad"
    deck.write(str(deck_file))

    # Parse rendered deck back
    blocks = read_deck(str(deck_file))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law163(block, model, log)

    assert len(log.errors) == 0
    assert 1 in model.mat_law163s
    assert 2 in model.mat_law163s
    assert 3 in model.mat_law163s

    m1 = model.mat_law163s[1]
    assert m1.id == 1
    assert m1.title == "Foam Law 163 Fixed"
    assert m1.rho == pytest.approx(1.2e-3)
    assert m1.e == pytest.approx(220.0)
    assert m1.nu == pytest.approx(0.28)
    assert m1.tsc == pytest.approx(4.5)
    assert m1.damp == pytest.approx(0.15)
    assert m1.ncycle == 14
    assert m1.tab_id == 202
    assert m1.epsd_ref == pytest.approx(0.02)
    assert m1.fscale == pytest.approx(1.1)
    assert m1.srclmt == pytest.approx(1.0e16)
    assert m1.nrs == 1

    m2 = model.mat_law163s[2]
    assert m2.id == 2
    assert m2.title == "Foam Alias Crushable"
    assert m2.rho == pytest.approx(1.8e-3)
    assert m2.e == pytest.approx(310.0)
    assert m2.tab_id == 303

    m3 = model.mat_law163s[3]
    assert m3.id == 3
    assert m3.title == "Foam Alias Crush"
    assert m3.rho == pytest.approx(0.9e-3)
    assert m3.e == pytest.approx(90.0)
    assert m3.tab_id == 404


def test_deck_writer_object_unpacking(tmp_path):
    """Verify StarterDeck.mat_law163 unpacks MatLaw163 instance directly."""
    m = MatLaw163(
        id=77,
        rho=1.4e-3,
        e=180.0,
        nu=0.22,
        tsc=3.5,
        damp=0.11,
        ncycle=10,
        tab_id=707,
        epsd_ref=0.01,
        fscale=1.25,
        srclmt=2.0e15,
        nrs=0,
        title="Unpacked Foam",
    )
    deck = StarterDeck("obj_test")
    deck.mat_law163(m)
    deck_file = tmp_path / "deck_obj_0000.rad"
    deck.write(str(deck_file))

    blocks = read_deck(str(deck_file))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law163(block, model, log)

    assert 77 in model.mat_law163s
    m_read = model.mat_law163s[77]
    assert m_read.title == "Unpacked Foam"
    assert m_read.rho == pytest.approx(1.4e-3)
    assert m_read.e == pytest.approx(180.0)
    assert m_read.nu == pytest.approx(0.22)
    assert m_read.tsc == pytest.approx(3.5)
    assert m_read.tab_id == 707


# ============================================================================
# 6. Law 163 Constitutive Physics (sigeps163.F90)
# ============================================================================

def test_law163_elastic_predictor_and_tangent():
    """Verify elastic predictor and algorithmic tangent matrix for LAW163."""
    mat = MatLaw163(id=1, rho=1.0e-3, e=100.0, nu=0.25, damp=0.10)
    # Voigt strain increment: pure volumetric strain deps_xx = -0.001
    deps = np.array([[-0.001, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sig = np.zeros((1, 6))

    # 1. Inviscid elastic response (dt = 0.0)
    extra = {"rho": np.array([1.0e-3]), "le": np.array([1.0])}
    sig_out, epsp_out, c = law163_crush_foam.solid_update(
        mat, sig, deps, dt=0.0, extra=extra, return_tuple=True
    )

    # cii = 120.0, cij = 40.0
    # Expected: sig_xx = -0.12, sig_yy = -0.04, sig_zz = -0.04
    assert sig_out[0, 0] == pytest.approx(-0.12, rel=1e-4)
    assert sig_out[0, 1] == pytest.approx(-0.04, rel=1e-4)
    assert sig_out[0, 2] == pytest.approx(-0.04, rel=1e-4)
    assert sig_out[0, 3] == pytest.approx(0.0)
    assert c[0] == pytest.approx(math.sqrt(120.0 / 1.0e-3), rel=1e-4)

    # 2. Dynamic response with viscous damping (dt = 1.0e-4)
    sig_dyn, _, c_dyn = law163_crush_foam.solid_update(
        mat, sig, deps, dt=1.0e-4, extra=extra, return_tuple=True
    )
    # Damping stress sigv_xx = -0.415692, sig_tot = -0.12 - 0.415692 = -0.535692
    assert sig_dyn[0, 0] == pytest.approx(-0.535692, rel=1e-4)
    assert extra["sigv"][0, 0] == pytest.approx(-0.415692, rel=1e-4)
    assert c_dyn[0] > c[0]

    # Tangent check
    D = law163_crush_foam.consistent_solid_tangent(mat, sig)
    assert D.shape == (1, 6, 6)
    assert D[0, 0, 0] == pytest.approx(120.0)
    assert D[0, 0, 1] == pytest.approx(40.0)
    assert D[0, 3, 3] == pytest.approx(40.0)  # G = 40


def test_law163_plastic_yield_and_tensile_cutoff():
    """Verify tabulated compressive yield capping and tensile cutoff tsc."""
    # Define simple yield curve: yield stress = 0.50 at all strains
    yield_table = (np.array([0.0, 1.0]), np.array([0.50, 0.50]))

    mat = MatLaw163(
        id=1,
        rho=1.0e-3,
        e=1000.0,
        nu=0.20,
        tsc=0.10,
        params={"table": yield_table},
    )

    # 1. Large compressive strain -> elastic trial gives sig_xx = -10.0, capped at -0.50
    deps_comp = np.array([[-0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sig_comp = np.zeros((1, 6))
    extra = {"rho": np.array([1.01e-3]), "le": np.array([1.0])}

    sig_out, epsp_out, c = law163_crush_foam.solid_update(
        mat, sig_comp, deps_comp, dt=0.0, extra=extra, return_tuple=True
    )
    # Principal stress should be capped at -0.50
    assert sig_out[0, 0] == pytest.approx(-0.50, rel=1e-3)

    # 2. Large tensile strain -> elastic trial gives sig_xx = +10.0, capped at tsc = +0.10
    deps_tens = np.array([[+0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])
    sig_tens = np.zeros((1, 6))
    extra_tens = {"rho": np.array([0.99e-3]), "le": np.array([1.0])}

    sig_tens_out, _, _ = law163_crush_foam.solid_update(
        mat, sig_tens, deps_tens, dt=0.0, extra=extra_tens, return_tuple=True
    )
    assert sig_tens_out[0, 0] == pytest.approx(0.10, rel=1e-3)


# ============================================================================
# 7. Solid Element Kernels Integration (solid_hexa8 & solid_tetra4)
# ============================================================================

def test_solid_hexa8_integration():
    """Verify solid_hexa8 force and critical time step computation with LAW163."""
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
                "mat_extra": {"uvar163": np.zeros((1, 2)), "epsd163": np.zeros(1)},
                "chk_fail": False,
                "qvw_pend": np.zeros(1),
            }

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

    mat = Material(id=1, rho0=1.0e-3, law=163)
    mat.params = {
        "rho": 1.0e-3, "rho0": 1.0e-3,
        "e": 100.0, "nu": 0.25, "tsc": 5.0, "damp": 0.10,
    }

    grp = DummyGroup()
    grp.state["slices"] = [(slice(0, 1), mat, None)]

    fint = np.zeros_like(x)
    mint = np.zeros_like(x)

    # 1. Critical time step calculation
    dt_crit = solid_hexa8.forces(grp, x=x, v=None, vr=None, dt=0.0, fint=fint, mint=mint)
    assert len(dt_crit) == 1
    assert dt_crit[0] > 0.0
    assert not np.isnan(dt_crit[0])

    # 2. Advance one cycle with compression in z
    v = np.zeros_like(x)
    v[4:, 2] = -1.0
    vr = np.zeros_like(x)
    dt = 1.0e-4

    dt_step = solid_hexa8.forces(grp, x=x, v=v, vr=vr, dt=dt, fint=fint, mint=mint)
    assert len(dt_step) == 1
    assert dt_step[0] > 0.0
    assert np.any(np.abs(fint) > 0.0)
    assert grp.state["sig"][0, 2] < 0.0  # Compressive stress in z


def test_solid_tetra4_integration():
    """Verify solid_tetra4 force and critical time step computation with LAW163."""
    class DummyTetGroup:
        def __init__(self):
            self.n = 1
            self.conn = np.array([[0, 1, 2, 3]])
            self.state = {
                "slices": [],
                "off": np.array([1.0]),
                "mass": np.array([1.2e-3 / 6.0]),
                "sig": np.zeros((1, 6)),
                "epsp": np.zeros(1),
                "eint": np.zeros(1),
                "ehour": np.zeros(1),
                "vol0": np.array([1.0 / 6.0]),
                "dtfac": np.array([0.9]),
                "lc_scale": np.array([1.0]),
                "mat_extra": {"uvar163": np.zeros((1, 2)), "epsd163": np.zeros(1)},
                "chk_fail": False,
                "qvw_pend": np.zeros(1),
            }

    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    mat = Material(id=2, rho0=1.2e-3, law=163)
    mat.params = {
        "rho": 1.2e-3, "rho0": 1.2e-3,
        "e": 150.0, "nu": 0.20, "tsc": 4.0, "damp": 0.10,
    }

    grp = DummyTetGroup()
    grp.state["slices"] = [(slice(0, 1), mat, None)]

    fint = np.zeros_like(x)
    mint = np.zeros_like(x)

    # 1. Critical time step calculation
    dt_crit = solid_tetra4.forces(grp, x=x, v=None, vr=None, dt=0.0, fint=fint, mint=mint)
    assert len(dt_crit) == 1
    assert dt_crit[0] > 0.0
    assert not np.isnan(dt_crit[0])

    # 2. Advance one cycle with compression on node 3
    v = np.zeros_like(x)
    v[3, 2] = -1.0
    vr = np.zeros_like(x)
    dt = 1.0e-4

    dt_step = solid_tetra4.forces(grp, x=x, v=v, vr=vr, dt=dt, fint=fint, mint=mint)
    assert len(dt_step) == 1
    assert dt_step[0] > 0.0
    assert np.any(np.abs(fint) > 0.0)
    assert grp.state["sig"][0, 2] < 0.0  # Compressive stress in z
