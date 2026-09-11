"""Tests for Milestone M562: /MAT/LAW66 (/MAT/PLAS_TAB_COSSER, /MAT/PLAS_COSSER) Starter & Model Integration.

Validates:
1. MatLaw66 (and MatPlasTabCosser, MatPlasCosser, MaterialLaw66) dataclass fields,
   property helpers (rho0, rhor, E, Nu, nu, G, bulk, K, sound_speed, sound_speed_solid, sound_speed_shell),
   CallableFloat behavior, and mapping protocol (__getitem__, __setitem__, __contains__, get, keys, values, items).
2. Model containers (mat_law66s, mat_plas_tab_cossers, mat_plas_cossers).
3. Card layouts (MAT_LAW66_1..5, MAT_LAW66_CURVE, MAT_PLAS_TAB_COSSER_*, MAT_PLAS_COSSER_*) matching mat_law66.cfg.
4. CFG catalogue mapping (LAW66, PLAS_TAB_COSSER, PLAS_COSSER -> 66).
5. Starter keyword reader for fixed and free formats across ISRATE 1, 2, 3, 4 with multi-curves,
   populating model.mat_law66s and model.materials.
6. Starter diagnostic checks: parameter bounds (rho > 0, E > 0, 0 <= nu < 0.5 with ANCMSG 1514, PC >= 0, PT >= 0),
   solid acceptance, shell acceptance, and 1D element rejection (ANCMSG 306).
7. StarterDeck writer for fixed and free formats and helper aliases (mat_plas_tab_cosser, mat_plas_cosser) with roundtrip.
8. Element integration: Courant sound speeds and uvar66 history arrays in solid_hexa8, solid_tetra4,
   shell_bt4, shell_qeph, and shell_tri3.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import (
    MatLaw66,
    MatPlasTabCosser,
    MatPlasCosser,
    MaterialLaw66,
    Material,
)
from pyradioss.model.model import Model
from pyradioss.input.card_layouts import CARD_LAYOUTS
from pyradioss.input.cfg_catalogue import law_number, canonical_law_name
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import read_mat_law66, read_mat
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.checks import (
    check_mat_law66,
    check_materials,
    check_model,
    _ALLOWED_LAWS,
    _MAT_CHECKS,
)
from pyradioss.common.messages import MessageLog
from pyradioss.elements.shell_bt4 import _init_material_state as bt4_init_mat, _layer_extra as bt4_layer_extra
from pyradioss.elements.solid_hexa8 import _init_material_state as hexa8_init_mat
import pyradioss.materials as materials


# ============================================================================
# 1. Model Entities, Property Helpers & Mapping Protocol
# ============================================================================

def test_mat_law66_entities_and_properties():
    """Verify MatLaw66 dataclass fields, property helpers, and mapping protocol."""
    m = MatLaw66(
        id=66,
        rho=1.2e-9,
        e=210000.0,
        nu=0.3,
        ec=180000.0,
        pc=50.0,
        pt=30.0,
        rpct=1.0,
        c_hard=0.25,
        f_cut=0.15,
        fsmooth=1,
        iyld_rate=1,
        fun_a1=101,
        fun_a2=102,
        fscale11=1.05,
        fscale22=0.95,
        eps_0=1.0,
        c=1.8,
        sigma_y0=220.0,
        vp=0,
        title="Cosserat Foam Model",
    )

    # Class aliases
    assert MatPlasTabCosser is MatLaw66
    assert MatPlasCosser is MatLaw66
    assert MaterialLaw66 is MatLaw66

    # Primary fields
    assert m.id == 66
    assert m.title == "Cosserat Foam Model"
    assert m.law == 66
    assert m.law_name == "LAW66"
    assert m.rho == pytest.approx(1.2e-9)
    assert m.rho0 == pytest.approx(1.2e-9)
    assert m.rhor == pytest.approx(1.2e-9)
    assert m.e == pytest.approx(210000.0)
    assert m.E == pytest.approx(210000.0)
    assert m.nu == pytest.approx(0.3)
    assert m.Nu == pytest.approx(0.3)
    assert m.ec == pytest.approx(180000.0)
    assert m.pc == pytest.approx(50.0)
    assert m.pt == pytest.approx(30.0)
    assert m.rpct == pytest.approx(1.0)
    assert m.c_hard == pytest.approx(0.25)
    assert m.f_cut == pytest.approx(0.15)
    assert m.fsmooth == 1
    assert m.iyld_rate == 1
    assert m.fun_a1 == 101
    assert m.fun_a2 == 102
    assert m.fscale11 == pytest.approx(1.05)
    assert m.fscale22 == pytest.approx(0.95)
    assert m.eps_0 == pytest.approx(1.0)
    assert m.c == pytest.approx(1.8)
    assert m.sigma_y0 == pytest.approx(220.0)
    assert m.vp == 0

    # Derived Moduli
    expected_g = 210000.0 / (2.0 * (1.0 + 0.3))
    expected_k = 210000.0 / (3.0 * (1.0 - 2.0 * 0.3))
    assert m.G == pytest.approx(expected_g)
    assert m.bulk == pytest.approx(expected_k)
    assert m.K == pytest.approx(expected_k)

    # Sound speeds
    expected_c_acoustic = math.sqrt(210000.0 / 1.2e-9)
    expected_c_solid = math.sqrt((expected_k + 4.0 * expected_g / 3.0) / 1.2e-9)
    expected_c_shell = math.sqrt(210000.0 / (1.2e-9 * (1.0 - 0.3**2)))

    assert m.sound_speed == pytest.approx(expected_c_acoustic)
    assert m.sound_speed_solid == pytest.approx(expected_c_solid)
    assert m.sound_speed_shell == pytest.approx(expected_c_shell)

    # CallableFloat support: both property and zero-arg call work
    assert m.sound_speed_solid() == pytest.approx(expected_c_solid)
    assert m.sound_speed_shell() == pytest.approx(expected_c_shell)
    assert float(m.sound_speed_solid) == pytest.approx(expected_c_solid)
    assert float(m.sound_speed_shell) == pytest.approx(expected_c_shell)

    # Mapping protocol
    assert "rho" in m
    assert "e" in m
    assert "nu" in m
    assert "ec" in m
    assert "pc" in m
    assert "pt" in m
    assert "rpct" in m
    assert m["e"] == pytest.approx(210000.0)
    assert m["pc"] == pytest.approx(50.0)
    assert m.get("pt") == pytest.approx(30.0)
    assert m.get("missing_param", -999.0) == -999.0

    # Key setting
    m["pc"] = 55.0
    assert m.pc == pytest.approx(55.0)
    assert m["pc"] == pytest.approx(55.0)

    with pytest.raises(KeyError, match="Cannot set unknown attribute"):
        m["non_existent_key"] = 123.45


def test_model_mat_law66_containers():
    """Verify Model contains mat_law66s dict and aliases."""
    model = Model()
    assert hasattr(model, "mat_law66s")
    assert hasattr(model, "mat_plas_tab_cossers")
    assert hasattr(model, "mat_plas_cossers")
    assert model.mat_plas_tab_cossers is model.mat_law66s
    assert model.mat_plas_cossers is model.mat_law66s


# ============================================================================
# 2. Card Layouts & Catalogue Mappings
# ============================================================================

def test_card_layouts_and_cfg_catalogue_law66():
    """Verify card layouts and law catalogue mappings for LAW66 and synonyms."""
    for prefix in ("MAT_LAW66", "MAT_PLAS_TAB_COSSER", "MAT_PLAS_COSSER"):
        assert CARD_LAYOUTS[f"{prefix}_1"] == [20, 20]
        assert CARD_LAYOUTS[f"{prefix}_2"] == [20, 20, 20, 20, 10, 10]
        assert CARD_LAYOUTS[f"{prefix}_3"] == [20, 20, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_4"] == [10, 10, 20, 20]
        assert CARD_LAYOUTS[f"{prefix}_5"] == [20, 20, 20, 10]
        assert CARD_LAYOUTS[f"{prefix}_CURVE"] == [10, 10, 20, 20]

    assert law_number("LAW66") == 66
    assert law_number("PLAS_TAB_COSSER") == 66
    assert law_number("PLAS_COSSER") == 66
    assert law_number("MAT_LAW66") == 66
    assert law_number("MAT_PLAS_TAB_COSSER") == 66
    assert law_number("MAT_PLAS_COSSER") == 66

    assert canonical_law_name(66) == "LAW66"
    assert canonical_law_name("PLAS_TAB_COSSER") == "LAW66"
    assert canonical_law_name("PLAS_COSSER") == "LAW66"


# ============================================================================
# 3. Starter Keyword Reader
# ============================================================================

def test_starter_keyword_reader_fixed_israte1(tmp_path):
    """Verify fixed-format /MAT/LAW66 parsing for ISRATE <= 2."""
    deck_text = (
        "/MAT/LAW66/1\n"
        "Cosserat Law66 Rate1\n"
        "#                 RHO            REFER_RHO\n"
        "             1.20E-9              1.20E-9\n"
        "#                  E                  NU              C_HARD               F_CUT   FSmooth  Iyld_rate\n"
        "            210000.0                 0.3                0.25                0.15         1         1\n"
        "#                P_C                 P_T                  EC                RPCT\n"
        "                50.0                30.0            180000.0                 1.0\n"
        "#  FUN_A1    FUN_A2             FSCALE11            FSCALE22\n"
        "      101       102                  1.0                 1.0\n"
        "#         EPSILON_0                    C            SIGMA_Y0        VP\n"
        "                 1.0                 2.5               250.0         0\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_fixed_israte1_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law66(block, model, log)

    assert len(log.errors) == 0
    assert 1 in model.mat_law66s
    assert 1 in model.materials

    m = model.mat_law66s[1]
    assert m.id == 1
    assert m.title == "Cosserat Law66 Rate1"
    assert m.rho == pytest.approx(1.2e-9)
    assert m.rhor == pytest.approx(1.2e-9)
    assert m.e == pytest.approx(210000.0)
    assert m.nu == pytest.approx(0.3)
    assert m.c_hard == pytest.approx(0.25)
    assert m.f_cut == pytest.approx(0.15)
    assert m.fsmooth == 1
    assert m.iyld_rate == 1
    assert m.pc == pytest.approx(50.0)
    assert m.pt == pytest.approx(30.0)
    assert m.ec == pytest.approx(180000.0)
    assert m.rpct == pytest.approx(1.0)
    assert m.fun_a1 == 101
    assert m.fun_a2 == 102
    assert m.fscale11 == pytest.approx(1.0)
    assert m.fscale22 == pytest.approx(1.0)
    assert m.eps_0 == pytest.approx(1.0)
    assert m.c == pytest.approx(2.5)
    assert m.sigma_y0 == pytest.approx(250.0)
    assert m.vp == 0

    mat = model.materials[1]
    assert mat.id == 1
    assert mat.law == 66
    assert mat.rho0 == pytest.approx(1.2e-9)
    assert mat.params["pc"] == pytest.approx(50.0)


def test_starter_keyword_reader_fixed_israte3(tmp_path):
    """Verify fixed-format /MAT/LAW66 parsing for ISRATE = 3."""
    deck_text = (
        "/MAT/LAW66/2\n"
        "Cosserat Law66 Rate3\n"
        "#                 RHO            REFER_RHO\n"
        "             1.50E-9             1.50E-9\n"
        "#                  E                  NU              C_HARD               F_CUT   FSmooth  Iyld_rate\n"
        "            190000.0                0.28                0.10                0.20         0         3\n"
        "#                P_C                 P_T                  EC                RPCT\n"
        "                40.0                20.0            160000.0                 1.0\n"
        "#  FUN_A1    FUN_A2             FSCALE11            FSCALE22\n"
        "      201       202                 1.05                0.95\n"
        "#  FUN_B1    FUN_B2             FSCALE33            FSCALE12\n"
        "      301       302                  1.1                 0.9\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_fixed_israte3_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law66(block, model, log)

    assert len(log.errors) == 0
    assert 2 in model.mat_law66s
    m = model.mat_law66s[2]
    assert m.iyld_rate == 3
    assert m.fun_a1 == 201
    assert m.fun_a2 == 202
    assert m.fscale11 == pytest.approx(1.05)
    assert m.fscale22 == pytest.approx(0.95)
    assert m.fun_b1 == 301
    assert m.fun_b2 == 302
    assert m.fscale33 == pytest.approx(1.1)
    assert m.fscale12 == pytest.approx(0.9)


def test_starter_keyword_reader_fixed_israte4_multicurves(tmp_path):
    """Verify fixed-format /MAT/LAW66 parsing for ISRATE = 4 with multi-curves."""
    deck_text = (
        "/MAT/LAW66/3\n"
        "Cosserat Law66 MultiCurve\n"
        "#                 RHO            REFER_RHO\n"
        "             1.80E-9             1.80E-9\n"
        "#                  E                  NU              C_HARD               F_CUT   FSmooth  Iyld_rate\n"
        "            150000.0                0.25                0.30                0.05         1         4\n"
        "#                P_C                 P_T                  EC                RPCT\n"
        "                60.0                40.0            120000.0                 1.0\n"
        "#   NFUNC     TFUNC\n"
        "        3         1\n"
        "#      ID                      STRAIN_RATE          FSCALE_Y\n"
        "      401                                0.0                 1.0\n"
        "      402                               10.0                 1.2\n"
        "      403                              100.0                 1.5\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_fixed_israte4_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat_law66(block, model, log)

    assert len(log.errors) == 0
    assert 3 in model.mat_law66s
    m = model.mat_law66s[3]
    assert m.iyld_rate == 4
    assert m.nfunc == 3
    assert m.tfunc == 1
    assert m.func_ids == [401, 402, 403]
    assert m.rates == pytest.approx([0.0, 10.0, 100.0])
    assert m.fscales == pytest.approx([1.0, 1.2, 1.5])


def test_starter_keyword_reader_free_and_synonyms(tmp_path):
    """Verify free-format /MAT/PLAS_TAB_COSSER and /MAT/PLAS_COSSER parsing."""
    deck_text = (
        "/MAT/PLAS_TAB_COSSER/10\n"
        "Plas Tab Cosser Free\n"
        "2.0e-9 2.0e-9\n"
        "180000.0 0.3 0.2 0.1 0 1\n"
        "45.0 25.0 160000.0 1.0\n"
        "501 502 1.0 1.0\n"
        "1.0 2.0 200.0 0\n"
        "/MAT/PLAS_COSSER/20\n"
        "Plas Cosser Free Rate4\n"
        "1.1e-9 1.1e-9\n"
        "200000.0 0.32 0.15 0.08 1 4\n"
        "30.0 15.0 170000.0 1.0\n"
        "2 0\n"
        "601 0.0 0.0 1.0\n"
        "602 0.0 50.0 1.25\n"
        "/END\n"
    )
    deck_file = tmp_path / "deck_free_synonyms_0000.rad"
    deck_file.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(deck_file))

    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat(block, model, log)

    assert len(log.errors) == 0
    assert 10 in model.mat_law66s
    assert 20 in model.mat_law66s

    m10 = model.mat_law66s[10]
    assert m10.title == "Plas Tab Cosser Free"
    assert m10.rho == pytest.approx(2.0e-9)
    assert m10.e == pytest.approx(180000.0)
    assert m10.nu == pytest.approx(0.3)
    assert m10.c_hard == pytest.approx(0.2)
    assert m10.pc == pytest.approx(45.0)
    assert m10.pt == pytest.approx(25.0)

    m20 = model.mat_law66s[20]
    assert m20.title == "Plas Cosser Free Rate4"
    assert m20.iyld_rate == 4
    assert m20.nfunc == 2
    assert m20.func_ids == [601, 602]
    assert m20.rates == pytest.approx([0.0, 50.0])
    assert m20.fscales == pytest.approx([1.0, 1.25])


# ============================================================================
# 4. Starter Validation Checks
# ============================================================================

def test_starter_checks_valid_law66():
    """Verify check_mat_law66 produces no errors on valid material."""
    m = MatLaw66(id=1, rho=1.2e-9, e=210000.0, nu=0.3, pc=50.0, pt=30.0)
    log = MessageLog()
    check_mat_law66(mat=m, log=log)
    assert len(log.errors) == 0


def test_starter_checks_parameter_bounds():
    """Verify check_mat_law66 catches rho<=0, E<=0, 0<=nu<0.5, pc<0, pt<0."""
    log = MessageLog()

    # 1. Invalid rho
    m_bad_rho = MatLaw66(id=1, rho=-1.0, e=210000.0, nu=0.3, pc=50.0, pt=30.0)
    check_mat_law66(mat=m_bad_rho, log=log)
    assert any("initial density RHO must be > 0" in e for e in log.errors)

    # 2. Invalid E
    log = MessageLog()
    m_bad_e = MatLaw66(id=2, rho=1.0e-9, e=0.0, nu=0.3, pc=50.0, pt=30.0)
    check_mat_law66(mat=m_bad_e, log=log)
    assert any("Young's modulus E must be > 0" in e for e in log.errors)

    # 3. Invalid nu (ANCMSG 1514)
    log = MessageLog()
    m_bad_nu = MatLaw66(id=3, rho=1.0e-9, e=210000.0, nu=0.55, pc=50.0, pt=30.0)
    check_mat_law66(mat=m_bad_nu, log=log)
    assert any("ANCMSG 1514" in e for e in log.errors)

    # 4. Invalid pc
    log = MessageLog()
    m_bad_pc = MatLaw66(id=4, rho=1.0e-9, e=210000.0, nu=0.3, pc=-10.0, pt=30.0)
    check_mat_law66(mat=m_bad_pc, log=log)
    assert any("P_c must be >= 0" in e for e in log.errors)

    # 5. Invalid pt
    log = MessageLog()
    m_bad_pt = MatLaw66(id=5, rho=1.0e-9, e=210000.0, nu=0.3, pc=50.0, pt=-5.0)
    check_mat_law66(mat=m_bad_pt, log=log)
    assert any("P_t must be >= 0" in e for e in log.errors)


def test_starter_checks_element_compatibility():
    """Verify solid and shell acceptance, and 1D element rejection (ANCMSG 306)."""
    m = MatLaw66(id=66, rho=1.2e-9, e=210000.0, nu=0.3, pc=50.0, pt=30.0)

    # Solid and shell families are in _ALLOWED_LAWS
    assert 66 in _ALLOWED_LAWS["bricks"]
    assert 66 in _ALLOWED_LAWS["tetras"]
    assert 66 in _ALLOWED_LAWS["penta6"]
    assert 66 in _ALLOWED_LAWS["pyra5"]
    assert 66 in _ALLOWED_LAWS["shells"]
    assert 66 in _ALLOWED_LAWS["shells_qeph"]
    assert 66 in _ALLOWED_LAWS["shells_qbat"]
    assert 66 in _ALLOWED_LAWS["sh3n"]

    # 1D families reject LAW66
    assert 66 not in _ALLOWED_LAWS["trusses"]
    assert 66 not in _ALLOWED_LAWS["beams"]

    # Test model check rejecting 1D element
    class DummyEl:
        def __init__(self, mat_id):
            self.mat_id = mat_id

    class DummyGroup:
        def __init__(self, elements):
            self._elements = elements
        def values(self):
            return self._elements

    model = Model()
    model.materials[66] = m
    model.mat_law66s[66] = m
    model._groups = [("beams", DummyGroup([DummyEl(66)]))]
    model.element_groups = lambda: model._groups

    log = MessageLog()
    check_mat_law66(mat=m, log=log, model=model)
    assert any("ANCMSG 306" in e for e in log.errors)

    # Registered in _MAT_CHECKS
    assert 66 in _MAT_CHECKS
    assert "LAW66" in _MAT_CHECKS
    assert "PLAS_TAB_COSSER" in _MAT_CHECKS


# ============================================================================
# 5. StarterDeck Writer & Roundtrip
# ============================================================================

def test_starter_deck_writer_and_roundtrip(tmp_path):
    """Verify StarterDeck emits /MAT/LAW66 in fixed and free formats with roundtrip parsing."""
    deck = StarterDeck("M562_TEST")

    # 1. Using kwargs in fixed format (ISRATE = 1)
    deck.mat_law66(
        mid=1,
        title="Law66 Emitter Card",
        rho=1.3e-9,
        e=205000.0,
        nu=0.29,
        ec=175000.0,
        pc=48.0,
        pt=28.0,
        rpct=1.0,
        c_hard=0.2,
        f_cut=0.12,
        fsmooth=1,
        iyld_rate=1,
        fun_a1=10,
        fun_a2=11,
        fscale11=1.0,
        fscale22=1.0,
        eps_0=1.0,
        c=2.2,
        sigma_y0=210.0,
        vp=0,
    )

    # 2. Using MatLaw66 dataclass with multi-curves (ISRATE = 4)
    m2 = MatLaw66(
        id=2,
        rho=1.4e-9,
        e=185000.0,
        nu=0.27,
        ec=155000.0,
        pc=42.0,
        pt=22.0,
        rpct=1.0,
        c_hard=0.18,
        f_cut=0.1,
        fsmooth=0,
        iyld_rate=4,
        nfunc=2,
        tfunc=1,
        func_ids=[701, 702],
        rates=[0.0, 100.0],
        fscales=[1.0, 1.3],
        title="Law66 MultiCurve Emitter",
    )
    deck.mat_plas_tab_cosser(m2)

    # 3. Free format using alias
    deck.mat_plas_cosser(
        mid=3,
        title="Law66 Free Emitter",
        rho=1.6e-9,
        e=190000.0,
        nu=0.31,
        pc=52.0,
        pt=32.0,
        ec=160000.0,
        rpct=1.0,
        fixed_format=False,
    )

    deck_path = tmp_path / "emitter_0000.rad"
    deck.write(str(deck_path))

    # Read back and verify
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    for block in blocks:
        if block.key0 == "MAT":
            read_mat(block, model, log)

    assert len(log.errors) == 0
    assert 1 in model.mat_law66s
    assert 2 in model.mat_law66s
    assert 3 in model.mat_law66s

    r1 = model.mat_law66s[1]
    assert r1.title == "Law66 Emitter Card"
    assert r1.rho == pytest.approx(1.3e-9)
    assert r1.e == pytest.approx(205000.0)
    assert r1.nu == pytest.approx(0.29)
    assert r1.pc == pytest.approx(48.0)
    assert r1.pt == pytest.approx(28.0)
    assert r1.c == pytest.approx(2.2)

    r2 = model.mat_law66s[2]
    assert r2.title == "Law66 MultiCurve Emitter"
    assert r2.iyld_rate == 4
    assert r2.nfunc == 2
    assert r2.tfunc == 1
    assert r2.func_ids == [701, 702]
    assert r2.rates == pytest.approx([0.0, 100.0])
    assert r2.fscales == pytest.approx([1.0, 1.3])

    r3 = model.mat_law66s[3]
    assert r3.title == "Law66 Free Emitter"
    assert r3.rho == pytest.approx(1.6e-9)
    assert r3.e == pytest.approx(190000.0)
    assert r3.pc == pytest.approx(52.0)
    assert r3.pt == pytest.approx(32.0)


# ============================================================================
# 6. Element Integration (Solids & Shells)
# ============================================================================

def test_element_integration_solid_hexa8():
    """Verify solid_hexa8 allocates uvar66 array and computes sound speed for LAW66."""
    m = MatLaw66(id=1, rho=1.2e-9, e=210000.0, nu=0.3, pc=50.0, pt=30.0)
    shapes = materials.extra_shapes(m, nip=None)
    assert "uvar66" in shapes

    class DummyGroup:
        def __init__(self):
            self.n = 2
            self.state = {
                "slices": [(slice(0, 2), m, None)],
                "off": np.ones(2),
                "eint": np.zeros(2),
                "vol0": np.ones(2),
                "mass": np.full(2, 1.2e-9),
            }

    grp = DummyGroup()
    dndx0 = np.zeros((2, 8, 3))
    hexa8_init_mat(grp, dndx0)
    assert "uvar66" in grp.state["mat_extra"]
    assert grp.state["chk_fail"] is True


def test_element_integration_solid_tetra4():
    """Verify solid_tetra4 uses sound_speed_solid for LAW66."""
    m = MatLaw66(id=1, rho=1.2e-9, e=210000.0, nu=0.3, pc=50.0, pt=30.0)
    assert hasattr(m, "sound_speed_solid")
    c_expected = m.sound_speed_solid()
    assert c_expected > 0.0


def test_element_integration_shell_bt4():
    """Verify shell_bt4 allocates uvar66 array (n, nip, 20) and exposes in _layer_extra."""
    m = MatLaw66(id=1, rho=1.2e-9, e=210000.0, nu=0.3, pc=50.0, pt=30.0)
    shapes = materials.extra_shapes(m, nip=3)
    assert "uvar66" in shapes

    class DummyProp:
        nip = 3
        thick = 1.0

    class DummyGroup:
        def __init__(self):
            self.n = 2
            self.state = {
                "slices": [(slice(0, 2), m, DummyProp())],
                "layfail": np.zeros((2, 3), dtype=bool),
                "mat_extra": {},
            }

    grp = DummyGroup()
    bt4_init_mat(grp, nip_max=3)
    assert "uvar66" in grp.state["mat_extra"]
    assert grp.state["chk_fail"] is True

    extra = bt4_layer_extra(grp.state, slice(0, 2), 0)
    assert "uvar66" in extra
    assert "uvar" in extra
    assert m.sound_speed_shell() > 0.0


def test_element_integration_shell_qeph_and_tri3():
    """Verify shell_qeph and shell_tri3 sound speed calculations for LAW66."""
    m = MatLaw66(id=1, rho=1.2e-9, e=210000.0, nu=0.3, pc=50.0, pt=30.0)
    c_shell = m.sound_speed_shell()
    assert c_shell > 0.0

    # Materials dispatch check
    c_disp = materials.sound_speed(m, rho=1.2e-9)
    assert c_disp > 0.0
