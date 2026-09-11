"""
Tests for M554 (/MAT/LAW52, /MAT/GURSON, /MAT/PLAS_GURS) input, starter, model, and checks integration.
"""

import math
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.card_layouts import (
    MAT_LAW52_1,
    MAT_LAW52_2,
    MAT_LAW52_3,
    MAT_LAW52_4,
    MAT_LAW52_5,
    MAT_LAW52_6,
    MAT_GURSON_1,
    MAT_GURSON_2,
    MAT_GURSON_3,
    MAT_GURSON_4,
    MAT_GURSON_5,
    MAT_GURSON_6,
    MAT_PLAS_GURS_1,
    MAT_PLAS_GURS_2,
    MAT_PLAS_GURS_3,
    MAT_PLAS_GURS_4,
    MAT_PLAS_GURS_5,
    MAT_PLAS_GURS_6,
    LAYOUTS,
)
from pyradioss.input.cfg_catalogue import LAW_MAP, LAW_SYNONYMS, law_number, canonical_law_name
from pyradioss.model.entities import MatLaw52, MatGurson, MatPlasGurs, Material
from pyradioss.model.model import Model
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck, read_mat_law52, KEYWORD_PARSERS
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.checks import (
    check_mat_law52,
    _check_mat_law52,
    check_materials,
    check_model,
    _ALLOWED_LAWS,
    _MAT_CHECKS,
)


def test_m554_card_layouts():
    assert tuple(MAT_LAW52_1) == (20, 20)
    assert tuple(MAT_LAW52_2) == (20, 20, 10, 10, 20, 10)
    assert tuple(MAT_LAW52_3) == (20, 20, 20, 20, 20)
    assert tuple(MAT_LAW52_4) == (20, 20, 20, 20, 20)
    assert tuple(MAT_LAW52_5) == (20, 20, 20, 20)
    assert tuple(MAT_LAW52_6) == (10, 10, 20, 20)

    for i in range(1, 7):
        l_law = globals()[f"MAT_LAW52_{i}"]
        l_gur = globals()[f"MAT_GURSON_{i}"]
        l_plas = globals()[f"MAT_PLAS_GURS_{i}"]
        assert l_gur == l_law
        assert l_plas == l_law
        assert f"MAT_LAW52_{i}" in LAYOUTS
        assert f"MAT_GURSON_{i}" in LAYOUTS
        assert f"MAT_PLAS_GURS_{i}" in LAYOUTS
        assert list(LAYOUTS[f"MAT_LAW52_{i}"]) == list(l_law)
        assert list(LAYOUTS[f"MAT_GURSON_{i}"]) == list(l_law)
        assert list(LAYOUTS[f"MAT_PLAS_GURS_{i}"]) == list(l_law)


def test_m554_cfg_catalogue():
    for name in ("LAW52", "GURSON", "PLAS_GURS", "MAT_GURSON", "MAT_PLAS_GURS", "MAT_LAW52", "LAW52_GURSON"):
        assert LAW_MAP[name] == 52
        assert law_number(name) == 52
        assert canonical_law_name(name) == "LAW52"

    assert LAW_SYNONYMS["LAW52"] == "LAW52"
    assert LAW_SYNONYMS["GURSON"] == "LAW52"
    assert LAW_SYNONYMS["PLAS_GURS"] == "LAW52"
    assert LAW_SYNONYMS["LAW52_GURSON"] == "LAW52"


def test_m554_keyword_parsers_dispatch():
    for k in ("LAW52", "MAT_LAW52", "GURSON", "MAT_GURSON", "PLAS_GURS", "MAT_PLAS_GURS"):
        assert KEYWORD_PARSERS[k] is read_mat_law52


def test_m554_model_entities_matlaw52():
    m = MatLaw52(
        id=52,
        rho=7.8e-3,
        refer_rho=7.8e-3,
        e=210000.0,
        nu=0.3,
        a=400.0,
        b=100.0,
        n=0.2,
        c=0.01,
        pc=2.0,
        q1=1.5,
        q2=1.0,
        q3=2.25,
        s_n=0.1,
        eps_n=0.3,
        f_i=0.001,
        f_n=0.04,
        f_c=0.05,
        f_f=0.15,
        iflag=1,
        fsmooth=0.0,
        fcut=1000.0,
        itable=10,
        xfac=1.05,
        yfac=1.02,
        title="Test Gurson Mat",
    )
    assert m.id == 52
    assert m.rho == pytest.approx(7.8e-3)
    assert m.refer_rho == pytest.approx(7.8e-3)
    assert m.rho0 == pytest.approx(7.8e-3)
    assert m.rhor == pytest.approx(7.8e-3)
    assert m.e == pytest.approx(210000.0)
    assert m.nu == pytest.approx(0.3)
    assert m.a == pytest.approx(400.0)
    assert m.yield_stress == pytest.approx(400.0)
    assert m.b == pytest.approx(100.0)
    assert m.hardening_b == pytest.approx(100.0)
    assert m.n == pytest.approx(0.2)
    assert m.hardening_n == pytest.approx(0.2)
    assert m.c == pytest.approx(0.01)
    assert m.pc == pytest.approx(2.0)
    assert m.q1 == pytest.approx(1.5)
    assert m.fu == pytest.approx(1.0 / 1.5)
    assert m.q2 == pytest.approx(1.0)
    assert m.q3 == pytest.approx(2.25)
    assert m.s_n == pytest.approx(0.1)
    assert m.eps_n == pytest.approx(0.3)
    assert m.f_i == pytest.approx(0.001)
    assert m.f_n == pytest.approx(0.04)
    assert m.f_c == pytest.approx(0.05)
    assert m.f_f == pytest.approx(0.15)
    assert m.iflag == 1
    assert m.fsmooth == pytest.approx(0.0)
    assert m.fcut == pytest.approx(1000.0)
    assert m.itable == 10
    assert m.xfac == pytest.approx(1.05)
    assert m.yfac == pytest.approx(1.02)
    assert m.title == "Test Gurson Mat"

    # Sound speed checks (property and callable)
    c_bulk_expected = math.sqrt(210000.0 / (7.8e-3))
    c_solid_expected = math.sqrt((210000.0 * (1.0 - 0.3)) / (7.8e-3 * (1.0 + 0.3) * (1.0 - 2.0 * 0.3)))
    c_shell_expected = math.sqrt(210000.0 / (7.8e-3 * (1.0 - 0.3 * 0.3)))

    assert m.sound_speed == pytest.approx(c_bulk_expected)
    assert m.sound_speed() == pytest.approx(c_bulk_expected)
    assert m.sound_speed_solid == pytest.approx(c_solid_expected)
    assert m.sound_speed_solid() == pytest.approx(c_solid_expected)
    assert m.sound_speed_shell == pytest.approx(c_shell_expected)
    assert m.sound_speed_shell() == pytest.approx(c_shell_expected)


def test_m554_model_entities_aliases_and_subscript():
    m_alias1 = MatGurson(id=1, rho=7.8e-3, e=210000.0, nu=0.3)
    assert isinstance(m_alias1, MatLaw52)
    m_alias2 = MatPlasGurs(id=2, rho=7.8e-3, e=210000.0, nu=0.3)
    assert isinstance(m_alias2, MatLaw52)

    # Dictionary-like subscript access
    assert m_alias1["rho"] == pytest.approx(7.8e-3)
    assert m_alias1["e"] == pytest.approx(210000.0)
    assert m_alias1["nu"] == pytest.approx(0.3)
    assert "rho" in m_alias1
    assert "e" in m_alias1
    assert m_alias1.get("rho") == pytest.approx(7.8e-3)
    assert m_alias1.get("nonexistent", 42) == 42

    # Model collection aliases
    model = Model()
    assert model.mat_gursons is model.mat_law52s
    assert model.mat_plas_gurs is model.mat_law52s


def test_m554_fixed_format_parsing(tmp_path):
    deck_text = """/BEGIN
Fixed format Gurson test
/MAT/LAW52/10
High strength steel
              0.0078              0.0078
            210000.0                 0.3         1         0              1000.0         5
               400.0               100.0                 0.2                0.01                 2.0
                 1.5                 1.0                2.25                 0.1                 0.3
               0.001                0.04                0.05                0.15
         5                          1.05                1.02
/END
"""
    f = tmp_path / "deck_fixed_0000.rad"
    f.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(f))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert 10 in model.mat_law52s
    assert 10 in model.mat_gursons
    assert 10 in model.mat_plas_gurs
    assert 10 in model.materials

    mat = model.mat_law52s[10]
    assert mat.id == 10
    assert mat.title == "High strength steel"
    assert mat.rho == pytest.approx(0.0078)
    assert mat.refer_rho == pytest.approx(0.0078)
    assert mat.e == pytest.approx(210000.0)
    assert mat.nu == pytest.approx(0.3)
    assert mat.iflag == 1
    assert mat.fsmooth == 0
    assert mat.fcut == pytest.approx(1000.0)
    assert mat.a == pytest.approx(400.0)
    assert mat.b == pytest.approx(100.0)
    assert mat.n == pytest.approx(0.2)
    assert mat.c == pytest.approx(0.01)
    assert mat.pc == pytest.approx(2.0)
    assert mat.q1 == pytest.approx(1.5)
    assert mat.q2 == pytest.approx(1.0)
    assert mat.q3 == pytest.approx(2.25)
    assert mat.s_n == pytest.approx(0.1)
    assert mat.eps_n == pytest.approx(0.3)
    assert mat.f_i == pytest.approx(0.001)
    assert mat.f_n == pytest.approx(0.04)
    assert mat.f_c == pytest.approx(0.05)
    assert mat.f_f == pytest.approx(0.15)
    assert mat.itable == 5
    assert mat.xfac == pytest.approx(1.05)
    assert mat.yfac == pytest.approx(1.02)


def test_m554_free_format_and_defaults(tmp_path):
    # Minimal fields to trigger Fortran defaults:
    # refer_rho -> rho
    # c -> 1e30
    # pc -> 1.0
    # fcut -> 1e30
    # q1 -> 1e-20 (fu -> 1/q1)
    # xfac -> 1.0, yfac -> 1.0
    deck_text = """/BEGIN
Free format Gurson defaults test
/MAT/GURSON/20
Minimal Gurson
7.8e-3
210000.0, 0.3, 0, 0, 0.0, 0
350.0, 80.0, 0.15, 0.0, 0.0
0.0, 1.0, 0.0, 0.0, 0.0
0.001, 0.04, 0.05, 0.10
/END
"""
    f = tmp_path / "deck_free_0000.rad"
    f.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(f))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert 20 in model.mat_law52s
    mat = model.mat_law52s[20]
    assert mat.rho == pytest.approx(7.8e-3)
    assert mat.refer_rho == pytest.approx(7.8e-3)
    assert mat.c == pytest.approx(1e30)
    assert mat.pc == pytest.approx(1.0)
    assert mat.fcut == pytest.approx(1e30)
    assert mat.q1 == pytest.approx(1e-20)
    assert mat.fu == pytest.approx(1.0 / 1e-20)
    assert mat.xfac == pytest.approx(1.0)
    assert mat.yfac == pytest.approx(1.0)
    # Check preserved raw inputs for diagnostics
    assert mat.raw_c == pytest.approx(0.0)
    assert mat.raw_pc == pytest.approx(0.0)
    assert mat.raw_fcut == pytest.approx(0.0)


def test_m554_plas_gurs_alias_deck(tmp_path):
    deck_text = """/BEGIN
PLAS_GURS alias test
/MAT/PLAS_GURS/30
PLAS_GURS title
7.8e-3, 7.8e-3
200000.0, 0.28, 0, 0, 0.0, 0
500.0, 150.0, 0.25, 0.0, 0.0
1.5, 1.0, 2.25, 0.1, 0.3
0.001, 0.04, 0.05, 0.15
/END
"""
    f = tmp_path / "deck_plas_gurs_0000.rad"
    f.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(f))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert 30 in model.mat_law52s
    assert 30 in model.mat_plas_gurs
    mat = model.mat_plas_gurs[30]
    assert mat.id == 30
    assert mat.e == pytest.approx(200000.0)
    assert mat.nu == pytest.approx(0.28)
    assert mat.a == pytest.approx(500.0)


def test_m554_deck_writer_fixed_and_roundtrip(tmp_path):
    deck = StarterDeck("GURSON_WRITER_TEST")
    deck.mat_law52(
        mid=1,
        rho=7.85e-3,
        refer_rho=7.85e-3,
        e=205000.0,
        nu=0.29,
        a=420.0,
        b=120.0,
        n=0.22,
        c=0.015,
        pc=2.5,
        q1=1.5,
        q2=1.0,
        q3=2.25,
        s_n=0.1,
        eps_n=0.3,
        f_i=0.002,
        f_n=0.04,
        f_c=0.06,
        f_f=0.18,
        iflag=1,
        fsmooth=0,
        fcut=500.0,
        itable=101,
        xfac=1.1,
        yfac=1.05,
        title="Steel Gurson",
    )

    rendered = deck.render()
    assert "/MAT/LAW52/1" in rendered
    assert "Steel Gurson" in rendered

    # Parse rendered back into a model
    deck_path = tmp_path / "roundtrip_0000.rad"
    deck_path.write_text(rendered, encoding="utf-8")
    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert 1 in model.mat_law52s
    m = model.mat_law52s[1]
    assert m.rho == pytest.approx(7.85e-3)
    assert m.e == pytest.approx(205000.0)
    assert m.nu == pytest.approx(0.29)
    assert m.a == pytest.approx(420.0)
    assert m.b == pytest.approx(120.0)
    assert m.n == pytest.approx(0.22)
    assert m.c == pytest.approx(0.015)
    assert m.pc == pytest.approx(2.5)
    assert m.q1 == pytest.approx(1.5)
    assert m.q2 == pytest.approx(1.0)
    assert m.q3 == pytest.approx(2.25)
    assert m.s_n == pytest.approx(0.1)
    assert m.eps_n == pytest.approx(0.3)
    assert m.f_i == pytest.approx(0.002)
    assert m.f_n == pytest.approx(0.04)
    assert m.f_c == pytest.approx(0.06)
    assert m.f_f == pytest.approx(0.18)
    assert m.iflag == 1
    assert m.fcut == pytest.approx(500.0)
    assert m.itable == 101
    assert m.xfac == pytest.approx(1.1)
    assert m.yfac == pytest.approx(1.05)


def test_m554_deck_writer_aliases():
    deck = StarterDeck("GURSON_ALIASES")
    deck.mat_gurson(mid=2, rho=7.8e-3, e=210000.0, nu=0.3, title="Gurson Alias")
    deck.mat_plas_gurs(mid=3, rho=7.8e-3, e=210000.0, nu=0.3, title="PlasGurs Alias")
    rendered = deck.render()
    assert "/MAT/GURSON/2" in rendered
    assert "Gurson Alias" in rendered
    assert "/MAT/PLAS_GURS/3" in rendered
    assert "PlasGurs Alias" in rendered


def test_m554_starter_checks():
    # 1. Valid material passes
    m_valid = MatLaw52(
        id=1,
        rho=7.8e-3,
        e=210000.0,
        nu=0.3,
        f_i=0.001,
        f_c=0.05,
        f_f=0.15,
    )
    log = MessageLog()
    check_mat_law52(mat=m_valid, log=log)
    assert not log.has_errors
    assert not log.has_warnings

    # 2. Alias _check_mat_law52 works with positional and kw args
    log = MessageLog()
    _check_mat_law52(m_valid, log)
    assert not log.has_errors

    # 3. Density rho <= 0
    m_bad_rho = MatLaw52(id=2, rho=0.0, e=210000.0, nu=0.3)
    log = MessageLog()
    check_mat_law52(mat=m_bad_rho, log=log)
    assert log.has_errors
    assert any("density" in str(e).lower() or "rho" in str(e).lower() for e in log.errors)

    # 4. Young's modulus E <= 0
    m_bad_e = MatLaw52(id=3, rho=7.8e-3, e=-100.0, nu=0.3)
    log = MessageLog()
    check_mat_law52(mat=m_bad_e, log=log)
    assert log.has_errors
    assert any("young" in str(e).lower() or "e" in str(e).lower() for e in log.errors)

    # 5. Poisson's ratio out of range: nu < 0 or nu >= 0.5
    m_bad_nu1 = MatLaw52(id=4, rho=7.8e-3, e=210000.0, nu=-0.1)
    log = MessageLog()
    check_mat_law52(mat=m_bad_nu1, log=log)
    assert log.has_errors
    assert any("poisson" in str(e).lower() or "nu" in str(e).lower() for e in log.errors)

    m_bad_nu2 = MatLaw52(id=5, rho=7.8e-3, e=210000.0, nu=0.5)
    log = MessageLog()
    check_mat_law52(mat=m_bad_nu2, log=log)
    assert log.has_errors
    assert any("poisson" in str(e).lower() or "nu" in str(e).lower() for e in log.errors)

    # 6. Void fraction consistency ANCMSG 1745: f_F < f_C or f_F < f_I or f_C < f_I
    m_bad_f1 = MatLaw52(id=6, rho=7.8e-3, e=210000.0, nu=0.3, f_i=0.05, f_c=0.02, f_f=0.10)
    log = MessageLog()
    check_mat_law52(mat=m_bad_f1, log=log)
    assert log.has_errors
    assert any("1745" in str(e) or "void" in str(e).lower() for e in log.errors)

    m_bad_f2 = MatLaw52(id=7, rho=7.8e-3, e=210000.0, nu=0.3, f_i=0.01, f_c=0.08, f_f=0.05)
    log = MessageLog()
    check_mat_law52(mat=m_bad_f2, log=log)
    assert log.has_errors
    assert any("1745" in str(e) or "void" in str(e).lower() for e in log.errors)

    # 7. Warning ANCMSG 1220: C > 0 and P > 0 with Fcut == 0
    m_warn = MatLaw52(
        id=8,
        rho=7.8e-3,
        e=210000.0,
        nu=0.3,
        c=0.05,
        pc=2.0,
        fcut=1e30,
        raw_c=0.05,
        raw_pc=2.0,
        raw_fcut=0.0,
        f_i=0.001,
        f_c=0.05,
        f_f=0.15,
    )
    log = MessageLog()
    check_mat_law52(mat=m_warn, log=log)
    assert not log.has_errors
    assert log.has_warnings
    assert any("1220" in str(w) or "strain rate filtering" in str(w).lower() for w in log.warnings)


def test_m554_element_compatibility_and_model_checks():
    # Check _ALLOWED_LAWS registry
    for family in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
        assert 52 in _ALLOWED_LAWS[family]
        assert "52" in _ALLOWED_LAWS[family]
        assert "LAW52" in _ALLOWED_LAWS[family]
        assert "GURSON" in _ALLOWED_LAWS[family]
        assert "PLAS_GURS" in _ALLOWED_LAWS[family]

    # Trusses, beams, springs must reject LAW52
    assert 52 not in _ALLOWED_LAWS["trusses"]
    assert 52 not in _ALLOWED_LAWS["beams"]
    assert _ALLOWED_LAWS["springs"] is None

    # Check _MAT_CHECKS dispatch
    assert _MAT_CHECKS[52] is check_mat_law52
    assert _MAT_CHECKS["52"] is check_mat_law52
    assert _MAT_CHECKS["LAW52"] is check_mat_law52
    assert _MAT_CHECKS["GURSON"] is check_mat_law52
    assert _MAT_CHECKS["PLAS_GURS"] is check_mat_law52

    # Check check_materials integration
    model = Model()
    m_valid = MatLaw52(id=1, rho=7.8e-3, e=210000.0, nu=0.3, f_i=0.001, f_c=0.05, f_f=0.15)
    m_valid.law = 52
    model.materials[1] = m_valid
    log = MessageLog()
    check_materials(model, log)
    assert not log.has_errors

    # Invalid Young's modulus detected via check_materials
    m_bad = MatLaw52(id=2, rho=7.8e-3, e=-500.0, nu=0.3)
    m_bad.law = 52
    model.materials[2] = m_bad
    log = MessageLog()
    check_materials(model, log)
    assert log.has_errors
    assert any("young" in str(e).lower() for e in log.errors)

    # 1D element rejection check
    class DummyElement:
        def __init__(self, mat_id):
            self.mat_id = mat_id

    class DummyGroup:
        def __init__(self, elements):
            self._elements = elements
        def values(self):
            return self._elements

    # Attaching LAW52 to trusses element group produces ANCMSG 306 error
    model_truss = Model()
    m52 = MatLaw52(id=1, rho=7.8e-3, e=210000.0, nu=0.3, f_i=0.001, f_c=0.05, f_f=0.15)
    m52.law = 52
    model_truss.materials[1] = m52
    model_truss._element_groups = [("trusses", DummyGroup([DummyElement(1)]))]
    model_truss.element_groups = lambda: model_truss._element_groups

    log = MessageLog()
    check_mat_law52(model=model_truss, mat_id=1, mat=m52, log=log)
    assert log.has_errors
    assert any("306" in str(e) or "1d" in str(e).lower() for e in log.errors)
