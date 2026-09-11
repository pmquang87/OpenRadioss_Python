"""
Tests for M551 (/MAT/LAW60 /MAT/PLAS_T3 /MAT/FABRIC) input, starter, model, and checks integration.
"""

from io import StringIO
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.input.card_layouts import (
    MAT_LAW60_1,
    MAT_LAW60_2,
    MAT_LAW60_3,
    MAT_LAW60_4,
    MAT_LAW60_5,
    MAT_LAW60_6,
    MAT_LAW60_7,
    MAT_LAW60_8,
    MAT_LAW60_9,
    MAT_LAW60_10,
    MAT_PLAS_T3_1,
    MAT_PLAS_T3_2,
    MAT_PLAS_T3_3,
    MAT_PLAS_T3_4,
    MAT_PLAS_T3_5,
    MAT_PLAS_T3_6,
    MAT_PLAS_T3_7,
    MAT_PLAS_T3_8,
    MAT_PLAS_T3_9,
    MAT_PLAS_T3_10,
    MAT_FABRIC_1,
    MAT_FABRIC_2,
    MAT_FABRIC_3,
    MAT_FABRIC_4,
    MAT_FABRIC_5,
    MAT_FABRIC_6,
    MAT_FABRIC_7,
    MAT_FABRIC_8,
    MAT_FABRIC_9,
    MAT_FABRIC_10,
    LAYOUTS,
)
from pyradioss.input.cfg_catalogue import LAW_MAP, LAW_SYNONYMS, law_number, canonical_law_name
from pyradioss.model.entities import MatLaw60, MatFabric
from pyradioss.model.model import Model
from pyradioss.input.starter_keywords import parse_starter_deck, read_mat_law60, KEYWORD_PARSERS
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck, _conv_mat
from pyradioss.starter.checks import check_mat_law60, _check_mat_law60, check_materials, check_model, _ALLOWED_LAWS


def test_m551_card_layouts():
    assert tuple(MAT_LAW60_1) == (20, 20)
    assert tuple(MAT_LAW60_2) == (20, 20, 20, 20, 20)
    assert tuple(MAT_LAW60_3) == (10, 10, 20, 20)
    assert tuple(MAT_LAW60_4) == (10, 20, 10, 20, 20)
    assert tuple(MAT_LAW60_5) == (10, 10, 10, 10, 10)
    assert tuple(MAT_LAW60_6) == (10, 10, 10, 10, 10)
    assert tuple(MAT_LAW60_7) == (20, 20, 20, 20, 20)
    assert tuple(MAT_LAW60_8) == (20, 20, 20, 20, 20)
    assert tuple(MAT_LAW60_9) == (20, 20, 20, 20, 20)
    assert tuple(MAT_LAW60_10) == (20, 20, 20, 20, 20)

    for i in range(1, 11):
        l_law = globals()[f"MAT_LAW60_{i}"]
        l_plas = globals()[f"MAT_PLAS_T3_{i}"]
        l_fab = globals()[f"MAT_FABRIC_{i}"]
        assert l_plas == l_law
        assert l_fab == l_law
        assert f"MAT_LAW60_{i}" in LAYOUTS
        assert f"MAT_PLAS_T3_{i}" in LAYOUTS
        assert f"MAT_FABRIC_{i}" in LAYOUTS
        assert list(LAYOUTS[f"MAT_LAW60_{i}"]) == list(l_law)
        assert list(LAYOUTS[f"MAT_PLAS_T3_{i}"]) == list(l_law)
        assert list(LAYOUTS[f"MAT_FABRIC_{i}"]) == list(l_law)


def test_m551_cfg_catalogue():
    for name in ("LAW60", "PLAS_T3", "MAT_PLAS_T3", "FABRIC", "MAT_FABRIC", "MAT_LAW60"):
        assert LAW_MAP[name] == 60
        assert law_number(name) == 60
        assert canonical_law_name(name) == "LAW60"

    assert LAW_SYNONYMS["LAW60"] == "LAW60"
    assert LAW_SYNONYMS["PLAS_T3"] == "LAW60"
    assert LAW_SYNONYMS["FABRIC"] == "LAW60"


def test_m551_keyword_parsers_dispatch():
    for k in ("LAW60", "MAT_LAW60", "PLAS_T3", "MAT_PLAS_T3", "MAT_FABRIC"):
        assert KEYWORD_PARSERS[k] is read_mat_law60


def test_m551_model_entities_matlaw60():
    m = MatLaw60(
        id=60,
        rho=1.5e-6,
        ref_rho=1.5e-6,
        e=2000.0,
        nu=0.35,
        eps_p_max=0.5,
        eps_t1=0.4,
        eps_t2=0.6,
        nfunc=5,
        fsmooth=1,
        mat_hard=0.2,
        fcut=100.0,
        xr_fun=2,
        ifunce=1,
        mat_fscale=1.2,
        einf=500.0,
        ce=10.0,
        funcs=[101, 102],
        fscales=[1.0, 1.1],
        rates=[0.0, 50.0],
        title="Test Fabric Mat",
    )
    assert m.id == 60
    assert m.rho == 1.5e-6
    assert m.rho0 == 1.5e-6
    assert m.ref_rho == 1.5e-6
    assert m.refer_rho == 1.5e-6
    assert m.e == 2000.0
    assert m.E == 2000.0
    assert m.nu == 0.35
    assert m.eps_p_max == 0.5
    assert m.eps_t1 == 0.4
    assert m.eps_t2 == 0.6
    assert m.nfunc == 5
    assert m.fsmooth == 1
    assert m.mat_hard == 0.2
    assert m.chard == 0.2
    assert m.fcut == 100.0
    assert m.xr_fun == 2
    assert m.ifunce == 1
    assert m.mat_fscale == 1.2
    assert m.fpscale == 1.2
    assert m.einf == 500.0
    assert m.ce == 10.0
    assert m.funcs == [101, 102]
    assert m.fun_ids == [101, 102]
    assert m.fscales == [1.0, 1.1]
    assert m.rates == [0.0, 50.0]
    assert m.eps_rates == [0.0, 50.0]
    assert m.title == "Test Fabric Mat"

    # Test alias
    assert MatFabric is MatLaw60

    # Test legacy keyword arguments initialization
    m_legacy = MatLaw60(
        id=61,
        rho=2.0e-6,
        refer_rho=2.0e-6,
        chard=0.15,
        fpscale=1.05,
        fun_ids=[201, 202],
        eps_rates=[1.0, 10.0],
    )
    assert m_legacy.ref_rho == 2.0e-6
    assert m_legacy.mat_hard == 0.15
    assert m_legacy.mat_fscale == 1.05
    assert m_legacy.funcs == [201, 202]
    assert m_legacy.rates == [1.0, 10.0]


def test_m551_parse_fixed_format_5curves(tmp_path):
    rad = """/BEGIN
TEST_FIXED_60_5
/MAT/LAW60/1
Law60 Fixed 5 Curves
         1.50000E-06         1.50000E-06
         2.00000E+03               0.350               0.500               0.400               0.600
         5         1         2.00000E-01         1.00000E+02
         2         1.20000E+00         1         5.00000E+02         1.00000E+01
        11        12        13
         1.00000E+00         1.10000E+00         1.20000E+00
         0.00000E+00         1.00000E+01         1.00000E+02
"""
    f = tmp_path / "deck_fixed_0000.rad"
    f.write_text(rad, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(f))
    parse_starter_deck(blocks, model, log)

    assert 1 in model.mat_law60s
    m = model.mat_law60s[1]
    assert m.id == 1
    assert m.rho == pytest.approx(1.5e-6)
    assert m.ref_rho == pytest.approx(1.5e-6)
    assert m.e == pytest.approx(2000.0)
    assert m.nu == pytest.approx(0.35)
    assert m.eps_p_max == pytest.approx(0.5)
    assert m.eps_t1 == pytest.approx(0.4)
    assert m.eps_t2 == pytest.approx(0.6)
    assert m.nfunc == 5
    assert m.fsmooth == 1
    assert m.mat_hard == pytest.approx(0.2)
    assert m.fcut == pytest.approx(100.0)
    assert m.xr_fun == 2
    assert m.mat_fscale == pytest.approx(1.2)
    assert m.ifunce == 1
    assert m.einf == pytest.approx(500.0)
    assert m.ce == pytest.approx(10.0)
    assert m.funcs == [11, 12, 13]
    assert m.fscales == [pytest.approx(1.0), pytest.approx(1.1), pytest.approx(1.2)]
    assert m.rates == [pytest.approx(0.0), pytest.approx(10.0), pytest.approx(100.0)]

    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 60
    assert mat.rho0 == pytest.approx(1.5e-6)
    assert mat.law_name == "LAW60"


def test_m551_parse_fixed_format_10curves(tmp_path):
    rad = """/BEGIN
TEST_FIXED_60_10
/MAT/LAW60/2
Law60 Fixed 10 Curves
         1.20000E-06
         3.00000E+03               0.300         1.00000E+30         1.00000E+30         2.00000E+30
        10         0         0.00000E+00         1.00000E+30
         0         1.00000E+00         0         0.00000E+00         0.00000E+00
         1         2         3         4         5
         6         7         8
         1.00000E+00         1.00000E+00         1.00000E+00         1.00000E+00         1.00000E+00
         1.00000E+00         1.00000E+00         1.00000E+00
         0.00000E+00         1.00000E+01         5.00000E+01         1.00000E+02         2.00000E+02
         5.00000E+02         1.00000E+03         2.00000E+03
"""
    f = tmp_path / "deck_fixed_10_0000.rad"
    f.write_text(rad, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(f))
    parse_starter_deck(blocks, model, log)

    assert 2 in model.mat_law60s
    m = model.mat_law60s[2]
    assert m.id == 2
    assert m.nfunc == 10
    assert m.funcs == [1, 2, 3, 4, 5, 6, 7, 8]
    assert len(m.fscales) == 8
    assert len(m.rates) == 8
    assert m.rates[7] == pytest.approx(2000.0)


def test_m551_parse_free_format_and_aliases(tmp_path):
    # Free format with space and comma delimiters, testing /MAT/PLAS_T3 and /MAT/FABRIC
    rad = """# RADIOSS STARTER
/BEGIN
TEST_FREE_60
/MAT/PLAS_T3/3
Plas T3 Free Format
1.4e-6, 1.4e-6
2500.0, 0.33, 0.45, 0.35, 0.55
5, 0, 0.1, 80.0
1, 1.0, 0, 0.0, 0.0
21, 22
1.0, 1.05
0.0, 50.0
/MAT/FABRIC/4
Fabric Alias Free Space
1.6e-6
1800.0 0.38 0.5 0.4 0.6
5 0 0.0 1.0e30
0 1.0 0 0.0 0.0
31
1.0
0.0
"""
    f = tmp_path / "deck_free_0000.rad"
    f.write_text(rad, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(f))
    parse_starter_deck(blocks, model, log)

    assert 3 in model.mat_law60s
    m3 = model.mat_law60s[3]
    assert m3.id == 3
    assert m3.rho == pytest.approx(1.4e-6)
    assert m3.e == pytest.approx(2500.0)
    assert m3.funcs == [21, 22]
    assert m3.rates == [pytest.approx(0.0), pytest.approx(50.0)]
    assert model.materials[3].law == 60

    assert 4 in model.mat_law60s
    m4 = model.mat_law60s[4]
    assert m4.id == 4
    assert m4.rho == pytest.approx(1.6e-6)
    assert m4.e == pytest.approx(1800.0)
    assert m4.funcs == [31]
    assert model.materials[4].law == 60


def test_m551_deck_writer_roundtrip(tmp_path):
    deck = StarterDeck("LAW60_ROUNDTRIP")
    # Fixed format with 5 curves
    deck.mat_law60(
        mid=1,
        title="Fabric Mat 1",
        rho=1.5e-6,
        ref_rho=1.5e-6,
        e=2200.0,
        nu=0.34,
        eps_p_max=0.45,
        eps_t1=0.35,
        eps_t2=0.55,
        nfunc=5,
        fsmooth=1,
        mat_hard=0.15,
        fcut=90.0,
        xr_fun=2,
        ifunce=1,
        mat_fscale=1.1,
        einf=600.0,
        ce=12.0,
        funcs=[10, 11, 12],
        fscales=[1.0, 1.05, 1.1],
        rates=[0.0, 20.0, 200.0],
        fixed_format=True,
    )
    # Free format with PLAS_T3 alias
    deck.mat_plas_t3(
        mid=2,
        title="Plas T3 Mat 2",
        rho=1.4e-6,
        e=2400.0,
        nu=0.32,
        funcs=[20],
        fscales=[1.0],
        rates=[0.0],
        fixed_format=False,
    )
    # FABRIC alias
    deck.mat_fabric(
        mid=3,
        title="Fabric Mat 3",
        rho=1.6e-6,
        e=1900.0,
        nu=0.36,
        funcs=[30, 31],
        fscales=[1.0, 1.0],
        rates=[0.0, 10.0],
        fixed_format=True,
    )

    deck_path = tmp_path / "test_law60_0000.rad"
    deck.write(str(deck_path))

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert 1 in model.mat_law60s
    m1 = model.mat_law60s[1]
    assert m1.rho == pytest.approx(1.5e-6)
    assert m1.ref_rho == pytest.approx(1.5e-6)
    assert m1.e == pytest.approx(2200.0)
    assert m1.nu == pytest.approx(0.34)
    assert m1.eps_p_max == pytest.approx(0.45)
    assert m1.eps_t1 == pytest.approx(0.35)
    assert m1.eps_t2 == pytest.approx(0.55)
    assert m1.nfunc == 5
    assert m1.fsmooth == 1
    assert m1.mat_hard == pytest.approx(0.15)
    assert m1.fcut == pytest.approx(90.0)
    assert m1.xr_fun == 2
    assert m1.mat_fscale == pytest.approx(1.1)
    assert m1.ifunce == 1
    assert m1.einf == pytest.approx(600.0)
    assert m1.ce == pytest.approx(12.0)
    assert m1.funcs == [10, 11, 12]
    assert m1.fscales == [pytest.approx(1.0), pytest.approx(1.05), pytest.approx(1.1)]
    assert m1.rates == [pytest.approx(0.0), pytest.approx(20.0), pytest.approx(200.0)]

    assert 2 in model.mat_law60s
    m2 = model.mat_law60s[2]
    assert m2.rho == pytest.approx(1.4e-6)
    assert m2.e == pytest.approx(2400.0)
    assert m2.funcs == [20]

    assert 3 in model.mat_law60s
    m3 = model.mat_law60s[3]
    assert m3.rho == pytest.approx(1.6e-6)
    assert m3.e == pytest.approx(1900.0)
    assert m3.funcs == [30, 31]


def test_m551_deck_writer_object_invocation(tmp_path):
    m = MatLaw60(
        id=5,
        rho=1.3e-6,
        ref_rho=1.3e-6,
        e=2100.0,
        nu=0.33,
        eps_p_max=0.6,
        eps_t1=0.5,
        eps_t2=0.7,
        nfunc=5,
        fsmooth=0,
        mat_hard=0.25,
        fcut=150.0,
        xr_fun=0,
        ifunce=0,
        mat_fscale=1.0,
        einf=0.0,
        ce=0.0,
        funcs=[50],
        fscales=[1.0],
        rates=[0.0],
        title="Direct Object Mat",
    )
    deck = StarterDeck("OBJ_TEST")
    deck.mat_law60(m)
    deck_path = tmp_path / "obj_test_0000.rad"
    deck.write(str(deck_path))

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert 5 in model.mat_law60s
    assert model.mat_law60s[5].e == pytest.approx(2100.0)
    assert model.mat_law60s[5].funcs == [50]


def test_m551_deck_writer_conv_mat(tmp_path):
    deck = StarterDeck("CONV_MAT_TEST")
    rad_lines = [
        "/BEGIN",
        "CONV_MAT_TEST",
        "/MAT/LAW60/10",
        "Test Conv Mat",
        "1.5e-6 1.5e-6",
        "2000.0 0.3 0.5 0.4 0.6",
        "5 0 0.1 100.0",
        "0 1.0 0 0.0 0.0",
        "1 2",
        "1.0 1.0",
        "0.0 10.0",
    ]
    deck_path = tmp_path / "conv_mat_0000.rad"
    deck_path.write_text("\n".join(rad_lines), encoding="utf-8")
    blocks = read_deck(str(deck_path))
    for b in blocks:
        if b.key0 == "MAT":
            _conv_mat(deck, b)

    rendered = deck.render()
    assert "/MAT/LAW60/10" in rendered
    assert "Test Conv Mat" in rendered


def test_m551_starter_checks():
    class DummyMat:
        def __init__(self, **kwargs):
            self.id = kwargs.get("id", 1)
            self.rho0 = kwargs.get("rho0", 1.5e-6)
            self.rho = self.rho0
            self.ref_rho = kwargs.get("ref_rho", self.rho0)
            self.e = kwargs.get("e", 2000.0)
            self.E = self.e
            self.nu = kwargs.get("nu", 0.3)
            self.eps_p_max = kwargs.get("eps_p_max", 0.5)
            self.eps_t1 = kwargs.get("eps_t1", 0.4)
            self.eps_t2 = kwargs.get("eps_t2", 0.6)
            self.nfunc = kwargs.get("nfunc", 5)
            self.fsmooth = kwargs.get("fsmooth", 0)
            self.mat_hard = kwargs.get("mat_hard", 0.0)
            self.fcut = kwargs.get("fcut", 1.0e30)
            self.xr_fun = kwargs.get("xr_fun", 0)
            self.ifunce = kwargs.get("ifunce", 0)
            self.mat_fscale = kwargs.get("mat_fscale", 1.0)
            self.einf = kwargs.get("einf", 0.0)
            self.ce = kwargs.get("ce", 0.0)
            self.funcs = kwargs.get("funcs", [1])
            self.fscales = kwargs.get("fscales", [1.0])
            self.rates = kwargs.get("rates", [0.0])

    # 1. Valid material with existing curve
    m_valid = DummyMat()
    model = Model()
    model.functions[1] = FunctTable(1, [0.0, 0.1], [0.0, 10.0])
    log = MessageLog()
    check_mat_law60(m_valid, log=log, model=model)
    assert not log.has_errors

    # 2. Alias _check_mat_law60
    log = MessageLog()
    _check_mat_law60(m_valid, model, log)
    assert not log.has_errors

    # 3. rho0 <= 0
    m_bad_rho = DummyMat(rho0=0.0)
    log = MessageLog()
    check_mat_law60(m_bad_rho, log)
    assert log.has_errors
    assert any("density" in str(e).lower() or "rho" in str(e).lower() for e in log.errors)

    # 4. E <= 0
    m_bad_e = DummyMat(e=0.0)
    log = MessageLog()
    check_mat_law60(m_bad_e, log)
    assert log.has_errors
    assert any("young" in str(e).lower() or "e" in str(e).lower() for e in log.errors)

    # 5. nu < 0 or nu >= 0.5
    m_bad_nu1 = DummyMat(nu=-0.1)
    log = MessageLog()
    check_mat_law60(m_bad_nu1, log)
    assert log.has_errors
    assert any("poisson" in str(e).lower() or "nu" in str(e).lower() for e in log.errors)

    m_bad_nu2 = DummyMat(nu=0.5)
    log = MessageLog()
    check_mat_law60(m_bad_nu2, log)
    assert log.has_errors
    assert any("poisson" in str(e).lower() or "nu" in str(e).lower() for e in log.errors)

    # 6. Failure strains: eps_t1 >= eps_t2
    m_bad_fail = DummyMat(eps_t1=0.6, eps_t2=0.4)
    log = MessageLog()
    check_mat_law60(m_bad_fail, log)
    assert log.has_errors
    assert any("eps_t1" in str(e).lower() for e in log.errors)

    # 7. Non-monotonic strain rates
    m_bad_rates = DummyMat(rates=[10.0, 5.0])
    log = MessageLog()
    check_mat_law60(m_bad_rates, log)
    assert log.has_errors
    assert any("strain rates" in str(e).lower() or "strictly increasing" in str(e).lower() for e in log.errors)

    # 8. Missing curve in functions/tables
    m_missing_curve = DummyMat(funcs=[99])
    log = MessageLog()
    check_mat_law60(m_missing_curve, log=log, model=model)
    assert log.has_errors
    assert any("99" in str(e) for e in log.errors)

    # 9. Allowed element families check
    for part in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
        assert 60 in _ALLOWED_LAWS[part]
        assert "60" in _ALLOWED_LAWS[part]
        assert "LAW60" in _ALLOWED_LAWS[part]
        assert "PLAS_T3" in _ALLOWED_LAWS[part]
        assert "FABRIC" in _ALLOWED_LAWS[part]
        assert "MAT_PLAS_T3" in _ALLOWED_LAWS[part]
        assert "MAT_FABRIC" in _ALLOWED_LAWS[part]

    # Trusses, beams, springs must reject LAW60
    for rejected in ("trusses", "beams"):
        assert 60 not in _ALLOWED_LAWS[rejected]
        assert "LAW60" not in _ALLOWED_LAWS[rejected]


def test_m551_check_materials_model():
    model = Model()
    m_valid = MatLaw60(id=1, rho=1.5e-6, e=2000.0, nu=0.3)
    m_valid.law = 60
    model.materials[1] = m_valid
    log = MessageLog()
    check_materials(model, log)
    assert not log.has_errors

    # Invalid Young's modulus
    m_bad = MatLaw60(id=2, rho=1.5e-6, e=-100.0, nu=0.3)
    m_bad.law = 60
    model.materials[2] = m_bad
    log = MessageLog()
    check_materials(model, log)
    assert log.has_errors
    assert any("young" in str(e).lower() or "e" in str(e).lower() for e in log.errors)
