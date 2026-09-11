"""
Tests for M552 (/MAT/LAW48 /MAT/ZHAO /MAT/PLAS_ZHAO) input, starter, model, and checks integration.
"""

from io import StringIO
import math
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.input.card_layouts import (
    MAT_LAW48_1,
    MAT_LAW48_2,
    MAT_LAW48_3,
    MAT_LAW48_4,
    MAT_LAW48_5,
    MAT_LAW48_6,
    MAT_ZHAO_1,
    MAT_ZHAO_2,
    MAT_ZHAO_3,
    MAT_ZHAO_4,
    MAT_ZHAO_5,
    MAT_ZHAO_6,
    MAT_PLAS_ZHAO_1,
    MAT_PLAS_ZHAO_2,
    MAT_PLAS_ZHAO_3,
    MAT_PLAS_ZHAO_4,
    MAT_PLAS_ZHAO_5,
    MAT_PLAS_ZHAO_6,
    LAYOUTS,
)
from pyradioss.input.cfg_catalogue import LAW_MAP, LAW_SYNONYMS, law_number, canonical_law_name
from pyradioss.model.entities import MatLaw48, MatZhao, MatPlasZhao, Material
from pyradioss.model.model import Model
from pyradioss.input.starter_keywords import parse_starter_deck, read_mat_law48, KEYWORD_PARSERS
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck, _conv_mat
from pyradioss.starter.checks import check_mat_law48, _check_mat_law48, check_materials, check_model, _ALLOWED_LAWS


def test_m552_card_layouts():
    assert tuple(MAT_LAW48_1) == (20, 20)
    assert tuple(MAT_LAW48_2) == (20, 20)
    assert tuple(MAT_LAW48_3) == (20, 20, 20, 20, 20)
    assert tuple(MAT_LAW48_4) == (20, 20, 20, 20, 20)
    assert tuple(MAT_LAW48_5) == (20, 20)
    assert tuple(MAT_LAW48_6) == (20, 20, 20)

    for i in range(1, 7):
        l_law = globals()[f"MAT_LAW48_{i}"]
        l_zhao = globals()[f"MAT_ZHAO_{i}"]
        l_plas = globals()[f"MAT_PLAS_ZHAO_{i}"]
        assert l_zhao == l_law
        assert l_plas == l_law
        assert f"MAT_LAW48_{i}" in LAYOUTS
        assert f"MAT_ZHAO_{i}" in LAYOUTS
        assert f"MAT_PLAS_ZHAO_{i}" in LAYOUTS
        assert list(LAYOUTS[f"MAT_LAW48_{i}"]) == list(l_law)
        assert list(LAYOUTS[f"MAT_ZHAO_{i}"]) == list(l_law)
        assert list(LAYOUTS[f"MAT_PLAS_ZHAO_{i}"]) == list(l_law)


def test_m552_cfg_catalogue():
    for name in ("LAW48", "ZHAO", "PLAS_ZHAO", "MAT_ZHAO", "MAT_PLAS_ZHAO", "MAT_LAW48", "LAW48_ZHAO"):
        assert LAW_MAP[name] == 48
        assert law_number(name) == 48
        assert canonical_law_name(name) == "LAW48"

    assert LAW_SYNONYMS["LAW48"] == "LAW48"
    assert LAW_SYNONYMS["ZHAO"] == "LAW48"
    assert LAW_SYNONYMS["PLAS_ZHAO"] == "LAW48"


def test_m552_keyword_parsers_dispatch():
    for k in ("LAW48", "MAT_LAW48", "ZHAO", "MAT_ZHAO", "PLAS_ZHAO", "MAT_PLAS_ZHAO"):
        assert KEYWORD_PARSERS[k] is read_mat_law48


def test_m552_model_entities_matlaw48():
    m = MatLaw48(
        id=48,
        rho=7.8e-6,
        refer_rho=7.8e-6,
        e=210000.0,
        nu=0.3,
        a=250.0,
        b=500.0,
        n=0.45,
        chard=0.1,
        sig_max=800.0,
        c=0.02,
        d=0.01,
        m=0.35,
        e1=100.0,
        k=0.8,
        eps_rate_0=1.0,
        fcut=1000.0,
        eps_max=0.5,
        eps_t1=0.4,
        eps_t2=0.6,
        title="Test Zhao Steel",
    )
    assert m.id == 48
    assert m.rho == pytest.approx(7.8e-6)
    assert m.rho0 == pytest.approx(7.8e-6)
    assert m.ref_rho == pytest.approx(7.8e-6)
    assert m.refer_rho == pytest.approx(7.8e-6)
    assert m.e == pytest.approx(210000.0)
    assert m.E == pytest.approx(210000.0)
    assert m.nu == pytest.approx(0.3)
    assert m.a == pytest.approx(250.0)
    assert m.sigy == pytest.approx(250.0)
    assert m.b == pytest.approx(500.0)
    assert m.n == pytest.approx(0.45)
    assert m.chard == pytest.approx(0.1)
    assert m.hard == pytest.approx(0.45)
    assert m.mat_hard == pytest.approx(0.1)
    assert m.fisokin == pytest.approx(0.1)
    assert m.sig_max == pytest.approx(800.0)
    assert m.sigma_max == pytest.approx(800.0)
    assert m.c == pytest.approx(0.02)
    assert m.d == pytest.approx(0.01)
    assert m.m == pytest.approx(0.35)
    assert m.e1 == pytest.approx(100.0)
    assert m.k == pytest.approx(0.8)
    assert m.eps_rate_0 == pytest.approx(1.0)
    assert m.eps0 == pytest.approx(1.0)
    assert m.fcut == pytest.approx(1000.0)
    assert m.scale == pytest.approx(1000.0)
    assert m.eps_max == pytest.approx(0.5)
    assert m.eps_t1 == pytest.approx(0.4)
    assert m.eta1 == pytest.approx(0.4)
    assert m.eps_t2 == pytest.approx(0.6)
    assert m.eta2 == pytest.approx(0.6)
    assert m.title == "Test Zhao Steel"

    # Computed elastic properties
    expected_G = 210000.0 / (2.0 * (1.0 + 0.3))
    expected_K = 210000.0 / (3.0 * (1.0 - 2.0 * 0.3))
    assert m.G == pytest.approx(expected_G)
    assert m.K == pytest.approx(expected_K)

    expected_c = math.sqrt(210000.0 / 7.8e-6)
    expected_c_shell = math.sqrt(210000.0 / ((1.0 - 0.3**2) * 7.8e-6))
    expected_c_solid = math.sqrt((expected_K + 4.0 * expected_G / 3.0) / 7.8e-6)
    assert m.sound_speed == pytest.approx(expected_c)
    assert m.sound_speed_shell() == pytest.approx(expected_c_shell)
    assert m.sound_speed_solid() == pytest.approx(expected_c_solid)

    # Test property setters
    m.ref_rho = 8.0e-6
    assert m.refer_rho == pytest.approx(8.0e-6)
    m.sigy = 300.0
    assert m.a == pytest.approx(300.0)
    m.hard = 0.5
    assert m.n == pytest.approx(0.5)
    m.mat_hard = 0.3
    assert m.chard == pytest.approx(0.3)
    m.fisokin = 0.4
    assert m.chard == pytest.approx(0.4)
    m.E = 200000.0
    assert m.e == pytest.approx(200000.0)
    m.eps0 = 2.0
    assert m.eps_rate_0 == pytest.approx(2.0)
    m.scale = 500.0
    assert m.fcut == pytest.approx(500.0)
    m.sigma_max = 900.0
    assert m.sig_max == pytest.approx(900.0)
    m.eta1 = 0.35
    assert m.eps_t1 == pytest.approx(0.35)
    m.eta2 = 0.65
    assert m.eps_t2 == pytest.approx(0.65)

    # Class aliases
    assert MatZhao is MatLaw48
    assert MatPlasZhao is MatLaw48


def test_m552_parse_fixed_format(tmp_path):
    rad = """/BEGIN
TEST_FIXED_48
/MAT/LAW48/1
Zhao Fixed Format Deck
         7.85000E-06         7.85000E-06
         2.10000E+05               0.300
         2.50000E+02         5.00000E+02               0.450               0.100         8.00000E+02
         2.00000E-02         1.00000E-02               0.350         1.00000E+02               0.800
         1.00000E+00         1.00000E+03
         5.00000E-01         4.00000E-01         6.00000E-01
"""
    f = tmp_path / "deck_fixed_0000.rad"
    f.write_text(rad, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(f))
    parse_starter_deck(blocks, model, log)

    assert 1 in model.mat_law48s
    m = model.mat_law48s[1]
    assert m.id == 1
    assert m.rho == pytest.approx(7.85e-6)
    assert m.refer_rho == pytest.approx(7.85e-6)
    assert m.e == pytest.approx(210000.0)
    assert m.nu == pytest.approx(0.3)
    assert m.a == pytest.approx(250.0)
    assert m.b == pytest.approx(500.0)
    assert m.n == pytest.approx(0.45)
    assert m.chard == pytest.approx(0.1)
    assert m.sig_max == pytest.approx(800.0)
    assert m.c == pytest.approx(0.02)
    assert m.d == pytest.approx(0.01)
    assert m.m == pytest.approx(0.35)
    assert m.e1 == pytest.approx(100.0)
    assert m.k == pytest.approx(0.8)
    assert m.eps_rate_0 == pytest.approx(1.0)
    assert m.fcut == pytest.approx(1000.0)
    assert m.eps_max == pytest.approx(0.5)
    assert m.eps_t1 == pytest.approx(0.4)
    assert m.eps_t2 == pytest.approx(0.6)

    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 48
    assert mat.rho0 == pytest.approx(7.85e-6)
    assert mat.params["sigy"] == pytest.approx(250.0)


def test_m552_parse_free_format_and_aliases(tmp_path):
    rad = """# RADIOSS STARTER
/BEGIN
TEST_FREE_48
/MAT/ZHAO/2
Zhao Free Comma Format
7.8e-6, 7.8e-6
200000.0, 0.28
300.0, 450.0, 0.5, 0.2, 750.0
0.015, 0.005, 0.4, 80.0, 0.9
1.0, 500.0
0.45, 0.35, 0.55
/MAT/PLAS_ZHAO/3
Plas Zhao Free Space Format
7.9e-6
190000.0 0.32
280.0 400.0 0.4 0.0 1.0e30
0.0 0.0 1.0 0.0 1.0
1.0 1.0e30
1.0e30 1.0e30 2.0e30
"""
    f = tmp_path / "deck_free_0000.rad"
    f.write_text(rad, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(f))
    parse_starter_deck(blocks, model, log)

    assert 2 in model.mat_law48s
    m2 = model.mat_law48s[2]
    assert m2.id == 2
    assert m2.rho == pytest.approx(7.8e-6)
    assert m2.e == pytest.approx(200000.0)
    assert m2.a == pytest.approx(300.0)
    assert m2.b == pytest.approx(450.0)
    assert model.materials[2].law == 48

    assert 3 in model.mat_law48s
    m3 = model.mat_law48s[3]
    assert m3.id == 3
    assert m3.rho == pytest.approx(7.9e-6)
    assert m3.e == pytest.approx(190000.0)
    assert m3.a == pytest.approx(280.0)
    assert model.materials[3].law == 48


def test_m552_deck_writer_roundtrip(tmp_path):
    deck = StarterDeck("LAW48_ROUNDTRIP")
    # Fixed format
    deck.mat_law48(
        mid=1,
        title="Zhao Mat 1 Fixed",
        rho=7.85e-6,
        ref_rho=7.85e-6,
        e=210000.0,
        nu=0.3,
        a=260.0,
        b=480.0,
        n=0.42,
        chard=0.15,
        sig_max=850.0,
        c=0.025,
        d=0.012,
        m=0.38,
        e1=120.0,
        k=0.75,
        eps_rate_0=1.0,
        fcut=1200.0,
        eps_max=0.55,
        eps_t1=0.42,
        eps_t2=0.62,
        fixed_format=True,
    )
    # Free format with ZHAO alias
    deck.mat_zhao(
        mid=2,
        title="Zhao Mat 2 Free",
        rho=7.8e-6,
        e=205000.0,
        nu=0.29,
        a=240.0,
        b=420.0,
        n=0.5,
        chard=0.0,
        sig_max=700.0,
        c=0.01,
        d=0.005,
        m=0.3,
        e1=50.0,
        k=1.0,
        eps_rate_0=1.0,
        fcut=800.0,
        eps_max=0.4,
        eps_t1=0.3,
        eps_t2=0.5,
        fixed_format=False,
    )
    # PLAS_ZHAO alias
    deck.mat_plas_zhao(
        mid=3,
        title="Plas Zhao Mat 3",
        rho=7.9e-6,
        e=195000.0,
        nu=0.31,
        a=270.0,
        b=460.0,
        n=0.44,
        chard=0.25,
        sig_max=780.0,
        c=0.02,
        d=0.01,
        m=0.36,
        e1=90.0,
        k=0.85,
        eps_rate_0=1.0,
        fcut=1100.0,
        eps_max=0.48,
        eps_t1=0.38,
        eps_t2=0.58,
        fixed_format=True,
    )

    deck_path = tmp_path / "test_law48_0000.rad"
    deck.write(str(deck_path))

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert 1 in model.mat_law48s
    m1 = model.mat_law48s[1]
    assert m1.rho == pytest.approx(7.85e-6)
    assert m1.refer_rho == pytest.approx(7.85e-6)
    assert m1.e == pytest.approx(210000.0)
    assert m1.nu == pytest.approx(0.3)
    assert m1.a == pytest.approx(260.0)
    assert m1.b == pytest.approx(480.0)
    assert m1.n == pytest.approx(0.42)
    assert m1.chard == pytest.approx(0.15)
    assert m1.sig_max == pytest.approx(850.0)
    assert m1.c == pytest.approx(0.025)
    assert m1.d == pytest.approx(0.012)
    assert m1.m == pytest.approx(0.38)
    assert m1.e1 == pytest.approx(120.0)
    assert m1.k == pytest.approx(0.75)
    assert m1.eps_rate_0 == pytest.approx(1.0)
    assert m1.fcut == pytest.approx(1200.0)
    assert m1.eps_max == pytest.approx(0.55)
    assert m1.eps_t1 == pytest.approx(0.42)
    assert m1.eps_t2 == pytest.approx(0.62)

    assert 2 in model.mat_law48s
    m2 = model.mat_law48s[2]
    assert m2.rho == pytest.approx(7.8e-6)
    assert m2.e == pytest.approx(205000.0)
    assert m2.a == pytest.approx(240.0)

    assert 3 in model.mat_law48s
    m3 = model.mat_law48s[3]
    assert m3.rho == pytest.approx(7.9e-6)
    assert m3.e == pytest.approx(195000.0)
    assert m3.a == pytest.approx(270.0)


def test_m552_deck_writer_object_invocation(tmp_path):
    m = MatLaw48(
        id=5,
        rho=7.82e-6,
        refer_rho=7.82e-6,
        e=208000.0,
        nu=0.295,
        a=255.0,
        b=470.0,
        n=0.43,
        chard=0.12,
        sig_max=820.0,
        c=0.018,
        d=0.008,
        m=0.34,
        e1=95.0,
        k=0.82,
        eps_rate_0=1.0,
        fcut=1050.0,
        eps_max=0.52,
        eps_t1=0.41,
        eps_t2=0.61,
        title="Direct Object Mat",
    )
    deck = StarterDeck("OBJ_TEST")
    deck.mat_law48(m)
    deck_path = tmp_path / "obj_test_0000.rad"
    deck.write(str(deck_path))

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert 5 in model.mat_law48s
    assert model.mat_law48s[5].e == pytest.approx(208000.0)
    assert model.mat_law48s[5].a == pytest.approx(255.0)


def test_m552_deck_writer_conv_mat(tmp_path):
    deck = StarterDeck("CONV_MAT_TEST")
    rad_lines = [
        "/BEGIN",
        "CONV_MAT_TEST",
        "/MAT/LAW48/10",
        "Test Conv Mat",
        "7.85e-6 7.85e-6",
        "210000.0 0.3",
        "250.0 500.0 0.45 0.1 800.0",
        "0.02 0.01 0.35 100.0 0.8",
        "1.0 1000.0",
        "0.5 0.4 0.6",
    ]
    deck_path = tmp_path / "conv_mat_0000.rad"
    deck_path.write_text("\n".join(rad_lines), encoding="utf-8")
    blocks = read_deck(str(deck_path))
    for b in blocks:
        if b.key0 == "MAT":
            _conv_mat(deck, b)

    rendered = deck.render()
    assert "/MAT/LAW48/10" in rendered
    assert "Test Conv Mat" in rendered


def test_m552_starter_checks():
    class DummyMat:
        def __init__(self, **kwargs):
            self.id = kwargs.get("id", 1)
            self.rho0 = kwargs.get("rho0", 7.85e-6)
            self.rho = self.rho0
            self.ref_rho = kwargs.get("ref_rho", self.rho0)
            self.e = kwargs.get("e", 210000.0)
            self.E = self.e
            self.nu = kwargs.get("nu", 0.3)
            self.a = kwargs.get("a", 250.0)
            self.sigy = self.a
            self.b = kwargs.get("b", 500.0)
            self.n = kwargs.get("n", 0.45)
            self.chard = kwargs.get("chard", 0.1)
            self.sig_max = kwargs.get("sig_max", 800.0)
            self.c = kwargs.get("c", 0.02)
            self.d = kwargs.get("d", 0.01)
            self.m = kwargs.get("m", 0.35)
            self.e1 = kwargs.get("e1", 100.0)
            self.k = kwargs.get("k", 0.8)
            self.eps_rate_0 = kwargs.get("eps_rate_0", 1.0)
            self.fcut = kwargs.get("fcut", 1000.0)
            self.eps_max = kwargs.get("eps_max", 0.5)
            self.eps_t1 = kwargs.get("eps_t1", 0.4)
            self.eps_t2 = kwargs.get("eps_t2", 0.6)

    # 1. Valid material
    m_valid = DummyMat()
    model = Model()
    log = MessageLog()
    check_mat_law48(m_valid, log=log, model=model)
    assert not log.has_errors

    # 2. Alias _check_mat_law48
    log = MessageLog()
    _check_mat_law48(m_valid, model, log)
    assert not log.has_errors

    # 3. rho0 <= 0
    m_bad_rho = DummyMat(rho0=0.0)
    log = MessageLog()
    check_mat_law48(m_bad_rho, log)
    assert log.has_errors
    assert any("density" in str(e).lower() or "rho" in str(e).lower() for e in log.errors)

    # 4. E <= 0
    m_bad_e = DummyMat(e=0.0)
    log = MessageLog()
    check_mat_law48(m_bad_e, log)
    assert log.has_errors
    assert any("young" in str(e).lower() or "e" in str(e).lower() for e in log.errors)

    # 5. nu < 0 or nu >= 0.5
    m_bad_nu1 = DummyMat(nu=-0.05)
    log = MessageLog()
    check_mat_law48(m_bad_nu1, log)
    assert log.has_errors
    assert any("poisson" in str(e).lower() or "nu" in str(e).lower() for e in log.errors)

    m_bad_nu2 = DummyMat(nu=0.5)
    log = MessageLog()
    check_mat_law48(m_bad_nu2, log)
    assert log.has_errors
    assert any("poisson" in str(e).lower() or "nu" in str(e).lower() for e in log.errors)

    # 6. Yield stress sigy <= 0
    m_bad_sigy = DummyMat(a=-10.0)
    log = MessageLog()
    check_mat_law48(m_bad_sigy, log)
    assert log.has_errors
    assert any("yield" in str(e).lower() or "sigy" in str(e).lower() for e in log.errors)

    # 7. Hardening exponent n < 0 or n > 1
    m_bad_n1 = DummyMat(n=-0.1)
    log = MessageLog()
    check_mat_law48(m_bad_n1, log)
    assert log.has_errors
    assert any("hardening exponent" in str(e).lower() or "n" in str(e).lower() for e in log.errors)

    m_bad_n2 = DummyMat(n=1.2)
    log = MessageLog()
    check_mat_law48(m_bad_n2, log)
    assert log.has_errors
    assert any("hardening exponent" in str(e).lower() or "n" in str(e).lower() for e in log.errors)

    # 8. Hardening parameter b < 0
    m_bad_b = DummyMat(b=-50.0)
    log = MessageLog()
    check_mat_law48(m_bad_b, log)
    assert log.has_errors
    assert any("hardening parameter" in str(e).lower() or "b" in str(e).lower() for e in log.errors)

    # 9. Maximum stress sig_max < 0
    m_bad_sigm = DummyMat(sig_max=-100.0)
    log = MessageLog()
    check_mat_law48(m_bad_sigm, log)
    assert log.has_errors
    assert any("maximum stress" in str(e).lower() or "sig_max" in str(e).lower() for e in log.errors)

    # 10. Failure strains: eps_t1 >= eps_t2 when eps_t1 < 1e20
    m_bad_fail = DummyMat(eps_t1=0.6, eps_t2=0.4)
    log = MessageLog()
    check_mat_law48(m_bad_fail, log)
    assert log.has_errors
    assert any("eps_t2" in str(e).lower() or "eps_t1" in str(e).lower() for e in log.errors)

    # 11. Warning for strain rate filtering cutoff frequency
    m_warn_fcut = DummyMat(c=0.05, eps_rate_0=1.0, fcut=0.0)
    log = MessageLog()
    check_mat_law48(m_warn_fcut, log)
    assert not log.has_errors
    assert log.has_warnings
    assert any("cutoff frequency" in str(w).lower() or "1220" in str(w) for w in log.warnings)

    # 12. Allowed element families check
    for part in ("bricks", "tetras", "penta6", "pyra5", "shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
        assert 48 in _ALLOWED_LAWS[part]
        assert "48" in _ALLOWED_LAWS[part]
        assert "LAW48" in _ALLOWED_LAWS[part]
        assert "ZHAO" in _ALLOWED_LAWS[part]
        assert "PLAS_ZHAO" in _ALLOWED_LAWS[part]
        assert "MAT_ZHAO" in _ALLOWED_LAWS[part]
        assert "MAT_PLAS_ZHAO" in _ALLOWED_LAWS[part]

    # Trusses, beams, springs must reject LAW48
    for rejected in ("trusses", "beams"):
        assert 48 not in _ALLOWED_LAWS[rejected]
        assert "LAW48" not in _ALLOWED_LAWS[rejected]


def test_m552_check_materials_model():
    model = Model()
    m_valid = MatLaw48(id=1, rho=7.85e-6, e=210000.0, nu=0.3, a=250.0, b=500.0, n=0.45)
    m_valid.law = 48
    model.materials[1] = m_valid
    log = MessageLog()
    check_materials(model, log)
    assert not log.has_errors

    # Invalid Young's modulus
    m_bad = MatLaw48(id=2, rho=7.85e-6, e=-100.0, nu=0.3, a=250.0, b=500.0, n=0.45)
    m_bad.law = 48
    model.materials[2] = m_bad
    log = MessageLog()
    check_materials(model, log)
    assert log.has_errors
    assert any("young" in str(e).lower() or "e" in str(e).lower() for e in log.errors)
