"""
Tests for M553 (/MAT/LAW58 /MAT/FABR_A /MAT/FABRIC_A) input, starter, model, and checks integration.
"""

from io import StringIO
import math
import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.input.card_layouts import (
    MAT_LAW58_1,
    MAT_LAW58_2,
    MAT_LAW58_3,
    MAT_LAW58_4,
    MAT_LAW58_5,
    MAT_LAW58_6,
    MAT_LAW58_7,
    MAT_FABR_A_1,
    MAT_FABR_A_2,
    MAT_FABR_A_3,
    MAT_FABR_A_4,
    MAT_FABR_A_5,
    MAT_FABR_A_6,
    MAT_FABR_A_7,
    MAT_FABRIC_A_1,
    MAT_FABRIC_A_2,
    MAT_FABRIC_A_3,
    MAT_FABRIC_A_4,
    MAT_FABRIC_A_5,
    MAT_FABRIC_A_6,
    MAT_FABRIC_A_7,
    LAYOUTS,
)
from pyradioss.input.cfg_catalogue import LAW_MAP, LAW_SYNONYMS, law_number, canonical_law_name
from pyradioss.model.entities import MatLaw58, MatFabrA, MatFabricA, Material
from pyradioss.model.model import Model
from pyradioss.input.starter_keywords import parse_starter_deck, read_mat_law58, KEYWORD_PARSERS
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.deck_writer import StarterDeck, _conv_mat
from pyradioss.starter.checks import check_mat_law58, _check_mat_law58, check_materials, check_model, _ALLOWED_LAWS


def test_m553_card_layouts():
    assert tuple(MAT_LAW58_1) == (20, 20)
    assert tuple(MAT_LAW58_2) == (20, 20, 20, 20, 20)
    assert tuple(MAT_LAW58_3) == (20, 20, 20, 20, 10, 10)
    assert tuple(MAT_LAW58_4) == (20, 20, 20, 20, 20)
    assert tuple(MAT_LAW58_5) == (10, 10, 20, 20, 20, 20)
    assert tuple(MAT_LAW58_6) == (10, 10, 20)
    assert tuple(MAT_LAW58_7) == (10, 10, 20, 20, 10, 20)

    for i in range(1, 8):
        l_law = globals()[f"MAT_LAW58_{i}"]
        l_fabr = globals()[f"MAT_FABR_A_{i}"]
        l_fabric = globals()[f"MAT_FABRIC_A_{i}"]
        assert l_fabr == l_law
        assert l_fabric == l_law
        assert f"MAT_LAW58_{i}" in LAYOUTS
        assert f"MAT_FABR_A_{i}" in LAYOUTS
        assert f"MAT_FABRIC_A_{i}" in LAYOUTS
        assert list(LAYOUTS[f"MAT_LAW58_{i}"]) == list(l_law)
        assert list(LAYOUTS[f"MAT_FABR_A_{i}"]) == list(l_law)
        assert list(LAYOUTS[f"MAT_FABRIC_A_{i}"]) == list(l_law)


def test_m553_cfg_catalogue():
    for name in ("LAW58", "FABR_A", "MAT_FABR_A", "FABRIC_A", "MAT_FABRIC_A", "MAT_LAW58", "LAW58_FABR_A"):
        assert LAW_MAP[name] == 58
        assert law_number(name) == 58
        assert canonical_law_name(name) == "LAW58"

    assert LAW_SYNONYMS["LAW58"] == "LAW58"
    assert LAW_SYNONYMS["FABR_A"] == "LAW58"
    assert LAW_SYNONYMS["FABRIC_A"] == "LAW58"


def test_m553_keyword_parsers_dispatch():
    for k in ("LAW58", "MAT_LAW58", "FABR_A", "MAT_FABR_A", "FABRIC_A", "MAT_FABRIC_A"):
        assert KEYWORD_PARSERS[k] is read_mat_law58


def test_m553_model_entities_matlaw58():
    m = MatLaw58(
        id=58,
        rho=1.2e-6,
        refer_rho=1.2e-6,
        e1=1500.0,
        b1=10.0,
        e2=800.0,
        b2=8.0,
        f=0.02,
        g0=50.0,
        gi=400.0,
        alpha=0.05,
        g5=20.0,
        isensor=12,
        df=0.03,
        ds=0.04,
        friction_phi=15.0,
        m58_zerostress=1,
        n1_warp=2,
        n2_weft=2,
        s1=0.15,
        s2=0.12,
        c4=1.2,
        c5=1.1,
        fun_a1=101,
        c1=1.0,
        fun_a2=102,
        c2=1.0,
        fun_a3=103,
        c3=1.0,
        fun_a4=104,
        scale4=1.05,
        fun_a5=105,
        scale5=1.02,
        fun_a6=106,
        scale6=0.98,
        title="Test Fabric A",
    )
    assert m.id == 58
    assert m.rho == pytest.approx(1.2e-6)
    assert m.rho0 == pytest.approx(1.2e-6)
    assert m.ref_rho == pytest.approx(1.2e-6)
    assert m.refer_rho == pytest.approx(1.2e-6)
    assert m.e1 == pytest.approx(1500.0)
    assert m.b1 == pytest.approx(10.0)
    assert m.e2 == pytest.approx(800.0)
    assert m.b2 == pytest.approx(8.0)
    assert m.f == pytest.approx(0.02)
    assert m.g0 == pytest.approx(50.0)
    assert m.gi == pytest.approx(400.0)
    assert m.alpha == pytest.approx(0.05)
    assert m.g5 == pytest.approx(20.0)
    assert m.isensor == 12
    assert m.df == pytest.approx(0.03)
    assert m.ds == pytest.approx(0.04)
    assert m.friction_phi == pytest.approx(15.0)
    assert m.m58_zerostress == 1
    assert m.n1_warp == 2
    assert m.n2_weft == 2
    assert m.s1 == pytest.approx(0.15)
    assert m.s2 == pytest.approx(0.12)
    assert m.c4 == pytest.approx(1.2)
    assert m.c5 == pytest.approx(1.1)
    assert m.fun_a1 == 101
    assert m.c1 == pytest.approx(1.0)
    assert m.fun_a2 == 102
    assert m.c2 == pytest.approx(1.0)
    assert m.fun_a3 == 103
    assert m.c3 == pytest.approx(1.0)
    assert m.fun_a4 == 104
    assert m.scale4 == pytest.approx(1.05)
    assert m.fun_a5 == 105
    assert m.scale5 == pytest.approx(1.02)
    assert m.fun_a6 == 106
    assert m.scale6 == pytest.approx(0.98)
    assert m.title == "Test Fabric A"

    # Backward compatibility properties
    assert m.flex == pytest.approx(0.02)
    assert m.gt == pytest.approx(400.0)
    assert m.alphat == pytest.approx(0.05)
    assert m.sensor_id == 12
    assert m.gfrot == pytest.approx(15.0)
    assert m.zero_stress == 1
    assert m.n1 == 2
    assert m.n2 == 2
    assert m.fun_id1 == 101
    assert m.fscale1 == pytest.approx(1.0)
    assert m.fun_id4 == 104
    assert m.fscale4 == pytest.approx(1.05)

    # Sound speed shell check: sqrt(max(E1/N1, E2/N2) / rho)
    # E1/N1 = 1500 / 2 = 750; E2/N2 = 800 / 2 = 400. max = 750.
    # c_shell = sqrt(750 / 1.2e-6) = 25000.0
    expected_c = math.sqrt(750.0 / 1.2e-6)
    # Test dual access (property and callable)
    assert float(m.sound_speed_shell) == pytest.approx(expected_c)
    assert m.sound_speed_shell() == pytest.approx(expected_c)

    # Property setters
    m.ref_rho = 1.3e-6
    assert m.refer_rho == pytest.approx(1.3e-6)
    m.rho0 = 1.25e-6
    assert m.rho == pytest.approx(1.25e-6)
    m.flex = 0.03
    assert m.f == pytest.approx(0.03)
    m.gt = 450.0
    assert m.gi == pytest.approx(450.0)
    m.alphat = 0.08
    assert m.alpha == pytest.approx(0.08)
    m.sensor_id = 99
    assert m.isensor == 99
    m.gfrot = 20.0
    assert m.friction_phi == pytest.approx(20.0)
    m.zero_stress = 0
    assert m.m58_zerostress == 0
    m.n1 = 3
    assert m.n1_warp == 3
    m.n2 = 4
    assert m.n2_weft == 4

    # Class aliases
    assert MatFabrA is MatLaw58
    assert MatFabricA is MatLaw58

    # Generic Material sound speed shell test
    gen_mat = Material(id=1, law=58, rho0=1.0e-6, params={"E1": 1000.0, "E2": 2000.0, "N1": 1, "N2": 1})
    assert gen_mat.sound_speed_shell() == pytest.approx(math.sqrt(2000.0 / 1.0e-6))


def test_m553_parse_fixed_format(tmp_path):
    rad = """/BEGIN
TEST_FIXED_58
/MAT/LAW58/1
Fabric Fixed Deck
         1.20000E-06         1.20000E-06
         1.50000E+03         1.00000E+01         8.00000E+02         8.00000E+00         2.00000E-02
         5.00000E+01         4.00000E+02         5.00000E-02         2.00000E+01                    12
         3.00000E-02         4.00000E-02         1.50000E+01                                       1
         2         2         1.50000E-01         1.20000E-01         1.20000E+00         1.10000E+00
       101                   1.00000E+00
       102                   1.00000E+00
       103                   1.00000E+00
       104       105         1.05000E+00         1.02000E+00       106         9.80000E-01
"""
    f = tmp_path / "deck_fixed_0000.rad"
    f.write_text(rad, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(f))
    parse_starter_deck(blocks, model, log)

    assert 1 in model.mat_law58s
    m = model.mat_law58s[1]
    assert m.id == 1
    assert m.rho == pytest.approx(1.2e-6)
    assert m.refer_rho == pytest.approx(1.2e-6)
    assert m.e1 == pytest.approx(1500.0)
    assert m.b1 == pytest.approx(10.0)
    assert m.e2 == pytest.approx(800.0)
    assert m.b2 == pytest.approx(8.0)
    assert m.f == pytest.approx(0.02)
    assert m.g0 == pytest.approx(50.0)
    assert m.gi == pytest.approx(400.0)
    assert m.alpha == pytest.approx(0.05)
    assert m.g5 == pytest.approx(20.0)
    assert m.isensor == 12
    assert m.df == pytest.approx(0.03)
    assert m.ds == pytest.approx(0.04)
    assert m.friction_phi == pytest.approx(15.0)
    assert m.m58_zerostress == 1
    assert m.n1_warp == 2
    assert m.n2_weft == 2
    assert m.s1 == pytest.approx(0.15)
    assert m.s2 == pytest.approx(0.12)
    assert m.c4 == pytest.approx(1.2)
    assert m.c5 == pytest.approx(1.1)
    assert m.fun_a1 == 101
    assert m.c1 == pytest.approx(1.0)
    assert m.fun_a2 == 102
    assert m.c2 == pytest.approx(1.0)
    assert m.fun_a3 == 103
    assert m.c3 == pytest.approx(1.0)
    assert m.fun_a4 == 104
    assert m.scale4 == pytest.approx(1.05)
    assert m.fun_a5 == 105
    assert m.scale5 == pytest.approx(1.02)
    assert m.fun_a6 == 106
    assert m.scale6 == pytest.approx(0.98)

    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 58
    assert mat.rho0 == pytest.approx(1.2e-6)
    assert mat.params["E1"] == pytest.approx(1500.0)
    assert mat.params["E2"] == pytest.approx(800.0)


def test_m553_parse_free_format_and_aliases(tmp_path):
    rad = """# RADIOSS STARTER
/BEGIN
TEST_FREE_58
/MAT/FABR_A/2
Fabric A Free Comma Format
1.1e-6, 1.1e-6
1200.0, 5.0, 700.0, 4.0, 0.015
40.0, 300.0, 0.04, 15.0, 5
0.02, 0.03, 10.0, 0
1, 1, 0.1, 0.1, 1.0, 1.0
201, 1.0
202, 1.0
203, 1.0
204, 205, 1.0, 1.0, 206, 1.0
/MAT/FABRIC_A/3
Fabric A Free Space Format
1.3e-6
1400.0 6.0 750.0 5.0 0.02
45.0 350.0 0.05 18.0 0
0.03 0.04 12.0 0
1 1 0.1 0.1
301 1.0
302 1.0
303 1.0
"""
    f = tmp_path / "deck_free_0000.rad"
    f.write_text(rad, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(f))
    parse_starter_deck(blocks, model, log)

    assert 2 in model.mat_law58s
    m2 = model.mat_law58s[2]
    assert m2.id == 2
    assert m2.rho == pytest.approx(1.1e-6)
    assert m2.e1 == pytest.approx(1200.0)
    assert m2.fun_a1 == 201
    assert m2.fun_a4 == 204
    assert m2.scale4 == pytest.approx(1.0)
    assert model.materials[2].law == 58

    assert 3 in model.mat_law58s
    m3 = model.mat_law58s[3]
    assert m3.id == 3
    assert m3.rho == pytest.approx(1.3e-6)
    assert m3.e1 == pytest.approx(1400.0)
    assert m3.fun_a1 == 301
    # Mat 3 has no unloading card, so unloading curves default to 0
    assert m3.fun_a4 == 0
    assert model.materials[3].law == 58


def test_m553_fortran_defaults(tmp_path):
    # Test card 5 c4 and c5 default logic:
    # c4 = c1 / (1 - s1) = 1.0 / (1.0 - 0.2) = 1.25
    # c5 = c2 / (1 - s2) = 1.0 / (1.0 - 0.2) = 1.25
    # gi = 0.25 * (e1 + e2) = 0.25 * (1000 + 1000) = 500
    rad = """/BEGIN
TEST_DEFAULTS_58
/MAT/LAW58/4
Defaults Check
         1.00000E-06
         1.00000E+03         0.00000E+00         1.00000E+03
         0.00000E+00         0.00000E+00
         0.00000E+00
                   0                   0         2.00000E-01         2.00000E-01
       401
       402
       403
"""
    f = tmp_path / "deck_defaults_0000.rad"
    f.write_text(rad, encoding="utf-8")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(f))
    parse_starter_deck(blocks, model, log)

    m = model.mat_law58s[4]
    assert m.n1_warp == 1
    assert m.n2_weft == 1
    assert m.s1 == pytest.approx(0.2)
    assert m.s2 == pytest.approx(0.2)
    assert m.c4 == pytest.approx(0.01)
    assert m.c5 == pytest.approx(0.01)
    assert m.gi == pytest.approx(500.0)
    assert m.f == pytest.approx(0.01)
    assert m.df == pytest.approx(0.05)


def test_m553_deck_writer_roundtrip(tmp_path):
    deck = StarterDeck("LAW58_ROUNDTRIP")
    # 1. Fixed format
    deck.mat_law58(
        mid=1,
        title="Fabric Mat 1 Fixed",
        rho=1.2e-6,
        ref_rho=1.2e-6,
        e1=1500.0,
        b1=10.0,
        e2=800.0,
        b2=8.0,
        f=0.02,
        g0=50.0,
        gi=400.0,
        alpha=0.05,
        g5=20.0,
        isensor=12,
        df=0.03,
        ds=0.04,
        friction_phi=15.0,
        m58_zerostress=1,
        n1_warp=2,
        n2_weft=2,
        s1=0.15,
        s2=0.12,
        c4=1.2,
        c5=1.1,
        fun_a1=101,
        c1=1.0,
        fun_a2=102,
        c2=1.0,
        fun_a3=103,
        c3=1.0,
        fun_a4=104,
        scale4=1.05,
        fun_a5=105,
        scale5=1.02,
        fun_a6=106,
        scale6=0.98,
        fixed_format=True,
    )
    # 2. Free format with mat_fabr_a
    deck.mat_fabr_a(
        mid=2,
        title="Fabric Mat 2 Free",
        rho=1.1e-6,
        e1=1200.0,
        b1=5.0,
        e2=700.0,
        b2=4.0,
        f=0.015,
        g0=40.0,
        gi=300.0,
        alpha=0.04,
        g5=15.0,
        isensor=5,
        df=0.02,
        ds=0.03,
        friction_phi=10.0,
        m58_zerostress=0,
        n1_warp=1,
        n2_weft=1,
        s1=0.1,
        s2=0.1,
        c4=1.0,
        c5=1.0,
        fun_a1=201,
        c1=1.0,
        fun_a2=202,
        c2=1.0,
        fun_a3=203,
        c3=1.0,
        fun_a4=204,
        scale4=1.0,
        fun_a5=205,
        scale5=1.0,
        fun_a6=206,
        scale6=1.0,
        fixed_format=False,
    )
    # 3. mat_fabric_a without unloading
    deck.mat_fabric_a(
        mid=3,
        title="Fabric Mat 3 Free No Unload",
        rho=1.3e-6,
        e1=1400.0,
        b1=6.0,
        e2=750.0,
        b2=5.0,
        f=0.02,
        g0=45.0,
        gi=350.0,
        alpha=0.05,
        g5=18.0,
        isensor=0,
        df=0.03,
        ds=0.04,
        friction_phi=12.0,
        m58_zerostress=0,
        n1_warp=1,
        n2_weft=1,
        s1=0.1,
        s2=0.1,
        c4=1.11,
        c5=1.11,
        fun_a1=301,
        c1=1.0,
        fun_a2=302,
        c2=1.0,
        fun_a3=303,
        c3=1.0,
        fixed_format=True,
    )

    deck_path = tmp_path / "test_law58_0000.rad"
    deck.write(str(deck_path))

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert 1 in model.mat_law58s
    m1 = model.mat_law58s[1]
    assert m1.rho == pytest.approx(1.2e-6)
    assert m1.refer_rho == pytest.approx(1.2e-6)
    assert m1.e1 == pytest.approx(1500.0)
    assert m1.b1 == pytest.approx(10.0)
    assert m1.e2 == pytest.approx(800.0)
    assert m1.b2 == pytest.approx(8.0)
    assert m1.f == pytest.approx(0.02)
    assert m1.g0 == pytest.approx(50.0)
    assert m1.gi == pytest.approx(400.0)
    assert m1.alpha == pytest.approx(0.05)
    assert m1.g5 == pytest.approx(20.0)
    assert m1.isensor == 12
    assert m1.df == pytest.approx(0.03)
    assert m1.ds == pytest.approx(0.04)
    assert m1.friction_phi == pytest.approx(15.0)
    assert m1.m58_zerostress == 1
    assert m1.n1_warp == 2
    assert m1.n2_weft == 2
    assert m1.s1 == pytest.approx(0.15)
    assert m1.s2 == pytest.approx(0.12)
    assert m1.c4 == pytest.approx(1.2)
    assert m1.c5 == pytest.approx(1.1)
    assert m1.fun_a1 == 101
    assert m1.c1 == pytest.approx(1.0)
    assert m1.fun_a2 == 102
    assert m1.c2 == pytest.approx(1.0)
    assert m1.fun_a3 == 103
    assert m1.c3 == pytest.approx(1.0)
    assert m1.fun_a4 == 104
    assert m1.scale4 == pytest.approx(1.05)
    assert m1.fun_a5 == 105
    assert m1.scale5 == pytest.approx(1.02)
    assert m1.fun_a6 == 106
    assert m1.scale6 == pytest.approx(0.98)

    assert 2 in model.mat_law58s
    m2 = model.mat_law58s[2]
    assert m2.rho == pytest.approx(1.1e-6)
    assert m2.e1 == pytest.approx(1200.0)
    assert m2.fun_a1 == 201
    assert m2.fun_a4 == 204

    assert 3 in model.mat_law58s
    m3 = model.mat_law58s[3]
    assert m3.rho == pytest.approx(1.3e-6)
    assert m3.e1 == pytest.approx(1400.0)
    assert m3.fun_a1 == 301
    assert m3.fun_a4 == 0


def test_m553_deck_writer_object_invocation(tmp_path):
    m = MatLaw58(
        id=5,
        rho=1.25e-6,
        refer_rho=1.25e-6,
        e1=1600.0,
        b1=12.0,
        e2=850.0,
        b2=9.0,
        f=0.025,
        g0=55.0,
        gi=420.0,
        alpha=0.06,
        g5=22.0,
        isensor=15,
        df=0.035,
        ds=0.045,
        friction_phi=16.0,
        m58_zerostress=1,
        n1_warp=2,
        n2_weft=2,
        s1=0.14,
        s2=0.11,
        c4=1.22,
        c5=1.12,
        fun_a1=501,
        c1=1.0,
        fun_a2=502,
        c2=1.0,
        fun_a3=503,
        c3=1.0,
        title="Direct Object Fabric",
    )
    deck = StarterDeck("OBJ_TEST")
    deck.mat_law58(m)
    deck_path = tmp_path / "obj_test_0000.rad"
    deck.write(str(deck_path))

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert 5 in model.mat_law58s
    assert model.mat_law58s[5].e1 == pytest.approx(1600.0)
    assert model.mat_law58s[5].fun_a1 == 501


def test_m553_deck_writer_conv_mat(tmp_path):
    deck = StarterDeck("CONV_MAT_TEST")
    rad_lines = [
        "/BEGIN",
        "CONV_MAT_TEST",
        "/MAT/LAW58/10",
        "Test Conv Mat",
        "1.2e-6 1.2e-6",
        "1500.0 10.0 800.0 8.0 0.02",
        "50.0 400.0 0.05 20.0 12",
        "0.03 0.04 15.0 0 1",
        "2 2 0.15 0.12 1.2 1.1",
        "101 1.0",
        "102 1.0",
        "103 1.0",
        "104 105 1.05 1.02 106 0.98",
    ]
    deck_path = tmp_path / "conv_mat_0000.rad"
    deck_path.write_text("\n".join(rad_lines), encoding="utf-8")
    blocks = read_deck(str(deck_path))
    for b in blocks:
        if b.key0 == "MAT":
            _conv_mat(deck, b)

    rendered = deck.render()
    assert "/MAT/LAW58/10" in rendered
    assert "Test Conv Mat" in rendered


def test_m553_starter_checks():
    class DummyFabric:
        def __init__(self, **kwargs):
            self.id = kwargs.get("id", 1)
            self.rho0 = kwargs.get("rho0", 1.2e-6)
            self.rho = self.rho0
            self.refer_rho = kwargs.get("refer_rho", self.rho0)
            self.e1 = kwargs.get("e1", 1500.0)
            self.b1 = kwargs.get("b1", 10.0)
            self.e2 = kwargs.get("e2", 800.0)
            self.b2 = kwargs.get("b2", 8.0)
            self.f = kwargs.get("f", 0.02)
            self.g0 = kwargs.get("g0", 50.0)
            self.gi = kwargs.get("gi", 400.0)
            self.alpha = kwargs.get("alpha", 0.05)
            self.g5 = kwargs.get("g5", 20.0)
            self.isensor = kwargs.get("isensor", 0)
            self.df = kwargs.get("df", 0.03)
            self.ds = kwargs.get("ds", 0.04)
            self.friction_phi = kwargs.get("friction_phi", 15.0)
            self.m58_zerostress = kwargs.get("m58_zerostress", 0)
            self.n1_warp = kwargs.get("n1_warp", 1)
            self.n2_weft = kwargs.get("n2_weft", 1)
            self.s1 = kwargs.get("s1", 0.1)
            self.s2 = kwargs.get("s2", 0.1)
            self.c4 = kwargs.get("c4", 1.25)
            self.c5 = kwargs.get("c5", 1.25)
            self.fun_a1 = kwargs.get("fun_a1", 101)
            self.c1 = kwargs.get("c1", 1.0)
            self.fun_a2 = kwargs.get("fun_a2", 102)
            self.c2 = kwargs.get("c2", 1.0)
            self.fun_a3 = kwargs.get("fun_a3", 103)
            self.c3 = kwargs.get("c3", 1.0)
            self.fun_a4 = kwargs.get("fun_a4", 104)
            self.scale4 = kwargs.get("scale4", 1.0)
            self.fun_a5 = kwargs.get("fun_a5", 105)
            self.scale5 = kwargs.get("scale5", 1.0)
            self.fun_a6 = kwargs.get("fun_a6", 106)
            self.scale6 = kwargs.get("scale6", 1.0)

    # 1. Valid material
    m_valid = DummyFabric()
    model = Model()
    log = MessageLog()
    check_mat_law58(m_valid, log=log, model=model)
    assert not log.has_errors

    # 2. Function alias _check_mat_law58
    log = MessageLog()
    _check_mat_law58(m_valid, model, log)
    assert not log.has_errors

    # 3. Density rho0 <= 0
    m_bad_rho = DummyFabric(rho0=0.0)
    log = MessageLog()
    check_mat_law58(m_bad_rho, log)
    assert log.has_errors
    assert any("density" in str(e).lower() or "rho" in str(e).lower() for e in log.errors)

    # 4. Moduli e1 <= 0, e2 <= 0
    m_bad_e1 = DummyFabric(e1=0.0)
    log = MessageLog()
    check_mat_law58(m_bad_e1, log)
    assert log.has_errors
    assert any("e1" in str(e).lower() or "young" in str(e).lower() for e in log.errors)

    m_bad_e2 = DummyFabric(e2=-5.0)
    log = MessageLog()
    check_mat_law58(m_bad_e2, log)
    assert log.has_errors
    assert any("e2" in str(e).lower() or "young" in str(e).lower() for e in log.errors)

    # 5. Curve consistency (ANCMSG 1578, 1579, 1580)
    # If unloading active (fun_a4/5/6 != 0) but fun_a1 == 0:
    m_no_a1 = DummyFabric(fun_a1=0, fun_a4=104)
    log = MessageLog()
    check_mat_law58(m_no_a1, log)
    assert log.has_errors
    assert any("1578" in str(e) or "fun_a1" in str(e).lower() for e in log.errors)

    # If fun_a2 == 0:
    m_no_a2 = DummyFabric(fun_a2=0, fun_a5=105)
    log = MessageLog()
    check_mat_law58(m_no_a2, log)
    assert log.has_errors
    assert any("1579" in str(e) or "fun_a2" in str(e).lower() for e in log.errors)

    # If fun_a3 == 0:
    m_no_a3 = DummyFabric(fun_a3=0, fun_a6=106)
    log = MessageLog()
    check_mat_law58(m_no_a3, log)
    assert log.has_errors
    assert any("1580" in str(e) or "fun_a3" in str(e).lower() for e in log.errors)

    # 6. Element family check in _ALLOWED_LAWS
    for shell_part in ("shells", "shells_qbat", "shells_qeph", "sh3n", "quads"):
        assert 58 in _ALLOWED_LAWS[shell_part]
        assert "58" in _ALLOWED_LAWS[shell_part]
        assert "LAW58" in _ALLOWED_LAWS[shell_part]
        assert "FABR_A" in _ALLOWED_LAWS[shell_part]
        assert "FABRIC_A" in _ALLOWED_LAWS[shell_part]
        assert "MAT_LAW58" in _ALLOWED_LAWS[shell_part]
        assert "MAT_FABR_A" in _ALLOWED_LAWS[shell_part]
        assert "MAT_FABRIC_A" in _ALLOWED_LAWS[shell_part]

    # Solids and 1D elements must NOT allow LAW58
    for rejected in ("bricks", "tetras", "penta6", "pyra5", "trusses", "beams"):
        assert 58 not in _ALLOWED_LAWS[rejected]
        assert "LAW58" not in _ALLOWED_LAWS[rejected]


def test_m553_check_materials_and_model_elements():
    model = Model()
    m_valid = MatLaw58(id=1, rho=1.2e-6, e1=1500.0, e2=800.0)
    m_valid.law = 58
    model.materials[1] = m_valid
    log = MessageLog()
    check_materials(model, log)
    assert not log.has_errors

    # Invalid Young's modulus E1
    m_bad = MatLaw58(id=2, rho=1.2e-6, e1=-10.0, e2=800.0)
    m_bad.law = 58
    model.materials[2] = m_bad
    log = MessageLog()
    check_materials(model, log)
    assert log.has_errors
    assert any("e1" in str(e).lower() for e in log.errors)

    # Element compatibility error in check_mat_law58
    class DummyElement:
        def __init__(self, mat_id):
            self.mat_id = mat_id

    class DummyGroup:
        def __init__(self, elements):
            self._elements = elements
        def values(self):
            return self._elements

    # Solid element group attaching LAW58 (ANCMSG 305)
    model2 = Model()
    m58 = MatLaw58(id=1, rho=1.2e-6, e1=1500.0, e2=800.0)
    m58.law = 58
    model2.materials[1] = m58
    model2._element_groups = [("bricks", DummyGroup([DummyElement(1)]))]
    model2.element_groups = lambda: model2._element_groups
    log = MessageLog()
    check_mat_law58(model=model2, mat_id=1, mat=m58, log=log)
    assert log.has_errors
    assert any("305" in str(e) or "solid" in str(e).lower() for e in log.errors)

    # 1D element group attaching LAW58 (ANCMSG 306)
    model3 = Model()
    model3.materials[1] = m58
    model3._element_groups = [("trusses", DummyGroup([DummyElement(1)]))]
    model3.element_groups = lambda: model3._element_groups
    log = MessageLog()
    check_mat_law58(model=model3, mat_id=1, mat=m58, log=log)
    assert log.has_errors
    assert any("306" in str(e) or "1d" in str(e).lower() for e in log.errors)
