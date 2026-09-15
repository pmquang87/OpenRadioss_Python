"""Tests for Milestone M189:
- /MAT/LAW52 (/MAT/GURSON, /MAT/PLAS_GURS)
- /MAT/LAW16 (/MAT/GRAY, /MAT/CAST_IRON)
- /MAT/LAW14 (/MAT/COMPSO, /MAT/COMP_SOL)
- /MAT/LAW59 (/MAT/CONNECT, /MAT/CONNECTOR)
- /MAT/LAW64 (/MAT/TRANSFO_MART, /MAT/MARTENSITE)
- /FAIL/LAD_DAMA (/FAIL/LADEVEZE)
- /FAIL/PUCK
- /FAIL/WIERZBICKI (/FAIL/MMC)
- /FAIL/WILKINS
- /FAIL/SPALLING (/FAIL/SPALL)
- /PROP/TYPE14 (/PROP/SOLID, /PROP/SOL_GENE)
- /PROP/TYPE8 (/PROP/SPR_GENE, /PROP/SPRING_GENE)
- /PROP/TYPE25 (/PROP/SPR_AXI, /PROP/SPRING_AXI)
- /PROP/TYPE32 (/PROP/SPR_PRE, /PROP/SPRING_PRE)
- /PROP/TYPE43 (/PROP/CONNECT, /PROP/PROP_CONNECT)
- Unit conversions, model containers, and alias access.
"""

from pathlib import Path
import pytest
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import (
    MatLaw52, MatLaw16, MatLaw14, MatLaw59, MatLaw64,
    FailLadDama, FailPuck, FailWierzbicki, FailWilkins, FailSpalling,
    PropType14, PropType8, PropType25, PropType32, PropType43,
)


def _parse(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_mat_law52_gurson_free_and_fixed(tmp_path: Path):
    # Free format
    deck_free = """/BEGIN
TEST_GURSON
/MAT/LAW52/10
Gurson Steel
7.85e-6 7.85e-6
210000.0 0.3 1 1 0.25
500.0 300.0 0.2 0.015 1.0e-3
1.5 1.0 2.25 0.1 0.3
0.001 0.04 0.15 0.25
/END
"""
    model_free, log = _parse(tmp_path, deck_free)
    assert len(log.errors) == 0
    assert 10 in model_free.mat_law52s
    assert 10 in model_free.mat_gursons
    m = model_free.mat_law52s[10]
    assert m.rho0 == pytest.approx(7.85e-6)
    assert m.e == pytest.approx(210000.0)
    assert m.nu == pytest.approx(0.3)
    assert m.iflag == 1
    assert m.fsmooth == 1
    assert m.fcut == pytest.approx(0.25)
    assert m.a == pytest.approx(500.0)
    assert m.b == pytest.approx(300.0)
    assert m.n == pytest.approx(0.2)
    assert m.c == pytest.approx(0.015)
    assert m.pc == pytest.approx(1.0e-3)
    assert m.q1 == pytest.approx(1.5)
    assert m.q2 == pytest.approx(1.0)
    assert m.q3 == pytest.approx(2.25)
    assert m.s_n == pytest.approx(0.1)
    assert m.eps_n == pytest.approx(0.3)
    assert m.f_i == pytest.approx(0.001)
    assert m.f_n == pytest.approx(0.04)
    assert m.f_c == pytest.approx(0.15)
    assert m.f_f == pytest.approx(0.25)
    assert m.title == "Gurson Steel"

    # Fixed format via /MAT/GURSON alias
    deck_fixed = """#---1---|----2---|----3---|----4---|----5---|----6---|----7---|----8---|----9---|---10---|
/MAT/GURSON/11
Gurson Steel Fixed
             7.85e-6             7.85e-6
            210000.0                 0.3         1         1                0.25
               500.0               300.0                 0.2               0.015              1.0e-3
                 1.5                 1.0                2.25                 0.1                 0.3
               0.001                0.04                0.15                0.25
/END
"""
    model_fixed, log_fix = _parse(tmp_path, deck_fixed)
    assert len(log_fix.errors) == 0
    assert 11 in model_fixed.mat_law52s
    m2 = model_fixed.mat_law52s[11]
    assert m2.rho0 == pytest.approx(7.85e-6)
    assert m2.e == pytest.approx(210000.0)
    assert m2.q1 == pytest.approx(1.5)
    assert m2.f_f == pytest.approx(0.25)


def test_mat_law16_gray_cast_iron(tmp_path: Path):
    deck = """/BEGIN
TEST_GRAY_IRON
/MAT/GRAY/20
Gray Cast Iron
7.2e-6 7.2e-6
1.0e5 4500.0 1.5 1.6 0.2
120000.0 0.25 110000.0 0.26 250.0
0.5 500.0 400.0 0.08
/END
"""
    model, log = _parse(tmp_path, deck)
    assert len(log.errors) == 0
    assert 20 in model.mat_law16s
    assert 20 in model.mat_grays
    m = model.mat_law16s[20]
    assert m.rho0 == pytest.approx(7.2e-6)
    assert m.p0 == pytest.approx(1.0e5)
    assert m.c == pytest.approx(4500.0)
    assert m.s == pytest.approx(1.5)
    assert m.gamma0 == pytest.approx(1.6)
    assert m.a == pytest.approx(0.2)
    assert m.e0 == pytest.approx(120000.0)
    assert m.v0 == pytest.approx(0.25)
    assert m.e == pytest.approx(110000.0)
    assert m.nu == pytest.approx(0.26)
    assert m.sig_y == pytest.approx(250.0)
    assert m.beta == pytest.approx(0.5)
    assert m.hard == pytest.approx(500.0)
    assert m.sig_max == pytest.approx(400.0)
    assert m.eps_max == pytest.approx(0.08)


def test_mat_law14_composite_solid(tmp_path: Path):
    deck = """/BEGIN
TEST_COMPSO
/MAT/COMPSO/30
Composite Solid
1.6e-6 1.6e-6
140000.0 10000.0 10000.0 0.3 0.35
0.02 5000.0 3500.0 5000.0
1500.0 50.0 50.0 0.8 0.1
100.0 2000.0 1200.0 40.0 150.0
150.0 80.0 40.0 90.0 45.0 0.05 1000.0 0.01 1
/END
"""
    model, log = _parse(tmp_path, deck)
    assert len(log.errors) == 0
    assert 30 in model.mat_law14s
    assert 30 in model.mat_compsos
    m = model.mat_law14s[30]
    assert m.rho0 == pytest.approx(1.6e-6)
    assert m.ea == pytest.approx(140000.0)
    assert m.eb == pytest.approx(10000.0)
    assert m.ec == pytest.approx(10000.0)
    assert m.prab == pytest.approx(0.3)
    assert m.prbc == pytest.approx(0.35)
    assert m.prca == pytest.approx(0.02)
    assert m.gab == pytest.approx(5000.0)
    assert m.gbc == pytest.approx(3500.0)
    assert m.gca == pytest.approx(5000.0)
    assert m.sigt1 == pytest.approx(1500.0)
    assert m.damage == pytest.approx(0.8)
    assert m.strflag == 1


def test_mat_law59_connector(tmp_path: Path):
    deck = """/BEGIN
TEST_CONNECTOR
/MAT/CONNECT/40
Spotweld Connector
7.8e-6 7.8e-6
210000.0 80000.0 1 0.1 2
101 1 5000.0 0.5
102 2 8000.0 0.8
/END
"""
    model, log = _parse(tmp_path, deck)
    assert len(log.errors) == 0
    assert 40 in model.mat_law59s
    assert 40 in model.mat_connects
    m = model.mat_law59s[40]
    assert m.rho0 == pytest.approx(7.8e-6)
    assert m.e == pytest.approx(210000.0)
    assert m.g0 == pytest.approx(80000.0)
    assert m.fsmooth == 1
    assert m.fcut == pytest.approx(0.1)
    assert m.iflag == 2
    assert len(m.functions) == 2
    assert m.functions[0]["ipt"] == 101
    assert m.functions[0]["fp1"] == pytest.approx(5000.0)
    assert m.functions[1]["ipt"] == 102
    assert m.functions[1]["fp2"] == pytest.approx(0.8)


def test_mat_law64_martensitic(tmp_path: Path):
    deck = """/BEGIN
TEST_MARTENSITE
/MAT/TRANSFO_MART/50
TRIP Steel
7.9e-6 7.9e-6
205000.0 0.3 450.0 2.5 0.4
80.0 0.1 0.8 293.15
201 202 1.0 1.2
/END
"""
    model, log = _parse(tmp_path, deck)
    assert len(log.errors) == 0
    assert 50 in model.mat_law64s
    assert 50 in model.mat_transfo_marts
    m = model.mat_law64s[50]
    assert m.rho0 == pytest.approx(7.9e-6)
    assert m.e == pytest.approx(205000.0)
    assert m.nu == pytest.approx(0.3)
    assert m.cp == pytest.approx(450.0)
    assert m.d == pytest.approx(2.5)
    assert m.n == pytest.approx(0.4)
    assert m.md == pytest.approx(80.0)
    assert m.v0 == pytest.approx(0.1)
    assert m.vmc == pytest.approx(0.8)
    assert m.t_ini == pytest.approx(293.15)
    assert m.funct_id_0 == 201
    assert m.funct_id_1 == 202
    assert m.scale_0 == pytest.approx(1.0)
    assert m.scale_1 == pytest.approx(1.2)


def test_fail_criteria_suite(tmp_path: Path):
    deck = """/BEGIN
TEST_FAIL_SUITE
/FAIL/LAD_DAMA/100
1000.0 200.0 300.0 0.5 0.25
1.5 25.0 0.8 1.2 80.0
1 2
/FAIL/PUCK/101
1200.0 45.0 70.0 800.0 150.0
0.25 0.20 0.22 90.0
/FAIL/WIERZBICKI/102
500.0 0.1 1.2 0.05 1.5
0.2 1 1 0
/FAIL/WILKINS/103
1.5 2.0 1.0e4 0.6
1 1
/FAIL/SPALLING/104
0.1 0.2 0.3 0.4 0.5
1.0 -500.0 1
/END
"""
    model, log = _parse(tmp_path, deck)
    assert len(log.errors) == 0

    # LAD_DAMA
    assert 100 in model.fail_lad_damas
    assert 100 in model.fail_ladevezes
    flad = model.fail_lad_damas[100]
    assert flad.k1 == pytest.approx(1000.0)
    assert flad.gamma1 == pytest.approx(0.5)
    assert flad.y0 == pytest.approx(1.5)
    assert flad.yc == pytest.approx(25.0)
    assert flad.k_lad == pytest.approx(0.8)
    assert flad.a_dama == pytest.approx(1.2)
    assert flad.tau_max == pytest.approx(80.0)
    assert flad.ifail_sh == 1
    assert flad.ifail_so == 2

    # PUCK
    assert 101 in model.fail_pucks
    fpuck = model.fail_pucks[101]
    assert fpuck.sigma_1t == pytest.approx(1200.0)
    assert fpuck.sigma_2t == pytest.approx(45.0)
    assert fpuck.sigma_12 == pytest.approx(70.0)
    assert fpuck.sigma_1c == pytest.approx(800.0)
    assert fpuck.sigma_2c == pytest.approx(150.0)
    assert fpuck.p12_pos == pytest.approx(0.25)
    assert fpuck.p12_neg == pytest.approx(0.20)
    assert fpuck.p22_neg == pytest.approx(0.22)
    assert fpuck.tau_max == pytest.approx(90.0)

    # WIERZBICKI
    assert 102 in model.fail_wierzbickis
    assert 102 in model.fail_mmcs
    fmmc = model.fail_wierzbickis[102]
    assert fmmc.c1 == pytest.approx(500.0)
    assert fmmc.c2 == pytest.approx(0.1)
    assert fmmc.c3 == pytest.approx(1.2)
    assert fmmc.c4 == pytest.approx(0.05)
    assert fmmc.m == pytest.approx(1.5)
    assert fmmc.n == pytest.approx(0.2)
    assert fmmc.ifail_sh == 1

    # WILKINS
    assert 103 in model.fail_wilkinss
    fwilk = model.fail_wilkinss[103]
    assert fwilk.alpha == pytest.approx(1.5)
    assert fwilk.beta == pytest.approx(2.0)
    assert fwilk.plim == pytest.approx(1.0e4)
    assert fwilk.df == pytest.approx(0.6)

    # SPALLING
    assert 104 in model.fail_spallings
    assert 104 in model.fail_spalls
    fspall = model.fail_spallings[104]
    assert fspall.d1 == pytest.approx(0.1)
    assert fspall.d2 == pytest.approx(0.2)
    assert fspall.d3 == pytest.approx(0.3)
    assert fspall.d4 == pytest.approx(0.4)
    assert fspall.d5 == pytest.approx(0.5)
    assert fspall.eps_dot_0 == pytest.approx(1.0)
    assert fspall.p_min == pytest.approx(-500.0)
    assert fspall.ifail_so == 1


def test_prop_type14_solid(tmp_path: Path):
    deck = """/BEGIN
TEST_PROP_SOLID
/PROP/TYPE14/200
Solid Prop
14 1 0 2 2 2 1 0 0.15
1.1 0.05 0.1
1.0e-6 1 0.0 0.0 0.0 0 1
/END
"""
    model, log = _parse(tmp_path, deck)
    assert len(log.errors) == 0
    assert 200 in model.prop_type14s
    assert 200 in model.prop_solids
    p = model.prop_type14s[200]
    assert p.isolid == 14
    assert p.ismstr == 1
    assert p.inpts_r == 2
    assert p.inpts_s == 2
    assert p.inpts_t == 2
    assert p.i_rot == 1
    assert p.dn == pytest.approx(0.15)
    assert p.qa == pytest.approx(1.1)
    assert p.qb == pytest.approx(0.05)
    assert p.h == pytest.approx(0.1)
    assert p.deltat_min == pytest.approx(1.0e-6)
    assert p.istrain == 1
    assert p.icstr == 1


def test_prop_type8_spr_gene(tmp_path: Path):
    deck = """/BEGIN
TEST_PROP_SPR_GENE
/PROP/SPR_GENE/300
6DOF Spring
1.5 0.05 10 20 1 0 1
1000.0 10.0 0.0 0.0 0.0
101 0 0 0 -10.0 10.0
1000.0 10.0 0.0 0.0 0.0
102 0 0 0 -10.0 10.0
1000.0 10.0 0.0 0.0 0.0
103 0 0 0 -10.0 10.0
500.0 5.0 0.0 0.0 0.0
104 0 0 0 -5.0 5.0
500.0 5.0 0.0 0.0 0.0
105 0 0 0 -5.0 5.0
500.0 5.0 0.0 0.0 0.0
106 0 0 0 -5.0 5.0
/END
"""
    model, log = _parse(tmp_path, deck)
    assert len(log.errors) == 0
    assert 300 in model.prop_type8s
    assert 300 in model.prop_spr_genes
    p = model.prop_type8s[300]
    assert p.mass == pytest.approx(1.5)
    assert p.inertia == pytest.approx(0.05)
    assert p.skew_id == 10
    assert p.sensor_id == 20
    assert p.isflag == 1
    assert p.iequil == 1
    assert len(p.dofs) == 6
    assert p.dofs["tx"]["stiff"] == pytest.approx(1000.0)
    assert p.dofs["tx"]["fun_a"] == 101
    assert p.dofs["rz"]["stiff"] == pytest.approx(500.0)
    assert p.dofs["rz"]["fun_a"] == 106


def test_prop_type25_spr_axi(tmp_path: Path):
    deck = """/BEGIN
TEST_PROP_SPR_AXI
/PROP/SPR_AXI/400
Axisymmetric Spring
2.0 0.1 5 15 1 0 1 1
1500.0 15.0 0.0 0.0 0.0
201 0 0 0 -20.0 20.0
800.0 8.0 0.0 0.0 0.0
202 0 0 0 -10.0 10.0
/END
"""
    model, log = _parse(tmp_path, deck)
    assert len(log.errors) == 0
    assert 400 in model.prop_type25s
    assert 400 in model.prop_spr_axis
    p = model.prop_type25s[400]
    assert p.mass == pytest.approx(2.0)
    assert p.inertia == pytest.approx(0.1)
    assert p.skew_id == 5
    assert p.sensor_id == 15
    assert p.isflag == 1
    assert p.ileng == 1
    assert p.ifail2 == 1
    assert p.tension["stiff"] == pytest.approx(1500.0)
    assert p.tension["fun_a"] == 201
    assert p.shear["stiff"] == pytest.approx(800.0)
    assert p.shear["fun_a"] == 202


def test_prop_type32_spr_pre(tmp_path: Path):
    deck = """/BEGIN
TEST_PROP_SPR_PRE
/PROP/SPR_PRE/500
Preloaded Spring
0.5 12 1
1200.0 500.0 10.0 5.0 1500.0
301 302 1.0 1.0 1.0
/END
"""
    model, log = _parse(tmp_path, deck)
    assert len(log.errors) == 0
    assert 500 in model.prop_type32s
    assert 500 in model.prop_spr_pres
    p = model.prop_type32s[500]
    assert p.mass == pytest.approx(0.5)
    assert p.sensor_id == 12
    assert p.ilock == 1
    assert p.stiff0 == pytest.approx(1200.0)
    assert p.f1 == pytest.approx(500.0)
    assert p.d1 == pytest.approx(-10.0)
    assert p.e1 == pytest.approx(5.0)
    assert p.stiff1 == pytest.approx(1500.0)
    assert p.fun_a1 == 301
    assert p.fun_b1 == 302
    assert p.scale_t == pytest.approx(1.0)


def test_prop_type43_connect(tmp_path: Path):
    deck = """/BEGIN
TEST_PROP_CONNECT
/PROP/CONNECT/600
Connector Element Property
/END
"""
    model, log = _parse(tmp_path, deck)
    assert len(log.errors) == 0
    assert 600 in model.prop_type43s
    assert 600 in model.prop_connects
    p = model.prop_type43s[600]
    assert p.id == 600
    assert p.title == "Connector Element Property"


def test_m189_all_aliases_and_dictionaries(tmp_path: Path):
    deck = """/BEGIN
TEST_M189_ALIASES
/MAT/PLAS_GURS/1
Plas Gurs
7.85e-6 7.85e-6
210000.0 0.3 1 1 0.25
500.0 300.0 0.2 0.015 1.0e-3
1.5 1.0 2.25 0.1 0.3
0.001 0.04 0.15 0.25
/MAT/CAST_IRON/2
Cast Iron
7.2e-6 7.2e-6
1.0e5 4500.0 1.5 1.6 0.2
120000.0 0.25 110000.0 0.26 250.0
0.5 500.0 400.0 0.08
/MAT/COMP_SOL/3
Comp Sol
1.6e-6 1.6e-6
140000.0 10000.0 10000.0 0.3 0.35
0.02 5000.0 3500.0 5000.0
1500.0 50.0 50.0 0.8 0.1
100.0 2000.0 1200.0 40.0 150.0
150.0 80.0 40.0 90.0 45.0 0.05 1000.0 0.01 1
/MAT/CONNECTOR/4
Connector
7.8e-6 7.8e-6
210000.0 80000.0 1 0.1 2
101 1 5000.0 0.5
/MAT/MARTENSITE/5
Martensite
7.9e-6 7.9e-6
205000.0 0.3 450.0 2.5 0.4
80.0 0.1 0.8 293.15
201 202 1.0 1.2
/FAIL/LADEVEZE/1
1000.0 200.0 300.0 0.5 0.25
1.5 25.0 0.8 1.2 80.0
1 2
/FAIL/MMC/2
500.0 0.1 1.2 0.05 1.5
0.2 1 1 0
/FAIL/SPALL/3
0.1 0.2 0.3 0.4 0.5
1.0 -500.0 1
/PROP/SOLID/1
Solid
14 0 0 1 1 1 0 0 0.1
1.1 0.05 0.1
1.0e-6 0 0.0 0.0 0.0 0 0
/PROP/SPRING_GENE/2
Spring Gene
1.0 0.01 0 0 0 0 0
1000.0 10.0 0.0 0.0 0.0
0 0 0 0 0.0 0.0
/PROP/SPRING_AXI/3
Spring Axi
1.0 0.01 0 0 0 0 0 0
1000.0 10.0 0.0 0.0 0.0
0 0 0 0 0.0 0.0
/PROP/SPRING_PRE/4
Spring Pre
1.0 0 0 1000.0
100.0 1.0 0.0 1000.0
0 0 1.0 1.0 1.0
/PROP/PROP_CONNECT/5
Prop Connect
/END
"""
    model, log = _parse(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.materials and model.materials[1].law == 52
    assert 2 in model.materials and model.materials[2].law == 16
    assert 3 in model.materials and model.materials[3].law == 14
    assert 4 in model.materials and model.materials[4].law == 59
    assert 5 in model.materials and model.materials[5].law == 64
    assert 1 in model.properties and model.properties[1].type == 14
    assert 2 in model.properties and model.properties[2].type == 8
    assert 3 in model.properties and model.properties[3].type == 25
    assert 4 in model.properties and model.properties[4].type == 32
    assert 5 in model.properties and model.properties[5].type == 43
    assert 1 in model.fail_ladevezes
    assert 2 in model.fail_mmcs
    assert 3 in model.fail_spalls

