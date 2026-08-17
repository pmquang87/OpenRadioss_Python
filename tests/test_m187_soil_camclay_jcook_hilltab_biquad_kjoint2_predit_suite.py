"""Tests for M187: Geotechnical, Hydrodynamic, Tabulated Plasticity & Advanced Joint/Interface Suite.

Keywords covered:
- /MAT/LAW3 (PLAS_BOST, BOSTEELS)
- /MAT/LAW4 (HYD_JCOOK, HYDRO_JCOOK)
- /MAT/LAW5 (JCOOK_TAB, JOHN_COOK_TAB)
- /MAT/LAW10 (SOIL, SOIL_CONC)
- /MAT/LAW14 (CAM_CLAY, CAMCLAY)
- /MAT/LAW21 (DUCKHUB, DRUCKER_PRAGER)
- /MAT/LAW32 (HILL_TAB, HILL_PLAS_TAB)
- /MAT/LAW37 (BIQUAD, BANABIC, BIPHAS)
- /PROP/TYPE45 (KJOINT2, KINEMATIC_JOINT2)
- /PROP/TYPE36 (PREDIT, DELAMINATION)
"""

from pathlib import Path
import pytest
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog


def _parse(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


# ---------------------------------------------------------------------------
# 1. /MAT/LAW3 (PLAS_BOST)
# ---------------------------------------------------------------------------

def test_mat_law3_fixed(tmp_path: Path):
    deck = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/LAW3/101
Bosteels Steel
#              RHO_I                   E                  NU
              7.8E-9              210000                 0.3
#              SIG_Y                 E_T                   C                   P
                 350                1500               40.0                 5.0
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 101 in model.mat_law3s
    m = model.mat_law3s[101]
    assert m.rho0 == pytest.approx(7.8e-9)
    assert m.e == pytest.approx(210000.0)
    assert m.nu == pytest.approx(0.3)
    assert m.sig_y == pytest.approx(350.0)
    assert m.e_t == pytest.approx(1500.0)
    assert m.c == pytest.approx(40.0)
    assert m.p == pytest.approx(5.0)
    assert 101 in model.materials
    assert model.materials[101].params["sig_y"] == pytest.approx(350.0)


def test_mat_law3_free(tmp_path: Path):
    deck = """\
/MAT/PLAS_BOST/102
Free Bosteels
7.85e-9 205000.0 0.28
420.0 2000.0 100.0 4.0
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 102 in model.mat_law3s
    m = model.mat_law3s[102]
    assert m.rho0 == pytest.approx(7.85e-9)
    assert m.e == pytest.approx(205000.0)
    assert m.nu == pytest.approx(0.28)
    assert m.sig_y == pytest.approx(420.0)
    assert m.e_t == pytest.approx(2000.0)
    assert m.c == pytest.approx(100.0)
    assert m.p == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 2. /MAT/LAW4 (HYD_JCOOK)
# ---------------------------------------------------------------------------

def test_mat_law4_fixed(tmp_path: Path):
    deck = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/LAW4/201
Hydro JCook
#              RHO_I                  C0                   S              GAMMA0                   A
              8.9e-6              3940.0                1.48                 2.0                 0.5
#                  A                   B                   N                   C             EPS_MAX             SIG_MAX
                90.0               292.0                0.31               0.025                 0.0              1000.0
#                 T0                  TM                   M                  CP                PMIN
               293.0              1356.0                1.09               385.0              -500.0
#                 C0                  C1                  C2                  C3                  C4                  C5
                 0.0                 0.0                 0.0                 0.0                 0.0                 0.0
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 201 in model.mat_law4s
    m = model.mat_law4s[201]
    assert m.rho_i == pytest.approx(8.9e-6)
    assert m.c0_eos == pytest.approx(3940.0)
    assert m.s_eos == pytest.approx(1.48)
    assert m.gamma0 == pytest.approx(2.0)
    assert m.a == pytest.approx(90.0)
    assert m.b == pytest.approx(292.0)
    assert m.n == pytest.approx(0.31)
    assert m.c == pytest.approx(0.025)
    assert m.sig_max == pytest.approx(1000.0)
    assert m.t0 == pytest.approx(293.0)
    assert m.tm == pytest.approx(1356.0)
    assert m.m == pytest.approx(1.09)
    assert m.cp == pytest.approx(385.0)
    assert m.pmin == pytest.approx(-500.0)


def test_mat_law4_free(tmp_path: Path):
    deck = """\
/MAT/HYD_JCOOK/202
Free Hydro JCook
8.9e-6 4000.0 1.5 2.0 0.5
100.0 300.0 0.3 0.02 0.0 1200.0
300.0 1400.0 1.0 400.0 -600.0
1.0 2.0 3.0 4.0 5.0 6.0
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 202 in model.mat_law4s
    m = model.mat_law4s[202]
    assert m.rho_i == pytest.approx(8.9e-6)
    assert m.c0_eos == pytest.approx(4000.0)
    assert m.s_eos == pytest.approx(1.5)
    assert m.a == pytest.approx(100.0)
    assert m.b == pytest.approx(300.0)
    assert m.pmin == pytest.approx(-600.0)
    assert m.c0 == pytest.approx(1.0)
    assert m.c5 == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# 3. /MAT/LAW5 (JCOOK_TAB)
# ---------------------------------------------------------------------------

def test_mat_law5_fixed(tmp_path: Path):
    deck = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/LAW5/301
Tabulated JCook
#              RHO_I                   E                  NU
              7.8E-9              210000                 0.3
#                  A                   B                   N                   C             SIG_MAX
               450.0               600.0                0.25               0.015               900.0
#   FCT_ID_1  FCT_ID_2  FCT_ID_3  FCT_ID_4  FCT_ID_5
          11        12        13        14        15
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 301 in model.mat_law5s
    m = model.mat_law5s[301]
    assert m.rho0 == pytest.approx(7.8e-9)
    assert m.e == pytest.approx(210000.0)
    assert m.nu == pytest.approx(0.3)
    assert m.a == pytest.approx(450.0)
    assert m.b == pytest.approx(600.0)
    assert m.n == pytest.approx(0.25)
    assert m.c == pytest.approx(0.015)
    assert m.sig_max == pytest.approx(900.0)
    assert m.fct_id1 == 11
    assert m.fct_id2 == 12
    assert m.fct_id3 == 13
    assert m.fct_id4 == 14
    assert m.fct_id5 == 15


def test_mat_law5_free(tmp_path: Path):
    deck = """\
/MAT/JCOOK_TAB/302
Free Tabulated JCook
7.85e-9 200000.0 0.29
400.0 500.0 0.2 0.01 800.0
21 22 23 24 25
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 302 in model.mat_law5s
    m = model.mat_law5s[302]
    assert m.rho0 == pytest.approx(7.85e-9)
    assert m.a == pytest.approx(400.0)
    assert m.sig_max == pytest.approx(800.0)
    assert m.fct_id1 == 21
    assert m.fct_id5 == 25


# ---------------------------------------------------------------------------
# 4. /MAT/LAW10 (SOIL)
# ---------------------------------------------------------------------------

def test_mat_law10_fixed(tmp_path: Path):
    deck = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/LAW10/401
Soil Concrete Model
#              RHO_I                   G                   K                  A0                  A1                  A2
              2.2E-9              12000.              25000.                10.0                 0.5                0.01
#              P_CUT               P_MIN  FCT_ID_P
               -10.0             -1000.0          42
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 401 in model.mat_law10s
    m = model.mat_law10s[401]
    assert m.rho0 == pytest.approx(2.2e-9)
    assert m.g == pytest.approx(12000.0)
    assert m.k == pytest.approx(25000.0)
    assert m.a0 == pytest.approx(10.0)
    assert m.a1 == pytest.approx(0.5)
    assert m.a2 == pytest.approx(0.01)
    assert m.p_cut == pytest.approx(-10.0)
    assert m.p_min == pytest.approx(-1000.0)
    assert m.fct_id_p == 42


def test_mat_law10_free(tmp_path: Path):
    deck = """\
/MAT/SOIL_CONC/402
Free Soil Concrete
2.3e-9 15000.0 30000.0 12.0 0.6 0.02
-15.0 -2000.0 99
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 402 in model.mat_law10s
    m = model.mat_law10s[402]
    assert m.rho0 == pytest.approx(2.3e-9)
    assert m.g == pytest.approx(15000.0)
    assert m.k == pytest.approx(30000.0)
    assert m.p_cut == pytest.approx(-15.0)
    assert m.fct_id_p == 99


# ---------------------------------------------------------------------------
# 5. /MAT/LAW14 (CAM_CLAY)
# ---------------------------------------------------------------------------

def test_mat_law14_fixed(tmp_path: Path):
    deck = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/LAW14/501
Modified Cam Clay
#              RHO_I                   G                  NU
              1.8E-9              5000.0                0.35
#                  M               LAMDA               KAPPA                  E0                 PC0
                 1.2                0.15                0.03                 0.8               120.0
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 501 in model.mat_law14s
    m = model.mat_law14s[501]
    assert m.rho0 == pytest.approx(1.8e-9)
    assert m.g == pytest.approx(5000.0)
    assert m.nu == pytest.approx(0.35)
    assert m.m == pytest.approx(1.2)
    assert m.lamda == pytest.approx(0.15)
    assert m.kappa == pytest.approx(0.03)
    assert m.e0 == pytest.approx(0.8)
    assert m.pc0 == pytest.approx(120.0)


def test_mat_law14_free(tmp_path: Path):
    deck = """\
/MAT/CAM_CLAY/502
Free Cam Clay
1.9e-9 6000.0 0.3
1.1 0.18 0.04 0.75 150.0
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 502 in model.mat_law14s
    m = model.mat_law14s[502]
    assert m.rho0 == pytest.approx(1.9e-9)
    assert m.g == pytest.approx(6000.0)
    assert m.m == pytest.approx(1.1)
    assert m.lamda == pytest.approx(0.18)
    assert m.kappa == pytest.approx(0.04)
    assert m.e0 == pytest.approx(0.75)
    assert m.pc0 == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# 6. /MAT/LAW21 (DUCKHUB)
# ---------------------------------------------------------------------------

def test_mat_law21_fixed(tmp_path: Path):
    deck = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/LAW21/601
Drucker Prager Cap
#              RHO_I                   E                  NU
              2.0E-9              30000.                 0.2
#                 A0                  A1                  A2                   W                   D                  X0
                15.0                 0.4               0.005                 0.1                 0.2               -50.0
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 601 in model.mat_law21s
    m = model.mat_law21s[601]
    assert m.rho0 == pytest.approx(2.0e-9)
    assert m.e == pytest.approx(30000.0)
    assert m.nu == pytest.approx(0.2)
    assert m.a0 == pytest.approx(15.0)
    assert m.a1 == pytest.approx(0.4)
    assert m.a2 == pytest.approx(0.005)
    assert m.w == pytest.approx(0.1)
    assert m.d == pytest.approx(0.2)
    assert m.x0 == pytest.approx(-50.0)


def test_mat_law21_free(tmp_path: Path):
    deck = """\
/MAT/DUCKHUB/602
Free Drucker Prager
2.1e-9 35000.0 0.22
18.0 0.45 0.006 0.12 0.25 -60.0
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 602 in model.mat_law21s
    m = model.mat_law21s[602]
    assert m.rho0 == pytest.approx(2.1e-9)
    assert m.e == pytest.approx(35000.0)
    assert m.a0 == pytest.approx(18.0)
    assert m.w == pytest.approx(0.12)
    assert m.x0 == pytest.approx(-60.0)


# ---------------------------------------------------------------------------
# 7. /MAT/LAW32 (HILL_TAB)
# ---------------------------------------------------------------------------

def test_mat_law32_fixed(tmp_path: Path):
    deck = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/LAW32/701
Hill Tabulated Anisotropic
#              RHO_I                  E1                  E2                  E3                NU12                NU23                NU31
              2.7E-9              70000.              68000.              65000.                 0.3                0.32                0.29
#                G12                 G23                 G31
              26000.              25000.              24000.
# FCT_ID11  FCT_ID22  FCT_ID33  FCT_ID12  FCT_ID23  FCT_ID31
       111       122       133       112       123       131
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 701 in model.mat_law32s
    m = model.mat_law32s[701]
    assert m.rho0 == pytest.approx(2.7e-9)
    assert m.e1 == pytest.approx(70000.0)
    assert m.e2 == pytest.approx(68000.0)
    assert m.e3 == pytest.approx(65000.0)
    assert m.nu12 == pytest.approx(0.3)
    assert m.nu23 == pytest.approx(0.32)
    assert m.nu31 == pytest.approx(0.29)
    assert m.g12 == pytest.approx(26000.0)
    assert m.g23 == pytest.approx(25000.0)
    assert m.g31 == pytest.approx(24000.0)
    assert m.fct_id11 == 111
    assert m.fct_id22 == 122
    assert m.fct_id33 == 133
    assert m.fct_id12 == 112
    assert m.fct_id23 == 123
    assert m.fct_id31 == 131


def test_mat_law32_free(tmp_path: Path):
    deck = """\
/MAT/HILL_TAB/702
Free Hill Tabulated
2.75e-9 72000.0 70000.0 67000.0 0.31 0.33 0.30
27000.0 26000.0 25000.0
211 222 233 212 223 231
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 702 in model.mat_law32s
    m = model.mat_law32s[702]
    assert m.rho0 == pytest.approx(2.75e-9)
    assert m.e1 == pytest.approx(72000.0)
    assert m.g12 == pytest.approx(27000.0)
    assert m.fct_id11 == 211
    assert m.fct_id31 == 231


# ---------------------------------------------------------------------------
# 8. /MAT/LAW37 (BIQUAD)
# ---------------------------------------------------------------------------

def test_mat_law37_fixed(tmp_path: Path):
    deck = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/MAT/LAW37/801
Biquad Banabic Model
#              RHO_I                   E                  NU                   A                   B                   N
              2.7E-9              70000.                0.33               150.0               300.0                0.22
#                 C1                  C2                  C3                  C4                  C5                  C6                  C7                  C8
                 1.0                 1.1                 0.9                 1.0                 1.05                0.95                 1.02                0.98
#                  P                   Q
                 2.0                 2.0
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 801 in model.mat_law37s
    m = model.mat_law37s[801]
    assert m.rho0 == pytest.approx(2.7e-9)
    assert m.e == pytest.approx(70000.0)
    assert m.nu == pytest.approx(0.33)
    assert m.a == pytest.approx(150.0)
    assert m.b == pytest.approx(300.0)
    assert m.n == pytest.approx(0.22)
    assert m.c1 == pytest.approx(1.0)
    assert m.c2 == pytest.approx(1.1)
    assert m.c8 == pytest.approx(0.98)
    assert m.p == pytest.approx(2.0)
    assert m.q == pytest.approx(2.0)


def test_mat_law37_free(tmp_path: Path):
    deck = """\
/MAT/BIQUAD/802
Free Biquad Banabic
2.8e-9 75000.0 0.32 160.0 320.0 0.25
1.0 1.2 0.8 1.1 1.0 0.9 1.0 1.0
2.5 2.5
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 802 in model.mat_law37s
    m = model.mat_law37s[802]
    assert m.rho0 == pytest.approx(2.8e-9)
    assert m.a == pytest.approx(160.0)
    assert m.c2 == pytest.approx(1.2)
    assert m.p == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# 9. /PROP/TYPE45 (KJOINT2)
# ---------------------------------------------------------------------------

def test_prop_type45_fixed(tmp_path: Path):
    deck = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/PROP/TYPE45/901
Kinematic Joint 2
#  PROP_TYPE                  KN               SCALE                  CR             ISENSOR
           2              1000.0                 1.0                 0.1                   5
#      SKEW1       SKEW2                 KTX                 KTY                 KTZ
           1           2             50000.0             50000.0             50000.0
#     XT_FUN      YT_FUN      ZT_FUN                  XN                  YN                  ZN                  XC                  YC                  ZC
          11          12          13               -10.0               -10.0               -10.0                10.0                10.0                10.0
#                CTX                 CTY                 CTZ             CTX_FUN             CTY_FUN             CTZ_FUN                  VX                  VY                  VZ                  FX                  FY                  FZ
                50.0                50.0                50.0                  21                  22                  23                 1.0                 0.0                 0.0               100.0                 0.0                 0.0
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 901 in model.prop_type45s
    p = model.prop_type45s[901]
    assert p.joint_type == 2
    assert p.kn == pytest.approx(1000.0)
    assert p.scale == pytest.approx(1.0)
    assert p.cr == pytest.approx(0.1)
    assert p.isensor == 5
    assert p.skew1 == 1
    assert p.skew2 == 2
    assert p.ktx == pytest.approx(50000.0)
    assert p.xt_fun == 11
    assert p.xn == pytest.approx(-10.0)
    assert p.xc == pytest.approx(10.0)
    assert p.ctx == pytest.approx(50.0)
    assert p.ctx_fun == 21
    assert p.vx == pytest.approx(1.0)
    assert p.fx == pytest.approx(100.0)
    assert 901 in model.properties
    assert model.properties[901].params["joint_type"] == 2


def test_prop_type45_free(tmp_path: Path):
    deck = """\
/PROP/KJOINT2/902
Free Kinematic Joint 2
3 2000.0 1.0 0.05 7
10 20 60000.0 60000.0 60000.0
101 102 103 -5.0 -5.0 -5.0 5.0 5.0 5.0
25.0 25.0 25.0 201 202 203 0.0 1.0 0.0 0.0 200.0 0.0
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 902 in model.prop_type45s
    p = model.prop_type45s[902]
    assert p.joint_type == 3
    assert p.kn == pytest.approx(2000.0)
    assert p.isensor == 7
    assert p.skew1 == 10
    assert p.skew2 == 20
    assert p.ktx == pytest.approx(60000.0)
    assert p.xt_fun == 101
    assert p.xn == pytest.approx(-5.0)
    assert p.xc == pytest.approx(5.0)
    assert p.ctx_fun == 201
    assert p.vy == pytest.approx(1.0)
    assert p.fy == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# 10. /PROP/TYPE36 (PREDIT)
# ---------------------------------------------------------------------------

def test_prop_type36_fixed(tmp_path: Path):
    deck = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/PROP/TYPE36/1001
Delamination Predit
#     LUTYPE   SKEW_CSID    PROP_ID1    PROP_ID2                  XK
           1           3          10          20               500.0
#     MAT_ID                AREA                 IXX                 IYY                 IZZ                 RAY
           5                25.0                12.5                12.5                25.0                 2.5
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 1001 in model.prop_type36s
    p = model.prop_type36s[1001]
    assert p.lutype == 1
    assert p.skew_csid == 3
    assert p.prop_id1 == 10
    assert p.prop_id2 == 20
    assert p.xk == pytest.approx(500.0)
    assert p.mat_id == 5
    assert p.area == pytest.approx(25.0)
    assert p.ixx == pytest.approx(12.5)
    assert p.iyy == pytest.approx(12.5)
    assert p.izz == pytest.approx(25.0)
    assert p.ray == pytest.approx(2.5)
    assert 1001 in model.properties
    assert model.properties[1001].params["area"] == pytest.approx(25.0)


def test_prop_type36_free(tmp_path: Path):
    deck = """\
/PROP/PREDIT/1002
Free Delamination
2 4 11 21 600.0
6 30.0 15.0 15.0 30.0 3.0
"""
    model, log = _parse(tmp_path, deck)
    assert not log.errors
    assert 1002 in model.prop_type36s
    p = model.prop_type36s[1002]
    assert p.lutype == 2
    assert p.skew_csid == 4
    assert p.prop_id1 == 11
    assert p.prop_id2 == 21
    assert p.xk == pytest.approx(600.0)
    assert p.mat_id == 6
    assert p.area == pytest.approx(30.0)
    assert p.ray == pytest.approx(3.0)
