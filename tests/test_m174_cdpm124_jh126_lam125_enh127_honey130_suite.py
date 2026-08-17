"""Unit test suite for Milestone M174: Advanced Concrete Damage, Laminated/Enhanced Composites & Modified Honeycomb Material Models Suite.

Tests cover:
1. /MAT/LAW124 & /MAT/CDPM2 (Concrete Damage Plastic Model 2)
2. /MAT/LAW126 & /MAT/JOHNSON_HOLMQUIST_CONCRETE / /MAT/JH_CONC / /MAT/JHC
3. /MAT/LAW125 & /MAT/LAMINATED_COMPOSITE / /MAT/LAM_COMP
4. /MAT/LAW127 & /MAT/ENHANCED_COMPOSITE / /MAT/ENH_COMP
5. /MAT/LAW130 & /MAT/MODIFIED_HONEYCOMB / /MAT/MOD_HONEYCOMB
"""

from pathlib import Path
import pytest
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog


def _parse_deck(tmp_path: Path, deck_text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="ascii")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(p))
    parse_starter_deck(blocks, model, log)
    return model, log


def test_mat_law124_cdpm2_parsing(tmp_path: Path):
    """Test /MAT/LAW124 and /MAT/CDPM2 parsing for concrete damage plasticity."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW124_TEST
/MAT/LAW124/101
Concrete Damage Plastic Model
#              RHO_I
              2.4E-9
#                  E                  NU                                            IRATE                FCUT
             30000.0                 0.2                                                1                 0.0
#                ECC                 QH0                  FT                  FC                  HP
                0.55                 0.3                 3.0                30.0                 0.5
#                 AH                  BH                  CH                  DH
                0.08               0.003                 2.0               1e-06
#                 AS                  BS                  DF                         DFLAG     DTYPE      IREG
                15.0                 1.0                0.85                             1         0         1
#                 WF                 WF1                 FT1                 EFC
              0.0001              0.0005                 0.5              0.0035
/MAT/CDPM2/102
CDPM2 Free Format
2.4e-9
35000.0 0.18 0 10.0
0.5 0.25 3.5 35.0 0.6
0.1 0.004 2.5 1e-5
12.0 0.8 0.8 1 1 0
0.0002 0.0008 0.6 0.004
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law124 entity 101
    assert 101 in model.mat_law124s
    m124_1 = model.mat_law124s[101]
    assert m124_1.id == 101
    assert m124_1.title == "Concrete Damage Plastic Model"
    assert pytest.approx(m124_1.rho0) == 2.4e-9
    assert pytest.approx(m124_1.e) == 30000.0
    assert pytest.approx(m124_1.nu) == 0.2
    assert m124_1.irate == 1
    assert pytest.approx(m124_1.ecc) == 0.55
    assert pytest.approx(m124_1.qh0) == 0.3
    assert pytest.approx(m124_1.ft) == 3.0
    assert pytest.approx(m124_1.fc) == 30.0
    assert pytest.approx(m124_1.hp) == 0.5
    assert pytest.approx(m124_1.ah) == 0.08
    assert pytest.approx(m124_1.bh) == 0.003
    assert pytest.approx(m124_1.ch) == 2.0
    assert pytest.approx(m124_1.dh) == 1e-6
    assert pytest.approx(m124_1.as_) == 15.0
    assert pytest.approx(m124_1.bs) == 1.0
    assert pytest.approx(m124_1.df) == 0.85
    assert m124_1.dflag == 1
    assert m124_1.dtype == 0
    assert m124_1.ireg == 1
    assert pytest.approx(m124_1.wf) == 0.0001
    assert pytest.approx(m124_1.wf1) == 0.0005
    assert pytest.approx(m124_1.ft1) == 0.5
    assert pytest.approx(m124_1.efc) == 0.0035

    # Verify Law124 entity 102
    assert 102 in model.mat_law124s
    m124_2 = model.mat_law124s[102]
    assert m124_2.id == 102
    assert pytest.approx(m124_2.e) == 35000.0
    assert pytest.approx(m124_2.fc) == 35.0
    assert m124_2.dtype == 1

    # Verify model.materials and GenericMaterialRecord
    mat1 = model.materials[101]
    assert mat1.law == 124
    assert mat1.params["MAT_E"] == 30000.0
    assert mat1.params["MAT_FC"] == 30.0
    assert mat1.params["MAT_AS"] == 15.0
    assert mat1.record.law_name == "LAW124"


def test_mat_law126_johnson_holmquist_concrete_parsing(tmp_path: Path):
    """Test /MAT/LAW126 and Johnson-Holmquist concrete/ceramic model parsing."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW126_TEST
/MAT/LAW126/201
JH Concrete Model
#              RHO_I
              2.4E-9
#                  G
             15000.0
#                  A                   B                   N                  FC                  T0
                0.79                 1.6                0.61                48.0                 4.0
#                  C                EPS0                FCUT               SFMAX               EFMIN
               0.007                 1.0                 0.0                 7.0                0.01
#                 PC                 MUC                  PL                 MUL
                16.0               0.001               800.0                 0.1
#                 K1                  K2                  K3
             85000.0            -17000.0             34000.0
#                 D1                  D2              IDEL             EPS_MAX             IFAILSO
                0.04                 1.0                 1                 0.5                   2
#                 CT                POWT                  CC                POWC
                 0.1                 0.8                 0.2                 0.9
/MAT/JH_CONC/202
JH Concrete Free Format
2.5e-9
18000.0
0.8 1.5 0.6 50.0 5.0
0.01 1.0 0.0 6.5 0.02
20.0 0.002 900.0 0.12
90000.0 -15000.0 30000.0
0.05 1.2 0 0.0 0
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law126 entity 201
    assert 201 in model.mat_law126s
    m126_1 = model.mat_law126s[201]
    assert m126_1.id == 201
    assert m126_1.title == "JH Concrete Model"
    assert pytest.approx(m126_1.rho0) == 2.4e-9
    assert pytest.approx(m126_1.g) == 15000.0
    assert pytest.approx(m126_1.a) == 0.79
    assert pytest.approx(m126_1.b) == 1.6
    assert pytest.approx(m126_1.n) == 0.61
    assert pytest.approx(m126_1.fc) == 48.0
    assert pytest.approx(m126_1.t0) == 4.0
    assert pytest.approx(m126_1.c) == 0.007
    assert pytest.approx(m126_1.eps0) == 1.0
    assert pytest.approx(m126_1.sfmax) == 7.0
    assert pytest.approx(m126_1.efmin) == 0.01
    assert pytest.approx(m126_1.pc) == 16.0
    assert pytest.approx(m126_1.muc) == 0.001
    assert pytest.approx(m126_1.pl) == 800.0
    assert pytest.approx(m126_1.mul) == 0.1
    assert pytest.approx(m126_1.k1) == 85000.0
    assert pytest.approx(m126_1.k2) == -17000.0
    assert pytest.approx(m126_1.k3) == 34000.0
    assert pytest.approx(m126_1.d1) == 0.04
    assert pytest.approx(m126_1.d2) == 1.0
    assert m126_1.idel == 1
    assert pytest.approx(m126_1.eps_max) == 0.5
    assert m126_1.ifailso == 2
    assert pytest.approx(m126_1.ct) == 0.1
    assert pytest.approx(m126_1.powt) == 0.8
    assert pytest.approx(m126_1.cc) == 0.2
    assert pytest.approx(m126_1.powc) == 0.9

    # Verify Law126 entity 202
    assert 202 in model.mat_law126s
    m126_2 = model.mat_law126s[202]
    assert m126_2.id == 202
    assert pytest.approx(m126_2.g) == 18000.0
    assert pytest.approx(m126_2.k1) == 90000.0

    # Verify material parameters dictionary
    mat1 = model.materials[201]
    assert mat1.law == 126
    assert mat1.params["MAT_G"] == 15000.0
    assert mat1.params["MAT_K1"] == 85000.0
    assert mat1.record.law_name == "LAW126"


def test_mat_law125_laminated_composite_parsing(tmp_path: Path):
    """Test /MAT/LAW125 and /MAT/LAMINATED_COMPOSITE parsing."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW125_TEST
/MAT/LAW125/301
Laminated Composite Mat
#              RHO_I
              1.6E-9
#                E11                 E22                 E33                                   Ifail
            140000.0             10000.0             10000.0                                       1
#                G12                 G31                 G23
              5000.0              5000.0              3500.0
#               PR21                PR31                PR32
                 0.3                 0.3                0.45
#     LCE11T                E11T   Fct_T11                 T11             SLIMT11
        1001                 0.0      1011              1500.0                 0.5
#     LCE11C                E11C   Fct_C11                 C11             SLIMC11
        1002                 0.0      1012              1200.0                 0.6
#     LCE22T                E22T   Fct_T22                 T22             SLIMT22
        1003                 0.0      1013                50.0                 0.4
#     LCE22C                E22C   Fct_C22                 C22             SLIMC22
        1004                 0.0      1014               200.0                 0.5
#     LCE33T                E33T   Fct_T33                 T33             SLIMT33
        1005                 0.0      1015                50.0                 0.4
#     LCE33C                E33C   Fct_C33                 C33             SLIMC33
        1006                 0.0      1016               200.0                 0.5
#               g12a                t12a                g12b                t12b             SLIMS12
                0.02                70.0                0.05                80.0                 0.3
#   Fct_g12a  Fct_t12a  Fct_g12b  Fct_t12b
        2001      2002      2003      2004
#               g31a                t31a                g31b                t31b             SLIMS31
                0.02                70.0                0.05                80.0                 0.3
#   Fct_g31a  Fct_t31a  Fct_g31b  Fct_t31b
        2005      2006      2007      2008
#               g23a                t23a                g23b                t23b             SLIMS23
                0.03                50.0                0.06                60.0                 0.3
#   Fct_g23a  Fct_t23a  Fct_g23b  Fct_t23b
        2009      2010      2011      2012
#               EPSF                EPSR                Dmax
                0.05                0.02                 0.9
#   Fct_Fail                Fail
        3001                 0.1
#               FCUT
                10.0
/MAT/LAM_COMP/302
Lam Comp Free Format
1.5e-9
120000.0 9000.0 9000.0 0
4500.0 4500.0 3000.0
0.28 0.28 0.4
0 0.0 0 1400.0 0.0
0 0.0 0 1100.0 0.0
0 0.0 0 45.0 0.0
0 0.0 0 180.0 0.0
0 0.0 0 45.0 0.0
0 0.0 0 180.0 0.0
0.015 65.0 0.04 75.0 0.2
0 0 0 0
0.015 65.0 0.04 75.0 0.2
0 0 0 0
0.025 45.0 0.05 55.0 0.2
0 0 0 0
0.04 0.015 0.95
0 0.0
0.0
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law125 entity 301
    assert 301 in model.mat_law125s
    m125_1 = model.mat_law125s[301]
    assert m125_1.id == 301
    assert m125_1.title == "Laminated Composite Mat"
    assert pytest.approx(m125_1.rho0) == 1.6e-9
    assert pytest.approx(m125_1.ea) == 140000.0
    assert pytest.approx(m125_1.eb) == 10000.0
    assert pytest.approx(m125_1.ec) == 10000.0
    assert m125_1.ifail == 1
    assert pytest.approx(m125_1.gab) == 5000.0
    assert pytest.approx(m125_1.prba) == 0.3
    assert m125_1.lce11t == 1001
    assert pytest.approx(m125_1.t11) == 1500.0
    assert pytest.approx(m125_1.slimt11) == 0.5
    assert m125_1.fct_c11 == 1012
    assert pytest.approx(m125_1.c11) == 1200.0
    assert pytest.approx(m125_1.g12a) == 0.02
    assert pytest.approx(m125_1.t12a) == 70.0
    assert m125_1.fct_g12a == 2001
    assert pytest.approx(m125_1.epsf) == 0.05
    assert pytest.approx(m125_1.dmax) == 0.9
    assert m125_1.fct_fail == 3001
    assert pytest.approx(m125_1.fcut) == 10.0

    # Verify Law125 entity 302
    assert 302 in model.mat_law125s
    m125_2 = model.mat_law125s[302]
    assert m125_2.id == 302
    assert pytest.approx(m125_2.ea) == 120000.0
    assert pytest.approx(m125_2.t11) == 1400.0

    # Verify material parameters dictionary
    mat1 = model.materials[301]
    assert mat1.law == 125
    assert mat1.params["LSD_MAT_EA"] == 140000.0
    assert mat1.params["lce11t"] == 1001
    assert mat1.record.law_name == "LAW125"


def test_mat_law127_enhanced_composite_parsing(tmp_path: Path):
    """Test /MAT/LAW127 and /MAT/ENHANCED_COMPOSITE parsing."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW127_TEST
/MAT/LAW127/401
Enhanced Composite Model
#              RHO_I
              1.5E-9
#                 EA                  EB                  EC
            150000.0             12000.0             12000.0
#                GAB                 GCA                 GBC
              6000.0              6000.0              4000.0
#               PRBA                PRCA                PRCB
                0.27                0.27                 0.4
#                 XT              SLIMT1                LCXT             SCALCXT
              1800.0                 0.5                1001                 1.2
#                 YT              SLIMT2                LCYT             SCALCYT
                60.0                 0.4                1002                 1.0
#                 SC              SLIMSC                LCSC             SCALCSC
                80.0                 0.3                1003                 1.1
#                 XC              SLIMC1                LCXC             SCALCXC
              1400.0                 0.6                1004                 1.0
#                 YC              SLIMC2                LCYC             SCALCYC
               220.0                 0.5                1005                 1.0
#               FCUT
                12.0
#               ALPH                BETA                2WAY                  TI
                 0.5                 1.0                   1                   2
#             DFAILT              DFAILC              DFAILS              DFAILM               RATIO
                0.05                0.03                0.04                0.02                 0.8
#             NCYRED               TFAIL                FBRT               YCFAC
                  10                0.01                 0.5                 2.0
#                EFS                EPSF                EPSR                TSMD
                0.02                0.06                0.03                0.05
/MAT/ENHANCED_COMPOSITE/402
Enhanced Composite Free Format
1.6e-9
160000.0 13000.0 13000.0
6500.0 6500.0 4200.0
0.25 0.25 0.38
2000.0 0.55 0 1.0
70.0 0.45 0 1.0
90.0 0.35 0 1.0
1500.0 0.65 0 1.0
240.0 0.55 0 1.0
0.0
0.0 0.0 0 0
0.04 0.02 0.03 0.01 0.7
5 0.0 0.0 1.5
0.0 0.05 0.02 0.0
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law127 entity 401
    assert 401 in model.mat_law127s
    m127_1 = model.mat_law127s[401]
    assert m127_1.id == 401
    assert m127_1.title == "Enhanced Composite Model"
    assert pytest.approx(m127_1.rho0) == 1.5e-9
    assert pytest.approx(m127_1.ea) == 150000.0
    assert pytest.approx(m127_1.eb) == 12000.0
    assert pytest.approx(m127_1.ec) == 12000.0
    assert pytest.approx(m127_1.gab) == 6000.0
    assert pytest.approx(m127_1.prba) == 0.27
    assert pytest.approx(m127_1.xt) == 1800.0
    assert pytest.approx(m127_1.slimt1) == 0.5
    assert m127_1.lcxt == 1001
    assert pytest.approx(m127_1.scalcxt) == 1.2
    assert pytest.approx(m127_1.yt) == 60.0
    assert pytest.approx(m127_1.sc) == 80.0
    assert pytest.approx(m127_1.xc) == 1400.0
    assert pytest.approx(m127_1.yc) == 220.0
    assert pytest.approx(m127_1.fcut) == 12.0
    assert pytest.approx(m127_1.alph) == 0.5
    assert pytest.approx(m127_1.beta) == 1.0
    assert m127_1.two_way == 1
    assert m127_1.ti == 2
    assert pytest.approx(m127_1.dfailt) == 0.05
    assert pytest.approx(m127_1.ratio) == 0.8
    assert m127_1.ncyred == 10
    assert pytest.approx(m127_1.tfail) == 0.01
    assert pytest.approx(m127_1.fbrt) == 0.5
    assert pytest.approx(m127_1.ycfac) == 2.0
    assert pytest.approx(m127_1.efs) == 0.02
    assert pytest.approx(m127_1.epsf) == 0.06
    assert pytest.approx(m127_1.tsmd) == 0.05

    # Verify Law127 entity 402
    assert 402 in model.mat_law127s
    m127_2 = model.mat_law127s[402]
    assert m127_2.id == 402
    assert pytest.approx(m127_2.ea) == 160000.0
    assert pytest.approx(m127_2.xt) == 2000.0
    assert m127_2.ncyred == 5

    # Verify material parameters dictionary
    mat1 = model.materials[401]
    assert mat1.law == 127
    assert mat1.params["LSDYNA_EA"] == 150000.0
    assert mat1.params["xt"] == 1800.0
    assert mat1.record.law_name == "LAW127"


def test_mat_law130_modified_honeycomb_parsing(tmp_path: Path):
    """Test /MAT/LAW130 and /MAT/MODIFIED_HONEYCOMB parsing."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW130_TEST
/MAT/LAW130/501
Modified Honeycomb Model
#              RHO_I
              0.1E-9
#                  E                  NU                SIGY                  VF                  MU
               500.0                 0.2                15.0                0.05                0.01
#      IFORM    SHDFLG       LCA       LCB       LCC       LCS      LCAB      LCBC      LCCA      LCSR
           1         0      1001      1002      1003      1004      1005      1006      1007        -1
#               EAAU                EBBU                ECCU                GABU                GBCU
               400.0               400.0              2500.0               150.0               150.0
#               GCAU                RFAC                TSEF                SSEF                 PRU
               200.0                 0.1                 0.0                 0.0                   2
#      LCSRA     LCSRB     LCSRC    LCSRAB    LCSRBC    LCSRCA
        2001      2002      2003      2004      2005      2006
#              PRUAB               PRUAC               PRUBC               PRUBA               PRUCA
                0.05                0.02                0.02                0.05                0.01
#              PRUCB
                0.01
/MAT/MOD_HONEYCOMB/502
Mod Honeycomb Free Format
0.12e-9
600.0 0.15 18.0 0.06 0.02
0 0 3001 3002 3003 3004 3005 3006 3007 0
500.0 500.0 3000.0 200.0 200.0
250.0 0.15 0.0 0.0 0
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law130 entity 501
    assert 501 in model.mat_law130s
    m130_1 = model.mat_law130s[501]
    assert m130_1.id == 501
    assert m130_1.title == "Modified Honeycomb Model"
    assert pytest.approx(m130_1.rho0) == 0.1e-9
    assert pytest.approx(m130_1.e) == 500.0
    assert pytest.approx(m130_1.nu) == 0.2
    assert pytest.approx(m130_1.sigy) == 15.0
    assert pytest.approx(m130_1.vf) == 0.05
    assert pytest.approx(m130_1.mu) == 0.01
    assert m130_1.iform == 1
    assert m130_1.shdflg == 0
    assert m130_1.lca == 1001
    assert m130_1.lcsr == -1
    assert pytest.approx(m130_1.eaau) == 400.0
    assert pytest.approx(m130_1.eccu) == 2500.0
    assert pytest.approx(m130_1.gcau) == 200.0
    assert pytest.approx(m130_1.rfac) == 0.1
    assert m130_1.pru == 2
    assert m130_1.lcsra == 2001
    assert m130_1.lcsrca == 2006
    assert pytest.approx(m130_1.pruab) == 0.05
    assert pytest.approx(m130_1.prucb) == 0.01

    # Verify Law130 entity 502
    assert 502 in model.mat_law130s
    m130_2 = model.mat_law130s[502]
    assert m130_2.id == 502
    assert pytest.approx(m130_2.e) == 600.0
    assert m130_2.lca == 3001
    assert m130_2.pru == 0

    # Verify material parameters dictionary
    mat1 = model.materials[501]
    assert mat1.law == 130
    assert mat1.params["MAT_E"] == 500.0
    assert mat1.params["LSDYNA_SIGY"] == 15.0
    assert mat1.params["LSD_LCID"] == 1001
    assert mat1.record.law_name == "LAW130"
