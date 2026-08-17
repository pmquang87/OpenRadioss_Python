# pyradioss - Python port of OpenRadioss explicit FEM solver
# Milestone M176: Advanced Anisotropic Plasticity, Powder-Burn, Johnson-Cook Phase Transform, Paper Plasticity, Vegter & Deshpande-Fleck Material Models Suite
from pathlib import Path
import pytest
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.common.messages import MessageLog
from pyradioss.model.model import Model


def _parse_deck(tmp_path: Path, deck_text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(deck_text, encoding="ascii")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(p))
    parse_starter_deck(blocks, model, log)
    return model, log


def test_mat_law104_drucker_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW104 and /MAT/JOHNS_VOCE_DRUCKER in fixed and free formats."""
    # Fixed format deck
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed LAW104
/MAT/LAW104/1041
Drucker Voce Material Fixed
#        Init. dens.
             7.8e-09
#                  E                  Nu      Ires
            210000.0                 0.3         1
#         sigma0_yld                   H                  Qv                  Bv                 Cdr
               450.0              1200.0               150.0                15.0                 1.5
#                Cjc                Eps0                Fcut
                0.02                 1.0             10000.0
#                 mu                Tref                Tini
               400.0               293.0               293.0
#                ETA                  Cp              EpsIso               EpsAd
                 0.9               450.0               1e+20               2e+20
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors parsing fixed LAW104: {log.errors}"
    assert 1041 in model.mat_law104s
    m1 = model.mat_law104s[1041]
    assert m1.rho0 == pytest.approx(7.8e-09)
    assert m1.young == pytest.approx(210000.0)
    assert m1.nu == pytest.approx(0.3)
    assert m1.ires == 1
    assert m1.sigma_r == pytest.approx(450.0)
    assert m1.h == pytest.approx(1200.0)
    assert m1.qv == pytest.approx(150.0)
    assert m1.bv == pytest.approx(15.0)
    assert m1.cdr == pytest.approx(1.5)
    assert m1.cjc == pytest.approx(0.02)
    assert m1.epsp0 == pytest.approx(1.0)
    assert m1.fcut == pytest.approx(10000.0)
    assert m1.tss == pytest.approx(400.0)
    assert m1.tref == pytest.approx(293.0)
    assert m1.tini == pytest.approx(293.0)
    assert m1.eta == pytest.approx(0.9)
    assert m1.cp == pytest.approx(450.0)
    assert 1041 in model.materials
    mat = model.materials[1041]
    assert mat.law == 104
    assert mat.record.params["MAT_PR"] == pytest.approx(150.0)

    # Free format deck
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free JOHNS_VOCE_DRUCKER
/MAT/JOHNS_VOCE_DRUCKER/1042
Drucker Voce Material Free
7.85e-09
205000.0 0.29 2
400.0 1000.0 120.0 12.0 1.2
0.015 1.0 12000.0
350.0 300.0 300.0
0.85 480.0 1e20 2e20
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors parsing free JOHNS_VOCE_DRUCKER: {log.errors}"
    assert 1042 in model.mat_law104s
    m2 = model.mat_law104s[1042]
    assert m2.rho0 == pytest.approx(7.85e-09)
    assert m2.young == pytest.approx(205000.0)
    assert m2.nu == pytest.approx(0.29)
    assert m2.ires == 2
    assert m2.sigma_r == pytest.approx(400.0)
    assert m2.h == pytest.approx(1000.0)
    assert m2.qv == pytest.approx(120.0)
    assert m2.bv == pytest.approx(12.0)
    assert m2.cdr == pytest.approx(1.2)
    assert m2.cjc == pytest.approx(0.015)
    assert m2.fcut == pytest.approx(12000.0)
    assert m2.tss == pytest.approx(350.0)


def test_mat_law105_powder_burn_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW105 and /MAT/POWDER_BURN in fixed and free formats."""
    # Fixed format deck
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed LAW105
/MAT/LAW105/1051
Powder Burn Propellant Fixed
#                RHO
             1.8e-09
#                  B                  P0                 PSH
              8500.0                 0.1                 0.0
#                  D                  EG
                 1.4                 4.2
#                 Gr                   C               alpha
                 2.5                 1.8                 0.7
#F_id_b(P)                       SCALE_B             SCALE_P
        12                       1000.00                1.00
#F_id_g(r)                   SCALE_GAMMA           SCALE_RHO                  C1                  C2
        34                          1.20                1.00               250.0                0.85
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors parsing fixed LAW105: {log.errors}"
    assert 1051 in model.mat_law105s
    m1 = model.mat_law105s[1051]
    assert m1.rho0 == pytest.approx(1.8e-09)
    assert m1.bulk == pytest.approx(8500.0)
    assert m1.p0 == pytest.approx(0.1)
    assert m1.gas_d == pytest.approx(1.4)
    assert m1.gas_eg == pytest.approx(4.2)
    assert m1.gr == pytest.approx(2.5)
    assert m1.c == pytest.approx(1.8)
    assert m1.alpha == pytest.approx(0.7)
    assert m1.func_b == 12
    assert m1.scale_b == pytest.approx(1000.0)
    assert m1.func_gam == 34
    assert m1.scale_gam == pytest.approx(1.2)
    assert m1.c1 == pytest.approx(250.0)
    assert m1.c2 == pytest.approx(0.85)
    assert 1051 in model.materials

    # Free format deck
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free POWDER_BURN
/MAT/POWDER_BURN/1052
Powder Burn Propellant Free
1.85e-09
9000.0 0.15 0.05
1.45 4.5
2.8 1.9 0.75
10 1200.0 1.05
20 1.25 1.05 275.0 0.9
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors parsing free POWDER_BURN: {log.errors}"
    assert 1052 in model.mat_law105s
    m2 = model.mat_law105s[1052]
    assert m2.rho0 == pytest.approx(1.85e-09)
    assert m2.bulk == pytest.approx(9000.0)
    assert m2.func_b == 10
    assert m2.scale_b == pytest.approx(1200.0)
    assert m2.func_gam == 20
    assert m2.c1 == pytest.approx(275.0)


def test_mat_law106_jcook_alm_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW106 and /MAT/JCOOK_ALM in fixed and free formats."""
    # Fixed format deck
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed LAW106
/MAT/LAW106/1061
Johnson Cook ALM Fixed
#              RHO_I               RHO_O
             7.9e-09             7.9e-09
#                  E                  nu   fct_ID1   fct_ID2   fct_ID3
            195000.0                 0.3        11        12        13
#                  A                   B                   n              epsmax              sigmax
               550.0               800.0                0.45                 0.4              1500.0
#               Fcut        VP      Nmax                 Tol                   C               deps0
             15000.0         2         4               1e-06               0.025               0.001
#                                                          m               Tmelt
                                                         1.1              1650.0
#             RHo_Cp                 Eta                  T0                  Tr
               500.0                 0.9               293.0               293.0
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors parsing fixed LAW106: {log.errors}"
    assert 1061 in model.mat_law106s
    m1 = model.mat_law106s[1061]
    assert m1.rho0 == pytest.approx(7.9e-09)
    assert m1.rhor == pytest.approx(7.9e-09)
    assert m1.young == pytest.approx(195000.0)
    assert m1.nu == pytest.approx(0.3)
    assert m1.fct_id1 == 11
    assert m1.fct_id2 == 12
    assert m1.fct_id3 == 13
    assert m1.sigy == pytest.approx(550.0)
    assert m1.beta == pytest.approx(800.0)
    assert m1.hard_n == pytest.approx(0.45)
    assert m1.ep_max == pytest.approx(0.4)
    assert m1.sig_max == pytest.approx(1500.0)
    assert m1.fcut == pytest.approx(15000.0)
    assert m1.vp == 2
    assert m1.nmax == 4
    assert m1.tol == pytest.approx(1e-06)
    assert m1.cjc == pytest.approx(0.025)
    assert m1.deps0 == pytest.approx(0.001)
    assert m1.m == pytest.approx(1.1)
    assert m1.tmelt == pytest.approx(1650.0)
    assert m1.spheat == pytest.approx(500.0)
    assert m1.eta == pytest.approx(0.9)
    assert 1061 in model.materials

    # Free format deck
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free JCOOK_ALM
/MAT/JCOOK_ALM/1062
Johnson Cook ALM Free
8.0e-09 8.0e-09
200000.0 0.28 1 2 3
600.0 850.0 0.42 0.35 1600.0
20000.0 1 5 1e-7 0.03 0.002
1.15 1700.0
520.0 0.95 300.0 300.0
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors parsing free JCOOK_ALM: {log.errors}"
    assert 1062 in model.mat_law106s
    m2 = model.mat_law106s[1062]
    assert m2.rho0 == pytest.approx(8.0e-09)
    assert m2.young == pytest.approx(200000.0)
    assert m2.vp == 1
    assert m2.nmax == 5
    assert m2.cjc == pytest.approx(0.03)


def test_mat_law107_paper_light_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW107 and /MAT/PAPER_LIGHT in fixed and free formats."""
    # Fixed format deck (Analytic, Itab=0)
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed LAW107
/MAT/LAW107/1071
Paper Light Material Fixed
#              RHO_I               RHO_O
             0.8e-09             0.8e-09
#                 E1                  E2                  E3      Ires      Itab   Ismooth
              7500.0              3500.0               100.0         1         0         0
#               nu21                 G12                 G23                 G13
                0.35              2000.0                80.0                90.0
#                XI1                 XI2                 g1c                  d1                 d2
                 0.1                 0.2                 1.5                 0.5                0.05
#                 k1                  k2                  k3
                 1.1                 1.2                 1.3
#                 k4                  k5                  k6
                 1.4                 1.5                 1.6
#              SIGY1               CINI1                  S1
                40.0                 5.0               250.0
#              SIGY2               CINI2                  S2
                25.0                 3.0               150.0
#             SIGY1C              CINI1C                 S1C
                20.0                 2.0               120.0
#             SIGY2C              CINI2C                 S2C
                15.0                 1.5                90.0
#              SIGYT               CINIT                  ST
                18.0                 2.5               110.0
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors parsing fixed LAW107: {log.errors}"
    assert 1071 in model.mat_law107s
    m1 = model.mat_law107s[1071]
    assert m1.rho0 == pytest.approx(0.8e-09)
    assert m1.e1 == pytest.approx(7500.0)
    assert m1.e2 == pytest.approx(3500.0)
    assert m1.e3 == pytest.approx(100.0)
    assert m1.nu21 == pytest.approx(0.35)
    assert m1.g12 == pytest.approx(2000.0)
    assert m1.xi1 == pytest.approx(0.1)
    assert m1.k1 == pytest.approx(1.1)
    assert m1.k6 == pytest.approx(1.6)
    assert m1.sigy1 == pytest.approx(40.0)
    assert m1.cini1 == pytest.approx(5.0)
    assert m1.s1 == pytest.approx(250.0)
    assert m1.sigyt == pytest.approx(18.0)
    assert 1071 in model.materials

    # Free format deck (Tabulated, Itab=1)
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free PAPER_LIGHT
/MAT/PAPER_LIGHT/1072
Paper Light Tabulated Free
0.85e-09 0.85e-09
8000.0 4000.0 120.0 1 1 0
0.34 2200.0 90.0 100.0
0.12 0.22 1.6 0.55 0.06
1.0 1.1 1.2
1.3 1.4 1.5
101 1.0 1.0
102 1.0 1.0
103 1.0 1.0
104 1.0 1.0
105 1.0 1.0
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors parsing free PAPER_LIGHT: {log.errors}"
    assert 1072 in model.mat_law107s
    m2 = model.mat_law107s[1072]
    assert m2.itab == 1
    assert m2.tab_yld1 == 101
    assert m2.tab_yld2 == 102
    assert m2.tab_yld1c == 103
    assert m2.tab_yld2c == 104
    assert m2.tab_yldt == 105


def test_mat_law110_vegter_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW110 and /MAT/VEGTER in fixed and free formats."""
    # Fixed format deck (Icrit=3: 0/45/90 reference tests)
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed LAW110
/MAT/LAW110/1101
Vegter Anisotropic Fixed
#              RHO_I               RHO_O
             7.8e-09             7.8e-09
#                  E                  nu      Ires
            210000.0                 0.3         1
#    Icrit   TAB_YLD          MAT_Xscale          MAT_Yscale                 fBI               rhoBI
         3         0                1.00                1.00                1.05                0.95
#               SIGMA_r               DSIGM                BETA               OMEGA                   n
                  280.0                50.0                 8.0                 0.2                0.22
#               EPS0                SIGS                 DG0               Deps0                   m
                  0.002               450.0               1e-19                 1.0                0.01
#               TINI              C_HARD               F_CUT        VP   Ismooth  TAB_TEMP
               293.0                 0.0             10000.0         0         0         0
#               RM_0               RM_45               RM_90                AG_0               AG_45
               350.0               340.0               360.0                18.0                22.0
#              AG_90                 R_0                R_45                R_90
                19.0                1.85                1.35                2.15
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors parsing fixed LAW110: {log.errors}"
    assert 1101 in model.mat_law110s
    m1 = model.mat_law110s[1101]
    assert m1.rho0 == pytest.approx(7.8e-09)
    assert m1.young == pytest.approx(210000.0)
    assert m1.icrit == 3
    assert m1.fbi == pytest.approx(1.05)
    assert m1.rhobi == pytest.approx(0.95)
    assert m1.sigma_r == pytest.approx(280.0)
    assert m1.dsigm == pytest.approx(50.0)
    assert m1.beta == pytest.approx(8.0)
    assert m1.omega == pytest.approx(0.2)
    assert m1.hard_n == pytest.approx(0.22)
    assert m1.rm_0 == pytest.approx(350.0)
    assert m1.rm_45 == pytest.approx(340.0)
    assert m1.rm_90 == pytest.approx(360.0)
    assert m1.ag_0 == pytest.approx(18.0)
    assert m1.ag_45 == pytest.approx(22.0)
    assert m1.ag_90 == pytest.approx(19.0)
    assert m1.r_0 == pytest.approx(1.85)
    assert m1.r_45 == pytest.approx(1.35)
    assert m1.r_90 == pytest.approx(2.15)
    assert 1101 in model.materials

    # Free format deck (Icrit=1: angle list)
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free VEGTER
/MAT/VEGTER/1102
Vegter Anisotropic Free
7.85e-09 7.85e-09
205000.0 0.29 0
1 0 1.0 1.0 1.02 0.98
300.0 45.0 7.5 0.18 0.24
0.002 480.0 1e-19 1.0 0.015
293.0 0.0 10000.0 0 0 0
1.0 1.8 1.15 1.12 0.58
0.98 1.3 1.12 1.10 0.56
1.04 2.1 1.18 1.16 0.60
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors parsing free VEGTER: {log.errors}"
    assert 1102 in model.mat_law110s
    m2 = model.mat_law110s[1102]
    assert m2.icrit == 1
    assert len(m2.angles_data) == 3
    assert m2.angles_data[0][0] == pytest.approx(1.0)
    assert m2.angles_data[0][1] == pytest.approx(1.8)


def test_mat_law115_deshfleck_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW115 and /MAT/DESHPANDE_FLECK in fixed and free formats."""
    # Fixed format deck (Istat=0: constant properties)
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed LAW115
/MAT/LAW115/1151
Deshpande Fleck Foam Fixed
#              RHO_I
             0.3e-09
#                  E                  nu      Ires     Istat
               800.0                0.25         2         0
#              ALPHA             EPSVP_F              SIGP_F
                0.35                 0.6                 5.0
#               SIGP               GAMMA                EPSD              ALPHA2                BETA
                 2.5               120.0                 0.7                15.0                 3.2
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors parsing fixed LAW115: {log.errors}"
    assert 1151 in model.mat_law115s
    m1 = model.mat_law115s[1151]
    assert m1.rho0 == pytest.approx(0.3e-09)
    assert m1.young == pytest.approx(800.0)
    assert m1.nu == pytest.approx(0.25)
    assert m1.ires == 2
    assert m1.istat == 0
    assert m1.alpha == pytest.approx(0.35)
    assert m1.cfail == pytest.approx(0.6)
    assert m1.pfail == pytest.approx(5.0)
    assert m1.sigp == pytest.approx(2.5)
    assert m1.gamma == pytest.approx(120.0)
    assert m1.epsd == pytest.approx(0.7)
    assert m1.alpha2 == pytest.approx(15.0)
    assert m1.beta == pytest.approx(3.2)
    assert 1151 in model.materials

    # Free format deck (Istat=1: statistical variation)
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free DESHPANDE_FLECK
/MAT/DESHPANDE_FLECK/1152
Deshpande Fleck Statistical Free
0.35e-09
850.0 0.22 2 1
0.4 0.65 5.5 2.7e-09
1.8 0.9 1.5
12.0 4.0 1.8
100.0 30.0 1.2
2.5 0.8 1.1
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors parsing free DESHPANDE_FLECK: {log.errors}"
    assert 1152 in model.mat_law115s
    m2 = model.mat_law115s[1152]
    assert m2.istat == 1
    assert m2.rhof0 == pytest.approx(2.7e-09)
    assert m2.sigp_c0 == pytest.approx(1.8)
    assert m2.sigp_c1 == pytest.approx(0.9)
    assert m2.sigp_n == pytest.approx(1.5)
    assert m2.alpha2_c0 == pytest.approx(12.0)
    assert m2.gamma_c0 == pytest.approx(100.0)
    assert m2.beta_c0 == pytest.approx(2.5)
