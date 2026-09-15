# pyradioss - Python port of OpenRadioss explicit FEM solver
# Milestone M177: Advanced Marlow Hyperelastic, Thermo-Viscoplastic Tabular, Full Anisotropic Paper Plasticity, Cohesive Hysteresis, Modified Ladevèze Delamination & Nonlinear Fabric Material Models Suite
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


def test_mat_law109_thermo_viscoplast_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW109 in fixed and free formats."""
    # Fixed format deck
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed LAW109
/MAT/LAW109/1091
Thermo-Viscoplastic Tabular Fixed
#              RHO_I
             7.8e-09
#                  E                  Nu
            210000.0                 0.3
#                C_p                 ETA               T_ref               T_ini
               450.0                 0.9               293.0               300.0
# tab_ID_h  tab_ID_t            Xscale_h            Yscale_h                                I_smooth
        11        12                1.05                1.02                                       2
#  TAB_ETA          Xscale_ETA
        13                1.00
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors parsing fixed LAW109: {log.errors}"
    assert 1091 in model.mat_law109s
    m1 = model.mat_law109s[1091]
    assert m1.rho0 == pytest.approx(7.8e-09)
    assert m1.young == pytest.approx(210000.0)
    assert m1.nu == pytest.approx(0.3)
    assert m1.cp == pytest.approx(450.0)
    assert m1.eta == pytest.approx(0.9)
    assert m1.tref == pytest.approx(293.0)
    assert m1.tini == pytest.approx(300.0)
    assert m1.tab_yld == 11
    assert m1.tab_temp == 12
    assert m1.xscale_h == pytest.approx(1.05)
    assert m1.yscale_h == pytest.approx(1.02)
    assert m1.ismooth == 2
    assert m1.tab_eta == 13
    assert m1.xscale_eta == pytest.approx(1.0)
    assert 1091 in model.materials
    mat = model.materials[1091]
    assert mat.law == 109
    assert mat.record.params["MAT_TAB_YLD"] == 11

    # Free format deck
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free LAW109
/MAT/LAW109/1092
Thermo-Viscoplastic Tabular Free
7.85e-09
205000.0 0.29
480.0 0.95 295.0 310.0
21 22 1.10 1.05 1
23 1.02
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors parsing free LAW109: {log.errors}"
    assert 1092 in model.mat_law109s
    m2 = model.mat_law109s[1092]
    assert m2.rho0 == pytest.approx(7.85e-09)
    assert m2.young == pytest.approx(205000.0)
    assert m2.tab_yld == 21
    assert m2.tab_temp == 22
    assert m2.xscale_h == pytest.approx(1.10)
    assert m2.tab_eta == 23


def test_mat_law111_marlow_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW111 and /MAT/MARLOW in fixed and free formats."""
    # Fixed format deck
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed LAW111
/MAT/LAW111/1111
Marlow Hyperelastic Fixed
#              Rho_i
             1.1e-09
#    Itype   Func_ID              Fscale                  Nu
         1        55                1.25               0.495
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors parsing fixed LAW111: {log.errors}"
    assert 1111 in model.mat_law111s
    m1 = model.mat_law111s[1111]
    assert m1.rho0 == pytest.approx(1.1e-09)
    assert m1.itype == 1
    assert m1.fct_id == 55
    assert m1.fscale == pytest.approx(1.25)
    assert m1.nu == pytest.approx(0.495)
    assert 1111 in model.materials

    # Free format deck
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free MARLOW
/MAT/MARLOW/1112
Marlow Hyperelastic Free
1.15e-09
2 60 1.50 0.49
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors parsing free MARLOW: {log.errors}"
    assert 1112 in model.mat_law111s
    m2 = model.mat_law111s[1112]
    assert m2.rho0 == pytest.approx(1.15e-09)
    assert m2.itype == 2
    assert m2.fct_id == 60
    assert m2.fscale == pytest.approx(1.50)
    assert m2.nu == pytest.approx(0.49)


def test_mat_law112_paper_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW112 and /MAT/PAPER in fixed and free formats."""
    # Fixed format deck (Analytic, Itab=0)
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed LAW112
/MAT/LAW112/1121
Paper Material Fixed
#              RHO_I               RHO_O
             0.8e-09             0.8e-09
#                 E1                  E2                  E3      Ires      Itab   Ismooth
              8000.0              4000.0               120.0         1         0         0
#               nu21                 G12                 G23                 G13
                0.35              2200.0                90.0               100.0
#                  K                 E3C                  CC
                 2.0               150.0                 0.5
#               nu1p                nu2p                nu4p                nu5p
                0.30                0.20                0.25                0.15
#                S01                 A01                 B01                 C01
                45.0                 6.0               280.0                10.0
#                S02                 A02                 B02                 C02
                30.0                 4.0               180.0                 8.0
#                S03                 A03                 B03                 C03
                25.0                 3.0               150.0                 6.0
#                S04                 A04                 B04                 C04
                40.0                 5.0               250.0                 9.0
#                S05                 A05                 B05                 C05
                28.0                 3.5               160.0                 7.0
#               ASIG                BSIG                CSIG
                50.0                20.0                 0.8
#               TAU0                ATAU                BTAU
                15.0                 5.0                 0.6
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors parsing fixed LAW112: {log.errors}"
    assert 1121 in model.mat_law112s
    m1 = model.mat_law112s[1121]
    assert m1.rho0 == pytest.approx(0.8e-09)
    assert m1.e1 == pytest.approx(8000.0)
    assert m1.e2 == pytest.approx(4000.0)
    assert m1.e3 == pytest.approx(120.0)
    assert m1.nu21 == pytest.approx(0.35)
    assert m1.g12 == pytest.approx(2200.0)
    assert m1.k == pytest.approx(2.0)
    assert m1.e3c == pytest.approx(150.0)
    assert m1.cc == pytest.approx(0.5)
    assert m1.s01 == pytest.approx(45.0)
    assert m1.a01 == pytest.approx(6.0)
    assert m1.b01 == pytest.approx(280.0)
    assert m1.c01 == pytest.approx(10.0)
    assert m1.asig == pytest.approx(50.0)
    assert m1.tau0 == pytest.approx(15.0)
    assert 1121 in model.materials

    # Free format deck (Tabulated, Itab=1)
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free PAPER
/MAT/PAPER/1122
Paper Material Tabulated Free
0.85e-09 0.85e-09
8500.0 4200.0 130.0 1 1 0
0.34 2300.0 95.0 105.0
2.1 160.0 0.55
0.32 0.22 0.26 0.16
101 1.0 1.0
102 1.0 1.0
103 1.0 1.0
104 1.0 1.0
105 1.0 1.0
106 1.0 1.0
107 1.0 1.0
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors parsing free PAPER: {log.errors}"
    assert 1122 in model.mat_law112s
    m2 = model.mat_law112s[1122]
    assert m2.itab == 1
    assert m2.tab_yld1 == 101
    assert m2.tab_yld2 == 102
    assert m2.tab_yld3 == 103
    assert m2.tab_yld4 == 104
    assert m2.tab_yld5 == 105
    assert m2.tab_yldc == 106
    assert m2.tab_ylds == 107


def test_mat_law116_coh_hyst_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW116 and /MAT/COH_HYST in fixed and free formats."""
    # Fixed format deck
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed LAW116
/MAT/LAW116/1161
Cohesive Hysteresis Fixed
#        Init. dens.
             1.2e-09
#                  E                  G             Thick     Imass     Idel     Icrit
              5000.0             2000.0              0.05         1        1         1
#       MAT_GC1_ini         MAT_GC1_inf           MAT_SRATG1            MAT_FG1
                 0.5                 1.2                 10.0                0.8
#       MAT_GC2_ini         MAT_GC2_inf           MAT_SRATG2            MAT_FG2
                 0.8                 2.0                 15.0                0.7
#         MAT_SIGA1           MAT_SIGB1           MAT_SRATE1 MAT_ORDER1 MAT_FAIL1
                30.0                45.0                100.0          1         1
#         MAT_SIGA2           MAT_SIGB2           MAT_SRATE2 MAT_ORDER2 MAT_FAIL2
                20.0                35.0                120.0          1         1
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors parsing fixed LAW116: {log.errors}"
    assert 1161 in model.mat_law116s
    m1 = model.mat_law116s[1161]
    assert m1.rho0 == pytest.approx(1.2e-09)
    assert m1.young == pytest.approx(5000.0)
    assert m1.g == pytest.approx(2000.0)
    assert m1.thick == pytest.approx(0.05)
    assert m1.imass == 1
    assert m1.gc1_ini == pytest.approx(0.5)
    assert m1.gc1_inf == pytest.approx(1.2)
    assert m1.sratg1 == pytest.approx(10.0)
    assert m1.fg1 == pytest.approx(0.8)
    assert m1.gc2_ini == pytest.approx(0.8)
    assert m1.gc2_inf == pytest.approx(2.0)
    assert m1.siga1 == pytest.approx(30.0)
    assert m1.sigb1 == pytest.approx(45.0)
    assert m1.srate1 == pytest.approx(100.0)
    assert 1161 in model.materials

    # Free format deck
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free COH_HYST
/MAT/COH_HYST/1162
Cohesive Hysteresis Free
1.25e-09
5500.0 2200.0 0.06 1 1 1
0.6 1.4 12.0 0.85
0.9 2.2 18.0 0.75
35.0 50.0 110.0 1 1
25.0 40.0 130.0 1 1
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors parsing free COH_HYST: {log.errors}"
    assert 1162 in model.mat_law116s
    m2 = model.mat_law116s[1162]
    assert m2.rho0 == pytest.approx(1.25e-09)
    assert m2.young == pytest.approx(5500.0)
    assert m2.g == pytest.approx(2200.0)
    assert m2.thick == pytest.approx(0.06)
    assert m2.gc1_ini == pytest.approx(0.6)
    assert m2.siga1 == pytest.approx(35.0)


def test_mat_law122_modified_ladeveze_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW122 and /MAT/MODIFIED_LADEVEZE in fixed and free formats."""
    # Fixed format deck
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed LAW122
/MAT/LAW122/1221
Modified Ladeveze Fixed
#        Init. dens.
             1.5e-09
#                 E1                  E2                  E3                 G12                 G23
            120000.0             10000.0             10000.0              5000.0              3500.0
#                G31                NU12                NU23                NU31
              5000.0                 0.3                 0.4                0.03
#                E1C               GAMMA                 ISH                 ITR                IRES
            110000.0                 0.1                   0                   0                   1
#              SIGY0                BETA                   M                   A
               200.0               800.0                 0.5                 1.2
#            EPS_FTI             EPS_FTU                DFTU
               0.015               0.025                 0.8
#            EPS_FCI             EPS_FCU                DCFU               IBUCK
               0.012               0.020                 0.7                   0
#            IFUNCD1               DSAT1                  Y0                  YC                   B
                  10                 0.9                 0.2                 1.5                 0.5
#               DMAX                  YR                 YSP
                 0.95                0.1                 2.0
#            IFUNCD2               DSAT2                 Y0P                 YCP
                  11                 0.8                 0.3                 1.8
#           IFUNCD2C              DSAT2C                Y0PC                YCPC
                  12                 0.7                 0.4                 2.2
#             EPSD11                 D11                 N11                D11U                N11U
              0.0001                0.02                 0.1                0.03                 0.2
#             EPSD12                 D22                 N22                 D12                 N12
              0.0002                0.04                 0.1                0.05                 0.2
#             EPSDR0                 DR0                 NR0             LTYPE11   LTYPE12   LTYPER0
              0.0001                0.02                 0.1                   0         0         0
#               FCUT
             10000.0
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors parsing fixed LAW122: {log.errors}"
    assert 1221 in model.mat_law122s
    m1 = model.mat_law122s[1221]
    assert m1.rho0 == pytest.approx(1.5e-09)
    assert m1.e1 == pytest.approx(120000.0)
    assert m1.e2 == pytest.approx(10000.0)
    assert m1.e3 == pytest.approx(10000.0)
    assert m1.g12 == pytest.approx(5000.0)
    assert m1.g23 == pytest.approx(3500.0)
    assert m1.g31 == pytest.approx(5000.0)
    assert m1.nu12 == pytest.approx(0.3)
    assert m1.e1c == pytest.approx(110000.0)
    assert m1.gamma == pytest.approx(0.1)
    assert m1.ires == 1
    assert m1.sigy0 == pytest.approx(200.0)
    assert m1.beta == pytest.approx(800.0)
    assert m1.hard_m == pytest.approx(0.5)
    assert m1.hard_a == pytest.approx(1.2)
    assert m1.eps_fti == pytest.approx(0.015)
    assert m1.eps_ftu == pytest.approx(0.025)
    assert m1.dftu == pytest.approx(0.8)
    assert m1.ifuncd1 == 10
    assert m1.dsat1 == pytest.approx(0.9)
    assert m1.y0 == pytest.approx(0.2)
    assert m1.yc == pytest.approx(1.5)
    assert m1.b == pytest.approx(0.5)
    assert m1.dmax == pytest.approx(0.95)
    assert m1.fcut == pytest.approx(10000.0)
    assert 1221 in model.materials

    # Free format deck
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free MODIFIED_LADEVEZE
/MAT/MODIFIED_LADEVEZE/1222
Modified Ladeveze Free
1.55e-09
125000.0 10500.0 10500.0 5200.0 3600.0
5200.0 0.29 0.39 0.029
115000.0 0.12 0 0 1
210.0 850.0 0.52 1.25
0.016 0.026 0.82
0.013 0.021 0.72 0
20 0.92 0.22 1.55 0.52
0.96 0.11 2.1
21 0.82 0.32 1.85
22 0.72 0.42 2.25
0.0001 0.02 0.1 0.03 0.2
0.0002 0.04 0.1 0.05 0.2
0.0001 0.02 0.1 0 0 0
12000.0
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors parsing free MODIFIED_LADEVEZE: {log.errors}"
    assert 1222 in model.mat_law122s
    m2 = model.mat_law122s[1222]
    assert m2.rho0 == pytest.approx(1.55e-09)
    assert m2.e1 == pytest.approx(125000.0)
    assert m2.sigy0 == pytest.approx(210.0)
    assert m2.ifuncd1 == 20
    assert m2.ifuncd2 == 21
    assert m2.ifuncd2c == 22
    assert m2.fcut == pytest.approx(12000.0)


def test_mat_law158_fabric_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW158 and /MAT/FABR_NL in fixed and free formats."""
    # Fixed format deck
    fixed_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Fixed LAW158
/MAT/LAW158/1581
Nonlinear Fabric Fixed
#              RHO_I
             0.5e-09
#       S1                  S2                FLEX               FLEX1               FLEX2
                 0.1                 0.1                50.0                10.0                10.0
#         ZERO_STRESS           sensor_ID
                 0.0                   1
#  FCT_ID1                       Fscale1
        10                          1.00
#  FCT_ID2                       Fscale2
        11                          1.00
#  FCT_ID3                       Fscale3
        12                          1.00
#  FCT_ID4   FCT_ID5
        13        14
/END
"""
    model, log = _parse_deck(tmp_path, fixed_deck)
    assert not log.errors, f"Errors parsing fixed LAW158: {log.errors}"
    assert 1581 in model.mat_law158s
    m1 = model.mat_law158s[1581]
    assert m1.rho0 == pytest.approx(0.5e-09)
    assert m1.s1 == pytest.approx(0.1)
    assert m1.s2 == pytest.approx(0.1)
    assert m1.flex == pytest.approx(50.0)
    assert m1.flex1 == pytest.approx(10.0)
    assert m1.flex2 == pytest.approx(10.0)
    assert m1.zerostress == pytest.approx(0.0)
    assert m1.sensor_id == 1
    assert m1.fun_a1 == 10
    assert m1.c1 == pytest.approx(1.0)
    assert m1.fun_a2 == 11
    assert m1.c2 == pytest.approx(1.0)
    assert m1.fun_a3 == 12
    assert m1.c3 == pytest.approx(1.0)
    assert m1.fun_a4 == 13
    assert m1.fun_a5 == 14
    assert 1581 in model.materials

    # Free format deck
    free_deck = """\
# OpenRadioss Starter Deck
/BEGIN
Title Free FABR_NL
/MAT/FABR_NL/1582
Nonlinear Fabric Free
0.55e-09
0.12 0.12 60.0 12.0 12.0
0.0 2
20 1.2
21 1.2
22 1.2
23 24
/END
"""
    model, log = _parse_deck(tmp_path, free_deck)
    assert not log.errors, f"Errors parsing free FABR_NL: {log.errors}"
    assert 1582 in model.mat_law158s
    m2 = model.mat_law158s[1582]
    assert m2.rho0 == pytest.approx(0.55e-09)
    assert m2.s1 == pytest.approx(0.12)
    assert m2.flex == pytest.approx(60.0)
    assert m2.sensor_id == 2
    assert m2.fun_a1 == 20
    assert m2.c1 == pytest.approx(1.2)
    assert m2.fun_a4 == 23
    assert m2.fun_a5 == 24
