"""Unit test suite for Milestone M175: Advanced Anisotropic Viscoplasticity, Thermal Creep, Daimler Damage & Viscous Foam Material Models Suite.

Tests cover:
1. /MAT/LAW128 & /MAT/HILL_VISC_PLAST / /MAT/HILL_VISCO_PLASTIC (Hill anisotropic viscoplastic material)
2. /MAT/LAW129 & /MAT/THERM_CREEP / /MAT/THERMAL_CREEP / /MAT/THERMO_ELASTO_VISCOPLASTIC_CREEP (Thermo-elasto-viscoplastic creep material)
3. /MAT/LAW123 & /MAT/DAIMLER_PINHO / /MAT/DAIMLER-PINHO / /MAT/LAMINATED_FRACTURE_DAIMLER_PINHO (Daimler-Pinho 3D composite damage model)
4. /MAT/LAW132 & /MAT/DAIMLER_CAMANHO / /MAT/DAIMLER-CAMANHO / /MAT/LAMINATED_FRACTURE_DAIMLER_CAMANHO (Daimler-Camanho composite failure model)
5. /MAT/LAW134 & /MAT/VISCOUS_FOAM / /MAT/VISC_FOAM (Viscous foam material)
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


def test_mat_law128_hill_visc_plast_parsing(tmp_path: Path):
    """Test /MAT/LAW128 and /MAT/HILL_VISC_PLAST parsing for Hill anisotropic viscoplasticity."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW128_TEST
/MAT/LAW128/101
Hill Viscoplastic Fixed Format
#              RHO_I
              7.8E-9
#                  E                  NU                SIGY               CHARD
            210000.0                 0.3               300.0                 0.5
#     tab_ID              Fscale              Xscale
          10                 1.0                 1.0
#                QR1                 CR1                 QR2                 CR2
               100.0                10.0                50.0                 5.0
#                QX1                 CX1                 QX2                 CX2
                80.0                 8.0                40.0                 4.0
#              EPSP0                  CP
              0.0001               100.0
#                R00                 R45                 R90
                 1.2                 1.5                 1.8
#                  F                   G                   H
                0.45                0.55                0.65
#                  L                   M                   N
                1.35                1.45                1.55
/MAT/HILL_VISC_PLAST/102
Hill Viscoplastic Free Format
7.85e-9
205000.0 0.29 320.0 0.6
20 1.2 1.1
120.0 12.0 60.0 6.0
90.0 9.0 45.0 4.5
0.0002 120.0
1.1 1.4 1.7
0.4 0.5 0.6
1.3 1.4 1.5
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law128 entity 101
    assert 101 in model.mat_law128s
    m128_1 = model.mat_law128s[101]
    assert m128_1.id == 101
    assert m128_1.title == "Hill Viscoplastic Fixed Format"
    assert pytest.approx(m128_1.rho0) == 7.8e-9
    assert pytest.approx(m128_1.e) == 210000.0
    assert pytest.approx(m128_1.nu) == 0.3
    assert pytest.approx(m128_1.sigy) == 300.0
    assert pytest.approx(m128_1.kin) == 0.5
    assert m128_1.tab_id == 10
    assert pytest.approx(m128_1.facy) == 1.0
    assert pytest.approx(m128_1.facx) == 1.0
    assert pytest.approx(m128_1.qr1) == 100.0
    assert pytest.approx(m128_1.cr1) == 10.0
    assert pytest.approx(m128_1.qr2) == 50.0
    assert pytest.approx(m128_1.cr2) == 5.0
    assert pytest.approx(m128_1.qx1) == 80.0
    assert pytest.approx(m128_1.cx1) == 8.0
    assert pytest.approx(m128_1.qx2) == 40.0
    assert pytest.approx(m128_1.cx2) == 4.0
    assert pytest.approx(m128_1.epsp0) == 0.0001
    assert pytest.approx(m128_1.cp) == 100.0
    assert pytest.approx(m128_1.r00) == 1.2
    assert pytest.approx(m128_1.r45) == 1.5
    assert pytest.approx(m128_1.r90) == 1.8
    assert pytest.approx(m128_1.f) == 0.45
    assert pytest.approx(m128_1.g) == 0.55
    assert pytest.approx(m128_1.h) == 0.65
    assert pytest.approx(m128_1.l) == 1.35
    assert pytest.approx(m128_1.m) == 1.45
    assert pytest.approx(m128_1.n) == 1.55

    # Verify Law128 entity 102
    assert 102 in model.mat_law128s
    m128_2 = model.mat_law128s[102]
    assert m128_2.id == 102
    assert pytest.approx(m128_2.rho0) == 7.85e-9
    assert pytest.approx(m128_2.e) == 205000.0
    assert pytest.approx(m128_2.sigy) == 320.0
    assert m128_2.tab_id == 20
    assert pytest.approx(m128_2.r00) == 1.1

    # Verify model.materials and GenericMaterialRecord
    mat1 = model.materials[101]
    assert mat1.law == 128
    assert pytest.approx(mat1.rho0) == 7.8e-9
    assert pytest.approx(mat1.params["E"]) == 210000.0
    assert mat1.record is not None
    assert mat1.record.law_name == "LAW128"
    assert mat1.record.law_number == 128


def test_mat_law129_therm_creep_parsing(tmp_path: Path):
    """Test /MAT/LAW129 and /MAT/THERM_CREEP parsing for thermo-elasto-viscoplastic creep."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW129_TEST
/MAT/LAW129/201
Thermo Creep Fixed Format
#              RHO_I
              7.9E-9
#                  E                  Nu                Sig0               alpha                Tref
            200000.0                 0.3               250.0              1.2E-5               293.0
#     Fct_E     Fct_Nu    Fct_YLD  Fct_alpha                                                   ISENSOR
          1          2          3          4                                                         5
#    Tab_ID               Fscale
         15                  1.0
#                QR1                 CR1                 QR2                 CR2      F_QR      F_CR
                60.0                 6.0                30.0                 3.0         6         7
#                QX1                 CX1                 QX2                 CX2      F_QX      F_CX
                50.0                 5.0                25.0                 2.5         8         9
#              EPSP0                  CP      F_CC      F_CP
              0.0005                50.0        11        12
#               CRPA                CRPN                CRPM  Fct_CPRA  Fct_CPRN  Fct_CPRM   CRP_Law
               1e-15                 4.5                 0.0        13        14        15         1
#              CRSIG                 CRT                CRPQ                EPS0  Fct_CPRQ Fct_CRSIG
               100.0               300.0             15000.0               1e-05        16        17
/MAT/THERM_CREEP/202
Thermo Creep Free Format
8.0e-9
195000.0 0.31 240.0 1.3e-5 300.0
21 22 23 24 0
25 1.5
70.0 7.0 35.0 3.5 26 27
55.0 5.5 27.5 2.75 28 29
0.0004 45.0 30 31
2e-15 4.8 0.1 32 33 34 2
110.0 310.0 16000.0 2e-5 35 36
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law129 entity 201
    assert 201 in model.mat_law129s
    m129_1 = model.mat_law129s[201]
    assert m129_1.id == 201
    assert m129_1.title == "Thermo Creep Fixed Format"
    assert pytest.approx(m129_1.rho0) == 7.9e-9
    assert pytest.approx(m129_1.e) == 200000.0
    assert pytest.approx(m129_1.nu) == 0.3
    assert pytest.approx(m129_1.sigy) == 250.0
    assert pytest.approx(m129_1.alpha) == 1.2e-5
    assert pytest.approx(m129_1.tref) == 293.0
    assert m129_1.f_young == 1
    assert m129_1.f_nu == 2
    assert m129_1.f_yld == 3
    assert m129_1.f_alpha == 4
    assert m129_1.isensor == 5
    assert m129_1.itab == 15
    assert pytest.approx(m129_1.facy) == 1.0
    assert pytest.approx(m129_1.qr1) == 60.0
    assert pytest.approx(m129_1.cr1) == 6.0
    assert pytest.approx(m129_1.qr2) == 30.0
    assert pytest.approx(m129_1.cr2) == 3.0
    assert m129_1.f_qr == 6
    assert m129_1.f_cr == 7
    assert pytest.approx(m129_1.qx1) == 50.0
    assert pytest.approx(m129_1.cx1) == 5.0
    assert pytest.approx(m129_1.qx2) == 25.0
    assert pytest.approx(m129_1.cx2) == 2.5
    assert m129_1.f_qx == 8
    assert m129_1.f_cx == 9
    assert pytest.approx(m129_1.epsp0) == 0.0005
    assert pytest.approx(m129_1.cp) == 50.0
    assert m129_1.f_cc == 11
    assert m129_1.f_cp == 12
    assert pytest.approx(m129_1.crpa) == 1e-15
    assert pytest.approx(m129_1.crpn) == 4.5
    assert pytest.approx(m129_1.crpm) == 0.0
    assert m129_1.f_a == 13
    assert m129_1.f_n == 14
    assert m129_1.f_m == 15
    assert m129_1.crp_law == 1
    assert pytest.approx(m129_1.crsig) == 100.0
    assert pytest.approx(m129_1.crt) == 300.0
    assert pytest.approx(m129_1.crpq) == 15000.0
    assert pytest.approx(m129_1.eps0) == 1e-5
    assert m129_1.f_q == 16
    assert m129_1.f_sig == 17

    # Verify Law129 entity 202
    assert 202 in model.mat_law129s
    m129_2 = model.mat_law129s[202]
    assert m129_2.id == 202
    assert pytest.approx(m129_2.e) == 195000.0
    assert m129_2.crp_law == 2

    # Verify model.materials and GenericMaterialRecord
    mat1 = model.materials[201]
    assert mat1.law == 129
    assert pytest.approx(mat1.rho0) == 7.9e-9
    assert pytest.approx(mat1.params["E"]) == 200000.0
    assert mat1.record is not None
    assert mat1.record.law_name == "LAW129"
    assert mat1.record.law_number == 129


def test_mat_law123_daimler_pinho_parsing(tmp_path: Path):
    """Test /MAT/LAW123 and /MAT/DAIMLER_PINHO parsing for Daimler-Pinho 3D composite damage."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW123_TEST
/MAT/LAW123/301
Daimler-Pinho Fixed Format
#                Rho
              1.6E-9
#                 E1                  E2                  E3
            140000.0             10000.0             10000.0
#                G12                 G13                 G23
              5000.0              5000.0              3500.0
#               Nu12                Nu31                Nu32
                0.32                0.02                0.45
#             ENKINK                 ENA                 ENB                 ENT                 ENL
                 1.1                 2.2                 3.3                 4.4                 5.5
#                 XC                  XT                  YC                  YT                  SL
              1200.0              2200.0               200.0                60.0                90.0
#                FIO                SIGY      LCSS                        BETA
                53.0               100.0       101                         0.1
#                EFS               RATIO                FCUT
                0.05                 0.8                 0.0
/MAT/DAIMLER_PINHO/302
Daimler-Pinho Free Format
1.55e-9
135000.0 9500.0 9500.0
4800.0 4800.0 3300.0
0.3 0.025 0.42
1.2 2.4 3.6 4.8 6.0
1100.0 2100.0 190.0 55.0 85.0
50.0 95.0 102 0.12
0.06 0.85 10.0
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law123 entity 301
    assert 301 in model.mat_law123s
    m123_1 = model.mat_law123s[301]
    assert m123_1.id == 301
    assert m123_1.title == "Daimler-Pinho Fixed Format"
    assert pytest.approx(m123_1.rho0) == 1.6e-9
    assert pytest.approx(m123_1.ea) == 140000.0
    assert pytest.approx(m123_1.eb) == 10000.0
    assert pytest.approx(m123_1.ec) == 10000.0
    assert pytest.approx(m123_1.gab) == 5000.0
    assert pytest.approx(m123_1.gca) == 5000.0
    assert pytest.approx(m123_1.gbc) == 3500.0
    assert pytest.approx(m123_1.prba) == 0.32
    assert pytest.approx(m123_1.prca) == 0.02
    assert pytest.approx(m123_1.prcb) == 0.45
    assert pytest.approx(m123_1.enkink) == 1.1
    assert pytest.approx(m123_1.ena) == 2.2
    assert pytest.approx(m123_1.enb) == 3.3
    assert pytest.approx(m123_1.ent) == 4.4
    assert pytest.approx(m123_1.enl) == 5.5
    assert pytest.approx(m123_1.xc) == 1200.0
    assert pytest.approx(m123_1.xt) == 2200.0
    assert pytest.approx(m123_1.yc) == 200.0
    assert pytest.approx(m123_1.yt) == 60.0
    assert pytest.approx(m123_1.sl) == 90.0
    assert pytest.approx(m123_1.fio) == 53.0
    assert pytest.approx(m123_1.sigy) == 100.0
    assert m123_1.lcss == 101
    assert pytest.approx(m123_1.beta) == 0.1
    assert pytest.approx(m123_1.efs) == 0.05
    assert pytest.approx(m123_1.ratio) == 0.8
    assert pytest.approx(m123_1.fcut) == 0.0

    # Verify Law123 entity 302
    assert 302 in model.mat_law123s
    m123_2 = model.mat_law123s[302]
    assert m123_2.id == 302
    assert pytest.approx(m123_2.ea) == 135000.0
    assert pytest.approx(m123_2.fio) == 50.0

    # Verify model.materials and GenericMaterialRecord
    mat1 = model.materials[301]
    assert mat1.law == 123
    assert pytest.approx(mat1.rho0) == 1.6e-9
    assert pytest.approx(mat1.params["E"]) == 140000.0
    assert mat1.record is not None
    assert mat1.record.law_name == "LAW123"
    assert mat1.record.law_number == 123


def test_mat_law132_daimler_camanho_parsing(tmp_path: Path):
    """Test /MAT/LAW132 and /MAT/DAIMLER_CAMANHO parsing for Daimler-Camanho composite damage."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW132_TEST
/MAT/LAW132/401
Daimler-Camanho Fixed Format
#                Rho
              1.5E-9
#                 E1                  E2                  E3
            150000.0              9000.0              9000.0
#                G12                 G13                 G23
              4500.0              4500.0              3000.0
#               Nu12                Nu31                Nu32
                0.30                0.03                0.40
#                GXC                 GXT                 GYC                 GYT                 GSL
                50.0                80.0                 1.5                 0.5                 2.0
#                 XC                  XT                  YC                  YT                  SL
              1300.0              2400.0               220.0                70.0                95.0
#               GXC0                GXT0                 XC0                 XT0
                40.0                70.0              1100.0              2000.0
#                FIO                SIGY                ETAN                BETA      LCSS
                53.0               110.0              1000.0                 0.2       201
#             EPSF23              EPSR23              TSMD23
                0.08                0.12                0.95
#             EPSF31              EPSR31              TSMD31
                0.07                0.11                0.90
#              EF11T               EF11C               EF22T               EF22c                EF12
                0.02                0.01                0.03                0.02                0.04
#               EF23                EF31                CF12                CF23                CF31
                0.05                0.04                 0.1                 0.3                 0.2
#          LRD_RATIO                FCUT
                 0.9                 0.0
/MAT/DAIMLER_CAMANHO/402
Daimler-Camanho Free Format
1.52e-9
145000.0 8800.0 8800.0
4400.0 4400.0 2900.0
0.29 0.032 0.39
48.0 78.0 1.4 0.45 1.9
1250.0 2350.0 210.0 65.0 90.0
38.0 68.0 1050.0 1950.0
52.0 105.0 950.0 0.18 202
0.075 0.115 0.92
0.065 0.105 0.88
0.019 0.009 0.029 0.019 0.039
0.049 0.039 0.12 0.32 0.22
0.88 5.0
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law132 entity 401
    assert 401 in model.mat_law132s
    m132_1 = model.mat_law132s[401]
    assert m132_1.id == 401
    assert m132_1.title == "Daimler-Camanho Fixed Format"
    assert pytest.approx(m132_1.rho0) == 1.5e-9
    assert pytest.approx(m132_1.ea) == 150000.0
    assert pytest.approx(m132_1.eb) == 9000.0
    assert pytest.approx(m132_1.ec) == 9000.0
    assert pytest.approx(m132_1.gab) == 4500.0
    assert pytest.approx(m132_1.gca) == 4500.0
    assert pytest.approx(m132_1.gbc) == 3000.0
    assert pytest.approx(m132_1.prba) == 0.30
    assert pytest.approx(m132_1.prca) == 0.03
    assert pytest.approx(m132_1.prcb) == 0.40
    assert pytest.approx(m132_1.gxc) == 50.0
    assert pytest.approx(m132_1.gxt) == 80.0
    assert pytest.approx(m132_1.gyc) == 1.5
    assert pytest.approx(m132_1.gyt) == 0.5
    assert pytest.approx(m132_1.gsl) == 2.0
    assert pytest.approx(m132_1.xc) == 1300.0
    assert pytest.approx(m132_1.xt) == 2400.0
    assert pytest.approx(m132_1.yc) == 220.0
    assert pytest.approx(m132_1.yt) == 70.0
    assert pytest.approx(m132_1.sl) == 95.0
    assert pytest.approx(m132_1.gxc0) == 40.0
    assert pytest.approx(m132_1.gxt0) == 70.0
    assert pytest.approx(m132_1.xc0) == 1100.0
    assert pytest.approx(m132_1.xt0) == 2000.0
    assert pytest.approx(m132_1.fio) == 53.0
    assert pytest.approx(m132_1.sigy) == 110.0
    assert pytest.approx(m132_1.etan) == 1000.0
    assert pytest.approx(m132_1.beta) == 0.2
    assert m132_1.lcss == 201
    assert pytest.approx(m132_1.epsf23) == 0.08
    assert pytest.approx(m132_1.epsr23) == 0.12
    assert pytest.approx(m132_1.tsmd23) == 0.95
    assert pytest.approx(m132_1.epsf31) == 0.07
    assert pytest.approx(m132_1.epsr31) == 0.11
    assert pytest.approx(m132_1.tsmd31) == 0.90
    assert pytest.approx(m132_1.ef11t) == 0.02
    assert pytest.approx(m132_1.ef11c) == 0.01
    assert pytest.approx(m132_1.ef22t) == 0.03
    assert pytest.approx(m132_1.ef22c) == 0.02
    assert pytest.approx(m132_1.ef12) == 0.04
    assert pytest.approx(m132_1.ef23) == 0.05
    assert pytest.approx(m132_1.ef31) == 0.04
    assert pytest.approx(m132_1.cf12) == 0.1
    assert pytest.approx(m132_1.cf23) == 0.3
    assert pytest.approx(m132_1.cf31) == 0.2
    assert pytest.approx(m132_1.ratio) == 0.9

    # Verify Law132 entity 402
    assert 402 in model.mat_law132s
    m132_2 = model.mat_law132s[402]
    assert m132_2.id == 402
    assert pytest.approx(m132_2.ea) == 145000.0
    assert m132_2.lcss == 202

    # Verify model.materials and GenericMaterialRecord
    mat1 = model.materials[401]
    assert mat1.law == 132
    assert pytest.approx(mat1.rho0) == 1.5e-9
    assert pytest.approx(mat1.params["E"]) == 150000.0
    assert mat1.record is not None
    assert mat1.record.law_name == "LAW132"
    assert mat1.record.law_number == 132


def test_mat_law134_viscous_foam_parsing(tmp_path: Path):
    """Test /MAT/LAW134 and /MAT/VISCOUS_FOAM parsing for viscous foam material."""
    deck = """\
# OpenRadioss Starter Deck
/BEGIN
LAW134_TEST
/MAT/LAW134/501
Viscous Foam Fixed Format
#              RHO_I
              1.2E-7
#                 E1                  N1                  NU
                15.0                 1.5                 0.3
#                 E2                  V2                  N2
                25.0                0.05                 1.2
/MAT/VISCOUS_FOAM/502
Viscous Foam Free Format
1.1e-7
12.0 1.4 0.28
20.0 0.04 1.1
/END
"""
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"

    # Verify Law134 entity 501
    assert 501 in model.mat_law134s
    m134_1 = model.mat_law134s[501]
    assert m134_1.id == 501
    assert m134_1.title == "Viscous Foam Fixed Format"
    assert pytest.approx(m134_1.rho0) == 1.2e-7
    assert pytest.approx(m134_1.e1) == 15.0
    assert pytest.approx(m134_1.n1) == 1.5
    assert pytest.approx(m134_1.nu) == 0.3
    assert pytest.approx(m134_1.e2) == 25.0
    assert pytest.approx(m134_1.v2) == 0.05
    assert pytest.approx(m134_1.n2) == 1.2

    # Verify Law134 entity 502
    assert 502 in model.mat_law134s
    m134_2 = model.mat_law134s[502]
    assert m134_2.id == 502
    assert pytest.approx(m134_2.rho0) == 1.1e-7
    assert pytest.approx(m134_2.e1) == 12.0
    assert pytest.approx(m134_2.n1) == 1.4
    assert pytest.approx(m134_2.nu) == 0.28
    assert pytest.approx(m134_2.e2) == 20.0
    assert pytest.approx(m134_2.v2) == 0.04
    assert pytest.approx(m134_2.n2) == 1.1

    # Verify model.materials and GenericMaterialRecord
    mat1 = model.materials[501]
    assert mat1.law == 134
    assert pytest.approx(mat1.rho0) == 1.2e-7
    assert pytest.approx(mat1.params["E"]) == 40.0  # e1 + e2 = 15.0 + 25.0
    assert mat1.record is not None
    assert mat1.record.law_name == "LAW134"
    assert mat1.record.law_number == 134
