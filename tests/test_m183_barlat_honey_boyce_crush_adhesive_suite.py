"""
Milestone M183 Test Suite:
Advanced Anisotropic Plasticity, Rate-Dependent Honeycomb, Bergstrom-Boyce Polymer,
Crushable Foam & 3D Adhesive Cohesive Material Models Suite:
- /MAT/LAW50, /MAT/VISC_HONEY, /MAT/HYP_FOAM
- /MAT/LAW57, /MAT/BARLAT3
- /MAT/LAW87, /MAT/BARLAT_YLD2000, /MAT/BARLAT2000
- /MAT/LAW95, /MAT/BERGSTROM_BOYCE, /MAT/HYP_VISC_PLAS, /MAT/FOAM_TAB
- /MAT/LAW163, /MAT/CRUSHABLE_FOAM, /MAT/CRUSH_FOAM
- /MAT/LAW169, /MAT/ARUP_ADHESIVE, /MAT/COH_TAB_3D, /MAT/COH_3D
"""

from pathlib import Path
import pytest
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog


def _parse_deck(tmp_path: Path, text: str, name: str = "TEST_0000.rad") -> tuple[Model, MessageLog]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    deck_path = tmp_path / name
    deck_path.write_text(text, encoding="ascii")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(deck_path))
    parse_starter_deck(blocks, model, log)
    return model, log


def test_mat_law50_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW50 & /MAT/VISC_HONEY in fixed and free formats."""
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
test_law50_fixed
                  10                   1
/MAT/LAW50/1
Honeycomb Material Law 50
#              RHO_I          Ref. dens.
              1.2E-3              1.2E-3
#                E11                 E22                 E33
               100.0               200.0               300.0
#                G12                 G23                 G31
                40.0                50.0                60.0
#             asrate
                10.0
#   Iflag1           Eps_max11           Eps_max22           Eps_max33
         0                0.15                0.25                0.35
#funID11-1 funID11-2 funID11-3 funID11-4 funID11-5
       101       102
#        Fscale_11-1         Fscale_11-2         Fscale_11-3         Fscale_11-4         Fscale_11-5
                 1.1                 1.2
#      Eps_rate_11-1       Eps_rate_11-2       Eps_rate_11-3       Eps_rate_11-4       Eps_rate_11-5
                 0.1                 1.0
#funID22-1 funID22-2 funId22-3 funID22-4 funID22-5
       201
#        Fscale_22-1         Fscale_22-2         Fscale_22-3         Fscale_22-4         Fscale_22-5
                 1.0
#      Eps_rate_22-1       Eps_rate_22-2       Eps_rate_22-3       Eps_rate_22-4       Eps_rate_22-5
                 0.0
#funID33-1 funID33-2 funID33-3 funID33-4 funID33-5
       301
#        Fscale_33-1         Fscale_33-2         Fscale_33-3         Fscale_33-4         Fscale_33-5
                 1.0
#      Eps_rate_33-1       Eps_rate_33-2       Eps_rate_33-3       Eps_rate_33-4       Eps_rate_33-5
                 0.0
#   Iflag2           Eps_max12           Eps_max23           Eps_max31
         0                0.12                0.23                0.31
#funID12-1 funID12-2 funID12-3 funID12-4 funID12-5
       401
#        Fscale_12-1         Fscale_12-2         Fscale_12-3         Fscale_12-4         Fscale_12-5
                 1.0
#      Eps_rate_12-1       Eps_rate_12-2       Eps_rate_12-3       Eps_rate_12-4       Eps_rate_12-5
                 0.0
#funID23-1 funID23-2 funID23-3 funID23-4 funID23-5
       501
#        Fscale_23-1         Fscale_23-2         Fscale_23-3         Fscale_23-4         Fscale_23-5
                 1.0
#      Eps_rate_23-1       Eps_rate_23-2       Eps_rate_23-3       Eps_rate_23-4       Eps_rate_23-5
                 0.0
#funID31-1 funID31-2 funID31-3 funID31-4 funID31-5
       601
#        Fscale_31-1         Fscale_31-2         Fscale_31-3         Fscale_31-4         Fscale_31-5
                 1.0
#      Eps_rate_31-1       Eps_rate_31-2       Eps_rate_31-3       Eps_rate_31-4       Eps_rate_31-5
                 0.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_fixed, "fixed.rad")

    assert 1 in model.mat_law50s
    m1 = model.mat_law50s[1]
    assert m1.rho == pytest.approx(1.2e-3)
    assert m1.rho0 == pytest.approx(1.2e-3)
    assert m1.ea == pytest.approx(100.0)
    assert m1.eb == pytest.approx(200.0)
    assert m1.ec == pytest.approx(300.0)
    assert m1.e11 == pytest.approx(100.0)
    assert m1.gab == pytest.approx(40.0)
    assert m1.g12 == pytest.approx(40.0)
    assert m1.asrate == pytest.approx(10.0)
    assert m1.eps_max11 == pytest.approx(0.15)
    assert m1.yfun11 == [101, 102]
    assert m1.sfac11 == [1.1, 1.2]
    assert m1.eps11 == [0.1, 1.0]
    assert m1.yfun22 == [201]
    assert m1.yfun33 == [301]
    assert m1.yfun12 == [401]
    assert m1.yfun23 == [501]
    assert m1.yfun31 == [601]

    # Free format /MAT/VISC_HONEY
    deck_free = """\
/BEGIN
test_law50_free
/MAT/VISC_HONEY/2
Visc Honeycomb Free
1.5E-3 0.0
150.0 250.0 350.0
45.0 55.0 65.0
12.0
1 0.1 0.2 0.3
11 12
1.0 1.5
0.05 0.5
21
1.0
0.0
31
1.0
0.0
1 0.05 0.06 0.07
41
1.0
0.0
51
1.0
0.0
61
1.0
0.0
/END
"""
    model2, log2 = _parse_deck(tmp_path, deck_free, "free.rad")

    assert 2 in model2.mat_law50s
    m2 = model2.mat_law50s[2]
    assert m2.rho == pytest.approx(1.5e-3)
    assert m2.ea == pytest.approx(150.0)
    assert m2.eb == pytest.approx(250.0)
    assert m2.ec == pytest.approx(350.0)
    assert m2.gflag == 1
    assert m2.yfun11 == [11, 12]
    assert m2.sfac11 == [1.0, 1.5]
    assert m2.eps11 == [0.05, 0.5]
    assert 2 in model2.materials
    assert model2.materials[2].params["e11"] == pytest.approx(150.0)


def test_mat_law57_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW57 & /MAT/BARLAT3 in fixed and free formats."""
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
test_law57_fixed
                  10                   1
/MAT/LAW57/1
Barlat 3 Fixed
#              RHO_I          Ref. dens.
              7.8E-3              7.8E-3
#                  E                  NU
            210000.0                 0.3
#                r00                 r45                 r90              C_hard                   m
                 1.5                 1.2                 1.8               500.0                 6.0
#           EPSP_max              EPS_t1              EPS_t2
                 0.4                 0.5                 0.6
# funct_ID                      Fscale_i               EPS_i
        10                           1.0                 0.0
        11                           1.1                 0.1
/END
"""
    model, log = _parse_deck(tmp_path, deck_fixed, "fixed57.rad")

    assert 1 in model.mat_law57s
    m1 = model.mat_law57s[1]
    assert m1.rho == pytest.approx(7.8e-3)
    assert m1.e == pytest.approx(210000.0)
    assert m1.nu == pytest.approx(0.3)
    assert m1.r00 == pytest.approx(1.5)
    assert m1.r45 == pytest.approx(1.2)
    assert m1.r90 == pytest.approx(1.8)
    assert m1.chard == pytest.approx(500.0)
    assert m1.m == pytest.approx(6.0)
    assert m1.epsp_max == pytest.approx(0.4)
    assert len(m1.curves) == 2
    assert m1.curves[0].fct_id == 10
    assert m1.curves[1].fscale == pytest.approx(1.1)
    assert m1.curves[1].eps == pytest.approx(0.1)

    # Free format /MAT/BARLAT3
    deck_free = """\
/BEGIN
test_law57_free
/MAT/BARLAT3/2
Barlat 3 Free
2.7E-3 0.0
70000.0 0.33
0.8 0.9 1.1 300.0 8.0
0.35 0.45 0.55
101 1.0 0.0
102 1.25 10.0
/END
"""
    model2, log2 = _parse_deck(tmp_path, deck_free, "free57.rad")

    assert 2 in model2.mat_law57s
    m2 = model2.mat_law57s[2]
    assert m2.rho == pytest.approx(2.7e-3)
    assert m2.e == pytest.approx(70000.0)
    assert m2.r00 == pytest.approx(0.8)
    assert m2.m == pytest.approx(8.0)
    assert len(m2.curves) == 2
    assert m2.curves[1].fct_id == 102
    assert m2.curves[1].fscale == pytest.approx(1.25)
    assert m2.curves[1].eps == pytest.approx(10.0)
    assert model2.materials[2].params["m"] == pytest.approx(8.0)


def test_mat_law87_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW87 & /MAT/BARLAT_YLD2000 in fixed and free formats."""
    # Fixed format with Ifit=0 (explicit alpha1..8) and Iflag=0 (curves)
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
test_law87_fixed
                  10                   1
/MAT/LAW87/1
Barlat Yld2000 Fixed
#              RHO_I               RHO_O
              7.8E-3              7.8E-3
#                  E                  Nu     IFlag        VP                   c                   P
            205000.0                0.29         0         1                40.0                 5.0
#                 a1                  a2                  a3                  a4
                0.95                1.05                0.98                1.02
#                 a5                  a6                  a7                  a8
                1.01                0.99                1.03                0.97
#              Chard      Ikin
               120.0         1
#              exp_a                                                       F_cut  F_smooth     Nrate
                 6.0                                                         0.0         0         2

#  Func_ID                        FSCALEi                Epsp
       501                                           1.0                 0.0
       502                                           1.1               100.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_fixed, "fixed87.rad")

    assert 1 in model.mat_law87s
    m1 = model.mat_law87s[1]
    assert m1.rho == pytest.approx(7.8e-3)
    assert m1.e == pytest.approx(205000.0)
    assert m1.nu == pytest.approx(0.29)
    assert m1.iflag == 0
    assert m1.vp == 1
    assert m1.strain1 == pytest.approx(40.0)
    assert m1.exp1 == pytest.approx(5.0)
    assert m1.ifit == 0
    assert m1.alpha[0] == pytest.approx(0.95)
    assert m1.alpha[7] == pytest.approx(0.97)
    assert m1.chard == pytest.approx(120.0)
    assert m1.ikin == 1
    assert m1.exp_a == pytest.approx(6.0)
    assert len(m1.curves) == 2
    assert m1.curves[0].fct_id == 501
    assert m1.curves[1].fct_id == 502
    assert m1.curves[1].fscale == pytest.approx(1.1)

    # Free format with Ifit=1 and Iflag=1 (Swift-Voce)
    deck_free = """\
/BEGIN
test_law87_free
/MAT/BARLAT_YLD2000/2
Barlat 2000 Free
2.7E-3 0.0
72000.0 0.33 1 0 0.0 0.0
200.0 220.0 240.0 250.0 1
0.85 0.95 1.15 1.05
50.0 0
8.0 0.0 0 0
500.0 0.002 150.0 15.0 200.0
/END
"""
    model2, log2 = _parse_deck(tmp_path, deck_free, "free87.rad")

    assert 2 in model2.mat_law87s
    m2 = model2.mat_law87s[2]
    assert m2.rho == pytest.approx(2.7e-3)
    assert m2.e == pytest.approx(72000.0)
    assert m2.iflag == 1
    assert m2.ifit == 1
    assert m2.sigma_00 == pytest.approx(200.0)
    assert m2.sigma_45 == pytest.approx(220.0)
    assert m2.sigma_90 == pytest.approx(240.0)
    assert m2.sigma_b == pytest.approx(250.0)
    assert m2.r_00 == pytest.approx(0.85)
    assert m2.r_45 == pytest.approx(0.95)
    assert m2.r_90 == pytest.approx(1.15)
    assert m2.r_b == pytest.approx(1.05)
    assert m2.aswift == pytest.approx(500.0)
    assert m2.eps0 == pytest.approx(0.002)
    assert m2.qvoce == pytest.approx(150.0)
    assert m2.beta == pytest.approx(15.0)
    assert m2.k0 == pytest.approx(200.0)


def test_mat_law95_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW95 & /MAT/BERGSTROM_BOYCE in fixed and free formats."""
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
test_law95_fixed
                  10                   1
/MAT/LAW95/1
Bergstrom Boyce Polymer Fixed
#              Rho_I
              1.1E-3
#                C10                 C01                 C20                 C11                 C02
                 2.5                 1.5                 0.5                 0.2                 0.1
#                C30                 C21                 C12                 C03                  Sb
                0.05                0.02                0.01               0.005                 1.2
#                 D1                  D2                  D3                  NU     IFORM
              0.0001              0.0002              0.0003                0.48         1
#                  A                   C                   M                 KSI             TAU_REF
               0.025               -0.35                 2.1                0.05                15.0
/END
"""
    model, log = _parse_deck(tmp_path, deck_fixed, "fixed95.rad")

    assert 1 in model.mat_law95s
    m1 = model.mat_law95s[1]
    assert m1.rho == pytest.approx(1.1e-3)
    assert m1.c10 == pytest.approx(2.5)
    assert m1.c01 == pytest.approx(1.5)
    assert m1.c20 == pytest.approx(0.5)
    assert m1.c11 == pytest.approx(0.2)
    assert m1.c02 == pytest.approx(0.1)
    assert m1.c30 == pytest.approx(0.05)
    assert m1.sb == pytest.approx(1.2)
    assert m1.d1 == pytest.approx(0.0001)
    assert m1.nu == pytest.approx(0.48)
    assert m1.iform == 1
    assert m1.a == pytest.approx(0.025)
    assert m1.c == pytest.approx(-0.35)
    assert m1.m == pytest.approx(2.1)
    assert m1.ksi == pytest.approx(0.05)
    assert m1.tau_ref == pytest.approx(15.0)

    # Free format /MAT/BERGSTROM_BOYCE
    deck_free = """\
/BEGIN
test_law95_free
/MAT/BERGSTROM_BOYCE/2
Bergstrom Boyce Free
0.95E-3
1.8 1.2 0.3 0.1 0.05
0.02 0.01 0.005 0.001 1.0
0.0005 0.0 0.0 0.495 0
0.015 -0.25 1.8 0.02 20.0
/END
"""
    model2, log2 = _parse_deck(tmp_path, deck_free, "free95.rad")

    assert 2 in model2.mat_law95s
    m2 = model2.mat_law95s[2]
    assert m2.rho == pytest.approx(0.95e-3)
    assert m2.c10 == pytest.approx(1.8)
    assert m2.c == pytest.approx(-0.25)
    assert m2.tau_ref == pytest.approx(20.0)
    assert model2.materials[2].params["c10"] == pytest.approx(1.8)


def test_mat_law163_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW163 & /MAT/CRUSHABLE_FOAM in fixed and free formats."""
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
test_law163_fixed
                  10                   1
/MAT/LAW163/1
Crushable Foam Fixed
#              RHO_I
              0.05E-3
#                  E                  NU                 TSC                DAMP              NCYCLE
                50.0                 0.1                 2.5                0.08                  10
#             TAB_ID            EPSD_REF              FSCALE           SRC_LIMIT                 NRS
                 101                 0.1                 1.2                50.0                   1
/END
"""
    model, log = _parse_deck(tmp_path, deck_fixed, "fixed163.rad")

    assert 1 in model.mat_law163s
    m1 = model.mat_law163s[1]
    assert m1.rho == pytest.approx(0.05e-3)
    assert m1.e == pytest.approx(50.0)
    assert m1.nu == pytest.approx(0.1)
    assert m1.tsc == pytest.approx(2.5)
    assert m1.damp == pytest.approx(0.08)
    assert m1.ncycle == 10
    assert m1.tab_id == 101
    assert m1.epsd_ref == pytest.approx(0.1)
    assert m1.fscale == pytest.approx(1.2)
    assert m1.srclmt == pytest.approx(50.0)
    assert m1.nrs == 1

    # Free format /MAT/CRUSHABLE_FOAM
    deck_free = """\
/BEGIN
test_law163_free
/MAT/CRUSHABLE_FOAM/2
Crushable Foam Free
0.08E-3
80.0 0.05 3.0 0.1 5
201 0.05 1.0 100.0 0
/END
"""
    model2, log2 = _parse_deck(tmp_path, deck_free, "free163.rad")

    assert 2 in model2.mat_law163s
    m2 = model2.mat_law163s[2]
    assert m2.rho == pytest.approx(0.08e-3)
    assert m2.e == pytest.approx(80.0)
    assert m2.tsc == pytest.approx(3.0)
    assert m2.tab_id == 201
    assert m2.fscale == pytest.approx(1.0)
    assert model2.materials[2].params["tsc"] == pytest.approx(3.0)


def test_mat_law169_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW169 & /MAT/ARUP_ADHESIVE in fixed and free formats."""
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
test_law169_fixed
                  10                   1
/MAT/LAW169/1
Arup Adhesive Fixed
#              Rho_I
              1.4E-3
#                  E                  PR              SHT_SL              TENMAX               GCTEN
              2500.0                0.38                 1.5                45.0                 2.8
#             SHRMAX               GCSHR      PWRT      PWRS                SHRP              
                35.0                 3.5         2         2                 0.4
/END
"""
    model, log = _parse_deck(tmp_path, deck_fixed, "fixed169.rad")

    assert 1 in model.mat_law169s
    m1 = model.mat_law169s[1]
    assert m1.rho == pytest.approx(1.4e-3)
    assert m1.e == pytest.approx(2500.0)
    assert m1.nu == pytest.approx(0.38)
    assert m1.pr == pytest.approx(0.38)
    assert m1.sht_sl == pytest.approx(1.5)
    assert m1.tenmax == pytest.approx(45.0)
    assert m1.gcten == pytest.approx(2.8)
    assert m1.shrmax == pytest.approx(35.0)
    assert m1.gcshr == pytest.approx(3.5)
    assert m1.pwrt == 2
    assert m1.pwrs == 2
    assert m1.shrp == pytest.approx(0.4)

    # Free format /MAT/ARUP_ADHESIVE
    deck_free = """\
/BEGIN
test_law169_free
/MAT/ARUP_ADHESIVE/2
Arup Adhesive Free
1.2E-3
1800.0 0.35 1.2 30.0 1.9
25.0 2.2 1 1 0.5
/END
"""
    model2, log2 = _parse_deck(tmp_path, deck_free, "free169.rad")

    assert 2 in model2.mat_law169s
    m2 = model2.mat_law169s[2]
    assert m2.rho == pytest.approx(1.2e-3)
    assert m2.e == pytest.approx(1800.0)
    assert m2.tenmax == pytest.approx(30.0)
    assert m2.shrmax == pytest.approx(25.0)
    assert m2.pwrt == 1
    assert m2.pwrs == 1
    assert m2.shrp == pytest.approx(0.5)
    assert model2.materials[2].params["tenmax"] == pytest.approx(30.0)


def test_m183_error_guards(tmp_path: Path):
    """Test error guards on missing cards."""
    deck_empty = """\
/BEGIN
test_empty
/MAT/LAW50/1
/MAT/LAW57/2
/MAT/LAW87/3
/MAT/LAW95/4
/MAT/LAW163/5
/MAT/LAW169/6
/END
"""
    model, log = _parse_deck(tmp_path, deck_empty, "empty.rad")

    assert len(model.mat_law50s) == 0
    assert len(model.mat_law57s) == 0
    assert len(model.mat_law87s) == 0
    assert len(model.mat_law95s) == 0
    assert len(model.mat_law163s) == 0
    assert len(model.mat_law169s) == 0
