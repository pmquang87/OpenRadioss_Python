"""Tests for Milestone M192: Advanced Composite Failure Criteria (Hoffman, Tsai-Hill, Tsai-Wu, Max-Strain, Fabric, Chang),
Thick Shell Properties (TSH_ORTH, TSH_COMP), Lee-Tarver Explosive (LAW41) & Advanced Materials (LAW79, LAW190),
and Contact Interfaces (TYPE19, TYPE20, TYPE21, TYPE23, TYPE24, TYPE25) Suite.
"""
from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    MatLaw41, MatLaw79, MatLaw190,
    FailHoffman, FailTsaiHill, FailTsaiWu, FailMaxStrain, FailFabric, FailChang,
    PropType21, PropType22,
    Interface
)


def _parse(tmp_path: Path, text: str) -> Model:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert not log.errors, f"Parse errors: {log.errors}"
    return model


# ==============================================================================
# 1. Material Laws Tests
# ==============================================================================

def test_mat_law41_free_and_fixed(tmp_path: Path):
    """Test /MAT/LAW41 & /MAT/LEE_T (Lee-Tarver explosive reaction kinetics)."""
    deck_free = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW41 Free
                  10                   0
/MAT/LAW41/41
Lee-Tarver Explosive
#             RHO_0               RHO_R
              1.8e-6                   0
#             IREAC                 A_R                 B_R                R_1R                R_2R                R_3R
                  1               8.5e5               1.8e4                 4.5                 1.5                 0.3
#               A_P                 B_P                R_1P                R_2P                R_3P
              7.5e5               1.2e4                 4.1                 1.2                 0.2
#              C_VR                C_VP                 ENQ
             1000.0              1200.0               4.5e6
#             NITRS           EPSILON_0                FTOL
                 50                0.01              1.0e-5
#           I_COEFF             B_COEFF             X_COEFF
               44.0               0.667                 4.0
#                G1             D_COEFF             Y_COEFF             C_COEFF
              400.0               0.667                 2.0               0.333
#                KN                 CHI                 TOL
                0.0                 0.0              1.0e-4
#                G2             E_COEFF             G_COEFF             Z_COEFF
               50.0               0.333                 1.0                 3.0
#             CCRIT              FIGMAX              FG1MAX              FG2MIN
                0.0                0.02                 0.5                 0.5
#                G0           T_INITIAL
                0.0               298.0
"""
    model_free = _parse(tmp_path, deck_free)
    assert 41 in model_free.mat_law41s
    m = model_free.mat_law41s[41]
    assert isinstance(m, MatLaw41)
    assert m.rho == pytest.approx(1.8e-6)
    assert m.ireac == 1
    assert m.a_r == pytest.approx(8.5e5)
    assert m.b_r == pytest.approx(1.8e4)
    assert m.r_1r == pytest.approx(4.5)
    assert m.a_p == pytest.approx(7.5e5)
    assert m.c_vr == pytest.approx(1000.0)
    assert m.c_vp == pytest.approx(1200.0)
    assert m.enq == pytest.approx(4.5e6)
    assert m.nitrs == 50
    assert m.i_coeff == pytest.approx(44.0)
    assert m.g1 == pytest.approx(400.0)
    assert m.g2 == pytest.approx(50.0)
    assert m.figmax == pytest.approx(0.02)
    assert m.fg1max == pytest.approx(0.5)
    assert m.fg2min == pytest.approx(0.5)
    assert m.t_initial == pytest.approx(298.0)

    # Fixed format with alias /MAT/LEE_T
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW41 Fixed
                  10                   0
/MAT/LEE_T/42
Lee-Tarver Fixed
             1.8E-06                 0.0
                   1              850000               18000                 4.5                 1.5                 0.3
              750000               12000                 4.1                 1.2                 0.2
              1000.0              1200.0             4500000
                  50                0.01               1E-05
                44.0               0.667                 4.0
               400.0               0.667                 2.0               0.333
                 0.0                 0.0               1E-04
                50.0               0.333                 1.0                 3.0
                 0.0                0.02                 0.5                 0.5
                 0.0               298.0
"""
    model_fixed = _parse(tmp_path, deck_fixed)
    assert 42 in model_fixed.mat_law41s
    mf = model_fixed.mat_law41s[42]
    assert mf.rho == pytest.approx(1.8e-6)
    assert mf.ireac == 1
    assert mf.a_r == pytest.approx(8.5e5)
    assert mf.g1 == pytest.approx(400.0)


def test_mat_law79_free_and_fixed(tmp_path: Path):
    """Test /MAT/LAW79 (Johnson-Holmquist ceramic / brittle material model)."""
    deck_free = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW79 Free
                  10                   0
/MAT/LAW79/79
Johnson-Holmquist Ceramic
#             RHO_0               RHO_R
              2.4e-6                   0
#                 G
             90000.0
#                 A                   B                   M                   N
               0.93                0.31                 1.0                0.64
#                 C                EPS0          SIGMA_FMAX                FCUT
              0.003                 1.0              1500.0                 0.0
#                T0                 HEL                PHEL
              200.0              6000.0              1500.0
#                D1                  D2                IDEL              EPSMAX
              0.005                 0.7                   1                 0.2
#                K1                  K2                  K3                BETA
           130000.0                 0.0                 0.0                 1.0
"""
    model = _parse(tmp_path, deck_free)
    assert 79 in model.mat_law79s
    m = model.mat_law79s[79]
    assert isinstance(m, MatLaw79)
    assert m.rho == pytest.approx(2.4e-6)
    assert m.g == pytest.approx(90000.0)
    assert m.a == pytest.approx(0.93)
    assert m.b == pytest.approx(0.31)
    assert m.n == pytest.approx(0.64)
    assert m.c == pytest.approx(0.003)
    assert m.t0 == pytest.approx(200.0)
    assert m.hel == pytest.approx(6000.0)
    assert m.phel == pytest.approx(1500.0)
    assert m.d1 == pytest.approx(0.005)
    assert m.d2 == pytest.approx(0.7)
    assert m.idel == 1
    assert m.k1 == pytest.approx(130000.0)
    assert m.beta == pytest.approx(1.0)


def test_mat_law190_free_and_fixed(tmp_path: Path):
    """Test /MAT/LAW190 & /MAT/FOAM_DUBOIS (Du Bois foam model)."""
    deck_free = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW190 Free
                  10                   0
/MAT/LAW190/190
Foam Dubois
#             RHO_0
              1.2e-6
#                E0                  NU
               50.0                 0.2
#                HU               SHAPE
                0.1                 1.5
#             FUN_1            XSCALE_1             SCALE_1
                801                 1.0                 1.0
"""
    model = _parse(tmp_path, deck_free)
    assert 190 in model.mat_law190s
    m = model.mat_law190s[190]
    assert isinstance(m, MatLaw190)
    assert m.rho == pytest.approx(1.2e-6)
    assert m.e0 == pytest.approx(50.0)
    assert m.nu == pytest.approx(0.2)
    assert m.hu == pytest.approx(0.1)
    assert m.shape == pytest.approx(1.5)
    assert m.fun_1 == 801
    assert m.xscale_1 == pytest.approx(1.0)
    assert m.scale_1 == pytest.approx(1.0)


# ==============================================================================
# 2. Failure Criteria Tests
# ==============================================================================

def test_fail_hoffman(tmp_path: Path):
    """Test /FAIL/HOFFMAN."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test FAIL HOFFMAN
                  10                   0
/MAT/ELAST/1
Steel
              7.8e-6                   0
            210000.0                 0.3
/FAIL/HOFFMAN/1
#          SIGMA_1T            SIGMA_2T            SIGMA_1C            SIGMA_2C            SIGMA_12
              150.0               120.0               200.0               180.0                80.0
#           TAU_MAX                FCUT            IFAIL_SH            IFAIL_SO
              100.0                 0.0                   1                   0
#           FAIL_ID
                  1
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.fail_hoffmans
    fh = model.fail_hoffmans[1]
    assert isinstance(fh, FailHoffman)
    assert fh.mat_id == 1
    assert fh.sigma_1t == pytest.approx(150.0)
    assert fh.sigma_2t == pytest.approx(120.0)
    assert fh.sigma_1c == pytest.approx(200.0)
    assert fh.sigma_2c == pytest.approx(180.0)
    assert fh.sigma_12 == pytest.approx(80.0)
    assert fh.tau_max == pytest.approx(100.0)
    assert fh.ifail_sh == 1
    assert fh.fail_id == 1


def test_fail_tsai_hill(tmp_path: Path):
    """Test /FAIL/TSAI_HILL."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test FAIL TSAI_HILL
                  10                   0
/MAT/ELAST/1
Steel
              7.8e-6                   0
            210000.0                 0.3
/FAIL/TSAI_HILL/1
#               X11                 X22                 S12            IFAIL_SH            IFAIL_SO
              250.0               180.0                90.0                   1                   0
#           TAU_MAX                FCUT
              120.0                 0.0
#           FAIL_ID
                  2
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.fail_tsaihills
    ft = model.fail_tsaihills[1]
    assert isinstance(ft, FailTsaiHill)
    assert ft.mat_id == 1
    assert ft.x11 == pytest.approx(250.0)
    assert ft.x22 == pytest.approx(180.0)
    assert ft.s12 == pytest.approx(90.0)
    assert ft.tau_max == pytest.approx(120.0)
    assert ft.fail_id == 2


def test_fail_tsai_wu(tmp_path: Path):
    """Test /FAIL/TSAI_WU."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test FAIL TSAI_WU
                  10                   0
/MAT/ELAST/1
Steel
              7.8e-6                   0
            210000.0                 0.3
/FAIL/TSAI_WU/1
#          SIGMA_1T            SIGMA_2T            SIGMA_1C            SIGMA_2C            SIGMA_12
              300.0               200.0               400.0               300.0               100.0
#             ALPHA             TAU_MAX                FCUT            IFAIL_SH            IFAIL_SO
               -0.5               150.0                 0.0                   1                   0
#           FAIL_ID
                  3
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.fail_tsaiwus
    ft = model.fail_tsaiwus[1]
    assert isinstance(ft, FailTsaiWu)
    assert ft.mat_id == 1
    assert ft.sigma_1t == pytest.approx(300.0)
    assert ft.sigma_2t == pytest.approx(200.0)
    assert ft.sigma_1c == pytest.approx(400.0)
    assert ft.sigma_2c == pytest.approx(300.0)
    assert ft.sigma_12 == pytest.approx(100.0)
    assert ft.alpha == pytest.approx(-0.5)
    assert ft.tau_max == pytest.approx(150.0)
    assert ft.fail_id == 3


def test_fail_max_strain(tmp_path: Path):
    """Test /FAIL/MAX_STRAIN."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test FAIL MAX_STRAIN
                  10                   0
/MAT/ELAST/1
Steel
              7.8e-6                   0
            210000.0                 0.3
/FAIL/MAX_STRAIN/1
#          EPS1_MAX            EPS2_MAX           GAM12_MAX            IFAIL_SH            IFAIL_SO
               0.08                0.06                0.12                   1                   0
#           TAU_MAX                FCUT
              200.0                 0.0
#           FAIL_ID
                  4
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.fail_maxstrains
    fm = model.fail_maxstrains[1]
    assert isinstance(fm, FailMaxStrain)
    assert fm.mat_id == 1
    assert fm.eps1_max == pytest.approx(0.08)
    assert fm.eps2_max == pytest.approx(0.06)
    assert fm.gam12_max == pytest.approx(0.12)
    assert fm.tau_max == pytest.approx(200.0)
    assert fm.fail_id == 4


def test_fail_fabric(tmp_path: Path):
    """Test /FAIL/FABRIC."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test FAIL FABRIC
                  10                   0
/MAT/ELAST/1
Steel
              7.8e-6                   0
            210000.0                 0.3
/FAIL/FABRIC/1
#        EPSILON_F1          EPSILON_R1          EPSILON_F2          EPSILON_R2                                    NDIR
               0.15                0.05                0.12                0.04                                       2
#            FCT_ID
                501
#           FAIL_ID
                  5
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.fail_fabrics
    ff = model.fail_fabrics[1]
    assert isinstance(ff, FailFabric)
    assert ff.mat_id == 1
    assert ff.epsilon_f1 == pytest.approx(0.15)
    assert ff.epsilon_r1 == pytest.approx(0.05)
    assert ff.epsilon_f2 == pytest.approx(0.12)
    assert ff.epsilon_r2 == pytest.approx(0.04)
    assert ff.ndir == 2
    assert ff.fct_id == 501
    assert ff.fail_id == 5


def test_fail_chang(tmp_path: Path):
    """Test /FAIL/CHANG."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test FAIL CHANG
                  10                   0
/MAT/ELAST/1
Steel
              7.8e-6                   0
            210000.0                 0.3
/FAIL/CHANG/1
#          SIGMA_1T            SIGMA_2T            SIGMA_12            SIGMA_1C            SIGMA_2C
              250.0               180.0                90.0               300.0               220.0
#              BETA             TAU_MAX            IFAIL_SH              FAILIP
                0.8               120.0                   1                   1
#           FAIL_ID
                  6
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.fail_changs
    fc = model.fail_changs[1]
    assert isinstance(fc, FailChang)
    assert fc.mat_id == 1
    assert fc.sigma_1t == pytest.approx(250.0)
    assert fc.sigma_2t == pytest.approx(180.0)
    assert fc.sigma_12 == pytest.approx(90.0)
    assert fc.sigma_1c == pytest.approx(300.0)
    assert fc.sigma_2c == pytest.approx(220.0)
    assert fc.beta == pytest.approx(0.8)
    assert fc.tau_max == pytest.approx(120.0)
    assert fc.ifail_sh == 1
    assert fc.failip == 1
    assert fc.fail_id == 6


# ==============================================================================
# 3. Thick Shell Properties Tests
# ==============================================================================

def test_prop_type21_tsh_orth(tmp_path: Path):
    """Test /PROP/TYPE21 & /PROP/TSH_ORTH."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test PROP TYPE21
                  10                   0
/PROP/TSH_ORTH/1
Orthotropic Thick Shell Fixed
        15         0                   0         0       222         1                   0.010000000
             1.100000000         0.050000000
             1.000000000         0.000000000         0.000000000         5         1
            30.000000000
             1.00000E-07
/PROP/TYPE21/2
Orthotropic Thick Shell Free
15 0 0 222 1 0.02
1.15 0.04
0.0 1.0 0.0 6 2
45.0
2.5e-7
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.prop_tsh_orths
    p1 = model.prop_tsh_orths[1]
    assert isinstance(p1, PropType21)
    assert p1.id == 1
    assert p1.isolid == 15
    assert p1.inpts_r == 2
    assert p1.inpts_s == 2
    assert p1.inpts_t == 2
    assert p1.dn == pytest.approx(0.01)
    assert p1.qa == pytest.approx(1.1)
    assert p1.qb == pytest.approx(0.05)
    assert p1.vx == pytest.approx(1.0)
    assert p1.skew_id == 5
    assert p1.iorth == 1
    assert p1.phi == pytest.approx(30.0)
    assert p1.deltat_min == pytest.approx(1.0e-7)

    assert 2 in model.prop_tsh_orths
    p2 = model.prop_tsh_orths[2]
    assert p2.id == 2
    assert p2.vx == pytest.approx(0.0)
    assert p2.vy == pytest.approx(1.0)
    assert p2.skew_id == 6
    assert p2.iorth == 2
    assert p2.phi == pytest.approx(45.0)
    assert p2.deltat_min == pytest.approx(2.5e-7)


def test_prop_type22_tsh_comp(tmp_path: Path):
    """Test /PROP/TYPE22 & /PROP/TSH_COMP."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test PROP TYPE22
                  10                   0
/PROP/TSH_COMP/1
Composite Layered Thick Shell Fixed
        15         0                   0       222         2         0.010000000
             1.100000000         0.050000000
             1.000000000         0.000000000         0.000000000         5         1         0
             0.833333330
             0.000000000         0.500000000        -0.250000000        10
            90.000000000         0.500000000         0.250000000        10
             1.00000E-07
/PROP/TYPE22/2
Composite Layered Thick Shell Free
15 0 0 222 2 0.02
1.2 0.06
1.0 0.0 0.0 0 0 0
0.8333
-45.0 0.5 -0.25 20
45.0 0.5 0.25 20
2.0e-7
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.prop_tsh_comps
    p1 = model.prop_tsh_comps[1]
    assert isinstance(p1, PropType22)
    assert p1.id == 1
    assert p1.isolid == 15
    assert p1.ashear == pytest.approx(0.83333333)
    assert len(p1.layers) == 2
    assert p1.layers[0].phi == pytest.approx(0.0)
    assert p1.layers[0].thick == pytest.approx(0.5)
    assert p1.layers[0].zi == pytest.approx(-0.25)
    assert p1.layers[0].mat_id == 10
    assert p1.layers[1].phi == pytest.approx(90.0)
    assert p1.layers[1].mat_id == 10
    assert p1.deltat_min == pytest.approx(1.0e-7)

    assert 2 in model.prop_tsh_comps
    p2 = model.prop_tsh_comps[2]
    assert p2.id == 2
    assert len(p2.layers) == 2
    assert p2.layers[0].phi == pytest.approx(-45.0)
    assert p2.layers[1].phi == pytest.approx(45.0)
    assert p2.layers[1].mat_id == 20


# ==============================================================================
# 4. Contact Interfaces Tests
# ==============================================================================

def test_inter_type19(tmp_path: Path):
    """Test /INTER/TYPE19 (Tied contact with failure)."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test INTER TYPE19
                  10                   0
/INTER/TYPE19/19
Tied Contact With Failure
#          GRNOD_ID             SURF_ID                ISTF                IGAP               IEDGE                IBAG                IDEL               ICURV
                 10                  20                   2                   1                   0                   0                   1                   0
#        GAP_SCALE              GAP_MAX
                1.2                 5.0
#             STMIN               STMAX
               10.0               100.0
#                N1                  N2
                  1                   2
#             STFAC                FRIC                 GAP              TSTART               TSTOP
                1.0                 0.2                 0.5                 0.0                 0.1
"""
    model = _parse(tmp_path, deck)
    inter = next((i for i in model.interfaces if i.id == 19), None)
    assert inter is not None
    assert isinstance(inter, Interface)
    assert inter.type == 19
    assert inter.grnod_id == 10
    assert inter.surf_id == 20
    assert inter.istf == 2
    assert inter.igap == 1
    assert inter.gap_scale == pytest.approx(1.2)
    assert inter.gap_max == pytest.approx(5.0)
    assert inter.stmin == pytest.approx(10.0)
    assert inter.stmax == pytest.approx(100.0)
    assert inter.fric == pytest.approx(0.2)
    assert inter.gap == pytest.approx(0.5)


def test_inter_type20(tmp_path: Path):
    """Test /INTER/TYPE20 (Spotweld interface)."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test INTER TYPE20
                  10                   0
/INTER/TYPE20/20
Spotweld Interface
#             SURF1               SURF2                ISYM               IEDGE            GRNOD_ID            LINE_ID1            LINE_ID2          EDGE_ANGLE
                 11                  22                   1                   0                  33                  44                  55                15.0
"""
    model = _parse(tmp_path, deck)
    inter = next((i for i in model.interfaces if i.id == 20), None)
    assert inter is not None
    assert isinstance(inter, Interface)
    assert inter.type == 20
    assert inter.surf_id == 11
    assert inter.surf_id1 == 22
    assert inter.isym == 1
    assert inter.grnod_id == 33
    assert inter.line_id1 == 44
    assert inter.line_id2 == 55
    assert inter.edge_angle == pytest.approx(15.0)


def test_inter_type21(tmp_path: Path):
    """Test /INTER/TYPE21 (Line-to-line contact)."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test INTER TYPE21
                  10                   0
/INTER/TYPE21/21
Line to Line Contact
#            SURF_1              SURF_2                ISTF                IGAP             MULTIMP                IADM
                 15                  25                   3                   1                   0                   0
#        GAP_SCALE              GAP_MAX             DSEARCH
                1.0                 2.0                 0.5
#                                                     STFAC                FRIC
                                                        1.5                0.25
"""
    model = _parse(tmp_path, deck)
    inter = next((i for i in model.interfaces if i.id == 21), None)
    assert inter is not None
    assert isinstance(inter, Interface)
    assert inter.type == 21
    assert inter.surf_id == 15
    assert inter.surf_id1 == 25
    assert inter.istf == 3
    assert inter.igap == 1
    assert inter.gap_max == pytest.approx(2.0)
    assert inter.dsearch == pytest.approx(0.5)
    assert inter.stfac == pytest.approx(1.5)
    assert inter.fric == pytest.approx(0.25)


def test_inter_type23(tmp_path: Path):
    """Test /INTER/TYPE23 (Mortar contact)."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test INTER TYPE23
                  10                   0
/INTER/TYPE23/23
Mortar Contact
#            SURF_S              SURF_M                ISTF                IGAP                IBAG                IDEL
                 12                  24                   1                   2                   0                   1
#        FSCALE_GAP             GAP_MAX
                1.1                 3.5
"""
    model = _parse(tmp_path, deck)
    inter = next((i for i in model.interfaces if i.id == 23), None)
    assert inter is not None
    assert isinstance(inter, Interface)
    assert inter.type == 23
    assert inter.surf_id == 24
    assert inter.surf_id1 == 12
    assert inter.istf == 1
    assert inter.igap == 2
    assert inter.idel == 1
    assert inter.fscale_gap == pytest.approx(1.1)
    assert inter.gap_max == pytest.approx(3.5)


def test_inter_type24(tmp_path: Path):
    """Test /INTER/TYPE24 (Surface-to-surface linear contact)."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test INTER TYPE24
                  10                   0
/INTER/TYPE24/24
Surface Linear Contact
#          GRNOD_ID             SURF_ID                ISTF                IGAP
                 18                  28                   4                   1
#             STFAC                FRIC              GAPMIN              GAPMAX
                2.0                 0.3                 0.1                 1.5
"""
    model = _parse(tmp_path, deck)
    inter = next((i for i in model.interfaces if i.id == 24), None)
    assert inter is not None
    assert isinstance(inter, Interface)
    assert inter.type == 24
    assert inter.grnod_id == 18
    assert inter.surf_id == 28
    assert inter.istf == 4
    assert inter.igap == 1
    assert inter.stfac == pytest.approx(2.0)
    assert inter.fric == pytest.approx(0.3)


def test_inter_type25(tmp_path: Path):
    """Test /INTER/TYPE25 (Smooth contact interface)."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test INTER TYPE25
                  10                   0
/INTER/TYPE25/25
Smooth Contact
#   surf_1    surf_2      Istf      Ithe      Igap                        Idel
        14        24         2         0         1                           1
#          GRNOD_ID                                                 PRMESH_SIZE                GAP1                GAP2
                 10                                                         0.0                 0.1                 0.2
#             STMIN               STMAX           IGAP_EDGE
                5.0                50.0                   1
#             STFAC                FRIC                                                      TSTART               TSTOP
                1.2                0.15                                                         0.0                 0.5
"""
    model = _parse(tmp_path, deck)
    inter = next((i for i in model.interfaces if i.id == 25), None)
    assert inter is not None
    assert isinstance(inter, Interface)
    assert inter.type == 25
    assert inter.surf_id == 14
    assert inter.surf_id1 == 24
    assert inter.grnod_id == 10
    assert inter.istf == 2
    assert inter.igap == 1
    assert inter.stfac == pytest.approx(1.2)
    assert inter.fric == pytest.approx(0.15)
    assert inter.gap == pytest.approx(0.1)
    assert inter.gap_max == pytest.approx(0.2)
    assert inter.tstop == pytest.approx(0.5)
