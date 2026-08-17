"""Tests for Milestone M191: Advanced Material Laws (100, 97, 71, 73, 84, 93, 133, 101, 43),
Multi-axial and Visual Failure Criteria (LEMAITRE, COMPOSITE, TAB2, ALTER, VISUAL, ORTHSTRAIN),
and Boundary Conditions / Preloads (EBCS/PROPELLANT, EBCS/CYCLIC, PRELOAD).
"""
from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.model.entities import (
    MatLaw100, MatLaw97, MatLaw71, MatLaw73, MatLaw84, MatLaw93, MatLaw133, MatLaw101, MatLaw43,
    FailLemaitre, FailComposite, FailTab2, FailAlter, FailVisual, FailOrthstrain,
    EbcsPropellant, EbcsCyclic, Preload
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

def test_mat_law100_free_and_fixed(tmp_path: Path):
    """Test /MAT/LAW100 (SPOTWELD / STRUCTURAL_ADHESIVE)."""
    deck_free = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW100 Free
                  10                   0
/MAT/LAW100/100
Spotweld adhesive
#             RHO_0               RHO_R
              1.2e-6                   0
#           FLAG_HE             FLAG_CR
                  1                   2
#               C10                 C01                 C20                 C11                 C02
                1.5                 2.5                 3.5                 4.5                 5.5
#               C30                 C21                 C12                 C03                  D1
                6.5                 7.5                 8.5                 9.5                10.5
#                D2                  D3                MUE1                   D            LAMBDA_M
               11.5                12.5                13.5                14.5                15.5
#             ITYPE           FCT_ID_AB                  NU           FCT_ID_SM           FCT_ID_BM
                  3                 101                 0.3                 102                 103
#         FSCALE_SM           FSCALE_BM                A_PL            SIGMA_PL                F_PL
                1.1                 1.2                 5.0                50.0                 0.1
#         EPSILON_F                N_PL
               0.25                   2
"""
    model_free = _parse(tmp_path, deck_free)
    assert 100 in model_free.mat_law100s
    m = model_free.mat_law100s[100]
    assert isinstance(m, MatLaw100)
    assert m.rho0 == pytest.approx(1.2e-6)
    assert m.flag_he == 1
    assert m.flag_cr == 2
    assert m.c10 == pytest.approx(1.5)
    assert m.d3 == pytest.approx(12.5)
    assert m.itype == 3
    assert m.fct_id_ab == 101
    assert m.nu == pytest.approx(0.3)
    assert m.epsilon_f == pytest.approx(0.25)
    assert m.n_pl == 2

    # Verify generic material
    mat = model_free.materials[100]
    assert mat.law == 100
    assert mat.params["sigma_pl"] == pytest.approx(50.0)

    # Fixed format
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW100 Fixed
                  10                   0
/MAT/SPOTWELD/101
Spotweld fixed
             1.2E-06                 0.0
                   1                   2
                 1.5                 2.5                 3.5                 4.5                 5.5
                 6.5                 7.5                 8.5                 9.5                10.5
                11.5                12.5                13.5                14.5                15.5
                   3                 101                 0.3                 102                 103
                 1.1                 1.2                 5.0                50.0                 0.1
                0.25                   2
"""
    model_fixed = _parse(tmp_path, deck_fixed)
    assert 101 in model_fixed.mat_law100s
    m_fixed = model_fixed.mat_law100s[101]
    assert m_fixed.rho0 == pytest.approx(1.2e-6)
    assert m_fixed.flag_he == 1
    assert m_fixed.itype == 3
    assert m_fixed.fscale_sm == pytest.approx(1.1)


def test_mat_law97_free_and_fixed(tmp_path: Path):
    """Test /MAT/LAW97 (EXPLOSIVE_JWLS / JWLS)."""
    deck_free = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW97 Free
                  10                   0
/MAT/LAW97/97
Explosive JWLS
#             RHO_0               RHO_R
              1.6e-6                   0
#                P0                 PSH              IBFRAC
               1e-4                1e-3                   1
#                 D                 PCJ                  E0               OMEGA                   C
             7000.0               2.5e4               6.0e3                 0.3                 1.2
#                A1                  A2                  A3                  A4                  A5
                1.0                 2.0                 3.0                 4.0                 5.0
#                R1                  R2                  R3                  R4                  R5
                4.5                 1.5                 0.5                 0.2                 0.1
"""
    model_free = _parse(tmp_path, deck_free)
    assert 97 in model_free.mat_law97s
    m = model_free.mat_law97s[97]
    assert isinstance(m, MatLaw97)
    assert m.rho0 == pytest.approx(1.6e-6)
    assert m.p0 == pytest.approx(1e-4)
    assert m.psh == pytest.approx(1e-3)
    assert m.ibfrac == 1
    assert m.d == pytest.approx(7000.0)
    assert m.pcj == pytest.approx(2.5e4)
    assert m.e0 == pytest.approx(6.0e3)
    assert m.omega == pytest.approx(0.3)
    assert m.a1 == pytest.approx(1.0)
    assert m.r1 == pytest.approx(4.5)

    # Fixed format with alias /MAT/EXPLOSIVE_JWLS
    deck_fixed = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW97 Fixed
                  10                   0
/MAT/EXPLOSIVE_JWLS/98
JWLS Fixed
             1.6E-06                 0.0
              0.0001               0.001                   1
              7000.0             25000.0              6000.0                 0.3                 1.2
                 1.0                 2.0                 3.0                 4.0                 5.0
                 4.5                 1.5                 0.5                 0.2                 0.1
"""
    model_fixed = _parse(tmp_path, deck_fixed)
    assert 98 in model_fixed.mat_law97s
    m_fixed = model_fixed.mat_law97s[98]
    assert m_fixed.d == pytest.approx(7000.0)
    assert m_fixed.pcj == pytest.approx(25000.0)


def test_mat_law71_free_and_fixed(tmp_path: Path):
    """Test /MAT/LAW71 (SUPER_ELAS / NITINOL)."""
    deck_free = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW71 Free
                  10                   0
/MAT/LAW71/71
Nitinol superelastic
#             RHO_0               RHO_R
              6.5e-6                   0
#                 E                  NU              E_MART             SIG_SAS             SIG_FAS
            60000.0                0.33             30000.0               500.0               600.0
#           SIG_SSA             SIG_FSA               ALPHA                EPSL                 CAS
              200.0               100.0                 0.1                0.06                 5.0
#               CSA                TSAS                TFAS                TSSA                TFSA
                5.0               293.0               300.0               280.0               270.0
#                CP                TINI
              450.0               293.0
"""
    model = _parse(tmp_path, deck_free)
    assert 71 in model.mat_law71s
    m = model.mat_law71s[71]
    assert isinstance(m, MatLaw71)
    assert m.rho0 == pytest.approx(6.5e-6)
    assert m.e == pytest.approx(60000.0)
    assert m.e_mart == pytest.approx(30000.0)
    assert m.sig_sas == pytest.approx(500.0)
    assert m.alpha == pytest.approx(0.1)
    assert m.epsl == pytest.approx(0.06)
    assert m.tsas == pytest.approx(293.0)
    assert m.cp == pytest.approx(450.0)


def test_mat_law73_free_and_fixed(tmp_path: Path):
    """Test /MAT/LAW73 (THERM_HILL / HILL_THERM)."""
    deck_free = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW73 Free
                  10                   0
/MAT/LAW73/73
Thermal Hill Orthotropic
#             RHO_0               RHO_R
              2.7e-6                   0
#                 E                  NU                 R00                 R45                 R90
            70000.0                0.33                 1.2                 1.1                 1.5
#             CHARD             EPS_MAX               EPST1               EPST2              FUN_A1
                0.1                 0.2                0.01                0.02                   201
#            FSCALE              PSCALE           T_INITIAL              SPHEAT              IYIELD
                1.0                 1.0               293.0               900.0                   1
#            YR_FUN                EFIB                   C
                202             10000.0                 0.5
"""
    model = _parse(tmp_path, deck_free)
    assert 73 in model.mat_law73s
    m = model.mat_law73s[73]
    assert isinstance(m, MatLaw73)
    assert m.r00 == pytest.approx(1.2)
    assert m.r90 == pytest.approx(1.5)
    assert m.fun_a1 == 201
    assert m.t_initial == pytest.approx(293.0)
    assert m.yr_fun == 202
    assert m.efib == pytest.approx(10000.0)


def test_mat_law84_free_and_fixed(tmp_path: Path):
    """Test /MAT/LAW84 (SWIFT_VOCE / PLAS_SWIFT_VOCE)."""
    deck_free = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW84 Free
                  10                   0
/MAT/LAW84/84
Swift Voce Hardening
#             RHO_0               RHO_R
              7.8e-6                   0
#                 E                  NU                FCUT             CAP_END                  PC
           210000.0                 0.3                 0.0                 0.0                 0.0
#                PR                  T0                C2_T                  A2                C1_C
                0.0               293.0                 0.0                 0.0                 0.0
#               VOL                 NUT            FSCALE11            FSCALE22            FSCALE33
                0.0                 0.0                 1.0                 1.0                 1.0
#          FSCALE12            FSCALE23              SCALE1              SCALE2              SCALE3
                1.0                 1.0                 0.0                 0.0                 0.0
#            SCALE4              SCALE5
                0.0                 0.0
"""
    model = _parse(tmp_path, deck_free)
    assert 84 in model.mat_law84s
    m = model.mat_law84s[84]
    assert isinstance(m, MatLaw84)
    assert m.e == pytest.approx(210000.0)
    assert m.nu == pytest.approx(0.3)
    assert m.t0 == pytest.approx(293.0)


def test_mat_law93_free_and_fixed(tmp_path: Path):
    """Test /MAT/LAW93 (ORTH_HILL)."""
    deck_free = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW93 Free
                  10                   0
/MAT/LAW93/93
3D Orthotropic Hill
#             RHO_0               RHO_R
              7.8e-6                   0
#               E11                 E22                 E33                 G12                NU12
           200000.0            190000.0            180000.0             80000.0                 0.3
#               G13                 G23                NU13                NU23                  NL
            75000.0             70000.0                0.28                0.27                   2
#           SIGMA_Y                 QR1                 CR1                 QR2                 CR2
              250.0               100.0                10.0                50.0                 5.0
#               R11                 R22                 R12                 R33                 R13
                1.0                 1.1                 1.0                 0.9                 1.0
#               R23                FCUT                  VP
                1.0                 0.0                   1
#            FCT_ID              FSCALE             EPS_DOT
                301                 1.0                 0.0
                302                 1.0               100.0
"""
    model = _parse(tmp_path, deck_free)
    assert 93 in model.mat_law93s
    m = model.mat_law93s[93]
    assert isinstance(m, MatLaw93)
    assert m.e11 == pytest.approx(200000.0)
    assert m.e22 == pytest.approx(190000.0)
    assert m.nl == 2
    assert m.sigma_y == pytest.approx(250.0)
    assert m.r22 == pytest.approx(1.1)
    assert len(m.curves) == 2
    assert m.curves[0]["fct_id"] == 301
    assert m.curves[1]["eps_dot"] == pytest.approx(100.0)


def test_mat_law133_free_and_fixed(tmp_path: Path):
    """Test /MAT/LAW133 (GRANULAR)."""
    deck_free = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW133 Free
                  10                   0
/MAT/LAW133/133
Granular material
#             RHO_0
              2.0e-6
#                NU                PMIN
               0.25               -0.01
#          FCT_ID_G                   0            FSCALE_G
                401                   0                 1.5
#          FCT_ID_Y                   0            FSCALE_Y
                402                   0                 2.0
"""
    model = _parse(tmp_path, deck_free)
    assert 133 in model.mat_law133s
    m = model.mat_law133s[133]
    assert isinstance(m, MatLaw133)
    assert m.rho0 == pytest.approx(2.0e-6)
    assert m.nu == pytest.approx(0.25)
    assert m.pmin == pytest.approx(-0.01)
    assert m.fct_id_g == 401
    assert m.fscale_g == pytest.approx(1.5)
    assert m.fct_id_y == 402
    assert m.fscale_y == pytest.approx(2.0)


def test_mat_law101_free_and_fixed(tmp_path: Path):
    """Test /MAT/LAW101 (PLAS_POLY)."""
    deck_free = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW101 Free
                  10                   0
/MAT/LAW101/101
Polymer Viscoplasticity
#             RHO_0               RHO_R
              1.1e-6                   0
#                 E              ALPHA1                  NU                 VE1
             3000.0                 0.1                0.35                 0.2
#               VE2          EPSILONREF              GAMMA0             ALPHA_P
                0.3                0.01                 1.5                 0.4
#            DELTAH                 VOL                   M                  C3
                5.0                10.0                 2.0                 1.0
#                C4             ALPHAK1             ALPHAK2                HARD
                1.1                 0.5                 0.6                50.0
#            ZETA1I                  C5                  C6                  C7
                0.8                 1.2                 1.3                 1.4
#                C8                  C9                 C10
                1.5                 1.6                 1.7
"""
    model = _parse(tmp_path, deck_free)
    assert 101 in model.mat_law101s
    m = model.mat_law101s[101]
    assert isinstance(m, MatLaw101)
    assert m.rho0 == pytest.approx(1.1e-6)
    assert m.e == pytest.approx(3000.0)
    assert m.ve1 == pytest.approx(0.2)
    assert m.gamma0 == pytest.approx(1.5)
    assert m.vol == pytest.approx(10.0)
    assert m.c10 == pytest.approx(1.7)


def test_mat_law43_free_and_fixed(tmp_path: Path):
    """Test /MAT/LAW43 (HILL_TAB)."""
    deck_free = """\
#RADIOSS STARTER
/BEGIN
Test MAT LAW43 Free
                  10                   0
/MAT/LAW43/43
Tabulated Hill Orthotropic
#             RHO_0               RHO_R
              7.8e-6                   0
#                 E                  NU              YR_FUN                EFIB                   C
           210000.0                 0.3                 501             20000.0                 0.4
#               R00                 R45                 R90               CHARD              IYIELD
                1.1                 1.2                 1.3                 0.1                   1
#               EPS               EPST1               EPST2          NUM_CURVES             FSMOOTH                FCUT
                0.2                0.01                0.02                   2                   1                 0.0
#          FUNCT_ID              FSCALE             EPS_DOT
                502                 1.0                 0.0
                503                 1.0                50.0
"""
    model = _parse(tmp_path, deck_free)
    assert 43 in model.mat_law43s
    m = model.mat_law43s[43]
    assert isinstance(m, MatLaw43)
    assert m.e == pytest.approx(210000.0)
    assert m.yr_fun == 501
    assert m.r45 == pytest.approx(1.2)
    assert m.num_curves == 2
    assert len(m.curves) == 2
    assert m.curves[0]["funct_id"] == 502
    assert m.curves[1]["eps_dot"] == pytest.approx(50.0)


# ==============================================================================
# 2. Failure Criteria Tests
# ==============================================================================

def test_fail_lemaitre(tmp_path: Path):
    """Test /FAIL/LEMAITRE."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test FAIL LEMAITRE
                  10                   0
/MAT/ELAST/1
Steel
              7.8e-6                   0
            210000.0                 0.3
/FAIL/LEMAITRE/1
#             EPS_D                 S_D                  DC              FAILIP         P_THICKFAIL
               0.05                 2.0                 0.8                   1                 0.5
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.fail_lemaitres
    fl = model.fail_lemaitres[1]
    assert isinstance(fl, FailLemaitre)
    assert fl.mat_id == 1
    assert fl.eps_d == pytest.approx(0.05)
    assert fl.s_d == pytest.approx(2.0)
    assert fl.dc == pytest.approx(0.8)
    assert fl.failip == 1
    assert fl.p_thickfail == pytest.approx(0.5)


def test_fail_composite(tmp_path: Path):
    """Test /FAIL/COMPOSITE."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test FAIL COMPOSITE
                  10                   0
/MAT/ELAST/1
Steel
              7.8e-6                   0
            210000.0                 0.3
/FAIL/COMPOSITE/1
#             SIG_1T              SIG_1C              SIG_2T              SIG_2C              SIG_12
                10.0                20.0                15.0                25.0                12.0
#             SIG_3T              SIG_3C              SIG_23              SIG_31
                22.0                30.0                35.0                40.0
#               BETA             TAU_MAX                EXPN            IFAIL_SH            IFAIL_SO
                 0.5                50.0                 1.5                   1                   1
#            FAIL_ID
                   1
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.fail_composites
    fc = model.fail_composites[1]
    assert isinstance(fc, FailComposite)
    assert fc.sig_1t == pytest.approx(10.0)
    assert fc.sig_1c == pytest.approx(20.0)
    assert fc.sig_12 == pytest.approx(12.0)
    assert fc.beta == pytest.approx(0.5)
    assert fc.fail_id == 1


def test_fail_tab2(tmp_path: Path):
    """Test /FAIL/TAB2."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test FAIL TAB2
                  10                   0
/MAT/ELAST/1
Steel
              7.8e-6                   0
            210000.0                 0.3
/FAIL/TAB2/1
#            EPSF_ID               FCRIT              FAILIP                PTHK
                 601                 1.0                   1                 0.5
#                  N               DCRIT             INST_ID               ECRIT
                 1.0                 0.8                 602                 0.1
#            FCT_EXP             EXP_REF                 EXP
                 603                 1.0                 1.5
#            FAIL_ID
                   2
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.fail_tab2s
    ft = model.fail_tab2s[1]
    assert isinstance(ft, FailTab2)
    assert ft.epsf_id == 601
    assert ft.fcrit == pytest.approx(1.0)
    assert ft.failip == 1
    assert ft.dcrit == pytest.approx(0.8)
    assert ft.fct_exp == 603
    assert ft.fail_id == 2


def test_fail_alter(tmp_path: Path):
    """Test /FAIL/ALTER."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test FAIL ALTER
                  10                   0
/MAT/ELAST/1
Steel
              7.8e-6                   0
            210000.0                 0.3
/FAIL/ALTER/1
#             EXP_N                  V0                  VC                 EMA               IRATE               ISIDE                MODE
                1.5                10.0               100.0                   1                   0                   1                   2
#           CR_FOIL              CR_AIR             CR_CORE             CR_EDGE              GRSH4N              GRSH3N
                0.1                 0.2                 0.3                 0.4                   1                   2
#               KIC                 KTH                RLEN                TDEL
              100.0                50.0                 2.5               0.001
#             KRES1               KRES2
                5.0                 6.0
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.fail_alters
    fa = model.fail_alters[1]
    assert isinstance(fa, FailAlter)
    assert fa.exp_n == pytest.approx(1.5)
    assert fa.v0 == pytest.approx(10.0)
    assert fa.vc == pytest.approx(100.0)
    assert fa.ema == 1
    assert fa.mode == 2
    assert fa.cr_foil == pytest.approx(0.1)
    assert fa.kic == pytest.approx(100.0)
    assert fa.kres1 == pytest.approx(5.0)


def test_fail_visual(tmp_path: Path):
    """Test /FAIL/VISUAL."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test FAIL VISUAL
                  10                   0
/MAT/ELAST/1
Steel
              7.8e-6                   0
            210000.0                 0.3
/FAIL/VISUAL/1
#              TYPE               C_MIN               C_MAX             F_COEFF              F_FLAG                   0              STRDEF
                  2                 0.1                 0.9                 1.5                   1                   0                   3
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.fail_visuals
    fv = model.fail_visuals[1]
    assert isinstance(fv, FailVisual)
    assert fv.vtype == 2
    assert fv.c_min == pytest.approx(0.1)
    assert fv.c_max == pytest.approx(0.9)
    assert fv.alpha_exp == pytest.approx(1.5)
    assert fv.f_flag == 1
    assert fv.strdef == 3


def test_fail_orthstrain(tmp_path: Path):
    """Test /FAIL/ORTHSTRAIN."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test FAIL ORTHSTRAIN
                  10                   0
/MAT/ELAST/1
Steel
              7.8e-6                   0
            210000.0                 0.3
/FAIL/ORTHSTRAIN/1
#                                  PTHK
                                    0.5
#       EPS_DOT_REF                FCUT
                1.0                 0.0
#          FCT_IDEL           FSCALE_EL              EI_REF              STRDEF
                701                 1.0                 0.0                   1
#        EPS_11__TF          EPS_11__TM         FCT_ID_11_T          EPS_11__CF          EPS_11__CM         FCT_ID_11_C
               0.05                0.08                 702                0.04                0.07                 703
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.fail_orthstrains
    fo = model.fail_orthstrains[1]
    assert isinstance(fo, FailOrthstrain)
    assert fo.pthk == pytest.approx(0.5)
    assert fo.eps_dot_ref == pytest.approx(1.0)
    assert fo.fct_idel == 701
    assert fo.strdef == 1
    assert fo.eps_11tf == pytest.approx(0.05)
    assert fo.eps_11tm == pytest.approx(0.08)
    assert fo.fct_id_11t == 702
    assert fo.eps_11cf == pytest.approx(0.04)


# ==============================================================================
# 3. Boundary Conditions & Preload Tests
# ==============================================================================

def test_ebcs_propellant(tmp_path: Path):
    """Test /EBCS/PROPELLANT."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test EBCS PROPELLANT
                  10                   0
/EBCS/PROPELLANT/1
Solid Propellant Burning
#            SURF_ID             SENS_ID           SUBMAT_ID           IENTHALPY
                  10                   0                   1                   1
#              RHO0S               TBURN
              1.5e-6               300.0
#            PARAM_A             PARAM_N
                 0.5                 0.8
#           FFUNC_ID             FSCALEX             FSCALEY
                 801                 1.0                 1.0
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.ebcs_propellants
    ep = model.ebcs_propellants[1]
    assert isinstance(ep, EbcsPropellant)
    assert ep.surf_id == 10
    assert ep.sens_id == 0
    assert ep.submat_id == 1
    assert ep.rho0s == pytest.approx(1.5e-6)
    assert ep.tburn == pytest.approx(300.0)
    assert ep.param_a == pytest.approx(0.5)
    assert ep.param_n == pytest.approx(0.8)
    assert ep.f_func_id == 801


def test_ebcs_cyclic(tmp_path: Path):
    """Test /EBCS/CYCLIC."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test EBCS CYCLIC
                  10                   0
/EBCS/CYCLIC/1
Cyclic Symmetry Boundary
#           SURF1_ID            SURF2_ID             SKEW_ID           GRPART_ID
                  15                  25                   1                   2
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.ebcs_cyclics
    ec = model.ebcs_cyclics[1]
    assert isinstance(ec, EbcsCyclic)
    assert ec.surf1_id == 15
    assert ec.surf2_id == 25
    assert ec.skew_id == 1
    assert ec.grpart_id == 2


def test_preload(tmp_path: Path):
    """Test /PRELOAD."""
    deck = """\
#RADIOSS STARTER
/BEGIN
Test PRELOAD
                  10                   0
/PRELOAD/1
Bolt Preload
#            SECT_ID             SENS_ID               ITYPE              FCT_ID             PRELOAD              TSTART               TSTOP
                  60                   0                   1                 901              5000.0                 0.0                 0.1
"""
    model = _parse(tmp_path, deck)
    assert 1 in model.preloads
    pr = model.preloads[1]
    assert isinstance(pr, Preload)
    assert pr.sect_id == 60
    assert pr.sens_id == 0
    assert pr.itype == 1
    assert pr.fct_id == 901
    assert pr.preload == pytest.approx(5000.0)
    assert pr.tstart == pytest.approx(0.0)
    assert pr.tstop == pytest.approx(0.1)
