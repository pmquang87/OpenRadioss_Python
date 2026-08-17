"""Tests for Milestone M193:
- Failure models: /FAIL/EMC, /FAIL/NXT, /FAIL/TBUTCHER, /FAIL/MULLINS (/FAIL/MULLINS_OR),
  /FAIL/COCKCROFT, /FAIL/GENE1, /FAIL/XFEM_FLD, /FAIL/XFEM_JOHNS, /FAIL/XFEM_TBUTC
- Material laws: /MAT/LAW53 (/MAT/TSAI_TAB), /MAT/LAW54 (/MAT/PREDIT),
  /MAT/LAW74 (/MAT/HILL_THERM), /MAT/LAW82 (/MAT/OGDEN)
- Property models: /PROP/TYPE18, /PROP/INT_BEAM
- Interface defaults: /DEF_INTER/TYPE11, /DEF_INTER/TYPE19, /DEF_INTER/TYPE25, /DEFAULT/INTER/...
- Element & Node state directives: /STATE/BEAM, /STATE/BRICK, /STATE/NODE, /STATE/SHELL, /STATE/SPRING, /STATE/TRUSS
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_fail_emc_fixed_and_free(tmp_path: Path):
    # Fixed format (no title card for /FAIL)
    deck_fixed = """# RADIOSS STARTER
/BEGIN
EMC Fixed
                 10                   0
/FAIL/EMC/1
#                  A                   N                  B0                   C
                 1.5                 2.5                 3.5                 4.5
#              GAMMA       EPSILON_DOT_0
                 5.5                 6.5
#  FAIL_ID
           1
/END
"""
    model, log = _parse(tmp_path, deck_fixed)
    assert 1 in model.fail_emcs
    f = model.fail_emcs[1]
    assert f.a_emc == 1.5
    assert f.n_emc == 2.5
    assert f.b0 == 3.5
    assert f.c == 4.5
    assert f.gamma == 5.5
    assert f.epsilon_dot_0 == 6.5
    assert f.fail_id == 1

    # Free format
    deck_free = """# RADIOSS STARTER
/BEGIN
EMC Free
                 10                   0
/FAIL/EMC/2
1.1, 2.2, 3.3, 4.4
5.5, 6.6
2
/END
"""
    model, log = _parse(tmp_path, deck_free)
    assert 2 in model.fail_emcs
    f2 = model.fail_emcs[2]
    assert f2.a_emc == 1.1
    assert f2.n_emc == 2.2
    assert f2.b0 == 3.3
    assert f2.c == 4.4
    assert f2.gamma == 5.5
    assert f2.epsilon_dot_0 == 6.6
    assert f2.fail_id == 2


def test_fail_nxt_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """# RADIOSS STARTER
/BEGIN
NXT Fixed
                 10                   0
/FAIL/NXT/1
#   FCT_ID_1  FCT_ID_2            IFAIL_
           1         2                 3
#  FAIL_ID
           4
/END
"""
    model, log = _parse(tmp_path, deck_fixed)
    assert 1 in model.fail_nxts
    f = model.fail_nxts[1]
    assert f.fct_id1 == 1
    assert f.fct_id2 == 2
    assert f.ifail_sh == 3
    assert f.fail_id == 4

    # Free format
    deck_free = """# RADIOSS STARTER
/BEGIN
NXT Free
                 10                   0
/FAIL/NXT/2
1, 0, 1
2
/END
"""
    model, log = _parse(tmp_path, deck_free)
    assert 2 in model.fail_nxts
    f2 = model.fail_nxts[2]
    assert f2.fct_id1 == 1
    assert f2.fct_id2 == 0
    assert f2.ifail_sh == 1
    assert f2.fail_id == 2


def test_fail_tbutcher_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """# RADIOSS STARTER
/BEGIN
TB Fixed
                 10                   0
/FAIL/TBUTCHER/1
#             LAMBDA                   K             SIGMA_R  IFAIL_SH  IFAIL_SO     IDUCT     IXFEM
               100.0                 2.0               0.005         1         2         3         4
#                  A                   B                DADV
                 0.1                 0.2                 0.3
#  FAIL_ID
           5
/END
"""
    model, log = _parse(tmp_path, deck_fixed)
    assert 1 in model.fail_tbutchers
    f = model.fail_tbutchers[1]
    assert f.lam == 100.0
    assert f.k == 2.0
    assert f.sigma_r == 0.005
    assert f.ifail_sh == 1
    assert f.ifail_so == 2
    assert f.iduct == 3
    assert f.ixfem == 4
    assert f.a == 0.1
    assert f.b == 0.2
    assert f.dadv == 0.3
    assert f.fail_id == 5

    # Free format
    deck_free = """# RADIOSS STARTER
/BEGIN
TB Free
                 10                   0
/FAIL/TBUTCHER/2
200.0, 1.5, 0.01, 1, 1, 1, 2
0.05, 0.06, 0.07
0
/END
"""
    model, log = _parse(tmp_path, deck_free)
    assert 2 in model.fail_tbutchers
    f2 = model.fail_tbutchers[2]
    assert f2.lam == 200.0
    assert f2.k == 1.5
    assert f2.sigma_r == 0.01
    assert f2.ifail_sh == 1
    assert f2.ifail_so == 1
    assert f2.iduct == 1
    assert f2.ixfem == 2
    assert f2.a == 0.05
    assert f2.b == 0.06
    assert f2.dadv == 0.07
    assert f2.fail_id == 0


def test_fail_mullins_fixed_and_free(tmp_path: Path):
    # Fixed format with alias /FAIL/MULLINS_OR
    deck_fixed = """# RADIOSS STARTER
/BEGIN
Mullins Fixed
                 10                   0
/FAIL/MULLINS_OR/1
#                  R                BETA                   M
                 0.5                 1.2                 2.3
#  FAIL_ID
           1
/END
"""
    model, log = _parse(tmp_path, deck_fixed)
    assert 1 in model.fail_mullins
    f = model.fail_mullins[1]
    assert f.coefr == 0.5
    assert f.beta == 1.2
    assert f.coefm == 2.3
    assert f.fail_id == 1

    # Free format with /FAIL/MULLINS
    deck_free = """# RADIOSS STARTER
/BEGIN
Mullins Free
                 10                   0
/FAIL/MULLINS/2
0.8, 2.0, 1.5
2
/END
"""
    model, log = _parse(tmp_path, deck_free)
    assert 2 in model.fail_mullins
    f2 = model.fail_mullins[2]
    assert f2.coefr == 0.8
    assert f2.beta == 2.0
    assert f2.coefm == 1.5
    assert f2.fail_id == 2


def test_fail_cockcroft_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """# RADIOSS STARTER
/BEGIN
Cockcroft Fixed
                 10                   0
/FAIL/COCKCROFT/1
#                 C0               ALPHA              FAILIP
                 5.0                 0.8                   1
#  FAIL_ID
           3
/END
"""
    model, log = _parse(tmp_path, deck_fixed)
    assert 1 in model.fail_cockcrofts
    f = model.fail_cockcrofts[1]
    assert f.c0 == 5.0
    assert f.alpha == 0.8
    assert f.failip == 1
    assert f.fail_id == 3

    # Free format
    deck_free = """# RADIOSS STARTER
/BEGIN
Cockcroft Free
                 10                   0
/FAIL/COCKCROFT/2
10.5, 0.9, 2
1
/END
"""
    model, log = _parse(tmp_path, deck_free)
    assert 2 in model.fail_cockcrofts
    f2 = model.fail_cockcrofts[2]
    assert f2.c0 == 10.5
    assert f2.alpha == 0.9
    assert f2.failip == 2
    assert f2.fail_id == 1


def test_fail_gene1_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """# RADIOSS STARTER
/BEGIN
Gene1 Fixed
                 10                   0
/FAIL/GENE1/1
#               PMIN                PMAX           SIGP1_MAX                TMAX               DTMIN
                 1.0                 2.0                 3.0                 4.0                 5.0
#  FCT_ID_SM                       EPS_DOT_SM             SIG_MAX                SIGR                  K
         101                                 0.1                10.0                 5.0                 0.5
#  FCT_ID_PS                       EPS_DOT_PS             EPS_MAX             EPS_EFF             EPS_VOL
         102                                 0.2                20.0                 0.3                 0.4
#            EPS_MIN              EPS_SH          FCT_ID_G12          FCT_ID_G13          FCT_ID_E1C
                 0.1                 0.2                   1                   2                   3
# TAB_ID_FLD      ITAB        EPS_DOT_FLD     NSTEP   ISMOOTH   ISTRAIN                      THINNING
           4         1                0.3         5         1         0                           0.5
#            VOLFRAC                PTHK                 NCS                                TEMP_MAX      FAILIP
                 0.2                 0.1                   2                                   150.0           1
#  FCT_ID_EL                       FSCALE_EL              EL_REF
         103                             1.2                 0.5
#  FAIL_ID
           2
/END
"""
    model, log = _parse(tmp_path, deck_fixed)
    assert 1 in model.fail_gene1s
    f = model.fail_gene1s[1]
    assert f.pmin == 1.0
    assert f.pmax == 2.0
    assert f.sigp1_max == 3.0
    assert f.tmax == 4.0
    assert f.dtmin == 5.0
    assert f.fct_idsm == 101
    assert f.eps_dot_sm == 0.1
    assert f.sig_max == 10.0
    assert f.fct_idps == 102
    assert f.eps_dot_ps == 0.2
    assert f.eps_min == 0.1
    assert f.tab_idfld == 4
    assert f.temp_max == 150.0
    assert f.fct_idel == 103
    assert f.fscale_el == 1.2
    assert f.fail_id == 2

    # Free format
    deck_free = """# RADIOSS STARTER
/BEGIN
Gene1 Free
                 10                   0
/FAIL/GENE1/2
1, 2, 3, 4, 5
101, 0.1, 10, 5, 0.5
102, 0.2, 20, 0.3, 0.4
0.1, 0.2, 1, 2, 3
4, 1, 0.3, 5, 1, 0, 0.5
0.2, 0.1, 2, 150, 1
103, 1.2, 0.5
2
/END
"""
    model, log = _parse(tmp_path, deck_free)
    assert 2 in model.fail_gene1s
    f2 = model.fail_gene1s[2]
    assert f2.pmin == 1.0
    assert f2.fct_idsm == 101
    assert f2.fail_id == 2


def test_fail_xfem_models(tmp_path: Path):
    deck = """# RADIOSS STARTER
/BEGIN
XFEM Models
                 10                   0
/FAIL/XFEM_FLD/1
#             FCT_ID            IFAIL_SH
                 101                   1
/FAIL/XFEM_JOHNS/2
#                 D1                  D2                  D3                  D4                  D5
                0.05               300.0                 0.1                 0.2                 0.3
#      EPS_DOT_0            IFAIL_SH
             1.0                   1
/FAIL/XFEM_TBUTC/3
#             LAMBDA                   K             SIGMA_R            IFAIL_SH               IDUCT
               250.0                 1.8               100.0                   1                   2
#                  A                   B
                 0.5                 0.6
/END
"""
    model, log = _parse(tmp_path, deck)
    assert 1 in model.fail_xfems
    assert model.fail_xfems[1].model_name == "XFEM_FLD"
    assert model.fail_xfems[1].fct_id == 101

    assert 2 in model.fail_xfems
    assert model.fail_xfems[2].model_name == "XFEM_JOHNS"
    assert model.fail_xfems[2].eps == 0.05
    assert model.fail_xfems[2].sigma == 300.0

    assert 3 in model.fail_xfems
    assert model.fail_xfems[3].model_name == "XFEM_TBUTC"
    assert model.fail_xfems[3].sig0 == 250.0
    assert model.fail_xfems[3].lam == 1.8


def test_mat_law53_tsai_tab(tmp_path: Path):
    # Fixed format with alias /MAT/TSAI_TAB
    deck_fixed = """# RADIOSS STARTER
/BEGIN
Tsai Fixed
                 10                   0
/MAT/TSAI_TAB/1
Tsai Tabulated
#              RHO_I           REFER_RHO
              7.8E-9              7.8E-9
#                EA1                 EA2
              150000              100000
#                GAB                 GBC
               50000               30000
#             FUN_A1              FUN_B1              FUN_A3              FUN_A5              FUN_A6
                   1                   2                   3                   4                   5
#            SFAC_11             SFAC_22             SFAC_12             SFAC_23             SFAC_45
                 1.0                 1.1                 1.2                 1.3                 1.4
/END
"""
    model, log = _parse(tmp_path, deck_fixed)
    assert 1 in model.mat_law53s
    m = model.mat_law53s[1]
    assert m.rho == 7.8e-9
    assert m.e1 == 150000.0
    assert m.e2 == 100000.0
    assert m.gab == 50000.0
    assert m.gbc == 30000.0
    assert m.fun_a1 == 1
    assert m.fun_a6 == 5
    assert m.sfac11 == 1.0
    assert m.sfac45 == 1.4
    assert 1 in model.materials
    assert model.materials[1].law == 53

    # Free format with /MAT/LAW53
    deck_free = """# RADIOSS STARTER
/BEGIN
Law 53 Free
                 10                   0
/MAT/LAW53/2
Law 53 Free
7.8E-9, 7.8E-9
120000, 90000
45000, 25000
10, 20, 30, 40, 50
1.0, 1.0, 1.0, 1.0, 1.0
/END
"""
    model, log = _parse(tmp_path, deck_free)
    assert 2 in model.mat_law53s
    m2 = model.mat_law53s[2]
    assert m2.e1 == 120000.0
    assert m2.fun_a1 == 10


def test_mat_law54_predit(tmp_path: Path):
    # Fixed format with alias /MAT/PREDIT
    deck_fixed = """# RADIOSS STARTER
/BEGIN
Predit Fixed
                 10                   0
/MAT/PREDIT/1
Predit Concrete
#              RHO_I           REFER_RHO
              2.4E-9              2.4E-9
#                  E                  NU
               30000                 0.2
#              IFUNC                   A                   B                   N                SFAC
                   1                 1.5                 2.5                 0.5                 1.0
#                 AY                  AZ                  BY                  BZ                  CX
                 0.1                 0.2                 0.3                 0.4                 0.5
#                 DC                  RC             EPS_MAX
                 0.6                 0.7                 0.8
/END
"""
    model, log = _parse(tmp_path, deck_fixed)
    assert 1 in model.mat_law54s
    m = model.mat_law54s[1]
    assert m.rho == 2.4e-9
    assert m.e == 30000.0
    assert m.nu == 0.2
    assert m.ifunc == 1
    assert m.a == 1.5
    assert m.b == 2.5
    assert m.ay == 0.1
    assert m.dc == 0.6
    assert m.eps_max == 0.8
    assert 1 in model.materials
    assert model.materials[1].law == 54

    # Free format with /MAT/LAW54
    deck_free = """# RADIOSS STARTER
/BEGIN
Law 54 Free
                 10                   0
/MAT/LAW54/2
Law 54 Free
2.4E-9, 2.4E-9
35000, 0.18
2, 1.8, 2.8, 0.6, 1.0
0.15, 0.25, 0.35, 0.45, 0.55
0.65, 0.75, 0.85
/END
"""
    model, log = _parse(tmp_path, deck_free)
    assert 2 in model.mat_law54s
    m2 = model.mat_law54s[2]
    assert m2.e == 35000.0
    assert m2.ifunc == 2
    assert m2.eps_max == 0.85


def test_mat_law74_hill_therm(tmp_path: Path):
    # Fixed format with alias /MAT/HILL_THERM
    deck_fixed = """# RADIOSS STARTER
/BEGIN
Hill Therm Fixed
                 10                   0
/MAT/HILL_THERM/1
Hill Thermal Orthotropic
#              RHO_I           REFER_RHO
              7.8E-9              7.8E-9
#                  E                  NU           EPS_P_MAX               EPS_T               EPS_M
              210000                0.28                0.05                0.01                0.02
#            FSMOOTH              C_HARD                FCUT
                   1               100.0                10.0
#             SIG11Y              SIG22Y              SIG33Y
               500.0               450.0               400.0
#             SIG12Y              SIG23Y              SIG31Y
               250.0               200.0               200.0
#             TAB_ID                         SIGMA_SCALE         EPSPT_SCALE
                 101                                 1.0                 1.0
#                 TI             RHO0_CP
               293.0               450.0
/END
"""
    model, log = _parse(tmp_path, deck_fixed)
    assert 1 in model.mat_law74s
    m = model.mat_law74s[1]
    assert m.rho == 7.8e-9
    assert m.e == 210000.0
    assert m.nu == 0.28
    assert m.eps_p_max == 0.05
    assert m.fsmooth == 1
    assert m.c_hard == 100.0
    assert m.sig11y == 500.0
    assert m.sig12y == 250.0
    assert m.tab_id == 101
    assert m.ti == 293.0
    assert m.rho0_cp == 450.0
    assert 1 in model.materials
    assert model.materials[1].law == 74

    # Free format with /MAT/LAW74
    deck_free = """# RADIOSS STARTER
/BEGIN
Law 74 Free
                 10                   0
/MAT/LAW74/2
Law 74 Free
7.8E-9, 7.8E-9
220000, 0.29, 0.06, 0.02, 0.03
0, 120.0, 15.0
520.0, 470.0, 420.0
260.0, 210.0, 210.0
102, 1.0, 1.0
300.0, 460.0
/END
"""
    model, log = _parse(tmp_path, deck_free)
    assert 2 in model.mat_law74s
    m2 = model.mat_law74s[2]
    assert m2.e == 220000.0
    assert m2.c_hard == 120.0
    assert m2.ti == 300.0


def test_mat_law82_ogden(tmp_path: Path):
    # Fixed format with /MAT/LAW82
    deck_fixed = """# RADIOSS STARTER
/BEGIN
Ogden Fixed
                 10                   0
/MAT/LAW82/1
Ogden Rubber
#              RHO_I           REFER_RHO
              1.0E-9              1.0E-9
#          N                          NU
           2                       0.495
#               MU_1                MU_2
                 0.5                 0.2
#            ALPHA_1             ALPHA_2
                 2.0                -2.0
#            GAMMA_1             GAMMA_2
                 0.0                 0.0
/END
"""
    model, log = _parse(tmp_path, deck_fixed)
    assert 1 in model.mat_law82s
    m = model.mat_law82s[1]
    assert m.rho == 1.0e-9
    assert m.order == 2
    assert m.nu == 0.495
    assert m.mu_arr == [0.5, 0.2]
    assert m.alpha_arr == [2.0, -2.0]
    assert m.gamma_arr == [0.0, 0.0]
    assert 1 in model.materials
    assert model.materials[1].law == 82

    # Free format with /MAT/LAW82
    deck_free = """# RADIOSS STARTER
/BEGIN
Law 82 Free
                 10                   0
/MAT/LAW82/2
Law 82 Free
1.2E-9, 1.2E-9
3, 0.498
0.6, 0.3, 0.1
1.5, -1.5, 3.0
0.01, 0.02, 0.03
/END
"""
    model, log = _parse(tmp_path, deck_free)
    assert 2 in model.mat_law82s
    m2 = model.mat_law82s[2]
    assert m2.order == 3
    assert m2.mu_arr == [0.6, 0.3, 0.1]
    assert m2.alpha_arr == [1.5, -1.5, 3.0]
    assert m2.gamma_arr == [0.01, 0.02, 0.03]


def test_prop_int_beam_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """# RADIOSS STARTER
/BEGIN
Prop Int Beam Fixed
                 10                   0
/PROP/TYPE18/1
Integrated Beam Property
#     ISFLAG      ISMSTR
           0           0
#                 DM                  DF
                0.01                0.02
#        NIP        IREF                  Y0                  Z0
           2           1                 0.5                 0.5
#                YIP                 ZIP                AREA
                 0.1                 0.2                10.0
#                YIP                 ZIP                AREA
                 0.3                 0.4                15.0
/END
"""
    model, log = _parse(tmp_path, deck_fixed)
    assert 1 in model.prop_int_beams
    p = model.prop_int_beams[1]
    assert p.isflag == 0
    assert p.ismstr == 0
    assert p.dm == 0.01
    assert p.df == 0.02
    assert p.nip == 2
    assert p.iref == 1
    assert p.y0 == 0.5
    assert p.z0 == 0.5
    assert len(p.ips) == 2
    assert p.ips[0].y == 0.1
    assert p.ips[0].z == 0.2
    assert p.ips[0].area == 10.0
    assert 1 in model.properties
    assert model.properties[1].type == 18

    # Free format with alias /PROP/INT_BEAM
    deck_free = """# RADIOSS STARTER
/BEGIN
Int Beam Free
                 10                   0
/PROP/INT_BEAM/2
Int Beam Free
0, 1
0.05, 0.06
1, 0, 0.0, 0.0
0.2, 0.3, 20.0
/END
"""
    model, log = _parse(tmp_path, deck_free)
    assert 2 in model.prop_int_beams
    p2 = model.prop_int_beams[2]
    assert p2.isflag == 0
    assert p2.nip == 1
    assert len(p2.ips) == 1
    assert p2.ips[0].y == 0.2


def test_def_inter_types_11_19_25(tmp_path: Path):
    # /DEF_INTER/TYPE11
    deck_11 = """# RADIOSS STARTER
/BEGIN
Def Inter 11
                 10                   0
/DEF_INTER/TYPE11
#                         Istf                Igap            Irem_gap      Idel
                             2                   3                   4         5
#                                                                                    Iform
                                                                                         1
#                                Inactiv
                                       0
/END
"""
    model, log = _parse(tmp_path, deck_11)
    assert model.def_inter_type11 is not None
    assert model.def_inter_type11.istf == 2
    assert model.def_inter_type11.igap == 3
    assert model.def_inter_type11.ikrem == 4
    assert model.def_inter_type11.noddel11 == 5

    # /DEF_INTER/TYPE19
    deck_19 = """# RADIOSS STARTER
/BEGIN
Def Inter 19
                 10                   0
/DEF_INTER/TYPE19
#       ISTF        IGAP       IEDGE        IBAG       IDEL7       ICURV
           1           2           3           4           5           6
#    INACTIV
           1
#      IFORM
           2
/END
"""
    model, log = _parse(tmp_path, deck_19)
    assert model.def_inter_type19 is not None
    assert model.def_inter_type19.istf == 1
    assert model.def_inter_type19.igap == 2
    assert model.def_inter_type19.iedge == 3
    assert model.def_inter_type19.ibag == 4
    assert model.def_inter_type19.idel7 == 5
    assert model.def_inter_type19.icurv == 6

    # /DEF_INTER/TYPE25
    deck_25 = """# RADIOSS STARTER
/BEGIN
Def Inter 25
                 10                   0
/DEF_INTER/TYPE25
#       ISTF        IGAP     IREM_I2        IDEL
           2           1           1           3
#      ITIED      ISHAPE
           1           2
#        IRS
           4
/END
"""
    model, log = _parse(tmp_path, deck_25)
    assert model.def_inter_type25 is not None
    assert model.def_inter_type25.istf == 2
    assert model.def_inter_type25.igap == 1
    assert model.def_inter_type25.irem_i2 == 1
    assert model.def_inter_type25.idel == 3
    assert model.def_inter_type25.itied == 1
    assert model.def_inter_type25.ishape == 2
    assert model.def_inter_type25.irs == 4


def test_state_directives(tmp_path: Path):
    deck = """# RADIOSS STARTER
/BEGIN
State Directives
                 10                   0
/STATE/BEAM/STRESS
/STATE/BRICK/STRAIN
/STATE/NODE/VEL
/STATE/SHELL/THICK/1.5
/STATE/SPRING/FORCE
/STATE/TRUSS/STRESS
/END
"""
    model, log = _parse(tmp_path, deck)
    assert len(model.state_directives) == 6
    kinds = [sd.kind for sd in model.state_directives]
    subtypes = [sd.subtype for sd in model.state_directives]
    assert "BEAM" in kinds
    assert "BRICK" in kinds
    assert "NODE" in kinds
    assert "SHELL" in kinds
    assert "SPRING" in kinds
    assert "TRUSS" in kinds
    assert "STRESS" in subtypes
    assert "STRAIN" in subtypes
    assert "VEL" in subtypes
    assert "THICK" in subtypes
    shell_sd = [sd for sd in model.state_directives if sd.kind == "SHELL"][0]
    assert shell_sd.val == 1.5
