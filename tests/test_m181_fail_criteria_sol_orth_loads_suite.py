"""Unit tests for Milestone M181:
- /FAIL/FABRIC & /FAIL/FABR
- /FAIL/HOFFMAN
- /FAIL/MAX_STRAIN & /FAIL/MAXSTRAIN
- /FAIL/TSAI_HILL & /FAIL/TSAIHILL
- /FAIL/TSAI_WU & /FAIL/TSAIWU
- /PROP/TYPE6 & /PROP/SOL_ORTH
- /LOAD/CLOAD & /CLOAD
- /LOAD/PLOAD & /PLOAD
- Both fixed-column format and free-format parsing
"""

import pytest
from pathlib import Path
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog


def _parse_deck(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    deck_path = tmp_path / "TEST_0000.rad"
    deck_path.write_text(text, encoding="ascii")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(deck_path))
    parse_starter_deck(blocks, model, log)
    return model, log


def test_fail_fabric_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """\
# OpenRadioss Starter Deck
/BEGIN
Test FAIL FABRIC Fixed
      2022         0
/MAT/LAW1/1
Mat1
#              RHO
             1.0E-6
#                E                    NU
           210000.0                   0.3
/FAIL/FABRIC/1/101
#         EPSILON_F1          EPSILON_R1          EPSILON_F2          EPSILON_R2                NDIR
                 0.1                 0.2                 0.3                 0.4                   2
#             FCT_ID
                  55
#  FAIL_ID
       101
"""
    model, log = _parse_deck(tmp_path, deck_fixed)

    assert 1 in model.fail_fabrics
    ff = model.fail_fabrics[1]
    assert ff.id == 101
    assert ff.mat_id == 1
    assert pytest.approx(ff.epsilon_f1) == 0.1
    assert pytest.approx(ff.epsilon_r1) == 0.2
    assert pytest.approx(ff.epsilon_f2) == 0.3
    assert pytest.approx(ff.epsilon_r2) == 0.4
    assert ff.ndir == 2
    assert ff.fct_id == 55
    assert ff.fail_id == 101

    # Free format
    deck_free = """\
/BEGIN
Test FAIL FABRIC Free
/MAT/LAW1/2
Mat2
1.0e-6
210000.0 0.3
/FAIL/FABRIC/2
0.15 0.25 0.35 0.45 1
66
202
"""
    model2, log2 = _parse_deck(tmp_path, deck_free)

    assert 2 in model2.fail_fabrics
    ff2 = model2.fail_fabrics[2]
    assert ff2.mat_id == 2
    assert pytest.approx(ff2.epsilon_f1) == 0.15
    assert pytest.approx(ff2.epsilon_r1) == 0.25
    assert pytest.approx(ff2.epsilon_f2) == 0.35
    assert pytest.approx(ff2.epsilon_r2) == 0.45
    assert ff2.ndir == 1
    assert ff2.fct_id == 66
    assert ff2.fail_id == 202


def test_fail_hoffman_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """\
# OpenRadioss Starter Deck
/BEGIN
Test FAIL HOFFMAN Fixed
      2022         0
/MAT/LAW1/10
Mat10
#              RHO
             1.0E-6
#                E                    NU
           210000.0                   0.3
/FAIL/HOFFMAN/10/301
#           SIGMA_1T            SIGMA_2T            SIGMA_1C            SIGMA_2C            SIGMA_12
               500.0                50.0               600.0                80.0                70.0
#            TAU_MAX                FCUT                                 IFAIL_SH  IFAIL_SO
              0.0001              1000.0                                        2         1
#  FAIL_ID
       301
"""
    model, log = _parse_deck(tmp_path, deck_fixed)

    assert 10 in model.fail_hoffmans
    fh = model.fail_hoffmans[10]
    assert fh.id == 301
    assert fh.mat_id == 10
    assert pytest.approx(fh.sigma_1t) == 500.0
    assert pytest.approx(fh.sigma_2t) == 50.0
    assert pytest.approx(fh.sigma_1c) == 600.0
    assert pytest.approx(fh.sigma_2c) == 80.0
    assert pytest.approx(fh.sigma_12) == 70.0
    assert pytest.approx(fh.tau_max) == 0.0001
    assert pytest.approx(fh.fcut) == 1000.0
    assert fh.ifail_sh == 2
    assert fh.ifail_so == 1
    assert fh.fail_id == 301

    # Free format
    deck_free = """\
/BEGIN
Test FAIL HOFFMAN Free
/MAT/LAW1/11
Mat11
1.0e-6
210000.0 0.3
/FAIL/HOFFMAN/11
400.0 40.0 500.0 70.0 60.0
0.0002 2000.0 1 2
302
"""
    model2, log2 = _parse_deck(tmp_path, deck_free)

    assert 11 in model2.fail_hoffmans
    fh2 = model2.fail_hoffmans[11]
    assert fh2.mat_id == 11
    assert pytest.approx(fh2.sigma_1t) == 400.0
    assert pytest.approx(fh2.sigma_2t) == 40.0
    assert pytest.approx(fh2.sigma_1c) == 500.0
    assert pytest.approx(fh2.sigma_2c) == 70.0
    assert pytest.approx(fh2.sigma_12) == 60.0
    assert pytest.approx(fh2.tau_max) == 0.0002
    assert pytest.approx(fh2.fcut) == 2000.0
    assert fh2.ifail_sh == 1
    assert fh2.ifail_so == 2
    assert fh2.fail_id == 302


def test_fail_maxstrain_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """\
# OpenRadioss Starter Deck
/BEGIN
Test FAIL MAXSTRAIN Fixed
      2022         0
/MAT/LAW1/20
Mat20
#              RHO
             1.0E-6
#                E                    NU
           210000.0                   0.3
/FAIL/MAX_STRAIN/20/401
#           EPS1_MAX            EPS2_MAX           GAM12_MAX                      IFAIL_SH  IFAIL_SO
                0.05                0.02                0.03                             1         2
#            TAU_MAX                FCUT
              0.0005              1500.0
#  FAIL_ID
       401
"""
    model, log = _parse_deck(tmp_path, deck_fixed)

    assert 20 in model.fail_maxstrains
    fms = model.fail_maxstrains[20]
    assert fms.id == 401
    assert fms.mat_id == 20
    assert pytest.approx(fms.eps1_max) == 0.05
    assert pytest.approx(fms.eps2_max) == 0.02
    assert pytest.approx(fms.gam12_max) == 0.03
    assert fms.ifail_sh == 1
    assert fms.ifail_so == 2
    assert pytest.approx(fms.tau_max) == 0.0005
    assert pytest.approx(fms.fcut) == 1500.0
    assert fms.fail_id == 401

    # Free format
    deck_free = """\
/BEGIN
Test FAIL MAXSTRAIN Free
/MAT/LAW1/21
Mat21
1.0e-6
210000.0 0.3
/FAIL/MAXSTRAIN/21
0.04 0.015 0.025 2 1
0.0003 1200.0
402
"""
    model2, log2 = _parse_deck(tmp_path, deck_free)

    assert 21 in model2.fail_maxstrains
    fms2 = model2.fail_maxstrains[21]
    assert fms2.mat_id == 21
    assert pytest.approx(fms2.eps1_max) == 0.04
    assert pytest.approx(fms2.eps2_max) == 0.015
    assert pytest.approx(fms2.gam12_max) == 0.025
    assert fms2.ifail_sh == 2
    assert fms2.ifail_so == 1
    assert pytest.approx(fms2.tau_max) == 0.0003
    assert pytest.approx(fms2.fcut) == 1200.0
    assert fms2.fail_id == 402


def test_fail_tsaihill_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """\
# OpenRadioss Starter Deck
/BEGIN
Test FAIL TSAIHILL Fixed
      2022         0
/MAT/LAW1/30
Mat30
#              RHO
             1.0E-6
#                E                    NU
           210000.0                   0.3
/FAIL/TSAI_HILL/30/501
#                X11                 X22                 S12                      IFAIL_SH  IFAIL_SO
               800.0                80.0                90.0                             2         1
#            TAU_MAX                FCUT
              0.0004              2500.0
#  FAIL_ID
       501
"""
    model, log = _parse_deck(tmp_path, deck_fixed)

    assert 30 in model.fail_tsaihills
    fth = model.fail_tsaihills[30]
    assert fth.id == 501
    assert fth.mat_id == 30
    assert pytest.approx(fth.x11) == 800.0
    assert pytest.approx(fth.x22) == 80.0
    assert pytest.approx(fth.s12) == 90.0
    assert fth.ifail_sh == 2
    assert fth.ifail_so == 1
    assert pytest.approx(fth.tau_max) == 0.0004
    assert pytest.approx(fth.fcut) == 2500.0
    assert fth.fail_id == 501

    # Free format
    deck_free = """\
/BEGIN
Test FAIL TSAIHILL Free
/MAT/LAW1/31
Mat31
1.0e-6
210000.0 0.3
/FAIL/TSAIHILL/31
750.0 70.0 85.0 1 0
0.0002 1800.0
502
"""
    model2, log2 = _parse_deck(tmp_path, deck_free)

    assert 31 in model2.fail_tsaihills
    fth2 = model2.fail_tsaihills[31]
    assert fth2.mat_id == 31
    assert pytest.approx(fth2.x11) == 750.0
    assert pytest.approx(fth2.x22) == 70.0
    assert pytest.approx(fth2.s12) == 85.0
    assert fth2.ifail_sh == 1
    assert fth2.ifail_so == 0
    assert pytest.approx(fth2.tau_max) == 0.0002
    assert pytest.approx(fth2.fcut) == 1800.0
    assert fth2.fail_id == 502


def test_fail_tsaiwu_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """\
# OpenRadioss Starter Deck
/BEGIN
Test FAIL TSAIWU Fixed
      2022         0
/MAT/LAW1/40
Mat40
#              RHO
             1.0E-6
#                E                    NU
           210000.0                   0.3
/FAIL/TSAI_WU/40/601
#           SIGMA_1T            SIGMA_2T            SIGMA_1C            SIGMA_2C            SIGMA_12
              1000.0               100.0              1200.0               150.0               120.0
#              ALPHA             TAU_MAX                FCUT                      IFAIL_SH  IFAIL_SO
                -0.5              0.0001              3000.0                             1         2
#  FAIL_ID
       601
"""
    model, log = _parse_deck(tmp_path, deck_fixed)

    assert 40 in model.fail_tsaiwus
    ftw = model.fail_tsaiwus[40]
    assert ftw.id == 601
    assert ftw.mat_id == 40
    assert pytest.approx(ftw.sigma_1t) == 1000.0
    assert pytest.approx(ftw.sigma_2t) == 100.0
    assert pytest.approx(ftw.sigma_1c) == 1200.0
    assert pytest.approx(ftw.sigma_2c) == 150.0
    assert pytest.approx(ftw.sigma_12) == 120.0
    assert pytest.approx(ftw.alpha) == -0.5
    assert pytest.approx(ftw.tau_max) == 0.0001
    assert pytest.approx(ftw.fcut) == 3000.0
    assert ftw.ifail_sh == 1
    assert ftw.ifail_so == 2
    assert ftw.fail_id == 601

    # Free format
    deck_free = """\
/BEGIN
Test FAIL TSAIWU Free
/MAT/LAW1/41
Mat41
1.0e-6
210000.0 0.3
/FAIL/TSAIWU/41
900.0 90.0 1100.0 130.0 110.0
-0.4 0.0002 2200.0 2 1
602
"""
    model2, log2 = _parse_deck(tmp_path, deck_free)

    assert 41 in model2.fail_tsaiwus
    ftw2 = model2.fail_tsaiwus[41]
    assert ftw2.mat_id == 41
    assert pytest.approx(ftw2.sigma_1t) == 900.0
    assert pytest.approx(ftw2.sigma_2t) == 90.0
    assert pytest.approx(ftw2.sigma_1c) == 1100.0
    assert pytest.approx(ftw2.sigma_2c) == 130.0
    assert pytest.approx(ftw2.sigma_12) == 110.0
    assert pytest.approx(ftw2.alpha) == -0.4
    assert pytest.approx(ftw2.tau_max) == 0.0002
    assert pytest.approx(ftw2.fcut) == 2200.0
    assert ftw2.ifail_sh == 2
    assert ftw2.ifail_so == 1
    assert ftw2.fail_id == 602


def test_prop_sol_orth_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """\
# OpenRadioss Starter Deck
/BEGIN
Test PROP SOL_ORTH Fixed
      2022         0
/PROP/SOL_ORTH/6
Solid Orthotropic Property Test
#   Isolid    Ismstr               Icpre  Itetra10     Inpts   Itetra4    Iframe                  Dn
        14         1                   0         0       222         0         1                 0.1
#                 qa                  qb                   h
                 1.2                0.04                 0.2
#                 Vx                  Vy                  Vz   skew_ID        Ip     Iorth
                 1.0                 0.0                 0.0        12         1         2
#                Phi                  Px                  Py                  Pz
                30.0                 0.5                 0.5                 0.0
#         deltaT_min            vdef_min            vdef_max             ASP_max             COL_min
              1.0E-7                -0.5                 2.0                 5.0                 0.1
#     Ndir  sphpartID
         2        88
"""
    model, log = _parse_deck(tmp_path, deck_fixed)

    assert 6 in model.prop_sol_orths
    p = model.prop_sol_orths[6]
    assert p.id == 6
    assert p.isolid == 14
    assert p.ismstr == 1
    assert p.inpts_r == 2
    assert p.inpts_s == 2
    assert p.inpts_t == 2
    assert p.iframe == 1
    assert pytest.approx(p.dn) == 0.1
    assert pytest.approx(p.qa) == 1.2
    assert pytest.approx(p.qb) == 0.04
    assert pytest.approx(p.h) == 0.2
    assert pytest.approx(p.vx) == 1.0
    assert p.skew_id == 12
    assert p.refplane == 1
    assert p.orthtrop == 2
    assert pytest.approx(p.mat_beta) == 30.0
    assert pytest.approx(p.px) == 0.5
    assert pytest.approx(p.py) == 0.5
    assert pytest.approx(p.deltat_min) == 1.0e-7
    assert pytest.approx(p.vdef_min) == -0.5
    assert pytest.approx(p.vdef_max) == 2.0
    assert pytest.approx(p.asp_max) == 5.0
    assert pytest.approx(p.col_min) == 0.1
    assert p.ndir == 2
    assert p.sphpart_id == 88

    # Free format
    deck_free = """\
/BEGIN
Test PROP SOL_ORTH Free
/PROP/TYPE6/7
Solid Orth Free
14 0 0 0 333 0 2 0.05
1.1 0.05 0.1
0.0 1.0 0.0 15 2 1
45.0 1.0 2.0 3.0
2.0e-7 -0.4 1.8 4.0 0.2
3 99
"""
    model2, log2 = _parse_deck(tmp_path, deck_free)

    assert 7 in model2.prop_sol_orths
    p2 = model2.prop_sol_orths[7]
    assert p2.id == 7
    assert p2.isolid == 14
    assert p2.inpts_r == 3
    assert p2.inpts_s == 3
    assert p2.inpts_t == 3
    assert p2.iframe == 2
    assert pytest.approx(p2.dn) == 0.05
    assert pytest.approx(p2.vy) == 1.0
    assert p2.skew_id == 15
    assert p2.refplane == 2
    assert p2.orthtrop == 1
    assert pytest.approx(p2.mat_beta) == 45.0
    assert pytest.approx(p2.px) == 1.0
    assert pytest.approx(p2.py) == 2.0
    assert pytest.approx(p2.pz) == 3.0
    assert pytest.approx(p2.deltat_min) == 2.0e-7
    assert p2.ndir == 3
    assert p2.sphpart_id == 99


def test_loads_cload_and_pload(tmp_path: Path):
    # Fixed format
    deck_fixed = """\
# OpenRadioss Starter Deck
/BEGIN
Test LOADS Fixed
      2022         0
/CLOAD/101
Nodal Concentrated Load
#funct_IDT       Dir   skew_ID sensor_ID  grnod_ID                       Ascalex             Fscaley
         5         X         2         1        10                           2.0               500.0
/PLOAD/201
Surface Pressure Load
#  surf_ID  functIDT sensor_ID    Ipinch      Idel   Itypfun            Ascale_x            Fscale_y
        12         6         3         1         1         2                 1.5                10.0
"""
    model, log = _parse_deck(tmp_path, deck_fixed)

    assert 101 in model.load_cloads
    cl = model.load_cloads[101]
    assert cl.id == 101
    assert cl.curve_id == 5
    assert cl.dir == "X"
    assert cl.skew_id == 2
    assert cl.sens_id == 1
    assert cl.grnod_id == 10
    assert pytest.approx(cl.xscale) == 2.0
    assert pytest.approx(cl.magnitude) == 500.0

    assert 201 in model.load_ploads
    pl = model.load_ploads[201]
    assert pl.id == 201
    assert pl.surf_id == 12
    assert pl.curve_id == 6
    assert pl.sens_id == 3
    assert pl.ipinch == 1
    assert pl.idel == 1
    assert pl.functype == 2
    assert pytest.approx(pl.xscale) == 1.5
    assert pytest.approx(pl.magnitude) == 10.0

    # Free format
    deck_free = """\
/BEGIN
Test LOADS Free
/LOAD/CLOAD/102
Free Cload
7 Y 20 750.0 4
/LOAD/PLOAD/202
Free Pload
15 8 5 0 1 1 2.5 15.0
"""
    model2, log2 = _parse_deck(tmp_path, deck_free)

    assert 102 in model2.load_cloads
    cl2 = model2.load_cloads[102]
    assert cl2.id == 102
    assert cl2.curve_id == 7
    assert cl2.dir == "Y"
    assert cl2.grnod_id == 20
    assert pytest.approx(cl2.magnitude) == 750.0
    assert cl2.sens_id == 4

    assert 202 in model2.load_ploads
    pl2 = model2.load_ploads[202]
    assert pl2.id == 202
    assert pl2.surf_id == 15
    assert pl2.curve_id == 8
    assert pl2.sens_id == 5
    assert pl2.ipinch == 0
    assert pl2.idel == 1
    assert pl2.functype == 1
    assert pytest.approx(pl2.xscale) == 2.5
    assert pytest.approx(pl2.magnitude) == 15.0
