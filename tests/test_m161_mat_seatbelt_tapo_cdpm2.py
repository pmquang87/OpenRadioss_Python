"""
Unit tests for Milestone M161:
Seatbelt, Tape Fabric & Advanced Concrete Damage Material Models Suite:
- /MAT/LAW114 (/MAT/SPR_SEATBELT)
- /MAT/LAW119 (/MAT/SH_SEATBELT)
- /MAT/LAW120 (/MAT/TAPO)
- /MAT/LAW121 (/MAT/PLAS_RATE)
- /MAT/LAW124 (/MAT/CDPM2)
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_deck(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    assert not log.errors, f"Parse errors: {log.errors}"
    return model, log


def test_mat_law114_spr_seatbelt_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """/BEGIN
LAW114_FIXED_TEST
                  00
/MAT/LAW114/10
Seatbelt Spring Mat Fixed
                7.8e-6                15.0
               12000.0               150.0
         5         6               1.5                 2.0
              210000.0                10.0                20.0             50000.0             10000.0
                  25.0                 0.5
/END
"""
    model_fixed, log_fixed = _parse_deck(tmp_path, deck_fixed)
    assert 10 in model_fixed.mat_spr_seatbelts
    assert 10 in model_fixed.materials
    m1 = model_fixed.mat_spr_seatbelts[10]
    assert m1.id == 10
    assert m1.title == "Seatbelt Spring Mat Fixed"
    assert pytest.approx(m1.rho) == 7.8e-6
    assert pytest.approx(m1.lmin) == 15.0
    assert pytest.approx(m1.k) == 12000.0
    assert pytest.approx(m1.c) == 150.0
    assert m1.fun_l == 5
    assert m1.fun_ul == 6
    assert pytest.approx(m1.xscale) == 1.5
    assert pytest.approx(m1.fscale) == 2.0
    assert pytest.approx(m1.e) == 210000.0
    assert pytest.approx(m1.i) == 10.0
    assert pytest.approx(m1.j) == 20.0
    assert pytest.approx(m1.fmax) == 50000.0
    assert pytest.approx(m1.mmax) == 10000.0
    assert pytest.approx(m1.as_) == 25.0
    assert pytest.approx(m1.r) == 0.5

    # Free format
    deck_free = """#RADIOSS STARTER
/BEGIN
LAW114_FREE_TEST
0 0
/MAT/SPR_SEATBELT/20
Seatbelt Spring Mat Free
7.85e-6 12.0
15000.0 200.0
7 8 1.2 1.8
200000.0 8.0 16.0 45000.0 9000.0
30.0 0.8
/END
"""
    model_free, log_free = _parse_deck(tmp_path, deck_free)
    assert 20 in model_free.mat_spr_seatbelts
    assert 20 in model_free.materials
    m2 = model_free.mat_spr_seatbelts[20]
    assert m2.id == 20
    assert m2.title == "Seatbelt Spring Mat Free"
    assert pytest.approx(m2.rho) == 7.85e-6
    assert pytest.approx(m2.lmin) == 12.0
    assert pytest.approx(m2.k) == 15000.0
    assert pytest.approx(m2.c) == 200.0
    assert m2.fun_l == 7
    assert m2.fun_ul == 8
    assert pytest.approx(m2.xscale) == 1.2
    assert pytest.approx(m2.fscale) == 1.8
    assert pytest.approx(m2.e) == 200000.0
    assert pytest.approx(m2.i) == 8.0
    assert pytest.approx(m2.j) == 16.0
    assert pytest.approx(m2.fmax) == 45000.0
    assert pytest.approx(m2.mmax) == 9000.0
    assert pytest.approx(m2.as_) == 30.0
    assert pytest.approx(m2.r) == 0.8


def test_mat_law119_sh_seatbelt_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """/BEGIN
LAW119_FIXED_TEST
                  00
/MAT/LAW119/101
Shell Seatbelt Mat Fixed
                1.2e-6                20.0
               10000.0               100.0                50.0
         1         2                 1.1                 1.2         1
               15000.0                 0.3              5000.0                 1.0
                2000.0                 0.4                 0.1
/END
"""
    model_fixed, log_fixed = _parse_deck(tmp_path, deck_fixed)
    assert 101 in model_fixed.mat_sh_seatbelts
    assert 101 in model_fixed.materials
    m1 = model_fixed.mat_sh_seatbelts[101]
    assert m1.id == 101
    assert m1.title == "Shell Seatbelt Mat Fixed"
    assert pytest.approx(m1.rho) == 1.2e-6
    assert pytest.approx(m1.lmin) == 20.0
    assert pytest.approx(m1.k) == 10000.0
    assert pytest.approx(m1.c) == 100.0
    assert pytest.approx(m1.re) == 50.0
    assert m1.fun_l == 1
    assert m1.fun_ul == 2
    assert pytest.approx(m1.fscale1) == 1.1
    assert pytest.approx(m1.fscale2) == 1.2
    assert m1.ireload == 1
    assert pytest.approx(m1.e22) == 15000.0
    assert pytest.approx(m1.nu12) == 0.3
    assert pytest.approx(m1.g12) == 5000.0
    assert pytest.approx(m1.fscale22) == 1.0
    assert pytest.approx(m1.ecoat) == 2000.0
    assert pytest.approx(m1.nucoat) == 0.4
    assert pytest.approx(m1.tcoat) == 0.1

    # Free format
    deck_free = """#RADIOSS STARTER
/BEGIN
LAW119_FREE_TEST
0 0
/MAT/SH_SEATBELT/102
Shell Seatbelt Mat Free
1.3e-6 25.0
12000.0 120.0 60.0
3 4 1.3 1.4 2
18000.0 0.35 6000.0 1.2
2500.0 0.42 0.15
/END
"""
    model_free, log_free = _parse_deck(tmp_path, deck_free)
    assert 102 in model_free.mat_sh_seatbelts
    assert 102 in model_free.materials
    m2 = model_free.mat_sh_seatbelts[102]
    assert m2.id == 102
    assert m2.title == "Shell Seatbelt Mat Free"
    assert pytest.approx(m2.rho) == 1.3e-6
    assert pytest.approx(m2.lmin) == 25.0
    assert pytest.approx(m2.k) == 12000.0
    assert pytest.approx(m2.c) == 120.0
    assert pytest.approx(m2.re) == 60.0
    assert m2.fun_l == 3
    assert m2.fun_ul == 4
    assert pytest.approx(m2.fscale1) == 1.3
    assert pytest.approx(m2.fscale2) == 1.4
    assert m2.ireload == 2
    assert pytest.approx(m2.e22) == 18000.0
    assert pytest.approx(m2.nu12) == 0.35
    assert pytest.approx(m2.g12) == 6000.0
    assert pytest.approx(m2.fscale22) == 1.2
    assert pytest.approx(m2.ecoat) == 2500.0
    assert pytest.approx(m2.nucoat) == 0.42
    assert pytest.approx(m2.tcoat) == 0.15


def test_mat_law120_tapo_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """/BEGIN
LAW120_FIXED_TEST
                  00
/MAT/LAW120/201
Tapo Mat Fixed
                1.5e-6              1.45e-6
               70000.0                 0.3         2         1         1                    0.25
        11                 1.2                 1.5
                 250.0                50.0                 1.2               300.0
                  10.0                20.0                30.0                40.0                50.0
                 500.0                0.05                 0.2
                  0.15                0.25                0.35                0.45
                  0.02                0.03                 1.5
/END
"""
    model_fixed, log_fixed = _parse_deck(tmp_path, deck_fixed)
    assert 201 in model_fixed.mat_tapos
    assert 201 in model_fixed.materials
    m1 = model_fixed.mat_tapos[201]
    assert m1.id == 201
    assert m1.title == "Tapo Mat Fixed"
    assert pytest.approx(m1.rho) == 1.5e-6
    assert pytest.approx(m1.refer_rho) == 1.45e-6
    assert pytest.approx(m1.e) == 70000.0
    assert pytest.approx(m1.nu) == 0.3
    assert m1.iform == 2
    assert m1.itrx == 1
    assert m1.idam == 1
    assert pytest.approx(m1.thick) == 0.25
    assert m1.tab_id == 11
    assert pytest.approx(m1.xscale) == 1.2
    assert pytest.approx(m1.yscale) == 1.5
    assert pytest.approx(m1.tau) == 250.0
    assert pytest.approx(m1.q) == 50.0
    assert pytest.approx(m1.beta) == 1.2
    assert pytest.approx(m1.h) == 300.0
    assert pytest.approx(m1.af1) == 10.0
    assert pytest.approx(m1.af2) == 20.0
    assert pytest.approx(m1.ah1) == 30.0
    assert pytest.approx(m1.ah2) == 40.0
    assert pytest.approx(m1.as_) == 50.0
    assert pytest.approx(m1.cc) == 500.0
    assert pytest.approx(m1.gam0) == 0.05
    assert pytest.approx(m1.gamf) == 0.2
    assert pytest.approx(m1.d1c) == 0.15
    assert pytest.approx(m1.d2c) == 0.25
    assert pytest.approx(m1.d1f) == 0.35
    assert pytest.approx(m1.d2f) == 0.45
    assert pytest.approx(m1.d_trx) == 0.02
    assert pytest.approx(m1.d_jc) == 0.03
    assert pytest.approx(m1.exp_n) == 1.5

    # Free format
    deck_free = """#RADIOSS STARTER
/BEGIN
LAW120_FREE_TEST
0 0
/MAT/TAPO/202
Tapo Mat Free
1.6e-6 1.55e-6
75000.0 0.32 1 0 0 0.3
12 1.0 1.0
280.0 60.0 1.0 350.0
15.0 25.0 35.0 45.0 55.0
600.0 0.06 0.25
0.18 0.28 0.38 0.48
0.025 0.035 1.8
/END
"""
    model_free, log_free = _parse_deck(tmp_path, deck_free)
    assert 202 in model_free.mat_tapos
    assert 202 in model_free.materials
    m2 = model_free.mat_tapos[202]
    assert m2.id == 202
    assert m2.title == "Tapo Mat Free"
    assert pytest.approx(m2.rho) == 1.6e-6
    assert pytest.approx(m2.refer_rho) == 1.55e-6
    assert pytest.approx(m2.e) == 75000.0
    assert pytest.approx(m2.nu) == 0.32
    assert m2.iform == 1
    assert m2.itrx == 0
    assert m2.idam == 0
    assert pytest.approx(m2.thick) == 0.3
    assert m2.tab_id == 12
    assert pytest.approx(m2.xscale) == 1.0
    assert pytest.approx(m2.yscale) == 1.0
    assert pytest.approx(m2.tau) == 280.0
    assert pytest.approx(m2.q) == 60.0
    assert pytest.approx(m2.beta) == 1.0
    assert pytest.approx(m2.h) == 350.0
    assert pytest.approx(m2.af1) == 15.0
    assert pytest.approx(m2.af2) == 25.0
    assert pytest.approx(m2.ah1) == 35.0
    assert pytest.approx(m2.ah2) == 45.0
    assert pytest.approx(m2.as_) == 55.0
    assert pytest.approx(m2.cc) == 600.0
    assert pytest.approx(m2.gam0) == 0.06
    assert pytest.approx(m2.gamf) == 0.25
    assert pytest.approx(m2.d1c) == 0.18
    assert pytest.approx(m2.d2c) == 0.28
    assert pytest.approx(m2.d1f) == 0.38
    assert pytest.approx(m2.d2f) == 0.48
    assert pytest.approx(m2.d_trx) == 0.025
    assert pytest.approx(m2.d_jc) == 0.035
    assert pytest.approx(m2.exp_n) == 1.8


def test_mat_law121_plas_rate_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """/BEGIN
LAW121_FIXED_TEST
                  00
/MAT/LAW121/301
Plas Rate Mat Fixed
                7.8e-6
              210000.0                 0.3         1         2               100.0              1.0e-5
        21                            1.05                1.15
        22                            1.02                1.04
        23                            1.01              1500.0
        24         1                  1.08                1.12
/END
"""
    model_fixed, log_fixed = _parse_deck(tmp_path, deck_fixed)
    assert 301 in model_fixed.mat_plas_rates
    assert 301 in model_fixed.materials
    m1 = model_fixed.mat_plas_rates[301]
    assert m1.id == 301
    assert m1.title == "Plas Rate Mat Fixed"
    assert pytest.approx(m1.rho) == 7.8e-6
    assert pytest.approx(m1.e) == 210000.0
    assert pytest.approx(m1.nu) == 0.3
    assert m1.ires == 1
    assert m1.ivisc == 2
    assert pytest.approx(m1.fcut) == 100.0
    assert pytest.approx(m1.tdel) == 1.0e-5
    assert m1.fct_sig0 == 21
    assert pytest.approx(m1.xscale_sig0) == 1.05
    assert pytest.approx(m1.yscale_sig0) == 1.15
    assert m1.fct_youn == 22
    assert pytest.approx(m1.xscale_youn) == 1.02
    assert pytest.approx(m1.yscale_youn) == 1.04
    assert m1.fct_tang == 23
    assert pytest.approx(m1.xscale_tang) == 1.01
    assert pytest.approx(m1.tang) == 1500.0
    assert m1.fct_fail == 24
    assert m1.ifail == 1
    assert pytest.approx(m1.xscale_fail) == 1.08
    assert pytest.approx(m1.yscale_fail) == 1.12

    # Free format
    deck_free = """#RADIOSS STARTER
/BEGIN
LAW121_FREE_TEST
0 0
/MAT/PLAS_RATE/302
Plas Rate Mat Free
7.85e-6
205000.0 0.28 0 1 120.0 2.0e-5
31 1.1 1.2
32 1.05 1.08
33 1.02 2000.0
34 2 1.15 1.25
/END
"""
    model_free, log_free = _parse_deck(tmp_path, deck_free)
    assert 302 in model_free.mat_plas_rates
    assert 302 in model_free.materials
    m2 = model_free.mat_plas_rates[302]
    assert m2.id == 302
    assert m2.title == "Plas Rate Mat Free"
    assert pytest.approx(m2.rho) == 7.85e-6
    assert pytest.approx(m2.e) == 205000.0
    assert pytest.approx(m2.nu) == 0.28
    assert m2.ires == 0
    assert m2.ivisc == 1
    assert pytest.approx(m2.fcut) == 120.0
    assert pytest.approx(m2.tdel) == 2.0e-5
    assert m2.fct_sig0 == 31
    assert pytest.approx(m2.xscale_sig0) == 1.1
    assert pytest.approx(m2.yscale_sig0) == 1.2
    assert m2.fct_youn == 32
    assert pytest.approx(m2.xscale_youn) == 1.05
    assert pytest.approx(m2.yscale_youn) == 1.08
    assert m2.fct_tang == 33
    assert pytest.approx(m2.xscale_tang) == 1.02
    assert pytest.approx(m2.tang) == 2000.0
    assert m2.fct_fail == 34
    assert m2.ifail == 2
    assert pytest.approx(m2.xscale_fail) == 1.15
    assert pytest.approx(m2.yscale_fail) == 1.25


def test_mat_law124_cdpm2_fixed_and_free(tmp_path: Path):
    # Fixed format
    deck_fixed = """/BEGIN
LAW124_FIXED_TEST
                  00
/MAT/LAW124/401
CDPM2 Mat Fixed
                2.4e-6
               30000.0                 0.2                             1               500.0
                   0.5                 0.2                 3.0                30.0                 0.1
                   0.1                 0.2                 0.3                 0.4
                   1.5                 2.5                 0.8                   1         2         1
                 0.001               0.002                 3.5                0.01
/END
"""
    model_fixed, log_fixed = _parse_deck(tmp_path, deck_fixed)
    assert 401 in model_fixed.mat_cdpm2s
    assert 401 in model_fixed.materials
    m1 = model_fixed.mat_cdpm2s[401]
    assert m1.id == 401
    assert m1.title == "CDPM2 Mat Fixed"
    assert pytest.approx(m1.rho) == 2.4e-6
    assert pytest.approx(m1.e) == 30000.0
    assert pytest.approx(m1.nu) == 0.2
    assert m1.irate == 1
    assert pytest.approx(m1.fcut) == 500.0
    assert pytest.approx(m1.ecc) == 0.5
    assert pytest.approx(m1.qh0) == 0.2
    assert pytest.approx(m1.ft) == 3.0
    assert pytest.approx(m1.fc) == 30.0
    assert pytest.approx(m1.hp) == 0.1
    assert pytest.approx(m1.ah) == 0.1
    assert pytest.approx(m1.bh) == 0.2
    assert pytest.approx(m1.ch) == 0.3
    assert pytest.approx(m1.dh) == 0.4
    assert pytest.approx(m1.as_) == 1.5
    assert pytest.approx(m1.bs) == 2.5
    assert pytest.approx(m1.df) == 0.8
    assert m1.dflag == 1
    assert m1.dtype == 2
    assert m1.ireg == 1
    assert pytest.approx(m1.wf) == 0.001
    assert pytest.approx(m1.wf1) == 0.002
    assert pytest.approx(m1.ft1) == 3.5
    assert pytest.approx(m1.efc) == 0.01

    # Free format
    deck_free = """#RADIOSS STARTER
/BEGIN
LAW124_FREE_TEST
0 0
/MAT/CDPM2/402
CDPM2 Mat Free
2.45e-6
32000.0 0.18 2 600.0
0.52 0.22 3.2 32.0 0.12
0.15 0.25 0.35 0.45
1.8 2.8 0.85 0 1 0
0.0015 0.0025 3.8 0.012
/END
"""
    model_free, log_free = _parse_deck(tmp_path, deck_free)
    assert 402 in model_free.mat_cdpm2s
    assert 402 in model_free.materials
    m2 = model_free.mat_cdpm2s[402]
    assert m2.id == 402
    assert m2.title == "CDPM2 Mat Free"
    assert pytest.approx(m2.rho) == 2.45e-6
    assert pytest.approx(m2.e) == 32000.0
    assert pytest.approx(m2.nu) == 0.18
    assert m2.irate == 2
    assert pytest.approx(m2.fcut) == 600.0
    assert pytest.approx(m2.ecc) == 0.52
    assert pytest.approx(m2.qh0) == 0.22
    assert pytest.approx(m2.ft) == 3.2
    assert pytest.approx(m2.fc) == 32.0
    assert pytest.approx(m2.hp) == 0.12
    assert pytest.approx(m2.ah) == 0.15
    assert pytest.approx(m2.bh) == 0.25
    assert pytest.approx(m2.ch) == 0.35
    assert pytest.approx(m2.dh) == 0.45
    assert pytest.approx(m2.as_) == 1.8
    assert pytest.approx(m2.bs) == 2.8
    assert pytest.approx(m2.df) == 0.85
    assert m2.dflag == 0
    assert m2.dtype == 1
    assert m2.ireg == 0
    assert pytest.approx(m2.wf) == 0.0015
    assert pytest.approx(m2.wf1) == 0.0025
    assert pytest.approx(m2.ft1) == 3.8
    assert pytest.approx(m2.efc) == 0.012
