"""Tests for Milestone M182:
- /MAT/LAW114 & /MAT/SPR_SEATBELT (1D Seatbelt spring material)
- /MAT/LAW117 & /MAT/COH_TAB (Tabulated cohesive zone material)
- /MAT/LAW119 & /MAT/SH_SEATBELT (2D Shell seatbelt material)
- /MAT/LAW120 & /MAT/TAPO (Tabulated orthotropic Pont-Pack material)
- /MAT/LAW121 & /MAT/PLAS_RATE (Tabulated rate-dependent elastoplastic material)
- /PROP/TYPE26 & /PROP/SPR_TAB (Tabulated spring property)
- /PROP/TYPE27 & /PROP/SPR_BDAMP (Bilinear/barycentric damping spring property)
"""

import pytest
from pathlib import Path
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog


def _parse_deck(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    deck_path = tmp_path / "TEST_0000.rad"
    deck_path.write_text(text, encoding="ascii")
    model = Model()
    log = MessageLog()
    blocks = read_deck(str(deck_path))
    parse_starter_deck(blocks, model, log)
    return model, log


def test_mat_law114_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW114 and /MAT/SPR_SEATBELT in fixed and free formats."""
    # Fixed format
    deck_fixed = """\
# OpenRadioss Starter Deck
/BEGIN
TEST_LAW114_FIXED
        2022         0
/MAT/LAW114/114
Seatbelt_1D_Fixed
#              RHO_I                LMIN
              7.8e-6                10.0
#                  K                   C
               500.0                10.0
#   fct_load  fct_uload             Xscale1             Fscale1
         1         2                 1.5                 2.0
#                  E               Ibend               Itors                FMAX                MMAX
             21000.0                0.05                0.02              1000.0               500.0
#         SHEAR_AREA                Rfac
                 0.8                 0.1
/END
"""
    model, log = _parse_deck(tmp_path / "fixed", deck_fixed)
    assert not log.errors
    assert 114 in model.mat_law114s
    m114 = model.mat_law114s[114]
    assert pytest.approx(m114.rho) == 7.8e-6
    assert pytest.approx(m114.lmin) == 10.0
    assert pytest.approx(m114.stiff1) == 500.0
    assert pytest.approx(m114.damp1) == 10.0
    assert m114.fun_l == 1
    assert m114.fun_ul == 2
    assert pytest.approx(m114.xcoeft1) == 1.5
    assert pytest.approx(m114.fcoeft1) == 2.0
    assert pytest.approx(m114.young) == 21000.0
    assert pytest.approx(m114.ibend) == 0.05
    assert pytest.approx(m114.itors) == 0.02
    assert pytest.approx(m114.fmax) == 1000.0
    assert pytest.approx(m114.mmax) == 500.0
    assert pytest.approx(m114.shear_area) == 0.8
    assert pytest.approx(m114.rfac) == 0.1

    # Free format alias /MAT/SPR_SEATBELT
    deck_free = """\
# OpenRadioss Starter Deck
/BEGIN
TEST_LAW114_FREE
/MAT/SPR_SEATBELT/214
Seatbelt_1D_Free
7.8e-6 12.0
600.0 15.0
3 4 1.2 2.5
22000.0 0.06 0.03 1200.0 600.0
0.9 0.2
/END
"""
    model_free, log_free = _parse_deck(tmp_path / "free", deck_free)
    assert not log_free.errors
    assert 214 in model_free.mat_law114s
    m214 = model_free.mat_law114s[214]
    assert pytest.approx(m214.rho) == 7.8e-6
    assert pytest.approx(m214.lmin) == 12.0
    assert pytest.approx(m214.stiff1) == 600.0
    assert pytest.approx(m214.damp1) == 15.0
    assert m214.fun_l == 3
    assert m214.fun_ul == 4
    assert pytest.approx(m214.young) == 22000.0


def test_mat_law117_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW117 and /MAT/COH_TAB in fixed and free formats."""
    deck_fixed = """\
# OpenRadioss Starter Deck
/BEGIN
TEST_LAW117_FIXED
        2022         0
/MAT/LAW117/117
Cohesive_Tabulated_Fixed
#        Init. dens.
              1.2e-6
#                 EN                  ES     Imass      Idel     Irupt
             15000.0             10000.0         1         0         2
#   FCT_TN    FCT_TT                  TN                  TS            Fscale_x
         5         6                50.0                40.0                 1.0
#                GIC                GIIC               EXP_G              EXP_BK               GAMMA
                 1.5                 2.0                 1.8                 2.2                 0.5
/END
"""
    model, log = _parse_deck(tmp_path / "fixed", deck_fixed)
    assert not log.errors
    assert 117 in model.mat_law117s
    m117 = model.mat_law117s[117]
    assert pytest.approx(m117.rho) == 1.2e-6
    assert pytest.approx(m117.en) == 15000.0
    assert pytest.approx(m117.es) == 10000.0
    assert m117.imass == 1
    assert m117.irupt == 2
    assert m117.fct_tn == 5
    assert m117.fct_tt == 6
    assert pytest.approx(m117.tn) == 50.0
    assert pytest.approx(m117.ts) == 40.0
    assert pytest.approx(m117.gic) == 1.5
    assert pytest.approx(m117.giic) == 2.0
    assert pytest.approx(m117.exp_bk) == 2.2

    # Free format alias /MAT/COH_TAB
    deck_free = """\
# OpenRadioss Starter Deck
/BEGIN
TEST_LAW117_FREE
/MAT/COH_TAB/217
Cohesive_Tabulated_Free
1.5e-6
16000.0 11000.0 2 1 1
7 8 55.0 45.0 1.2
1.8 2.5 1.9 2.4 0.6
/END
"""
    model_free, log_free = _parse_deck(tmp_path / "free", deck_free)
    assert not log_free.errors
    assert 217 in model_free.mat_law117s
    m217 = model_free.mat_law117s[217]
    assert pytest.approx(m217.rho) == 1.5e-6
    assert pytest.approx(m217.en) == 16000.0
    assert m217.fct_tn == 7
    assert pytest.approx(m217.gic) == 1.8


def test_mat_law119_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW119 and /MAT/SH_SEATBELT in fixed and free formats."""
    deck_fixed = """\
# OpenRadioss Starter Deck
/BEGIN
TEST_LAW119_FIXED
        2022         0
/MAT/LAW119/119
Shell_Seatbelt_Fixed
#              RHO_I                LMIN
              1.0e-6                 5.0
#                  K                   C                  RE
               300.0                 5.0                 0.5
# fct_load fct_uload             Fscale1             Fscale2   Ireload
         9        10                 1.0                 1.0         1
#                E22                 V12                 G12            Fscale22
             18000.0                 0.3              7000.0                 1.0
#                 EC                  VC                  TC
              5000.0                0.25                 0.2
/END
"""
    model, log = _parse_deck(tmp_path / "fixed", deck_fixed)
    assert not log.errors
    assert 119 in model.mat_law119s
    m119 = model.mat_law119s[119]
    assert pytest.approx(m119.rho) == 1.0e-6
    assert pytest.approx(m119.lmin) == 5.0
    assert pytest.approx(m119.stiff1) == 300.0
    assert pytest.approx(m119.re) == 0.5
    assert m119.fun_l == 9
    assert m119.fun_ul == 10
    assert m119.ireload == 1
    assert pytest.approx(m119.e22) == 18000.0
    assert pytest.approx(m119.nu12) == 0.3
    assert pytest.approx(m119.g12) == 7000.0
    assert pytest.approx(m119.ecoat) == 5000.0
    assert pytest.approx(m119.tcoat) == 0.2

    # Free format alias /MAT/SH_SEATBELT
    deck_free = """\
# OpenRadioss Starter Deck
/BEGIN
TEST_LAW119_FREE
/MAT/SH_SEATBELT/219
Shell_Seatbelt_Free
1.1e-6 6.0
350.0 6.0 0.6
11 12 1.1 1.2 2
19000.0 0.32 7500.0 1.1
5500.0 0.28 0.25
/END
"""
    model_free, log_free = _parse_deck(tmp_path / "free", deck_free)
    assert not log_free.errors
    assert 219 in model_free.mat_law119s
    m219 = model_free.mat_law119s[219]
    assert pytest.approx(m219.rho) == 1.1e-6
    assert m219.fun_l == 11
    assert pytest.approx(m219.e22) == 19000.0


def test_mat_law120_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW120 and /MAT/TAPO in fixed and free formats."""
    deck_fixed = """\
# OpenRadioss Starter Deck
/BEGIN
TEST_LAW120_FIXED
        2022         0
/MAT/LAW120/120
Tapo_Fixed
#        Init. dens.          Ref. dens.
              2.7e-6              2.7e-6
#                  E                  nu     Iform      Itrx      Idam                         THICK
             70000.0                0.33         1         0         1                           1.5
# Table_Id              Xscale              Yscale
        15                 1.0                 1.0
#                 T0                   Q                Beta                   H
               150.0                50.0                 8.0               300.0
#                AF1                 AF2                 AH1                 AH2                  AS
                 0.1                 0.2                 0.3                 0.4                 0.5
#                  C               EPSD0               EPSDF
                 0.0                1e-3                 0.2
#                D1C                 D2C                 D1F                 D2F
                0.05                0.02                 0.1                0.08
#               Dtrx                 Djc               EXP_N
                 0.5                 0.6                 2.0
/END
"""
    model, log = _parse_deck(tmp_path / "fixed", deck_fixed)
    assert not log.errors
    assert 120 in model.mat_law120s
    m120 = model.mat_law120s[120]
    assert pytest.approx(m120.rho) == 2.7e-6
    assert pytest.approx(m120.e) == 70000.0
    assert pytest.approx(m120.nu) == 0.33
    assert m120.iform == 1
    assert m120.idam == 1
    assert pytest.approx(m120.thick) == 1.5
    assert m120.tab_id == 15
    assert pytest.approx(m120.tau0) == 150.0
    assert pytest.approx(m120.q) == 50.0
    assert pytest.approx(m120.beta) == 8.0
    assert pytest.approx(m120.af1) == 0.1
    assert pytest.approx(m120.dtrx) == 0.5
    assert pytest.approx(m120.exp_n) == 2.0

    # Free format alias /MAT/TAPO
    deck_free = """\
# OpenRadioss Starter Deck
/BEGIN
TEST_LAW120_FREE
/MAT/TAPO/220
Tapo_Free
2.8e-6 2.8e-6
72000.0 0.34 2 1 2 2.0
18 1.1 1.2
160.0 55.0 9.0 320.0
0.12 0.22 0.32 0.42 0.52
0.01 2e-3 0.25
0.06 0.03 0.12 0.09
0.55 0.65 2.1
/END
"""
    model_free, log_free = _parse_deck(tmp_path / "free", deck_free)
    assert not log_free.errors
    assert 220 in model_free.mat_law120s
    m220 = model_free.mat_law120s[220]
    assert pytest.approx(m220.rho) == 2.8e-6
    assert pytest.approx(m220.e) == 72000.0
    assert m220.tab_id == 18
    assert pytest.approx(m220.tau0) == 160.0


def test_mat_law121_fixed_and_free(tmp_path: Path):
    """Test /MAT/LAW121 and /MAT/PLAS_RATE in fixed and free formats."""
    deck_fixed = """\
# OpenRadioss Starter Deck
/BEGIN
TEST_LAW121_FIXED
        2022         0
/MAT/LAW121/121
Plas_Rate_Fixed
#              RHO_I
              7.8e-6
#                  E                  Nu      Ires     Ivisc                Fcut               DTMIN
            210000.0                 0.3         2         1              1000.0              1.0e-5
# Fct_SIG0                   Xscale_SIG0         Yscale_SIG0
        21                           1.0                 1.0
# Fct_YOUN                   Xscale_YOUN         Yscale_YOUN
        22                           1.0                 1.0
# Fct_TANG                   Xscale_TANG                TANG
        23                           1.0              1500.0
# Fct_FAIL     Ifail         Xscale_FAIL         Yscale_FAIL
        24         1                 1.0                 0.3
/END
"""
    model, log = _parse_deck(tmp_path / "fixed", deck_fixed)
    assert not log.errors
    assert 121 in model.mat_law121s
    m121 = model.mat_law121s[121]
    assert pytest.approx(m121.rho) == 7.8e-6
    assert pytest.approx(m121.e) == 210000.0
    assert pytest.approx(m121.nu) == 0.3
    assert m121.ires == 2
    assert m121.ivisc == 1
    assert pytest.approx(m121.fcut) == 1000.0
    assert pytest.approx(m121.dtmin) == 1.0e-5
    assert m121.fct_sig0 == 21
    assert m121.fct_youn == 22
    assert m121.fct_tang == 23
    assert pytest.approx(m121.tang) == 1500.0
    assert m121.fct_fail == 24
    assert m121.ifail == 1
    assert pytest.approx(m121.yscale_fail) == 0.3

    # Free format alias /MAT/PLAS_RATE
    deck_free = """\
# OpenRadioss Starter Deck
/BEGIN
TEST_LAW121_FREE
/MAT/PLAS_RATE/221
Plas_Rate_Free
7.9e-6
215000.0 0.29 1 0 1100.0 2.0e-5
25 1.1 1.2
26 1.1 1.2
27 1.1 1600.0
28 2 1.2 0.35
/END
"""
    model_free, log_free = _parse_deck(tmp_path / "free", deck_free)
    assert not log_free.errors
    assert 221 in model_free.mat_law121s
    m221 = model_free.mat_law121s[221]
    assert pytest.approx(m221.rho) == 7.9e-6
    assert pytest.approx(m221.e) == 215000.0
    assert m221.fct_sig0 == 25
    assert pytest.approx(m221.tang) == 1600.0


def test_prop_spr_tab_fixed_and_free(tmp_path: Path):
    """Test /PROP/TYPE26 and /PROP/SPR_TAB with multi-curve loading and unloading."""
    deck_fixed = """\
# OpenRadioss Starter Deck
/BEGIN
TEST_PROP_SPR_TAB_FIXED
        2022         0
/PROP/TYPE26/26
Spring_Tabulated_Fixed
#                  M                                 sens_ID    Isflag     Ileng                Dmin
                 0.5                                       1         0         1                -5.0
#    Nfunc     Nfund              Lscale                Kmax                Dmax               Alpha
         2         1                 1.0               500.0                10.0                 0.9
#  fct_ID1              Fscale         Strain_rate
        31                 1.0                 0.0
        32                 1.2                10.0
#  fct_ID1              Fscale         Strain_rate
        33                 1.0                 0.0
/END
"""
    model, log = _parse_deck(tmp_path / "fixed", deck_fixed)
    assert not log.errors
    assert 26 in model.prop_spr_tabs
    p26 = model.prop_spr_tabs[26]
    assert pytest.approx(p26.mass) == 0.5
    assert p26.sens_id == 1
    assert p26.isflag == 0
    assert p26.ileng == 1
    assert pytest.approx(p26.dmin) == -5.0
    assert p26.nfunc == 2
    assert p26.nfund == 1
    assert pytest.approx(p26.lscale) == 1.0
    assert pytest.approx(p26.kmax) == 500.0
    assert pytest.approx(p26.dmax) == 10.0
    assert pytest.approx(p26.alpha) == 0.9
    assert len(p26.loading_curves) == 2
    assert p26.loading_curves[0].fct_id == 31
    assert pytest.approx(p26.loading_curves[0].fscale) == 1.0
    assert p26.loading_curves[1].fct_id == 32
    assert pytest.approx(p26.loading_curves[1].strain_rate) == 10.0
    assert len(p26.unloading_curves) == 1
    assert p26.unloading_curves[0].fct_id == 33

    # Free format alias /PROP/SPR_TAB
    deck_free = """\
# OpenRadioss Starter Deck
/BEGIN
TEST_PROP_SPR_TAB_FREE
/PROP/SPR_TAB/126
Spring_Tabulated_Free
0.6 2 1 0 -6.0
1 1 1.1 550.0 12.0 0.95
34 1.1 0.0
35 1.1 0.0
/END
"""
    model_free, log_free = _parse_deck(tmp_path / "free", deck_free)
    assert not log_free.errors
    assert 126 in model_free.prop_spr_tabs
    p126 = model_free.prop_spr_tabs[126]
    assert pytest.approx(p126.mass) == 0.6
    assert p126.sens_id == 2
    assert len(p126.loading_curves) == 1
    assert p126.loading_curves[0].fct_id == 34
    assert len(p126.unloading_curves) == 1
    assert p126.unloading_curves[0].fct_id == 35


def test_prop_spr_bdamp_fixed_and_free(tmp_path: Path):
    """Test /PROP/TYPE27 and /PROP/SPR_BDAMP in fixed and free formats."""
    deck_fixed = """\
# OpenRadioss Starter Deck
/BEGIN
TEST_PROP_SPR_BDAMP_FIXED
        2022         0
/PROP/TYPE27/27
Spring_Bilinear_Damp_Fixed
#               Mass                                 sens_ID    Isflag     Ileng     Itens     Ifail
                 0.2                                       1         0         0         1         2
#                  K                   C                   n           Delta_min           Delta_max
               200.0                10.0                 1.5                -2.0                 5.0
#                gap                                                     Fsmooth                Fcut
                 0.1                                                           1               100.0
#  fct_ID1   fct_ID2             Ascale1             Fscale1             Ascale2             Fscale2
        41        42                 1.0                 1.0                 1.0                 1.0
/END
"""
    model, log = _parse_deck(tmp_path / "fixed", deck_fixed)
    assert not log.errors
    assert 27 in model.prop_spr_bdamps
    p27 = model.prop_spr_bdamps[27]
    assert pytest.approx(p27.mass) == 0.2
    assert p27.sens_id == 1
    assert p27.itens == 1
    assert p27.ifail == 2
    assert pytest.approx(p27.stiff) == 200.0
    assert pytest.approx(p27.damp) == 10.0
    assert pytest.approx(p27.nexp) == 1.5
    assert pytest.approx(p27.delta_min) == -2.0
    assert pytest.approx(p27.delta_max) == 5.0
    assert pytest.approx(p27.gap) == 0.1
    assert p27.fsmooth == 1
    assert pytest.approx(p27.fcut) == 100.0
    assert p27.fct_id1 == 41
    assert p27.fct_id2 == 42

    # Free format alias /PROP/SPR_BDAMP
    deck_free = """\
# OpenRadioss Starter Deck
/BEGIN
TEST_PROP_SPR_BDAMP_FREE
/PROP/SPR_BDAMP/127
Spring_Bilinear_Damp_Free
0.25 2 1 1 0 1
250.0 12.0 1.6 -3.0 6.0
0.15 2 120.0
43 44 1.2 1.3 1.4 1.5
/END
"""
    model_free, log_free = _parse_deck(tmp_path / "free", deck_free)
    assert not log_free.errors
    assert 127 in model_free.prop_spr_bdamps
    p127 = model_free.prop_spr_bdamps[127]
    assert pytest.approx(p127.mass) == 0.25
    assert p127.sens_id == 2
    assert pytest.approx(p127.stiff) == 250.0
    assert p127.fct_id1 == 43
    assert p127.fct_id2 == 44
    assert pytest.approx(p127.ascale1) == 1.2
    assert pytest.approx(p127.fscale2) == 1.5
