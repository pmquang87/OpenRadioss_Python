"""Tests for Milestone M170: Anisotropic Barlat 2000, Concrete, Spring-Joint & Metallurgical Steel Material Models Suite.

Covers:
- /MAT/LAW24 & /MAT/CONC: Concrete material model (fixed & free formats)
- /MAT/LAW87 & /MAT/BARLAT: Barlat 2000 anisotropic plasticity model (Ifit=0 alphas, Ifit=1 fit parameters, Swift/Voce hardening)
- /MAT/LAW83 & /MAT/SPR_JOU: Non-linear spring/joint material model (fixed & free formats)
- /MAT/LAW80 & /MAT/TRANSFO: Metallurgical phase transformation steel model (fixed & free formats)
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
    return model, log


def test_m170_mat_conc_fixed(tmp_path: Path):
    """Test /MAT/LAW24 & /MAT/CONC in fixed format."""
    deck = (
        "#STARTER\n"
        "/MAT/LAW24/1\n"
        "Concrete Mat 1\n"
        "#              RHO_I               RHO_O\n"
        "             2.40E-6             2.40E-6\n"
        "#                E_C                  NU\n"
        "             30000.0                 0.2\n"
        "#                F_C            FT_ON_FC            FB_ON_FC            F2_ON_FC            S0_ON_FC\n"
        "                35.0                 0.1                 1.15                1.5                 5.0\n"
        "#                H_T               D_SUP             EPS_MAX\n"
        "              2500.0                 0.8                0.05\n"
        "#                K_Y                 R_T                 R_C                H_BP\n"
        "                 0.3                 5.0                10.0              5000.0\n"
        "#            ALPHA_Y             ALPHA_F               V_MAX\n"
        "                 0.1                 0.2               -0.03\n"
        "#                F_K                  F0                H_V0\n"
        "                15.0                20.0              1000.0\n"
        "#                  E             SIGMA_Y                 E_T\n"
        "            210000.0               500.0              2000.0\n"
        "#             ALPHA1              ALPHA2              ALPHA3\n"
        "                0.01                0.01                0.00\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 1 in model.mat_concs
    mc = model.mat_concs[1]
    assert mc.id == 1
    assert mc.title == "Concrete Mat 1"
    assert pytest.approx(mc.rho0) == 2.40e-6
    assert pytest.approx(mc.refer_rho) == 2.40e-6
    assert pytest.approx(mc.e_c) == 30000.0
    assert pytest.approx(mc.nu) == 0.2
    assert pytest.approx(mc.f_c) == 35.0
    assert pytest.approx(mc.ft_on_fc) == 0.1
    assert pytest.approx(mc.fb_on_fc) == 1.15
    assert pytest.approx(mc.f2_on_fc) == 1.5
    assert pytest.approx(mc.s0_on_fc) == 5.0
    assert pytest.approx(mc.h_t) == 2500.0
    assert pytest.approx(mc.d_sup) == 0.8
    assert pytest.approx(mc.eps_max) == 0.05
    assert pytest.approx(mc.k_y) == 0.3
    assert pytest.approx(mc.r_t) == 5.0
    assert pytest.approx(mc.r_c) == 10.0
    assert pytest.approx(mc.h_bp) == 5000.0
    assert pytest.approx(mc.alpha_y) == 0.1
    assert pytest.approx(mc.alpha_f) == 0.2
    assert pytest.approx(mc.v_max) == -0.03
    assert pytest.approx(mc.f_k) == 15.0
    assert pytest.approx(mc.f0) == 20.0
    assert pytest.approx(mc.h_v0) == 1000.0
    assert pytest.approx(mc.e2) == 210000.0
    assert pytest.approx(mc.ssig) == 500.0
    assert pytest.approx(mc.setan) == 2000.0
    assert pytest.approx(mc.alpha1) == 0.01
    assert pytest.approx(mc.alpha2) == 0.01
    assert pytest.approx(mc.alpha3) == 0.00

    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 24
    assert mat.params["E"] == mc.e_c


def test_m170_mat_conc_free(tmp_path: Path):
    """Test /MAT/CONC in free format."""
    deck = (
        "/MAT/CONC/2\n"
        "Concrete Free\n"
        "2.35e-6 2.35e-6\n"
        "28000.0 0.18\n"
        "30.0 0.09 1.12 1.4 4.5\n"
        "2200.0 0.75 0.04\n"
        "0.25 4.5 9.0 4500.0\n"
        "0.08 0.18 -0.025\n"
        "12.0 18.0 900.0\n"
        "200000.0 450.0 1800.0\n"
        "0.015 0.015 0.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 2 in model.mat_concs
    mc = model.mat_concs[2]
    assert mc.id == 2
    assert mc.title == "Concrete Free"
    assert pytest.approx(mc.rho0) == 2.35e-6
    assert pytest.approx(mc.e_c) == 28000.0
    assert pytest.approx(mc.nu) == 0.18
    assert pytest.approx(mc.f_c) == 30.0
    assert pytest.approx(mc.e2) == 200000.0
    assert pytest.approx(mc.ssig) == 450.0
    assert pytest.approx(mc.alpha1) == 0.015


def test_m170_mat_barlat_alphas_fixed(tmp_path: Path):
    """Test /MAT/LAW87 (Barlat 2000) with explicit alpha coefficients in fixed format."""
    deck = (
        "#STARTER\n"
        "/MAT/LAW87/10\n"
        "Barlat Alphas Mat\n"
        "#              RHO_I               RHO_O\n"
        "             7.85E-6             7.85E-6\n"
        "#                  E                  Nu     IFlag        VP                   c                   P\n"
        "            210000.0                 0.3         1         0                 0.0                 0.0\n"
        "#                 a1                  a2                  a3                  a4\n"
        "                 0.9                 0.95                1.05                1.1\n"
        "#                 a5                  a6                  a7                  a8\n"
        "                 0.92                0.98                1.02                1.08\n"
        "#              Chard\n"
        "                 0.0\n"
        "#              exp_a               alpha                   n               F_cut  F_smooth\n"
        "                   6                 1.0                0.22                10.0         1\n"
        "#             ASwift                Eps0               Qvoce                Beta                  KO\n"
        "               550.0               0.005               200.0                15.0               250.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 10 in model.mat_barlats
    mb = model.mat_barlats[10]
    assert mb.id == 10
    assert mb.title == "Barlat Alphas Mat"
    assert pytest.approx(mb.rho0) == 7.85e-6
    assert pytest.approx(mb.e) == 210000.0
    assert pytest.approx(mb.nu) == 0.3
    assert mb.iflag == 1
    assert mb.ifit == 0
    assert pytest.approx(mb.alphas[0]) == 0.9
    assert pytest.approx(mb.alphas[1]) == 0.95
    assert pytest.approx(mb.alphas[2]) == 1.05
    assert pytest.approx(mb.alphas[3]) == 1.1
    assert pytest.approx(mb.alphas[4]) == 0.92
    assert pytest.approx(mb.alphas[5]) == 0.98
    assert pytest.approx(mb.alphas[6]) == 1.02
    assert pytest.approx(mb.alphas[7]) == 1.08
    assert mb.a_exp == 6
    assert pytest.approx(mb.alpha_vol) == 1.0
    assert pytest.approx(mb.n_hard) == 0.22
    assert pytest.approx(mb.fcut) == 10.0
    assert mb.fsmooth == 1
    assert pytest.approx(mb.a_swift) == 550.0
    assert pytest.approx(mb.eps0) == 0.005
    assert pytest.approx(mb.q_voce) == 200.0
    assert pytest.approx(mb.beta) == 15.0
    assert pytest.approx(mb.k0) == 250.0

    assert 10 in model.materials
    mat = model.materials[10]
    assert mat.law == 87
    assert mat.params["alphas"] == mb.alphas


def test_m170_mat_barlat_fit_free(tmp_path: Path):
    """Test /MAT/BARLAT with Lankford / yield stress fit parameters in free format."""
    deck = (
        "/MAT/BARLAT/11\n"
        "Barlat Fit Mat\n"
        "2.70e-6 2.70e-6\n"
        "70000.0 0.33 1 0 0.0 0.0\n"
        "280.0 270.0 290.0 300.0 1\n"
        "0.65 0.55 0.75 0.85\n"
        "0.0\n"
        "8 1.0 0.18 5.0 0\n"
        "420.0 0.002 150.0 12.0 180.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 11 in model.mat_barlats
    mb = model.mat_barlats[11]
    assert mb.id == 11
    assert mb.title == "Barlat Fit Mat"
    assert pytest.approx(mb.rho0) == 2.70e-6
    assert pytest.approx(mb.e) == 70000.0
    assert mb.ifit == 1
    assert pytest.approx(mb.sigma_00) == 280.0
    assert pytest.approx(mb.sigma_45) == 270.0
    assert pytest.approx(mb.sigma_90) == 290.0
    assert pytest.approx(mb.sigma_b) == 300.0
    assert pytest.approx(mb.r_00) == 0.65
    assert pytest.approx(mb.r_45) == 0.55
    assert pytest.approx(mb.r_90) == 0.75
    assert pytest.approx(mb.r_b) == 0.85
    assert mb.a_exp == 8
    assert pytest.approx(mb.a_swift) == 420.0


def test_m170_mat_law83_fixed(tmp_path: Path):
    """Test /MAT/LAW83 in fixed format."""
    deck = (
        "#STARTER\n"
        "/MAT/LAW83/20\n"
        "Spring Joint Mat Fixed\n"
        "#              RHO_I               RHO_O\n"
        "             1.00E-6             1.00E-6\n"
        "#                  E                         Imass\n"
        "              1500.0                             1\n"
        "#  Fct_ID1                      Y_scale1            X_scale1               ALPHA                BETA\n"
        "       101                           1.5                 2.0                 0.1                 0.2\n"
        "#                 RN                  RS   Fsmooth                Fcut\n"
        "                 0.5                 0.6         1                25.0\n"
        "#  Fct_IDN   Fct_IDS              XSCALE\n"
        "       102       103                 1.2\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 20 in model.mat_law83s
    m83 = model.mat_law83s[20]
    assert m83.id == 20
    assert m83.title == "Spring Joint Mat Fixed"
    assert pytest.approx(m83.rho0) == 1.00e-6
    assert pytest.approx(m83.e) == 1500.0
    assert m83.imass == 1
    assert m83.fun_a1 == 101
    assert pytest.approx(m83.fscale11) == 1.5
    assert pytest.approx(m83.fscale22) == 2.0
    assert pytest.approx(m83.alpha) == 0.1
    assert pytest.approx(m83.beta) == 0.2
    assert pytest.approx(m83.rn) == 0.5
    assert pytest.approx(m83.rs) == 0.6
    assert m83.fsmooth == 1
    assert pytest.approx(m83.fcut) == 25.0
    assert m83.fun_a2 == 102
    assert m83.fun_a3 == 103
    assert pytest.approx(m83.fscale33) == 1.2

    assert 20 in model.materials
    mat = model.materials[20]
    assert mat.law == 83
    assert mat.params["fun_a1"] == 101


def test_m170_mat_law83_free(tmp_path: Path):
    """Test /MAT/SPR_JOU in free format."""
    deck = (
        "/MAT/SPR_JOU/21\n"
        "Spring Joint Free\n"
        "1.2e-6 1.2e-6\n"
        "1800.0 0\n"
        "201 1.0 1.0 0.05 0.1\n"
        "0.4 0.45 0 0.0\n"
        "202 203 1.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 21 in model.mat_law83s
    m83 = model.mat_law83s[21]
    assert m83.id == 21
    assert m83.title == "Spring Joint Free"
    assert pytest.approx(m83.e) == 1800.0
    assert m83.fun_a1 == 201
    assert m83.fun_a2 == 202
    assert m83.fun_a3 == 203


def test_m170_mat_law80_fixed(tmp_path: Path):
    """Test /MAT/LAW80 & /MAT/TRANSFO in fixed format."""
    deck = (
        "#STARTER\n"
        "/MAT/LAW80/30\n"
        "Phase Transfo Mat Fixed\n"
        "#              RHO_I               RHO_O\n"
        "             7.80E-6             7.80E-6\n"
        "#                  E                  Nu   Fct_IDE             YscaleE           Time_unit\n"
        "            205000.0                0.29       301                 1.1              3600.0\n"
        "#            Fsmooth                Fcut                Ceps                Peps\n"
        "                   1                50.0                40.0                 5.0\n"
        "#  ID_YLD1   ID_YLD2   ID_YLD3   ID_YLD4   ID_YLD5\n"
        "       311       312       313       314       315\n"
        "#            Yscale1             Yscale2             Yscale3             Yscale4             Yscale5\n"
        "                 1.0                 1.1                 1.2                 1.3                 1.4\n"
        "#            Xscale1             Xscale2             Xscale3             Xscale4             Xscale5\n"
        "                 1.0                 1.0                 1.0                 1.0                 1.0\n"
        "#             Theta2              Theta3              Theta4              Theta5\n"
        "                 0.1                 0.2                 0.3                 0.4\n"
        "#             Alpha1              Alpha2\n"
        "             1.20E-5             1.50E-5\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 30 in model.mat_law80s
    m80 = model.mat_law80s[30]
    assert m80.id == 30
    assert m80.title == "Phase Transfo Mat Fixed"
    assert pytest.approx(m80.rho0) == 7.80e-6
    assert pytest.approx(m80.e) == 205000.0
    assert pytest.approx(m80.nu) == 0.29
    assert m80.fct_ide == 301
    assert pytest.approx(m80.scale_e) == 1.1
    assert pytest.approx(m80.time_unit) == 3600.0
    assert m80.fsmooth == 1
    assert pytest.approx(m80.fcut) == 50.0
    assert pytest.approx(m80.ceps) == 40.0
    assert pytest.approx(m80.peps) == 5.0
    assert m80.fun_a == [311, 312, 313, 314, 315]
    assert pytest.approx(m80.fscale_y[0]) == 1.0
    assert pytest.approx(m80.fscale_y[4]) == 1.4
    assert pytest.approx(m80.theta[0]) == 0.1
    assert pytest.approx(m80.theta[3]) == 0.4
    assert pytest.approx(m80.alpha1) == 1.20e-5
    assert pytest.approx(m80.alpha2) == 1.50e-5

    assert 30 in model.materials
    mat = model.materials[30]
    assert mat.law == 80
    assert mat.params["fun_a"] == [311, 312, 313, 314, 315]


def test_m170_mat_law80_free(tmp_path: Path):
    """Test /MAT/TRANSFO in free format."""
    deck = (
        "/MAT/TRANSFO/31\n"
        "Phase Transfo Free\n"
        "7.85e-6 7.85e-6\n"
        "210000.0 0.3 0 1.0 3600.0\n"
        "0 0.0 0.0 0.0\n"
        "401 402 403 404 405\n"
        "1.0 1.0 1.0 1.0 1.0\n"
        "1.0 1.0 1.0 1.0 1.0\n"
        "0.0 0.0 0.0 0.0\n"
        "1.1e-5 1.4e-5\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 31 in model.mat_law80s
    m80 = model.mat_law80s[31]
    assert m80.id == 31
    assert m80.title == "Phase Transfo Free"
    assert pytest.approx(m80.e) == 210000.0
    assert m80.fun_a == [401, 402, 403, 404, 405]
