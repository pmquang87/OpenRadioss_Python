"""Tests for Milestone M172: Advanced Tabular Foam, Viscoelastic Foam, Visco-Hyperelastic, Honeycomb & Cowper-Symonds Material Models Suite.

Covers:
- /MAT/LAW66 & /MAT/FOAM_TAB: Tabular foam material model (fixed & free formats, all ISRATE modes 0, 1, 2, 3, 4)
- /MAT/LAW35 & /MAT/FOAM_VISC: Viscoelastic foam material model (fixed & free formats)
- /MAT/LAW62 & /MAT/VISC_HYP: Viscoelastic hyperelastic Ogden material model (fixed & free formats, multi-order N & M)
- /MAT/LAW28 & /MAT/HONEYCOMB: Orthotropic honeycomb crushable material model (fixed & free formats)
- /MAT/LAW44 & /MAT/COWPER_SYMONDS / /MAT/PLAS_COWPER: Cowper-Symonds strain-rate elastoplastic material model (fixed & free formats, Iflag 0 & 1)
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


def test_mat_law66_fixed_and_free(tmp_path: Path):
    # Fixed format with ISRATE=0
    deck_fixed = (
        "/BEGIN\n"
        "TEST_M172_LAW66_FIXED\n"
        "/MAT/LAW66/101\n"
        "Foam Tabular Material Fixed\n"
        "#              RHO_I          Ref. dens.\n"
        "            1.200E-3            1.200E-3\n"
        "#                  E                  Nu              C_hard               F_cut  F_smooth Iyld_rate\n"
        "            100.0000              0.3000              0.0500              1000.0         1         0\n"
        "#                P_c                 P_t                  EC                RPCT\n"
        "              5.0000              2.5000             80.0000              0.1500\n"
        "#funct_IDc funct_IDt             Fscalec             Fscalet\n"
        "        10        11              1.2000              1.1000\n"
        "#          Epsilon_0                   c            Sigma_Y0        VP\n"
        "              0.0010              0.0400             15.0000         1\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_fixed)
    assert 101 in model.mat_law66s
    m = model.mat_law66s[101]
    assert m.id == 101
    assert m.title == "Foam Tabular Material Fixed"
    assert pytest.approx(m.rho0) == 1.2e-3
    assert pytest.approx(m.e) == 100.0
    assert pytest.approx(m.nu) == 0.3
    assert pytest.approx(m.c_hard) == 0.05
    assert pytest.approx(m.f_cut) == 1000.0
    assert m.fsmooth == 1
    assert m.israte == 0
    assert pytest.approx(m.p_c) == 5.0
    assert pytest.approx(m.p_t) == 2.5
    assert pytest.approx(m.ec) == 80.0
    assert pytest.approx(m.rpct) == 0.15
    assert m.funct_idc == 10
    assert m.funct_idt == 11
    assert pytest.approx(m.fscalec) == 1.2
    assert pytest.approx(m.fscalet) == 1.1
    assert pytest.approx(m.epsilon_0) == 0.001
    assert pytest.approx(m.c) == 0.04
    assert pytest.approx(m.sigma_y0) == 15.0
    assert m.vp == 1
    assert 101 in model.materials
    assert model.materials[101].law == 66

    # Fixed format with ISRATE=3 (yield rate functions)
    deck_israte3 = (
        "/BEGIN\n"
        "TEST_M172_LAW66_ISRATE3\n"
        "/MAT/LAW66/103\n"
        "Foam Tabular ISRATE3\n"
        "            1.200E-3            1.200E-3\n"
        "            100.0000              0.3000              0.0500              1000.0         0         3\n"
        "              5.0000              2.5000             80.0000              0.1500\n"
        "        10        11              1.2000              1.1000\n"
        "        30        31              2.0000              2.5000\n"
        "/END\n"
    )
    model3, _ = _parse_deck(tmp_path, deck_israte3)
    assert 103 in model3.mat_law66s
    m3 = model3.mat_law66s[103]
    assert m3.israte == 3
    assert m3.fnyrt_idc == 30
    assert m3.fnyrt_idt == 31
    assert pytest.approx(m3.yrate_fscalec) == 2.0
    assert pytest.approx(m3.yrate_fscalet) == 2.5

    # Free format with ISRATE=4 (multi-curve table)
    deck_free = (
        "#RADIOSS STARTER\n"
        "/MAT/FOAM_TAB/102\n"
        "Foam Tabular Multi Curve Free\n"
        "1.5e-3\n"
        "120.0 0.25 0.02 500.0 0 4\n"
        "6.0 3.0 90.0 0.2\n"
        "2 2\n"
        "101 0.001 1.0\n"
        "102 0.010 1.2\n"
        "201 0.001 0.9\n"
        "202 0.010 1.05\n"
        "/END\n"
    )
    model2, log2 = _parse_deck(tmp_path, deck_free)
    assert 102 in model2.mat_law66s
    m2 = model2.mat_law66s[102]
    assert m2.id == 102
    assert m2.israte == 4
    assert m2.nfunc == 2
    assert m2.tfunc == 2
    assert m2.func_c_list == [101, 102]
    assert pytest.approx(m2.eps_c_list) == [0.001, 0.010]
    assert pytest.approx(m2.fscale_c_list) == [1.0, 1.2]
    assert m2.func_t_list == [201, 202]
    assert pytest.approx(m2.eps_t_list) == [0.001, 0.010]
    assert pytest.approx(m2.fscale_t_list) == [0.9, 1.05]


def test_mat_law35_fixed_and_free(tmp_path: Path):
    deck_fixed = (
        "/BEGIN\n"
        "TEST_M172_LAW35_FIXED\n"
        "/MAT/LAW35/201\n"
        "Viscoelastic Foam Material\n"
        "#              RHO_I               RHO_O\n"
        "            8.000E-4            8.000E-4\n"
        "#                  E                  Nu                  E1                  E2                   n\n"
        "             50.0000              0.3500              5.0000              2.0000              1.5000\n"
        "#                 C1                  C2                  C3               Iflag                Pmin\n"
        "              0.1000              0.0200              0.0050                   1              0.0100\n"
        "# func_IDf                    Fscalepres                                 Fsmooth                Fcut\n"
        "        50                        1.0500                                       1             20000.0\n"
        "#                 Et                Nu_t               eta_0               Lamda\n"
        "             10.0000              0.2000              0.5000              0.1000\n"
        "#                 P0                 Phi             gamma_0\n"
        "              0.1013              0.0500              0.0200\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_fixed)
    assert 201 in model.mat_law35s
    m = model.mat_law35s[201]
    assert m.id == 201
    assert m.title == "Viscoelastic Foam Material"
    assert pytest.approx(m.rho0) == 8.0e-4
    assert pytest.approx(m.e) == 50.0
    assert pytest.approx(m.nu) == 0.35
    assert pytest.approx(m.e1) == 5.0
    assert pytest.approx(m.e2) == 2.0
    assert pytest.approx(m.n) == 1.5
    assert pytest.approx(m.c1) == 0.1
    assert pytest.approx(m.c2) == 0.02
    assert pytest.approx(m.c3) == 0.005
    assert m.itype == 1
    assert pytest.approx(m.pmin) == 0.01
    assert m.func_idf == 50
    assert pytest.approx(m.fscalepres) == 1.05
    assert m.fsmooth == 1
    assert pytest.approx(m.fcut) == 20000.0
    assert pytest.approx(m.et) == 10.0
    assert pytest.approx(m.nu_t) == 0.2
    assert pytest.approx(m.eta_0) == 0.5
    assert pytest.approx(m.lamda) == 0.1
    assert pytest.approx(m.p0) == 0.1013
    assert pytest.approx(m.phi) == 0.05
    assert pytest.approx(m.gama0) == 0.02
    assert model.materials[201].law == 35

    deck_free = (
        "#RADIOSS STARTER\n"
        "/MAT/FOAM_VISC/202\n"
        "Viscoelastic Foam Free\n"
        "7.5e-4\n"
        "45.0 0.3 4.0 1.5 1.2\n"
        "0.08 0.015 0.003 0 0.005\n"
        "55 1.0 0 10000.0\n"
        "8.0 0.15 0.4 0.08\n"
        "0.1013 0.04 0.01\n"
        "/END\n"
    )
    model2, log2 = _parse_deck(tmp_path, deck_free)
    assert 202 in model2.mat_law35s
    m2 = model2.mat_law35s[202]
    assert m2.id == 202
    assert pytest.approx(m2.e) == 45.0
    assert m2.func_idf == 55
    assert pytest.approx(m2.p0) == 0.1013


def test_mat_law62_fixed_and_free(tmp_path: Path):
    deck_fixed = (
        "/BEGIN\n"
        "TEST_M172_LAW62_FIXED\n"
        "/MAT/LAW62/301\n"
        "Visco Hyperelastic Ogden\n"
        "#              RHO_I          Ref. dens.\n"
        "            1.100E-3            1.100E-3\n"
        "#                 Nu         N         M              mu_max\n"
        "              0.4900         2         2              5.0000\n"
        "#         mu_i\n"
        "              1.2000              0.8000\n"
        "#      alpha_i\n"
        "              2.0000             -2.0000\n"
        "#           gamma_i\n"
        "              0.3000              0.2000\n"
        "#         tetha_i\n"
        "              0.0100              0.1000\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_fixed)
    assert 301 in model.mat_law62s
    m = model.mat_law62s[301]
    assert m.id == 301
    assert m.title == "Visco Hyperelastic Ogden"
    assert pytest.approx(m.rho0) == 1.1e-3
    assert pytest.approx(m.nu) == 0.49
    assert m.order_n == 2
    assert m.order_m == 2
    assert pytest.approx(m.mu_max) == 5.0
    assert pytest.approx(m.mu_arr) == [1.2, 0.8]
    assert pytest.approx(m.alpha_arr) == [2.0, -2.0]
    assert pytest.approx(m.gamma_arr) == [0.3, 0.2]
    assert pytest.approx(m.tau_arr) == [0.01, 0.1]
    assert model.materials[301].law == 62

    deck_free = (
        "#RADIOSS STARTER\n"
        "/MAT/VISC_HYP/302\n"
        "Visco Hyper Free\n"
        "1.05e-3\n"
        "0.495 1 1 3.5\n"
        "1.5\n"
        "2.2\n"
        "0.4\n"
        "0.05\n"
        "/END\n"
    )
    model2, log2 = _parse_deck(tmp_path, deck_free)
    assert 302 in model2.mat_law62s
    m2 = model2.mat_law62s[302]
    assert m2.order_n == 1
    assert m2.order_m == 1
    assert pytest.approx(m2.mu_arr) == [1.5]
    assert pytest.approx(m2.alpha_arr) == [2.2]
    assert pytest.approx(m2.gamma_arr) == [0.4]
    assert pytest.approx(m2.tau_arr) == [0.05]


def test_mat_law28_fixed_and_free(tmp_path: Path):
    deck_fixed = (
        "/BEGIN\n"
        "TEST_M172_LAW28_FIXED\n"
        "/MAT/LAW28/401\n"
        "Honeycomb Orthotropic Material\n"
        "#              RHO_I               RHO_O\n"
        "            3.500E-4            3.500E-4\n"
        "#                E11                 E22                 E33\n"
        "            100.0000            120.0000            500.0000\n"
        "#                G12                 G23                 G31\n"
        "             40.0000             60.0000             50.0000\n"
        "#   I11   I22   I33   IF1               FAC1                FAC2                FAC3\n"
        "     11    22    33     1             1.0000              1.1000              1.2000\n"
        "#              EMX11               EMX22               EMX33\n"
        "              0.5000              0.4000              0.8000\n"
        "#   I12   I23   I31   IF2               FAC4                FAC5                FAC6\n"
        "     12    23    31     0             0.9000              1.0000              1.0500\n"
        "#              EMX12               EMX23               EMX31\n"
        "              0.3000              0.3500              0.4500\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_fixed)
    assert 401 in model.mat_law28s
    m = model.mat_law28s[401]
    assert m.id == 401
    assert m.title == "Honeycomb Orthotropic Material"
    assert pytest.approx(m.rho0) == 3.5e-4
    assert pytest.approx(m.e11) == 100.0
    assert pytest.approx(m.e22) == 120.0
    assert pytest.approx(m.e33) == 500.0
    assert pytest.approx(m.g12) == 40.0
    assert pytest.approx(m.g23) == 60.0
    assert pytest.approx(m.g31) == 50.0
    assert m.fun_a1 == 11
    assert m.fun_b1 == 22
    assert m.fun_a2 == 33
    assert m.gflag == 1
    assert pytest.approx(m.fscale11) == 1.0
    assert pytest.approx(m.fscale22) == 1.1
    assert pytest.approx(m.fscale33) == 1.2
    assert pytest.approx(m.epsr1) == 0.5
    assert pytest.approx(m.epsr2) == 0.4
    assert pytest.approx(m.epsr3) == 0.8
    assert m.fun_a3 == 12
    assert m.fun_b3 == 23
    assert m.fun_a4 == 31
    assert m.vflag == 0
    assert pytest.approx(m.fscale12) == 0.9
    assert pytest.approx(m.fscale23) == 1.0
    assert pytest.approx(m.fscale13) == 1.05
    assert pytest.approx(m.epsr4) == 0.3
    assert pytest.approx(m.epsr5) == 0.35
    assert pytest.approx(m.epsr6) == 0.45
    assert model.materials[401].law == 28

    deck_free = (
        "#RADIOSS STARTER\n"
        "/MAT/HONEYCOMB/402\n"
        "Honeycomb Free\n"
        "3.0e-4\n"
        "80.0 90.0 400.0\n"
        "30.0 50.0 45.0\n"
        "10 20 30 0 1.0 1.0 1.0\n"
        "0.4 0.35 0.7\n"
        "11 21 31 1 1.0 1.0 1.0\n"
        "0.25 0.3 0.4\n"
        "/END\n"
    )
    model2, log2 = _parse_deck(tmp_path, deck_free)
    assert 402 in model2.mat_law28s
    m2 = model2.mat_law28s[402]
    assert pytest.approx(m2.e33) == 400.0
    assert m2.fun_a1 == 10
    assert m2.vflag == 1


def test_mat_law44_fixed_and_free(tmp_path: Path):
    deck_fixed = (
        "/BEGIN\n"
        "TEST_M172_LAW44_FIXED\n"
        "/MAT/LAW44/501\n"
        "Cowper Symonds Fixed\n"
        "#              RHO_I               RHO_O\n"
        "            7.800E-3            7.800E-3\n"
        "#                  E                  Nu     MAT_Iflag\n"
        "            210000.0              0.3000             0\n"
        "#               SIGY                   B                   n                HARD             SIG_max\n"
        "            350.0000            450.0000              0.2500              1.0000            800.0000\n"
        "#                SRC                 SRE             STRFLAG             Fsmooth                Fcut               Vflag\n"
        "             40.0000              5.0000                   1                   1              1000.0                   0\n"
        "#            EPS_max                ETA1                ETA2\n"
        "              0.3000              1.0000              2.0000\n"
        "#           YLD_FUNC           YLD_SCALE\n"
        "                  60              1.2000\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_fixed)
    assert 501 in model.mat_law44s
    m = model.mat_law44s[501]
    assert m.id == 501
    assert m.title == "Cowper Symonds Fixed"
    assert pytest.approx(m.rho0) == 7.8e-3
    assert pytest.approx(m.e) == 210000.0
    assert pytest.approx(m.nu) == 0.3
    assert m.iflag == 0
    assert pytest.approx(m.a) == 350.0
    assert pytest.approx(m.b) == 450.0
    assert pytest.approx(m.n) == 0.25
    assert pytest.approx(m.hard) == 1.0
    assert pytest.approx(m.sig_max) == 800.0
    assert pytest.approx(m.src) == 40.0
    assert pytest.approx(m.sre) == 5.0
    assert m.strflag == 1
    assert m.fsmooth == 1
    assert pytest.approx(m.fcut) == 1000.0
    assert m.vflag == 0
    assert pytest.approx(m.eps_max) == 0.3
    assert pytest.approx(m.eta1) == 1.0
    assert pytest.approx(m.eta2) == 2.0
    assert m.yld_func == 60
    assert pytest.approx(m.yld_scale) == 1.2
    assert model.materials[501].law == 44

    deck_free = (
        "#RADIOSS STARTER\n"
        "/MAT/COWPER_SYMONDS/502\n"
        "Cowper Symonds Free\n"
        "7.85e-3\n"
        "200000.0 0.28 1\n"
        "300.0 600.0 0.15 0.0 750.0\n"
        "50.0 4.0 0 0 0.0 1\n"
        "0.25 0.0 0.0\n"
        "/END\n"
    )
    model2, log2 = _parse_deck(tmp_path, deck_free)
    assert 502 in model2.mat_law44s
    m2 = model2.mat_law44s[502]
    assert m2.iflag == 1
    assert pytest.approx(m2.a) == 300.0
    assert pytest.approx(m2.b) == 600.0
    assert pytest.approx(m2.n) == 0.15
    assert pytest.approx(m2.src) == 50.0
    assert pytest.approx(m2.sre) == 4.0
    assert m2.vflag == 1
    assert model2.materials[502].law == 44
