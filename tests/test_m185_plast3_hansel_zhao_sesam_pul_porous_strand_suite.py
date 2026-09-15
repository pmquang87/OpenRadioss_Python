"""Test suite for Milestone M185: Tabulated Temperature/Rate Plasticity (LAW60/PLAS_T3),
Hänsel Martensite Plasticity (LAW63/HANSEL), Zhao Hardening Plasticity (LAW48/ZHAO),
SESAME Hydrodynamic EOS Material (LAW26/SESAM), Pulley Spring Property (PROP TYPE12/SPR_PUL),
Porous Solid Property (PROP TYPE15/POROUS), and Multi-Strand Cable Property (PROP TYPE28/NSTRAND).
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


def test_mat_law60_plas_t3(tmp_path: Path):
    """Verify /MAT/LAW60 and /MAT/PLAS_T3 parsing in both fixed and free format."""
    # 1. Fixed format
    deck_fixed = (
        "/BEGIN\n"
        "TEST_MAT_LAW60\n"
        "/MAT/LAW60/1\n"
        "Law60 Tabulated Plasticity Fixed\n"
        "#              RHO_I               RHO_O\n"
        "             7.85e-9              0.0000\n"
        "#                  E                  NU           EPS_P_MAX              EPS_T1              EPS_T2\n"
        "            210000.0                 0.3                 0.5                 0.4                 0.6\n"
        "#    N_FUNCT   F_SMOOTH               CHARD                FCUT\n"
        "           6           1                 0.1              1000.0\n"
        "#      IPFUN             FPSCALE\n"
        "           5                 1.0\n"
        "#      FUN_1       FUN_2       FUN_3       FUN_4       FUN_5\n"
        "         101         102         103         104         105\n"
        "#      FUN_6\n"
        "         106\n"
        "#            FSCALE1             FSCALE2             FSCALE3             FSCALE4             FSCALE5\n"
        "                 1.0                 1.1                 1.2                 1.3                 1.4\n"
        "#            FSCALE6\n"
        "                 1.5\n"
        "#              EPSR1               EPSR2               EPSR3               EPSR4               EPSR5\n"
        "                0.01                 0.1                 1.0                10.0               100.0\n"
        "#              EPSR6\n"
        "              1000.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_fixed, "law60_fixed_0000.rad")
    assert 1 in model.mat_law60s
    m = model.mat_law60s[1]
    assert m.id == 1
    assert m.title == "Law60 Tabulated Plasticity Fixed"
    assert pytest.approx(m.rho, rel=1e-5) == 7.85e-9
    assert pytest.approx(m.e, rel=1e-5) == 210000.0
    assert pytest.approx(m.nu, rel=1e-5) == 0.3
    assert pytest.approx(m.eps_p_max, rel=1e-5) == 0.5
    assert m.nfunc == 6
    assert m.fsmooth == 1
    assert pytest.approx(m.chard, rel=1e-5) == 0.1
    assert pytest.approx(m.fcut, rel=1e-5) == 1000.0
    assert m.xr_fun == 5
    assert m.fun_ids == [101, 102, 103, 104, 105, 106]
    assert len(m.fscales) == 6
    assert pytest.approx(m.fscales[1], rel=1e-5) == 1.1
    assert len(m.eps_rates) == 6
    assert pytest.approx(m.eps_rates[3], rel=1e-5) == 10.0
    assert 1 in model.materials

    # 2. Free format alias /MAT/PLAS_T3
    deck_free = (
        "/MAT/PLAS_T3/2\n"
        "Free Format Law60\n"
        "2.7e-9 0.0\n"
        "70000.0 0.33 0.3 0.25 0.35\n"
        "5 0 0.05 500.0\n"
        "0 1.0\n"
        "201 202 203 204 205\n"
        "1.0 1.0 1.0 1.0 1.0\n"
        "0.001 0.01 0.1 1.0 10.0\n"
    )
    model2, log2 = _parse_deck(tmp_path, deck_free, "law60_free_0000.rad")
    assert 2 in model2.mat_law60s
    m2 = model2.mat_law60s[2]
    assert m2.id == 2
    assert pytest.approx(m2.rho, rel=1e-5) == 2.7e-9
    assert pytest.approx(m2.e, rel=1e-5) == 70000.0
    assert m2.fun_ids == [201, 202, 203, 204, 205]


def test_mat_law63_hansel(tmp_path: Path):
    """Verify /MAT/LAW63 and /MAT/HANSEL parsing in both fixed and free format."""
    # 1. Fixed format
    deck_fixed = (
        "/BEGIN\n"
        "TEST_MAT_LAW63\n"
        "/MAT/LAW63/1\n"
        "Law63 Hansel Plasticity Fixed\n"
        "#              RHO_I               RHO_O\n"
        "             7.85e-9              0.0000\n"
        "#                  E                  NU                  CP\n"
        "            210000.0                 0.3               460.0\n"
        "#                  A                   B                   Q                   C                   D\n"
        "               200.0               500.0               100.0               0.015                0.25\n"
        "#                  P                 AHS                 BHS                   M                   N\n"
        "                 1.5                 2.0                 0.5                 1.2                 0.8\n"
        "#                 K1                  K2                  DH                 VM0                EPS0\n"
        "               0.001                0.02               10.0                 0.05               0.002\n"
        "#              TEMP0\n"
        "               293.15\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_fixed, "law63_fixed_0000.rad")
    assert 1 in model.mat_law63s
    m = model.mat_law63s[1]
    assert m.id == 1
    assert m.title == "Law63 Hansel Plasticity Fixed"
    assert pytest.approx(m.rho, rel=1e-5) == 7.85e-9
    assert pytest.approx(m.e, rel=1e-5) == 210000.0
    assert pytest.approx(m.cp, rel=1e-5) == 460.0
    assert pytest.approx(m.a, rel=1e-5) == 200.0
    assert pytest.approx(m.b, rel=1e-5) == 500.0
    assert pytest.approx(m.q, rel=1e-5) == 100.0
    assert pytest.approx(m.c, rel=1e-5) == 0.015
    assert pytest.approx(m.p, rel=1e-5) == 1.5
    assert pytest.approx(m.ahs, rel=1e-5) == 2.0
    assert pytest.approx(m.k1, rel=1e-5) == 0.001
    assert pytest.approx(m.vm0, rel=1e-5) == 0.05
    assert pytest.approx(m.t0, rel=1e-5) == 293.15
    assert 1 in model.materials

    # 2. Free format alias /MAT/HANSEL
    deck_free = (
        "/MAT/HANSEL/2\n"
        "Free Format Hansel\n"
        "7.85e-9\n"
        "200000.0 0.3 500.0\n"
        "250.0 400.0 80.0 0.02 0.3\n"
        "1.0 1.5 0.4 1.0 0.5\n"
        "0.002 0.01 5.0 0.01 0.001\n"
        "300.0\n"
    )
    model2, log2 = _parse_deck(tmp_path, deck_free, "law63_free_0000.rad")
    assert 2 in model2.mat_law63s
    m2 = model2.mat_law63s[2]
    assert pytest.approx(m2.a, rel=1e-5) == 250.0
    assert pytest.approx(m2.t0, rel=1e-5) == 300.0


def test_mat_law48_zhao(tmp_path: Path):
    """Verify /MAT/LAW48 and /MAT/ZHAO parsing in both fixed and free format."""
    # 1. Fixed format
    deck_fixed = (
        "/BEGIN\n"
        "TEST_MAT_LAW48\n"
        "/MAT/LAW48/1\n"
        "Law48 Zhao Plasticity Fixed\n"
        "#              RHO_I               RHO_O\n"
        "             7.85e-9              0.0000\n"
        "#                  E                  NU\n"
        "            210000.0                 0.3\n"
        "#                  A                   B                   N               CHARD             SIG_MAX\n"
        "               300.0               600.0                 0.4                 0.2               800.0\n"
        "#                  C                   D                   M                   C                   K\n"
        "                0.05                0.01                 0.5                10.0                 1.5\n"
        "#         EPS_RATE_0                FCUT\n"
        "               0.001              5000.0\n"
        "#            EPS_MAX              EPS_T1              EPS_T2\n"
        "                 0.8                 0.7                 0.9\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_fixed, "law48_fixed_0000.rad")
    assert 1 in model.mat_law48s
    m = model.mat_law48s[1]
    assert m.id == 1
    assert m.title == "Law48 Zhao Plasticity Fixed"
    assert pytest.approx(m.rho, rel=1e-5) == 7.85e-9
    assert pytest.approx(m.e, rel=1e-5) == 210000.0
    assert pytest.approx(m.a, rel=1e-5) == 300.0
    assert pytest.approx(m.b, rel=1e-5) == 600.0
    assert pytest.approx(m.n, rel=1e-5) == 0.4
    assert pytest.approx(m.chard, rel=1e-5) == 0.2
    assert pytest.approx(m.sig_max, rel=1e-5) == 800.0
    assert pytest.approx(m.c, rel=1e-5) == 0.05
    assert pytest.approx(m.d, rel=1e-5) == 0.01
    assert pytest.approx(m.m, rel=1e-5) == 0.5
    assert pytest.approx(m.e1, rel=1e-5) == 10.0
    assert pytest.approx(m.k, rel=1e-5) == 1.5
    assert pytest.approx(m.eps_rate_0, rel=1e-5) == 0.001
    assert pytest.approx(m.fcut, rel=1e-5) == 5000.0
    assert pytest.approx(m.eps_max, rel=1e-5) == 0.8
    assert 1 in model.materials

    # 2. Free format alias /MAT/ZHAO
    deck_free = (
        "/MAT/ZHAO/2\n"
        "Free Format Zhao\n"
        "7.85e-9\n"
        "210000.0 0.3\n"
        "350.0 500.0 0.35 0.0 900.0\n"
        "0.02 0.005 0.4 5.0 1.2\n"
        "0.01 1000.0\n"
        "0.5 0.45 0.55\n"
    )
    model2, log2 = _parse_deck(tmp_path, deck_free, "law48_free_0000.rad")
    assert 2 in model2.mat_law48s
    m2 = model2.mat_law48s[2]
    assert pytest.approx(m2.a, rel=1e-5) == 350.0
    assert pytest.approx(m2.sig_max, rel=1e-5) == 900.0


def test_mat_law26_sesam(tmp_path: Path):
    """Verify /MAT/LAW26 and /MAT/SESAM parsing in both fixed and free format."""
    # 1. Fixed format
    deck_fixed = (
        "/BEGIN\n"
        "TEST_MAT_LAW26\n"
        "/MAT/LAW26/1\n"
        "Law26 Sesame Hydrodynamic Fixed\n"
        "#              RHO_I               RHO_O\n"
        "              2.7e-9              0.0000\n"
        "#                  E                  NU\n"
        "             70000.0                0.33\n"
        "#                  A                   B                   N              EPSMAX              SIGMAX\n"
        "               150.0               300.0                0.35                 0.5               450.0\n"
        "#                 E0\n"
        "                 0.0\n"
        "#                                                                                           SESAM301\n"
        "aluminum_sesame_301.dat\n"
        "#                  C                EPS0                   M               TMELT                TMAX\n"
        "                0.01                0.001                0.8               933.0              1500.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_fixed, "law26_fixed_0000.rad")
    assert 1 in model.mat_law26s
    m = model.mat_law26s[1]
    assert m.id == 1
    assert m.title == "Law26 Sesame Hydrodynamic Fixed"
    assert pytest.approx(m.rho, rel=1e-5) == 2.7e-9
    assert pytest.approx(m.e, rel=1e-5) == 70000.0
    assert pytest.approx(m.a, rel=1e-5) == 150.0
    assert pytest.approx(m.b, rel=1e-5) == 300.0
    assert pytest.approx(m.n, rel=1e-5) == 0.35
    assert pytest.approx(m.eps_max, rel=1e-5) == 0.5
    assert pytest.approx(m.sig_max, rel=1e-5) == 450.0
    assert m.sesam301 == "aluminum_sesame_301.dat"
    assert pytest.approx(m.tmelt, rel=1e-5) == 933.0
    assert pytest.approx(m.tmax, rel=1e-5) == 1500.0
    assert 1 in model.materials

    # 2. Free format alias /MAT/SESAM
    deck_free = (
        "/MAT/SESAM/2\n"
        "Free Format Sesam\n"
        "2.7e-9\n"
        "70000.0 0.33\n"
        "120.0 250.0 0.3 0.4 400.0\n"
        "10.0\n"
        "eos_table_301.dat\n"
        "0.02 0.01 1.0 1000.0 2000.0\n"
    )
    model2, log2 = _parse_deck(tmp_path, deck_free, "law26_free_0000.rad")
    assert 2 in model2.mat_law26s
    m2 = model2.mat_law26s[2]
    assert m2.sesam301 == "eos_table_301.dat"
    assert pytest.approx(m2.e0, rel=1e-5) == 10.0


def test_prop_type12_spr_pul(tmp_path: Path):
    """Verify /PROP/TYPE12 and /PROP/SPR_PUL parsing in both fixed and free format."""
    # 1. Fixed format
    deck_fixed = (
        "/BEGIN\n"
        "TEST_PROP_TYPE12\n"
        "/PROP/TYPE12/1\n"
        "Prop Type12 Pulley Spring Fixed\n"
        "#               MASS                                 SENS_ID    ISFLAG     ILENG                FRIC\n"
        "              0.0025                                      10         1         0                0.15\n"
        "#                  K                   C                   A                   B                   D\n"
        "              5000.0                10.0                 1.0                 0.0                 1.0\n"
        "#  FCT_ID1         H   FCT_ID2                                          DELTAMIN            DELTAMAX\n"
        "        51         1        52                                             -50.0               150.0\n"
        "#            FSCALE1                   E             ASCALEX\n"
        "                 1.2                 0.0                 1.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_fixed, "type12_fixed_0000.rad")
    assert 1 in model.prop_type12s
    p = model.prop_type12s[1]
    assert p.id == 1
    assert p.title == "Prop Type12 Pulley Spring Fixed"
    assert pytest.approx(p.mass, rel=1e-5) == 0.0025
    assert p.isensor == 10
    assert p.isflag == 1
    assert p.ileng == 0
    assert pytest.approx(p.fric, rel=1e-5) == 0.15
    assert pytest.approx(p.stiff1, rel=1e-5) == 5000.0
    assert pytest.approx(p.damp1, rel=1e-5) == 10.0
    assert p.fun_a1 == 51
    assert p.hflag1 == 1
    assert p.fun_b1 == 52
    assert pytest.approx(p.min_rup1, rel=1e-5) == -50.0
    assert pytest.approx(p.max_rup1, rel=1e-5) == 150.0
    assert pytest.approx(p.prop_x_f, rel=1e-5) == 1.2
    assert 1 in model.properties

    # 2. Free format alias /PROP/SPR_PUL
    deck_free = (
        "/PROP/SPR_PUL/2\n"
        "Free Format Pulley\n"
        "0.005 20 0 1 0.2\n"
        "10000.0 20.0 1.0 0.0 1.0\n"
        "61 0 0 -100.0 200.0\n"
        "1.0 0.0 1.0\n"
    )
    model2, log2 = _parse_deck(tmp_path, deck_free, "type12_free_0000.rad")
    assert 2 in model2.prop_type12s
    p2 = model2.prop_type12s[2]
    assert p2.isensor == 20
    assert pytest.approx(p2.fric, rel=1e-5) == 0.2
    assert pytest.approx(p2.stiff1, rel=1e-5) == 10000.0


def test_prop_type15_porous(tmp_path: Path):
    """Verify /PROP/TYPE15 and /PROP/POROUS parsing in both fixed and free format."""
    # 1. Fixed format
    deck_fixed = (
        "/BEGIN\n"
        "TEST_PROP_TYPE15\n"
        "/PROP/TYPE15/1\n"
        "Prop Type15 Porous Solid Fixed\n"
        "#                 QA                  QB                   H\n"
        "                 0.0                 0.0                 0.1\n"
        "#                POR\n"
        "                0.75\n"
        "#                 R1                  R2                  R3\n"
        "               100.0               200.0               300.0\n"
        "# SKEW_IDR      IHON\n"
        "         5         1\n"
        "#      ITU               ALPHA               L_MIX\n"
        "         1                 0.2                 5.0\n"
        "#RBODY_IDS\n"
        "        12\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_fixed, "type15_fixed_0000.rad")
    assert 1 in model.prop_type15s
    p = model.prop_type15s[1]
    assert p.id == 1
    assert p.title == "Prop Type15 Porous Solid Fixed"
    assert pytest.approx(p.h, rel=1e-5) == 0.1
    assert pytest.approx(p.poros, rel=1e-5) == 0.75
    assert pytest.approx(p.r1, rel=1e-5) == 100.0
    assert pytest.approx(p.r2, rel=1e-5) == 200.0
    assert pytest.approx(p.r3, rel=1e-5) == 300.0
    assert p.skew_csid == 5
    assert p.ihon == 1
    assert p.itu == 1
    assert pytest.approx(p.alpha, rel=1e-5) == 0.2
    assert pytest.approx(p.l_mix, rel=1e-5) == 5.0
    assert p.irby == 12
    assert 1 in model.properties

    # 2. Free format alias /PROP/POROUS
    deck_free = (
        "/PROP/POROUS/2\n"
        "Free Format Porous\n"
        "0.0 0.0 0.15\n"
        "0.6\n"
        "50.0 50.0 150.0\n"
        "0 0\n"
        "0 0.1 0.0\n"
        "0\n"
    )
    model2, log2 = _parse_deck(tmp_path, deck_free, "type15_free_0000.rad")
    assert 2 in model2.prop_type15s
    p2 = model2.prop_type15s[2]
    assert pytest.approx(p2.poros, rel=1e-5) == 0.6
    assert pytest.approx(p2.r3, rel=1e-5) == 150.0


def test_prop_type28_nstrand(tmp_path: Path):
    """Verify /PROP/TYPE28 and /PROP/NSTRAND parsing in both fixed and free format."""
    # 1. Fixed format
    deck_fixed = (
        "/BEGIN\n"
        "TEST_PROP_TYPE28\n"
        "/PROP/TYPE28/1\n"
        "Prop Type28 Multi-Strand Cable Fixed\n"
        "#               MASS                   K                   C\n"
        "              0.0012              2500.0                 5.0\n"
        "#    FUN_ID1   FUN_ID2         EPSILON_MIN         EPSILON_MAX\n"
        "          11        12               -0.05                0.25\n"
        "#                MU1                 MU2\n"
        "                0.12                0.18\n"
        "#     TYPE         K                  MU\n"
        "PULLEY             1                0.10\n"
        "STRAND             2                0.15\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck_fixed, "type28_fixed_0000.rad")
    assert 1 in model.prop_type28s
    p = model.prop_type28s[1]
    assert p.id == 1
    assert p.title == "Prop Type28 Multi-Strand Cable Fixed"
    assert pytest.approx(p.mass, rel=1e-5) == 0.0012
    assert pytest.approx(p.k, rel=1e-5) == 2500.0
    assert pytest.approx(p.c, rel=1e-5) == 5.0
    assert p.fun_a1 == 11
    assert p.fun_b1 == 12
    assert pytest.approx(p.strain1, rel=1e-5) == -0.05
    assert pytest.approx(p.strain2, rel=1e-5) == 0.25
    assert pytest.approx(p.mu1, rel=1e-5) == 0.12
    assert pytest.approx(p.mu2, rel=1e-5) == 0.18
    assert len(p.layers) == 2
    assert p.layers[0].type_name == "PULLEY"
    assert p.layers[0].k_id == 1
    assert pytest.approx(p.layers[0].mu, rel=1e-5) == 0.10
    assert p.layers[1].type_name == "STRAND"
    assert p.layers[1].k_id == 2
    assert pytest.approx(p.layers[1].mu, rel=1e-5) == 0.15
    assert 1 in model.properties

    # 2. Free format alias /PROP/NSTRAND
    deck_free = (
        "/PROP/NSTRAND/2\n"
        "Free Format Strand\n"
        "0.002 5000.0 10.0\n"
        "21 22 -0.1 0.3\n"
        "0.1 0.2\n"
        "PULLEY 3 0.12\n"
        "STRAND 4 0.22\n"
    )
    model2, log2 = _parse_deck(tmp_path, deck_free, "type28_free_0000.rad")
    assert 2 in model2.prop_type28s
    p2 = model2.prop_type28s[2]
    assert pytest.approx(p2.k, rel=1e-5) == 5000.0
    assert len(p2.layers) == 2
    assert p2.layers[0].type_name == "PULLEY"
    assert p2.layers[0].k_id == 3


def test_m185_error_guards(tmp_path: Path):
    """Test defaults and edge cases for M185 materials and properties."""
    # Blank/minimal cards
    deck_minimal = (
        "/MAT/LAW60/10\n"
        "Minimal Law60\n"
        "\n"
        "/MAT/LAW63/20\n"
        "Minimal Law63\n"
        "\n"
        "/MAT/LAW48/30\n"
        "Minimal Law48\n"
        "\n"
        "/MAT/LAW26/40\n"
        "Minimal Law26\n"
        "\n"
        "/PROP/TYPE12/50\n"
        "Minimal Prop12\n"
        "\n"
        "/PROP/TYPE15/60\n"
        "Minimal Prop15\n"
        "\n"
        "/PROP/TYPE28/70\n"
        "Minimal Prop28\n"
        "\n"
    )
    model, log = _parse_deck(tmp_path, deck_minimal, "m185_minimal_0000.rad")
    assert 10 in model.mat_law60s
    assert 20 in model.mat_law63s
    assert 30 in model.mat_law48s
    assert 40 in model.mat_law26s
    assert 50 in model.prop_type12s
    assert 60 in model.prop_type15s
    assert 70 in model.prop_type28s

