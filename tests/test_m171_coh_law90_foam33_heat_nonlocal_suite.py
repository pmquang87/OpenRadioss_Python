"""Tests for Milestone M171: Cohesive Zone, Tabular Rate/Foam, Crushable Foam & Thermal/Nonlocal Material Modifiers Suite.

Covers:
- /MAT/LAW117 & /MAT/COH_MC: Cohesive element material model (fixed & free formats)
- /MAT/LAW90 & /MAT/PLAS_TAB: Strain-rate dependent tabular foam/plasticity material model (fixed & free formats)
- /MAT/LAW33 & /MAT/FOAM_PLAS: Crushable foam plasticity material model (fixed & free formats)
- /MAT/HEAT & /HEAT/MAT: Material thermal property modifier (fixed & free formats)
- /MAT/NONLOCAL & /NONLOCAL/MAT: Non-local damage regularization modifier (fixed & free formats)
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


def test_m171_mat_law117_fixed(tmp_path: Path):
    """Test /MAT/LAW117 in fixed format."""
    deck = (
        "#STARTER\n"
        "/MAT/LAW117/1\n"
        "Cohesive Law117 Mat\n"
        "#        Init. dens.          Ref. dens.\n"
        "             1.50E-6             1.50E-6\n"
        "#                 EN                  ES     Imass      Idel     Irupt\n"
        "             12000.0              8000.0         1         2         1\n"
        "#   FCT_TN    FCT_TT                  TN                  TS            Fscale_x\n"
        "        10        20                45.0                30.0                 1.5\n"
        "#                GIC                GIIC               EXP_G              EXP_BK               GAMMA\n"
        "                 0.5                 1.2                 1.8                 2.2                 1.5\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 1 in model.mat_law117s
    m = model.mat_law117s[1]
    assert m.id == 1
    assert m.title == "Cohesive Law117 Mat"
    assert pytest.approx(m.rho0) == 1.50e-6
    assert pytest.approx(m.refer_rho) == 1.50e-6
    assert pytest.approx(m.e_elas_n) == 12000.0
    assert pytest.approx(m.e_elas_s) == 8000.0
    assert m.imass == 1
    assert m.idel == 2
    assert m.irupt == 1
    assert m.fct_tn == 10
    assert m.fct_tt == 20
    assert pytest.approx(m.tmax_n) == 45.0
    assert pytest.approx(m.tmax_s) == 30.0
    assert pytest.approx(m.fscale_x) == 1.5
    assert pytest.approx(m.gic) == 0.5
    assert pytest.approx(m.giic) == 1.2
    assert pytest.approx(m.exp_g) == 1.8
    assert pytest.approx(m.exp_bk) == 2.2
    assert pytest.approx(m.gamma) == 1.5

    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 117
    assert mat.params["E_elas_n"] == 12000.0


def test_m171_mat_coh_mc_free(tmp_path: Path):
    """Test /MAT/COH_MC in free format."""
    deck = (
        "/MAT/COH_MC/2\n"
        "Cohesive Free\n"
        "1.6e-6 1.6e-6\n"
        "10000.0 7000.0 0 0 2\n"
        "0 0 50.0 35.0 1.0\n"
        "0.6 1.4 2.0 2.0 1.2\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 2 in model.mat_law117s
    m = model.mat_law117s[2]
    assert m.id == 2
    assert m.title == "Cohesive Free"
    assert pytest.approx(m.e_elas_n) == 10000.0
    assert pytest.approx(m.tmax_n) == 50.0
    assert pytest.approx(m.gic) == 0.6


def test_m171_mat_law90_fixed(tmp_path: Path):
    """Test /MAT/LAW90 in fixed format."""
    deck = (
        "#STARTER\n"
        "/MAT/LAW90/10\n"
        "Tabular Foam Mat Fixed\n"
        "#              Rho_I               Rho_O\n"
        "             5.00E-8             5.00E-8\n"
        "#                 E0              MAT_NU\n"
        "                80.0                0.05\n"
        "#       NL   Ismooth                Fcut               Shape                 Hys\n"
        "         2         1                50.0                 2.5                 0.3\n"
        "#  fct_idL             eps_dot              Fscale\n"
        "       101                 0.0                 1.0\n"
        "       102                10.0                 1.2\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 10 in model.mat_law90s
    m = model.mat_law90s[10]
    assert m.id == 10
    assert m.title == "Tabular Foam Mat Fixed"
    assert pytest.approx(m.rho0) == 5.00e-8
    assert pytest.approx(m.e0) == 80.0
    assert pytest.approx(m.nu) == 0.05
    assert m.nl == 2
    assert m.ismooth == 1
    assert pytest.approx(m.fcut) == 50.0
    assert pytest.approx(m.shape) == 2.5
    assert pytest.approx(m.hys) == 0.3
    assert m.fct_ids == [101, 102]
    assert m.eps_dots == [0.0, 10.0]
    assert m.fscales == [1.0, 1.2]

    assert 10 in model.materials
    mat = model.materials[10]
    assert mat.law == 90
    assert mat.params["NL"] == 2


def test_m171_mat_tab_foam_free(tmp_path: Path):
    """Test /MAT/TAB_FOAM in free format."""
    deck = (
        "/MAT/TAB_FOAM/11\n"
        "Tabular Free\n"
        "6.0e-8 6.0e-8\n"
        "100.0 0.0\n"
        "1 0 0.0 3.0 0.2\n"
        "201 0.0 1.0\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 11 in model.mat_law90s
    m = model.mat_law90s[11]
    assert m.id == 11
    assert m.title == "Tabular Free"
    assert pytest.approx(m.e0) == 100.0
    assert m.nl == 1
    assert m.fct_ids == [201]


def test_m171_mat_law33_fixed(tmp_path: Path):
    """Test /MAT/LAW33 & /MAT/FOAM_PLAS in fixed format."""
    deck = (
        "#STARTER\n"
        "/MAT/LAW33/20\n"
        "Crushable Foam Mat Fixed\n"
        "#        Init. dens.          Ref. dens.\n"
        "             4.50E-8             4.50E-8\n"
        "#                  E        Ka        If              Fscale\n"
        "                40.0         1       301                 1.1\n"
        "#                 P0                 Phi             Gamma_0\n"
        "                 0.1                0.05                0.02\n"
        "#                  A                   B                   C\n"
        "                 1.2                 0.8                 0.4\n"
        "#                 E1                  E2                  Et            eta_comp           eta_shear\n"
        "                10.0                 5.0                 2.0                 0.1                0.05\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 20 in model.mat_law33s
    m = model.mat_law33s[20]
    assert m.id == 20
    assert m.title == "Crushable Foam Mat Fixed"
    assert pytest.approx(m.rho0) == 4.50e-8
    assert pytest.approx(m.e) == 40.0
    assert m.itype == 1
    assert m.fun_a1 == 301
    assert pytest.approx(m.ifscale) == 1.1
    assert pytest.approx(m.p0) == 0.1
    assert pytest.approx(m.phi) == 0.05
    assert pytest.approx(m.gama0) == 0.02
    assert pytest.approx(m.a0) == 1.2
    assert pytest.approx(m.a1) == 0.8
    assert pytest.approx(m.a2) == 0.4
    assert pytest.approx(m.e1) == 10.0
    assert pytest.approx(m.e2) == 5.0
    assert pytest.approx(m.etan) == 2.0
    assert pytest.approx(m.eta1) == 0.1
    assert pytest.approx(m.eta2) == 0.05

    assert 20 in model.materials
    mat = model.materials[20]
    assert mat.law == 33
    assert mat.params["E"] == 40.0


def test_m171_mat_foam_plas_free(tmp_path: Path):
    """Test /MAT/FOAM_PLAS in free format."""
    deck = (
        "/MAT/FOAM_PLAS/21\n"
        "Foam Plas Free\n"
        "4.0e-8 4.0e-8\n"
        "35.0 0 401 1.0\n"
        "0.08 0.04 0.015\n"
        "1.0 0.7 0.3\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 21 in model.mat_law33s
    m = model.mat_law33s[21]
    assert m.id == 21
    assert m.title == "Foam Plas Free"
    assert pytest.approx(m.e) == 35.0
    assert m.itype == 0
    assert m.fun_a1 == 401
    assert pytest.approx(m.p0) == 0.08


def test_m171_mat_heat_fixed_and_free(tmp_path: Path):
    """Test /MAT/HEAT and /HEAT/MAT in fixed & free formats."""
    deck = (
        "#STARTER\n"
        "/MAT/HEAT/50\n"
        "Heat Mod 50\n"
        "#                 T0             RHO0_CP                  AS                  BS\n"
        "               293.0              2.4E-3                45.0                 0.1\n"
        "#                 T1                  AL                  BL               EFRAC\n"
        "              1800.0                30.0                0.05                 0.9\n"
        "/HEAT/MAT/51\n"
        "Heat Mod 51 Free\n"
        "298.0 2.5e-3 50.0 0.12\n"
        "1900.0 32.0 0.06 0.95\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 50 in model.mat_heat_modifiers
    h50 = model.mat_heat_modifiers[50]
    assert h50.id == 50
    assert pytest.approx(h50.t0) == 293.0
    assert pytest.approx(h50.rho0_cp) == 2.4e-3
    assert pytest.approx(h50.as_solid) == 45.0
    assert pytest.approx(h50.bs_solid) == 0.1
    assert pytest.approx(h50.t1) == 1800.0
    assert pytest.approx(h50.al_liquid) == 30.0
    assert pytest.approx(h50.bl_liquid) == 0.05
    assert pytest.approx(h50.efrac) == 0.9

    assert 51 in model.mat_heat_modifiers
    h51 = model.mat_heat_modifiers[51]
    assert h51.id == 51
    assert pytest.approx(h51.t0) == 298.0
    assert pytest.approx(h51.efrac) == 0.95


def test_m171_mat_nonlocal_fixed_and_free(tmp_path: Path):
    """Test /MAT/NONLOCAL in fixed and free formats."""
    deck = (
        "#STARTER\n"
        "/MAT/NONLOCAL/60\n"
        "Nonlocal Mod 60\n"
        "#             LENGTH              LE_MAX\n"
        "                 2.5                 0.5\n"
        "/MAT/NONLOCAL/61\n"
        "Nonlocal Mod 61\n"
        "3.0 0.6\n"
        "/END\n"
    )
    model, log = _parse_deck(tmp_path, deck)
    assert not log.errors, f"Errors: {log.errors}"
    assert 60 in model.mat_nonlocal_modifiers
    nl60 = model.mat_nonlocal_modifiers[60]
    assert nl60.id == 60
    assert pytest.approx(nl60.length) == 2.5
    assert pytest.approx(nl60.le_max) == 0.5

    assert 61 in model.mat_nonlocal_modifiers
    nl61 = model.mat_nonlocal_modifiers[61]
    assert nl61.id == 61
    assert pytest.approx(nl61.length) == 3.0
    assert pytest.approx(nl61.le_max) == 0.6
