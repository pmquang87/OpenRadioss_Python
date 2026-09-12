"""Tests for /MAT/LAW104 Starter reader, Model representation, and Diagnostics checks (M574)."""

import math
from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import MaterialLaw104, Part
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_mat_law104, check_materials, check_model


def _parse_deck(tmp_path: Path, text: str) -> Model:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert not log.errors, f"Parse errors: {log.errors}"
    return model


def test_starter_keyword_reader_and_aliases(tmp_path: Path):
    """Test reading /MAT/LAW104 and its aliases (/MAT/DRUCKER, /MAT/JOHNS_VOCE_DRUCKER, /MAT/PLAS_DRUCK)."""
    deck_text = """# OpenRadioss Starter Deck
/BEGIN
Test LAW104
/MAT/LAW104/1041
Mild Steel Drucker
#    rho_initial        refer_rho
         7.8e-09          7.8e-09
#              E               nu     Ires
        210000.0              0.3        1
#          yld_0                H       Qv        Bv       Cdr
           350.0            400.0    120.0      15.0       0.5
#            Cjc             Eps0     Fcut
            0.04              1.0  15000.0
#             mu             Tref     Tini
           0.001           293.15   293.15
#            ETA               Cp   EpsIso     EpsAd
             0.9         450.0e06      1.0     100.0

/MAT/DRUCKER/1042
Drucker Alias
         7.8e-09
        210000.0              0.3        2
           400.0            500.0    100.0      10.0      -1.2
            0.02              1.0  10000.0
             0.0           293.15   293.15
             0.9         450.0e06      1.0     100.0

/MAT/JOHNS_VOCE_DRUCKER/1043
Johns Voce Drucker Alias
         7.8e-09
        210000.0              0.3        1
           300.0            200.0     80.0      25.0       1.8
            0.05              0.5  12000.0
           0.002           293.15   300.00
            0.85         500.0e06      0.5      50.0

/MAT/PLAS_DRUCK/1044
Plas Druck Alias
         7.8e-09
        200000.0             0.28        1
           500.0            100.0     50.0       8.0       0.0
            0.03              1.0  10000.0
             0.0           293.15   293.15
             0.9         450.0e06      1.0     100.0
/END
"""
    model = _parse_deck(tmp_path, deck_text)

    assert 1041 in model.mat_law104s
    assert 1042 in model.mat_law104s
    assert 1043 in model.mat_law104s
    assert 1044 in model.mat_law104s

    # Verify alias collections on model
    assert 1041 in model.mat_druckers
    assert 1042 in model.mat_johns_voce_druckers
    assert 1043 in model.mat_plas_drucks

    # Verify parsed parameters for mat 1041
    m1 = model.mat_law104s[1041]
    assert m1.id == 1041
    assert m1.title == "Mild Steel Drucker"
    assert math.isclose(m1.rho0, 7.8e-09, rel_tol=1e-6)
    assert math.isclose(m1.young, 210000.0, rel_tol=1e-6)
    assert math.isclose(m1.nu, 0.3, rel_tol=1e-6)
    assert m1.ires == 1
    assert math.isclose(m1.sigma0_yld, 350.0, rel_tol=1e-6)
    assert math.isclose(m1.h, 400.0, rel_tol=1e-6)
    assert math.isclose(m1.q_voce, 120.0, rel_tol=1e-6)
    assert math.isclose(m1.b_voce, 15.0, rel_tol=1e-6)
    assert math.isclose(m1.c_dr, 0.5, rel_tol=1e-6)
    assert math.isclose(m1.c_jc, 0.04, rel_tol=1e-6)
    assert math.isclose(m1.eps0, 1.0, rel_tol=1e-6)
    assert math.isclose(m1.fcut, 15000.0, rel_tol=1e-6)
    assert math.isclose(m1.tss, 0.001, rel_tol=1e-6)
    assert math.isclose(m1.tref, 293.15, rel_tol=1e-6)
    assert math.isclose(m1.tini, 293.15, rel_tol=1e-6)
    assert math.isclose(m1.eta, 0.9, rel_tol=1e-6)
    assert math.isclose(m1.cp, 450.0e06, rel_tol=1e-6)
    assert math.isclose(m1.eps_iso, 1.0, rel_tol=1e-6)
    assert math.isclose(m1.eps_ad, 100.0, rel_tol=1e-6)

    # Verify model entity derived properties
    assert math.isclose(m1.rho, 7.8e-09, rel_tol=1e-6)
    assert math.isclose(m1.rhor, 7.8e-09, rel_tol=1e-6)
    assert math.isclose(m1.e, 210000.0, rel_tol=1e-6)
    assert math.isclose(m1.E, 210000.0, rel_tol=1e-6)
    expected_g = 210000.0 / (2.0 * 1.3)
    expected_k = 210000.0 / (3.0 * (1.0 - 0.6))
    assert math.isclose(m1.G, expected_g, rel_tol=1e-6)
    assert math.isclose(m1.K, expected_k, rel_tol=1e-6)
    assert math.isclose(m1.bulk, expected_k, rel_tol=1e-6)
    expected_c_solid = math.sqrt((expected_k + 4.0 / 3.0 * expected_g) / 7.8e-09)
    assert math.isclose(m1.sound_speed, expected_c_solid, rel_tol=1e-6)
    expected_c_shell = math.sqrt((210000.0 / (1.0 - 0.09)) / 7.8e-09)
    assert math.isclose(m1.sound_speed_shell, expected_c_shell, rel_tol=1e-6)


def test_starter_checks_diagnostics():
    """Test starter diagnostic checks for /MAT/LAW104."""
    log = MessageLog()

    # 1. Zero density (ANCMSG 1514)
    mat_bad_rho = MaterialLaw104(id=1, young=210000.0, nu=0.3, rho0=0.0)
    check_mat_law104(mat=mat_bad_rho, log=log)
    assert any("ANCMSG 1514" in msg or "ZERO OR NEGATIVE DENSITY" in msg for msg in log.messages)

    # 2. Zero Young modulus (ANCMSG 276)
    log2 = MessageLog()
    mat_bad_e = MaterialLaw104(id=2, young=0.0, nu=0.3, rho0=7.8e-09)
    check_mat_law104(mat=mat_bad_e, log=log2)
    assert any("ANCMSG 276" in msg or "ZERO OR NEGATIVE YOUNG" in msg for msg in log2.messages)

    # 3. Invalid Poisson ratio (ANCMSG 300)
    log3 = MessageLog()
    mat_bad_nu = MaterialLaw104(id=3, young=210000.0, nu=0.52, rho0=7.8e-09)
    check_mat_law104(mat=mat_bad_nu, log=log3)
    assert any("ANCMSG 300" in msg or "INVALID POISSON" in msg for msg in log3.messages)

    # 4. Drucker coefficient upper bound warning (ANCMSG 1651)
    log4 = MessageLog()
    mat_high_cdr = MaterialLaw104(id=4, young=210000.0, nu=0.3, rho0=7.8e-09, cdr=3.5)
    check_mat_law104(mat=mat_high_cdr, log=log4)
    assert any("ANCMSG 1651" in msg or "EXCEEDS 2.25" in msg for msg in log4.messages)

    # 5. Drucker coefficient lower bound warning (ANCMSG 1652)
    log5 = MessageLog()
    mat_low_cdr = MaterialLaw104(id=5, young=210000.0, nu=0.3, rho0=7.8e-09, cdr=-4.0)
    check_mat_law104(mat=mat_low_cdr, log=log5)
    assert any("ANCMSG 1652" in msg or "LESS THAN -3.375" in msg for msg in log5.messages)

    # 6. Resolution method Ires warning (ANCMSG 1731)
    log6 = MessageLog()
    mat_bad_ires = MaterialLaw104(id=6, young=210000.0, nu=0.3, rho0=7.8e-09, ires=3)
    check_mat_law104(mat=mat_bad_ires, log=log6)
    assert any("ANCMSG 1731" in msg or "INVALID RESOLUTION METHOD" in msg for msg in log6.messages)

    # 7. Self-heating rate error (ANCMSG 1655)
    log7 = MessageLog()
    mat_bad_heating = MaterialLaw104(id=7, young=210000.0, nu=0.3, rho0=7.8e-09, eps_iso=100.0, eps_ad=10.0)
    check_mat_law104(mat=mat_bad_heating, log=log7)
    assert any("ANCMSG 1655" in msg or "EXCEEDS ADIABATIC RATE" in msg for msg in log7.messages)


def test_starter_checks_element_compatibility():
    """Test that 1D elements are rejected (ANCMSG 306) while 2D shells and 3D solids are accepted."""
    log = MessageLog()
    model = Model()

    # Material LAW104
    mat1 = MaterialLaw104(id=10, young=210000.0, nu=0.3, rho0=7.8e-09)
    model.mat_law104s[10] = mat1

    # Part 1 assigned to 2D shell
    p1 = Part(id=1, mat_id=10, prop_id=1)
    model.parts[1] = p1
    class DummyShell:
        part_id = 1
    model.shells = {101: DummyShell()}

    # Check: Shell elements should NOT produce errors
    check_mat_law104(model=model, mat_id=10, mat=mat1, log=log)
    shell_errors = [m for m in log.messages if "305" in m or "shell elements" in m]
    assert len(shell_errors) == 0

    # Part 2 assigned to 1D truss
    log_1d = MessageLog()
    p2 = Part(id=2, mat_id=10, prop_id=2)
    model.parts[2] = p2
    class DummyTruss:
        part_id = 2
    model.trusses = {201: DummyTruss()}

    check_mat_law104(model=model, mat_id=10, mat=mat1, log=log_1d)
    oned_errors = [m for m in log_1d.messages if "ANCMSG 306" in m or "1D elements" in m]
    assert len(oned_errors) > 0
