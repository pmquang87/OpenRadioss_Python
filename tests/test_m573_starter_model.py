"""Tests for /MAT/LAW103 Starter parsing, Model entities, and Diagnostic checks (M573).

Verifies:
- Fixed format parsing of 6 cards.
- Free format parsing with keywords /MAT/LAW103, /MAT/HENSEL_SPITTEL, /MAT/PLAS_HENS.
- Default values (Pmin=-1e30, RhoCp=1e30, Refer_Rho=Rho, eta <= 1.0).
- Model entity properties (E, G, K, bulk, sound_speed, aliases).
- Starter diagnostic checks (ANCMSG 276, 300, 1514, 305, 306).
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import (
    MatLaw103,
    MatHenselSpittel,
    MatPlasHens,
    Part,
    Property,
)
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_mat_law103, check_model


def _parse_deck(tmp_path: Path, text: str) -> Model:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    assert not log.errors, f"Parse errors: {log.errors}"
    return model


def test_fixed_format_parsing(tmp_path: Path):
    """Parse a complete fixed-format 6-card /MAT/LAW103 deck."""
    deck_text = """#RADIOSS STARTER
/BEGIN
Test MAT LAW103 6 Cards
                  10                   0
/MAT/LAW103/10
Hot Forming Steel 42CrMo4
#        Init. dens.          Ref. dens.
              7.8e-9              7.8e-9
#                  E                  Nu
            210000.0                 0.3
#                 A0                  m1                  m2                  m3                  m4
               550.0             -0.0025                0.15               0.045              -0.001
#                 m5                  m7
             -0.0001               0.015
#            Fsmooth                Fcut               EPS_0                Pmin
                   1              1000.0               0.002             -1.0e20
#              RhoCp                  T0                 ETA
              3.8e-3              1073.0                 0.9
"""
    model = _parse_deck(tmp_path, deck_text)

    assert 10 in model.mat_law103s
    mat = model.mat_law103s[10]

    assert mat.id == 10
    assert "Hot Forming Steel" in mat.title
    assert math.isclose(mat.rho, 7.8e-9, rel_tol=1.0e-9)
    assert math.isclose(mat.refer_rho, 7.8e-9, rel_tol=1.0e-9)
    assert math.isclose(mat.e, 210000.0, rel_tol=1.0e-9)
    assert math.isclose(mat.nu, 0.3, rel_tol=1.0e-9)
    assert math.isclose(mat.a0, 550.0, rel_tol=1.0e-9)
    assert math.isclose(mat.m1, -0.0025, rel_tol=1.0e-9)
    assert math.isclose(mat.m2, 0.15, rel_tol=1.0e-9)
    assert math.isclose(mat.m3, 0.045, rel_tol=1.0e-9)
    assert math.isclose(mat.m4, -0.001, rel_tol=1.0e-9)
    assert math.isclose(mat.m5, -0.0001, rel_tol=1.0e-9)
    assert math.isclose(mat.m7, 0.015, rel_tol=1.0e-9)
    assert mat.fsmooth == 1
    assert math.isclose(mat.fcut, 1000.0, rel_tol=1.0e-9)
    assert math.isclose(mat.eps_0, 0.002, rel_tol=1.0e-9)
    assert math.isclose(mat.pmin, -1.0e20, rel_tol=1.0e-9)
    assert math.isclose(mat.rhocp, 3.8e-3, rel_tol=1.0e-9)
    assert math.isclose(mat.t0, 1073.0, rel_tol=1.0e-9)
    assert math.isclose(mat.eta, 0.9, rel_tol=1.0e-9)

    # Check properties
    assert math.isclose(mat.E, 210000.0, rel_tol=1.0e-9)
    g_expected = 210000.0 / (2.0 * 1.3)
    k_expected = 210000.0 / (3.0 * (1.0 - 0.6))
    assert math.isclose(mat.G, g_expected, rel_tol=1.0e-9)
    assert math.isclose(mat.K, k_expected, rel_tol=1.0e-9)
    assert math.isclose(mat.bulk, k_expected, rel_tol=1.0e-9)
    assert mat.sound_speed > 5000.0


def test_free_format_and_aliases(tmp_path: Path):
    """Parse free format decks with /MAT/HENSEL_SPITTEL and /MAT/PLAS_HENS."""
    deck1 = """#RADIOSS STARTER
/BEGIN
Test MAT HENSEL_SPITTEL
                  10                   0
/MAT/HENSEL_SPITTEL/5
Title Hensel Spittel Free Format
7.85e-9 7.85e-9
205000.0 0.29
480.0 -0.0018 0.12 0.035 0.0
-0.00005 0.01
0 0.0 0.001 -1.0e30
3.5e-3 800.0 0.85
"""
    model1 = _parse_deck(tmp_path, deck1)

    assert 5 in model1.mat_law103s
    assert 5 in model1.mat_hensel_spittels
    m1 = model1.mat_law103s[5]
    assert math.isclose(m1.a0, 480.0, rel_tol=1.0e-9)
    assert math.isclose(m1.eta, 0.85, rel_tol=1.0e-9)

    deck2 = """#RADIOSS STARTER
/BEGIN
Test MAT PLAS_HENS
                  10                   0
/MAT/PLAS_HENS/8
Title Plas Hens Free Format
8.0e-9
190000.0 0.31
600.0 -0.003 0.2 0.05 -0.002
0.0 0.02
0 0.0 0.0 0.0
0.0 293.15 1.5
"""
    p2 = tmp_path / "TEST2_0000.rad"
    p2.write_text(deck2.strip() + "\n", encoding="ascii")
    blocks2 = read_deck(str(p2))
    model2 = Model()
    log2 = MessageLog()
    parse_starter_deck(blocks2, model2, log2)
    assert not log2.errors

    assert 8 in model2.mat_law103s
    m2 = model2.mat_law103s[8]
    # Check default pmin: 0.0 -> -1.0e30
    assert m2.pmin == -1.0e30
    # Check default rhocp: 0.0 -> 1.0e30
    assert m2.rhocp == 1.0e30
    # Check default refer_rho: 0.0 -> rho = 8.0e-9
    assert m2.refer_rho == 8.0e-9
    # Check clamped eta: 1.5 -> 1.0
    assert m2.eta == 1.0


def test_checks_valid():
    """Valid parameters produce zero errors in check_mat_law103."""
    model = Model()
    log = MessageLog()
    mat = MatLaw103(
        id=1, rho=7.8e-9, e=210000.0, nu=0.3, a0=500.0
    )
    model.mat_law103s[1] = mat
    check_mat_law103(model=model, mat_id=1, mat=mat, log=log)
    assert not log.has_errors


def test_checks_invalid_bounds():
    """Invalid bounds (rho <= 0, E <= 0, nu <= -1 or nu >= 0.5) trigger errors."""
    # Zero density -> ANCMSG 1514
    model1 = Model()
    log1 = MessageLog()
    mat1 = MatLaw103(id=1, rho=0.0, e=210000.0, nu=0.3, a0=500.0)
    check_mat_law103(model=model1, mat_id=1, mat=mat1, log=log1)
    assert any("ANCMSG 1514" in str(e) and "DENSITY" in str(e) for e in log1.errors)

    # Negative Young's modulus -> ANCMSG 276
    model2 = Model()
    log2 = MessageLog()
    mat2 = MatLaw103(id=2, rho=7.8e-9, e=-1000.0, nu=0.3, a0=500.0)
    check_mat_law103(model=model2, mat_id=2, mat=mat2, log=log2)
    assert any("ANCMSG 276" in str(e) for e in log2.errors)

    # Invalid Poisson's ratio <= -1 -> ANCMSG 300
    model3 = Model()
    log3 = MessageLog()
    mat3 = MatLaw103(id=3, rho=7.8e-9, e=210000.0, nu=-1.5, a0=500.0)
    check_mat_law103(model=model3, mat_id=3, mat=mat3, log=log3)
    assert any("ANCMSG 300" in str(e) for e in log3.errors)

    # Invalid Poisson's ratio >= 0.5 -> ANCMSG 300
    model4 = Model()
    log4 = MessageLog()
    mat4 = MatLaw103(id=4, rho=7.8e-9, e=210000.0, nu=0.5, a0=500.0)
    check_mat_law103(model=model4, mat_id=4, mat=mat4, log=log4)
    assert any("ANCMSG 300" in str(e) for e in log4.errors)


def test_checks_element_compatibility():
    """Reject 2D shells (ANCMSG 305) and 1D elements (ANCMSG 306)."""
    model = Model()
    mat = MatLaw103(id=1, rho=7.8e-9, e=210000.0, nu=0.3, a0=500.0)
    model.mat_law103s[1] = mat
    model.parts[1] = Part(id=1, mat_id=1, prop_id=1, title="P1")

    # Add a shell element referencing this part -> ANCMSG 305
    class DummyShell:
        part_id = 1
    model.shells = {1: DummyShell()}
    log_shell = MessageLog()
    check_mat_law103(model=model, mat_id=1, mat=mat, log=log_shell)
    assert any("305" in str(e) for e in log_shell.errors)

    # Add a 1D element referencing this part -> ANCMSG 306
    model.shells.clear()
    class DummyBeam:
        part_id = 1
    model.beams = {1: DummyBeam()}
    log_1d = MessageLog()
    check_mat_law103(model=model, mat_id=1, mat=mat, log=log_1d)
    assert any("306" in str(e) for e in log_1d.errors)


def test_checks_2d_analysis():
    """Reject 2D plane analysis (N2D > 0) with ANCMSG 305."""
    model = Model()
    model.n2d = 1
    model.mat_law103s[1] = MatLaw103(id=1, rho=7.8e-9, e=210000.0, nu=0.3, a0=500.0)
    log = MessageLog()
    check_model(model, log)
    assert any("ANCMSG 305" in str(e) and "N2D" in str(e) for e in log.errors)
