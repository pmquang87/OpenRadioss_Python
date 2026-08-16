"""Tests for Milestone M145: 2D Bivariate Function Tables (/FUNC_2D, /FUNC2D), Non-Local Damage Regularization Models (/NONLOCAL),
Anisotropic Friction Orientations (/FRIC_ORIENT), SPH Cell Initial States (/INISPHCEL), and Implicit Mode Flag / TH Requests (/IMPLICIT, /THPART).
"""
from __future__ import annotations

from pathlib import Path
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str):
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text, encoding="ascii")
    blocks = read_deck(str(p))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m145_func_2d(tmp_path):
    deck = """/BEGIN
Test 2D Function Tables
/FUNC_2D/1
Surface Map 1
1 2
0.0 0.0 10.0
1.0 1.0 20.0
/FUNC2D/2
Surface Map 2
1 2
0.5 0.5 15.0
1.5 1.5 25.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.func2d_tables
    f1 = model.func2d_tables[1]
    assert f1.dim == 1
    assert len(f1.x_vals) == 2
    assert f1.x_vals[0] == pytest.approx(0.0)
    assert f1.y_vals[0] == pytest.approx(0.0)
    assert f1.z_vals[0] == pytest.approx(10.0)
    assert f1.x_vals[1] == pytest.approx(1.0)
    assert f1.y_vals[1] == pytest.approx(1.0)
    assert f1.z_vals[1] == pytest.approx(20.0)

    assert 2 in model.func2d_tables
    f2 = model.func2d_tables[2]
    assert f2.x_vals[0] == pytest.approx(0.5)
    assert f2.z_vals[1] == pytest.approx(25.0)


def test_m145_nonlocal(tmp_path):
    deck = """/BEGIN
Test Nonlocal Material Regularization
/NONLOCAL/10
Regularization Model
1.5 0.5 7.8e-9 0.05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 10 in model.nonlocal_models
    nl = model.nonlocal_models[10]
    assert nl.length == pytest.approx(1.5)
    assert nl.le_max == pytest.approx(0.5)
    assert nl.dens == pytest.approx(7.8e-9)
    assert nl.damp == pytest.approx(0.05)


def test_m145_fric_orient(tmp_path):
    deck = """/BEGIN
Test Friction Orientation
/FRIC_ORIENT/1
Anisotropic Friction Direction
5 2 45.0 1.0 0.0 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.fric_orients
    fo = model.fric_orients[1]
    assert fo.grpart_id == 5
    assert fo.skew_id == 2
    assert fo.phi == pytest.approx(45.0)
    assert fo.vx == pytest.approx(1.0)
    assert fo.vy == pytest.approx(0.0)
    assert fo.vz == pytest.approx(0.0)


def test_m145_inisphcel(tmp_path):
    deck = """/BEGIN
Test SPH Cell Initial State
/INISPHCEL/3
Cell Initial State
1.0e5 1000.0 2.5e5 10.0 0.0 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 3 in model.ini_sphcels
    ic = model.ini_sphcels[3]
    assert ic.p == pytest.approx(1.0e5)
    assert ic.rho == pytest.approx(1000.0)
    assert ic.e == pytest.approx(2.5e5)
    assert ic.vx == pytest.approx(10.0)
    assert ic.vy == pytest.approx(0.0)
    assert ic.vz == pytest.approx(0.0)


def test_m145_implicit_and_thpart(tmp_path):
    deck = """/BEGIN
Test Implicit Mode and THPART
/IMPLICIT
/THPART/1
Part Time History
DEF
1 2 3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert model.implicit_flag is True
    assert len(model.th_requests) == 1
    assert model.th_requests[0].kind == "PART"
    assert model.th_requests[0].ids == [1, 2, 3]
