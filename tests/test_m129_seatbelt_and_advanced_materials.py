"""Tests for Milestone M129: Extended Seatbelt & Advanced Material Laws Suite
(/MAT/LAW114, /MAT/LAW117, /MAT/LAW119, /MAT/LAW120, /MAT/LAW121, /MAT/LAW124, /MAT/LAW90).
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


def test_m129_law114_spring_seatbelt(tmp_path):
    deck = """/BEGIN
Test LAW114 Seatbelt Spring Material
/MAT/LAW114/1
Belt Spring
7.85e-9 10.0
500.0 0.05
1 2 1.0 1000.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 1 in model.materials
    mat = model.materials[1]
    assert mat.law == 114
    assert mat.rho0 == pytest.approx(7.85e-9)
    assert mat.params["stiff1"] == pytest.approx(500.0)
    assert mat.params["damp1"] == pytest.approx(0.05)
    assert mat.params["fun_l"] == 1
    assert mat.params["fun_ul"] == 2
    assert mat.params["fscale"] == pytest.approx(1000.0)


def test_m129_law119_shell_seatbelt(tmp_path):
    deck = """/BEGIN
Test LAW119 Seatbelt Shell Material
/MAT/LAW119/2
Belt Shell
7.85e-9 5.0
1000.0 0.1 0.2
1 2 1.0 1.0 1
2000.0 0.3 800.0 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 2 in model.materials
    mat = model.materials[2]
    assert mat.law == 119
    assert mat.rho0 == pytest.approx(7.85e-9)
    assert mat.params["stiff1"] == pytest.approx(1000.0)
    assert mat.params["re"] == pytest.approx(0.2)
    assert mat.params["ireload"] == 1
    assert mat.params["e22"] == pytest.approx(2000.0)
    assert mat.params["g12"] == pytest.approx(800.0)


def test_m129_law117_cohesive(tmp_path):
    deck = """/BEGIN
Test LAW117 Cohesive Material
/MAT/LAW117/3
Cohesive Law
1.0e-9
10000.0 5000.0 1 1 1
1 2 50.0 25.0 1.0
1.5 2.5 1.0 1.0 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 3 in model.materials
    mat = model.materials[3]
    assert mat.law == 117
    assert mat.rho0 == pytest.approx(1.0e-9)
    assert mat.params["e_elas_n"] == pytest.approx(10000.0)
    assert mat.params["e_elas_s"] == pytest.approx(5000.0)
    assert mat.params["tmax_n"] == pytest.approx(50.0)
    assert mat.params["tmax_s"] == pytest.approx(25.0)
    assert mat.params["gic"] == pytest.approx(1.5)
    assert mat.params["giic"] == pytest.approx(2.5)


def test_m129_law120_tapo(tmp_path):
    deck = """/BEGIN
Test LAW120 Tape Orientation Composite
/MAT/LAW120/4
TAPO Composite
1.5e-9
50000.0 0.3 0 0 0 0.25
10 1.0 1.0
30.0 100.0 5.0 50.0
1.0 0.5 0.2 0.1 0.8
0.01 0.001 0.05
0.1 0.2 0.3 0.4
0.5 0.6 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 4 in model.materials
    mat = model.materials[4]
    assert mat.law == 120
    assert mat.rho0 == pytest.approx(1.5e-9)
    assert mat.params["E"] == pytest.approx(50000.0)
    assert mat.params["nu"] == pytest.approx(0.3)
    assert mat.params["thick"] == pytest.approx(0.25)
    assert mat.params["tab_id"] == 10
    assert mat.params["tau"] == pytest.approx(30.0)
    assert mat.params["q"] == pytest.approx(100.0)


def test_m129_law121_plasrate(tmp_path):
    deck = """/BEGIN
Test LAW121 Strain-Rate Plasticity
/MAT/LAW121/5
Rate Dependent Plas
7.85e-9
210000.0 0.3 1 0 100.0 0.05
1 1.0 1.0
2 1.0 1.0
3 1.0 1500.0
4 1 1.0 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 5 in model.materials
    mat = model.materials[5]
    assert mat.law == 121
    assert mat.rho0 == pytest.approx(7.85e-9)
    assert mat.params["E"] == pytest.approx(210000.0)
    assert mat.params["fct_sig0"] == 1
    assert mat.params["fct_tang"] == 3
    assert mat.params["tang"] == pytest.approx(1500.0)


def test_m129_law124_cdpm2(tmp_path):
    deck = """/BEGIN
Test LAW124 Concrete Damage Plasticity
/MAT/LAW124/6
CDPM2 Concrete
2.4e-9
30000.0 0.2 0 500.0
0.5 0.1 3.5 35.0 1000.0
0.1 0.2 0.3 0.4
0.5 0.6 0.7 1 2 1
0.01 0.02 2.0 0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 6 in model.materials
    mat = model.materials[6]
    assert mat.law == 124
    assert mat.rho0 == pytest.approx(2.4e-9)
    assert mat.params["E"] == pytest.approx(30000.0)
    assert mat.params["ft"] == pytest.approx(3.5)
    assert mat.params["fc"] == pytest.approx(35.0)
    assert mat.params["hp"] == pytest.approx(1000.0)


def test_m129_law90_tabfoam(tmp_path):
    deck = """/BEGIN
Test LAW90 Tabulated Foam
/MAT/LAW90/7
Foam Law 90
5.0e-11
50.0 0.1
0 0 100.0 1.5 0.2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert 7 in model.materials
    mat = model.materials[7]
    assert mat.law == 90
    assert mat.rho0 == pytest.approx(5.0e-11)
    assert mat.params["E"] == pytest.approx(50.0)
    assert mat.params["nu"] == pytest.approx(0.1)
