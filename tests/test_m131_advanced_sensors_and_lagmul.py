"""Tests for Milestone M131: Advanced Sensor Suite & Lagrange Multiplier Constraints
(/SENSOR/ACCE, /SENSOR/SENS, /SENSOR/PYTHON, /LAGMUL, /RBODY/LAGMUL, /GEAR, /RACK, /DIFF).
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


def test_m131_sensor_acce(tmp_path):
    deck = """/BEGIN
Test SENSOR ACCE
/SENSOR/ACCE/1
Accelerometer Sensor 1
0.01 2
101 X 50.0 0.005
102 Z 100.0 0.002
/SENSOR/TYPE1/2
Accelerometer Type 1
0.00 1
201 Y 25.0 0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.sensors) == 2

    s1 = next(s for s in model.sensors if s.id == 1)
    assert s1.kind == "ACCE"
    assert s1.tdelay == pytest.approx(0.01)
    assert len(s1.acc_entries) == 2
    assert s1.acc_entries[0] == (101, "X", 50.0, 0.005)
    assert s1.acc_entries[1] == (102, "Z", 100.0, 0.002)

    s2 = next(s for s in model.sensors if s.id == 2)
    assert s2.kind == "ACCE"
    assert s2.tdelay == pytest.approx(0.0)
    assert len(s2.acc_entries) == 1
    assert s2.acc_entries[0] == (201, "Y", 25.0, 0.001)


def test_m131_sensor_sens(tmp_path):
    deck = """/BEGIN
Test SENSOR SENS
/SENSOR/SENS/1
State Switch Sensor
0.02
10 20
/SENSOR/TYPE3/2
Type3 Sensor
0.05 30 40
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.sensors) == 2

    s1 = next(s for s in model.sensors if s.id == 1)
    assert s1.kind == "SENS"
    assert s1.tdelay == pytest.approx(0.02)
    assert s1.sens_id1 == 10
    assert s1.sens_id2 == 20

    s2 = next(s for s in model.sensors if s.id == 2)
    assert s2.kind == "SENS"
    assert s2.tdelay == pytest.approx(0.05)
    assert s2.sens_id1 == 30
    assert s2.sens_id2 == 40


def test_m131_sensor_python(tmp_path):
    deck = """/BEGIN
Test SENSOR PYTHON
/SENSOR/PYTHON/1
Custom Python Callback
0.005
user_sensor evaluate_trigger
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.sensors) == 1

    s = model.sensors[0]
    assert s.kind == "PYTHON"
    assert s.tdelay == pytest.approx(0.005)
    assert s.script_name == "user_sensor"
    assert s.func_name == "evaluate_trigger"


def test_m131_lagmul_global(tmp_path):
    deck = """/BEGIN
Test LAGMUL Global
/LAGMUL
Global Options
2 3 1.0e-9 1.0e-3 1.0e-4
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert model.lagmul_global is not None
    assert model.lagmul_global.lagmod == 2
    assert model.lagmul_global.lagopt == 3
    assert model.lagmul_global.tol == pytest.approx(1.0e-9)
    assert model.lagmul_global.alpha == pytest.approx(1.0e-3)
    assert model.lagmul_global.alpha_s == pytest.approx(1.0e-4)


def test_m131_rbody_lagmul(tmp_path):
    deck = """/BEGIN
Test RBODY LAGMUL
/RBODY/LAGMUL/1
Lagmul Rigid Body
100 200 50.0 1
10.0 10.0 10.0
/RBODY/2
Standard Rigid Body
101 201 25.0 1
5.0 5.0 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"
    assert len(model.rbodies) == 2

    rb1 = next(rb for rb in model.rbodies if rb.id == 1)
    assert rb1.master_id == 100
    assert rb1.grnod_id == 200
    assert rb1.lagmul is True

    rb2 = next(rb for rb in model.rbodies if rb.id == 2)
    assert rb2.master_id == 101
    assert rb2.grnod_id == 201
    assert rb2.lagmul is False


def test_m131_gear_rack_diff(tmp_path):
    deck = """/BEGIN
Test Kinematic Constraints
/GEAR/1
Gear Transmission
10 20 2.5 1 2 0 0
/LAGMUL/RACK/2
Rack and Pinion Steering
30 40 0.15 1 3 0 0
/LAGMUL/DIFF/3
Differential Unit
50 60 70 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert 1 in model.gears
    gear = model.gears[1]
    assert gear.node1 == 10
    assert gear.node2 == 20
    assert gear.ratio == pytest.approx(2.5)
    assert gear.dir1 == 1
    assert gear.dir2 == 2

    assert 2 in model.racks
    rack = model.racks[2]
    assert rack.node1 == 30
    assert rack.node2 == 40
    assert rack.pitch_radius == pytest.approx(0.15)
    assert rack.dir1 == 1
    assert rack.dir2 == 3

    assert 3 in model.diffs
    diff = model.diffs[3]
    assert diff.node0 == 50
    assert diff.node1 == 60
    assert diff.node2 == 70
    assert diff.ratio == pytest.approx(1.0)
