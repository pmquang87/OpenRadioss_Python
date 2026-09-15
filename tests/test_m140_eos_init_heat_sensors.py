"""Tests for Milestone M140: Extended High-Explosive Equations of State, Unified Initial State Dispatcher & Sensor Suite
(/EOS/JWL, /EOS/TYPE5, /EOS/COMPACT, /EOS/SESAME, /EOS/IGNITION_GROWTH, /INIT/VEL, /INIT/BRI, /INIT/SHE, /INIT/TRU, /INIT/BEA, /INIT/SPR, /INIT/TEMP, /HEAT/MAT, /HEAT/SOLVER, /SENSOR/RWALL_PLANE, /SENSOR/FORCE, /SENSOR/MOMENT, /SENSOR/RWALL_CYL).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
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


def test_m140_eos_jwl_compact_sesame_ig(tmp_path):
    deck = """/BEGIN
Test Extended EOS
/MAT/LAW1/1
Explosive Material
1.63e-6
200.0 0.3
/EOS/JWL/1
JWL High Explosive
371.2 3.23 4.15 0.95 0.30
7.0 0.0 1.63e-6
/MAT/LAW1/2
Porous Material
2.0e-6
100.0 0.3
/EOS/COMPACT/2
Porous Compaction EOS
10.0 20.0 30.0 40.0 50.0
/MAT/LAW1/3
Sesame Material
2.7e-6
70.0 0.33
/EOS/SESAME/3
Sesame Table
1001 1.0 1.0
/MAT/LAW1/4
Ignition Growth
1.8e-6
150.0 0.25
/EOS/IGNITION_GROWTH/4
Ignition Growth EOS
1.0 2.0 3.0 4.0 5.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert len(model.raw_eos) == 4
    mat_id1, eos1, _ = model.raw_eos[0]
    assert mat_id1 == 1
    assert eos1.kind == "JWL"
    assert eos1.params["a"] == pytest.approx(371.2)
    assert eos1.params["b"] == pytest.approx(3.23)
    assert eos1.params["r1"] == pytest.approx(4.15)
    assert eos1.params["r2"] == pytest.approx(0.95)
    assert eos1.params["omega"] == pytest.approx(0.30)
    assert eos1.params["e0"] == pytest.approx(7.0)

    mat_id2, eos2, _ = model.raw_eos[1]
    assert mat_id2 == 2
    assert eos2.kind == "COMPACT"
    assert np.allclose(eos2.params["values"], [10.0, 20.0, 30.0, 40.0, 50.0])

    mat_id3, eos3, _ = model.raw_eos[2]
    assert mat_id3 == 3
    assert eos3.kind == "SESAME"

    mat_id4, eos4, _ = model.raw_eos[3]
    assert mat_id4 == 4
    assert eos4.kind == "IGNITION_GROWTH"


def test_m140_init_unified_dispatcher(tmp_path):
    deck = """/BEGIN
Test Unified /INIT Dispatcher
/NODE/1
1 0.0 0.0 0.0
/NODE/2
2 1.0 0.0 0.0
/NODE/3
3 1.0 1.0 0.0
/NODE/4
4 0.0 1.0 0.0
/NODE/5
5 0.0 0.0 1.0
/NODE/6
6 1.0 0.0 1.0
/NODE/7
7 1.0 1.0 1.0
/NODE/8
8 0.0 1.0 1.0
/BRICK/1/1
1 1 2 3 4 5 6 7 8
/SHELL/1/1
1 1 2 3 4
/INIT/VEL/TRA/1
Velocity translation
10.0 0.0 0.0
/INIT/BRI/TEMP/1
1 350.0
/INIT/SHE/ENER/1
1 125.0
/INIT/TEMP/1
1 300.0
/INIT/TRU/TEMP/1
1 250.0
/INIT/BEA/TEMP/1
1 260.0
/INIT/SPR/TEMP/1
1 270.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert len(model.inivel) == 1
    assert np.allclose(model.inivel[0].v, [10.0, 0.0, 0.0])

    assert 1 in model.ini_bricks
    assert model.ini_bricks[1].temp == pytest.approx(300.0)

    assert 1 in model.ini_shells
    assert model.ini_shells[1].em == pytest.approx(125.0)

    assert 1 in model.ini_trusses
    assert model.ini_trusses[1].temp == pytest.approx(250.0)

    assert 1 in model.ini_beams
    assert model.ini_beams[1].temp == pytest.approx(260.0)

    assert 1 in model.ini_springs
    assert model.ini_springs[1].temp == pytest.approx(270.0)


def test_m140_heat_and_sensors(tmp_path):
    deck = """/BEGIN
Test Heat and Sensors
/HEAT/MAT/1
Thermal Material modifier
100.0 200.0
/HEAT/SOLVER/1
Thermal Solver Config
1.0 1e-4
/SENSOR/RWALL_PLANE/1
Rigid Wall Plane Sensor
10 500.0 0.0 0.01
/SENSOR/FORCE/2
Force Threshold Sensor
20 1500.0 0.0 0.005
/SENSOR/MOMENT/3
Moment Threshold Sensor
25 3000.0 0.0 0.002
/SENSOR/RWALL_CYL/4
Rigid Wall Cylinder Sensor
40 750.0 0.0 0.02
/SENSOR/BOX/5
Box Sensor
30 1.0 2.0 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0, f"Errors: {log.errors}"

    assert len(model.sensors) == 5
    s1 = model.sensors[0]
    assert s1.id == 1
    assert s1.kind == "RWALL_PLANE"
    assert s1.target_id == 10
    assert s1.dmin == pytest.approx(500.0)

    s2 = model.sensors[1]
    assert s2.id == 2
    assert s2.kind == "FORCE"
    assert s2.target_id == 20
    assert s2.dmin == pytest.approx(1500.0)

    s3 = model.sensors[2]
    assert s3.id == 3
    assert s3.kind == "MOMENT"
    assert s3.target_id == 25
    assert s3.dmin == pytest.approx(3000.0)

    s4 = model.sensors[3]
    assert s4.id == 4
    assert s4.kind == "RWALL_CYL"
    assert s4.target_id == 40
    assert s4.dmin == pytest.approx(750.0)

    s5 = model.sensors[4]
    assert s5.id == 5
    assert s5.kind == "BOX"
    assert s5.target_id == 30
