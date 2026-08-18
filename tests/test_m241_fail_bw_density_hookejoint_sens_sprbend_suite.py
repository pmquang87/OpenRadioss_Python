"""Tests for Milestone M241:
- Bao-Wierzbicki Failure Model: /FAIL/BAO_WIERZBICKI, /FAIL/BW
- Engine Density Output Directive: /DENSITY, /ENG/DENSITY
- Hooke Joint Aliases: /LAGMUL/HOOKE_JOINT, /LAGMUL/HOOKE_AXIS
- Spring Bending Moment Sensor: /SENSOR/SPRING_BEND, /SENSOR/BEND_SPRING
"""

from pathlib import Path
import pytest
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model


def _parse_starter(tmp_path: Path, text: str) -> tuple[Model, MessageLog]:
    p = tmp_path / "TEST_0000.rad"
    p.write_text(text.strip() + "\n", encoding="ascii")
    blocks = read_deck(str(p))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_m241_fail_bao_wierzbicki(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Bao-Wierzbicki Failure Test
1 1
/FAIL/BAO_WIERZBICKI/66
Bao-Wierzbicki Fracture Locus Model
                0.35                0.12                0.08               0.333         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 66 in model.fail_bao_wierzbickis
    fbw = model.fail_bao_wierzbickis[66]
    assert pytest.approx(fbw.c1) == 0.35
    assert pytest.approx(fbw.c2) == 0.12
    assert pytest.approx(fbw.c3) == 0.08
    assert pytest.approx(fbw.eta0) == 0.333
    assert fbw.ifail_sh == 2


def test_m241_eng_density(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Density Output Directive Test
1 1
/ENG/DENSITY/1
Density History Output Control
               0.002         7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_densities
    ed = model.eng_densities[1]
    assert pytest.approx(ed.dt_dens) == 0.002
    assert ed.sens_id == 7


def test_m241_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Hooke Joint and Hooke Axis Aliases Test
1 1
/LAGMUL/HOOKE_JOINT/91
Hooke Universal Joint Constraint
        15        25         1         0             1.5e-05
/LAGMUL/HOOKE_AXIS/92
Hooke Axis Joint Constraint
        35        45         2         0             2.5e-05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 91 in model.cardan_joints
    assert model.cardan_joints[91].node1 == 15
    assert model.cardan_joints[91].node2 == 25
    assert model.cardan_joints[91].axis_dir == 1
    assert pytest.approx(model.cardan_joints[91].tol) == 1.5e-5

    assert 92 in model.cardan_joints
    assert model.cardan_joints[92].node1 == 35
    assert model.cardan_joints[92].node2 == 45
    assert model.cardan_joints[92].axis_dir == 2
    assert pytest.approx(model.cardan_joints[92].tol) == 2.5e-5


def test_m241_sensor_spring_bend(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Bending Moment Sensor Test
1 1
/SENSOR/SPRING_BEND/150
Spring Bending Moment Threshold Sensor
       654              2500.0               0.006
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 150 in model.sensor_spring_bends
    assert any(s.id == 150 and s.kind == "SPRING_BEND" for s in model.sensors)
    ssb = model.sensor_spring_bends[150]
    assert ssb.spring_id == 654
    assert pytest.approx(ssb.mbend_max) == 2500.0
    assert pytest.approx(ssb.t_delay) == 0.006
