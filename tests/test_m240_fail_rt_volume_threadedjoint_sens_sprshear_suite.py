"""Tests for Milestone M240:
- Rice-Tracey Failure Model: /FAIL/RICE_TRACEY, /FAIL/RT
- Engine Volume Output Directive: /VOLUME, /ENG/VOLUME
- Threaded Joint Aliases: /LAGMUL/THREADED_JOINT, /LAGMUL/THREADED_AXIS
- Spring Shear Force Sensor: /SENSOR/SPRING_SHEAR, /SENSOR/SHEAR_SPRING
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


def test_m240_fail_rice_tracey(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Rice-Tracey Failure Test
1 1
/FAIL/RICE_TRACEY/77
Rice-Tracey Void Growth Failure Model
                0.01                 1.8               0.283         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 77 in model.fail_rice_traceys
    frt = model.fail_rice_traceys[77]
    assert pytest.approx(frt.r0) == 0.01
    assert pytest.approx(frt.rc_r0) == 1.8
    assert pytest.approx(frt.alpha_rt) == 0.283
    assert frt.ifail_sh == 2


def test_m240_eng_volume(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Volume Output Directive Test
1 1
/ENG/VOLUME/1
Volume History Output Control
               0.005         6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_volumes
    ev = model.eng_volumes[1]
    assert pytest.approx(ev.dt_vol) == 0.005
    assert ev.sens_id == 6


def test_m240_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Threaded Joint and Threaded Axis Aliases Test
1 1
/LAGMUL/THREADED_JOINT/81
Threaded Joint Constraint
        11        21         1         0             1.5e-03             2.0e-05
/LAGMUL/THREADED_AXIS/82
Threaded Axis Joint Constraint
        31        41         2         0             2.5e-03             3.0e-05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 81 in model.screw_joints
    assert model.screw_joints[81].node1 == 11
    assert model.screw_joints[81].node2 == 21
    assert pytest.approx(model.screw_joints[81].pitch) == 1.5e-3
    assert pytest.approx(model.screw_joints[81].tol) == 2.0e-5

    assert 82 in model.screw_joints
    assert model.screw_joints[82].node1 == 31
    assert model.screw_joints[82].node2 == 41
    assert pytest.approx(model.screw_joints[82].pitch) == 2.5e-3
    assert pytest.approx(model.screw_joints[82].tol) == 3.0e-5


def test_m240_sensor_spring_shear(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Shear Force Sensor Test
1 1
/SENSOR/SPRING_SHEAR/140
Spring Shear Force Threshold Sensor
       892              1800.0               0.004
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 140 in model.sensor_spring_shears
    assert any(s.id == 140 and s.kind == "SPRING_SHEAR" for s in model.sensors)
    sss = model.sensor_spring_shears[140]
    assert sss.spring_id == 892
    assert pytest.approx(sss.fsh_max) == 1800.0
    assert pytest.approx(sss.t_delay) == 0.004
