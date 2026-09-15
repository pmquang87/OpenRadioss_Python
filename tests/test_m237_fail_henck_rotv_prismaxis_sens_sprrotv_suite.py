"""Tests for Milestone M237:
- Hencky Logarithmic Principal Strain Failure Model: /FAIL/HENCKY, /FAIL/LOG_STRAIN
- Engine Rotational Velocity Output Directive: /ROTV, /ENG/ROTV
- Prismatic/Translational Axis Joint Aliases: /LAGMUL/PRISMATIC_AXIS, /LAGMUL/TRANSLATIONAL_AXIS
- Spring Rotational Velocity Sensor: /SENSOR/SPRING_ROTV, /SENSOR/ROTV_SPRING
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


def test_m237_fail_hencky(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Hencky Principal Strain Failure Test
1 1
/FAIL/HENCKY/97
Hencky Logarithmic Strain Failure Model
                 0.3                 0.4                 0.5                0.35         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 97 in model.fail_henckys
    fh = model.fail_henckys[97]
    assert pytest.approx(fh.eps_1_max) == 0.3
    assert pytest.approx(fh.eps_2_max) == 0.4
    assert pytest.approx(fh.eps_3_max) == 0.5
    assert pytest.approx(fh.eps_eff_max) == 0.35
    assert fh.ifail_sh == 2


def test_m237_eng_rotv(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Rotational Velocity Output Directive Test
1 1
/ENG/ROTV/1
Rotational Velocity Vector History Output Control
               0.001         6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_rotvs
    er = model.eng_rotvs[1]
    assert pytest.approx(er.dt_rotv) == 0.001
    assert er.sens_id == 6


def test_m237_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Prismatic Axis and Translational Axis Joint Aliases Test
1 1
/LAGMUL/PRISMATIC_AXIS/91
Prismatic Axis Joint Constraint
        17        27         1         3             1.0e-05
/LAGMUL/TRANSLATIONAL_AXIS/92
Translational Axis Joint Constraint
        37        47         2         4             2.0e-05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 91 in model.slider_joints
    assert model.slider_joints[91].node1 == 17
    assert model.slider_joints[91].node2 == 27
    assert model.slider_joints[91].axis_dir == 1
    assert model.slider_joints[91].skew_id == 3
    assert pytest.approx(model.slider_joints[91].tol) == 1.0e-5

    assert 92 in model.slider_joints
    assert model.slider_joints[92].node1 == 37
    assert model.slider_joints[92].node2 == 47
    assert model.slider_joints[92].axis_dir == 2
    assert model.slider_joints[92].skew_id == 4
    assert pytest.approx(model.slider_joints[92].tol) == 2.0e-5


def test_m237_sensor_spring_rotv(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Rotational Velocity Sensor Test
1 1
/SENSOR/SPRING_ROTV/137
Spring Rotational Speed Threshold Sensor
       780                15.7               0.005
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 137 in model.sensor_spring_rotvs
    assert any(s.id == 137 and s.kind == "SPRING_ROTV" for s in model.sensors)
    ssrv = model.sensor_spring_rotvs[137]
    assert ssrv.spring_id == 780
    assert pytest.approx(ssrv.rotv_max) == 15.7
    assert pytest.approx(ssrv.t_delay) == 0.005
