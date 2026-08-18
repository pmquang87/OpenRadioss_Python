"""Tests for Milestone M247:
- Ghosh Failure Model: /FAIL/GHOSH, /FAIL/GHOSH_LAW, /FAIL/GHOSH_DAMAGE, /FAIL/GHOSH_MODEL
- Engine Numerical Dissipation Output Directive: /NUMERICAL_DISSIPATION, /ENG/NUMERICAL_DISSIPATION, /ENG/NUM_DISS, /ENG/NUM_ENERGY, /ENG/NUMERICAL_ENERGY, /ENG/DISSIPATION
- Pin Axis Joint Aliases: /LAGMUL/PIN_AXIS, /PIN_AXIS, /LAGMUL/REVOLUTE_PIN_AXIS, /REVOLUTE_PIN_AXIS, /PIN_JOINT_AXIS
- Spring Numerical Dissipation Sensor: /SENSOR/SPRING_NUMERICAL_DISSIPATION, /SENSOR/SPRING_NUM_DISS, /SENSOR/NUM_DISS_SPRING, /SENSOR/SPRING_DISSIPATION
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


def test_m247_fail_ghosh(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Ghosh Failure Model Test
1 1
/FAIL/GHOSH/99
Ghosh Power-Law Hardening Failure Model
               320.0               540.0               0.005                0.22                0.40         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 99 in model.fail_ghoshs
    fg = model.fail_ghoshs[99]
    assert pytest.approx(fg.sigma0) == 320.0
    assert pytest.approx(fg.k_coeff) == 540.0
    assert pytest.approx(fg.eps0) == 0.005
    assert pytest.approx(fg.n_exp) == 0.22
    assert pytest.approx(fg.eps_max) == 0.40
    assert fg.ifail_sh == 1


def test_m247_eng_numerical_dissipation(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Numerical Dissipation Output Directive Test
1 1
/ENG/NUMERICAL_DISSIPATION/1
Numerical Dissipation Energy History Output Control
              0.0001         8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_numerical_dissipations
    end_obj = model.eng_numerical_dissipations[1]
    assert pytest.approx(end_obj.dt_num) == 0.0001
    assert end_obj.sens_id == 8


def test_m247_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Pin Axis Joint Aliases Test
1 1
/LAGMUL/PIN_AXIS/401
Pin Axis Joint Constraint
        12        22         3         0              0.0001
/REVOLUTE_PIN_AXIS/402
Revolute Pin Axis Joint Constraint
        32        42         1         0              0.0002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 401 in model.pin_joints
    assert model.pin_joints[401].node1 == 12
    assert model.pin_joints[401].node2 == 22
    assert model.pin_joints[401].axis_dir == 3
    assert pytest.approx(model.pin_joints[401].tol) == 0.0001

    assert 402 in model.pin_joints
    assert model.pin_joints[402].node1 == 32
    assert model.pin_joints[402].node2 == 42
    assert model.pin_joints[402].axis_dir == 1
    assert pytest.approx(model.pin_joints[402].tol) == 0.0002


def test_m247_sensor_spring_numerical_dissipation(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Numerical Dissipation Sensor Test
1 1
/SENSOR/SPRING_NUMERICAL_DISSIPATION/199
Spring Element Numerical Dissipation Energy Threshold Sensor
        890                45.0               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 199 in model.sensor_spring_numerical_dissipations
    assert any(s.id == 199 and s.kind == "SPRING_NUMERICAL_DISSIPATION" for s in model.sensors)
    ssnd = model.sensor_spring_numerical_dissipations[199]
    assert ssnd.spring_id == 890
    assert pytest.approx(ssnd.enum_max) == 45.0
    assert pytest.approx(ssnd.t_delay) == 0.002
