"""Tests for Milestone M239:
- Energy Ratio Failure Model: /FAIL/ENERGY_RATIO, /FAIL/ERATIO
- Engine Force Output Directive: /FORCE, /ENG/FORCE
- Ball Socket Joint Aliases: /LAGMUL/BALL_SOCKET, /LAGMUL/BALL_AND_SOCKET
- Spring Axial Force Sensor: /SENSOR/SPRING_AXIAL, /SENSOR/AXIAL_SPRING
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


def test_m239_fail_energy_ratio(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Energy Ratio Failure Test
1 1
/FAIL/ENERGY_RATIO/99
Energy Ratio Failure Model
                  1.25                50.0         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 99 in model.fail_energy_ratios
    fer = model.fail_energy_ratios[99]
    assert pytest.approx(fer.eratio_max) == 1.25
    assert pytest.approx(fer.eint_min) == 50.0
    assert fer.ifail_sh == 2


def test_m239_eng_force(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Force Output Directive Test
1 1
/ENG/FORCE/1
Force Vector History Output Control
               0.001         8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_forces
    ef = model.eng_forces[1]
    assert pytest.approx(ef.dt_force) == 0.001
    assert ef.sens_id == 8


def test_m239_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Ball Socket and Ball-and-Socket Joint Aliases Test
1 1
/LAGMUL/BALL_SOCKET/95
Ball Socket Joint Constraint
        19        29             1.5e-05
/LAGMUL/BALL_AND_SOCKET/96
Ball and Socket Joint Constraint
        39        49             2.5e-05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 95 in model.ball_joints
    assert model.ball_joints[95].node1 == 19
    assert model.ball_joints[95].node2 == 29
    assert pytest.approx(model.ball_joints[95].tol) == 1.5e-5

    assert 96 in model.ball_joints
    assert model.ball_joints[96].node1 == 39
    assert model.ball_joints[96].node2 == 49
    assert pytest.approx(model.ball_joints[96].tol) == 2.5e-5


def test_m239_sensor_spring_axial(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Axial Force Sensor Test
1 1
/SENSOR/SPRING_AXIAL/139
Spring Axial Force Threshold Sensor
       791              2500.0               0.007
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 139 in model.sensor_spring_axials
    assert any(s.id == 139 and s.kind == "SPRING_AXIAL" for s in model.sensors)
    ssfa = model.sensor_spring_axials[139]
    assert ssfa.spring_id == 791
    assert pytest.approx(ssfa.fax_max) == 2500.0
    assert pytest.approx(ssfa.t_delay) == 0.007
