"""Tests for Milestone M223:
- Weibull Statistical Brittle Failure Criterion: /FAIL/WEIBULL, /FAIL/WEIBULL_BRITTLE
- Engine Momentum Tracking Directive: /MOMENT, /ENG/MOMENT
- Revolute Kinematic Joint Aliases: /LAGMUL/REVOLUTE, /REVOLUTE_JOINT
- Total Energy Error Percentage Sensor: /SENSOR/ENERGY_ERROR, /SENSOR/ENG_ERROR
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


def test_m223_fail_weibull(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Weibull Brittle Fracture Model Test
1 1
/FAIL/WEIBULL/44
Weibull Ceramic Failure Criterion
               350.0                 8.5                 1.0         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 44 in model.fail_weibulls
    fw = model.fail_weibulls[44]
    assert pytest.approx(fw.sigma_0) == 350.0
    assert pytest.approx(fw.m_mod) == 8.5
    assert pytest.approx(fw.v_0) == 1.0
    assert fw.ifail_sh == 1


def test_m223_eng_moment(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Momentum Tracking Test
1 1
/ENG/MOMENT/1
Momentum Tracking Directive
                0.02         8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_moments
    em = model.eng_moments[1]
    assert pytest.approx(em.dt_mom) == 0.02
    assert em.sens_id == 8


def test_m223_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Revolute Kinematic Joint Aliases Test
1 1
/LAGMUL/REVOLUTE/22
Revolute Joint Constraint
        15        16         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 22 in model.pin_joints
    assert model.pin_joints[22].node1 == 15
    assert model.pin_joints[22].node2 == 16


def test_m223_sensor_energy_error(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Energy Error Percentage Sensor Test
1 1
/SENSOR/ENERGY_ERROR/66
Energy Error Trigger Sensor
                15.0               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 66 in model.sensor_energy_errors
    assert any(s.id == 66 and s.kind == "ENERGY_ERROR" for s in model.sensors)
    see = model.sensor_energy_errors[66]
    assert pytest.approx(see.err_max) == 15.0
    assert pytest.approx(see.t_delay) == 0.002
