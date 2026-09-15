"""Tests for Milestone M225:
- Griffith Brittle Fracture Criterion: /FAIL/GRIFFITH, /FAIL/GRIF
- Engine Contact Surface Directive: /SURF, /ENG/SURF
- Homokinetic and Cylinder Joint Aliases: /LAGMUL/HOMOKINETIC, /LAGMUL/CYLINDER
- Spring Force/Moment Sensor: /SENSOR/SPRING, /SENSOR/SPRING_FORCE
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


def test_m225_fail_griffith(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Griffith Brittle Fracture Test
1 1
/FAIL/GRIFFITH/88
Griffith Failure Criterion
                45.0               250.0                60.0         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 88 in model.fail_griffiths
    fg = model.fail_griffiths[88]
    assert pytest.approx(fg.sigma_0) == 45.0
    assert pytest.approx(fg.sigma_c) == 250.0
    assert pytest.approx(fg.tau_max) == 60.0
    assert fg.ifail_sh == 1


def test_m225_eng_surf(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Contact Surface Force Tracking Test
1 1
/ENG/SURF/1
Surface Tracking Directive
               0.002         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_surfs
    es = model.eng_surfs[1]
    assert pytest.approx(es.dt_surf) == 0.002
    assert es.sens_id == 5


def test_m225_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Homokinetic and Cylinder Joint Aliases Test
1 1
/LAGMUL/HOMOKINETIC/12
Homokinetic Joint Constraint
        10        20         1         2
/LAGMUL/CYLINDER/14
Cylinder Joint Constraint
        15        25         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 12 in model.cv_joints
    assert model.cv_joints[12].node1 == 10
    assert model.cv_joints[12].node2 == 20
    assert 14 in model.cyl_joints
    assert model.cyl_joints[14].node1 == 15
    assert model.cyl_joints[14].node2 == 25


def test_m225_sensor_spring(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Force Sensor Test
1 1
/SENSOR/SPRING/99
Spring Force Threshold Sensor
        42              1500.0               500.0               0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 99 in model.sensor_springs
    assert any(s.id == 99 and s.kind == "SPRING" for s in model.sensors)
    ss = model.sensor_springs[99]
    assert ss.spring_id == 42
    assert pytest.approx(ss.f_max) == 1500.0
    assert pytest.approx(ss.m_max) == 500.0
    assert pytest.approx(ss.t_delay) == 0.001
