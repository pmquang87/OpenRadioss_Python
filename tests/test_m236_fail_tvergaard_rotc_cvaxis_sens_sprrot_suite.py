"""Tests for Milestone M236:
- Tvergaard-Needleman Void Shear Coalescence Failure Model: /FAIL/TVERGAARD, /FAIL/TVERGAARD_NEEDLEMAN
- Engine Rotational Displacement Output Directive: /ROTC, /ENG/ROTC
- CV/Homokinetic Axis Joint Aliases: /LAGMUL/CV_AXIS, /LAGMUL/HOMOKINETIC_AXIS
- Spring Rotation Sensor: /SENSOR/SPRING_ROT, /SENSOR/ROT_SPRING
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


def test_m236_fail_tvergaard(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Tvergaard-Needleman Failure Test
1 1
/FAIL/TVERGAARD/96
Tvergaard-Needleman Void Coalescence Failure Model
                 1.6                 1.1                2.56
                 0.4                0.14                0.28         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 96 in model.fail_tvergaards
    ft = model.fail_tvergaards[96]
    assert pytest.approx(ft.q1) == 1.6
    assert pytest.approx(ft.q2) == 1.1
    assert pytest.approx(ft.q3) == 2.56
    assert pytest.approx(ft.kw) == 0.4
    assert pytest.approx(ft.f_c) == 0.14
    assert pytest.approx(ft.f_f) == 0.28
    assert ft.ifail_sh == 2


def test_m236_eng_rotc(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Rotational Displacement Output Directive Test
1 1
/ENG/ROTC/1
Rotational Displacement Vector History Output Control
               0.002         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_rotcs
    er = model.eng_rotcs[1]
    assert pytest.approx(er.dt_rotc) == 0.002
    assert er.sens_id == 5


def test_m236_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
CV Axis and Homokinetic Axis Joint Aliases Test
1 1
/LAGMUL/CV_AXIS/89
Constant Velocity Joint Constraint
        16        26         2         4             1.0e-05
/LAGMUL/HOMOKINETIC_AXIS/90
Homokinetic Axis Joint Constraint
        36        46         3         5             2.0e-05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 89 in model.cv_joints
    assert model.cv_joints[89].node1 == 16
    assert model.cv_joints[89].node2 == 26
    assert model.cv_joints[89].axis_dir == 2
    assert model.cv_joints[89].skew_id == 4
    assert pytest.approx(model.cv_joints[89].tol) == 1.0e-5

    assert 90 in model.cv_joints
    assert model.cv_joints[90].node1 == 36
    assert model.cv_joints[90].node2 == 46
    assert model.cv_joints[90].axis_dir == 3
    assert model.cv_joints[90].skew_id == 5
    assert pytest.approx(model.cv_joints[90].tol) == 2.0e-5


def test_m236_sensor_spring_rot(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Rotation Sensor Test
1 1
/SENSOR/SPRING_ROT/136
Spring Twist Angle Threshold Sensor
       770                0.785              0.004
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 136 in model.sensor_spring_rots
    assert any(s.id == 136 and s.kind == "SPRING_ROT" for s in model.sensors)
    ssr = model.sensor_spring_rots[136]
    assert ssr.spring_id == 770
    assert pytest.approx(ssr.rot_max) == 0.785
    assert pytest.approx(ssr.t_delay) == 0.004
