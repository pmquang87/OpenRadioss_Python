"""Tests for Milestone M218:
- Cohesive Failure Criterion: /FAIL/COHESIVE, /FAIL/COH
- Engine Trajectory Tracking Output Directive: /TRACK, /ENG/TRACK
- Spherical Kinematic Joint Aliases: /LAGMUL/SPHERICAL, /SPHERICAL_JOINT, /SPHERICAL
- Cross-Section Force/Moment Sensor Trigger: /SENSOR/CROSSSECTION, /SENSOR/SEC_FORCE
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


def test_m218_fail_cohesive(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Cohesive Failure Test
1 1
/FAIL/COHESIVE/45
Interface Delamination Failure
               0.520                1.45                25.0                40.0                1.85
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 45 in model.fail_cohesives
    fc = model.fail_cohesives[45]
    assert pytest.approx(fc.g1c) == 0.520
    assert pytest.approx(fc.g2c) == 1.45
    assert pytest.approx(fc.t1) == 25.0
    assert pytest.approx(fc.t2) == 40.0
    assert pytest.approx(fc.alpha) == 1.85


def test_m218_eng_track(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Track Test
1 1
/ENG/TRACK/1
Nodal Trajectory Tracking
       101         2               0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_tracks
    tr = model.eng_tracks[1]
    assert tr.node_id == 101
    assert tr.skew_id == 2
    assert pytest.approx(tr.dt_track) == 0.001


def test_m218_spherical_joint_alias(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spherical Joint Alias Test
1 1
/LAGMUL/SPHERICAL/7
Spherical Kinematic Constraint 7
         1         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 7 in model.ball_joints
    bj = model.ball_joints[7]
    assert bj.node1 == 1
    assert bj.node2 == 2


def test_m218_sensor_crosssection(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Cross-Section Sensor Test
1 1
/SENSOR/CROSSSECTION/12
Cross-Section Resultant Limit Sensor
         4              5000.0             12000.0               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 12 in model.sensor_cross_sections
    assert any(s.id == 12 and s.kind == "CROSSSECTION" for s in model.sensors)
    scs = model.sensor_cross_sections[12]
    assert scs.sec_id == 4
    assert pytest.approx(scs.f_cut) == 5000.0
    assert pytest.approx(scs.m_cut) == 12000.0
    assert pytest.approx(scs.t_delay) == 0.002
