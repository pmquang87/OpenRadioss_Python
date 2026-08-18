"""Tests for Milestone M232:
- GTN Porous Metal Failure Criterion: /FAIL/GTN, /FAIL/GURSON_TVERGAARD
- Engine Plastic Strain Output Directive: /PLASTIC, /ENG/PLASTIC
- Perpendicular/Normal Axis Joint Aliases: /LAGMUL/PERPENDICULAR_AXIS, /LAGMUL/NORMAL_AXIS
- Beam Force Sensor: /SENSOR/BEAM_FORCE, /SENSOR/FORCE_BEAM
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


def test_m232_fail_gtn(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
GTN Porous Metal Failure Test
1 1
/FAIL/GTN/77
GTN Void Nucleation and Growth Model
                 1.5                 1.0                 0.3                 0.1
                0.04                0.15                0.25                0.01         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 77 in model.fail_gtns
    fg = model.fail_gtns[77]
    assert pytest.approx(fg.q1) == 1.5
    assert pytest.approx(fg.q2) == 1.0
    assert pytest.approx(fg.eps_n) == 0.3
    assert pytest.approx(fg.s_n) == 0.1
    assert pytest.approx(fg.f_n) == 0.04
    assert pytest.approx(fg.f_c) == 0.15
    assert pytest.approx(fg.f_f) == 0.25
    assert pytest.approx(fg.f_0) == 0.01
    assert fg.ifail_sh == 2


def test_m232_eng_plastic(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Plastic Strain Output Directive Test
1 1
/ENG/PLASTIC/1
Plastic Strain History Output Control
               0.005         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_plastics
    ep = model.eng_plastics[1]
    assert pytest.approx(ep.dt_plastic) == 0.005
    assert ep.sens_id == 2


def test_m232_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Perpendicular and Normal Axis Joint Aliases Test
1 1
/LAGMUL/PERPENDICULAR_AXIS/71
Perpendicular Axis Joint Constraint
        12        22         1         2         0         0             1.0e-05
/LAGMUL/NORMAL_AXIS/72
Normal Axis Joint Constraint
        32        42         2         3         0         0             2.0e-05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 71 in model.perpendicular_joints
    assert model.perpendicular_joints[71].node1 == 12
    assert model.perpendicular_joints[71].node2 == 22
    assert model.perpendicular_joints[71].axis1_dir == 1
    assert model.perpendicular_joints[71].axis2_dir == 2
    assert pytest.approx(model.perpendicular_joints[71].tol) == 1.0e-5

    assert 72 in model.perpendicular_joints
    assert model.perpendicular_joints[72].node1 == 32
    assert model.perpendicular_joints[72].node2 == 42
    assert model.perpendicular_joints[72].axis1_dir == 2
    assert model.perpendicular_joints[72].axis2_dir == 3
    assert pytest.approx(model.perpendicular_joints[72].tol) == 2.0e-5


def test_m232_sensor_beam_force(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Beam Force Sensor Test
1 1
/SENSOR/BEAM_FORCE/132
Beam Force/Moment Threshold Sensor
       500              1200.0               450.0               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 132 in model.sensor_beam_forces
    assert any(s.id == 132 and s.kind == "BEAM_FORCE" for s in model.sensors)
    sbf = model.sensor_beam_forces[132]
    assert sbf.beam_id == 500
    assert pytest.approx(sbf.f_max) == 1200.0
    assert pytest.approx(sbf.m_max) == 450.0
    assert pytest.approx(sbf.t_delay) == 0.002
