"""Tests for Milestone M235:
- Gurson Porous Metal Failure Criterion: /FAIL/GURSON, /FAIL/GURSON_MODEL
- Engine Displacement Output Directive: /DISP, /ENG/DISP
- Distance/Rod Joint Aliases: /LAGMUL/DISTANCE_JOINT, /LAGMUL/ROD
- Spring Deflection Sensor: /SENSOR/SPRING_DEFL, /SENSOR/DEF_SPRING
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


def test_m235_fail_gurson(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Gurson Failure Test
1 1
/FAIL/GURSON/92
Gurson Porous Metal Failure Model
                0.01                0.12                0.22
                0.28                0.08                0.03         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 92 in model.fail_gursons
    fg = model.fail_gursons[92]
    assert pytest.approx(fg.f_0) == 0.01
    assert pytest.approx(fg.f_c) == 0.12
    assert pytest.approx(fg.f_u) == 0.22
    assert pytest.approx(fg.eps_n) == 0.28
    assert pytest.approx(fg.s_n) == 0.08
    assert pytest.approx(fg.f_n) == 0.03
    assert fg.ifail_sh == 2


def test_m235_eng_disp(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Displacement Output Directive Test
1 1
/ENG/DISP/1
Displacement Vector History Output Control
               0.005         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_disps
    ed = model.eng_disps[1]
    assert pytest.approx(ed.dt_disp) == 0.005
    assert ed.sens_id == 3


def test_m235_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Distance and Rod Joint Aliases Test
1 1
/LAGMUL/DISTANCE_JOINT/87
Distance Joint Constraint
        15        25               125.0             1.0e-05
/LAGMUL/ROD/88
Rod Joint Constraint
        35        45               250.0             2.0e-05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 87 in model.distance_joints
    assert model.distance_joints[87].node1 == 15
    assert model.distance_joints[87].node2 == 25
    assert pytest.approx(model.distance_joints[87].dist) == 125.0
    assert pytest.approx(model.distance_joints[87].tol) == 1.0e-5

    assert 88 in model.distance_joints
    assert model.distance_joints[88].node1 == 35
    assert model.distance_joints[88].node2 == 45
    assert pytest.approx(model.distance_joints[88].dist) == 250.0
    assert pytest.approx(model.distance_joints[88].tol) == 2.0e-5


def test_m235_sensor_spring_defl(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Deflection Sensor Test
1 1
/SENSOR/SPRING_DEFL/135
Spring Elongation Threshold Sensor
       760                15.5               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 135 in model.sensor_spring_defls
    assert any(s.id == 135 and s.kind == "SPRING_DEFL" for s in model.sensors)
    ssd = model.sensor_spring_defls[135]
    assert ssd.spring_id == 760
    assert pytest.approx(ssd.defl_max) == 15.5
    assert pytest.approx(ssd.t_delay) == 0.003
