"""Tests for Milestone M233:
- Tabulated 3D Failure Criterion: /FAIL/TAB3, /FAIL/TABULATED3
- Engine Velocity Output Directive: /VELOCITY, /ENG/VELOCITY
- Gimbal Axis/Universal Joint Aliases: /LAGMUL/GIMBAL_AXIS, /LAGMUL/UNIVERSAL_GIMBAL
- Truss Force Sensor: /SENSOR/TRUSS_FORCE, /SENSOR/FORCE_TRUSS
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


def test_m233_fail_tab3(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Tabulated 3D Failure Test
1 1
/FAIL/TAB3/88
3D Tabulated Triaxiality Failure Model
        12                 1.2                 1.1                 0.9
                0.35                0.05         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 88 in model.fail_tab3s
    ft = model.fail_tab3s[88]
    assert ft.table_id == 12
    assert pytest.approx(ft.scale_x) == 1.2
    assert pytest.approx(ft.scale_y) == 1.1
    assert pytest.approx(ft.scale_z) == 0.9
    assert pytest.approx(ft.eps_max) == 0.35
    assert pytest.approx(ft.d_adv) == 0.05
    assert ft.ifail_sh == 2


def test_m233_eng_velocity(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Velocity Output Directive Test
1 1
/ENG/VELOCITY/1
Velocity Vector History Output Control
               0.002         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_velocities
    ev = model.eng_velocities[1]
    assert pytest.approx(ev.dt_vel) == 0.002
    assert ev.sens_id == 3


def test_m233_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Gimbal and Universal Gimbal Axis Joint Aliases Test
1 1
/LAGMUL/GIMBAL_AXIS/81
Gimbal Axis Joint Constraint
        15        25         1         2         0         0             1.0e-05
/LAGMUL/UNIVERSAL_GIMBAL/82
Universal Gimbal Joint Constraint
        35        45         2         3         0         0             2.0e-05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 81 in model.gimbal_joints
    assert model.gimbal_joints[81].node1 == 15
    assert model.gimbal_joints[81].node2 == 25
    assert model.gimbal_joints[81].axis1_dir == 1
    assert model.gimbal_joints[81].axis2_dir == 2
    assert pytest.approx(model.gimbal_joints[81].tol) == 1.0e-5

    assert 82 in model.gimbal_joints
    assert model.gimbal_joints[82].node1 == 35
    assert model.gimbal_joints[82].node2 == 45
    assert model.gimbal_joints[82].axis1_dir == 2
    assert model.gimbal_joints[82].axis2_dir == 3
    assert pytest.approx(model.gimbal_joints[82].tol) == 2.0e-5


def test_m233_sensor_truss_force(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Truss Force Sensor Test
1 1
/SENSOR/TRUSS_FORCE/133
Truss Axial Force Threshold Sensor
       600              3500.0               0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 133 in model.sensor_truss_forces
    assert any(s.id == 133 and s.kind == "TRUSS_FORCE" for s in model.sensors)
    stf = model.sensor_truss_forces[133]
    assert stf.truss_id == 600
    assert pytest.approx(stf.f_max) == 3500.0
    assert pytest.approx(stf.t_delay) == 0.001
