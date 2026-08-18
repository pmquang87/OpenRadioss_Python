"""Tests for Milestone M242:
- Lou-Huhn Failure Model: /FAIL/LOU_HUHN, /FAIL/LH
- Engine Internal Energy Output Directive: /INTERNAL_ENERGY, /ENG/INTERNAL_ENERGY
- Constant Velocity Joint Aliases: /LAGMUL/CONSTANT_VELOCITY_JOINT, /LAGMUL/CONSTANT_VELOCITY_AXIS
- Spring Torsional Moment Sensor: /SENSOR/SPRING_TORSION, /SENSOR/TORSION_SPRING
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


def test_m242_fail_lou_huhn(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Lou-Huhn Failure Test
1 1
/FAIL/LOU_HUHN/77
Lou-Huhn Shear Ductile Fracture Model
                0.42                0.18                1.25               0.333         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 77 in model.fail_lou_huhns
    flh = model.fail_lou_huhns[77]
    assert pytest.approx(flh.c1) == 0.42
    assert pytest.approx(flh.c2) == 0.18
    assert pytest.approx(flh.l_param) == 1.25
    assert pytest.approx(flh.eta0) == 0.333
    assert flh.ifail_sh == 1


def test_m242_eng_internal_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Internal Energy Output Directive Test
1 1
/ENG/INTERNAL_ENERGY/1
Internal Energy History Output Control
              0.0005         8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_internal_energies
    eie = model.eng_internal_energies[1]
    assert pytest.approx(eie.dt_ie) == 0.0005
    assert eie.sens_id == 8


def test_m242_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Constant Velocity Joint and Axis Aliases Test
1 1
/LAGMUL/CONSTANT_VELOCITY_JOINT/93
Constant Velocity Universal Joint Constraint
        16        26         1         0             1.6e-05
/LAGMUL/CONSTANT_VELOCITY_AXIS/94
Constant Velocity Axis Joint Constraint
        36        46         2         0             2.6e-05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 93 in model.cv_joints
    assert model.cv_joints[93].node1 == 16
    assert model.cv_joints[93].node2 == 26
    assert model.cv_joints[93].axis_dir == 1
    assert pytest.approx(model.cv_joints[93].tol) == 1.6e-5

    assert 94 in model.cv_joints
    assert model.cv_joints[94].node1 == 36
    assert model.cv_joints[94].node2 == 46
    assert model.cv_joints[94].axis_dir == 2
    assert pytest.approx(model.cv_joints[94].tol) == 2.6e-5


def test_m242_sensor_spring_torsion(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Torsional Moment Sensor Test
1 1
/SENSOR/SPRING_TORSION/151
Spring Torsional Moment Threshold Sensor
       765              1800.0               0.007
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 151 in model.sensor_spring_torsions
    assert any(s.id == 151 and s.kind == "SPRING_TORSION" for s in model.sensors)
    sst = model.sensor_spring_torsions[151]
    assert sst.spring_id == 765
    assert pytest.approx(sst.mtor_max) == 1800.0
    assert pytest.approx(sst.t_delay) == 0.007
