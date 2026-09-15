"""Tests for Milestone M251:
- Gologanu-Leblond-Devaux Void Shape Evolution Failure Model: /FAIL/GOLOGANU, /FAIL/GLD, /FAIL/GOLOGANU_LEBLOND_DEVAUX, /FAIL/GOLOGANU_MODEL, /FAIL/GOLOGANU_LAW, /FAIL/GOLOGANU_DAMAGE
- Engine Mass Change / Variation Output Directive: /MASS_CHANGE, /ENG/MASS_CHANGE, /ENG/DMASS, /ENG/DELTA_MASS, /ENG/MASS_VARIATION
- Belt Pulley Kinematic Joint Constraint: /LAGMUL/BELT_PULLEY, /BELT_PULLEY, /LAGMUL/PULLEY_JOINT, /PULLEY_JOINT, /LAGMUL/BELT_JOINT, /BELT_JOINT, /LAGMUL/PULLEY, /PULLEY
- Spring Mass Variation Sensor: /SENSOR/SPRING_MASS_CHANGE, /SENSOR/SPRING_DMASS, /SENSOR/SPRING_DELTA_MASS, /SENSOR/MASS_CHANGE_SPRING
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


def test_m251_fail_gologanu(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Gologanu Void Shape Evolution Failure Model Test
1 1
/FAIL/GOLOGANU/112
Gologanu-Leblond-Devaux Void Growth Model
               0.002                 0.8                0.12                0.22         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 112 in model.fail_gologanus
    fgld = model.fail_gologanus[112]
    assert pytest.approx(fgld.f0) == 0.002
    assert pytest.approx(fgld.s0) == 0.8
    assert pytest.approx(fgld.fc) == 0.12
    assert pytest.approx(fgld.ff) == 0.22
    assert fgld.ifail_sh == 1


def test_m251_eng_mass_change(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Mass Change Output Directive Test
1 1
/ENG/MASS_CHANGE/1
Engine Delta Mass History Output Control
              0.0005         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_mass_changes
    emc = model.eng_mass_changes[1]
    assert pytest.approx(emc.dt_dmass) == 0.0005
    assert emc.sens_id == 2


def test_m251_belt_pulley_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Belt Pulley Kinematic Joint Constraint Test
1 1
/LAGMUL/BELT_PULLEY/801
Belt Pulley Constraint 1
        10        20                 0.1                 0.2         1         1         0              1.0e-5
/PULLEY_JOINT/802
Belt Pulley Constraint 2
        30        40                 0.15                0.3         2         2         1              2.0e-5
/BELT_JOINT/803
Belt Pulley Constraint 3
        50        60                 0.05                0.25        3         3         2              3.0e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 801 in model.belt_pulley_joints
    assert model.belt_pulley_joints[801].node1 == 10
    assert model.belt_pulley_joints[801].node2 == 20
    assert pytest.approx(model.belt_pulley_joints[801].radius1) == 0.1
    assert pytest.approx(model.belt_pulley_joints[801].radius2) == 0.2
    assert model.belt_pulley_joints[801].axis1_dir == 1
    assert model.belt_pulley_joints[801].axis2_dir == 1
    assert model.belt_pulley_joints[801].skew_id == 0
    assert pytest.approx(model.belt_pulley_joints[801].tol) == 1.0e-5

    assert 802 in model.belt_pulley_joints
    assert model.belt_pulley_joints[802].node1 == 30
    assert model.belt_pulley_joints[802].node2 == 40
    assert pytest.approx(model.belt_pulley_joints[802].radius1) == 0.15
    assert pytest.approx(model.belt_pulley_joints[802].radius2) == 0.3
    assert model.belt_pulley_joints[802].axis1_dir == 2
    assert model.belt_pulley_joints[802].axis2_dir == 2
    assert model.belt_pulley_joints[802].skew_id == 1
    assert pytest.approx(model.belt_pulley_joints[802].tol) == 2.0e-5

    assert 803 in model.belt_pulley_joints
    assert model.belt_pulley_joints[803].node1 == 50
    assert model.belt_pulley_joints[803].node2 == 60
    assert pytest.approx(model.belt_pulley_joints[803].radius1) == 0.05
    assert pytest.approx(model.belt_pulley_joints[803].radius2) == 0.25
    assert model.belt_pulley_joints[803].axis1_dir == 3
    assert model.belt_pulley_joints[803].axis2_dir == 3
    assert model.belt_pulley_joints[803].skew_id == 2
    assert pytest.approx(model.belt_pulley_joints[803].tol) == 3.0e-5


def test_m251_sensor_spring_mass_change(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Mass Change Sensor Test
1 1
/SENSOR/SPRING_MASS_CHANGE/405
Spring Mass Variation Threshold Sensor
        920                50.0               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 405 in model.sensor_spring_mass_changes
    assert any(s.id == 405 and s.kind == "SPRING_MASS_CHANGE" for s in model.sensors)
    ssmc = model.sensor_spring_mass_changes[405]
    assert ssmc.spring_id == 920
    assert pytest.approx(ssmc.dmass_max) == 50.0
    assert pytest.approx(ssmc.t_delay) == 0.002
