"""Tests for Milestone M252:
- Rousselier Ductile Damage and Void Growth Fracture Model: /FAIL/ROUSSELIER, /FAIL/ROUSSELIER_MODEL, /FAIL/ROUSSELIER_LAW, /FAIL/ROUSSELIER_DAMAGE, /FAIL/ROUSS
- Engine Hydrostatic Pressure Output Directive: /ENG/PRESSURE, /ENG_PRESSURE, /ENG/PRESS, /ENG/HYDROSTATIC_PRESSURE, /HYDROSTATIC_PRESSURE
- Oldham Coupling Kinematic Joint Constraint: /LAGMUL/OLDHAM, /OLDHAM, /LAGMUL/OLDHAM_JOINT, /OLDHAM_JOINT, /LAGMUL/OLDHAM_COUPLING, /OLDHAM_COUPLING
- Spring Pressure / Normal Force Sensor: /SENSOR/SPRING_PRESSURE, /SENSOR/SPRING_PRESS, /SENSOR/PRESSURE_SPRING, /SENSOR/SPRING_HYDROSTATIC_PRESSURE
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


def test_m252_fail_rousselier(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Rousselier Ductile Fracture Model Test
1 1
/FAIL/ROUSSELIER/115
Rousselier Void Growth Failure Model
              0.0005               450.0                 0.9                0.05               1.5e5         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 115 in model.fail_rousseliers
    fr = model.fail_rousseliers[115]
    assert pytest.approx(fr.d0) == 0.0005
    assert pytest.approx(fr.sigma1) == 450.0
    assert pytest.approx(fr.d_crit) == 0.9
    assert pytest.approx(fr.eps_init) == 0.05
    assert pytest.approx(fr.eps_max) == 1.5e5
    assert fr.ifail_sh == 1


def test_m252_eng_pressure(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Pressure Output Directive Test
1 1
/ENG/PRESSURE/1
Engine Hydrostatic Pressure Output Control
              0.0002         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_pressures
    ep = model.eng_pressures[1]
    assert pytest.approx(ep.dt_press) == 0.0002
    assert ep.sens_id == 3


def test_m252_oldham_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Oldham Coupling Kinematic Joint Constraint Test
1 1
/LAGMUL/OLDHAM/810
Oldham Coupling Constraint 1
        10        20         1         0              1.0e-5
/OLDHAM_JOINT/811
Oldham Coupling Constraint 2
        30        40         2         1              2.0e-5
/OLDHAM_COUPLING/812
Oldham Coupling Constraint 3
        50        60         3         2              3.0e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 810 in model.oldham_joints
    assert model.oldham_joints[810].node1 == 10
    assert model.oldham_joints[810].node2 == 20
    assert model.oldham_joints[810].axis_dir == 1
    assert model.oldham_joints[810].skew_id == 0
    assert pytest.approx(model.oldham_joints[810].tol) == 1.0e-5

    assert 811 in model.oldham_joints
    assert model.oldham_joints[811].node1 == 30
    assert model.oldham_joints[811].node2 == 40
    assert model.oldham_joints[811].axis_dir == 2
    assert model.oldham_joints[811].skew_id == 1
    assert pytest.approx(model.oldham_joints[811].tol) == 2.0e-5

    assert 812 in model.oldham_joints
    assert model.oldham_joints[812].node1 == 50
    assert model.oldham_joints[812].node2 == 60
    assert model.oldham_joints[812].axis_dir == 3
    assert model.oldham_joints[812].skew_id == 2
    assert pytest.approx(model.oldham_joints[812].tol) == 3.0e-5


def test_m252_sensor_spring_pressure(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Pressure Sensor Test
1 1
/SENSOR/SPRING_PRESSURE/410
Spring Normal Pressure Threshold Sensor
        930               250.0               0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 410 in model.sensor_spring_pressures
    assert any(s.id == 410 and s.kind == "SPRING_PRESSURE" for s in model.sensors)
    ssp = model.sensor_spring_pressures[410]
    assert ssp.spring_id == 930
    assert pytest.approx(ssp.press_max) == 250.0
    assert pytest.approx(ssp.t_delay) == 0.001
