"""Tests for Milestone M238:
- Strain Energy Density Failure Model: /FAIL/ENERGY_DENSITY, /FAIL/SED
- Engine Rotational Acceleration Output Directive: /ROTA, /ENG/ROTA
- Planar/Plane Axis Joint Aliases: /LAGMUL/PLANE_JOINT, /LAGMUL/PLANAR_AXIS
- Spring Rotational Acceleration Sensor: /SENSOR/SPRING_ROTA, /SENSOR/ROTA_SPRING
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


def test_m238_fail_energy_density(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Strain Energy Density Failure Test
1 1
/FAIL/ENERGY_DENSITY/98
Strain Energy Density Failure Model
                 12.5                25.0         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 98 in model.fail_energy_densitys
    fed = model.fail_energy_densitys[98]
    assert pytest.approx(fed.w_crit) == 12.5
    assert pytest.approx(fed.w_rupt) == 25.0
    assert fed.ifail_sh == 2


def test_m238_eng_rota(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Rotational Acceleration Output Directive Test
1 1
/ENG/ROTA/1
Rotational Acceleration Vector History Output Control
               0.002         7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_rotas
    er = model.eng_rotas[1]
    assert pytest.approx(er.dt_rota) == 0.002
    assert er.sens_id == 7


def test_m238_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Plane Joint and Planar Axis Joint Aliases Test
1 1
/LAGMUL/PLANE_JOINT/93
Plane Joint Constraint
        18        28         3         5             1.5e-05
/LAGMUL/PLANAR_AXIS/94
Planar Axis Joint Constraint
        38        48         1         6             2.5e-05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 93 in model.planar_joints
    assert model.planar_joints[93].node1 == 18
    assert model.planar_joints[93].node2 == 28
    assert model.planar_joints[93].axis_dir == 3
    assert model.planar_joints[93].skew_id == 5
    assert pytest.approx(model.planar_joints[93].tol) == 1.5e-5

    assert 94 in model.planar_joints
    assert model.planar_joints[94].node1 == 38
    assert model.planar_joints[94].node2 == 48
    assert model.planar_joints[94].axis_dir == 1
    assert model.planar_joints[94].skew_id == 6
    assert pytest.approx(model.planar_joints[94].tol) == 2.5e-5


def test_m238_sensor_spring_rota(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Rotational Acceleration Sensor Test
1 1
/SENSOR/SPRING_ROTA/138
Spring Angular Acceleration Threshold Sensor
       781               314.0               0.006
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 138 in model.sensor_spring_rotas
    assert any(s.id == 138 and s.kind == "SPRING_ROTA" for s in model.sensors)
    ssra = model.sensor_spring_rotas[138]
    assert ssra.spring_id == 781
    assert pytest.approx(ssra.rota_max) == 314.0
    assert pytest.approx(ssra.t_delay) == 0.006
