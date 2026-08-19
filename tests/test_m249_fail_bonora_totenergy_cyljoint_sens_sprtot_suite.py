"""Tests for Milestone M249:
- Bonora Continuous Ductile Damage Failure Model: /FAIL/BONORA, /FAIL/BONORA_LAW, /FAIL/BONORA_DAMAGE, /FAIL/BONORA_MODEL
- Engine Total System Energy Output Directive: /TOT_ENERGY, /ENG/TOT_ENERGY, /ENG/ENERGY_TOTAL, /ENG/TOTAL_SYSTEM_ENERGY, /ENG/ETOT
- Cylindrical Axis Joint Aliases: /LAGMUL/CYLINDRICAL_AXIS, /CYLINDRICAL_AXIS, /LAGMUL/CYL_JOINT_AXIS, /CYL_JOINT_AXIS, /LAGMUL/CYL_AXIS, /CYL_AXIS
- Spring Total Energy Sensor: /SENSOR/SPRING_TOT_ENERGY, /SENSOR/SPRING_TOTAL_ENERGY, /SENSOR/TOT_ENERGY_SPRING, /SENSOR/SPRING_ETOT
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


def test_m249_fail_bonora(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Bonora Failure Model Test
1 1
/FAIL/BONORA/105
Bonora Ductile Damage Model
                0.08                0.85                0.90                0.05                0.60         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 105 in model.fail_bonoras
    fb = model.fail_bonoras[105]
    assert pytest.approx(fb.p_th) == 0.08
    assert pytest.approx(fb.p_cr) == 0.85
    assert pytest.approx(fb.d_cr) == 0.90
    assert pytest.approx(fb.d_0) == 0.05
    assert pytest.approx(fb.alpha) == 0.60
    assert fb.ifail_sh == 1


def test_m249_eng_tot_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Total System Energy Output Directive Test
1 1
/ENG/TOT_ENERGY/1
Total System Energy History Output Control
             0.00025         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_tot_energies
    ete = model.eng_tot_energies[1]
    assert pytest.approx(ete.dt_etot) == 0.00025
    assert ete.sens_id == 5


def test_m249_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Cylindrical Axis Joint Aliases Test
1 1
/LAGMUL/CYLINDRICAL_AXIS/601
Cylindrical Axis Joint Constraint
        15        25         1         0              0.0001
/CYL_JOINT_AXIS/602
Cylindrical Joint Axis Constraint
        35        45         2         0              0.0002
/CYL_AXIS/603
Cyl Axis Constraint
        55        65         3         0              0.0003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 601 in model.cyl_joints
    assert model.cyl_joints[601].node1 == 15
    assert model.cyl_joints[601].node2 == 25
    assert model.cyl_joints[601].axis_dir == 1
    assert pytest.approx(model.cyl_joints[601].tol) == 0.0001

    assert 602 in model.cyl_joints
    assert model.cyl_joints[602].node1 == 35
    assert model.cyl_joints[602].node2 == 45
    assert model.cyl_joints[602].axis_dir == 2
    assert pytest.approx(model.cyl_joints[602].tol) == 0.0002

    assert 603 in model.cyl_joints
    assert model.cyl_joints[603].node1 == 55
    assert model.cyl_joints[603].node2 == 65
    assert model.cyl_joints[603].axis_dir == 3
    assert pytest.approx(model.cyl_joints[603].tol) == 0.0003


def test_m249_sensor_spring_tot_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Total Energy Sensor Test
1 1
/SENSOR/SPRING_TOT_ENERGY/205
Spring Element Total Energy Threshold Sensor
        920               750.0               0.005
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 205 in model.sensor_spring_tot_energies
    assert any(s.id == 205 and s.kind == "SPRING_TOT_ENERGY" for s in model.sensors)
    sste = model.sensor_spring_tot_energies[205]
    assert sste.spring_id == 920
    assert pytest.approx(sste.etot_max) == 750.0
    assert pytest.approx(sste.t_delay) == 0.005
