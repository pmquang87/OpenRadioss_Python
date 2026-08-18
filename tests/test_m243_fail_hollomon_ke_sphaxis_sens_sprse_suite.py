"""Tests for Milestone M243:
- Hollomon Failure Model: /FAIL/HOLLOMON, /FAIL/POWER_LAW
- Engine Kinetic Energy Output Directive: /KINETIC_ENERGY, /ENG/KINETIC_ENERGY
- Spherical / Ball Axis Joint Aliases: /LAGMUL/SPHERICAL_AXIS, /LAGMUL/BALL_AXIS
- Spring Strain Energy Sensor: /SENSOR/SPRING_STRAIN_ENERGY, /SENSOR/STRAIN_ENERGY_SPRING
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


def test_m243_fail_hollomon(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Hollomon Failure Test
1 1
/FAIL/HOLLOMON/88
Hollomon Power-Law Failure Model
               0.002                0.22               850.0                0.65         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 88 in model.fail_hollomons
    fh = model.fail_hollomons[88]
    assert pytest.approx(fh.eps0) == 0.002
    assert pytest.approx(fh.n_exp) == 0.22
    assert pytest.approx(fh.k_coeff) == 850.0
    assert pytest.approx(fh.eps_max) == 0.65
    assert fh.ifail_sh == 1


def test_m243_eng_kinetic_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Kinetic Energy Output Directive Test
1 1
/ENG/KINETIC_ENERGY/1
Kinetic Energy History Output Control
              0.0002         9
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_kinetic_energies
    eke = model.eng_kinetic_energies[1]
    assert pytest.approx(eke.dt_ke) == 0.0002
    assert eke.sens_id == 9


def test_m243_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spherical and Ball Axis Aliases Test
1 1
/LAGMUL/SPHERICAL_AXIS/95
Spherical Axis Joint Constraint
        17        27             1.7e-05
/LAGMUL/BALL_AXIS/96
Ball Axis Joint Constraint
        37        47             2.7e-05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 95 in model.ball_joints
    assert model.ball_joints[95].node1 == 17
    assert model.ball_joints[95].node2 == 27
    assert pytest.approx(model.ball_joints[95].tol) == 1.7e-5

    assert 96 in model.ball_joints
    assert model.ball_joints[96].node1 == 37
    assert model.ball_joints[96].node2 == 47
    assert pytest.approx(model.ball_joints[96].tol) == 2.7e-5


def test_m243_sensor_spring_strain_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Strain Energy Sensor Test
1 1
/SENSOR/SPRING_STRAIN_ENERGY/152
Spring Element Strain Energy Threshold Sensor
       876               450.0               0.009
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 152 in model.sensor_spring_strain_energies
    assert any(s.id == 152 and s.kind == "SPRING_STRAIN_ENERGY" for s in model.sensors)
    ssse = model.sensor_spring_strain_energies[152]
    assert ssse.spring_id == 876
    assert pytest.approx(ssse.estrain_max) == 450.0
    assert pytest.approx(ssse.t_delay) == 0.009
