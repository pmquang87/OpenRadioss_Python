"""Tests for Milestone M250:
- Ritchie-Knott-Rice (RKR) Cleavage Fracture Model: /FAIL/RITCHIE_KNOTT_RICE, /FAIL/RKR, /FAIL/RKR_MODEL, /FAIL/RKR_LAW, /FAIL/RKR_DAMAGE, /FAIL/RITCHIE
- Engine Added Mass Energy Output Directive: /MASS_ENERGY, /ENG/MASS_ENERGY, /ENG/MASS_ENER, /ENG/EMASS, /ENG/MASS_CORRECTION_ENERGY
- Rack and Pinion Kinematic Joint Constraint: /LAGMUL/RACK_PINION, /RACK_PINION, /LAGMUL/RACK_PINION_JOINT, /RACK_PINION_JOINT, /LAGMUL/RACK_PINION_AXIS, /RACK_PINION_AXIS, /LAGMUL/RACK_JOINT, /RACK_JOINT, /LAGMUL/PINION_JOINT, /PINION_JOINT
- Spring Added Mass Energy Sensor: /SENSOR/SPRING_MASS_ENERGY, /SENSOR/SPRING_MASS_ENER, /SENSOR/SPRING_EMASS, /SENSOR/MASS_ENERGY_SPRING
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


def test_m250_fail_ritchie(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
RKR Cleavage Failure Model Test
1 1
/FAIL/RITCHIE_KNOTT_RICE/110
RKR Stress Based Cleavage Fracture Model
              1850.0               0.050                0.02                0.95                0.40         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 110 in model.fail_ritchies
    fr = model.fail_ritchies[110]
    assert pytest.approx(fr.sigma_c) == 1850.0
    assert pytest.approx(fr.l_star) == 0.050
    assert pytest.approx(fr.eps_init) == 0.02
    assert pytest.approx(fr.d_crit) == 0.95
    assert pytest.approx(fr.eps_max) == 0.40
    assert fr.ifail_sh == 1


def test_m250_eng_mass_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Mass Energy Output Directive Test
1 1
/ENG/MASS_ENERGY/1
Added Mass Kinetic Energy Output Control
              0.0001         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_mass_energies
    eme = model.eng_mass_energies[1]
    assert pytest.approx(eme.dt_emass) == 0.0001
    assert eme.sens_id == 3


def test_m250_rack_pinion_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Rack and Pinion Kinematic Joint Test
1 1
/LAGMUL/RACK_PINION/701
Rack and Pinion Joint Constraint 1
        10        20                15.0         1         0              1.0e-5
/RACK_PINION_JOINT/702
Rack and Pinion Joint Constraint 2
        30        40                25.0         2         1              2.0e-5
/RACK_JOINT/703
Rack Joint Constraint 3
        50        60                35.0         3         2              3.0e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 701 in model.rack_pinion_joints
    assert model.rack_pinion_joints[701].node1 == 10
    assert model.rack_pinion_joints[701].node2 == 20
    assert pytest.approx(model.rack_pinion_joints[701].pitch_radius) == 15.0
    assert model.rack_pinion_joints[701].axis_dir == 1
    assert model.rack_pinion_joints[701].skew_id == 0
    assert pytest.approx(model.rack_pinion_joints[701].tol) == 1.0e-5

    assert 702 in model.rack_pinion_joints
    assert model.rack_pinion_joints[702].node1 == 30
    assert model.rack_pinion_joints[702].node2 == 40
    assert pytest.approx(model.rack_pinion_joints[702].pitch_radius) == 25.0
    assert model.rack_pinion_joints[702].axis_dir == 2
    assert model.rack_pinion_joints[702].skew_id == 1
    assert pytest.approx(model.rack_pinion_joints[702].tol) == 2.0e-5

    assert 703 in model.rack_pinion_joints
    assert model.rack_pinion_joints[703].node1 == 50
    assert model.rack_pinion_joints[703].node2 == 60
    assert pytest.approx(model.rack_pinion_joints[703].pitch_radius) == 35.0
    assert model.rack_pinion_joints[703].axis_dir == 3
    assert model.rack_pinion_joints[703].skew_id == 2
    assert pytest.approx(model.rack_pinion_joints[703].tol) == 3.0e-5


def test_m250_sensor_spring_mass_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Mass Energy Sensor Test
1 1
/SENSOR/SPRING_MASS_ENERGY/305
Spring Added Mass Energy Threshold Sensor
        880               120.0               0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 305 in model.sensor_spring_mass_energies
    assert any(s.id == 305 and s.kind == "SPRING_MASS_ENERGY" for s in model.sensors)
    ssme = model.sensor_spring_mass_energies[305]
    assert ssme.spring_id == 880
    assert pytest.approx(ssme.emass_max) == 120.0
    assert pytest.approx(ssme.t_delay) == 0.001
