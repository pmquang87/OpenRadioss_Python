"""Tests for Milestone M246:
- Voce Failure Model: /FAIL/VOCE, /FAIL/VOCE_LAW, /FAIL/VOCE_DAMAGE, /FAIL/VOCE_MODEL
- Engine Contact Energy Output Directive: /CONTACT_ENERGY, /ENG/CONTACT_ENERGY, /ENG/CONT_ENERGY, /ENG/CE, /ENG/CONTACT
- Slot Axis Joint Aliases: /LAGMUL/SLOT_AXIS, /SLOT_AXIS, /LAGMUL/SLOT_LINE_AXIS, /SLOT_LINE_AXIS
- Spring Contact Energy Sensor: /SENSOR/SPRING_CONTACT_ENERGY, /SENSOR/SPRING_CE, /SENSOR/CONTACT_ENERGY_SPRING, /SENSOR/SPRING_CONTACT
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


def test_m246_fail_voce(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Voce Failure Model Test
1 1
/FAIL/VOCE/88
Voce Isotropic Saturation Hardening Failure Model
               400.0               750.0                15.5                0.35         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 88 in model.fail_voces
    fv = model.fail_voces[88]
    assert pytest.approx(fv.sigma0) == 400.0
    assert pytest.approx(fv.sigma_inf) == 750.0
    assert pytest.approx(fv.beta) == 15.5
    assert pytest.approx(fv.eps_max) == 0.35
    assert fv.ifail_sh == 1


def test_m246_eng_contact_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Contact Energy Output Directive Test
1 1
/ENG/CONTACT_ENERGY/1
Contact Energy History Output Control
              0.0002         6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_contact_energies
    ece = model.eng_contact_energies[1]
    assert pytest.approx(ece.dt_ce) == 0.0002
    assert ece.sens_id == 6


def test_m246_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Slot Axis Joint Aliases Test
1 1
/LAGMUL/SLOT_AXIS/301
Slot Axis Joint Constraint
        15        25         0         1               -10.0                10.0
/SLOT_LINE_AXIS/302
Slot Line Axis Joint Constraint
        35        45         0         2               -20.0                20.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 301 in model.slot_joints
    assert model.slot_joints[301].node1 == 15
    assert model.slot_joints[301].node2 == 25
    assert pytest.approx(model.slot_joints[301].d_min) == -10.0
    assert pytest.approx(model.slot_joints[301].d_max) == 10.0

    assert 302 in model.slot_joints
    assert model.slot_joints[302].node1 == 35
    assert model.slot_joints[302].node2 == 45
    assert pytest.approx(model.slot_joints[302].d_min) == -20.0
    assert pytest.approx(model.slot_joints[302].d_max) == 20.0


def test_m246_sensor_spring_contact_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Contact Energy Sensor Test
1 1
/SENSOR/SPRING_CONTACT_ENERGY/188
Spring Element Contact Energy Threshold Sensor
        789               150.0               0.008
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 188 in model.sensor_spring_contact_energies
    assert any(s.id == 188 and s.kind == "SPRING_CONTACT_ENERGY" for s in model.sensors)
    ssce = model.sensor_spring_contact_energies[188]
    assert ssce.spring_id == 789
    assert pytest.approx(ssce.ece_max) == 150.0
    assert pytest.approx(ssce.t_delay) == 0.008
