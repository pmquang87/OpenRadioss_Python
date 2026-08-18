"""Tests for Milestone M245:
- Ludwik Failure Model: /FAIL/LUDWIK, /FAIL/LUDWIK_LAW, /FAIL/LUDWIK_DAMAGE, /FAIL/LUDWIK_MODEL
- Engine Hourglass Energy Output Directive: /HOURGLASS_ENERGY, /ENG/HOURGLASS_ENERGY, /ENG/HG_ENERGY, /ENG/HE, /ENG/HOURGLASS
- Screw / Helical Axis Joint Aliases: /LAGMUL/SCREW_AXIS, /SCREW_AXIS, /LAGMUL/HELICAL_AXIS, /HELICAL_AXIS, /LAGMUL/HELICAL_JOINT, /HELICAL_JOINT
- Spring Hourglass Energy Sensor: /SENSOR/SPRING_HOURGLASS_ENERGY, /SENSOR/SPRING_HE, /SENSOR/HOURGLASS_ENERGY_SPRING, /SENSOR/SPRING_HOURGLASS
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


def test_m245_fail_ludwik(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Ludwik Failure Model Test
1 1
/FAIL/LUDWIK/77
Ludwik Power-Law Strain Hardening Failure Model
               350.0               600.0                0.28                0.45         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 77 in model.fail_ludwiks
    fl = model.fail_ludwiks[77]
    assert pytest.approx(fl.sigma0) == 350.0
    assert pytest.approx(fl.k_coeff) == 600.0
    assert pytest.approx(fl.n_exp) == 0.28
    assert pytest.approx(fl.eps_max) == 0.45
    assert fl.ifail_sh == 1


def test_m245_eng_hourglass_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Hourglass Energy Output Directive Test
1 1
/ENG/HOURGLASS_ENERGY/1
Hourglass Energy Output Directive
              0.0005         4
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_hourglass_energies
    ehe = model.eng_hourglass_energies[1]
    assert pytest.approx(ehe.dt_he) == 0.0005
    assert ehe.sens_id == 4


def test_m245_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Screw and Helical Axis Joint Aliases Test
1 1
/LAGMUL/SCREW_AXIS/201
Screw Axis Joint Constraint
        10        20         1         0                 5.0
/HELICAL_AXIS/202
Helical Axis Joint Constraint
        30        40         2         0                 8.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 201 in model.screw_joints
    assert model.screw_joints[201].node1 == 10
    assert model.screw_joints[201].node2 == 20
    assert pytest.approx(model.screw_joints[201].pitch) == 5.0

    assert 202 in model.screw_joints
    assert model.screw_joints[202].node1 == 30
    assert model.screw_joints[202].node2 == 40
    assert pytest.approx(model.screw_joints[202].pitch) == 8.0


def test_m245_sensor_spring_hourglass_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Hourglass Energy Sensor Test
1 1
/SENSOR/SPRING_HOURGLASS_ENERGY/175
Spring Element Hourglass Energy Threshold Sensor
        654                85.0               0.004
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 175 in model.sensor_spring_hourglass_energies
    assert any(s.id == 175 and s.kind == "SPRING_HOURGLASS_ENERGY" for s in model.sensors)
    sshe = model.sensor_spring_hourglass_energies[175]
    assert sshe.spring_id == 654
    assert pytest.approx(sshe.ehe_max) == 85.0
    assert pytest.approx(sshe.t_delay) == 0.004
