"""Tests for Milestone M244:
- Swift Failure Model: /FAIL/SWIFT, /FAIL/SWIFT_LAW
- Engine Total Energy Output Directive: /TOTAL_ENERGY, /ENG/TOTAL_ENERGY
- Differential Joint / Axis Aliases: /LAGMUL/DIFF_JOINT, /DIFF_JOINT, /LAGMUL/DIFF_AXIS, /DIFF_AXIS
- Spring Kinetic Energy Sensor: /SENSOR/SPRING_KINETIC_ENERGY, /SENSOR/KINETIC_ENERGY_SPRING
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


def test_m244_fail_swift(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Swift Failure Test
1 1
/FAIL/SWIFT/99
Swift Power-Law Failure Model
               0.005                0.19               920.0                0.55         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 99 in model.fail_swifts
    fs = model.fail_swifts[99]
    assert pytest.approx(fs.eps0) == 0.005
    assert pytest.approx(fs.n_exp) == 0.19
    assert pytest.approx(fs.k_coeff) == 920.0
    assert pytest.approx(fs.eps_max) == 0.55
    assert fs.ifail_sh == 1


def test_m244_eng_total_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Total Energy Output Directive Test
1 1
/ENG/TOTAL_ENERGY/1
Total Energy History Output Control
              0.0001         8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_total_energies
    ete = model.eng_total_energies[1]
    assert pytest.approx(ete.dt_te) == 0.0001
    assert ete.sens_id == 8


def test_m244_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Differential Joint and Axis Aliases Test
1 1
/LAGMUL/DIFF_JOINT/101
Differential Joint Constraint
        11        22        33                 1.8
/DIFF_AXIS/102
Differential Axis Constraint
        44        55        66                 2.4
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 101 in model.diffs
    assert model.diffs[101].node0 == 11
    assert model.diffs[101].node1 == 22
    assert model.diffs[101].node2 == 33
    assert pytest.approx(model.diffs[101].ratio) == 1.8

    assert 102 in model.diffs
    assert model.diffs[102].node0 == 44
    assert model.diffs[102].node1 == 55
    assert model.diffs[102].node2 == 66
    assert pytest.approx(model.diffs[102].ratio) == 2.4


def test_m244_sensor_spring_kinetic_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Kinetic Energy Sensor Test
1 1
/SENSOR/SPRING_KINETIC_ENERGY/162
Spring Element Kinetic Energy Threshold Sensor
       987               320.0               0.006
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 162 in model.sensor_spring_kinetic_energies
    assert any(s.id == 162 and s.kind == "SPRING_KINETIC_ENERGY" for s in model.sensors)
    sske = model.sensor_spring_kinetic_energies[162]
    assert sske.spring_id == 987
    assert pytest.approx(sske.ekin_max) == 320.0
    assert pytest.approx(sske.t_delay) == 0.006
