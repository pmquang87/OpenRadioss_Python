"""Tests for Milestone M222:
- Bammann-Chiesa-Johnson Failure Criterion: /FAIL/BAMMAN, /FAIL/BCJ
- Engine Energy Balance Directive: /ENERGY, /ENG/ENERGY
- Planar Kinematic Joint Aliases: /LAGMUL/PLANAR, /PLANAR_JOINT
- Added Mass Ratio Sensor: /SENSOR/MASS, /SENSOR/MASS_RATIO
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


def test_m222_fail_bamman(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Bammann BCJ Failure Model Test
1 1
/FAIL/BAMMAN/99
Bammann-Chiesa-Johnson Void Damage Model
                0.01                0.05                0.85                 2.0         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 99 in model.fail_bammans
    fb = model.fail_bammans[99]
    assert pytest.approx(fb.v0) == 0.01
    assert pytest.approx(fb.an) == 0.05
    assert pytest.approx(fb.bn) == 0.85
    assert pytest.approx(fb.cn) == 2.0
    assert fb.ifail_sh == 1


def test_m222_eng_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Energy Balance Test
1 1
/ENG/ENERGY/1
Energy Balance Tracking Directive
                0.01         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_energies
    ee = model.eng_energies[1]
    assert pytest.approx(ee.dt_energy) == 0.01
    assert ee.sens_id == 5


def test_m222_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Planar Kinematic Joint Aliases Test
1 1
/LAGMUL/PLANAR/16
Planar Joint Constraint
        31        32         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 16 in model.planar_joints
    assert model.planar_joints[16].node1 == 31
    assert model.planar_joints[16].node2 == 32


def test_m222_sensor_mass(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Added Mass Ratio Sensor Test
1 1
/SENSOR/MASS/55
Added Mass Ratio Trigger Sensor
         2                 0.05                0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 55 in model.sensor_mass_ratios
    assert any(s.id == 55 and s.kind == "MASS_RATIO" for s in model.sensors)
    sm = model.sensor_mass_ratios[55]
    assert sm.part_id == 2
    assert pytest.approx(sm.dmass_max) == 0.05
    assert pytest.approx(sm.t_delay) == 0.001
