"""Tests for Milestone M228:
- Hill Anisotropic Failure Criterion: /FAIL/HILL, /FAIL/HILL_PLASTIC
- Engine Geometry Update Directive: /GEO, /ENG/GEO
- Pin-Slot & Revolute Axis Joint Aliases: /LAGMUL/PIN_SLOT, /LAGMUL/REVOLUTE_AXIS
- Beam Strain Sensor: /SENSOR/BEAM_STRAIN, /SENSOR/STRAIN_BEAM
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


def test_m228_fail_hill(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Hill Anisotropic Failure Test
1 1
/FAIL/HILL/77
Hill Anisotropic Plasticity Failure
                 0.6                 0.4                 0.5                 1.4
                 1.6                 1.3               450.0         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 77 in model.fail_hills
    fh = model.fail_hills[77]
    assert pytest.approx(fh.F) == 0.6
    assert pytest.approx(fh.G) == 0.4
    assert pytest.approx(fh.H) == 0.5
    assert pytest.approx(fh.L) == 1.4
    assert pytest.approx(fh.M) == 1.6
    assert pytest.approx(fh.N) == 1.3
    assert pytest.approx(fh.sigma_fail) == 450.0
    assert fh.ifail_sh == 1


def test_m228_eng_geo(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Geometry Update Directive Test
1 1
/ENG/GEO/1
Geometry Refresh Output Control
                0.05         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_geos
    eg = model.eng_geos[1]
    assert pytest.approx(eg.dt_geo) == 0.05
    assert eg.sens_id == 3


def test_m228_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Pin Slot and Revolute Axis Joint Aliases Test
1 1
/LAGMUL/PIN_SLOT/31
Pin Slot Joint Constraint
        12        22         4
/LAGMUL/REVOLUTE_AXIS/32
Revolute Axis Joint Constraint
        32        42         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 31 in model.slot_joints
    assert model.slot_joints[31].node1 == 12
    assert model.slot_joints[31].node2 == 22

    assert 32 in model.pin_joints
    assert model.pin_joints[32].node1 == 32
    assert model.pin_joints[32].node2 == 42


def test_m228_sensor_beam_strain(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Beam Strain Sensor Test
1 1
/SENSOR/BEAM_STRAIN/120
Beam Strain Threshold Sensor
       150                 0.18         2               0.004
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 120 in model.sensor_beam_strains
    assert any(s.id == 120 and s.kind == "BEAM_STRAIN" for s in model.sensors)
    sbs = model.sensor_beam_strains[120]
    assert sbs.beam_id == 150
    assert pytest.approx(sbs.eps_max) == 0.18
    assert sbs.ip == 2
    assert pytest.approx(sbs.t_delay) == 0.004
