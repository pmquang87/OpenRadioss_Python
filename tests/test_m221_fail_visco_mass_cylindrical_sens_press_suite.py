"""Tests for Milestone M221:
- Viscoplastic Failure Criterion: /FAIL/VISCO, /FAIL/VISCO_PLASTIC
- Engine Mass Summary Directive: /MASS, /ENG/MASS
- Cylindrical Kinematic Joint Aliases: /LAGMUL/CYLINDRICAL, /CYLINDRICAL_JOINT
- Pressure Sensor: /SENSOR/PRESSURE, /SENSOR/PRESS
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


def test_m221_fail_visco(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Viscoplastic Failure Model Test
1 1
/FAIL/VISCO/88
Viscoplastic Strain-Rate Sensitive Failure Law
                0.25               100.0                0.15         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 88 in model.fail_viscos
    fv = model.fail_viscos[88]
    assert pytest.approx(fv.eps_f0) == 0.25
    assert pytest.approx(fv.eps_rate0) == 100.0
    assert pytest.approx(fv.m_rate) == 0.15
    assert fv.ifail_sh == 2


def test_m221_eng_mass(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Mass Summary Test
1 1
/ENG/MASS/1
Mass Summary Output Directive
                0.05         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_masses
    em = model.eng_masses[1]
    assert pytest.approx(em.dt_mass) == 0.05
    assert em.sens_id == 3


def test_m221_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Cylindrical Kinematic Joint Aliases Test
1 1
/LAGMUL/CYLINDRICAL/14
Cylindrical Joint Constraint
        21        22         1         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 14 in model.cyl_joints
    assert model.cyl_joints[14].node1 == 21
    assert model.cyl_joints[14].node2 == 22


def test_m221_sensor_pressure(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Pressure Sensor Test
1 1
/SENSOR/PRESSURE/44
Element Hydrostatic Pressure Sensor Trigger
       701              -100.0               500.0                0.02
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 44 in model.sensor_pressures
    assert any(s.id == 44 and s.kind == "PRESSURE" for s in model.sensors)
    sp = model.sensor_pressures[44]
    assert sp.elem_id == 701
    assert pytest.approx(sp.p_min) == -100.0
    assert pytest.approx(sp.p_max) == 500.0
    assert pytest.approx(sp.t_delay) == 0.02
