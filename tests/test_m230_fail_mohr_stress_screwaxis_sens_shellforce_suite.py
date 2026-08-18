"""Tests for Milestone M230:
- Mohr-Coulomb Failure Criterion: /FAIL/MOHR, /FAIL/MOHR_COULOMB
- Engine Stress Output Directive: /STRESS, /ENG/STRESS
- Screw/Helical Axis Joint Aliases: /LAGMUL/SCREW_AXIS, /LAGMUL/HELICAL_AXIS
- Shell Force Sensor: /SENSOR/SHELL_FORCE, /SENSOR/FORCE_SHELL
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


def test_m230_fail_mohr(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Mohr-Coulomb Failure Test
1 1
/FAIL/MOHR/77
Mohr-Coulomb Concrete Failure Model
                50.0                35.0                15.0         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 77 in model.fail_mohrs
    fm = model.fail_mohrs[77]
    assert pytest.approx(fm.c) == 50.0
    assert pytest.approx(fm.phi) == 35.0
    assert pytest.approx(fm.sigma_t) == 15.0
    assert fm.ifail_sh == 2


def test_m230_eng_stress(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Stress Output Directive Test
1 1
/ENG/STRESS/1
Stress History Output Control
                 0.05         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_stresses
    es = model.eng_stresses[1]
    assert pytest.approx(es.dt_stress) == 0.05
    assert es.sens_id == 2


def test_m230_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Screw Axis and Helical Axis Joint Aliases Test
1 1
/LAGMUL/SCREW_AXIS/51
Screw Axis Joint Constraint
        12        22         1         0                 1.5
/LAGMUL/HELICAL_AXIS/52
Helical Axis Joint Constraint
        32        42         2         0                 2.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 51 in model.screw_joints
    assert model.screw_joints[51].node1 == 12
    assert model.screw_joints[51].node2 == 22
    assert pytest.approx(model.screw_joints[51].pitch) == 1.5

    assert 52 in model.screw_joints
    assert model.screw_joints[52].node1 == 32
    assert model.screw_joints[52].node2 == 42
    assert pytest.approx(model.screw_joints[52].pitch) == 2.0


def test_m230_sensor_shell_force(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Shell Force Sensor Test
1 1
/SENSOR/SHELL_FORCE/130
Shell Force Threshold Sensor
       300               500.0               120.0               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 130 in model.sensor_shell_forces
    assert any(s.id == 130 and s.kind == "SHELL_FORCE" for s in model.sensors)
    ssf = model.sensor_shell_forces[130]
    assert ssf.shell_id == 300
    assert pytest.approx(ssf.f_max) == 500.0
    assert pytest.approx(ssf.m_max) == 120.0
    assert pytest.approx(ssf.t_delay) == 0.002
