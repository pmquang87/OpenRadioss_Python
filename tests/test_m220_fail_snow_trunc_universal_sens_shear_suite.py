"""Tests for Milestone M220:
- Snow Failure Criterion: /FAIL/SNOW, /FAIL/BRITTLE_SNOW
- Engine Truncation Directive: /TRUNC, /ENG/TRUNC
- Universal Kinematic Joint Aliases: /LAGMUL/UNIVERSAL, /UNIVERSAL_JOINT
- Shear Stress Sensor: /SENSOR/SHEAR, /SENSOR/SHEAR_STRESS
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


def test_m220_fail_snow(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Snow Failure Model Test
1 1
/FAIL/SNOW/77
Snow / Ice Failure Law
                 0.5                 0.2                 1.8         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 77 in model.fail_snows
    fs = model.fail_snows[77]
    assert pytest.approx(fs.p_tens) == 0.5
    assert pytest.approx(fs.eps_comp) == 0.2
    assert pytest.approx(fs.sig_shear) == 1.8
    assert fs.ifail_sh == 2


def test_m220_eng_trunc(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Truncation Directive Test
1 1
/ENG/TRUNC/1
Engine Step Truncation Control
               1.0E-4              1.0E-7        10
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_truncs
    tr = model.eng_truncs[1]
    assert pytest.approx(tr.tol_trunc) == 1.0e-4
    assert pytest.approx(tr.dt_min) == 1.0e-7
    assert tr.n_cycle == 10


def test_m220_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Universal Kinematic Joint Aliases Test
1 1
/LAGMUL/UNIVERSAL/9
Universal Cardan Joint Constraint
        12        13
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 9 in model.cardan_joints
    assert model.cardan_joints[9].node1 == 12
    assert model.cardan_joints[9].node2 == 13


def test_m220_sensor_shear(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Shear Stress Sensor Test
1 1
/SENSOR/SHEAR/33
Element Critical Shear Threshold
       501               250.0                0.01
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 33 in model.sensor_shears
    assert any(s.id == 33 and s.kind == "SHEAR_STRESS" for s in model.sensors)
    ss = model.sensor_shears[33]
    assert ss.elem_id == 501
    assert pytest.approx(ss.tau_max) == 250.0
    assert pytest.approx(ss.t_delay) == 0.01
