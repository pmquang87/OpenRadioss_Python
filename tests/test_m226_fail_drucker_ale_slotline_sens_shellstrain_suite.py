"""Tests for Milestone M226:
- Drucker-Prager Failure Criterion: /FAIL/DRUCKER, /FAIL/DRUCKER_PRAGER
- Engine ALE Grid Smoothing Directive: /ALE, /ENG/ALE
- Slot Line Joint Aliases: /LAGMUL/SLOT_LINE, /SLOT_LINE_JOINT
- Shell Strain Sensor: /SENSOR/SHELL_STRAIN, /SENSOR/STRAIN_SHELL
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


def test_m226_fail_drucker(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Drucker-Prager Failure Test
1 1
/FAIL/DRUCKER/77
Drucker-Prager Failure Criterion
                 0.25                35.0                80.0         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 77 in model.fail_druckers
    fd = model.fail_druckers[77]
    assert pytest.approx(fd.alpha) == 0.25
    assert pytest.approx(fd.k) == 35.0
    assert pytest.approx(fd.sigma_t) == 80.0
    assert fd.ifail_sh == 1


def test_m226_eng_ale(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine ALE Smoothing Directive Test
1 1
/ENG/ALE/1
ALE Smoothing Control
               0.005         8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_ales
    ea = model.eng_ales[1]
    assert pytest.approx(ea.dt_ale) == 0.005
    assert ea.sens_id == 8


def test_m226_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Slot Line Joint Aliases Test
1 1
/LAGMUL/SLOT_LINE/18
Slot Line Joint Constraint
        10        20         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 18 in model.slot_joints
    assert model.slot_joints[18].node1 == 10
    assert model.slot_joints[18].node2 == 20


def test_m226_sensor_shell_strain(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Shell Plastic Strain Sensor Test
1 1
/SENSOR/SHELL_STRAIN/105
Shell Strain Threshold Sensor
       120                 0.18         3               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 105 in model.sensor_shell_strains
    assert any(s.id == 105 and s.kind == "SHELL_STRAIN" for s in model.sensors)
    sss = model.sensor_shell_strains[105]
    assert sss.shell_id == 120
    assert pytest.approx(sss.eps_max) == 0.18
    assert sss.ip == 3
    assert pytest.approx(sss.t_delay) == 0.002
