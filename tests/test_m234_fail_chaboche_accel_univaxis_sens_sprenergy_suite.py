"""Tests for Milestone M234:
- Lemaitre-Chaboche Failure Criterion: /FAIL/CHABOCHE, /FAIL/LEMAITRE_CHABOCHE
- Engine Acceleration Output Directive: /ACCEL, /ENG/ACCEL
- Universal/Cardan Axis Joint Aliases: /LAGMUL/UNIVERSAL_AXIS, /LAGMUL/CARDAN_AXIS
- Spring Energy Sensor: /SENSOR/SPRING_ENERGY, /SENSOR/ENERGY_SPRING
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


def test_m234_fail_chaboche(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Lemaitre-Chaboche Failure Test
1 1
/FAIL/CHABOCHE/91
Chaboche Ductile Damage Model
                15.0                 2.5                 1.5
                0.85                0.45         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 91 in model.fail_chaboches
    fc = model.fail_chaboches[91]
    assert pytest.approx(fc.s_0) == 15.0
    assert pytest.approx(fc.s_1) == 2.5
    assert pytest.approx(fc.beta) == 1.5
    assert pytest.approx(fc.d_crit) == 0.85
    assert pytest.approx(fc.eps_crit) == 0.45
    assert fc.ifail_sh == 2


def test_m234_eng_accel(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Acceleration Output Directive Test
1 1
/ENG/ACCEL/1
Acceleration Vector History Output Control
               0.001         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_accels
    ea = model.eng_accels[1]
    assert pytest.approx(ea.dt_acc) == 0.001
    assert ea.sens_id == 2


def test_m234_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Universal and Cardan Axis Joint Aliases Test
1 1
/LAGMUL/UNIVERSAL_AXIS/85
Universal Axis Joint Constraint
        12        22         1         2             1.0e-05
/LAGMUL/CARDAN_AXIS/86
Cardan Axis Joint Constraint
        32        42         2         3             2.0e-05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 85 in model.cardan_joints
    assert model.cardan_joints[85].node1 == 12
    assert model.cardan_joints[85].node2 == 22
    assert model.cardan_joints[85].axis_dir == 1
    assert model.cardan_joints[85].skew_id == 2
    assert pytest.approx(model.cardan_joints[85].tol) == 1.0e-5

    assert 86 in model.cardan_joints
    assert model.cardan_joints[86].node1 == 32
    assert model.cardan_joints[86].node2 == 42
    assert model.cardan_joints[86].axis_dir == 2
    assert model.cardan_joints[86].skew_id == 3
    assert pytest.approx(model.cardan_joints[86].tol) == 2.0e-5



def test_m234_sensor_spring_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Internal Energy Sensor Test
1 1
/SENSOR/SPRING_ENERGY/134
Spring Internal Energy Threshold Sensor
       750               500.0               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 134 in model.sensor_spring_energies
    assert any(s.id == 134 and s.kind == "SPRING_ENERGY" for s in model.sensors)
    sse = model.sensor_spring_energies[134]
    assert sse.spring_id == 750
    assert pytest.approx(sse.e_max) == 500.0
    assert pytest.approx(sse.t_delay) == 0.002
