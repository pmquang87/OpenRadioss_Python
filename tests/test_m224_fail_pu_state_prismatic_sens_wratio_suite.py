"""Tests for Milestone M224:
- Polyurethane Foam Failure Criterion: /FAIL/PU, /FAIL/POLYURETHANE
- Engine State Variable Tracking Directive: /STATE, /ENG/STATE
- Prismatic Kinematic Joint Aliases: /LAGMUL/PRISMATIC, /PRISMATIC_JOINT
- Work Ratio Sensor: /SENSOR/WORK_RATIO, /SENSOR/WRATIO
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


def test_m224_fail_pu(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Polyurethane Foam Failure Test
1 1
/FAIL/PU/55
Polyurethane Failure Criterion
                0.25               -0.45                12.5               -25.0         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 55 in model.fail_pus
    fpu = model.fail_pus[55]
    assert pytest.approx(fpu.eps_t) == 0.25
    assert pytest.approx(fpu.eps_c) == -0.45
    assert pytest.approx(fpu.sigma_t) == 12.5
    assert pytest.approx(fpu.sigma_c) == -25.0
    assert fpu.ifail_sh == 1


def test_m224_eng_state(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine State Variable Tracking Test
1 1
/ENG/STATE/1
State Variable Directive
               0.005         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_states
    es = model.eng_states[1]
    assert pytest.approx(es.dt_state) == 0.005
    assert es.sens_id == 3


def test_m224_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Prismatic Kinematic Joint Aliases Test
1 1
/LAGMUL/PRISMATIC/33
Prismatic Joint Constraint
        17        18         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 33 in model.slider_joints
    assert model.slider_joints[33].node1 == 17
    assert model.slider_joints[33].node2 == 18


def test_m224_sensor_work_ratio(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Work Ratio Sensor Test
1 1
/SENSOR/WORK_RATIO/77
Work Ratio Trigger Sensor
                 2.5               0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 77 in model.sensor_work_ratios
    assert any(s.id == 77 and s.kind == "WORK_RATIO" for s in model.sensors)
    swr = model.sensor_work_ratios[77]
    assert pytest.approx(swr.w_ratio_max) == 2.5
    assert pytest.approx(swr.t_delay) == 0.001
