"""Tests for Milestone M229:
- Norton Creep Rupture Failure Criterion: /FAIL/NORTON, /FAIL/CREEP
- Engine Tensor Tracking Directive: /TENS, /ENG/TENS
- Cylinder/Slider Axis Joint Aliases: /LAGMUL/CYLINDER_AXIS, /LAGMUL/SLIDER_AXIS
- Truss Strain Sensor: /SENSOR/TRUSS_STRAIN, /SENSOR/STRAIN_TRUSS
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


def test_m229_fail_norton(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Norton Creep Rupture Failure Test
1 1
/FAIL/NORTON/88
Norton Bailey Creep Failure
               1.2e-5                 4.5                 0.2                0.25
                100.0         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 88 in model.fail_nortons
    fn = model.fail_nortons[88]
    assert pytest.approx(fn.A) == 1.2e-5
    assert pytest.approx(fn.n) == 4.5
    assert pytest.approx(fn.m) == 0.2
    assert pytest.approx(fn.eps_rupt) == 0.25
    assert pytest.approx(fn.t_rupt) == 100.0
    assert fn.ifail_sh == 2


def test_m229_eng_tens(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Tensor Output Directive Test
1 1
/ENG/TENS/1
Tensor History Output Control
                 0.02         4
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_tenses
    et = model.eng_tenses[1]
    assert pytest.approx(et.dt_tens) == 0.02
    assert et.sens_id == 4


def test_m229_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Cylinder Axis and Slider Axis Joint Aliases Test
1 1
/LAGMUL/CYLINDER_AXIS/41
Cylinder Axis Joint Constraint
        15        25         2
/LAGMUL/SLIDER_AXIS/42
Slider Axis Joint Constraint
        35        45         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 41 in model.cyl_joints
    assert model.cyl_joints[41].node1 == 15
    assert model.cyl_joints[41].node2 == 25

    assert 42 in model.slider_joints
    assert model.slider_joints[42].node1 == 35
    assert model.slider_joints[42].node2 == 45


def test_m229_sensor_truss_strain(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Truss Strain Sensor Test
1 1
/SENSOR/TRUSS_STRAIN/125
Truss Strain Threshold Sensor
       200                 0.12               0.005
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 125 in model.sensor_truss_strains
    assert any(s.id == 125 and s.kind == "TRUSS_STRAIN" for s in model.sensors)
    sts = model.sensor_truss_strains[125]
    assert sts.truss_id == 200
    assert pytest.approx(sts.eps_max) == 0.12
    assert pytest.approx(sts.t_delay) == 0.005
