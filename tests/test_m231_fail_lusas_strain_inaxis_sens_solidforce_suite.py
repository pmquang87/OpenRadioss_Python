"""Tests for Milestone M231:
- Lusas Composite Failure Criterion: /FAIL/LUSAS, /FAIL/COMPOSITE_LUSAS
- Engine Strain Output Directive: /STRAIN, /ENG/STRAIN
- Inline/Parallel Axis Joint Aliases: /LAGMUL/INLINE_AXIS, /LAGMUL/PARALLEL_AXIS
- Solid Force Sensor: /SENSOR/SOLID_FORCE, /SENSOR/FORCE_SOLID
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


def test_m231_fail_lusas(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Lusas Composite Failure Test
1 1
/FAIL/LUSAS/88
Lusas 3D Composite Failure Model
              1200.0               900.0                80.0               150.0
                60.0                50.0                70.0         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 88 in model.fail_lusases
    fl = model.fail_lusases[88]
    assert pytest.approx(fl.xt) == 1200.0
    assert pytest.approx(fl.xc) == 900.0
    assert pytest.approx(fl.yt) == 80.0
    assert pytest.approx(fl.yc) == 150.0
    assert pytest.approx(fl.s12) == 60.0
    assert pytest.approx(fl.s23) == 50.0
    assert pytest.approx(fl.s31) == 70.0
    assert fl.ifail_sh == 2


def test_m231_eng_strain(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Strain Output Directive Test
1 1
/ENG/STRAIN/1
Strain History Output Control
                0.02         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_strains
    es = model.eng_strains[1]
    assert pytest.approx(es.dt_strain) == 0.02
    assert es.sens_id == 3


def test_m231_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Inline Axis and Parallel Axis Joint Aliases Test
1 1
/LAGMUL/INLINE_AXIS/61
Inline Axis Joint Constraint
        15        25         1         0             1.0e-05
/LAGMUL/PARALLEL_AXIS/62
Parallel Axis Joint Constraint
        35        45         2         0             2.0e-05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 61 in model.inline_joints
    assert model.inline_joints[61].node1 == 15
    assert model.inline_joints[61].node2 == 25
    assert pytest.approx(model.inline_joints[61].tol) == 1.0e-5

    assert 62 in model.parallel_joints
    assert model.parallel_joints[62].node1 == 35
    assert model.parallel_joints[62].node2 == 45
    assert pytest.approx(model.parallel_joints[62].tol) == 2.0e-5


def test_m231_sensor_solid_force(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Solid Force Sensor Test
1 1
/SENSOR/SOLID_FORCE/131
Solid Force Threshold Sensor
       400              2500.0               0.005
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 131 in model.sensor_solid_forces
    assert any(s.id == 131 and s.kind == "SOLID_FORCE" for s in model.sensors)
    ssf = model.sensor_solid_forces[131]
    assert ssf.solid_id == 400
    assert pytest.approx(ssf.f_max) == 2500.0
    assert pytest.approx(ssf.t_delay) == 0.005
