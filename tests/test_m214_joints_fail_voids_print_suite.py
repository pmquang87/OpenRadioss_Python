"""Tests for Milestone M214:
- Gimbal and Constant Distance Kinematic Joints: /LAGMUL/GIMBAL, /GIMBAL, /LAGMUL/DISTANCE, /DISTANCE
- Void Coalescence Porosity Failure Criterion: /FAIL/VOIDS, /FAIL/VOID, /FAIL/POROSITY
- Engine Print Directives: /PRINT, /ENG/PRINT
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


def test_m214_gimbal_and_distance_joints(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Gimbal and Distance Joints Test
1 1
/LAGMUL/GIMBAL/701
Gimbal Universal Joint
        15        25         1         3         2         4              1.5e-5
/LAGMUL/DISTANCE/702
Constant Distance Joint
        35        45                125.0              2.5e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 701 in model.gimbal_joints
    gj = model.gimbal_joints[701]
    assert gj.node1 == 15
    assert gj.node2 == 25
    assert gj.axis1_dir == 1
    assert gj.axis2_dir == 3
    assert gj.skew1_id == 2
    assert gj.skew2_id == 4
    assert pytest.approx(gj.tol) == 1.5e-5

    assert 702 in model.distance_joints
    dj = model.distance_joints[702]
    assert dj.node1 == 35
    assert dj.node2 == 45
    assert pytest.approx(dj.dist) == 125.0
    assert pytest.approx(dj.tol) == 2.5e-5


def test_m214_fail_voids(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Void Coalescence Porosity Failure Test
1 1
/FAIL/VOIDS/33
Gurson Porosity Failure Model
                0.02                0.18         2                1.45                0.95                0.90
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 33 in model.fail_voids
    fv = model.fail_voids[33]
    assert pytest.approx(fv.f_0) == 0.02
    assert pytest.approx(fv.f_c) == 0.18
    assert fv.ifail_sh == 2
    assert pytest.approx(fv.q1) == 1.45
    assert pytest.approx(fv.q2) == 0.95
    assert pytest.approx(fv.d_max) == 0.90


def test_m214_eng_print(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Print Directives Test
1 1
/ENG/PRINT
       200              1.0e-3         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert model.print_ncycle == 200
    assert pytest.approx(model.print_dt) == 1.0e-3
    assert model.print_sens_id == 3
