"""Tests for Milestone M212:
- Constant Velocity and In-Line Kinematic Joints: /LAGMUL/CV_JOINT, /CV_JOINT, /CONSTANT_VELOCITY, /LAGMUL/INLINE, /INLINE
- Hoop Stress Failure Criterion: /FAIL/HOOP, /FAIL/HOOP_STRESS
- Engine Stop and Time-History Directives: /STOP, /ENG/STOP, /TFILE, /ENG/TFILE
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


def test_m212_cv_and_inline_joints(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
CV and In-Line Joints Test
1 1
/LAGMUL/CV_JOINT/501
Constant Velocity Homokinetic Joint
        15        25         3         4              1.5e-5
/LAGMUL/INLINE/502
In-Line Translational Joint
        35        45         1         0              2.5e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 501 in model.cv_joints
    cv = model.cv_joints[501]
    assert cv.node1 == 15
    assert cv.node2 == 25
    assert cv.axis_dir == 3
    assert cv.skew_id == 4
    assert pytest.approx(cv.tol) == 1.5e-5

    assert 502 in model.inline_joints
    inj = model.inline_joints[502]
    assert inj.node1 == 35
    assert inj.node2 == 45
    assert inj.axis_dir == 1
    assert inj.skew_id == 0
    assert pytest.approx(inj.tol) == 2.5e-5


def test_m212_fail_hoop(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Hoop Stress Failure Test
1 1
/FAIL/HOOP/18
Pipe Hoop Stress Failure Model
               450.0         2                0.15                 0.9
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 18 in model.fail_hoops
    fh = model.fail_hoops[18]
    assert pytest.approx(fh.sigma_hoop_max) == 450.0
    assert fh.ifail_sh == 2
    assert pytest.approx(fh.eps_p_max) == 0.15
    assert pytest.approx(fh.d_max) == 0.9


def test_m212_eng_stop_and_tfile(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Stop and TFILE Directives Test
1 1
/ENG/STOP
        10     50000                0.05
/ENG/TFILE
              1.0e-4         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert model.stop_sens_id == 10
    assert model.stop_cycle_max == 50000
    assert pytest.approx(model.stop_time_max) == 0.05
    assert pytest.approx(model.tfile_dt) == 1.0e-4
    assert model.tfile_sens_id == 3
