"""Tests for Milestone M213:
- Parallel and Perpendicular Kinematic Joints: /LAGMUL/PARALLEL, /PARALLEL, /LAGMUL/PERPENDICULAR, /PERPENDICULAR
- Spalling Hydrostatic Cutoff Failure Criterion: /FAIL/SPALLING_CUT, /FAIL/SPALL_CUT, /FAIL/HYDRO_CUT
- Engine Restart File Directives: /RFILE, /ENG/RFILE
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


def test_m213_parallel_and_perpendicular_joints(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Parallel and Perpendicular Joints Test
1 1
/LAGMUL/PARALLEL/601
Parallel Axes Joint
        12        22         2         3              1.0e-5
/LAGMUL/PERPENDICULAR/602
Perpendicular Axes Joint
        32        42         1         2         5         6              2.0e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 601 in model.parallel_joints
    pj = model.parallel_joints[601]
    assert pj.node1 == 12
    assert pj.node2 == 22
    assert pj.axis_dir == 2
    assert pj.skew_id == 3
    assert pytest.approx(pj.tol) == 1.0e-5

    assert 602 in model.perpendicular_joints
    perp = model.perpendicular_joints[602]
    assert perp.node1 == 32
    assert perp.node2 == 42
    assert perp.axis1_dir == 1
    assert perp.axis2_dir == 2
    assert perp.skew1_id == 5
    assert perp.skew2_id == 6
    assert pytest.approx(perp.tol) == 2.0e-5


def test_m213_fail_spalling_cut(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spalling Cutoff Failure Test
1 1
/FAIL/SPALLING_CUT/25
Hydrostatic Tension Cutoff Spalling Model
              -250.0         1                0.08                 0.85
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 25 in model.fail_spalling_cuts
    fc = model.fail_spalling_cuts[25]
    assert pytest.approx(fc.p_min) == -250.0
    assert fc.ifail_sh == 1
    assert pytest.approx(fc.eps_v_max) == 0.08
    assert pytest.approx(fc.d_max) == 0.85


def test_m213_eng_rfile(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Restart File Directives Test
1 1
/ENG/RFILE
              5.0e-3      1000         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert pytest.approx(model.rfile_dt) == 5.0e-3
    assert model.rfile_ncycle == 1000
    assert model.rfile_sens_id == 2
