"""Tests for Milestone M210:
- Planar and Cardan Kinematic Joints: /LAGMUL/PLANAR, /PLANAR, /LAGMUL/CARDAN, /CARDAN
- Tabular Failure Criterion: /FAIL/TBID, /FAIL/TABLE
- Engine Animation Directives: /ANIM/DT, /ENG/ANIM/DT
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


def test_m210_planar_and_cardan_joints(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Planar and Cardan Joints Test
1 1
/LAGMUL/PLANAR/301
Planar Kinematic Joint
        15        25         3         0              1.5e-5
/LAGMUL/CARDAN/302
Cardan Universal Joint
        35        45         1         4              2.5e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 301 in model.planar_joints
    pj = model.planar_joints[301]
    assert pj.node1 == 15
    assert pj.node2 == 25
    assert pj.axis_dir == 3
    assert pj.skew_id == 0
    assert pytest.approx(pj.tol) == 1.5e-5

    assert 302 in model.cardan_joints
    cj = model.cardan_joints[302]
    assert cj.node1 == 35
    assert cj.node2 == 45
    assert cj.axis_dir == 1
    assert cj.skew_id == 4
    assert pytest.approx(cj.tol) == 2.5e-5


def test_m210_fail_tbid(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Tabular Failure Test
1 1
/FAIL/TBID/10
Tabular Failure Criterion
        42         2                0.01                0.95                0.05
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 10 in model.fail_tbids
    ft = model.fail_tbids[10]
    assert ft.fct_id == 42
    assert ft.ifail_sh == 2
    assert pytest.approx(ft.eps_dot_0) == 0.01
    assert pytest.approx(ft.d_max) == 0.95
    assert pytest.approx(ft.f_smooth) == 0.05


def test_m210_anim_dt(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Animation Output Test
1 1
/ENG/ANIM/DT
                0.01                0.05         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert pytest.approx(model.anim_tstart) == 0.01
    assert pytest.approx(model.anim_dt) == 0.05
    assert model.anim_sens_id == 3
