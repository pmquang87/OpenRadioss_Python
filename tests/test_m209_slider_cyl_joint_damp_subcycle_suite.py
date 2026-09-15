"""Tests for Milestone M209:
- Prismatic Slider and Cylindrical Kinematic Joints: /LAGMUL/SLIDER, /SLIDER, /LAGMUL/CYL_JOINT, /CYL_JOINT
- Part-Level Rayleigh Damping: /DAMP/PART
- Engine Sub-cycling Directives: /ENG/SUB_CYCLE, /SUB_CYCLE
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


def test_m209_slider_and_cyl_joints(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Slider and Cylindrical Joints Test
1 1
/LAGMUL/SLIDER/201
Prismatic Slider Joint
        10        20         1         0              1.0e-5
/LAGMUL/CYL_JOINT/202
Cylindrical Joint
        30        40         2         5              2.0e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 201 in model.slider_joints
    sj = model.slider_joints[201]
    assert sj.node1 == 10
    assert sj.node2 == 20
    assert sj.axis_dir == 1
    assert sj.skew_id == 0
    assert pytest.approx(sj.tol) == 1.0e-5

    assert 202 in model.cyl_joints
    cj = model.cyl_joints[202]
    assert cj.node1 == 30
    assert cj.node2 == 40
    assert cj.axis_dir == 2
    assert cj.skew_id == 5
    assert pytest.approx(cj.tol) == 2.0e-5


def test_m209_damp_part(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Part Damping Test
1 1
/DAMP/PART/501
Part Specific Damping
         4                0.05                0.01                0.02                0.80
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 501 in model.damp_parts
    dp = model.damp_parts[501]
    assert dp.part_id == 4
    assert pytest.approx(dp.alpha) == 0.05
    assert pytest.approx(dp.beta) == 0.01
    assert pytest.approx(dp.tstart) == 0.02
    assert pytest.approx(dp.tstop) == 0.80


def test_m209_eng_sub_cycle(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Sub-cycling Test
1 1
/ENG/SUB_CYCLE
         4                 0.0         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert model.sub_cycle_enabled is True
    assert model.sub_cycle_ratio == 4
    assert model.sub_cycle_inter is True
