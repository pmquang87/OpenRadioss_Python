"""Tests for Milestone M211:
- Rigid Link and Screw Kinematic Joints: /LAGMUL/RIGID, /RIGID_JOINT, /LAGMUL/SCREW, /SCREW
- S-N Fatigue Failure Criterion: /FAIL/SN_CURVE, /FAIL/SNCURVE, /FAIL/WOHLER
- Engine Run Directives: /RUN, /ENG/RUN
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


def test_m211_rigid_and_screw_joints(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Rigid Link and Screw Joints Test
1 1
/LAGMUL/RIGID/401
Rigid Link Kinematic Joint
        10        20              1.0e-5
/LAGMUL/SCREW/402
Helical Screw Joint
        30        40         2         5                 2.5              2.0e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 401 in model.rigid_joints
    rj = model.rigid_joints[401]
    assert rj.node1 == 10
    assert rj.node2 == 20
    assert pytest.approx(rj.tol) == 1.0e-5

    assert 402 in model.screw_joints
    sj = model.screw_joints[402]
    assert sj.node1 == 30
    assert sj.node2 == 40
    assert sj.axis_dir == 2
    assert sj.skew_id == 5
    assert pytest.approx(sj.pitch) == 2.5
    assert pytest.approx(sj.tol) == 2.0e-5


def test_m211_fail_sn_curve(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
SN Fatigue Failure Test
1 1
/FAIL/SN_CURVE/15
SN Curve Fatigue Model
        12         1         2                 0.8             1.0e8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 15 in model.fail_sn_curves
    sn = model.fail_sn_curves[15]
    assert sn.fct_id == 12
    assert sn.ifail_sh == 1
    assert sn.s_mean_corr == 2
    assert pytest.approx(sn.d_crit) == 0.8
    assert pytest.approx(sn.n_cutoff) == 1.0e8


def test_m211_eng_run(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Run Directive Test
1 1
/ENG/RUN
Simulation Run Title
                0.25      100000
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert model.run_title == "Simulation Run Title"
    assert pytest.approx(model.run_tstop) == 0.25
    assert model.run_cycle_max == 100000
