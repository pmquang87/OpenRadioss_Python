"""Tests for Milestone M208:
- Ball/Pin Kinematic Joints: /LAGMUL/BALL_JOINT, /BALL_JOINT, /LAGMUL/PIN_JOINT, /PIN_JOINT
- Layered Composite Thick Shell Property: /PROP/TYPE54, /PROP/TSH_P54
- Global Engine Rayleigh Damping: /ENG/DAMP, /DAMP
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


def test_m208_ball_and_pin_joints(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Ball and Pin Joints Test
1 1
/LAGMUL/BALL_JOINT/101
Spherical Ball Joint
        10        20              1.0e-5
/LAGMUL/PIN_JOINT/102
Revolute Pin Joint
        30        40         3         2              2.5e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 101 in model.ball_joints
    bj = model.ball_joints[101]
    assert bj.node1 == 10
    assert bj.node2 == 20
    assert pytest.approx(bj.tol) == 1.0e-5

    assert 102 in model.pin_joints
    pj = model.pin_joints[102]
    assert pj.node1 == 30
    assert pj.node2 == 40
    assert pj.axis_dir == 3
    assert pj.skew_id == 2
    assert pytest.approx(pj.tol) == 2.5e-5


def test_m208_prop_type54_layered_thick_shell(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Type 54 Thick Shell Test
1 1
/PROP/TYPE54/301
Composite Layered Thick Shell
        15         0         0         3         3         3         1                 0.0
                 1.2                0.06
                 0.0                 0.0                 0.0         0         0         0
                0.83
                45.0                 0.5                -0.5         1
               -45.0                 0.5                 0.5         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 301 in model.props_type54
    p54 = model.props_type54[301]
    assert p54.isolid == 15
    assert p54.inpts_r == 3
    assert p54.inpts_s == 3
    assert p54.inpts_t == 3
    assert p54.iint == 1
    assert pytest.approx(p54.qa) == 1.2
    assert pytest.approx(p54.qb) == 0.06
    assert pytest.approx(p54.ashear) == 0.83
    assert len(p54.layers) == 2
    assert pytest.approx(p54.layers[0].phi) == 45.0
    assert pytest.approx(p54.layers[0].thick) == 0.5
    assert pytest.approx(p54.layers[0].zi) == -0.5
    assert p54.layers[0].mat_id == 1
    assert pytest.approx(p54.layers[1].phi) == -45.0
    assert pytest.approx(p54.layers[1].thick) == 0.5
    assert pytest.approx(p54.layers[1].zi) == 0.5
    assert p54.layers[1].mat_id == 2
    assert model.props_tsh_p54 is model.props_type54


def test_m208_eng_damp(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Damping Test
1 1
/ENG/DAMP
                0.05                0.01                0.02                0.50
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert pytest.approx(model.damp_alpha) == 0.05
    assert pytest.approx(model.damp_beta) == 0.01
    assert pytest.approx(model.damp_tstart) == 0.02
    assert pytest.approx(model.damp_tstop) == 0.50
