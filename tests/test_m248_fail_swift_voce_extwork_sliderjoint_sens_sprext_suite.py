"""Tests for Milestone M248:
- Swift-Voce Combined Failure Model: /FAIL/SWIFT_VOCE, /FAIL/SWIFT_VOCE_LAW, /FAIL/SWIFT_VOCE_DAMAGE, /FAIL/SWIFT_VOCE_MODEL, /FAIL/SV
- Engine External Work Output Directive: /EXT_WORK, /ENG/EXT_WORK, /ENG/EXTERNAL_WORK, /ENG/EXTWORK, /ENG/WEXT
- Prismatic Slider Axis Joint Aliases: /LAGMUL/PRISMATIC_SLIDER_AXIS, /PRISMATIC_SLIDER_AXIS, /LAGMUL/SLIDER_JOINT_AXIS, /SLIDER_JOINT_AXIS, /LAGMUL/TRANSLATIONAL_SLIDER_AXIS, /TRANSLATIONAL_SLIDER_AXIS
- Spring External Work Sensor: /SENSOR/SPRING_EXT_WORK, /SENSOR/SPRING_EXTERNAL_WORK, /SENSOR/EXT_WORK_SPRING, /SENSOR/SPRING_WEXT
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


def test_m248_fail_swift_voce(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Swift-Voce Combined Failure Model Test
1 1
/FAIL/SWIFT_VOCE/101
Swift-Voce Combined Hardening Failure Model
                 0.6               600.0               0.008                0.25
               350.0               800.0                18.0                0.45         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 101 in model.fail_swift_voces
    fsv = model.fail_swift_voces[101]
    assert pytest.approx(fsv.alpha) == 0.6
    assert pytest.approx(fsv.k_coeff) == 600.0
    assert pytest.approx(fsv.eps0) == 0.008
    assert pytest.approx(fsv.n_exp) == 0.25
    assert pytest.approx(fsv.sigma0) == 350.0
    assert pytest.approx(fsv.sigma_inf) == 800.0
    assert pytest.approx(fsv.beta) == 18.0
    assert pytest.approx(fsv.eps_max) == 0.45
    assert fsv.ifail_sh == 1


def test_m248_eng_ext_work(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine External Work Output Directive Test
1 1
/ENG/EXT_WORK/1
External Work History Output Control
             0.00015         4
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_ext_works
    eew = model.eng_ext_works[1]
    assert pytest.approx(eew.dt_wext) == 0.00015
    assert eew.sens_id == 4


def test_m248_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Prismatic Slider Axis Joint Aliases Test
1 1
/LAGMUL/PRISMATIC_SLIDER_AXIS/501
Prismatic Slider Axis Joint Constraint
        14        24         2         0              0.0003
/SLIDER_JOINT_AXIS/502
Slider Joint Axis Constraint
        34        44         3         0              0.0004
/TRANSLATIONAL_SLIDER_AXIS/503
Translational Slider Axis Constraint
        54        64         1         0              0.0005
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 501 in model.slider_joints
    assert model.slider_joints[501].node1 == 14
    assert model.slider_joints[501].node2 == 24
    assert model.slider_joints[501].axis_dir == 2
    assert pytest.approx(model.slider_joints[501].tol) == 0.0003

    assert 502 in model.slider_joints
    assert model.slider_joints[502].node1 == 34
    assert model.slider_joints[502].node2 == 44
    assert model.slider_joints[502].axis_dir == 3
    assert pytest.approx(model.slider_joints[502].tol) == 0.0004

    assert 503 in model.slider_joints
    assert model.slider_joints[503].node1 == 54
    assert model.slider_joints[503].node2 == 64
    assert model.slider_joints[503].axis_dir == 1
    assert pytest.approx(model.slider_joints[503].tol) == 0.0005


def test_m248_sensor_spring_ext_work(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring External Work Sensor Test
1 1
/SENSOR/SPRING_EXT_WORK/201
Spring Element External Work Threshold Sensor
        910               550.0               0.015
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 201 in model.sensor_spring_ext_works
    assert any(s.id == 201 and s.kind == "SPRING_EXT_WORK" for s in model.sensors)
    ssew = model.sensor_spring_ext_works[201]
    assert ssew.spring_id == 910
    assert pytest.approx(ssew.ewext_max) == 550.0
    assert pytest.approx(ssew.t_delay) == 0.015
