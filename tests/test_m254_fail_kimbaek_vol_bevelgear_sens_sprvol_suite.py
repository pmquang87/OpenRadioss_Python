"""Tests for Milestone M254:
- Kim-Baek Ductile Fracture Model: /FAIL/KIM_BAEK, /FAIL/KIM_BAEK_MODEL, /FAIL/KIM_BAEK_LAW, /FAIL/KIM_BAEK_DAMAGE, /FAIL/KB, /FAIL/KIM
- Engine Volume Output Directive: /ENG/VOLUME, /ENG_VOLUME, /ENG/VOL, /ENG_VOL, /ENG/VOLUME_CHANGE, /VOLUME
- Bevel Gear Kinematic Joint Constraint: /LAGMUL/BEVEL_GEAR, /BEVEL_GEAR, /LAGMUL/BEVEL_GEAR_JOINT, /BEVEL_GEAR_JOINT, /LAGMUL/BEVEL, /BEVEL
- Spring Volume Sensor: /SENSOR/SPRING_VOLUME, /SENSOR/SPRING_VOL, /SENSOR/VOLUME_SPRING, /SENSOR/SPRING_VOL_CHANGE
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


def test_m254_fail_kim_baek(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Kim-Baek Failure Model Test
1 1
/FAIL/KIM_BAEK/130
Kim-Baek Ductile Damage Law
               250.0                 0.5                0.25               0.015                 1.0                0.65         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 130 in model.fail_kim_baeks
    fkb = model.fail_kim_baeks[130]
    assert pytest.approx(fkb.sigma0) == 250.0
    assert pytest.approx(fkb.k_coeff) == 0.5
    assert pytest.approx(fkb.n_exp) == 0.25
    assert pytest.approx(fkb.c_rate) == 0.015
    assert pytest.approx(fkb.eps0_dot) == 1.0
    assert pytest.approx(fkb.eps_max) == 0.65
    assert fkb.ifail_sh == 1


def test_m254_eng_volume(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Volume Output Directive Test
1 1
/ENG/VOLUME/1
Engine Part and Element Volume History Tracking
              0.0001         3
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_volumes
    ev = model.eng_volumes[1]
    assert pytest.approx(ev.dt_vol) == 0.0001
    assert ev.sens_id == 3


def test_m254_bevel_gear_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Bevel Gear Joint Test
1 1
/LAGMUL/BEVEL_GEAR/830
Bevel Gear Kinematic Joint 1
        15        25                 2.5         1         2              1.0e-5
/BEVEL_GEAR_JOINT/831
Bevel Gear Kinematic Joint 2
        35        45                 1.75         0         0              2.0e-5
/BEVEL/832
Bevel Gear Kinematic Joint 3
        55        65                 0.5         3         4              3.0e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 830 in model.bevel_gear_joints
    assert model.bevel_gear_joints[830].node1 == 15
    assert model.bevel_gear_joints[830].node2 == 25
    assert pytest.approx(model.bevel_gear_joints[830].ratio) == 2.5
    assert model.bevel_gear_joints[830].skew1_id == 1
    assert model.bevel_gear_joints[830].skew2_id == 2
    assert pytest.approx(model.bevel_gear_joints[830].tol) == 1.0e-5

    assert 831 in model.bevel_gear_joints
    assert model.bevel_gear_joints[831].node1 == 35
    assert model.bevel_gear_joints[831].node2 == 45
    assert pytest.approx(model.bevel_gear_joints[831].ratio) == 1.75
    assert model.bevel_gear_joints[831].skew1_id == 0
    assert model.bevel_gear_joints[831].skew2_id == 0
    assert pytest.approx(model.bevel_gear_joints[831].tol) == 2.0e-5

    assert 832 in model.bevel_gear_joints
    assert model.bevel_gear_joints[832].node1 == 55
    assert model.bevel_gear_joints[832].node2 == 65
    assert pytest.approx(model.bevel_gear_joints[832].ratio) == 0.5
    assert model.bevel_gear_joints[832].skew1_id == 3
    assert model.bevel_gear_joints[832].skew2_id == 4
    assert pytest.approx(model.bevel_gear_joints[832].tol) == 3.0e-5


def test_m254_sensor_spring_volume(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Volume Sensor Test
1 1
/SENSOR/SPRING_VOLUME/430
Spring Volumetric Deformation Threshold Sensor
        950               1250.0              0.0015
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 430 in model.sensor_spring_volumes
    assert any(s.id == 430 and s.kind == "SPRING_VOLUME" for s in model.sensors)
    ssv = model.sensor_spring_volumes[430]
    assert ssv.spring_id == 950
    assert pytest.approx(ssv.vol_max) == 1250.0
    assert pytest.approx(ssv.t_delay) == 0.0015
