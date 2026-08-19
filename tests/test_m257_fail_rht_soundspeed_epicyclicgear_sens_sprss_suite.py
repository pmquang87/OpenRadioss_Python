"""Tests for Milestone M257:
- Riedel-Hiermaier-Thoma Concrete/Rock Damage Failure Model: /FAIL/RHT, /FAIL/RHT_MODEL, /FAIL/RHT_LAW, /FAIL/RHT_DAMAGE, /FAIL/RIEDEL_HIERMAIER_THOMA
- Engine Sound Speed Output Directive: /ENG/SOUND_SPEED, /ENG_SOUND_SPEED, /ENG/C_SOUND, /ENG/C_SPEED, /SOUND_SPEED
- Epicyclic Planetary Gear Kinematic Joint Constraint: /LAGMUL/EPICYCLIC_GEAR, /EPICYCLIC_GEAR, /LAGMUL/PLANETARY_GEAR, /PLANETARY_GEAR, /LAGMUL/EPICYCLIC, /EPICYCLIC
- Spring Sound Speed Sensor: /SENSOR/SPRING_SOUND_SPEED, /SENSOR/SPRING_SS, /SENSOR/SOUND_SPEED_SPRING, /SENSOR/SPRING_C_SOUND
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


def test_m257_fail_rht(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
RHT Failure Model Test
1 1
/FAIL/RHT/155
RHT Concrete Damage Model
                0.04                 1.0               0.015                0.01         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 155 in model.fail_rhts
    frht = model.fail_rhts[155]
    assert pytest.approx(frht.d1) == 0.04
    assert pytest.approx(frht.d2) == 1.0
    assert pytest.approx(frht.p_spall) == 0.015
    assert pytest.approx(frht.eps_min) == 0.01
    assert frht.ifail_sh == 1


def test_m257_eng_sound_speed(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Sound Speed Output Directive Test
1 1
/ENG/SOUND_SPEED/1
Engine Sound Speed Output
              0.0002         6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_sound_speeds
    ess = model.eng_sound_speeds[1]
    assert pytest.approx(ess.dt_sound) == 0.0002
    assert ess.sens_id == 6


def test_m257_epicyclic_gear_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Epicyclic Gear Joint Test
1 1
/LAGMUL/EPICYCLIC_GEAR/860
Epicyclic Gear Kinematic Joint 1
        10        20        30                24.0                72.0         1         0              1.0e-5
/PLANETARY_GEAR/861
Epicyclic Gear Kinematic Joint 2
        40        50        60                20.0                80.0         2         1              2.0e-5
/EPICYCLIC/862
Epicyclic Gear Kinematic Joint 3
        70        80        90                30.0                90.0         3         2              3.0e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 860 in model.epicyclic_gear_joints
    assert model.epicyclic_gear_joints[860].node1 == 10
    assert model.epicyclic_gear_joints[860].node2 == 20
    assert model.epicyclic_gear_joints[860].node3 == 30
    assert pytest.approx(model.epicyclic_gear_joints[860].ratio_sun) == 24.0
    assert pytest.approx(model.epicyclic_gear_joints[860].ratio_ring) == 72.0
    assert model.epicyclic_gear_joints[860].axis_dir == 1
    assert model.epicyclic_gear_joints[860].skew_id == 0
    assert pytest.approx(model.epicyclic_gear_joints[860].tol) == 1.0e-5

    assert 861 in model.epicyclic_gear_joints
    assert model.epicyclic_gear_joints[861].node1 == 40
    assert model.epicyclic_gear_joints[861].node2 == 50
    assert model.epicyclic_gear_joints[861].node3 == 60
    assert pytest.approx(model.epicyclic_gear_joints[861].ratio_sun) == 20.0
    assert pytest.approx(model.epicyclic_gear_joints[861].ratio_ring) == 80.0
    assert model.epicyclic_gear_joints[861].axis_dir == 2
    assert model.epicyclic_gear_joints[861].skew_id == 1
    assert pytest.approx(model.epicyclic_gear_joints[861].tol) == 2.0e-5

    assert 862 in model.epicyclic_gear_joints
    assert model.epicyclic_gear_joints[862].node1 == 70
    assert model.epicyclic_gear_joints[862].node2 == 80
    assert model.epicyclic_gear_joints[862].node3 == 90
    assert pytest.approx(model.epicyclic_gear_joints[862].ratio_sun) == 30.0
    assert pytest.approx(model.epicyclic_gear_joints[862].ratio_ring) == 90.0
    assert model.epicyclic_gear_joints[862].axis_dir == 3
    assert model.epicyclic_gear_joints[862].skew_id == 2
    assert pytest.approx(model.epicyclic_gear_joints[862].tol) == 3.0e-5


def test_m257_sensor_spring_sound_speed(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Sound Speed Sensor Test
1 1
/SENSOR/SPRING_SOUND_SPEED/460
Spring Sound Speed Threshold Sensor
        980               3400.0               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 460 in model.sensor_spring_sound_speeds
    assert any(s.id == 460 and s.kind == "SPRING_SOUND_SPEED" for s in model.sensors)
    ssss = model.sensor_spring_sound_speeds[460]
    assert ssss.spring_id == 980
    assert pytest.approx(ssss.sound_max) == 3400.0
    assert pytest.approx(ssss.t_delay) == 0.002
