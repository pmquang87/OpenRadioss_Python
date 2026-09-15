"""Tests for Milestone M253:
- Hockett-Sherby Saturation Hardening Failure Model: /FAIL/HOCKETT_SHERBY, /FAIL/HOCKETT_SHERBY_MODEL, /FAIL/HOCKETT_SHERBY_LAW, /FAIL/HOCKETT_SHERBY_DAMAGE, /FAIL/HS, /FAIL/HOCKETT
- Engine Temperature Output Directive: /ENG/TEMPERATURE, /ENG_TEMPERATURE, /ENG/TEMP, /ENG_TEMP, /ENG/THERMAL_TEMPERATURE, /TEMPERATURE
- Tripod Plunging CV Kinematic Joint Constraint: /LAGMUL/TRIPOD, /TRIPOD, /LAGMUL/TRIPOD_JOINT, /TRIPOD_JOINT, /LAGMUL/PLUNGING_CV, /PLUNGING_CV
- Spring Temperature Sensor: /SENSOR/SPRING_TEMPERATURE, /SENSOR/SPRING_TEMP, /SENSOR/TEMPERATURE_SPRING, /SENSOR/SPRING_THERMAL_TEMP
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


def test_m253_fail_hockett_sherby(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Hockett-Sherby Failure Model Test
1 1
/FAIL/HOCKETT_SHERBY/120
Hockett-Sherby Saturation Hardening Law
               180.0               420.0                 3.5                 0.8                0.45         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 120 in model.fail_hockett_sherbys
    fhs = model.fail_hockett_sherbys[120]
    assert pytest.approx(fhs.sigma0) == 180.0
    assert pytest.approx(fhs.sigma_s) == 420.0
    assert pytest.approx(fhs.m_exp) == 3.5
    assert pytest.approx(fhs.n_exp) == 0.8
    assert pytest.approx(fhs.eps_max) == 0.45
    assert fhs.ifail_sh == 1


def test_m253_eng_temperature(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Temperature Output Directive Test
1 1
/ENG/TEMPERATURE/1
Engine Thermal Temperature Output Control
              0.0005         2
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_temperatures
    et = model.eng_temperatures[1]
    assert pytest.approx(et.dt_temp) == 0.0005
    assert et.sens_id == 2


def test_m253_tripod_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Tripod Plunging CV Joint Test
1 1
/LAGMUL/TRIPOD/820
Tripod Joint Constraint 1
        10        20         1         0                15.0              1.0e-5
/TRIPOD_JOINT/821
Tripod Joint Constraint 2
        30        40         2         1                25.0              2.0e-5
/PLUNGING_CV/822
Tripod Plunging CV Joint Constraint 3
        50        60         3         2                35.0              3.0e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 820 in model.tripod_joints
    assert model.tripod_joints[820].node1 == 10
    assert model.tripod_joints[820].node2 == 20
    assert model.tripod_joints[820].axis_dir == 1
    assert model.tripod_joints[820].skew_id == 0
    assert pytest.approx(model.tripod_joints[820].plunge_limit) == 15.0
    assert pytest.approx(model.tripod_joints[820].tol) == 1.0e-5

    assert 821 in model.tripod_joints
    assert model.tripod_joints[821].node1 == 30
    assert model.tripod_joints[821].node2 == 40
    assert model.tripod_joints[821].axis_dir == 2
    assert model.tripod_joints[821].skew_id == 1
    assert pytest.approx(model.tripod_joints[821].plunge_limit) == 25.0
    assert pytest.approx(model.tripod_joints[821].tol) == 2.0e-5

    assert 822 in model.tripod_joints
    assert model.tripod_joints[822].node1 == 50
    assert model.tripod_joints[822].node2 == 60
    assert model.tripod_joints[822].axis_dir == 3
    assert model.tripod_joints[822].skew_id == 2
    assert pytest.approx(model.tripod_joints[822].plunge_limit) == 35.0
    assert pytest.approx(model.tripod_joints[822].tol) == 3.0e-5


def test_m253_sensor_spring_temperature(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Temperature Sensor Test
1 1
/SENSOR/SPRING_TEMPERATURE/420
Spring Thermal Temperature Threshold Sensor
        940               350.0               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 420 in model.sensor_spring_temperatures
    assert any(s.id == 420 and s.kind == "SPRING_TEMPERATURE" for s in model.sensors)
    sst = model.sensor_spring_temperatures[420]
    assert sst.spring_id == 940
    assert pytest.approx(sst.temp_max) == 350.0
    assert pytest.approx(sst.t_delay) == 0.002
