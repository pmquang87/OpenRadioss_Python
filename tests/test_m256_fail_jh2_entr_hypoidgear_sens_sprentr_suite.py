"""Tests for Milestone M256:
- Johnson-Holmquist Ceramic/Brittle Failure Model: /FAIL/JH2, /FAIL/JH2_MODEL, /FAIL/JH2_LAW, /FAIL/JH2_DAMAGE, /FAIL/JOHNSON_HOLMQUIST, /FAIL/JH
- Engine Entropy Output Directive: /ENG/ENTROPY, /ENG_ENTROPY, /ENG/THERMAL_ENTROPY, /ENTROPY
- Hypoid Gear Kinematic Joint Constraint: /LAGMUL/HYPOID_GEAR, /HYPOID_GEAR, /LAGMUL/HYPOID_GEAR_JOINT, /HYPOID_GEAR_JOINT, /LAGMUL/HYPOID, /HYPOID
- Spring Entropy Sensor: /SENSOR/SPRING_ENTROPY, /SENSOR/SPRING_ENTR, /SENSOR/ENTROPY_SPRING, /SENSOR/SPRING_THERMAL_ENTROPY
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


def test_m256_fail_jh2(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Johnson-Holmquist Failure Model Test
1 1
/FAIL/JH2/150
Johnson-Holmquist Ceramic Damage Model
               0.045                 1.0               0.007                0.25                 1.0         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 150 in model.fail_jh2s
    fjh = model.fail_jh2s[150]
    assert pytest.approx(fjh.d1) == 0.045
    assert pytest.approx(fjh.d2) == 1.0
    assert pytest.approx(fjh.c_rate) == 0.007
    assert pytest.approx(fjh.t_star) == 0.25
    assert pytest.approx(fjh.eps0_dot) == 1.0
    assert fjh.ifail_sh == 1


def test_m256_eng_entropy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Entropy Output Directive Test
1 1
/ENG/ENTROPY/1
Engine Entropy and Dissipation Output
              0.0005         8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_entropies
    ee = model.eng_entropies[1]
    assert pytest.approx(ee.dt_entr) == 0.0005
    assert ee.sens_id == 8


def test_m256_hypoid_gear_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Hypoid Gear Joint Test
1 1
/LAGMUL/HYPOID_GEAR/850
Hypoid Gear Kinematic Joint 1
        17        27                 3.73                25.0         1         2              1.0e-5
/HYPOID_GEAR_JOINT/851
Hypoid Gear Kinematic Joint 2
        37        47                 4.10                30.0         0         0              2.0e-5
/HYPOID/852
Hypoid Gear Kinematic Joint 3
        57        67                 5.25                35.0         3         4              3.0e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 850 in model.hypoid_gear_joints
    assert model.hypoid_gear_joints[850].node1 == 17
    assert model.hypoid_gear_joints[850].node2 == 27
    assert pytest.approx(model.hypoid_gear_joints[850].ratio) == 3.73
    assert pytest.approx(model.hypoid_gear_joints[850].offset) == 25.0
    assert model.hypoid_gear_joints[850].skew1_id == 1
    assert model.hypoid_gear_joints[850].skew2_id == 2
    assert pytest.approx(model.hypoid_gear_joints[850].tol) == 1.0e-5

    assert 851 in model.hypoid_gear_joints
    assert model.hypoid_gear_joints[851].node1 == 37
    assert model.hypoid_gear_joints[851].node2 == 47
    assert pytest.approx(model.hypoid_gear_joints[851].ratio) == 4.10
    assert pytest.approx(model.hypoid_gear_joints[851].offset) == 30.0
    assert model.hypoid_gear_joints[851].skew1_id == 0
    assert model.hypoid_gear_joints[851].skew2_id == 0
    assert pytest.approx(model.hypoid_gear_joints[851].tol) == 2.0e-5

    assert 852 in model.hypoid_gear_joints
    assert model.hypoid_gear_joints[852].node1 == 57
    assert model.hypoid_gear_joints[852].node2 == 67
    assert pytest.approx(model.hypoid_gear_joints[852].ratio) == 5.25
    assert pytest.approx(model.hypoid_gear_joints[852].offset) == 35.0
    assert model.hypoid_gear_joints[852].skew1_id == 3
    assert model.hypoid_gear_joints[852].skew2_id == 4
    assert pytest.approx(model.hypoid_gear_joints[852].tol) == 3.0e-5


def test_m256_sensor_spring_entropy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Entropy Sensor Test
1 1
/SENSOR/SPRING_ENTROPY/450
Spring Entropy Dissipation Threshold Sensor
        970                500.0               0.001
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 450 in model.sensor_spring_entropies
    assert any(s.id == 450 and s.kind == "SPRING_ENTROPY" for s in model.sensors)
    sse = model.sensor_spring_entropies[450]
    assert sse.spring_id == 970
    assert pytest.approx(sse.entr_max) == 500.0
    assert pytest.approx(sse.t_delay) == 0.001
