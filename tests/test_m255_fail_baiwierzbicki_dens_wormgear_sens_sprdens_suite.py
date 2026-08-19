"""Tests for Milestone M255:
- Bai-Wierzbicki Asymmetric Fracture Model: /FAIL/BAI_WIERZBICKI, /FAIL/BAI_WIERZBICKI_MODEL, /FAIL/BAI_WIERZBICKI_LAW, /FAIL/BAI_WIERZBICKI_DAMAGE, /FAIL/BW, /FAIL/BAI
- Engine Density Output Directive: /ENG/DENSITY, /ENG_DENSITY, /ENG/RHO, /ENG_RHO, /ENG/DENS, /DENSITY
- Worm Gear Kinematic Joint Constraint: /LAGMUL/WORM_GEAR, /WORM_GEAR, /LAGMUL/WORM_GEAR_JOINT, /WORM_GEAR_JOINT, /LAGMUL/WORM, /WORM
- Spring Density Sensor: /SENSOR/SPRING_DENSITY, /SENSOR/SPRING_DENS, /SENSOR/DENSITY_SPRING, /SENSOR/SPRING_RHO
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


def test_m255_fail_bai_wierzbicki(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Bai-Wierzbicki Failure Model Test
1 1
/FAIL/BAI_WIERZBICKI/140
Bai-Wierzbicki Asymmetric Damage Law
                0.25                1.25                0.15                0.85                 0.3                0.75         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 140 in model.fail_bai_wierzbickis
    fbw = model.fail_bai_wierzbickis[140]
    assert pytest.approx(fbw.c1) == 0.25
    assert pytest.approx(fbw.c2) == 1.25
    assert pytest.approx(fbw.c3) == 0.15
    assert pytest.approx(fbw.c4) == 0.85
    assert pytest.approx(fbw.c_theta) == 0.3
    assert pytest.approx(fbw.eps_max) == 0.75
    assert fbw.ifail_sh == 1


def test_m255_eng_density(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Density Output Directive Test
1 1
/ENG/DENSITY/1
Engine Part and Element Density History Tracking
              0.0002         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 1 in model.eng_densities
    ed = model.eng_densities[1]
    assert pytest.approx(ed.dt_dens) == 0.0002
    assert ed.sens_id == 5


def test_m255_worm_gear_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Worm Gear Joint Test
1 1
/LAGMUL/WORM_GEAR/840
Worm Gear Kinematic Joint 1
        16        26                20.0         1         2              1.0e-5
/WORM_GEAR_JOINT/841
Worm Gear Kinematic Joint 2
        36        46                30.0         0         0              2.0e-5
/WORM/842
Worm Gear Kinematic Joint 3
        56        66                40.0         3         4              3.0e-5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 840 in model.worm_gear_joints
    assert model.worm_gear_joints[840].node1 == 16
    assert model.worm_gear_joints[840].node2 == 26
    assert pytest.approx(model.worm_gear_joints[840].ratio) == 20.0
    assert model.worm_gear_joints[840].skew1_id == 1
    assert model.worm_gear_joints[840].skew2_id == 2
    assert pytest.approx(model.worm_gear_joints[840].tol) == 1.0e-5

    assert 841 in model.worm_gear_joints
    assert model.worm_gear_joints[841].node1 == 36
    assert model.worm_gear_joints[841].node2 == 46
    assert pytest.approx(model.worm_gear_joints[841].ratio) == 30.0
    assert model.worm_gear_joints[841].skew1_id == 0
    assert model.worm_gear_joints[841].skew2_id == 0
    assert pytest.approx(model.worm_gear_joints[841].tol) == 2.0e-5

    assert 842 in model.worm_gear_joints
    assert model.worm_gear_joints[842].node1 == 56
    assert model.worm_gear_joints[842].node2 == 66
    assert pytest.approx(model.worm_gear_joints[842].ratio) == 40.0
    assert model.worm_gear_joints[842].skew1_id == 3
    assert model.worm_gear_joints[842].skew2_id == 4
    assert pytest.approx(model.worm_gear_joints[842].tol) == 3.0e-5


def test_m255_sensor_spring_density(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Density Sensor Test
1 1
/SENSOR/SPRING_DENSITY/440
Spring Material Density Threshold Sensor
        960               7850.0               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert 440 in model.sensor_spring_densities
    assert any(s.id == 440 and s.kind == "SPRING_DENSITY" for s in model.sensors)
    ssd = model.sensor_spring_densities[440]
    assert ssd.spring_id == 960
    assert pytest.approx(ssd.dens_max) == 7850.0
    assert pytest.approx(ssd.t_delay) == 0.002
