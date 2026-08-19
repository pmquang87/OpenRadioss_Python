# -*- coding: utf-8 -*-
"""Tests for Milestone M262:
- Puck Composite Failure Model: /FAIL/PUCK, /FAIL/PUCK_MODEL, /FAIL/PUCK_LAW, /FAIL/PUCK_CRITERION
- Engine Stress Triaxiality Output Directive: /ENG/STRESS_TRI, /ENG/TRIAXIALITY, /ENG/ETA, /ENG/STRESS_TRIAXIALITY
- Screw / Leadscrew Joint Constraint: /LAGMUL/SCREW_JOINT, /SCREW_JOINT, /LAGMUL/LEADSCREW, /LEADSCREW, /SCREW
- Spring Force Impulse Sensor: /SENSOR/SPRING_FORCE_IMPULSE, /SENSOR/SPRING_IMPULSE, /SENSOR/SPRING_J, /SENSOR/IMPULSE_SPRING
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


# ============================================================================
# 1. /FAIL/PUCK Tests
# ============================================================================

def test_m262_fail_puck_fixed(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Puck Failure Model Fixed Format Test
2022 0
/FAIL/PUCK/195
Puck Carbon Fiber Failure Model
             1200.00               50.00               80.00              800.00              150.00
                0.35                0.30                0.25               0.001         1         1
              5000.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 195 in model.fail_pucks
    fp = model.fail_pucks[195]
    assert pytest.approx(fp.sigma_1t) == 1200.00
    assert pytest.approx(fp.sigma_2t) == 50.00
    assert pytest.approx(fp.sigma_12) == 80.00
    assert pytest.approx(fp.sigma_1c) == 800.00
    assert pytest.approx(fp.sigma_2c) == 150.00
    assert pytest.approx(fp.p12_pos) == 0.35
    assert pytest.approx(fp.p12_neg) == 0.30
    assert pytest.approx(fp.p22_neg) == 0.25
    assert pytest.approx(fp.tau_max) == 0.001
    assert fp.ifail_sh == 1
    assert fp.ifail_so == 1
    assert pytest.approx(fp.fcut) == 5000.0


def test_m262_fail_puck_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Puck Failure Model Defaults Test
2022 0
/FAIL/PUCK/196
Puck Defaults Model
/END
"""
    # Missing card error check
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m262_fail_puck_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Puck Aliases Test
2022 0
/FAIL/PUCK_MODEL/230
Puck Model Alias
             1000.00               40.00               70.00              700.00              120.00
/FAIL/PUCK_LAW/231
Puck Law Alias
             1100.00               45.00               75.00              750.00              130.00
/FAIL/PUCK_CRITERION/232
Puck Criterion Alias
             1150.00               48.00               78.00              780.00              140.00
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 230 in model.fail_pucks
    assert 231 in model.fail_pucks
    assert 232 in model.fail_pucks
    assert pytest.approx(model.fail_pucks[230].sigma_1t) == 1000.00
    assert pytest.approx(model.fail_pucks[231].sigma_1t) == 1100.00
    assert pytest.approx(model.fail_pucks[232].sigma_1t) == 1150.00


# ============================================================================
# 2. /ENG/STRESS_TRI Tests
# ============================================================================

def test_m262_eng_stress_tri(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Stress Triaxiality Output Test
1 1
/ENG/STRESS_TRI/1
Stress Triaxiality Field Output
              0.0002        12
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_stress_tris
    est = model.eng_stress_tris[1]
    assert pytest.approx(est.dt_triax) == 0.0002
    assert est.sens_id == 12


def test_m262_eng_stress_tri_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Stress Triaxiality Aliases Test
1 1
/ENG/TRIAXIALITY/2
Triaxiality Alias
              0.0001         0
/ENG/ETA/3
Eta Alias
              0.0003         6
/ENG/STRESS_TRIAXIALITY/4
Stress Triaxiality Alias
              0.0004         9
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_stress_tris
    assert 3 in model.eng_stress_tris
    assert 4 in model.eng_stress_tris
    assert pytest.approx(model.eng_stress_tris[2].dt_triax) == 0.0001
    assert pytest.approx(model.eng_stress_tris[3].dt_triax) == 0.0003
    assert pytest.approx(model.eng_stress_tris[4].dt_triax) == 0.0004


# ============================================================================
# 3. /LAGMUL/SCREW_JOINT Tests
# ============================================================================

def test_m262_screw_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Screw Joint Test
1 1
/LAGMUL/SCREW_JOINT/780
Screw Linear Actuator Mechanism
        15        25                 5.0              2.0e+6         1              1.0e-6
                 0.0                 0.0                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 780 in model.lagmul_screw_joints
    sj = model.lagmul_screw_joints[780]
    assert sj.node1 == 15
    assert sj.node2 == 25
    assert pytest.approx(sj.lead_pitch) == 5.0
    assert pytest.approx(sj.stiff) == 2.0e+6
    assert sj.skew_id == 1
    assert pytest.approx(sj.tol) == 1.0e-6
    assert pytest.approx(sj.axis_x) == 0.0
    assert pytest.approx(sj.axis_y) == 0.0
    assert pytest.approx(sj.axis_z) == 1.0


def test_m262_screw_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Screw Joint Aliases Test
1 1
/SCREW_JOINT/781
SJ Alias 1
         1         2                 4.0              1.0e+6         0              1.0e-6
/LAGMUL/LEADSCREW/782
SJ Alias 2
         3         4                 6.0              1.5e+6         0              1.0e-6
/LEADSCREW/783
SJ Alias 3
         5         6                 8.0              2.5e+6         0              1.0e-6
/SCREW/784
SJ Alias 4
         7         8                10.0              3.5e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 781 in model.lagmul_screw_joints
    assert 782 in model.lagmul_screw_joints
    assert 783 in model.lagmul_screw_joints
    assert 784 in model.lagmul_screw_joints
    assert pytest.approx(model.lagmul_screw_joints[781].lead_pitch) == 4.0
    assert pytest.approx(model.lagmul_screw_joints[782].lead_pitch) == 6.0
    assert pytest.approx(model.lagmul_screw_joints[783].lead_pitch) == 8.0
    assert pytest.approx(model.lagmul_screw_joints[784].lead_pitch) == 10.0


# ============================================================================
# 4. /SENSOR/SPRING_FORCE_IMPULSE Tests
# ============================================================================

def test_m262_sensor_spring_force_impulse(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Force Impulse Sensor Test
1 1
/SENSOR/SPRING_FORCE_IMPULSE/950
Spring Force Impulse Sensor 1
       350              1.2e+4               0.005
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 950 in model.sensor_spring_force_impulses
    sens = model.sensor_spring_force_impulses[950]
    assert sens.id == 950
    assert sens.spring_id == 350
    assert pytest.approx(sens.j_max) == 1.2e+4
    assert pytest.approx(sens.t_delay) == 0.005

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 950]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_FORCE_IMPULSE"
    assert pytest.approx(matching[0].tdelay) == 0.005


def test_m262_sensor_spring_force_impulse_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Force Impulse Sensor Aliases Test
1 1
/SENSOR/SPRING_IMPULSE/951
Spring Impulse Alias 1
       351              2.2e+4               0.001
/SENSOR/SPRING_J/952
Spring J Alias 2
       352              3.2e+4               0.002
/SENSOR/IMPULSE_SPRING/953
Impulse Spring Alias 3
       353              4.2e+4               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 951 in model.sensor_spring_force_impulses
    assert 952 in model.sensor_spring_force_impulses
    assert 953 in model.sensor_spring_force_impulses
    assert pytest.approx(model.sensor_spring_force_impulses[951].j_max) == 2.2e+4
    assert pytest.approx(model.sensor_spring_force_impulses[952].j_max) == 3.2e+4
    assert pytest.approx(model.sensor_spring_force_impulses[953].j_max) == 4.2e+4
