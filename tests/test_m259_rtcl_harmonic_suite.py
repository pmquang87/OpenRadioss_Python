# -*- coding: utf-8 -*-
"""Tests for Milestone M259:
- Rice-Tracey & Cockcroft-Latham Combined Failure Model: /FAIL/RTCL, /FAIL/RTCL_MODEL, /FAIL/RTCL_LAW
- Engine Yield Stress Output Directive: /ENG/YIELD_STRESS, /ENG/YIELD, /ENG/SIGY, /YIELD_STRESS
- Harmonic Drive Kinematic Joint Constraint: /LAGMUL/HARMONIC_DRIVE, /HARMONIC_DRIVE, /LAGMUL/STRAIN_WAVE, /STRAIN_WAVE_GEAR
- Spring Yield Stress Sensor: /SENSOR/SPRING_YIELD_STRESS, /SENSOR/SPRING_YIELD, /SENSOR/SPRING_SIGY, /SENSOR/YIELD_SPRING
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
# 1. /FAIL/RTCL Tests
# ============================================================================

def test_m259_fail_rtcl_fixed(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
RTCL Failure Model Test
1 1
/FAIL/RTCL/155
RTCL Combined Ductile Failure Model
                0.45         1                0.25
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 155 in model.fail_rtcls
    frtcl = model.fail_rtcls[155]
    assert pytest.approx(frtcl.epscal) == 0.45
    assert frtcl.inst == 1
    assert pytest.approx(frtcl.n_exp) == 0.25
    assert frtcl.ifail_sh == 1


def test_m259_fail_rtcl_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
RTCL Default Test
1 1
/FAIL/RTCL/156
RTCL Defaults
                 0.0         0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 156 in model.fail_rtcls
    frtcl = model.fail_rtcls[156]
    assert pytest.approx(frtcl.epscal) == 0.3   # Default steel
    assert frtcl.inst == 2                      # Default shell necking flag
    assert pytest.approx(frtcl.n_exp) == 0.0


def test_m259_fail_rtcl_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
RTCL Aliases Test
1 1
/FAIL/RTCL_MODEL/201
RTCL Model Alias
                0.35         2                0.15
/FAIL/RTCL_LAW/202
RTCL Law Alias
                0.50         1                0.30
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 201 in model.fail_rtcls
    assert 202 in model.fail_rtcls
    assert pytest.approx(model.fail_rtcls[201].epscal) == 0.35
    assert pytest.approx(model.fail_rtcls[202].epscal) == 0.50


# ============================================================================
# 2. /ENG/YIELD_STRESS Tests
# ============================================================================

def test_m259_eng_yield_stress(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Yield Stress Output Directive Test
1 1
/ENG/YIELD_STRESS/1
Engine Yield Stress Output
              0.0005        12
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_yield_stresses
    eys = model.eng_yield_stresses[1]
    assert pytest.approx(eys.dt_yield) == 0.0005
    assert eys.sens_id == 12


def test_m259_eng_yield_stress_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Yield Aliases Test
1 1
/ENG/YIELD/2
Yield Alias
               0.001         5
/ENG/SIGY/3
Sigy Alias
               0.002         0
/YIELD_STRESS/4
Yield Stress Alias
               0.005         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_yield_stresses
    assert 3 in model.eng_yield_stresses
    assert 4 in model.eng_yield_stresses
    assert pytest.approx(model.eng_yield_stresses[2].dt_yield) == 0.001
    assert pytest.approx(model.eng_yield_stresses[3].dt_yield) == 0.002
    assert pytest.approx(model.eng_yield_stresses[4].dt_yield) == 0.005


# ============================================================================
# 3. /LAGMUL/HARMONIC_DRIVE Tests
# ============================================================================

def test_m259_harmonic_drive_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Harmonic Drive Joint Test
1 1
/LAGMUL/HARMONIC_DRIVE/870
Harmonic Drive Joint 1
        10        20        30               160.0              5.0e+7         2              1.0e-5
                 0.0                 0.0                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 870 in model.lagmul_harmonic_drives
    hd = model.lagmul_harmonic_drives[870]
    assert hd.node1 == 10
    assert hd.node2 == 20
    assert hd.node3 == 30
    assert pytest.approx(hd.ratio) == 160.0
    assert pytest.approx(hd.stiff) == 5.0e+7
    assert hd.skew_id == 2
    assert pytest.approx(hd.tol) == 1.0e-5
    assert pytest.approx(hd.axis_x) == 0.0
    assert pytest.approx(hd.axis_y) == 0.0
    assert pytest.approx(hd.axis_z) == 1.0


def test_m259_harmonic_drive_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Harmonic Drive Aliases Test
1 1
/HARMONIC_DRIVE/871
HD Alias 1
         1         2         0               100.0              1.0e+6         0              1.0e-6
/LAGMUL/STRAIN_WAVE/872
Strain Wave Alias 2
         3         4         5                80.0              2.0e+6         0              1.0e-6
/STRAIN_WAVE_GEAR/873
Strain Wave Gear Alias 3
         6         7         8               120.0              3.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 871 in model.lagmul_harmonic_drives
    assert 872 in model.lagmul_harmonic_drives
    assert 873 in model.lagmul_harmonic_drives
    assert pytest.approx(model.lagmul_harmonic_drives[871].ratio) == 100.0
    assert pytest.approx(model.lagmul_harmonic_drives[872].ratio) == 80.0
    assert pytest.approx(model.lagmul_harmonic_drives[873].ratio) == 120.0


# ============================================================================
# 4. /SENSOR/SPRING_YIELD_STRESS Tests
# ============================================================================

def test_m259_sensor_spring_yield_stress(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Yield Stress Sensor Test
1 1
/SENSOR/SPRING_YIELD_STRESS/920
Spring Yield Sensor 1
       100               450.0               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 920 in model.sensor_spring_yield_stresses
    sens = model.sensor_spring_yield_stresses[920]
    assert sens.id == 920
    assert sens.spring_id == 100
    assert pytest.approx(sens.sigy_max) == 450.0
    assert pytest.approx(sens.t_delay) == 0.002

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 920]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_YIELD_STRESS"
    assert pytest.approx(matching[0].tdelay) == 0.002


def test_m259_sensor_spring_yield_stress_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Yield Stress Sensor Aliases Test
1 1
/SENSOR/SPRING_YIELD/921
Spring Yield Alias 1
       101               300.0               0.001
/SENSOR/SPRING_SIGY/922
Spring Sigy Alias 2
       102               350.0               0.003
/SENSOR/YIELD_SPRING/923
Yield Spring Alias 3
       103               400.0               0.004
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 921 in model.sensor_spring_yield_stresses
    assert 922 in model.sensor_spring_yield_stresses
    assert 923 in model.sensor_spring_yield_stresses
    assert pytest.approx(model.sensor_spring_yield_stresses[921].sigy_max) == 300.0
    assert pytest.approx(model.sensor_spring_yield_stresses[922].sigy_max) == 350.0
    assert pytest.approx(model.sensor_spring_yield_stresses[923].sigy_max) == 400.0
