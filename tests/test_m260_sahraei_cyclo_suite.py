# -*- coding: utf-8 -*-
"""Tests for Milestone M260:
- Sahraei Battery Cell / Separator Failure Model: /FAIL/SAHRAEI, /FAIL/SAHRAEI_MODEL, /FAIL/SAHRAEI_LAW, /FAIL/BATTERY_SEPARATOR
- Engine Plastic Work Output Directive: /ENG/PLASTIC_WORK, /ENG/WPLAS, /ENG/PLAS_WORK, /PLASTIC_WORK
- Cycloidal Speed Reducer Kinematic Joint Constraint: /LAGMUL/CYCLOIDAL_DRIVE, /CYCLOIDAL_DRIVE, /LAGMUL/CYCLO_DRIVE, /CYCLO_GEAR
- Spring Plastic Work Sensor: /SENSOR/SPRING_PLASTIC_WORK, /SENSOR/SPRING_WPLAS, /SENSOR/SPRING_PW, /SENSOR/WPLAS_SPRING
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
# 1. /FAIL/SAHRAEI Tests
# ============================================================================

def test_m260_fail_sahraei_fixed(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Sahraei Failure Model Test
1 1
/FAIL/SAHRAEI/180
Sahraei Battery Separator Failure Model
        12         2         3         1                0.25                    15                1.25
         3         1                0.40                0.85
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 180 in model.fail_sahraeis
    fs = model.fail_sahraeis[180]
    assert fs.fct_ratio == 12
    assert fs.num == 2
    assert fs.den == 3
    assert fs.ordi == 1
    assert pytest.approx(fs.vol_strain) == 0.25
    assert fs.fct_elsize == 15
    assert pytest.approx(fs.el_ref) == 1.25
    assert fs.comp_dir == 3
    assert fs.idel == 1
    assert pytest.approx(fs.max_comp_strain) == 0.40
    assert pytest.approx(fs.ratio) == 0.85
    assert fs.ifail_sh == 1


def test_m260_fail_sahraei_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Sahraei Defaults Test
1 1
/FAIL/SAHRAEI/181
Sahraei Defaults
         0         0         0         0                 0.0                     0                 0.0
         0         0                 0.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 181 in model.fail_sahraeis
    fs = model.fail_sahraeis[181]
    assert fs.num == 1
    assert fs.den == 1
    assert fs.ordi == 1                     # default 1
    assert pytest.approx(fs.vol_strain) == 0.0
    assert pytest.approx(fs.max_comp_strain) == 1e30  # default infinity
    assert pytest.approx(fs.ratio) == 1.0   # default 1.0


def test_m260_fail_sahraei_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Sahraei Aliases Test
1 1
/FAIL/SAHRAEI_MODEL/210
Sahraei Model Alias
         5         1         1         2                0.15                     0                 0.0
         1         0                0.30                 1.0
/FAIL/SAHRAEI_LAW/211
Sahraei Law Alias
         6         2         2         3                0.20                     0                 0.0
         2         1                0.35                 1.0
/FAIL/BATTERY_SEPARATOR/212
Battery Separator Alias
         7         3         3         4                0.18                     0                 0.0
         3         0                0.28                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 210 in model.fail_sahraeis
    assert 211 in model.fail_sahraeis
    assert 212 in model.fail_sahraeis
    assert model.fail_sahraeis[210].fct_ratio == 5
    assert model.fail_sahraeis[211].fct_ratio == 6
    assert model.fail_sahraeis[212].fct_ratio == 7


# ============================================================================
# 2. /ENG/PLASTIC_WORK Tests
# ============================================================================

def test_m260_eng_plastic_work(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Plastic Work Output Test
1 1
/ENG/PLASTIC_WORK/1
Plastic Work Output
              0.0001        10
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_plastic_works
    epw = model.eng_plastic_works[1]
    assert pytest.approx(epw.dt_wplas) == 0.0001
    assert epw.sens_id == 10


def test_m260_eng_plastic_work_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Plastic Work Aliases Test
1 1
/ENG/WPLAS/2
Wplas Alias
              0.0002         5
/ENG/PLAS_WORK/3
Plas Work Alias
              0.0003         0
/PLASTIC_WORK/4
Plastic Work Alias
              0.0004         1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_plastic_works
    assert 3 in model.eng_plastic_works
    assert 4 in model.eng_plastic_works
    assert pytest.approx(model.eng_plastic_works[2].dt_wplas) == 0.0002
    assert pytest.approx(model.eng_plastic_works[3].dt_wplas) == 0.0003
    assert pytest.approx(model.eng_plastic_works[4].dt_wplas) == 0.0004


# ============================================================================
# 3. /LAGMUL/CYCLOIDAL_DRIVE Tests
# ============================================================================

def test_m260_cycloidal_drive_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Cycloidal Drive Joint Test
1 1
/LAGMUL/CYCLOIDAL_DRIVE/880
Cycloidal Speed Reducer Joint 1
        10        20        30                29.0              1.0e+7         1              1.0e-5
                 0.0                 0.0                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 880 in model.lagmul_cycloidal_drives
    cd = model.lagmul_cycloidal_drives[880]
    assert cd.node1 == 10
    assert cd.node2 == 20
    assert cd.node3 == 30
    assert pytest.approx(cd.ratio) == 29.0
    assert pytest.approx(cd.stiff) == 1.0e+7
    assert cd.skew_id == 1
    assert pytest.approx(cd.tol) == 1.0e-5
    assert pytest.approx(cd.axis_x) == 0.0
    assert pytest.approx(cd.axis_y) == 0.0
    assert pytest.approx(cd.axis_z) == 1.0


def test_m260_cycloidal_drive_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Cycloidal Drive Aliases Test
1 1
/CYCLOIDAL_DRIVE/881
Cyclo Alias 1
         1         2         0                35.0              2.0e+6         0              1.0e-6
/LAGMUL/CYCLO_DRIVE/882
Cyclo Drive Alias 2
         3         4         5                41.0              3.0e+6         0              1.0e-6
/CYCLO_GEAR/883
Cyclo Gear Alias 3
         6         7         8                59.0              4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 881 in model.lagmul_cycloidal_drives
    assert 882 in model.lagmul_cycloidal_drives
    assert 883 in model.lagmul_cycloidal_drives
    assert pytest.approx(model.lagmul_cycloidal_drives[881].ratio) == 35.0
    assert pytest.approx(model.lagmul_cycloidal_drives[882].ratio) == 41.0
    assert pytest.approx(model.lagmul_cycloidal_drives[883].ratio) == 59.0


# ============================================================================
# 4. /SENSOR/SPRING_PLASTIC_WORK Tests
# ============================================================================

def test_m260_sensor_spring_plastic_work(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Plastic Work Sensor Test
1 1
/SENSOR/SPRING_PLASTIC_WORK/930
Spring Plastic Work Sensor 1
       200              1250.0               0.005
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 930 in model.sensor_spring_plastic_works
    sens = model.sensor_spring_plastic_works[930]
    assert sens.id == 930
    assert sens.spring_id == 200
    assert pytest.approx(sens.wplas_max) == 1250.0
    assert pytest.approx(sens.t_delay) == 0.005

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 930]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_PLASTIC_WORK"
    assert pytest.approx(matching[0].tdelay) == 0.005


def test_m260_sensor_spring_plastic_work_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Plastic Work Sensor Aliases Test
1 1
/SENSOR/SPRING_WPLAS/931
Spring Wplas Alias 1
       201               800.0               0.001
/SENSOR/SPRING_PW/932
Spring PW Alias 2
       202               950.0               0.002
/SENSOR/WPLAS_SPRING/933
Wplas Spring Alias 3
       203              1100.0               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 931 in model.sensor_spring_plastic_works
    assert 932 in model.sensor_spring_plastic_works
    assert 933 in model.sensor_spring_plastic_works
    assert pytest.approx(model.sensor_spring_plastic_works[931].wplas_max) == 800.0
    assert pytest.approx(model.sensor_spring_plastic_works[932].wplas_max) == 950.0
    assert pytest.approx(model.sensor_spring_plastic_works[933].wplas_max) == 1100.0
