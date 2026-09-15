# -*- coding: utf-8 -*-
"""Tests for Milestone M264:
- Johnson-Cook Failure Model: /FAIL/JOHNSON_COOK, /FAIL/JOHNSON_COOK_MODEL, /FAIL/JOHNSON_COOK_LAW, /FAIL/JC, /FAIL/JC_DAMAGE
- Engine Maximum Shear Stress Output Directive: /ENG/MAX_SHEAR, /ENG/TMAX, /ENG/MAX_SHEAR_STRESS, /ENG/TAUMAX
- Transfer Case Joint Constraint: /LAGMUL/TRANSFER_CASE, /TRANSFER_CASE, /LAGMUL/TCASE, /TCASE, /TRANSFER_GEAR
- Spring Moment Impulse Sensor: /SENSOR/SPRING_MOMENT_IMPULSE, /SENSOR/SPRING_MOM_IMPULSE, /SENSOR/SPRING_ANGULAR_IMPULSE, /SENSOR/ANGULAR_IMPULSE_SPRING
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
# 1. /FAIL/JOHNSON_COOK Tests
# ============================================================================

def test_m264_fail_johnson_cook_fixed(tmp_path: Path):
    c1 = f"{0.05:>20.4f}{3.44:>20.4f}{-2.12:>20.4f}{0.002:>20.4f}{0.61:>20.4f}"
    c2 = f"{1.0:>20.2f}{293.15:>20.2f}{1800.0:>20.2f}{1.0:>20.2f}{2:>10d}{1.0:>10.2f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Johnson Cook Failure Model Fixed Format Test
2022 0
/FAIL/JOHNSON_COOK/205
Johnson Cook Ductile Damage Model
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 205 in model.fail_johnson_cooks
    fjc = model.fail_johnson_cooks[205]
    assert pytest.approx(fjc.d1) == 0.05
    assert pytest.approx(fjc.d2) == 3.44
    assert pytest.approx(fjc.d3) == -2.12
    assert pytest.approx(fjc.d4) == 0.002
    assert pytest.approx(fjc.d5) == 0.61
    assert pytest.approx(fjc.eps_dot_0) == 1.0
    assert pytest.approx(fjc.t_room) == 293.15
    assert pytest.approx(fjc.t_melt) == 1800.0
    assert pytest.approx(fjc.m_exp) == 1.0
    assert fjc.ifail_sh == 2
    assert pytest.approx(fjc.d_max) == 1.0


def test_m264_fail_johnson_cook_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Johnson Cook Defaults Test
2022 0
/FAIL/JOHNSON_COOK/206
Johnson Cook Default Values
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m264_fail_johnson_cook_aliases(tmp_path: Path):
    c = f"{0.1:>20.4f}{0.2:>20.4f}{0.3:>20.4f}{0.001:>20.4f}{0.5:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Johnson Cook Aliases Test
2022 0
/FAIL/JOHNSON_COOK_MODEL/250
JC Model Alias
{c}
/FAIL/JOHNSON_COOK_LAW/251
JC Law Alias
{c}
/FAIL/JC/252
JC Short Alias
{c}
/FAIL/JC_DAMAGE/253
JC Damage Alias
{c}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 250 in model.fail_johnson_cooks
    assert 251 in model.fail_johnson_cooks
    assert 252 in model.fail_johnson_cooks
    assert 253 in model.fail_johnson_cooks
    assert pytest.approx(model.fail_johnson_cooks[250].d1) == 0.1
    assert pytest.approx(model.fail_johnson_cooks[251].d1) == 0.1
    assert pytest.approx(model.fail_johnson_cooks[252].d1) == 0.1
    assert pytest.approx(model.fail_johnson_cooks[253].d1) == 0.1


# ============================================================================
# 2. /ENG/MAX_SHEAR Tests
# ============================================================================

def test_m264_eng_max_shear(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Max Shear Output Test
1 1
/ENG/MAX_SHEAR/1
Max Shear Field Output
              0.0004         7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_max_shears
    ems = model.eng_max_shears[1]
    assert pytest.approx(ems.dt_tmax) == 0.0004
    assert ems.sens_id == 7


def test_m264_eng_max_shear_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Max Shear Aliases Test
1 1
/ENG/TMAX/2
Tmax Alias
              0.0001         0
/ENG/MAX_SHEAR_STRESS/3
Max Shear Stress Alias
              0.0002         4
/ENG/TAUMAX/4
Taumax Alias
              0.0003         6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_max_shears
    assert 3 in model.eng_max_shears
    assert 4 in model.eng_max_shears
    assert pytest.approx(model.eng_max_shears[2].dt_tmax) == 0.0001
    assert pytest.approx(model.eng_max_shears[3].dt_tmax) == 0.0002
    assert pytest.approx(model.eng_max_shears[4].dt_tmax) == 0.0003


# ============================================================================
# 3. /LAGMUL/TRANSFER_CASE Tests
# ============================================================================

def test_m264_transfer_case(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Transfer Case Joint Test
1 1
/LAGMUL/TRANSFER_CASE/795
AWD Center Differential Transfer Case
        10        20        30                 0.4              8.0e+6         1              1.0e-6
                 0.0                 1.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 795 in model.lagmul_transfer_cases
    tc = model.lagmul_transfer_cases[795]
    assert tc.node1 == 10
    assert tc.node2 == 20
    assert tc.node3 == 30
    assert pytest.approx(tc.front_split) == 0.4
    assert pytest.approx(tc.stiff) == 8.0e+6
    assert tc.skew_id == 1
    assert pytest.approx(tc.tol) == 1.0e-6
    assert pytest.approx(tc.axis_x) == 0.0
    assert pytest.approx(tc.axis_y) == 1.0
    assert pytest.approx(tc.axis_z) == 0.0


def test_m264_transfer_case_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Transfer Case Aliases Test
1 1
/TRANSFER_CASE/796
TCase Alias 1
         1         2         3                 0.5              1.0e+6         0              1.0e-6
/LAGMUL/TCASE/797
TCase Alias 2
         4         5         6                 0.35             2.0e+6         0              1.0e-6
/TCASE/798
TCase Alias 3
         7         8         9                 0.45             3.0e+6         0              1.0e-6
/TRANSFER_GEAR/799
TCase Alias 4
        11        12        13                 0.55             4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 796 in model.lagmul_transfer_cases
    assert 797 in model.lagmul_transfer_cases
    assert 798 in model.lagmul_transfer_cases
    assert 799 in model.lagmul_transfer_cases
    assert pytest.approx(model.lagmul_transfer_cases[796].front_split) == 0.5
    assert pytest.approx(model.lagmul_transfer_cases[797].front_split) == 0.35
    assert pytest.approx(model.lagmul_transfer_cases[798].front_split) == 0.45
    assert pytest.approx(model.lagmul_transfer_cases[799].front_split) == 0.55


# ============================================================================
# 4. /SENSOR/SPRING_MOMENT_IMPULSE Tests
# ============================================================================

def test_m264_sensor_spring_moment_impulse(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Moment Impulse Sensor Test
1 1
/SENSOR/SPRING_MOMENT_IMPULSE/970
Spring Moment Impulse Sensor 1
       450              8.5e+4               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 970 in model.sensor_spring_moment_impulses
    sens = model.sensor_spring_moment_impulses[970]
    assert sens.id == 970
    assert sens.spring_id == 450
    assert pytest.approx(sens.h_max) == 8.5e+4
    assert pytest.approx(sens.t_delay) == 0.003

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 970]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_MOMENT_IMPULSE"
    assert pytest.approx(matching[0].tdelay) == 0.003


def test_m264_sensor_spring_moment_impulse_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Moment Impulse Sensor Aliases Test
1 1
/SENSOR/SPRING_MOM_IMPULSE/971
Spring Mom Impulse Alias 1
       451              1.5e+4               0.001
/SENSOR/SPRING_ANGULAR_IMPULSE/972
Spring Angular Impulse Alias 2
       452              2.5e+4               0.002
/SENSOR/ANGULAR_IMPULSE_SPRING/973
Angular Impulse Spring Alias 3
       453              3.5e+4               0.004
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 971 in model.sensor_spring_moment_impulses
    assert 972 in model.sensor_spring_moment_impulses
    assert 973 in model.sensor_spring_moment_impulses
    assert pytest.approx(model.sensor_spring_moment_impulses[971].h_max) == 1.5e+4
    assert pytest.approx(model.sensor_spring_moment_impulses[972].h_max) == 2.5e+4
    assert pytest.approx(model.sensor_spring_moment_impulses[973].h_max) == 3.5e+4
