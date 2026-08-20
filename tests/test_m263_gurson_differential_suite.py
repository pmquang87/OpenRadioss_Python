# -*- coding: utf-8 -*-
"""Tests for Milestone M263:
- Gurson-Tvergaard-Needleman Porous Plasticity Failure Model: /FAIL/GURSON, /FAIL/GURSON_MODEL, /FAIL/GURSON_LAW, /FAIL/GTN, /FAIL/GURSON_TVERGAARD_NEEDLEMAN
- Engine Lode Angle Output Directive: /ENG/LODE_ANGLE, /ENG/LODE, /ENG/LODE_PARAM, /ENG/LODE_ANGLE_PARAM
- Differential Gear Joint Constraint: /LAGMUL/DIFFERENTIAL_GEAR, /DIFFERENTIAL_GEAR, /LAGMUL/DIFF_GEAR, /DIFF_GEAR, /LAGMUL/DIFFERENTIAL
- Spring Moment-Rate Sensor: /SENSOR/SPRING_MOMENT_RATE, /SENSOR/SPRING_DM, /SENSOR/SPRING_MOMENTRATE, /SENSOR/DM_SPRING
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
# 1. /FAIL/GURSON Tests
# ============================================================================

def test_m263_fail_gurson_fixed(tmp_path: Path):
    card1 = f"{1.45:>20.2f}{0.95:>20.2f}{'':50s}{2:>10d}"
    card2 = f"{0.20:>20.2f}{0.05:>20.2f}{0.10:>20.2f}"
    card3 = f"{0.12:>20.2f}{0.22:>20.2f}{0.01:>20.2f}"
    card4 = f"{1.50:>20.2f}{5000.0:>20.1f}{0.25:>20.2f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Gurson GTN Failure Model Fixed Format Test
2022 0
/FAIL/GURSON/200
Gurson Porous Plasticity Model
{card1}
{card2}
{card3}
{card4}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 200 in model.fail_gursons
    fg = model.fail_gursons[200]
    assert pytest.approx(fg.q1) == 1.45
    assert pytest.approx(fg.q2) == 0.95
    assert fg.i_loc == 2
    assert pytest.approx(fg.eps_n) == 0.20
    assert pytest.approx(fg.a_s) == 0.05
    assert pytest.approx(fg.k_w) == 0.10
    assert pytest.approx(fg.f_c) == 0.12
    assert pytest.approx(fg.f_r) == 0.22
    assert pytest.approx(fg.f_0) == 0.01
    assert pytest.approx(fg.r_len) == 1.50
    assert pytest.approx(fg.h_chi) == 5000.0
    assert pytest.approx(fg.le_max) == 0.25


def test_m263_fail_gurson_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Gurson Defaults Test
2022 0
/FAIL/GURSON/201
Gurson Default Values
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m263_fail_gurson_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Gurson Aliases Test
2022 0
/FAIL/GURSON_MODEL/240
Gurson Model Alias
                1.50                1.00                                                                     1
/FAIL/GURSON_LAW/241
Gurson Law Alias
                1.55                1.05                                                                     1
/FAIL/GTN/242
GTN Alias
                1.60                1.10                                                                     1
/FAIL/GURSON_TVERGAARD_NEEDLEMAN/243
GTN Full Name Alias
                1.65                1.15                                                                     1
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 240 in model.fail_gursons
    assert 241 in model.fail_gursons
    assert 242 in model.fail_gursons
    assert 243 in model.fail_gursons
    assert pytest.approx(model.fail_gursons[240].q1) == 1.50
    assert pytest.approx(model.fail_gursons[241].q1) == 1.55
    assert pytest.approx(model.fail_gursons[242].q1) == 1.60
    assert pytest.approx(model.fail_gursons[243].q1) == 1.65


# ============================================================================
# 2. /ENG/LODE_ANGLE Tests
# ============================================================================

def test_m263_eng_lode_angle(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Lode Angle Output Test
1 1
/ENG/LODE_ANGLE/1
Lode Angle Field Output
              0.0005         8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_lode_angles
    ela = model.eng_lode_angles[1]
    assert pytest.approx(ela.dt_lode) == 0.0005
    assert ela.sens_id == 8


def test_m263_eng_lode_angle_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Lode Angle Aliases Test
1 1
/ENG/LODE/2
Lode Alias
              0.0001         0
/ENG/LODE_PARAM/3
Lode Param Alias
              0.0002         5
/ENG/LODE_ANGLE_PARAM/4
Lode Angle Param Alias
              0.0003         7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_lode_angles
    assert 3 in model.eng_lode_angles
    assert 4 in model.eng_lode_angles
    assert pytest.approx(model.eng_lode_angles[2].dt_lode) == 0.0001
    assert pytest.approx(model.eng_lode_angles[3].dt_lode) == 0.0002
    assert pytest.approx(model.eng_lode_angles[4].dt_lode) == 0.0003


# ============================================================================
# 3. /LAGMUL/DIFFERENTIAL_GEAR Tests
# ============================================================================

def test_m263_differential_gear(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Differential Gear Joint Test
1 1
/LAGMUL/DIFFERENTIAL_GEAR/790
Automotive Differential Joint
        10        20        30                3.73              5.0e+6         2              1.0e-6
                 1.0                 0.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 790 in model.lagmul_differential_gears
    dg = model.lagmul_differential_gears[790]
    assert dg.node1 == 10
    assert dg.node2 == 20
    assert dg.node3 == 30
    assert pytest.approx(dg.ratio) == 3.73
    assert pytest.approx(dg.stiff) == 5.0e+6
    assert dg.skew_id == 2
    assert pytest.approx(dg.tol) == 1.0e-6
    assert pytest.approx(dg.axis_x) == 1.0
    assert pytest.approx(dg.axis_y) == 0.0
    assert pytest.approx(dg.axis_z) == 0.0


def test_m263_differential_gear_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Differential Gear Aliases Test
1 1
/DIFFERENTIAL_GEAR/791
Diff Gear Alias 1
         1         2         3                 3.5              1.0e+6         0              1.0e-6
/LAGMUL/DIFF_GEAR/792
Diff Gear Alias 2
         4         5         6                 4.1              2.0e+6         0              1.0e-6
/DIFF_GEAR/793
Diff Gear Alias 3
         7         8         9                 4.5              3.0e+6         0              1.0e-6
/LAGMUL/DIFFERENTIAL/794
Diff Gear Alias 4
        11        12        13                 2.8              4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 791 in model.lagmul_differential_gears
    assert 792 in model.lagmul_differential_gears
    assert 793 in model.lagmul_differential_gears
    assert 794 in model.lagmul_differential_gears
    assert pytest.approx(model.lagmul_differential_gears[791].ratio) == 3.5
    assert pytest.approx(model.lagmul_differential_gears[792].ratio) == 4.1
    assert pytest.approx(model.lagmul_differential_gears[793].ratio) == 4.5
    assert pytest.approx(model.lagmul_differential_gears[794].ratio) == 2.8


# ============================================================================
# 4. /SENSOR/SPRING_MOMENT_RATE Tests
# ============================================================================

def test_m263_sensor_spring_moment_rate(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Moment-Rate Sensor Test
1 1
/SENSOR/SPRING_MOMENT_RATE/960
Spring Moment Rate Sensor 1
       400              5.5e+5               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 960 in model.sensor_spring_moment_rates
    sens = model.sensor_spring_moment_rates[960]
    assert sens.id == 960
    assert sens.spring_id == 400
    assert pytest.approx(sens.dm_max) == 5.5e+5
    assert pytest.approx(sens.t_delay) == 0.002

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 960]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_MOMENT_RATE"
    assert pytest.approx(matching[0].tdelay) == 0.002


def test_m263_sensor_spring_moment_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Moment Rate Sensor Aliases Test
1 1
/SENSOR/SPRING_DM/961
Spring DM Alias 1
       401              6.5e+5               0.001
/SENSOR/SPRING_MOMENTRATE/962
Spring MomentRate Alias 2
       402              7.5e+5               0.003
/SENSOR/DM_SPRING/963
DM Spring Alias 3
       403              8.5e+5               0.004
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 961 in model.sensor_spring_moment_rates
    assert 962 in model.sensor_spring_moment_rates
    assert 963 in model.sensor_spring_moment_rates
    assert pytest.approx(model.sensor_spring_moment_rates[961].dm_max) == 6.5e+5
    assert pytest.approx(model.sensor_spring_moment_rates[962].dm_max) == 7.5e+5
    assert pytest.approx(model.sensor_spring_moment_rates[963].dm_max) == 8.5e+5
