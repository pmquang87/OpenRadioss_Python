# -*- coding: utf-8 -*-
"""Tests for Milestone M261:
- Syazwan Hosford-Coulomb Failure Model: /FAIL/SYAZWAN, /FAIL/SYAZWAN_MODEL, /FAIL/SYAZWAN_LAW, /FAIL/HOSFORD_COULOMB
- Engine Temperature Output Directive: /ENG/TEMPERATURE, /ENG/TEMP, /ENG/THERMAL_TEMP, /ENG/NODE_TEMP
- Rack and Pinion Kinematic Joint Constraint: /LAGMUL/RACK_AND_PINION, /RACK_AND_PINION, /LAGMUL/RACK_PINION, /RACK_PINION_GEAR, /RACK_PINION
- Spring Force-Rate Sensor: /SENSOR/SPRING_FORCE_RATE, /SENSOR/SPRING_DF, /SENSOR/SPRING_FORCERATE, /SENSOR/DF_SPRING
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
# 1. /FAIL/SYAZWAN Tests
# ============================================================================

def test_m261_fail_syazwan_icard1_fixed(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Syazwan Failure Model ICARD=1 Test
2022 0
/FAIL/SYAZWAN/190
Syazwan Direct Constants Model
                   1                0.02
                0.15                0.25                0.35                0.45                0.55
                0.65
                   1                0.05                0.95
         1         2                0.22                 2.5
                   3                1.50                 0.9
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 190 in model.fail_syazwans
    fs = model.fail_syazwans[190]
    assert fs.icard == 1
    assert pytest.approx(fs.epfmin) == 0.02
    assert pytest.approx(fs.c1) == 0.15
    assert pytest.approx(fs.c2) == 0.25
    assert pytest.approx(fs.c3) == 0.35
    assert pytest.approx(fs.c4) == 0.45
    assert pytest.approx(fs.c5) == 0.55
    assert pytest.approx(fs.c6) == 0.65
    assert fs.dinit == 1
    assert pytest.approx(fs.dam_sf) == 0.05
    assert pytest.approx(fs.max_dam) == 0.95
    assert fs.inst == 1
    assert fs.iform == 2
    assert pytest.approx(fs.n_val) == 0.22
    assert pytest.approx(fs.softexp) == 2.5
    assert fs.reg_func == 3
    assert pytest.approx(fs.ref_len) == 1.50
    assert pytest.approx(fs.reg_scale) == 0.9
    assert fs.ifail_sh == 1


def test_m261_fail_syazwan_icard2_fixed(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Syazwan Failure Model ICARD=2 Test
2022 0
/FAIL/SYAZWAN/191
Syazwan Calibrated Strains Model
                   2                0.01
                0.40                0.30                0.50                0.20                0.60
                   0                 0.0                 1.0
         0         0                 0.0                 0.0
                   0                 0.0                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 191 in model.fail_syazwans
    fs = model.fail_syazwans[191]
    assert fs.icard == 2
    assert pytest.approx(fs.epfmin) == 0.01
    assert pytest.approx(fs.epf_comp) == 0.40
    assert pytest.approx(fs.epf_shear) == 0.30
    assert pytest.approx(fs.epf_tens) == 0.50
    assert pytest.approx(fs.epf_plstrn) == 0.20
    assert pytest.approx(fs.epf_biax) == 0.60
    assert pytest.approx(fs.max_dam) == 1.0


def test_m261_fail_syazwan_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Syazwan Aliases Test
2022 0
/FAIL/SYAZWAN_MODEL/220
Syazwan Model Alias
                   1                0.01
                0.10                0.20                0.30                0.40                0.50
                0.60
/FAIL/SYAZWAN_LAW/221
Syazwan Law Alias
                   1                0.02
                0.11                0.21                0.31                0.41                0.51
                0.61
/FAIL/HOSFORD_COULOMB/222
Hosford Coulomb Alias
                   1                0.03
                0.12                0.22                0.32                0.42                0.52
                0.62
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 220 in model.fail_syazwans
    assert 221 in model.fail_syazwans
    assert 222 in model.fail_syazwans
    assert pytest.approx(model.fail_syazwans[220].c1) == 0.10
    assert pytest.approx(model.fail_syazwans[221].c1) == 0.11
    assert pytest.approx(model.fail_syazwans[222].c1) == 0.12


# ============================================================================
# 2. /ENG/TEMPERATURE Tests
# ============================================================================

def test_m261_eng_temperature(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Temperature Output Test
1 1
/ENG/TEMPERATURE/1
Temperature Output
              0.0005        15
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_temperatures
    et = model.eng_temperatures[1]
    assert pytest.approx(et.dt_temp) == 0.0005
    assert et.sens_id == 15


def test_m261_eng_temperature_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Temperature Aliases Test
1 1
/ENG/TEMP/2
Temp Alias
              0.0001         0
/ENG/THERMAL_TEMP/3
Thermal Temp Alias
              0.0002         5
/ENG/NODE_TEMP/4
Node Temp Alias
              0.0003         8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_temperatures
    assert 3 in model.eng_temperatures
    assert 4 in model.eng_temperatures
    assert pytest.approx(model.eng_temperatures[2].dt_temp) == 0.0001
    assert pytest.approx(model.eng_temperatures[3].dt_temp) == 0.0002
    assert pytest.approx(model.eng_temperatures[4].dt_temp) == 0.0003


# ============================================================================
# 3. /LAGMUL/RACK_AND_PINION Tests
# ============================================================================

def test_m261_rack_and_pinion_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Rack and Pinion Joint Test
1 1
/LAGMUL/RACK_AND_PINION/770
Rack and Pinion Steering Mechanism
        10        20                25.0              5.0e+6         2              1.0e-5
                 0.0                 0.0                 1.0                 1.0                 0.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 770 in model.lagmul_rack_pinions
    rp = model.lagmul_rack_pinions[770]
    assert rp.node1 == 10
    assert rp.node2 == 20
    assert pytest.approx(rp.pitch_radius) == 25.0
    assert pytest.approx(rp.stiff) == 5.0e+6
    assert rp.skew_id == 2
    assert pytest.approx(rp.tol) == 1.0e-5
    assert pytest.approx(rp.axis_rot_x) == 0.0
    assert pytest.approx(rp.axis_rot_y) == 0.0
    assert pytest.approx(rp.axis_rot_z) == 1.0
    assert pytest.approx(rp.axis_tra_x) == 1.0
    assert pytest.approx(rp.axis_tra_y) == 0.0
    assert pytest.approx(rp.axis_tra_z) == 0.0


def test_m261_rack_and_pinion_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Rack and Pinion Aliases Test
1 1
/RACK_AND_PINION/771
RP Alias 1
         1         2                15.0              2.0e+6         0              1.0e-6
/LAGMUL/RACK_PINION/772
RP Alias 2
         3         4                18.0              3.0e+6         0              1.0e-6
/RACK_PINION_GEAR/773
RP Alias 3
         5         6                22.0              4.0e+6         0              1.0e-6
/RACK_PINION/774
RP Alias 4
         7         8                30.0              6.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 771 in model.lagmul_rack_pinions
    assert 772 in model.lagmul_rack_pinions
    assert 773 in model.lagmul_rack_pinions
    assert 774 in model.lagmul_rack_pinions
    assert pytest.approx(model.lagmul_rack_pinions[771].pitch_radius) == 15.0
    assert pytest.approx(model.lagmul_rack_pinions[772].pitch_radius) == 18.0
    assert pytest.approx(model.lagmul_rack_pinions[773].pitch_radius) == 22.0
    assert pytest.approx(model.lagmul_rack_pinions[774].pitch_radius) == 30.0


# ============================================================================
# 4. /SENSOR/SPRING_FORCE_RATE Tests
# ============================================================================

def test_m261_sensor_spring_force_rate(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Force Rate Sensor Test
1 1
/SENSOR/SPRING_FORCE_RATE/940
Spring Force Rate Sensor 1
       300              5.0e+5               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 940 in model.sensor_spring_force_rates
    sens = model.sensor_spring_force_rates[940]
    assert sens.id == 940
    assert sens.spring_id == 300
    assert pytest.approx(sens.df_max) == 5.0e+5
    assert pytest.approx(sens.t_delay) == 0.002

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 940]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_FORCE_RATE"
    assert pytest.approx(matching[0].tdelay) == 0.002


def test_m261_sensor_spring_force_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Force Rate Sensor Aliases Test
1 1
/SENSOR/SPRING_DF/941
Spring DF Alias 1
       301              2.0e+5               0.001
/SENSOR/SPRING_FORCERATE/942
Spring Forcerate Alias 2
       302              3.0e+5               0.003
/SENSOR/DF_SPRING/943
DF Spring Alias 3
       303              4.0e+5               0.004
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 941 in model.sensor_spring_force_rates
    assert 942 in model.sensor_spring_force_rates
    assert 943 in model.sensor_spring_force_rates
    assert pytest.approx(model.sensor_spring_force_rates[941].df_max) == 2.0e+5
    assert pytest.approx(model.sensor_spring_force_rates[942].df_max) == 3.0e+5
    assert pytest.approx(model.sensor_spring_force_rates[943].df_max) == 4.0e+5
