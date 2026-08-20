# -*- coding: utf-8 -*-
"""Tests for Milestone M277:
- Alter Subcritical Crack Growth Failure Model: /FAIL/ALTER and aliases
- Engine Heat Exchange Energy Output Directive: /ENG/HEAT_EXCHANGE and aliases
- Tripod Ball Joint Constraint: /LAGMUL/TRIPOD_BALL_JOINT and aliases
- Spring Thermal Dissipation Energy Sensor: /SENSOR/SPRING_THERMAL_DISSIPATION and aliases
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
# 1. /FAIL/ALTER Tests
# ============================================================================

def test_m277_fail_alter_fixed(tmp_path: Path):
    c1 = f"{16.5:>20.4f}{0.002:>20.4f}{1500.0:>20.4f}{10:>10d}{1:>10d}{0:>10d}{2:>10d}"
    c2 = f"{0.15:>20.4f}{0.12:>20.4f}{0.08:>20.4f}{0.25:>20.4f}"
    c3 = f"{75.0:>20.4f}{12.5:>20.4f}{2.5:>20.4f}{0.005:>20.4f}"
    c4 = f"{2:>10d}{1:>10d}{0.95:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Alter Subcritical Crack Growth Fixed Format Test
2022 0
/FAIL/ALTER/830
Alter Glass Fracture Model
{c1}
{c2}
{c3}
{c4}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 830 in model.fail_alters
    fa = model.fail_alters[830]
    assert pytest.approx(fa.exp_n) == 16.5
    assert pytest.approx(fa.v0) == 0.002
    assert pytest.approx(fa.vc) == 1500.0
    assert fa.ema == 10
    assert fa.irate == 1
    assert fa.iside == 0
    assert fa.mode == 2
    assert pytest.approx(fa.cr_foil) == 0.15
    assert pytest.approx(fa.cr_air) == 0.12
    assert pytest.approx(fa.cr_core) == 0.08
    assert pytest.approx(fa.cr_edge) == 0.25
    assert pytest.approx(fa.kic) == 75.0
    assert pytest.approx(fa.kth) == 12.5
    assert pytest.approx(fa.rlen) == 2.5
    assert pytest.approx(fa.tdel) == 0.005
    assert fa.ifail_sh == 2
    assert fa.ifail_so == 1
    assert pytest.approx(fa.d_max) == 0.95


def test_m277_fail_alter_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Alter Defaults Test
2022 0
/FAIL/ALTER/831
Alter Default Model
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 831 in model.fail_alters
    fa = model.fail_alters[831]
    assert pytest.approx(fa.exp_n) == 1.0
    assert pytest.approx(fa.v0) == 0.0
    assert pytest.approx(fa.vc) == 1e30
    assert fa.ema == 0
    assert fa.irate == 0
    assert fa.iside == 0
    assert fa.mode == 0
    assert fa.ifail_sh == 1
    assert fa.ifail_so == 1
    assert pytest.approx(fa.d_max) == 1.0


def test_m277_fail_alter_aliases(tmp_path: Path):
    c1_1 = f"{18.0:>20.4f}{0.005:>20.4f}{1800.0:>20.4f}{5:>10d}{0:>10d}{1:>10d}{0:>10d}"
    c1_2 = f"{0.2:>20.4f}{0.1:>20.4f}{0.05:>20.4f}{0.3:>20.4f}"
    c1_3 = f"{80.0:>20.4f}{15.0:>20.4f}{3.0:>20.4f}{0.001:>20.4f}"
    c1_4 = f"{1:>10d}{1:>10d}{1.0:>20.4f}"

    c2_1 = f"{20.0:>20.4f}{0.008:>20.4f}{2000.0:>20.4f}{8:>10d}{1:>10d}{0:>10d}{1:>10d}"
    c2_2 = f"{0.3:>20.4f}{0.15:>20.4f}{0.1:>20.4f}{0.4:>20.4f}"
    c2_3 = f"{90.0:>20.4f}{18.0:>20.4f}{4.0:>20.4f}{0.002:>20.4f}"
    c2_4 = f"{2:>10d}{2:>10d}{0.85:>20.4f}"

    c3_1 = f"{22.0:>20.4f}{0.01:>20.4f}{2200.0:>20.4f}{12:>10d}{0:>10d}{0:>10d}{2:>10d}"
    c3_2 = f"{0.25:>20.4f}{0.18:>20.4f}{0.12:>20.4f}{0.35:>20.4f}"
    c3_3 = f"{95.0:>20.4f}{20.0:>20.4f}{5.0:>20.4f}{0.003:>20.4f}"
    c3_4 = f"{1:>10d}{2:>10d}{0.9:>20.4f}"

    c4_1 = f"{25.0:>20.4f}{0.015:>20.4f}{2500.0:>20.4f}{15:>10d}{1:>10d}{1:>10d}{3:>10d}"
    c4_2 = f"{0.35:>20.4f}{0.22:>20.4f}{0.15:>20.4f}{0.45:>20.4f}"
    c4_3 = f"{100.0:>20.4f}{22.0:>20.4f}{6.0:>20.4f}{0.004:>20.4f}"
    c4_4 = f"{2:>10d}{1:>10d}{0.98:>20.4f}"

    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Alter Aliases Test
2022 0
/FAIL/ALTER_MODEL/832
Alter Model Alias
{c1_1}
{c1_2}
{c1_3}
{c1_4}
/FAIL/ALTER_GLASS/833
Alter Glass Alias
{c2_1}
{c2_2}
{c2_3}
{c2_4}
/FAIL/ALTER_FRACTURE/834
Alter Fracture Alias
{c3_1}
{c3_2}
{c3_3}
{c3_4}
/FAIL/ALTER_LAW/835
Alter Law Alias
{c4_1}
{c4_2}
{c4_3}
{c4_4}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 832 in model.fail_alters
    assert pytest.approx(model.fail_alters[832].exp_n) == 18.0
    assert pytest.approx(model.fail_alters[832].kic) == 80.0
    assert 833 in model.fail_alters
    assert pytest.approx(model.fail_alters[833].exp_n) == 20.0
    assert pytest.approx(model.fail_alters[833].kic) == 90.0
    assert 834 in model.fail_alters
    assert pytest.approx(model.fail_alters[834].exp_n) == 22.0
    assert pytest.approx(model.fail_alters[834].kic) == 95.0
    assert 835 in model.fail_alters
    assert pytest.approx(model.fail_alters[835].exp_n) == 25.0
    assert pytest.approx(model.fail_alters[835].kic) == 100.0


# ============================================================================
# 2. /ENG/HEAT_EXCHANGE Tests
# ============================================================================

def test_m277_eng_heat_exchange(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Heat Exchange Output Test
2022 0
/ENG/HEAT_EXCHANGE/1
Engine Heat Exchange Directive
              0.0002         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_heat_exchanges
    ehe = model.eng_heat_exchanges[1]
    assert pytest.approx(ehe.dt_heat) == 0.0002
    assert ehe.sens_id == 5


def test_m277_eng_heat_exchange_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Heat Exchange Aliases Test
2022 0
/ENG/HEAT_ENERGY/2
Heat Energy Alias
              0.0004         6
/ENG/THERMAL_ENERGY/3
Thermal Energy Alias
              0.0006         7
/ENG/EHEAT/4
EHeat Alias
              0.0008         8
/ENG/HEAT_WORK/5
Heat Work Alias
              0.0010         9
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_heat_exchanges
    assert pytest.approx(model.eng_heat_exchanges[2].dt_heat) == 0.0004
    assert model.eng_heat_exchanges[2].sens_id == 6
    assert 3 in model.eng_heat_exchanges
    assert pytest.approx(model.eng_heat_exchanges[3].dt_heat) == 0.0006
    assert model.eng_heat_exchanges[3].sens_id == 7
    assert 4 in model.eng_heat_exchanges
    assert pytest.approx(model.eng_heat_exchanges[4].dt_heat) == 0.0008
    assert model.eng_heat_exchanges[4].sens_id == 8
    assert 5 in model.eng_heat_exchanges
    assert pytest.approx(model.eng_heat_exchanges[5].dt_heat) == 0.0010
    assert model.eng_heat_exchanges[5].sens_id == 9


# ============================================================================
# 3. /LAGMUL/TRIPOD_BALL_JOINT Tests
# ============================================================================

def test_m277_tripod_ball_joint(tmp_path: Path):
    c1 = f"{12:>10d}{22:>10d}{32:>10d}{8.5e6:>20.4e}{4:>10d}{2.5e-6:>20.4e}"
    c2 = f"{0.0:>20.4f}{1.0:>20.4f}{0.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Tripod Ball Joint Test
2022 0
/LAGMUL/TRIPOD_BALL_JOINT/910
Tripod Ball Joint 1
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 910 in model.lagmul_tripod_ball_joints
    j = model.lagmul_tripod_ball_joints[910]
    assert j.node1 == 12
    assert j.node2 == 22
    assert j.node3 == 32
    assert pytest.approx(j.stiff) == 8.5e6
    assert j.skew_id == 4
    assert pytest.approx(j.tol) == 2.5e-6
    assert pytest.approx(j.axis_x) == 0.0
    assert pytest.approx(j.axis_y) == 1.0
    assert pytest.approx(j.axis_z) == 0.0


def test_m277_tripod_ball_joint_aliases(tmp_path: Path):
    c1_1 = f"{14:>10d}{24:>10d}{34:>10d}{9.0e6:>20.4e}{5:>10d}{3.0e-6:>20.4e}"
    c1_2 = f"{1.0:>20.4f}{0.0:>20.4f}{0.0:>20.4f}"

    c2_1 = f"{16:>10d}{26:>10d}{36:>10d}{9.5e6:>20.4e}{6:>10d}{3.5e-6:>20.4e}"
    c2_2 = f"{0.0:>20.4f}{0.0:>20.4f}{1.0:>20.4f}"

    c3_1 = f"{18:>10d}{28:>10d}{38:>10d}{7.5e6:>20.4e}{7:>10d}{4.0e-6:>20.4e}"
    c3_2 = f"{0.7071:>20.4f}{0.7071:>20.4f}{0.0:>20.4f}"

    c4_1 = f"{20:>10d}{30:>10d}{40:>10d}{6.5e6:>20.4e}{8:>10d}{5.0e-6:>20.4e}"
    c4_2 = f"{0.0:>20.4f}{0.7071:>20.4f}{0.7071:>20.4f}"

    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Tripod Ball Joint Aliases Test
2022 0
/TRIPOD_BALL_JOINT/911
Tripod Ball Joint Alias
{c1_1}
{c1_2}
/LAGMUL/TRIPOD_BALL/912
Tripod Ball Alias 1
{c2_1}
{c2_2}
/TRIPOD_BALL/913
Tripod Ball Alias 2
{c3_1}
{c3_2}
/TRIPOD_BALL_MECHANISM/914
Tripod Ball Mechanism Alias
{c4_1}
{c4_2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 911 in model.lagmul_tripod_ball_joints
    assert model.lagmul_tripod_ball_joints[911].node1 == 14
    assert pytest.approx(model.lagmul_tripod_ball_joints[911].stiff) == 9.0e6
    assert 912 in model.lagmul_tripod_ball_joints
    assert model.lagmul_tripod_ball_joints[912].node1 == 16
    assert pytest.approx(model.lagmul_tripod_ball_joints[912].stiff) == 9.5e6
    assert 913 in model.lagmul_tripod_ball_joints
    assert model.lagmul_tripod_ball_joints[913].node1 == 18
    assert pytest.approx(model.lagmul_tripod_ball_joints[913].stiff) == 7.5e6
    assert 914 in model.lagmul_tripod_ball_joints
    assert model.lagmul_tripod_ball_joints[914].node1 == 20
    assert pytest.approx(model.lagmul_tripod_ball_joints[914].stiff) == 6.5e6


# ============================================================================
# 4. /SENSOR/SPRING_THERMAL_DISSIPATION Tests
# ============================================================================

def test_m277_sensor_spring_thermal_dissipation(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Thermal Dissipation Sensor Test
2022 0
/SENSOR/SPRING_THERMAL_DISSIPATION/41
Thermal Dissipation Sensor 1
       205            3500.0               0.015
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 41 in model.sensor_spring_thermal_dissipations
    s = model.sensor_spring_thermal_dissipations[41]
    assert s.spring_id == 205
    assert pytest.approx(s.u_therm_max) == 3500.0
    assert pytest.approx(s.t_delay) == 0.015


def test_m277_sensor_spring_thermal_dissipation_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Thermal Dissipation Sensor Aliases Test
2022 0
/SENSOR/SPRING_THERM_DISS/42
Therm Diss Alias
       206            4000.0               0.020
/SENSOR/SPRING_HEAT_ENERGY/43
Heat Energy Alias
       207            4500.0               0.025
/SENSOR/THERMAL_DISSIPATION_SPRING/44
Thermal Dissipation Spring Alias
       208            5000.0               0.030
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 42 in model.sensor_spring_thermal_dissipations
    assert model.sensor_spring_thermal_dissipations[42].spring_id == 206
    assert pytest.approx(model.sensor_spring_thermal_dissipations[42].u_therm_max) == 4000.0
    assert 43 in model.sensor_spring_thermal_dissipations
    assert model.sensor_spring_thermal_dissipations[43].spring_id == 207
    assert pytest.approx(model.sensor_spring_thermal_dissipations[43].u_therm_max) == 4500.0
    assert 44 in model.sensor_spring_thermal_dissipations
    assert model.sensor_spring_thermal_dissipations[44].spring_id == 208
    assert pytest.approx(model.sensor_spring_thermal_dissipations[44].u_therm_max) == 5000.0
