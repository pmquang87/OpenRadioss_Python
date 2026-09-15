# -*- coding: utf-8 -*-
"""Tests for Milestone M265:
- Cockcroft-Latham Ductile Failure Model: /FAIL/COCKCROFT_LATHAM, /FAIL/COCKCROFT_LATHAM_MODEL, /FAIL/COCKCROFT_LATHAM_LAW, /FAIL/CL, /FAIL/CL_DAMAGE
- Engine Effective Stress Output Directive: /ENG/EFFECTIVE_STRESS, /ENG/SIG_EFF, /ENG/VON_MISES, /ENG/SIGVM, /ENG/EFF_STRESS
- Torque Split Gear Joint Constraint: /LAGMUL/TORQUE_SPLIT_GEAR, /TORQUE_SPLIT_GEAR, /LAGMUL/SPLIT_GEAR, /SPLIT_GEAR, /TORQUE_SPLITTER
- Spring Torsional Energy Sensor: /SENSOR/SPRING_TORSIONAL_ENERGY, /SENSOR/SPRING_TOR_ENERGY, /SENSOR/SPRING_TORSION_ENERGY, /SENSOR/TORSION_ENERGY_SPRING
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
# 1. /FAIL/COCKCROFT_LATHAM Tests
# ============================================================================

def test_m265_fail_cockcroft_latham_fixed(tmp_path: Path):
    c1 = f"{450.0:>20.4f}{0.015:>20.4f}{1.0:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}{0.95:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Cockcroft Latham Failure Model Fixed Format Test
2022 0
/FAIL/COCKCROFT_LATHAM/205
Cockcroft Latham Normalized Ductile Model
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 205 in model.fail_cockcroft_lathams
    fcl = model.fail_cockcroft_lathams[205]
    assert pytest.approx(fcl.w_crit) == 450.0
    assert pytest.approx(fcl.c_rate) == 0.015
    assert pytest.approx(fcl.eps_dot_0) == 1.0
    assert fcl.ifail_sh == 2
    assert fcl.ifail_so == 1
    assert pytest.approx(fcl.d_max) == 0.95


def test_m265_fail_cockcroft_latham_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Cockcroft Latham Defaults Test
2022 0
/FAIL/COCKCROFT_LATHAM/206
Cockcroft Latham Default Values
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m265_fail_cockcroft_latham_aliases(tmp_path: Path):
    c = f"{320.0:>20.4f}{0.005:>20.4f}{1.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Cockcroft Latham Aliases Test
2022 0
/FAIL/COCKCROFT_LATHAM_MODEL/250
CL Model Alias
{c}
/FAIL/COCKCROFT_LATHAM_LAW/251
CL Law Alias
{c}
/FAIL/CL/252
CL Short Alias
{c}
/FAIL/CL_DAMAGE/253
CL Damage Alias
{c}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 250 in model.fail_cockcroft_lathams
    assert 251 in model.fail_cockcroft_lathams
    assert 252 in model.fail_cockcroft_lathams
    assert 253 in model.fail_cockcroft_lathams
    assert pytest.approx(model.fail_cockcroft_lathams[250].w_crit) == 320.0
    assert pytest.approx(model.fail_cockcroft_lathams[251].w_crit) == 320.0
    assert pytest.approx(model.fail_cockcroft_lathams[252].w_crit) == 320.0
    assert pytest.approx(model.fail_cockcroft_lathams[253].w_crit) == 320.0


# ============================================================================
# 2. /ENG/EFFECTIVE_STRESS Tests
# ============================================================================

def test_m265_eng_effective_stress(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Effective Stress Output Test
1 1
/ENG/EFFECTIVE_STRESS/1
Effective Stress Field Output
              0.0005         8
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_effective_stresses
    ees = model.eng_effective_stresses[1]
    assert pytest.approx(ees.dt_sigeff) == 0.0005
    assert ees.sens_id == 8


def test_m265_eng_effective_stress_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Effective Stress Aliases Test
1 1
/ENG/SIG_EFF/2
Sig Eff Alias
              0.0001         0
/ENG/VON_MISES/3
Von Mises Alias
              0.0002         3
/ENG/SIGVM/4
Sigvm Alias
              0.0003         5
/ENG/EFF_STRESS/5
Eff Stress Alias
              0.0004         7
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_effective_stresses
    assert 3 in model.eng_effective_stresses
    assert 4 in model.eng_effective_stresses
    assert 5 in model.eng_effective_stresses
    assert pytest.approx(model.eng_effective_stresses[2].dt_sigeff) == 0.0001
    assert pytest.approx(model.eng_effective_stresses[3].dt_sigeff) == 0.0002
    assert pytest.approx(model.eng_effective_stresses[4].dt_sigeff) == 0.0003
    assert pytest.approx(model.eng_effective_stresses[5].dt_sigeff) == 0.0004


# ============================================================================
# 3. /LAGMUL/TORQUE_SPLIT_GEAR Tests
# ============================================================================

def test_m265_torque_split_gear(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Torque Split Gear Joint Test
1 1
/LAGMUL/TORQUE_SPLIT_GEAR/810
Dual Output Power Take-Off Splitter
        15        25        35                 0.35             7.5e+6         2              1.0e-6
                 1.0                 0.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 810 in model.lagmul_torque_split_gears
    tsg = model.lagmul_torque_split_gears[810]
    assert tsg.node1 == 15
    assert tsg.node2 == 25
    assert tsg.node3 == 35
    assert pytest.approx(tsg.split_ratio) == 0.35
    assert pytest.approx(tsg.stiff) == 7.5e+6
    assert tsg.skew_id == 2
    assert pytest.approx(tsg.tol) == 1.0e-6
    assert pytest.approx(tsg.axis_x) == 1.0
    assert pytest.approx(tsg.axis_y) == 0.0
    assert pytest.approx(tsg.axis_z) == 0.0


def test_m265_torque_split_gear_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Torque Split Gear Aliases Test
1 1
/TORQUE_SPLIT_GEAR/811
Split Gear Alias 1
         1         2         3                 0.5              1.0e+6         0              1.0e-6
/LAGMUL/SPLIT_GEAR/812
Split Gear Alias 2
         4         5         6                 0.4              2.0e+6         0              1.0e-6
/SPLIT_GEAR/813
Split Gear Alias 3
         7         8         9                 0.6              3.0e+6         0              1.0e-6
/TORQUE_SPLITTER/814
Split Gear Alias 4
        10        11        12                 0.45             4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 811 in model.lagmul_torque_split_gears
    assert 812 in model.lagmul_torque_split_gears
    assert 813 in model.lagmul_torque_split_gears
    assert 814 in model.lagmul_torque_split_gears
    assert pytest.approx(model.lagmul_torque_split_gears[811].split_ratio) == 0.5
    assert pytest.approx(model.lagmul_torque_split_gears[812].split_ratio) == 0.4
    assert pytest.approx(model.lagmul_torque_split_gears[813].split_ratio) == 0.6
    assert pytest.approx(model.lagmul_torque_split_gears[814].split_ratio) == 0.45


# ============================================================================
# 4. /SENSOR/SPRING_TORSIONAL_ENERGY Tests
# ============================================================================

def test_m265_sensor_spring_torsional_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Torsional Energy Sensor Test
1 1
/SENSOR/SPRING_TORSIONAL_ENERGY/980
Spring Torsional Energy Sensor 1
       550              1.2e+5               0.002
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 980 in model.sensor_spring_torsional_energies
    sens = model.sensor_spring_torsional_energies[980]
    assert sens.id == 980
    assert sens.spring_id == 550
    assert pytest.approx(sens.e_tor_max) == 1.2e+5
    assert pytest.approx(sens.t_delay) == 0.002

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 980]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_TORSIONAL_ENERGY"
    assert pytest.approx(matching[0].tdelay) == 0.002


def test_m265_sensor_spring_torsional_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Torsional Energy Sensor Aliases Test
1 1
/SENSOR/SPRING_TOR_ENERGY/981
Spring Tor Energy Alias 1
       551              2.5e+4               0.001
/SENSOR/SPRING_TORSION_ENERGY/982
Spring Torsion Energy Alias 2
       552              3.5e+4               0.003
/SENSOR/TORSION_ENERGY_SPRING/983
Torsion Energy Spring Alias 3
       553              4.5e+4               0.005
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 981 in model.sensor_spring_torsional_energies
    assert 982 in model.sensor_spring_torsional_energies
    assert 983 in model.sensor_spring_torsional_energies
    assert pytest.approx(model.sensor_spring_torsional_energies[981].e_tor_max) == 2.5e+4
    assert pytest.approx(model.sensor_spring_torsional_energies[982].e_tor_max) == 3.5e+4
    assert pytest.approx(model.sensor_spring_torsional_energies[983].e_tor_max) == 4.5e+4
