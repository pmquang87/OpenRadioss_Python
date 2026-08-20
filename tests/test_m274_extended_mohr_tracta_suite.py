# -*- coding: utf-8 -*-
"""Tests for Milestone M274:
- Extended Mohr-Coulomb Failure Model: /FAIL/EXTENDED_MOHR and aliases
- Engine Joint Energy Output Directive: /ENG/JOINT_ENERGY and aliases
- Tracta Joint Constraint: /LAGMUL/TRACTA_JOINT and aliases
- Spring Bending Energy Sensor: /SENSOR/SPRING_BENDING_ENERGY and aliases
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
# 1. /FAIL/EXTENDED_MOHR Tests
# ============================================================================

def test_m274_fail_extended_mohr_fixed(tmp_path: Path):
    c1 = f"{0.50:>20.4f}{0.75:>20.4f}{0.30:>20.4f}{0.15:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}{0.88:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Extended Mohr-Coulomb Failure Fixed Format Test
2022 0
/FAIL/EXTENDED_MOHR/801
Extended Mohr-Coulomb Model
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 801 in model.fail_extended_mohrs
    fem = model.fail_extended_mohrs[801]
    assert pytest.approx(fem.c_0) == 0.50
    assert pytest.approx(fem.c_1) == 0.75
    assert pytest.approx(fem.c_2) == 0.30
    assert pytest.approx(fem.c_theta) == 0.15
    assert fem.ifail_sh == 2
    assert fem.ifail_so == 1
    assert pytest.approx(fem.d_max) == 0.88


def test_m274_fail_extended_mohr_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Extended Mohr Defaults Test
2022 0
/FAIL/EXTENDED_MOHR/802
Extended Mohr Default Values
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m274_fail_extended_mohr_aliases(tmp_path: Path):
    c1 = f"{0.40:>20.4f}{0.60:>20.4f}{0.20:>20.4f}{0.10:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Extended Mohr Aliases Test
2022 0
/FAIL/EXTENDED_MOHR_COULOMB/850
Ext Mohr Coulomb Alias
{c1}
/FAIL/EXT_MOHR/851
Ext Mohr Alias
{c1}
/FAIL/EMC_FAILURE/852
EMC Failure Alias
{c1}
/FAIL/EXTENDED_MC/853
Extended MC Alias
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 850 in model.fail_extended_mohrs
    assert 851 in model.fail_extended_mohrs
    assert 852 in model.fail_extended_mohrs
    assert 853 in model.fail_extended_mohrs
    assert pytest.approx(model.fail_extended_mohrs[850].c_0) == 0.40
    assert pytest.approx(model.fail_extended_mohrs[851].c_0) == 0.40
    assert pytest.approx(model.fail_extended_mohrs[852].c_0) == 0.40
    assert pytest.approx(model.fail_extended_mohrs[853].c_0) == 0.40


# ============================================================================
# 2. /ENG/JOINT_ENERGY Tests
# ============================================================================

def test_m274_eng_joint_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Joint Energy Output Test
1 1
/ENG/JOINT_ENERGY/1
Joint Energy Field Output
              0.0011        40
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_joint_energies
    eje = model.eng_joint_energies[1]
    assert pytest.approx(eje.dt_joint) == 0.0011
    assert eje.sens_id == 40


def test_m274_eng_joint_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Joint Energy Aliases Test
1 1
/ENG/JNT_ENERGY/2
Jnt Energy Alias
              0.0001         0
/ENG/JOINT_WORK/3
Joint Work Alias
              0.0002         1
/ENG/EJNT/4
Ejnt Alias
              0.0003         3
/ENG/JOINT_ENER/5
Joint Ener Alias
              0.0004         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_joint_energies
    assert 3 in model.eng_joint_energies
    assert 4 in model.eng_joint_energies
    assert 5 in model.eng_joint_energies
    assert pytest.approx(model.eng_joint_energies[2].dt_joint) == 0.0001
    assert pytest.approx(model.eng_joint_energies[3].dt_joint) == 0.0002
    assert pytest.approx(model.eng_joint_energies[4].dt_joint) == 0.0003
    assert pytest.approx(model.eng_joint_energies[5].dt_joint) == 0.0004


# ============================================================================
# 3. /LAGMUL/TRACTA_JOINT Tests
# ============================================================================

def test_m274_tracta_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Tracta Joint Test
1 1
/LAGMUL/TRACTA_JOINT/990
Tracta Sliding Yoke Joint
        65        75        85              8.5e+6         9              2.0e-6
                 0.0                 1.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 990 in model.lagmul_tracta_joints
    tj = model.lagmul_tracta_joints[990]
    assert tj.node1 == 65
    assert tj.node2 == 75
    assert tj.node3 == 85
    assert pytest.approx(tj.stiff) == 8.5e+6
    assert tj.skew_id == 9
    assert pytest.approx(tj.tol) == 2.0e-6
    assert pytest.approx(tj.axis_x) == 0.0
    assert pytest.approx(tj.axis_y) == 1.0
    assert pytest.approx(tj.axis_z) == 0.0


def test_m274_tracta_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Tracta Joint Aliases Test
1 1
/TRACTA_JOINT/991
Tracta Joint Alias 1
         1         2         3              1.0e+6         0              1.0e-6
/LAGMUL/TRACTA_COUPLING/992
Tracta Coupling Alias 2
         4         5         6              2.0e+6         0              1.0e-6
/TRACTA_COUPLING/993
Tracta Coupling Alias 3
         7         8         9              3.0e+6         0              1.0e-6
/TRACTA_MECHANISM/994
Tracta Mechanism Alias 4
        10        11        12              4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 991 in model.lagmul_tracta_joints
    assert 992 in model.lagmul_tracta_joints
    assert 993 in model.lagmul_tracta_joints
    assert 994 in model.lagmul_tracta_joints
    assert model.lagmul_tracta_joints[991].node3 == 3
    assert model.lagmul_tracta_joints[992].node3 == 6
    assert model.lagmul_tracta_joints[993].node3 == 9
    assert model.lagmul_tracta_joints[994].node3 == 12


# ============================================================================
# 4. /SENSOR/SPRING_BENDING_ENERGY Tests
# ============================================================================

def test_m274_sensor_spring_bending_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Bending Energy Sensor Test
1 1
/SENSOR/SPRING_BENDING_ENERGY/2100
Spring Bending Energy Sensor 1
       990              1.2e+5               0.0040
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2100 in model.sensor_spring_bending_energies
    sens = model.sensor_spring_bending_energies[2100]
    assert sens.id == 2100
    assert sens.spring_id == 990
    assert pytest.approx(sens.u_bend_max) == 1.2e+5
    assert pytest.approx(sens.t_delay) == 0.0040

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 2100]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_BENDING_ENERGY"
    assert pytest.approx(matching[0].tdelay) == 0.0040


def test_m274_sensor_spring_bending_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Bending Energy Sensor Aliases Test
1 1
/SENSOR/SPRING_BEND_ENERGY/2200
Spring Bend Energy Alias 1
       801              4.0e+4               0.001
/SENSOR/SPRING_FLEX_ENERGY/2201
Spring Flex Energy Alias 2
       802              5.0e+4               0.002
/SENSOR/BENDING_ENERGY_SPRING/2202
Bending Energy Spring Alias 3
       803              6.0e+4               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2200 in model.sensor_spring_bending_energies
    assert 2201 in model.sensor_spring_bending_energies
    assert 2202 in model.sensor_spring_bending_energies
    assert pytest.approx(model.sensor_spring_bending_energies[2200].u_bend_max) == 4.0e+4
    assert pytest.approx(model.sensor_spring_bending_energies[2201].u_bend_max) == 5.0e+4
    assert pytest.approx(model.sensor_spring_bending_energies[2202].u_bend_max) == 6.0e+4
