# -*- coding: utf-8 -*-
"""Tests for Milestone M272:
- Wilkins Cumulative Damage Failure Model: /FAIL/WILKINS_CUMULATIVE and aliases
- Engine Contact Energy Output Directive: /ENG/CONTACT_ENERGY and aliases
- Tripod Joint Constraint: /LAGMUL/TRIPOD_JOINT and aliases
- Spring Coupling Energy Sensor: /SENSOR/SPRING_COUPLING_ENERGY and aliases
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
# 1. /FAIL/WILKINS_CUMULATIVE Tests
# ============================================================================

def test_m272_fail_wilkins_cumulative_fixed(tmp_path: Path):
    c1 = f"{0.75:>20.4f}{2.0:>20.4f}{0.50:>20.4f}{-100.0:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}{0.90:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Wilkins Cumulative Failure Fixed Format Test
2022 0
/FAIL/WILKINS_CUMULATIVE/601
Wilkins Cumulative Damage Model
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 601 in model.fail_wilkins_cumulatives
    fwc = model.fail_wilkins_cumulatives[601]
    assert pytest.approx(fwc.d_crit) == 0.75
    assert pytest.approx(fwc.a_wk) == 2.0
    assert pytest.approx(fwc.b_wk) == 0.50
    assert pytest.approx(fwc.p_min) == -100.0
    assert fwc.ifail_sh == 2
    assert fwc.ifail_so == 1
    assert pytest.approx(fwc.d_max) == 0.90


def test_m272_fail_wilkins_cumulative_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Wilkins Cumulative Defaults Test
2022 0
/FAIL/WILKINS_CUMULATIVE/602
Wilkins Cumulative Default Values
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m272_fail_wilkins_cumulative_aliases(tmp_path: Path):
    c1 = f"{0.50:>20.4f}{1.5:>20.4f}{0.30:>20.4f}{-50.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Wilkins Cumulative Aliases Test
2022 0
/FAIL/WILKINS_CUMUL/650
Wilkins Cumul Alias
{c1}
/FAIL/WILKINS_DAMAGE/651
Wilkins Damage Alias
{c1}
/FAIL/WK_CUMULATIVE/652
WK Cumulative Alias
{c1}
/FAIL/WILKINS_CUM_LAW/653
Wilkins Cum Law Alias
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 650 in model.fail_wilkins_cumulatives
    assert 651 in model.fail_wilkins_cumulatives
    assert 652 in model.fail_wilkins_cumulatives
    assert 653 in model.fail_wilkins_cumulatives
    assert pytest.approx(model.fail_wilkins_cumulatives[650].d_crit) == 0.50
    assert pytest.approx(model.fail_wilkins_cumulatives[651].d_crit) == 0.50
    assert pytest.approx(model.fail_wilkins_cumulatives[652].d_crit) == 0.50
    assert pytest.approx(model.fail_wilkins_cumulatives[653].d_crit) == 0.50


# ============================================================================
# 2. /ENG/CONTACT_ENERGY Tests
# ============================================================================

def test_m272_eng_contact_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Contact Energy Output Test
1 1
/ENG/CONTACT_ENERGY/1
Contact Energy Field Output
              0.0007        30
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_contact_energies
    ece = model.eng_contact_energies[1]
    assert pytest.approx(ece.dt_contact) == 0.0007
    assert ece.sens_id == 30


def test_m272_eng_contact_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Contact Energy Aliases Test
1 1
/ENG/CNT_ENERGY/2
Cnt Energy Alias
              0.0001         0
/ENG/CONTACT_WORK/3
Contact Work Alias
              0.0002         1
/ENG/ECNT/4
Ecnt Alias
              0.0003         3
/ENG/INTERFACE_ENERGY/5
Interface Energy Alias
              0.0004         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_contact_energies
    assert 3 in model.eng_contact_energies
    assert 4 in model.eng_contact_energies
    assert 5 in model.eng_contact_energies
    assert pytest.approx(model.eng_contact_energies[2].dt_contact) == 0.0001
    assert pytest.approx(model.eng_contact_energies[3].dt_contact) == 0.0002
    assert pytest.approx(model.eng_contact_energies[4].dt_contact) == 0.0003
    assert pytest.approx(model.eng_contact_energies[5].dt_contact) == 0.0004


# ============================================================================
# 3. /LAGMUL/TRIPOD_JOINT Tests
# ============================================================================

def test_m272_tripod_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Tripod Joint Test
1 1
/LAGMUL/TRIPOD_JOINT/970
Tripod Tulip Spider Joint
        35        45        55              6.0e+6         7              5.0e-6
                 0.0                 0.0                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 970 in model.lagmul_tripod_joints
    tj = model.lagmul_tripod_joints[970]
    assert tj.node1 == 35
    assert tj.node2 == 45
    assert tj.node3 == 55
    assert pytest.approx(tj.stiff) == 6.0e+6
    assert tj.skew_id == 7
    assert pytest.approx(tj.tol) == 5.0e-6
    assert pytest.approx(tj.axis_x) == 0.0
    assert pytest.approx(tj.axis_y) == 0.0
    assert pytest.approx(tj.axis_z) == 1.0


def test_m272_tripod_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Tripod Joint Aliases Test
1 1
/TRIPOD_JOINT/971
Tripod Joint Alias 1
         1         2         3              1.0e+6         0              1.0e-6
/LAGMUL/TRIPOD_COUPLING/972
Tripod Coupling Alias 2
         4         5         6              2.0e+6         0              1.0e-6
/TRIPOD_COUPLING/973
Tripod Coupling Alias 3
         7         8         9              3.0e+6         0              1.0e-6
/TRIPOD_MECHANISM/974
Tripod Mechanism Alias 4
        10        11        12              4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 971 in model.lagmul_tripod_joints
    assert 972 in model.lagmul_tripod_joints
    assert 973 in model.lagmul_tripod_joints
    assert 974 in model.lagmul_tripod_joints
    assert model.lagmul_tripod_joints[971].node3 == 3
    assert model.lagmul_tripod_joints[972].node3 == 6
    assert model.lagmul_tripod_joints[973].node3 == 9
    assert model.lagmul_tripod_joints[974].node3 == 12


# ============================================================================
# 4. /SENSOR/SPRING_COUPLING_ENERGY Tests
# ============================================================================

def test_m272_sensor_spring_coupling_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Coupling Energy Sensor Test
1 1
/SENSOR/SPRING_COUPLING_ENERGY/1700
Spring Coupling Energy Sensor 1
       970              2.0e+5               0.0030
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1700 in model.sensor_spring_coupling_energies
    sens = model.sensor_spring_coupling_energies[1700]
    assert sens.id == 1700
    assert sens.spring_id == 970
    assert pytest.approx(sens.u_coup_max) == 2.0e+5
    assert pytest.approx(sens.t_delay) == 0.0030

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 1700]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_COUPLING_ENERGY"
    assert pytest.approx(matching[0].tdelay) == 0.0030


def test_m272_sensor_spring_coupling_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Coupling Energy Sensor Aliases Test
1 1
/SENSOR/SPRING_COUP_ENERGY/1800
Spring Coup Energy Alias 1
       801              4.0e+4               0.001
/SENSOR/SPRING_COUPLE_ENERGY/1801
Spring Couple Energy Alias 2
       802              5.0e+4               0.002
/SENSOR/COUPLING_ENERGY_SPRING/1802
Coupling Energy Spring Alias 3
       803              6.0e+4               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1800 in model.sensor_spring_coupling_energies
    assert 1801 in model.sensor_spring_coupling_energies
    assert 1802 in model.sensor_spring_coupling_energies
    assert pytest.approx(model.sensor_spring_coupling_energies[1800].u_coup_max) == 4.0e+4
    assert pytest.approx(model.sensor_spring_coupling_energies[1801].u_coup_max) == 5.0e+4
    assert pytest.approx(model.sensor_spring_coupling_energies[1802].u_coup_max) == 6.0e+4
