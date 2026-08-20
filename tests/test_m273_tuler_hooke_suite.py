# -*- coding: utf-8 -*-
"""Tests for Milestone M273:
- Tuler-Butcher Spall Failure Model: /FAIL/TULER_BUTCHER and aliases
- Engine Spring Energy Output Directive: /ENG/SPRING_ENERGY and aliases
- Hooke Joint Constraint: /LAGMUL/HOOKE_JOINT and aliases
- Spring Torsional Energy Sensor: /SENSOR/SPRING_TORSIONAL_ENERGY and aliases
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
# 1. /FAIL/TULER_BUTCHER Tests
# ============================================================================

def test_m273_fail_tuler_butcher_fixed(tmp_path: Path):
    c1 = f"{500.0:>20.4f}{1.0e-3:>20.4e}{3.0:>20.4f}{0.80:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}{0.95:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Tuler-Butcher Spall Failure Fixed Format Test
2022 0
/FAIL/TULER_BUTCHER/701
Tuler-Butcher Spall Model
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 701 in model.fail_tuler_butchers
    ftb = model.fail_tuler_butchers[701]
    assert pytest.approx(ftb.sigma_spall) == 500.0
    assert pytest.approx(ftb.k_tb) == 1.0e-3
    assert pytest.approx(ftb.lambda_tb) == 3.0
    assert pytest.approx(ftb.d_crit) == 0.80
    assert ftb.ifail_sh == 2
    assert ftb.ifail_so == 1
    assert pytest.approx(ftb.d_max) == 0.95


def test_m273_fail_tuler_butcher_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Tuler-Butcher Defaults Test
2022 0
/FAIL/TULER_BUTCHER/702
Tuler-Butcher Default Values
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m273_fail_tuler_butcher_aliases(tmp_path: Path):
    c1 = f"{200.0:>20.4f}{2.0e-4:>20.4e}{2.5:>20.4f}{0.70:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Tuler-Butcher Aliases Test
2022 0
/FAIL/TULER_BUTCHER_SPALL/750
TB Spall Alias
{c1}
/FAIL/TB_SPALL/751
TB Spall Short Alias
{c1}
/FAIL/TULER_SPALL/752
Tuler Spall Alias
{c1}
/FAIL/BUTCHER_SPALL/753
Butcher Spall Alias
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 750 in model.fail_tuler_butchers
    assert 751 in model.fail_tuler_butchers
    assert 752 in model.fail_tuler_butchers
    assert 753 in model.fail_tuler_butchers
    assert pytest.approx(model.fail_tuler_butchers[750].sigma_spall) == 200.0
    assert pytest.approx(model.fail_tuler_butchers[751].sigma_spall) == 200.0
    assert pytest.approx(model.fail_tuler_butchers[752].sigma_spall) == 200.0
    assert pytest.approx(model.fail_tuler_butchers[753].sigma_spall) == 200.0


# ============================================================================
# 2. /ENG/SPRING_ENERGY Tests
# ============================================================================

def test_m273_eng_spring_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Spring Energy Output Test
1 1
/ENG/SPRING_ENERGY/1
Spring Energy Field Output
              0.0009        35
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_spring_energies
    ese = model.eng_spring_energies[1]
    assert pytest.approx(ese.dt_spring) == 0.0009
    assert ese.sens_id == 35


def test_m273_eng_spring_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Spring Energy Aliases Test
1 1
/ENG/SPR_ENERGY/2
Spr Energy Alias
              0.0001         0
/ENG/SPRING_WORK/3
Spring Work Alias
              0.0002         1
/ENG/ESPR/4
Espr Alias
              0.0003         3
/ENG/SPRING_ENER/5
Spring Ener Alias
              0.0004         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_spring_energies
    assert 3 in model.eng_spring_energies
    assert 4 in model.eng_spring_energies
    assert 5 in model.eng_spring_energies
    assert pytest.approx(model.eng_spring_energies[2].dt_spring) == 0.0001
    assert pytest.approx(model.eng_spring_energies[3].dt_spring) == 0.0002
    assert pytest.approx(model.eng_spring_energies[4].dt_spring) == 0.0003
    assert pytest.approx(model.eng_spring_energies[5].dt_spring) == 0.0004


# ============================================================================
# 3. /LAGMUL/HOOKE_JOINT Tests
# ============================================================================

def test_m273_hooke_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Hooke Joint Test
1 1
/LAGMUL/HOOKE_JOINT/980
Hooke Universal Cardan Joint
        60        70        80              7.5e+6         8              3.0e-6
                 1.0                 0.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 980 in model.lagmul_hooke_joints
    hj = model.lagmul_hooke_joints[980]
    assert hj.node1 == 60
    assert hj.node2 == 70
    assert hj.node3 == 80
    assert pytest.approx(hj.stiff) == 7.5e+6
    assert hj.skew_id == 8
    assert pytest.approx(hj.tol) == 3.0e-6
    assert pytest.approx(hj.axis_x) == 1.0
    assert pytest.approx(hj.axis_y) == 0.0
    assert pytest.approx(hj.axis_z) == 0.0


def test_m273_hooke_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Hooke Joint Aliases Test
1 1
/HOOKE_JOINT/981
Hooke Joint Alias 1
         1         2         3              1.0e+6         0              1.0e-6
/LAGMUL/HOOKE_COUPLING/982
Hooke Coupling Alias 2
         4         5         6              2.0e+6         0              1.0e-6
/HOOKE_COUPLING/983
Hooke Coupling Alias 3
         7         8         9              3.0e+6         0              1.0e-6
/HOOKE_MECHANISM/984
Hooke Mechanism Alias 4
        10        11        12              4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 981 in model.lagmul_hooke_joints
    assert 982 in model.lagmul_hooke_joints
    assert 983 in model.lagmul_hooke_joints
    assert 984 in model.lagmul_hooke_joints
    assert model.lagmul_hooke_joints[981].node3 == 3
    assert model.lagmul_hooke_joints[982].node3 == 6
    assert model.lagmul_hooke_joints[983].node3 == 9
    assert model.lagmul_hooke_joints[984].node3 == 12


# ============================================================================
# 4. /SENSOR/SPRING_TORSIONAL_ENERGY Tests
# ============================================================================

def test_m273_sensor_spring_torsional_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Torsional Energy Sensor Test
1 1
/SENSOR/SPRING_TORSIONAL_ENERGY/1900
Spring Torsional Energy Sensor 1
       980              1.5e+5               0.0035
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1900 in model.sensor_spring_torsional_energies
    sens = model.sensor_spring_torsional_energies[1900]
    assert sens.id == 1900
    assert sens.spring_id == 980
    assert pytest.approx(sens.u_tors_max) == 1.5e+5
    assert pytest.approx(sens.t_delay) == 0.0035

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 1900]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_TORSIONAL_ENERGY"
    assert pytest.approx(matching[0].tdelay) == 0.0035


def test_m273_sensor_spring_torsional_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Torsional Energy Sensor Aliases Test
1 1
/SENSOR/SPRING_TORS_ENERGY/2000
Spring Tors Energy Alias 1
       801              4.0e+4               0.001
/SENSOR/SPRING_TWIST_ENERGY/2001
Spring Twist Energy Alias 2
       802              5.0e+4               0.002
/SENSOR/TORSIONAL_ENERGY_SPRING/2002
Torsional Energy Spring Alias 3
       803              6.0e+4               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2000 in model.sensor_spring_torsional_energies
    assert 2001 in model.sensor_spring_torsional_energies
    assert 2002 in model.sensor_spring_torsional_energies
    assert pytest.approx(model.sensor_spring_torsional_energies[2000].u_tors_max) == 4.0e+4
    assert pytest.approx(model.sensor_spring_torsional_energies[2001].u_tors_max) == 5.0e+4
    assert pytest.approx(model.sensor_spring_torsional_energies[2002].u_tors_max) == 6.0e+4
