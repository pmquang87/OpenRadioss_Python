# -*- coding: utf-8 -*-
"""Tests for Milestone M271:
- Biquadratic Anisotropic Yield Failure Model: /FAIL/BIQUAD_ANISO and aliases
- Engine Hourglass Energy Output Directive: /ENG/HOURGLASS_ENERGY and aliases
- Birfield Joint Constraint: /LAGMUL/BIRFIELD_JOINT and aliases
- Spring Damping Energy Sensor: /SENSOR/SPRING_DAMPING_ENERGY and aliases
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
# 1. /FAIL/BIQUAD_ANISO Tests
# ============================================================================

def test_m271_fail_biquad_aniso_fixed(tmp_path: Path):
    c1 = f"{300.0:>20.4f}{350.0:>20.4f}{200.0:>20.4f}{250.0:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}{0.95:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Biquadratic Anisotropic Failure Fixed Format Test
2022 0
/FAIL/BIQUAD_ANISO/501
Biquadratic Anisotropic Yield Model
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 501 in model.fail_biquad_anisos
    fba = model.fail_biquad_anisos[501]
    assert pytest.approx(fba.sigma_1t) == 300.0
    assert pytest.approx(fba.sigma_1c) == 350.0
    assert pytest.approx(fba.sigma_2t) == 200.0
    assert pytest.approx(fba.sigma_2c) == 250.0
    assert fba.ifail_sh == 2
    assert fba.ifail_so == 1
    assert pytest.approx(fba.d_max) == 0.95


def test_m271_fail_biquad_aniso_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Biquadratic Anisotropic Defaults Test
2022 0
/FAIL/BIQUAD_ANISO/502
Biquadratic Anisotropic Default Values
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m271_fail_biquad_aniso_aliases(tmp_path: Path):
    c1 = f"{100.0:>20.4f}{150.0:>20.4f}{80.0:>20.4f}{120.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Biquadratic Anisotropic Aliases Test
2022 0
/FAIL/BIQUAD_ANISO_MODEL/550
BQ Aniso Model Alias
{c1}
/FAIL/BQ_ANISO/551
BQ Aniso Alias
{c1}
/FAIL/BIQUADRATIC_ANISO/552
Biquadratic Aniso Alias
{c1}
/FAIL/BIQUAD_ANISOTROPIC/553
Biquad Anisotropic Alias
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 550 in model.fail_biquad_anisos
    assert 551 in model.fail_biquad_anisos
    assert 552 in model.fail_biquad_anisos
    assert 553 in model.fail_biquad_anisos
    assert pytest.approx(model.fail_biquad_anisos[550].sigma_1t) == 100.0
    assert pytest.approx(model.fail_biquad_anisos[551].sigma_1t) == 100.0
    assert pytest.approx(model.fail_biquad_anisos[552].sigma_1t) == 100.0
    assert pytest.approx(model.fail_biquad_anisos[553].sigma_1t) == 100.0


# ============================================================================
# 2. /ENG/HOURGLASS_ENERGY Tests
# ============================================================================

def test_m271_eng_hourglass_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Hourglass Energy Output Test
1 1
/ENG/HOURGLASS_ENERGY/1
Hourglass Energy Field Output
              0.0005        25
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_hourglass_energies
    ehe = model.eng_hourglass_energies[1]
    assert pytest.approx(ehe.dt_hg) == 0.0005
    assert ehe.sens_id == 25


def test_m271_eng_hourglass_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Hourglass Energy Aliases Test
1 1
/ENG/HG_ENERGY/2
HG Energy Alias
              0.0001         0
/ENG/HOURG_ENERGY/3
Hourg Energy Alias
              0.0002         1
/ENG/HOURGLASS/4
Hourglass Alias
              0.0003         3
/ENG/EHG/5
EHG Alias
              0.0004         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_hourglass_energies
    assert 3 in model.eng_hourglass_energies
    assert 4 in model.eng_hourglass_energies
    assert 5 in model.eng_hourglass_energies
    assert pytest.approx(model.eng_hourglass_energies[2].dt_hg) == 0.0001
    assert pytest.approx(model.eng_hourglass_energies[3].dt_hg) == 0.0002
    assert pytest.approx(model.eng_hourglass_energies[4].dt_hg) == 0.0003
    assert pytest.approx(model.eng_hourglass_energies[5].dt_hg) == 0.0004


# ============================================================================
# 3. /LAGMUL/BIRFIELD_JOINT Tests
# ============================================================================

def test_m271_birfield_joint(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Birfield Joint Test
1 1
/LAGMUL/BIRFIELD_JOINT/960
Birfield Plunging CV Joint
        30        40        50              9.0e+6         6              4.0e-6
                 0.0                 1.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 960 in model.lagmul_birfield_joints
    bj = model.lagmul_birfield_joints[960]
    assert bj.node1 == 30
    assert bj.node2 == 40
    assert bj.node3 == 50
    assert pytest.approx(bj.stiff) == 9.0e+6
    assert bj.skew_id == 6
    assert pytest.approx(bj.tol) == 4.0e-6
    assert pytest.approx(bj.axis_x) == 0.0
    assert pytest.approx(bj.axis_y) == 1.0
    assert pytest.approx(bj.axis_z) == 0.0


def test_m271_birfield_joint_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Birfield Joint Aliases Test
1 1
/BIRFIELD_JOINT/961
Birfield Joint Alias 1
         1         2         3              1.0e+6         0              1.0e-6
/LAGMUL/BIRFIELD_COUPLING/962
Birfield Coupling Alias 2
         4         5         6              2.0e+6         0              1.0e-6
/BIRFIELD_COUPLING/963
Birfield Coupling Alias 3
         7         8         9              3.0e+6         0              1.0e-6
/BIRFIELD_MECHANISM/964
Birfield Mechanism Alias 4
        10        11        12              4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 961 in model.lagmul_birfield_joints
    assert 962 in model.lagmul_birfield_joints
    assert 963 in model.lagmul_birfield_joints
    assert 964 in model.lagmul_birfield_joints
    assert model.lagmul_birfield_joints[961].node3 == 3
    assert model.lagmul_birfield_joints[962].node3 == 6
    assert model.lagmul_birfield_joints[963].node3 == 9
    assert model.lagmul_birfield_joints[964].node3 == 12


# ============================================================================
# 4. /SENSOR/SPRING_DAMPING_ENERGY Tests
# ============================================================================

def test_m271_sensor_spring_damping_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Damping Energy Sensor Test
1 1
/SENSOR/SPRING_DAMPING_ENERGY/1500
Spring Damping Energy Sensor 1
       960              3.0e+5               0.0025
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1500 in model.sensor_spring_damping_energies
    sens = model.sensor_spring_damping_energies[1500]
    assert sens.id == 1500
    assert sens.spring_id == 960
    assert pytest.approx(sens.u_damp_max) == 3.0e+5
    assert pytest.approx(sens.t_delay) == 0.0025

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 1500]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_DAMPING_ENERGY"
    assert pytest.approx(matching[0].tdelay) == 0.0025


def test_m271_sensor_spring_damping_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Damping Energy Sensor Aliases Test
1 1
/SENSOR/SPRING_DAMP_ENERGY/1600
Spring Damp Energy Alias 1
       801              4.0e+4               0.001
/SENSOR/SPRING_DISSIPATION/1601
Spring Dissipation Alias 2
       802              5.0e+4               0.002
/SENSOR/DAMPING_ENERGY_SPRING/1602
Damping Energy Spring Alias 3
       803              6.0e+4               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1600 in model.sensor_spring_damping_energies
    assert 1601 in model.sensor_spring_damping_energies
    assert 1602 in model.sensor_spring_damping_energies
    assert pytest.approx(model.sensor_spring_damping_energies[1600].u_damp_max) == 4.0e+4
    assert pytest.approx(model.sensor_spring_damping_energies[1601].u_damp_max) == 5.0e+4
    assert pytest.approx(model.sensor_spring_damping_energies[1602].u_damp_max) == 6.0e+4
