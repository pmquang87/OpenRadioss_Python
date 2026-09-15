# -*- coding: utf-8 -*-
"""Tests for Milestone M269:
- Drucker-Prager Failure Model: /FAIL/DRUCKER_PRAGER, /FAIL/DRUCKER_PRAGER_MODEL, /FAIL/DP_DAMAGE, /FAIL/DP_FAILURE, /FAIL/DRUCKER_PRAGER_LAW
- Engine Strain Rate Output Directive: /ENG/STRAIN_RATE, /ENG/EPSDOT, /ENG/STRAIN_RATE_FIELD, /ENG/RATE_STRAIN, /ENG/EDOT
- Schmidt Coupling Joint Constraint: /LAGMUL/SCHMIDT_COUPLING, /SCHMIDT_COUPLING, /LAGMUL/SCHMIDT_JOINT, /SCHMIDT_JOINT, /SCHMIDT_MECHANISM
- Spring Shear Energy Sensor: /SENSOR/SPRING_SHEAR_ENERGY, /SENSOR/SPRING_SH_ENERGY, /SENSOR/SPRING_SHEAR_DEFORM, /SENSOR/SHEAR_ENERGY_SPRING
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
# 1. /FAIL/DRUCKER_PRAGER Tests
# ============================================================================

def test_m269_fail_drucker_prager_fixed(tmp_path: Path):
    c1 = f"{0.35:>20.4f}{180.0:>20.4f}{250.0:>20.4f}{500.0:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}{0.90:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Drucker-Prager Failure Model Fixed Format Test
2022 0
/FAIL/DRUCKER_PRAGER/301
Drucker-Prager Pressure Dependent Model
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 301 in model.fail_drucker_pragers
    fdp = model.fail_drucker_pragers[301]
    assert pytest.approx(fdp.alpha) == 0.35
    assert pytest.approx(fdp.k_dp) == 180.0
    assert pytest.approx(fdp.tens_limit) == 250.0
    assert pytest.approx(fdp.comp_limit) == 500.0
    assert fdp.ifail_sh == 2
    assert fdp.ifail_so == 1
    assert pytest.approx(fdp.d_max) == 0.90


def test_m269_fail_drucker_prager_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Drucker-Prager Defaults Test
2022 0
/FAIL/DRUCKER_PRAGER/302
Drucker-Prager Default Values
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m269_fail_drucker_prager_aliases(tmp_path: Path):
    c1 = f"{0.20:>20.4f}{120.0:>20.4f}{200.0:>20.4f}{400.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Drucker-Prager Aliases Test
2022 0
/FAIL/DRUCKER_PRAGER_MODEL/350
Drucker Prager Model Alias
{c1}
/FAIL/DP_DAMAGE/351
DP Damage Alias
{c1}
/FAIL/DP_FAILURE/352
DP Failure Alias
{c1}
/FAIL/DRUCKER_PRAGER_LAW/353
Drucker Prager Law Alias
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 350 in model.fail_drucker_pragers
    assert 351 in model.fail_drucker_pragers
    assert 352 in model.fail_drucker_pragers
    assert 353 in model.fail_drucker_pragers
    assert pytest.approx(model.fail_drucker_pragers[350].alpha) == 0.20
    assert pytest.approx(model.fail_drucker_pragers[351].alpha) == 0.20
    assert pytest.approx(model.fail_drucker_pragers[352].alpha) == 0.20
    assert pytest.approx(model.fail_drucker_pragers[353].alpha) == 0.20


# ============================================================================
# 2. /ENG/STRAIN_RATE Tests
# ============================================================================

def test_m269_eng_strain_rate(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Strain Rate Output Test
1 1
/ENG/STRAIN_RATE/1
Strain Rate Field Output
              0.0008        15
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_strain_rates
    esr = model.eng_strain_rates[1]
    assert pytest.approx(esr.dt_epsdot) == 0.0008
    assert esr.sens_id == 15


def test_m269_eng_strain_rate_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Strain Rate Aliases Test
1 1
/ENG/EPSDOT/2
Epsdot Alias
              0.0001         0
/ENG/STRAIN_RATE_FIELD/3
Strain Rate Field Alias
              0.0002         1
/ENG/RATE_STRAIN/4
Rate Strain Alias
              0.0003         3
/ENG/EDOT/5
Edot Alias
              0.0004         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_strain_rates
    assert 3 in model.eng_strain_rates
    assert 4 in model.eng_strain_rates
    assert 5 in model.eng_strain_rates
    assert pytest.approx(model.eng_strain_rates[2].dt_epsdot) == 0.0001
    assert pytest.approx(model.eng_strain_rates[3].dt_epsdot) == 0.0002
    assert pytest.approx(model.eng_strain_rates[4].dt_epsdot) == 0.0003
    assert pytest.approx(model.eng_strain_rates[5].dt_epsdot) == 0.0004


# ============================================================================
# 3. /LAGMUL/SCHMIDT_COUPLING Tests
# ============================================================================

def test_m269_schmidt_coupling(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Schmidt Coupling Joint Test
1 1
/LAGMUL/SCHMIDT_COUPLING/900
Double Cardan Schmidt Shaft Coupling
        20        30        40              8.0e+6         5              2.0e-6
                 0.0                 0.0                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 900 in model.lagmul_schmidt_couplings
    sc = model.lagmul_schmidt_couplings[900]
    assert sc.node1 == 20
    assert sc.node2 == 30
    assert sc.node3 == 40
    assert pytest.approx(sc.stiff) == 8.0e+6
    assert sc.skew_id == 5
    assert pytest.approx(sc.tol) == 2.0e-6
    assert pytest.approx(sc.axis_x) == 0.0
    assert pytest.approx(sc.axis_y) == 0.0
    assert pytest.approx(sc.axis_z) == 1.0


def test_m269_schmidt_coupling_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Schmidt Coupling Aliases Test
1 1
/SCHMIDT_COUPLING/901
Schmidt Coupling Alias 1
         1         2         3              1.0e+6         0              1.0e-6
/LAGMUL/SCHMIDT_JOINT/902
Schmidt Joint Alias 2
         4         5         6              2.0e+6         0              1.0e-6
/SCHMIDT_JOINT/903
Schmidt Joint Alias 3
         7         8         9              3.0e+6         0              1.0e-6
/SCHMIDT_MECHANISM/904
Schmidt Mechanism Alias 4
        10        11        12              4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 901 in model.lagmul_schmidt_couplings
    assert 902 in model.lagmul_schmidt_couplings
    assert 903 in model.lagmul_schmidt_couplings
    assert 904 in model.lagmul_schmidt_couplings
    assert model.lagmul_schmidt_couplings[901].node3 == 3
    assert model.lagmul_schmidt_couplings[902].node3 == 6
    assert model.lagmul_schmidt_couplings[903].node3 == 9
    assert model.lagmul_schmidt_couplings[904].node3 == 12


# ============================================================================
# 4. /SENSOR/SPRING_SHEAR_ENERGY Tests
# ============================================================================

def test_m269_sensor_spring_shear_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Shear Energy Sensor Test
1 1
/SENSOR/SPRING_SHEAR_ENERGY/1100
Spring Shear Energy Sensor 1
       900              2.5e+5               0.0015
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1100 in model.sensor_spring_shear_energies
    sens = model.sensor_spring_shear_energies[1100]
    assert sens.id == 1100
    assert sens.spring_id == 900
    assert pytest.approx(sens.u_shear_max) == 2.5e+5
    assert pytest.approx(sens.t_delay) == 0.0015

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 1100]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_SHEAR_ENERGY"
    assert pytest.approx(matching[0].tdelay) == 0.0015


def test_m269_sensor_spring_shear_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Shear Energy Sensor Aliases Test
1 1
/SENSOR/SPRING_SH_ENERGY/1200
Spring Sh Energy Alias 1
       801              4.0e+4               0.001
/SENSOR/SPRING_SHEAR_DEFORM/1201
Spring Shear Deform Alias 2
       802              5.0e+4               0.002
/SENSOR/SHEAR_ENERGY_SPRING/1202
Shear Energy Spring Alias 3
       803              6.0e+4               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1200 in model.sensor_spring_shear_energies
    assert 1201 in model.sensor_spring_shear_energies
    assert 1202 in model.sensor_spring_shear_energies
    assert pytest.approx(model.sensor_spring_shear_energies[1200].u_shear_max) == 4.0e+4
    assert pytest.approx(model.sensor_spring_shear_energies[1201].u_shear_max) == 5.0e+4
    assert pytest.approx(model.sensor_spring_shear_energies[1202].u_shear_max) == 6.0e+4
