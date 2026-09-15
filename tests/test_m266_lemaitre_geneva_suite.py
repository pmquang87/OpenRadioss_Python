# -*- coding: utf-8 -*-
"""Tests for Milestone M266:
- Lemaitre Continuum Ductile Damage Model: /FAIL/LEMAITRE_DAMAGE, /FAIL/LEMAITRE_MODEL, /FAIL/LEM_DAMAGE, /FAIL/LEM_MODEL, /FAIL/LEMAITRE_LAW
- Engine Hydrostatic Pressure Output Directive: /ENG/HYDROSTATIC_PRESSURE, /ENG/HYDRO_PRES, /ENG/P_HYDRO, /ENG/PRESSURE_HYDRO, /ENG/HYD_PRES
- Geneva Drive Joint Constraint: /LAGMUL/GENEVA_DRIVE, /GENEVA_DRIVE, /LAGMUL/GENEVA_MECHANISM, /GENEVA_MECHANISM, /GENEVA_WHEEL
- Spring Bending Energy Sensor: /SENSOR/SPRING_BENDING_ENERGY, /SENSOR/SPRING_BEND_ENERGY, /SENSOR/SPRING_FLEX_ENERGY, /SENSOR/BENDING_ENERGY_SPRING
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
# 1. /FAIL/LEMAITRE_DAMAGE Tests
# ============================================================================

def test_m266_fail_lemaitre_damage_fixed(tmp_path: Path):
    c1 = f"{12.5:>20.4f}{1.5:>20.4f}{0.02:>20.4f}{0.85:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Lemaitre Failure Model Fixed Format Test
2022 0
/FAIL/LEMAITRE_DAMAGE/205
Lemaitre Ductile Damage Model
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 205 in model.fail_lemaitre_damages
    fld = model.fail_lemaitre_damages[205]
    assert pytest.approx(fld.s_coeff) == 12.5
    assert pytest.approx(fld.s_exp) == 1.5
    assert pytest.approx(fld.eps_d) == 0.02
    assert pytest.approx(fld.d_c) == 0.85
    assert fld.ifail_sh == 2
    assert fld.ifail_so == 1


def test_m266_fail_lemaitre_damage_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Lemaitre Defaults Test
2022 0
/FAIL/LEMAITRE_DAMAGE/206
Lemaitre Default Values
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m266_fail_lemaitre_damage_aliases(tmp_path: Path):
    c = f"{8.0:>20.4f}{1.2:>20.4f}{0.01:>20.4f}{0.9:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Lemaitre Aliases Test
2022 0
/FAIL/LEMAITRE_MODEL/250
Lemaitre Model Alias
{c}
/FAIL/LEM_DAMAGE/251
Lem Damage Alias
{c}
/FAIL/LEM_MODEL/252
Lem Model Alias
{c}
/FAIL/LEMAITRE_LAW/253
Lemaitre Law Alias
{c}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 250 in model.fail_lemaitre_damages
    assert 251 in model.fail_lemaitre_damages
    assert 252 in model.fail_lemaitre_damages
    assert 253 in model.fail_lemaitre_damages
    assert pytest.approx(model.fail_lemaitre_damages[250].s_coeff) == 8.0
    assert pytest.approx(model.fail_lemaitre_damages[251].s_coeff) == 8.0
    assert pytest.approx(model.fail_lemaitre_damages[252].s_coeff) == 8.0
    assert pytest.approx(model.fail_lemaitre_damages[253].s_coeff) == 8.0


# ============================================================================
# 2. /ENG/HYDROSTATIC_PRESSURE Tests
# ============================================================================

def test_m266_eng_hydrostatic_pressure(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Hydrostatic Pressure Output Test
1 1
/ENG/HYDROSTATIC_PRESSURE/1
Hydrostatic Pressure Field Output
              0.0006         9
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_hydrostatic_pressures
    ehp = model.eng_hydrostatic_pressures[1]
    assert pytest.approx(ehp.dt_phyd) == 0.0006
    assert ehp.sens_id == 9


def test_m266_eng_hydrostatic_pressure_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Hydrostatic Pressure Aliases Test
1 1
/ENG/HYDRO_PRES/2
Hydro Pres Alias
              0.0001         0
/ENG/P_HYDRO/3
P Hydro Alias
              0.0002         2
/ENG/PRESSURE_HYDRO/4
Pressure Hydro Alias
              0.0003         4
/ENG/HYD_PRES/5
Hyd Pres Alias
              0.0004         6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_hydrostatic_pressures
    assert 3 in model.eng_hydrostatic_pressures
    assert 4 in model.eng_hydrostatic_pressures
    assert 5 in model.eng_hydrostatic_pressures
    assert pytest.approx(model.eng_hydrostatic_pressures[2].dt_phyd) == 0.0001
    assert pytest.approx(model.eng_hydrostatic_pressures[3].dt_phyd) == 0.0002
    assert pytest.approx(model.eng_hydrostatic_pressures[4].dt_phyd) == 0.0003
    assert pytest.approx(model.eng_hydrostatic_pressures[5].dt_phyd) == 0.0004


# ============================================================================
# 3. /LAGMUL/GENEVA_DRIVE Tests
# ============================================================================

def test_m266_geneva_drive(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Geneva Drive Joint Test
1 1
/LAGMUL/GENEVA_DRIVE/820
Intermittent Indexing Geneva Wheel Mechanism
        12        22         6              5.0e+6         1              1.0e-6
                 0.0                 0.0                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 820 in model.lagmul_geneva_drives
    gd = model.lagmul_geneva_drives[820]
    assert gd.node1 == 12
    assert gd.node2 == 22
    assert gd.num_slots == 6
    assert pytest.approx(gd.stiff) == 5.0e+6
    assert gd.skew_id == 1
    assert pytest.approx(gd.tol) == 1.0e-6
    assert pytest.approx(gd.axis_x) == 0.0
    assert pytest.approx(gd.axis_y) == 0.0
    assert pytest.approx(gd.axis_z) == 1.0


def test_m266_geneva_drive_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Geneva Drive Aliases Test
1 1
/GENEVA_DRIVE/821
Geneva Drive Alias 1
         1         2         4              1.0e+6         0              1.0e-6
/LAGMUL/GENEVA_MECHANISM/822
Geneva Drive Alias 2
         3         4         5              2.0e+6         0              1.0e-6
/GENEVA_MECHANISM/823
Geneva Drive Alias 3
         5         6         8              3.0e+6         0              1.0e-6
/GENEVA_WHEEL/824
Geneva Drive Alias 4
         7         8         6              4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 821 in model.lagmul_geneva_drives
    assert 822 in model.lagmul_geneva_drives
    assert 823 in model.lagmul_geneva_drives
    assert 824 in model.lagmul_geneva_drives
    assert model.lagmul_geneva_drives[821].num_slots == 4
    assert model.lagmul_geneva_drives[822].num_slots == 5
    assert model.lagmul_geneva_drives[823].num_slots == 8
    assert model.lagmul_geneva_drives[824].num_slots == 6


# ============================================================================
# 4. /SENSOR/SPRING_BENDING_ENERGY Tests
# ============================================================================

def test_m266_sensor_spring_bending_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Bending Energy Sensor Test
1 1
/SENSOR/SPRING_BENDING_ENERGY/990
Spring Bending Energy Sensor 1
       600              9.5e+4               0.0025
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 990 in model.sensor_spring_bending_energies
    sens = model.sensor_spring_bending_energies[990]
    assert sens.id == 990
    assert sens.spring_id == 600
    assert pytest.approx(sens.e_bend_max) == 9.5e+4
    assert pytest.approx(sens.t_delay) == 0.0025

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 990]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_BENDING_ENERGY"
    assert pytest.approx(matching[0].tdelay) == 0.0025


def test_m266_sensor_spring_bending_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Bending Energy Sensor Aliases Test
1 1
/SENSOR/SPRING_BEND_ENERGY/991
Spring Bend Energy Alias 1
       601              1.5e+4               0.001
/SENSOR/SPRING_FLEX_ENERGY/992
Spring Flex Energy Alias 2
       602              2.5e+4               0.002
/SENSOR/BENDING_ENERGY_SPRING/993
Bending Energy Spring Alias 3
       603              3.5e+4               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 991 in model.sensor_spring_bending_energies
    assert 992 in model.sensor_spring_bending_energies
    assert 993 in model.sensor_spring_bending_energies
    assert pytest.approx(model.sensor_spring_bending_energies[991].e_bend_max) == 1.5e+4
    assert pytest.approx(model.sensor_spring_bending_energies[992].e_bend_max) == 2.5e+4
    assert pytest.approx(model.sensor_spring_bending_energies[993].e_bend_max) == 3.5e+4
