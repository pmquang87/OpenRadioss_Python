# -*- coding: utf-8 -*-
"""Tests for Milestone M268:
- Mohr-Coulomb Failure Model: /FAIL/MOHR_COULOMB, /FAIL/MOHR_COULOMB_MODEL, /FAIL/MC_DAMAGE, /FAIL/MC_FAILURE, /FAIL/MOHR_COULOMB_LAW
- Engine Deviatoric Energy Output Directive: /ENG/DEVIATORIC_ENERGY, /ENG/DEV_ENERGY, /ENG/W_DEV, /ENG/DEVIATORIC_WORK, /ENG/EDEV
- Oldham Coupling Joint Constraint: /LAGMUL/OLDHAM_COUPLING, /OLDHAM_COUPLING, /LAGMUL/OLDHAM_JOINT, /OLDHAM_JOINT, /OLDHAM_MECHANISM
- Spring Volumetric Energy Sensor: /SENSOR/SPRING_VOLUMETRIC_ENERGY, /SENSOR/SPRING_VOL_ENERGY, /SENSOR/SPRING_HYDRO_ENERGY, /SENSOR/VOLUMETRIC_ENERGY_SPRING
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
# 1. /FAIL/MOHR_COULOMB Tests
# ============================================================================

def test_m268_fail_mohr_coulomb_fixed(tmp_path: Path):
    c1 = f"{25.0:>20.4f}{32.5:>20.4f}{150.0:>20.4f}{10.0:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}{0.95:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Mohr-Coulomb Failure Model Fixed Format Test
2022 0
/FAIL/MOHR_COULOMB/205
Mohr-Coulomb Pressure Dependent Model
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 205 in model.fail_mohr_coulombs
    fmc = model.fail_mohr_coulombs[205]
    assert pytest.approx(fmc.cohesion) == 25.0
    assert pytest.approx(fmc.phi) == 32.5
    assert pytest.approx(fmc.tens_limit) == 150.0
    assert pytest.approx(fmc.dilatancy) == 10.0
    assert fmc.ifail_sh == 2
    assert fmc.ifail_so == 1
    assert pytest.approx(fmc.d_max) == 0.95


def test_m268_fail_mohr_coulomb_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Mohr-Coulomb Defaults Test
2022 0
/FAIL/MOHR_COULOMB/206
Mohr-Coulomb Default Values
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m268_fail_mohr_coulomb_aliases(tmp_path: Path):
    c1 = f"{15.0:>20.4f}{28.0:>20.4f}{80.0:>20.4f}{5.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Mohr-Coulomb Aliases Test
2022 0
/FAIL/MOHR_COULOMB_MODEL/250
Mohr Coulomb Model Alias
{c1}
/FAIL/MC_DAMAGE/251
MC Damage Alias
{c1}
/FAIL/MC_FAILURE/252
MC Failure Alias
{c1}
/FAIL/MOHR_COULOMB_LAW/253
Mohr Coulomb Law Alias
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 250 in model.fail_mohr_coulombs
    assert 251 in model.fail_mohr_coulombs
    assert 252 in model.fail_mohr_coulombs
    assert 253 in model.fail_mohr_coulombs
    assert pytest.approx(model.fail_mohr_coulombs[250].cohesion) == 15.0
    assert pytest.approx(model.fail_mohr_coulombs[251].cohesion) == 15.0
    assert pytest.approx(model.fail_mohr_coulombs[252].cohesion) == 15.0
    assert pytest.approx(model.fail_mohr_coulombs[253].cohesion) == 15.0


# ============================================================================
# 2. /ENG/DEVIATORIC_ENERGY Tests
# ============================================================================

def test_m268_eng_deviatoric_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Deviatoric Energy Output Test
1 1
/ENG/DEVIATORIC_ENERGY/1
Deviatoric Strain Energy Field Output
              0.0005        12
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_deviatoric_energies
    ede = model.eng_deviatoric_energies[1]
    assert pytest.approx(ede.dt_wdev) == 0.0005
    assert ede.sens_id == 12


def test_m268_eng_deviatoric_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Deviatoric Energy Aliases Test
1 1
/ENG/DEV_ENERGY/2
Dev Energy Alias
              0.0001         0
/ENG/W_DEV/3
W Dev Alias
              0.0002         1
/ENG/DEVIATORIC_WORK/4
Deviatoric Work Alias
              0.0003         3
/ENG/EDEV/5
Edev Alias
              0.0004         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_deviatoric_energies
    assert 3 in model.eng_deviatoric_energies
    assert 4 in model.eng_deviatoric_energies
    assert 5 in model.eng_deviatoric_energies
    assert pytest.approx(model.eng_deviatoric_energies[2].dt_wdev) == 0.0001
    assert pytest.approx(model.eng_deviatoric_energies[3].dt_wdev) == 0.0002
    assert pytest.approx(model.eng_deviatoric_energies[4].dt_wdev) == 0.0003
    assert pytest.approx(model.eng_deviatoric_energies[5].dt_wdev) == 0.0004


# ============================================================================
# 3. /LAGMUL/OLDHAM_COUPLING Tests
# ============================================================================

def test_m268_oldham_coupling(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Oldham Coupling Joint Test
1 1
/LAGMUL/OLDHAM_COUPLING/840
Parallel Offset Oldham Shaft Coupling
        15        25        35              9.0e+6         3              1.0e-6
                 0.0                 0.0                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 840 in model.lagmul_oldham_couplings
    oc = model.lagmul_oldham_couplings[840]
    assert oc.node1 == 15
    assert oc.node2 == 25
    assert oc.node3 == 35
    assert pytest.approx(oc.stiff) == 9.0e+6
    assert oc.skew_id == 3
    assert pytest.approx(oc.tol) == 1.0e-6
    assert pytest.approx(oc.axis_x) == 0.0
    assert pytest.approx(oc.axis_y) == 0.0
    assert pytest.approx(oc.axis_z) == 1.0


def test_m268_oldham_coupling_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Oldham Coupling Aliases Test
1 1
/OLDHAM_COUPLING/841
Oldham Coupling Alias 1
         1         2         3              1.0e+6         0              1.0e-6
/LAGMUL/OLDHAM_JOINT/842
Oldham Coupling Alias 2
         4         5         6              2.0e+6         0              1.0e-6
/OLDHAM_JOINT/843
Oldham Coupling Alias 3
         7         8         9              3.0e+6         0              1.0e-6
/OLDHAM_MECHANISM/844
Oldham Coupling Alias 4
        10        11        12              4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 841 in model.lagmul_oldham_couplings
    assert 842 in model.lagmul_oldham_couplings
    assert 843 in model.lagmul_oldham_couplings
    assert 844 in model.lagmul_oldham_couplings
    assert model.lagmul_oldham_couplings[841].node3 == 3
    assert model.lagmul_oldham_couplings[842].node3 == 6
    assert model.lagmul_oldham_couplings[843].node3 == 9
    assert model.lagmul_oldham_couplings[844].node3 == 12


# ============================================================================
# 4. /SENSOR/SPRING_VOLUMETRIC_ENERGY Tests
# ============================================================================

def test_m268_sensor_spring_volumetric_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Volumetric Energy Sensor Test
1 1
/SENSOR/SPRING_VOLUMETRIC_ENERGY/999
Spring Volumetric Energy Sensor 1
       800              3.5e+5               0.0012
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 999 in model.sensor_spring_volumetric_energies
    sens = model.sensor_spring_volumetric_energies[999]
    assert sens.id == 999
    assert sens.spring_id == 800
    assert pytest.approx(sens.u_vol_max) == 3.5e+5
    assert pytest.approx(sens.t_delay) == 0.0012

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 999]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_VOLUMETRIC_ENERGY"
    assert pytest.approx(matching[0].tdelay) == 0.0012


def test_m268_sensor_spring_volumetric_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Volumetric Energy Sensor Aliases Test
1 1
/SENSOR/SPRING_VOL_ENERGY/1000
Spring Vol Energy Alias 1
       801              4.0e+4               0.001
/SENSOR/SPRING_HYDRO_ENERGY/1001
Spring Hydro Energy Alias 2
       802              5.0e+4               0.002
/SENSOR/VOLUMETRIC_ENERGY_SPRING/1002
Volumetric Energy Spring Alias 3
       803              6.0e+4               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1000 in model.sensor_spring_volumetric_energies
    assert 1001 in model.sensor_spring_volumetric_energies
    assert 1002 in model.sensor_spring_volumetric_energies
    assert pytest.approx(model.sensor_spring_volumetric_energies[1000].u_vol_max) == 4.0e+4
    assert pytest.approx(model.sensor_spring_volumetric_energies[1001].u_vol_max) == 5.0e+4
    assert pytest.approx(model.sensor_spring_volumetric_energies[1002].u_vol_max) == 6.0e+4
