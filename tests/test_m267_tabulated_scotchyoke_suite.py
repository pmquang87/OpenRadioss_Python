# -*- coding: utf-8 -*-
"""Tests for Milestone M267:
- Tabulated Multi-Axial Plasticity Failure Model: /FAIL/TABULATED_PLASTICITY, /FAIL/TAB_PLAS, /FAIL/TAB_DAMAGE, /FAIL/TABULATED_DAMAGE, /FAIL/TAB_FAILURE
- Engine Octahedral Shear Stress Output Directive: /ENG/OCTAHEDRAL_SHEAR, /ENG/OCT_SHEAR, /ENG/TAU_OCT, /ENG/OCTAHEDRAL_STRESS, /ENG/OCT_TAU
- Scotch Yoke Joint Constraint: /LAGMUL/SCOTCH_YOKE, /SCOTCH_YOKE, /LAGMUL/SCOTCH_YOKE_MECHANISM, /SCOTCH_YOKE_MECHANISM, /SCOTCH_YOKE_JOINT
- Spring Total Strain Energy Sensor: /SENSOR/SPRING_TOTAL_STRAIN_ENERGY, /SENSOR/SPRING_STRAIN_ENERGY, /SENSOR/SPRING_TOTAL_ENERGY, /SENSOR/STRAIN_ENERGY_SPRING
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
# 1. /FAIL/TABULATED_PLASTICITY Tests
# ============================================================================

def test_m267_fail_tabulated_plasticity_fixed(tmp_path: Path):
    c1 = f"{101:>10d}{102:>10d}{103:>10d}{2:>10d}{1:>10d}{0.92:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Tabulated Plasticity Failure Model Fixed Format Test
2022 0
/FAIL/TABULATED_PLASTICITY/205
Tabulated Multi Axial Plasticity Model
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 205 in model.fail_tabulated_plasticities
    ftp = model.fail_tabulated_plasticities[205]
    assert ftp.fct_id_triax == 101
    assert ftp.fct_id_lode == 102
    assert ftp.fct_id_rate == 103
    assert ftp.ifail_sh == 2
    assert ftp.ifail_so == 1
    assert pytest.approx(ftp.d_max) == 0.92


def test_m267_fail_tabulated_plasticity_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Tabulated Plasticity Defaults Test
2022 0
/FAIL/TABULATED_PLASTICITY/206
Tabulated Plasticity Default Values
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m267_fail_tabulated_plasticity_aliases(tmp_path: Path):
    c = f"{201:>10d}{202:>10d}{203:>10d}{1:>10d}{1:>10d}{1.0:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Tabulated Plasticity Aliases Test
2022 0
/FAIL/TAB_PLAS/250
Tab Plas Alias
{c}
/FAIL/TAB_DAMAGE/251
Tab Damage Alias
{c}
/FAIL/TABULATED_DAMAGE/252
Tabulated Damage Alias
{c}
/FAIL/TAB_FAILURE/253
Tab Failure Alias
{c}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 250 in model.fail_tabulated_plasticities
    assert 251 in model.fail_tabulated_plasticities
    assert 252 in model.fail_tabulated_plasticities
    assert 253 in model.fail_tabulated_plasticities
    assert model.fail_tabulated_plasticities[250].fct_id_triax == 201
    assert model.fail_tabulated_plasticities[251].fct_id_triax == 201
    assert model.fail_tabulated_plasticities[252].fct_id_triax == 201
    assert model.fail_tabulated_plasticities[253].fct_id_triax == 201


# ============================================================================
# 2. /ENG/OCTAHEDRAL_SHEAR Tests
# ============================================================================

def test_m267_eng_octahedral_shear(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Octahedral Shear Stress Output Test
1 1
/ENG/OCTAHEDRAL_SHEAR/1
Octahedral Shear Stress Field Output
              0.0004        10
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_octahedral_shears
    eos = model.eng_octahedral_shears[1]
    assert pytest.approx(eos.dt_toct) == 0.0004
    assert eos.sens_id == 10


def test_m267_eng_octahedral_shear_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Octahedral Shear Stress Aliases Test
1 1
/ENG/OCT_SHEAR/2
Oct Shear Alias
              0.0001         0
/ENG/TAU_OCT/3
Tau Oct Alias
              0.0002         1
/ENG/OCTAHEDRAL_STRESS/4
Octahedral Stress Alias
              0.0003         3
/ENG/OCT_TAU/5
Oct Tau Alias
              0.0004         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_octahedral_shears
    assert 3 in model.eng_octahedral_shears
    assert 4 in model.eng_octahedral_shears
    assert 5 in model.eng_octahedral_shears
    assert pytest.approx(model.eng_octahedral_shears[2].dt_toct) == 0.0001
    assert pytest.approx(model.eng_octahedral_shears[3].dt_toct) == 0.0002
    assert pytest.approx(model.eng_octahedral_shears[4].dt_toct) == 0.0003
    assert pytest.approx(model.eng_octahedral_shears[5].dt_toct) == 0.0004


# ============================================================================
# 3. /LAGMUL/SCOTCH_YOKE Tests
# ============================================================================

def test_m267_scotch_yoke(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Scotch Yoke Joint Test
1 1
/LAGMUL/SCOTCH_YOKE/830
Harmonic Scotch Yoke Rotary-to-Linear Mechanism
        14        24                0.15              8.0e+6         2              1.0e-6
                 0.0                 0.0                 1.0                 1.0                 0.0                 0.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 830 in model.lagmul_scotch_yokes
    sy = model.lagmul_scotch_yokes[830]
    assert sy.node1 == 14
    assert sy.node2 == 24
    assert pytest.approx(sy.crank_radius) == 0.15
    assert pytest.approx(sy.stiff) == 8.0e+6
    assert sy.skew_id == 2
    assert pytest.approx(sy.tol) == 1.0e-6
    assert pytest.approx(sy.rot_x) == 0.0
    assert pytest.approx(sy.rot_y) == 0.0
    assert pytest.approx(sy.rot_z) == 1.0
    assert pytest.approx(sy.trans_x) == 1.0
    assert pytest.approx(sy.trans_y) == 0.0
    assert pytest.approx(sy.trans_z) == 0.0


def test_m267_scotch_yoke_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Scotch Yoke Aliases Test
1 1
/SCOTCH_YOKE/831
Scotch Yoke Alias 1
         1         2                 0.1              1.0e+6         0              1.0e-6
/LAGMUL/SCOTCH_YOKE_MECHANISM/832
Scotch Yoke Alias 2
         3         4                 0.2              2.0e+6         0              1.0e-6
/SCOTCH_YOKE_MECHANISM/833
Scotch Yoke Alias 3
         5         6                 0.3              3.0e+6         0              1.0e-6
/SCOTCH_YOKE_JOINT/834
Scotch Yoke Alias 4
         7         8                 0.4              4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 831 in model.lagmul_scotch_yokes
    assert 832 in model.lagmul_scotch_yokes
    assert 833 in model.lagmul_scotch_yokes
    assert 834 in model.lagmul_scotch_yokes
    assert pytest.approx(model.lagmul_scotch_yokes[831].crank_radius) == 0.1
    assert pytest.approx(model.lagmul_scotch_yokes[832].crank_radius) == 0.2
    assert pytest.approx(model.lagmul_scotch_yokes[833].crank_radius) == 0.3
    assert pytest.approx(model.lagmul_scotch_yokes[834].crank_radius) == 0.4


# ============================================================================
# 4. /SENSOR/SPRING_TOTAL_STRAIN_ENERGY Tests
# ============================================================================

def test_m267_sensor_spring_total_strain_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Total Strain Energy Sensor Test
1 1
/SENSOR/SPRING_TOTAL_STRAIN_ENERGY/995
Spring Total Strain Energy Sensor 1
       700              2.5e+5               0.0015
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 995 in model.sensor_spring_total_strain_energies
    sens = model.sensor_spring_total_strain_energies[995]
    assert sens.id == 995
    assert sens.spring_id == 700
    assert pytest.approx(sens.u_total_max) == 2.5e+5
    assert pytest.approx(sens.t_delay) == 0.0015

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 995]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_TOTAL_STRAIN_ENERGY"
    assert pytest.approx(matching[0].tdelay) == 0.0015


def test_m267_sensor_spring_total_strain_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Total Strain Energy Sensor Aliases Test
1 1
/SENSOR/SPRING_STRAIN_ENERGY/996
Spring Strain Energy Alias 1
       701              5.0e+4               0.001
/SENSOR/SPRING_TOTAL_ENERGY/997
Spring Total Energy Alias 2
       702              6.0e+4               0.002
/SENSOR/STRAIN_ENERGY_SPRING/998
Strain Energy Spring Alias 3
       703              7.0e+4               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 996 in model.sensor_spring_total_strain_energies
    assert 997 in model.sensor_spring_total_strain_energies
    assert 998 in model.sensor_spring_total_strain_energies
    assert pytest.approx(model.sensor_spring_total_strain_energies[996].u_total_max) == 5.0e+4
    assert pytest.approx(model.sensor_spring_total_strain_energies[997].u_total_max) == 6.0e+4
    assert pytest.approx(model.sensor_spring_total_strain_energies[998].u_total_max) == 7.0e+4
