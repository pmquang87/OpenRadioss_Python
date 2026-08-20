# -*- coding: utf-8 -*-
"""Tests for Milestone M275:
- Oyane Porous Ductile Fracture Failure Model: /FAIL/OYANE and aliases
- Engine Rigid Wall Energy Output Directive: /ENG/RWALL_ENERGY and aliases
- Thompson CV Coupling Joint Constraint: /LAGMUL/THOMPSON_COUPLING and aliases
- Spring Pinching Energy Sensor: /SENSOR/SPRING_PINCHING_ENERGY and aliases
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
# 1. /FAIL/OYANE Tests
# ============================================================================

def test_m275_fail_oyane_fixed(tmp_path: Path):
    c1 = f"{1.25:>20.4f}{0.45:>20.4f}{850.0:>20.4f}{0.02:>20.4f}"
    c2 = f"{2:>10d}{1:>10d}{0.92:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Oyane Porous Ductile Fracture Fixed Format Test
2022 0
/FAIL/OYANE/810
Oyane Fracture Model
{c1}
{c2}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 810 in model.fail_oyanes
    fo = model.fail_oyanes[810]
    assert pytest.approx(fo.c_oyane) == 1.25
    assert pytest.approx(fo.b_oyane) == 0.45
    assert pytest.approx(fo.sigma_cut) == 850.0
    assert pytest.approx(fo.eps_p_min) == 0.02
    assert fo.ifail_sh == 2
    assert fo.ifail_so == 1
    assert pytest.approx(fo.d_max) == 0.92


def test_m275_fail_oyane_defaults(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Oyane Defaults Test
2022 0
/FAIL/OYANE/811
Oyane Default Values
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 1
    assert "missing data card" in log.errors[0]


def test_m275_fail_oyane_aliases(tmp_path: Path):
    c1 = f"{0.80:>20.4f}{0.30:>20.4f}{600.0:>20.4f}{0.01:>20.4f}"
    deck = f"""# RADIOSS STARTER DECK
/BEGIN
Oyane Aliases Test
2022 0
/FAIL/OYANE_MODEL/860
Oyane Model Alias
{c1}
/FAIL/OYANE_DUCTILE/861
Oyane Ductile Alias
{c1}
/FAIL/OYANE_FRACTURE/862
Oyane Fracture Alias
{c1}
/FAIL/OYANE_LAW/863
Oyane Law Alias
{c1}
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 860 in model.fail_oyanes
    assert 861 in model.fail_oyanes
    assert 862 in model.fail_oyanes
    assert 863 in model.fail_oyanes
    assert pytest.approx(model.fail_oyanes[860].c_oyane) == 0.80
    assert pytest.approx(model.fail_oyanes[861].c_oyane) == 0.80
    assert pytest.approx(model.fail_oyanes[862].c_oyane) == 0.80
    assert pytest.approx(model.fail_oyanes[863].c_oyane) == 0.80


# ============================================================================
# 2. /ENG/RWALL_ENERGY Tests
# ============================================================================

def test_m275_eng_rwall_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Rigid Wall Energy Output Test
1 1
/ENG/RWALL_ENERGY/1
Rigid Wall Energy Field Output
              0.0015        45
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.eng_rwall_energies
    erw = model.eng_rwall_energies[1]
    assert pytest.approx(erw.dt_rwall) == 0.0015
    assert erw.sens_id == 45


def test_m275_eng_rwall_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Engine Rigid Wall Energy Aliases Test
1 1
/ENG/RWALL_WORK/2
Rwall Work Alias
              0.0001         0
/ENG/ERWALL/3
Erwall Alias
              0.0002         1
/ENG/RWALL_ENER/4
Rwall Ener Alias
              0.0003         3
/ENG/RWALL_ENG/5
Rwall Eng Alias
              0.0004         5
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2 in model.eng_rwall_energies
    assert 3 in model.eng_rwall_energies
    assert 4 in model.eng_rwall_energies
    assert 5 in model.eng_rwall_energies
    assert pytest.approx(model.eng_rwall_energies[2].dt_rwall) == 0.0001
    assert pytest.approx(model.eng_rwall_energies[3].dt_rwall) == 0.0002
    assert pytest.approx(model.eng_rwall_energies[4].dt_rwall) == 0.0003
    assert pytest.approx(model.eng_rwall_energies[5].dt_rwall) == 0.0004


# ============================================================================
# 3. /LAGMUL/THOMPSON_COUPLING Tests
# ============================================================================

def test_m275_thompson_coupling(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Thompson Coupling Test
1 1
/LAGMUL/THOMPSON_COUPLING/995
Thompson CV Joint Coupling
        70        80        90              9.0e+6        10              1.5e-6
                 0.0                 0.0                 1.0
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 995 in model.lagmul_thompson_couplings
    tc = model.lagmul_thompson_couplings[995]
    assert tc.node1 == 70
    assert tc.node2 == 80
    assert tc.node3 == 90
    assert pytest.approx(tc.stiff) == 9.0e+6
    assert tc.skew_id == 10
    assert pytest.approx(tc.tol) == 1.5e-6
    assert pytest.approx(tc.axis_x) == 0.0
    assert pytest.approx(tc.axis_y) == 0.0
    assert pytest.approx(tc.axis_z) == 1.0


def test_m275_thompson_coupling_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Thompson Coupling Aliases Test
1 1
/THOMPSON_COUPLING/996
Thompson Coupling Alias 1
         1         2         3              1.0e+6         0              1.0e-6
/LAGMUL/THOMPSON_JOINT/997
Thompson Joint Alias 2
         4         5         6              2.0e+6         0              1.0e-6
/THOMPSON_JOINT/998
Thompson Joint Alias 3
         7         8         9              3.0e+6         0              1.0e-6
/THOMPSON_MECHANISM/999
Thompson Mechanism Alias 4
        10        11        12              4.0e+6         0              1.0e-6
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 996 in model.lagmul_thompson_couplings
    assert 997 in model.lagmul_thompson_couplings
    assert 998 in model.lagmul_thompson_couplings
    assert 999 in model.lagmul_thompson_couplings
    assert model.lagmul_thompson_couplings[996].node3 == 3
    assert model.lagmul_thompson_couplings[997].node3 == 6
    assert model.lagmul_thompson_couplings[998].node3 == 9
    assert model.lagmul_thompson_couplings[999].node3 == 12


# ============================================================================
# 4. /SENSOR/SPRING_PINCHING_ENERGY Tests
# ============================================================================

def test_m275_sensor_spring_pinching_energy(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Pinching Energy Sensor Test
1 1
/SENSOR/SPRING_PINCHING_ENERGY/2300
Spring Pinching Energy Sensor 1
       995              1.8e+5               0.0050
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2300 in model.sensor_spring_pinching_energies
    sens = model.sensor_spring_pinching_energies[2300]
    assert sens.id == 2300
    assert sens.spring_id == 995
    assert pytest.approx(sens.u_pinch_max) == 1.8e+5
    assert pytest.approx(sens.t_delay) == 0.0050

    # Global sensor list check
    matching = [s for s in model.sensors if s.id == 2300]
    assert len(matching) == 1
    assert matching[0].kind == "SPRING_PINCHING_ENERGY"
    assert pytest.approx(matching[0].tdelay) == 0.0050


def test_m275_sensor_spring_pinching_energy_aliases(tmp_path: Path):
    deck = """# RADIOSS STARTER DECK
/BEGIN
Spring Pinching Energy Sensor Aliases Test
1 1
/SENSOR/SPRING_PINCH_ENERGY/2400
Spring Pinch Energy Alias 1
       801              4.0e+4               0.001
/SENSOR/SPRING_SQUEEZE_ENERGY/2401
Spring Squeeze Energy Alias 2
       802              5.0e+4               0.002
/SENSOR/PINCHING_ENERGY_SPRING/2402
Pinching Energy Spring Alias 3
       803              6.0e+4               0.003
/END
"""
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 2400 in model.sensor_spring_pinching_energies
    assert 2401 in model.sensor_spring_pinching_energies
    assert 2402 in model.sensor_spring_pinching_energies
    assert pytest.approx(model.sensor_spring_pinching_energies[2400].u_pinch_max) == 4.0e+4
    assert pytest.approx(model.sensor_spring_pinching_energies[2401].u_pinch_max) == 5.0e+4
    assert pytest.approx(model.sensor_spring_pinching_energies[2402].u_pinch_max) == 6.0e+4
